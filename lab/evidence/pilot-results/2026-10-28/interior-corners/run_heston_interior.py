"""One predeclared interior development observation; existing PyMC and BF APIs."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify(plan):
    for name, digest in plan['frozen_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Frozen source/input changed: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    sys.path[:0] = [str(Path(plan['repo']) / 'src'), str(Path(plan['repo']) / 'scripts')]
    def stop(signum, frame):
        raise TimeoutError('1740-second interior-case deadline; preserve partial traces, no automatic retry')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(1740)
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 4 or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('One declared four-CPU Slurm allocation is required')
    verify(plan)
    import numpy as np
    from sbi4finance.calibration import bank, reference, tasks
    from calibration_reference import PartialDrawWriter
    from check_heston_reference_precision import check as precision_check

    root = Path(plan['root'])
    out = root / 'reference'
    out.mkdir(exist_ok=False)
    task = plan['task']
    truth = np.asarray(plan['truth'])
    status = {'format': 'heston_interior_development_run_v1', 'status': 'generating_observation',
              'scientific_acceptance': False, 'manifest_sha256': sha(args.manifest),
              'selection': plan['case_selection'], 'stages': {}}
    path = root / 'run.json'
    save(path, status)
    started = time.perf_counter()
    try:
        prices, _, _, difference, _ = tasks.price(task, truth, np.empty(0), seed=0)
        prices = np.asarray(prices)
        noise = np.random.default_rng(np.random.SeedSequence(plan['observation_entropy'])).normal(size=63)
        observed = prices + tasks.sigma(task) * noise
        np.savez_compressed(root / 'observation.npz', truth=truth, clean_prices=prices,
                            observed_prices=observed, standardized_quote_noise=noise,
                            checked_fourier_difference=np.asarray(difference))
        likelihood = reference.heston_log_likelihood(task, observed)
        source = Path(reference.__file__).parent
        record = {'format': 'full_surface_pymc_reference_v1', 'task': task,
                  'target': 'independent Gaussian quote noise around deterministic refined Fourier prices',
                  'likelihood_kind': 'fourier', 'scientific_acceptance': False,
                  'sampler': 'PyMC DEMetropolisZ; native four independent chains with data-only initialization',
                  'bank_manifest_sha256': None, 'surface_id': 'development:midpoint-20261028',
                  'split': 'development', 'generating_parameters': truth.tolist(),
                  'observed_prices': observed.tolist(), 'observation_sha256': sha(root / 'observation.npz'),
                  'settings': plan['mcmc'], 'source_sha256': {
                      n: sha(source/n) for n in ('reference.py', 'pymc_reference_op.py',
                                                 'reference_initialization.py', 'variance_functionals.py')},
                  'reference_pricing_source_sha256': bank.source_hashes(task),
                  'versions': {n: metadata.version(n) for n in ('pymc', 'pytensor', 'arviz-stats', 'numpy', 'scipy')},
                  'approximation_scope': 'Checked finite Fourier price model; pointwise refinement is a separate check.',
                  'case_selection': plan['case_selection']}
        save(out / 'reference.json', record)
        status['status'] = 'initialization'
        save(path, status)
        settings = plan['mcmc']
        initialization = reference.data_fit_initialization(
            likelihood, task['prior_bounds'], 4, settings['seed'], fit_starts=4, fit_budget=2000)
        record['initialization'] = initialization
        save(out / 'reference.json', record)
        callback = PartialDrawWriter(out / 'partial', chains=4, tune=4000, draws=8000, every=100)
        status['status'] = 'mcmc'
        save(path, status)
        idata = reference.sample_deterministic_reference(
            likelihood, task['prior_bounds'], task['parameter_names'], draws=8000, tune=4000,
            chains=4, cores=4, mp_context='spawn', seed=settings['seed'], progressbar=False,
            callback=callback, initialization=initialization)
        chains = idata['posterior']['parameters'].transpose('chain', 'draw', 'parameter').values
        arrays = {'parameters': chains}
        for name, value in idata['sample_stats'].data_vars.items():
            if set(value.dims) == {'chain', 'draw'}:
                arrays['sample_stats_' + name] = value.transpose('chain', 'draw').values
        np.savez_compressed(out / 'chains.npz', **arrays)
        diagnostics = reference.chain_diagnostics(idata)
        diagnostics['variance_functionals'] = reference.variance_functional_diagnostics(idata, task)
        diagnostics['elapsed_seconds'] = time.perf_counter() - started
        save(out / 'diagnostics.json', diagnostics)
        status['stages']['reference'] = {'status': 'completed', 'diagnostics_passed': diagnostics['mcmc_diagnostics_passed']}
        status['status'] = 'precision'
        save(path, status)
        precision = precision_check(out, points=16)
        save(root / 'precision.json', precision)
        status['stages']['precision'] = {'status': 'completed', 'all_points_priced': precision['all_points_priced']}
        bounds = np.asarray(task['prior_bounds'])
        unit = (chains.reshape(-1,4) - bounds[:,0]) / (bounds[:,1]-bounds[:,0])
        quantiles = np.quantile(unit, [.005,.995], axis=0)
        interior = bool(np.all(quantiles[0] > .05) and np.all(quantiles[1] < .95))
        save(root / 'interior-check.json', {'criterion': plan['interior_criterion'],
            'central_99_percent_unit_prior_quantiles': quantiles.tolist(), 'interior_passed': interior,
            'fraction_any_parameter_outside_inner_90_percent_prior_box': float(np.mean(np.any((unit<.05)|(unit>.95),axis=1))),
            'scientific_acceptance': False})
        status['stages']['interior'] = {'status': 'completed', 'interior_passed': interior}
        status['status'] = 'neural_readout'
        save(path, status)
        # The frozen candidate is read only after independent sampling. It never
        # supplies MCMC initialization, proposal geometry, or case selection.
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import bayesflow as bf
        from sbi4finance.calibration import full_surface
        if bf.__version__ != '2.1.0':
            raise ValueError('Pinned BayesFlow 2.1 required')
        actual_task, workflow, training = full_surface.load_checkpoint(Path(plan['checkpoint_directory']))
        workflow.load_estimator(Path(plan['candidate']))
        if actual_task != task or dict(workflow.inference_network.integrate_kwargs) != {'method':'tsit5','steps':'adaptive'}:
            raise ValueError('Frozen task/native sampling policy mismatch')
        draws = full_surface.sample(workflow, task, observed[None,:,None], 10000, seed=plan['neural_seed'])
        neural = root / 'neural'
        neural.mkdir(exist_ok=False)
        np.savez_compressed(neural/'draws.npz', parameters=draws.reshape(10000,4))
        save(neural/'run.json', {'candidate_sha256': plan['candidate_sha256'], 'seed':plan['neural_seed'],
            'draws':10000, 'sampler':{'method':'tsit5','steps':'adaptive'}, 'role':'development readout',
            'source_checkpoint_training':training['checkpoint_sha256'], 'scientific_acceptance':False})
        status['stages']['neural_readout'] = {'status':'completed'}
        verify(plan)
        status.update(status='completed_development_case', elapsed_seconds=time.perf_counter()-started,
                      frozen_inputs_verified=True, outputs={str(p.relative_to(root)):sha(p)
                      for p in (root/'observation.npz', out/'reference.json',out/'chains.npz',out/'diagnostics.json',
                                root/'precision.json',root/'interior-check.json',neural/'draws.npz',neural/'run.json')})
        save(path,status)
        print(json.dumps({'status':status['status'],'mcmc_passed':diagnostics['mcmc_diagnostics_passed'],
                          'interior_passed':interior,'elapsed_seconds':status['elapsed_seconds']}),flush=True)
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__,error=str(error),
                      elapsed_seconds=time.perf_counter()-started)
        save(path,status)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()

"""One frozen Heston candidate: held-out rank screen and cached analytic products."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify(manifest):
    for path, digest in manifest['frozen_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen input changed: ' + path)


def classify(error, uncertainty, tolerance):
    if error + uncertainty <= tolerance and uncertainty <= tolerance / 2:
        return 'within_declared_tolerance'
    if error - uncertainty > tolerance:
        return 'resolved_excess'
    return 'inconclusive'


def quantile_check(neural, reference, probability, epsilon):
    import arviz as az
    import numpy as np
    reference_mcse = float(az.mcse({'value': reference}, method='quantile', prob=probability)['value'])
    neural_mcse = float(az.mcse({'value': neural[None, :]}, method='quantile', prob=probability)['value'])
    reference_value = float(np.quantile(reference, probability))
    neural_value = float(np.quantile(neural, probability))
    width = float(np.diff(np.quantile(reference, [.05, .95]))[0])
    tolerance = max(.10 * width, epsilon)
    error = abs(neural_value - reference_value)
    uncertainty = 1.959963984540054 * float(np.hypot(reference_mcse, neural_mcse))
    return {'probability': probability, 'reference': reference_value, 'neural': neural_value,
            'absolute_error': error, 'reference_quantile_mcse': reference_mcse,
            'neural_quantile_mcse': neural_mcse, 'combined_95_percent_half_width': uncertainty,
            'reference_90_percent_width': width, 'tolerance': tolerance,
            'status': classify(error, uncertainty, tolerance)}


def products(manifest, out):
    import numpy as np
    from sbi4finance.calibration import variance_functionals
    out.mkdir(parents=True, exist_ok=False)
    report = {'format': 'heston_cached_analytic_products_v1', 'scientific_acceptance': False,
              'case_role': 'previously inspected development references',
              'candidate_sha256': manifest['candidate_sha256'], 'cases': [],
              'runtime': {name: metadata.version(name) for name in ('numpy', 'arviz', 'xarray')},
              'rules': manifest['products'],
              'limitations': 'Analytic mean-variance functionals test v0/theta only. No path product, '
                             'vol-of-vol/leverage check, new MCMC, or independent conditional acceptance. '
                             'Quantile errors use pointwise approximate normal uncertainty, not simultaneous bounds.'}
    for case in manifest['development_cases']:
        record = json.loads(Path(case['reference']).joinpath('reference.json').read_text())
        diagnostics = json.loads(Path(case['reference']).joinpath('diagnostics.json').read_text())
        if record['task'] != manifest['task']:
            raise ValueError('Reference task differs from frozen candidate task')
        with np.load(Path(case['reference']) / 'chains.npz', allow_pickle=False) as archive:
            chains = archive['parameters']
        with np.load(case['neural_draws'], allow_pickle=False) as archive:
            neural = archive['final']
        if chains.shape != (4, 8000, 4) or neural.shape != (10000, 4):
            raise ValueError('Cached reference/neural axes changed')
        reference_values = variance_functionals.heston_variance_functionals(manifest['task'], chains)
        neural_values = variance_functionals.heston_variance_functionals(manifest['task'], neural)
        rows = {}
        for name in manifest['products']['names']:
            diagnostic = diagnostics['variance_functionals']['names'][name]
            converged = (diagnostic['r_hat'] <= 1.01 and diagnostic['ess_bulk'] >= 400
                         and diagnostic['ess_tail'] >= 400)
            quantiles = [quantile_check(neural_values[name], reference_values[name], q,
                                       manifest['products']['epsilon_annual_variance']) for q in (.1, .9)]
            if not converged:
                for q in quantiles:
                    q['status'] = 'inconclusive_reference'
            rows[name] = {'reference_diagnostics': diagnostic, 'reference_converged': converged,
                          'reference_mean': float(reference_values[name].mean()),
                          'neural_mean': float(neural_values[name].mean()), 'quantiles': quantiles}
        report['cases'].append({'case': case['case'], 'products': rows})
        write(out / 'products.json', report)
    verify(manifest)
    report['immutable_inputs_verified'] = True
    write(out / 'products.json', report)


def dry_products():
    import numpy as np
    from scipy.integrate import quad
    from sbi4finance.calibration import tasks, variance_functionals
    task = tasks.heston_task()
    theta = np.array([.04, .08, .7, -.6])
    values = variance_functionals.heston_variance_functionals(task, theta)
    curve = lambda t: .08 + (.04 - .08) * np.exp(-2 * t)
    assert np.isclose(values['variance_swap_1y'], quad(curve, 0, 1)[0], atol=1e-14)
    assert np.isclose(values['forward_variance_0_5_to_1y'], 2 * quad(curve, .5, 1)[0], atol=1e-14)
    reference = np.random.default_rng(1).normal(size=(4, 1000))
    exact = quantile_check(reference.ravel(), reference, .1, .1)
    shifted = quantile_check(reference.ravel() + 2, reference, .1, .1)
    assert exact['status'] == 'within_declared_tolerance'
    assert shifted['status'] == 'resolved_excess'
    assert classify(.1, .1, .15) == 'inconclusive'
    print(json.dumps({'synthetic_product_checks_passed': True, 'identical': exact, 'shifted': shifted}))


def evaluate(manifest, manifest_path):
    import numpy as np
    import bayesflow as bf
    import jax
    from sbi4finance.calibration import bank_collection, full_surface, validation
    if (bf.__version__ != '2.1.0' or not any(d.platform == 'gpu' for d in jax.devices())
            or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 4):
        raise ValueError('Pinned BF2.1 / GPU / four-CPU allocation required')
    if {name: metadata.version(name) for name in manifest['expected_gpu_runtime']} != manifest['expected_gpu_runtime']:
        raise ValueError('Pinned runtime versions changed since declaration')
    out = Path(manifest['root']) / 'results/acceptance'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'partial').mkdir()
    task, workflow, training = full_surface.load_checkpoint(Path(manifest['checkpoint_directory']))
    workflow.load_estimator(Path(manifest['candidate']))
    if task != manifest['task'] or dict(workflow.inference_network.integrate_kwargs) != manifest['sampler']:
        raise ValueError('Candidate task or native sampler differs')
    bank_manifest, splits = bank_collection.load_bank(Path(manifest['bank']), splits=('test',),
                                                    roles=('observations', 'parameters', 'example_id'))
    split = splits['test']
    ids = [f'test:{i}' for i in range(2000)]
    if (bank_manifest['configuration']['task'] != task or split['example_id'][:2000].tolist() != ids
            or training['run_contract']['dataset']['manifest_sha256'] != manifest['bank_sha256']):
        raise ValueError('Exact task, snapshot and test panel required')
    report = {'format': 'heston_frozen_candidate_gate_v1', 'status': 'sampling',
              'scientific_acceptance': False, 'candidate_role': 'post-hoc development-selected final500, frozen before test',
              'checkpoint_sha256': manifest['candidate_sha256'], 'manifest_sha256': sha(manifest_path),
              'bank_manifest_sha256': manifest['bank_sha256'], 'evaluated_split': 'test',
              'reserved_test_evaluated': True, 'example_ids': ids, 'completed_surfaces': 0,
              'rules': manifest['acceptance'], 'sampler': manifest['sampler'], 'targets': task['parameter_names'],
              'runtime': {name: metadata.version(name) for name in ('bayesflow', 'keras', 'jax', 'jaxlib', 'numpy')},
              'scope': 'One frozen candidate, four marginal rank screens; not three-seed confirmation, '
                       'joint calibration, full conditional/library acceptance or market adequacy.'}
    path = out / 'run.json'
    write(path, report)
    started = time.perf_counter()
    chunks = []
    try:
        for first in range(0, 2000, 100):
            draws = full_surface.sample(workflow, task, split['observations'][first:first + 100],
                                        1000, seed=280028 + first)
            np.savez_compressed(out / 'partial' / f'batch-{first:04d}.npz', draws=draws,
                                truth=split['parameters'][first:first + 100], example_ids=split['example_id'][first:first + 100])
            chunks.append(draws)
            report.update(completed_surfaces=first + 100, sampling_seconds=time.perf_counter() - started)
            write(path, report)
            print(json.dumps({'completed_surfaces': first + 100, 'seconds': report['sampling_seconds']}), flush=True)
        draws = np.concatenate(chunks)
        truth = split['parameters'][:2000]
        sbc = validation.sbc(draws, truth, seed=280028, alpha=.05)
        evaluation = {**report, 'status': 'completed', 'sbc': {k: v for k, v in sbc.items() if k != 'fractional_ranks'},
                      'coverage': validation.coverage(draws, truth),
                      'contraction_logit': dict(zip(task['parameter_names'], validation.contraction(draws, np.asarray(task['prior_bounds'])))),
                      'recovery_correlation': dict(zip(task['parameter_names'], validation.recovery(draws, truth)['correlation'])),
                      'rank_screen_passed': all(sbc['passed']), 'seconds': time.perf_counter() - started}
        verify(manifest)
        evaluation['immutable_inputs_verified'] = True
        write(out / 'evaluation.json', evaluation)
        np.savez_compressed(out / 'draws.npz', draws=draws, truth=truth)
        report.update(status='completed', sbc=evaluation['sbc'], coverage=evaluation['coverage'],
                      rank_screen_passed=evaluation['rank_screen_passed'], immutable_inputs_verified=True)
        write(path, report)
        subprocess.run([manifest['reference_python'], __file__, '--manifest', str(manifest_path),
                        '--products-only'], check=True, env={**os.environ, 'JAX_PLATFORMS': 'cpu'})
        verify(manifest)
        print(json.dumps({'status': 'completed', 'sbc': evaluation['sbc']}), flush=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      sampling_seconds=time.perf_counter() - started)
        write(path, report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--products-only', action='store_true')
    parser.add_argument('--dry-products', action='store_true')
    parser.add_argument('--repo', type=Path)
    args = parser.parse_args()
    def stop(signum, frame):
        raise TimeoutError('1470-second acceptance deadline; retain partial batches and do not retry')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(1470)
    if args.dry_products:
        sys.path.insert(0, str(args.repo / 'src'))
        dry_products()
        return
    manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, str(Path(manifest['repo']) / 'src'))
    verify(manifest)
    if args.products_only:
        products(manifest, Path(manifest['root']) / 'results/products')
    else:
        evaluate(manifest, args.manifest)
    signal.alarm(0)


if __name__ == '__main__':
    main()

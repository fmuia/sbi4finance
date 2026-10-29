"""One fixed-point independent pricing diagnostic; no training or posterior fitting."""
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

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def coupled_controlled_block(parameters, curve, contracts, *, horizon, steps, n_pairs, seed):
    """Bank estimator on two CRN grids, retaining bounded raw antithetic pairs.

    Finest-grid innovations exactly replay rough_block(seed). Coarse normals
    are normalized finest-channel sums, preserving iid standard-normal
    marginals; this is not an exact singular-Wiener-integral bridge.
    """
    from sbi4finance.bergomi import conditional_put_payoffs, curve_values
    from sbi4finance.bergomi_refinement import coarsen_innovations
    from sbi4finance.calibration import pricing
    from sbi4finance.rough_bergomi import simulate_rough

    puts = [dict(c, option_type='put') for c in contracts]
    rng = np.random.default_rng(seed)
    chunks = [([], []) for _ in steps]
    indices, expected = [], []
    for n in steps:
        dt, nodes = pricing._block_contracts(contracts, horizon, n)
        indices.append(nodes)
        xi = curve_values(curve, dt * np.arange(n))
        expected.append({i: float(np.sum(xi[:i]) * dt) for i in sorted(set(nodes.tolist()))})
    remaining = n_pairs
    while remaining:
        size = min(pricing._PAIR_CHUNK, remaining)
        finest = rng.standard_normal((size, steps[-1], 3))
        for grid, n in enumerate(steps):
            z = coarsen_innovations(finest, n, model='rough', dt_years=horizon / steps[-1])
            pair = []
            for sign in (1., -1.):
                paths = simulate_rough(parameters, curve, horizon / n, sign * z,
                                       method='hybrid', conditional_only=True)
                payoff = conditional_put_payoffs(paths, parameters, puts, indices[grid],
                                                  spot=1., rate=0., dividend=0.)
                controls = []
                for i in expected[grid]:
                    iv = paths['integrated_variance'][:, i - 1]
                    iw = paths['volatility_driver_integral'][:, i - 1]
                    controls.extend([np.exp(parameters['rho'] * iw - .5 * parameters['rho']**2 * iv) - 1.,
                                     iv - expected[grid][i]])
                pair.append((payoff, np.column_stack(controls)))
            chunks[grid][0].append(.5 * (pair[0][0] + pair[1][0]))
            chunks[grid][1].append(.5 * (pair[0][1] + pair[1][1]))
        remaining -= size
    raw, adjusted, moments = [], [], []
    parity = np.asarray([1. - c['strike'] if c['option_type'] == 'call' else 0. for c in contracts])
    for payoff_chunks, control_chunks in chunks:
        payoffs, controls = np.concatenate(payoff_chunks), np.concatenate(control_chunks)
        coefficients = np.linalg.lstsq(controls - controls.mean(0), payoffs - payoffs.mean(0), rcond=None)[0]
        controlled = payoffs - controls @ coefficients
        raw.append(payoffs)
        adjusted.append(controlled)
        residual = controlled - controlled.mean(0)
        dof = n_pairs - controls.shape[1] - 1
        moments.append({'coefficient_matrix': coefficients.tolist(), 'control_mean': controls.mean(0).tolist(),
                        'dof': dof, 'bank_covariance': (residual.T @ residual / dof / n_pairs).tolist()})
    # Fit coefficients separately on each grid, as in the bank. Independent
    # replicate dispersion is authoritative for uncertainty in fitted CVs.
    values = np.concatenate([*adjusted, *raw], axis=1)
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered / ((n_pairs - 1) * n_pairs)
    width = len(contracts)
    difference = adjusted[1] - adjusted[0]
    difference_centered = difference - difference.mean(0)
    difference_covariance = difference_centered.T @ difference_centered / ((n_pairs - 1) * n_pairs)
    return {'controlled_prices': np.stack([a.mean(0) + parity for a in adjusted]),
            'raw_prices': np.stack([a.mean(0) + parity for a in raw]),
            'joint_covariance': covariance.reshape(4, width, 4, width),
            'difference': difference.mean(0), 'difference_covariance': difference_covariance,
            'moments': moments}


def prepare(args):
    from sbi4finance.calibration import bank_collection, pricer_net, tasks
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    (root / 'results').mkdir(); (root / 'logs').mkdir(); (root / 'cache').mkdir()
    repo = args.repo
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9':
        raise ValueError('Frozen source56 required')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True):
        raise ValueError('Frozen source is modified')
    manifest, splits = bank_collection.load_bank(args.bank, splits=('validation',),
        roles=('parameters', 'context_values', 'observations', 'reference_prices', 'example_id'))
    task = manifest['configuration']['task']; split = splits['validation']
    if (sha(args.bank / 'manifest.json') != '7cf0314dc6371fd045ad6b314dd6dd605db2f3343dbccf4e35b4df9ddf9397eb'
            or task['task_id'] != 'rough_calibration_v1'):
        raise ValueError('Exact known-curve20k bank required')
    pricer = pricer_net.Pricer.load(args.pricer)
    frozen = [args.bank / 'manifest.json', args.pricer / 'pricer.npz', args.pricer / 'pricer.json',
              args.pricer / 'task.json', repo / 'pixi.lock', repo / 'pyproject.toml', Path(__file__)]
    for name in ['calibration/pricing.py', 'calibration/tasks.py', 'calibration/pricer_net.py',
                 'rough_bergomi.py', 'bergomi.py', 'bergomi_refinement.py']:
        frozen.append(repo / 'src/sbi4finance' / name)
    points, jobs = [], []
    for case in (0, 2):
        if split['example_id'][case] != f'validation:{case}': raise ValueError('Validation IDs differ')
        reference = args.references / 'runs' / f'case-{case}' / 'chains.npz'
        neural = args.references / 'runs/comparisons' / f'case-{case}' / 'neural_draws.npz'
        comparison = args.references / 'runs/comparisons' / f'case-{case}' / 'comparison.json'
        frozen.extend([reference, neural, comparison])
        with np.load(reference, allow_pickle=False) as archive: chains = archive['parameters']
        with np.load(neural, allow_pickle=False) as archive: draws = archive['parameters']
        curves = split['context_values'][case].tolist()
        observation = np.asarray(split['observations'][case]).reshape(-1).tolist()
        for role, theta in [('truth', split['parameters'][case]), ('neural_mean', draws.mean(0)),
                            ('reference_mean', chains.reshape(-1, 3).mean(0))]:
            point = {'case': case, 'role': role, 'theta': theta.tolist(), 'curve_levels': curves,
                     'observed_prices': observation,
                     'emulator_scaled_prices': pricer(pricer_net.features(task, theta, curves))[0].tolist()}
            if role == 'truth': point['stored_bank_prices'] = split['reference_prices'][case].tolist()
            points.append(point)
            for replicate in range(8):
                jobs.append({'index': len(jobs), 'point_index': len(points) - 1, 'replicate': replicate,
                             'stream_entropy_by_block': [[0x52505631, 20261028, case, replicate, b] for b in range(3)]})
    state_digests = []
    for case in (0, 2):
        for replicate in range(8):
            for b in range(3):
                entropy = [0x52505631, 20261028, case, replicate, b]
                state_digests.append(hashlib.sha256(np.random.SeedSequence(entropy).generate_state(8).tobytes()).hexdigest())
    if len(state_digests) != len(set(state_digests)): raise ValueError('Seed collision')
    plan = {'format': 'known_rough_independent_pricing_v1', 'scientific_acceptance': False,
            'reserved_test_evaluated': False, 'source_revision': revision, 'repo': str(repo),
            'root': str(root), 'task': task, 'points': points, 'jobs': jobs, 'paths_per_batch': 65536,
            'replicates_per_point': 8, 'coarse_steps': [80, 504, 504], 'fine_steps': [160, 1008, 1008],
            'seed_namespace': 'RPV1; distinct from bank OPT namespace; shared intentionally across point roles',
            'unique_stream_state_sha256': state_digests, 'frozen_sha256': {str(p): sha(p) for p in frozen},
            'interpretation': 'Fixed-point emulator, pricing-grid and Monte Carlo diagnostics; no posterior acceptance.',
            'external_uncertainty': '8 independent batches, Student t7 intervals; no inversion of 63x63 batch covariance.',
            'coupling': 'Hybrid Gaussian channel block CRN, not an exact singular-integral common-Wiener bridge.',
            'runtime': {'python': sys.version, 'executable': sys.executable,
                        **{name: metadata.version(name) for name in ('numpy', 'scipy')}},
            'within_batch_covariance_scope': 'Conditional on same-sample fitted CV coefficients; external batch dispersion checks calibration.'}
    save(root / 'plan.json', plan)
    print(json.dumps({'status': 'prepared', 'points': len(points), 'jobs': len(jobs), 'plan_sha256': sha(root/'plan.json')}))


def worker(args):
    from sbi4finance.calibration import tasks
    started = time.perf_counter()
    def stop(signum, frame): raise TimeoutError('1100-second pricing deadline; retain failed batch without retry')
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop); signal.alarm(1100)
    plan = json.loads((args.root / 'plan.json').read_text())
    if {name: metadata.version(name) for name in ('numpy','scipy')} != {name:plan['runtime'][name] for name in ('numpy','scipy')}:
        raise ValueError('Prepared numerical runtime changed')
    for name, digest in plan['frozen_sha256'].items():
        if sha(name) != digest: raise ValueError('Frozen input changed: ' + name)
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 1 or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('One allocated CPU required')
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', args.index))
    job = plan['jobs'][index]; point = plan['points'][job['point_index']]; task = plan['task']
    directory = args.root / 'results' / f'batch-{index:02d}'
    directory.mkdir(exist_ok=False)
    status = {'format': 'known_rough_pricing_batch_v1', 'status': 'running', 'index': index,
              'job': job, 'point': point, 'plan_sha256': sha(args.root/'plan.json'),
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'scientific_acceptance': False}
    save(directory / 'run.json', status)
    try:
        contracts = task['context']['contracts']; maturities = np.asarray([c['maturity'] for c in contracts])
        assignment = np.full(len(contracts), -1)
        for b, block in reversed(list(enumerate(task['numerics']['blocks']))):
            assignment[maturities <= block['horizon'] + 1e-12] = b
        parameters = tasks.model_parameters(task, point['theta'])
        curve = tasks.curve(task, point['curve_levels'])
        controlled = np.zeros((2, len(contracts))); raw = controlled.copy()
        covariance = np.zeros((4, len(contracts), 4, len(contracts)))
        difference = np.zeros(len(contracts)); dcov = np.zeros((len(contracts), len(contracts)))
        moments = []
        for b, block in enumerate(task['numerics']['blocks']):
            chosen = np.flatnonzero(assignment == b)
            found = coupled_controlled_block(parameters, curve, [contracts[i] for i in chosen],
                horizon=block['horizon'], steps=(plan['coarse_steps'][b], plan['fine_steps'][b]),
                n_pairs=plan['paths_per_batch']//2, seed=np.random.SeedSequence(job['stream_entropy_by_block'][b]))
            controlled[:, chosen], raw[:, chosen] = found['controlled_prices'], found['raw_prices']
            for a in range(4):
                for c in range(4): covariance[a, chosen[:,None], c, chosen[None,:]] = found['joint_covariance'][a,:,c,:]
            difference[chosen] = found['difference']; dcov[np.ix_(chosen, chosen)] = found['difference_covariance']
            moments.append({'block': b, 'contracts': chosen.tolist(), 'grids': found['moments']})
            np.savez_compressed(directory / f'block-{b}.npz', **{k:v for k,v in found.items() if k!='moments'})
            status['blocks_done'] = b+1; status['elapsed_seconds'] = time.perf_counter()-started; save(directory/'run.json',status)
        np.savez_compressed(directory/'prices.npz', controlled=controlled,raw=raw,covariance=covariance,
                            grid_difference=difference,grid_difference_covariance=dcov)
        for name,digest in plan['frozen_sha256'].items():
            if sha(name)!=digest:raise ValueError('Frozen input changed during pricing: '+name)
        status.update(status='completed', elapsed_seconds=time.perf_counter()-started, moments=moments,
                      prices_sha256=sha(directory/'prices.npz'), frozen_inputs_verified=True)
        save(directory/'run.json',status)
        print(json.dumps({k:status[k] for k in ('index','status','elapsed_seconds')}))
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
        save(directory/'run.json',status);raise
    finally:signal.alarm(0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker'])
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bank',type=Path);parser.add_argument('--pricer',type=Path);parser.add_argument('--references',type=Path)
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args();sys.path.insert(0,str(args.repo/'src'))
    prepare(args) if args.mode=='prepare' else worker(args)


if __name__=='__main__':main()

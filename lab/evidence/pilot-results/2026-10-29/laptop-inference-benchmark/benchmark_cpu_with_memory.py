"""Time saved core checkpoints on one fixed observation; no fitting/simulation."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

STARTED = time.perf_counter()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(120)  # Bound this local CPU measurement; no automatic retry.
    if args.out.exists():
        raise FileExistsError(args.out)
    files = {str(p): sha(p) for p in [args.checkpoint / f for f in ('model.keras', 'task.json', 'training.json')]}
    files[str(args.case)] = sha(args.case)
    source = sorted((args.repo / 'src' / 'sbi4finance' / 'calibration').glob('*.py'))
    source.extend([args.repo / 'src' / 'sbi4finance' / 'gbm_bank.py', Path(__file__)])
    files.update({str(p): sha(p) for p in source})
    case = json.loads(args.case.read_text())
    sys.path.insert(0, str(args.repo / 'src'))
    import numpy as np
    from sbi4finance.calibration import full_surface
    source_import_seconds = time.perf_counter() - STARTED
    peak_before_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    before = time.perf_counter()
    task, workflow, record = full_surface.load_checkpoint(args.checkpoint)
    load_seconds = time.perf_counter() - before
    peak_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import jax
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU benchmark must not use a GPU backend')
    if task != case['task'] or case['split'] != 'validation':
        raise ValueError('Case/model task mismatch or non-validation case')
    if record['network']['family'] != 'flow_matching' or record['network']['flow_steps'] is not None:
        raise ValueError('This benchmark requires BayesFlow native FlowMatching integration')
    versions = {k: metadata.version(k) for k in ('bayesflow', 'keras', 'jax', 'jaxlib', 'numpy', 'scipy')}
    if versions['bayesflow'] != '2.1.0':
        raise ValueError('BayesFlow version differs')
    observation = np.asarray(case['observed_prices'], dtype=np.float64)[None]
    context = None if case.get('known_curve_levels') is None else np.asarray(case['known_curve_levels'], dtype=np.float64)[None]
    timings = []
    cold_first_wall = None
    for label, seed in [('first', 271400), ('warm_1', 271401), ('warm_2', 271402)]:
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        before = time.perf_counter()
        draws = full_surface.sample(workflow, task, observation, 1000,
            encoder=record['network']['encoder'], seed=seed, context_values=context)
        elapsed = time.perf_counter() - before
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        if cold_first_wall is None:
            cold_first_wall = time.perf_counter() - STARTED
        # sample converts backend arrays to NumPy; this forces completion before timing stops.
        if draws.shape != (1, 1000, len(task['parameter_names'])) or not np.isfinite(draws).all():
            raise ValueError('Invalid posterior output')
        bounds = np.asarray(task['prior_bounds'])
        if np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1]):
            raise ValueError('Posterior samples outside model support')
        timings.append({'label': label, 'seed': seed, 'seconds': elapsed,
            'shape': list(draws.shape), 'all_finite': True,
            'cpu_user_seconds': cpu_after.ru_utime - cpu_before.ru_utime,
            'cpu_system_seconds': cpu_after.ru_stime - cpu_before.ru_stime,
            'process_peak_RSS_bytes': cpu_after.ru_maxrss if platform.system() == 'Darwin' else 1024 * cpu_after.ru_maxrss,
            'draws_array_sha256': hashlib.sha256(draws.tobytes()).hexdigest()})
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Input or source bytes changed during measurement: ' + path)
    result = {'format': 'core_laptop_cpu_inference_benchmark_v1', 'status': 'completed',
        'case_id': case['surface_id'], 'split': case['split'], 'checkpoint_sha256': record['checkpoint_sha256'],
        'task_fingerprint': record['task_fingerprint'], 'parameter_names': task['parameter_names'],
        'network': record['network'], 'native_integration': dict(workflow.inference_network.integrate_kwargs),
        'source_import_seconds': source_import_seconds, 'checkpoint_load_seconds': load_seconds,
        'cold_start_to_first_draws_seconds': cold_first_wall,
        'resource_usage': {'units': 'Darwin ru_maxrss bytes; whole-process high-water mark, includes runtime imports and model',
            'peak_RSS_before_load_bytes': peak_before_load, 'peak_RSS_after_load_bytes': peak_after_load,
            'peak_RSS_after_all_sampling_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        'inference_calls': timings, 'total_process_seconds': time.perf_counter() - STARTED,
        'runtime': {'python': platform.python_version(), **versions, 'keras_backend': os.environ.get('KERAS_BACKEND'),
            'jax_backend': jax.default_backend(), 'jax_enable_x64': jax.config.read('jax_enable_x64'), 'jax_devices': list(map(str, jax.devices()))},
        'platform': {'platform': platform.platform(), 'machine': platform.machine(), 'logical_cpu_count': os.cpu_count()},
        'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'JAX_NUM_THREADS', 'XLA_FLAGS', 'JAX_PLATFORMS')},
        'inputs_and_source_sha256': files, 'stable_before_after': True,
        'scope': 'One existing validation observation per model, 1000 posterior draws per call, two warm repeats. No training, numerical simulation, MCMC or notebook execution; no scientific validation/promotion. No disk XLA cache configured.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('case_id', 'checkpoint_load_seconds', 'cold_start_to_first_draws_seconds', 'inference_calls', 'total_process_seconds')}))


if __name__ == '__main__':
    main()

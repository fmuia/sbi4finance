"""Read-only paired Monte Carlo validation loss; never fit, sample posteriors, or price."""
import argparse, hashlib, inspect, json, os, platform, signal, subprocess, sys, time
from importlib import metadata
from pathlib import Path

REV = '56dcb13e1d2bf5b6dc96e573dfcaa6240a7104a9'
BANK_SHA = '4b5929b0d0b23698c340eec6defabfd629fe23eca2ed78fe9bd46e4de35c244e'
CHECKPOINT_SHA = {'best402': '7f037dee0fde78cd5a33f1107e13031ff01fdcb5481dab76fc6c8f7d07598f39',
                  'final500': '8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db'}
SEEDS = tuple(range(281001, 281033))  # Declared before either checkpoint is evaluated.
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    q = p.with_suffix('.json.tmp'); q.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n'); q.replace(p)
def array_sha(x):
    x = np.asarray(x)
    return hashlib.sha256(str(x.shape).encode() + x.dtype.str.encode() + x.tobytes(order='C')).hexdigest()
def arrays_sha(d): return {k: array_sha(v) for k, v in d.items()}
def values(vs): return [np.asarray(v).copy() for v in vs]
def variables(vs): return [{'path': v.path, 'sha256': array_sha(v), 'shape': list(v.shape)} for v in vs]
def api(f):
    p = Path(inspect.getsourcefile(f))
    return {'qualname': f.__qualname__, 'path': str(p), 'file_sha256': sha(p),
            'method_sha256': hashlib.sha256(inspect.getsource(f).encode()).hexdigest()}

def make_loss_call(estimator):
    """JIT the real BF stateless validation entry point; state is an explicit argument."""
    def call(trainable, nontrainable, metrics, batch):
        loss, (_, updated, _) = estimator.stateless_compute_metrics(
            trainable, nontrainable, metrics, data=batch, stage='validation')
        return loss, updated
    return jax.jit(call)

def loss_epoch(estimator, call, data, seed, batch_size=256):
    trainable, original = values(estimator.trainable_variables), values(estimator.non_trainable_variables)
    metrics = values(estimator.metrics_variables)
    generator = estimator.inference_network.seed_generator.state
    index = next(i for i, v in enumerate(estimator.non_trainable_variables) if v is generator)
    state = list(original); state[index] = np.asarray([seed, 0], dtype=original[index].dtype)
    total = 0.; count = len(next(iter(data.values())))
    for first in range(0, count, batch_size):
        batch = {k: v[first:first + batch_size] for k, v in data.items()}
        loss, state = call(trainable, state, metrics, batch)
        total += float(loss) * len(next(iter(batch.values())))
    # Validation must advance just the FM generator twice per batch: Gaussian base and time.
    expected = [seed, 2 * ((count + batch_size - 1) // batch_size)]
    if np.asarray(state[index]).tolist() != expected: raise ValueError('Unexpected validation RNG consumption')
    for i, (before, after) in enumerate(zip(original, state, strict=True)):
        if i != index and not np.array_equal(before, np.asarray(after)):
            raise ValueError('Validation changed non-RNG model/standardization state')
    if any(not np.array_equal(x, np.asarray(v)) for x, v in zip(original, estimator.non_trainable_variables, strict=True)):
        raise ValueError('Stateless validation mutated the saved live model')
    return total / count

def dry_check():
    """Synthetic CPU replay, including the actual noise/time draws; no fit or financial input."""
    keras.utils.set_random_seed(281000)
    net = bf.networks.FlowMatching(subnet_kwargs={'widths': (8, 8)})
    wf = bf.BasicWorkflow(inference_network=net, standardize=['inference_variables', 'inference_conditions'])
    data = {'inference_variables': np.arange(68, dtype=np.float32).reshape(17, 4) / 100,
            'inference_conditions': np.ones((17, 63), dtype=np.float32)}
    wf.estimator.build_from_data(data)
    wf.estimator.compile(optimizer=None)  # Only construct the framework's validation loss tracker.
    original = values(wf.estimator.non_trainable_variables)
    captured = []; normal, uniform = net.base_distribution.sample, keras.random.uniform
    def base(*args, **kwargs):
        if kwargs.get('seed') is not net.seed_generator: raise ValueError('Base noise uses another RNG')
        x = normal(*args, **kwargs); captured.append(('base', array_sha(x))); return x
    def time_draw(*args, **kwargs):
        if kwargs.get('seed') is not net.seed_generator: raise ValueError('Time uses another RNG')
        x = uniform(*args, **kwargs); captured.append(('time', array_sha(x))); return x
    try:
        net.base_distribution.sample, keras.random.uniform = base, time_draw
        net.seed_generator.state.assign(np.asarray([SEEDS[0], 0], dtype=np.uint32))
        eager = float(wf.estimator.compute_metrics(**data, stage='validation')['loss'])
        first = list(captured); captured.clear()
        net.seed_generator.state.assign(np.asarray([SEEDS[0], 0], dtype=np.uint32))
        replay = float(wf.estimator.compute_metrics(**data, stage='validation')['loss'])
        if first != captured or [x[0] for x in first] != ['base', 'time'] or eager != replay:
            raise ValueError('Noise/time exact replay failed')
    finally:
        net.base_distribution.sample, keras.random.uniform = normal, uniform
        for variable, value in zip(wf.estimator.non_trainable_variables, original, strict=True): variable.assign(value)
    call = make_loss_call(wf.estimator)
    x = loss_epoch(wf.estimator, call, data, SEEDS[0])
    y = loss_epoch(wf.estimator, call, data, SEEDS[0])
    z = loss_epoch(wf.estimator, call, data, SEEDS[1])
    if x != y or x == z or not np.isclose(x, eager, rtol=1e-6, atol=1e-7):
        raise ValueError('Stateless/eager or independent seed check failed')
    # Check a partial tail against the framework's own loss-tracker reduction.
    est = wf.estimator; nontrainable = values(est.non_trainable_variables)
    gi = next(i for i, v in enumerate(est.non_trainable_variables) if v is net.seed_generator.state)
    nontrainable[gi] = np.asarray([SEEDS[0], 0], dtype=np.uint32)
    state = (values(est.trainable_variables), nontrainable, [np.zeros_like(v) for v in est.metrics_variables])
    for first in (0, 10):
        _, state = est.stateless_test_step(state, {k: v[first:first + 10] for k, v in data.items()})
    with keras.StatelessScope(list(zip(est.metrics_variables, state[2], strict=True))):
        framework_reduction = float(est._loss_tracker.result())
    weighted = loss_epoch(est, call, data, SEEDS[0], batch_size=10)
    if not np.isclose(weighted, framework_reduction, rtol=1e-6, atol=1e-7):
        raise ValueError('Framework partial-tail reduction differs')
    print(json.dumps({'synthetic_replay_passed': True, 'eager': eager, 'stateless': x, 'other_seed': z,
                      'captured_rng_calls': captured, 'live_state_preserved': True,
                      'partial_tail_row_weighted_loss': weighted, 'framework_loss_tracker': framework_reduction}))

def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--bank', type=Path); p.add_argument('--out', type=Path); p.add_argument('--dry-check', action='store_true')
    a = p.parse_args(); started = time.perf_counter()
    def stop(signum, frame): raise TimeoutError('270-second read-only audit deadline; no retry')
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop); signal.alarm(270)
    if a.repo: sys.path.insert(0, str(a.repo / 'src'))
    global np, jax, keras, bf
    import numpy as np
    import bayesflow as bf, jax, keras
    if bf.__version__ != '2.1.0' or keras.backend.backend() != 'jax': raise ValueError('Pinned BF2.1/JAX required')
    if a.dry_check: dry_check(); signal.alarm(0); return
    if any(x is None for x in (a.repo, a.checkpoint, a.bank, a.out)): raise ValueError('All audit paths required')
    if (not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 4
            or jax.default_backend() != 'gpu' or len([d for d in jax.devices() if d.platform == 'gpu']) != 1):
        raise ValueError('One 4-CPU/1-GPU Slurm allocation required')
    revision = subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(a.repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != REV or dirty: raise ValueError('Wrong or modified frozen repository revision')
    from sbi4finance.calibration import bank_collection, full_surface
    a.out.mkdir(parents=True, exist_ok=False)
    path = a.out / 'run.json'; report = {'format': 'heston_paired_validation_loss_mc_v1', 'status': 'preflight',
        'scientific_acceptance': False, 'development_audit': True, 'seeds': list(SEEDS), 'replicates': 32,
        'loss_difference_convention': 'final500 - best402', 'completed_pairs': [], 'source_revision': revision,
        'runtime': {'python': platform.python_version(), **{n: metadata.version(n) for n in ('bayesflow', 'keras', 'jax', 'jaxlib', 'numpy')},
                    'devices': [str(d) for d in jax.devices()]}, 'driver_sha256': sha(__file__)}
    save(path, report)
    try:
        resume = json.loads((a.checkpoint / 'resume.json').read_text())
        files = {'best402': a.checkpoint / 'model.keras', 'final500': a.checkpoint / resume['checkpoint_files']['last']}
        if resume['epochs_done'] != 500 or {k: sha(v) for k, v in files.items()} != CHECKPOINT_SHA:
            raise ValueError('Exact selected402/final500 checkpoints required')
        task, selected, training = full_surface.load_checkpoint(a.checkpoint)
        final = full_surface.build_workflow(task, **training['network']); final.load_estimator(files['final500'])
        workflows = {'best402': selected, 'final500': final}
        if any(w.inference_network.use_optimal_transport or w.summary_network is not None for w in workflows.values()):
            raise ValueError('This audit requires the original vector FM path without optimal transport')
        if (sha(a.bank / 'manifest.json') != BANK_SHA or training['run_contract']['dataset']['manifest_sha256'] != BANK_SHA
                or training['selected_epoch'] != 402 or training['network']['family'] != 'flow_matching'
                or training['network']['encoder'] != 'vector'):
            raise ValueError('Wrong validation bank or architecture')
        manifest, splits = bank_collection.load_bank(a.bank, splits=('validation',), roles=('observations', 'parameters', 'example_id'))
        split = splits['validation']
        if manifest['configuration']['task'] != task or split['example_id'].tolist() != [f'validation:{i}' for i in range(5000)]:
            raise ValueError('Exact 5000 validation rows required')
        raw = full_surface.training_data(split, task, clean=False)
        data = {name: {k: np.asarray(v) for k, v in wf.adapter(raw).items()} for name, wf in workflows.items()}
        if arrays_sha(data['best402']) != arrays_sha(data['final500']): raise ValueError('Saved adapters disagree')
        for d in [split, raw, *data.values()]:
            for value in d.values(): np.asarray(value).setflags(write=False)
        live = {name: variables(wf.estimator.variables) for name, wf in workflows.items()}
        frozen = {str(x): sha(x) for x in [*files.values(), a.bank / 'manifest.json', a.checkpoint / 'task.json',
                                          a.checkpoint / 'training.json', a.checkpoint / 'resume.json', Path(__file__)]}
        est = selected.estimator; net = selected.inference_network
        report.update(status='evaluating', checkpoint_sha256=CHECKPOINT_SHA, frozen_sha256=frozen,
            validation_input_sha256=arrays_sha(split), adapted_input_sha256=arrays_sha(data['best402']),
            standardizer_variables={n: [{**item, 'value': np.asarray(v).tolist()}
                for item, v in zip(variables(w.estimator.standardizer.variables), w.estimator.standardizer.variables, strict=True)]
                for n, w in workflows.items()},
            adapter_configs={n: w.adapter.get_config() for n, w in workflows.items()},
            api={k: api(f) for k, f in {'estimator': est.compute_metrics, 'stateless': est.stateless_compute_metrics,
                'framework_test_step': est.stateless_test_step, 'framework_metric_reduction': est._update_metrics,
                'network': net.compute_metrics, 'base_noise': net.base_distribution.sample,
                'uniform_time': keras.random.uniform, 'seed_advance': net.seed_generator.next,
                'standardizer': est.standardizer.maybe_standardize, 'training_data': full_surface.training_data}.items()},
            reduction={'batch_size': 256, 'batch_rows': [256] * 19 + [136], 'row_weighted_mean': True,
                'differences_from_historical_val_loss': ['Canonical fixed row order replaces per-epoch validation shuffle.',
                    '32 independent explicit seeds replace the historical evolving training/validation RNG stream.',
                    'Same BF compute_metrics and row-weighting; Python float64 accumulation may differ at float32 roundoff.'],
                'scope': 'Same fixed noisy validation quotes/targets, no augmentation, stage=validation; model-specific saved standardization.'})
        save(path, report); calls = {n: make_loss_call(w.estimator) for n, w in workflows.items()}
        for i, seed in enumerate(SEEDS):
            # Alternate execution order to avoid consistently charging one role for warmup.
            order = ('best402', 'final500') if i % 2 == 0 else ('final500', 'best402')
            losses = {n: loss_epoch(workflows[n].estimator, calls[n], data[n], seed) for n in order}
            if not all(np.isfinite(v) for v in losses.values()): raise ValueError('Nonfinite validation loss')
            report['completed_pairs'].append({'replicate': i, 'seed': seed, **losses, 'difference': losses['final500'] - losses['best402']})
            report['elapsed_seconds'] = time.perf_counter() - started; save(path, report)
        if any(sha(p) != h for p, h in frozen.items()) or arrays_sha(split) != report['validation_input_sha256']:
            raise ValueError('Frozen model/input bytes changed')
        if any(variables(w.estimator.variables) != live[n] or arrays_sha(data[n]) != report['adapted_input_sha256'] for n, w in workflows.items()):
            raise ValueError('Live model or adapted inputs changed')
        rows = report['completed_pairs']; delta = np.asarray([r['difference'] for r in rows])
        report.update(status='completed', summaries={n: {'mean': float(np.mean([r[n] for r in rows])),
            'sd': float(np.std([r[n] for r in rows], ddof=1))} for n in workflows},
            paired_difference={'mean': float(delta.mean()), 'sd': float(delta.std(ddof=1)),
                               'mcse': float(delta.std(ddof=1) / np.sqrt(32))},
            interpretation_caveat='Post-hoc development comparison. A lower averaged final loss supports noisy selection as a contributor; a higher final loss alongside better posterior diagnostics indicates surrogate/proxy misalignment. Neither establishes a universally optimal checkpoint or independent scientific acceptance.',
            immutable_model_inputs_verified=True, elapsed_seconds=time.perf_counter() - started)
        save(path, report); print(json.dumps({k: report[k] for k in ('status', 'summaries', 'paired_difference', 'elapsed_seconds')}))
    except BaseException as e:
        report.update(status='failed', error_type=type(e).__name__, error=str(e), elapsed_seconds=time.perf_counter() - started)
        save(path, report); raise
    finally: signal.alarm(0)
if __name__ == '__main__': main()

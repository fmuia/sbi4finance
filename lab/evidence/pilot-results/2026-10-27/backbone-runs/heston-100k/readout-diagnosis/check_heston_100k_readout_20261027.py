"""One bounded readout/preprocessing diagnosis; native tutorial policy is unchanged."""
import argparse, hashlib, importlib, inspect, json, os, signal, subprocess, sys, time
from importlib import metadata
from pathlib import Path

DRAW_COUNT = 10000
SEEDS = [271026 + i for i in range(3)]
TASK_SHA = '5e499526ff8f29a72f49785a1a36006b1d769d26bca8a6f83f7a8c8ac4d431fa'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'); temporary.replace(path)
def arrays(path, **values):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **values)
    temporary.replace(path)
def agreement(draws, reference, names):
    result = validation.agreement(draws, reference)
    rows = {name: {'standardized_mean_error': float(result['standardized_mean_error'][i]),
                  'width_ratio_90': float(result['width_ratio'][i]),
                  'mean_error_within_0_3_reference_sd': bool(result['standardized_mean_error'][i] <= .3),
                  'relative_width_error_within_25_percent': bool(abs(result['width_ratio'][i] - 1) <= .25)}
            for i, name in enumerate(names)}
    return {'parameters': rows, 'all_checks_passed': all(
        r['mean_error_within_0_3_reference_sd'] and r['relative_width_error_within_25_percent'] for r in rows.values()),
        'scientific_acceptance': False, 'scope': 'Existing proposed development rules; no numerical posterior stability/product acceptance or quantile MCSE claim.'}
def delta(left, right, reference_sd):
    d = left - right
    width = lambda x: np.diff(np.quantile(x, [.05, .95], axis=0), axis=0)[0]
    return {'signed_mean_delta_over_reference_sd': (d.mean(0) / reference_sd).tolist(),
            'signed_width_delta_over_reference_sd': ((width(left) - width(right)) / reference_sd).tolist(),
            'width_ratio_left_over_right': (width(left) / width(right)).tolist(),
            'paired_draw_rms_delta_over_reference_sd': (np.sqrt(np.mean(d*d, axis=0)) / reference_sd).tolist(),
            'paired_draw_max_abs_delta_over_reference_sd': (np.max(np.abs(d), axis=0) / reference_sd).tolist(),
            'max_abs_physical_delta': float(np.max(np.abs(d)))}
def audit(workflow, task, quotes, parameters, out):
    bounds = np.asarray(task['prior_bounds']); unit = (parameters - bounds[:, 0]) / np.diff(bounds, axis=1)[:, 0]
    expected = {'inference_variables': (np.log(unit) - np.log1p(-unit)).astype('float32'),
                'inference_conditions': (quotes / np.asarray(task['context']['quote_noise_std'])).astype('float32')}
    data = {'observations': quotes / np.asarray(task['context']['quote_noise_std']),
            **{name: parameters[:, i:i+1] for i, name in enumerate(task['parameter_names'])}}
    actual = workflow.adapter(data, strict=False, stage='inference')
    report, payload = {}, {}
    standardizer = workflow.estimator.standardizer
    for key, wanted in expected.items():
        value = np.asarray(actual[key]); layer = standardizer.standardize_layers[key]
        mean, sd, count = np.asarray(layer.moving_mean[0]), np.asarray(layer.moving_std(0)), float(np.asarray(layer.count[0]))
        if not (np.isfinite(mean).all() and np.isfinite(sd).all() and np.all(sd > 0) and count > 0):
            raise ValueError('invalid stored standardization moments')
        normalized = np.asarray(standardizer.maybe_standardize(value, key=key, stage='inference'))
        restored = np.asarray(standardizer.maybe_standardize(normalized, key=key, stage='inference', forward=False))
        manual = (value - mean) / sd
        passed = all(np.allclose(a, b, rtol=2e-6, atol=2e-6) for a, b in
                     ((value, wanted), (normalized, manual), (restored, value)))
        report[key] = {'passed': bool(passed), 'adapter_manual_max_abs_delta': float(np.max(np.abs(value-wanted))),
            'standardization_manual_max_abs_delta': float(np.max(np.abs(normalized-manual))),
            'roundtrip_max_abs_delta': float(np.max(np.abs(restored-value))), 'stored_mean': mean.tolist(),
            'stored_sd': sd.tolist(), 'stored_count': count, 'float32_audit_rtol': 2e-6, 'float32_audit_atol': 2e-6}
        payload.update({key+'_adapted': value, key+'_manual': wanted, key+'_standardized': normalized})
    arrays(out/'preprocessing-audit.npz', **payload)
    if not all(row['passed'] for row in report.values()): raise ValueError('preprocessing audit differs; retained audit arrays')
    return report

def run(args):
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 4:
        raise ValueError('one 4-CPU GPU Slurm allocation required')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('numerical library thread caps must all equal 1')
    started = time.perf_counter(); args.out.mkdir(parents=True, exist_ok=False)
    state = {'format': 'heston_100k_readout_diagnosis_v1', 'status': 'preflight', 'scientific_acceptance': False,
        'native_policy_unchanged': True, 'no_fit_data_pricing_mcmc_or_test': True,
        'internal_wall_seconds': 330, 'outer_slurm_wall_seconds': 360, 'cleanup_seconds': 240,
        'cpus': 4, 'gpus': 1, 'full_limit_plus_cleanup_cpu_seconds': 2400,
        'full_limit_plus_cleanup_gpu_seconds': 600, 'draws_per_case': DRAW_COUNT, 'seeds': SEEDS, 'driver_sha256': sha(__file__),
        'input_sha256': {}, 'cases': [], 'sampling': {'native': {}, 'tighter': {'atol': 1e-7, 'rtol': 1e-5}}}
    path = args.out/'run.json'; write(path, state)
    def stop(signum, frame): raise TimeoutError(f'signal {signum}; preserve outputs, no retry')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGALRM, stop); signal.alarm(330)
    try:
        global np, validation
        import numpy as np
        sys.path[:0] = [str(args.repo/'src'), str(args.repo/'scripts')]
        import jax, keras
        from sbi4finance.calibration import full_surface, validation
        from compare_full_surface_reference import _reference_issues
        devices = jax.devices()
        if len(devices) != 1 or any(d.platform != 'gpu' for d in devices): raise ValueError('GPU backend required')
        state['devices'] = [{'name': str(d), 'kind': d.device_kind, 'platform': d.platform} for d in devices]
        state['python'] = sys.version; state['keras_backend_env'] = os.environ.get('KERAS_BACKEND')
        state['versions'] = {name: metadata.version(name) for name in ('bayesflow','keras','jax','numpy','scipy')}
        state['source_revision'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.repo, text=True).strip()
        state['source_sha256'] = {str(p.relative_to(args.repo)): sha(p) for p in
            [*sorted((args.repo/'src/sbi4finance/calibration').glob('*.py')), args.repo/'scripts/compare_full_surface_reference.py']}
        state['environment_sha256'] = {n: sha(args.repo/n) for n in ('pyproject.toml','pixi.lock')}
        modules = [importlib.import_module('bayesflow.'+n) for n in ('utils.integrate',
          'networks.inference.flow_matching.flow_matching','networks.inference.inference_network',
          'estimators.basic_estimator','networks.helpers.standardization.standardize','adapters.transforms.constrain')]
        state['library_source_sha256'] = {str(Path(m.__file__)): sha(m.__file__) for m in modules}
        integration = modules[0]; code = inspect.getsource(integration.integrate_adaptive)
        if 'kwargs.get("atol", 1e-6)' not in code or 'kwargs.get("rtol", 1e-4 if len(state) == 1 else 1e-5)' not in code:
            raise ValueError('unrecognized native sampling tolerance defaults')
        state['effective_native_sampling'] = {'atol': 1e-6, 'rtol': 1e-4,
            **{n: inspect.signature(integration.integrate).parameters[n].default for n in ('min_steps','max_steps')}}
        for filename in ('task.json','training.json','model.keras'):
            p = args.checkpoint/filename; state['input_sha256'][str(p)] = sha(p)
        task, workflow, training = full_surface.load_checkpoint(args.checkpoint)
        if (training['task_fingerprint'] != TASK_SHA or training['training_surfaces'] != 100000
            or training['network']['family'] != 'flow_matching' or training['network']['encoder'] != 'vector'
            or workflow.inference_network.integrate_kwargs != {'method':'tsit5','steps':'adaptive'}):
            raise ValueError('frozen native 100k Heston checkpoint required')
        for name in ('full_surface.py','tasks.py','validation.py'):
            key = 'src/sbi4finance/calibration/'+name
            if training['run_contract']['source_sha256'][key] != state['source_sha256'][key]:
                raise ValueError('readout core source differs from checkpoint training contract')
        native_policy = dict(workflow.inference_network.integrate_kwargs)
        inputs = []
        for i, seed in enumerate(SEEDS):
            case = args.references/f'case-{i}'; root = case/'reference'; old = args.checkpoint.parent/'comparisons'/f'case-{i}'
            for p in [root/'reference.json',root/'diagnostics.json',root/'chains.npz',case/'precision/precision.json',old/'comparison.json',old/'neural_draws.npz']:
                state['input_sha256'][str(p)] = sha(p)
            reference, diagnostics = (json.loads((root/n).read_text()) for n in ('reference.json','diagnostics.json'))
            previous = json.loads((old/'comparison.json').read_text())
            if (reference['task'] != task or _reference_issues(diagnostics, task['parameter_names'])
                or previous['checkpoint_sha256'] != training['checkpoint_sha256'] or previous['sampling_seed'] != seed
                or previous['neural_draws_requested'] != DRAW_COUNT or previous['observed_prices'] != reference['observed_prices']
                or any(previous['reference_input_sha256'].get(n) != sha(root/n) for n in ('reference.json','diagnostics.json','chains.npz'))):
                raise ValueError('exact existing observations, references and native comparisons required')
            with np.load(root/'chains.npz', allow_pickle=False) as z: chains = z['parameters']
            with np.load(old/'neural_draws.npz', allow_pickle=False) as z: stored = z['parameters']
            if chains.shape != (4,8000,4) or stored.shape != (DRAW_COUNT,4) or not np.isfinite(chains).all() or not np.isfinite(stored).all():
                raise ValueError('existing reference/native draw shape or values differ')
            inputs.append((np.asarray(reference['observed_prices']), chains.reshape(-1,4), stored))
        state['status'] = 'preprocessing_audit'; write(path, state)
        state['preprocessing_audit'] = audit(workflow, task, np.stack([q for q,r,s in inputs]),
            np.stack([r.mean(0) for q,r,s in inputs]), args.out)
        moments_before = [np.array(w) for layer in workflow.estimator.standardizer.standardize_layers.values() for w in layer.weights]
        for i, (quotes, reference, stored) in enumerate(inputs):
            case_out = args.out/f'case-{i}'; case_out.mkdir(); row = {'case': i, 'seed': SEEDS[i], 'status': 'native'}
            state['cases'].append(row); state['status'] = f'case-{i}-native'; write(path,state)
            conditions = full_surface.conditioning_data(quotes[None,:], task)
            def sample(**kwargs):
                draws = workflow.sample(num_samples=DRAW_COUNT, conditions=conditions, sample_shape=(), seed=SEEDS[i], **kwargs)
                result = np.concatenate([np.asarray(draws[n],dtype=float) for n in task['parameter_names']],axis=-1)[0]
                if result.shape != (DRAW_COUNT,4) or not np.isfinite(result).all(): raise ValueError('invalid sampled draws')
                return result
            native = sample(); arrays(case_out/'native.npz', parameters=native)
            row['native_agreement'] = agreement(native,reference,task['parameter_names'])
            sd = reference.std(0)
            row['native_vs_stored'] = {'array_equal': bool(np.array_equal(native,stored)),
                'allclose_rtol_1e_6_atol_1e_8': bool(np.allclose(native,stored,rtol=1e-6,atol=1e-8)),
                **delta(native,stored,sd), 'scope':'Cross-host equality is descriptive, not required.'}
            row['status'] = 'tighter'; state['status'] = f'case-{i}-tighter'; write(path,state)
            tight = sample(atol=1e-7, rtol=1e-5); arrays(case_out/'tighter.npz', parameters=tight)
            row.update(status='completed', tighter_agreement=agreement(tight,reference,task['parameter_names']),
                       tighter_minus_native=delta(tight,native,sd))
            write(case_out/'comparison.json',row); write(path,state)
        moments_after = [np.array(w) for layer in workflow.estimator.standardizer.standardize_layers.values() for w in layer.weights]
        if workflow.inference_network.integrate_kwargs != native_policy or any(not np.array_equal(a,b) for a,b in zip(moments_before,moments_after)):
            raise ValueError('native policy or stored standardization moments mutated')
        if any(sha(p)!=h for p,h in state['input_sha256'].items()) or sha(__file__) != state['driver_sha256']:
            raise ValueError('frozen inputs or driver changed')
        if (any(sha(args.repo/n)!=h for n,h in state['source_sha256'].items())
            or any(sha(args.repo/n)!=h for n,h in state['environment_sha256'].items())
            or any(sha(n)!=h for n,h in state['library_source_sha256'].items())):
            raise ValueError('frozen source/environment changed')
        state.update(status='completed_development_diagnosis',seconds=time.perf_counter()-started,
            interpretation='Numerical readout and preprocessing diagnosis only. Native tutorial policy remains unchanged; no new fit, data, posterior reference or scientific acceptance.')
        write(path,state); return state
    except BaseException as error:
        state.update(status='incomplete' if isinstance(error,(TimeoutError,SystemExit)) else 'failed',
                     seconds=time.perf_counter()-started,failure={'type':type(error).__name__,'error':str(error)})
        write(path,state); raise
    finally: signal.alarm(0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','checkpoint','references','out'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for name in ('repo','checkpoint','references','out'): setattr(args,name,getattr(args,name).resolve())
    state=run(args); print(json.dumps({'status':state['status'],'scientific_acceptance':False,'native_policy_unchanged':True}))
if __name__ == '__main__': main()

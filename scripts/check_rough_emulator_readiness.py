"""Fixed-point emulator checks; reuse price batches and never fit a posterior.

The differentiated target is the existing mean-emulator likelihood with its
physical uniform prior transformed to logits. Pricing-label covariance is
reported separately and is deliberately not inserted into that likelihood.
"""
import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.special import erf, expit
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration import correction, pricer_net, tasks
from sbi4finance.gbm_bank import _json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_consumer(path):
    """Load an immutable evaluator separately; keep numerical imports intact."""
    spec=importlib.util.spec_from_file_location('sbi4finance.calibration.readiness_pricer_consumer',path)
    consumer=importlib.util.module_from_spec(spec);spec.loader.exec_module(consumer)
    return consumer


def load_bound_pricer(directory, task, *, consumer=None):
    """The current reference loader's binding checks, without its newer API.

    The frozen numerical checkout predates reference.load_forward_pricer.
    The evaluator comes from the checkpoint's separately frozen consumer.
    """
    tasks.validate_task(task)
    if task['model']!='rough': raise ValueError('A rough task is required')
    directory=Path(directory).resolve()
    paths={name:directory/name for name in ('pricer.npz','pricer.json','task.json')}
    hashes={name:sha(path) for name,path in paths.items()}
    record=json.loads(paths['pricer.json'].read_text())
    fingerprint=hashlib.sha256(_json(task).encode()).hexdigest()
    if (json.loads(paths['task.json'].read_text())!=task or record.get('task_fingerprint')!=fingerprint
            or record.get('pricer_sha256')!=hashes['pricer.npz']):
        raise ValueError('Forward-pricer task/fingerprint/checkpoint binding differs')
    consumer=pricer_net if consumer is None else consumer
    pricer=consumer.Pricer.load(directory)
    return pricer,{'directory':str(directory),'task_fingerprint':fingerprint,
        'input_sha256':{str(paths[name]):digest for name,digest in hashes.items()},
        'activation':pricer.activation,'training_record':record,
        'consumer_source_sha256':sha(consumer.__file__)}


def value_and_jacobian(pricer, x):
    """Independent float64 chain rule for the exact saved MLP activation."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.shape != pricer.x_mean.shape or not np.isfinite(x).all():
        raise ValueError("one finite feature vector is required")
    h = (x - pricer.x_mean) / pricer.x_sd
    jacobian = np.diag(1.0 / pricer.x_sd)
    for w, b in pricer.weights[:-1]:
        a = h @ w + b
        jacobian = jacobian @ w
        if pricer.activation == "gelu_erf":
            derivative = .5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-.5*a*a) / np.sqrt(2*np.pi)
            h = .5 * a * (1 + erf(a / np.sqrt(2)))
        else:
            u = .7978845608028654 * (a + .044715*a**3)
            tangent = np.tanh(u)
            derivative = .5 * (1+tangent) + .5*a*(1-tangent*tangent)*.7978845608028654*(1+3*.044715*a*a)
            h = .5 * a * (1+tangent)
        jacobian *= derivative
    w, b = pricer.weights[-1]
    return (h @ w + b) * pricer.y_sd + pricer.y_mean, (jacobian @ w) * pricer.y_sd


def target_and_gradient(pricer, task, point, z):
    z = np.asarray(z, dtype=np.float64)
    bounds = correction.bounds_of(task)
    known = tasks.context_size(task) and not tasks.is_joint_curve_task(task)
    x = np.r_[z, np.log(point['curve_levels'])] if known else z
    values, jacobian = value_and_jacobian(pricer, x)
    residual = values - np.asarray(point['observed_prices']) / tasks.sigma(task)
    value = -.5 * (residual @ residual) + float(correction.log_jacobian(z[None], bounds)[0])
    gradient = -(jacobian[:len(z)] @ residual) + 1 - 2*expit(z)
    return value, gradient, values


def pytensor_target(pricer, task, point):
    """A native differentiable expression; no custom gradient Op or sampler."""
    import pytensor
    import pytensor.tensor as pt
    z = pt.dvector('z')
    known = tasks.context_size(task) and not tasks.is_joint_curve_task(task)
    x = pt.concatenate([z, np.log(point['curve_levels'])]) if known else z
    h = (x - np.asarray(pricer.x_mean, dtype=np.float64)) / np.asarray(pricer.x_sd, dtype=np.float64)
    for w, b in pricer.weights[:-1]:
        a = pt.dot(h, np.asarray(w, dtype=np.float64)) + np.asarray(b, dtype=np.float64)
        h = (.5*a*(1+pt.erf(a/np.sqrt(2))) if pricer.activation == 'gelu_erf'
             else .5*a*(1+pt.tanh(.7978845608028654*(a+.044715*a**3))))
    w, b = pricer.weights[-1]
    prediction = (pt.dot(h, np.asarray(w, dtype=np.float64)) + np.asarray(b, dtype=np.float64)) * pricer.y_sd + pricer.y_mean
    residual = prediction - np.asarray(point['observed_prices']) / tasks.sigma(task)
    widths = np.diff(correction.bounds_of(task), axis=1).reshape(-1)
    target = -.5 * pt.sum(residual**2) + pt.sum(np.log(widths)-pt.softplus(-z)-pt.softplus(z))
    return pytensor.function([z], [target, pt.grad(target, z), prediction])


def check_gradient(pricer, task, point, *, check_pytensor=False, consumer=None):
    theta = np.asarray(point['theta'], dtype=float)
    bounds = correction.bounds_of(task)
    z = correction.to_unbounded(theta[None], bounds)[0]
    value, gradient, prediction = target_and_gradient(pricer, task, point, z)
    consumer=pricer_net if consumer is None else consumer
    original = consumer.log_posterior(pricer, task, point['observed_prices'],
                                       point.get('curve_levels'))
    original_value = float(original(z)[0])
    finite = []
    for epsilon in (1e-3, 1e-4, 1e-5):
        differences = []
        for i in range(len(z)):
            delta = np.zeros_like(z); delta[i] = epsilon
            differences.append((float(original(z+delta)[0])-float(original(z-delta)[0]))/(2*epsilon))
        differences = np.asarray(differences)
        finite.append({'step': epsilon, 'gradient': differences.tolist(),
                       'max_scaled_error': float(np.max(np.abs(differences-gradient)/(1+np.abs(gradient))))})
    result = {'case': point['case'], 'role': point['role'], 'theta': theta.tolist(),
              'prior_fraction': ((theta-bounds[:,0])/(bounds[:,1]-bounds[:,0])).tolist(),
              'target_value': value, 'original_numpy_target_value': original_value,
              'gradient_logit': gradient.tolist(), 'finite_difference_checks': finite,
              'value_parity_passed': bool(np.isclose(value, original_value, rtol=1e-12, atol=1e-9)),
              'finite_difference_passed': all(r['max_scaled_error'] <= 5e-6 for r in finite[1:])}
    if check_pytensor:
        symbolic = pytensor_target(pricer, task, point)
        other_value, other_gradient, other_prices = symbolic(z)
        result['pytensor'] = {'target_value': float(other_value), 'gradient_logit': other_gradient.tolist(),
                              'max_scaled_gradient_error': float(np.max(np.abs(other_gradient-gradient)/(1+np.abs(gradient)))),
                              'max_price_error_quote_sigma': float(np.max(np.abs(other_prices-prediction)))}
        result['pytensor']['passed'] = bool(np.isclose(value, other_value, rtol=1e-11, atol=1e-8)
            and result['pytensor']['max_scaled_gradient_error'] <= 1e-7
            and result['pytensor']['max_price_error_quote_sigma'] <= 1e-7)
    return result


def interval(rows):
    rows = np.asarray(rows)
    mean = rows.mean(axis=0)
    se = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
    radius = t.ppf(.975, len(rows)-1) * se
    return {'mean': mean.tolist(), 'mcse': se.tolist(), 'pointwise_t95_lower': (mean-radius).tolist(),
            'pointwise_t95_upper': (mean+radius).tolist(), 'degrees_of_freedom': len(rows)-1}


def rescore_batches(pricer, plan, controlled, *, consumer=None):
    controlled = np.asarray(controlled)
    if controlled.shape != (len(plan['points']), 8, 2, 63) or not np.isfinite(controlled).all():
        raise ValueError('exact fixed-point ×8batch ×2grid ×63price panel required')
    task = plan['task']; sigma = tasks.sigma(task)
    consumer=pricer_net if consumer is None else consumer
    result = []
    for point, batches in zip(plan['points'], controlled):
        prediction = pricer(consumer.features(task, point['theta'], point.get('curve_levels')))[0]
        errors = prediction[None,None,:] - batches/sigma
        old_errors = np.asarray(point['emulator_scaled_prices'])[None,None,:] - batches/sigma
        result.append({'case': point['case'], 'role': point['role'], 'theta': point['theta'],
            'new_emulator_minus_controlled_quote_sigma': interval(errors),
            'new_rms_error_by_grid': np.sqrt(np.mean(errors.mean(0)**2, axis=-1)).tolist(),
            'old_rms_error_by_grid': np.sqrt(np.mean(old_errors.mean(0)**2, axis=-1)).tolist(),
            'new_max_abs_error_by_grid': np.max(np.abs(errors.mean(0)), axis=-1).tolist(),
            'replicate_mean_price_covariance_quote_units_by_grid': [
                (np.cov(batches[:,g,:]/sigma, rowvar=False, ddof=1)/8).tolist() for g in (0,1)],
            'fine_minus_coarse_quote_sigma': interval((batches[:,1]-batches[:,0])/sigma)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pricer', type=Path, required=True); p.add_argument('--points', type=Path, required=True)
    p.add_argument('--batch-summaries', type=Path); p.add_argument('--pytensor', action='store_true')
    p.add_argument('--consumer-file',type=Path)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    plan = json.loads(a.points.read_text()); task = plan['task']
    consumer=load_consumer(a.consumer_file) if a.consumer_file else pricer_net
    pricer, identity = load_bound_pricer(a.pricer, task, consumer=consumer)
    report = {'format': 'rough_emulator_readiness_v1', 'scientific_acceptance': False,
        'reserved_test_evaluated': False, 'mcmc_run': False, 'new_pricing_paths': 0,
        'pricer': identity, 'inputs_sha256': {'points': sha(a.points), 'script': sha(__file__)},
        'runtime': {'python': platform.python_version(), 'system': platform.system(),
                    'machine': platform.machine(), 'numpy': np.__version__, 'scipy': metadata.version('scipy'),
                    'pytensor': metadata.version('pytensor') if a.pytensor else None},
        'target': 'existing deterministic mean-emulator Gaussian quote likelihood plus exact prior-box logit Jacobian',
        'gradient_checks': [check_gradient(pricer, task, q, check_pytensor=a.pytensor,consumer=consumer) for q in plan['points']],
        'limitations': ['Fixed-point checks are not posterior accuracy or prior-wide acceptance.',
            'Rank7 eight-batch covariance is retained; it is not inverted or inserted into a likelihood.',
            'MC label randomness, mean-emulator approximation and grid shifts remain separate errors.',
            'Existing PyMC reference CLI uses DEMetropolisZ; this check does not change its sampler or add NUTS.']}
    if a.batch_summaries:
        report['inputs_sha256']['batch_summaries'] = sha(a.batch_summaries)
        with np.load(a.batch_summaries, allow_pickle=False) as archive:
            report['reused_price_panel'] = rescore_batches(pricer, plan, archive['controlled'],consumer=consumer)
    report['all_gradient_checks_passed'] = all(q['value_parity_passed'] and q['finite_difference_passed']
        and (not a.pytensor or q['pytensor']['passed']) for q in report['gradient_checks'])
    report['seconds'] = time.perf_counter()-started
    (a.out/'readiness.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'all_gradient_checks_passed': report['all_gradient_checks_passed'],
                      'reused_points': len(report.get('reused_price_panel', [])), 'seconds': report['seconds']}))
    if not report['all_gradient_checks_passed']: raise SystemExit(1)


if __name__ == '__main__': main()

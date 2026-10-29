"""Bounded price-level challenge to the exponential initial-variance curve.

This is a development screen, not a posterior comparison or an acceptance
test for either curve family. The published alternative is
xi(t)=b0+b1 exp(-t/tau1)+b2 (t/tau2) exp(-t/tau2).
"""

import copy
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from ..gbm_bank import _json, _sha256
from . import bank, pricing, tasks


def _coefficients(coefficients):
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all() or np.any(values[3:] <= 0):
        raise ValueError("five finite coefficients with positive tau1/tau2 required")
    return values


def hump_values(times, coefficients):
    """Evaluate the published family; positivity is checked separately on a horizon."""
    b0, b1, b2, tau1, tau2 = _coefficients(coefficients)
    t = np.asarray(times, dtype=np.float64)
    if not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("nonnegative finite times required")
    return b0 + b1 * np.exp(-t / tau1) + b2 * (t / tau2) * np.exp(-t / tau2)


def hump_integrated_variance(times, coefficients):
    """Analytic integral, using a series for the small-time hump contribution."""
    b0, b1, b2, tau1, tau2 = _coefficients(coefficients)
    t = np.asarray(times, dtype=np.float64)
    if not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("nonnegative finite times required")
    x = t / tau2
    tail = -np.expm1(-x) - x * np.exp(-x)
    series = x*x * (0.5 + x * (-1/3 + x * (1/8 + x * (-1/30 + x/144))))
    tail = np.where(x < 1e-3, series, tail)
    return b0*t - b1*tau1*np.expm1(-t/tau1) + b2*tau2*tail


def positive_hump_minimum(coefficients, horizon):
    """Check endpoints and all interior stationary points, not just a grid.

    After division by exp(-t/tau2), the derivative is a linear function minus
    one exponential. Its derivative has at most one zero; splitting there
    brackets all possible stationary points of xi on [0,horizon].
    """
    b0, b1, b2, tau1, tau2 = _coefficients(coefficients)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("positive finite horizon required")
    a = 1/tau2 - 1/tau1
    partitions = [0.0, float(horizon)]
    if a != 0 and b1 != 0 and b2 != 0 and np.sign(-b2) == np.sign(a*b1):
        log_ratio = np.log(abs(b2)) + np.log(tau1) - np.log(abs(a)) - np.log(abs(b1)) - 2*np.log(tau2)
        turning = log_ratio / a
        if 0 < turning < horizon:
            partitions.append(float(turning))
    partitions.sort()

    def derivative(t):
        # Normalize signed terms in log space. The ordinary derivative can
        # underflow to zero at the far endpoint and hide an interior dip.
        terms = []
        if b1 != 0:
            terms.append((-np.sign(b1), np.log(abs(b1))-np.log(tau1)-t/tau1))
        linear = 1-t/tau2
        if not np.isfinite(linear):
            raise ValueError("curve timescales are numerically unresolved on the declared horizon")
        if b2 != 0 and linear != 0:
            terms.append((np.sign(b2)*np.sign(linear),
                          np.log(abs(b2))-np.log(tau2)+np.log(abs(linear))-t/tau2))
        if not terms:
            return 0.0
        largest = max(value for _, value in terms)
        if not np.isfinite(largest):
            raise ValueError("curve derivative is numerically unresolved on the declared horizon")
        return float(sum(sign*np.exp(value-largest) for sign, value in terms))

    candidates = list(partitions)
    if b1 == 0 and b2 != 0 and 0 < tau2 < horizon:
        candidates.append(float(tau2))
    for lo, hi in zip(partitions[:-1], partitions[1:]):
        left, right = derivative(lo), derivative(hi)
        if (left < 0 < right) or (right < 0 < left):
            candidates.append(float(brentq(derivative, lo, hi, xtol=1e-13)))
    values = hump_values(np.asarray(candidates), coefficients)
    minimum = float(values.min())
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError(f"curve is not positive throughout [0,{horizon}]; minimum {minimum:.6g}")
    return minimum


def _nodes(blocks):
    return np.unique(np.concatenate([b["horizon"] / b["n_steps"] * np.arange(b["n_steps"] + 1)
                                     for b in blocks]))


def hump_curve(coefficients, blocks):
    horizon = max(b["horizon"] for b in blocks)
    positive_hump_minimum(coefficients, horizon)
    nodes = _nodes(blocks)
    return {"times": nodes.tolist(), "values": hump_values(nodes, coefficients).tolist()}


def panel():
    """Three named controls/challenges, declared before observing method rankings."""
    common = {"hurst": 0.12, "eta": 1.5, "rho": -0.7}
    return [
        {"name": "flat", "rough_parameters": common.copy(),
         "curve_coefficients": [0.04, 0.0, 0.0, 0.5, 0.3]},
        {"name": "monotone", "rough_parameters": common.copy(),
         "curve_coefficients": [0.025, 0.045, 0.0, 0.5, 0.3]},
        {"name": "hump", "rough_parameters": common.copy(),
         "curve_coefficients": [0.035, -0.015, 0.12, 0.25, 0.3]},
    ]


def seed_for(root_seed, case_name, purpose, replicate=0):
    """Stable semantic streams; case scheduling does not choose its randomness."""
    if type(root_seed) is not int or root_seed < 0 or type(replicate) is not int or replicate < 0:
        raise ValueError("nonnegative integer root seed and replicate required")
    digest = hashlib.sha256(f"{case_name}:{purpose}".encode()).digest()
    words = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
    return int(np.random.SeedSequence([0x43555256, root_seed, *words, replicate]).generate_state(1, dtype=np.uint64)[0])


def price_curve(task, rough_parameters, curve, paths, seed):
    """Fixed path count for bounded computation; no error-budget acceptance implied."""
    if type(paths) is not int or paths < 64 or paths % 2:
        raise ValueError("an even fixed path count of at least 64 required")
    contracts = task["context"]["contracts"]
    return pricing.rough_surface(rough_parameters, curve, contracts, task["numerics"]["blocks"],
                                 [paths], tasks.sigma(task), np.inf, seed, cap=np.inf)


def _product_values(total_variance):
    return {"variance_swap_1y": float(total_variance(1.0)),
            "forward_variance_0.5_to_1y": float((total_variance(1.0)-total_variance(0.5))/0.5)}


def _replay_observations(task, case, source, seed):
    """Reuse an exact development observation; only parameter bounds may change."""
    if (not isinstance(source, dict) or source.get("format") != "curve_family_price_challenge_v1"
            or source.get("status") != "completed_price_screen" or source.get("case") != case
            or source.get("root_seed") != seed):
        raise ValueError("replay requires a completed report with the same truth case and root seed")
    if ({k: v for k, v in task.items() if k != "prior_bounds"}
            != {k: v for k, v in source["task"].items() if k != "prior_bounds"}):
        raise ValueError("replay must preserve the exact observation design, noise and numerical grid")
    observed = np.asarray(source.get("observations"), dtype=np.float64)
    if observed.shape != (63,) or not np.isfinite(observed).all():
        raise ValueError("replay requires the original complete finite 63-price observation")
    return observed


def run_case(task, case, *, seed=271026, fit_paths=2048, audit_paths=32768,
             fit_evaluations=80, audit_replicates=2, grid_factor=1,
             replay_report=None, initial_parameters=None, audit_namespace=""):
    """Fit all six baseline parameters, then audit with new independent prices.

    Fitting reuses one fixed Monte Carlo stream for a deterministic objective.
    The separate truth and fitted-price audit streams are mutually independent
    and absent from optimization. Finite fitting effort is never proof of the
    global best achievable quote fit.
    """
    from .full_surface import validate_task
    validate_task(task)
    if not tasks.is_joint_curve_task(task):
        raise ValueError("the joint rough baseline task is required")
    for value, name in ((fit_evaluations, "fit_evaluations"), (audit_replicates, "audit_replicates"),
                        (grid_factor, "grid_factor")):
        if type(value) is not int or value < 1:
            raise ValueError(f"positive integer {name} required")
    task = copy.deepcopy(task)
    for block in task["numerics"]["blocks"]:
        block["n_steps"] *= grid_factor
    if not isinstance(audit_namespace, str) or len(audit_namespace) > 128:
        raise ValueError("audit_namespace must be a string of at most 128 characters")
    replayed = None if replay_report is None else _replay_observations(task, case, replay_report, seed)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    supplied_initial = None
    if initial_parameters is not None:
        supplied_initial = np.asarray(initial_parameters, dtype=np.float64)
        if supplied_initial.shape != (6,) or not np.isfinite(supplied_initial).all():
            raise ValueError("six finite physical initial parameters required")
        initial_unit = (supplied_initial-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])
        if np.any(initial_unit < 1e-6) or np.any(initial_unit > 1-1e-6):
            raise ValueError("initial parameters must lie inside the finite numerical optimizer bounds")
    coefficients = _coefficients(case["curve_coefficients"])
    minimum = positive_hump_minimum(coefficients, 1.0)
    target_curve = hump_curve(coefficients, task["numerics"]["blocks"])
    sigma = tasks.sigma(task)
    name = case["name"]
    started, cpu_started = time.perf_counter(), time.process_time()
    record = {"format": "curve_family_price_challenge_v1", "case": case, "task": task,
              "root_seed": seed, "curve_minimum_on_0_1y": minimum,
              "versions": bank.versions(),
              "settings": {"fit_paths": fit_paths, "audit_paths": audit_paths,
                           "fit_evaluations": fit_evaluations, "audit_replicates": audit_replicates,
                           "grid_factor": grid_factor, "audit_namespace": audit_namespace},
              "scope": "bounded price-fit challenge only; no posterior or product-uncertainty acceptance",
              "status": "started"}
    prefix = f"{audit_namespace}:" if audit_namespace else ""
    streams = {"target_label": seed_for(seed, name, "target_label"),
               "quote_noise": seed_for(seed, name, "quote_noise"),
               "fit_objective": seed_for(seed, name, "fit_objective"),
               "audit_truth": [seed_for(seed, name, prefix+"audit_truth", r) for r in range(audit_replicates)],
               "audit_fit": [seed_for(seed, name, prefix+"audit_fit", r) for r in range(audit_replicates)]}
    if replay_report is not None:
        streams["target_label"] = replay_report["streams"]["target_label"]
        streams["quote_noise"] = replay_report["streams"]["quote_noise"]
        record["replay"] = {"source_report_canonical_sha256": hashlib.sha256(_json(replay_report).encode()).hexdigest(),
                            "observation_vector_exactly_reused": True,
                            "source_target_label_paths": replay_report["settings"]["audit_paths"],
                            "source_prior_bounds": replay_report["task"]["prior_bounds"],
                            "parameter_bounds_changed": task["prior_bounds"] != replay_report["task"]["prior_bounds"],
                            "limitations": "changed normalized bounds also change optimizer geometry; no global optimum claim"}
    record["streams"] = streams
    try:
        if replayed is None:
            target, target_se, _, _ = price_curve(task, case["rough_parameters"], target_curve,
                                                  audit_paths, streams["target_label"])
            y = target + sigma*np.random.default_rng(streams["quote_noise"]).standard_normal(len(target))
            record["target_label_max_mc_se_over_quote_sigma"] = float(np.max(target_se/sigma))
        else:
            y = replayed.copy()
            record["target_label_max_mc_se_over_quote_sigma"] = replay_report["target_label_max_mc_se_over_quote_sigma"]
        record["observations"] = y.tolist()

        if supplied_initial is None:
            times = tasks.knots(task)
            projection = least_squares(lambda x: pricing.heston_variance_values(times, *x)-hump_values(times, coefficients),
                                       np.mean(bounds[3:], axis=1), bounds=(bounds[3:, 0], bounds[3:, 1]))
            initial = np.r_[[case["rough_parameters"][n] for n in ("hurst", "eta", "rho")], projection.x]
            unit = np.clip((initial-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0]), 1e-5, 1-1e-5)
            initial_source = "curve projection plus generating rough parameters"
        else:
            unit = (supplied_initial-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])
            initial_source = "supplied physical parameters"
        record["initial_parameters"] = (bounds[:, 0]+unit*(bounds[:, 1]-bounds[:, 0])).tolist()
        record["initial_source"] = initial_source
        evaluation_count = 0
        best = {"rms": np.inf}

        class EvaluationLimit(Exception):
            pass

        def residual(unit_parameters):
            nonlocal evaluation_count, best
            if evaluation_count >= fit_evaluations:
                raise EvaluationLimit
            evaluation_count += 1
            row = bounds[:, 0] + unit_parameters*(bounds[:, 1]-bounds[:, 0])
            record["fit_progress"] = {"price_evaluations": evaluation_count, "last_parameters": row.tolist()}
            predicted, se, _, _ = price_curve(task, tasks.model_parameters(task, row),
                                               tasks.joint_curve(task, row), fit_paths, streams["fit_objective"])
            error = (predicted-y)/sigma
            rms = float(np.sqrt(np.mean(error**2)))
            if rms < best["rms"]:
                best = {"rms": rms, "parameters": row.copy(), "max_mc_se_over_quote_sigma": float(np.max(se/sigma))}
            return error

        try:
            fitted = least_squares(residual, unit, bounds=(np.full(6, 1e-6), np.full(6, 1-1e-6)),
                                   diff_step=1e-3, max_nfev=fit_evaluations, ftol=1e-5, xtol=1e-5, gtol=1e-5)
            optimizer = {"terminated": bool(fitted.success), "message": str(fitted.message), "status": int(fitted.status)}
        except EvaluationLimit:
            optimizer = {"terminated": False, "message": "explicit price-evaluation budget exhausted", "status": None}
        row = best["parameters"]
        record["fit"] = {"optimizer": optimizer, "price_evaluations": evaluation_count,
                         "parameters": row.tolist(), "objective_rms_over_quote_sigma": best["rms"],
                         "objective_max_mc_se_over_quote_sigma": best["max_mc_se_over_quote_sigma"],
                         "warning": "fixed low-path Monte Carlo objective; convergence is not global-optimality evidence"}
        differences, covariances, audit_details = [], [], []
        for replicate in range(audit_replicates):
            true_price, true_se, true_cov, _ = price_curve(task, case["rough_parameters"], target_curve,
                                                          audit_paths, streams["audit_truth"][replicate])
            fit_price, fit_se, fit_cov, _ = price_curve(task, tasks.model_parameters(task, row),
                                                       tasks.joint_curve(task, row), audit_paths,
                                                       streams["audit_fit"][replicate])
            differences.append(fit_price-true_price)
            covariances.append(fit_cov+true_cov)
            audit_details.append({"replicate": replicate, "truth_max_mc_se_over_quote_sigma": float(np.max(true_se/sigma)),
                                  "fit_max_mc_se_over_quote_sigma": float(np.max(fit_se/sigma))})
        difference = np.mean(differences, axis=0)
        covariance = np.sum(covariances, axis=0)/audit_replicates**2
        difference_se = np.sqrt(np.diag(covariance))
        z = norm.ppf(1-0.05/(2*len(sigma)))
        upper = (np.abs(difference)+z*difference_se)/sigma
        lower = np.maximum(0, np.abs(difference)-z*difference_se)/sigma
        record["audit"] = {"replicates": audit_details, "mean_price_difference": difference.tolist(),
                           "difference_covariance": covariance.tolist(),
                           "difference_se_over_quote_sigma": (difference_se/sigma).tolist(),
                           "rms_price_difference_over_quote_sigma": float(np.sqrt(np.mean((difference/sigma)**2))),
                           "max_price_difference_over_quote_sigma": float(np.max(np.abs(difference)/sigma)),
                           "max_abs_difference_lower_over_quote_sigma": float(lower.max()),
                           "max_abs_difference_upper_over_quote_sigma": float(upper.max()),
                           "band": "approximate 95% familywise normal/Bonferroni band over 63 contracts; MC covariance estimated",
                           "interpretation": "independent price discrepancy, not a posterior or universal model-adequacy verdict"}
        truth_product = _product_values(lambda t: hump_integrated_variance(t, coefficients))
        fit_product = _product_values(lambda t: pricing.heston_integrated_variance(t, *row[3:]))
        record["plug_in_products"] = {"truth": truth_product, "fitted": fit_product,
                                      "scope": "curve-derived point values only; posterior/product uncertainty remains untested"}
        record["status"] = "completed_price_screen"
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        record.update(status="inconclusive_failure", error=str(error))
    record["timing_seconds"] = {"wall": time.perf_counter()-started, "cpu": time.process_time()-cpu_started}
    record["source_sha256"] = {**bank.source_hashes(task), "calibration/curve_adequacy.py": _sha256(Path(__file__))}
    record["task_fingerprint"] = hashlib.sha256(_json(task).encode()).hexdigest()
    record["scientific_acceptance"] = False
    return record

"""Predeclared uncertainty checks for the bounded Heston training study.

These checks compare estimators on fixed synthetic cases. They do not establish
MCMC convergence, numerical target accuracy, product accuracy or the posterior
copula. Chain/draw structure is retained for Monte Carlo uncertainty; observations
are the resampling unit for the separate final-panel comparison.
"""

import math
import hashlib
import json

import numpy as np
from scipy.stats import norm


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value, name):
    if np.iscomplexobj(value) or (np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value))):
        raise ValueError(f"{name} must contain finite unmasked real values")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _status(interval, limits):
    lower, upper = interval
    low_limit, high_limit = limits
    if lower >= low_limit and upper <= high_limit:
        return "pass"
    if upper < low_limit or lower > high_limit:
        return "fail"
    return "inconclusive"


def _combined_status(statuses):
    if "fail" in statuses:
        return "fail"
    return "pass" if all(status == "pass" for status in statuses) else "inconclusive"


def _summaries(values):
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    quantiles = np.quantile(values, [.05, .95], axis=0)
    width = quantiles[1] - quantiles[0]
    if np.any(sd <= 0) or np.any(width <= 0):
        raise ValueError("nondegenerate posterior SD and central 90% width required for every parameter")
    return mean, sd, quantiles, width


def _conditional_bootstrap(neural, chains, *, length, replicates, rng):
    """Bound memory by resampling one replicate at a time, preserving paired targets."""
    count, draws, parameters = chains.shape
    offsets = np.arange(length)
    blocks = math.ceil(draws / length)
    reference_means = np.empty((replicates, parameters))
    reference_sds = np.empty_like(reference_means)
    reference_quantiles = np.empty((replicates, 2, parameters))
    neural_means = np.empty_like(reference_means)
    neural_quantiles = np.empty_like(reference_quantiles)
    for replicate in range(replicates):
        # Circular blocks are sampled independently within each chain; neither
        # chains nor individual parameters are pooled before selecting blocks.
        starts = rng.integers(draws, size=(count, blocks))
        indices = ((starts[..., None] + offsets) % draws).reshape(count, -1)[:, :draws]
        reference = chains[np.arange(count)[:, None], indices].reshape(-1, parameters)
        simulated = neural[rng.integers(len(neural), size=len(neural))]
        ref_mean, ref_sd, ref_quantile, _ = _summaries(reference)
        net_mean, _, net_quantile, _ = _summaries(simulated)
        reference_means[replicate] = ref_mean
        reference_sds[replicate] = ref_sd
        reference_quantiles[replicate] = ref_quantile
        neural_means[replicate] = net_mean
        neural_quantiles[replicate] = net_quantile
    mean_difference = (neural_means - reference_means) / reference_sds
    log_width_ratio = np.log((neural_quantiles[:, 1] - neural_quantiles[:, 0]) /
                            (reference_quantiles[:, 1] - reference_quantiles[:, 0]))
    return {
        "block_length": length,
        "reference_mean_mcse": reference_means.std(axis=0, ddof=1),
        "reference_sd_mcse": reference_sds.std(axis=0, ddof=1),
        "reference_quantile_mcse": reference_quantiles.std(axis=0, ddof=1),
        "neural_mean_mcse": neural_means.std(axis=0, ddof=1),
        "neural_quantile_mcse": neural_quantiles.std(axis=0, ddof=1),
        "standardized_mean_mcse": mean_difference.std(axis=0, ddof=1),
        "log_width_ratio_mcse": log_width_ratio.std(axis=0, ddof=1),
    }


def conditional_agreement(neural, chains, diagnostics, names, *, seed, replicates=400,
                          family_checks=24, mean_limit=.3, width_limits=(.75, 1.25)):
    """Compare means and central widths with chain-aware, simultaneous uncertainty.

    ``neural`` is [draw, parameter], ``chains`` is [chain, draw, parameter].
    Diagnostics are the saved reference dictionary with ``parameters`` entries
    keyed by a name or ``parameters[name]``. Bulk/tail ESS and mean MCSE must be
    supplied; this function never estimates their values from flattened chains.

    Two circular-block lengths are checked. Each chain is resampled separately,
    preserving paired parameter rows, and neural rows are resampled IID. The
    larger MCSE is used. Normal intervals with a Bonferroni adjustment over the
    predeclared family are approximate, not exact simultaneous guarantees.
    """
    seed = _integer(seed, "seed", 0)
    replicates = _integer(replicates, "replicates", 100)
    family_checks = _integer(family_checks, "family_checks", 1)
    names = list(names)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("unique nonempty parameter names required")
    neural, chains = _array(neural, "neural draws"), _array(chains, "reference chains")
    parameters = len(names)
    if (neural.ndim != 2 or neural.shape[1] != parameters or neural.shape[0] < 32
            or chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[1] < 32
            or chains.shape[2] != parameters):
        raise ValueError("neural [L>=32,P] and reference [C>=2,D>=32,P] draws required")
    if family_checks < 2 * parameters:
        raise ValueError("family_checks must cover both metrics for every declared parameter/case")
    if not np.isfinite(mean_limit) or mean_limit <= 0:
        raise ValueError("positive finite mean_limit required")
    width_limits = _array(width_limits, "width limits")
    if width_limits.shape != (2,) or not 0 < width_limits[0] < 1 < width_limits[1]:
        raise ValueError("width limits must bracket one with positive endpoints")
    entries = diagnostics.get("parameters", {}) if isinstance(diagnostics, dict) else {}
    if not isinstance(entries, dict):
        raise ValueError("saved reference parameter diagnostics must be a dictionary")
    floors, effective = [], []
    for name in names:
        entry = entries.get(name, entries.get(f"parameters[{name}]"))
        if not isinstance(entry, dict):
            raise ValueError(f"missing saved reference diagnostics for {name}")
        values = _array([entry.get("ess_bulk", np.nan), entry.get("ess_tail", np.nan),
                         entry.get("mcse_mean", np.nan)], "reference ESS and mean MCSE")
        if np.any(values[:2] <= 0) or values[2] < 0:
            raise ValueError("positive reference ESS and nonnegative mean MCSE required")
        effective.extend(values[:2])
        floors.append(values[2])
    floors = np.asarray(floors)
    reference = chains.reshape(-1, parameters)
    ref_mean, ref_sd, ref_quantiles, ref_width = _summaries(reference)
    net_mean, net_sd, net_quantiles, net_width = _summaries(neural)
    count, draws, _ = chains.shape
    length = max(math.ceil(draws ** (1 / 3)), math.ceil(2 * count * draws / min(effective)))
    lengths = [length, 2 * length]
    result = {
        "format": "heston_budget_conditional_agreement_v1",
        "status": "inconclusive", "parameters": {}, "neural_shape": list(neural.shape),
        "reference_chain_shape": list(chains.shape), "seed": seed,
        "replicates_per_block_length": replicates, "block_lengths": lengths,
        "block_length_rule": "max(ceil(D^(1/3)), ceil(2*C*D/min(saved bulk and tail ESS across parameters))); also double L",
        "family_checks": family_checks, "family_confidence": .95,
        "interval_method": "approximate normal, two-sided Bonferroni; maximum MCSE over L and 2L",
        "reference_diagnostics_passed": diagnostics.get("mcmc_diagnostics_passed"),
        "limits": {"absolute_standardized_mean": float(mean_limit), "width_ratio_90": width_limits.tolist()},
        "scope": "Monte Carlo agreement on one fixed case; no numerical-target, product, copula or MCMC convergence certification",
    }
    if lengths[-1] >= draws:
        result["reason"] = "doubled block reaches chain length; reference has insufficient effective blocks for this uncertainty rule"
        return result
    children = np.random.SeedSequence(seed).spawn(2)
    bootstraps = [_conditional_bootstrap(neural, chains, length=block, replicates=replicates,
                                        rng=np.random.default_rng(child))
                  for block, child in zip(lengths, children, strict=True)]
    critical = float(norm.ppf(1 - .05 / (2 * family_checks)))
    result["normal_critical_value"] = critical
    result["block_length_sensitivity"] = [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                            for key, value in bootstrap.items()} for bootstrap in bootstraps]
    # The standardized-mean bootstrap varies both reference mean and SD.
    # Restore missing reference-mean variance whenever a short block bootstrap
    # understates the saved mean MCSE; also retain the analytic independent
    # neural/ref mean-error floor. This is deliberately conservative.
    mean_mcses = [np.sqrt(boot["standardized_mean_mcse"] ** 2 +
                         np.maximum(floors ** 2 - boot["reference_mean_mcse"] ** 2, 0) / ref_sd ** 2)
                  for boot in bootstraps]
    analytic_floor = np.sqrt((net_sd / np.sqrt(len(neural))) ** 2 + floors ** 2) / ref_sd
    mean_mcse = np.maximum(np.maximum.reduce(mean_mcses), analytic_floor)
    log_width_mcse = np.maximum.reduce([boot["log_width_ratio_mcse"] for boot in bootstraps])
    ref_mean_mcse = np.maximum(np.maximum.reduce([boot["reference_mean_mcse"] for boot in bootstraps]), floors)
    ref_quantile_mcse = np.maximum.reduce([boot["reference_quantile_mcse"] for boot in bootstraps])
    net_quantile_mcse = np.maximum.reduce([boot["neural_quantile_mcse"] for boot in bootstraps])
    signed_mean = (net_mean - ref_mean) / ref_sd
    width_ratio = net_width / ref_width
    statuses = []
    for index, name in enumerate(names):
        mean_interval = (signed_mean[index] + np.array([-1, 1]) * critical * mean_mcse[index]).tolist()
        log_interval = np.log(width_ratio[index]) + np.array([-1, 1]) * critical * log_width_mcse[index]
        width_interval = np.exp(log_interval).tolist()
        mean_status = _status(mean_interval, (-mean_limit, mean_limit))
        width_status = _status(width_interval, width_limits)
        status = _combined_status([mean_status, width_status])
        statuses.append(status)
        result["parameters"][name] = {
            "status": status, "neural_mean": float(net_mean[index]), "reference_mean": float(ref_mean[index]),
            "reference_sd": float(ref_sd[index]), "neural_width_90": float(net_width[index]),
            "reference_width_90": float(ref_width[index]),
            "signed_standardized_mean_difference": float(signed_mean[index]),
            "absolute_standardized_mean_difference": float(abs(signed_mean[index])),
            "standardized_mean_mcse": float(mean_mcse[index]),
            "signed_standardized_mean_interval": mean_interval, "mean_status": mean_status,
            "width_ratio_90": float(width_ratio[index]), "log_width_ratio_mcse": float(log_width_mcse[index]),
            "width_ratio_interval": width_interval, "width_status": width_status,
            "reference_mean_mcse": float(ref_mean_mcse[index]), "saved_reference_mean_mcse": float(floors[index]),
            "neural_mean_mcse_iid": float(net_sd[index] / np.sqrt(len(neural))),
            "reference_q05_q95": ref_quantiles[:, index].tolist(),
            "neural_q05_q95": net_quantiles[:, index].tolist(),
            "reference_quantile_mcse": ref_quantile_mcse[:, index].tolist(),
            "neural_quantile_mcse_iid": net_quantile_mcse[:, index].tolist(),
        }
    result["status"] = _combined_status(statuses)
    if diagnostics.get("mcmc_diagnostics_passed") is not True:
        result["status"] = "inconclusive"
        result["reason"] = "saved MCMC diagnostics have no explicit passing verdict; agreement cannot admit the study"
    return result


def _panel_ids(values, count):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != count:
        raise ValueError("one ordered test ID per observation required")
    identifiers = values.tolist()
    if (any(not isinstance(identifier, str) or not identifier.startswith("test:")
            or len(identifier) <= len("test:") for identifier in identifiers)
            or len(set(identifiers)) != count):
        raise ValueError("unique split-qualified test IDs required")
    return identifiers


def _identified_draws(record, identifiers, parameters, bounds):
    if not isinstance(record, dict) or not {"draws", "example_ids"} <= set(record):
        raise ValueError("draw records require draws and explicit ordered example_ids")
    if _panel_ids(record["example_ids"], len(identifiers)) != identifiers:
        raise ValueError("draw/truth test IDs differ or have a different order")
    draws = _array(record["draws"], "posterior draws")
    if (draws.ndim != 3 or draws.shape[0] != len(identifiers) or draws.shape[1] < 2
            or draws.shape[2] != parameters):
        raise ValueError("posterior draws [observations, draws>=2, parameters] required")
    if np.any(draws <= bounds[:, 0]) or np.any(draws >= bounds[:, 1]):
        raise ValueError("posterior draws must lie strictly inside the declared prior; never clip")
    return draws


def _panel_metrics(draws, truth, variance, levels):
    # Quantile estimates use exactly the supplied finite posterior sample; the
    # outer bootstrap resamples observations, not individual posterior draws.
    quantiles = np.quantile(draws, np.concatenate([.5 - levels / 2, .5 + levels / 2]), axis=1)
    low, high = quantiles[:len(levels)], quantiles[len(levels):]
    coverage = ((truth[None, ...] >= low) & (truth[None, ...] <= high)).transpose(1, 2, 0)
    normalized_squared_error = (draws.mean(axis=1) - truth) ** 2 / variance
    return coverage, normalized_squared_error


def _simultaneous_bounds(point, bootstrap):
    """Basic max-stat bounds with the correct upper-bound centered-error sign."""
    standard_error = bootstrap.std(axis=0, ddof=1)
    active = standard_error > 0
    if np.any(~active):
        tolerance = 16 * np.finfo(float).eps * np.maximum(1, np.abs(point[~active]))
        if np.any(np.abs(bootstrap[:, ~active] - point[~active]) > tolerance):
            raise ValueError("zero bootstrap SE with nonzero centered error; no confidence bound is available")
    if np.any(active):
        centered = (bootstrap[:, active] - point[active]) / standard_error[active]
        upper_critical = max(0., float(np.quantile(np.max(-centered, axis=1), .95, method="higher")))
        lower_critical = max(0., float(np.quantile(np.max(centered, axis=1), .95, method="higher")))
    else:
        upper_critical = lower_critical = 0.
    return (point - lower_critical * standard_error, point + upper_critical * standard_error,
            standard_error, upper_critical, lower_critical, int(np.count_nonzero(~active)))


def paired_final_comparison(candidates, control, truth, prior_bounds, example_ids, names, *, seed,
                            replicates=2000, levels=(.5, .8, .9, .95), coverage_margin=.02,
                            mse_ratio_limit=1.10):
    """Simultaneous one-sided paired-bootstrap bounds for a frozen final panel.

    Each value of ``candidates`` and ``control`` must be a dictionary containing
    ``draws`` [observation, draw, parameter] and the corresponding ``example_ids``.
    Their order must match the explicitly supplied truth IDs. A label identifies
    one frozen recipe/seed; every supplied label enters the simultaneous family.

    The max-statistic uses centered bootstrap errors with the upper-bound sign
    ``(point - bootstrap) / SE``. Its 95th percentile supplies all upper bounds.
    Zero bootstrap SE gives the constant point bound. These are approximate
    studentized bootstrap bounds; absolute coverage error has a kink at nominal
    coverage, so no exact finite-sample coverage claim is made. Neither final
    panel non-use nor the required separate rank/seed checks is certified here.
    """
    seed = _integer(seed, "seed", 0)
    replicates = _integer(replicates, "replicates", 200)
    truth, bounds = _array(truth, "truth"), _array(prior_bounds, "prior bounds")
    names = list(names)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or truth.ndim != 2
            or truth.shape[0] < 2 or truth.shape[1] != len(names)
            or bounds.shape != (len(names), 2) or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("at least two truth rows, unique names and increasing prior bounds required")
    if np.any(truth <= bounds[:, 0]) or np.any(truth >= bounds[:, 1]):
        raise ValueError("truth must lie strictly inside the declared prior bounds")
    identifiers = _panel_ids(example_ids, len(truth))
    if (not isinstance(candidates, dict) or not 1 <= len(candidates) <= 6
            or any(not isinstance(label, str) or not label for label in candidates)):
        raise ValueError("one to six explicitly labelled frozen candidate/seed records required")
    levels = _array(levels, "coverage levels")
    if (levels.ndim != 1 or not len(levels) or np.any(levels <= 0) or np.any(levels >= 1)
            or not np.all(np.diff(levels) > 0)):
        raise ValueError("strictly increasing coverage levels inside (0,1) required")
    if (not np.isfinite(coverage_margin) or coverage_margin < 0
            or not np.isfinite(mse_ratio_limit) or mse_ratio_limit <= 0):
        raise ValueError("nonnegative finite coverage margin and positive MSE ratio limit required")
    variance = (bounds[:, 1] - bounds[:, 0]) ** 2 / 12
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("finite positive uniform-prior variances required")
    control_draws = _identified_draws(control, identifiers, len(names), bounds)
    control_covered, control_error = _panel_metrics(control_draws, truth, variance, levels)
    candidate_covered, candidate_error = [], []
    for record in candidates.values():
        draws = _identified_draws(record, identifiers, len(names), bounds)
        covered, error = _panel_metrics(draws, truth, variance, levels)
        candidate_covered.append(covered)
        candidate_error.append(error)
    candidate_covered = np.stack(candidate_covered)
    candidate_error = np.stack(candidate_error)
    control_mse = control_error.mean(axis=0)
    if np.any(control_mse <= 0):
        raise ValueError("control MSE is zero; a per-parameter risk ratio is undefined")
    control_coverage = control_covered.mean(axis=0)
    candidate_coverage = candidate_covered.mean(axis=1)
    candidate_mse = candidate_error.mean(axis=1)
    increase = np.abs(candidate_coverage - levels) - np.abs(control_coverage - levels)
    ratio = candidate_mse / control_mse
    # The final axis stores all coverage levels and then one recovery metric.
    point = np.concatenate([increase, ratio[..., None]], axis=-1)
    bootstrap = np.empty((replicates, *point.shape))
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        index = rng.integers(len(truth), size=len(truth))
        baseline_coverage = control_covered[index].mean(axis=0)
        baseline_mse = control_error[index].mean(axis=0)
        if np.any(baseline_mse <= 0):
            raise ValueError("degenerate resampled control MSE; no observations may be dropped/redrawn")
        cov = candidate_covered[:, index].mean(axis=1)
        risk = candidate_error[:, index].mean(axis=1)
        bootstrap[replicate, ..., :-1] = np.abs(cov - levels) - np.abs(baseline_coverage - levels)
        bootstrap[replicate, ..., -1] = risk / baseline_mse
    lower, upper, standard_error, upper_critical, lower_critical, zero_se = _simultaneous_bounds(point, bootstrap)
    results = {}
    for model_index, label in enumerate(candidates):
        parameters = {}
        for parameter_index, name in enumerate(names):
            coverage = {}
            statuses = []
            for level_index, level in enumerate(levels):
                observed = float(point[model_index, parameter_index, level_index])
                hi = float(upper[model_index, parameter_index, level_index])
                lo = float(lower[model_index, parameter_index, level_index])
                status = ("pass" if hi <= coverage_margin else
                          "fail" if lo > coverage_margin else "inconclusive")
                statuses.append(status)
                coverage[f"{level:g}"] = {
                    "candidate_coverage": float(candidate_coverage[model_index, parameter_index, level_index]),
                    "control_coverage": float(control_coverage[parameter_index, level_index]),
                    "absolute_error_increase": observed, "upper_bound": hi, "lower_bound": lo,
                    "bootstrap_se": float(standard_error[model_index, parameter_index, level_index]), "status": status}
            hi, lo = float(upper[model_index, parameter_index, -1]), float(lower[model_index, parameter_index, -1])
            recovery_status = "pass" if hi <= mse_ratio_limit else "fail" if lo > mse_ratio_limit else "inconclusive"
            statuses.append(recovery_status)
            parameters[name] = {
                "status": _combined_status(statuses), "coverage": coverage,
                "recovery": {"candidate_normalized_mse": float(candidate_mse[model_index, parameter_index]),
                             "control_normalized_mse": float(control_mse[parameter_index]),
                             "mse_ratio": float(ratio[model_index, parameter_index]), "upper_bound": hi,
                             "lower_bound": lo, "bootstrap_se": float(standard_error[model_index, parameter_index, -1]),
                             "status": recovery_status}}
        results[label] = {"status": _combined_status([value["status"] for value in parameters.values()]),
                          "parameters": parameters}
    return {
        "format": "heston_budget_final_comparison_v1",
        "status": _combined_status([result["status"] for result in results.values()]),
        "candidates": results, "observations": len(truth), "parameter_names": names,
        "example_ids_sha256": hashlib.sha256(json.dumps(identifiers, separators=(",", ":")).encode()).hexdigest(),
        "levels": levels.tolist(), "seed": seed, "replicates": replicates,
        "family_checks": int(point.size), "family_confidence": .95,
        "upper_max_stat_critical_value": upper_critical,
        "lower_max_stat_critical_value": lower_critical,
        "zero_se_checks": zero_se,
        "method": "centered, studentized paired-observation bootstrap max-statistic; one-sided 95% upper bounds admit candidates",
        "lower_bound_scope": "separate one-sided 95% family for descriptive clear-failure status; not an additional admission condition",
        "resampling_unit": "one shared observation-index vector across every candidate/seed, control, parameter and level",
        "recovery_estimator": "posterior mean, physical squared error divided by independent-uniform prior variance",
        "limits": {"coverage_absolute_error_increase": float(coverage_margin), "normalized_mse_ratio": float(mse_ratio_limit)},
        "scope": "approximate finite-draw final-panel comparison; absolute-coverage-error kink has no exact bootstrap guarantee; untouched-panel audit and separate rank/seed gates remain required",
    }

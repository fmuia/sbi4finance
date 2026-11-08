"""Predeclared uncertainty checks for the bounded Heston training study.

These checks compare estimators on fixed synthetic cases. They do not establish
MCMC convergence, numerical target accuracy, product accuracy or the posterior
copula. Chain/draw structure is retained for Monte Carlo uncertainty; observations
are the resampling unit for the separate final-panel comparison.
"""

import math

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

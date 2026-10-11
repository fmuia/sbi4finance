"""Finite-Monte-Carlo reference for the restricted rough-Bergomi pilot.

Each independent prior row contains (theta, U, C_hat(theta, U)). Given a panel y,
Gaussian quote-error weights target the joint posterior of theta and pricing
randomness U. Marginalizing the weighted rows gives a parameter reference for
this *numerical* observation law, not a continuum-price posterior. Clean price
estimates enter the weights; noisy observed training panels do not. Adding the
price-estimator covariance to quote variance would count U uncertainty twice.

These self-normalized importance estimates have finite-bank error and possible
finite-bank bias. ESS and independent-block agreement do not prove that a bank
has found all important posterior regions. Marginal SBC is likewise neither a
conditional nor a joint validation certificate.
"""

import numpy as np
from scipy.stats import norm


_PROBABILITIES = np.array([0.05, 0.5, 0.95])


def _rows(values, *, columns=None, minimum=2, name="values"):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or len(values) < minimum or values.shape[1] < 1
            or (columns is not None and values.shape[1] != columns)
            or not np.isfinite(values).all()):
        shape = f"[N,{columns}]" if columns is not None else "[N,D]"
        raise ValueError(f"finite {name} {shape}, N>={minimum}, required")
    return values


def _weights(weights, count):
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (count,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("finite nonnegative weights [N] with positive total required")
    # Scaling first avoids overflow when callers provide unnormalized weights.
    weights = weights / weights.max()
    return weights / weights.sum()


def _snis_mean_mcse(values, weights):
    """Delta-method SE for a ratio of IID sample means, including random U/V."""
    count = len(values)
    mean = weights @ values
    return np.sqrt(count / (count - 1) * np.sum(
        (weights[:, None] * (values - mean)) ** 2, axis=0))


def importance_weights(clean_prices, y, sigma):
    """Normalized Gaussian quote-error weights for clean numerical panels.

    ``clean_prices`` is [N,C], ``y`` is [C], and positive ``sigma`` is scalar
    or [C]. Rows must be independent prior/pricing-randomness draws. Constants
    cancel because every row uses the same quote-error scale. Pricing covariance
    is deliberately absent: pricing randomness is integrated by the rows.
    """
    clean_prices = _rows(clean_prices, minimum=1, name="clean_prices")
    y, sigma = np.asarray(y, dtype=float), np.asarray(sigma, dtype=float)
    contracts = clean_prices.shape[1]
    if y.shape != (contracts,) or not np.isfinite(y).all():
        raise ValueError("finite y [C] matching clean_prices required")
    if (sigma.shape not in ((), (contracts,)) or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)):
        raise ValueError("positive finite sigma, scalar or [C], required")
    with np.errstate(over="ignore", invalid="ignore"):
        log_weights = -0.5 * np.sum(((clean_prices - y) / sigma) ** 2, axis=1)
    if not np.isfinite(log_weights).all():
        raise ValueError("quote residuals too large to evaluate finite log weights")
    return _weights(np.exp(log_weights - log_weights.max()), len(clean_prices))


def weighted_summary(parameters, weights):
    """JSON-ready population moments and inverse-empirical-CDF quantiles.

    Parameters are paired rows [N,2]; quantiles have shape [2,3] at probabilities
    0.05/0.5/0.95. Weighted posterior variance is a distributional moment, without
    a Bessel correction. It is different from uncertainty in the estimated mean.
    """
    parameters = _rows(parameters, columns=2, name="parameters")
    weights = _weights(weights, len(parameters))
    mean = weights @ parameters
    centered = parameters - mean
    covariance = (centered * weights[:, None]).T @ centered
    std = np.sqrt(np.diag(covariance))
    if not np.isfinite(covariance).all() or np.any(std <= 0):
        raise ValueError("nondegenerate finite weighted parameter marginals required")
    correlation = float(np.clip(covariance[0, 1] / np.prod(std), -1.0, 1.0))
    quantiles = []
    for values in parameters.T:
        order = np.argsort(values, kind="stable")
        order = order[weights[order] > 0]
        cumulative = np.cumsum(weights[order])
        cumulative[-1] = 1.0
        quantiles.append(values[order[np.searchsorted(cumulative, _PROBABILITIES)]].tolist())
    return {"mean": mean.tolist(), "std": std.tolist(), "covariance": covariance.tolist(),
            "correlation": correlation, "probabilities": _PROBABILITIES.tolist(),
            "quantiles": quantiles}


def reference_diagnostics(parameters, weights):
    """Report finite-bank precision and agreement of two independent row blocks.

    Contiguous halves must have been generated independently; this function
    cannot infer independence from the arrays. Mean SEs use the IID SNIS delta
    method. Block ratios compare second block / first block; differences compare
    second minus first. Failed block support is explicit, never silently removed.
    """
    parameters = _rows(parameters, columns=2, minimum=4, name="parameters")
    weights = _weights(weights, len(parameters))
    summary = weighted_summary(parameters, weights)
    mean_mcse = _snis_mean_mcse(parameters, weights)
    blocks = []
    midpoint = len(parameters) // 2
    for indices in (slice(0, midpoint), slice(midpoint, len(parameters))):
        block_weights, block_parameters = weights[indices], parameters[indices]
        try:
            block_weights = _weights(block_weights, len(block_parameters))
            block_summary = weighted_summary(block_parameters, block_weights)
        except ValueError as error:
            blocks.append({"count": len(block_parameters), "valid": False, "reason": str(error)})
        else:
            blocks.append({"count": len(block_parameters), "valid": True,
                           "ess": float(1 / np.sum(block_weights**2)),
                           "max_weight": float(block_weights.max()), "summary": block_summary})
    blocks_valid = all(block["valid"] for block in blocks)
    comparison = None
    if blocks_valid:
        first, second = (block["summary"] for block in blocks)
        comparison = {
            "mean_difference_in_ref_sd": ((np.asarray(second["mean"]) - first["mean"])
                                          / summary["std"]).tolist(),
            "std_ratio": (np.asarray(second["std"]) / first["std"]).tolist(),
            "correlation_difference": float(second["correlation"] - first["correlation"])}
    return {"count": len(parameters), "ess": float(1 / np.sum(weights**2)),
            "max_weight": float(weights.max()), "summary": summary,
            "mean_mcse": mean_mcse.tolist(),
            "mean_mcse_in_ref_sd": (mean_mcse / summary["std"]).tolist(),
            "blocks_valid": blocks_valid, "blocks": blocks, "block_comparison": comparison,
            "scope": "IID prior-importance approximation to finite-Monte-Carlo observation law; ESS is not a support certificate"}


def conditional_metrics(draws, parameters, weights, *, prior_bounds=None):
    """Compare neural joint draws with an independent weighted reference.

    Marginal 90% interval mass is summed over the empirical weighted reference.
    Optional ``prior_bounds`` [2,2] supplies [lower,upper] for each independent
    uniform prior. Contraction is 1 - posterior SD / prior SD and may be negative.
    Mean update is measured from the prior mean in prior SD units. Neither is
    evidence of accurate conditioning without the independent reference checks.
    """
    draws = _rows(draws, columns=2, minimum=3, name="draws")
    parameters = _rows(parameters, columns=2, name="parameters")
    weights = _weights(weights, len(parameters))
    reference = weighted_summary(parameters, weights)
    ref_mean, ref_std = np.asarray(reference["mean"]), np.asarray(reference["std"])
    mean, std = draws.mean(axis=0), draws.std(axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError("nondegenerate neural draws required for joint correlation")
    correlation = float(np.corrcoef(draws.T)[0, 1])
    intervals = np.quantile(draws, [0.05, 0.95], axis=0).T
    masses = [float(weights[(parameters[:, d] >= low) & (parameters[:, d] <= high)].sum())
              for d, (low, high) in enumerate(intervals)]
    result = {"neural_mean": mean.tolist(), "neural_std": std.tolist(),
              "neural_correlation": correlation,
              "mean_error_in_ref_sd": ((mean - ref_mean) / ref_std).tolist(),
              "relative_std_error": (std / ref_std - 1).tolist(),
              "correlation_error": float(correlation - reference["correlation"]),
              "parameter_quantiles": np.quantile(draws, _PROBABILITIES, axis=0).T.tolist(),
              "parameter_intervals_90": intervals.tolist(),
              "reference_mass_of_90_interval": masses,
              "mean_mcse_in_ref_sd": (std / np.sqrt(len(draws)) / ref_std).tolist()}
    if prior_bounds is not None:
        bounds = np.asarray(prior_bounds, dtype=float)
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 1] <= bounds[:, 0])):
            raise ValueError("finite ordered prior_bounds [2,2] required")
        prior_mean, prior_std = bounds.mean(axis=1), np.diff(bounds, axis=1)[:, 0] / np.sqrt(12)
        result.update({"prior_mean": prior_mean.tolist(), "prior_std": prior_std.tolist(),
                       "neural_contraction": (1 - std / prior_std).tolist(),
                       "reference_contraction": (1 - ref_std / prior_std).tolist(),
                       "neural_mean_update_in_prior_sd": ((mean - prior_mean) / prior_std).tolist(),
                       "reference_mean_update_in_prior_sd": ((ref_mean - prior_mean) / prior_std).tolist()})
    return result


def weighted_target_mean(target_estimates, weights):
    """Estimate posterior mean target prices and their joint-row SNIS MCSE.

    Target estimates [N,K] must use fresh pricing randomness V, conditionally
    independent of observation-price randomness U given theta. Reusing U would
    instead average under U|y and generally change the price target. Independent
    (theta,U,V) rows allow the ratio-estimator SE to include both reference-bank
    and target-pricing randomness. These are mean-price estimates, not samples
    or quantiles of a posterior distribution of exact prices. Numerical target
    time-grid bias is not part of the reported sampling SE.
    """
    values = _rows(target_estimates, name="target_estimates")
    weights = _weights(weights, len(values))
    return {"mean": (weights @ values).tolist(),
            "mcse": _snis_mean_mcse(values, weights).tolist(),
            "count": len(values), "ess": float(1 / np.sum(weights**2)),
            "scope": "posterior mean numerical target prices; IID joint-row SNIS sampling uncertainty"}


def randomized_sbc(draws, truth, *, seed, alpha=0.05):
    """Marginal randomized SBC ranks and 90% coverage for [B,L,2] draws.

    Ties receive uniformly random rank placement, followed by within-rank
    randomization. The DKW threshold is union-adjusted over both marginals.
    Wilson coverage intervals are individual approximate binomial intervals.
    """
    draws, truth = np.asarray(draws, dtype=float), np.asarray(truth, dtype=float)
    if (draws.ndim != 3 or not all(draws.shape) or draws.shape[2] != 2
            or truth.shape != (draws.shape[0], 2)
            or not np.isfinite(draws).all() or not np.isfinite(truth).all()):
        raise ValueError("finite draws [B,L,2] and truth [B,2] required")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    count, samples, dimensions = draws.shape
    rng = np.random.default_rng(seed)
    lower = np.sum(draws < truth[:, None, :], axis=1)
    ties = np.sum(draws == truth[:, None, :], axis=1)
    ranks = lower + rng.integers(0, ties + 1, size=ties.shape)
    randomized = (ranks + rng.random(ranks.shape)) / (samples + 1)
    ordered = np.sort(randomized, axis=0)
    ks = np.maximum(np.max(np.arange(1, count + 1)[:, None] / count - ordered, axis=0),
                    np.max(ordered - np.arange(count)[:, None] / count, axis=0))
    bound = float(np.sqrt(np.log(2 * dimensions / alpha) / (2 * count)))
    intervals = np.quantile(draws, [0.05, 0.95], axis=1)
    covered = (truth >= intervals[0]) & (truth <= intervals[1])
    coverage = covered.mean(axis=0)
    z = norm.ppf(1 - alpha / 2)
    denominator = 1 + z * z / count
    center = (coverage + z * z / (2 * count)) / denominator
    half = z * np.sqrt(coverage * (1 - coverage) / count + z * z / (4 * count**2)) / denominator
    return {"cases": count, "draws_per_case": samples, "ranks": ranks.tolist(),
            "randomized_ranks": randomized.tolist(), "rank_ks": ks.tolist(),
            "rank_dkw_bound": bound, "sbc_family_alpha": float(alpha),
            "rank_diagnostic_passed": bool(np.all(ks <= bound)),
            "coverage_90": coverage.tolist(), "coverage_90_counts": covered.sum(axis=0).tolist(),
            "coverage_90_wilson_interval": np.stack([center-half, center+half], axis=1).tolist(),
            "coverage_interval_confidence": float(1-alpha),
            "coverage_interval_scope": "individual approximate binomial intervals"}

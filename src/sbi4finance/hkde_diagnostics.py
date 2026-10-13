"""Small numerical diagnostics for the restricted two-parameter HKDE task.

SBC is a prior-predictive marginal check, not a conditional/joint certificate.
Conditional comparisons use the separately refined numerical reference; their
Monte Carlo uncertainty is reported without silently changing acceptance gates.
"""

import numpy as np
from scipy.stats import binom, norm


def randomized_sbc(draws, truth, *, seed, alpha=0.05):
    """JSON-ready marginal SBC and 90% coverage for [B,L,2] joint draws.

Random placement among ties followed by within-rank randomization preserves
uniform ranks even for a discrete exchangeable target. The DKW threshold is
union-adjusted over both targets; Wilson intervals are individual approximate
binomial intervals, not simultaneous intervals or conditional guarantees.
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


def conditional_metrics(draws, reference_summary, target_draws, *, probabilities=(0.05, 0.5, 0.95)):
    """Compare joint draws and matched downstream prices with one reference.

``reference_summary`` is the output of ``hkde_reference.summarize``. Parameter
and price quantiles have shape [2,P], matching that reference. Interval masses
interpolate its integrated marginal CDFs, rather than summing endpoint masses.
Price-quantile uncertainty uses individual 95% order-statistic intervals under
IID draws; this is sampling uncertainty, not a confidence interval for neural
bias. An unbounded order-statistic endpoint is returned as JSON null.
"""
    draws, targets = np.asarray(draws, dtype=float), np.asarray(target_draws, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if (draws.ndim != 2 or draws.shape[1] != 2 or len(draws) < 3
            or targets.shape != draws.shape or not np.isfinite(draws).all()
            or not np.isfinite(targets).all()):
        raise ValueError("finite parameter and target-price draws [L,2], L>=3 required")
    if (probs.ndim != 1 or not len(probs) or not np.isfinite(probs).all()
            or np.any((probs <= 0) | (probs >= 1))
            or not np.array_equal(probs, reference_summary["probabilities"])):
        raise ValueError("probabilities must match the reference and lie in (0,1)")
    ref_mean, ref_std = (np.asarray(reference_summary[key], dtype=float) for key in ("mean", "std"))
    ref_correlation = reference_summary["correlation"]
    ref_targets = np.asarray(reference_summary["target_quantiles"], dtype=float)
    if (ref_mean.shape != (2,) or ref_std.shape != (2,)
            or not np.isfinite(ref_mean).all() or not np.isfinite(ref_std).all()
            or np.any(ref_std <= 0) or ref_correlation is None
            or not np.isfinite(ref_correlation) or ref_targets.shape != (2, len(probs))
            or not np.isfinite(ref_targets).all()):
        raise ValueError("finite nondegenerate two-target reference required")
    mean, std = draws.mean(axis=0), draws.std(axis=0, ddof=1)
    if np.any(std == 0):
        raise ValueError("degenerate posterior draws cannot define joint correlation")
    correlation = float(np.corrcoef(draws.T)[0, 1])
    intervals = np.quantile(draws, [0.05, 0.95], axis=0).T
    masses = []
    for dimension, (axis, cdf) in enumerate(zip(reference_summary["marginal_axes"],
                                               reference_summary["marginal_cdfs"], strict=True)):
        axis, cdf = np.asarray(axis), np.asarray(cdf)
        if (axis.ndim != 1 or len(axis) < 3 or cdf.shape != axis.shape
                or not np.isfinite(axis).all() or not np.isfinite(cdf).all()
                or np.any(np.diff(axis) <= 0) or np.any(np.diff(cdf) < 0)
                or not np.allclose(cdf[[0, -1]], [0, 1], rtol=0, atol=1e-12)):
            raise ValueError("reference marginal axes and integrated CDFs required")
        values = np.interp(intervals[dimension], axis, cdf, left=0.0, right=1.0)
        masses.append(float(values[1] - values[0]))
    if len(masses) != 2:
        raise ValueError("exactly two reference marginals required")
    target_quantiles = np.quantile(targets, probs, axis=0).T
    samples = len(draws)
    # For K~Binomial(L,p), [X_(l),X_(u)] covers the population p quantile
    # when l<=K<u; indices are one-based, with 0/L+1 unbounded sentinels.
    lower = binom.ppf(0.025, samples, probs).astype(int)
    upper = binom.ppf(0.975, samples, probs).astype(int) + 1
    ordered_targets = np.sort(targets, axis=0)
    uncertainty = [[[float(ordered_targets[l-1, d]) if l > 0 else None,
                     float(ordered_targets[u-1, d]) if u <= samples else None]
                    for l, u in zip(lower, upper, strict=True)] for d in range(2)]
    return {"neural_mean": mean.tolist(), "neural_std": std.tolist(),
            "neural_correlation": correlation,
            "mean_error_in_ref_sd": ((mean-ref_mean)/ref_std).tolist(),
            "relative_std_error": (std/ref_std-1).tolist(),
            "correlation_error": float(correlation-ref_correlation),
            "parameter_quantiles": np.quantile(draws, probs, axis=0).T.tolist(),
            "parameter_intervals_90": intervals.tolist(),
            "reference_mass_of_90_interval": masses,
            "mean_mcse_in_ref_sd": (std/np.sqrt(samples)/ref_std).tolist(),
            "target_quantiles": target_quantiles.tolist(),
            "target_quantile_error_currency": (target_quantiles-ref_targets).tolist(),
            "target_quantile_mc95_interval": uncertainty,
            "target_quantile_mc95_order_indices": np.stack([lower, upper], axis=1).tolist(),
            "target_quantile_mc_interval_scope": "individual IID 95% order-statistic intervals; not neural bias uncertainty"}

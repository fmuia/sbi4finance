"""Checks of an amortized posterior on held-out simulated surfaces, for any number of parameters.

SBC ranks are randomized for ties and judged with a simultaneous DKW band,
union-adjusted over parameters, as in rough_reference (which is fixed at two
parameters). Coverage is reported at several central levels with Wilson
intervals. Contraction is computed in the logit coordinates of the prior box,
where a uniform prior has the logistic variance pi^2 / 3. Agreement with a
reference posterior uses the design document's metrics: mean difference in
reference standard deviations and the ratio of 90% interval widths.
"""

import numpy as np
from scipy.special import logit
from scipy.stats import norm


def _checked(draws, truth):
    draws, truth = np.asarray(draws, dtype=float), np.asarray(truth, dtype=float)
    if (draws.ndim != 3 or truth.shape != (draws.shape[0], draws.shape[2])
            or not np.isfinite(draws).all() or not np.isfinite(truth).all()):
        raise ValueError("finite draws [B, L, P] and truth [B, P] required")
    return draws, truth


def sbc(draws, truth, *, seed, alpha=0.05):
    draws, truth = _checked(draws, truth)
    count, samples, dims = draws.shape
    rng = np.random.default_rng(seed)
    lower = np.sum(draws < truth[:, None, :], axis=1)
    ties = np.sum(draws == truth[:, None, :], axis=1)
    ranks = lower + rng.integers(0, ties + 1, size=ties.shape)
    unit = (ranks + rng.random(ranks.shape)) / (samples + 1)
    ordered = np.sort(unit, axis=0)
    grid = np.arange(1, count + 1)[:, None] / count
    ks = np.maximum(np.max(grid - ordered, axis=0), np.max(ordered - (grid - 1 / count), axis=0))
    bound = float(np.sqrt(np.log(2 * dims / alpha) / (2 * count)))
    return {"surfaces": count, "draws_per_surface": samples, "rank_ks": ks.tolist(), "dkw_bound": bound,
            "family_alpha": float(alpha), "passed": [bool(x <= bound) for x in ks],
            "fractional_ranks": unit.tolist()}


def ecdf_difference(fractional_ranks, points=201):
    """ECDF of fractional ranks minus the uniform CDF, on an even grid, per parameter."""
    unit = np.asarray(fractional_ranks, dtype=float)
    grid = np.linspace(0, 1, points)
    return grid, np.stack([np.searchsorted(np.sort(unit[:, j]), grid, side="right") / len(unit) - grid
                           for j in range(unit.shape[1])], axis=1)


def coverage(draws, truth, levels=(0.5, 0.8, 0.9, 0.95), confidence=0.95):
    draws, truth = _checked(draws, truth)
    count = len(truth)
    z = norm.ppf(0.5 + confidence / 2)
    result = {}
    for level in levels:
        bounds = np.quantile(draws, [0.5 - level / 2, 0.5 + level / 2], axis=1)
        share = ((truth >= bounds[0]) & (truth <= bounds[1])).mean(axis=0)
        centre = (share + z * z / (2 * count)) / (1 + z * z / count)
        half = z * np.sqrt(share * (1 - share) / count + z * z / (4 * count**2)) / (1 + z * z / count)
        result[f"{level:g}"] = {"coverage": share.tolist(), "wilson": np.stack([centre - half, centre + half], 1).tolist()}
    return result


def contraction(draws, prior_bounds):
    """1 - mean posterior variance / prior variance, in the logit coordinates of the prior box."""
    draws = np.asarray(draws, dtype=float)
    bounds = np.asarray(prior_bounds, dtype=float)
    unit = np.clip((draws - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]), 1e-12, 1 - 1e-12)
    posterior_variance = logit(unit).var(axis=1).mean(axis=0)
    return (1 - posterior_variance / (np.pi**2 / 3)).tolist()


def recovery(draws, truth):
    draws, truth = _checked(draws, truth)
    median = np.median(draws, axis=1)
    q05, q95 = np.quantile(draws, [0.05, 0.95], axis=1)
    correlation = [float(np.corrcoef(truth[:, j], median[:, j])[0, 1]) for j in range(truth.shape[1])]
    return {"median": median, "q05": q05, "q95": q95, "correlation": correlation}


def agreement(draws, reference_draws=None, *, reference_mean=None, reference_sd=None, reference_q05=None,
              reference_q95=None):
    """Per parameter: |mean - reference mean| / reference sd, and 90% width / reference 90% width."""
    draws = np.asarray(draws, dtype=float)
    if reference_draws is not None:
        reference = np.asarray(reference_draws, dtype=float)
        reference_mean, reference_sd = reference.mean(axis=0), reference.std(axis=0)
        reference_q05, reference_q95 = np.quantile(reference, [0.05, 0.95], axis=0)
    q05, q95 = np.quantile(draws, [0.05, 0.95], axis=0)
    return {"standardized_mean_error": (np.abs(draws.mean(axis=0) - reference_mean) / reference_sd).tolist(),
            "width_ratio": ((q95 - q05) / (np.asarray(reference_q95) - np.asarray(reference_q05))).tolist()}

"""Checks of an amortized posterior on held-out simulated surfaces, for any number of parameters.

SBC ranks are randomized for ties and judged with a simultaneous DKW band,
union-adjusted over parameters, as in rough_reference (which is fixed at two
parameters). Coverage is reported at several central levels with Wilson
intervals. Contraction is computed in the logit coordinates of the prior box,
where a uniform prior has the logistic variance pi^2 / 3. Agreement with a
reference posterior uses the tutorial's metrics: mean difference in
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


def contraction(draws, prior_bounds, prior_variance=None):
    """1 - mean posterior variance / prior variance, in the logit coordinates of the prior box.

    A target uniform on its bounds has the logistic variance pi^2/3 in these coordinates. A
    derived target whose prior is not uniform on its range, such as rho * eta, needs its own
    prior variance in prior_variance (one entry per target; None keeps pi^2/3), else its
    contraction is overstated or understated.
    """
    draws = np.asarray(draws, dtype=float)
    bounds = np.asarray(prior_bounds, dtype=float)
    unit = np.clip((draws - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]), 1e-12, 1 - 1e-12)
    posterior_variance = logit(unit).var(axis=1).mean(axis=0)
    prior = np.full(draws.shape[-1], np.pi**2 / 3)
    if prior_variance is not None:
        prior = np.array([np.pi**2 / 3 if v is None else v for v in prior_variance], dtype=float)
    return (1 - posterior_variance / prior).tolist()


def posterior_correlations(draws, prior_bounds):
    """Within-surface correlation of every pair of targets, in the logit coordinates of the prior box.

    A ridge in the posterior shows as a correlation near +1 or -1 on most surfaces.
    Returns {(a, b): {"median", "q10", "q90"}} over surfaces.
    """
    draws = np.asarray(draws, dtype=float)
    bounds = np.asarray(prior_bounds, dtype=float)
    unit = np.clip((draws - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]), 1e-12, 1 - 1e-12)
    z = logit(unit)
    z = z - z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1)
    out = {}
    for a in range(z.shape[-1]):
        for b in range(a + 1, z.shape[-1]):
            values = np.mean(z[..., a] * z[..., b], axis=1) / (sd[:, a] * sd[:, b])
            out[(a, b)] = {"median": float(np.median(values)), "q10": float(np.quantile(values, 0.1)),
                           "q90": float(np.quantile(values, 0.9))}
    return out


def product_prior_variance(bounds_a, bounds_b, product_bounds, draws=1_000_000, seed=0):
    """Prior variance of a * b, for independent uniforms a and b, in the logit coordinates of product_bounds."""
    rng = np.random.default_rng(seed)
    product = rng.uniform(*bounds_a, draws) * rng.uniform(*bounds_b, draws)
    low, high = product_bounds
    unit = np.clip((product - low) / (high - low), 1e-12, 1 - 1e-12)
    return float(logit(unit).var())


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

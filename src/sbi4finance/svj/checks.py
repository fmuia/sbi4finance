"""Checks on the amortized posterior: calibration on simulated data and misspecification on real windows.

Simulation-based calibration (SBC) ranks each true value among the posterior draws of its own series;
for an exact posterior the fractional ranks are uniform. Their ECDF minus the uniform CDF must stay
inside a 95% simultaneous band. Recovery, coverage and contraction say what four years of returns
identify. None of these see real data, so the misspecification test asks, in the space of the 32 learned
summary statistics, whether an observed window looks like the training data (Schmitt et al. 2024).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bayesflow.utils.ecdf import simultaneous_ecdf_bands

from . import model

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


def fractional_ranks(draws, truths, seed=0):
    """Randomized fractional ranks (k + U) / (L + 1), k = number of the L draws below the truth: (n, d).

    For an exact posterior k is uniform on {0, ..., L}, so the randomized rank is exactly uniform on (0, 1),
    which removes the false alarms a discrete rank causes at the ends of the continuous band.
    """
    draws, truths = np.asarray(draws), np.asarray(truths)
    below = np.sum(draws < truths[:, None, :], axis=1)
    jitter = np.random.default_rng(seed).random(below.shape)
    return (below + jitter) / (draws.shape[1] + 1)


def sbc(draws, truths, confidence=0.95, seed=0):
    """ECDF of the fractional ranks minus the uniform CDF, and whether it stays inside the simultaneous band."""
    ranks = fractional_ranks(draws, truths, seed)
    n = len(ranks)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        _, z, lower, upper = simultaneous_ecdf_bands(n, confidence=confidence)
    finally:
        np.random.set_state(state)
    ecdf = np.mean(ranks[None, :, :] <= z[:, None, None], axis=1)  # (K, d)
    inside = (ecdf >= lower[:, None]) & (ecdf <= upper[:, None])
    return {
        "z": z,
        "ecdf_minus_uniform": ecdf - z[:, None],
        "band_half_width_at_centre": float((upper - lower)[len(z) // 2] / 2),
        "lower": lower - z,
        "upper": upper - z,
        "inside": inside.all(axis=0),
        "max_abs_deviation": np.abs(ecdf - z[:, None]).max(axis=0),
    }


def coverage(draws, truths, levels=COVERAGE_LEVELS):
    """Share of series whose central posterior interval at each level contains the truth, shape (levels, d)."""
    draws, truths = np.asarray(draws), np.asarray(truths)
    out = []
    for level in levels:
        lo = np.quantile(draws, (1 - level) / 2, axis=1)
        hi = np.quantile(draws, (1 + level) / 2, axis=1)
        out.append(np.mean((truths >= lo) & (truths <= hi), axis=0))
    return np.array(out)


def contraction(draws, truths):
    """1 - average posterior variance / prior variance, in the unbounded coordinates, per target."""
    return 1 - np.asarray(draws).var(axis=1).mean(axis=0) / np.asarray(truths).var(axis=0)


@dataclass
class MisspecificationTest:
    """MMD between one observed summary vector and the summaries of simulated series (Appendix G.5).

    D(s) = 1 - (2 / R) sum_j k(s, r_j) with the inverse multiquadric kernel k(a, b) = 1 / sqrt(1 + |a - b|² / l);
    the p-value compares D(s) with its distribution over further simulated series.
    """

    reference: np.ndarray  # (R, 32)
    null: np.ndarray  # (n_null, 32) summaries of further simulated series
    scale: float  # l, the median squared distance between 200 reference summaries

    def statistic(self, summaries):
        s = np.atleast_2d(summaries)
        sq = ((s[:, None, :] - self.reference[None, :, :]) ** 2).sum(-1)
        return 1 - 2 * np.mean(1 / np.sqrt(1 + sq / self.scale), axis=1)

    def pvalue(self, summaries):
        null = self.statistic(self.null)
        observed = self.statistic(summaries)
        return (1 + (null[None, :] >= observed[:, None]).sum(axis=1)) / (1 + len(null))

    @classmethod
    def build(cls, posterior, n_reference=2_000, n_null=500, seed=0):
        """Simulate from the prior and summarize with the trained network (`AmortizedPosterior`)."""
        sims = model.simulate(n_reference + n_null, seed=seed)
        summaries = posterior.summaries(sims["returns"])
        reference, null = summaries[:n_reference], summaries[n_reference:]
        sub = reference[:200]
        sq = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        return cls(reference, null, float(np.median(sq[np.triu_indices(len(sub), k=1)])))

    def save(self, path):
        np.savez_compressed(path, reference=self.reference, null=self.null, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(Path(path)) as data:
            return cls(data["reference"], data["null"], float(data["scale"]))

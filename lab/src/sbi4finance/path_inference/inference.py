"""Masked, arbitrary-prefix BayesFlow inference from observed returns only.

All lengths 1..T are supported. Every training batch draws an independent
integer prefix for every trajectory, pads to T, and forwards a boolean mask
through BayesFlow 2.1 to the causal GRU. Fixed physical scaling preserves
between-history return amplitude; no data-dependent standardizer is fitted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.trajectories.inference import (
    FORMAT,
    DEFAULT_NETWORK,
    LOGISTIC_SD,
    PREFIX_POLICIES,
    BACKEND_SEED_MAPPING,
    fingerprint,
    sha256,
    integer,
    backend_seed,
    validate_task,
    _bounds,
    inverse_parameters,
    conditioning_data,
    build_workflow,
    sample,
    load_checkpoint,
    transform_parameters,
    training_data,
    draw_prefix_lengths,
    random_prefixes,
)


def prior_samples(task, count, seed):
    rng = np.random.default_rng(seed)
    bounds, logged = _bounds(task)
    result = rng.uniform(bounds[:, 0], bounds[:, 1], (count, len(bounds)))
    result[..., logged] = np.exp(result[..., logged])
    return result


def diagnostics(samples, truth, prior, *, seed=0):
    """Physical-coordinate diagnostics. No variance expansion is clipped away."""
    samples, truth, prior = (np.asarray(v, dtype=np.float64) for v in (samples, truth, prior))
    if samples.ndim != 3 or samples.shape[0] != len(truth) or samples.shape[2] != truth.shape[1]:
        raise ValueError("posterior/truth shapes do not align")
    mean = samples.mean(axis=1)
    prior_mean, prior_variance = prior.mean(axis=0), prior.var(axis=0)
    mse = np.mean((mean - truth) ** 2, axis=0)
    baseline = np.mean((prior_mean - truth) ** 2, axis=0)
    rng = np.random.default_rng(seed)
    wrong = truth[rng.permutation(len(truth))]
    ranks = np.sum(samples < truth[:, None, :], axis=1)
    u = (ranks + rng.uniform(size=ranks.shape)) / (samples.shape[1] + 1)
    ordered = np.sort(u, axis=0)
    n = len(truth)
    ks = np.maximum(np.max(np.arange(1, n + 1)[:, None] / n - ordered, axis=0),
                    np.max(ordered - np.arange(n)[:, None] / n, axis=0))
    coverage = {}
    for level in (.50, .80, .90, .95):
        lower, upper = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
        hits = (truth >= lower) & (truth <= upper)
        rate = hits.mean(axis=0)
        coverage[str(level)] = {"coverage": rate.tolist(),
                                "binomial_se": np.sqrt(rate * (1 - rate) / n).tolist()}
    lower, upper = np.quantile(samples, [.05, .95], axis=1)
    prior_width = np.diff(np.quantile(prior, [.05, .95], axis=0), axis=0)[0]
    corr = []
    for j in range(truth.shape[1]):
        corr.append(float(np.corrcoef(mean[:, j], truth[:, j])[0, 1])
                    if np.std(mean[:, j]) > 0 and np.std(truth[:, j]) > 0 else None)
    return {"cases": n, "draws_per_case": samples.shape[1], "rank_ks": ks.tolist(),
            "coverage": coverage, "mean_squared_error": mse.tolist(),
            "prior_mean_squared_error": baseline.tolist(),
            "mse_to_prior_mean_ratio": (mse / baseline).tolist(),
            "wrong_pair_mean_squared_error": np.mean((mean - wrong) ** 2, axis=0).tolist(),
            "posterior_mean_truth_correlation": corr,
            "mean_posterior_to_prior_variance": (samples.var(axis=1).mean(axis=0) / prior_variance).tolist(),
            "mean_90_interval_to_prior_width": ((upper - lower).mean(axis=0) / prior_width).tolist(),
            "rank_screen_note": "descriptive marginal diagnostics; no conditional/joint accuracy certification"}

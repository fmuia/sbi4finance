"""Finite-rank repeated-simulation checks, kept separate from predictive adequacy."""

import numpy as np

from . import model


def rank_statistic(ranks, draws):
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or not np.isin(ranks, np.arange(draws + 1)).all():
        raise ValueError("ranks must be integers on the finite posterior-rank support")
    empirical = np.cumsum(np.bincount(ranks.astype(int), minlength=draws + 1)) / len(ranks)
    null = np.arange(1, draws + 2) / (draws + 1)
    return float(np.max(np.abs(empirical - null))), empirical - null


def null_statistics(rule):
    rng = np.random.default_rng(rule["null_seed"])
    n, draws = rule["histories_per_resolution"], rule["posterior_draws"]
    # Batched to bound memory. The rule fixes this distribution before evaluation.
    return np.array([rank_statistic(rng.integers(0, draws + 1, n), draws)[0]
                     for _ in range(rule["null_replications"])])


def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z*z / n
    middle = (p + z*z / (2*n)) / denominator
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return float(middle - half), float(middle + half)


def evaluate(posterior, truths, *, null):
    values = np.asarray(posterior)
    truth = np.asarray(truths)
    if values.ndim != 3 or values.shape[::2] != (len(truth), 4) or truth.shape != (len(values), 4) or not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise ValueError("require [histories,draws,4] posterior and [histories,4] truth")
    ranks = np.sum(values < truth[:, None, :], axis=1)
    median = np.median(values, axis=1)
    low, high = np.quantile(values, [.05, .95], axis=1)
    covered = (truth >= low) & (truth <= high)
    records, deviations = [], []
    for j, parameter in enumerate(model.PARAMETERS):
        stat, delta = rank_statistic(ranks[:, j], values.shape[1])
        deviations.append(delta)
        mse = np.mean((median[:, j] - truth[:, j])**2)
        # Analytic prior means; ν−2 has a log-uniform prior.
        prior_mean = (0., .015, .71, 2 + (10 - .05) / np.log(10/.05))[j]
        baseline = np.mean((prior_mean - truth[:, j])**2)
        k = int(covered[:, j].sum())
        records.append({"parameter": parameter, "rank_deviation": stat,
                        "rank_p_value": float((1 + np.sum(null >= stat)) / (len(null) + 1)),
                        "coverage90": k / len(truth), "coverage90_wilson95": wilson(k, len(truth)),
                        "mean_interval_width90": float(np.mean(high[:, j] - low[:, j])),
                        "mse_to_prior_mean_ratio": float(mse / baseline)})
    return {"records": records, "ranks": ranks, "rank_deviations": np.array(deviations),
            "median": median, "lower90": low, "upper90": high, "truth": truth}


def holm(records, alpha=.05):
    """Adjusted p-values for the declared family; return original display order."""
    count = len(records)
    order = np.argsort([row["rank_p_value"] for row in records])
    previous = 0.
    result = [dict(row) for row in records]
    for i, index in enumerate(order):
        previous = max(previous, min(1., (count-i) * records[index]["rank_p_value"]))
        result[index].update(rank_holm_p=previous, rank_holm_reject=bool(previous <= alpha))
    return result

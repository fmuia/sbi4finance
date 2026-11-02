"""Independent one-return reference for the joint rough discrete task.

At its declared origin, V_0 is short_variance and the standardized price
innovation is N(0,1) for every rho. Only short_variance is updated by the first
return. Its log-uniform prior gives a scalar posterior integrated here by
deterministic Gauss-Legendre quadrature; all other independent priors stay exact.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from . import bank


@lru_cache(maxsize=8)
def nodes_and_weights(count):
    if not isinstance(count, int) or count < 16:
        raise ValueError("at least 16 quadrature nodes required")
    return leggauss(count)


def prior_cdf(values, low, high, kind):
    values = np.asarray(values, dtype=np.float64)
    if kind == "log_uniform":
        result = (np.log(np.maximum(values, low)) - np.log(low)) / np.log(high / low)
    else:
        result = (values - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


class ShortVariancePosterior:
    def __init__(self, returns, task, nodes=256):
        bank.validate_task(task)
        if task["model"] != "rough-joint":
            raise ValueError("one-return scalar reference requires the joint rough path task")
        if "drift" in task["parameter_names"] or task["numerics"].get("observation_substeps", 1) != 1:
            raise ValueError("legacy scalar reference requires fixed drift and a single internal step per observation")
        self.returns = np.asarray(returns, dtype=np.float64)
        if self.returns.ndim != 1 or not len(self.returns) or not np.isfinite(self.returns).all():
            raise ValueError("one finite first return per case required")
        self.index = task["parameter_names"].index("short_variance")
        if task["prior_types"][self.index] != "log_uniform":
            raise ValueError("short-variance reference requires its declared log-uniform prior")
        self.low, self.high = task["prior_bounds"][self.index]
        self.a, self.b = np.log([self.low, self.high])
        self.dt, self.mu = task["dt_years"], task["fixed"]["drift"]
        self.nodes, self.weights = nodes_and_weights(nodes)
        log_s = self.a + (self.nodes + 1) * (self.b - self.a) / 2
        self.variance_nodes = np.exp(log_s)
        likelihood = self._log_likelihood(self.variance_nodes[None, :])
        log_weights = np.log(self.weights) + np.log((self.b - self.a) / 2)
        self.log_normalizer = logsumexp(likelihood + log_weights, axis=1)
        self.posterior_weights = np.exp(likelihood + log_weights - self.log_normalizer[:, None])

    def _log_likelihood(self, variance):
        return -.5 * (np.log(2 * np.pi * variance * self.dt)
                      + (self.returns[:, None] - (self.mu - .5 * variance) * self.dt) ** 2
                      / (variance * self.dt))

    def cdf(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.returns.shape or not np.isfinite(values).all():
            raise ValueError("CDF arguments must align with observed cases")
        upper = np.log(np.clip(values, self.low, self.high))
        half_width = (upper - self.a) / 2
        log_s = self.a + (self.nodes[None, :] + 1) * half_width[:, None]
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights)[None, :] + np.log(half_width)[:, None]
        log_mass = logsumexp(self._log_likelihood(np.exp(log_s)) + log_weights, axis=1)
        return np.clip(np.exp(log_mass - self.log_normalizer), 0.0, 1.0)

    def quantile(self, probability):
        if not 0 < probability < 1:
            raise ValueError("quantile probability must lie strictly between zero and one")
        low = np.full(len(self.returns), self.a)
        high = np.full(len(self.returns), self.b)
        for _ in range(42):
            middle = (low + high) / 2
            below = self.cdf(np.exp(middle)) < probability
            low, high = np.where(below, middle, low), np.where(below, high, middle)
        return np.exp((low + high) / 2)

    def moments(self):
        mean = np.sum(self.posterior_weights * self.variance_nodes[None, :], axis=1)
        variance = np.sum(self.posterior_weights * (self.variance_nodes[None, :] - mean[:, None]) ** 2,
                          axis=1)
        return mean, np.sqrt(variance)


def reference(returns, task, nodes=256):
    scalar = ShortVariancePosterior(returns, task, nodes=nodes)
    count, dimensions = len(returns), len(task["parameter_names"])
    mean, sd, low_quantile, high_quantile = (np.empty((count, dimensions)) for _ in range(4))
    for j, ((low, high), kind) in enumerate(zip(task["prior_bounds"], task["prior_types"], strict=True)):
        if kind == "uniform":
            mean[:, j], sd[:, j] = (low + high) / 2, (high - low) / np.sqrt(12)
            low_quantile[:, j], high_quantile[:, j] = low + .05 * (high - low), low + .95 * (high - low)
        else:
            first = (high - low) / np.log(high / low)
            second = (high ** 2 - low ** 2) / (2 * np.log(high / low))
            mean[:, j], sd[:, j] = first, np.sqrt(second - first ** 2)
            low_quantile[:, j], high_quantile[:, j] = low * (high / low) ** .05, low * (high / low) ** .95
    mean[:, scalar.index], sd[:, scalar.index] = scalar.moments()
    low_quantile[:, scalar.index], high_quantile[:, scalar.index] = scalar.quantile(.05), scalar.quantile(.95)
    return {"mean": mean, "sd": sd, "q05": low_quantile, "q95": high_quantile}, scalar


def compare(draws, truth, returns, task, nodes=256):
    draws, truth = np.asarray(draws), np.asarray(truth)
    exact, scalar = reference(returns, task, nodes=nodes)
    coarser, _ = reference(returns, task, nodes=max(16, nodes // 2))
    lower, upper = np.quantile(draws, [.05, .95], axis=1)
    mass = np.empty_like(truth)
    for j, ((low, high), kind) in enumerate(zip(task["prior_bounds"], task["prior_types"], strict=True)):
        if j == scalar.index:
            mass[:, j] = scalar.cdf(upper[:, j]) - scalar.cdf(lower[:, j])
        else:
            mass[:, j] = prior_cdf(upper[:, j], low, high, kind) - prior_cdf(lower[:, j], low, high, kind)
    mean_errors = (draws.mean(axis=1) - exact["mean"]) / exact["sd"]
    width_ratio = (upper - lower) / (exact["q95"] - exact["q05"])
    report = {"reference_scope": "exact one-return posterior; scalar short variance plus five independent priors",
              "quadrature_nodes": nodes, "cases": len(truth), "draws_per_case": draws.shape[1],
              "targets": task["parameter_names"],
              "mean_absolute_mean_error_over_reference_sd": np.mean(np.abs(mean_errors), axis=0).tolist(),
              "rms_mean_error_over_reference_sd": np.sqrt(np.mean(mean_errors ** 2, axis=0)).tolist(),
              "largest_mean_error_over_reference_sd": np.max(np.abs(mean_errors), axis=0).tolist(),
              "mean_neural_90_width_over_reference_width": width_ratio.mean(axis=0).tolist(),
              "mean_reference_mass_in_neural_90_interval": mass.mean(axis=0).tolist(),
              "neural_90_truth_coverage": np.mean((truth >= lower) & (truth <= upper), axis=0).tolist(),
              "reference_90_truth_coverage": np.mean((truth >= exact["q05"]) & (truth <= exact["q95"]), axis=0).tolist(),
              "quadrature_refinement": {
                  "max_mean_shift_over_reference_sd": float(np.max(np.abs(exact["mean"] - coarser["mean"]) / exact["sd"])),
                  "max_quantile_shift_over_reference_sd": float(max(
                      np.max(np.abs(exact[q] - coarser[q]) / exact["sd"]) for q in ("q05", "q95")))},
              "full_history_or_joint_copula_accuracy_certified": False}
    return report, {**exact, "reference_mass_in_neural_interval": mass, "neural_mean_errors_reference_sd": mean_errors}

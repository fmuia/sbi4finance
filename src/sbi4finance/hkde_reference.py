"""Numerical two-parameter HKDE posterior on a rectangular quadrature grid.

The likelihood is the option bank's independent Gaussian currency quote model.
Fourier prices and the posterior integral both remain numerical: refine their
controls separately before treating this as a conditional inference reference.
No market data, latent market truth or physical-measure forecast is implied.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp

from . import hkde
from .option_bank import validate_config
from .option_panel import validate_context


DEFAULT_TARGET_CONTRACTS = (
    {"strike": 95.0, "maturity": 0.75, "option_type": "put"},
    {"strike": 80.0, "maturity": 1.0, "option_type": "put"},
)


def _validate_task(task):
    validate_config({"task": task, "root_seed": 0,
                     "splits": {"train": 1, "validation": 1, "test": 1}})
    if task["model"] != "hkde" or task["parameter_names"] != ["theta", "jump_intensity"]:
        raise ValueError("reference requires HKDE parameter order theta, jump_intensity")


def price_parameters(task, parameter_rows, contracts=None, *, nodes=None, upper_limit=None):
    """Price [N,2] parameter rows into [N,C] currency expectations.

    Defaults to the bank's *refined* Fourier label map. Parameters remain scalar
    per call; contracts of the same option type broadcast together. Optional
    controls support a separate Fourier refinement check, not changed training
    labels. Rows must lie in the declared two-dimensional prior rectangle.
    """
    _validate_task(task)
    rows = np.asarray(parameter_rows, dtype=np.float64)
    bounds = np.asarray(task["prior_bounds"])
    if (rows.ndim != 2 or rows.shape[1] != 2 or not len(rows)
            or not np.isfinite(rows).all() or np.any(rows < bounds[:, 0])
            or np.any(rows > bounds[:, 1])):
        raise ValueError("parameter_rows must be finite [N,2] inside the declared prior")
    contracts = task["context"]["contracts"] if contracts is None else list(contracts)
    context = dict(task["context"], contracts=contracts)
    validate_context(context)
    nodes = task["numerics"]["refined_nodes"] if nodes is None else nodes
    cutoff = task["numerics"]["refined_upper_limit"] if upper_limit is None else upper_limit
    maturities = np.array([c["maturity"] for c in contracts])
    strikes = np.array([c["strike"] for c in contracts])
    kinds = np.array([c["option_type"] for c in contracts])
    forwards = context["spot"] * np.exp((context["rate"] - context["dividend"]) * maturities)
    discounts = np.exp(-context["rate"] * maturities)
    result = np.empty((len(rows), len(contracts)))
    for row, values in enumerate(rows):
        parameters = dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], values)))
        for kind in np.unique(kinds):
            chosen = kinds == kind
            result[row, chosen] = hkde.european_price(
                forwards[chosen], strikes[chosen], discounts[chosen], maturities[chosen],
                parameters, task["model_context"]["initial_variance"], kind=kind,
                nodes=nodes, upper_limit=cutoff)
    return result


def _trapezoid_weights(axis):
    gaps = np.diff(axis)
    return np.r_[gaps[0], gaps[:-1] + gaps[1:], gaps[-1]] / 2


def build_grid(task, n_points, target_contracts=None):
    """Build an endpoint-inclusive tensor grid and uniform-prior quadrature.

    Flattening is C order with theta on axis 0 and jump intensity on axis 1.
    ``quadrature_weights`` integrate against the normalized uniform prior;
    endpoint weights are half the interior weights along each axis. Target
    prices are excluded from conditioning and only propagate posterior draws.
    """
    _validate_task(task)
    if (isinstance(n_points, (bool, np.bool_))
            or not isinstance(n_points, (int, np.integer)) or n_points < 3):
        raise ValueError("n_points must be an integer >=3")
    axes = tuple(np.linspace(low, high, n_points) for low, high in task["prior_bounds"])
    parameters = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    one_dimensional = [_trapezoid_weights(axis) / (axis[-1] - axis[0]) for axis in axes]
    quadrature_weights = np.outer(*one_dimensional).ravel()
    targets = [dict(c) for c in (DEFAULT_TARGET_CONTRACTS if target_contracts is None else target_contracts)]
    if not targets:
        raise ValueError("at least one target contract required")
    contracts = task["context"]["contracts"]
    prices = price_parameters(task, parameters, [*contracts, *targets])
    return {"axes": axes, "shape": (n_points, n_points), "parameters": parameters,
            "parameter_names": list(task["parameter_names"]),
            "prices": prices[:, :len(contracts)], "target_prices": prices[:, len(contracts):],
            "target_contracts": targets, "quadrature_weights": quadrature_weights}


def _weights(weights, count):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (count,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError("finite nonnegative weights with positive total required")
    return weights / weights.sum()


def posterior(grid, observation, sigma):
    """Return normalized flattened quadrature masses for one complete panel.

    These are density times integration weights, not raw likelihood values.
    A scalar positive ``sigma`` is the declared independent quote-noise standard
    deviation in currency; numerical pricing uncertainty is not silently added.
    """
    prices = np.asarray(grid["prices"], dtype=np.float64)
    observed = np.asarray(observation, dtype=np.float64)
    if observed.ndim == 2 and observed.shape[1] == 1:
        observed = observed[:, 0]
    if (prices.ndim != 2 or not all(prices.shape) or not np.isfinite(prices).all()
            or observed.shape != (prices.shape[1],) or not np.isfinite(observed).all()):
        raise ValueError("finite prices [N,C] and one observation [C] or [C,1] required")
    if isinstance(sigma, (bool, np.bool_)) or np.ndim(sigma) != 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite scalar")
    quadrature = _weights(grid["quadrature_weights"], len(prices))
    if np.any(quadrature == 0):
        raise ValueError("quadrature weights must be strictly positive")
    with np.errstate(over="ignore"):
        log_mass = np.log(quadrature) - 0.5 * np.sum(((prices - observed) / sigma) ** 2, axis=1)
    normalizer = logsumexp(log_mass)
    if not np.isfinite(normalizer):
        raise ValueError("nonfinite likelihood normalization; observation/sigma outside numerical range")
    masses = np.exp(log_mass - normalizer)
    return masses / masses.sum()


def _probabilities(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or not probs.size or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("probs must be a nonempty vector in [0,1]")
    return probs


def weighted_quantiles(values, weights, probs=(0.05, 0.5, 0.95)):
    """Inverse weighted empirical CDF, without interpolation between prices.

    The output is [P] for values [N], or [D,P] for values [N,D]. Each target
    uses the same *joint* parameter weights. This finite-grid convention has
    resolution error, assessed by grid refinement; it is not the continuous
    marginal-density convention used for parameter quantiles below.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (1, 2) or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("finite nonempty values [N] or [N,D] required")
    probabilities = _probabilities(probs)
    weights = _weights(weights, len(values))
    columns = values[:, None] if values.ndim == 1 else values
    result = []
    positive = weights > 0
    for column in columns.T:
        order = np.argsort(column[positive], kind="stable")
        ordered = column[positive][order]
        cdf = np.cumsum(weights[positive][order])
        cdf[-1] = 1.0
        index = np.minimum(np.searchsorted(cdf, probabilities, side="left"), len(cdf) - 1)
        result.append(ordered[index])
    result = np.asarray(result)
    return result[0] if values.ndim == 1 else result


def summarize(grid, weights, probs=(0.05, 0.5, 0.95)):
    """Joint moments, continuous marginal CDFs and propagated target summaries.

    Marginal densities remove their one-dimensional trapezoid integration
    weights before cumulative trapezoid integration. Quantiles interpolate
    those CDFs, avoiding the endpoint bias of a point-mass parameter CDF.
    All moments and target prices retain the joint posterior dependence.
    """
    axes = tuple(np.asarray(axis, dtype=np.float64) for axis in grid["axes"])
    if len(axes) != 2 or any(axis.ndim != 1 or len(axis) < 3
                             or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0)
                             for axis in axes):
        raise ValueError("two finite strictly increasing axes with at least three points required")
    parameters = np.asarray(grid["parameters"], dtype=np.float64)
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    if not np.array_equal(parameters, expected):
        raise ValueError("parameters must follow the axes' C-order tensor grid")
    weights = _weights(weights, len(parameters))
    probabilities = _probabilities(probs)
    mean = weights @ parameters
    centered = parameters - mean
    covariance = centered.T @ (weights[:, None] * centered)
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    correlation = float(covariance[0, 1] / (std[0] * std[1])) if np.all(std > 0) else None
    joint = weights.reshape(tuple(len(axis) for axis in axes))
    densities, cdfs, quantiles = [], [], []
    for dimension, axis in enumerate(axes):
        density = joint.sum(axis=1 - dimension) / _trapezoid_weights(axis)
        cdf = cumulative_trapezoid(density, axis, initial=0)
        cdf /= cdf[-1]
        quantile = np.interp(probabilities, cdf, axis)
        quantile[probabilities == 0] = axis[0]
        quantile[probabilities == 1] = axis[-1]
        densities.append(density)
        cdfs.append(cdf)
        quantiles.append(quantile)
    targets = np.asarray(grid["target_prices"], dtype=np.float64)
    if targets.ndim != 2 or targets.shape[0] != len(weights) or not targets.shape[1] or not np.isfinite(targets).all():
        raise ValueError("finite target_prices [N,D] required")
    target_mean = weights @ targets
    target_std = np.sqrt(weights @ ((targets - target_mean) ** 2))
    return {"mean": mean, "std": std, "covariance": covariance, "correlation": correlation,
            "probabilities": probabilities, "marginal_quantiles": np.asarray(quantiles),
            "marginal_axes": axes, "marginal_densities": tuple(densities),
            "marginal_cdfs": tuple(cdfs), "target_mean": target_mean,
            "target_std": target_std, "target_quantiles": weighted_quantiles(targets, weights, probabilities)}

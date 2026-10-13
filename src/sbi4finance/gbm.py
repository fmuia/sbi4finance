"""Exact GBM simulation and small analytic inference/pricing references.

Parameters are annualized and ``dt`` contains year fractions. Histories always
have shape [batch, time, 1] and contain decimal log returns, not simple returns.
These functions implement explicit mathematical inputs; they do not draw priors,
read data, train networks or infer a physical/risk-neutral measure relationship.
"""

import math

import numpy as np


def _vector(value, name, *, positive=False, nonnegative=False):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim > 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or nonempty 1D array")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return np.atleast_1d(array)


def _scalar(value, name, **kwargs):
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(_vector(value, name, **kwargs)[0])


def _time_steps(dt):
    if np.ndim(dt) != 1:
        raise ValueError("dt must be a nonempty 1D array of positive year fractions")
    return _vector(dt, "dt", positive=True)


def _returns(log_returns):
    values = np.asarray(log_returns, dtype=np.float64)
    if (values.ndim != 3 or values.shape[-1] != 1
            or values.shape[0] == 0 or values.shape[1] == 0
            or not np.isfinite(values).all()):
        raise ValueError("log_returns must be finite with shape [batch, time, 1]")
    return values


def _history(log_returns, dt):
    values, steps = _returns(log_returns), _time_steps(dt)
    if values.shape[1] != len(steps):
        raise ValueError("dt must have one year fraction per return")
    return values[:, :, 0], steps


def simulate_log_returns(drift, volatility, dt, *, rng):
    """Draw exact increments, returning float64 [batch, time, 1].

    ``drift`` is proportional price drift, so the log-return drift is
    ``drift - volatility**2 / 2``. Drift/volatility are scalars or broadcastable
    1D batch vectors, constant through each path. ``dt`` is shared by the batch.
    Under Q the caller explicitly supplies the appropriate price drift.

    Zero volatility is supported. An explicit NumPy Generator owns randomness;
    time-major draws allow splitting a path across calls with the same generator
    and batch order. Independent example-ID streams belong in the task layer.
    """
    mu, sigma = np.broadcast_arrays(
        _vector(drift, "drift"),
        _vector(volatility, "volatility", nonnegative=True),
    )
    steps = _time_steps(dt)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    z = rng.standard_normal((len(steps), len(mu))).T
    result = ((mu[:, None] - 0.5 * sigma[:, None] ** 2) * steps
              + sigma[:, None] * np.sqrt(steps) * z)
    return result[:, :, None]


def prices_from_log_returns(log_returns, initial_price):
    """Return [batch, time + 1, 1] prices, including the initial price.

    ``initial_price`` is a positive scalar or one value per batch member. To
    continue a simulated price path, pass its terminal price and future returns.
    GBM has no hidden variance state when volatility is known and constant.
    """
    values = _returns(log_returns)
    spot = np.broadcast_to(
        _vector(initial_price, "initial_price", positive=True), (values.shape[0],)
    )
    cumulative = np.concatenate(
        (np.zeros((len(spot), 1, 1)), np.cumsum(values, axis=1)), axis=1
    )
    with np.errstate(over="raise"):
        return spot[:, None, None] * np.exp(cumulative)


def drift_log_likelihood(log_returns, drift, volatility, dt):
    """Gaussian log likelihood per history for known positive volatility.

    Histories and drift/volatility batch vectors broadcast. In particular a
    single history can be evaluated at a vector of candidate drifts. The result
    is a 1D array. Zero volatility is deliberately excluded from this density.
    """
    values, steps = _history(log_returns, dt)
    mu, sigma = np.broadcast_arrays(
        _vector(drift, "drift"), _vector(volatility, "volatility", positive=True)
    )
    residual = values - (mu[:, None] - 0.5 * sigma[:, None] ** 2) * steps
    standardized = residual / (sigma[:, None] * np.sqrt(steps))
    log_variance = 2 * np.log(sigma[:, None]) + np.log(steps)
    return -0.5 * np.sum(np.log(2 * np.pi) + log_variance + standardized**2, axis=1)


def drift_posterior(log_returns, volatility, dt, *, prior_mean, prior_std):
    """Exact posterior (mean, standard deviation), both shape [batch].

    The inferred proportional drift has prior N(prior_mean, prior_std**2).
    Volatility and prior parameters are shared scalars, with positive volatility
    and prior standard deviation. All observations must be present. Irregular
    positive intervals are supported; uncertainty depends on total elapsed time.
    """
    values, steps = _history(log_returns, dt)
    sigma = _scalar(volatility, "volatility", positive=True)
    m0 = _scalar(prior_mean, "prior_mean")
    s0 = _scalar(prior_std, "prior_std", positive=True)
    duration = steps.sum()
    variance = 1 / (1 / s0**2 + duration / sigma**2)
    mean = variance * (m0 / s0**2 + (values.sum(axis=1) + 0.5 * sigma**2 * duration) / sigma**2)
    return mean, np.full_like(mean, np.sqrt(variance))


def predictive_log_return(posterior_mean, posterior_std, volatility, horizon):
    """Mean and standard deviation of the total posterior-predictive log return.

    Integrates one fixed uncertain drift per future path. The variance is
    sigma**2 * horizon + posterior_std**2 * horizon**2. Inputs broadcast as 1D
    batch vectors; horizon is a nonnegative scalar in years. This is a P forecast
    only when the supplied posterior describes P dynamics.
    """
    mean, std, sigma = np.broadcast_arrays(
        _vector(posterior_mean, "posterior_mean"),
        _vector(posterior_std, "posterior_std", nonnegative=True),
        _vector(volatility, "volatility", nonnegative=True),
    )
    h = _scalar(horizon, "horizon", nonnegative=True)
    return (mean - 0.5 * sigma**2) * h, np.sqrt(sigma**2 * h + std**2 * h**2)


def _normal_cdf(values):
    # The small reference pricer needs only NumPy + the standard library.
    return np.fromiter(
        (0.5 * math.erfc(-x / math.sqrt(2)) for x in values.flat),
        dtype=np.float64, count=values.size,
    ).reshape(values.shape)


def european_price(forward, strike, discount, integrated_variance, *, option="call"):
    """European Q reference price for deterministic variance and financing.

    Inputs broadcast across contracts: F > 0, K > 0, D > 0 and w >= 0.
    With constant coefficients, F = spot * exp((rate - dividend_yield) * T),
    D = exp(-rate * T), and w = volatility**2 * T. More generally w is the
    integral of instantaneous variance, not squared integrated volatility.

    At w = 0 return D times the deterministic payoff. Expiry is represented by
    F = spot, D = 1, w = 0. Returns an array of the broadcast input shape (0D for
    scalar inputs). This pricing context is distinct from the P drift posterior.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    fwd, k, disc, variance = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float64)
          for x in (forward, strike, discount, integrated_variance)]
    )
    for name, value in (("forward", fwd), ("strike", k), ("discount", disc)):
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("integrated_variance must be finite and nonnegative")
    root = np.sqrt(variance)
    d1 = (np.log(fwd) - np.log(k)) / np.where(root > 0, root, 1) + 0.5 * root
    d2 = d1 - root
    if option == "call":
        value = disc * (fwd * _normal_cdf(d1) - k * _normal_cdf(d2))
        deterministic = disc * np.maximum(fwd - k, 0)
    else:
        value = disc * (k * _normal_cdf(-d2) - fwd * _normal_cdf(-d1))
        deterministic = disc * np.maximum(k - fwd, 0)
    return np.where(variance > 0, value, deterministic)

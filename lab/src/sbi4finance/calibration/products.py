"""Products the smile does not pin down, priced under posterior draws with common random numbers.

All prices are in forward units with zero carry. Paths carry, per time point,
the integrated variance I(t) = int_0^t V ds and the variance-driver integral
J(t) = int_0^t sqrt(V) dW, where W drives the variance. Conditional on the
variance path, log(S_T2 / S_T1) is Gaussian with

    mean  rho (J(T2) - J(T1)) - rho^2/2 (I(T2) - I(T1)) - (1 - rho^2)/2 (I(T2) - I(T1)),
    var   (1 - rho^2) (I(T2) - I(T1)),

so every product below is a Black–Scholes formula per path (design document,
Appendix D.3 and H). Two zero-mean controls per horizon remove the linear part of
the remaining noise: the effective forward minus one, and I minus its mean where
the scheme's mean of I is exact (rough Bergomi's left-point hybrid scheme). Under
Heston's full-truncation Euler scheme the continuous-time mean of I is not the
scheme's (1.8% off at vol-of-vol 1.5), so the second control is the increment of
J, whose mean is zero under any scheme.

The same normals are used for every posterior draw (common random numbers), so
the spread of prices across draws is the spread due to the parameters.
"""

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..bergomi import curve_values
from ..rough_bergomi import simulate_rough

PRODUCTS = ("variance_swap", "volatility_swap", "forward_start_call", "put_out_of_grid")


def _normals(n_paths, n_steps, seed):
    """Antithetic standard normals [n_paths, n_steps, 3]: the second half is minus the first."""
    z = np.random.default_rng(seed).standard_normal((n_paths // 2, n_steps, 3))
    return np.concatenate([z, -z])


def rough_paths(parameters, curve, horizon, n_steps, n_paths, seed):
    dt = horizon / n_steps
    out = simulate_rough(parameters, curve, dt, _normals(n_paths, n_steps, seed), method="hybrid")
    zeros = np.zeros((n_paths, 1))
    times = dt * np.arange(n_steps + 1)
    mean_i = np.r_[0.0, np.cumsum(curve_values(curve, times[:-1]) * dt)]
    return {"times": times, "I": np.hstack([zeros, out["integrated_variance"]]),
            "J": np.hstack([zeros, out["volatility_driver_integral"]]), "mean_I": mean_i}


def heston_paths(parameters, initial_variance, horizon, n_steps, n_paths, seed):
    """Full-truncation Euler for Heston, the same output layout as rough_paths."""
    kappa, theta, xi = parameters["kappa"], parameters["theta"], parameters["vol_of_vol"]
    dt = horizon / n_steps
    z = _normals(n_paths, n_steps, seed)[..., 0]
    v = np.full(n_paths, float(initial_variance))
    integrated, driver = np.zeros((n_paths, n_steps + 1)), np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        positive = np.maximum(v, 0.0)
        root = np.sqrt(positive * dt)
        integrated[:, i + 1] = integrated[:, i] + positive * dt
        driver[:, i + 1] = driver[:, i] + root * z[:, i]
        v = v + kappa * (theta - positive) * dt + xi * root * z[:, i]
    times = dt * np.arange(n_steps + 1)
    mean_v = theta + (initial_variance - theta) * np.exp(-kappa * times)
    return {"times": times, "I": integrated, "J": driver,
            "mean_I": theta * times + (initial_variance - theta) * (-np.expm1(-kappa * times)) / kappa,
            "mean_I_is_exact": False, "_mean_v": mean_v}


def _index(paths, t):
    i = int(np.rint(t / paths["times"][1]))
    if not np.isclose(paths["times"][i], t, rtol=0, atol=1e-9):
        raise ValueError("product dates must land on the path grid")
    return i


def _controlled_mean(values, controls):
    """Mean of values with zero-mean controls removed by least squares, and its standard error.

    Antithetic halves are averaged first, so pairs are the iid units.
    """
    half = len(values) // 2
    values = 0.5 * (values[:half] + values[half:])
    controls = 0.5 * (controls[:half] + controls[half:])
    centred = controls - controls.mean(axis=0)
    beta = np.linalg.lstsq(centred, values - values.mean(), rcond=None)[0]
    adjusted = values - controls @ beta
    return float(adjusted.mean()), float(adjusted.std(ddof=controls.shape[1] + 1) / np.sqrt(len(adjusted)))


def _conditional(paths, rho, start, end):
    """The conditional law of log(S_end / S_start) and the second control for the interval."""
    i, j = _index(paths, start), _index(paths, end)
    di, dj = paths["I"][:, j] - paths["I"][:, i], paths["J"][:, j] - paths["J"][:, i]
    log_forward = rho * dj - 0.5 * rho**2 * di
    control = di - (paths["mean_I"][j] - paths["mean_I"][i]) if paths.get("mean_I_is_exact", True) else dj
    return log_forward, (1 - rho**2) * di, control


def _bs_call(log_forward, log_strike, total_variance):
    sd = np.sqrt(np.maximum(total_variance, 1e-300))
    d1 = (log_forward - log_strike + 0.5 * total_variance) / sd
    return np.exp(log_forward + log_ndtr(d1)) - np.exp(log_strike) * ndtr(d1 - sd)


def forward_start_call(paths, rho, start, end, moneyness=1.0):
    """E[(S_end / S_start - moneyness)^+], the strike set at start as moneyness times S_start."""
    log_forward, variance, control = _conditional(paths, rho, start, end)
    payoff = _bs_call(log_forward, np.log(moneyness), variance)
    controls = np.column_stack([np.exp(log_forward) - 1.0, control])
    return _controlled_mean(payoff, controls)


def vanilla(paths, rho, strike, maturity, kind):
    log_forward, variance, control = _conditional(paths, rho, 0.0, maturity)
    call = _bs_call(log_forward, np.log(strike), variance)
    controls = np.column_stack([np.exp(log_forward) - 1.0, control])
    price, se = _controlled_mean(call, controls)
    return (price - 1.0 + strike if kind == "put" else price), se


def volatility_swap(paths, maturity):
    """E[sqrt(I(T) / T)]: the fair strike of a volatility swap on continuous realized variance."""
    i = _index(paths, maturity)
    realized = paths["I"][:, i] / maturity
    control = realized - paths["mean_I"][i] / maturity if paths.get("mean_I_is_exact", True) else paths["J"][:, i]
    return _controlled_mean(np.sqrt(realized), control[:, None])


def variance_swap(paths, maturity):
    """E[I(T) / T], from the exact mean of the path law; zero range under rough Bergomi by construction."""
    return float(paths["mean_I"][_index(paths, maturity)] / maturity), 0.0


def default_specification():
    """The design document's products: 63-day swaps, a 63-to-126-day forward start, a 2-year 90% put."""
    return {"variance_swap": {"maturity": 63 / 252}, "volatility_swap": {"maturity": 63 / 252},
            "forward_start_call": {"start": 63 / 252, "end": 126 / 252, "moneyness": 1.0},
            "put_out_of_grid": {"strike": 0.9, "maturity": 2.0},
            "steps_per_year": 252}


def price_draw(model, parameters, context, specification, n_paths, seed):
    """Prices and standard errors of every product for one parameter set.

    model is 'rough' (parameters hurst, eta, rho; context a curve dict that must
    cover two years) or 'heston' (parameters as hkde with initial_variance).
    """
    spec = specification
    steps_per_year = spec["steps_per_year"]
    horizon_short = spec["forward_start_call"]["end"]
    horizon_long = spec["put_out_of_grid"]["maturity"]
    if model == "rough":
        def make(horizon):
            return rough_paths(parameters, context, horizon, int(round(horizon * steps_per_year)), n_paths, seed)
    else:
        heston = {k: parameters[k] for k in ("kappa", "theta", "vol_of_vol")}
        v0 = parameters["initial_variance"]

        def make(horizon):
            return heston_paths(heston, v0, horizon, int(round(horizon * steps_per_year)), n_paths, seed)
    short, long = make(horizon_short), make(horizon_long)
    rho = parameters["rho"]
    results = {
        "variance_swap": variance_swap(short, spec["variance_swap"]["maturity"]),
        "volatility_swap": volatility_swap(short, spec["volatility_swap"]["maturity"]),
        "forward_start_call": forward_start_call(short, rho, spec["forward_start_call"]["start"],
                                                 spec["forward_start_call"]["end"],
                                                 spec["forward_start_call"]["moneyness"]),
        "put_out_of_grid": vanilla(long, rho, spec["put_out_of_grid"]["strike"], horizon_long, "put"),
    }
    return np.array([results[p][0] for p in PRODUCTS]), np.array([results[p][1] for p in PRODUCTS])


def extend_curve(curve, horizon):
    """Hold the last forward-variance level flat out to the horizon (the extrapolation is stated, not inferred)."""
    times, values = list(curve["times"]), list(curve["values"])
    if times[-1] < horizon:
        times.append(float(horizon))
        values.append(values[-1])
    return {"times": times, "values": values}


def reserve(prices, point, side="short", level=0.9):
    """Distance from the point price to the prudent quantile on the side that hurts the position."""
    prices = np.asarray(prices, dtype=float)
    if side == "short":
        return float(np.quantile(prices, level) - point)
    return float(point - np.quantile(prices, 1 - level))

"""What rough Bergomi shares with one-factor Bergomi: the forward-variance curve, the price path and the
conditional put estimator.

A put is priced conditionally on the variance path: given int v dt and int sqrt(v) dW, the price is lognormal,
so Black's formula integrates the independent noise out. The estimator is bounded, so its Monte Carlo standard
errors are the ordinary ones. Rough simulation and lab production workers
reuse these same curve, correlation and conditional-payoff functions.
"""

import numpy as np
from scipy.special import log_ndtr, ndtr


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def curve_values(curve, times):
    """Evaluate an explicit, positive piecewise-linear initial variance curve.

    Require {times, values}, times[0] == 0, and coverage of all queried times.
    No extrapolation or market curve estimation is performed.
    """
    if not isinstance(curve, dict) or set(curve) != {"times", "values"}:
        raise ValueError("curve needs exactly times and values")
    x, y = np.asarray(curve["times"], float), np.asarray(curve["values"], float)
    t = np.asarray(times, float)
    if (x.ndim != 1 or len(x) < 2 or y.shape != x.shape or x[0] != 0
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0) or np.any(y <= 0)
            or not np.isfinite(t).all() or np.any(t < 0) or np.any(t > x[-1] + 1e-12)):
        raise ValueError("positive finite curve, increasing times from zero, and full horizon coverage required")
    return np.interp(t, x, y)


def _model_parameters(parameters, kind):
    names = {"eta", "rho", "kappa" if kind == "one_factor" else "hurst"}
    if not isinstance(parameters, dict) or set(parameters) != names:
        raise ValueError(f"{kind} parameters require exactly {sorted(names)}")
    p = {name: _scalar(parameters[name], name) for name in names}
    if p["eta"] < 0 or not -1 <= p["rho"] <= 0:
        raise ValueError("require eta >= 0 and -1 <= rho <= 0 (supported pricing domain)")
    if kind == "one_factor" and p["kappa"] <= 0:
        raise ValueError("require kappa > 0")
    if kind == "rough" and not 0 < p["hurst"] < 0.5:
        raise ValueError("require 0 < hurst < 0.5")
    return p


def _normals(innovations, n_steps=None):
    z = np.asarray(innovations, dtype=float)
    if (z.ndim != 3 or z.shape[0] < 1 or z.shape[1] < 1 or z.shape[2] != 3
            or (n_steps is not None and z.shape[1] != n_steps) or not np.isfinite(z).all()):
        raise ValueError("innovations must be finite independent standard normals [paths,steps,3]")
    return z


def _q_inputs(spot, rate, dividend):
    s, r, q = [_scalar(v, n) for v, n in ((spot, "spot"), (rate, "rate"), (dividend, "dividend"))]
    if s <= 0:
        raise ValueError("spot must be positive")
    return s, r, q


def _asset_paths(variance, dt, normals, rho, spot, rate, dividend, *, conditional_only=False):
    """Adapted left-point log Euler; same W drives leverage and variance."""
    left = variance[:, :-1]
    integrated = np.cumsum(left * dt, axis=1)
    driver_integral = np.cumsum(np.sqrt(left * dt) * normals[..., 0], axis=1)
    if conditional_only:
        if not all(np.isfinite(a).all() for a in (variance, integrated, driver_integral)):
            raise ValueError("nonfinite trajectory; numerical support exceeded, do not redraw silently")
        # Conditional European pricing integrates W-perp out. Avoid materializing
        # asset returns/log spots when a conservative scalar bound certifies that
        # the omitted calculations would remain finite. The integrated variance
        # bounds every nonnegative left*dt term. A generous floating-point margin
        # covers rounding in the bound and cumulative sums. Extreme/uncertain
        # inputs fall through to the original calculations and failure checks.
        with np.errstate(over="ignore", invalid="ignore"):
            normal_bound = max(np.max(normals), -np.min(normals))
            driver_bound = (abs(rho) + np.sqrt(1 - rho * rho)) * normal_bound
            drift_bound = abs(rate) + abs(dividend) + 0.5 * np.max(variance)
            return_bound = drift_bound * np.max(dt) + np.sqrt(np.max(integrated)) * driver_bound
            path_bound = abs(np.log(spot)) + normals.shape[1] * return_bound
        bounds = (driver_bound, drift_bound, return_bound, path_bound)
        if np.isfinite(bounds).all() and max(bounds) < np.finfo(float).max / 8:
            return {"integrated_variance": integrated, "volatility_driver_integral": driver_integral}
    returns = ((rate - dividend - 0.5 * left) * dt
               + np.sqrt(left * dt) * (rho * normals[..., 0]
                                      + np.sqrt(1 - rho * rho) * normals[..., 2]))
    log_spot = np.column_stack((np.full(len(normals), np.log(spot)),
                               np.log(spot) + np.cumsum(returns, axis=1)))
    if not all(np.isfinite(a).all() for a in (variance, returns, integrated, driver_integral, log_spot)):
        raise ValueError("nonfinite trajectory; numerical support exceeded, do not redraw silently")
    if conditional_only:
        return {"integrated_variance": integrated, "volatility_driver_integral": driver_integral}
    return {"log_returns": returns[..., None], "log_spot": log_spot,
            "variance": variance, "integrated_variance": integrated,
            "volatility_driver_integral": driver_integral}


def conditional_put_payoffs(paths, parameters, contracts, indices, *, spot, rate, dividend):
    """Bounded discounted conditional puts, integrating independent W-perp.

    Each column lies in [0, K exp(-rT)]. This finite-variance estimator justifies
    ordinary iid Monte Carlo standard errors even where raw calls lack a second
    moment. It is a price estimator conditional on one driver path, not one
    realized payoff used as a noiseless price label.
    """
    rho = parameters["rho"]
    results = []
    for contract, index in zip(contracts, indices):
        k, maturity = contract["strike"], contract["maturity"]
        iv = paths["integrated_variance"][:, index - 1]
        iw = paths["volatility_driver_integral"][:, index - 1]
        log_forward = np.log(spot) + (rate - dividend) * maturity + rho * iw - 0.5 * rho ** 2 * iv
        residual_variance = (1 - rho ** 2) * iv
        if abs(rho) == 1:
            value = np.exp(-rate * maturity) * (k - np.exp(np.minimum(log_forward, np.log(k))))
        else:
            sd = np.sqrt(residual_variance)
            d1 = (log_forward - np.log(k) + 0.5 * residual_variance) / sd
            d2 = d1 - sd
            value = k * np.exp(-rate * maturity) * ndtr(-d2) - np.exp(log_forward - rate * maturity + log_ndtr(-d1))
            # Roundoff in a nonnegative bounded analytic expectation only.
            if np.any(value < -1e-12 * max(k, spot)):
                raise ValueError("conditional put arithmetic lost precision")
            value = np.maximum(value, 0)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite conditional puts; numerical support exceeded")
        results.append(value)
    return np.column_stack(results)

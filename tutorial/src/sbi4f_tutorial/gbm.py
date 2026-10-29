"""Black's formula for a known integrated variance: the lab's sbi4finance/gbm.py, the functions hkde.py uses.

hkde.european_price returns it when the vol of vol and the jump intensity are zero, where the integrated
variance is deterministic. The functions are copied unchanged; lab/tests/test_tutorial_parity.py checks them.
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

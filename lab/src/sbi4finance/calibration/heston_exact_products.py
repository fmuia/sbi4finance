"""Exact Heston prices of the SPX lesson's forward-start call and volatility swap.

Both products are priced by deterministic quadrature, with no time stepping and no Monte
Carlo error, in forward units with zero carry. The variance follows

    dV = kappa (theta - V) dt + vol_of_vol sqrt(V) dW,   V(0) = initial_variance,

correlated with the index by rho.

Forward-start call, E[max(S_expiry / S_reset - 1, 0)]. Given the variance at the reset date,
the payoff is a Heston at-the-money call over expiry - reset, whose characteristic function is
exp(A(u) + B(u) V_reset). V_reset has the exact CIR law (a scaled noncentral chi-square), so
its moment generating function integrates the variance out in closed form. The price is then
one Lewis integral,

    1 - (1/pi) int_0^inf Re[exp(A(u - i/2)) E exp(B(u - i/2) V_reset)] / (u^2 + 1/4) du.

Volatility swap, E[sqrt(I_T / T)] with I_T the integrated variance, from the closed-form CIR
Laplace transform L(s) = E[exp(-s I_T / T)] and

    E[sqrt(X)] = (1 / sqrt(pi)) int_0^inf (1 - L(x^2)) / x^2 dx.

Each integral is split at a scale point, and its infinite part is mapped onto (0, 1] by
x = scale / t, so nothing is truncated. Gauss-Legendre nodes are applied, then doubled; the
functions return the doubled-node value and the absolute change, a quadrature diagnostic
that callers can report or bound.

The tutorial runtime's Heston characteristic function returns exp(A + B v0) only, and its
source is verified byte for byte by the pretrained SPX bundle, so it is not refactored to
expose A and B. ``heston_exponent`` re-derives them in the same little-trap form, and
``tests/test_heston_exact_products.py`` checks it against that function.
"""

from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

_CHUNK = 512  # parameter sets per vectorized block


def _broadcast(parameters, initial_variance, keys):
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of Heston parameters")
    values = [np.asarray(parameters[key], dtype=float) for key in keys]
    values.append(np.asarray(initial_variance, dtype=float))
    try:
        values = np.broadcast_arrays(*values)
    except ValueError as error:
        raise ValueError("parameter batches must broadcast") from error
    shape = values[0].shape
    if len(shape) > 1 or any(not np.isfinite(value).all() for value in values):
        raise ValueError("parameters must be finite scalars or 1-D batches")
    values = [np.atleast_1d(value).astype(float) for value in values]
    named = dict(zip((*keys, "initial_variance"), values))
    if (np.any(named["kappa"] <= 0) or np.any(named["theta"] <= 0) or np.any(named["vol_of_vol"] <= 0)
            or np.any(named["initial_variance"] < 0)
            or ("rho" in named and np.any(np.abs(named["rho"]) >= 1))):
        raise ValueError("require kappa, theta, vol_of_vol > 0, initial_variance >= 0 and |rho| < 1")
    return named, shape


def _horizon(value, label, *, positive):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


@lru_cache(maxsize=16)
def _legendre(count):
    """Gauss-Legendre nodes and weights on [0, 1]."""
    roots, weights = roots_legendre(int(count))
    nodes, weights = 0.5 * (roots + 1), 0.5 * weights
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def heston_exponent(u, maturity, kappa, theta, vol_of_vol, rho):
    """A(u), B(u) with E[exp(i u log(S_T / S_0))] = exp(A + B v0) under Heston, zero carry.

    The little-trap form of the tutorial runtime's characteristic function, broadcasting over
    frequencies and parameters. Valid for -1 <= Im(u) <= 0 away from u = 0 and u = -i, where
    Re(B) <= 0.
    """
    iu = 1j * u
    b = kappa - rho * vol_of_vol * iu
    d = np.sqrt(b * b + vol_of_vol * vol_of_vol * (u * u + iu))
    a = -(u * u + iu) / (b + d)  # (b - d) / vol_of_vol^2 without cancellation
    g = vol_of_vol * vol_of_vol * a / (b + d)
    decay = np.exp(-d * maturity)
    log_ratio = np.log1p(-g * decay) - np.log1p(-g)
    exponent_a = kappa * theta * (a * maturity - 2 * log_ratio / (vol_of_vol * vol_of_vol))
    exponent_b = a * (-np.expm1(-d * maturity)) / (1 - g * decay)
    return exponent_a, exponent_b


def cir_log_mgf(z, horizon, initial_variance, kappa, theta, vol_of_vol):
    """log E[exp(z V_horizon)] for the exact CIR transition law, for Re(z) <= 0.

    V_horizon is s times a noncentral chi-square with 4 kappa theta / vol_of_vol^2 degrees of
    freedom and noncentrality initial_variance exp(-kappa horizon) / s, where
    s = vol_of_vol^2 (1 - exp(-kappa horizon)) / (4 kappa).
    """
    scale = vol_of_vol * vol_of_vol * (-np.expm1(-kappa * horizon)) / (4 * kappa)
    return (initial_variance * np.exp(-kappa * horizon) * z / (1 - 2 * z * scale)
            - 2 * kappa * theta / (vol_of_vol * vol_of_vol) * np.log1p(-2 * z * scale))


def _forward_start_block(p, reset, tenor, nodes, split):
    def integrand(u):
        z = u[None, :] - 0.5j
        column = {key: value[:, None] for key, value in p.items()}
        exponent_a, exponent_b = heston_exponent(z, tenor, column["kappa"], column["theta"],
                                                 column["vol_of_vol"], column["rho"])
        exponent = exponent_a + cir_log_mgf(exponent_b, reset, column["initial_variance"], column["kappa"],
                                            column["theta"], column["vol_of_vol"])
        return np.real(np.exp(exponent)) / (u[None, :] ** 2 + 0.25)

    near, near_weights = _legendre(nodes[0])
    far, far_weights = _legendre(nodes[1])
    u = split * near
    integral = integrand(u) @ (split * near_weights)
    u = split / far  # [split, inf) onto (0, 1]
    integral += integrand(u) @ (far_weights * split / far ** 2)
    return 1 - integral / np.pi


def forward_start_call(parameters, initial_variance, reset, expiry, *, nodes=(128, 256), split=25.0):
    """Price of max(S_expiry / S_reset - 1, 0) per unit notional, and its quadrature change.

    parameters maps kappa, theta, vol_of_vol and rho to scalars or equal-length vectors;
    initial_variance broadcasts with them. Returns (price, change): the price with doubled
    nodes and its absolute change from the base nodes, both in units of notional.
    """
    p, shape = _broadcast(parameters, initial_variance, ("kappa", "theta", "vol_of_vol", "rho"))
    reset = _horizon(reset, "reset", positive=False)
    tenor = _horizon(expiry, "expiry", positive=True) - reset
    if tenor <= 0:
        raise ValueError("expiry must follow reset")
    if split <= 0 or len(nodes) != 2 or min(nodes) < 8:
        raise ValueError("require a positive split and two node counts of at least 8")
    price, change = np.empty_like(p["kappa"]), np.empty_like(p["kappa"])
    for start in range(0, price.size, _CHUNK):
        block = {key: value[start:start + _CHUNK] for key, value in p.items()}
        base = _forward_start_block(block, reset, tenor, nodes, split)
        refined = _forward_start_block(block, reset, tenor, (2 * nodes[0], 2 * nodes[1]), split)
        price[start:start + _CHUNK], change[start:start + _CHUNK] = refined, np.abs(refined - base)
    if not (np.isfinite(price).all() and np.all((price > 0) & (price < 1))):
        raise ValueError("forward-start quadrature left the price bounds; parameters unsupported")
    return price.reshape(shape), change.reshape(shape)


def integrated_variance_log_laplace(lam, maturity, initial_variance, kappa, theta, vol_of_vol):
    """log E[exp(-lam int_0^T V dt)] for CIR variance, lam >= 0.

    The bond-price formula of Cox, Ingersoll and Ross with the short rate lam V, rearranged so
    that it stays accurate for small lam and small vol_of_vol and finite for large lam.
    """
    gamma = np.sqrt(kappa * kappa + 2 * vol_of_vol * vol_of_vol * lam)
    gap = 2 * vol_of_vol * vol_of_vol * lam / (gamma + kappa)  # gamma - kappa
    growth = -np.expm1(-gamma * maturity)  # 1 - exp(-gamma T)
    denominator = gamma * (2 - growth) + kappa * growth
    b = 2 * lam * growth / denominator
    a = (-2 * kappa * theta * lam * maturity / (gamma + kappa)
         - 2 * kappa * theta / (vol_of_vol * vol_of_vol) * np.log1p(-growth * gap / (2 * gamma)))
    return a - b * initial_variance


def _volatility_swap_block(p, maturity, nodes):
    mean = (p["theta"] + (p["initial_variance"] - p["theta"]) * (-np.expm1(-p["kappa"] * maturity))
            / (p["kappa"] * maturity))  # E[I_T / T]
    scale = (1 / np.sqrt(mean))[:, None]
    column = {key: value[:, None] for key, value in p.items()}

    def one_minus_laplace(x):
        return -np.expm1(integrated_variance_log_laplace(x * x / maturity, maturity, column["initial_variance"],
                                                         column["kappa"], column["theta"], column["vol_of_vol"]))

    near, near_weights = _legendre(nodes[0])
    far, far_weights = _legendre(nodes[1])
    x = scale * near[None, :]
    inner = np.sum(one_minus_laplace(x) / x ** 2 * near_weights, axis=1) * scale[:, 0]
    outer = np.sum(one_minus_laplace(scale / far[None, :]) * far_weights, axis=1) / scale[:, 0]
    return (inner + outer) / np.sqrt(np.pi)


def volatility_swap(parameters, initial_variance, maturity, *, nodes=(64, 128)):
    """E[sqrt(I_T / T)], the fair strike of a volatility swap on continuous realized variance.

    Annualized volatility units (multiply by 100 for points). parameters maps kappa, theta
    and vol_of_vol. Returns (value, change) as for forward_start_call.
    """
    p, shape = _broadcast(parameters, initial_variance, ("kappa", "theta", "vol_of_vol"))
    maturity = _horizon(maturity, "maturity", positive=True)
    if len(nodes) != 2 or min(nodes) < 8:
        raise ValueError("require two node counts of at least 8")
    if np.any(p["initial_variance"] == 0):
        raise ValueError("volatility swap requires positive initial variance")
    value, change = np.empty_like(p["kappa"]), np.empty_like(p["kappa"])
    for start in range(0, value.size, _CHUNK):
        block = {key: array[start:start + _CHUNK] for key, array in p.items()}
        base = _volatility_swap_block(block, maturity, nodes)
        refined = _volatility_swap_block(block, maturity, (2 * nodes[0], 2 * nodes[1]))
        value[start:start + _CHUNK], change[start:start + _CHUNK] = refined, np.abs(refined - base)
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError("volatility-swap quadrature failed; parameters unsupported")
    return value.reshape(shape), change.reshape(shape)

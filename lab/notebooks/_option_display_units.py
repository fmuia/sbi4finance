"""Display option prices in volatility points without changing the price model.

An implied volatility of 0.20 is 20 volatility points; one point is one
percentage point of annualized volatility. Exact Black implied volatilities
and first-order, reference-vega price differences are deliberately separate
APIs. A Gaussian price replicate can have no Black implied volatility.

The affine residual/SD conversions preserve every input coordinate, including
non-finite values. They never clip prices, filter draws or recompute quantiles.
Black prices, vega and inversion are owned by the canonical tutorial runtime.
"""

import numpy as np

from sbi4f_tutorial.shared import surface


def _geometry(forward, strike, maturity, discount):
    arrays = np.broadcast_arrays(*[np.atleast_1d(np.asarray(value, dtype=float))
                                   for value in (forward, strike, maturity, discount)])
    if arrays[0].ndim != 1 or not arrays[0].size:
        raise ValueError("option geometry must describe a nonempty ordered contract vector")
    if any(not np.isfinite(value).all() or np.any(value <= 0) for value in arrays):
        raise ValueError("forwards, strikes, maturities and discounts must be finite and positive")
    return arrays


def _contracts(value, count, label):
    array = np.asarray(value, dtype=float)
    if array.ndim < 1 or array.shape[-1] != count:
        raise ValueError(label + " must have the contract vector as its last dimension")
    return array


def _vega(vega):
    array = np.asarray(vega, dtype=float)
    if (array.ndim != 1 or not array.size or not np.isfinite(array).all()
            or np.any(array <= 0) or not np.isfinite(100 / array).all()):
        raise ValueError("reference vega must be a finite positive value for each contract")
    return array


def black_vega(forward, strike, maturity, discount, iv):
    """Raw-price vega, per unit absolute volatility, at supplied reference IVs.

    For example, divide a raw-price residual by this vega and multiply by 100
    to obtain first-order volatility-point equivalents. No vega floor is used.
    """
    forward, strike, maturity, discount = _geometry(forward, strike, maturity, discount)
    iv = _contracts(iv, forward.size, "reference implied volatility")
    if iv.ndim != 1 or not np.isfinite(iv).all() or np.any(iv <= 0):
        raise ValueError("reference implied volatility must be finite and positive for every contract")
    return _vega(discount * forward * surface.bs_vega(np.log(strike / forward), maturity, iv))


def volatility_point_residuals(prices, baseline, vega):
    """100 * (prices - baseline) / fixed reference vega, preserving all rows.

    These are local Black-vega equivalents, not exact differences of implied
    volatilities. Negative and out-of-Black-domain Gaussian prices remain
    visible. NaNs/infinities propagate rather than being silently omitted.
    """
    vega = _vega(vega)
    prices = _contracts(prices, vega.size, "prices")
    baseline = _contracts(baseline, vega.size, "baseline prices")
    return (prices - baseline) * (100 / vega)


def volatility_point_sd(sd_prices, vega):
    """Convert price SDs to first-order volatility points at fixed vega."""
    vega = _vega(vega)
    sd = _contracts(sd_prices, vega.size, "price standard deviations")
    if np.any(sd < 0):
        raise ValueError("a price standard deviation cannot be negative")
    return sd * (100 / vega)


def equivalent_volatility_points(prices, reference_prices, reference_iv, vega):
    """Affine levels anchored on reference IV, for explicitly labelled displays.

    At the reference price the value is its exact IV in points. Away from it,
    this is a tangent approximation, which may be poor for large price moves
    and may produce negative equivalents. It is never a Black inverse.
    """
    vega = _vega(vega)
    reference_iv = _contracts(reference_iv, vega.size, "reference implied volatility")
    if reference_iv.ndim != 1 or not np.isfinite(reference_iv).all() or np.any(reference_iv <= 0):
        raise ValueError("reference implied volatility must be finite and positive for every contract")
    return 100 * reference_iv + volatility_point_residuals(prices, reference_prices, vega)


def implied_volatility_points(prices, forward, strike, maturity, discount, kind,
                              *, lower=1e-4, upper=5., tol=1e-12, max_iter=200):
    """Exact Black IV in points and its validity mask, with no bound clipping.

    The output has exactly the input shape. Prices outside Black bounds or the
    declared inversion bracket, or inversions that fail a repricing check,
    receive NaN/False. In particular, we do not silently map an above-bracket
    price to the upper volatility limit. Callers must retain/report the
    validity mask; using NaN-dropping quantiles would condition on
    invertibility and can remove predictive tails.
    """
    if (not np.isfinite([lower, upper, tol]).all() or not 0 < lower < upper
            or tol <= 0 or not isinstance(max_iter, int) or max_iter <= 0):
        raise ValueError("Black inversion requires an ordered positive bracket and positive tolerance/budget")
    forward, strike, maturity, discount = _geometry(forward, strike, maturity, discount)
    prices = _contracts(prices, forward.size, "prices")
    kind = np.broadcast_to(np.asarray(kind), forward.shape)
    if not np.all(np.isin(kind, ["put", "call"])):
        raise ValueError("option type must be put or call for every contract")
    k = np.log(strike / forward)
    normalized = prices / (discount * forward)
    low_price = surface.bs_price(k, maturity, lower, kind)
    high_price = surface.bs_price(k, maturity, upper, kind)
    iv = surface.implied_vol(normalized, k, maturity, kind,
                             lower=lower, upper=upper, tol=tol, max_iter=max_iter)
    repriced = surface.bs_price(k, maturity, iv, kind)
    # The solver tolerance bounds the volatility interval. Convert it to a
    # normalized-price bound using max Black vega, and allow roundoff in the
    # canonical formula's forward/strike terms. A depleted iteration budget
    # must not label an unconverged inversion as valid.
    price_tolerance = (tol * np.sqrt(maturity / (2 * np.pi))
                       + 32 * np.finfo(float).eps * np.maximum(1., strike / forward))
    valid = (np.isfinite(normalized) & np.isfinite(iv)
             & (normalized >= low_price) & (normalized <= high_price)
             & (np.abs(repriced - normalized) <= price_tolerance))
    return np.where(valid, 100 * iv, np.nan), valid


def forward_start_volatility_points(price_percent_notional, reset, expiry):
    """Black-equivalent points for the lesson's normalized forward-start payoff.

    This is a display benchmark for its zero-carry, undiscounted payoff
    ``max(X_expiry/X_reset - 1, 0)``. The input is its price in percent of
    notional. A flat-volatility Black ATM call over ``expiry-reset`` defines
    the equivalent volatility; the stochastic-volatility product is not
    repriced or replaced. Returns same-shaped points and validity mask.
    """
    if not np.isfinite([reset, expiry]).all() or not 0 <= reset < expiry:
        raise ValueError("forward-start dates must satisfy 0 <= reset < expiry")
    prices = np.asarray(price_percent_notional, dtype=float)
    shape = prices.shape
    points, valid = implied_volatility_points(prices.reshape(-1, 1) / 100,
                                              [1.], [1.], [expiry - reset], [1.], ["call"])
    return points.reshape(shape), valid.reshape(shape)

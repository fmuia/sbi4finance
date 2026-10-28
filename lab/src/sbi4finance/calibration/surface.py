"""Black–Scholes in forward units, implied volatility, the design grid and smile statistics.

k = ln(K / F) is log-moneyness and delta = k / (sigma_ref sqrt(T)) standardized
moneyness. A call is c = N(d1) - e^k N(d2) and a put p = c - 1 + e^k. Quotes on
the design grid are out of the money: puts at delta <= 0, calls above.
"""

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

TRADING_DAYS = 252
DESIGN_DAYS = (5, 10, 21, 42, 63, 126, 252)
DESIGN_DELTAS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
SIGMA_REF = 0.20
NOISE_ATM = 0.003  # 0.3 volatility points at delta = 0
NOISE_WING = 0.010  # 1.0 volatility point at |delta| = 2


def _d1(k, maturity, sigma):
    total = sigma * np.sqrt(maturity)
    return (-k + 0.5 * total**2) / total, total


def bs_call(k, maturity, sigma):
    k, maturity, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (k, maturity, sigma)))
    d1, total = _d1(k, maturity, sigma)
    return ndtr(d1) - np.exp(k) * ndtr(d1 - total)


def bs_put(k, maturity, sigma):
    return bs_call(k, maturity, sigma) - 1.0 + np.exp(np.asarray(k, dtype=float))


def bs_price(k, maturity, sigma, kind):
    """kind is 'call', 'put' or an array of them, broadcast with k."""
    call = bs_call(k, maturity, sigma)
    is_put = np.asarray(kind) == "put"
    return np.where(is_put, call - 1.0 + np.exp(np.asarray(k, dtype=float)), call)


def bs_vega(k, maturity, sigma):
    """d price / d sigma in forward units, the same for calls and puts."""
    k, maturity, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (k, maturity, sigma)))
    d1, _ = _d1(k, maturity, sigma)
    return np.sqrt(maturity) * norm.pdf(d1)


def implied_vol(price, k, maturity, kind, *, lower=1e-4, upper=5.0, tol=1e-12, max_iter=200):
    """Implied volatility by bisection on [lower, upper]; NaN outside the no-arbitrage bounds.

    Bisection is slow but cannot fail inside the bounds, which matters for the
    deep wings where vega is tiny and Newton steps overshoot.
    """
    price, k, maturity = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, k, maturity)))
    kind = np.broadcast_to(np.asarray(kind), price.shape)
    is_put = kind == "put"
    intrinsic = np.where(is_put, np.maximum(np.exp(k) - 1.0, 0.0), np.maximum(1.0 - np.exp(k), 0.0))
    cap = np.where(is_put, np.exp(k), 1.0)
    valid = np.isfinite(price) & (price > intrinsic) & (price < cap)
    lo = np.full(price.shape, lower)
    hi = np.full(price.shape, upper)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        high = bs_price(k, maturity, mid, kind) > price
        hi = np.where(high, mid, hi)
        lo = np.where(high, lo, mid)
        if np.all(hi - lo < tol):
            break
    result = 0.5 * (lo + hi)
    return np.where(valid, result, np.nan)


def design_grid(days=DESIGN_DAYS, deltas=DESIGN_DELTAS, sigma_ref=SIGMA_REF):
    """Ordered contracts, maturity-major, spot 1 and zero carry: strikes are e^k."""
    contracts = []
    for day in days:
        maturity = day / TRADING_DAYS
        for delta in deltas:
            k = delta * sigma_ref * np.sqrt(maturity)
            contracts.append({"strike": float(np.exp(k)), "maturity": float(maturity),
                              "option_type": "put" if delta <= 0 else "call"})
    return contracts


def contract_arrays(contracts):
    k = np.log(np.array([c["strike"] for c in contracts], dtype=float))
    maturity = np.array([c["maturity"] for c in contracts], dtype=float)
    kind = np.array([c["option_type"] for c in contracts])
    return k, maturity, kind


def standardized_moneyness(k, maturity, sigma_ref=SIGMA_REF):
    return np.asarray(k, dtype=float) / (sigma_ref * np.sqrt(np.asarray(maturity, dtype=float)))


def implied_vol_noise(delta, atm=NOISE_ATM, wing=NOISE_WING):
    """Quote noise in volatility units, linear in |delta| from atm to wing at |delta| = 2."""
    return atm + (wing - atm) * np.minimum(np.abs(np.asarray(delta, dtype=float)), 2.0) / 2.0


def quote_noise(contracts, sigma_ref=SIGMA_REF, atm=NOISE_ATM, wing=NOISE_WING):
    """Price-noise standard deviation per contract: vega at sigma_ref times the volatility noise."""
    k, maturity, _ = contract_arrays(contracts)
    delta = standardized_moneyness(k, maturity, sigma_ref)
    return [float(x) for x in bs_vega(k, maturity, sigma_ref) * implied_vol_noise(delta, atm, wing)]


def smile_statistics(implied, k, maturity):
    """ATM level, skew d sigma / dk and curvature d2 sigma / dk2 per maturity.

    A quadratic in k is fitted by least squares to the five quotes nearest the
    money of each maturity; the skew and curvature are its derivatives at k = 0.
    Returns sorted maturities and three arrays.
    """
    implied, k, maturity = (np.asarray(x, dtype=float) for x in (implied, k, maturity))
    out = []
    for value in np.unique(maturity):
        chosen = np.flatnonzero((maturity == value) & np.isfinite(implied))
        nearest = chosen[np.argsort(np.abs(k[chosen]))[:5]]
        if len(nearest) < 3:
            out.append((value, np.nan, np.nan, np.nan))
            continue
        x = k[nearest]
        design = np.column_stack([np.ones_like(x), x, x**2])
        c0, c1, c2 = np.linalg.lstsq(design, implied[nearest], rcond=None)[0]
        out.append((value, c0, c1, 2.0 * c2))
    out = np.array(out)
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def skew_power_law(maturities, skews):
    """Fit |skew| = a T^b on log-log axes; for rough Bergomi b = H - 1/2 at short T."""
    maturities, skews = np.asarray(maturities, dtype=float), np.abs(np.asarray(skews, dtype=float))
    ok = np.isfinite(skews) & (skews > 0)
    b, log_a = np.polyfit(np.log(maturities[ok]), np.log(skews[ok]), 1)
    return float(b), float(np.exp(log_a))


def total_variance_by_replication(k, otm_prices):
    """w(T) = 2 * integral of out-of-the-money prices e^{-k} dk, by the trapezoid rule.

    Inputs are one maturity's sorted log-moneyness and out-of-the-money prices
    in forward units (puts for k < 0, calls for k > 0). The strip is truncated
    at the outermost quotes; the caller reports the truncation separately.
    """
    k, otm_prices = np.asarray(k, dtype=float), np.asarray(otm_prices, dtype=float)
    order = np.argsort(k)
    k, otm_prices = k[order], otm_prices[order]
    return float(2.0 * np.trapezoid(otm_prices * np.exp(-k), k))


def forward_variances(maturities, total_variances):
    """Piecewise-constant forward variance on (0, T1], (T1, T2], ... from total variances."""
    maturities = np.asarray(maturities, dtype=float)
    total_variances = np.asarray(total_variances, dtype=float)
    return np.diff(np.r_[0.0, total_variances]) / np.diff(np.r_[0.0, maturities])


def parity_forward(strikes, calls, puts):
    """C - P = D (F - K): regress C - P on K across strikes; slope -D, intercept D F."""
    strikes, difference = np.asarray(strikes, dtype=float), np.asarray(calls, dtype=float) - np.asarray(puts, dtype=float)
    slope, intercept = np.polyfit(strikes, difference, 1)
    discount = -slope
    return float(intercept / discount), float(discount)

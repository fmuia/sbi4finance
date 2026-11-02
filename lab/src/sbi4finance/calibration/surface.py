"""Black–Scholes in forward units, implied volatility, the design grid and smile statistics.

k = ln(K / F) is log-moneyness and delta = k / (sigma_ref sqrt(T)) standardized
moneyness. A call is c = N(d1) - e^k N(d2) and a put p = c - 1 + e^k. Quotes on
the design grid are out of the money: puts at delta <= 0, calls above.
"""

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.shared.surface import (
    TRADING_DAYS,
    DESIGN_DAYS,
    DESIGN_DELTAS,
    SIGMA_REF,
    NOISE_ATM,
    NOISE_WING,
    _d1,
    bs_call,
    bs_put,
    bs_price,
    bs_vega,
    implied_vol,
    design_grid,
    contract_arrays,
    standardized_moneyness,
    implied_vol_noise,
    quote_noise,
    smile_statistics,
    skew_power_law,
)


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

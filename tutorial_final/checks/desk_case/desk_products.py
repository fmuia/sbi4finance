"""Desk products under rough Bergomi, priced on one set of 1-year paths with common random numbers.

All prices are in forward units with zero carry, so the forward is 1 and no discounting applies.
Given the variance path, the log-return over [t_(k-1), t_k] is Gaussian (products.py):

    rho (J(t_k) - J(t_(k-1))) - (I(t_k) - I(t_(k-1))) / 2 + sqrt((1 - rho^2) (I(t_k) - I(t_(k-1)))) Z_k,

with Z_k independent standard normals. Monthly returns (21 trading days) are simulated jointly from it, so every
path-dependent product below sees the same joint law. The Z_k are antithetic and drawn from their own seed, the
same for every parameter set (common random numbers), like the variance normals.

Products (the specification is SPEC):
- autocallable: 1 year, quarterly observations; called at quarter k if S >= 100% of the initial level, paying
  1 + k c; if never called, pays 1 + 4c at 1 year when S >= 100%, 1 when S >= 70% (the barrier), S otherwise;
- forward_start: (S(1y) / S(6m) - 1)^+, the strike set at six months at the money;
- put_1y_80: (0.8 - S(1y))^+;
- cliquet: max(sum over 12 months of min(R_i, cap), 0), R_i the monthly return (locally capped, globally floored).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sbi4finance.calibration import products  # noqa: E402

NAMES = ("autocallable", "forward_start", "put_1y_80", "cliquet")
SPEC = {"steps_per_year": 252, "months": 12, "autocall_coupon": 0.0125, "autocall_barrier": 0.7,
        "put_strike": 0.8, "cliquet_cap": 0.02}
_Z_TAG = 0x5A5A


def _period_normals(n_paths, n_periods, seed):
    z = np.random.default_rng(np.random.SeedSequence([_Z_TAG, int(seed)])).standard_normal((n_paths // 2, n_periods))
    return np.concatenate([z, -z])


def monthly_log_returns(parameters, curve, n_paths, seed):
    """Joint monthly log-returns [paths, 12] and the integrated variance per month."""
    spy = SPEC["steps_per_year"]
    paths = products.rough_paths(parameters, products.extend_curve(curve, 1.0), 1.0, spy, n_paths, seed)
    idx = np.arange(0, spy + 1, spy // SPEC["months"])
    di = np.diff(paths["I"][:, idx], axis=1)
    dj = np.diff(paths["J"][:, idx], axis=1)
    rho = parameters["rho"]
    z = _period_normals(n_paths, SPEC["months"], seed)
    return rho * dj - 0.5 * di + np.sqrt((1 - rho**2) * di) * z, di


def _mean_se(values, controls):
    return products._controlled_mean(values, controls)


def price(parameters, curve, n_paths=16384, seed=0):
    """Prices and standard errors of the four products, in forward units."""
    r, _ = monthly_log_returns(parameters, curve, n_paths, seed)
    log_s = np.cumsum(r, axis=1)
    s = np.exp(log_s)
    quarters = s[:, [2, 5, 8, 11]]  # 3, 6, 9, 12 months
    c, barrier = SPEC["autocall_coupon"], SPEC["autocall_barrier"]
    alive = np.ones(len(s), bool)
    payoff = np.zeros(len(s))
    for k in range(4):
        called = alive & (quarters[:, k] >= 1.0)
        payoff[called] = 1 + (k + 1) * c
        alive &= ~called
    final = quarters[:, 3]
    payoff[alive] = np.where(final[alive] >= barrier, 1.0, final[alive])
    controls = quarters - 1.0  # each has mean zero: the forward is 1 at every date
    out = {"autocallable": _mean_se(payoff, controls)}
    s6, s12 = s[:, 5], s[:, 11]
    out["forward_start"] = _mean_se(np.maximum(s12 / s6 - 1.0, 0.0), np.column_stack([s12 / s6 - 1.0, s12 - 1.0]))
    out["put_1y_80"] = _mean_se(np.maximum(SPEC["put_strike"] - s12, 0.0), (s12 - 1.0)[:, None])
    monthly = np.expm1(r)
    out["cliquet"] = _mean_se(np.maximum(np.minimum(monthly, SPEC["cliquet_cap"]).sum(axis=1), 0.0),
                              monthly.sum(axis=1)[:, None])
    return np.array([out[n][0] for n in NAMES]), np.array([out[n][1] for n in NAMES])

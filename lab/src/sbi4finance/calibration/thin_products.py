"""The thin-chain products (studies S10 and S11), priced under rough Bergomi on one set of one-year paths.

A thin layout quotes up to six months (126 trading days). Five products mature there; the sixth starts there and
matures at one year, so its price depends on the forward-variance curve beyond the last quote:

- variance_swap_6m: E[I(T) / T], T = 6 months; a function of the curve only;
- volatility_swap_6m: E[sqrt(I(T) / T)];
- forward_start_3m_6m: (S(6m) / S(3m) - 1)^+, struck at the money at three months;
- put_6m_80: (0.8 - S(6m))^+, beyond the layout's six-month strikes;
- cliquet_6m: max(sum of min(R_i, 2%) over six monthly returns R_i, 0);
- forward_start_6m_1y: (S(1y) / S(6m) - 1)^+, beyond the quotes.

Prices are in forward units with zero carry, as in products.py: given the variance path, the log-return over a
period is Gaussian with mean rho dJ - dI / 2 and variance (1 - rho^2) dI. Paths and the period normals are common
random numbers for every parameter set priced with the same seed.
"""

import numpy as np

from . import products, tasks

NAMES = ("variance_swap_6m", "volatility_swap_6m", "forward_start_3m_6m", "put_6m_80", "cliquet_6m",
         "forward_start_6m_1y")
SPEC = {"steps_per_year": 252, "month_steps": 21, "put_strike": 0.8, "cliquet_cap": 0.02}
_Z_TAG = 0x544850  # "THP"


def _period_normals(n_paths, n_periods, seed):
    z = np.random.default_rng(np.random.SeedSequence([_Z_TAG, int(seed)])).standard_normal((n_paths // 2, n_periods))
    return np.concatenate([z, -z])


def price(task, parameter_values, log_levels, n_paths=8192, seed=0):
    """Prices and standard errors of the six products for one draw of (H, eta, rho) and the log levels."""
    parameters = tasks.model_parameters(task, parameter_values)
    curve = tasks.curve(task, np.exp(np.asarray(log_levels, dtype=float)))
    spy = SPEC["steps_per_year"]
    paths = products.rough_paths(parameters, curve, 1.0, spy, n_paths, seed)
    rho = parameters["rho"]
    out = {"variance_swap_6m": products.variance_swap(paths, 0.5),
           "volatility_swap_6m": products.volatility_swap(paths, 0.5),
           "forward_start_3m_6m": products.forward_start_call(paths, rho, 0.25, 0.5),
           "put_6m_80": products.vanilla(paths, rho, SPEC["put_strike"], 0.5, "put"),
           "forward_start_6m_1y": products.forward_start_call(paths, rho, 0.5, 1.0)}
    idx = np.arange(0, 6 * SPEC["month_steps"] + 1, SPEC["month_steps"])
    di = np.diff(paths["I"][:, idx], axis=1)
    dj = np.diff(paths["J"][:, idx], axis=1)
    r = rho * dj - 0.5 * di + np.sqrt((1 - rho**2) * di) * _period_normals(n_paths, 6, seed)
    monthly = np.expm1(r)
    out["cliquet_6m"] = products._controlled_mean(
        np.maximum(np.minimum(monthly, SPEC["cliquet_cap"]).sum(axis=1), 0.0), monthly.sum(axis=1)[:, None])
    return np.array([out[n][0] for n in NAMES]), np.array([out[n][1] for n in NAMES])

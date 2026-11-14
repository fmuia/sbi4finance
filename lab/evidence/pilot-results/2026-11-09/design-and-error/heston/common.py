"""Shared helpers for the design-grid tests (scratch only; not part of the repo).

Grids are 7 maturities x 9 moneyness points (63 quotes), out of the money:
puts at delta <= 0, calls above. A grid is defined by its delta list and the
volatility used to standardize log-moneyness, k = delta * sigma_scale * sqrt(T).
"""
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6] / "tutorial" / "src"))
from sbi4f_tutorial.shared import surface  # noqa: E402
from sbi4f_tutorial.heston_options.model import european_price  # noqa: E402

DAYS = surface.DESIGN_DAYS
T_DESIGN = np.array(DAYS) / surface.TRADING_DAYS
SIGMA_REF = surface.SIGMA_REF
KAPPA = 2.0
PRIOR = np.array([[0.01, 0.16], [0.01, 0.16], [0.1, 1.5], [-0.95, -0.1]])
NAMES = ["v0", "theta", "vol_of_vol", "rho"]

D_CURRENT = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
D_PROPOSED = np.array([-4.0, -3.25, -2.5, -1.75, -1.0, -0.5, 0.0, 0.5, 1.0])

# name -> (deltas, scaling): scaling "ref" = fixed 20% vol, "atm" = each maturity's ATM vol
GRIDS = {
    "G0 current (+-2, 20% vol)": (D_CURRENT, "ref"),
    "G1 proposed (-4..+1, ATM vol)": (D_PROPOSED, "atm"),
    "G2 shift only (-4..+1, 20% vol)": (D_PROPOSED, "ref"),
    "G3 scale only (+-2, ATM vol)": (D_CURRENT, "atm"),
}


def heston_params(row):
    v0, theta, xi, rho = (float(x) for x in row)
    p = {"kappa": KAPPA, "theta": theta, "vol_of_vol": xi, "rho": rho, "jump_intensity": 0.0,
         "jump_up_probability": 0.5, "jump_up_rate": 10.0, "jump_down_rate": 10.0}
    return p, v0


def heston_prices(row, k, T, kind, nodes=2048, upper=1500.0):
    p, v0 = heston_params(row)
    k, T, kind = np.asarray(k, float), np.asarray(T, float), np.asarray(kind)
    out = np.empty(k.shape)
    for value in ("call", "put"):
        m = kind == value
        if m.any():
            out[m] = european_price(1.0, np.exp(k[m]), 1.0, T[m], p, v0, kind=value, nodes=nodes, upper_limit=upper)
    return out


def heston_atm_vol(row, T=T_DESIGN):
    T = np.asarray(T, float)
    price = heston_prices(row, np.zeros_like(T), T, np.array(["put"] * len(T)))
    return surface.implied_vol(price, np.zeros_like(T), T, "put")


def grid_contracts(deltas, scaling, atm_vol=None, T=T_DESIGN):
    """k, T, kind, delta (in the grid's own units) and sigma_scale per contract, maturity-major."""
    ks, Ts, kinds, ds, scales = [], [], [], [], []
    for j, t in enumerate(T):
        s = SIGMA_REF if scaling == "ref" else float(atm_vol[j])
        for d in deltas:
            ks.append(d * s * np.sqrt(t)); Ts.append(t); kinds.append("put" if d <= 0 else "call")
            ds.append(d); scales.append(s)
    return np.array(ks), np.array(Ts), np.array(kinds), np.array(ds), np.array(scales)


def s_iv(delta):
    """The tutorial's vol-point noise profile: 0.3 at the money, 1.0 at |delta| >= 2."""
    return surface.implied_vol_noise(delta)


def noise_design(k, T, delta, scale):
    """Price-noise sd as the tutorial builds it: vega at the grid's scaling vol x s_iv(delta in grid units)."""
    return surface.bs_vega(k, T, scale) * s_iv(delta)


def noise_local(k, T, kind, prices, atm_vol_per_contract):
    """Same vol-point noise at a given strike whatever the grid: vega at the quote's own IV x s_iv(delta_atm)."""
    iv = surface.implied_vol(prices, k, T, kind)
    d_atm = k / (atm_vol_per_contract * np.sqrt(T))
    return surface.bs_vega(k, T, iv) * s_iv(d_atm), iv


def jacobian(fun, row, rel=1e-4, absolute=(2e-5, 2e-5, 1e-3, 1e-3)):
    row = np.asarray(row, float)
    cols = []
    for i in range(len(row)):
        h = absolute[i]
        up, dn = row.copy(), row.copy()
        up[i] += h; dn[i] -= h
        cols.append((fun(up) - fun(dn)) / (2 * h))
    return np.column_stack(cols)


def heston_varswap_vol(row, T=T_DESIGN):
    v0, theta = row[0], row[1]
    T = np.asarray(T, float)
    w = theta * T + (v0 - theta) * (1 - np.exp(-KAPPA * T)) / KAPPA
    return np.sqrt(w / T)


def varswap_jacobian(row, T=T_DESIGN):
    """d vol_var(T) / d(v0, theta, xi, rho); only v0 and theta enter."""
    return jacobian(lambda r: heston_varswap_vol(r, T), row, absolute=(1e-6, 1e-6, 1e-3, 1e-3))

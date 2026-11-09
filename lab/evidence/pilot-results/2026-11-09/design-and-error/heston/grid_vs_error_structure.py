"""Where should the grid go once the error model includes model error?

Heston (kappa free) at the SPX 9-Nov fit. Observables: implied vols on a grid. Laplace posterior sd
(flat prior) = sqrt(diag (J' S^-1 J)^-1), J = d IV / d theta, S = error covariance in IV units.
The discrepancy is a field on the physical location (delta_atm, log T), so every grid sees the same field.

Error structures:
  spread : SPX bid-ask half-spreads only (measured profile), independent
  iid    : spread + independent model error, tau = 1 vol point
  corr   : spread + smooth model error, tau = 1 vol point, squared-exponential kernel with
           l_delta = 0.75 ATM-sd and l_logT = 1.15 (matches the measured residual correlations 0.80
           between neighbouring strikes 0.5 ATM-sd apart and 0.85 between neighbouring design maturities)
"""
import json
import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent),
                str(pathlib.Path(__file__).resolve().parents[6] / "tutorial" / "src")]
import numpy as np  # noqa: E402
from sbi4f_tutorial.heston_options.model import european_price  # noqa: E402
from common import DAYS, T_DESIGN, SIGMA_REF, surface  # noqa: E402

spx = json.load(open(pathlib.Path(__file__).resolve().parent / "spx_noise_and_model_error.json"))
X0 = np.array(spx["heston_fit"])                       # kappa, v0, theta, xi, rho
NAMES = ["kappa", "v0", "theta", "xi", "rho"]
LOGT = np.log(T_DESIGN)


def heston_iv(x, k, T, kind, nodes=2048, upper=1500.0):
    kappa, v0, theta, xi, rho = x
    p = {"kappa": kappa, "theta": theta, "vol_of_vol": xi, "rho": rho, "jump_intensity": 0.0,
         "jump_up_probability": 0.5, "jump_up_rate": 10.0, "jump_down_rate": 10.0}
    price = np.empty(len(k))
    for kd in ("call", "put"):
        m = kind == kd
        if m.any():
            price[m] = european_price(1.0, np.exp(k[m]), 1.0, T[m], p, v0, kind=kd, nodes=nodes, upper_limit=upper)
    return surface.implied_vol(price, k, T, kind)


def varswap_1y(x):
    kappa, v0, theta = x[:3]
    return np.sqrt(theta + (v0 - theta) * (1 - np.exp(-kappa)) / kappa)


ATM = heston_iv(X0, np.zeros(7), T_DESIGN, np.array(["put"] * 7))

# measured SPX half-spread profile (vol pts): median by delta_atm bin and expiry, filled and interpolated
bins = np.array(spx["bins"], float); centres = 0.5 * (bins[:-1] + bins[1:])
table = np.array([[np.nan if v is None else v for v in spx["half_spread_table"][str(d)]] for d in DAYS])
for row in table:                                       # fill gaps with the nearest measured bin
    ok = np.isfinite(row)
    row[~ok] = np.interp(centres[~ok], centres[ok], row[ok])


def half_spread(d_atm, T):
    j = np.clip(np.searchsorted(T_DESIGN, np.asarray(T) - 1e-12), 0, 6)
    return np.array([np.interp(d, centres, table[jj]) for d, jj in zip(d_atm, j)]) / 100.0


def grid(deltas, scale):
    k, T, kind, d_atm = [], [], [], []
    for j, t in enumerate(T_DESIGN):
        s = SIGMA_REF if scale == "ref" else ATM[j]
        for d in deltas:
            kk = d * s * np.sqrt(t)
            k.append(kk); T.append(t); kind.append("put" if kk <= 0 else "call"); d_atm.append(kk / (ATM[j] * np.sqrt(t)))
    return np.array(k), np.array(T), np.array(kind), np.array(d_atm)


GRIDS = {
    "current: +-2 at 20% vol, 9/maturity": (np.linspace(-2, 2, 9), "ref"),
    "current range, 17/maturity (denser)": (np.linspace(-2, 2, 17), "ref"),
    "wide put-heavy: -6..+2 ATM-sd, 9/maturity": (np.linspace(-6, 2, 9), "atm"),
    "wide put-heavy: -6..+2 ATM-sd, 17/maturity": (np.linspace(-6, 2, 17), "atm"),
}


def covariance(d_atm, T, half, kind_of_error, tau=0.01, l_d=0.75, l_t=1.15):
    S = np.diag(half ** 2)
    if kind_of_error == "iid":
        S = S + tau ** 2 * np.eye(len(T))
    elif kind_of_error == "corr":
        dd = (d_atm[:, None] - d_atm[None, :]) / l_d
        dt = (np.log(T)[:, None] - np.log(T)[None, :]) / l_t
        S = S + tau ** 2 * np.exp(-0.5 * (dd ** 2 + dt ** 2))
    return S


steps = np.array([0.02, 2e-5, 2e-5, 2e-3, 1e-3])
out = {}
for name, (deltas, scale) in GRIDS.items():
    k, T, kind, d_atm = grid(deltas, scale)
    J = np.column_stack([(heston_iv(X0 + e, k, T, kind) - heston_iv(X0 - e, k, T, kind)) / (2 * h)
                         for e, h in ((np.eye(5)[i] * steps[i], steps[i]) for i in range(5))])
    gv = np.array([(varswap_1y(X0 + np.eye(5)[i] * steps[i]) - varswap_1y(X0 - np.eye(5)[i] * steps[i])) / (2 * steps[i])
                   for i in range(5)])
    half = half_spread(d_atm, T)
    res = {"n": len(k), "d_atm_range_1w": [float(d_atm[0]), float(d_atm[len(deltas) - 1])],
           "d_atm_range_1y": [float(d_atm[-len(deltas)]), float(d_atm[-1])]}
    for err in ("spread", "iid", "corr"):
        S = covariance(d_atm, T, half, err)
        Si = np.linalg.inv(S)
        C = np.linalg.inv(J.T @ Si @ J)
        res[err] = {"sd": np.sqrt(np.diag(C)).tolist(), "varswap_1y_sd_volpts": float(np.sqrt(gv @ C @ gv) * 100)}
    out[name] = res
    print(f"\n{name}  (n={len(k)}; delta_atm 1w {res['d_atm_range_1w'][0]:.1f}..{res['d_atm_range_1w'][1]:.1f}, "
          f"1y {res['d_atm_range_1y'][0]:.1f}..{res['d_atm_range_1y'][1]:.1f})")
    for err in ("spread", "iid", "corr"):
        r = res[err]
        print(f"   {err:7s} " + "  ".join(f"{n}={s:.3g}" for n, s in zip(NAMES, r["sd"]))
              + f"   varswap1y={r['varswap_1y_sd_volpts']:.3f} vol pts")

# how the answer depends on the size of the model error (wide 17/maturity vs current 9/maturity, smooth error)
print("\nsd ratio wide-17 / current-9 under smooth model error of size tau (vol pts):")
for tau in (0.05, 0.1, 0.3, 1.0, 2.0):
    ratios = []
    for name in ("current: +-2 at 20% vol, 9/maturity", "wide put-heavy: -6..+2 ATM-sd, 17/maturity"):
        deltas, scale = GRIDS[name]
        k, T, kind, d_atm = grid(deltas, scale)
        J = np.column_stack([(heston_iv(X0 + np.eye(5)[i] * steps[i], k, T, kind)
                              - heston_iv(X0 - np.eye(5)[i] * steps[i], k, T, kind)) / (2 * steps[i]) for i in range(5)])
        C = np.linalg.inv(J.T @ np.linalg.inv(covariance(d_atm, T, half_spread(d_atm, T), "corr", tau=tau / 100)) @ J)
        gv = np.array([(varswap_1y(X0 + np.eye(5)[i] * steps[i]) - varswap_1y(X0 - np.eye(5)[i] * steps[i])) / (2 * steps[i])
                       for i in range(5)])
        ratios.append(np.r_[np.sqrt(np.diag(C)), np.sqrt(gv @ C @ gv)])
    r = ratios[1] / ratios[0]
    print(f"   tau={tau:4.2f}: " + "  ".join(f"{n}={v:.2f}" for n, v in zip(NAMES + ["varswap1y"], r)))
json.dump(out, open("grid_vs_error_structure.json", "w"), indent=1)

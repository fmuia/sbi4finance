"""Verify claims about the design grid against surface.py and SPY's surface.json (read-only)."""
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "tutorial/src"))
from sbi4f_tutorial.shared import surface as S  # noqa: E402

spy = json.loads((ROOT / "tutorial/assets/results/calibration/spy/surface.json").read_text())
dg = spy["diagnostics"]
days = np.array(S.DESIGN_DAYS)
deltas = np.array(S.DESIGN_DELTAS)
T_design = days / 252
atm = np.array(spy["atm_implied_vol"])
iv = np.array(dg["implied_vol_grid"]).reshape(7, 9)
noise = np.array(spy["noise_sd"]).reshape(7, 9)
prices = np.array(spy["prices_forward_units"]).reshape(7, 9)

print("== Q1 grid ==")
print("deltas", deltas, "sigma_ref", S.SIGMA_REF)
contracts = S.design_grid()
k, T, kind = S.contract_arrays(contracts)
k = k.reshape(7, 9)
print("k range per maturity:", [(round(k[i, 0], 4), round(k[i, -1], 4)) for i in range(7)])
print("calls:", int(np.sum(deltas > 0)), "of 9 columns; delta>=1:", int(np.sum(deltas >= 1)))
# compare stored noise with design noise (floor)
design_noise = np.array(S.quote_noise(contracts)).reshape(7, 9)
print("SPY noise_sd == design noise everywhere:", np.allclose(noise, design_noise, rtol=1e-9))
task = json.loads((ROOT / "tutorial/assets/backbones/heston-final500/task.json").read_text())
print("task quote_noise_std == design noise:", np.allclose(np.array(task["context"]["quote_noise_std"]).reshape(7, 9), design_noise))

print("\n== BS market delta of each grid point at SPY iv ==")
Tm = T_design[:, None]
d1 = (-k + 0.5 * iv**2 * Tm) / (iv * np.sqrt(Tm))
mkt_delta = np.where(deltas[None, :] <= 0, norm.cdf(d1) - 1, norm.cdf(d1))
np.set_printoptions(linewidth=200, precision=3, suppress=True)
print("rows=days", days, "cols=", deltas)
print(mkt_delta)
print("SPY iv grid (%):\n", 100 * iv)

print("\n== ATM-vol standardized delta of grid columns: delta*0.2/atm ==")
print(np.outer(0.2 / atm, deltas))

print("\n== Q2 strip deltas per expiration, 20% units and ATM-vol units ==")
exps = dg["expirations"]
Te = np.array([e["days"] for e in exps]) / 252
atm_e = np.interp(Te, T_design, atm)  # ATM vol interpolated (flat beyond ends)
for e, a in zip(exps, atm_e):
    lo, hi = e["strip_deltas"]
    print(f"{e['expiration']} d={e['days']:3d} n={e['quotes_kept']:3d} strip20=[{lo:7.2f},{hi:5.2f}] stripATM=[{lo*0.2/a:7.2f},{hi*0.2/a:5.2f}] k_min={lo*0.2*np.sqrt(e['days']/252):.3f} cov={e['replication_covered']}")
mins = np.array([e["strip_deltas"][0] for e in exps])
print("min strip delta (20% units): range", mins.min(), mins.max(), "; among replication-covered:",
      min(m for m, e in zip(mins, exps) if e["replication_covered"]), max(m for m, e in zip(mins, exps) if e["replication_covered"]))
print("ATM-units min:", (mins * 0.2 / atm_e).min(), (mins * 0.2 / atm_e).max())

print("\n== Q4 replicate reading logic from strip_deltas ==")
Ts = Te
dmin = np.array([e["strip_deltas"][0] for e in exps]); dmax = np.array([e["strip_deltas"][1] for e in exps])
def covers(i, Tq, d):
    return dmin[i] <= d * np.sqrt(Tq / Ts[i]) <= dmax[i]
readings = {"two": 0, "one": 0, "extrap": 0}
for r, Tq in enumerate(T_design):
    j = int(np.searchsorted(Ts, Tq))
    for c, d in enumerate(deltas):
        below = [i for i in range(len(Ts)) if Ts[i] <= Tq + 1e-12 and covers(i, Tq, d)]
        above = [i for i in range(len(Ts)) if Ts[i] >= Tq - 1e-12 and covers(i, Tq, d)]
        if below and above:
            lo, hi = below[-1], above[0]; readings["two" if lo != hi else "one"] += 1
        elif below or above:
            lo = hi = below[-1] if below else above[0]; readings["one"] += 1
        else:
            readings["extrap"] += 1; lo = hi = None
        # nearest bracketing expirations regardless of cover
        nb = [i for i in range(len(Ts)) if Ts[i] <= Tq + 1e-12][-1]
        na = [i for i in range(len(Ts)) if Ts[i] >= Tq - 1e-12][0]
        skipped = [exps[i]["days"] for i in {nb, na} if i not in {lo, hi}]
        if skipped:
            print(f"T={days[r]:3d} delta={d:+.1f}: nearest expirations {sorted({exps[nb]['days'], exps[na]['days']})} "
                  f"do not all reach; read from {sorted({exps[lo]['days'], exps[hi]['days']})} (skipped {skipped})")
print(readings, "stored:", dg["grid_point_readings"])

print("\n== Q3 noise in iv units at SPY's actual iv ==")
vega_actual = S.bs_vega(k, Tm, iv)
vega_ref = S.bs_vega(k, Tm, 0.2)
iv_sd = noise / vega_actual
print("design iv noise s_iv(delta) (vol pts):", 100 * S.implied_vol_noise(deltas))
print("effective iv sd at SPY iv (vol pts):\n", 100 * iv_sd)
print("ratio vega(0.2)/vega(SPY iv):\n", vega_ref / vega_actual)
print("column mean of effective iv sd:", 100 * iv_sd.mean(0))
print("price relative to noise (y/sigma):\n", prices / noise)

# Rough Bergomi Fisher information per point, from the committed Gauss-Newton Jacobian at SPY's fit
npz = np.load(ROOT / "lab/checks/rough_spy_is/rough_spy_is_seed0.npz")
J = npz["jacobian"]  # (63, 3) d price / d (H, eta, rho)
sig = np.array(spy["noise_sd"])
Js = J / sig[:, None]
F = Js.T @ Js
print("\n== Rough Bergomi (SPY fit, seed0 Jacobian): share of Fisher information by column ==")
names = ["H", "eta", "rho"]
per_point = (Js**2).reshape(7, 9, 3)
col = per_point.sum(0)
print("cols", deltas)
for p in range(3):
    print(names[p], np.round(col[:, p] / col[:, p].sum(), 3))
# leave-columns-out: posterior sd inflation when dropping delta>=1 columns
def sd(mask):
    Jm = Js[mask]
    return np.sqrt(np.diag(np.linalg.inv(Jm.T @ Jm)))
full = np.ones(63, bool)
drop_calls = np.tile(deltas < 1, 7)
drop_far = np.tile(deltas < 1.5, 7)
drop_putwing = np.tile(deltas > -1.5, 7)
print("GN sd full:", sd(full))
print("GN sd without delta>=1 (21 pts):", sd(drop_calls), "ratio", sd(drop_calls) / sd(full))
print("GN sd without delta>=1.5 (14 pts):", sd(drop_far), "ratio", sd(drop_far) / sd(full))
print("GN sd without delta<=-1.5 (14 pts):", sd(drop_putwing), "ratio", sd(drop_putwing) / sd(full))

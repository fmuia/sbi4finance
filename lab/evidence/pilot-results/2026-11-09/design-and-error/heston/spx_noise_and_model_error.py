"""SPX: (1) bid-ask half-spreads in vol points by moneyness (ATM-vol units) and maturity;
(2) Heston (kappa free) fitted in implied-vol space on a wide put-heavy grid: size and shape of the
model error compared with the half-spreads.

Usage: python -I spx_noise_and_model_error.py /path/to/SPX.json
"""
import json
import pathlib
import sys
from datetime import date

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent),
                str(pathlib.Path(__file__).resolve().parents[6] / "lab" / "src"), str(pathlib.Path(__file__).resolve().parents[6] / "tutorial" / "src")]
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from sbi4finance.calibration import chain  # noqa: E402
from sbi4f_tutorial.heston_options.model import european_price  # noqa: E402
from common import DAYS, T_DESIGN, SIGMA_REF, surface  # noqa: E402

payload = json.load(open(sys.argv[1]))
data = payload["data"]
valuation = date.fromisoformat(data["last_trade_time"][:10])
def _row(o):
    tail = o["option"][-15:]
    return {"expiration": date(2000 + int(tail[:2]), int(tail[2:4]), int(tail[4:6])),
            "strike": int(tail[7:]) / 1000.0, "type": "call" if tail[6] == "C" else "put",
            "bid": float(o["bid"]), "ask": float(o["ask"]), "date": valuation}


# PM-settled SPXW wherever it exists; AM-settled SPX (root 'SPX' + date) only on dates SPXW does not list,
# which supplies the expiries beyond SPXW's last date. SPXO is a different product and is excluded.
pm = [_row(o) for o in data["options"] if o["option"].startswith("SPXW")]
pm_dates = {r["expiration"] for r in pm}
am = [_row(o) for o in data["options"] if o["option"][:3] == "SPX" and o["option"][3].isdigit()]
rows = pm + [r for r in am if r["expiration"] not in pm_dates]
rate = chain.rate_curve([(0.25, 0.0426), (2.0, 0.0479)])
print(f"SPX valuation {valuation}, spot {data.get('current_price')}, SPXW rows {len(rows)}")

# ---- (1) half-spreads by moneyness, at the expiration nearest each design maturity ----
by_exp = {}
for r in rows:
    if r["expiration"] > valuation:
        by_exp.setdefault(r["expiration"], []).append(r)
exps = []
for e, rr in sorted(by_exp.items()):
    d = chain.trading_days(valuation, e)
    if d < 3 or d > 300:
        continue
    T = d / 252
    disc = float(np.exp(-rate(T) * T))
    fwd = chain.forward_given_discount(rr, disc)
    if fwd is None:
        continue
    q = chain.clean_expiration(rr, fwd, disc, T)
    if len(q) < 20:
        continue
    exps.append((d, T, q))
bins = [-12, -8, -6, -4, -3, -2, -1, -0.5, 0.5, 1, 1.5, 2, 3, 5]
labels = [f"[{a},{b})" for a, b in zip(bins[:-1], bins[1:])]
print("\nmedian half-spread (vol pts) by moneyness in ATM-vol units; expiry nearest each design maturity")
print(f"{'days':>5s} {'ATM%':>5s} {'reach':>13s}  " + " ".join(f"{l:>9s}" for l in labels))
half_table = {}
for day in DAYS:
    d, T, q = min(exps, key=lambda x: abs(x[0] - day))
    k = np.array([x["k"] for x in q]); iv = np.array([x["iv"] for x in q]); hs = np.array([x["iv_half_spread"] for x in q])
    atm = float(np.interp(0.0, k, iv))
    da = k / (atm * np.sqrt(T))
    med = [np.median(hs[(da >= a) & (da < b)]) * 100 if np.any((da >= a) & (da < b)) else np.nan
           for a, b in zip(bins[:-1], bins[1:])]
    half_table[day] = med
    print(f"{d:5d} {atm*100:5.1f} {da.min():6.1f}..{da.max():4.1f}  " + " ".join(f"{m:9.2f}" for m in med))

# ---- (2) Heston, kappa free, fitted to a wide put-heavy grid in IV space ----
chain.surface.implied_vol_noise = lambda dd: 0.0 * np.asarray(dd, float)     # noise_sd -> vega(20%) x raw half-spread
base = chain.build_surface(rows, rate=rate, deltas=(0.0,))
atm_d = np.array(base["diagnostics"]["implied_vol_grid"])
D = np.linspace(-5.0, 2.0, 15)                                                # ATM-vol units, put-heavy
K, TT, KIND, IV, HALF, DA = [], [], [], [], [], []
for j, (day, t) in enumerate(zip(DAYS, T_DESIGN)):
    dref = D * atm_d[j] / SIGMA_REF
    out = chain.build_surface(rows, rate=rate, design_days=(day,), deltas=tuple(dref))
    k = dref * SIGMA_REF * np.sqrt(t)
    K += list(k); TT += [t] * len(D); KIND += ["put" if x <= 0 else "call" for x in D]; DA += list(D)
    IV += list(out["diagnostics"]["implied_vol_grid"]); HALF += list(out["noise_sd"] / surface.bs_vega(k, t, SIGMA_REF))
    assert out["diagnostics"]["grid_points_outside_quoted_strikes"] == 0, (day, out["diagnostics"]["grid_point_readings"])
K, TT, KIND, IV, HALF, DA = map(np.array, (K, TT, KIND, IV, HALF, DA))
n_m = len(D)


def heston_iv(x):
    kappa, v0, theta, xi, rho = x
    p = {"kappa": kappa, "theta": theta, "vol_of_vol": xi, "rho": rho, "jump_intensity": 0.0,
         "jump_up_probability": 0.5, "jump_up_rate": 10.0, "jump_down_rate": 10.0}
    price = np.empty(len(K))
    for kind in ("call", "put"):
        m = KIND == kind
        price[m] = european_price(1.0, np.exp(K[m]), 1.0, TT[m], p, v0, kind=kind, nodes=2048, upper_limit=1500.0)
    return surface.implied_vol(price, K, TT, KIND)


lo = np.array([0.1, 0.001, 0.001, 0.05, -0.99]); hi = np.array([15.0, 0.30, 0.30, 6.0, 0.0])
rng = np.random.default_rng(3)
best = None
for s in range(6):
    x0 = lo + (hi - lo) * (0.1 + 0.5 * rng.random(5))
    try:
        r = least_squares(lambda x: np.nan_to_num(heston_iv(x) - IV, nan=0.5), x0, bounds=(lo, hi),
                          x_scale=hi - lo, max_nfev=300)
    except ValueError:
        continue
    best = r if best is None or r.cost < best.cost else best
x = best.x
res = (heston_iv(x) - IV).reshape(7, n_m) * 100        # vol pts
half = HALF.reshape(7, n_m) * 100
print(f"\nHeston IV-space fit, kappa free: kappa={x[0]:.2f} v0={x[1]:.4f} theta={x[2]:.4f} xi={x[3]:.2f} rho={x[4]:.2f}")
print(f"  grid: {n_m} points per maturity, delta_atm from {D[0]} to {D[-1]}; 0 points outside quoted strikes")
print(f"  RMS model error {np.sqrt(np.mean(res**2)):.2f} vol pts; median half-spread {np.median(half):.2f} vol pts "
      f"(ratio {np.sqrt(np.mean(res**2))/np.median(half):.0f}x)")
print("  residual (model - market, vol pts), rows 1w..1y, columns delta_atm", np.round(D, 1))
for j in range(7):
    print("    " + " ".join(f"{v:6.2f}" for v in res[j]))
print("  half-spread (vol pts):")
for j in range(7):
    print("    " + " ".join(f"{v:6.2f}" for v in half[j]))
# shape of the model error: smooth or white?
lag1 = np.mean([np.corrcoef(res[j, :-1], res[j, 1:])[0, 1] for j in range(7)])
X = np.column_stack([np.ones(n_m), D, D**2])
fitted = np.vstack([X @ np.linalg.lstsq(X, res[j], rcond=None)[0] for j in range(7)])
r2 = 1 - np.sum((res - fitted) ** 2) / np.sum((res - res.mean()) ** 2)
mat_corr = np.mean([np.corrcoef(res[j], res[j + 1])[0, 1] for j in range(6)])
print(f"  lag-1 correlation along strikes {lag1:.2f}; adjacent-maturity correlation {mat_corr:.2f}; "
      f"share explained by a quadratic in delta per maturity {r2:.2f}")
json.dump({"valuation": str(valuation), "half_spread_table": {str(k): v for k, v in half_table.items()},
           "bins": bins, "heston_fit": x.tolist(), "residual_volpts": res.tolist(), "half_volpts": half.tolist(),
           "grid_delta_atm": D.tolist(), "lag1": lag1, "r2_quadratic": r2},
          open("spx_noise_and_model_error.json", "w"), indent=1)

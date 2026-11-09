"""Real quotes: the live SPY chain (Cboe delayed quotes), read onto the current and proposed grids
with the lab's own cleaning and spline reading (lab/src/sbi4finance/calibration/chain.py), then a
Heston (kappa = 2) weighted least-squares fit on each grid.

Usage: python -I test_spy_live.py /path/to/SPY.json
"""
import json
import pathlib
import sys

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent),
                str(pathlib.Path(__file__).resolve().parents[6] / "lab" / "src"), str(pathlib.Path(__file__).resolve().parents[6] / "tutorial" / "src")]
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from sbi4finance.calibration import chain  # noqa: E402
from common import (D_CURRENT, D_PROPOSED, PRIOR, NAMES, T_DESIGN, DAYS, SIGMA_REF, heston_prices,  # noqa: E402
                    heston_varswap_vol, jacobian, s_iv, surface)

if len(sys.argv) > 2:
    PRIOR[2, 1] = float(sys.argv[2])  # relax the vol-of-vol bound
rows = chain.parse_cboe_json(open(sys.argv[1]).read())
valuation = rows[0]["date"]
# Rates of 20 Oct 2026 (the tutorial's curve); the forward comes from parity, so this barely matters.
rate = chain.rate_curve([(0.25, 0.0426), (2.0, 0.0479)])

_orig_noise = chain.surface.implied_vol_noise
chain.surface.implied_vol_noise = lambda d: 0.0 * np.asarray(d, float)   # so noise_sd = vega(20%) x raw half-spread


def read(days, deltas_ref):
    out = chain.build_surface(rows, rate=rate, design_days=days, deltas=tuple(deltas_ref))
    return out


base = read(DAYS, D_CURRENT)
diag = base["diagnostics"]
iv0 = np.array(diag["implied_vol_grid"]).reshape(7, 9)
atm = iv0[:, 4]
w_rep = np.array(diag["design_total_variance"])
vs_rep = np.sqrt(w_rep / T_DESIGN)
print(f"valuation {diag['valuation']}; expirations used {len(diag['expirations'])}; "
      f"G0 readings {diag['grid_point_readings']}; outside quoted strikes {diag['grid_points_outside_quoted_strikes']}")
print("ATM vol %:", np.round(atm * 100, 1))
print("variance-swap vol % replicated from strips reaching delta<=-8:", np.round(vs_rep * 100, 2))
print("strip reach (20%-vol delta) of expirations near design maturities:")
for e in diag["expirations"]:
    if e["days"] <= 260:
        print(f"   {e['expiration']} {e['days']:4d}d  {e['strip_deltas'][0]:7.2f} .. {e['strip_deltas'][1]:5.2f}  "
              f"(ATM-vol units {e['strip_deltas'][0]*SIGMA_REF/np.interp(e['days']/252, T_DESIGN, atm):6.2f})")


def grid_data(kind_name):
    """Prices, raw half-spreads (vol pts) and contracts for one grid, read with the lab pipeline."""
    K, TT, KIND, IV, HALF, DREF, DATM, MISS = ([] for _ in range(8))
    readings = {"two expirations": 0, "one expiration": 0, "extrapolated": 0}
    for j, (day, t) in enumerate(zip(DAYS, T_DESIGN)):
        if kind_name == "G0":
            d_ref = D_CURRENT
        else:  # G1: proposed deltas in ATM-vol units, converted to the pipeline's 20%-vol units
            d_ref = D_PROPOSED * atm[j] / SIGMA_REF
        out = read((day,), d_ref)
        for key, v in out["diagnostics"]["grid_point_readings"].items():
            readings[key] += v
        k = d_ref * SIGMA_REF * np.sqrt(t)
        iv = np.array(out["diagnostics"]["implied_vol_grid"])
        half = out["noise_sd"] / surface.bs_vega(k, t, SIGMA_REF)
        K += list(k); TT += [t] * 9; KIND += ["put" if d <= 0 else "call" for d in d_ref]
        IV += list(iv); HALF += list(half); DREF += list(d_ref); DATM += list(k / (atm[j] * np.sqrt(t)))
        MISS += list(out["missing"])
    a = {n: np.array(v) for n, v in zip(("k", "T", "kind", "iv", "half", "d_ref", "d_atm", "missing"),
                                        (K, TT, KIND, IV, HALF, DREF, DATM, MISS))}
    a["price"] = surface.bs_price(a["k"], a["T"], a["iv"], a["kind"])
    a["readings"] = readings
    return a


def fit(a, sd, label, starts=8, seed=1):
    def resid(x):
        return (heston_prices(x, a["k"], a["T"], a["kind"], nodes=1024, upper=600.0) - a["price"]) / sd
    rng = np.random.default_rng(seed)
    best = None
    for s in range(starts):
        x0 = PRIOR[:, 0] + (PRIOR[:, 1] - PRIOR[:, 0]) * (0.15 + 0.7 * rng.random(4))
        try:
            r = least_squares(resid, x0, bounds=(PRIOR[:, 0], PRIOR[:, 1]), x_scale=PRIOR[:, 1] - PRIOR[:, 0],
                              xtol=1e-10, ftol=1e-10, max_nfev=400)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    res = (heston_prices(x, a["k"], a["T"], a["kind"]) - a["price"]) / sd
    J = jacobian(lambda r: heston_prices(r, a["k"], a["T"], a["kind"]), x) / sd[:, None]
    C = np.linalg.inv(J.T @ J)
    model_iv = surface.implied_vol(heston_prices(x, a["k"], a["T"], a["kind"]), a["k"], a["T"], a["kind"])
    iv_res = (model_iv - a["iv"]).reshape(7, 9) * 100
    vs = heston_varswap_vol(x)
    feller = x[2] / np.sqrt(2 * 2.0 * x[1])
    print(f"\n--- {label} ---")
    print("  fit:", "  ".join(f"{n}={v:.4f}±{s:.4f}" for n, v, s in zip(NAMES, x, np.sqrt(np.diag(C)))),
          f"  vol_of_vol / Feller limit = {feller:.2f}")
    print(f"  RMS residual {np.sqrt(np.mean(res**2)):.2f} noise sd; |res|>2: {int(np.sum(np.abs(res) > 2))}/63; "
          f"max |res| {np.max(np.abs(res)):.1f}")
    print("  mean residual by column (noise sd):", np.round(res.reshape(7, 9).mean(axis=0), 2))
    print("  model-minus-market IV (vol pts), rows 1w..1y:")
    for j in range(7):
        print("     ", " ".join(f"{v:6.2f}" for v in iv_res[j]))
    print("  variance-swap vol: model", np.round(vs * 100, 2))
    print("                     gap   ", np.round((vs - vs_rep) * 100, 2), "vol pts (model - replicated)")
    return {"x": x.tolist(), "sd": np.sqrt(np.diag(C)).tolist(), "rms": float(np.sqrt(np.mean(res**2))),
            "n_over_2": int(np.sum(np.abs(res) > 2)), "col_resid": res.reshape(7, 9).mean(axis=0).tolist(),
            "iv_resid": iv_res.tolist(), "varswap_gap": ((vs - vs_rep) * 100).tolist(), "feller_ratio": float(feller)}


results = {"valuation": diag["valuation"], "atm": atm.tolist(), "varswap_replicated": vs_rep.tolist()}
grids = {"G0": grid_data("G0"), "G1": grid_data("G1")}
for name, a in grids.items():
    print(f"\n{name}: readings {a['readings']}; points outside quoted strikes {int(a['missing'].sum())}")
    print("  20%-vol delta range 1w:", np.round(a["d_ref"][[0, 8]], 2), " 1y:", np.round(a["d_ref"][[-9, -1]], 2))
    print("  raw half-spread (vol pts) by column, median over maturities:",
          np.round(np.median(a["half"].reshape(7, 9), axis=0) * 100, 2))
    print("  ... max over maturities:", np.round(np.max(a["half"].reshape(7, 9), axis=0) * 100, 2))
    # how much of the variance swap the 9 grid points alone replicate
    w_grid = [chain.surface.total_variance_by_replication(a["k"][9*j:9*j+9], a["price"][9*j:9*j+9]) for j in range(7)]
    print("  variance-swap vol from the 9 grid points only:", np.round(np.sqrt(np.array(w_grid) / T_DESIGN) * 100, 2))

a0, a1 = grids["G0"], grids["G1"]
# noise conventions
sd_current = surface.bs_vega(a0["k"], a0["T"], SIGMA_REF) * np.maximum(a0["half"], _orig_noise(a0["d_ref"]))
results["G0_current_noise"] = fit(a0, sd_current, "G0 current grid, current noise (vega at 20%, the tutorial's)")
for name, a in grids.items():
    sd_local = surface.bs_vega(a["k"], a["T"], a["iv"]) * np.maximum(a["half"], s_iv(a["d_atm"]))
    results[f"{name}_local_noise"] = fit(a, sd_local, f"{name}, local noise (vega at the quote's own IV)")
json.dump(results, open(f"spy_live_xi{PRIOR[2, 1]:g}.json", "w"), indent=1)

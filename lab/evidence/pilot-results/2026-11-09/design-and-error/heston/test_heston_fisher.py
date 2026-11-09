"""Heston: information each grid carries about (v0, theta, vol_of_vol, rho), kappa = 2 fixed.

Laplace (flat-prior) posterior sd = sqrt(diag F^-1), F = J_w' J_w, J_w = d price / d theta / noise sd.
Strikes of ATM-scaled grids are fixed at the evaluated parameters' own ATM vols (the observed contracts).
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json
import sys
import numpy as np
from common import (GRIDS, PRIOR, NAMES, T_DESIGN, heston_prices, heston_atm_vol, grid_contracts,
                    noise_design, noise_local, jacobian, varswap_jacobian, surface)

rng = np.random.default_rng(20261109)
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
points = {"spy_21oct_median": np.array([0.01817, 0.06565, 1.1405, -0.6311]),
          "prior_centre": PRIOR.mean(axis=1)}
draws = PRIOR[:, 0] + (PRIOR[:, 1] - PRIOR[:, 0]) * rng.random((N_DRAWS, 4))


def analyse(row):
    atm = heston_atm_vol(row)
    out = {"atm_vol": atm.tolist()}
    gv = varswap_jacobian(row)[-1]          # gradient of the 1y variance-swap vol
    for name, (deltas, scaling) in GRIDS.items():
        k, T, kind, d, scale = grid_contracts(deltas, scaling, atm)
        prices = heston_prices(row, k, T, kind)
        J = jacobian(lambda r: heston_prices(r, k, T, kind), row)
        atm_c = np.repeat(atm, len(deltas))
        sd_design = noise_design(k, T, d, scale)
        sd_local, iv = noise_local(k, T, kind, prices, atm_c)
        res = {"ref_delta_range_1w": [float(k[0] / (0.2 * np.sqrt(T[0]))), float(k[8] / (0.2 * np.sqrt(T[0])))],
               "ref_delta_range_1y": [float(k[-9] / (0.2 * np.sqrt(T[-1]))), float(k[-1] / (0.2 * np.sqrt(T[-1])))]}
        for conv, sd in (("design", sd_design), ("local", sd_local)):
            Jw = J / sd[:, None]
            F = Jw.T @ Jw
            try:
                C = np.linalg.inv(F)
                post_sd = np.sqrt(np.diag(C))
                H = Jw @ C @ Jw.T
                lev = np.diag(H)
                col = lev.reshape(7, 9).sum(axis=0)
                put = float(lev[d < 0].sum()); atmc = float(lev[d == 0].sum()); call = float(lev[d > 0].sum())
                vs_sd = float(np.sqrt(gv @ C @ gv))
            except np.linalg.LinAlgError:
                post_sd = np.full(4, np.nan); col = np.full(9, np.nan); put = atmc = call = vs_sd = np.nan
            # effective vol-point noise each quote carries (price sd / vega at its own IV)
            eff = sd / surface.bs_vega(k, T, iv)
            res[conv] = {"post_sd": post_sd.tolist(), "logdet_F": float(np.linalg.slogdet(F)[1]),
                         "leverage_by_column": col.tolist(), "leverage_put_atm_call": [put, atmc, call],
                         "varswap_1y_sd": vs_sd,
                         "eff_iv_noise_by_column_volpts": (100 * eff.reshape(7, 9).mean(axis=0)).tolist()}
        out[name] = res
    return out


results = {"points": {}, "draws": []}
for key, row in points.items():
    results["points"][key] = {"row": row.tolist(), **analyse(row)}
for i, row in enumerate(draws):
    try:
        results["draws"].append({"row": row.tolist(), **analyse(row)})
    except Exception as e:  # report and continue
        results["draws"].append({"row": row.tolist(), "error": repr(e)})
json.dump(results, open("heston_fisher.json", "w"), indent=1)

# ---- summary ----
g0 = list(GRIDS)[0]
for conv in ("design", "local"):
    print(f"\n=== noise convention: {conv} ===")
    for key in points:
        r = results["points"][key]
        print(f"\n[{key}] ATM vols {np.round(np.array(r['atm_vol'])*100,1)}")
        print(f"{'grid':34s} " + " ".join(f"{n:>11s}" for n in NAMES) + "   varswap1y  put/atm/call leverage")
        for name in GRIDS:
            s = r[name][conv]
            print(f"{name:34s} " + " ".join(f"{x:11.2e}" for x in s["post_sd"])
                  + f"  {s['varswap_1y_sd']*100:8.3f}vp  " + "/".join(f"{x:.2f}" for x in s["leverage_put_atm_call"]))
        print("  effective vol-point noise by column (G0):", np.round(r[g0][conv]["eff_iv_noise_by_column_volpts"], 2))
        print("  leverage by column (G0):", np.round(r[g0][conv]["leverage_by_column"], 2))
    ok = [d for d in results["draws"] if "error" not in d]
    print(f"\n[prior draws: {len(ok)} ok of {len(results['draws'])}] median ratio of posterior sd to G0 (<1 = tighter)")
    for name in list(GRIDS)[1:]:
        ratios = np.array([np.array(d[name][conv]["post_sd"]) / np.array(d[g0][conv]["post_sd"]) for d in ok])
        vs = np.array([d[name][conv]["varswap_1y_sd"] / d[g0][conv]["varswap_1y_sd"] for d in ok])
        q = lambda a: f"{np.nanmedian(a):.2f} [{np.nanquantile(a, .1):.2f},{np.nanquantile(a, .9):.2f}]"
        print(f"{name:34s} " + "  ".join(f"{n}:{q(ratios[:, j])}" for j, n in enumerate(NAMES)) + f"  varswap1y:{q(vs)}")
    lev = np.array([d[g0][conv]["leverage_put_atm_call"] for d in ok])
    print("G0 leverage put/atm/call, median over draws:", np.round(np.nanmedian(lev, axis=0), 2))
    lev1 = np.array([d[list(GRIDS)[1]][conv]["leverage_put_atm_call"] for d in ok])
    print("G1 leverage put/atm/call, median over draws:", np.round(np.nanmedian(lev1, axis=0), 2))

"""Print compact report tables from results.json."""
import json
import os

import numpy as np

R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")))
cols = ["H", "eta", "rho", "xi_s", "xi_l", "decay", "VSvol1y", "ATMiv1y"]
print("posterior-sd ratio vs G0 (same noise convention): median over points [SPY-like] {min-max}")
print("grid/conv  | " + " | ".join(f"{c:>17s}" for c in cols))
for key, t in R["ratio_vs_G0"]["table"].items():
    if key.startswith("G0"):
        continue
    cells = [f"{m:5.2f} [{s:5.2f}] {{{lo:4.2f}-{hi:4.2f}}}" for m, s, lo, hi in
             zip(t["median_all"], t["spy_like"], t["min"], t["max"])]
    print(f"{key:10s} | " + " | ".join(cells))
print("\nG0 local/design sd ratio, median [SPY]:")
M = np.array(list(R["G0_local_over_design"].values()))
print(" ", np.round(np.nanmedian(M, 0), 2), np.round(R["G0_local_over_design"]["spy_like"], 2))
print("\nabsolute G0 posterior sd (local | design) and products:")
for n, p in R["points"].items():
    gl, gd = p["grids"]["G0/local"], p["grids"]["G0/design"]
    print(f"  {n:9s} theta={np.round(p['theta'], 3).tolist()} atm1y={p['sigma_atm'][6]:.3f} vs1y={p['vs_vol_1y']:.3f}")
    print(f"            local {np.round(gl['post_sd'], 4).tolist()} prod {gl['products_sd']}")
    print(f"            design{np.round(gd['post_sd'], 4).tolist()} prod {gd['products_sd']}")
print("\nleverage shares put/atm/call: median over points [SPY]")
for conv in ("design", "local"):
    for g in ("G0", "G1", "G2", "G3"):
        key = f"{g}/{conv}"
        sh = np.array([[p["grids"][key]["leverage_share"][s] for s in ("put", "atm", "call")]
                       for p in R["points"].values()])
        spy = [R["points"]["spy_like"]["grids"][key]["leverage_share"][s] for s in ("put", "atm", "call")]
        print(f"  {key:10s} {np.round(np.median(sh, 0), 2).tolist()}  [{np.round(spy, 2).tolist()}]")
if "mc_check" in R:
    print("\nMC check (seed A vs B) at", R["mc_check"]["point"])
    for key, v in R["mc_check"]["result"].items():
        j = max(v["jacobian_col_rel_diff"].values())
        s = max(abs(x) for x in v["post_sd_rel_diff"].values())
        print(f"  {key:10s} max whitened-J column rel diff {j:.3f}   max |post-sd rel change| {s:.3f}")

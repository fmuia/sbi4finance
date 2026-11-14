import pathlib
"""Effective vol-point noise per column under the tutorial's noise (vega at 20% x max(half, s_iv)), two dates."""
import json, sys
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent),
                str(pathlib.Path(__file__).resolve().parents[6] / "lab" / "src"), str(pathlib.Path(__file__).resolve().parents[6] / "tutorial" / "src")]
import numpy as np
from sbi4finance.calibration import chain
from common import D_CURRENT, T_DESIGN, SIGMA_REF, surface
k = np.concatenate([D_CURRENT * SIGMA_REF * np.sqrt(t) for t in T_DESIGN]); T = np.repeat(T_DESIGN, 9)
kind = np.where(np.tile(D_CURRENT, 7) <= 0, "put", "call")
def report(label, iv, noise):
    eff = noise / surface.bs_vega(k, T, iv)
    print(f"{label}: ATM vol {np.round(iv.reshape(7,9)[:,4]*100,1)}")
    print("   effective noise (vol pts) by column, median over maturities:", np.round(np.median(eff.reshape(7,9),0)*100, 2))
    print("   max over maturities:                                       ", np.round(np.max(eff.reshape(7,9),0)*100, 2))
d = json.load(open(str(pathlib.Path(__file__).resolve().parents[6]) + "/tutorial/assets/results/calibration/spy/surface.json"))
report("SPY 21 Oct (tutorial asset)", np.array(d["diagnostics"]["implied_vol_grid"]), np.array(d["noise_sd"]))
rows = chain.parse_cboe_json(open(sys.argv[1]).read())
out = chain.build_surface(rows, rate=chain.rate_curve([(0.25, 0.0426), (2.0, 0.0479)]))
report("SPY 6 Nov (live chain)", np.array(out["diagnostics"]["implied_vol_grid"]), out["noise_sd"])

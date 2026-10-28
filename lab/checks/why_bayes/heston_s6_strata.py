"""Study S6's 200 Heston smiles split by an observable: the fitted vol-of-vol below or above 0.3.

Reads the accepted S6 result (exact-Fourier least squares from ten starts, Gauss-Newton 90% intervals, and the
BayesFlow 2.0 network's 90% intervals) and reports, per stratum, the one-sided breaches of each method's 90%
interval, the cases with no Gauss-Newton interval, optima on a bound or two or more optima, and the ratio of the
Gauss-Newton width to the posterior's. The stratum is chosen from the data (the fitted vol-of-vol), so the
posterior's coverage within it is not disturbed by the choice.

Run from lab/: python checks/why_bayes/heston_s6_strata.py
"""

import json
from pathlib import Path

import numpy as np

SOURCE = Path("../tutorial/assets/results/calibration/heston/s6/classical.json")
OUT = Path("checks/why_bayes/heston_s6_strata.json")
NAMES = ["initial_variance", "theta", "vol_of_vol", "rho"]
CUT = 0.3

d = json.loads(SOURCE.read_text())
s = d["surfaces"]
T = np.array([r["truth"] for r in s])
B = np.array([r["best_theta"] for r in s])
gn = [r["gauss_newton_90"] for r in s]
po = np.array([r["posterior_90"] for r in s])
modes = np.array([r["modes"] for r in s])
onb = np.array([r["on_bound"] for r in s])
sing = np.array([g is None for g in gn])
small = B[:, 2] < CUT
result = {"source": str(SOURCE), "stratum": f"least-squares vol_of_vol below {CUT}", "strata": {}}
for label, m in ((f"vol_of_vol below {CUT}", small), (f"vol_of_vol {CUT} and above", ~small)):
    row = {"smiles": int(m.sum()), "optimum_on_bound": int((onb & m).sum()), "no_gauss_newton": int((sing & m).sum()),
           "two_or_more_optima": int(((modes > 1) & m).sum()), "parameters": {}}
    print(f"\n{label}: {row}")
    for j, name in enumerate(NAMES):
        n_g = lo_g = hi_g = lo_p = hi_p = 0
        ratio = []
        for i in np.flatnonzero(m):
            lo_p += T[i, j] < po[i, j, 0]
            hi_p += T[i, j] > po[i, j, 1]
            if gn[i] is None:
                continue
            a, b = gn[i][j]
            n_g += 1
            lo_g += T[i, j] < a
            hi_g += T[i, j] > b
            ratio.append((b - a) / (po[i, j, 1] - po[i, j, 0]))
        row["parameters"][name] = {
            "posterior_breach_below": lo_p / m.sum(), "posterior_breach_above": hi_p / m.sum(),
            "gauss_newton_breach_below": lo_g / max(n_g, 1), "gauss_newton_breach_above": hi_g / max(n_g, 1),
            "gauss_newton_smiles": n_g, "width_ratio_median": float(np.median(ratio)),
            "width_ratio_10": float(np.percentile(ratio, 10)), "width_ratio_90": float(np.percentile(ratio, 90))}
        p = row["parameters"][name]
        print(f"  {name:16s} posterior below/above {p['posterior_breach_below']:.2f}/{p['posterior_breach_above']:.2f} | "
              f"Gauss-Newton {p['gauss_newton_breach_below']:.2f}/{p['gauss_newton_breach_above']:.2f} on {n_g} | "
              f"width ratio {p['width_ratio_median']:.2f} ({p['width_ratio_10']:.2f}-{p['width_ratio_90']:.2f})")
    result["strata"][label] = row
OUT.write_text(json.dumps(result, indent=1) + "\n")

"""Which region of the SPX smile informs which quantity? Wide grid (-6..+2 ATM-sd, 17/maturity), smooth
model error (tau = 1 vol pt) + measured half-spreads. Unique contribution of a region = % increase in
posterior sd when that region is left out."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import grid_vs_error_structure as g   # reuses the model, error model and grids (prints its own tables first)

deltas, scale = g.GRIDS["wide put-heavy: -6..+2 ATM-sd, 17/maturity"]
k, T, kind, d = g.grid(deltas, scale)
E = np.eye(5)
J = np.column_stack([(g.heston_iv(g.X0 + E[i]*g.steps[i], k, T, kind) - g.heston_iv(g.X0 - E[i]*g.steps[i], k, T, kind))
                     / (2*g.steps[i]) for i in range(5)])
gv = np.array([(g.varswap_1y(g.X0 + E[i]*g.steps[i]) - g.varswap_1y(g.X0 - E[i]*g.steps[i])) / (2*g.steps[i]) for i in range(5)])
S = g.covariance(d, T, g.half_spread(d, T), "corr")
def sds(mask):
    Si = np.linalg.inv(S[np.ix_(mask, mask)])
    C = np.linalg.inv(J[mask].T @ Si @ J[mask])
    return np.r_[np.sqrt(np.diag(C)), np.sqrt(gv @ C @ gv)]
full = sds(np.ones(len(k), bool))
regions = {"deep puts (< -3 ATM-sd)": d < -3, "near puts (-3..-0.5)": (d >= -3) & (d < -0.5),
           "at the money (|d| <= 0.5)": np.abs(d) <= 0.5, "calls (> +0.5)": d > 0.5,
           "short maturities (<= 1m)": T <= 21/252 + 1e-9, "long maturities (>= 2m)": T >= 42/252 - 1e-9}
print("\n% increase in posterior sd when the region is removed (bigger = the region carries unique information)")
print(f"{'region':30s} " + " ".join(f"{n:>9s}" for n in g.NAMES + ["varswap1y"]))
for name, m in regions.items():
    r = sds(~m) / full - 1
    print(f"{name:30s} " + " ".join(f"{100*v:8.0f}%" for v in r))

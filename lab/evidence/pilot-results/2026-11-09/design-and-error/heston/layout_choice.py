"""Pick a 9-point-per-maturity layout (63 contracts, so 7x9 code keeps working) for SPX.
Heston (kappa free) at the SPX fit; correlated model error + measured SPX half-spreads.
Deltas are in ATM-sd units at the fit (SPX ATM 15.6-17.1%, i.e. a fixed 16% scale in practice)."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import grid_vs_error_structure as g
E = np.eye(5)
LAYOUTS = {
    "current (+-2 at 20%)": (np.linspace(-2, 2, 9), "ref"),
    "A: -6 -4.5 -3.25 -2.25 -1.5 -0.75 0 +0.75 +1.5": (np.array([-6, -4.5, -3.25, -2.25, -1.5, -.75, 0, .75, 1.5]), "atm"),
    "B: -6 -4 -2.75 -1.75 -1 -0.5 0 +0.75 +1.75": (np.array([-6, -4, -2.75, -1.75, -1, -.5, 0, .75, 1.75]), "atm"),
    "C: -5 -3.75 -2.5 -1.5 -0.75 0 +0.5 +1 +2": (np.array([-5, -3.75, -2.5, -1.5, -.75, 0, .5, 1, 2]), "atm"),
    "D: -6 -5 -4 -3 -2 -1 0 +1 +2": (np.linspace(-6, 2, 9), "atm"),
    "E (chosen): -5 -3.5 -2.25 -1.25 -0.5 0 +0.5 +1 +2": (np.array([-5, -3.5, -2.25, -1.25, -.5, 0, .5, 1, 2]), "atm"),
    "F: -4 -3 -2 -1.25 -0.5 0 +0.5 +1 +2": (np.array([-4, -3, -2, -1.25, -.5, 0, .5, 1, 2]), "atm"),
    "G: -5 -3.5 -2.5 -1.5 -0.75 -0.25 +0.25 +1 +2": (np.array([-5, -3.5, -2.5, -1.5, -.75, -.25, .25, 1, 2]), "atm"),
}
def widths(deltas, scale, tau, l_d, l_t):
    k, T, kind, d = g.grid(deltas, scale)
    J = np.column_stack([(g.heston_iv(g.X0 + E[i]*g.steps[i], k, T, kind) - g.heston_iv(g.X0 - E[i]*g.steps[i], k, T, kind))
                         / (2*g.steps[i]) for i in range(5)])
    gv = np.array([(g.varswap_1y(g.X0 + E[i]*g.steps[i]) - g.varswap_1y(g.X0 - E[i]*g.steps[i])) / (2*g.steps[i]) for i in range(5)])
    C = np.linalg.inv(J.T @ np.linalg.inv(g.covariance(d, T, g.half_spread(d, T), "corr", tau=tau/100, l_d=l_d, l_t=l_t)) @ J)
    return np.r_[np.sqrt(np.diag(C)), np.sqrt(gv @ C @ gv)]
for tau, l_d, l_t in [(1, .75, 1.15), (2.5, 1.6, 2.0)]:
    print(f"\nsd ratio vs current, smooth model error tau={tau} vp, l_delta={l_d}, l_logT={l_t}")
    print(f"{'layout':52s} " + " ".join(f"{n:>8s}" for n in g.NAMES + ["varswap"]))
    base = widths(*LAYOUTS["current (+-2 at 20%)"], tau, l_d, l_t)
    for name, (dl, sc) in LAYOUTS.items():
        r = widths(dl, sc, tau, l_d, l_t) / base
        print(f"{name:52s} " + " ".join(f"{v:8.2f}" for v in r))

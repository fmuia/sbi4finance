"""Does 'coverage beats density' survive a larger, smoother model error (noise session's joint fits:
tau_g ~ 2.1-2.6 vol pts, l_delta ~ 1.6, l_T ~ 1.5-2.2)?"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import grid_vs_error_structure as g
E = np.eye(5)
def widths(name, tau, l_d, l_t):
    deltas, scale = g.GRIDS[name]
    k, T, kind, d = g.grid(deltas, scale)
    J = np.column_stack([(g.heston_iv(g.X0 + E[i]*g.steps[i], k, T, kind) - g.heston_iv(g.X0 - E[i]*g.steps[i], k, T, kind))
                         / (2*g.steps[i]) for i in range(5)])
    gv = np.array([(g.varswap_1y(g.X0 + E[i]*g.steps[i]) - g.varswap_1y(g.X0 - E[i]*g.steps[i])) / (2*g.steps[i]) for i in range(5)])
    C = np.linalg.inv(J.T @ np.linalg.inv(g.covariance(d, T, g.half_spread(d, T), "corr", tau=tau/100, l_d=l_d, l_t=l_t)) @ J)
    return np.r_[np.sqrt(np.diag(C)), np.sqrt(gv @ C @ gv) * 100]
cur, dense, wide9, wide17 = list(g.GRIDS)
print(f"\n{'tau':>4s} {'l_d':>4s} {'l_t':>4s}  {'varswap sd cur':>14s} | ratio dense/cur: xi rho varswap | ratio wide17/cur: kappa theta xi rho varswap")
for tau, l_d, l_t in [(1, .75, 1.15), (1, 1.6, 2.0), (2.5, 1.6, 2.0), (2.5, 2.2, 2.0), (2.5, 1.6, 1.5)]:
    c, dn, w = widths(cur, tau, l_d, l_t), widths(dense, tau, l_d, l_t), widths(wide17, tau, l_d, l_t)
    print(f"{tau:4.1f} {l_d:4.2f} {l_t:4.2f}  {c[5]:11.2f} vp | {dn[3]/c[3]:.2f} {dn[4]/c[4]:.2f} {dn[5]/c[5]:.2f}"
          f"            | {w[0]/c[0]:.2f} {w[2]/c[2]:.2f} {w[3]/c[3]:.2f} {w[4]/c[4]:.2f} {w[5]/c[5]:.2f}")

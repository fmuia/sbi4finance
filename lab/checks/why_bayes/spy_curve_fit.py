"""Fit the joint task's Heston-shaped curve to SPY's replicated forward-variance levels (21 Oct 2026).

The levels are interval averages of xi_0 between the design maturities, read by variance-swap replication
(tutorial/assets/results/calibration/spy/surface.json). This is a curve-only least-squares fit with 5% relative
error per level, the long-end quick check's assumption (lab/checks/long_end); it only places SPY in the decay
regimes of the joint-task screen and is not a calibration of rough Bergomi. The LEAP comparison uses the 2.23-year
variance-swap volatility of lab/checks/long_end/long_end_extra.json.

Run from lab/: python checks/why_bayes/spy_curve_fit.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

surface = json.loads(Path("../tutorial/assets/results/calibration/spy/surface.json").read_text())
leap_vol = json.loads(Path("checks/long_end/long_end_extra.json").read_text())["leaps"]["2.23"] / 100
OUT = Path("checks/why_bayes/spy_curve_fit.json")
days = np.array(surface["design"]["days"], dtype=float)
knots = np.r_[0.0, days / 252]
levels = np.array(surface["forward_variance_levels"])


def integrated(t, s, l, k):
    t = np.asarray(t, dtype=float)
    return l * t + (s - l) * (-np.expm1(-k * t)) / k


def averages(p):
    s, l, k = p
    w = integrated(knots, s, l, k)
    return np.diff(w) / np.diff(knots)


def residual(p):
    return (averages(p) - levels) / (0.05 * levels)


best = None
for k0 in (0.3, 1.0, 3.0, 8.0):
    r = least_squares(residual, [levels[0], levels[-1] * 1.1, k0], bounds=([1e-4, 1e-4, 0.05], [0.5, 0.5, 50]))
    if best is None or r.cost < best.cost:
        best = r
s, l, k = best.x
J = best.jac
cov = np.linalg.inv(J.T @ J)
sd = np.sqrt(np.diag(cov))
print("levels", np.round(levels, 4).tolist())
print("fit", np.round(averages(best.x), 4).tolist(), "chi2 %.2f on %d levels" % (2 * best.cost, len(levels)))
print(f"short {s:.4f} +- {sd[0]:.4f}, long {l:.4f} +- {sd[1]:.4f}, decay {k:.3f} +- {sd[2]:.3f} per year;"
      f" corr(long, decay) {cov[1, 2] / sd[1] / sd[2]:.2f}")
w1, w2, w223 = integrated([1.0, 2.0, 2.23], s, l, k)
# The LEAP's total variance to 2.23 years less the replicated total to one year, per year.
leap_forward = (leap_vol**2 * 2.23 - surface["diagnostics"]["design_total_variance"][-1]) / 1.23
print(f"implied 2y variance-swap vol {np.sqrt(w2 / 2):.4f}; 1y->2y forward variance {w2 - w1:.4f};"
      f" 1y->2.23y forward variance {(w223 - w1) / 1.23:.4f} against the LEAP-implied {leap_forward:.4f}")
result = {"levels": levels.tolist(), "fitted_levels": averages(best.x).tolist(), "chi2": 2 * best.cost,
          "relative_error_per_level": 0.05, "short_variance": s, "long_variance": l, "decay": k,
          "sd": sd.tolist(), "forward_variance_1y_to_2y": w2 - w1, "forward_variance_1y_to_2.23y": (w223 - w1) / 1.23,
          "leap_forward_variance_1y_to_2.23y": leap_forward, "decay_fixed": []}
for fixed_k in (0.25, 0.5, 1.0, 2.0, 4.0):
    r = least_squares(lambda q: residual([q[0], q[1], fixed_k]), [levels[0], levels[-1]], bounds=([1e-4, 1e-4], [0.5, 0.5]))
    a, b = r.x
    w1, w2 = integrated([1.0, 2.0], a, b, fixed_k)
    result["decay_fixed"].append({"decay": fixed_k, "chi2": 2 * r.cost, "short_variance": a, "long_variance": b,
                                  "forward_variance_1y_to_2y": w2 - w1})
    print(f"  decay fixed at {fixed_k}: chi2 {2 * r.cost:.2f}, short {a:.4f}, long {b:.4f}, 1y->2y forward variance {w2 - w1:.4f}")
OUT.write_text(json.dumps(result, indent=1) + "\n")

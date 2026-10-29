"""Local Gauss-Newton screen of the joint-curve rough-Bergomi task (PR 48's rough_joint_calibration_v1).

At a few true parameter vectors it computes, with the task's own Monte Carlo pricer and common random numbers:
the expected Fisher information on the 63 design quotes, the Gauss-Newton covariance with the curve inferred
(6x6) and with the curve given exactly (the 3x3 (H, eta, rho) block), the weakest directions in prior-width
units, delta-method intervals for curve products beyond the last quote against their hard bounds, and a slice of
chi-square along the weakest direction against the quadratic the Gauss-Newton approximation predicts.

This is a screen for where to look, not a posterior: it uses noise-free quotes at the truth.

Needs the joint task of PR #48 (tasks.joint_rough_task). Run from lab/:
python checks/why_bayes/joint_gn_screen.py checks/why_bayes/joint_gn_screen.json
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sbi4finance.calibration import pricing, products, tasks

TASK = tasks.joint_rough_task()
CONTRACTS = TASK["context"]["contracts"]
SIGMA = np.asarray(TASK["context"]["quote_noise_std"], dtype=float)
BLOCKS = TASK["numerics"]["blocks"]
LO, HI = np.array(TASK["prior_bounds"], dtype=float).T
WIDTH = HI - LO
NAMES = tasks.JOINT_ROUGH_PARAMETERS
PATHS = 32768
PRODUCT_PATHS = 8192
SEED = 20261028
Z90 = 1.6449
POINTS = {
    "A rising, slow decay (SPY-like dynamics)": [0.09, 2.6, -0.73, 0.02, 0.06, 0.5],
    "B rising, fast decay": [0.09, 2.6, -0.73, 0.02, 0.06, 4.0],
    "C flat curve": [0.09, 2.6, -0.73, 0.04, 0.04, 2.0],
    "D falling curve (stressed)": [0.09, 2.6, -0.73, 0.10, 0.04, 1.5],
    "E rho near its bound": [0.07, 1.9, -0.95, 0.02, 0.06, 0.5],
    "F low vol-of-vol": [0.15, 0.8, -0.5, 0.03, 0.05, 1.0],
    "G rising, very slow decay": [0.09, 2.6, -0.73, 0.02, 0.08, 0.3],
    "H rising, slow decay, low eta": [0.12, 1.2, -0.6, 0.02, 0.06, 0.5],
}


def surface(theta):
    theta = np.asarray(theta, dtype=float)
    p, _, _, _ = pricing.rough_surface(tasks.model_parameters(TASK, theta), tasks.joint_curve(TASK, theta),
                                       CONTRACTS, BLOCKS, [PATHS], SIGMA, 0.25, SEED, cap=np.inf)
    return p


def curve_products(theta):
    """2-year variance-swap strike and the 1y->2y forward variance: deterministic functions of the curve."""
    s, l, k = theta[3:]
    w = pricing.heston_integrated_variance(np.array([1.0, 2.0]), s, l, k)
    return np.array([w[1] / 2.0, w[1] - w[0]])


def mc_products(theta):
    """The tutorial's forward start (63->126 days) and 2-year 90% put, on one seed."""
    spec = products.default_specification()
    times = np.arange(0, 2 * 252 + 1) / 252
    curve = pricing.heston_variance_curve(times, *theta[3:])
    prices, _ = products.price_draw("rough", tasks.model_parameters(TASK, theta), curve, spec, PRODUCT_PATHS, SEED)
    return prices[[2, 3]]  # forward_start_call, put_out_of_grid


def jacobian(f, theta, rel=1e-3):
    theta = np.asarray(theta, dtype=float)
    cols = []
    for j in range(len(theta)):
        h = rel * WIDTH[j]
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        cols.append((f(up) - f(dn)) / (2 * h))
    return np.column_stack(cols)


def analyse(item):
    name, theta = item
    theta = np.asarray(theta, dtype=float)
    t0 = time.time()
    base = surface(theta)
    J = jacobian(surface, theta) / SIGMA[:, None]
    F = J.T @ J
    cov = np.linalg.inv(F)
    sd_joint = np.sqrt(np.diag(cov))
    sd_known = np.sqrt(np.diag(np.linalg.inv(F[:3, :3])))
    Fs = F * np.outer(WIDTH, WIDTH)  # information in prior-width units
    evals, evecs = np.linalg.eigh(Fs)
    weakest = evecs[:, 0] * np.sign(evecs[np.argmax(np.abs(evecs[:, 0])), 0])
    gn90 = np.column_stack([theta - Z90 * sd_joint, theta + Z90 * sd_joint])
    crosses = {NAMES[j]: [bool(gn90[j, 0] < LO[j]), bool(gn90[j, 1] > HI[j])] for j in range(6)}
    # Curve products beyond the last quote, with their hard bounds implied by the prior box (levels in [0.01, 0.16]).
    g_curve = jacobian(curve_products, theta, rel=1e-4)
    cp = curve_products(theta)
    cp_sd = np.sqrt(np.einsum("ij,jk,ik->i", g_curve, cov, g_curve))
    curve_rows = []
    for i, label in enumerate(["2y variance swap strike (variance)", "1y->2y forward variance"]):
        lo, hi = cp[i] - Z90 * cp_sd[i], cp[i] + Z90 * cp_sd[i]
        curve_rows.append({"product": label, "value": float(cp[i]), "gn_sd": float(cp_sd[i]),
                           "gn90": [float(lo), float(hi)], "below_0.01": bool(lo < 0.01),
                           "above_0.16": bool(hi > 0.16), "rel_sd": float(cp_sd[i] / cp[i])})
    # Exotics priced by Monte Carlo: delta-method sd with the curve inferred and with it given.
    g_mc = jacobian(mc_products, theta)
    mc = mc_products(theta)
    mc_sd_joint = np.sqrt(np.einsum("ij,jk,ik->i", g_mc, cov, g_mc))
    cov_known = np.linalg.inv(F[:3, :3])
    mc_sd_known = np.sqrt(np.einsum("ij,jk,ik->i", g_mc[:, :3], cov_known, g_mc[:, :3]))
    # Slice of chi-square along the weakest direction, in units of the Gauss-Newton sd along it.
    step = weakest * WIDTH / np.sqrt(evals[0])  # one Gauss-Newton sd along the weakest direction, natural units
    slice_rows = []
    for t in (-3, -2, -1.6449, -1, 1, 1.6449, 2, 3):
        th = theta + t * step
        inside = bool(np.all(th >= LO) and np.all(th <= HI))
        valid = bool(th[0] > 0 and th[0] < 0.5 and abs(th[2]) < 1 and np.all(th[3:] > 0) and th[1] > 0)
        chi2 = float(np.sum(((surface(th) - base) / SIGMA) ** 2)) if valid else None
        slice_rows.append({"t_sd": t, "inside_prior": inside, "chi2": chi2, "quadratic": t * t})
    return {
        "name": name, "theta": theta.tolist(), "seconds": time.time() - t0,
        "sd_joint": dict(zip(NAMES, sd_joint.tolist())),
        "sd_known_curve": dict(zip(NAMES[:3], sd_known.tolist())),
        "sd_ratio_joint_over_known": dict(zip(NAMES[:3], (sd_joint[:3] / sd_known).tolist())),
        "sd_over_prior_width": dict(zip(NAMES, (sd_joint / WIDTH).tolist())),
        "correlation": np.round(cov / np.outer(sd_joint, sd_joint), 3).tolist(),
        "scaled_eigenvalues": evals.tolist(),
        "weakest_direction": dict(zip(NAMES, np.round(weakest, 3).tolist())),
        "weakest_sd_in_prior_widths": float(1 / np.sqrt(evals[0])),
        "gn90_crosses_prior_box": crosses,
        "curve_products": curve_rows,
        "mc_products": {"names": ["forward_start_call", "put_2y_90"], "price": mc.tolist(),
                        "gn_sd_joint": mc_sd_joint.tolist(), "gn_sd_known_curve": mc_sd_known.tolist(),
                        "rel_sd_joint": (mc_sd_joint / mc).tolist(), "rel_sd_known_curve": (mc_sd_known / mc).tolist()},
        "slice_weakest": slice_rows,
    }


if __name__ == "__main__":
    out = sys.argv[1]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(analyse, POINTS.items()))
    json.dump({"paths": PATHS, "product_paths": PRODUCT_PATHS, "seed": SEED, "names": NAMES,
               "prior_bounds": TASK["prior_bounds"], "results": results}, open(out, "w"), indent=1)
    for r in results:
        print(f"\n== {r['name']}  theta={r['theta']}  ({r['seconds']:.0f}s)")
        print("  sd joint:", {k: round(v, 4) for k, v in r["sd_joint"].items()})
        print("  sd ratio joint/known:", {k: round(v, 2) for k, v in r["sd_ratio_joint_over_known"].items()})
        print("  sd / prior width:", {k: round(v, 3) for k, v in r["sd_over_prior_width"].items()})
        print("  weakest direction:", r["weakest_direction"], "sd in prior widths %.3f" % r["weakest_sd_in_prior_widths"])
        print("  GN90 crosses box:", {k: v for k, v in r["gn90_crosses_prior_box"].items() if any(v)})
        for c in r["curve_products"]:
            print(f"  {c['product']}: {c['value']:.4f} sd {c['gn_sd']:.4f} ({100*c['rel_sd']:.1f}%) gn90 [{c['gn90'][0]:.4f}, {c['gn90'][1]:.4f}]"
                  f"{'  BELOW 0.01' if c['below_0.01'] else ''}{'  ABOVE 0.16' if c['above_0.16'] else ''}")
        m = r["mc_products"]
        for i, n in enumerate(m["names"]):
            print(f"  {n}: price {m['price'][i]:.5f}, rel sd joint {100*m['rel_sd_joint'][i]:.2f}%, known-curve {100*m['rel_sd_known_curve'][i]:.2f}%")
        print("  chi2 slice along weakest (t sd: chi2 vs t^2):",
              [(s["t_sd"], None if s["chi2"] is None else round(s["chi2"], 2), s["inside_prior"]) for s in r["slice_weakest"]])

"""Desk case, validation: how often is each method's one-sided 90% reserve breached on simulated thin chains?

    python validate.py OUT_DIR [--replicas 80] [--scale 1.0]

Truths: parameter sets resampled from the full 63-quote SPY posterior (rough_spy_is.py, seed 0), the values this
market is believed to have. For each truth, a thin chain is simulated: the 27 front-month quotes priced by an
independent Monte Carlo (seed 7, 32,768 paths) plus Gaussian quote noise at the given scale. Each method then
marks the four products and sets one-sided 90% reserves from that chain alone:
  Gauss–Newton and the delta method (least squares started at the best priced proposal);
  desk practice, H fixed at 0.10 (least squares on eta and rho);
  the profile-likelihood range (chi-square within 1.64 of its minimum), read off the desk-case proposals;
  the posterior, by importance sampling of the desk-case proposals (priced once, reweighted per chain).
A breach is a true price below the long prudent value (mark - long reserve) or above the short one (mark + short
reserve). A correct 90% one-sided reserve is breached 10% of the time on each side.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import desk_case as dc  # noqa: E402
import desk_products as dp  # noqa: E402
import rough_spy_is as base  # noqa: E402
from sbi4finance.calibration import correction  # noqa: E402

DATA_SEED = 7
CASE = {}


def _load(out_dir):
    case = np.load(out_dir / "desk_case.npz")
    CASE.update(theta=case["theta"], log_q=case["log_q"], prices=case["prices"], products=case["products"])


def replica_job(args):
    """Gauss–Newton and H-fixed marks and delta-method sds for one simulated chain; the truth's product prices."""
    r, truth, y, scale, start = args
    out = {"replica": r}
    rows = dc.FRONT

    def fit(free, x0):
        lo, hi = base.BOUNDS[free, 0], base.BOUNDS[free, 1]

        def theta_of(z):
            t = x0.copy()
            t[free] = lo + (hi - lo) / (1 + np.exp(-z))
            return t

        def residuals(z):
            return (base.surface(theta_of(z), dc.SEED)[0][rows] - y[rows]) / base.SIGMA[rows]

        from scipy.optimize import least_squares
        z0 = np.log((x0[free] - lo) / (hi - x0[free]))
        res = least_squares(residuals, z0, method="trf", diff_step=1e-4, max_nfev=200, x_scale="jac")
        return theta_of(np.clip(res.x, -30, 30)), float(2 * res.cost)

    def jac(theta, free, job):
        cols = []
        for i in free:
            up, dn = theta.copy(), theta.copy()
            up[i] += dc.STEP[i]
            dn[i] -= dc.STEP[i]
            a, b = job((up, dc.SEED)), job((dn, dc.SEED))
            a, b = (a[0], b[0]) if isinstance(a, tuple) else (a, b)
            cols.append((a - b) / (2 * dc.STEP[i]))
        return np.stack(cols, axis=1)

    theta_g, chi2_g = fit([0, 1, 2], start)
    jn = jac(theta_g, [0, 1, 2], base.surface_job)[rows] / (scale * base.SIGMA[rows, None])
    cov = np.linalg.inv(jn.T @ jn)
    grad = jac(theta_g, [0, 1, 2], dc.product_job)
    out.update(theta_gn=theta_g.tolist(), chi2_gn=chi2_g, mark_gn=dc.product_job((theta_g, dc.SEED)).tolist(),
               sd_gn=np.sqrt(np.einsum("pi,ij,pj->p", grad, cov, grad)).tolist())
    x0 = theta_g.copy()
    x0[0] = dc.H_DESK
    theta_x, chi2_x = fit([1, 2], x0)
    jn = jac(theta_x, [1, 2], base.surface_job)[rows] / (scale * base.SIGMA[rows, None])
    cov = np.linalg.inv(jn.T @ jn)
    grad = jac(theta_x, [1, 2], dc.product_job)
    out.update(theta_fix=theta_x.tolist(), chi2_fix=chi2_x, mark_fix=dc.product_job((theta_x, dc.SEED)).tolist(),
               sd_fix=np.sqrt(np.einsum("pi,ij,pj->p", grad, cov, grad)).tolist(),
               truth=truth.tolist(), truth_products=dc.product_job((truth, dc.SEED)).tolist())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--replicas", type=int, default=80)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    _load(out_dir)
    started = time.time()
    full = np.load(HERE.parent / "rough_spy_is" / "rough_spy_is_seed0.npz")
    truths = correction.resample(full["theta"], full["w_post"], args.replicas, 101)
    rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, int(100 * args.scale)]))
    noise = rng.standard_normal((args.replicas, len(base.Y)))
    theta, log_q, prices, prod = CASE["theta"], CASE["log_q"], CASE["prices"], CASE["products"]
    rows = dc.FRONT
    with ProcessPoolExecutor(args.workers) as pool:
        clean = list(pool.map(base.surface_job, [(t, DATA_SEED) for t in truths]))
        ys = np.stack([p for p, _ in clean]) + args.scale * base.SIGMA * noise
        print(f"simulated {args.replicas} chains ({time.time() - started:.0f} s)", flush=True)
        chi2 = np.stack([np.sum(((prices[:, rows] - y[rows]) / base.SIGMA[rows]) ** 2, axis=1) for y in ys])
        starts = theta[np.argmin(chi2, axis=1)]
        jobs = [(r, truths[r], ys[r], args.scale, starts[r]) for r in range(args.replicas)]
        results = []
        for res in pool.map(replica_job, jobs):
            results.append(res)
            if len(results) % 10 == 0:
                print(f"  {len(results)} replicas ({time.time() - started:.0f} s)", flush=True)
    names = dp.NAMES
    z = dc.ONE_SIDED
    breaches = {m: {"long": np.zeros((args.replicas, len(names)), bool), "short": np.zeros((args.replicas, len(names)), bool)}
                for m in ("posterior", "gauss_newton", "h_fixed", "profile")}
    widths = {m: np.zeros((args.replicas, len(names), 2)) for m in breaches}
    ess = np.zeros(args.replicas)
    for r, res in enumerate(results):
        truth_p = np.asarray(res["truth_products"])
        c2 = chi2[r] / args.scale**2
        w = correction.correct(-0.5 * c2, log_q, base.TASK)
        ess[r] = w["ess"]
        mark_g, sd_g = np.asarray(res["mark_gn"]), np.asarray(res["sd_gn"])
        mark_x, sd_x = np.asarray(res["mark_fix"]), np.asarray(res["sd_fix"])
        region = c2 - min(c2.min(), res["chi2_gn"] / args.scale**2) <= z**2
        for j in range(len(names)):
            q = base.weighted_quantiles(prod[:, j], w["weights"], (0.1, 0.9))
            lows = {"posterior": q[0], "gauss_newton": mark_g[j] - z * sd_g[j], "h_fixed": mark_x[j] - z * sd_x[j],
                    "profile": prod[region, j].min() if region.any() else mark_g[j]}
            highs = {"posterior": q[1], "gauss_newton": mark_g[j] + z * sd_g[j], "h_fixed": mark_x[j] + z * sd_x[j],
                     "profile": prod[region, j].max() if region.any() else mark_g[j]}
            for m in breaches:
                breaches[m]["long"][r, j] = truth_p[j] < lows[m]
                breaches[m]["short"][r, j] = truth_p[j] > highs[m]
                widths[m][r, j] = (lows[m], highs[m])
    report = {"replicas": args.replicas, "noise_scale": args.scale, "products": list(names),
              "ess": {"median": float(np.median(ess)), "min": float(ess.min()), "share_below_50": float(np.mean(ess < 50))},
              "breach_rate": {}, "mean_interval_width_over_truth": {}, "wall_seconds": time.time() - started}
    truth_all = np.array([res["truth_products"] for res in results])
    for m in breaches:
        report["breach_rate"][m] = {side: {n: float(breaches[m][side][:, j].mean()) for j, n in enumerate(names)}
                                    for side in ("long", "short")}
        report["mean_interval_width_over_truth"][m] = {
            n: float(np.mean((widths[m][:, j, 1] - widths[m][:, j, 0]) / truth_all[:, j])) for j, n in enumerate(names)}
    np.savez_compressed(out_dir / f"validate_x{args.scale:g}.npz", truths=truths, ys=ys, ess=ess,
                        **{f"breach_{m}_{s}": breaches[m][s] for m in breaches for s in ("long", "short")},
                        **{f"interval_{m}": widths[m] for m in widths}, truth_products=truth_all)
    (out_dir / f"validate_x{args.scale:g}.json").write_text(json.dumps({"report": report, "replicas_detail": results},
                                                                       indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""Desk case: marking 1-year products from front-month smiles only, under rough Bergomi.

    python desk_case.py OUT_DIR [--proposals 1536]

The market: SPY's 21 October 2026 smile, restricted to what a thinner market quotes reliably, the front three
monthly smiles (21, 42 and 63 days, 9 strikes each: 27 quotes). The forward-variance curve to one year is the
desk's input (its variance-swap marks), as everywhere in the tutorial. Two noise levels: the design quote noise
(0.3 vol points at the money, 0.4 to 0.7 in the put wing) and 1.5 times it, a thinner market.
Products (desk_products.py): a 1-year quarterly autocallable with a 70% barrier, a 6-month-to-1-year forward-start
at-the-money call, a 1-year 80% put and a 1-year monthly cliquet capped at 2% a month and floored at zero.
Methods, each giving a mark and one-sided 90% reserves (long: mark - 10% quantile; short: 90% quantile - mark):
1. Gauss–Newton and the delta method, at the least-squares fit on the 27 quotes;
2. the profile-likelihood range: the extremes of the product over the parameters whose chi-square is within
   1.2816^2 = 1.64 of the minimum (a one-sided 90% bound), read off the priced proposals;
3. desk practice: H fixed at 0.10, least squares and Gauss–Newton on (eta, rho), delta method;
4. the posterior: importance sampling with the exact Gaussian quote-noise likelihood on the 27 quotes, flat prior
   on the box; reserves around the least-squares mark.
Reference: the full 63-quote posterior of rough_spy_is.py (seed 0), with the products priced on its proposals:
what the full market would have said.
Pricer: 32,768 paths per block, common random numbers (seed 0); products 16,384 paths, seed 0.
Proposals: a defensive mixture, 45% a Student t (5 degrees of freedom, scale 1.5x) fitted in logit space to the
Gauss–Newton draws inside the box at each noise level, 10% the prior. All 63 quotes are priced for every proposal,
so other quote sets can be reweighted from the same file.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import desk_products as dp  # noqa: E402
import rough_spy_is as base  # noqa: E402
from rough_spy_is_cases import _DAYS, inside, mixture  # noqa: E402
from sbi4finance.calibration import correction, tasks  # noqa: E402

SEED = 0
FRONT = np.isin(_DAYS, [21, 42, 63])
SCALES = (1.0, 1.5)
H_DESK = 0.10
PRODUCT_PATHS = int(__import__("os").environ.get("DESK_PRODUCT_PATHS", 16384))  # override only for smoke tests
ONE_SIDED = 1.2815515655446004  # z of a one-sided 90% bound
STEP = base.STEP


def product_job(args):
    theta, seed = args
    return dp.price(tasks.model_parameters(base.TASK, theta), base.CURVE, PRODUCT_PATHS, seed)[0]


def least_squares_fit(rows, start, free):
    """Least squares on the quotes in rows over the free coordinates (logit), the others held at start."""
    lo, hi = base.BOUNDS[free, 0], base.BOUNDS[free, 1]

    def theta_of(z):
        theta = start.copy()
        theta[free] = lo + (hi - lo) / (1 + np.exp(-z))
        return theta

    def residuals(z):
        return (base.surface(theta_of(z), SEED)[0][rows] - base.Y[rows]) / base.SIGMA[rows]

    z0 = np.log((start[free] - lo) / (hi - start[free]))
    result = least_squares(residuals, z0, method="trf", diff_step=1e-4, max_nfev=200, x_scale="jac")
    return theta_of(np.clip(result.x, -30, 30)), float(2 * result.cost), int(result.nfev)


def central(pool, job, theta, free):
    """Central differences of job's output with respect to the free coordinates, columns in free order."""
    points = []
    for i in free:
        for sign in (1, -1):
            x = theta.copy()
            x[i] += sign * STEP[i]
            points.append((x, SEED))
    values = [v[0] if isinstance(v, tuple) else v for v in pool.map(job, points)]
    return np.stack([(values[2 * k] - values[2 * k + 1]) / (2 * STEP[i]) for k, i in enumerate(free)], axis=1)


def one_sided(mark, values, weights):
    q = base.weighted_quantiles(values, weights, (0.1, 0.5, 0.9))
    return mark - q[0], q[2] - mark, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--proposals", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    full = np.load(HERE.parent / "rough_spy_is" / "rough_spy_is_seed0.npz")
    theta63 = np.asarray(full["theta_hat"], dtype=float)
    with ProcessPoolExecutor(args.workers) as pool:
        theta_f, chi2_f, n_f = least_squares_fit(FRONT, theta63, [0, 1, 2])
        print(f"[fit] front-month least squares {np.round(theta_f, 4)} chi2 {chi2_f:.2f} ({n_f} evaluations, "
              f"{time.time() - started:.0f} s)", flush=True)
        start_fix = theta_f.copy()
        start_fix[0] = H_DESK
        theta_x, chi2_x, n_x = least_squares_fit(FRONT, start_fix, [1, 2])
        print(f"[fit] H fixed at {H_DESK}: {np.round(theta_x, 4)} chi2 {chi2_x:.2f} ({n_x} evaluations, "
              f"{time.time() - started:.0f} s)", flush=True)
        jac_f = central(pool, base.surface_job, theta_f, [0, 1, 2])
        jac_x = central(pool, base.surface_job, theta_x, [1, 2])
        mark_f, mark_x = product_job((theta_f, SEED)), product_job((theta_x, SEED))
        grad_f = central(pool, product_job, theta_f, [0, 1, 2])
        grad_x = central(pool, product_job, theta_x, [1, 2])
        jn_f = jac_f[FRONT] / base.SIGMA[FRONT, None]
        jn_x = jac_x[FRONT] / base.SIGMA[FRONT, None]
        cov1_f, cov1_x = np.linalg.inv(jn_f.T @ jn_f), np.linalg.inv(jn_x.T @ jn_x)
        rng = np.random.default_rng(2027)
        draw_sets, outside = [], {}
        for k in SCALES:
            g = rng.multivariate_normal(theta_f, k**2 * cov1_f, 40000)
            ok = inside(g)
            outside[k] = {"total": float(1 - ok.mean()),
                          "below": [float(np.mean(g[:, i] <= base.BOUNDS[i, 0])) for i in range(3)],
                          "above": [float(np.mean(g[:, i] >= base.BOUNDS[i, 1])) for i in range(3)]}
            draw_sets.append(g[ok][:4000])
        theta, log_q, comp = mixture(draw_sets, args.proposals, 23)
        priced = list(pool.map(base.surface_job, [(row, SEED) for row in theta], chunksize=8))
        prices = np.stack([p for p, _ in priced])
        se = np.stack([s for _, s in priced])
        print(f"[proposals] priced {len(theta)} on 63 quotes ({time.time() - started:.0f} s)", flush=True)
        prod = np.stack(list(pool.map(product_job, [(row, SEED) for row in theta], chunksize=8)))
        full_prod = np.stack(list(pool.map(product_job, [(row, SEED) for row in full["theta"]], chunksize=8)))
        print(f"[products] priced on {len(theta)} proposals and {len(full['theta'])} full-surface draws "
              f"({time.time() - started:.0f} s)", flush=True)
    chi2_front = np.sum(((prices[:, FRONT] - base.Y[FRONT]) / base.SIGMA[FRONT]) ** 2, axis=1)
    w_full = np.asarray(full["w_post"], dtype=float)
    report = {"design": "front three monthly smiles (21, 42, 63 days) x 9 strikes = 27 quotes; xi0 to 1 year given",
              "products": list(dp.NAMES), "spec": dp.SPEC, "proposals": int(len(theta)),
              "components": np.bincount(comp, minlength=3).tolist(),
              "fit_front": {"theta": theta_f.tolist(), "chi2": chi2_f},
              "fit_h_fixed": {"theta": theta_x.tolist(), "chi2": chi2_x, "h": H_DESK},
              "max_se_over_sigma_front": float(np.max(se[:, FRONT] / base.SIGMA[FRONT])),
              "marks": {"front_fit": mark_f.tolist(), "h_fixed": mark_x.tolist()},
              "full_surface": {}, "scales": {}}
    for j, n in enumerate(dp.NAMES):
        report["full_surface"][n] = base.weighted_quantiles(full_prod[:, j], w_full, (0.05, 0.1, 0.5, 0.9, 0.95)).tolist()
    for k in SCALES:
        cov_f, cov_x = k**2 * cov1_f, k**2 * cov1_x
        log_lik = -0.5 * chi2_front / k**2
        res = correction.correct(log_lik, log_q, base.TASK)
        w = res["weights"]
        w_gauss = base.normalize(np.where(inside(theta), multivariate_normal(theta_f, cov_f).logpdf(theta), -np.inf)
                                 - log_q)
        delta2 = chi2_front / k**2 - min(chi2_front.min() / k**2, chi2_f / k**2)
        region = delta2 <= ONE_SIDED**2
        sd_f, sd_x = np.sqrt(np.diag(cov_f)), np.sqrt(np.diag(cov_x))
        params = {}
        for i, n in enumerate(base.NAMES):
            q = base.weighted_quantiles(theta[:, i], w)
            params[n] = {"least_squares": float(theta_f[i]), "gauss_newton_90": [float(theta_f[i] - base.Z90 * sd_f[i]),
                                                                               float(theta_f[i] + base.Z90 * sd_f[i])],
                         "posterior_q05_q50_q95": q.tolist(),
                         "posterior_bootstrap_sd": base.bootstrap(
                             lambda idx, wp: base.weighted_quantiles(theta[idx, i], wp), len(theta), [w]).tolist()}
        prods = {}
        for j, n in enumerate(dp.NAMES):
            m = mark_f[j]
            pl, ps, pq = one_sided(m, prod[:, j], w)
            gl, gs, gq = one_sided(m, prod[:, j], w_gauss)

            def stat(idx, wp, wg, j=j, m=m):
                a = one_sided(m, prod[idx, j], wp)
                b = one_sided(m, prod[idx, j], wg)
                return np.array([a[0], a[1], b[0], b[1]])

            err = base.bootstrap(stat, len(theta), [w, w_gauss])
            d_sd = float(np.sqrt(grad_f[j] @ cov_f @ grad_f[j]))
            x_sd = float(np.sqrt(grad_x[j] @ cov_x @ grad_x[j]))
            prof = prod[region, j]
            prods[n] = {"mark": float(m), "mark_h_fixed": float(mark_x[j]),
                        "posterior_q10_q50_q90": pq.tolist(),
                        "posterior": {"long": float(pl), "short": float(ps), "long_sd": float(err[0]),
                                      "short_sd": float(err[1])},
                        "truncated_gaussian": {"long": float(gl), "short": float(gs), "long_sd": float(err[2]),
                                               "short_sd": float(err[3])},
                        "delta": {"long": ONE_SIDED * d_sd, "short": ONE_SIDED * d_sd},
                        "profile": {"long": float(m - prof.min()), "short": float(prof.max() - m),
                                    "points": int(region.sum())},
                        "h_fixed_delta": {"long": ONE_SIDED * x_sd, "short": ONE_SIDED * x_sd}}
        report["scales"][str(k)] = {"noise_scale": k, "gaussian_mass_outside_box": outside[k],
                                    "importance": {key: res[key] for key in ("efficiency", "ess", "max_weight", "khat")},
                                    "gauss_newton_covariance": cov_f.tolist(),
                                    "h_fixed_covariance": cov_x.tolist(), "parameters": params, "products": prods}
        print(f"\n== noise x{k}: importance {json.dumps(report['scales'][str(k)]['importance'])}; Gaussian mass outside "
              f"the box {100 * outside[k]['total']:.1f}%", flush=True)
        for n, e in params.items():
            print(f"  {n:6s} posterior {np.round(e['posterior_q05_q50_q95'], 4)}  GN {np.round(e['gauss_newton_90'], 4)}",
                  flush=True)
        for n, e in prods.items():
            m = e["mark"]
            row = "  ".join(f"{key} {100 * e[key]['long'] / m:5.2f}/{100 * e[key]['short'] / m:5.2f}"
                            for key in ("posterior", "truncated_gaussian", "delta", "profile", "h_fixed_delta"))
            print(f"  {n:14s} mark {m:.5f}  long/short reserve, % of mark: {row}", flush=True)
    for n, q in report["full_surface"].items():
        print(f"  full surface {n:14s} q05 q10 q50 q90 q95 {np.round(q, 5)}", flush=True)
    report["wall_seconds"] = time.time() - started
    np.savez_compressed(out / "desk_case.npz", theta=theta, log_q=log_q, component=comp, prices=prices, se=se,
                        products=prod, full_products=full_prod, jac_front=jac_f, jac_h_fixed=jac_x,
                        grad_front=grad_f, grad_h_fixed=grad_x, mark_front=mark_f, mark_h_fixed=mark_x)
    (out / "desk_case.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

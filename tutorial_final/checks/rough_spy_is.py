"""Rough Bergomi on SPY: the posterior by importance sampling, against Gauss–Newton, on the same pricer.

    python rough_spy_is.py OUT_DIR [--seeds 0 1] [--proposals 1024 512] [--products 1 0]

The question: on SPY's 63 quotes, is rough Bergomi's posterior the Gauss–Newton Gaussian, or not?
Both are computed on one pricer per seed: Monte Carlo with 32,768 paths per block and common random numbers,
so the price surface is a smooth deterministic function of (H, eta, rho) and the comparison isolates shape.
Per seed:
1. least squares on that pricer, started at the 8,192-path fit of rough_checks.py;
2. Gauss–Newton covariance (J' S^-1 J)^-1 from central differences at the optimum (steps 0.005, 0.03, 0.01);
3. proposals: the audit's defensive mixture (Student t, 5 degrees of freedom, scale inflated 1.5x, 10% prior)
   fitted in logit space to 4,000 draws of the Gauss–Newton Gaussian;
4. each proposal priced on all 63 quotes; weights = prior x likelihood / proposal, with the Gaussian quote-noise
   likelihood (variance sigma^2; the variant sigma^2 - s^2 is stored too); efficiency, ESS and Pareto k-hat;
5. products (variance and volatility swaps, forward start, 2-year 90% put; 16,384 paths, common random numbers)
   priced on every proposal: posterior quantiles with the posterior weights, the Gauss–Newton Gaussian pushed
   through the pricer with weights N(theta; theta_hat, C) / q, and the delta method; one-sided 90% reserves
   (mark - q05 for a long position, q95 - mark for a short one), with bootstrap errors over proposals.
A second seed repeats steps 1 to 4: the change between seeds is the pricer's Monte Carlo error on the result.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration import classical, correction, pricing, products, tasks  # noqa: E402

TASK = json.loads((ROOT / "configs/tasks/calibration_rough_pilot.json").read_text())["task"]
SPY = json.loads((ROOT / "tutorial_final/results/calibration/spy/surface.json").read_text())
START = np.asarray(json.loads((HERE / "rough_checks.json").read_text())["best"]["theta"], dtype=float)
Y = np.asarray(SPY["prices_forward_units"], dtype=float)
CURVE = tasks.curve(TASK, np.asarray(SPY["forward_variance_levels"], dtype=float))
CURVE_2Y = products.extend_curve(CURVE, 2.0)
SIGMA = tasks.sigma(TASK)
CONTRACTS = TASK["context"]["contracts"]
BLOCKS = TASK["numerics"]["blocks"]
NAMES = TASK["parameter_names"]
BOUNDS = correction.bounds_of(TASK)
SPEC = products.default_specification()
PATHS = int(os.environ.get("ROUGH_IS_PATHS", 32768))  # override only for smoke tests
PRODUCT_PATHS = int(os.environ.get("ROUGH_IS_PRODUCT_PATHS", 16384))
STEP = np.array([0.005, 0.03, 0.01])
Z90 = 1.6448536269514722
LEVELS = (0.05, 0.5, 0.95)


def surface(theta, seed):
    p, se, _, _ = pricing.rough_surface(tasks.model_parameters(TASK, theta), CURVE, CONTRACTS, BLOCKS, [PATHS],
                                        SIGMA, np.inf, seed)
    return p, se


def surface_job(args):
    return surface(*args)


def product_job(args):
    theta, seed = args
    return products.price_draw("rough", tasks.model_parameters(TASK, theta), CURVE_2Y, SPEC, PRODUCT_PATHS, seed)[0]


def central_jacobian(pool, job, theta, seed):
    points = []
    for i in range(len(theta)):
        for sign in (1, -1):
            x = theta.copy()
            x[i] += sign * STEP[i]
            points.append((x, seed))
    values = [v[0] if isinstance(v, tuple) else v for v in pool.map(job, points)]
    return np.stack([(values[2 * i] - values[2 * i + 1]) / (2 * STEP[i]) for i in range(len(theta))], axis=1)


def weighted_quantiles(values, weights, levels=LEVELS):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    return np.array([values[order[min(np.searchsorted(cdf, p), len(cdf) - 1)]] for p in levels])


def normalize(log_w):
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    return np.exp(log_w - logsumexp(log_w))


def bootstrap(stat, n, weights_list, reps=400, seed=5):
    """sd of stat(index, *weights) over resamples of the proposals (with replacement)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        out.append(stat(idx, *[w[idx] for w in weights_list]))
    return np.std(np.array(out, dtype=float), axis=0)


def run_seed(pool, seed, n_proposals, with_products, out_dir):
    started = time.time()
    fit = classical.fit(TASK, Y, lambda th: surface(th, seed)[0], [START])[0]
    theta_hat = np.asarray(fit["theta"], dtype=float)
    print(f"[seed {seed}] fit {np.round(theta_hat, 4)} chi2 {fit['chi2']:.2f} ({fit['evaluations']} evaluations, "
          f"{time.time() - started:.0f} s)", flush=True)
    jac = central_jacobian(pool, surface_job, theta_hat, seed)
    jn = jac / SIGMA[:, None]
    cov = np.linalg.inv(jn.T @ jn)
    sd = np.sqrt(np.diag(cov))
    rng = np.random.default_rng(1000 + seed)
    gn = rng.multivariate_normal(theta_hat, cov, 4000)
    inside = np.all((gn > BOUNDS[:, 0]) & (gn < BOUNDS[:, 1]), axis=1)
    theta, log_q, from_prior = correction.propose(gn[inside], TASK, n_proposals, 11 + seed, inflation=1.5)
    priced = list(pool.map(surface_job, [(row, seed) for row in theta], chunksize=8))
    prices = np.stack([p for p, _ in priced])
    se = np.stack([s for _, s in priced])
    print(f"[seed {seed}] priced {len(theta)} proposals ({time.time() - started:.0f} s)", flush=True)
    chi2 = np.sum(((prices - Y) / SIGMA) ** 2, axis=1)
    log_lik = correction.gaussian_loglik(TASK, Y, prices)
    variance_u = SIGMA**2 - se**2
    ok_u = np.all(se < 0.9 * SIGMA, axis=1)
    log_lik_u = np.where(ok_u, -0.5 * np.sum((prices - Y) ** 2 / np.where(variance_u > 0, variance_u, 1.0)
                                              + np.log(2 * np.pi * np.where(variance_u > 0, variance_u, 1.0)),
                                              axis=1), -np.inf)
    res = correction.correct(log_lik, log_q, TASK)
    res_u = correction.correct(log_lik_u, log_q, TASK)
    w_post, w_u = res["weights"], res_u["weights"]
    log_gauss = multivariate_normal(theta_hat, cov).logpdf(theta)
    w_gauss = normalize(np.where(np.all((theta > BOUNDS[:, 0]) & (theta < BOUNDS[:, 1]), axis=1), log_gauss, -np.inf)
                        - log_q)
    gauss_check = {"efficiency": float(1 / np.sum(w_gauss**2) / len(w_gauss)),
                   "khat": correction.pareto_khat(log_gauss - log_q)}

    def param_stat(idx, wp):
        return np.concatenate([weighted_quantiles(theta[idx, i], wp) for i in range(3)])

    boot = bootstrap(param_stat, len(theta), [w_post]).reshape(3, 3)
    params = {}
    for i, n in enumerate(NAMES):
        q = weighted_quantiles(theta[:, i], w_post)
        qu = weighted_quantiles(theta[:, i], w_u)
        qg = weighted_quantiles(theta[:, i], w_gauss)
        params[n] = {"least_squares": float(theta_hat[i]), "gauss_newton_sd": float(sd[i]),
                     "gauss_newton_90": [float(theta_hat[i] - Z90 * sd[i]), float(theta_hat[i] + Z90 * sd[i])],
                     "gaussian_reweighted_q05_q50_q95": qg.tolist(),
                     "posterior_q05_q50_q95": q.tolist(), "posterior_bootstrap_sd": boot[i].tolist(),
                     "posterior_unbiased_variant_q05_q50_q95": qu.tolist(),
                     "posterior_mean": float(np.sum(w_post * theta[:, i])),
                     "posterior_sd": float(np.sqrt(np.sum(w_post * (theta[:, i] - np.sum(w_post * theta[:, i])) ** 2))),
                     "width_ratio_gn_over_posterior": float(2 * Z90 * sd[i] / (q[2] - q[0])),
                     "lower_ratio_gn_over_posterior": float(Z90 * sd[i] / (theta_hat[i] - q[0])),
                     "upper_ratio_gn_over_posterior": float(Z90 * sd[i] / (q[2] - theta_hat[i]))}
    report = {"seed": seed, "paths": PATHS, "fit": {"theta": theta_hat.tolist(), "chi2": fit["chi2"],
                                                     "evaluations": fit["evaluations"]},
              "gauss_newton_covariance": cov.tolist(), "gauss_newton_correlation":
                  (cov / np.outer(sd, sd)).tolist(), "gaussian_mass_outside_box": float(1 - inside.mean()),
              "proposals": int(len(theta)), "from_prior": int(from_prior.sum()),
              "max_se_over_sigma": {"median": float(np.median(np.max(se / SIGMA, axis=1))),
                                    "max": float(np.max(se / SIGMA))},
              "unbiased_variant_excluded": int((~ok_u).sum()),
              "importance": {k: res[k] for k in ("efficiency", "ess", "max_weight", "khat")},
              "importance_unbiased_variant": {k: res_u[k] for k in ("efficiency", "ess", "max_weight", "khat")},
              "proposal_vs_gaussian": gauss_check, "parameters": params}
    arrays = {"theta": theta, "log_q": log_q, "from_prior": from_prior, "prices": prices, "se": se, "chi2": chi2,
              "log_lik": log_lik, "log_lik_unbiased": log_lik_u, "w_post": w_post, "w_gauss": w_gauss,
              "theta_hat": theta_hat, "cov": cov, "jacobian": jac}
    if with_products:
        mark = product_job((theta_hat, seed))
        grad = central_jacobian(pool, product_job, theta_hat, seed)  # [products, 3]
        delta_sd = np.sqrt(np.einsum("pi,ij,pj->p", grad, cov, grad))
        prod = np.stack(list(pool.map(product_job, [(row, seed) for row in theta], chunksize=8)))
        print(f"[seed {seed}] priced products ({time.time() - started:.0f} s)", flush=True)

        def reserves(idx, wp, wg):
            out = []
            for j in range(prod.shape[1]):
                pq, gq = weighted_quantiles(prod[idx, j], wp), weighted_quantiles(prod[idx, j], wg)
                out += [mark[j] - pq[0], pq[2] - mark[j], mark[j] - gq[0], gq[2] - mark[j]]
            return np.array(out)

        point = reserves(np.arange(len(theta)), w_post, w_gauss).reshape(-1, 4)
        err = bootstrap(reserves, len(theta), [w_post, w_gauss]).reshape(-1, 4)
        prods = {}
        for j, n in enumerate(products.PRODUCTS):
            pl, ps, gl, gs = point[j]
            prods[n] = {"mark": float(mark[j]),
                        "posterior_q05_q50_q95": weighted_quantiles(prod[:, j], w_post).tolist(),
                        "gaussian_pushforward_q05_q50_q95": weighted_quantiles(prod[:, j], w_gauss).tolist(),
                        "delta_method_90": [float(mark[j] - Z90 * delta_sd[j]), float(mark[j] + Z90 * delta_sd[j])],
                        "reserve_long": {"posterior": float(pl), "gaussian_pushforward": float(gl),
                                         "delta": float(Z90 * delta_sd[j]), "posterior_bootstrap_sd": float(err[j, 0]),
                                         "gaussian_bootstrap_sd": float(err[j, 2])},
                        "reserve_short": {"posterior": float(ps), "gaussian_pushforward": float(gs),
                                          "delta": float(Z90 * delta_sd[j]), "posterior_bootstrap_sd": float(err[j, 1]),
                                          "gaussian_bootstrap_sd": float(err[j, 3])}}
        report["products"] = prods
        report["product_paths"] = PRODUCT_PATHS
        arrays.update(products=prod, product_mark=mark, product_gradient=grad)
    report["wall_seconds"] = time.time() - started
    np.savez_compressed(out_dir / f"rough_spy_is_seed{seed}.npz", **arrays)
    (out_dir / f"rough_spy_is_seed{seed}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("importance", "importance_unbiased_variant", "proposal_vs_gaussian",
                                             "max_se_over_sigma", "wall_seconds")}, indent=1), flush=True)
    for n, e in params.items():
        print(f"  {n:6s} LS {e['least_squares']:.4f}  posterior {np.round(e['posterior_q05_q50_q95'], 4)}  GN "
              f"{np.round(e['gauss_newton_90'], 4)}  width GN/post {e['width_ratio_gn_over_posterior']:.2f}  "
              f"lower {e['lower_ratio_gn_over_posterior']:.2f}  upper {e['upper_ratio_gn_over_posterior']:.2f}",
              flush=True)
    for n, e in report.get("products", {}).items():
        print(f"  {n:20s} mark {e['mark']:.5f}  long: post {e['reserve_long']['posterior']:.5f} gauss "
              f"{e['reserve_long']['gaussian_pushforward']:.5f} delta {e['reserve_long']['delta']:.5f} | short: post "
              f"{e['reserve_short']['posterior']:.5f} gauss {e['reserve_short']['gaussian_pushforward']:.5f}",
              flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--proposals", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--products", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for seed, n, p in zip(args.seeds, args.proposals, args.products):
            run_seed(pool, seed, n, bool(p), out_dir)


if __name__ == "__main__":
    main()

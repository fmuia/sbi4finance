"""Rough Bergomi on SPY when the quotes say less: the posterior by importance sampling against Gauss–Newton.

    python rough_spy_is_cases.py OUT_DIR [--proposals 1024]

Two cases, on the pricer of rough_spy_is.py (32,768 paths per block, common random numbers, seed 0):
  A. sparse quotes: SPY's 15 quotes at 21, 63 and 126 days and standardized moneyness -1, -0.5, 0, 0.5, 1
     (the sparse set of rough_checks.py), with the design quote noise;
  B. wide spreads: all 63 quotes with three times the design quote noise.
Gauss–Newton per case: A refits least squares on its 15 quotes and takes the central-difference Jacobian there;
B has the full grid's optimum and nine times its covariance (the noise scales every residual alike).
Proposals: one defensive mixture for both cases, 45% a Student t (5 degrees of freedom, scale 1.5x) fitted in
logit space to each case's Gauss–Newton draws inside the prior box, 10% the prior. Every proposal is priced once
on the 63 quotes and once on the products; each case reweights the same proposals with its own likelihood.
Per case: efficiency, ESS and Pareto k-hat; the posterior's quantiles against the Gauss–Newton interval and the
Gaussian's mass outside the prior box; one-sided 90% reserves from the posterior, from the Gauss–Newton normal
truncated to the box and pushed through the pricer, and from the delta method (which ignores the box).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rough_spy_is as base  # noqa: E402
from sbi4finance.calibration import correction, products, surface  # noqa: E402

SEED = 0
NOISE_SCALE = 3.0
_LOGK, _T, _ = surface.contract_arrays(base.CONTRACTS)
_DAYS = np.rint(np.asarray(_T) * surface.TRADING_DAYS).astype(int)
_DELTA = np.round(2 * surface.standardized_moneyness(_LOGK, _T)) / 2
SPARSE = np.isin(_DAYS, [21, 63, 126]) & np.isin(_DELTA, [-1, -0.5, 0, 0.5, 1])
FULL = np.ones(len(base.Y), bool)
CASES = {"A": {"label": "15 quotes (3 maturities x 5 strikes), design noise", "rows": SPARSE, "scale": 1.0},
         "B": {"label": "63 quotes, 3x the design noise", "rows": FULL, "scale": NOISE_SCALE}}


def refit(rows, start):
    """Least squares on a subset of quotes, in logit coordinates as classical.fit does."""
    def residuals(z):
        return (base.surface(correction.to_bounded(z, base.BOUNDS), SEED)[0][rows] - base.Y[rows]) / base.SIGMA[rows]

    result = least_squares(residuals, correction.to_unbounded(start, base.BOUNDS), method="trf", diff_step=1e-4,
                           max_nfev=200, x_scale="jac")
    return correction.to_bounded(np.clip(result.x, -30, 30), base.BOUNDS), float(2 * result.cost), int(result.nfev)


def mixture(draw_sets, n, seed, prior_fraction=0.1, df=5, inflation=1.5):
    """IID proposals from sum_j (1 - f)/k t_j + f prior, and their log densities in natural units."""
    ts = [correction.fit_proposal(d, base.TASK, df=df, inflation=inflation) for d in draw_sets]
    k = len(ts)
    rng = np.random.default_rng(np.random.SeedSequence([7, seed]))
    comp = rng.choice(k + 1, size=n, p=[(1 - prior_fraction) / k] * k + [prior_fraction])
    z = np.empty((n, len(base.BOUNDS)))
    for j, t in enumerate(ts):
        m = comp == j
        if m.any():
            z[m] = np.atleast_2d(t.rvs(size=int(m.sum()), random_state=rng))
    m = comp == k
    z[m] = correction.to_unbounded(rng.uniform(base.BOUNDS[:, 0], base.BOUNDS[:, 1], (int(m.sum()), 3)), base.BOUNDS)
    z = np.clip(z, -30, 30)
    theta = correction.to_bounded(z, base.BOUNDS)
    log_jac = correction.log_jacobian(z, base.BOUNDS)
    parts = [np.log((1 - prior_fraction) / k) + t.logpdf(z) - log_jac for t in ts]
    parts.append(np.full(n, np.log(prior_fraction) + correction.log_prior(base.TASK)))
    return theta, logsumexp(np.stack(parts), axis=0), comp


def inside(theta):
    return np.all((theta > base.BOUNDS[:, 0]) & (theta < base.BOUNDS[:, 1]), axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--proposals", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    full = json.loads((out_dir / "rough_spy_is_seed0.json").read_text())
    full_arrays = np.load(out_dir / "rough_spy_is_seed0.npz")
    theta_full = np.asarray(full["fit"]["theta"], dtype=float)
    cov_full = np.asarray(full["gauss_newton_covariance"], dtype=float)
    with ProcessPoolExecutor(args.workers) as pool:
        fits = {}
        theta_a, chi2_a, nfev = refit(SPARSE, theta_full)
        print(f"[A] refit {np.round(theta_a, 4)} chi2 {chi2_a:.2f} on {SPARSE.sum()} quotes ({nfev} evaluations, "
              f"{time.time() - started:.0f} s)", flush=True)
        jac_a = base.central_jacobian(pool, base.surface_job, theta_a, SEED)
        jn = jac_a[SPARSE] / base.SIGMA[SPARSE, None]
        fits["A"] = {"theta": theta_a, "cov": np.linalg.inv(jn.T @ jn), "chi2": chi2_a}
        fits["B"] = {"theta": theta_full, "cov": NOISE_SCALE**2 * cov_full,
                     "chi2": float(full["fit"]["chi2"]) / NOISE_SCALE**2}
        rng = np.random.default_rng(2026)
        draw_sets, outside = [], {}
        for key in ("A", "B"):
            g = rng.multivariate_normal(fits[key]["theta"], fits[key]["cov"], 20000)
            ok = inside(g)
            outside[key] = {"total": float(1 - ok.mean()),
                            "per_parameter_below": [float(np.mean(g[:, i] <= base.BOUNDS[i, 0])) for i in range(3)],
                            "per_parameter_above": [float(np.mean(g[:, i] >= base.BOUNDS[i, 1])) for i in range(3)]}
            draw_sets.append(g[ok][:4000])
        theta, log_q, comp = mixture(draw_sets, args.proposals, 11)
        priced = list(pool.map(base.surface_job, [(row, SEED) for row in theta], chunksize=8))
        prices = np.stack([p for p, _ in priced])
        se = np.stack([s for _, s in priced])
        print(f"priced {len(theta)} proposals ({time.time() - started:.0f} s)", flush=True)
        prod = np.stack(list(pool.map(base.product_job, [(row, SEED) for row in theta], chunksize=8)))
        marks = {"A": base.product_job((theta_a, SEED)), "B": np.asarray(full_arrays["product_mark"])}
        grads = {"A": base.central_jacobian(pool, base.product_job, theta_a, SEED),
                 "B": np.asarray(full_arrays["product_gradient"])}
        print(f"priced products ({time.time() - started:.0f} s)", flush=True)
    report = {"proposals": int(len(theta)), "components": np.bincount(comp, minlength=3).tolist(),
              "max_se_over_sigma": float(np.max(se / base.SIGMA)), "cases": {}}
    arrays = {"theta": theta, "log_q": log_q, "component": comp, "prices": prices, "se": se, "products": prod}
    for key, case in CASES.items():
        rows, scale = case["rows"], case["scale"]
        mean, cov = fits[key]["theta"], fits[key]["cov"]
        sd = np.sqrt(np.diag(cov))
        log_lik = -0.5 * np.sum(((prices[:, rows] - base.Y[rows]) / (scale * base.SIGMA[rows])) ** 2, axis=1)
        res = correction.correct(log_lik, log_q, base.TASK)
        w_post = res["weights"]
        w_gauss = base.normalize(np.where(inside(theta), multivariate_normal(mean, cov).logpdf(theta), -np.inf)
                                 - log_q)

        def param_stat(idx, wp):
            return np.concatenate([base.weighted_quantiles(theta[idx, i], wp) for i in range(3)])

        boot = base.bootstrap(param_stat, len(theta), [w_post]).reshape(3, 3)
        params = {}
        for i, n in enumerate(base.NAMES):
            q = base.weighted_quantiles(theta[:, i], w_post)
            lo, hi = mean[i] - base.Z90 * sd[i], mean[i] + base.Z90 * sd[i]
            params[n] = {"least_squares": float(mean[i]), "gauss_newton_sd": float(sd[i]),
                         "gauss_newton_90": [float(lo), float(hi)],
                         "gauss_newton_90_leaves_box": bool(lo < base.BOUNDS[i, 0] or hi > base.BOUNDS[i, 1]),
                         "posterior_q05_q50_q95": q.tolist(), "posterior_bootstrap_sd": boot[i].tolist(),
                         "truncated_gaussian_q05_q50_q95": base.weighted_quantiles(theta[:, i], w_gauss).tolist(),
                         "width_ratio_gn_over_posterior": float((hi - lo) / (q[2] - q[0])),
                         "lower_ratio_gn_over_posterior": float((mean[i] - lo) / (mean[i] - q[0])),
                         "upper_ratio_gn_over_posterior": float((hi - mean[i]) / (q[2] - mean[i]))}
        mark, grad = marks[key], grads[key]
        delta_sd = np.sqrt(np.einsum("pi,ij,pj->p", grad, cov, grad))

        def reserves(idx, wp, wg):
            out = []
            for j in range(prod.shape[1]):
                pq, gq = base.weighted_quantiles(prod[idx, j], wp), base.weighted_quantiles(prod[idx, j], wg)
                out += [mark[j] - pq[0], pq[2] - mark[j], mark[j] - gq[0], gq[2] - mark[j]]
            return np.array(out)

        point = reserves(np.arange(len(theta)), w_post, w_gauss).reshape(-1, 4)
        err = base.bootstrap(reserves, len(theta), [w_post, w_gauss]).reshape(-1, 4)
        prods = {}
        for j, n in enumerate(products.PRODUCTS):
            pl, ps, gl, gs = point[j]
            prods[n] = {"mark": float(mark[j]),
                        "reserve_long": {"posterior": float(pl), "truncated_gaussian": float(gl),
                                         "delta": float(base.Z90 * delta_sd[j]), "posterior_bootstrap_sd": float(err[j, 0]),
                                         "gaussian_bootstrap_sd": float(err[j, 2])},
                        "reserve_short": {"posterior": float(ps), "truncated_gaussian": float(gs),
                                          "delta": float(base.Z90 * delta_sd[j]), "posterior_bootstrap_sd": float(err[j, 1]),
                                          "gaussian_bootstrap_sd": float(err[j, 3])}}
        report["cases"][key] = {"label": case["label"], "quotes": int(rows.sum()), "noise_scale": scale,
                                "fit": {"theta": mean.tolist(), "chi2": fits[key]["chi2"]},
                                "gauss_newton_covariance": cov.tolist(), "gaussian_mass_outside_box": outside[key],
                                "importance": {k: res[k] for k in ("efficiency", "ess", "max_weight", "khat")},
                                "parameters": params, "products": prods}
        arrays.update({f"w_post_{key}": w_post, f"w_gauss_{key}": w_gauss, f"log_lik_{key}": log_lik,
                       f"mark_{key}": mark, f"gradient_{key}": grad})
        print(f"\n== case {key}: {case['label']}", flush=True)
        print(json.dumps({"importance": report["cases"][key]["importance"], "outside": outside[key]}), flush=True)
        for n, e in params.items():
            print(f"  {n:6s} LS {e['least_squares']:.4f}  posterior {np.round(e['posterior_q05_q50_q95'], 4)}  GN "
                  f"{np.round(e['gauss_newton_90'], 4)}  width GN/post {e['width_ratio_gn_over_posterior']:.2f}  "
                  f"lower {e['lower_ratio_gn_over_posterior']:.2f}  upper {e['upper_ratio_gn_over_posterior']:.2f}",
                  flush=True)
        for n, e in prods.items():
            if n == "variance_swap":
                continue
            m = e["mark"]
            print(f"  {n:20s} long: post {100 * e['reserve_long']['posterior'] / m:.2f}% trunc-gauss "
                  f"{100 * e['reserve_long']['truncated_gaussian'] / m:.2f}% delta {100 * e['reserve_long']['delta'] / m:.2f}%"
                  f" | short: post {100 * e['reserve_short']['posterior'] / m:.2f}% trunc-gauss "
                  f"{100 * e['reserve_short']['truncated_gaussian'] / m:.2f}%", flush=True)
    report["wall_seconds"] = time.time() - started
    np.savez_compressed(out_dir / "rough_spy_is_cases.npz", **arrays)
    (out_dir / "rough_spy_is_cases.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""Desk case: the analysis of a set of priced proposals, shared by desk_case_round2.py and the notebook.

analyse() takes proposals priced on the 63 quotes and on the products, with their proposal log densities, and
returns, per noise scale, the posterior's parameter quantiles and the products' marks and one-sided 90% reserves
under the five methods of desk_case.py. fits() reads the least-squares fits, Gauss–Newton covariances and product
gradients that desk_case.py stored. mixture_logpdf() evaluates a defensive mixture's density at any points.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import desk_products as dp  # noqa: E402
import rough_spy_is as base  # noqa: E402
from desk_case import FRONT, ONE_SIDED, SCALES  # noqa: E402
from rough_spy_is_cases import inside  # noqa: E402
from sbi4finance.calibration import correction  # noqa: E402


def fits(out_dir):
    """The round-1 fits: least squares (all three and H fixed), unit-noise covariances, marks and gradients."""
    r = json.loads((Path(out_dir) / "desk_case.json").read_text())
    a = np.load(Path(out_dir) / "desk_case.npz")
    return {"theta_f": np.asarray(r["fit_front"]["theta"]), "chi2_f": r["fit_front"]["chi2"],
            "theta_x": np.asarray(r["fit_h_fixed"]["theta"]), "chi2_x": r["fit_h_fixed"]["chi2"],
            "cov1_f": np.asarray(r["scales"]["1.0"]["gauss_newton_covariance"]),
            "cov1_x": np.asarray(r["scales"]["1.0"]["h_fixed_covariance"]),
            "mark_f": np.asarray(a["mark_front"]), "mark_x": np.asarray(a["mark_h_fixed"]),
            "grad_f": np.asarray(a["grad_front"]), "grad_x": np.asarray(a["grad_h_fixed"]), "h_fixed": r["fit_h_fixed"]["h"]}


def mixture_logpdf(ts, fractions, theta):
    """log density, in natural units, of sum_j fractions[j] t_j (in logit space) + fractions[-1] prior, at theta."""
    z = correction.to_unbounded(np.clip(theta, base.BOUNDS[:, 0] + 1e-12, base.BOUNDS[:, 1] - 1e-12), base.BOUNDS)
    log_jac = correction.log_jacobian(z, base.BOUNDS)
    parts = [np.log(f) + t.logpdf(z) - log_jac for f, t in zip(fractions[:-1], ts)]
    parts.append(np.full(len(theta), np.log(fractions[-1]) + correction.log_prior(base.TASK)))
    return logsumexp(np.stack(parts), axis=0)


def round1_components(fit):
    """The t components of desk_case.py's proposal, rebuilt from its stored fit (same draws, same fit)."""
    rng = np.random.default_rng(2027)
    ts = []
    for k in SCALES:
        g = rng.multivariate_normal(fit["theta_f"], k**2 * fit["cov1_f"], 40000)
        ts.append(correction.fit_proposal(g[inside(g)][:4000], base.TASK, df=5, inflation=1.5))
    return ts, [0.45, 0.45, 0.10]


def weighted_t(theta, weights, inflation, df=5):
    """A Student t in logit space with the weighted mean and covariance of the points."""
    from scipy.stats import multivariate_t
    z = correction.to_unbounded(theta, base.BOUNDS)
    w = weights / weights.sum()
    mean = w @ z
    cov = (z - mean).T @ ((z - mean) * w[:, None])
    return multivariate_t(loc=mean, shape=cov * inflation**2 + 1e-9 * np.eye(3), df=df)


def posterior_weights(prices, log_q, scale, rows=FRONT):
    chi2 = np.sum(((prices[:, rows] - base.Y[rows]) / base.SIGMA[rows]) ** 2, axis=1)
    return correction.correct(-0.5 * chi2 / scale**2, log_q, base.TASK), chi2


def analyse(theta, log_q, prices, prod, full_prod, w_full, fit):
    """Per noise scale: importance diagnostics, parameter quantiles, product reserves under every method."""
    report = {"proposals": int(len(theta)), "fit_front": {"theta": fit["theta_f"].tolist(), "chi2": fit["chi2_f"]},
              "fit_h_fixed": {"theta": fit["theta_x"].tolist(), "chi2": fit["chi2_x"], "h": fit["h_fixed"]},
              "full_surface": {n: base.weighted_quantiles(full_prod[:, j], w_full, (0.05, 0.1, 0.5, 0.9, 0.95)).tolist()
                               for j, n in enumerate(dp.NAMES)}, "scales": {}}
    for k in SCALES:
        res, chi2 = posterior_weights(prices, log_q, k)
        w = res["weights"]
        cov_f, cov_x = k**2 * fit["cov1_f"], k**2 * fit["cov1_x"]
        w_gauss = base.normalize(np.where(inside(theta), multivariate_normal(fit["theta_f"], cov_f).logpdf(theta),
                                          -np.inf) - log_q)
        region = chi2 / k**2 - min(chi2.min(), fit["chi2_f"]) / k**2 <= ONE_SIDED**2
        g = np.random.default_rng(5).multivariate_normal(fit["theta_f"], cov_f, 200000)
        sd_f = np.sqrt(np.diag(cov_f))
        params = {}
        for i, n in enumerate(base.NAMES):
            q = base.weighted_quantiles(theta[:, i], w)
            params[n] = {"least_squares": float(fit["theta_f"][i]),
                         "gauss_newton_90": [float(fit["theta_f"][i] - base.Z90 * sd_f[i]),
                                             float(fit["theta_f"][i] + base.Z90 * sd_f[i])],
                         "posterior_q05_q50_q95": q.tolist(),
                         "posterior_bootstrap_sd": base.bootstrap(
                             lambda idx, wp, i=i: base.weighted_quantiles(theta[idx, i], wp), len(theta), [w]).tolist()}
        prods = {}
        for j, n in enumerate(dp.NAMES):
            m = fit["mark_f"][j]

            def sides(idx, wp, wg, j=j, m=m):
                a = base.weighted_quantiles(prod[idx, j], wp, (0.1, 0.9))
                b = base.weighted_quantiles(prod[idx, j], wg, (0.1, 0.9))
                return np.array([m - a[0], a[1] - m, m - b[0], b[1] - m])

            point = sides(np.arange(len(theta)), w, w_gauss)
            err = base.bootstrap(sides, len(theta), [w, w_gauss])
            d_sd = float(np.sqrt(fit["grad_f"][j] @ cov_f @ fit["grad_f"][j]))
            x_sd = float(np.sqrt(fit["grad_x"][j] @ cov_x @ fit["grad_x"][j]))
            prof = prod[region, j]
            prods[n] = {"mark": float(m), "mark_h_fixed": float(fit["mark_x"][j]),
                        "posterior_q10_q50_q90": base.weighted_quantiles(prod[:, j], w, (0.1, 0.5, 0.9)).tolist(),
                        "posterior": {"long": float(point[0]), "short": float(point[1]), "long_sd": float(err[0]),
                                      "short_sd": float(err[1])},
                        "truncated_gaussian": {"long": float(point[2]), "short": float(point[3]),
                                               "long_sd": float(err[2]), "short_sd": float(err[3])},
                        "delta": {"long": ONE_SIDED * d_sd, "short": ONE_SIDED * d_sd},
                        "profile": {"long": float(m - prof.min()), "short": float(prof.max() - m),
                                    "points": int(region.sum())},
                        "h_fixed_delta": {"long": ONE_SIDED * x_sd, "short": ONE_SIDED * x_sd}}
        report["scales"][str(k)] = {
            "noise_scale": k, "importance": {key: res[key] for key in ("efficiency", "ess", "max_weight", "khat")},
            "gaussian_mass_outside_box": float(1 - inside(g).mean()),
            "gaussian_below_bounds": [float(np.mean(g[:, i] <= base.BOUNDS[i, 0])) for i in range(3)],
            "gauss_newton_covariance": cov_f.tolist(), "h_fixed_covariance": cov_x.tolist(),
            "parameters": params, "products": prods}
    return report


def summary(report):
    lines = []
    for key, s in report["scales"].items():
        lines.append(f"== noise x{key}: importance {json.dumps(s['importance'])}; Gaussian mass outside the box "
                     f"{100 * s['gaussian_mass_outside_box']:.1f}%")
        for n, e in s["parameters"].items():
            lines.append(f"  {n:6s} posterior {np.round(e['posterior_q05_q50_q95'], 4)} (+-{np.round(e['posterior_bootstrap_sd'], 4)})"
                         f"  GN {np.round(e['gauss_newton_90'], 4)}")
        for n, e in s["products"].items():
            m = e["mark"]
            row = "  ".join(f"{k2} {100 * e[k2]['long'] / m:5.2f}/{100 * e[k2]['short'] / m:5.2f}"
                            for k2 in ("posterior", "truncated_gaussian", "delta", "profile", "h_fixed_delta"))
            lines.append(f"  {n:14s} mark {m:.5f}  long/short % of mark: {row}")
    for n, q in report["full_surface"].items():
        lines.append(f"  full surface {n:14s} q05 q10 q50 q90 q95 {np.round(q, 5)}")
    return "\n".join(lines)

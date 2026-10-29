"""Quick check: where does the exact Heston posterior on SPY differ from the Gauss–Newton route?

    python heston_checks.py OUT.json

Cases: the full grid with kappa fixed at 2 (slot 4) and free in [0.25, 8]; maturities up to 42 days; a sparse
3 x 5 grid with kappa fixed and free. For each case: multi-start least squares with its Gauss–Newton interval;
the exact posterior (random-walk Metropolis on the Fourier likelihood, uniform prior box; slot 4's 12,000 draws
for the first case); the 90% ranges of the four products under the posterior (300 draws, 8,192 paths, common
random numbers) and by the delta method (Gauss–Newton covariance, central-difference gradients at 32,768 paths).
Also a profile in kappa on the full grid.
"""

import functools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "src"))
sys.path.insert(0, str(ROOT / "lab" / "studies"))
sys.path.insert(0, str(ROOT / "lab" / "checks"))
from sbi4finance.calibration import classical, correction, products, tasks  # noqa: E402
import heston_check  # noqa: E402
from heston_spy_exact import rhat_ess  # noqa: E402

SPY = json.loads((ROOT / "tutorial/assets/results/calibration/spy/surface.json").read_text())
Y_FULL = np.asarray(SPY["prices_forward_units"], dtype=float)
FULL_CONTRACTS = tasks.heston_task()["context"]["contracts"]
EXACT_DRAWS = ROOT / "tutorial/assets/data/heston-spy-exact.npy"
FULL = {"days": (5, 10, 21, 42, 63, 126, 252), "deltas": (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)}
SHORT = {"days": (5, 10, 21, 42), "deltas": FULL["deltas"]}
SPARSE = {"days": (21, 63, 126), "deltas": (-1.0, -0.5, 0.0, 0.5, 1.0)}
KAPPA_BOUNDS = [0.25, 8.0]
SPEC = products.default_specification()
CASES = [("full grid, kappa fixed", FULL, False), ("full grid, kappa free", FULL, True),
         ("up to 42 days, kappa fixed", SHORT, False), ("3 x 5 grid, kappa fixed", SPARSE, False),
         ("3 x 5 grid, kappa free", SPARSE, True)]
SLOT4_OPTIMUM = [0.0182, 0.0657, 1.14, -0.632]


def make_task(grid, kappa_free, kappa=2.0):
    task = tasks.heston_task(days=grid["days"], deltas=grid["deltas"], kappa=kappa)
    if kappa_free:
        task["parameter_names"] = task["parameter_names"] + ["kappa"]
        task["prior_bounds"] = task["prior_bounds"] + [KAPPA_BOUNDS]
        task["fixed_parameters"] = {k: v for k, v in task["fixed_parameters"].items() if k != "kappa"}
    return task


def observation(task):
    index = [next(i for i, f in enumerate(FULL_CONTRACTS)
                  if np.isclose(f["strike"], c["strike"]) and np.isclose(f["maturity"], c["maturity"]))
             for c in task["context"]["contracts"]]
    return Y_FULL[index]


def loglik(task, y, theta):
    try:
        value = float(heston_check.exact_loglik(task, y, theta)[0])
    except Exception:
        return -np.inf
    return value if np.isfinite(value) else -np.inf


def log_target(task, y, z):
    bounds = correction.bounds_of(task)
    return loglik(task, y, correction.to_bounded(z, bounds)) + float(correction.log_jacobian(z, bounds))


def chain_job(args):
    task, y, start, scale, steps, seed = args
    return correction.random_walk_chain(functools.partial(log_target, task, y), start, scale, steps, seed)


def fit_job(args):
    task, y, start = args
    return classical.fit(task, y, lambda th: heston_check.exact_prices(task, th), [start])[0]


def product_job(args):
    task, row, paths, seed = args
    return products.price_draw("heston", tasks.model_parameters(task, row), None, SPEC, paths, seed)[0]


def least_squares(task, y, pool, extra_starts=()):
    starts = list(classical.latin_starts(task, 12, 3)) + [np.asarray(s, float) for s in extra_starts]
    fits = list(pool.map(fit_job, [(task, y, s) for s in starts]))
    return fits, min(fits, key=lambda f: f["chi2"])


def posterior(task, y, best, pool, rng):
    bounds = correction.bounds_of(task)
    d = len(bounds)
    z0 = best["z"]
    if best["covariance"] is not None:
        unit = expit(z0)
        dz = (bounds[:, 1] - bounds[:, 0]) * unit * (1 - unit)
        cov_z = np.asarray(best["covariance"]) / np.outer(dz, dz)
        sd = np.minimum(np.sqrt(np.diag(cov_z)), 1.0)
        cov_z = cov_z / np.outer(np.sqrt(np.diag(cov_z)), np.sqrt(np.diag(cov_z))) * np.outer(sd, sd)
    else:
        cov_z = np.diag(np.full(d, 0.05**2))
    scale = 2.38**2 / d * cov_z
    for _ in range(2):  # two pilot rounds to adapt the proposal
        pilots = list(pool.map(chain_job, [(task, y, z0, scale, 1500, s + 100 * _) for s in range(2)]))
        stacked = np.concatenate([p[0][500:] for p in pilots])
        cov_z = np.cov(stacked.T)
        scale = 2.38**2 / d * cov_z
    sd = np.sqrt(np.diag(cov_z))
    starts = [z0 + 2 * sd * rng.standard_normal(d) for _ in range(4)]
    mains = list(pool.map(chain_job, [(task, y, s, scale, 4000, 10 + i) for i, s in enumerate(starts)]))
    kept = np.stack([correction.to_bounded(m[0][1000:], bounds) for m in mains])
    diagnostics = [rhat_ess(kept[:, :, i]) for i in range(d)]
    return kept.reshape(-1, d), diagnostics, [float(m[1]) for m in mains]


def delta_method(task, theta, covariance, pool):
    bounds = correction.bounds_of(task)
    h = 0.002 * (bounds[:, 1] - bounds[:, 0])
    jobs = [(task, theta, 32768, 0)]
    for i in range(len(theta)):
        for sign in (1, -1):
            x = theta.copy()
            x[i] = np.clip(x[i] + sign * h[i], bounds[i, 0], bounds[i, 1])
            jobs.append((task, x, 32768, 0))
    values = list(pool.map(product_job, jobs))
    point = values[0]
    gradient = np.stack([(values[1 + 2 * i] - values[2 + 2 * i]) / (2 * h[i]) for i in range(len(theta))], axis=1)
    sd = np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", gradient, covariance, gradient), 0)) if covariance is not None else None
    return point, gradient, sd


def profile_job(args):
    kappa, starts = args
    task = make_task(FULL, False, kappa)
    y = observation(task)
    fits = classical.fit(task, y, lambda th: heston_check.exact_prices(task, th), starts)
    best = min(fits, key=lambda f: f["chi2"])
    return {"kappa": kappa, "chi2": best["chi2"], "theta": best["theta"].tolist(), "on_bound": best["on_bound"],
            "products": product_job((task, best["theta"], 32768, 0)).tolist()}


def summarize(task, y, fits, best, draws, diagnostics, acceptance, pool):
    names = task["parameter_names"]
    modes = classical.distinct_modes(fits)
    wald = classical.wald_interval(best)
    q = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    bounds = correction.bounds_of(task)
    params = {}
    for i, n in enumerate(names):
        entry = {"best": float(best["theta"][i]), "posterior_q05_q50_q95": q[:, i].tolist(),
                 "posterior_mean": float(draws[:, i].mean()), "posterior_sd": float(draws[:, i].std()),
                 "rhat_ess": diagnostics[i] if diagnostics else None}
        if wald is not None:
            entry["gauss_newton_90"] = wald[i].tolist()
            entry["width_ratio_gn_over_posterior"] = float((wald[i, 1] - wald[i, 0]) / (q[2, i] - q[0, i]))
            entry["gn_leaves_prior_box"] = bool(wald[i, 0] < bounds[i, 0] or wald[i, 1] > bounds[i, 1])
        params[n] = entry
    rows = draws[np.linspace(0, len(draws) - 1, 300).astype(int)]
    prices = np.stack(list(pool.map(product_job, [(task, r, 8192, 0) for r in rows], chunksize=10)))
    point, gradient, delta_sd = delta_method(task, np.asarray(best["theta"]), best["covariance"], pool)
    prods = {}
    for j, p in enumerate(products.PRODUCTS):
        pq = np.quantile(prices[:, j], [0.05, 0.5, 0.95])
        entry = {"least_squares_point": float(point[j]), "posterior_q05_q50_q95": pq.tolist(),
                 "posterior_range_90_relative": float((pq[2] - pq[0]) / pq[1])}
        if delta_sd is not None:
            entry["delta_range_90_relative"] = float(2 * 1.645 * delta_sd[j] / point[j])
            entry["delta_over_posterior_range"] = float(2 * 1.645 * delta_sd[j] / (pq[2] - pq[0]))
        prods[p] = entry
    return {"names": names, "n_quotes": len(y), "best_chi2": best["chi2"], "best_on_bound": best["on_bound"],
            "condition": best["condition"], "distinct_modes": len(modes),
            "modes": [{"theta": m["theta"].tolist(), "chi2": m["chi2"]} for m in modes],
            "acceptance": acceptance, "parameters": params, "products": prods,
            "posterior_product_draws": prices.tolist()}


def main(out):
    started = time.time()
    rng = np.random.default_rng(2026_10_23)
    report = {"kappa_bounds": KAPPA_BOUNDS, "cases": {}}
    with ProcessPoolExecutor(2) as pool:
        base = make_task(FULL, False)
        report["kappa_profile"] = list(pool.map(profile_job, [
            (k, list(classical.latin_starts(base, 3, 5)) + [np.asarray(SLOT4_OPTIMUM)])
            for k in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]))
        print("profile", [(p["kappa"], round(p["chi2"], 1), np.round(p["products"], 5).tolist())
                          for p in report["kappa_profile"]], flush=True)
        for name, grid, free in CASES:
            t0 = time.time()
            task = make_task(grid, free)
            y = observation(task)
            extra = [SLOT4_OPTIMUM + [2.0]] if free else [SLOT4_OPTIMUM]
            fits, best = least_squares(task, y, pool, extra)
            if name == "full grid, kappa fixed":
                draws, diagnostics, acceptance = np.load(EXACT_DRAWS).astype(float), None, None
            else:
                draws, diagnostics, acceptance = posterior(task, y, best, pool, rng)
            report["cases"][name] = summarize(task, y, fits, best, draws, diagnostics, acceptance, pool)
            report["cases"][name]["wall_seconds"] = time.time() - t0
            np.save(Path(out).with_name(Path(out).stem + "-" + name.replace(" ", "_").replace(",", "") + ".npy"),
                    draws.astype(np.float32))
            c = report["cases"][name]
            print(f"\n== {name}: {c['n_quotes']} quotes, chi2 {c['best_chi2']:.1f}, modes {c['distinct_modes']}, "
                  f"on bound {c['best_on_bound']}, {c['wall_seconds']:.0f} s", flush=True)
            for n, e in c["parameters"].items():
                print(f"  {n:16s} best {e['best']:.4f} post {np.round(e['posterior_q05_q50_q95'], 4)} "
                      f"GN {np.round(e.get('gauss_newton_90', [np.nan, np.nan]), 4)} ratio "
                      f"{e.get('width_ratio_gn_over_posterior', np.nan):.2f} leaves box {e.get('gn_leaves_prior_box')} "
                      f"rhat/ess {e['rhat_ess']}", flush=True)
            for p, e in c["products"].items():
                print(f"  {p:18s} post range {100 * e['posterior_range_90_relative']:.2f}%  delta "
                      f"{100 * e.get('delta_range_90_relative', np.nan):.2f}%  ratio {e.get('delta_over_posterior_range', np.nan):.2f}",
                      flush=True)
            Path(out).write_text(json.dumps(report, indent=1, default=float))
    report["wall_seconds"] = time.time() - started
    Path(out).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])

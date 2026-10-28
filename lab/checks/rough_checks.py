"""Quick check: is rough Bergomi weakly identified on SPY's design grid when xi0 is uncertain or quotes are few?

    python rough_checks.py OUT.json

1. Least squares on SPY (Monte Carlo prices, 8,192 paths, fixed seed), xi0 = SPY's replicated forward variances.
2. Jacobian of the 63 prices at the optimum with respect to (H, eta, rho) and the 7 log forward-variance
   levels: central differences, 32,768 paths, common random numbers; two seeds to check the noise.
3. Local (Laplace) posterior covariance (J' S^-1 J + P^-1)^-1 for three quote sets (full grid, maturities up
   to 42 days, a sparse 3 x 5 grid) and four prior widths on log xi0 (0 = known, 5%, 10%, 20% per level).
Local only: it shows where a direction is weak relative to the prior box, not the posterior's shape.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration import classical, correction, pricing, surface, tasks  # noqa: E402

TASK = json.loads((ROOT / "configs/tasks/calibration_rough_pilot.json").read_text())["task"]
SPY = json.loads((ROOT / "tutorial_final/results/calibration/spy/surface.json").read_text())
Y = np.asarray(SPY["prices_forward_units"], dtype=float)
LOG_XI = np.log(np.asarray(SPY["forward_variance_levels"], dtype=float))
SIGMA = tasks.sigma(TASK)
CONTRACTS = TASK["context"]["contracts"]
BLOCKS = TASK["numerics"]["blocks"]
NAMES = TASK["parameter_names"]
B = correction.bounds_of(TASK)
LOGK, T, _ = surface.contract_arrays(CONTRACTS)
DAYS = np.rint(np.asarray(T) * surface.TRADING_DAYS).astype(int)
DELTA = np.round(2 * surface.standardized_moneyness(LOGK, T)) / 2
SETS = {
    "full grid (63 quotes)": np.ones(len(Y), bool),
    "maturities up to 42 days (36)": DAYS <= 42,
    "3 maturities x 5 strikes (15)": np.isin(DAYS, [21, 63, 126]) & np.isin(DELTA, [-1, -0.5, 0, 0.5, 1]),
}
XI_SD = [0.0, 0.05, 0.10, 0.20]
STEP = np.array([0.005, 0.03, 0.01] + [0.02] * len(LOG_XI))


def prices(theta, log_xi, paths, seed):
    curve = tasks.curve(TASK, np.exp(log_xi))
    return pricing.rough_surface(tasks.model_parameters(TASK, theta), curve, CONTRACTS, BLOCKS, [paths], SIGMA,
                                 np.inf, seed)[0]


def fit_start(start):
    started = time.process_time()
    f = classical.fit(TASK, Y, lambda th: prices(th, LOG_XI, 8192, 0), [start])[0]
    return {"theta": f["theta"].tolist(), "chi2": f["chi2"], "on_bound": f["on_bound"],
            "covariance": None if f["covariance"] is None else f["covariance"].tolist(),
            "cpu_seconds": time.process_time() - started}


def evaluate(args):
    x, paths, seed = args
    return prices(x[:3], x[3:], paths, seed)


def jacobian(x0, pool, seed, paths=32768):
    points = []
    for i in range(len(x0)):
        for sign in (1, -1):
            x = x0.copy()
            x[i] += sign * STEP[i]
            points.append((x, paths, seed))
    values = list(pool.map(evaluate, points))
    return np.stack([(values[2 * i] - values[2 * i + 1]) / (2 * STEP[i]) for i in range(len(x0))], axis=1)


def local_posterior(jac, rows, xi_sd):
    """Covariance of (H, eta, rho) under the Laplace approximation, xi0 known (xi_sd 0) or Gaussian in log."""
    jn = jac[rows] / SIGMA[rows, None]
    if xi_sd == 0:
        info = jn[:, :3].T @ jn[:, :3]
    else:
        info = jn.T @ jn + np.diag([0.0] * 3 + [1 / xi_sd**2] * (jac.shape[1] - 3))
    return np.linalg.inv(info)[:3, :3]


def describe(cov, theta):
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    width = B[:, 1] - B[:, 0]
    lower, upper = theta - 1.645 * sd, theta + 1.645 * sd
    return {"sd": sd.tolist(), "sd_over_prior_width": (sd / width).tolist(),
            "contraction": (1 - sd**2 / (width**2 / 12)).tolist(),
            "corr_eta_rho": float(corr[1, 2]), "corr_hurst_eta": float(corr[0, 1]),
            "corr_hurst_rho": float(corr[0, 2]),
            "interval_leaves_prior_box": [bool(lo < b[0] or up > b[1]) for lo, up, b in zip(lower, upper, B)]}


def main(out):
    started = time.time()
    starts = classical.latin_starts(TASK, 4, 7)
    with ProcessPoolExecutor(2) as pool:
        fits = list(pool.map(fit_start, starts))
        best = min(fits, key=lambda f: f["chi2"])
        theta = np.asarray(best["theta"])
        x0 = np.concatenate([theta, LOG_XI])
        jacs = [jacobian(x0, pool, seed) for seed in (1, 2)]
    scenarios = []
    for name, rows in SETS.items():
        for xi_sd in XI_SD:
            entry = {"quotes": name, "n_quotes": int(rows.sum()), "xi0_log_sd": xi_sd}
            for s, jac in zip((1, 2), jacs):
                entry[f"seed{s}"] = describe(local_posterior(jac, rows, xi_sd), theta)
            scenarios.append(entry)
    report = {"names": NAMES, "prior_bounds": B.tolist(), "fits": fits, "best": best,
              "chi2_per_quote": best["chi2"] / len(Y), "scenarios": scenarios,
              "jacobian_seed1": jacs[0].tolist(), "wall_seconds": time.time() - started}
    Path(out).write_text(json.dumps(report, indent=1))
    print("best", np.round(theta, 3), "chi2", round(best["chi2"], 1), "starts",
          [(np.round(f["theta"], 3).tolist(), round(f["chi2"], 1)) for f in fits])
    for e in scenarios:
        a, b = e["seed1"], e["seed2"]
        print(f"{e['quotes']:32s} xi_sd {e['xi0_log_sd']:.2f}  sd/width H,eta,rho "
              f"{np.round(a['sd_over_prior_width'], 3)} (seed2 {np.round(b['sd_over_prior_width'], 3)})  "
              f"corr(eta,rho) {a['corr_eta_rho']:+.2f} ({b['corr_eta_rho']:+.2f})  leaves box {a['interval_leaves_prior_box']}")


if __name__ == "__main__":
    main(sys.argv[1])

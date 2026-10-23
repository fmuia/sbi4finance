"""Exact Heston posterior (kappa = 2 fixed) on the SPY smile of 21 Oct 2026: a feasibility check for slot 4.

    python scripts/curriculum/heston_spy_exact.py OUT.json    (writes OUT.json and the draws as OUT.npy)

The curriculum's tutorial_final/curriculum/data/heston-spy-exact.json was built from this run's output,
with the variance-swap quantiles added from the draws. Not an accepted study: study S8-H replaces it.

Random-walk Metropolis in logit coordinates with the Fourier likelihood, 4 chains started near the
least-squares fit of Lecture 6 (kappa = 2), scale from a short pilot chain. Writes a JSON summary.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tutorial_final" / "studies"))

from sbi4finance.calibration import correction  # noqa: E402
import heston_check  # noqa: E402

TASK = json.loads((ROOT / "configs/tasks/calibration_heston_bank.json").read_text())["task"]
SPY = json.loads((ROOT / "tutorial_final/results/calibration/spy/surface.json").read_text())
Y = np.asarray(SPY["prices_forward_units"], dtype=float)
B = correction.bounds_of(TASK)
START = np.array([0.0182, 0.0657, 1.14, -0.631])


def log_target(z):
    theta = correction.to_bounded(z, B)
    try:
        ll = heston_check.exact_loglik(TASK, Y, theta)[0]
    except Exception:
        return -np.inf
    return float(ll) + float(correction.log_jacobian(z, B))


def run(args):
    seed, steps, scale, start = args
    t = time.process_time()
    chain, acc = correction.random_walk_chain(log_target, start, scale, steps, seed)
    return chain, acc, time.process_time() - t


if __name__ == "__main__":
    z0 = correction.to_unbounded(START, B)
    # Pilot: diagonal scale, one chain, to estimate the covariance.
    pilot_scale = np.diag([0.02, 0.05, 0.05, 0.02]) ** 2
    with ProcessPoolExecutor(4) as pool:
        pilots = list(pool.map(run, [(s, 1500, pilot_scale, z0) for s in range(4)]))
    stacked = np.concatenate([c[500:] for c, _, _ in pilots])
    cov = np.cov(stacked.T) * 2.38**2 / 4
    starts = [c[-1] for c, _, _ in pilots]
    with ProcessPoolExecutor(4) as pool:
        mains = list(pool.map(run, [(10 + s, 3000, cov, starts[s]) for s in range(4)]))
    draws = np.concatenate([correction.to_bounded(c[500:], B) for c, _, _ in mains])
    names = TASK["parameter_names"]
    q = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    out = {
        "pilot_acceptance": [a for _, a, _ in pilots],
        "acceptance": [a for _, a, _ in mains],
        "cpu_seconds": sum(t for _, _, t in pilots) + sum(t for _, _, t in mains),
        "draws": int(len(draws)),
        "summary": {n: {"mean": float(draws[:, i].mean()), "sd": float(draws[:, i].std()),
                        "q05": float(q[0, i]), "q50": float(q[1, i]), "q95": float(q[2, i])}
                    for i, n in enumerate(names)},
        "feller_ratio_median": float(np.median(draws[:, 2] ** 2 / (2 * 2.0 * draws[:, 1]))),
        "chain_means": [correction.to_bounded(c[500:], B).mean(axis=0).tolist() for c, _, _ in mains],
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    np.save(Path(sys.argv[1]).with_suffix(".npy"), draws)
    print(json.dumps(out, indent=1))

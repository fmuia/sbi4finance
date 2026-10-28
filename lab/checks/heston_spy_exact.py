"""Exact Heston posterior (kappa = 2 fixed) on the SPY smile of 21 Oct 2026: a feasibility check for slot 4.

    python lab/checks/heston_spy_exact.py OUT.json    (writes OUT.json and the draws as OUT.npy)

Three steps, all with the repository's Fourier likelihood (one quadrature rung of 1,024 nodes, the full
ladder where that rung fails) and the prior box of configs/tasks/calibration_heston_bank.json:

1. Multi-start: Nelder-Mead on the negative log-likelihood in logit coordinates from 8 starting points
   drawn uniformly over the prior box, to count distinct optima.
2. Pilot: 4 random-walk Metropolis chains of 1,500 steps from the best optimum, for the proposal scale.
3. Main: 4 chains of 4,000 steps started at overdispersed points (the optimum plus three pilot standard
   deviations of noise in logit coordinates), the first 1,000 steps discarded; split R-hat and bulk ESS
   per parameter (as in sbi4finance.svj.pmmh.rhat_ess).

Not an accepted study: study S8-H, through lab/studies/, replaces it.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "src"))
sys.path.insert(0, str(ROOT / "lab" / "studies"))

from sbi4finance.calibration import correction  # noqa: E402
import heston_check  # noqa: E402

TASK = json.loads((ROOT / "lab/configs/tasks/calibration_heston_bank.json").read_text())["task"]
SPY = json.loads((ROOT / "tutorial/assets/results/calibration/spy/surface.json").read_text())
Y = np.asarray(SPY["prices_forward_units"], dtype=float)
B = correction.bounds_of(TASK)
KAPPA = float(TASK["fixed_parameters"]["kappa"])
DAYS = np.array([5, 10, 21, 42, 63, 126, 252])
SPY_VARIANCE_SWAP_VOL = [13.6, 15.0, 16.2, 17.6, 18.4, 20.1, 21.8]  # study S8: from the replicated forward-variance levels in results/calibration/spy/surface.json


# Rank-normalized split R-hat and bulk ESS, copied from sbi4finance.svj.pmmh, which imports jax.
def _rank_normalize(chains):
    ranks = rankdata(chains, method="average").reshape(chains.shape)
    return ndtri((ranks - 3 / 8) / (chains.size + 1 / 4))


def _split(chains):
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)


def _rhat(chains):
    n = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean()
    between_over_n = chains.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * within + between_over_n) / within)


def _ess(chains):
    m, n = chains.shape
    centred = chains - chains.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n] / n
    within = (acov[:, 0] * n / (n - 1)).mean()
    var_plus = within * (n - 1) / n + chains.mean(axis=1).var(ddof=1)
    rho = 1 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[:-1:2] + rho[1::2]  # Geyer's initial positive, monotone sequence
    positive = np.argmax(pairs < 0) if np.any(pairs < 0) else len(pairs)
    pairs = np.minimum.accumulate(pairs[:positive])
    tau = max(-1 + 2 * pairs.sum(), 1 / np.log10(m * n))
    return m * n / tau




def rhat_ess(chains):
    split = _split(np.asarray(chains, dtype=float))
    bulk = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    return float(max(_rhat(bulk), _rhat(folded))), float(_ess(bulk))


def loglik(theta):
    try:
        return float(heston_check.exact_loglik(TASK, Y, theta)[0])
    except Exception:
        return -np.inf


def log_target(z):
    """Log-likelihood plus the log-Jacobian of the logit map; the uniform prior is a constant."""
    return loglik(correction.to_bounded(z, B)) + float(correction.log_jacobian(z, B))


def optimize(z0):
    t = time.process_time()
    result = minimize(lambda z: -loglik(correction.to_bounded(z, B)), z0, method="Nelder-Mead",
                      options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-3})
    return result.x, -result.fun, time.process_time() - t


def chain(args):
    seed, steps, scale, start = args
    t = time.process_time()
    draws, acceptance = correction.random_walk_chain(log_target, start, scale, steps, seed)
    return draws, acceptance, time.process_time() - t


def main(out):
    rng = np.random.default_rng(2026_10_23)
    starts = correction.to_unbounded(B[:, 0] + (B[:, 1] - B[:, 0]) * rng.uniform(0.05, 0.95, (8, 4)), B)
    with ProcessPoolExecutor(4) as pool:
        optima = list(pool.map(optimize, starts))
    best = max(optima, key=lambda o: o[1])
    endpoints = [correction.to_bounded(z, B) for z, _, _ in optima]
    distinct = []
    for theta, (_, ll, _) in zip(endpoints, optima):
        # Two optima are the same if every parameter agrees to 2% of its prior range.
        if not any(np.all(np.abs(theta - d["theta"]) < 0.02 * (B[:, 1] - B[:, 0])) for d in distinct):
            distinct.append({"theta": theta, "loglik": ll})

    pilot_scale = np.diag([0.02, 0.05, 0.05, 0.02]) ** 2
    with ProcessPoolExecutor(4) as pool:
        pilots = list(pool.map(chain, [(s, 1500, pilot_scale, best[0]) for s in range(4)]))
    stacked = np.concatenate([d[500:] for d, _, _ in pilots])
    scale = np.cov(stacked.T) * 2.38**2 / 4
    sd = np.sqrt(np.diag(np.cov(stacked.T)))
    dispersed = [best[0] + 3 * sd * rng.standard_normal(4) for _ in range(4)]
    with ProcessPoolExecutor(4) as pool:
        mains = list(pool.map(chain, [(10 + s, 4000, scale, dispersed[s]) for s in range(4)]))

    kept = np.stack([correction.to_bounded(d[1000:], B) for d, _, _ in mains])  # (chains, steps, 4)
    draws = kept.reshape(-1, 4)
    names = TASK["parameter_names"]
    q = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    feller = draws[:, 2] ** 2 / (2 * KAPPA * draws[:, 1])
    T = DAYS / 252
    w = draws[:, 1:2] + (draws[:, 0:1] - draws[:, 1:2]) * (1 - np.exp(-KAPPA * T)) / (KAPPA * T)
    vs = np.quantile(100 * np.sqrt(w), [0.05, 0.5, 0.95], axis=0)
    diagnostics = {n: dict(zip(("rhat", "ess"), rhat_ess(kept[:, :, i]))) for i, n in enumerate(names)}
    report = {
        "what": ("Feasibility run of 23 Oct 2026 for the curriculum: exact Heston posterior (kappa = 2 fixed, "
                 "the prior box of configs/tasks/calibration_heston_bank.json) on the SPY smile of 21 Oct 2026. "
                 "Not an accepted study; study S8-H replaces it."),
        "script": "lab/checks/heston_spy_exact.py",
        "likelihood": "Fourier, one quadrature rung (1,024 nodes), full ladder where that rung fails",
        "multi_start": {"starts": len(starts), "distinct_optima": len(distinct),
                        "optima": [{"theta": d["theta"].tolist(), "loglik": d["loglik"]} for d in distinct],
                        "all_endpoints_loglik": [o[1] for o in optima]},
        "chains": {"pilot_steps": 1500, "main_steps": 4000, "burn_in": 1000, "chains": 4,
                   "starts": "best optimum plus 3 pilot standard deviations of noise, in logit coordinates",
                   "acceptance": [a for _, a, _ in mains]},
        "cpu_seconds": sum(o[2] for o in optima) + sum(t for _, _, t in pilots) + sum(t for _, _, t in mains),
        "draws": int(len(draws)),
        "summary": {n: {"mean": float(draws[:, i].mean()), "sd": float(draws[:, i].std()),
                        "q05": float(q[0, i]), "q50": float(q[1, i]), "q95": float(q[2, i]),
                        **diagnostics[n]} for i, n in enumerate(names)},
        "feller_ratio": {"definition": "vol_of_vol^2 / (2 kappa theta)", "min": float(feller.min()),
                         "median": float(np.median(feller)), "max": float(feller.max()),
                         "share_at_most_1": float(np.mean(feller <= 1))},
        "feller_limit_vol_of_vol_median": float(np.median(np.sqrt(2 * KAPPA * draws[:, 1]))),
        "variance_swap_vol_pct": {"days": DAYS.tolist(), "q05": vs[0].round(3).tolist(),
                                  "q50": vs[1].round(3).tolist(), "q95": vs[2].round(3).tolist(),
                                  "spy_replicated": SPY_VARIANCE_SWAP_VOL,
                                  "spy_source": ("study S8, tutorial/assets/results/calibration/spy/surface.json: "
                                                 "replication from the whole chain, strikes down to delta = -8")},
    }
    Path(out).write_text(json.dumps(report, indent=1))
    np.save(Path(out).with_suffix(".npy"), draws.astype(np.float32))
    print(json.dumps({k: report[k] for k in ("multi_start", "summary", "feller_ratio", "cpu_seconds")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

"""Rerun the 3 x 5 grid, kappa-free Heston case with longer chains; replace its entry in heston_checks.json."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heston_checks as h  # noqa: E402
from sbi4finance.calibration import correction  # noqa: E402

NAME = "3 x 5 grid, kappa free"


def main():
    rng = np.random.default_rng(7)
    task = h.make_task(h.SPARSE, True)
    y = h.observation(task)
    bounds = correction.bounds_of(task)
    d = len(bounds)
    with ProcessPoolExecutor(2) as pool:
        fits, best = h.least_squares(task, y, pool, [h.SLOT4_OPTIMUM + [2.0]])
        z0 = best["z"]
        scale = 2.38**2 / d * np.diag(np.full(d, 0.1**2))
        start = [z0, z0]
        for r in range(3):  # three adaptation rounds, each from where the last ended
            pilots = list(pool.map(h.chain_job, [(task, y, start[s], scale, 3000, 200 + 10 * r + s) for s in range(2)]))
            stacked = np.concatenate([p[0][1000:] for p in pilots])
            cov = np.cov(stacked.T)
            scale = 2.38**2 / d * cov
            start = [p[0][-1] for p in pilots]
        sd = np.sqrt(np.diag(cov))
        starts = [stacked[rng.integers(len(stacked))] + 0.5 * sd * rng.standard_normal(d) for _ in range(4)]
        mains = list(pool.map(h.chain_job, [(task, y, s, scale, 25000, 300 + i) for i, s in enumerate(starts)]))
        kept = np.stack([correction.to_bounded(m[0][5000:], bounds) for m in mains])
        diagnostics = [h.rhat_ess(kept[:, :, i]) for i in range(d)]
        draws = kept.reshape(-1, d)
        summary = h.summarize(task, y, fits, best, draws, diagnostics, [float(m[1]) for m in mains], pool)
    summary["chains"] = "3 adaptation rounds of 2 x 3,000; 4 chains of 25,000 steps, first 5,000 discarded"
    np.save(HERE / "heston_checks-3_x_5_grid_kappa_free.npy", draws.astype(np.float32))
    report = json.loads((HERE / "heston_checks.json").read_text())
    report["cases"][NAME] = summary
    (HERE / "heston_checks.json").write_text(json.dumps(report, indent=1, default=float))
    for n, e in summary["parameters"].items():
        print(f"  {n:16s} best {e['best']:.4f} post {np.round(e['posterior_q05_q50_q95'], 4)} "
              f"GN {np.round(e.get('gauss_newton_90', [np.nan, np.nan]), 4)} ratio "
              f"{e.get('width_ratio_gn_over_posterior', np.nan):.2f} leaves box {e.get('gn_leaves_prior_box')} "
              f"rhat/ess {np.round(e['rhat_ess'], 3)}")
    for p, e in summary["products"].items():
        print(f"  {p:18s} post range {100 * e['posterior_range_90_relative']:.2f}%  delta "
              f"{100 * e.get('delta_range_90_relative', np.nan):.2f}%  ratio {e.get('delta_over_posterior_range', np.nan):.2f}")
    print("acceptance", np.round(summary["acceptance"], 3))


if __name__ == "__main__":
    main()

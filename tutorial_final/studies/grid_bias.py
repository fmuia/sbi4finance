"""The bank's time-grid bias: each surface priced on the bank's grid and on a finer one (study S1-G).

    python tutorial_final/studies/grid_bias.py --bank BANK --out NEW_DIR \
        [--first 6000 --surfaces 50 --paths 131072 --refine 2 --workers 4 --seed 0]

The bank prices with the hybrid scheme on fixed grids (tasks.rough_task: 8 steps a day to 63 days, 2 a day beyond).
The Monte Carlo standard error the bank stores does not include the discretization bias. For each test surface of
rows [first, first + surfaces), this study prices the 63 design contracts at the surface's parameters and curve on
the bank's grid and on a grid refine times finer, each at paths paths on its own seed, and reports the difference
in units of the quote noise sd, with its Monte Carlo standard error. The bank's model is the scheme on its grids;
the difference says how far that model is from a finer discretization of rough Bergomi. Test rows 6000 to 6049 are
S1-G's (tutorial_final/studies/README.md). Writes grid-bias.json and grid-bias.png.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sbi4finance.calibration import bank, pricing, tasks  # noqa: E402

_TAG = 0x53314742  # "S1GB"


def job(args):
    try:
        return job_inner(args)
    except Exception as error:  # noqa: BLE001 - reported in the summary
        return {"test_row": int(args[1]), "failed": f"{type(error).__name__}: {error}"}


def job_inner(args):
    task, row, parameters, context, paths, refine, seed = args
    started = time.process_time()
    curve = tasks.curve(task, context)
    contracts = task["context"]["contracts"]
    sigma = tasks.sigma(task)
    model = tasks.model_parameters(task, parameters)
    blocks = task["numerics"]["blocks"]
    finer = [{"horizon": b["horizon"], "n_steps": refine * b["n_steps"]} for b in blocks]
    out = {}
    for name, grid, purpose in (("bank", blocks, 0), ("finer", finer, 1)):
        stream = int(np.random.SeedSequence([_TAG, int(seed), int(row), purpose]).generate_state(1, dtype=np.uint32)[0])
        prices, errors, _, _ = pricing.rough_surface(model, curve, contracts, grid, [paths], sigma, np.inf, stream,
                                                     cap=None)
        out[name] = (prices, errors)
    difference = (out["bank"][0] - out["finer"][0]) / sigma
    se = np.sqrt(out["bank"][1] ** 2 + out["finer"][1] ** 2) / sigma
    return {"test_row": int(row), "parameters": np.asarray(parameters).tolist(),
            "difference_over_sigma": difference.tolist(), "se_over_sigma": se.tolist(),
            "cpu_seconds": time.process_time() - started}


def figure(report, task, path):
    from sbi4finance.calibration.plots import INK, SERIES, plt, style

    style()
    records = report["surfaces"]
    hurst = np.array([r["parameters"][task["parameter_names"].index("hurst")] for r in records])
    diff = np.array([r["difference_over_sigma"] for r in records])
    se = np.array([r["se_over_sigma"] for r in records])
    worst = np.abs(diff).max(axis=1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
    left.scatter(hurst, worst, s=18, color=SERIES[0])
    left.set_xlabel("H")
    left.set_ylabel("largest |bank − finer| / σ over the 63 quotes")
    left.set_title("Grid bias per surface", loc="left", fontsize=9.5, color=INK)
    days = np.rint(np.array([c["maturity"] for c in task["context"]["contracts"]]) * 252).astype(int)
    for k, d in enumerate(sorted(set(days))):
        sel = days == d
        right.errorbar(np.full(sel.sum(), k) + np.linspace(-0.3, 0.3, sel.sum()), diff[:, sel].mean(axis=0),
                       yerr=2 * np.sqrt((se[:, sel] ** 2).mean(axis=0) / len(records)), fmt="o", ms=3,
                       color=SERIES[1])
    right.axhline(0, color="0.5", lw=0.8)
    right.set_xticks(range(len(set(days))), [f"{d} d" for d in sorted(set(days))])
    right.set_ylabel("mean (bank − finer) / σ, ±2 SE")
    right.set_title("Mean bias by maturity, deltas left to right", loc="left", fontsize=9.5, color=INK)
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=6000)
    parser.add_argument("--surfaces", type=int, default=50)
    parser.add_argument("--paths", type=int, default=131072)
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank, splits=("test",))
    task = manifest["configuration"]["task"]
    test = splits["test"]
    rows = np.arange(args.first, min(args.first + args.surfaces, len(test["parameters"])))
    if not len(rows):
        raise ValueError("no test rows in range")
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = [(task, r, test["parameters"][r], test["context_values"][r], args.paths, args.refine, args.seed) for r in rows]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(job, jobs))
    failed = [r for r in records if "failed" in r]
    records = [r for r in records if "failed" not in r]
    diff = np.array([r["difference_over_sigma"] for r in records])
    se = np.array([r["se_over_sigma"] for r in records])
    chi2 = np.sum(diff**2, axis=1) - np.sum(se**2, axis=1)  # minus the Monte Carlo part of the expected sum
    summary = {"surfaces": len(records), "failed": failed, "paths": args.paths, "refine": args.refine,
               "abs_difference_over_sigma_quantiles": dict(zip(("q50", "q90", "q99", "max"),
                                                               np.quantile(np.abs(diff), [0.5, 0.9, 0.99, 1.0]).tolist())),
               "share_of_quotes_with_abs_z_above_3": float(np.mean(np.abs(diff) > 3 * se)),
               "bias_chi2_per_surface_quantiles": dict(zip(("q50", "q90", "max"),
                                                           np.quantile(chi2, [0.5, 0.9, 1.0]).tolist())),
               "median_se_over_sigma": float(np.median(se)),
               "cpu_seconds_per_surface_median": float(np.median([r["cpu_seconds"] for r in records]))}
    report = {"study": "S1-G", "task_fingerprint": manifest["task_fingerprint"], "summary": summary,
              "surfaces": records}
    (args.out / "grid-bias.json").write_text(json.dumps(report, indent=1) + "\n")
    figure(report, task, args.out / "grid-bias.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

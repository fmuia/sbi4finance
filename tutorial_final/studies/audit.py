"""The audit of rough-Bergomi surfaces with the Monte Carlo likelihood (study S5).

    python tutorial_final/studies/audit.py --bank ROUGH_BANK --checkpoint RUN --out NEW_DIR \
        [--first 2500 --surfaces 500 --proposals 512 --paths 8192 --workers 4]

For each surface: the network's draws give a Student t proposal (mixed with
10% of the prior); each proposal's design surface is priced by Monte Carlo
(antithetic paths, control variates; common random numbers across proposals);
the likelihood adds each price's Monte Carlo variance to the quote noise
(design document, Appendix G.3); the weights give the efficiency and k-hat.
Reported: their distributions, the flag rate at k-hat > 0.7, the cost per
surface, and the network's error against the reweighted posterior.

Each finished surface is appended to OUT.partial.jsonl, next to the output
directory; rerunning the same command after an interruption audits only the
surfaces still missing, and the file is removed once audit.json is written.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sbi4finance.calibration import bank, correction, pricing, tasks, validation  # noqa: E402
from evaluate_calibration import load_network, posterior_draws  # noqa: E402
from heston_check import jsonable  # noqa: E402


def audit_job(job):
    task, observation, context, draws, proposals, paths, seed = job
    curve = tasks.curve(task, context)
    contracts = task["context"]["contracts"]
    started = time.process_time()
    theta, log_q, _ = correction.propose(draws, task, proposals, seed)
    prices, variances = [], []
    for row in theta:
        p, se, _, _ = pricing.rough_surface(tasks.model_parameters(task, row), curve, contracts,
                                            task["numerics"]["blocks"], [paths], tasks.sigma(task), np.inf, seed)
        prices.append(p)
        variances.append(se**2)
    log_lik = correction.gaussian_loglik(task, observation, np.array(prices), np.array(variances))
    result = correction.correct(log_lik, log_q, task)
    summary = correction.weighted_summary(theta, result["weights"])
    agree = validation.agreement(draws, reference_mean=summary["mean"], reference_sd=summary["sd"],
                                 reference_q05=summary["quantiles"][:, 0], reference_q95=summary["quantiles"][:, 2])
    return {"efficiency": result["efficiency"], "khat": result["khat"], "max_weight": result["max_weight"],
            "network_vs_audited": agree, "cpu_seconds": time.process_time() - started}


def figure(report, path):
    """Figure S5-1: each surface's audit efficiency against the network's largest standardized error."""
    from sbi4finance.calibration.plots import BASELINE, INK, SERIES, plt, style

    style()
    surfaces = report["surfaces"]
    error = np.array([max(v for v in s["network_vs_audited"]["standardized_mean_error"] if v is not None)
                      for s in surfaces])
    efficiency = np.array([s["efficiency"] for s in surfaces])
    flagged = np.array([s["khat"] is None or s["khat"] > 0.7 for s in surfaces])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    grid = np.linspace(0, max(4.0, float(np.quantile(error, 0.95))), 200)
    ax.plot(grid, np.exp(-grid**2), color=BASELINE, linestyle="--", label="e^(−δ²)")
    ax.scatter(error[~flagged], efficiency[~flagged], s=22, color=SERIES[0], label="k̂ ≤ 0.7")
    ax.scatter(error[flagged], efficiency[flagged], s=22, color=SERIES[1], label="k̂ > 0.7, flagged")
    ax.set_yscale("log")
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(max(1e-4, float(efficiency.min()) / 3), 1.2)
    ax.set_xlabel("network's largest standardized error against the audited posterior, δ")
    ax.set_ylabel("audit efficiency")
    ax.legend(fontsize=8)
    ax.set_title(f"The audit on {len(surfaces)} held-out surfaces", loc="left", color=INK, fontsize=10)
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=2500)
    parser.add_argument("--surfaces", type=int, default=500)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--proposals", type=int, default=512)
    parser.add_argument("--paths", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank)
    task, workflow, _ = load_network(args.checkpoint)
    if task["model"] != "rough":
        raise ValueError("the Monte Carlo audit is for rough Bergomi; Heston's exact audit is study S3")
    test = splits["test"]
    rows = np.arange(args.first, min(args.first + args.surfaces, len(test["parameters"])))
    args.out.mkdir(parents=True, exist_ok=False)
    network = posterior_draws(workflow, task, test, rows, args.draws, args.seed)
    partial = args.out.with_name(args.out.name + ".partial.jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            record = json.loads(line)
            done[record["test_row"]] = record["result"]
    pending = [i for i, r in enumerate(rows) if int(r) not in done]
    jobs = [(task, test["observations"][rows[i], :, 0], test["context_values"][rows[i]], network[i], args.proposals,
             args.paths, args.seed + int(rows[i])) for i in pending]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool, \
            partial.open("a") as log:
        for i, result in zip(pending, pool.map(audit_job, jobs)):
            done[int(rows[i])] = json.loads(json.dumps(jsonable(result)))
            log.write(json.dumps({"test_row": int(rows[i]), "result": done[int(rows[i])]}) + "\n")
            log.flush()
    results = [done[int(r)] for r in rows]
    efficiency = np.array([r["efficiency"] for r in results], dtype=float)
    khat = np.array([np.nan if r["khat"] is None else r["khat"] for r in results], dtype=float)
    error = np.array([max(r["network_vs_audited"]["standardized_mean_error"]) for r in results], dtype=float)
    summary = {"surfaces": len(rows), "proposals": args.proposals, "paths": args.paths,
               "efficiency_quantiles": dict(zip(("q05", "q50", "q95"), np.quantile(efficiency, [0.05, 0.5, 0.95]).tolist())),
               "flag_rate_khat_above_0_7": float(np.mean(np.nan_to_num(khat, nan=0.0) > 0.7)),
               "flag_rate_efficiency_below_0_02": float(np.mean(efficiency < 0.02)),
               "cpu_seconds_per_surface_median": float(np.median([r["cpu_seconds"] for r in results])),
               "network_error_vs_audited_sd_quantiles": dict(zip(("q50", "q90", "max"),
                                                                  [float(np.median(error)), float(np.quantile(error, 0.9)), float(error.max())]))}
    report = {"study": "S5", "task_fingerprint": manifest["task_fingerprint"], "summary": summary,
              "surfaces": [{"test_row": int(r), **res} for r, res in zip(rows, results)]}
    (args.out / "audit.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    figure(json.loads((args.out / "audit.json").read_text()), args.out / "audit.png")
    partial.unlink(missing_ok=True)
    print(json.dumps(jsonable(summary), indent=1))


if __name__ == "__main__":
    main()

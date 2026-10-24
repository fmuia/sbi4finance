"""The audit of rough-Bergomi surfaces with the Monte Carlo likelihood (study S5).

    python tutorial_final/studies/audit.py --bank ROUGH_BANK --checkpoint RUN --out NEW_DIR \
        [--first 2500 --surfaces 200 --proposals 512 --paths 32768 --workers 4]

For each surface: the network's draws give a Student t proposal (mixed with 10% of the prior). Each proposal's
design surface is priced by Monte Carlo on its own seed, so the Monte Carlo errors are independent across
proposals (antithetic paths and control variates within a proposal). The likelihood is the unbiased estimate
N(y; P_hat, diag(sigma^2) + C - S) of the exact likelihood N(y; P, diag(sigma^2) + C): S is the proposal's
Monte Carlo covariance (quotes priced on the same paths have correlated errors) and C the bank's own Monte Carlo
covariance of the surface's reference prices, since y = P_ref + noise (correction.gaussian_loglik_mc). A
proposal is priced again on a fresh stream with twice the paths, up to eight times, while a quote's error exceeds
half its noise sd, the covariance's largest eigenvalue exceeds half the noise variance, or the sum of (SE/sigma)^2
exceeds 1; one that still fails gets zero weight and is counted, instead of the whole surface being dropped. The weights give the efficiency and k-hat.

Earlier versions priced every proposal of a surface with one seed (common random numbers), with which this
estimate is not unbiased: the weights converge to the posterior given that seed's error field, which is shifted
and too narrow. Results computed with them (the pilot's S5 of 22 October 2026) must be rerun.

Reported: the distributions of efficiency and k-hat, the flag rates at k-hat above 0.7 and above the
sample-size threshold min(1 - 1/log10 M, 0.7), the proposals that could not be priced, the cost per surface, and
the network's error against the reweighted posterior, on all surfaces and on those with k-hat at most 0.7 (the
reliable audits). Each surface's record keeps its parameters and curve, so flag rates can be read by region.

Each finished surface is appended to OUT.partial.jsonl, next to the output directory; rerunning the same command
after an interruption audits only the surfaces still missing, and the file is removed once audit.json is written.
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


MC_BUDGET = 0.5  # largest Monte Carlo SE / quote-noise sd per quote; the paths double up to 8x to meet it
_TAG = 0x53354155  # "S5AU"


def price_proposal(row, seed, task, curve, contracts, paths, max_doublings=3):
    """(prices, covariance, paths) on the proposal's own seed, or (None, None, reason).

    Prices at paths, then 2, 4 and 8 times as many, each on a fresh stream, until every quote's SE is at most
    MC_BUDGET of its noise sd, the covariance's largest eigenvalue relative to the noise variance is at most
    correction.MAX_RELATIVE_EIGENVALUE and its trace at most correction.MAX_RELATIVE_TRACE; the decision uses the
    estimated errors only, so it does not bias the likelihood estimate.
    """
    sigma = tasks.sigma(task)
    model = tasks.model_parameters(task, row)
    reason = None
    for rung in range(max_doublings + 1):
        n = paths * 2**rung
        stream = int(np.random.SeedSequence([_TAG, int(seed), rung]).generate_state(1, dtype=np.uint32)[0])
        try:
            prices, errors, cov, _ = pricing.rough_surface(model, curve, contracts, task["numerics"]["blocks"], [n],
                                                           sigma, np.inf, stream, cap=None)
        except ValueError as error:
            reason = str(error)
            continue
        ratio = correction.relative_mc_eigenvalue(sigma, cov)
        worst = float(np.max(errors / sigma))
        trace = float(np.sum((errors / sigma) ** 2))
        if worst <= MC_BUDGET and ratio <= correction.MAX_RELATIVE_EIGENVALUE and trace <= correction.MAX_RELATIVE_TRACE:
            return prices, cov, int(n)
        reason = f"at {n} paths: largest SE/sigma {worst:.2f}, covariance eigenvalue {ratio:.2f}, trace {trace:.2f}"
    return None, None, reason


def audit_job(job):
    """One surface; an error is recorded instead of stopping the run (and its row is not retried on resume)."""
    try:
        return audit_job_inner(job)
    except Exception as error:  # noqa: BLE001 - reported in the summary
        return {"failed": f"{type(error).__name__}: {error}", "efficiency": 0.0, "khat": None,
                "unpriced_proposals": 0, "cpu_seconds": 0.0}


def audit_job_inner(job):
    task, observation, context, bank_covariance, draws, proposals, paths, seed = job
    curve = tasks.curve(task, context)
    contracts = task["context"]["contracts"]
    sigma = tasks.sigma(task)
    started = time.process_time()
    theta, log_q, from_prior = correction.propose(draws, task, proposals, seed)
    log_lik = np.full(len(theta), -np.inf)
    unpriced, unpriced_prior, reasons, escalated = 0, 0, [], 0
    for j, row in enumerate(theta):
        proposal_seed = int(np.random.SeedSequence([_TAG, int(seed), 1, j]).generate_state(1, dtype=np.uint32)[0])
        prices, cov, info = price_proposal(row, proposal_seed, task, curve, contracts, paths)
        if prices is None:
            unpriced += 1
            unpriced_prior += int(from_prior[j])
            reasons.append(info)
            continue
        escalated += int(info > paths)
        log_lik[j] = correction.gaussian_loglik_mc(observation, prices, sigma, cov, data_covariance=bank_covariance)
    result = correction.correct(log_lik, log_q, task)
    summary = correction.weighted_summary(theta, result["weights"])
    agree = validation.agreement(draws, reference_mean=summary["mean"], reference_sd=summary["sd"],
                                 reference_q05=summary["quantiles"][:, 0], reference_q95=summary["quantiles"][:, 2])
    return {"efficiency": result["efficiency"], "khat": result["khat"], "max_weight": result["max_weight"],
            "unpriced_proposals": unpriced, "unpriced_from_prior": unpriced_prior, "unpriced_reasons": reasons[:3],
            "escalated_proposals": escalated, "network_vs_audited": agree,
            "cpu_seconds": time.process_time() - started}


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


def error_quantiles(results):
    error = np.array([max(v for v in r["network_vs_audited"]["standardized_mean_error"] if v is not None)
                      for r in results], dtype=float)
    if not len(error):
        return None
    return dict(zip(("q50", "q90", "max"), [float(np.median(error)), float(np.quantile(error, 0.9)),
                                            float(error.max())]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=2500)
    parser.add_argument("--surfaces", type=int, default=200)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--proposals", type=int, default=512)
    parser.add_argument("--paths", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest, splits = bank.load_bank(args.bank, splits=("test",))
    task, workflow, _ = load_network(args.checkpoint)
    if task["model"] != "rough":
        raise ValueError("the Monte Carlo audit is for rough Bergomi; Heston's exact audit is study S3")
    if task != manifest["configuration"]["task"]:
        raise ValueError("checkpoint and bank tasks differ")
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
    jobs = [(task, test["observations"][rows[i], :, 0], test["context_values"][rows[i]],
             test["pricing_covariance"][rows[i]], network[i], args.proposals, args.paths, args.seed + int(rows[i]))
            for i in pending]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool, \
            partial.open("a") as log:
        for i, result in zip(pending, pool.map(audit_job, jobs)):
            done[int(rows[i])] = json.loads(json.dumps(jsonable(result)))
            log.write(json.dumps({"test_row": int(rows[i]), "result": done[int(rows[i])]}) + "\n")
            log.flush()
    failed = [int(r) for r in rows if "failed" in done[int(r)]]
    results = [done[int(r)] for r in rows if "failed" not in done[int(r)]]
    efficiency = np.array([r["efficiency"] for r in results], dtype=float)
    khat = np.array([np.inf if r["khat"] is None else r["khat"] for r in results], dtype=float)
    reliable = [r for r, k in zip(results, khat) if k <= 0.7]
    threshold = min(1 - 1 / np.log10(args.proposals), 0.7)  # PSIS sample-size-dependent threshold
    summary = {"surfaces": len(results), "failed_surfaces": failed, "proposals": args.proposals, "paths": args.paths,
               "likelihood": "unbiased N(y; P_hat, diag(sigma^2) + C_bank - S), one seed per proposal",
               "mc_budget": MC_BUDGET, "max_relative_eigenvalue": correction.MAX_RELATIVE_EIGENVALUE,
               "unpriced_proposals_total": int(sum(r["unpriced_proposals"] for r in results)),
               "surfaces_with_unpriced_proposals": int(sum(r["unpriced_proposals"] > 0 for r in results)),
               "khat_threshold": float(threshold),
               "flag_rate_khat_above_threshold": float(np.mean(khat > threshold)),
               "efficiency_quantiles": dict(zip(("q05", "q50", "q95"), np.quantile(efficiency, [0.05, 0.5, 0.95]).tolist())),
               "flag_rate_khat_above_0_7": float(np.mean(khat > 0.7)),
               "flag_rate_efficiency_below_0_02": float(np.mean(efficiency < 0.02)),
               "cpu_seconds_per_surface_median": float(np.median([r["cpu_seconds"] for r in results])),
               "network_error_vs_audited_sd_quantiles_reliable": error_quantiles(reliable),
               "network_error_vs_audited_sd_quantiles_all": error_quantiles(results),
               "reliable_surfaces": len(reliable)}
    report = {"study": "S5", "task_fingerprint": manifest["task_fingerprint"], "summary": summary,
              "surfaces": [{"test_row": int(r), "parameters": test["parameters"][r].tolist(),
                            "forward_variance_levels": test["context_values"][r].tolist(), **done[int(r)]}
                           for r in rows if "failed" not in done[int(r)]]}
    (args.out / "audit.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    figure(json.loads((args.out / "audit.json").read_text()), args.out / "audit.png")
    partial.unlink(missing_ok=True)
    print(json.dumps(jsonable(summary), indent=1))


if __name__ == "__main__":
    main()

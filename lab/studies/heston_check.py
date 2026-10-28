"""The Heston check: the amortized posterior against the exact posterior (study S3).

    python lab/studies/heston_check.py --bank HESTON_BANK --checkpoint RUN --out NEW_DIR \
        [--first 2000 --surfaces 50 --proposals 2048 --chain-surfaces 4 --chains 4 --steps 6000 --workers 4]

Heston has an exact likelihood through the Fourier pricer, so its posterior can
be computed in two independent ways and compared with the network's:

* importance reweighting of a defensive proposal fitted to the network's draws,
  with the exact likelihood (the audit, here exact), on --surfaces test surfaces;
* random-walk Metropolis chains with the exact likelihood, started from
  dispersed network draws, on the first --chain-surfaces of them.

Test surfaces from --first onwards are used, so they do not overlap the S2 set.
Agreement uses the tutorial's criterion: means within 0.3 posterior sd
and 90% widths within 25%, for every parameter.

Each finished audit and chain is saved in OUT.partial/, next to the output
directory; rerunning the same command after an interruption runs only the
missing jobs, and the folder is removed once the report is written.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import pickle
import shutil
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import bank, correction, pricing, tasks, validation  # noqa: E402

LABELS = ["v₀", "θ", "σ_v", "ρ"]
MEAN_LIMIT, WIDTH_LIMIT = 0.3, 0.25


def exact_prices(task, row):
    """Fourier prices for one parameter row; the 1024-node rung, or the full ladder where it fails."""
    parameters, v0 = tasks._heston_split(tasks.model_parameters(task, row))
    try:
        return pricing.heston_surface(parameters, v0, task["context"]["contracts"], nodes=1024, upper_limit=600.0)
    except ValueError:
        return tasks.heston_prices(task, parameters, v0)[0]


def exact_loglik(task, observation, rows):
    rows = np.atleast_2d(rows)
    return correction.gaussian_loglik(task, observation, np.stack([exact_prices(task, r) for r in rows]))


def audit_job(job):
    task, observation, draws, proposals, seed = job
    started = time.process_time()
    theta, log_q, _ = correction.propose(draws, task, proposals, seed)
    result = correction.correct(exact_loglik(task, observation, theta), log_q, task)
    resampled = correction.resample(theta, result["weights"], len(draws), seed)
    summary = correction.weighted_summary(theta, result["weights"])
    return {"efficiency": result["efficiency"], "khat": result["khat"], "max_weight": result["max_weight"],
            "draws": resampled, "summary": summary, "cpu_seconds": time.process_time() - started}


def chain_job(job):
    task, observation, start, scale, steps, seed = job
    bounds = correction.bounds_of(task)

    def log_target(z):
        theta = correction.to_bounded(z, bounds)
        return float(exact_loglik(task, observation, theta)[0] + correction.log_jacobian(z, bounds))

    started = time.process_time()
    chain, acceptance = correction.random_walk_chain(log_target, correction.to_unbounded(start, bounds), scale, steps, seed)
    return correction.to_bounded(chain, bounds), acceptance, time.process_time() - started


def jsonable(value):
    """Non-finite numbers (k-hat with a short tail, R-hat of a stuck chain) become null."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def passes(agreement):
    return bool(max(agreement["standardized_mean_error"]) < MEAN_LIMIT
                and max(abs(r - 1) for r in agreement["width_ratio"]) < WIDTH_LIMIT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=2000)
    parser.add_argument("--surfaces", type=int, default=50)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--proposals", type=int, default=2048)
    parser.add_argument("--chain-surfaces", type=int, default=4)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_calibration import load_network, posterior_draws
    from sbi4finance.svj.pmmh import rhat_ess
    from sbi4finance.calibration import plots

    manifest, splits = bank.load_bank(args.bank)
    task, workflow, record = load_network(args.checkpoint)
    if task != manifest["configuration"]["task"] or task["model"] != "hkde":
        raise ValueError("the Heston check needs the Heston bank and its checkpoint")
    test = splits["test"]
    rows = np.arange(args.first, args.first + args.surfaces)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    network = posterior_draws(workflow, task, test, rows, args.draws, args.seed)
    network_seconds = (time.perf_counter() - started) / len(rows)
    observations = test["observations"][rows, :, 0]
    partial = args.out.with_name(args.out.name + ".partial")
    partial.mkdir(exist_ok=True)

    def resumable(pool, function, jobs, names):
        """pool.map over the jobs not yet saved in partial/, saving each result as it arrives."""
        results = {name: pickle.loads((partial / name).read_bytes()) for name in names if (partial / name).exists()}
        todo = [k for k, name in enumerate(names) if name not in results]
        for k, result in zip(todo, pool.map(function, [jobs[k] for k in todo])):
            temporary = partial / (names[k] + ".tmp")
            temporary.write_bytes(pickle.dumps(result))
            temporary.replace(partial / names[k])
            results[names[k]] = result
        return [results[name] for name in names]

    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        audits = resumable(pool, audit_job, [(task, observations[i], network[i], args.proposals, args.seed + i)
                                             for i in range(len(rows))],
                           [f"audit-{int(r)}.pkl" for r in rows])
        chain_jobs, owners, names = [], [], []
        bounds = correction.bounds_of(task)
        for i in range(min(args.chain_surfaces, len(rows))):
            z = correction.to_unbounded(network[i], bounds)
            scale = np.cov(z.T) * 2.38**2 / z.shape[1]
            starts = network[i][np.random.default_rng(args.seed + i).choice(len(network[i]), args.chains, replace=False)]
            for c in range(args.chains):
                chain_jobs.append((task, observations[i], starts[c], scale, args.steps, 1000 * i + c))
                owners.append(i)
                names.append(f"chain-{int(rows[i])}-{c}.pkl")
        chains = resumable(pool, chain_job, chain_jobs, names)
    surfaces, panels, saved = [], [], {}
    for i, row in enumerate(rows):
        entry = {"test_row": int(row), "truth": test["parameters"][row].tolist(),
                 "audit": {k: audits[i][k] for k in ("efficiency", "khat", "max_weight", "cpu_seconds")}}
        weighted = audits[i]["summary"]
        entry["reweighted"] = {"mean": weighted["mean"].tolist(), "sd": weighted["sd"].tolist(),
                               "q05": weighted["quantiles"][:, 0].tolist(), "q95": weighted["quantiles"][:, 2].tolist()}
        entry["network_vs_reweighted"] = validation.agreement(
            network[i], reference_mean=weighted["mean"], reference_sd=weighted["sd"],
            reference_q05=weighted["quantiles"][:, 0], reference_q95=weighted["quantiles"][:, 2])
        entry["network_vs_reweighted"]["passed"] = passes(entry["network_vs_reweighted"])
        mine = [chains[j] for j, owner in enumerate(owners) if owner == i]
        if mine:
            stacked = np.stack([c[0][args.burn:] for c in mine])
            diagnostics = [rhat_ess(stacked[:, :, d]) for d in range(stacked.shape[2])]
            pooled = stacked.reshape(-1, stacked.shape[2])
            entry["chains"] = {"rhat": [d[0] for d in diagnostics], "ess": [d[1] for d in diagnostics],
                               "acceptance": [c[1] for c in mine], "cpu_seconds": sum(c[2] for c in mine)}
            for name, draws in (("network_vs_chain", network[i]), ("reweighted_vs_chain", audits[i]["draws"])):
                entry[name] = validation.agreement(draws, pooled)
                entry[name]["passed"] = passes(entry[name])
            panels.append((f"surface {int(row)}", pooled, network[i], audits[i]["draws"], test["parameters"][row]))
            saved.update({f"chain_{int(row)}": pooled, f"network_{int(row)}": network[i],
                          f"reweighted_{int(row)}": audits[i]["draws"], f"truth_{int(row)}": test["parameters"][row],
                          f"observation_{int(row)}": observations[i]})
        surfaces.append(entry)
    summary = {
        "network_vs_reweighted_pass_rate": float(np.mean([s["network_vs_reweighted"]["passed"] for s in surfaces])),
        "max_standardized_mean_error": np.max([s["network_vs_reweighted"]["standardized_mean_error"] for s in surfaces], axis=0).tolist(),
        "median_standardized_mean_error": np.median([s["network_vs_reweighted"]["standardized_mean_error"] for s in surfaces], axis=0).tolist(),
        "width_ratio_range": [np.min([s["network_vs_reweighted"]["width_ratio"] for s in surfaces], axis=0).tolist(),
                              np.max([s["network_vs_reweighted"]["width_ratio"] for s in surfaces], axis=0).tolist()],
        "audit_efficiency_median": float(np.median([s["audit"]["efficiency"] for s in surfaces])),
        "audit_khat_above_0_7": int(np.sum([s["audit"]["khat"] > 0.7 for s in surfaces])),
        "audit_cpu_seconds_median": float(np.median([s["audit"]["cpu_seconds"] for s in surfaces])),
        "network_seconds_per_surface": network_seconds,
        "chain_checks": [{k: s[k]["passed"] for k in ("network_vs_chain", "reweighted_vs_chain")}
                         for s in surfaces if "chains" in s],
        "chain_rhat_max": max(max(s["chains"]["rhat"]) for s in surfaces if "chains" in s),
        "chain_ess_min": min(min(s["chains"]["ess"]) for s in surfaces if "chains" in s),
    }
    report = {"study": "S3", "task_fingerprint": manifest["task_fingerprint"],
              "checkpoint_sha256": hashlib.sha256((args.checkpoint / "model.keras").read_bytes()).hexdigest(),
              "criterion": {"mean_within_sd": MEAN_LIMIT, "width_within": WIDTH_LIMIT},
              "settings": {k: v for k, v in vars(args).items() if k not in ("bank", "checkpoint", "out")},
              "summary": summary, "surfaces": surfaces}
    (args.out / "heston-check.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    if panels:
        plots.agreement(panels, LABELS, "exact (Fourier MCMC)").savefig(args.out / "agreement.png", dpi=160)
        np.savez_compressed(args.out / "posterior-draws.npz", **saved)
    shutil.rmtree(partial)
    print(json.dumps(jsonable(summary), indent=1))


if __name__ == "__main__":
    main()

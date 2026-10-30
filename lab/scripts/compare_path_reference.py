"""Guarded single-case Heston comparison after four-chain convergence checks.

A failed or unavailable reference gate produces an inconclusive report without
loading or sampling either neural model. Passing gates permit this development
comparison; they do not certify particle precision or accept a reference.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(out, report):
    temporary = out / "comparison.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "comparison.json")


def run(args):
    reduced = json.loads(args.reference_summary.read_text())
    current = reduced.get("current", {})
    gates = current.get("convergence_gates", {})
    report = {"format": "path_single_case_reference_comparison_v1",
              "reference_summary_sha256": sha256(args.reference_summary),
              "reference_summary": str(args.reference_summary),
              "chains_directory": str(args.chains),
              "reference_accepted": False, "global_accuracy_claim": False,
              "reserved_test_evaluated": False, "neural_models_sampled": False,
              "scope": "one validation history at 252 days; development comparison only; particle-number precision remains a separate requirement"}
    args.out.mkdir(parents=True, exist_ok=False)
    if gates.get("passed") is not True:
        report.update(status="inconclusive_reference", convergence_gates=gates,
                      reason="Reference convergence gates failed or were unavailable; no neural models loaded or sampled.")
        write_report(args.out, report)
        return report

    # Import numerical/source checks only after the gate; importing BayesFlow
    # remains deferred until actual chain/bank identities have been verified.
    from sbi4finance.path_inference import bank, inference
    from summarize_path_pmmh import _load_chains

    if (current.get("input_valid") is not True or gates.get("evaluated") is not True
            or gates.get("rank_rhat_max") != 1.01 or gates.get("bulk_ess_min") != 400
            or gates.get("tail_ess_min") != 400 or gates.get("tail_quantiles") != [.05, .95]):
        raise ValueError("reference gate configuration differs from the declared convergence screen")
    if Path(current["chains_directory"]).resolve() != args.chains.resolve():
        raise ValueError("reference summary describes a different chain directory")
    samples, identity, chain_records = _load_chains(args.chains)
    if current.get("identity") != identity or current.get("shape") != list(samples.shape):
        raise ValueError("current chain identity/shape differs from the reduced reference")
    reduced_chains = {item["chain_id"]: item for item in current.get("chains", [])}
    if set(reduced_chains) != {0, 1, 2, 3}:
        raise ValueError("reducer must describe four complete unique chains")
    for record in chain_records:
        reduced_record = reduced_chains[record["chain_id"]]
        for field in ("summary_sha256", "draws_sha256"):
            if record[field] != reduced_record.get(field):
                raise ValueError("chain files changed after convergence reduction")
    if identity.get("length") != 252 or identity.get("particles") != 512 or identity.get("split") != "validation":
        raise ValueError("comparison requires the declared 252-return, 512-particle validation reference")
    manifest = bank.load_manifest(args.bank)
    task = manifest["task"]
    if (task["model"] != "heston" or task["numerics"].get("substeps") != 8
            or identity["task_sha256"] != bank.task_fingerprint(task)
            or identity["parameter_names"] != task["parameter_names"]):
        raise ValueError("reference and bank must share the eight-substep Heston task")
    first = next(item for item in chain_records if item["chain_id"] == 0)
    first_directory = Path(first["directory"])
    chain_report = json.loads((first_directory / "pmmh-summary.json").read_text())
    observation = chain_report["observation"]
    if (observation.get("case_index") != 0 or observation.get("reserved_test_evaluated") is not False
            or observation.get("truth_used_for_initialization_or_likelihood") is not False):
        raise ValueError("reference must use validation row zero without truth-based initialization")
    part = next(item for item in manifest["splits"]["validation"]
                if item["start"] == 0 and item["count"] > 0)
    if (observation["source_shard"] != part["path"]
            or observation["source_shard_sha256"] != part["sha256"]):
        raise ValueError("reference observation shard differs from the bank")
    arrays = bank._load_part(args.bank, manifest, "validation", part)
    observed = arrays["observations"][0, :252, 0].copy()
    if str(arrays["trajectory_id"][0]) != identity["trajectory_id"]:
        raise ValueError("reference trajectory identity differs from validation row zero")
    with np.load(first_directory / "observations.npz", allow_pickle=False) as stored:
        if (not np.array_equal(stored["log_returns"], observed)
                or not np.array_equal(stored["dt_years"], np.full(252, task["dt_years"]))):
            raise ValueError("reference actual returns/time grid differ from the bank prefix")
    if args.draws != 4096 or args.seed < 0:
        raise ValueError("this fixed comparison requires 4096 neural draws and a nonnegative seed")
    pooled = samples.reshape(-1, samples.shape[-1])
    mean, sd = pooled.mean(axis=0), pooled.std(axis=0, ddof=1)
    quantiles = np.quantile(pooled, [.05, .5, .95], axis=0)
    if not np.isfinite(sd).all() or np.any(sd <= 0) or np.any(quantiles[2] <= quantiles[0]):
        raise ValueError("reference samples must have finite positive marginal spread")
    # The job is explicitly a CPU comparison. No accelerator initialization is
    # needed, and no neural backend is imported along the inconclusive path.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["KERAS_BACKEND"] = "jax"
    report.update(convergence_gates=gates, task_sha256=manifest["task_sha256"],
                  bank_manifest_sha256=sha256(args.bank / "manifest.json"), observation=observation,
                  targets=task["parameter_names"], reference_shape=list(samples.shape),
                  reference={"mean": mean.tolist(), "sd": sd.tolist(),
                             "quantiles_05_50_95": quantiles.tolist(), "chains": chain_records},
                  gates={"maximum_abs_mean_error_reference_sd": .3,
                         "central_90_width_ratio_range": [.75, 1.25]},
                  model_comparisons={})
    saved = {"reference_parameters": samples, "observed_log_returns": observed,
             "dt_years": np.full(252, task["dt_years"]),
             "parameter_names": np.asarray(task["parameter_names"])}
    for label, directory in (("baseline", args.baseline_checkpoint), ("candidate", args.candidate_checkpoint)):
        checkpoint_task, workflow, training = inference.load_checkpoint(directory)
        if (inference.fingerprint(checkpoint_task) != manifest["task_sha256"]
                or training["bank_manifest_sha256"] != report["bank_manifest_sha256"]):
            raise ValueError(f"{label} checkpoint task/bank identity differs from the reference")
        draws = inference.sample(workflow, task, observed[None, :, None], args.draws, seed=args.seed)[0]
        neural_quantiles = np.quantile(draws, [.05, .5, .95], axis=0)
        mean_error = (draws.mean(axis=0) - mean) / sd
        width_ratio = (neural_quantiles[2] - neural_quantiles[0]) / (quantiles[2] - quantiles[0])
        passed = bool(np.all(np.abs(mean_error) <= .3)
                      and np.all((width_ratio >= .75) & (width_ratio <= 1.25)))
        report["model_comparisons"][label] = {
            "checkpoint_directory": str(directory), "checkpoint_sha256": training["model_sha256"],
            "training_report_sha256": sha256(directory / "training.json"),
            "draws": args.draws, "seed": args.seed,
            "mean": draws.mean(axis=0).tolist(), "sd": draws.std(axis=0, ddof=1).tolist(),
            "mean_error_reference_sd": mean_error.tolist(),
            "central_90_width_ratio": width_ratio.tolist(),
            "quantiles_05_50_95": neural_quantiles.tolist(),
            "quantile_error_reference_sd": ((neural_quantiles - quantiles) / sd).tolist(),
            "passes_point_estimate_screen": passed,
            "status": "development_case_agreement" if passed else "development_case_discrepancy"}
        saved[f"{label}_parameters"] = draws
    target = args.out / "comparison-arrays.npz"
    np.savez_compressed(target, **saved)
    report.update(neural_models_sampled=True, arrays_sha256=sha256(target),
                  sampling_note="same declared seed used for both neural estimators; no reference subsampling",
                  monte_carlo_note="point estimates from finite neural and correlated PMMH draws; quantile/mean MCSE and particle-number sensitivity are not certified",
                  status=("development_case_agreement" if all(item["passes_point_estimate_screen"]
                          for item in report["model_comparisons"].values()) else "development_case_discrepancy"))
    write_report(args.out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "chains", "reference-summary", "baseline-checkpoint", "candidate-checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261030231)
    report = run(parser.parse_args())
    print(json.dumps({"status": report["status"], "neural_models_sampled": report["neural_models_sampled"],
                      "reference_accepted": False, "global_accuracy_claim": False}))


if __name__ == "__main__":
    main()

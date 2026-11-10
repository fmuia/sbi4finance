"""Freeze compact larger-batch results beside unchanged historical comparisons.

No model loading, inference or pricing. Native checkpoints are hash-checked;
only compact training records are copied. Historical manifests are referenced,
never extended or overwritten.
"""
import argparse
from pathlib import Path
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import summarize_heston_spread_scale_capacity as shared

fixed, read, binding = shared.fixed, shared.read, shared.binding


def normalizer_counts(checkpoint):
    """Read only the two hash-identified scalar counters, not learned weights."""
    checkpoint = Path(checkpoint)
    metadata = read(checkpoint/"checkpoint.json")
    state_path = checkpoint/"state.npz"
    if fixed.digest(state_path) != metadata["state_sha256"]:
        raise ValueError("normalizer-count state payload differs")
    rows = [(index, row) for index, row in enumerate(metadata["state"]["non_trainable"])
        if re.search(r"/standardization(?:_\d+)?/standardize(?:_\d+)?/", row["path"])
        and row["shape"] == [] and row["dtype"] == "float32"]
    if len(rows) != 2:
        raise ValueError("two saved target/input standardizer counters required")
    counters = []
    with np.load(state_path, allow_pickle=False) as arrays:
        for index, row in rows:
            value = arrays[f"non_trainable_{index:04d}"].copy()
            if (value.shape != () or value.dtype != np.float32 or not np.isfinite(value)
                    or fixed.array_summary(value)["sha256"] != row["sha256"] or value < 0):
                raise ValueError("saved standardizer counter shape, dtype or hash differs")
            counters.append({"path": row["path"], "value": float(value), "dtype": str(value.dtype),
                "sha256": row["sha256"]})
    if counters[0]["value"] != counters[1]["value"]:
        raise ValueError("target and input standardizer counter values differ")
    return {"state": binding(state_path), "counters": counters, "value": counters[0]["value"]}


def build(study_path, old_capacity_manifest, mcmc_manifest, out):
    study_path, old_capacity_manifest, mcmc_manifest, out = map(lambda p: Path(p).resolve(),
        (study_path, old_capacity_manifest, mcmc_manifest, out))
    study = read(study_path)
    if (study.get("format") != "heston_spread_scale_large_batch_study_v1"
            or sorted(a["batch_size"] for a in study["arms"]) != [1024, 4098]):
        raise ValueError("the exact new1024/4098 study is required")
    if out.exists() or (out.parent/"report-manifest.json").exists():
        raise FileExistsError("larger-batch summary and report identities must be new")
    historical = read(old_capacity_manifest)
    mcmc = read(mcmc_manifest)
    if (historical.get("format") != "heston_spread_scale_capacity_report_v1"
            or mcmc.get("format") != "heston_spread_mcmc_recovery_report_v1"):
        raise ValueError("the frozen capacity and selected-MCMC manifests are required")
    old_protocol_path = Path(historical["protocol"]["path"])
    if fixed.digest(old_protocol_path) != historical["protocol"]["sha256"]:
        raise ValueError("historical protocol binding differs")
    if (binding(old_protocol_path) != study["previous_training_study"]
            or binding(mcmc_manifest) != study["matched_mcmc_report"]):
        raise ValueError("historical report arguments differ from the predeclared comparison")
    old = read(old_protocol_path)
    if (old["joint_task"] != study["joint_task"]
            or old["parent_checkpoint"] != study["parent_checkpoint"]
            or old["bank"] != study["bank"]):
        raise ValueError("the larger-batch comparison changed target, parent or bank")
    baseline_path = Path(study["baseline_evaluation"]["path"])
    baseline = read(baseline_path)
    if fixed.digest(baseline_path) != study["baseline_evaluation"]["sha256"]:
        raise ValueError("original60k population baseline differs")
    parent_counts = normalizer_counts(study["parent_checkpoint"]["path"])
    prepared = []
    for arm in study["arms"]:
        batch, payload, contract = arm["batch_size"], Path(arm["output"]), arm["contract"]
        per_pass = (100000+batch-1)//batch
        training = read(payload/"training.json")
        if (training["status"] != "complete" or training["completed_bank_passes"] != 500
                or contract["batches_per_pass"] != per_pass
                or training["continued_updates"] != 500*per_pass
                or training["completed_updates"] != 60000+500*per_pass
                or training["continued_row_presentations"] != 50000000
                or training["same_seed_reload_exact"] is not True
                or training["all_parent_array_hashes_restored"] is not True):
            raise ValueError("both new arms must complete the declared500-pass exposure budget")
        with np.load(payload/"exposure.npz", allow_pickle=False) as arrays:
            if (arrays["continuation_presentations"].shape != (100000,)
                    or arrays["continuation_presentations"].dtype != np.int64
                    or not np.all(arrays["continuation_presentations"] == 500)
                    or not np.array_equal(arrays["row_id"], np.array([f"train:{i:08d}" for i in range(100000)]))):
                raise ValueError("larger-batch training-row exposure counts differ")
        endpoints, checkpoints, evaluations = [], [], []
        for passes in (25, 100, 250, 500):
            checkpoint = payload/"checkpoints"/f"pass-{passes:04d}"
            metadata = read(checkpoint/"checkpoint.json")
            path = study_path.parent/"evaluations"/f"batch-{batch}"/f"pass-{passes:04d}"/"evaluation.json"
            record = read(path)
            if (metadata.get("format") != "heston_spread_scale_large_batch_checkpoint_v1"
                    or metadata["continuation_protocol"] != contract
                    or metadata["protocol_sha256"] != fixed.digest(study_path)
                    or fixed.digest(checkpoint/"model.keras") != metadata["model_sha256"]
                    or fixed.digest(checkpoint/"state.npz") != metadata["state_sha256"]
                    or record.get("format") != "heston_spread_scale_large_batch_evaluation_v1"
                    or record["status"] != "complete" or record["batch_size"] != batch
                    or record["actual_optimizer_updates"] != 60000+passes*per_pass
                    or metadata["actual_optimizer_updates"] != record["actual_optimizer_updates"]
                    or record["completed_bank_passes"] != passes or metadata["completed_bank_passes"] != passes
                    or record["family_checks"] != 384 or record["study_protocol"] != binding(study_path)
                    or record["checkpoint_record_sha256"] != fixed.digest(checkpoint/"checkpoint.json")
                    or record["task"] != study["joint_task"]
                    or record["population"]["truth_sha256"] != baseline["population"]["truth_sha256"]
                    or record["population"]["observations_sha256"] != baseline["population"]["observations_sha256"]
                    or fixed.digest(path.parent/"recovery.npz") != record["population"]["recovery_sha256"]):
                raise ValueError("new endpoint checkpoint or matched population binding differs")
            evaluations.append({"epochs": passes, "evaluation": binding(path), "recovery": binding(path.parent/"recovery.npz")})
            counters = normalizer_counts(checkpoint)
            logical_count = parent_counts["value"]+passes*100000
            counters.update(logical_parent_plus_examples=logical_count,
                difference_from_logical=counters["value"]-logical_count,
                relative_difference=(counters["value"]-logical_count)/logical_count)
            checkpoints.append({"bank_passes": passes, "record": binding(checkpoint/"checkpoint.json"),
                "model": binding(checkpoint/"model.keras"), "state": binding(checkpoint/"state.npz"),
                "standardizer_count": counters})
            recovery = record["population"]["recovery"]
            endpoints.append({"bank_passes": passes, "actual_optimizer_updates": record["actual_optimizer_updates"],
                "c_median_rmse": recovery["median_rmse"][-1],
                "c_truth_median_correlation": recovery["truth_posterior_median_correlation"][-1],
                "c_median_width90": recovery["median_posterior_width90"][-1],
                "marginal_sbc_screens_passed": int(sum(record["population"]["rank_dkw"]["passed"])),
                "synthetic_conditional_status": record["synthetic_conditional_status"],
                "market_conditional_status": record["cases"]["market"]["status"], "evaluation": binding(path)})
        prepared.append((arm, training, evaluations, checkpoints, endpoints))
    # Preserve every input identity before creating the compact report exports.
    arms, outcomes = [], []
    for arm, training, evaluations, checkpoints, endpoints in prepared:
        batch, payload = arm["batch_size"], Path(arm["output"])
        compact = study_path.parent/"training"/f"batch-{batch}"
        compact.mkdir(parents=True, exist_ok=True)
        compact_exports = {}
        for name in ("run.json", "training.json", "learning-curve.json", "progress.json",
                "initial-probes.json", "parent-probes.json"):
            source, target = payload/name, compact/name
            if target.exists():
                if fixed.digest(source) != fixed.digest(target):
                    raise ValueError("existing compact larger-batch export differs")
            else:
                shutil.copyfile(source, target)
            compact_exports[name] = {"source": binding(source), "copy": binding(target)}
        arms.append({"batch_size": batch, "run": binding(compact/"run.json"),
            "training": binding(compact/"training.json"), "curve": binding(compact/"learning-curve.json"),
            "evaluations": evaluations, "checkpoints": checkpoints, "exposure_counts": binding(payload/"exposure.npz"),
            "compact_exports": compact_exports})
        outcomes.append({"batch_size": batch, "training": training, "endpoints": endpoints})
    manifest = {"format": "heston_spread_scale_large_batch_report_v1", "interim": False,
        "protocol": binding(study_path), "old_capacity_manifest": binding(old_capacity_manifest),
        "mcmc_recovery_manifest": binding(mcmc_manifest), "arms": arms,
        "parent_standardizer_count": parent_counts,
        "sources": {"summary_producer": binding(__file__)},
        "comparison_scope": "Four shared-parent recipes at equal50m additional examples; old and new conditional checks retain separate384-check families."}
    manifest_path = out.parent/"report-manifest.json"
    fixed.write_json(manifest_path, manifest)
    result = {"format": "heston_spread_scale_large_batch_summary_v1", "status": "complete",
        "producer": binding(__file__), "protocol": binding(study_path),
        "report_manifest": binding(manifest_path), "arms": outcomes,
        "historical_comparisons": {"capacity": binding(old_capacity_manifest), "selected_mcmc": binding(mcmc_manifest)},
        "standardizer_count_scope": "Float32 running-counter arithmetic can depend on grouping. Exact int64 training-row counts remain50m presentations/500 per row; no counter or normalizer reset is performed.",
        "reserved_arrays_opened": False, "new_pricing_labels": False, "new_reference_sampling": False,
        "scientific_acceptance": False}
    fixed.write_json(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "old-capacity-manifest", "mcmc-manifest", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    print(build(args.study, args.old_capacity_manifest, args.mcmc_manifest, args.out)["report_manifest"])

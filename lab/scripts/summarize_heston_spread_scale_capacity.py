"""Collect settled paired runs and prepare a portable, hash-bound report manifest.

No fitting, pricing or posterior sampling. Only compact run records and named
row-exposure counts are copied; native models, states and draws stay ignored.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_heston_spread_baseline as fixed


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": fixed.digest(path)}


def build(study_path, out):
    study_path, out = Path(study_path).resolve(), Path(out).resolve()
    study = read(study_path)
    if study["format"] != "heston_spread_scale_precision_recovery_study_v1":
        raise ValueError("completed precision-repaired study required")
    if out.exists() or (out.parent/"report-manifest.json").exists():
        raise FileExistsError("summary/report manifest identity must be new")
    evidence, lab = study_path.parent, Path(__file__).resolve().parents[1]
    baseline_path = Path(study["baseline_evaluation"]["path"])
    baseline = read(baseline_path)
    if fixed.digest(baseline_path) != study["baseline_evaluation"]["sha256"]:
        raise ValueError("baseline evaluation changed")
    arms, outcomes, exports = [], [], []
    prepared = []
    for arm in study["arms"]:
        batch, payload = arm["batch_size"], Path(arm["output"])
        batches_per_pass = (100000+batch-1)//batch
        training = read(payload/"training.json")
        if (training["status"] != "complete" or training["completed_bank_passes"] != 500
                or arm["contract"]["batches_per_pass"] != batches_per_pass
                or training["continued_updates"] != 500*batches_per_pass
                or training["completed_updates"] != 60000+500*batches_per_pass
                or training["continued_row_presentations"] != 50000000
                or training["same_seed_reload_exact"] is not True
                or training["all_parent_array_hashes_restored"] is not True):
            raise ValueError("both declared 500-pass runs must complete before final reporting")
        with np.load(payload/"exposure.npz", allow_pickle=False) as exposure:
            counts = exposure["continuation_presentations"]
            if (counts.shape != (100000,) or not np.all(counts == 500)
                    or not np.array_equal(exposure["row_id"],
                        np.array([f"train:{i:08d}" for i in range(100000)]))):
                raise ValueError("100k physical-row exposure accounting differs")
        checkpoints, evaluations, arm_outcomes = [], [], []
        for passes in (25, 100, 250, 500):
            checkpoint = payload/"checkpoints"/f"pass-{passes:04d}"
            metadata = read(checkpoint/"checkpoint.json")
            if (fixed.digest(checkpoint/"model.keras") != metadata["model_sha256"]
                    or fixed.digest(checkpoint/"state.npz") != metadata["state_sha256"]):
                raise ValueError("native checkpoint payload changed")
            path = evidence/"evaluations"/f"batch-{batch}"/f"pass-{passes:04d}"/"evaluation.json"
            record = read(path)
            if (record["status"] != "complete" or record["batch_size"] != batch
                    or metadata["format"] != "heston_spread_scale_precision_checkpoint_v1"
                    or record["format"] != "heston_spread_scale_precision_evaluation_v1"
                    or record["actual_optimizer_updates"] != 60000+passes*batches_per_pass
                    or metadata["actual_optimizer_updates"] != record["actual_optimizer_updates"]
                    or metadata["completed_bank_passes"] != passes
                    or record["completed_bank_passes"] != passes or record["family_checks"] != 384
                    or record["checkpoint_record_sha256"] != fixed.digest(checkpoint/"checkpoint.json")
                    or record["study_protocol"] != binding(study_path)
                    or record["task"] != study["joint_task"]
                    or record["population"]["truth_sha256"] != baseline["population"]["truth_sha256"]
                    or record["population"]["observations_sha256"] != baseline["population"]["observations_sha256"]
                    or fixed.digest(path.parent/"recovery.npz") != record["population"]["recovery_sha256"]):
                raise ValueError("endpoint or matched development population changed")
            evaluations.append({"epochs": passes, "evaluation": binding(path),
                                "recovery": binding(path.parent/"recovery.npz")})
            checkpoints.append({"bank_passes": passes, "record": binding(checkpoint/"checkpoint.json"),
                "model": binding(checkpoint/"model.keras"), "state": binding(checkpoint/"state.npz")})
            recovery = record["population"]["recovery"]
            arm_outcomes.append({"bank_passes": passes, "actual_optimizer_updates": record["actual_optimizer_updates"],
                "c_median_rmse": recovery["median_rmse"][-1],
                "c_truth_median_correlation": recovery["truth_posterior_median_correlation"][-1],
                "c_median_width90": recovery["median_posterior_width90"][-1],
                "c_prior_width90": recovery["prior_width90"][-1],
                "parameter_median_rmse": recovery["median_rmse"],
                "parameter_truth_median_correlation": recovery["truth_posterior_median_correlation"],
                "marginal_sbc_screens_passed": int(sum(record["population"]["rank_dkw"]["passed"])),
                "synthetic_conditional_status": record["synthetic_conditional_status"],
                "market_conditional_status": record["cases"]["market"]["status"],
                "evaluation": binding(path)})
        prepared.append((batch, payload, training, evaluations, checkpoints, arm_outcomes))
    # All settled inputs are validated before creating any new compact export.
    for batch, payload, training, evaluations, checkpoints, arm_outcomes in prepared:
        compact = evidence/"training"/f"batch-{batch}"
        compact.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ("run.json", "training.json", "learning-curve.json", "progress.json",
                     "initial-probes.json", "parent-probes.json"):
            source, target = payload/name, compact/name
            if target.exists():
                if fixed.digest(target) != fixed.digest(source):
                    raise ValueError("existing compact export differs")
            else:
                shutil.copyfile(source, target)
            files[name] = {"source": binding(source), "copy": binding(target)}
        exports.append({"batch_size": batch, "files": files, "checkpoints": checkpoints,
                        "exposure_counts": binding(payload/"exposure.npz")})
        arms.append({"batch_size": batch, "run": binding(compact/"run.json"),
            "training": binding(compact/"training.json"), "curve": binding(compact/"learning-curve.json"),
            "evaluations": evaluations})
        outcomes.append({"batch_size": batch, "training": training, "endpoints": arm_outcomes})
    manifest = {"format": "heston_spread_scale_capacity_report_v1", "interim": False,
        "protocol": binding(study_path), "recovery": study["precision_recovery"],
        "baseline": {"evaluation": binding(baseline_path), "recovery": binding(baseline_path.parent/"recovery.npz")},
        "arms": arms, "original_arms": [{"batch_size": batch,
            "evaluation": binding(out.parent/"evaluations"/f"batch-{batch}"/"pass-0025"/"evaluation.json"),
            "recovery": binding(out.parent/"evaluations"/f"batch-{batch}"/"pass-0025"/"recovery.npz")}
            for batch in (128,512)],
        "sources": {key: binding(lab/"scripts"/name) for key, name in {
            "trainer": "continue_heston_spread_scale_precision.py", "parent_trainer": "train_heston_spread_scale.py",
            "fixed_helper": "train_heston_spread_baseline.py", "full_state_helper": "continue_heston_spread_baseline.py",
            "original_expansion": "continue_heston_spread_scale.py", "evaluator": "evaluate_heston_spread_scale_precision.py"}.items()},
        "input_conditioning": binding(out.parent/"diagnostics"/"input-conditioning.json"),
        "input_conditioning_source": binding(out.parent/"diagnostics"/"input_conditioning.py")}
    manifest_path = out.parent/"report-manifest.json"
    manifest["sources"]["summary_producer"] = binding(__file__)
    fixed.write_json(manifest_path, manifest)
    result = {"format": "heston_spread_scale_capacity_summary_v1", "status": "complete",
        "producer": binding(__file__), "protocol": binding(study_path),
        "report_manifest": binding(manifest_path), "arms": outcomes, "exports": exports,
        "comparison_scope": "One paired warm-start recipe at equal additional examples; no across-seed or factorial attribution.",
        "reserved_arrays_opened": False, "new_pricing_labels": False,
        "new_reference_sampling": False, "scientific_acceptance": False}
    fixed.write_json(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.study, args.out)
    print(json.dumps({"status": summary["status"], "report_manifest": summary["report_manifest"]}))

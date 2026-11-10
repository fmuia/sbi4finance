"""Freeze the new 1024/4098 experiment against existing Heston study identities."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import continue_heston_spread_scale_large_batch as trainer
from scripts import evaluate_heston_spread_scale_large_batch as evaluator
from scripts import run_heston_spread_scale_large_batch_campaign as controller


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": controller.shared.digest(path)}


def prepare(previous_path, evidence, payload, *, preflight):
    previous_path, evidence, payload, preflight = map(Path, (previous_path, evidence, payload, preflight))
    if evidence.exists() or payload.exists():
        raise FileExistsError("large-batch study identities must be new")
    previous = controller.shared.read(previous_path)
    if previous.get("format") != "heston_spread_scale_precision_recovery_study_v1":
        raise ValueError("completed precision-safe comparison must be bound")
    for source in previous["sources"].values():
        if controller.shared.digest(source["path"]) != source["sha256"]:
            raise ValueError("historical producer changed")
    check = controller.shared.read(preflight)
    if check.get("status") != "passed" or check.get("production_batch_sizes") != [1024,4098]:
        raise ValueError("actual requested batch-shape preflight must pass")
    sources = deepcopy(previous["sources"])
    sources.update(precision_helper=previous["sources"]["trainer"],
        previous_evaluator=previous["sources"]["evaluator"],
        trainer=identity(trainer.__file__), evaluator=identity(evaluator.__file__),
        controller=identity(controller.__file__), preparer=identity(__file__))
    parent, bank = previous["parent_checkpoint"], previous["bank"]
    arms = []
    for batch in controller.BATCH_SIZES:
        contract = trainer.protocol_contract(parent["path"], bank["path"], batch)
        arms.append({"batch_size": batch, "contract": contract,
            "output": str((payload / f"batch-{batch}").resolve()),
            "log": str((payload / f"batch-{batch}.log").resolve())})
    study = {key: deepcopy(previous[key]) for key in (
        "joint_task", "parent_checkpoint", "bank", "original_study", "reference_recovery",
        "baseline_evaluation", "precision_recovery")}
    study.update(format="heston_spread_scale_large_batch_study_v1", predeclared_before_fit=True,
        declared_utc=datetime.now(timezone.utc).isoformat(),
        authorized_request="One run with batch 1024 and one with exact batch 4098.",
        sources=sources, arms=arms, previous_training_study=identity(previous_path),
        native_batch_preflight={"path": str((evidence / "native-batch-preflight.json").resolve()),
            "sha256": controller.shared.digest(preflight)},
        evaluation=evaluator.evaluation_contract(study["joint_task"]),
        campaign={"worker_wall_seconds": 5400, "whole_training_wall_seconds": 5700,
            "max_concurrent_workers": 2, "automatic_retry": False, "automatic_extension": False},
        comparison={
            "primary_budget_axis": "500 complete 100k-bank passes = 50 million additional examples per arm",
            "pairing": "Same original 60k checkpoint, all Adam/normalization state, per-pass rows, c and Gaussian errors as previous 128/512 runs",
            "batch_sizes": [1024,4098], "additional_optimizer_updates": [49000,12500],
            "learning_rate": "Same 5e-5 restart, cosine by examples to 5e-6 over 400 passes, then hold through 500",
            "architecture": "Unchanged six-block spline flow, conditioner widths [128,128], effective dropout zero",
            "scope": "Batch grouping changes update count at equal examples; one warm start and augmentation stream, not an across-seed experiment"},
        matched_mcmc_report=identity(evidence.parent / "mcmc-recovery/report-manifest.json"),
        reserved_arrays_opened=False, new_pricing_labels=False, new_reference_sampling=False,
        scientific_acceptance=False, automatic_extension=False,
        stop_after="Complete the two runs and diagnostics, then reflect before grid or Bergomi work")
    controller.validate(study)
    evaluator.validate_study(study, study["joint_task"])
    evidence.mkdir(parents=True); payload.mkdir(parents=True)
    (evidence / "native-batch-preflight.json").write_bytes(preflight.read_bytes())
    path = evidence / "protocol.json"
    controller.shared.write(path, study)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "evidence", "payload", "preflight"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.previous, args.evidence, args.payload, preflight=args.preflight))

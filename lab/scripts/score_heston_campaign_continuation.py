"""Physical-density scores for the authorized 75k/100k update continuation.

Read the same saved512 development observations and frozen checkpoints. Reuse
the recorded100k-row/50k-update physical log densities as the paired baseline.
No fitting, simulation, original-record modification or reserved-test reads.
"""

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
import time

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "src"))

import numpy as np

from scripts.evaluate_heston_pilot_campaign import digest, fingerprint, values_digest
from scripts.run_heston_observation_pilot import write_json
from scripts.score_heston_campaign_density import paired_difference
from scripts.train_heston_pilot import training_data
from scripts.train_heston_pilot_campaign import load_checkpoint


def run(args):
    if args.out.exists() or args.payload.exists():
        raise FileExistsError("continuation density score outputs must be new")
    design = json.loads(args.design.read_text())
    baseline = json.loads(args.baseline_score.read_text())
    original = baseline["cells"]["n100000-u050000"]
    if baseline["design_sha256"] != digest(args.design):
        raise ValueError("baseline density score and frozen design differ")
    original_payload = Path(baseline["per_observation_logp_payload_path"])
    if digest(original_payload) != baseline["per_observation_logp_payload_sha256"]:
        raise ValueError("baseline per-observation physical densities changed")
    with np.load(original_payload, allow_pickle=False) as archive:
        original_logp = archive["n100000-u050000"]
        original_ids = archive["validation_row_id"]
    logp = {50000: original_logp}
    report = {"format": "heston_continuation_physical_density_development_score_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "scientific_acceptance": False,
              "scope": __doc__, "method": "negative public BasicEstimator.log_prob on original training_data; adapter and target-standardization Jacobians included",
              "units": "joint physical-parameter NLL, nats per observation; lower is better",
              "script_sha256": digest(__file__), "design_sha256": digest(args.design),
              "baseline_score_sha256": digest(args.baseline_score), "baseline_cell": original,
              "baseline_logp_payload_sha256": digest(original_payload), "cells": {}}
    started = time.perf_counter()
    for updates, checkpoint, evaluation_directory in args.cell:
        updates, checkpoint, evaluation_directory = int(updates), Path(checkpoint), Path(evaluation_directory)
        if updates not in (75000, 100000) or updates in logp:
            raise ValueError("distinct declared75k/100k continuation endpoints required")
        evaluation_path = evaluation_directory / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        if digest(evaluation_directory / "draws.npz") != evaluation["draws_sha256"]:
            raise ValueError("continuation evaluation payload changed")
        with np.load(evaluation_directory / "draws.npz", allow_pickle=False) as archive:
            truth, observed, ids = (archive[key] for key in ("validation_truth", "validation_observations", "validation_row_id"))
        if (truth.shape != (512, 6) or observed.shape != (512, 54) or not np.array_equal(ids, original_ids)
                or values_digest(truth) != original["truth_sha256"]
                or values_digest(observed) != original["observations_sha256"]
                or fingerprint(ids.tolist()) != original["row_ids_sha256"]
                or evaluation["cohort_sha256"] != original["cohort_sha256"]):
            raise ValueError("continuation scoring must reuse the exact512 original truths, observations and IDs")
        workflow, record = load_checkpoint(checkpoint)
        old = evaluation["checkpoint"]
        if (record["model_sha256"] != old["model_sha256"]
                or record["task_fingerprint"] != original["task_fingerprint"]
                or record["task_fingerprint"] != old["task_fingerprint"]
                or record["design_sha256"] != report["design_sha256"]
                or record["actual_optimizer_updates"] != updates
                or record["train_rows"] != 100000 or record["schedule_updates"] != 50000):
            raise ValueError("continuation checkpoint, schedule, task or evaluation identity differs")
        raw = training_data(observed, np.ones(512), truth, design["records"], np.ones(54, bool), record["names"])
        values = np.asarray(workflow.estimator.log_prob(raw))
        if values.size != 512 or not np.isfinite(values).all():
            raise ValueError("finite physical log density required for each saved development observation")
        values = values.reshape(512).astype(float)
        logp[updates] = values
        report["cells"][str(updates)] = {"mean_physical_nll": float(-values.mean()),
            "observation_standard_error": float(values.std(ddof=1) / np.sqrt(512)),
            "checkpoint_record_sha256": digest(checkpoint / "checkpoint.json"), "model_sha256": record["model_sha256"],
            "task_fingerprint": record["task_fingerprint"], "actual_optimizer_updates": updates,
            "planned_updates": record["planned_updates"], "schedule_updates": record["schedule_updates"],
            "evaluation_record_sha256": digest(evaluation_path), "evaluation_payload_sha256": evaluation["draws_sha256"],
            "truth_sha256": values_digest(truth), "observations_sha256": values_digest(observed),
            "row_ids_sha256": fingerprint(ids.tolist()),
            "official_basic_estimator_source_sha256": digest(inspect.getfile(type(workflow.estimator)))}
        print(json.dumps({"updates": updates, "mean_physical_nll": float(-values.mean())}), flush=True)
    args.payload.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.payload, **{f"u{updates:06d}": values for updates, values in logp.items()}, validation_row_id=original_ids)
    report["paired_later_minus_50k"] = {str(updates): paired_difference(-logp[50000], -values)
                                           for updates, values in logp.items() if updates != 50000}
    if 75000 in logp and 100000 in logp:
        report["paired_100k_minus_75k"] = paired_difference(-logp[75000], -logp[100000])
    report["logp_payload_sha256"] = digest(args.payload)
    report["logp_payload_path"] = str(args.payload.resolve())
    report["wall_seconds"] = time.perf_counter() - started
    write_json(args.out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design", "baseline-score", "out", "payload"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cell", nargs=3, action="append", required=True,
                        metavar=("UPDATES", "CHECKPOINT", "EVALUATION_DIRECTORY"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()

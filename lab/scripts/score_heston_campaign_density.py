"""Comparable physical-coordinate NLL on the saved development observations.

Read-only scoring of frozen checkpoints and saved evaluation truths/quotes.
No simulation, fitting or reserved-test member reads occur. BayesFlow's public
BasicEstimator.log_prob applies adapter and target-standardization Jacobians.
"""

import argparse
from datetime import datetime, timezone
from importlib import metadata
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
from scripts.train_heston_pilot import training_data
from scripts.train_heston_pilot_campaign import load_checkpoint


def paired_difference(first, second):
    """Second-minus-first NLL; negative favors the second frozen model."""
    delta = np.asarray(second) - np.asarray(first)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("paired finite per-observation NLL vectors required")
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {"mean_nll_difference": mean, "paired_observation_standard_error": se,
            "normal_approximation_ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "cases": len(delta), "negative_favors": "second model", "scientific_acceptance": False}


def run(args):
    design = json.loads(args.design.read_text())
    design_hash = digest(args.design)
    records = design["records"]
    mask = np.ones(len(records["mid"]), dtype=bool)
    baseline = None
    nll = {}
    output = {"format": "heston_campaign_physical_density_development_score_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "scientific_acceptance": False,
              "scope": __doc__, "method": "negative BasicEstimator.log_prob(original training_data dict), including constrain-adapter and learned target-standardization log Jacobians",
              "units": "nats per observation; joint density in the six declared physical parameter coordinates",
              "paired_uncertainty": "Direct standard error across the same512 saved validation observations; approximate normal interval, conditional on these fixed models. Development comparison, no acceptance claim.",
              "design_sha256": design_hash, "script_sha256": digest(__file__), "cells": {}}
    started = time.perf_counter()
    for rows in (20000, 100000):
        for updates in (10000, 50000):
            name = f"n{rows:06d}-u{updates:06d}"
            directory = args.root / "evaluations" / name
            evaluation_path = directory / "evaluation.json"
            evaluation = json.loads(evaluation_path.read_text())
            if digest(directory / "draws.npz") != evaluation["draws_sha256"]:
                raise ValueError("evaluation payload differs from its recorded hash")
            with np.load(directory / "draws.npz", allow_pickle=False) as archive:
                truth = archive["validation_truth"]
                observed = archive["validation_observations"]
                ids = archive["validation_row_id"]
            if truth.shape != (512, 6) or observed.shape != (512, 54) or len(ids) != 512:
                raise ValueError("exactly the declared512 saved development observations required")
            if baseline is None:
                baseline = (truth.copy(), observed.copy(), ids.copy(), evaluation["cohort_sha256"])
            elif (not np.array_equal(truth, baseline[0]) or not np.array_equal(observed, baseline[1])
                  or not np.array_equal(ids, baseline[2]) or evaluation["cohort_sha256"] != baseline[3]):
                raise ValueError("four physical-density comparisons must use identical saved truths, observations and row IDs")
            checkpoint = args.root / f"train-n{rows:06d}"
            if updates == 10000:
                checkpoint = checkpoint / "checkpoints/update-010000"
            workflow, checkpoint_record = load_checkpoint(checkpoint)
            old = evaluation["checkpoint"]
            if (checkpoint_record["model_sha256"] != old["model_sha256"]
                    or checkpoint_record["task_fingerprint"] != old["task_fingerprint"]
                    or checkpoint_record["names"] != old["names"] or checkpoint_record["bounds"] != old["bounds"]
                    or checkpoint_record["design_sha256"] != design_hash
                    or checkpoint_record["actual_optimizer_updates"] != updates
                    or checkpoint_record["train_rows"] != rows):
                raise ValueError("checkpoint differs from the scored evaluation's model/task/endpoint")
            raw = training_data(observed, np.ones(len(truth)), truth, records, mask, checkpoint_record["names"])
            logp = np.asarray(workflow.estimator.log_prob(raw))
            if logp.size != len(truth) or not np.isfinite(logp).all():
                raise ValueError("finite physical log density required for every saved observation")
            logp = logp.reshape(len(truth)).astype(float)
            nll[name] = -logp
            estimator_source = Path(inspect.getfile(type(workflow.estimator)))
            output["cells"][name] = {"mean_physical_nll": float(-logp.mean()),
                "observation_standard_error": float(logp.std(ddof=1) / np.sqrt(len(logp))),
                "physical_logp_min_max": [float(logp.min()), float(logp.max())],
                "checkpoint_record_sha256": digest(checkpoint / "checkpoint.json"),
                "model_sha256": checkpoint_record["model_sha256"],
                "task_fingerprint": checkpoint_record["task_fingerprint"],
                "evaluation_record_sha256": digest(evaluation_path), "evaluation_payload_sha256": evaluation["draws_sha256"],
                "cohort_sha256": evaluation["cohort_sha256"], "truth_sha256": values_digest(truth),
                "observations_sha256": values_digest(observed), "row_ids_sha256": fingerprint(ids.tolist()),
                "official_basic_estimator_source_sha256": digest(estimator_source),
                "actual_optimizer_updates": updates, "training_rows": rows}
            print(json.dumps({"cell": name, "mean_physical_nll": float(-logp.mean())}), flush=True)
    args.payload.parent.mkdir(parents=True, exist_ok=True)
    if args.payload.exists() or args.out.exists():
        raise FileExistsError("physical density score outputs must be new")
    np.savez_compressed(args.payload, **{name: -value for name, value in nll.items()}, validation_row_id=baseline[2])
    output["paired_100k_minus_20k"] = {str(updates): paired_difference(
        nll[f"n020000-u{updates:06d}"], nll[f"n100000-u{updates:06d}"]) for updates in (10000, 50000)}
    output["per_observation_logp_payload_sha256"] = digest(args.payload)
    output["per_observation_logp_payload_path"] = str(args.payload.resolve())
    output["versions"] = {name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy")}
    output["source_sha256"] = {"checkpoint_loader": digest(Path(inspect.getfile(load_checkpoint))),
                               "training_data": digest(Path(inspect.getfile(training_data)))}
    output["wall_seconds"] = time.perf_counter() - started
    write_json(args.out, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

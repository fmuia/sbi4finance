"""Versioned 100k PR87 FlowMatching diagnostics on unchanged development panels.

The preserved 100-case/300-draw native validation recovery runs first. The second panel
reuses the exact historical 512 raw quote/truth rows, and the final three cases
reuse the existing joint MCMC chains and posterior draw seeds. Raw unbounded
FlowMatching draws are never clipped, filtered or resampled for support.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_spread_scale as previous
from scripts import evaluate_heston_spread_whitening as whitening

fixed, baseline, validation = previous.fixed, previous.baseline, previous.validation
SEED, ROWS, DRAWS, CONDITIONAL_DRAWS = previous.SEED, previous.ROWS, previous.DRAWS, previous.CONDITIONAL_DRAWS
NATIVE_ROWS, NATIVE_DRAWS, NATIVE_SEED = 100, 300, 9011 + 700001
WALL_SECONDS = 900


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": fixed.digest(path)}


def evaluation_contract(target):
    whitening.evaluation_contract(target)
    return {"native_validation_rows": NATIVE_ROWS, "native_validation_draws": NATIVE_DRAWS,
        "native_sampling_seed": NATIVE_SEED, "historical_population_rows": ROWS,
        "historical_draws_per_case": DRAWS, "historical_seed": SEED,
        "historical_population_batch_rows": 16,
        "historical_population_sampling_seed": "261112 + 1000 + zero_based_batch_start",
        "historical_population_sbc_seed": SEED + 2000,
        "conditional_cases": list(previous.CASES), "conditional_draws": CONDITIONAL_DRAWS,
        "conditional_sampling_seed": "261112 + zero_based_case_index",
        "order": ["native_validation", "historical_population", "historical_conditionals"],
        "raw_draw_policy": "retain every raw unbounded draw; no clipping, filtering, rejection or resampling",
        "support": "per-sample and per-dimension outside-prior, nonphysical and nonfinite counts",
        "calibration": "all raw finite draws; if any nonfinite draw, report the whole panel's calibration undefined",
        "checkpoint_selection": "primary final epoch200; one predeclared minimum-validation secondary only if its epoch differs; no other posterior queries",
        "sbc_scope": "six marginal DKW screens per endpoint and panel; descriptive development evidence, no simultaneous cross-panel or endpoint acceptance",
        "max_posterior_evaluation_seconds_per_endpoint": WALL_SECONDS,
        "primary_train_rows": 100000, "epochs": 200, "primary_presentations": 20000000,
        "parent_presentations": 2000000, "equal_compute_comparison": False,
        "secondary_selection": "native minimum validation loss, predeclared; evaluate only when distinct from epoch200",
        "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False}


def checked_binding(spec, protected):
    if not {"path", "sha256"} <= set(spec) or binding(spec["path"]) != {key: spec[key] for key in ("path", "sha256")}:
        raise ValueError("declared input identity differs")
    protected[str(Path(spec["path"]).resolve())] = spec["sha256"]
    return Path(spec["path"])


# Reuse pure raw-draw diagnostics and checked historical-input readers without
# changing the archived evaluator or replacing its trainer at runtime.
from scripts import evaluate_heston_pr87_reproduction as parent_evaluation
raw_support = parent_evaluation.raw_support
recovery_summary = parent_evaluation.recovery_summary
population_summary = parent_evaluation.population_summary
dependence = parent_evaluation.dependence
descriptive_comparison = parent_evaluation.descriptive_comparison
historical_inputs = parent_evaluation.historical_inputs


def checkpoint_endpoint(checkpoint, metadata, run):
    """Only the frozen final endpoint and distinct predeclared minimum are legal."""
    checkpoint = Path(checkpoint).resolve()
    epoch = metadata.get("native_epoch")
    contract = run.get("protocol", {})
    if (run.get("format") != "heston_pr87_100k_run_v1" or run.get("status") != "complete"
            or run.get("completed_epochs") != 200 or run.get("actual_optimizer_updates") != 78200
            or contract.get("train_rows") != 100000 or contract.get("validation_rows") != 100
            or contract.get("epochs") != 200 or contract.get("batch_size") != 256
            or type(epoch) is not int or not 1 <= epoch <= 200
            or metadata.get("actual_optimizer_updates") != epoch * 391):
        raise ValueError("completed100k/200epoch native training accounting required")
    primary = Path(run["primary_checkpoint"]).resolve()
    if checkpoint == primary and epoch == 200:
        return "primary_final"
    best = run.get("best_validation_archive", {})
    if (epoch != 200 and best.get("epoch") == epoch and best.get("path")
            and checkpoint == Path(best["path"]).resolve()):
        return "secondary_best_validation"
    raise ValueError("only primary final or distinct predeclared best-validation endpoint is permitted")


def evaluate(checkpoint, protocol_path, out, payload_out, *, max_seconds=WALL_SECONDS):
    started = time.monotonic()
    checkpoint, protocol_path, out, payload_out = map(Path, (checkpoint, protocol_path, out, payload_out))
    if out.exists() or payload_out.exists():
        raise FileExistsError("100k PR87 evaluation output identities must be new")
    protocol, protected = read(protocol_path), {}
    if not 0 < max_seconds <= WALL_SECONDS:
        raise ValueError("posterior evaluation budget must be positive and at most900 seconds")
    from scripts import reproduce_heston_pr87_100k as trainer
    workflow, metadata = trainer.load_checkpoint(checkpoint)
    task, target = metadata["task"], metadata["task"]["target"]
    if (metadata.get("format") != "heston_pr87_100k_checkpoint_v1"
            or protocol.get("evaluation") != evaluation_contract(target) or protocol.get("task") != task):
        raise ValueError("100k PR87 evaluation protocol or canonical task differs")
    if (protocol.get("sources", {}).get("evaluator") != binding(__file__)
            or protocol["sources"].get("parent_evaluator") != binding(parent_evaluation.__file__)):
        raise ValueError("executed PR87 evaluator was not frozen before fit")
    for spec in protocol["sources"].values():
        checked_binding(spec, protected)
    checked_binding(binding(protocol_path), protected)
    if (Path(metadata["protocol_path"]).resolve() != protocol_path.resolve()
            or metadata["protocol_sha256"] != fixed.digest(protocol_path)):
        raise ValueError("checkpoint training protocol differs from evaluation protocol")
    training_dir = checkpoint.parent.parent
    run_path = training_dir / "run.json"
    run = read(run_path)
    endpoint = checkpoint_endpoint(checkpoint, metadata, run)
    parent_path = checked_binding(protocol["parent_10k_evaluation"], protected)
    parent = read(parent_path)
    if (parent.get("format") != "heston_pr87_reproduction_evaluation_v1"
            or parent.get("status") != "complete" or parent.get("task") != target
            or parent.get("native_epoch") != 200):
        raise ValueError("preserved complete 10k parent evaluation required")
    data_path = training_dir / run["data_payload"]
    if (run.get("status") != "complete" or run["task"] != task or data_path.resolve().parent != training_dir.resolve()
            or fixed.digest(data_path) != run["data_payload_sha256"]):
        raise ValueError("native validation dataset binding differs")
    protected[str(data_path.resolve())] = run["data_payload_sha256"]
    protected[str(run_path.resolve())] = fixed.digest(run_path)
    for name in ("checkpoint.json", "model.keras", "state.npz", "whitening.json"):
        path = checkpoint / name
        if path.is_file(): protected[str(path.resolve())] = fixed.digest(path)
    quotes, truth, row_ids, old_study, panel_path, panel, ref_protocol = historical_inputs(protocol, task, protected)
    with np.load(data_path, allow_pickle=False) as archive:
        native_quotes = archive["validation_quotes"].copy()
        native_observations = archive["validation_observations"].copy()
        native_truth = np.column_stack([archive["validation_" + name].reshape(-1) for name in target["parameter_names"]])
    reconstructed_observations = native_quotes / np.asarray(task["records"]["half_spread"])
    if (native_quotes.shape != (NATIVE_ROWS, 54) or native_truth.shape != (NATIVE_ROWS, 6)
            or not np.array_equal(native_observations, reconstructed_observations)
            or baseline.observations_digest(native_quotes) != parent["native_validation"]["observations_sha256"]
            or baseline.observations_digest(native_truth) != parent["native_validation"]["truth_sha256"]):
        raise ValueError("notebook native100-row validation panel or exact condition reconstruction differs")
    out.mkdir(parents=True); payload_out.mkdir(parents=True)
    record = {"format": "heston_pr87_100k_evaluation_v1", "status": "started", "task": target,
        "checkpoint": binding(checkpoint / "checkpoint.json"), "native_epoch": metadata["native_epoch"],
        "endpoint": endpoint, "actual_optimizer_updates": metadata["actual_optimizer_updates"],
        "training_rows": 100000, "training_presentations": metadata["native_epoch"] * 100000,
        "parent_10k_evaluation": protocol["parent_10k_evaluation"],
        "training_run": binding(run_path),
        "protocol": binding(protocol_path), "seed": SEED, "native_seed": NATIVE_SEED,
        "historical_population_inputs": protocol["historical_population_inputs"],
        "original_whitening_study": protocol["original_whitening_study"],
        "matched_mcmc_report": old_study["matched_mcmc_report"],
        "native_data_payload": binding(data_path),
        "native_input_replication": {"observations_dtype": str(native_observations.dtype),
            "reconstructed_observations_dtype": str(reconstructed_observations.dtype),
            "raw_quote_wrapper_conditions_array_equal_to_native": True,
            "native_observations": fixed.array_summary(native_observations)},
        "source_sha256": {"evaluator": binding(__file__), "trainer": binding(trainer.__file__)},
        "cases": {}, "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False,
        "scope": "100k data extension: unchanged 10k training prefix, same100 validation rows and frozen parent whitening; 20m primary presentations versus2m parent, not equal compute",
        "loss_interpretation": "native FlowMatching objective, not density NLL; absolute values are not comparable with spline losses",
        "max_seconds": max_seconds}
    fixed.write_json(out / "run.json", record)
    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("bounded 100k PR87 posterior evaluation wall budget exhausted")
    def query(raw_quotes, count, seed):
        deadline()
        result = np.asarray(trainer.sample(workflow, task, raw_quotes, count, seed=seed))
        deadline()
        if result.shape != (len(raw_quotes), count, 6):
            raise ValueError("raw six-target FlowMatching posterior shape differs")
        return result
    def population(label, raw_quotes, truths, ids, count, sampling_seed, sbc_seed, batch_rows):
        draws = np.concatenate([query(raw_quotes[start:start + batch_rows], count, sampling_seed + start)
            for start in range(0, len(raw_quotes), batch_rows)])
        raw_path = payload_out / (label + "-draws.npz")
        fixed.write_npz(raw_path, {"draws": draws})
        summary, quantiles = population_summary(draws, truths, raw_quotes, target, sbc_seed)
        summary["raw_draw_payload"] = binding(raw_path)
        if quantiles is not None:
            recovery_path = out / ("recovery.npz" if label == "population" else "native-recovery.npz")
            fixed.write_npz(recovery_path, {"truth": truths, "quantiles": quantiles.astype(np.float32), "row_id": ids})
            summary["recovery_payload"] = binding(recovery_path)
        record[label] = summary
        fixed.write_json(out / "run.json", record)
    try:
        # Same single100-row/300-draw notebook recovery call, before old panels.
        population("native_validation", native_quotes, native_truth,
            np.array([f"native-validation:{i:08d}" for i in range(NATIVE_ROWS)]),
            NATIVE_DRAWS, NATIVE_SEED, NATIVE_SEED + 2000, NATIVE_ROWS)
        population("population", quotes, truth, row_ids, DRAWS, SEED + 1000, SEED + 2000, 16)
        conditional = {}
        for index, case_id in enumerate(previous.CASES):
            deadline()
            _, _, _, observation = previous.reference.load_case(panel_path, case_id, infer_c=True)
            draws = query(observation[None], CONDITIONAL_DRAWS, SEED + index)[0]
            conditional[case_id] = draws
            selected, reference_binding = previous.joint_reference(ref_protocol, case_id, panel_path, target, panel["products"], observation)
            entry = {"observation_sha256": baseline.observations_digest(observation),
                "support": raw_support(draws, target["prior_bounds"], target["parameter_names"]), **reference_binding}
            for prefix in ("reference_run", "reference_diagnostics", "reference_chain_payload"):
                if prefix + "_path" in reference_binding:
                    protected[reference_binding[prefix + "_path"]] = reference_binding[prefix + "_sha256"]
            if selected is not None:
                chains, _ = selected
                entry["comparison"] = descriptive_comparison(draws, chains, target["parameter_names"], panel["products"], target["prior_bounds"])
            record["cases"][case_id] = entry
            fixed.write_json(out / "run.json", record)
        path = payload_out / "conditional-draws.npz"
        fixed.write_npz(path, conditional)
        record["conditional_draws"] = {**binding(path), "draws_per_case": CONDITIONAL_DRAWS}
        if any(fixed.digest(path) != digest for path, digest in protected.items()):
            raise ValueError("protected historical input, checkpoint, data or source changed")
        deadline()
        record.update(status="complete", seconds=time.monotonic() - started, protected_inputs_unchanged=True)
        fixed.write_json(out / "evaluation.json", record); fixed.write_json(out / "run.json", record)
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic() - started)
        fixed.write_json(out / "failure.json", record); fixed.write_json(out / "run.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "protocol", "out", "payload-out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=WALL_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= WALL_SECONDS: parser.error("wall-seconds must be in(0,900]")
    if not args.worker:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        try: return subprocess.run(command, timeout=args.wall_seconds, check=False).returncode
        except subprocess.TimeoutExpired:
            print("100k PR87 posterior evaluation stopped at its hard wall limit", file=sys.stderr)
            return 124
    result = evaluate(args.checkpoint, args.protocol, args.out, args.payload_out, max_seconds=args.wall_seconds)
    print(json.dumps({"status": result["status"], "epoch": result["native_epoch"], "seconds": result["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

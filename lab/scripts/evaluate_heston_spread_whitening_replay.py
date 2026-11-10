"""Exploratory diagnostics for exact-prefix ZCA replay checkpoints.

The historical population, seeds, conditional references and financial target
are unchanged. The minimum validation epoch was already seen, so these checks
are development evidence rather than an independent acceptance experiment.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_spread_scale as previous
from scripts import compare_heston_spread_scale as saved
from scripts import evaluate_heston_spread_scale_continuation as historical
from scripts import train_heston_spread_whitening as frozen
from scripts import evaluate_heston_spread_whitening as original
from scripts import replay_heston_spread_whitening as trainer

fixed, baseline = previous.fixed, previous.baseline
CHECKPOINT_UPDATES = (2000, 3000, 3320, 4000)
SEED, ROWS, DRAWS, CONDITIONAL_DRAWS = previous.SEED, previous.ROWS, previous.DRAWS, previous.CONDITIONAL_DRAWS
WALL_SECONDS = 300


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": fixed.digest(path)}


def evaluation_contract(target):
    original.evaluation_contract(target)
    return {"mode": "zca", "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "reused_final_updates": 8000, "seed": SEED,
        "population_rows": ROWS, "posterior_draws_per_population_case": DRAWS,
        "conditional_cases": list(previous.CASES), "conditional_draws": CONDITIONAL_DRAWS,
        "population_c": "reuse draw_c(512,0,261112,namespace=3); c is never a condition",
        "population_gaussian": "reuse PCG64 SeedSequence([0x53504556,261112]) standard errors",
        "preprocessing": "load the original training-only frozen sidecar and apply exactly once; never refit",
        "conditional_metrics": "descriptive absolute mean error / reference SD and central90% width ratio; no pass/fail family",
        "sbc_scope": "six full-prior marginal DKW screens per checkpoint; exploratory across checkpoints",
        "checkpoint_selection": "epochs50/75/83/100 fixed before replay; original minimum83 already observed, posthoc development comparison",
        "primary_quantity": "c", "max_wall_seconds": WALL_SECONDS,
        "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False}


def validate_checkpoint(metadata, study, study_path, checkpoint, replay, replay_path):
    original.validate_study(study, metadata["task"]["target"])
    if replay.get("original_study") != binding(study_path):
        raise ValueError("replay original whitening study binding differs")
    for name, source in replay.get("sources", {}).items():
        if fixed.digest(source["path"]) != source["sha256"]:
            raise ValueError("replay declared source differs: "+name)
    if replay.get("sources", {}).get("evaluator") != binding(__file__):
        raise ValueError("replay evaluator was not frozen in the training protocol")
    contract = replay.get("contract", {})
    if (replay.get("format") != "heston_spread_whitening_checkpoint_replay_study_v1"
            or metadata.get("format") != "heston_spread_whitening_checkpoint_replay_checkpoint_v1"
            or metadata.get("mode") != "zca" or metadata.get("training_protocol") != contract
            or metadata.get("actual_optimizer_updates") not in CHECKPOINT_UPDATES
            or contract.get("planned_updates") != 4000 or contract.get("schedule_updates") != 8000
            or contract.get("checkpoint_updates") != list(CHECKPOINT_UPDATES)
            or contract.get("train_rows") != 10000 or contract.get("batch_size") != 256
            or contract.get("standardize") != ["inference_variables"]
            or contract.get("architecture") != {"family": "spline", "depth": 6, "widths": [128, 128], "dropout": 0.}
            or Path(metadata["protocol_path"]).resolve() != Path(replay_path).resolve()
            or metadata["protocol_sha256"] != fixed.digest(replay_path)
            or metadata.get("reserved_arrays_opened") is not False
            or fixed.digest(Path(checkpoint)/"preprocessing.json") != metadata["preprocessing_sha256"]):
        raise ValueError("replay checkpoint, schedule, sidecar or development-data boundary differs")
    if replay.get("evaluation") != evaluation_contract(metadata["task"]["target"]):
        raise ValueError("replay evaluation declaration differs")


preprocessing_summary = original.preprocessing_summary
descriptive_comparison = original.descriptive_comparison


def evaluate(checkpoint, study_path, replay_path, out, payload_out):
    started = time.monotonic()
    checkpoint, study_path, replay_path, out, payload_out = map(Path, (checkpoint, study_path, replay_path, out, payload_out))
    if out.exists() or payload_out.exists():
        raise FileExistsError("whitening evaluation identities must be new")
    study, replay = read(study_path), read(replay_path)
    workflow, metadata = trainer.load_checkpoint(checkpoint)
    task, target = metadata["task"], metadata["task"]["target"]
    validate_checkpoint(metadata, study, study_path, checkpoint, replay, replay_path)
    preprocessing_path = checkpoint/"preprocessing.json"
    preprocessing = read(preprocessing_path)
    old_study_path = Path(study["original_study"]["path"])
    old_study = read(old_study_path)
    previous.validate_study(old_study, target)
    recovery_path = Path(study["reference_recovery"]["path"])
    ref_path, protected = previous.selected_reference_protocol(old_study, old_study_path, recovery_path)
    ref_protocol = read(ref_path)
    recovery = read(recovery_path)
    if read(recovery["progress_path"])["status"] != "completed":
        raise ValueError("the existing reference campaign must already be completed")
    previous.scale_reference.validate_protocol(ref_protocol, ref_protocol["jobs"][0]["id"])
    panel_path = Path(old_study["panel"]["path"])
    if fixed.digest(panel_path) != old_study["panel"]["sha256"]:
        raise ValueError("historical conditional panel differs")
    panel = read(panel_path)
    if panel["joint_task"] != target:
        raise ValueError("existing references target a different posterior")
    bank = Path(metadata["bank_path"])
    bank_record = read(bank/"bank.json")
    payload = bank/bank_record["payload"]
    if (payload.resolve().parent != bank.resolve() or fixed.digest(bank/"bank.json") != metadata["bank_record_sha256"]
            or bank_record["identity"] != metadata["bank_identity"]
            or bank_record["payload_sha256"] != metadata["bank_payload_sha256"]
            or fixed.digest(payload) != metadata["bank_payload_sha256"]):
        raise ValueError("whitening validation bank binding differs")
    sources = {**historical.source_bindings(trainer), "evaluator": binding(__file__),
        "whitener": frozen.sources()["whitening"],
        "original_evaluator": binding(original.__file__), "original_trainer": binding(frozen.__file__)}
    saved.verify_artifacts(sources, protected)
    saved.verify_artifacts({key: study[key] for key in
        ("original_study", "reference_recovery", "baseline_evaluation", "matched_mcmc_report")}, protected)
    for path in (study_path, replay_path, panel_path, ref_path, checkpoint/"checkpoint.json", checkpoint/"model.keras",
            checkpoint/"state.npz", preprocessing_path, Path(metadata["preprocessing_path"]), bank/"bank.json", payload):
        protected[str(path.resolve())] = fixed.digest(path)
    baseline_record = read(study["baseline_evaluation"]["path"])
    if baseline_record["task"] != target:
        raise ValueError("old population baseline uses a different statistical task")
    out.mkdir(parents=True); payload_out.mkdir(parents=True)
    record = {"format": "heston_spread_whitening_checkpoint_replay_evaluation_v1", "status": "started", "mode": metadata["mode"],
        "task": target, "task_fingerprint": metadata["task_fingerprint"], "actual_optimizer_updates": metadata["actual_optimizer_updates"],
        "checkpoint": str(checkpoint.resolve()), "checkpoint_record_sha256": fixed.digest(checkpoint/"checkpoint.json"),
        "model_sha256": metadata["model_sha256"], "state_sha256": metadata["state_sha256"],
        "preprocessing": binding(preprocessing_path), "preprocessing_summary": preprocessing_summary(preprocessing),
        "training_dataset_bindings": metadata["splits"], "training_recipe": metadata["training_protocol"],
        "study_protocol": binding(study_path), "replay_protocol": binding(replay_path), "reference_protocol": binding(ref_path),
        "conditional_panel": binding(panel_path), "matched_mcmc_report": study["matched_mcmc_report"],
        "source_sha256": sources, "seed": SEED, "cases": {}, "reserved_arrays_opened": False,
        "new_reference_sampling": False, "scientific_acceptance": False,
        "scope": "Posthoc exact-prefix checkpoint diagnosis; descriptive development evidence, no scientific acceptance or independent checkpoint confirmation."}
    fixed.write_json(out/"run.json", record)
    def deadline():
        if time.monotonic()-started >= WALL_SECONDS:
            raise TimeoutError("bounded whitening evaluation wall limit exhausted")
    try:
        conditional = {}
        for index, case_id in enumerate(previous.CASES):
            deadline()
            _, _, _, y = previous.reference.load_case(panel_path, case_id, infer_c=True)
            draws = trainer.sample(workflow, task, y[None], CONDITIONAL_DRAWS, seed=SEED+index)[0]
            support = baseline.support_report(draws, np.asarray(target["prior_bounds"]))
            if draws.shape != (CONDITIONAL_DRAWS, 6) or support["material_outside_value_count"]:
                raise ValueError("conditional whitening posterior dimensions or support differ")
            conditional[case_id] = draws
            selected, reference_binding = previous.joint_reference(ref_protocol, case_id, panel_path, target, panel["products"], y)
            entry = {"observation_sha256": baseline.observations_digest(y), "neural_support": support,
                "neural_quantiles_05_50_95": np.quantile(draws, [.05, .5, .95], axis=0).tolist(), **reference_binding}
            for prefix in ("reference_run", "reference_diagnostics", "reference_chain_payload"):
                if prefix+"_path" in reference_binding:
                    protected[reference_binding[prefix+"_path"]] = reference_binding[prefix+"_sha256"]
            if selected is not None:
                chains, _ = selected
                entry["comparison"] = descriptive_comparison(draws, chains, target["parameter_names"], panel["products"])
            record["cases"][case_id] = entry
        fixed.write_npz(payload_out/"conditional-draws.npz", conditional)
        record["conditional_draws"] = {**binding(payload_out/"conditional-draws.npz"), "draws_per_case": CONDITIONAL_DRAWS}
        with np.load(payload, allow_pickle=False) as arrays:
            physical = arrays["validation_parameters"][:ROWS].copy()
            clean = arrays["validation_clean"][:ROWS].copy()
        noisy, truth = previous.population_observations(task, physical, clean)
        if (baseline.observations_digest(noisy) != baseline_record["population"]["observations_sha256"]
                or baseline.observations_digest(truth) != baseline_record["population"]["truth_sha256"]):
            raise ValueError("the whitening population differs from the original fixed c/Z cases")
        raw_conditions = frozen.scale.condition_data(task, noisy)["observations"]
        features = workflow.transform.transform(raw_conditions).astype(np.float32)
        record["evaluation_input_bindings"] = {"raw_conditions": fixed.array_summary(raw_conditions),
            "transformed_conditions": fixed.array_summary(features), "sidecar_applied_once": True,
            "preprocessing_refit": False}
        draws = np.concatenate([trainer.sample(workflow, task, noisy[start:start+16], DRAWS,
            seed=SEED+1000+start) for start in range(0, ROWS, 16)])
        deadline()
        support = baseline.support_report(draws, np.asarray(target["prior_bounds"]))
        if draws.shape != (ROWS, DRAWS, 6) or support["material_outside_value_count"]:
            raise ValueError("population whitening posterior dimensions or support differ")
        ranks = previous.validation.sbc(draws, truth, seed=SEED+2000)
        fractional = ranks.pop("fractional_ranks")
        quantiles = np.quantile(draws, [.05, .5, .95], axis=1)
        fixed.write_npz(out/"recovery.npz", {"truth": truth, "quantiles": quantiles.astype(np.float32),
            "row_id": np.array([f"validation:{i:08d}" for i in range(ROWS)])})
        record["population"] = {"rows": ROWS, "draws_per_case": DRAWS, "parameter_names": target["parameter_names"],
            "support": support, "rank_dkw": ranks, "fractional_ranks": fractional,
            "coverage": previous.validation.coverage(draws, truth),
            "recovery": historical.recovery_summary(truth, quantiles, target["prior_bounds"]),
            "mean_posterior_sd": draws.std(axis=1, ddof=1).mean(axis=0).tolist(),
            "recovery_sha256": fixed.digest(out/"recovery.npz"),
            "observations_sha256": baseline.observations_digest(noisy), "truth_sha256": baseline.observations_digest(truth),
            "scope": replay["evaluation"]["sbc_scope"]}
        if any(fixed.digest(path) != value for path, value in protected.items()):
            raise ValueError("a protected whitening checkpoint, sidecar, data or source changed")
        deadline()
        record.update(status="complete", protected_inputs_unchanged=True, seconds=time.monotonic()-started)
        fixed.write_json(out/"evaluation.json", record); fixed.write_json(out/"run.json", record)
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic()-started)
        fixed.write_json(out/"failure.json", record); fixed.write_json(out/"run.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "study", "replay-protocol", "out", "payload-out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.study, args.replay_protocol, args.out, args.payload_out)
    print({"mode": result["mode"], "status": result["status"], "c_rmse": result["population"]["recovery"]["median_rmse"][-1]})

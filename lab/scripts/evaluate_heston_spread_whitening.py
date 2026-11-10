"""Read-only final diagnostics for paired diagonal and ZCA preprocessing.

New networks are queried once on the historical 512-case population and three
existing conditional cases. The training-fitted sidecar is loaded, never fit
again. All posterior targets remain the same raw-quote six-dimensional task.
This is a bounded preprocessing pilot, without a formal acceptance family.
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
from scripts import train_heston_spread_whitening as trainer

fixed, baseline = previous.fixed, previous.baseline
MODES = ("diagonal", "zca")
SEED, ROWS, DRAWS, CONDITIONAL_DRAWS = previous.SEED, previous.ROWS, previous.DRAWS, previous.CONDITIONAL_DRAWS
WALL_SECONDS = 300


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": fixed.digest(path)}


def evaluation_contract(target):
    if (target.get("infer_c") is not True or target.get("parameter_names", [])[-1:] != ["c"]
            or target.get("prior_bounds", [])[-1:] != [[.5, 2.]] or target.get("quote_count") != 54):
        raise ValueError("the unchanged complete six-target posterior is required")
    return {"modes": list(MODES), "final_updates": 8000, "seed": SEED,
        "population_rows": ROWS, "posterior_draws_per_population_case": DRAWS,
        "conditional_cases": list(previous.CASES), "conditional_draws": CONDITIONAL_DRAWS,
        "population_c": "reuse draw_c(512,0,261112,namespace=3); c is never a condition",
        "population_gaussian": "reuse PCG64 SeedSequence([0x53504556,261112]) standard errors",
        "preprocessing": "load the training-only frozen sidecar and apply exactly once; never fit on evaluation data",
        "conditional_metrics": "descriptive absolute mean error / reference SD and central90% width ratio; no pass/fail family",
        "sbc_scope": "six full-prior marginal DKW screens within each final arm, not a release acceptance test",
        "primary_quantity": "c", "max_wall_seconds": WALL_SECONDS,
        "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False}


def validate_study(study, target):
    if (study.get("format") != "heston_spread_whitening_study_v1"
            or study.get("joint_task") != target or study.get("evaluation") != evaluation_contract(target)
            or [arm.get("mode") for arm in study.get("arms", [])] != list(MODES)):
        raise ValueError("paired whitening evaluation declaration differs")
    for key in ("original_study", "reference_recovery", "baseline_evaluation", "matched_mcmc_report"):
        value = study.get(key, {})
        if set(value) != {"path", "sha256"} or fixed.digest(value["path"]) != value["sha256"]:
            raise ValueError("whitening evaluation historical binding differs: "+key)
    for name, value in study["sources"].items():
        if fixed.digest(value["path"]) != value["sha256"]:
            raise ValueError("declared whitening source differs: "+name)


def validate_checkpoint(metadata, study, study_path, checkpoint):
    mode = metadata.get("mode")
    contract = trainer.selected_contract(study, mode)["contract"]
    if (mode not in MODES or metadata.get("format") != "heston_spread_whitening_checkpoint_v1"
            or metadata["training_protocol"] != contract or metadata["actual_optimizer_updates"] != 8000
            or contract["train_rows"] != 10000 or contract["batch_size"] != 256
            or contract["bank_passes"] != 200 or contract["planned_presentations"] != 2000000
            or contract["planned_updates"] != 8000 or contract["standardize"] != ["inference_variables"]
            or contract["architecture"] != {"family": "spline", "depth": 6, "widths": [128, 128], "dropout": 0.}
            or Path(metadata["protocol_path"]).resolve() != Path(study_path).resolve()
            or metadata["protocol_sha256"] != fixed.digest(study_path)
            or metadata["paired_initialization"]["all_trainable_hashes_exact"] is not True
            or metadata.get("reserved_arrays_opened") is not False
            or fixed.digest(Path(checkpoint)/"preprocessing.json") != metadata["preprocessing_sha256"]):
        raise ValueError("final whitening checkpoint recipe, sidecar or training-data boundary differs")


def preprocessing_summary(record):
    """Inspect existing fitted state without refitting or reading validation."""
    transform = trainer.preprocessing_transform(record)
    covariance = np.asarray(record["noisy_training_covariance"], dtype=np.float64)
    matrix = transform.whitening_matrix if record["mode"] == "zca" else np.diag(1/transform.standard_deviation)
    transformed_covariance = matrix.T @ covariance @ matrix
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {"mode": record["mode"], "fit_split": record["fit"]["split"], "fit_rows": record["fit"]["rows"],
        "feature_dimension": 54, "fit_conditions": record["fit"]["conditions"], "fit_targets": record["fit"]["targets"],
        "fit_row_ids": record["fit"]["row_id"], "validation_used_for_fit": record["validation_used_for_fit"],
        "raw_covariance_condition_number": float(np.linalg.cond(covariance)),
        "transformed_covariance_condition_number": float(np.linalg.cond(transformed_covariance)),
        "raw_covariance_eigenvalues": eigenvalues.tolist(),
        "relative_eigenvalue_floor": record["transform"].get("relative_eigenvalue_floor"),
        "floored_eigenvalues": int(np.sum(transform.safe_eigenvalues > transform.eigenvalues)) if record["mode"] == "zca" else 0,
        "noise_scope": record["noise_scope"], "invertible_all_features_retained": True}


def descriptive_comparison(neural, chains, names, products):
    neural_values = np.concatenate([neural, previous.reference.product_values(neural, products)], axis=-1)
    reference_values = np.concatenate([chains, previous.reference.product_values(chains, products)], axis=-1).reshape(-1, len(names)+len(products))
    metrics = previous.validation.agreement(neural_values, reference_values)
    nq = np.quantile(neural_values, [.05, .5, .95], axis=0)
    rq = np.quantile(reference_values, [.05, .5, .95], axis=0)
    return {"quantities": [*names, *products], **metrics,
        "neural_quantiles_05_50_95": nq.tolist(), "reference_quantiles_05_50_95": rq.tolist(),
        "neural_mean": neural_values.mean(axis=0).tolist(), "reference_mean": reference_values.mean(axis=0).tolist(),
        "reference_sd": reference_values.std(axis=0).tolist(),
        "scope": "Descriptive point summaries of saved ready joint references; no Monte Carlo acceptance classification or simultaneous family."}


def evaluate(checkpoint, study_path, out, payload_out):
    started = time.monotonic()
    checkpoint, study_path, out, payload_out = map(Path, (checkpoint, study_path, out, payload_out))
    if out.exists() or payload_out.exists():
        raise FileExistsError("whitening evaluation identities must be new")
    study = read(study_path)
    workflow, metadata = trainer.load_checkpoint(checkpoint)
    task, target = metadata["task"], metadata["task"]["target"]
    validate_study(study, target)
    validate_checkpoint(metadata, study, study_path, checkpoint)
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
        "whitener": trainer.sources()["whitening"]}
    saved.verify_artifacts(sources, protected)
    saved.verify_artifacts({key: study[key] for key in
        ("original_study", "reference_recovery", "baseline_evaluation", "matched_mcmc_report")}, protected)
    for path in (study_path, panel_path, ref_path, checkpoint/"checkpoint.json", checkpoint/"model.keras",
            checkpoint/"state.npz", preprocessing_path, Path(metadata["preprocessing_path"]), bank/"bank.json", payload):
        protected[str(path.resolve())] = fixed.digest(path)
    baseline_record = read(study["baseline_evaluation"]["path"])
    if baseline_record["task"] != target:
        raise ValueError("old population baseline uses a different statistical task")
    out.mkdir(parents=True); payload_out.mkdir(parents=True)
    record = {"format": "heston_spread_whitening_evaluation_v1", "status": "started", "mode": metadata["mode"],
        "task": target, "task_fingerprint": metadata["task_fingerprint"], "actual_optimizer_updates": 8000,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_record_sha256": fixed.digest(checkpoint/"checkpoint.json"),
        "model_sha256": metadata["model_sha256"], "state_sha256": metadata["state_sha256"],
        "preprocessing": binding(preprocessing_path), "preprocessing_summary": preprocessing_summary(preprocessing),
        "training_dataset_bindings": metadata["splits"], "training_recipe": metadata["training_protocol"],
        "study_protocol": binding(study_path), "reference_protocol": binding(ref_path),
        "conditional_panel": binding(panel_path), "matched_mcmc_report": study["matched_mcmc_report"],
        "source_sha256": sources, "seed": SEED, "cases": {}, "reserved_arrays_opened": False,
        "new_reference_sampling": False, "scientific_acceptance": False,
        "scope": "Final paired preprocessing pilot; descriptive inference checks, no acceptance family or historical training-budget attribution."}
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
        raw_conditions = trainer.scale.condition_data(task, noisy)["observations"]
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
            "scope": study["evaluation"]["sbc_scope"]}
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
    for key in ("checkpoint", "study", "out", "payload-out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.study, args.out, args.payload_out)
    print({"mode": result["mode"], "status": result["status"], "c_rmse": result["population"]["recovery"]["median_rmse"][-1]})

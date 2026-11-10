"""Evaluate the two larger-bank joint-c continuations against unchanged data.

The two arms and four bank-pass checkpoints form one predeclared development
family. Population observations and sample seeds match the earlier experiment.
Only validation arrays are decompressed; existing joint references are reused.
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

fixed, baseline = previous.fixed, previous.baseline
ARMS, BANK_PASSES = (128, 512), (25, 100, 250, 500)
FAMILY_CHECKS, WALL_SECONDS = 384, 600


def read(path):
    return json.loads(Path(path).read_text())


def evaluation_contract(target):
    if (target.get("infer_c") is not True or target.get("parameter_names", [])[-1:] != ["c"]
            or target.get("prior_bounds", [])[-1:] != [[.5, 2.]]):
        raise ValueError("unchanged six-target joint-c task required")
    return {"batch_sizes": list(ARMS), "checkpoint_bank_passes": list(BANK_PASSES),
        "conditional_cases": list(previous.CASES), "family_checks": FAMILY_CHECKS,
        "family_definition": "2 metrics * 8 quantities * 3 cases * 4 endpoints * 2 batch-size arms",
        "mean_tolerance_reference_sd": .3, "width_ratio90_limits": [.75, 1.25],
        "block_bootstrap_replicates": 400, "seed": previous.SEED,
        "population_rows": previous.ROWS,
        "posterior_draws_per_population_case": previous.DRAWS,
        "conditional_draws": previous.CONDITIONAL_DRAWS,
        "population_c": "reuse draw_c(512,0,261112,namespace=3); generating c is never a condition",
        "population_gaussian": "reuse PCG64 SeedSequence([0x53504556,261112]) standard errors",
        "sbc_scope": "six marginal full-prior DKW checks per endpoint; no simultaneous claim across endpoints or arms",
        "baseline_scope": "original 60k endpoint is a descriptive population baseline outside the 384 conditional checks",
        "reserved_arrays_opened": False, "new_reference_sampling": False,
        "scientific_acceptance": False}


def validate_study(study, target):
    if (study.get("format") != "heston_spread_scale_capacity_study_v1"
            or study.get("joint_task") != target
            or study.get("evaluation") != evaluation_contract(target)):
        raise ValueError("continuation evaluation declaration differs")
    for key in ("original_study", "reference_recovery", "baseline_evaluation"):
        value = study.get(key, {})
        if set(value) != {"path", "sha256"} or fixed.digest(value["path"]) != value["sha256"]:
            raise ValueError("continuation evaluation binding differs: " + key)


def recovery_summary(truth, quantiles, bounds):
    truth, quantiles, bounds = map(np.asarray, (truth, quantiles, bounds))
    if (truth.shape != (previous.ROWS, 6) or quantiles.shape != (3, previous.ROWS, 6)
            or bounds.shape != (6, 2) or not np.isfinite(truth).all()
            or not np.isfinite(quantiles).all() or np.any(np.diff(quantiles, axis=0) < 0)
            or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1])):
        raise ValueError("finite ordered six-parameter population recovery required")
    median = quantiles[1]
    correlation = [float(np.corrcoef(truth[:, j], median[:, j])[0, 1])
        if np.std(truth[:, j]) > 0 and np.std(median[:, j]) > 0 else None for j in range(6)]
    width = quantiles[2] - quantiles[0]
    prior_width = .9 * (bounds[:, 1] - bounds[:, 0])
    return {"median_rmse": np.sqrt(np.mean((median-truth)**2, axis=0)).tolist(),
        "truth_posterior_median_correlation": correlation,
        "median_posterior_width90": np.median(width, axis=0).tolist(),
        "prior_width90": prior_width.tolist(),
        "median_width90_over_prior": (np.median(width, axis=0)/prior_width).tolist(),
        "prior_midpoint_rmse": np.sqrt(np.mean((bounds.mean(axis=1)-truth)**2, axis=0)).tolist(),
        "scope": "Descriptive recovery over the same 512 validation rows and fresh fixed population c/Z; no reserved-test claim."}


def validate_endpoint(metadata):
    if (metadata.get("batch_size") not in ARMS
            or metadata.get("completed_bank_passes") not in BANK_PASSES
            or metadata.get("train_rows") != 100000
            or metadata.get("architecture", {}).get("effective_dropout") != 0.):
        raise ValueError("continuation endpoint, bank size or effective dropout differs")


def source_bindings(continuation):
    sources = {"evaluator": Path(__file__), "continuation_trainer": Path(continuation.__file__),
        "population_exporter": Path(previous.__file__), "comparison_consumer": Path(saved.__file__),
        "scale_trainer": Path(previous.trainer.__file__), "fixed_helper": Path(fixed.__file__),
        "comparison_helper": Path(baseline.__file__), "validation": Path(previous.validation.__file__),
        "reference_producer": Path(previous.reference.__file__),
        "reference_consumer": Path(previous.scale_reference.__file__),
        "comparison_metrics": Path(baseline.heston_budget_diagnostics.__file__)}
    sources.update({"reference_"+key: Path(value["path"])
        for key, value in previous.reference.source_hashes().items()})
    return {key: {"path": str(path.resolve()), "sha256": fixed.digest(path)}
        for key, path in sources.items()}


def baseline_population(study, target, protected):
    path = Path(study["baseline_evaluation"]["path"])
    record = read(path)
    recovery = path.parent / "recovery.npz"
    if (record.get("status") != "complete" or record.get("task") != target
            or record.get("actual_optimizer_updates") != 60000
            or record.get("seed") != previous.SEED
            or fixed.digest(recovery) != record["population"]["recovery_sha256"]):
        raise ValueError("original 60k population baseline differs")
    protected[str(recovery.resolve())] = fixed.digest(recovery)
    with np.load(recovery, allow_pickle=False) as arrays:
        truth, quantiles = arrays["truth"].copy(), arrays["quantiles"].copy()
    return {"evaluation": study["baseline_evaluation"],
        "recovery": {"path": str(recovery.resolve()), "sha256": fixed.digest(recovery)},
        "summary": recovery_summary(truth, quantiles, target["prior_bounds"]),
        "outside_new_conditional_family": True}, record["population"]


def evaluate(checkpoint, study_path, out, payload_out):
    from scripts import continue_heston_spread_scale as continuation
    started = time.monotonic()
    checkpoint, study_path, out, payload_out = map(Path, (checkpoint, study_path, out, payload_out))
    if out.exists() or payload_out.exists():
        raise FileExistsError("continuation evaluation identities must be new")
    study = read(study_path)
    workflow, metadata = continuation.load_checkpoint(checkpoint)
    task, target = metadata["task"], metadata["task"]["target"]
    validate_study(study, target)
    validate_endpoint(metadata)
    training_protocol = Path(metadata["protocol_path"])
    if fixed.digest(training_protocol) != metadata["protocol_sha256"]:
        raise ValueError("continuation checkpoint training protocol differs")
    old_study_path = Path(study["original_study"]["path"])
    recovery_path = Path(study["reference_recovery"]["path"])
    old_study = read(old_study_path)
    previous.validate_study(old_study, target)
    ref_path, protected = previous.selected_reference_protocol(old_study, old_study_path, recovery_path)
    ref_protocol = read(ref_path)
    recovery = read(recovery_path)
    campaign_path = Path(recovery["progress_path"])
    if read(campaign_path)["status"] != "completed":
        raise ValueError("existing reference campaign must be completed")
    previous.scale_reference.validate_protocol(ref_protocol, ref_protocol["jobs"][0]["id"])
    panel_path = Path(old_study["panel"]["path"])
    if fixed.digest(panel_path) != old_study["panel"]["sha256"]:
        raise ValueError("unchanged reference panel differs")
    panel = read(panel_path)
    if panel["joint_task"] != target or ref_protocol["panel_sha256"] != fixed.digest(panel_path):
        raise ValueError("unchanged references target a different posterior")
    bank = Path(metadata["bank_path"])
    bank_record = read(bank/"bank.json")
    payload = bank / bank_record["payload"]
    if (payload.resolve().parent != bank.resolve()
            or fixed.digest(bank/"bank.json") != metadata["bank_record_sha256"]
            or bank_record["identity"] != metadata["bank_identity"]
            or bank_record["payload_sha256"] != metadata["bank_payload_sha256"]
            or fixed.digest(payload) != metadata["bank_payload_sha256"]
            or task["physical_clean_price_task"]["target"]["prior_bounds"] != target["prior_bounds"][:5]):
        raise ValueError("continuation evaluation bank/physical projection differs")
    sources = source_bindings(continuation)
    saved.verify_artifacts(sources, protected)
    saved.verify_artifacts({key: study[key] for key in
        ("original_study", "reference_recovery", "baseline_evaluation")}, protected)
    for path in (study_path, old_study_path, recovery_path, campaign_path, ref_path,
            panel_path, training_protocol, checkpoint/"checkpoint.json", checkpoint/"model.keras",
            checkpoint/"state.npz", bank/"bank.json", payload):
        protected[str(path.resolve())] = fixed.digest(path)
    old_population, old_observation_binding = baseline_population(study, target, protected)
    record = {"format": "heston_spread_scale_capacity_evaluation_v1", "status": "started",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_record_sha256": fixed.digest(checkpoint/"checkpoint.json"),
        "model_sha256": metadata["model_sha256"], "state_sha256": metadata["state_sha256"],
        "actual_optimizer_updates": metadata["actual_optimizer_updates"],
        "batch_size": metadata["batch_size"], "completed_bank_passes": metadata["completed_bank_passes"],
        "task": target, "study_protocol": {"path": str(study_path.resolve()), "sha256": fixed.digest(study_path)},
        "training_protocol": {"path": str(training_protocol.resolve()), "sha256": fixed.digest(training_protocol)},
        "reference_protocol": {"path": str(ref_path.resolve()), "sha256": fixed.digest(ref_path)},
        "reference_recovery": study["reference_recovery"], "panel": old_study["panel"],
        "source_sha256": sources, "seed": previous.SEED, "family_checks": FAMILY_CHECKS,
        "baseline_population": old_population, "cases": {}, "scientific_acceptance": False,
        "reserved_arrays_opened": False, "new_reference_sampling": False,
        "scope": "Development continuation comparison; market application is separate from synthetic fidelity."}
    out.mkdir(parents=True)
    payload_out.mkdir(parents=True)
    fixed.write_json(out/"run.json", record)

    def deadline():
        if time.monotonic()-started >= WALL_SECONDS:
            raise TimeoutError("continuation evaluation wall budget exhausted")

    try:
        conditional = {}
        for index, case_id in enumerate(previous.CASES):
            deadline()
            _, _, _, y = previous.reference.load_case(panel_path, case_id, infer_c=True)
            neural = previous.trainer.sample(workflow, task, y[None], previous.CONDITIONAL_DRAWS,
                seed=previous.SEED+index)[0]
            support = baseline.support_report(neural, np.asarray(target["prior_bounds"]))
            if neural.shape != (4096, 6) or support["material_outside_value_count"]:
                raise ValueError("continuation conditional draw dimensions/support differ")
            conditional[case_id] = neural
            entry = {"role": "synthetic development fidelity" if case_id != "market" else "separate market application diagnostic",
                "observation_sha256": baseline.observations_digest(y), "neural_support": support,
                "neural_quantiles_05_50_95": np.quantile(neural, [.05, .5, .95], axis=0).tolist(),
                "c_mean": float(neural[:, -1].mean()), "c_sd": float(neural[:, -1].std(ddof=1)),
                "neural_c_physical_correlations": saved.c_correlations(neural, target["parameter_names"])}
            selected, binding = previous.joint_reference(ref_protocol, case_id, panel_path,
                target, panel["products"], y)
            entry.update(binding)
            for prefix in ("reference_run", "reference_diagnostics", "reference_chain_payload"):
                if prefix+"_path" in binding:
                    protected[binding[prefix+"_path"]] = binding[prefix+"_sha256"]
            if selected is not None:
                chains, diagnostics = selected
                flat = chains.reshape(-1, 6)
                entry["reference_c_marginal"] = {"mean": float(flat[:, -1].mean()),
                    "sd": float(flat[:, -1].std(ddof=1)),
                    "quantiles_05_50_95": np.quantile(flat[:, -1], [.05, .5, .95]).tolist(),
                    "probability_c_le_055": float(np.mean(flat[:, -1] <= .55)),
                    "probability_c_ge_195": float(np.mean(flat[:, -1] >= 1.95))}
                entry["comparison"] = baseline.compare_draws(neural, chains, diagnostics,
                    target["parameter_names"], panel["products"], seed=previous.SEED+100+index,
                    replicates=400, family_checks=FAMILY_CHECKS)
                entry["status"] = entry["comparison"]["status"]
            record["cases"][case_id] = entry
        fixed.write_npz(payload_out/"conditional-draws.npz", conditional)
        record["conditional_draws"] = {"path": str((payload_out/"conditional-draws.npz").resolve()),
            "sha256": fixed.digest(payload_out/"conditional-draws.npz"), "draws_per_case": previous.CONDITIONAL_DRAWS}
        with np.load(payload, allow_pickle=False) as arrays:
            physical = arrays["validation_parameters"][:previous.ROWS].copy()
            clean = arrays["validation_clean"][:previous.ROWS].copy()
        noisy, truth = previous.population_observations(task, physical, clean)
        if (baseline.observations_digest(noisy) != old_observation_binding["observations_sha256"]
                or baseline.observations_digest(truth) != old_observation_binding["truth_sha256"]):
            raise ValueError("continuation population differs from original fixed c/Z and validation rows")
        draws = np.concatenate([previous.trainer.sample(workflow, task,
            noisy[start:start+16], previous.DRAWS, seed=previous.SEED+1000+start)
            for start in range(0, previous.ROWS, 16)])
        deadline()
        support = baseline.support_report(draws, np.asarray(target["prior_bounds"]))
        if draws.shape != (512, 512, 6) or support["material_outside_value_count"]:
            raise ValueError("continuation population draw dimensions/support differ")
        ranks = previous.validation.sbc(draws, truth, seed=previous.SEED+2000)
        fractional = ranks.pop("fractional_ranks")
        quantiles = np.quantile(draws, [.05, .5, .95], axis=1)
        fixed.write_npz(out/"recovery.npz", {"truth": truth,
            "quantiles": quantiles.astype(np.float32),
            "row_id": np.array([f"validation:{i:08d}" for i in range(previous.ROWS)])})
        record["population"] = {"rows": previous.ROWS, "draws_per_case": previous.DRAWS,
            "parameter_names": target["parameter_names"], "support": support,
            "rank_dkw": ranks, "fractional_ranks": fractional,
            "coverage": previous.validation.coverage(draws, truth),
            "recovery": recovery_summary(truth, quantiles, target["prior_bounds"]),
            "mean_posterior_sd": np.std(draws, axis=1, ddof=1).mean(axis=0).tolist(),
            "recovery_sha256": fixed.digest(out/"recovery.npz"),
            "observations_sha256": baseline.observations_digest(noisy),
            "truth_sha256": baseline.observations_digest(truth),
            "generating_c": {"prior": "independent Uniform(.5,2)", "namespace": 3, "count": previous.ROWS},
            "scope": study["evaluation"]["sbc_scope"]}
        combine = lambda items: "fail" if "fail" in items else "pass" if all(x == "pass" for x in items) else "inconclusive"
        record.update(status="complete",
            conditional_status=combine([entry["status"] for entry in record["cases"].values()]),
            synthetic_conditional_status=combine([record["cases"][case]["status"] for case in previous.CASES[:2]]),
            seconds=time.monotonic()-started)
        if any(fixed.digest(path) != value for path, value in protected.items()):
            raise ValueError("a protected continuation evaluation input changed")
        deadline()
        record["protected_inputs_unchanged"] = True
        fixed.write_json(out/"evaluation.json", record)
        fixed.write_json(out/"run.json", record)
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
            seconds=time.monotonic()-started)
        fixed.write_json(out/"failure.json", record)
        fixed.write_json(out/"run.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "study", "out", "payload-out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.study, args.out, args.payload_out)
    print(result["synthetic_conditional_status"])


if __name__ == "__main__":
    main()

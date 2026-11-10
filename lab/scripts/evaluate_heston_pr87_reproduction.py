"""Bounded diagnostics for one frozen PR87 native FlowMatching recipe.

The notebook's fresh 100-case/300-draw recovery runs first. The second panel
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
        "checkpoint_selection": "primary final epoch200 only; earlier and minimum-validation checkpoints archived without posterior queries",
        "sbc_scope": "six marginal DKW screens per endpoint and panel; descriptive development evidence, no simultaneous cross-panel or endpoint acceptance",
        "max_total_posterior_evaluation_seconds": WALL_SECONDS,
        "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False}


def checked_binding(spec, protected):
    if not {"path", "sha256"} <= set(spec) or binding(spec["path"]) != {key: spec[key] for key in ("path", "sha256")}:
        raise ValueError("declared input identity differs")
    protected[str(Path(spec["path"]).resolve())] = spec["sha256"]
    return Path(spec["path"])


def raw_support(draws, bounds, names):
    """Support diagnostics without changing or rejecting a single draw."""
    values = np.asarray(draws)
    bounds = np.asarray(bounds)
    if values.ndim not in (2, 3) or values.shape[-1] != 6 or bounds.shape != (6, 2) or len(names) != 6:
        raise ValueError("six raw posterior targets and prior bounds required")
    flat = values.reshape(-1, 6)
    finite = np.isfinite(flat)
    below, above = flat < bounds[:, 0], flat > bounds[:, 1]
    outside = below | above
    nonphysical = np.column_stack((flat[:, 0] <= 0, flat[:, 1] <= 0, flat[:, 2] <= 0,
        flat[:, 3] <= 0, (flat[:, 4] < -1) | (flat[:, 4] > 1), flat[:, 5] <= 0)) | ~finite
    def counts(mask):
        return {"sample_count": int(mask.any(axis=1).sum()), "sample_fraction": float(mask.any(axis=1).mean()),
            "value_count": int(mask.sum()), "per_dimension_count": mask.sum(axis=0).tolist(),
            "per_dimension_fraction": mask.mean(axis=0).tolist()}
    return {"raw_shape": list(values.shape), "samples": len(flat), "parameter_names": list(names),
        "outside_prior": counts(outside), "below_prior": counts(below), "above_prior": counts(above),
        "nonphysical": counts(nonphysical), "nonfinite": counts(~finite),
        "physical_definition": "v0,kappa,theta,vol_of_vol,c strictly positive; -1<=rho<=1; nonfinite is invalid",
        "draws_discarded": 0, "draws_clipped": 0, "draws_resampled": 0,
        "outside_prior_draws_retained_for_calibration": True}


def recovery_summary(truth, quantiles, bounds):
    truth, quantiles, bounds = map(np.asarray, (truth, quantiles, bounds))
    if truth.ndim != 2 or truth.shape[1] != 6 or quantiles.shape != (3, len(truth), 6):
        raise ValueError("six-target recovery shapes differ")
    median = quantiles[1]
    width = quantiles[2] - quantiles[0]
    prior_width = .9 * (bounds[:, 1] - bounds[:, 0])
    return {"median_rmse": np.sqrt(np.mean((median - truth)**2, axis=0)).tolist(),
        "mean_error": (median - truth).mean(axis=0).tolist(),
        "truth_posterior_median_correlation": [float(np.corrcoef(truth[:, j], median[:, j])[0, 1])
            if np.std(truth[:, j]) > 0 and np.std(median[:, j]) > 0 else None for j in range(6)],
        "median_posterior_width90": np.median(width, axis=0).tolist(),
        "prior_width90": prior_width.tolist(), "median_width90_over_prior": (np.median(width, axis=0) / prior_width).tolist(),
        "prior_midpoint_rmse": np.sqrt(np.mean((bounds.mean(axis=1) - truth)**2, axis=0)).tolist()}


def population_summary(draws, truth, quotes, target, seed):
    values, truth = np.asarray(draws), np.asarray(truth)
    if values.ndim != 3 or truth.shape != (len(values), 6) or values.shape[-1] != 6:
        raise ValueError("raw population posterior dimensions differ")
    support = raw_support(values, target["prior_bounds"], target["parameter_names"])
    result = {"rows": len(values), "draws_per_case": values.shape[1], "parameter_names": target["parameter_names"],
        "support": support, "observations_sha256": baseline.observations_digest(quotes),
        "truth_sha256": baseline.observations_digest(truth), "sbc_seed": seed,
        "raw_draws_used": int(values.shape[0] * values.shape[1]), "calibration_status": "defined"}
    if support["nonfinite"]["value_count"]:
        result.update(calibration_status="undefined_nonfinite_raw_draws", coverage=None, rank_dkw=None,
            fractional_ranks=None, recovery=None, mean_posterior_sd=None)
        return result, None
    ranks = validation.sbc(values, truth, seed=seed)
    fractional = ranks.pop("fractional_ranks")
    quantiles = np.quantile(values, [.05, .5, .95], axis=1)
    result.update(rank_dkw=ranks, fractional_ranks=fractional, coverage=validation.coverage(values, truth),
        recovery=recovery_summary(truth, quantiles, target["prior_bounds"]),
        mean_posterior_sd=values.std(axis=1, ddof=1).mean(axis=0).tolist())
    return result, quantiles


def dependence(neural, reference):
    a, b = [np.asarray(values).reshape(-1, 6) for values in (neural, reference)]
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a.std(axis=0) == 0) or np.any(b.std(axis=0) == 0):
        return {"status": "undefined", "reason": "nonfinite values or constant dimensions; raw draws retained"}
    ac, bc = np.corrcoef(a, rowvar=False), np.corrcoef(b, rowvar=False)
    difference = ac - bc
    unique = np.abs(difference[np.triu_indices(6, k=1)])
    return {"status": "defined", "neural_correlation": ac.tolist(), "reference_correlation": bc.tolist(),
        "signed_correlation_difference": difference.tolist(), "unique_parameter_pairs": 15,
        "max_absolute_correlation_difference": float(unique.max()),
        "median_absolute_correlation_difference": float(np.median(unique)),
        "scope": "descriptive physical-coordinate Pearson correlation, not a full-joint equality test"}


def descriptive_comparison(neural, chains, names, products, bounds):
    """Always compare the six raw targets; invalid derived products are skipped."""
    neural, reference = np.asarray(neural), np.asarray(chains).reshape(-1, 6)
    support = raw_support(neural, bounds, names)
    if support["nonfinite"]["value_count"]:
        return {"status": "undefined_nonfinite_raw_draws", "quantities": list(names),
            "derived_products": {"status": "skipped", "reason": "nonfinite raw draws"}}
    nq, rq = np.quantile(neural, [.05, .5, .95], axis=0), np.quantile(reference, [.05, .5, .95], axis=0)
    result = {"status": "defined", "quantities": list(names), **validation.agreement(neural, reference),
        "neural_quantiles_05_50_95": nq.tolist(), "reference_quantiles_05_50_95": rq.tolist(),
        "neural_mean": neural.mean(axis=0).tolist(), "reference_mean": reference.mean(axis=0).tolist(),
        "reference_sd": reference.std(axis=0).tolist(), "dependence": dependence(neural, reference),
        "scope": "descriptive six-target raw-draw mean and central90% width comparison to the existing ready joint reference"}
    if support["nonphysical"]["sample_count"]:
        result["derived_products"] = {"status": "skipped", "reason": "raw samples include nonphysical parameters; no filtering or target alteration"}
    else:
        a, b = previous.reference.product_values(neural, products), previous.reference.product_values(reference, products)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            result["derived_products"] = {"status": "skipped", "reason": "derived product calculation nonfinite; raw six-target analysis retained"}
        else:
            result["derived_products"] = {"status": "defined", "quantities": list(products), **validation.agreement(a, b)}
    return result


def historical_inputs(protocol, task, protected):
    study_path = checked_binding(protocol["original_whitening_study"], protected)
    study = read(study_path)
    whitening.validate_study(study, task["target"])
    for spec in study["sources"].values():
        checked_binding(spec, protected)
    checked_binding(study["matched_mcmc_report"], protected)
    input_path = checked_binding(protocol["historical_population_inputs"], protected)
    inputs = read(input_path)
    payload = input_path.parent / inputs["payload"]
    if (inputs.get("format") != "heston_spread_scale_recovery_inputs_v1" or inputs["task"] != task
            or payload.resolve().parent != input_path.parent.resolve() or fixed.digest(payload) != inputs["payload_sha256"]):
        raise ValueError("historical raw population input target or payload differs")
    protected[str(payload.resolve())] = inputs["payload_sha256"]
    old_baseline = read(checked_binding(study["baseline_evaluation"], protected))
    if old_baseline["task"] != task["target"]:
        raise ValueError("historical baseline target differs")
    with np.load(payload, allow_pickle=False) as archive:
        quotes, truth, row_ids = [archive[key].copy() for key in ("quotes", "truth", "row_id")]
    if (quotes.shape != (ROWS, 54) or truth.shape != (ROWS, 6)
            or not np.array_equal(row_ids, np.array([f"validation:{i:08d}" for i in range(ROWS)]))
            or not np.isfinite(quotes).all() or not np.isfinite(truth).all()
            or baseline.observations_digest(quotes) != inputs["observations_sha256"]
            or baseline.observations_digest(truth) != inputs["truth_sha256"]
            or inputs["observations_sha256"] != old_baseline["population"]["observations_sha256"]
            or inputs["truth_sha256"] != old_baseline["population"]["truth_sha256"]):
        raise ValueError("historical raw quote/truth rows differ from frozen baseline")
    old_study_path = checked_binding(study["original_study"], protected)
    old_study = read(old_study_path)
    previous.validate_study(old_study, task["target"])
    recovery_path = checked_binding(study["reference_recovery"], protected)
    ref_path, recovered = previous.selected_reference_protocol(old_study, old_study_path, recovery_path)
    protected.update(recovered)
    recovery = read(recovery_path)
    if read(recovery["progress_path"])["status"] != "completed":
        raise ValueError("existing reference campaign is not completed")
    panel_path = checked_binding(old_study["panel"], protected)
    panel, ref_protocol = read(panel_path), read(ref_path)
    if panel["joint_task"] != task["target"]:
        raise ValueError("existing reference panel target differs")
    previous.scale_reference.validate_protocol(ref_protocol, ref_protocol["jobs"][0]["id"])
    protected[str(ref_path.resolve())] = fixed.digest(ref_path)
    return quotes, truth, row_ids, study, panel_path, panel, ref_protocol


def evaluate(checkpoint, protocol_path, out, payload_out, *, max_seconds=WALL_SECONDS):
    started = time.monotonic()
    checkpoint, protocol_path, out, payload_out = map(Path, (checkpoint, protocol_path, out, payload_out))
    if out.exists() or payload_out.exists():
        raise FileExistsError("PR87 evaluation output identities must be new")
    protocol, protected = read(protocol_path), {}
    if not 0 < max_seconds <= WALL_SECONDS:
        raise ValueError("posterior evaluation budget must be positive and at most900 seconds")
    from scripts import reproduce_heston_pr87 as trainer
    workflow, metadata = trainer.load_checkpoint(checkpoint)
    task, target = metadata["task"], metadata["task"]["target"]
    if (metadata.get("native_epoch") != 200 or metadata.get("actual_optimizer_updates") != 8000
            or protocol.get("evaluation") != evaluation_contract(target) or protocol.get("task") != task):
        raise ValueError("PR87 evaluation protocol or canonical task differs")
    if protocol.get("sources", {}).get("evaluator") != binding(__file__):
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
            or not np.array_equal(native_observations, reconstructed_observations)):
        raise ValueError("notebook native100-row validation panel or exact condition reconstruction differs")
    out.mkdir(parents=True); payload_out.mkdir(parents=True)
    record = {"format": "heston_pr87_reproduction_evaluation_v1", "status": "started", "task": target,
        "checkpoint": binding(checkpoint / "checkpoint.json"), "native_epoch": metadata["native_epoch"],
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
        "scope": "whole native notebook recipe with seed9011 added to an originally unseeded notebook; fresh training data prevents controlled architecture attribution",
        "loss_interpretation": "native FlowMatching objective, not density NLL; absolute values are not comparable with spline losses",
        "max_seconds": max_seconds}
    fixed.write_json(out / "run.json", record)
    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("bounded PR87 posterior evaluation wall budget exhausted")
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
            print("PR87 posterior evaluation stopped at its hard wall limit", file=sys.stderr)
            return 124
    result = evaluate(args.checkpoint, args.protocol, args.out, args.payload_out, max_seconds=args.wall_seconds)
    print(json.dumps({"status": result["status"], "epoch": result["native_epoch"], "seconds": result["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

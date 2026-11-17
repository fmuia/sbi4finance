"""Curate a historical paired preprocessing experiment for the Heston lesson.

Read the preserved whitening-pilot directory supplied with --study. Only the
shared generating c values, marginal c quantiles, descriptive metrics and
source hashes are written. No pricing, training, posterior sampling or private
payload publication occurs, and no financial kernel is implemented here.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


MODES = ("diagonal", "zca")
NAMES = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c"]
BOUNDS = [[.01, .16], [.1, 10.], [.01, .16], [.1, 3.], [-.95, -.1], [.5, 2.]]


def digest(path):
    value = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def c_metrics(truth, quantiles):
    return {
        "median_rmse": float(np.sqrt(np.mean((quantiles[1] - truth) ** 2))),
        "truth_median_correlation": float(np.corrcoef(truth, quantiles[1])[0, 1]),
        "median_width90": float(np.median(quantiles[2] - quantiles[0])),
        "coverage90": float(np.mean((quantiles[0] <= truth) & (truth <= quantiles[2]))),
    }


def plotting_json(value, level=0):
    if isinstance(value, list):
        if all(isinstance(item, (int, float)) for item in value):
            return json.dumps(value, allow_nan=False)
        rows = [plotting_json(item, level + 1) for item in value]
        opening, closing = "[", "]"
    elif isinstance(value, dict):
        rows = [json.dumps(key) + ": " + plotting_json(item, level + 1) for key, item in value.items()]
        opening, closing = "{", "}"
    else:
        return json.dumps(value, allow_nan=False)
    padding = "  " * (level + 1)
    return opening + "\n" + padding + (",\n" + padding).join(rows) + "\n" + "  " * level + closing


def curate(study):
    study = Path(study).resolve()
    protocol = read(study / "protocol.json")
    verification = read(study / "results-verification.json")
    manifest = read(study / "report-manifest.json")
    contracts = {arm["mode"]: arm["contract"] for arm in protocol["arms"]}
    require(set(contracts) == set(MODES), "the declared diagonal/ZCA pair is required")
    different = sorted(key for key in set(contracts["diagonal"]) | set(contracts["zca"])
                       if contracts["diagonal"].get(key) != contracts["zca"].get(key))
    require(different == ["mode", "preprocessing_fingerprint"], "the two contracts differ beyond preprocessing")
    task = protocol["joint_task"]
    require(task["parameter_names"] == NAMES and task["prior_bounds"] == BOUNDS
            and task["quote_count"] == 54 and task["infer_c"] is True
            and task["observation_model"] == "independent Gaussian with price SD c * half_spread",
            "the historical 54-quote c-times-half-spread task is required")
    require(protocol["predeclared_before_fit"] is True and protocol["scientific_acceptance"] is False,
            "this is a predeclared developmental pilot")
    require(verification["status"] == "passed"
            and verification["actual_pairing"]["only_input_preprocessing_differs"] is True,
            "the independent pairing verification must pass")
    require(digest(study / "protocol.json") == verification["protocol"]["sha256"]
            == manifest["protocol"]["sha256"], "the source protocol changed")
    source_bindings = {"protocol": digest(study / "protocol.json"),
                       "independent_verification": digest(study / "results-verification.json"),
                       "report_manifest": digest(study / "report-manifest.json"),
                       "curation_source": digest(__file__)}
    runs, evaluations, arms, cache_truths, cache_ids = {}, {}, {}, {}, {}
    for mode in MODES:
        contract = contracts[mode]
        run_path = study / "training" / mode / "run.json"
        training_path = study / "training" / mode / "training.json"
        evaluation_path = study / "evaluations" / mode / "evaluation.json"
        curve_path = study / "training" / mode / "learning-curve.json"
        recovery_path = evaluation_path.parent / "recovery.npz"
        run, training, evaluation = read(run_path), read(training_path), read(evaluation_path)
        runs[mode] = run
        evaluations[mode] = evaluation
        require(run["training_protocol"] == contract and run["task"]["target"] == task,
                "run contract or task differs")
        require(run["initial_state"] and run["paired_initialization"]["fresh"] is True
                and run["seed"] == 9011 and run["train_rows"] == 10000 and run["batch_size"] == 256,
                "the fresh 10k paired training specification differs")
        require(contract["architecture"] == {"family": "spline", "depth": 6, "widths": [128, 128], "dropout": 0.}
                and contract["standardize"] == ["inference_variables"]
                and contract["optimizer"] == {"class_name": "Adam", "schedule": {"class_name": "CosineDecay",
                    "initial_learning_rate": .0005, "decay_steps": 8000, "alpha": .1}}
                and contract["validation_rows"] == 100 and contract["batches_per_pass"] == 40
                and contract["tail_batch_size"] == 16,
                "the fixed architecture, optimizer, monitor or exposure plan differs")
        require(contract["planned_updates"] == training["completed_updates"] == 8000
                and training["completed_bank_passes"] == 200 and training["presentations"] == 2000000
                and training["status"] == "complete", "completed fixed exposure budget differs")
        checkpoint = Path(training["last_checkpoint"])
        saved = read(checkpoint / "checkpoint.json")
        require(saved["actual_optimizer_updates"] == evaluation["actual_optimizer_updates"] == 8000,
                "the predeclared final checkpoint is required")
        for name, key in (("model.keras", "model_sha256"), ("state.npz", "state_sha256")):
            require(digest(checkpoint / name) == saved[key] == evaluation[key], "native checkpoint changed")
        require(digest(checkpoint / "checkpoint.json") == evaluation["checkpoint_record_sha256"],
                "evaluation checkpoint record changed")
        require(digest(recovery_path) == evaluation["population"]["recovery_sha256"], "recovery cache changed")
        with np.load(recovery_path, allow_pickle=False) as cache:
            truth, quantiles, ids = cache["truth"].copy(), cache["quantiles"].astype(float), cache["row_id"].copy()
        require(truth.shape == (512, 6) and quantiles.shape == (3, 512, 6)
                and np.isfinite(truth).all() and np.isfinite(quantiles).all()
                and np.all(np.diff(quantiles, axis=0) >= 0), "invalid cached recovery population")
        cache_truths[mode], cache_ids[mode] = truth, ids
        metrics = c_metrics(truth[:, -1], quantiles[:, :, -1])
        population = evaluation["population"]
        require(evaluation["seed"] == 261112 and population["rows"] == 512
                and population["draws_per_case"] == 512 and population["parameter_names"] == NAMES,
                "the matched evaluation budget or target order differs")
        official = population["recovery"]
        for label, official_key in (("median_rmse", "median_rmse"),
                                    ("truth_median_correlation", "truth_posterior_median_correlation"),
                                    ("median_width90", "median_posterior_width90")):
            require(np.isclose(metrics[label], official[official_key][-1], atol=1e-6, rtol=1e-6),
                    "plot-cache metrics disagree with the original full-precision evaluation")
        coverage = population["coverage"]["0.9"]["coverage"]
        require(metrics["coverage90"] == coverage[-1], "cached c coverage differs")
        preprocessing = evaluation["preprocessing_summary"]
        require(preprocessing["mode"] == mode and preprocessing["fit_split"] == "train"
                and preprocessing["fit_rows"] == 10000 and preprocessing["validation_used_for_fit"] is False,
                "preprocessing was not frozen from the same training-only rows")
        curve = read(curve_path)
        require(len(curve) == 200 and curve[-1]["updates"] == 8000, "the complete fixed-budget loss curve is required")
        best = min(curve, key=lambda row: row["validation_probe_loss"])
        arms[mode] = {"c_q05_q50_q95": quantiles[:, :, -1].tolist(), "metrics": metrics,
                      "sbc_passed": population["rank_dkw"]["passed"], "physical_coverage90": coverage[:5],
                      "checkpoint": {"epoch": 200, "optimizer_updates": 8000, "selection": "predeclared_final",
                                     "validation_min_epoch": best["completed_bank_passes"],
                                     "validation_min_loss": best["validation_probe_loss"],
                                     "final_validation_loss": curve[-1]["validation_probe_loss"],
                                     "best_validation_weights_saved": False},
                      "preprocessing": {"mode": mode, "fit_rows": 10000, "fit_split": "train",
                                        "covariance_condition_number": preprocessing["transformed_covariance_condition_number"],
                                        "all_features_retained": True, "eigenvalues_floored": preprocessing["floored_eigenvalues"]}}
        for label, path in (("run", run_path), ("training", training_path), ("evaluation", evaluation_path),
                            ("recovery", recovery_path), ("learning_curve", curve_path),
                            ("checkpoint", checkpoint / "checkpoint.json")):
            source_bindings[mode + "_" + label] = digest(path)
    require(runs["diagonal"]["initial_state"] == runs["zca"]["initial_state"], "initial state differs")
    require(runs["diagonal"]["splits"] == runs["zca"]["splits"], "training or monitor observations differ")
    require(evaluations["diagonal"]["evaluation_input_bindings"]["raw_conditions"]
            == evaluations["zca"]["evaluation_input_bindings"]["raw_conditions"]
            and evaluations["diagonal"]["population"]["observations_sha256"]
            == evaluations["zca"]["population"]["observations_sha256"],
            "evaluation quote vectors differ despite matching generating values")
    require(all(evaluation["evaluation_input_bindings"]["sidecar_applied_once"] is True
                and evaluation["evaluation_input_bindings"]["preprocessing_refit"] is False
                for evaluation in evaluations.values()), "evaluation must apply each frozen transform exactly once")
    require(np.array_equal(cache_truths["diagonal"], cache_truths["zca"])
            and np.array_equal(cache_ids["diagonal"], cache_ids["zca"]), "evaluation datasets or order differ")
    population_c = cache_truths["diagonal"][:, -1].tolist()
    contract = contracts["diagonal"]
    source_bindings.update(task=fingerprint(task), bank=contract["bank_payload_sha256"],
                           initial_state=fingerprint(runs["diagonal"]["initial_state"]),
                           training_inputs=contract["splits"]["train"]["conditions"]["sha256"],
                           evaluation_c=fingerprint(population_c))
    return {
        "format": "heston_whitening_lesson_v1", "schema_version": 1, "scientific_acceptance": False,
        "study": {"date": "2026-11-10", "quote_count": 54, "parameter_names": NAMES, "prior_bounds": BOUNDS,
                  "observation_model": {"family": "gaussian", "sd": "c*h_i", "independent": True, "c_bounds": [.5, 2.]},
                  "conditioning": "noisy quotes divided by fixed observed half-spreads",
                  "architecture": {"family": "CouplingFlow", "design": "spline", "depth": 6,
                                   "conditioner_widths": [128, 128], "dropout": 0.},
                  "distinct_from_current_63_quote_flow_matching": True},
        "training": {"rows": 10000, "monitor_rows": 100, "epochs": 200, "optimizer_updates": 8000,
                     "batch_size": 256, "tail_batch_size": 16, "presentations": 2000000, "seed": 9011,
                     "noise": "fixed noisy rows repeated each pass", "checkpoint_selection": "predeclared_final",
                     "optimizer": contract["optimizer"]},
        "pairing": {"only_contract_differences": different, "same_initial_state": True,
                    "same_training_inputs_and_targets": True, "same_evaluation_datasets_and_order": True},
        "population": {"case_count": 512, "posterior_draws_per_case": 512, "seed": 261112,
                       "c_truth": population_c, "scope": "same 512 historical validation quote sets in both arms"},
        "arms": arms, "source_bindings": source_bindings,
        "qualification": "ZCA improves c recovery in this one matched historical pilot; the final whitened posterior is miscalibrated. This is not an ablation of the current 63-quote FlowMatching checkpoint.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/pilot-results/2026-11-14/heston-whitening-lesson/results.json"))
    args = parser.parse_args()
    record = curate(args.study)
    payload = plotting_json(record) + "\n"
    require(json.loads(payload) == record, "serialization changed plotting values")
    require(not any(token in payload for token in ("/Users/", "/scratch/", "row_id", '"draws"')), "private payload leaked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(json.dumps({"bytes": args.output.stat().st_size, "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()

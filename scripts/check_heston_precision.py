"""Repeat frozen Heston validation checks with more Monte Carlo draws.

This fixed, post-hoc diagnostic uses validation examples 0..15 only. It does
not train, access the reserved test bank, perform SBC, or override a failed
acceptance gate. Exit status zero means the diagnostic completed successfully.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

os.environ.setdefault("KERAS_BACKEND", "jax")

import bayesflow as bf
import jax
import keras
import numpy as np

from sbi4finance import heston, heston_bank, heston_forecast, heston_inference, heston_reference
import check_heston_reference
import train_heston


CASES = 16
DRAWS = 16384
NEURAL_SEEDS = (11101, 11201, 11301)
FINE_REFERENCE_SEEDS = (27101, 27102)
RANGE_KEYS = ("mean_error_in_reference_sd", "relative_std_error", "correlation_error",
              "neural_mean_mcse", "neural_mean_mcse_in_reference_sd",
              "predictive_std_relative_error", "predictive_quantile_error_in_sd")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moment_difference(left, right):
    scale = np.asarray(right["std"])
    return {
        "mean_error_in_reference_sd": ((np.asarray(left["mean"]) - right["mean"]) / scale).tolist(),
        "relative_std_error": (np.asarray(left["std"]) / scale - 1).tolist(),
        "correlation_error": float(left["correlation"] - right["correlation"]),
    }


def compare_draws(joint, reference, task, *, reference_seed, future_seed):
    started = perf_counter()
    reference_moments = check_heston_reference.summary(reference)
    neural_moments = {"mean": joint.mean(axis=0).tolist(),
                      "std": joint.std(axis=0, ddof=1).tolist(),
                      "correlation": float(np.corrcoef(joint.T)[0, 1])}
    reference_draws = check_heston_reference.sample_reference(reference, DRAWS, seed=reference_seed)
    # Future seeds are shared within each pair, independent of posterior draws.
    neural_losses = heston_forecast.simulate_futures(joint, task, seed=future_seed)["relative_loss"]
    reference_losses = heston_forecast.simulate_futures(reference_draws, task, seed=future_seed)["relative_loss"]
    neural_quantiles = np.quantile(neural_losses, [0.05, 0.5, 0.95])
    reference_quantiles = np.quantile(reference_losses, [0.05, 0.5, 0.95])
    neural_loss_std = float(neural_losses.std(ddof=1))
    reference_loss_std = float(reference_losses.std(ddof=1))
    mcse = np.asarray(neural_moments["std"]) / np.sqrt(DRAWS)
    result = {
        "reference": reference_moments, "neural": neural_moments,
        **moment_difference(neural_moments, reference_moments),
        "neural_mean_mcse": mcse.tolist(),
        "neural_mean_mcse_in_reference_sd": (mcse / reference_moments["std"]).tolist(),
        "predictive_std_relative_error": neural_loss_std / reference_loss_std - 1,
        "predictive_quantile_error_in_sd": ((neural_quantiles - reference_quantiles) / reference_loss_std).tolist(),
        "neural_loss_q05_median_q95": neural_quantiles.tolist(),
        "reference_loss_q05_median_q95": reference_quantiles.tolist(),
        "neural_loss_std": neural_loss_std, "reference_loss_std": reference_loss_std,
        "reference_draw_seed": reference_seed, "shared_future_seed": future_seed,
        "seconds": perf_counter() - started,
    }
    return result, reference_draws, neural_losses, reference_losses


def ranges(records, keys=RANGE_KEYS):
    result = {}
    for key in keys:
        values = np.asarray([record[key] for record in records])
        result[key] = {"minimum": values.min(axis=0).tolist(),
                       "maximum": values.max(axis=0).tolist()}
    return result


def aggregate_metrics(cases):
    mean_errors = np.asarray([case["mean_error_in_reference_sd"] for case in cases])
    return {
        "mean_rmse_in_reference_sd": float(np.sqrt(np.mean(mean_errors**2))),
        "maximum_abs_mean_error_in_reference_sd": float(np.abs(mean_errors).max()),
        "maximum_relative_std_error": float(np.abs([case["relative_std_error"] for case in cases]).max()),
        "maximum_correlation_error": float(np.abs([case["correlation_error"] for case in cases]).max()),
        "maximum_predictive_std_relative_error": float(np.abs([case["predictive_std_relative_error"] for case in cases]).max()),
        "maximum_predictive_interval_error_in_sd": float(np.abs([
            np.asarray(case["predictive_quantile_error_in_sd"])[[0, 2]] for case in cases]).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    original_path = root / "docs/results/heston-time-context/neural-validation.json"
    manifest_path = root / "docs/results/heston-inference-followup/bank-manifest.json"
    original = json.loads(original_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(args.config.read_text())
    saved = json.loads((args.checkpoint / "task.json").read_text())
    reference_path = args.references / "report.json"
    reference_report = json.loads(reference_path.read_text())
    heston_bank.validate_config(config)
    if config != manifest["configuration"] or config["root_seed"] != 43022:
        raise ValueError("this diagnostic requires the original root-43022 configuration")
    task = config["task"]
    expected_id = heston_inference.TIME_ADAPTER_ID
    if (saved != {"task": task, "adapter_id": expected_id}
            or original["task"] != task or original["adapter_id"] != expected_id
            or original["split"] != "validation"
            or original["checkpoint_sha256"] != file_hash(args.checkpoint / "model.keras")
            or original["bank_manifest_sha256"] != file_hash(manifest_path)
            or original["reference_report_sha256"] != file_hash(reference_path)
            or original["evaluation"]["pilot_passed"]):
        raise ValueError("checkpoint or original failed validation evidence does not match")
    if (not reference_report["passed"] or reference_report["task"] != task
            or reference_report["split"] != "validation" or len(reference_report["cases"]) != CASES
            or reference_report["bank_manifest_sha256"] != file_hash(manifest_path)
            or original["evaluation"]["thresholds"] != train_heston.THRESHOLDS):
        raise ValueError("reference evidence or unchanged thresholds do not match")

    # No load_bank call: only validation examples 0..15 are generated.
    validation = heston_bank.generate_examples(config, "validation", range(CASES))
    references, source_paths = [], {}
    for case, record in enumerate(reference_report["cases"]):
        identity = str(validation["example_id"][case])
        original_case = original["evaluation"]["reference_cases"][case]
        if record["example_id"] != identity or original_case["example_id"] != identity:
            raise ValueError("reference IDs do not match the fixed validation prefix")
        final_run = record["runs"][-1]
        if (final_run["grid_size"], final_run["particles"]) != (48, 4096):
            raise ValueError("expected the original 48-grid/4096-particle references")
        path = args.references / f"case-{case}.npz"
        with np.load(path, allow_pickle=False) as archive:
            reference = dict(archive)
        if not np.array_equal(reference.pop("observations"), validation["observations"][case]):
            raise ValueError("reference observations differ from regenerated validation inputs")
        moments = check_heston_reference.summary(reference)
        if (moments != final_run["summary"] or moments != original_case["reference"]
                or reference["terminal_particles"].shape != (48, 4096)):
            raise ValueError("saved reference arrays differ from their recorded weighted moments")
        references.append(reference)
        source_paths[f"original-reference/case-{case}.npz"] = path

    args.out.mkdir(parents=True, exist_ok=False)
    workflow = heston_inference.build_workflow(task, adapter_id=expected_id)
    load_started = perf_counter()
    workflow.load_approximator(args.checkpoint / "model.keras")
    load_seconds = perf_counter() - load_started
    conditions = heston_inference.conditioning_data(
        validation["observations"], validation["dt_years"], task, adapter_id=expected_id)
    repeats, case0_neural = [], []
    for neural_seed in NEURAL_SEEDS:
        sampling_started = perf_counter()
        joint = train_heston.joint_array(workflow.sample(
            num_samples=DRAWS, conditions=conditions, batch_size=16, seed=neural_seed), task)
        sampling_seconds = perf_counter() - sampling_started
        case0_neural.append(joint[0].copy())
        cases = []
        for case, reference in enumerate(references):
            comparison, _, _, _ = compare_draws(
                joint[case], reference, task,
                reference_seed=neural_seed + 1000 + case, future_seed=neural_seed + 2000 + case)
            comparison["example_id"] = str(validation["example_id"][case])
            cases.append(comparison)
        repeats.append({"neural_seed": neural_seed, "sampling_seconds": sampling_seconds,
                        "cases": cases, "aggregate_metrics": aggregate_metrics(cases)})
        print(json.dumps({"completed_neural_seed": neural_seed,
                          "aggregate_metrics": repeats[-1]["aggregate_metrics"]}), flush=True)

    fine_records, fine_summaries = [], []
    for fine_index, reference_seed in enumerate(FINE_REFERENCE_SEEDS):
        reference_started = perf_counter()
        fine = heston_reference.reference_posterior(
            validation["observations"][0], validation["dt_years"], task,
            grid_size=96, particles=16384, seed=reference_seed)
        reference_seconds = perf_counter() - reference_started
        moments = check_heston_reference.summary(fine)
        fine_summaries.append(moments)
        comparisons = []
        for repeat_index, neural_seed in enumerate(NEURAL_SEEDS):
            comparison, ref_draws, neural_losses, ref_losses = compare_draws(
                case0_neural[repeat_index], fine, task,
                reference_seed=neural_seed + 30000 + 100 * fine_index,
                future_seed=neural_seed + 40000 + 100 * fine_index)
            comparison["neural_seed"] = neural_seed
            comparisons.append(comparison)
            if fine_index == 1 and repeat_index == 0:
                np.savez(args.out / "case-0-draws.npz", observations=validation["observations"][0],
                         neural_joint=case0_neural[repeat_index], reference_joint=ref_draws,
                         neural_future_loss=neural_losses, reference_future_loss=ref_losses)
        fine_records.append({
            "example_id": "validation:0", "grid_size": 96, "particles": 16384,
            "seed": reference_seed, "seconds": reference_seconds, "summary": moments,
            "difference_from_original": moment_difference(moments, check_heston_reference.summary(references[0])),
            "neural_comparisons": comparisons, "comparison_ranges": ranges(comparisons),
        })
        print(json.dumps({"completed_fine_reference_seed": reference_seed,
                          "difference_from_original": fine_records[-1]["difference_from_original"]}), flush=True)

    source_paths.update({
        "config.json": args.config, "checkpoint/model.keras": args.checkpoint / "model.keras",
        "checkpoint/task.json": args.checkpoint / "task.json", "bank/manifest.json": manifest_path,
        "original-neural-validation.json": original_path, "original-reference/report.json": reference_path,
        "scripts/check_heston_precision.py": Path(__file__),
        "scripts/train_heston.py": Path(train_heston.__file__),
        "scripts/check_heston_reference.py": Path(check_heston_reference.__file__),
        **{f"src/sbi4finance/{module.__name__.split('.')[-1]}.py": Path(module.__file__)
           for module in (heston, heston_bank, heston_forecast, heston_inference, heston_reference)},
    })
    report = {
        "purpose": "post-hoc frozen-checkpoint Monte Carlo precision diagnostic",
        "data_origin": "synthetic", "measure": "P", "split": "validation",
        "case_selection": "unchanged first 16 validation histories; no cases dropped",
        "task": task, "root_seed": config["root_seed"], "adapter_id": expected_id,
        "training_performed": False, "reserved_test_accessed": False, "is_sbc": False,
        "draws_per_case_per_repeat": DRAWS, "neural_seeds": list(NEURAL_SEEDS),
        "forecast_steps": 10, "quantile_order": [0.05, 0.5, 0.95],
        "target_order": ["theta", "signed_terminal_auxiliary"],
        "original_gate_passed": False, "acceptance_status": "unchanged failed; diagnostic only",
        "unchanged_thresholds": original["evaluation"]["thresholds"],
        "original_metrics": original["evaluation"]["metrics"],
        "repeats": repeats,
        "aggregate_metric_ranges": ranges([repeat["aggregate_metrics"] for repeat in repeats],
                                           keys=tuple(train_heston.THRESHOLDS)),
        "case_ranges": [{"example_id": str(validation["example_id"][case]),
                         **ranges([repeat["cases"][case] for repeat in repeats])} for case in range(CASES)],
        "case0_fine_references": fine_records,
        "case0_fine_repeat_difference_second_vs_first": moment_difference(fine_summaries[1], fine_summaries[0]),
        "plot_arrays": {"file": "case-0-draws.npz", "sha256": file_hash(args.out / "case-0-draws.npz"),
                        "neural_seed": NEURAL_SEEDS[0], "reference_filter_seed": FINE_REFERENCE_SEEDS[1],
                        "reference_draw_seed": NEURAL_SEEDS[0] + 30100,
                        "shared_future_seed": NEURAL_SEEDS[0] + 40100},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "bayesflow": bf.__version__, "keras": keras.__version__, "jax": jax.__version__},
        "source_sha256": {key: file_hash(path) for key, path in source_paths.items()},
        "timing_seconds": {"checkpoint_load": load_seconds, "total": perf_counter() - started},
        "limits": "Mean Monte Carlo SE assumes independent neural draws from the frozen approximate posterior; "
                  "it excludes neural approximation and reference error. Three seed ranges are descriptive, "
                  "not confidence intervals. Shared future innovations do not couple posterior draws. "
                  "Finite grid/particle refinement is not an exact posterior certificate. These inspected "
                  "validation histories are not an independent SBC sample or a new held-out acceptance test.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"diagnostic_completed": True, "acceptance_status": report["acceptance_status"],
                      "report": str(args.out / "report.json")}, indent=2))


if __name__ == "__main__":
    main()

"""Train a small joint Heston posterior, or evaluate a frozen saved checkpoint.

First use validation references for development. A separate --checkpoint run
with independently checked test references evaluates the frozen result. All
observations are synthetic; future outcomes are evaluation-only.
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

from sbi4finance import heston_forecast, heston_inference
from sbi4finance.heston_bank import load_bank
from sbi4finance.heston_inference import build_workflow, conditioning_data, training_data
from check_heston_reference import sample_reference, summary


THRESHOLDS = {
    "mean_rmse_in_reference_sd": 0.20,
    "maximum_abs_mean_error_in_reference_sd": 0.40,
    "maximum_relative_std_error": 0.30,
    "maximum_correlation_error": 0.20,
    "maximum_predictive_std_relative_error": 0.15,
    "maximum_predictive_interval_error_in_sd": 0.15,
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def joint_array(samples, task):
    if set(samples) != {"theta", "terminal_state"}:
        raise ValueError("checkpoint must return matched theta and terminal_state draws")
    joint = np.concatenate([samples["theta"], samples["terminal_state"]], axis=-1).astype(np.float64)
    low, high = task["theta_bounds"]
    if not np.isfinite(joint).all() or np.any(joint[..., 0] <= low) or np.any(joint[..., 0] >= high):
        raise ValueError("invalid posterior output; no clipping or selective redraw is allowed")
    return joint


def evaluate(workflow, data, task, reference_dir, reference_report, *, cases, draws, seed, out,
             adapter_id=heston_inference.ADAPTER_ID):
    count = min(cases, len(data["observations"]))
    conditions = conditioning_data(data["observations"][:count], data["dt_years"], task,
                                   adapter_id=adapter_id)
    started = perf_counter()
    joint = joint_array(workflow.sample(num_samples=draws, conditions=conditions,
                                       batch_size=16, seed=seed), task)
    sampled_seconds = perf_counter() - started
    truth = np.concatenate([data["parameters"][:count], data["terminal_state"][:count]], axis=1)
    intervals = np.quantile(joint, [0.05, 0.95], axis=1)
    coverage = ((truth >= intervals[0]) & (truth <= intervals[1])).mean(axis=0)
    ranks = (joint < truth[:, None, :]).sum(axis=1)
    # Randomized finite-draw ranks; DKW is a diagnostic here, not validation of
    # all joint features. Development/validation cases are not held-out claims.
    rank_uniform = (ranks + np.random.default_rng(seed + 1).random(ranks.shape)) / (draws + 1)
    ordered = np.sort(rank_uniform, axis=0)
    ks = np.maximum(np.max(np.arange(1, count + 1)[:, None] / count - ordered, axis=0),
                    np.max(ordered - np.arange(count)[:, None] / count, axis=0))
    dkw = float(np.sqrt(np.log(4 / 0.05) / (2 * count)))
    reference_comparisons = []
    for case, record in enumerate(reference_report["cases"]):
        if case >= count or record["example_id"] != str(data["example_id"][case]):
            raise ValueError("reference case selection does not match the evaluation split")
        with np.load(reference_dir / f"case-{case}.npz", allow_pickle=False) as saved:
            reference = dict(saved)
        if not np.array_equal(reference.pop("observations"), data["observations"][case]):
            raise ValueError("reference observation differs from the evaluated history")
        exact = summary(reference)
        approximate = joint[case]
        mean_error = (approximate.mean(axis=0) - exact["mean"]) / exact["std"]
        std_error = approximate.std(axis=0, ddof=1) / exact["std"] - 1
        correlation_error = np.corrcoef(approximate.T)[0, 1] - exact["correlation"]
        # Same fresh forward innovations reduce noise in this distribution
        # comparison. The posterior draws remain independent of those inputs.
        reference_draws = sample_reference(reference, draws, seed=seed + 1000 + case)
        losses = heston_forecast.simulate_futures(approximate, task, seed=seed + 2000 + case)["relative_loss"]
        reference_losses = heston_forecast.simulate_futures(reference_draws, task, seed=seed + 2000 + case)["relative_loss"]
        predictive_std_error = losses.std(ddof=1) / reference_losses.std(ddof=1) - 1
        predictive_interval_error = ((np.quantile(losses, [0.05, 0.95])
                                     - np.quantile(reference_losses, [0.05, 0.95])) / reference_losses.std(ddof=1))
        reference_comparisons.append({
            "example_id": record["example_id"], "reference": exact,
            "neural_mean": approximate.mean(axis=0).tolist(),
            "neural_std": approximate.std(axis=0, ddof=1).tolist(),
            "mean_error_in_reference_sd": mean_error.tolist(), "relative_std_error": std_error.tolist(),
            "correlation_error": float(correlation_error),
            "predictive_std_relative_error": float(predictive_std_error),
            "predictive_interval_error_in_sd": predictive_interval_error.tolist(),
            "neural_negative_state_probability": float(np.mean(approximate[:, 1] < 0)),
            "reference_negative_state_probability": float(reference["grid_probabilities"] @ (reference["terminal_particles"] < 0).mean(axis=1)),
        })
    mean_errors = np.asarray([r["mean_error_in_reference_sd"] for r in reference_comparisons])
    metrics = {
        "mean_rmse_in_reference_sd": float(np.sqrt(np.mean(mean_errors**2))),
        "maximum_abs_mean_error_in_reference_sd": float(np.max(np.abs(mean_errors))),
        "maximum_relative_std_error": float(np.max(np.abs([r["relative_std_error"] for r in reference_comparisons]))),
        "maximum_correlation_error": float(max(abs(r["correlation_error"]) for r in reference_comparisons)),
        "maximum_predictive_std_relative_error": float(max(abs(r["predictive_std_relative_error"]) for r in reference_comparisons)),
        "maximum_predictive_interval_error_in_sd": float(np.max(np.abs([r["predictive_interval_error_in_sd"] for r in reference_comparisons]))),
    }
    reference_passed = all(metrics[key] <= threshold for key, threshold in THRESHOLDS.items())
    # One independently drawn future per held-out history. This evaluates the
    # predictive mixture, not the distribution/average of conditional VaRs.
    realized_loss = heston_forecast.simulate_futures(truth, task, seed=seed + 3000)["relative_loss"]
    forecast_intervals = np.empty((count, 2))
    forecast_started = perf_counter()
    for case in range(count):
        losses = heston_forecast.simulate_futures(joint[case], task, seed=seed + 4000 + case)["relative_loss"]
        forecast_intervals[case] = np.quantile(losses, [0.05, 0.95])
    predictive_coverage = np.mean((realized_loss >= forecast_intervals[:, 0]) & (realized_loss <= forecast_intervals[:, 1]))
    np.savez(out / "checks.npz", example_id=data["example_id"][:count], truth=truth,
             posterior_mean=joint.mean(axis=1), posterior_std=joint.std(axis=1, ddof=1),
             interval_low=intervals[0], interval_high=intervals[1], ranks=ranks,
             evaluation_future_loss=realized_loss, future_interval=forecast_intervals)
    return {
        "cases": count, "draws_per_case": draws, "seed": seed,
        "target_order": ["theta", "signed_terminal_auxiliary"],
        "reference_cases": reference_comparisons, "metrics": metrics, "thresholds": THRESHOLDS,
        "reference_comparisons_passed": bool(reference_passed),
        "coverage_90": coverage.tolist(), "coverage_nominal_binomial_se": float(np.sqrt(0.09 / count)),
        "rank_uniformity_max_ecdf_error": ks.tolist(), "rank_dkw_bound_two_targets": dkw,
        "rank_diagnostic_passed": bool(np.all(ks <= dkw)),
        "predictive_loss_interval_90_coverage": float(predictive_coverage),
        "negative_state_draw_fraction": float(np.mean(joint[..., 1] < 0)),
        "forecast_steps": 10, "forecast_horizon_years": 10 * task["dt_years"],
        "timing_seconds": {"posterior_sampling_including_compile": sampled_seconds,
                           "predictive_intervals": perf_counter() - forecast_started},
        "pilot_passed": bool(reference_passed and np.all(ks <= dkw)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--checkpoint", type=Path, help="evaluate frozen model.keras + task.json; no training")
    parser.add_argument("--density-estimator", choices=("flow_matching", "spline"), default="flow_matching",
                        help="training architecture; --checkpoint restores its own saved architecture")
    parser.add_argument("--explicit-time", action="store_true",
                        help="train with known end times and globally scaled Time2Vec inputs; frozen checkpoints use their saved adapter")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--seed", type=int, default=871)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.cases) < 1 or args.draws < 64 or args.seed < 0:
        parser.error("positive counts, draws>=64 and a nonnegative seed are required")
    manifest, splits = load_bank(args.bank)
    task = manifest["configuration"]["task"]
    reference_report = json.loads((args.references / "report.json").read_text())
    if (not reference_report["passed"] or reference_report["task"] != task
            or reference_report["bank_manifest_sha256"] != file_hash(args.bank / "manifest.json")):
        raise ValueError("a passing reference report for this exact bank/task is required")
    split = reference_report["split"]
    if args.checkpoint is None and split != "validation":
        raise ValueError("develop on validation; evaluate test only with a frozen --checkpoint")
    adapter_id = heston_inference.TIME_ADAPTER_ID if args.explicit_time else heston_inference.ADAPTER_ID
    if args.checkpoint is not None:
        checkpoint_design = json.loads((args.checkpoint / "task.json").read_text())
        if set(checkpoint_design) != {"task", "adapter_id"} or checkpoint_design["task"] != task:
            raise ValueError("saved checkpoint task differs in prior, parameters, initialization or observation design")
        adapter_id = checkpoint_design["adapter_id"]
    else:
        checkpoint_design = {"task": task, "adapter_id": adapter_id}
    heston_inference._check_adapter(adapter_id)
    args.out.mkdir(parents=True, exist_ok=False)
    keras.utils.set_random_seed(args.seed)
    workflow = build_workflow(task, args.out if args.checkpoint is None else None,
                              density_estimator=args.density_estimator, adapter_id=adapter_id)
    training_started = perf_counter()
    if args.checkpoint is None:
        history = workflow.fit_offline(data=training_data(splits["train"], task, adapter_id=adapter_id),
                                       validation_data=training_data(splits["validation"], task, adapter_id=adapter_id),
                                       epochs=args.epochs, batch_size=args.batch_size, verbose=2)
        workflow.load_approximator()
        (args.out / "task.json").write_text(json.dumps(checkpoint_design, indent=2) + "\n")
        history_values = {key: [float(x) for x in value] for key, value in history.history.items()}
        checkpoint = args.out / "model.keras"
    else:
        checkpoint = args.checkpoint / "model.keras"
        workflow.load_approximator(checkpoint)
        history_values = None
    training_seconds = perf_counter() - training_started
    small = conditioning_data(splits[split]["observations"][:2], splits[split]["dt_years"], task,
                              adapter_id=adapter_id)
    original = joint_array(workflow.sample(num_samples=64, conditions=small, seed=args.seed + 1), task)
    reloaded = build_workflow(task, adapter_id=adapter_id)
    reloaded.load_approximator(checkpoint)
    repeated = joint_array(reloaded.sample(num_samples=64, conditions=small, seed=args.seed + 1), task)
    if not np.array_equal(original, repeated):
        raise RuntimeError("same-seed checkpoint reload changed the joint posterior draws")
    evaluation = evaluate(workflow, splits[split], task, args.references, reference_report,
                          cases=args.cases, draws=args.draws, seed=args.seed + 2, out=args.out,
                          adapter_id=adapter_id)
    report = {
        "data_origin": "synthetic", "measure": "P", "evidence_scope": "S bounded joint-inference pilot",
        "task": task, "split": split, "frozen_checkpoint_evaluation": args.checkpoint is not None,
        "bank_manifest_sha256": file_hash(args.bank / "manifest.json"),
        "reference_report_sha256": file_hash(args.references / "report.json"),
        "checkpoint_sha256": file_hash(checkpoint),
        "source_sha256": {"train_heston.py": file_hash(__file__),
                          "heston_inference.py": file_hash(heston_inference.__file__),
                          "heston_forecast.py": file_hash(heston_forecast.__file__)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "bayesflow": bf.__version__, "keras": keras.__version__, "jax": jax.__version__},
        "inference_network_class": type(workflow.inference_network).__name__,
        "adapter_id": adapter_id,
        "checkpoint_load_seconds": training_seconds if history_values is None else None,
        "training": {"performed": True, "seed": args.seed, "epochs": args.epochs,
                     "batch_size": args.batch_size, "history": history_values,
                     "selection": "minimum validation loss", "normalization": "training batches only",
                     "adapter_id": adapter_id,
                     "architecture": "causal GRU32 summary16 + eight observed window moments; adapter_id specifies time context; full density architecture saved in model.keras",
                     "inference_network_class": type(workflow.inference_network).__name__,
                     "density_estimator_requested_for_training": args.density_estimator if history_values is not None else None,
                     "seconds_including_compile_or_load": training_seconds} if history_values is not None else {"performed": False},
        "same_seed_reload_exact": True, "evaluation": evaluation,
        "limits": "Two-target, fixed-parameter, one-step discrete Heston pilot. Validation results are development only; finite-particle reference, finite draws and finite histories limit conclusions. No full SVJ, continuous-time, market, regulatory-risk or tutorial-wide acceptance.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in evaluation.items() if key != "reference_cases"}, indent=2))
    if not evaluation["pilot_passed"]:
        raise SystemExit("Joint inference pilot failed; checkpoint remains provisional. Inspect saved diagnostics.")


if __name__ == "__main__":
    main()

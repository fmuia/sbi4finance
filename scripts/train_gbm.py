"""Load a saved GBM bank, train, reload and check against its analytic posterior."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import bayesflow as bf
import jax
import keras
import numpy as np

from sbi4finance.gbm import drift_posterior
from sbi4finance import gbm_inference
from sbi4finance.gbm_bank import load_bank
from sbi4finance.gbm_inference import build_workflow, conditioning_data, training_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new checkpoint/report directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=194)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--case-batch-size", type=int, default=16, help="bound posterior-sampling memory")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.case_batch_size) < 1 or args.draws < 2 or args.seed < 0:
        parser.error("epochs and batch sizes must be positive; draws >= 2; seed nonnegative")
    start = time.perf_counter()
    manifest, splits = load_bank(args.bank)
    task = manifest["configuration"]["task"]
    train = training_data(splits["train"], task)
    validation = training_data(splits["validation"], task)
    args.out.mkdir(parents=True, exist_ok=False)
    keras.utils.set_random_seed(args.seed)
    workflow = build_workflow(args.out)
    ready = time.perf_counter()
    history = workflow.fit_offline(data=train, validation_data=validation,
                                   epochs=args.epochs, batch_size=args.batch_size, verbose=0)
    trained = time.perf_counter()
    # The best validation epoch may differ from the final in-memory epoch.
    workflow.load_approximator()
    test = splits["test"]
    conditions = conditioning_data(test["observations"], test["dt_years"], task)
    sampled_start = time.perf_counter()
    draws = workflow.sample(num_samples=args.draws, conditions=conditions,
                            batch_size=args.case_batch_size, seed=args.seed + 1)["mu"][:, :, 0]
    sampled_end = time.perf_counter()
    restored = build_workflow()
    restored.load_approximator(args.out / "model.keras")
    small = {key: value[:2] for key, value in conditions.items()}
    original_draws = workflow.sample(num_samples=128, conditions=small, seed=args.seed + 2)["mu"]
    restored_draws = restored.sample(num_samples=128, conditions=small, seed=args.seed + 2)["mu"]
    if not np.array_equal(original_draws, restored_draws):
        raise RuntimeError("same-seed checkpoint reload changed posterior draws")
    exact_mean, exact_std = drift_posterior(test["observations"], task["volatility"], test["dt_years"],
                                           prior_mean=task["prior_mean"], prior_std=task["prior_std"])
    standardized_mean_error = (draws.mean(axis=1) - exact_mean) / exact_std
    relative_std_error = draws.std(axis=1, ddof=1) / exact_std - 1
    low, high = np.quantile(draws, [0.05, 0.95], axis=1)
    coverage = np.mean((test["parameters"][:, 0] >= low) & (test["parameters"][:, 0] <= high))
    exact_covered = np.abs(test["parameters"][:, 0] - exact_mean) <= 1.6448536269514722 * exact_std
    reference_mass = np.asarray([
        0.5 * (math.erf((hi - mean) / (std * math.sqrt(2)))
               - math.erf((lo - mean) / (std * math.sqrt(2))))
        for lo, hi, mean, std in zip(low, high, exact_mean, exact_std)
    ])
    metrics = {"mean_rmse_in_reference_sd": float(np.sqrt(np.mean(standardized_mean_error**2))),
               "max_abs_mean_error_in_reference_sd": float(np.max(np.abs(standardized_mean_error))),
               "median_abs_relative_std_error": float(np.median(np.abs(relative_std_error))),
               "coverage_90": float(coverage), "exact_posterior_coverage_90_same_cases": float(exact_covered.mean()),
               "mean_exact_mass_of_inferred_90_intervals": float(reference_mass.mean())}
    # Small predeclared pilot gate; this is not full SBC or tutorial acceptance.
    passed = (metrics["mean_rmse_in_reference_sd"] <= 0.10
              and metrics["max_abs_mean_error_in_reference_sd"] <= 0.30
              and metrics["median_abs_relative_std_error"] <= 0.10)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    report = {
        "data_origin": "synthetic", "evidence_scope": "S", "measure": "P",
        "scope": "local saved-bank/training/reload and conditional moment pilot; not full SBC, market adequacy or cluster evidence",
        "bank_manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((args.out / "model.keras").read_bytes()).hexdigest(),
        "source_sha256": {"train_gbm.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "gbm_inference.py": hashlib.sha256(Path(gbm_inference.__file__).read_bytes()).hexdigest()},
        "bank_configuration": manifest["configuration"],
        "versions": {"bayesflow": bf.__version__, "keras": keras.__version__,
                     "jax": jax.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "training": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                     "checkpoint": "model.keras", "selection": "minimum validation loss", "normalization": "training batches only",
                     "adapter": "gbm_sum_log_returns_v1", "network": "FlowMatching widths=[32,32], integration steps=16"},
        "evaluation": {"held_out_cases": len(draws), "draws_per_case": args.draws,
                       "sampling_seed": args.seed + 1, "metrics": metrics,
                       "sampling_case_batch_size": args.case_batch_size,
                       "thresholds": {"mean_rmse_in_reference_sd": 0.10, "max_abs_mean_error_in_reference_sd": 0.30,
                                      "median_abs_relative_std_error": 0.10},
                       "conditional_moment_pilot_passed": passed, "same_seed_reload_exact": True,
                       "coverage_binomial_se_at_nominal": float(np.sqrt(0.9 * 0.1 / len(draws))),
                       "mean_draw_mc_se_in_reference_sd_approx": float(1 / np.sqrt(args.draws))},
        "timing_seconds": {"load_and_setup": ready - start, "training_including_compile": trained - ready,
                           "posterior_sampling_including_compile": sampled_end - sampled_start},
        "memory": {"bank_array_bytes": sum(a.nbytes for s in splits.values() for a in s.values()),
                   "process_peak_rss_bytes": peak_bytes},
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    np.savez(args.out / "reference_checks.npz", example_id=test["example_id"], exact_mean=exact_mean,
             exact_std=exact_std, inferred_mean=draws.mean(axis=1), inferred_std=draws.std(axis=1, ddof=1),
             interval_low=low, interval_high=high, exact_interval_mass=reference_mass)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["evaluation"], indent=2))
    if not passed:
        raise SystemExit("Conditional moment pilot failed; inspect the saved report before using this checkpoint.")


if __name__ == "__main__":
    main()

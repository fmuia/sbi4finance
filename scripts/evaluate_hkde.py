"""Audit a frozen HKDE checkpoint against conditional grids and marginal SBC."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from sbi4finance import hkde_diagnostics as diagnostics, hkde_inference as inference
from sbi4finance import hkde_reference as ref, option_bank
from check_hkde_reference import compare, jsonable, load_grid, sha


def joint(samples):
    return np.concatenate([samples[name] for name in inference.PARAMETER_NAMES], axis=-1).astype(np.float64)


def aggregate(cases):
    mean_errors = np.asarray([c["mean_error_in_ref_sd"] for c in cases])
    masses = np.asarray([c["reference_mass_of_90_interval"] for c in cases]).mean(axis=0)
    return {"mean_rmse_in_ref_sd": float(np.sqrt(np.mean(mean_errors**2))),
            "max_abs_mean_error_in_ref_sd": float(np.max(np.abs(mean_errors))),
            "max_abs_relative_std_error": float(np.max(np.abs([c["relative_std_error"] for c in cases]))),
            "max_abs_correlation_error": float(np.max(np.abs([c["correlation_error"] for c in cases]))),
            "max_abs_target_quantile_error_currency": float(np.max(np.abs([c["target_quantile_error_currency"] for c in cases]))),
            "mean_reference_mass_of_90_interval_abs_error": float(np.max(np.abs(masses - 0.9)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("configs/hkde_inference_protocol.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(Path(protocol["bank_config"]).read_text())
    task, evaluation = config["task"], protocol["evaluation"]
    training = json.loads((args.checkpoint / "training.json").read_text())
    checkpoint_hash = sha(args.checkpoint / "model.keras")
    reference_report = json.loads((args.reference / "reference.json").read_text())
    reference_report_hash = sha(args.reference / "reference.json")
    if (training["protocol_sha256"] != sha(args.protocol)
            or training["bank_configuration"] != config or training["checkpoint_sha256"] != checkpoint_hash
            or not training["same_seed_reload_exact"] or not reference_report["passed"]
            or reference_report["protocol_sha256"] != sha(args.protocol)
            or training["reference_report_sha256"] != reference_report_hash
            or reference_report["task"] != task
            or reference_report["reference_grid_sha256"] != sha(args.reference / "reference-grid.npz")
            or json.loads((args.checkpoint / "task.json").read_text()) != task):
        raise ValueError("matching frozen task, checkpoint, protocol and accepted reference required")
    expected_sources = {"check_hkde_reference.py", "hkde_reference.py", *option_bank.source_files("hkde")}
    if set(reference_report["source_sha256"]) != expected_sources:
        raise ValueError("accepted reference has an incomplete or unexpected source set")
    for name, digest in reference_report["source_sha256"].items():
        source = (Path(__file__).with_name(name) if name == "check_hkde_reference.py"
                  else Path(option_bank.__file__).with_name(name))
        if digest != sha(source):
            raise ValueError(f"accepted reference source changed: {name}")
    for name, path in (("hkde_inference.py", inference.__file__),):
        if training["source_sha256"][name] != sha(path):
            raise ValueError("checkpoint inference source changed")
    if args.split == "test":
        if args.validation_report is None:
            parser.error("--split test requires the matching passed --validation-report")
        previous = json.loads(args.validation_report.read_text())
        if (not previous["passed"] or previous["split"] != "validation"
                or previous["checkpoint_sha256"] != checkpoint_hash
                or previous["protocol_sha256"] != sha(args.protocol)
                or previous["bank_manifest_sha256"] != training["bank_manifest_sha256"]
                or previous["reference_report_sha256"] != reference_report_hash):
            raise ValueError("reserved test cannot select a new candidate or bypass failed validation")
    manifest, splits = option_bank.load_bank(args.bank)
    if manifest["configuration"] != config or sha(args.bank / "manifest.json") != training["bank_manifest_sha256"]:
        raise ValueError("evaluation bank differs from training record")
    grid, saved = load_grid(args.reference / "reference-grid.npz", task, protocol["targets"])
    sigma, probabilities = task["context"]["quote_noise_std"], evaluation["quantiles"]
    final_reference = None
    if args.split == "validation":
        lo, hi = protocol["audit_validation_ids"]
        audit = {key: value[lo:hi] for key, value in splits["validation"].items()}
        observations, truth, case_ids, weights = [saved[key] for key in ("observations", "truth", "case_ids", "weights")]
    else:
        audit = splits["test"]
        ids = protocol["final_test"]["conditional_ids"]
        observations, truth, case_ids = [audit[key][ids] for key in ("observations", "parameters", "example_id")]
        previous_size = reference_report["grid_comparisons"][-1]["coarse"]
        coarse = ref.build_grid(task, previous_size, protocol["targets"])
        differences, _, weights = compare(coarse, grid, observations, sigma, probabilities)
        metrics = {key: max(row[key] for row in differences) for key in differences[0]}
        final_reference = {"metrics": metrics, "passed": all(value <= protocol["reference"]["gates"][key] for key, value in metrics.items())}
        if not final_reference["passed"]:
            raise ValueError("reserved-test reference resolution failed; no neural interpretation or tuning")
    args.out.mkdir(parents=True, exist_ok=False)
    workflow = inference.build_workflow(task)
    workflow.load_approximator(args.checkpoint / "model.keras")
    loaded = time.perf_counter()
    draws = joint(workflow.sample(num_samples=evaluation["conditional_draws"],
                                  conditions=inference.conditioning_data(observations, task),
                                  batch_size=8, seed=evaluation["sample_seed"]))
    conditional_sampled = time.perf_counter()
    cases, summaries, price_draws = [], [], []
    for index, (one_draws, one_weights) in enumerate(zip(draws, weights, strict=True)):
        prices = ref.price_parameters(task, one_draws, protocol["targets"])
        summary = ref.summarize(grid, one_weights, probabilities)
        metrics = diagnostics.conditional_metrics(one_draws, summary, prices, probabilities=probabilities)
        metrics["case_id"] = str(case_ids[index])
        cases.append(metrics)
        summaries.append(summary)
        price_draws.append(prices)
        print(f"Priced conditional case {index + 1}/{len(draws)}", flush=True)
    priced = time.perf_counter()
    audit_draws = joint(workflow.sample(num_samples=evaluation["sbc_draws"],
                                        conditions=inference.conditioning_data(audit["observations"], task),
                                        batch_size=16, seed=evaluation["sample_seed"] + 1))
    sbc = diagnostics.randomized_sbc(audit_draws, audit["parameters"], seed=evaluation["rank_randomization_seed"],
                                    alpha=evaluation["sbc_family_alpha"])
    sbc_finished = time.perf_counter()
    metrics = aggregate(cases)
    gates = {key: value <= evaluation["gates"][key] for key, value in metrics.items()}
    gates["marginal_sbc"] = sbc["rank_diagnostic_passed"]
    bounds = np.asarray(task["prior_bounds"])
    prior_mean, prior_std = bounds.mean(axis=1), np.diff(bounds)[:, 0] / np.sqrt(12)
    ref_mean, ref_std = np.array([s["mean"] for s in summaries]), np.array([s["std"] for s in summaries])
    prior_errors = (prior_mean - ref_mean) / ref_std
    prior = {"mean": prior_mean, "std": prior_std, "correlation": 0.0,
             "mean_rmse_in_ref_sd": float(np.sqrt(np.mean(prior_errors**2))),
             "mean_error_in_ref_sd": prior_errors,
             "reference_std_over_prior_std": ref_std / prior_std,
             "target_quantiles": ref.weighted_quantiles(grid["target_prices"], grid["quadrature_weights"], probabilities)}
    # Conditional reference variability at the same sample size; cached grid
    # prices preserve each reference draw's joint target dependence.
    repetitions = evaluation["reference_repeated_draws"]
    rng, null_metrics = np.random.default_rng(repetitions["seed"]), []
    for _ in range(repetitions["replicates"]):
        repeated_cases = []
        for summary, one_weights in zip(summaries, weights, strict=True):
            indices = rng.choice(len(one_weights), size=evaluation["conditional_draws"], p=one_weights)
            repeated_cases.append(diagnostics.conditional_metrics(grid["parameters"][indices], summary,
                                  grid["target_prices"][indices], probabilities=probabilities))
        null_metrics.append(aggregate(repeated_cases))
    null_ranges = {key: np.quantile([m[key] for m in null_metrics], [.025, .5, .975]) for key in metrics}
    # Rank arrays live in the compact checks artifact; the JSON keeps summaries.
    ranks, randomized = [np.asarray(sbc.pop(key)) for key in ("ranks", "randomized_ranks")]
    checks_path = args.out / f"{args.split}-checks.npz"
    np.savez_compressed(checks_path, case_ids=case_ids, observations=observations, truth=truth,
                        conditional_draws=draws, target_draws=np.asarray(price_draws),
                        reference_mean=ref_mean, reference_std=ref_std,
                        reference_correlation=[s["correlation"] for s in summaries],
                        reference_target_quantiles=[s["target_quantiles"] for s in summaries],
                        audit_ids=audit["example_id"], audit_truth=audit["parameters"],
                        ranks=ranks, randomized_ranks=randomized)
    report = {
        "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q", "split": args.split,
        "protocol_sha256": sha(args.protocol), "checkpoint_sha256": checkpoint_hash,
        "reference_report_sha256": reference_report_hash,
        "bank_manifest_sha256": sha(args.bank / "manifest.json"), "checks_sha256": sha(checks_path),
        "passed": bool(all(gates.values())), "metrics": metrics, "thresholds": evaluation["gates"], "gates": gates,
        "case_ids": case_ids, "conditional_draws_per_case": len(draws[0]), "cases": cases,
        "sbc": sbc, "prior_baseline": prior,
        "reference_draw_variability": {"replicates": repetitions["replicates"], "seed": repetitions["seed"],
                                       "metric_025_50_975_percentiles": null_ranges,
                                       "scope": "finite draws from discrete reference grid; contextual only, thresholds unchanged"},
        "mean_reference_mass_of_90_interval_by_parameter": np.mean([c["reference_mass_of_90_interval"] for c in cases], axis=0),
        "timing_seconds": {"loading_including_reference": loaded - started,
                           "conditional_sampling_including_compile": conditional_sampled - loaded,
                           "conditional_price_propagation_and_comparison": priced - conditional_sampled,
                           "sbc_sampling_and_checks": sbc_finished - priced, "total": time.perf_counter() - started},
        "source_sha256": {"evaluate_hkde.py": sha(__file__), "hkde_diagnostics.py": sha(diagnostics.__file__),
                          "hkde_reference.py": sha(ref.__file__), "hkde_inference.py": sha(inference.__file__)},
        "final_test_reference_check": final_reference,
        "reserved_test_evaluated": args.split == "test",
        "limits": ["SBC is marginal prior-predictive calibration, not conditional or joint acceptance.",
                   "Price distributions are Q model-price uncertainty, not future realized payoffs or P risks.",
                   "Numerical reference and finite-draw uncertainty remain visible; no real-market evidence.",
                   "Single frozen architecture/seed/budget; preserve failed gates without automatic retuning."],
    }
    destination = args.out / f"neural-{args.split}.json"
    destination.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "metrics": metrics, "gates": gates, "sbc": sbc}, indent=2))
    if not report["passed"]:
        raise SystemExit("Frozen neural gates failed; checkpoint remains provisional and final test stays reserved.")


if __name__ == "__main__":
    main()

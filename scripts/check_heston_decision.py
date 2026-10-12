"""Audit saved Heston #25 development evidence; never simulate, train or open a bank.

This checks arithmetic and artifact identities, not scientific acceptance.
Only the root-43022 validation summaries/arrays and saved precision draws are read.
Use the pinned environment and a new --out JSON path.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def audit(root):
    sources = {}

    def source(name):
        path = root / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(name):
        return json.loads(source(name).read_text())

    config = read("configs/tasks/heston_inference_followup.json")
    bank_name = "docs/results/heston-inference-followup/bank-manifest.json"
    bank = read(bank_name)
    reference_name = "docs/results/heston-inference-followup/reference-validation.json"
    reference = read(reference_name)
    ids = [f"validation:{i}" for i in range(16)]
    require(config["root_seed"] == 43022 and bank["configuration"] == config,
            "Expected the frozen root-43022 configuration")
    require(reference["split"] == "validation" and reference["task"] == config["task"]
            and reference["bank_manifest_sha256"] == sources[bank_name]
            and [case["example_id"] for case in reference["cases"]] == ids,
            "Reference task, bank or case selection differs")
    for case in reference["cases"]:
        for label in ("coarse_to_fine", "independent_fine_repeat"):
            require(case[label]["passed"] and all(
                case[label][key] <= limit for key, limit in reference["thresholds"].items()
            ), "A recorded reference stability check failed")
        require(case["passed"], "Reference case status differs")
    require(reference["passed"], "Reference report status differs")

    candidates = {}
    reports = {}
    for label, directory, checkpoint in (
        ("v2", "heston-time-context", "heston-time-context-provisional"),
        ("v3", "heston-state-transform", "heston-asinh-state-provisional"),
    ):
        report_name = f"docs/results/{directory}/neural-validation.json"
        report = reports[label] = read(report_name)
        evaluation = report["evaluation"]
        checkpoint_name = f"checkpoints/{checkpoint}/model.keras"
        source(checkpoint_name)
        design = read(f"checkpoints/{checkpoint}/task.json")
        require(report["split"] == "validation" and report["task"] == config["task"]
                and report["bank_manifest_sha256"] == sources[bank_name]
                and report["reference_report_sha256"] == sources[reference_name]
                and report["checkpoint_sha256"] == sources[checkpoint_name]
                and design == {"task": config["task"], "adapter_id": report["adapter_id"]},
                "Candidate artifact identity differs")
        cases = evaluation["reference_cases"]
        require([case["example_id"] for case in cases] == ids, "All 16 cases must remain")
        with np.load(source(f"docs/results/{directory}/validation-checks.npz"), allow_pickle=False) as saved:
            arrays = dict(saved)
        count, draws = evaluation["cases"], evaluation["draws_per_case"]
        require((count, draws) == (256, 2048) and arrays["example_id"].tolist()
                == [f"validation:{i}" for i in range(count)], "Validation array identity differs")
        means = np.asarray([case["reference"]["mean"] for case in cases])
        stds = np.asarray([case["reference"]["std"] for case in cases])
        for case, ref in zip(cases, reference["cases"]):
            require(case["reference"] == ref["runs"][-1]["summary"], "Reference summary differs")
        close(arrays["posterior_mean"][:16], [case["neural_mean"] for case in cases])
        close(arrays["posterior_std"][:16], [case["neural_std"] for case in cases])
        mean_errors = (arrays["posterior_mean"][:16] - means) / stds
        sd_errors = arrays["posterior_std"][:16] / stds - 1
        close(mean_errors, [case["mean_error_in_reference_sd"] for case in cases])
        close(sd_errors, [case["relative_std_error"] for case in cases])
        metrics = {
            "mean_rmse_in_reference_sd": float(np.sqrt(np.mean(mean_errors**2))),
            "maximum_abs_mean_error_in_reference_sd": float(np.abs(mean_errors).max()),
            "maximum_relative_std_error": float(np.abs(sd_errors).max()),
            "maximum_correlation_error": float(np.abs([c["correlation_error"] for c in cases]).max()),
            "maximum_predictive_std_relative_error": float(np.abs([c["predictive_std_relative_error"] for c in cases]).max()),
            "maximum_predictive_interval_error_in_sd": float(np.abs([c["predictive_interval_error_in_sd"] for c in cases]).max()),
        }
        for key, value in metrics.items():
            close(value, evaluation["metrics"][key])
        coverage = ((arrays["truth"] >= arrays["interval_low"]) &
                    (arrays["truth"] <= arrays["interval_high"])).mean(axis=0)
        loss_coverage = ((arrays["evaluation_future_loss"] >= arrays["future_interval"][:, 0]) &
                         (arrays["evaluation_future_loss"] <= arrays["future_interval"][:, 1])).mean()
        close(coverage, evaluation["coverage_90"])
        close(loss_coverage, evaluation["predictive_loss_interval_90_coverage"])
        ranks = arrays["ranks"]
        require(ranks.shape == (count, 2) and np.issubdtype(ranks.dtype, np.integer)
                and np.all((ranks >= 0) & (ranks <= draws)), "Invalid saved ranks")
        ordered = np.sort((ranks + np.random.default_rng(evaluation["seed"] + 1).random(ranks.shape)) / (draws + 1), axis=0)
        ks = np.maximum(np.max(np.arange(1, count + 1)[:, None] / count - ordered, axis=0),
                        np.max(ordered - np.arange(count)[:, None] / count, axis=0))
        bound = float(np.sqrt(np.log(4 / 0.05) / (2 * count)))
        close(ks, evaluation["rank_uniformity_max_ecdf_error"])
        close(bound, evaluation["rank_dkw_bound_two_targets"])
        failed = [key for key, value in metrics.items() if value > evaluation["thresholds"][key]]
        require(bool(failed) and not evaluation["reference_comparisons_passed"]
                and not evaluation["pilot_passed"] and evaluation["rank_diagnostic_passed"]
                and bool(np.all(ks <= bound)), "Recorded provisional status differs")
        candidates[label] = {
            "adapter_id": report["adapter_id"], "conditional_cases": ids,
            "metrics": metrics, "thresholds": evaluation["thresholds"], "failed_conditional_gates": failed,
            "coverage_90": coverage.tolist(), "predictive_coverage_90": float(loss_coverage),
            "development_sbc_ecdf_error": ks.tolist(), "development_sbc_bound": bound,
            "scientific_acceptance": False,
        }
    require(candidates["v2"]["thresholds"] == candidates["v3"]["thresholds"], "Thresholds changed")

    prior = read("docs/results/heston-diagnostic-review/prior-comparison.json")
    require(prior["neural_report_sha256"] == sources["docs/results/heston-time-context/neural-validation.json"]
            and prior["bank_manifest_sha256"] == sources[bank_name], "Prior binding differs")
    prior_rmse = float(np.sqrt(np.mean(((np.asarray(prior["prior_mean"]) - means) / stds)**2)))
    close(prior_rmse, prior["prior_copy_reference_mean_rmse_in_sd"])
    close(candidates["v2"]["metrics"]["mean_rmse_in_reference_sd"], prior["neural_reference_mean_rmse_in_sd"])

    precision = read("docs/results/heston-precision/report.json")
    require(precision["split"] == "validation" and precision["root_seed"] == 43022
            and precision["task"] == config["task"] and not precision["training_performed"]
            and not precision["reserved_test_accessed"] and not precision["is_sbc"]
            and not precision["original_gate_passed"], "Precision scope differs")
    for recorded, current in {
        "original-neural-validation.json": "docs/results/heston-time-context/neural-validation.json",
        "bank/manifest.json": bank_name, "original-reference/report.json": reference_name,
        "checkpoint/model.keras": "checkpoints/heston-time-context-provisional/model.keras",
    }.items():
        require(precision["source_sha256"][recorded] == sources[current], "Precision binding differs")
    artifact = precision["plot_arrays"]
    require(artifact["file"] == "case-0-draws.npz" and artifact["neural_seed"] == 11101
            and artifact["reference_filter_seed"] == 27102, "Precision selection differs")
    array_name = "docs/results/heston-precision/case-0-draws.npz"
    with np.load(source(array_name), allow_pickle=False) as saved:
        arrays = dict(saved)
    require(sources[array_name] == artifact["sha256"], "Precision array hash differs")
    fine = precision["case0_fine_references"][1]
    selected = fine["neural_comparisons"][0]
    require(fine["seed"] == 27102 and selected["neural_seed"] == 11101, "Precision repeat differs")
    joint = arrays["neural_joint"]
    require(joint.shape == arrays["reference_joint"].shape == (16384, 2), "Expected paired joint draws")
    close(joint.mean(axis=0), selected["neural"]["mean"])
    close(joint.std(axis=0, ddof=1), selected["neural"]["std"])
    close(np.corrcoef(joint.T)[0, 1], selected["neural"]["correlation"])
    errors = (joint.mean(axis=0) - selected["reference"]["mean"]) / selected["reference"]["std"]
    close(errors, selected["mean_error_in_reference_sd"])
    losses = {}
    for label in ("neural", "reference"):
        values = arrays[f"{label}_future_loss"]
        require(values.shape == (16384,) and np.isfinite(values).all(), "Invalid saved losses")
        quantiles = np.quantile(values, precision["quantile_order"])
        close(quantiles, selected[f"{label}_loss_q05_median_q95"])
        close(values.std(ddof=1), selected[f"{label}_loss_std"])
        losses[label] = (100 * quantiles).tolist()
    shift = losses["neural"][2] - losses["reference"][2]
    return {
        "scope": "Arithmetic review of saved validation evidence; no scientific rerun or acceptance",
        "decision": "retain optional diagnostic lesson; defer further acceptance experiments pending #26",
        "scientific_status": "provisional", "scientific_acceptance": False,
        "training_performed": False, "reserved_test_accessed": False,
        "raw_banks_accessed": False, "new_scientific_examples": 0,
        "reference_cases_with_recorded_stability_pass": 16, "candidates": candidates,
        "prior_copy_mean_rmse_in_reference_sd": prior_rmse,
        "precision_case0": {
            "neural_seed": 11101, "reference_filter_seed": 27102,
            "weighted_reference_mean": selected["reference"]["mean"],
            "neural_mean": joint.mean(axis=0).tolist(), "mean_error_in_reference_sd": errors.tolist(),
            "loss_q05_median_q95_percent_initial_wealth": losses,
            "upper_loss_endpoint_difference_percentage_points": shift,
            "negative_state_draw_counts": {name: int(np.sum(arrays[f"{name}_joint"][:, 1] < 0))
                                            for name in ("neural", "reference")},
        },
        "limits": [
            "Mean/SD errors use saved moments; coverage and SBC use saved validation arrays, not regenerated draws.",
            "Correlation/predictive maxima aggregate saved per-case errors; full historical draws are unavailable.",
            "Reference stability flags/inequalities checked from summaries, not reconstructed particle filters.",
            "Precision neural moments and both loss distributions recomputed from saved draws. Reference moments are full weighted-particle summaries, not sample moments.",
            "State-prior moments are the saved training estimate; no training bank was opened.",
            "Validation is reused development evidence. Finite reference precision and marginal SBC do not establish joint or held-out accuracy.",
        ],
        "source_sha256": sources,
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new JSON path")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new path")
    result = audit(ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "arithmetic_checks": "passed", "scientific_acceptance": False}))


if __name__ == "__main__":
    main()

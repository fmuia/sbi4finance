"""Compact source-bound report for the four full-panel training checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_comparison(record):
    fields = ("status", "mean_status", "width_status", "absolute_standardized_mean_difference",
              "signed_standardized_mean_difference", "signed_standardized_mean_interval",
              "width_ratio_90", "width_ratio_interval", "reference_sd", "reference_width_90",
              "neural_width_90", "reference_mean", "neural_mean")
    parameters = {name: {key: value[key] for key in fields if key in value}
                  for name, value in record.get("parameters", {}).items()}
    return {"status": record["status"], "parameters": parameters,
            **{key: record[key] for key in ("reason", "identity", "reference_diagnostics_passed", "observation_binding") if key in record}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells, curves, missing = {}, {}, []
    bank_path = args.root / "bank/bank.json"
    bank = json.loads(bank_path.read_text())
    for rows in (20000, 100000):
        directory = args.root / f"train-n{rows:06d}"
        if (directory / "curve.json").exists():
            curves[str(rows)] = json.loads((directory / "curve.json").read_text())["points"]
        for updates in (10000, 50000):
            name = f"n{rows:06d}-u{updates:06d}"
            evaluation = args.root / "evaluations" / name / "evaluation.json"
            if not evaluation.exists():
                missing.append(name)
                continue
            raw = json.loads(evaluation.read_text())
            checkpoint = raw["checkpoint"]
            if checkpoint["train_rows"] != rows or checkpoint["actual_optimizer_updates"] != updates:
                raise ValueError("evaluation is labelled with the wrong comparison cell")
            if checkpoint["bank_payload_sha256"] != bank["payload_sha256"]:
                raise ValueError("comparison cells use different paired banks")
            if digest(evaluation.parent / "draws.npz") != raw["draws_sha256"]:
                raise ValueError("evaluation draw payload changed after its analysis")
            with np.load(evaluation.parent / "draws.npz", allow_pickle=False) as archive:
                observations = np.asarray(archive["validation_observations"], dtype="<f8")
                truth = np.asarray(archive["validation_truth"], dtype="<f8")
            if observations.shape != (raw["settings"]["cases"], 54) or truth.shape != (raw["settings"]["cases"], 6):
                raise ValueError("development observation or truth dimensions differ from evaluation")
            def calibration(group):
                return {key: raw[group][key] for key in ("names", "cases", "draws_per_case", "coverage",
                            "coordinate_ks", "posterior_mean_rmse", "scope")}
            cells[name] = {"training_rows": rows, "updates": updates,
                "task_fingerprint": checkpoint["task_fingerprint"], "model_sha256": checkpoint["model_sha256"],
                "training_seconds_to_checkpoint": checkpoint["seconds_since_fit_started"],
                "curve_probe_metrics": checkpoint["curve_probe_metrics"],
                "parameter_calibration": calibration("parameter_calibration"),
                "product_calibration": calibration("product_calibration"),
                "market": compact_comparison(raw["market"]),
                "conditional_references": {key: compact_comparison(value) for key, value in raw.get("conditional_references", {}).items()},
                "support_audit": raw["neural_support"],
                "numeric_support_boundary_draw_count": raw["numeric_support_boundary_draw_count"],
                "cohort_sha256": raw["cohort_sha256"],
                "development_observations_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
                "development_truth_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
                "evaluation_seed": raw["settings"]["seed"],
                "evaluation_sha256": digest(evaluation), "evaluation_source_sha256": raw["script_sha256"],
                "draws_sha256": raw["draws_sha256"],
                "evaluation_seconds": raw["wall_seconds"], "scientific_acceptance": False}
    identities = {(cell["task_fingerprint"], cell["cohort_sha256"], cell["development_observations_sha256"],
                   cell["development_truth_sha256"], cell["evaluation_seed"], cell["evaluation_source_sha256"])
                  for cell in cells.values()}
    if len(identities) > 1:
        raise ValueError("training cells do not share the same task and development observations")
    density_path = args.out / "physical-density-score.json"
    density = None
    if density_path.exists():
        density = json.loads(density_path.read_text())
        for name, cell in cells.items():
            score = density["cells"][name]
            if (score["model_sha256"] != cell["model_sha256"]
                    or score["evaluation_record_sha256"] != cell["evaluation_sha256"]
                    or score["truth_sha256"] != cell["development_truth_sha256"]
                    or score["observations_sha256"] != cell["development_observations_sha256"]):
                raise ValueError("physical-density score and checkpoint comparison differ")
            cell["physical_validation_nll"] = {key: score[key] for key in ("mean_physical_nll", "observation_standard_error")}
    references = {}
    for index in (0, 2, 4, 6, 8, 10):
        directory = args.root / "references" / f"case-{index:03d}"
        diagnostics = directory / "diagnostics.json"
        failure = directory / "failure.json"
        if diagnostics.exists():
            record = json.loads(diagnostics.read_text())
            case = json.loads((directory / "case.json").read_text())
            entries = [*record["parameters"].values(), *record["products"]["parameters"].values()]
            references[str(index)] = {"mcmc_diagnostics_passed": record["mcmc_diagnostics_passed"],
                "r_hat_max": max(value["r_hat"] for value in entries),
                "ess_bulk_min": min(value["ess_bulk"] for value in entries),
                "ess_tail_min": min(value["ess_tail"] for value in entries),
                "chain_payload_sha256": record["chain_payload_sha256"],
                "diagnostics_sha256": digest(diagnostics), "case_sha256": digest(directory / "case.json"),
                "category": case["category"], "truth": case["parameters"],
                "negative_observation_count": sum(value < 0 for value in case["observations"])}
        else:
            references[str(index)] = {"status": "failed" if failure.exists() else "incomplete",
                                     **({"failure": json.loads(failure.read_text())} if failure.exists() else {})}
    record = {"format": "heston_full_training_study_results_v1", "scientific_acceptance": False,
              "scope": "Two training trajectories/four predeclared checkpoints; one seed, fixed full54 quote task, conditional and prior-predictive development evidence only.",
              "producer_commit": "76cef0d", "paired_bank": bank, "bank_record_sha256": digest(bank_path),
              "cells": cells, "missing_evaluations": missing, "curves": curves, "synthetic_references": references,
              "reserved_final_2000_opened": False,
              "physical_density_score_sha256": digest(density_path) if density is not None else None,
              "physical_density_data_volume_comparison": density["paired_100k_minus_20k"] if density is not None else None,
              "limitations": ["No capacity comparison, seed confirmation or short-panel training in this unit.",
                    "The prior and independent discrepancy are an engineering benchmark; market-model adequacy remains unresolved.",
                    "cumulative_training_loss is a preserved mislabel for the current batch metric under this BayesFlow version; plots use fixed train/validation probe NLL.",
                    "Curve probes are losses in each checkpoint's standardized logit coordinates; comparable physical-coordinate density scores are reported separately.",
                    "A single training seed and inspected development observations do not establish fresh acceptance."],
              "source_sha256": digest(Path(__file__))}
    (args.out / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if cells:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for rows, points in curves.items():
            axes[0, 0].plot([p["updates"] for p in points], [p["validation_probe_loss"] for p in points], label=f"{int(rows)//1000}k surfaces")
        axes[0, 0].set(title="Fixed validation-probe objective", xlabel="Optimizer updates", ylabel="NLL in checkpoint-standardized coordinates")
        axes[0, 0].legend(); axes[0, 0].grid(alpha=.2)
        names = next(iter(cells.values()))["parameter_calibration"]["names"] + ["control", "affected"]
        display = ["v₀", "κ", "θ", "ξ", "ρ", "τ", "near var.", "fwd var."]
        colors = ["#8ab6d6", "#135d8b", "#edb47e", "#b6500c"]
        for number, (name, cell) in enumerate(cells.items()):
            label = f"{cell['training_rows']//1000}k surfaces / {cell['updates']//1000}k updates"
            x = np.arange(len(names)) + (number - 1.5) * .15
            metrics = cell["market"]["parameters"]
            if metrics:
                width = np.array([metrics[key]["width_ratio_90"] for key in names])
                interval = np.array([metrics[key]["width_ratio_interval"] for key in names]).T
                axes[0, 1].errorbar(x, width, yerr=np.maximum(0, np.array([width-interval[0], interval[1]-width])), fmt=".", color=colors[number], label=label, capsize=2)
                mean = np.array([metrics[key]["signed_standardized_mean_difference"] for key in names])
                ci = np.array([metrics[key]["signed_standardized_mean_interval"] for key in names]).T
                axes[1, 0].errorbar(x, mean, yerr=np.maximum(0, np.array([mean-ci[0], ci[1]-mean])), fmt=".", color=colors[number], label=label, capsize=2)
            coverage = np.array(cell["parameter_calibration"]["coverage"]["0.9"]["coverage"] + cell["product_calibration"]["coverage"]["0.9"]["coverage"])
            axes[1, 1].plot(x, coverage, ".", color=colors[number], label=label)
        axes[0, 1].axhspan(.75, 1.25, color="green", alpha=.12)
        axes[0, 1].set(yscale="log", title="Market 90% interval width / reference width", ylabel="Ratio; shaded target [.75,1.25]")
        axes[1, 0].axhspan(-.3, .3, color="green", alpha=.12)
        axes[1, 0].set(title="Market posterior mean error", ylabel="Reference posterior SD")
        axes[1, 1].axhline(.9, color="black", linewidth=1)
        axes[1, 1].set(title="512-case development 90% coverage", ylabel="Coverage fraction", ylim=(.5, 1.01))
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.set(xticks=np.arange(len(names)), xticklabels=display)
            ax.grid(axis="y", alpha=.2)
        axes[1, 1].legend(fontsize=8, loc="lower right")
        fig.suptitle("Full-panel Heston training study: data coverage and optimizer exposure\nFixed wider-support task; common 50k learning-rate horizon; development evidence", fontsize=13)
        fig.savefig(args.out / "training-comparison.png", dpi=170)
        plt.close(fig)
    print(json.dumps({"evaluations": len(cells), "missing": missing, "references": {key: value.get("mcmc_diagnostics_passed", value.get("status")) for key, value in references.items()}}))


if __name__ == "__main__":
    main()

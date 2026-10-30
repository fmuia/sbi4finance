"""Paired offline learning-curve diagnostics; always report incomplete lanes safely."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sbi4finance.path_inference import bank, inference
from compare_path_refinement import paired_risk, interval_screen, prior_mean


def load_evaluation(directory):
    report = json.loads((directory / "evaluation.json").read_text())
    path = directory / report["case_summary_file"]
    if inference.sha256(path) != report["case_summary_sha256"]:
        raise ValueError("evaluation arrays differ from their recorded hash")
    if report.get("split") != "validation" or report.get("offset") != 0 or report.get("reserved_test_evaluated"):
        raise ValueError("only the declared validation cohort is allowed")
    with np.load(path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    if len(arrays["truth"]) != 1024 or report["cases"] != 1024:
        raise ValueError("the matched comparison requires 1,024 validation histories")
    return report, arrays


def save(out, result):
    temporary = out / "comparison.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "comparison.json")


def comparison(args):
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"format": "path_offline_scaleup_comparison_v1", "scientific_acceptance": False,
              "baseline_replaced": False, "reserved_test_evaluated": False,
              "scope": "one fixed offline learning-curve comparison; marginal development evidence only",
              "reference_status": "no conditional reference is certified for the new inferred-drift daily tasks"}
    try:
        manifest = bank.load_manifest(args.bank)
        task = manifest["task"]
        if bank.task_fingerprint(task) != bank.task_fingerprint(bank.teaching_task(task["model"])):
            raise ValueError("comparison requires the harmonized inferred-drift teaching task")
        old_report, old = load_evaluation(args.small_evaluation)
        new_report, new = load_evaluation(args.large_evaluation)
        for field in ("task_sha256", "bank_manifest_sha256", "targets", "case_identities_sha256", "draws"):
            if old_report[field] != new_report[field]:
                raise ValueError(f"paired evaluations differ in {field}")
        if old_report["bank_manifest_sha256"] != inference.sha256(args.bank / "manifest.json"):
            raise ValueError("evaluation describes a different bank")
        if old_report["task_sha256"] != bank.task_fingerprint(task):
            raise ValueError("evaluated task differs from the bank task")
        if (not np.array_equal(old["truth"], new["truth"])
                or not np.array_equal(old["trajectory_id"], new["trajectory_id"])):
            raise ValueError("the compared trajectories or truths differ")
        training = []
        for directory, expected, evaluation in ((args.small, 100_000, old_report), (args.large, 1_000_000, new_report)):
            record = json.loads((directory / "training.json").read_text())
            if (record["model_sha256"] != evaluation["checkpoint_sha256"]
                    or inference.sha256(directory / "model.keras") != record["model_sha256"]
                    or record["training_trajectories"] != expected
                    or record.get("same_seed_reload_exact") is not True):
                raise ValueError("checkpoint, subset count or native reload contract differs")
            training.append(record)
        for field in ("network", "seed", "batch_size", "prefix_policy", "learning_rate", "checkpoint_selection"):
            if training[0].get(field) != training[1].get(field):
                raise ValueError(f"matched training protocol differs in {field}")
        presentations = [record["training_trajectories"] * record["epochs_completed"] for record in training]
        result.update(status="development_comparison_complete", model=task["model"], targets=task["parameter_names"],
                      task_sha256=manifest["task_sha256"], case_identities_sha256=old_report["case_identities_sha256"],
                      training_presentations=presentations,
                      matched_training_exposure=bool(min(presentations) > 0 and max(presentations) / min(presentations) <= 1.02),
                      optimizer_steps=[r["optimizer_steps"] for r in training],
                      epochs_completed=[r["epochs_completed"] for r in training],
                      checkpoint_sha256=[r["model_sha256"] for r in training], cohorts={})
        lengths = sorted(set(old_report["lengths"]) & set(new_report["lengths"]), key=int)
        if set(old_report["lengths"]) != set(new_report["lengths"]) or not lengths:
            raise ValueError("observation checkpoints differ")
        result["finite_and_support_pass"] = all(
            row["nonfinite_count"] == 0 and not any(row["out_of_prior_fraction"])
            for report in (old_report, new_report) for row in report["lengths"].values())
        for cohort, rows in (("selection_reused", slice(0, 256)), ("new_validation", slice(256, 1024))):
            truth = old["truth"][rows]
            panel = {"cases": len(truth), "lengths": {}}
            for n in lengths:
                value = paired_risk(old[f"posterior_mean_{n}"][rows], new[f"posterior_mean_{n}"][rows],
                                    truth, prior_mean(task), seed=20261031211 + int(n), bootstraps=1000)
                value["interval_screen"] = interval_screen(old[f"lower90_{n}"][rows], old[f"upper90_{n}"][rows],
                    new[f"lower90_{n}"][rows], new[f"upper90_{n}"][rows], truth)
                value["elapsed_years"] = int(n) / 252
                panel["lengths"][n] = value
            result["cohorts"][cohort] = panel
        result["drift_target_index"] = task["parameter_names"].index("drift")
        save(args.out, result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="inconclusive_missing_or_invalid_inputs", reason=str(error))
        save(args.out, result)
        return result
    try:
        plot_histories(args, task, old_report, new_report)
        result["teaching_case_plot"] = "validation-case-000000.png"
    except Exception as error:
        result["plot_error"] = str(error)
    save(args.out, result)
    return result


def plot_histories(args, task, old_report, new_report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lengths = sorted(map(int, old_report["lengths"]))
    times = np.asarray(lengths) / 252
    panels = []
    truth = None
    for directory, report in ((args.small_evaluation, old_report), (args.large_evaluation, new_report)):
        path = directory / "teaching-cases.npz"
        if inference.sha256(path) != report["teaching_cases_sha256"]:
            raise ValueError("teaching cases changed after evaluation")
        with np.load(path, allow_pickle=False) as values:
            current_truth = values["truth"][0]
            if truth is not None and not np.array_equal(truth, current_truth):
                raise ValueError("plotted validation cases differ")
            truth = current_truth.copy()
            bands = [np.quantile(values[f"draws_{n}"][0], [.05, .5, .95], axis=0) for n in lengths]
        panels.append(np.stack(bands))
    dimensions = len(truth)
    figure, axes = plt.subplots((dimensions + 1) // 2, 2, figsize=(10, 2.8 * ((dimensions + 1) // 2)), squeeze=False)
    for j, axis in enumerate(axes.flat):
        if j >= dimensions:
            axis.axis("off")
            continue
        for label, color, bands in zip(("100k training histories", "1M training histories"), ("#2367a2", "#d4831c"), panels):
            axis.fill_between(times, bands[:, 0, j], bands[:, 2, j], color=color, alpha=.15)
            axis.plot(times, bands[:, 1, j], color=color, label=label)
        axis.axhline(truth[j], color="black", linestyle="--", linewidth=1, label="Generating parameter")
        axis.set(xscale="log", title=task["parameter_names"][j], xlabel="Observed trading years")
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    figure.suptitle(f"{task['model']}: validation case 0, posterior median and 90% intervals\nDevelopment illustration; no conditional reference accepted", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(args.out / "validation-case-000000.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "small", "large", "small-evaluation", "large-evaluation", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    result = comparison(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("status", "scientific_acceptance", "reserved_test_evaluated")}))


if __name__ == "__main__":
    main()

"""Package a paired Heston dropout development comparison without loading models.

Only small metadata/curve/evaluation records enter the output folder. Full model
and state files are hashed as bytes and remain under their immutable run paths.
Conditional results are copied with their pass/fail/inconclusive scope intact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def evaluate_identity(path, checkpoint, task):
    evaluation = read(path)
    declared = Path(evaluation["checkpoint"])
    if not declared.is_absolute():
        declared = Path(__file__).resolve().parents[1]/declared
    declared_record = read(declared/"checkpoint.json")
    if (evaluation["task"] != task["target"]
            or evaluation.get("scientific_acceptance") is not False
            or declared_record["model_sha256"] != checkpoint["model_sha256"]
            or declared_record["state_sha256"] != checkpoint["state_sha256"]
            or digest(declared/"model.keras") != checkpoint["model_sha256"]
            or digest(declared/"state.npz") != checkpoint["state_sha256"]
            or evaluation.get("checkpoint_model_sha256", checkpoint["model_sha256"]) != checkpoint["model_sha256"]
            or evaluation.get("checkpoint_state_sha256", checkpoint["state_sha256"]) != checkpoint["state_sha256"]):
        raise ValueError("evaluation target/model/state differs from the paired checkpoint")
    return evaluation


def case_summary(evaluation, names):
    result = {}
    for name, case in evaluation["cases"].items():
        comparison = case.get("comparison", {})
        parameters = comparison.get("parameters", {})
        result[name] = {"status": case["status"],
            "reason": case.get("reason"),
            "reference_diagnostics_passed": case.get("reference_diagnostics_passed"),
            "parameters": {parameter: {key: values.get(key) for key in (
                "status", "signed_standardized_mean_difference", "signed_standardized_mean_interval",
                "width_ratio_90", "width_ratio_interval", "mean_status", "width_status")}
                for parameter, values in parameters.items()},
            "physical_width_range": ([min(parameters[p]["width_ratio_90"] for p in names),
                                       max(parameters[p]["width_ratio_90"] for p in names)] if parameters else None)}
    return result


def package(baseline_run, control_run, baseline_evaluation, control_evaluation, out,
            baseline_2k_evaluation=None, control_2k_evaluation=None):
    baseline_run, control_run, out = map(Path, (baseline_run, control_run, out))
    baseline, control = read(baseline_run/"run.json"), read(control_run/"run.json")
    btrain, ctrain = read(baseline_run/"training.json"), read(control_run/"training.json")
    initial_b, initial_c = read(baseline_run/"initial-probes.json"), read(control_run/"initial-probes.json")
    reference = control["paired_default_reference"]
    proof = control["dropout_initialization_control"]
    if (baseline["architecture"]["subnet_dropout"] != .05 or control["architecture"]["subnet_dropout"] != 0.
            or control["dropout_zero_control"] is not True
            or digest(baseline_run/"run.json") != reference["run_sha256"]
            or digest(baseline_run/"initial-probes.json") != reference["initial_probes_sha256"]
            or initial_b != initial_c or reference["initial_deterministic_probes_exact"] is not True
            or proof["array_state_unchanged"] is not True
            or any(row["before_sha256"] != row["after_sha256"] for row in proof["state_groups"].values())
            or proof["rates_before"] != [.05]*24 or proof["rates_after"] != [0.]*24):
        raise ValueError("initialization/dropout control identity differs")
    for key in reference["paired_recipe_fields"]:
        if baseline[key] != control[key]:
            raise ValueError("paired recipe differs: "+key)
    for key in ("completed_updates", "unique_rows_seen", "row_presentations", "batch_size_counts", "ordered_row_indices_sha256"):
        if btrain[key] != ctrain[key]:
            raise ValueError("actual row exposure differs: "+key)
    if (btrain["status"] != "complete" or ctrain["status"] != "complete"
            or ctrain["completed_updates"] != 10000 or ctrain["reserved_arrays_opened"]
            or not btrain["same_seed_reload_exact"] or not ctrain["same_seed_reload_exact"]):
        raise ValueError("paired completed endpoint/reload/reserved-pool checks failed")
    payloads = {}
    for updates in (2000, 10000):
        checkpoint_dir = control_run/"checkpoints"/f"update-{updates:06d}"
        checkpoint = read(checkpoint_dir/"checkpoint.json")
        if (checkpoint["actual_optimizer_updates"] != updates
                or checkpoint["task_fingerprint"] != control["task_fingerprint"]
                or checkpoint["architecture"] != control["architecture"]
                or checkpoint["source_sha256"] != control["source_sha256"]
                or digest(checkpoint_dir/"model.keras") != checkpoint["model_sha256"]
                or digest(checkpoint_dir/"state.npz") != checkpoint["state_sha256"]):
            raise ValueError("control checkpoint target/source/payload differs")
        payloads[str(updates)] = {"actual_optimizer_updates": updates,
            "checkpoint_record_sha256": digest(checkpoint_dir/"checkpoint.json"),
            **{key: checkpoint[key] for key in ("model_sha256", "state_sha256", "task_fingerprint", "probe_metrics", "source_sha256", "architecture", "resume_supported")},
            "payload_bytes": {name: (checkpoint_dir/name).stat().st_size for name in ("model.keras", "state.npz")}}
    bcheckpoint = read(baseline_run/"checkpoints/update-010000/checkpoint.json")
    ccheckpoint = read(control_run/"checkpoints/update-010000/checkpoint.json")
    beval = evaluate_identity(baseline_evaluation, bcheckpoint, baseline["task"])
    ceval = evaluate_identity(control_evaluation, ccheckpoint, control["task"])
    if (beval["panel_sha256"] != ceval["panel_sha256"] or beval["seed"] != ceval["seed"]
            or beval["population"]["rows"] != ceval["population"]["rows"]
            or beval["population"]["draws_per_case"] != ceval["population"]["draws_per_case"]):
        raise ValueError("conditional panel or population evaluation protocol differs")
    names = control["names"]
    cases_b, cases_c = case_summary(beval, names), case_summary(ceval, names)
    evaluation_files = {"baseline_10k": str(baseline_evaluation), "control_10k": str(control_evaluation)}
    optional = {}
    for label, path, run in (("baseline_2k", baseline_2k_evaluation, baseline_run),
                             ("control_2k", control_2k_evaluation, control_run)):
        if path is not None:
            evaluation_files[label] = str(path)
            evaluation = evaluate_identity(path, read(run/"checkpoints/update-002000/checkpoint.json"), control["task"])
            optional[label] = {"conditional_status": evaluation["conditional_status"],
                               "cases": case_summary(evaluation, names), "population": evaluation["population"]}
    out.mkdir(parents=True, exist_ok=True)
    compact = {"format": "heston_spread_compact_dropout_control_run_v1",
        "original_run_path": str(control_run.resolve()), "original_run_sha256": digest(control_run/"run.json"),
        "target": control["task"]["target"], **{key: control[key] for key in (
            "task_fingerprint", "names", "bounds", "design_sha256", "bank_record_sha256", "bank_payload_sha256",
            "train_rows", "planned_updates", "schedule_updates", "seed", "batch_size", "validation_rows_available",
            "checkpoint_updates", "wall_seconds", "architecture", "dropout_zero_control", "source_sha256",
            "environment_sha256", "git_revision", "versions", "devices", "optimizer_schedule", "numeric_configuration",
            "augmentation", "batch_plan", "probe_rows", "checkpoint_selection", "resume_supported",
            "dropout_initialization_control", "paired_default_reference")},
        "scientific_acceptance": False, "reserved_arrays_opened": False,
        "scope": "compact development record; full quote arrays and checkpoint state stay under original identities"}
    write(out/"run.json", compact)
    copied = {}
    for name in ("learning-curve.json", "training.json", "initial-probes.json"):
        contents = (control_run/name).read_bytes()
        temporary = out/(name+".tmp")
        temporary.write_bytes(contents)
        temporary.replace(out/name)
        copied[name] = {"original_sha256": digest(control_run/name), "copied_sha256": digest(out/name)}
    write(out/"checkpoint-payloads.json", {"format": "heston_spread_checkpoint_payload_summary_v1",
        "checkpoints": payloads, "payload_scope": "full model/state files and full checkpoint variable lists remain outside Git; no asset promoted"})
    comparison = {"format": "heston_spread_paired_dropout_development_comparison_v1", "scientific_acceptance": False,
        "status": "improved_but_B0_unaccepted", "parameter_names": names,
        "pairing": {"initial_probes_exact": True, "within_control_toggle_all_arrays_unchanged": True,
            "original_full_initial_archive_available": False, "same_row_order_and_exposure": True,
            "ordered_row_indices_sha256": ctrain["ordered_row_indices_sha256"],
            "reference_panel_sha256": ceval["panel_sha256"], "evaluation_seed": ceval["seed"],
            "claim_limit": "original full initial arrays were not persisted; control replays construction and matches saved initial deterministic probes"},
        "evaluation_binding_scope": "source evaluation hashes plus declared checkpoint paths and currently verified immutable payloads; older evaluations did not persist explicit model/state hashes",
        "at_10k": {"baseline": {"worker_seconds": btrain["seconds"], "cases": cases_b, "population": beval["population"]},
                    "dropout_zero": {"worker_seconds": ctrain["seconds"], "cases": cases_c, "population": ceval["population"]}},
        "other_declared_endpoints": optional,
        "interpretation": ["Substantial improvement under the matched zero-dropout intervention, with remaining synthetic conditional failures.",
            "The changed objective is a supported explanation for part of the earlier error, not proof of a universal dropout variance floor.",
            "Market reference disagreement remains separate from adequacy of Heston and the one-scale observation law.",
            "Pooled development coverage is not conditional posterior acceptance; incomplete references remain inconclusive.",
            "Worker times include imports/compile/probes/checkpoints and concurrent activity; they are not a dedicated throughput benchmark."],
        "next_decision": "one separately frozen continuation of the selected zero-dropout checkpoint, same task/data/architecture and original 10k cosine floor, before data-density or capacity changes",
        "reserved_arrays_opened": False}
    write(out/"comparison.json", comparison)
    bcurve, ccurve = read(baseline_run/"learning-curve.json"), read(control_run/"learning-curve.json")
    colors = {"Dropout .05": "#a54f24", "Dropout 0": "#176d9c"}
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.3), layout="constrained")
    for label, curve, initial in (("Dropout .05", bcurve, initial_b), ("Dropout 0", ccurve, initial_c)):
        x = np.array([0]+[row["updates"] for row in curve])
        y = np.array([initial["validation_probe_loss"]]+[row["validation_probe_loss"] for row in curve])
        axes[0].plot(x/1000, y, color=colors[label], label=label, lw=2)
        chosen = np.isin(x, [2000,10000])
        axes[0].scatter(x[chosen]/1000, y[chosen], color=colors[label], s=35, zorder=3)
    axes[0].set(xlabel="Optimizer updates (thousands)", ylabel="Fixed validation NLL (transformed target)",
                title="Same data, noise, schedule and initial probes")
    axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False)
    for offset, label, cases in ((-.14, "Dropout .05", cases_b), (.14, "Dropout 0", cases_c)):
        parameters = cases["synthetic-broad"]["parameters"]
        values = np.array([parameters[name]["width_ratio_90"] for name in names])
        bounds = np.array([parameters[name]["width_ratio_interval"] for name in names])
        axes[1].errorbar(np.arange(5)+offset, values, yerr=np.stack([values-bounds[:,0], bounds[:,1]-values]),
                         fmt="o", capsize=3, color=colors[label], label=label)
    axes[1].axhspan(.75, 1.25, color="#3c9b55", alpha=.17, label="Declared width band")
    axes[1].axhline(1., color="#426548", lw=.8)
    axes[1].set(xticks=np.arange(5), xticklabels=[r"$v_0$", r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$"],
                yscale="log", ylabel="Neural / reference central-90% width",
                title="Broad synthetic case at 10k: still fails")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed-c Heston: dropout control improves fidelity; B0 remains unaccepted", fontsize=12)
    fig.savefig(out/"development-comparison.png", dpi=180)
    plt.close(fig)
    source_files = {"producer": str(Path(__file__).resolve()), "baseline_run": str(baseline_run/"run.json"),
        "control_run": str(control_run/"run.json"), "baseline_training": str(baseline_run/"training.json"),
        "control_training": str(control_run/"training.json"), "baseline_curve": str(baseline_run/"learning-curve.json"),
        "control_curve": str(control_run/"learning-curve.json"),
        "baseline_initial_probes": str(baseline_run/"initial-probes.json"),
        "control_initial_probes": str(control_run/"initial-probes.json"),
        "baseline_10k_checkpoint_record": str(baseline_run/"checkpoints/update-010000/checkpoint.json"),
        "control_10k_checkpoint_record": str(control_run/"checkpoints/update-010000/checkpoint.json"), **evaluation_files}
    write(out/"manifest.json", {"format": "heston_spread_development_package_manifest_v1",
        "source_files": {name: {"path": path, "sha256": digest(path)} for name, path in source_files.items()},
        "copied_records": copied,
        "generated_files_sha256": {name: digest(out/name) for name in ("run.json", "training.json", "learning-curve.json", "initial-probes.json", "checkpoint-payloads.json", "comparison.json", "development-comparison.png")},
        "scientific_acceptance": False, "reserved_arrays_opened": False})
    coverage_b = beval["population"]["coverage"]["0.9"]["coverage"]
    coverage_c = ceval["population"]["coverage"]["0.9"]["coverage"]
    rows = ["| Parameter | .05 broad width | Zero broad width | Zero mean error (ref SD) | 90% coverage .05 → zero |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for index, name in enumerate(names):
        b, c = cases_b["synthetic-broad"]["parameters"][name], cases_c["synthetic-broad"]["parameters"][name]
        rows.append(f"| {name} | {b['width_ratio_90']:.2f} | {c['width_ratio_90']:.2f} | {c['signed_standardized_mean_difference']:+.2f} | {coverage_b[index]:.3f} → {coverage_c[index]:.3f} |")
    text = "\n".join(["# Fixed-c Heston zero-dropout control", "",
        "**Provisional development evidence: substantially improved, B0 still unaccepted.** The same fixed c=1 likelihood, physical prior, 54 contracts, 20k row prefix, Gaussian errors, row schedule, batch protocol and 10k optimizer schedule were retained. Only effective hidden-layer dropout changed from .05 to zero after replaying the original construction.", "",
        "The control completed 10,000 finite updates in %.2f worker seconds; the original took %.2f. These are observed worker clocks with concurrent work, not a dedicated performance benchmark. Both runs saw all 20,000 physical rows and 1,273,952 presentations, with identical row-order hashes and exact seeded checkpoint reloads. Reserved test arrays remain unopened." % (ctrain['seconds'], btrain['seconds']), "",
        "The rate toggle leaves all initialized arrays unchanged and matches the original saved deterministic initial probes. The original full initial state was not persisted; the evidence does not claim a comparison against an unavailable initial archive. [Run bindings](run.json) and [payload hashes](checkpoint-payloads.json) preserve this limit.", "",
        "The table uses the same broad synthetic reference at 10k; width intervals in the figure include the evaluator's Monte Carlo uncertainty. Width acceptance is [.75,1.25] and the MC-aware mean margin is .3 reference SD. All five broad widths remain above the band, and several means fail. Development coverage is closer to nominal but does not establish conditional accuracy.", "", *rows, "",
        "The family-adjusted rank checks improve from 0/5 to 3/5 passing parameters; theta and rho still fail. These checks use the same 512 development cases, not the reserved acceptance population.", "",
        "![Matched development comparison](development-comparison.png)", "",
        "[The complete provisional comparison](comparison.json) retains market, market-like and boundary entries exactly as available in the source evaluations, including fail/inconclusive status. Some references were unavailable when the original baseline evaluation snapshot was made. A missing comparison is not a pass, and improvements on one case do not accept the whole task. Market-model adequacy remains a separate question.", "",
        "The selected next experiment is one separately frozen, bounded optimization continuation of the zero-dropout checkpoint, preserving its task, data, architecture and original schedule floor. More data and capacity remain later isolated choices; no automatic grid, inferred-c or Bergomi production is justified by these results.", "",
        "Reproduce with `lab/scripts/plot_heston_spread_development.py` using the source paths/hashes in [manifest.json](manifest.json). The producer loads no model or state arrays and opens no bank members. Full checkpoints remain outside Git; no tutorial asset is promoted.", ""])
    (out/"README.md").write_text(text)
    manifest = read(out/"manifest.json")
    manifest["generated_files_sha256"]["README.md"] = digest(out/"README.md")
    write(out/"manifest.json", manifest)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-run", "control-run", "baseline-evaluation", "control-evaluation", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--baseline-2k-evaluation", type=Path)
    parser.add_argument("--control-2k-evaluation", type=Path)
    args = parser.parse_args()
    package(**vars(args))
    print(json.dumps({"out": str(args.out), "scientific_acceptance": False}))


if __name__ == "__main__":
    main()

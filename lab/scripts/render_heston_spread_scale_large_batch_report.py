"""Compare four frozen batch recipes and the existing selected MCMC references.

Only compact JSON, source files and named recovery quantiles are opened. No
models, banks or chain payloads are loaded; no inference or pricing occurs.
Historical manifests, reference readiness and selected rows remain unchanged.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_spread_scale_capacity_report as capacity
from scripts import render_heston_spread_mcmc_recovery as selected

shared = capacity.shared
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BATCHES = (128, 512, 1024, 4098)
EPOCHS = (25, 100, 250, 500)
NAMES, LABELS = shared.NAMES, shared.LABELS
COLORS = {128: "#287b6a", 512: "#654a97", 1024: "#246da0", 4098: "#b57920",
    "mcmc": "#b44b37", "prior": "#7e8790"}
MARKERS = {128: "^", 512: "s", 1024: "D", 4098: "v", "mcmc": "o"}
esc, fmt, table, section, chart = shared.esc, shared.fmt, shared.table, shared.section, shared.chart


def load_arm(inputs, spec, target, protocol_spec, baseline, *, new):
    batch = spec["batch_size"]
    if batch not in ((1024, 4098) if new else (128, 512)):
        raise ValueError("unexpected exact batch size in report")
    run = inputs.json(spec["run"], f"Batch {batch} complete run")
    training = inputs.json(spec["training"], f"Batch {batch} training ledger")
    curve = inputs.json(spec["curve"], f"Batch {batch} learning curve")
    per_pass = (100000+batch-1)//batch
    if (run["task"]["target"] != target or run["train_rows"] != 100000
            or run["batch_size"] != batch
            or run["architecture"] != {"family": "spline", "depth": 6, "widths": [128, 128],
                "construction_dropout": .05, "effective_dropout": 0.}
            or run.get("reserved_arrays_opened") is not False
            or training["status"] != "complete" or training["completed_bank_passes"] != 500
            or training["continued_row_presentations"] != 50000000
            or training["continued_updates"] != 500*per_pass
            or training["completed_updates"] != 60000+500*per_pass
            or training["same_seed_reload_exact"] is not True
            or training["all_parent_array_hashes_restored"] is not True):
        raise ValueError("report recipe or complete exposure ledger differs")
    dropout = run.get("initialization_control", {})
    if (len(dropout.get("rates_after", [])) != 24 or any(x != 0. for x in dropout["rates_after"])
            or dropout.get("array_state_unchanged") is not True):
        raise ValueError("the inherited 24-layer zero-dropout control differs")
    if [p["completed_bank_passes"] for p in curve] != list(range(1, 501)):
        raise ValueError("one recorded loss point per complete bank pass required")
    for point in curve:
        passes = point["completed_bank_passes"]
        if (point["additional_presentations"] != passes*100000
                or point["updates"] != 60000+passes*per_pass):
            raise ValueError("learning-curve example or optimizer count differs")
    records = []
    if [item["epochs"] for item in spec["evaluations"]] != list(EPOCHS):
        raise ValueError("every predeclared endpoint must remain in the report")
    for item in spec["evaluations"]:
        passes = item["epochs"]
        evaluation = inputs.json(item["evaluation"], f"Batch {batch},pass{passes} evaluation")
        expected = "heston_spread_scale_large_batch_evaluation_v1" if new else "heston_spread_scale_precision_evaluation_v1"
        if (evaluation.get("format") != expected or evaluation["status"] != "complete"
                or evaluation["task"] != target or evaluation["batch_size"] != batch
                or evaluation["completed_bank_passes"] != passes
                or evaluation["actual_optimizer_updates"] != 60000+passes*per_pass
                or evaluation["study_protocol"] != protocol_spec or evaluation["family_checks"] != 384
                or evaluation["population"]["truth_sha256"] != baseline["population"]["truth_sha256"]
                or evaluation["population"]["observations_sha256"] != baseline["population"]["observations_sha256"]
                or evaluation.get("reserved_arrays_opened") is not False):
            raise ValueError("endpoint changed task, data, family or study identity")
        recovery = shared.recovery_data(item["recovery"], inputs, evaluation, passes)
        records.append({"passes": passes, "evaluation": evaluation, "recovery": recovery,
            "metrics": capacity.scale_metrics(*recovery)})
    return {"batch_size": batch, "run": run, "training": training, "curve": curve,
        "evaluations": records, "final": records[-1], "spec": spec}


def verify_compact_counter(inputs, checkpoint, parent_value):
    metadata = inputs.json(checkpoint["record"], "Observed standardizer-counter checkpoint metadata")
    counters = checkpoint["standardizer_count"]
    if counters["state"]["sha256"] != metadata["state_sha256"] or len(counters["counters"]) != 2:
        raise ValueError("compact counter record does not bind the saved full state")
    expected_paths = {row["path"] for row in metadata["state"]["non_trainable"]
        if re.search(r"/standardization(?:_\d+)?/standardize(?:_\d+)?/", row["path"])
        and row["shape"] == [] and row["dtype"] == "float32"}
    if len(expected_paths) != 2 or {c["path"] for c in counters["counters"]} != expected_paths:
        raise ValueError("both distinct input/target standardizer counters must be recorded")
    for counter in counters["counters"]:
        rows = [row for row in metadata["state"]["non_trainable"] if row["path"] == counter["path"]]
        encoded = np.asarray(counter["value"], dtype=np.float32)
        if (len(rows) != 1 or rows[0]["shape"] != [] or rows[0]["dtype"] != "float32"
                or rows[0]["sha256"] != counter["sha256"]
                or selected.hashlib.sha256(encoded.tobytes()).hexdigest() != counter["sha256"]):
            raise ValueError("observed scalar counter does not match its recorded array hash")
    if any(counter["value"] != counters["value"] for counter in counters["counters"]):
        raise ValueError("reported counter total differs from its two observed scalars")
    logical = parent_value+checkpoint["bank_passes"]*100000
    if (counters["logical_parent_plus_examples"] != logical
            or counters["difference_from_logical"] != counters["value"]-logical
            or counters["relative_difference"] != (counters["value"]-logical)/logical):
        raise ValueError("logical example count or observed difference differs")
    return counters


def load_data(summary_path):
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text())
    if summary.get("format") != "heston_spread_scale_large_batch_report_v1" or summary.get("interim") is not False:
        raise ValueError("complete source-bound larger-batch report manifest required")
    inputs = shared.Artifacts(summary_path)
    inputs.sources.append({"label": "Four-batch report manifest", "path": str(summary_path), "sha256": shared.digest(summary_path)})
    protocol = inputs.json(summary["protocol"], "New 1024/4098 study protocol")
    historical = inputs.json(summary["old_capacity_manifest"], "Unchanged 128/512 report manifest")
    old_protocol = inputs.json(historical["protocol"], "Original 128/512 precision-safe study protocol")
    if (protocol.get("format") != "heston_spread_scale_large_batch_study_v1"
            or historical["protocol"] != protocol["previous_training_study"]
            or summary["mcmc_recovery_manifest"] != protocol["matched_mcmc_report"]
            or protocol["joint_task"] != old_protocol["joint_task"]
            or protocol["parent_checkpoint"] != old_protocol["parent_checkpoint"]
            or protocol["bank"] != old_protocol["bank"]):
        raise ValueError("historical comparisons changed from the predeclared target or parent")
    for name, spec in protocol.get("sources", {}).items():
        selected.source_binding(inputs, spec, "Larger-batch executed source: "+name)
    baseline = inputs.json(historical["baseline"]["evaluation"], "Original 60k descriptive baseline")
    target = protocol["joint_task"]
    if baseline["task"] != target:
        raise ValueError("original 60k baseline uses a different posterior")
    arms = [load_arm(inputs, arm, target, historical["protocol"], baseline, new=False) for arm in historical["arms"]]
    arms += [load_arm(inputs, arm, target, summary["protocol"], baseline, new=True) for arm in summary["arms"]]
    if sorted(arm["batch_size"] for arm in arms) != list(BATCHES):
        raise ValueError("the report must retain all four exact batch sizes")
    arms = sorted(arms, key=lambda a: a["batch_size"])
    reference_contract = arms[0]["run"]["continuation_protocol"]
    for arm in arms:
        contract = arm["run"]["continuation_protocol"]
        if (arm["run"]["versions"] != arms[0]["run"]["versions"]
                or arm["run"]["numeric_configuration"] != arms[0]["run"]["numeric_configuration"]):
            raise ValueError("software or floating-point environment changed across arms")
        for key in ("train_rows", "additional_bank_passes", "additional_presentations", "seed", "schedule",
                "augmentation_namespace", "permutation_namespace", "parent_model_sha256", "parent_state_sha256"):
            if contract[key] != reference_contract[key]:
                raise ValueError("four-batch comparison changed more than grouping: "+key)
    mcmc_path = inputs.path(summary["mcmc_recovery_manifest"], "Unchanged 8-case MCMC recovery manifest")
    mcmc = selected.load_data(mcmc_path)
    if mcmc["target"] != target:
        raise ValueError("selected MCMC uses a different joint posterior")
    indices = mcmc["selection"]["selection"]["zero_based_population_indices"]
    for arm in arms:
        truth, q = arm["final"]["recovery"]
        if (not np.array_equal(truth[indices], mcmc["truth"])
                or arm["final"]["evaluation"]["population"]["observations_sha256"] != mcmc["selection"]["population_observations_sha256"]):
            raise ValueError("new neural quantiles use different selected truths or observations")
        mcmc["quantiles"][f"nn{arm['batch_size']}"] = q[:, indices, :]
    inputs.sources.extend(mcmc["inputs"].sources)
    parent_value = summary["parent_standardizer_count"]["value"]
    parent_record = {"bank_passes": 0, "record": protocol["parent_checkpoint"]["record"],
        "standardizer_count": {**summary["parent_standardizer_count"],
            "logical_parent_plus_examples": parent_value, "difference_from_logical": 0., "relative_difference": 0.}}
    verify_compact_counter(inputs, parent_record, parent_value)
    if summary["parent_standardizer_count"]["state"] != protocol["parent_checkpoint"]["state"]:
        raise ValueError("parent counter record refers to a different original state")
    counter_records = {}
    for arm in arms:
        if arm["batch_size"] in (1024, 4098):
            counter_records[arm["batch_size"]] = [verify_compact_counter(inputs, cp, parent_value) for cp in arm["spec"]["checkpoints"]]
    for name, spec in summary.get("sources", {}).items():
        selected.source_binding(inputs, spec, "New report source: "+name)
    return {"inputs": inputs, "summary": summary, "protocol": protocol, "target": target,
        "arms": arms, "baseline": baseline, "selected": mcmc, "counters": counter_records,
        "parent_counter": parent_value}


def save_plot(fig, directory, name, title, description):
    return selected.save_figure(fig, Path(directory)/name, title, description)


def plot_learning(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), layout="constrained")
    for arm in data["arms"]:
        bs = arm["batch_size"]; curve = arm["curve"]
        x = np.array([p["additional_presentations"] for p in curve])/1e6
        for key, style in (("train_probe_loss", ":"), ("validation_probe_loss", "-")):
            axes[0].plot(x, [p[key] for p in curve], style, color=COLORS[bs], lw=1.1,
                label=f"Batch {bs}" if style == "-" else None)
    passes = np.linspace(0, 500, 501)
    contract = data["arms"][0]["run"]["continuation_protocol"]
    axes[1].plot(passes*.1, [capacity.implied_rate(p, contract) for p in passes], color="#246da0")
    axes[0].set(xlabel="Additional examples (millions)", ylabel="Fixed-probe joint negative log density",
        title="Same frozen probes; dotted training, solid validation")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Additional examples (millions)", ylabel="Learning rate", title="Same exposure-based schedule in every arm")
    return save_plot(fig, directory, "learning-curves", "Four batch recipes at equal additional examples",
        "All four arms retain the original model, Adam slots and normalization state before an explicit common learning-rate restart.")


def plot_scale_progress(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4), layout="constrained")
    for arm in data["arms"]:
        bs = arm["batch_size"]
        for axis, key in zip(axes, ("rmse", "median_width90"), strict=True):
            axis.plot(EPOCHS, [e["metrics"][key] for e in arm["evaluations"]], marker=MARKERS[bs],
                color=COLORS[bs], label=f"Batch {bs}")
    prior_error = data["arms"][0]["final"]["metrics"]["prior_rmse"]
    axes[0].axhline(prior_error, ls=":", color=COLORS["prior"], label="Constant prior median")
    axes[1].axhline(1.35, ls=":", color=COLORS["prior"], label="Uniform prior 90% width")
    axes[0].set(ylabel="c posterior-median RMSE", title="Recovery across the same 512 datasets")
    axes[1].set(ylabel="Median central 90% posterior width", title="Remaining c uncertainty")
    for axis in axes:
        axis.set(xlabel="Additional 100k-bank passes", xticks=EPOCHS)
        axis.legend(fontsize=8)
    return save_plot(fig, directory, "c-progress", "Does c improve with training exposure?",
        "Each checkpoint retains the same generating c values, observations and posterior-sampling seeds; this is a development comparison.")


def plot_population(data, directory):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8), layout="constrained")
    chosen = np.arange(0, 512, 16)
    for axis, arm in zip(axes.flat, data["arms"], strict=True):
        bs = arm["batch_size"]; truth, q = arm["final"]["recovery"]
        axis.scatter(truth[:, -1], q[1, :, -1], s=10, color=COLORS[bs], alpha=.45, edgecolors="none")
        axis.errorbar(truth[chosen, -1], q[1, chosen, -1],
            yerr=np.stack([q[1, chosen, -1]-q[0, chosen, -1], q[2, chosen, -1]-q[1, chosen, -1]]),
            fmt="none", color=COLORS[bs], alpha=.8, capsize=2, elinewidth=.7)
        axis.plot([.5, 2], [.5, 2], "--", color="#405562", lw=1)
        axis.axhline(1.25, color=COLORS["prior"], ls=":")
        axis.set(title=f"Batch {bs}  · 500 passes", xlabel="Generating c", ylabel="Posterior median c",
            xlim=(.47, 2.03), ylim=(.47, 2.03), aspect="equal")
    return save_plot(fig, directory, "c-population-recovery", "Spread-scale recovery on the full 512-case population",
        "All 512 posterior medians are shown; central 90% intervals use 32 fixed index-selected rows, not selected by error.")


def plot_new_parameters(data, directory):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.5), layout="constrained")
    bounds = np.asarray(data["target"]["prior_bounds"]); chosen = np.arange(0, 512, 32)
    for j, axis in enumerate(axes.flat):
        lo, hi = bounds[j]; axis.plot([lo, hi], [lo, hi], "--", color="#405562", lw=1)
        for arm in data["arms"]:
            bs = arm["batch_size"]
            if bs not in (1024, 4098): continue
            truth, q = arm["final"]["recovery"]
            axis.scatter(truth[:, j], q[1, :, j], s=9, color=COLORS[bs], alpha=.3, edgecolors="none")
            axis.errorbar(truth[chosen, j], q[1, chosen, j],
                yerr=np.stack([q[1, chosen, j]-q[0, chosen, j], q[2, chosen, j]-q[1, chosen, j]]),
                fmt=MARKERS[bs], ms=2.5, color=COLORS[bs], capsize=2, elinewidth=.6)
        margin=.025*(hi-lo)
        axis.set(title=LABELS[NAMES[j]], xlabel="Generating value", ylabel="Posterior median",
            xlim=(lo-margin, hi+margin), ylim=(lo-margin, hi+margin), aspect="equal")
    fig.legend(handles=[Line2D([], [], color=COLORS[b], marker=MARKERS[b], ls="none", label=f"Batch {b}") for b in (1024, 4098)],
        loc="outside lower center", ncol=2, frameon=False)
    return save_plot(fig, directory, "new-parameter-recovery", "Parameterwise recovery for both larger batches",
        "All 512 medians per method; 16 fixed-index 90% intervals per method and parameter. Intervals may be narrower than markers.")


def plot_sbc(data, directory):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.4), layout="constrained")
    grid = np.linspace(0, 1, 201); bounds = []
    for arm in data["arms"]:
        bs = arm["batch_size"]; p = arm["final"]["evaluation"]["population"]
        _, ranks = shared.population_records({0: arm["final"]["evaluation"]})[0]
        bounds.append(p["rank_dkw"]["dkw_bound"])
        for j, axis in enumerate(axes.flat):
            difference = np.searchsorted(np.sort(ranks[:, j]), grid, side="right")/512-grid
            axis.plot(grid, difference, color=COLORS[bs], label=f"Batch {bs}", lw=1.2)
    if not np.allclose(bounds, bounds[0], rtol=0, atol=1e-14):
        raise ValueError("SBC comparison bands differ")
    for j, axis in enumerate(axes.flat):
        axis.axhspan(-bounds[0], bounds[0], color="#dbece1", zorder=0)
        axis.axhline(0, color="#7e8790", lw=.7)
        axis.set(title=LABELS[NAMES[j]], xlabel="Fractional posterior rank", ylabel="ECDF−uniform CDF", xlim=(0, 1))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    return save_plot(fig, directory, "sbc-ecdf", "Marginal SBC on the same 512 validation observations",
        "Each band adjusts over six marginals within one endpoint, not across fits or checkpoints. Prior-like c predictions can pass this screen.")


def method_style(method):
    if method == "mcmc": return "Joint MCMC", COLORS["mcmc"], MARKERS["mcmc"]
    batch = int(method.removeprefix("nn"))
    return f"Network · batch {batch}", COLORS[batch], MARKERS[batch]


def plot_selected_scale(data, directory):
    d = data["selected"]; methods = ["mcmc", *[f"nn{b}" for b in BATCHES]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.3), layout="constrained")
    truth = d["truth"][:, -1]
    for axis, method in zip(axes.flat, methods, strict=False):
        label, color, marker = method_style(method)
        mask = d["ready"] if method == "mcmc" else np.ones(8, dtype=bool)
        q = d["quantiles"][method][:, mask, -1]
        axis.plot([.5, 2], [.5, 2], "--", color="#405562", lw=1)
        axis.axhline(1.25, color=COLORS["prior"], ls=":")
        axis.errorbar(truth[mask], q[1], yerr=np.stack([q[1]-q[0], q[2]-q[1]]),
            fmt=marker, color=color, capsize=3, ms=5, elinewidth=1)
        if method == "mcmc":
            axis.text(.03, .96, f"{int(mask.sum())}/8 ready; unready case omitted", transform=axis.transAxes, va="top", fontsize=8)
        axis.set(title=label, xlabel="Generating c", ylabel="Posterior median c",
            xlim=(.47, 2.03), ylim=(.47, 2.03), aspect="equal")
    axis = axes.flat[-1]
    axis.axhline(1.35, color=COLORS["prior"], ls=":", label="Uniform prior 90% width")
    for method in methods:
        label, color, marker = method_style(method)
        mask = d["ready"] if method == "mcmc" else np.ones(8, dtype=bool)
        q = d["quantiles"][method][:, mask, -1]
        axis.plot(truth[mask], q[2]-q[0], marker=marker, color=color, label=label, lw=1)
    axis.set(title="Remaining c uncertainty", xlabel="Generating c", ylabel="Central 90% posterior width", ylim=(0, 1.52))
    axis.set_box_aspect(1)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    return save_plot(fig, directory, "selected-c-recovery", "Four neural batches against the unchanged joint-MCMC recovery panel",
        "All four networks use the same 8 preselected rows; only the 7 ready joint-MCMC references supply posterior points.")


def plot_selected_parameters(data, directory):
    d = data["selected"]; methods = [*[f"nn{b}" for b in reversed(BATCHES)], "mcmc"]
    bounds = np.asarray(data["target"]["prior_bounds"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.9), layout="constrained")
    for j, axis in enumerate(axes.flat):
        lo, hi = bounds[j]; axis.plot([lo, hi], [lo, hi], "--", color="#405562", lw=1)
        for method in methods:
            label, color, marker = method_style(method)
            mask = d["ready"] if method == "mcmc" else np.ones(8, dtype=bool)
            q = d["quantiles"][method][:, mask, j]
            axis.errorbar(d["truth"][mask, j], q[1], yerr=np.stack([q[1]-q[0], q[2]-q[1]]),
                fmt=marker, color=color, alpha=.8, ms=4, capsize=2, elinewidth=.7)
        margin = .025*(hi-lo)
        axis.set(title=LABELS[NAMES[j]], xlabel="Generating value", ylabel="Posterior median",
            xlim=(lo-margin, hi+margin), ylim=(lo-margin, hi+margin), aspect="equal")
    fig.legend(handles=[Line2D([], [], color=method_style(m)[1], marker=method_style(m)[2], ls="none", label=method_style(m)[0])
        for m in ["mcmc", *[f"nn{b}" for b in BATCHES]]], loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    return save_plot(fig, directory, "selected-parameter-recovery", "All six parameters on the same selected datasets",
        "Every neural case and every ready MCMC case has its central 90% interval. Narrow intervals may overlap or be smaller than markers.")


def plot_conditional(data, directory):
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8), layout="constrained")
    for axis, case in zip(axes, shared.CASES, strict=True):
        reference = None; estimates = []
        for arm in data["arms"]:
            bs = arm["batch_size"]; entry = arm["final"]["evaluation"]["cases"][case]
            q = np.asarray(entry["neural_quantiles_05_50_95"])[:, -1]
            candidate = np.asarray(entry["reference_c_marginal"]["quantiles_05_50_95"])
            if reference is not None and not np.array_equal(reference, candidate):
                raise ValueError("same-observation conditional MCMC reference changed")
            reference = candidate
            estimates.append((f"Network · batch {bs}", COLORS[bs], q))
        estimates = [("Uniform prior", COLORS["prior"], np.array([.575, 1.25, 1.925])),
            ("Joint MCMC reference", COLORS["mcmc"], reference), *estimates]
        for index, (_, color, q) in enumerate(estimates):
            axis.errorbar(q[1], index, xerr=np.array([[q[1]-q[0]], [q[2]-q[1]]]), color=color, fmt="o", capsize=3)
        axis.set(title=shared.CASE_LABELS[case], xlabel="c: q05, median, q95", xlim=(.48, 2.02),
            yticks=np.arange(len(estimates)), yticklabels=[x[0] for x in estimates])
        axis.invert_yaxis()
    return save_plot(fig, directory, "conditional-c", "Three unchanged conditional observation vectors",
        "All six parameters are inferred jointly. Market pressure against c=2 is model/prior tension, not known market measurement noise.")


def network_excerpt():
    path = Path(__file__).with_name("train_heston_spread_scale.py")
    source = path.read_text(); tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_workflow")
    assignment = next(node for node in function.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "flow" for target in node.targets))
    return path, ast.get_source_segment(source, assignment)


def figure_links(name):
    return f'<p><a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></p>'


def build(summary_path, out):
    out = Path(out).resolve(); data = load_data(summary_path); inputs = data["inputs"]
    out.parent.mkdir(parents=True, exist_ok=True)
    for label, path in (("Four-batch presentation producer", Path(__file__)),
            ("Shared capacity presentation helpers", Path(capacity.__file__)),
            ("Shared selected-MCMC helpers", Path(selected.__file__)),
            ("Shared compact figures", Path(shared.__file__).with_name("heston_spread_report_figures.py"))):
        inputs.sources.append({"label": label, "path": str(path.resolve()), "sha256": shared.digest(path)})
    csspath = Path(shared.__file__).with_name("render_heston_spread_report.py")
    inputs.sources.append({"label": "Shared stylesheet", "path": str(csspath.resolve()), "sha256": shared.digest(csspath)})
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "heston-four-batch-v1", "font.family": "DejaVu Sans", "font.size": 9}):
        plots = {"learning-curves": plot_learning(data, out.parent), "c-progress": plot_scale_progress(data, out.parent),
            "c-population-recovery": plot_population(data, out.parent), "new-parameter-recovery": plot_new_parameters(data, out.parent),
            "sbc-ecdf": plot_sbc(data, out.parent), "selected-c-recovery": plot_selected_scale(data, out.parent),
            "selected-parameter-recovery": plot_selected_parameters(data, out.parent), "conditional-c": plot_conditional(data, out.parent)}
    stats = selected.statistics(data["selected"])
    final_rows = []
    for arm in data["arms"]:
        bs = arm["batch_size"]; e = arm["final"]["evaluation"]; m = arm["final"]["metrics"]
        final_rows.append([str(bs), fmt(arm["curve"][-1]["validation_probe_loss"]), fmt(m["rmse"]),
            f'{m["correlation"]:+.4f}' if m["correlation"] is not None else "Undefined", fmt(m["median_width90"]),
            f'{sum(e["population"]["rank_dkw"]["passed"])}/6', capacity.count_text(capacity.check_counts(e, shared.CASES[:2]))])
    new_metrics = {arm["batch_size"]: arm["final"]["metrics"] for arm in data["arms"] if arm["batch_size"] in (1024, 4098)}
    lead = (f"After 500 bank passes, c posterior-median RMSE is {new_metrics[1024]['rmse']:.4f} for batch 1024 "
        f"and {new_metrics[4098]['rmse']:.4f} for batch 4098. Their median 90% interval widths are "
        f"{new_metrics[1024]['median_width90']:.4f} and {new_metrics[4098]['median_width90']:.4f}, against the prior width 1.35.")
    if all(abs(m["median_width90"]/1.35-1.) < .05 for m in new_metrics.values()):
        lead += " Both larger-batch outputs retain near-prior c uncertainty in this paired experiment."
    pieces = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Four Heston batch sizes on the same data</title><style>'+shared.stylesheet(csspath)+'</style></head><body><main>',
        '<header><p class="eyebrow">100k clean surfaces · 50m additional examples per arm</p><h1>Heston recovery with larger batches</h1><p class="lead">'+esc(lead)+'</p><button onclick="window.print()">Print report</button></header>',
        '<nav><a href="#overview">Results</a><a href="#recipe">Training</a><a href="#recovery">Recovery</a><a href="#sbc">SBC</a><a href="#mcmc">Matched MCMC</a><a href="#conditional">Conditional checks</a><a href="#sources">Sources</a></nav>']
    pieces.append(section("Final 500-pass results", table(["Batch", "Validation NLL", "c median RMSE· 512 cases", "c truth–median correlation", "Median c 90% width", "SBC screens", "Synthetic conditional metrics"], final_rows)+
        '<div class="equation">Yᵢ=Pᵢ(φ)+c hᵢ Zᵢ, &nbsp; Zᵢ∼N(0,1)</div><p>Here hᵢ is the frozen quote half-spread in price units. One unknown c multiplies the Gaussian standard deviation across all 54 quotes in a dataset. The five physical Heston parameters and c are inferred jointly under independent physical-coordinate Uniform priors; c has support (.5,2).</p>'+
        '<p>The constant prior-median predictor has RMSE '+fmt(data["arms"][0]["final"]["metrics"]["prior_rmse"])+'. The Uniform(.5,2) prior has 90% width 1.35. Near-prior intervals can pass marginal SBC while ignoring information about c. Point recovery, pooled SBC and local posterior agreement answer different questions.</p>'+chart(plots["c-progress"], "Same 512 generating truths and noisy observations at every endpoint; no cases selected by neural error.")+figure_links("c-progress"), "overview"))
    recipe_rows = []
    for arm in data["arms"]:
        bs = arm["batch_size"]; contract = arm["run"]["continuation_protocol"]
        recipe_rows.append([str(bs), str(contract["batches_per_pass"]), str(contract["tail_batch_size"]),
            f'{contract["additional_optimizer_updates"]:,}', f'{arm["training"]["completed_updates"]:,}', "50,000,000", fmt(arm["training"]["seconds"]/60)])
    network_path, snippet = network_excerpt()
    schedule_path = Path(__file__).with_name("continue_heston_spread_scale.py")
    precision_path = Path(__file__).with_name("continue_heston_spread_scale_precision.py")
    for label, path in (("Frozen network construction", network_path), ("Frozen exposure schedule", schedule_path), ("Frozen precision-safe target transform", precision_path)):
        inputs.sources.append({"label": label, "path": str(path.resolve()), "sha256": shared.digest(path)})
    pieces.append(section("What changes with batch grouping", table(["Batch", "Updates/pass", "Tail size", "Additional updates", "Final optimizer step", "Additional examples", "Observed run minutes"], recipe_rows)+
        '<p>Every arm starts from the same 60k model and full Adam/RNG/normalization state. The primary budget is equal examples, not equal updates. Larger batches average more surfaces per gradient and receive fewer optimizer updates. These runs do not isolate gradient variance from update count and are one paired stream, not an across-seed stability study. Observed runtimes came from shared-machine campaigns and are not a controlled speed benchmark.</p>'+chart(plots["learning-curves"], "The fixed training and validation probes use the same observations and targets. The learning rate restarts at 5×10⁻⁵, decays by example exposure to 5×10⁻⁶ over 400 passes, then stays there through 500; no linear batch-size learning-rate scaling is applied.")+figure_links("learning-curves")+
        '<p>Network: BayesFlow 2.1 spline CouplingFlow, depth 6, twelve 128×128 conditioners. The 24 existing Dropout layers have effective rate 0. Adam uses the inherited moments and iteration count. Inputs remain 54 ordered noisy prices divided by their frozen half-spreads; c is a sixth target and never an input. Geometry and the spread profile stay fixed.</p><details><summary>Exact network, schedule and target-transform code</summary><p>The historical constructor below is restored from its native checkpoint; no fresh initialization is used for these continuations. Existing dropout rates are set to 0 in the preserved parent.</p><pre><code>'+esc(snippet)+'</code></pre><h3>Exposure-based schedule</h3><pre><code>'+esc(capacity.code_excerpt(schedule_path, "schedule_class"))+'</code></pre><h3>Finite target transform</h3><pre><code>'+esc(capacity.code_excerpt(precision_path, "host_adapt_training"))+'</code></pre></details>', "recipe"))
    counter_rows = []
    for batch, records in sorted(data["counters"].items()):
        final = records[-1]
        counter_rows.append([str(batch), f'{data["parent_counter"]:,.0f}', "50,000,000", f'{final["logical_parent_plus_examples"]:,.0f}',
            f'{final["value"]:,.0f}', f'{final["difference_from_logical"]:+,.0f}', f'{100*final["relative_difference"]:+.6f}%'])
    pieces.append(section("Exact exposure ledger and observed normalizer counters", table(["New batch", "Parent counter", "Additional int64 examples", "Logical total", "Observed float32 counter", "Difference", "Relative difference"], counter_rows)+
        '<p>The exact int64 ledger verifies 500 presentations for every one of 100,000 physical rows in each new arm. BayesFlow stores its running-standardization counters in float32, whose rounding can depend on batch grouping. The compact records above read both observed scalar counters from hash-verified checkpoint state, and both agree within each model. They are not assumed mathematically identical across batch sizes. No counters, moments or normalization arrays were reset or corrected.</p>', "counters"))
    pieces.append(section("Full-population recovery", chart(plots["c-population-recovery"], "All 512 medians; 90% intervals for 32 rows chosen by fixed index. The horizontal dotted line is the prior median 1.25, and the diagonal is the unit line.")+figure_links("c-population-recovery")+
        chart(plots["new-parameter-recovery"], "All 512 medians for both new networks; 16 fixed-index intervals per network and parameter. Historical 128/512 recovery is retained in its original report; all four methods are compared on the selected MCMC cases below.")+figure_links("new-parameter-recovery"), "recovery"))
    sbc_rows = [[str(a["batch_size"]), f'{sum(a["final"]["evaluation"]["population"]["rank_dkw"]["passed"])}/6',
        fmt(a["final"]["evaluation"]["population"]["coverage"]["0.9"]["coverage"][-1])] for a in data["arms"]]
    pieces.append(section("Marginal prior-predictive SBC", table(["Batch", "Marginal screens", "c 90% coverage · 512 cases"], sbc_rows)+chart(plots["sbc-ecdf"],
        "The band adjusts over six marginal ranks within each endpoint. It is not simultaneous across all runs/checkpoints. The same 512 development observations are reused; these are not fresh reserved tests.")+figure_links("sbc-ecdf"), "sbc"))
    metric_rows = []
    for method in ["mcmc", *[f"nn{b}" for b in BATCHES], "prior"]:
        s = stats[method]; correlation = s["truth_median_correlation"][-1]
        label = "Uniform prior baseline" if method == "prior" else method_style(method)[0]
        metric_rows.append([esc(label), str(s["cases"]), fmt(s["median_rmse"][-1]),
            "Undefined/constant" if correlation is None else f'{correlation:+.4f}', fmt(s["median_width90"][-1])])
    ready = int(data["selected"]["ready"].sum())
    pieces.append(section("Same selected observations, existing joint MCMC", '<p>The original 8 cases were selected by generating-c rank alone before the reference campaign. No failed case was replaced. All four neural populations are joined to those exact row IDs and observation hashes. The '+str(ready)+' ready references infer all six parameters jointly; the remaining reference stays explicitly unready. No MCMC or additional neural query is performed for this report.</p>'+chart(plots["selected-c-recovery"],
        "Every available neural case has a 90% credible interval. Only ready MCMC cases supply posterior points. The prior median is 1.25 and its 90% width is 1.35.")+figure_links("selected-c-recovery")+
        table(["Method", "Matched ready cases", "c median RMSE", "c truth–median correlation", "Median c 90% width"], metric_rows)+
        '<p>All methods in this table use the same ready-reference subset. These few c-stratified cases provide descriptive recovery, not SBC or a population coverage estimate. Narrower intervals alone do not establish correctness. The reference readiness decisions and data selection are inherited unchanged from the completed campaign.</p>'+chart(plots["selected-parameter-recovery"],
        "Parameterwise unit-line recovery for the same selected datasets. Intervals may be smaller than markers or overlap across methods.")+figure_links("selected-parameter-recovery"), "mcmc"))
    status_rows = []
    for case, reference in zip(data["selected"]["cases"], data["selected"]["references"], strict=True):
        d = reference["stats"]
        status_rows.append([esc(case["row_id"]), fmt(case["truth"][-1]), "Ready" if reference["ready"] else "NOT READY",
            fmt(d["max_rhat"]), fmt(d["min_bulk_ess"]), fmt(d["min_tail_ess"]), esc(reference["reason"])])
    pieces.append(section("Inherited MCMC convergence decisions", table(["Row", "Generating c", "Status", "Max R̂", "Min bulk ESS", "Min tail ESS", "Reason"], status_rows)+
        '<p>References use native PyMC DEMetropolisZ with four sequential chains. Ready requiresR̂ ≤ 1.01 and bulk/tail ESS ≥ 400 across parameters and recorded products, plus sampled pricing/likelihood refinement checks. Unready references are not treated as reliable posterior recovery results.</p>', "reference-status"))
    endpoint_rows = []
    for arm in data["arms"]:
        for item in arm["evaluations"]:
            e = item["evaluation"]; m = item["metrics"]
            endpoint_rows.append([str(arm["batch_size"]), str(item["passes"]), f'{e["actual_optimizer_updates"]:,}',
                fmt(m["rmse"]), fmt(m["median_width90"]), f'{sum(e["population"]["rank_dkw"]["passed"])}/6',
                capacity.count_text(capacity.check_counts(e, shared.CASES[:2])), capacity.count_text(capacity.check_counts(e, ("market",)))])
    conditional = chart(plots["conditional-c"], "Each row conditions on one fixed 54-quote observation vector, with all six parameters inferred jointly. The actual market case is a separate model-adequacy diagnostic.")+figure_links("conditional-c")
    conditional += table(["Batch", "Passes", "Optimizer step", "c RMSE", "Median c 90% width", "SBC", "Synthetic metrics", "Market metrics"], endpoint_rows)
    conditional += '<p>The historical 128/512 runs retain their original 384-check family; the new 1024/4098 runs form a separate predeclared 384-check family. No pooled 768-test simultaneous claim is made. Each family uses approximate Monte Carlo intervals from 400 block-bootstrap replicates, mean tolerance 0.3 reference SD and 90% width-ratio limits [.75,1.25]. These are development screens, not domain-wide acceptance rates or rigorous simultaneous guarantees.</p>'
    for arm in data["arms"]:
        conditional += '<details><summary>Final local mean/width checks·batch'+str(arm["batch_size"])+'</summary>'+capacity.conditional_table(arm["final"]["evaluation"])+"</details>"
    pieces.append(section("Existing conditional reference checks", conditional, "conditional"))
    for name, spec in data["summary"].get("sources", {}).items(): selected.source_binding(inputs, spec, "Summary source: "+name)
    source_rows = [[esc(s["label"]), '<code>'+esc(s["path"])+'</code>', '<code>'+esc(s["sha256"])+'</code>'] for s in inputs.sources]
    pieces.append(section("Hash-bound sources and scope", '<p>This offline report loads compact JSON, source code and named recovery quantiles. It does not open banks, learned weights, full optimizer states or MCMC chain archives, and performs no pricing, fitting or posterior sampling. The summary producer separately verifies native checkpoint hashes and reads only the two scalar standardizer counters. Old reports, tasks, checkpoints and reference decisions remain unchanged. No grid, spread-profile, masking or Bergomi changes are introduced.</p>'+table(["Input", "Path", "SHA256"], source_rows), "sources"))
    figures = {name+suffix: {"path": str((out.parent/(name+suffix)).resolve()), "sha256": shared.digest(out.parent/(name+suffix))}
        for name in plots for suffix in (".png", ".svg")}
    inventory = {"format": "heston_spread_scale_large_batch_report_inventory_v1", "sources": inputs.sources,
        "batch_sizes": list(BATCHES), "new_batches": [1024, 4098], "population_rows": 512,
        "selected_cases": 8, "ready_selected_cases": ready, "selected_statistics": stats,
        "figures": figures, "scientific_acceptance": False,
        "comparison_scope": "Equal additional examples, different optimizer counts and float32 normalization rounding; two separate 384-check development families."}
    pieces.append('<script type="application/json" id="report-inventory">'+json.dumps(inventory).replace("<", "\\u003c")+'</script></main></body></html>')
    out.write_text("\n".join(pieces))
    result = {**inventory, "report": {"path": str(out), "sha256": shared.digest(out)}}
    out.with_name("plot-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.summary, args.out)["report"])

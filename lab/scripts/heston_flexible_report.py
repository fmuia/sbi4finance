"""Export aggregate Heston bottleneck evidence and a self-contained saved-data report.

No estimator is loaded, no posterior is sampled and no pricing is performed.
Raw quotes, parameter draws, models and machine-specific paths stay in the
private run directories. Completed endpoints are immutable in the export;
rerunning with another completed MLP run extends the study index and report.
"""

import argparse
from hashlib import sha256
from html import escape
import io
import json
from pathlib import Path
import re

import numpy as np


FORMAT = "heston_flexible_compact_evaluation_v1"
NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c")
LABELS = ("v₀", "κ", "v∞", "σᵥ", "ρ", "c")
CASES = ("synthetic-broad", "synthetic-market-like", "market")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / (
    "lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/"
    "recovery-bank-independent/300k-300/evaluation/evaluation.json")


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def portable(value):
    """Reject machine paths instead of silently removing accidental disclosure."""
    if isinstance(value, dict):
        for item in value.values():
            portable(item)
    elif isinstance(value, list):
        for item in value:
            portable(item)
    elif isinstance(value, str) and (value.startswith(("/", "file://")) or re.match(r"^[A-Za-z]:[\\/]", value)):
        raise ValueError("absolute paths are not portable evidence")
    return value


def write(path, value, *, immutable=False):
    path = Path(path)
    payload = json.dumps(portable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists() and path.read_text() != payload:
        raise ValueError("exported endpoint identity already exists with different evidence: " + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def bound_json(binding, *, override=None, protected=None):
    path = Path(override if override is not None else binding["path"])
    if digest(path) != binding["sha256"]:
        raise ValueError("bound source record checksum differs: " + path.name)
    if protected is not None:
        protected[path] = binding["sha256"]
    return read(path)


def population_summary(population):
    """Keep aggregate diagnostics and hashes, excluding rowwise ranks and truths."""
    if tuple(population.get("parameter_names", ())) != NAMES:
        raise ValueError("six canonical parameter names are required")
    keys = ("rows", "draws_per_case", "parameter_names", "support", "observations_sha256", "truth_sha256",
            "sbc_seed", "raw_draws_used", "calibration_status", "rank_dkw", "coverage", "recovery", "mean_posterior_sd")
    return {key: population[key] for key in keys if key in population}


def compact_evaluation(evaluation, *, endpoint, epoch, source_sha256):
    population = population_summary(evaluation["population"])
    parameters = {}
    for index, name in enumerate(NAMES):
        parameters[name] = {
            "rmse": population["recovery"]["median_rmse"][index],
            "coverage90": population["coverage"]["0.9"]["coverage"][index],
            "median_width90": population["recovery"]["median_posterior_width90"][index],
        }
        # The historical baseline stores median widths, but not mean widths.
        # Compare like quantities; never relabel a median as a mean.
        if "mean_width90" in evaluation.get("parameters", {}).get(name, {}):
            parameters[name]["mean_width90"] = evaluation["parameters"][name]["mean_width90"]
    cases = {}
    for case_id in CASES:
        source = evaluation["cases"][case_id]
        comparison = source.get("comparison", {})
        if source.get("reference_ready") is not True or comparison.get("status") != "defined":
            raise ValueError("complete matched conditional comparisons are required")
        if tuple(comparison.get("quantities", ())) != NAMES:
            raise ValueError("conditional parameter ordering differs")
        cases[case_id] = {
            "reference_ready": True,
            "parameters": {name: {
                "mean_error_in_reference_sd": comparison["standardized_mean_error"][index],
                "width90_ratio": comparison["width_ratio"][index],
            } for index, name in enumerate(NAMES)},
            "support": source["support"],
            "reference_chain_sha256": source.get("reference_chain_sha256", source.get("reference_chain_payload", {}).get("sha256")),
        }
        products = comparison.get("derived_products", {})
        if products.get("status") == "defined":
            cases[case_id]["derived_products"] = {key: products[key] for key in
                                                    ("status", "quantities", "standardized_mean_error", "width_ratio")}
        dependence = comparison.get("dependence", {})
        cases[case_id]["dependence"] = {key: dependence[key] for key in
                                         ("status", "max_absolute_correlation_difference", "median_absolute_correlation_difference")
                                         if key in dependence}
    return portable({
        "format": FORMAT, "status": "complete", "endpoint": endpoint, "epoch": epoch,
        "population_rows": population["rows"], "draws_per_case": population["draws_per_case"],
        "parameters": parameters, "population": population, "cases": cases,
        "source_evaluation_sha256": source_sha256,
        "conditional_mean_metric": "canonical saved comparison.standardized_mean_error; absolute difference in reference posterior SD units",
        "scientific_acceptance": False,
        "scope": "Reused development panels; aggregate diagnostics only; no fresh acceptance or sufficient-statistic claim.",
    })


def compact_run(run, *, endpoint, epoch, hashes):
    keys = ("status", "seed", "train_rows", "validation_rows", "epochs", "completed_epochs", "smoke_test_only",
            "batch_size", "elapsed_seconds", "optimizer_updates", "presentations", "best_epoch", "best_validation_loss",
            "configuration", "data_payload_sha256", "task_sha256", "preprocessing_sha256", "versions", "platform",
            "devices", "checkpoint_policy", "noise_policy", "support_policy", "evaluation_scope", "pricing_calls")
    result = {key: run[key] for key in keys if key in run}
    result.update(format="heston_flexible_compact_run_v1", endpoint=endpoint, evaluated_epoch=epoch,
                  scientific_acceptance=False, payloads_included=False, source_hashes=hashes)
    if "source" in run:
        result["source"] = {key: run["source"][key] for key in ("commit", "runner_sha256", "inference_sha256")
                            if key in run["source"]}
    return portable(result)


def _entry(label, directory, run):
    return {"label": label, "directory": directory, "status": "complete",
            "representation": run.get("configuration", {}).get("representation", "direct"),
            "summary_dim": run.get("configuration", {}).get("summary_dim"),
            "smoke_test_only": run.get("smoke_test_only", False),
            "endpoint": run["endpoint"], "epoch": run["evaluated_epoch"],
            "train_rows": run["train_rows"], "model_available": False}


def export_study(run_directories, baseline_path, out, *, baseline_training=None, baseline_bank=None):
    """Append checked completed runs; never copy payloads or absolute provenance."""
    out, baseline_path = Path(out).resolve(), Path(baseline_path).resolve()
    protected = {baseline_path: digest(baseline_path)}
    baseline = read(baseline_path)
    if (baseline.get("format") != "heston_pr87_grid_evaluation_v1" or baseline.get("status") != "complete"
            or baseline.get("training_rows") != 300000 or baseline.get("native_epoch") != 300
            or baseline.get("endpoint") != "primary_final" or baseline.get("scientific_acceptance") is not False):
        raise ValueError("the preserved actual63 300k final-300 baseline is required")
    old_run = bound_json(baseline["training_run"], override=baseline_training, protected=protected)
    bank = bound_json(baseline["common_bank"], override=baseline_bank, protected=protected)
    if old_run["data_payload_sha256"] != bank["data_payload_sha256"]:
        raise ValueError("baseline training and bank identities differ")
    old_evaluation = compact_evaluation(baseline, endpoint="primary_final", epoch=300,
                                        source_sha256=protected[baseline_path])
    old_metadata = compact_run({
        "status": "complete", "configuration": {"representation": "direct", "summary_dim": None},
        "train_rows": 300000, "epochs": 300, "completed_epochs": 300, "smoke_test_only": False,
        "elapsed_seconds": old_run["elapsed_seconds"], "optimizer_updates": old_run["actual_optimizer_updates"],
        "presentations": baseline["training_presentations"], "data_payload_sha256": bank["data_payload_sha256"],
        "checkpoint_policy": "final epoch 300 primary", "noise_policy": "immutable stored noisy rows",
        "support_policy": "raw draws retained without clipping, rejection or resampling",
    }, endpoint="primary_final", epoch=300, hashes={"evaluation_sha256": protected[baseline_path],
                                                   "training_run_sha256": baseline["training_run"]["sha256"],
                                                   "bank_record_sha256": baseline["common_bank"]["sha256"]})
    pending = [("baseline-final/run.json", old_metadata), ("baseline-final/evaluation.json", old_evaluation)]
    entries = [_entry("Direct 63-input baseline — final 300 (primary)", "baseline-final", old_metadata)]
    for directory in run_directories:
        directory = Path(directory).resolve()
        run_id = directory.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
            raise ValueError("safe run directory name required")
        run_path, history_path = directory / "run.json", directory / "history.json"
        run, history = read(run_path), read(history_path)
        protected.update({run_path: digest(run_path), history_path: digest(history_path)})
        if (run.get("status") != "complete" or run.get("smoke_test_only") is not False
                or run.get("train_rows") != 300000 or run.get("completed_epochs") != 300 or run.get("epochs") != 300
                or run.get("configuration", {}).get("representation") != "mlp"
                or run.get("task", {}).get("target") != baseline["task"]
                or run.get("data_payload_sha256") != bank["data_payload_sha256"]):
            raise ValueError("matched complete 300k × 300-epoch MLP development fit required")
        if (history.get("epoch") != list(range(1, 301)) or len(history.get("loss", [])) != 300
                or len(history.get("val_loss", [])) != 300 or not np.isfinite(history["loss"] + history["val_loss"]).all()):
            raise ValueError("complete finite 300-epoch history required")
        for filename, key in (("task.json", "task_sha256"), ("preprocessing.json", "preprocessing_sha256")):
            if digest(directory / filename) != run[key]:
                raise ValueError("run input checksum differs: " + filename)
            protected[directory / filename] = run[key]
        endpoints = [("final", "primary_final", 300, "evaluation.json", "draws.npz", "model.keras", "model_sha256")]
        if run["best_epoch"] != 300:
            endpoints.append(("best", "secondary_validation_minimum", run["best_epoch"],
                              "best-evaluation.json", "best-draws.npz", "best.keras", "best_model_sha256"))
        for role, endpoint, epoch, filename, draws_name, model_name, model_key in endpoints:
            source = directory / filename
            evaluation = read(source)
            protected[source] = digest(source)
            if evaluation.get("status") != "complete" or evaluation.get("endpoint") != role:
                raise ValueError("completed endpoint evaluation required: " + filename)
            for key in ("observations_sha256", "truth_sha256"):
                if evaluation["population"][key] != baseline["population"][key]:
                    raise ValueError("population pairing differs: " + key)
            if evaluation["population"]["rows"] != 512 or evaluation["population"]["draws_per_case"] != 512:
                raise ValueError("the common 512 × 512 development panel is required")
            for case_id in CASES:
                if evaluation["cases"][case_id]["reference_chain_sha256"] != baseline["cases"][case_id]["reference_chain_payload"]["sha256"]:
                    raise ValueError("matched reference chain identity differs")
            for payload_name, expected in ((draws_name, evaluation["draws_sha256"]), (model_name, run[model_key])):
                if digest(directory / payload_name) != expected:
                    raise ValueError("private endpoint payload checksum differs: " + payload_name)
                protected[directory / payload_name] = expected
            normalized = compact_evaluation(evaluation, endpoint=endpoint, epoch=epoch, source_sha256=protected[source])
            metadata = compact_run(run, endpoint=endpoint, epoch=epoch, hashes={
                "run_sha256": protected[run_path], "history_sha256": protected[history_path],
                "evaluation_sha256": protected[source], "model_sha256": run[model_key],
                "draws_sha256": evaluation["draws_sha256"],
            })
            relative = f"runs/{run_id}/{role}"
            pending.extend([(relative + "/run.json", metadata), (relative + "/evaluation.json", normalized),
                            (relative + "/history.json", history)])
            label = f"MLP {run['configuration']['summary_dim']} — " + (
                "final 300 (primary)" if role == "final" else f"validation minimum {epoch} (secondary)")
            entries.append(_entry(label, relative, metadata))
    # Reject collisions before writing any part of a changed endpoint.
    for relative, value in pending:
        payload = json.dumps(portable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
        path = out / relative
        if path.exists() and path.read_text() != payload:
            raise ValueError("exported endpoint identity already exists with different evidence: " + relative)
    existing = read(out / "study.json") if (out / "study.json").exists() else {"runs": []}
    if existing.get("format", "heston_flexible_study_v1") != "heston_flexible_study_v1":
        raise ValueError("unrecognized study index")
    by_directory = {entry["directory"]: entry for entry in existing["runs"]}
    by_directory.update({entry["directory"]: entry for entry in entries})
    study = {"format": "heston_flexible_study_v1", "stage": "S1", "scientific_acceptance": False,
             "scope": "Completed fixed-grid MLP development comparisons; masks and moved grids remain untrained.",
             "payloads_included": False, "runs": list(by_directory.values())}
    for path, value in pending:
        write(out / path, value, immutable=True)
    if any(digest(path) != expected for path, expected in protected.items()):
        raise ValueError("a protected source changed while exporting")
    write(out / "study.json", study)
    render_report(out)
    return study


def _table(headers, rows):
    def text(value):
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value)
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(text(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _svg(figure):
    import matplotlib.pyplot as plt
    stream = io.StringIO()
    figure.savefig(stream, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    svg = stream.getvalue()
    return svg[svg.index("<svg"):]


def render_report(directory):
    """Render only checked compact exported summaries; no private payload access."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    study = read(directory / "study.json")
    rows = []
    for entry in study["runs"]:
        relative = Path(entry["directory"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("study entries must stay inside the portable export")
        root = directory / relative
        run, evaluation = read(root / "run.json"), read(root / "evaluation.json")
        if evaluation.get("format") != FORMAT or evaluation.get("scientific_acceptance") is not False:
            raise ValueError("compact development evaluation required")
        rows.append((entry, run, evaluation, read(root / "history.json") if (root / "history.json").exists() else None))
    primary = [row for row in rows if row[1]["endpoint"] == "primary_final"]
    palette = {None: "#1f77b4", 32: "#9467bd", 16: "#2ca02c", 8: "#ff7f0e", 4: "#d62728", 64: "#8c564b"}
    primary_colors = [palette.get(run.get("configuration", {}).get("summary_dim"), "#7f7f7f")
                      for _, run, _, _ in primary]
    css = "body{margin:0;background:#f6f7f8;color:#233340;font:15px/1.55 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:30px 22px 65px}h1{line-height:1.2}h2{margin-top:36px}.notice{padding:14px 18px;background:#fff4d9;border-left:4px solid #b98324}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:12px}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#e8edf2}figure{margin:20px 0;background:white;padding:12px;border:1px solid #dce2e7;border-radius:8px}svg{width:100%;height:auto}figcaption,footer{font-size:12px;color:#5a6874}code{overflow-wrap:anywhere}"
    chunks = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Heston learned-summary development</title><style>' + css + '</style></head><body><main>',
              '<h1>Heston learned summaries: complete fixed-grid comparisons</h1>',
              '<p class="notice"><strong>Development evidence only.</strong> Final epoch 300 is primary. The minimum native validation-loss checkpoint is secondary and is shown separately; it was not selected by posterior diagnostics. These results reuse historical development cases, retain every raw draw and do not establish fresh acceptance or a sufficient statistic.</p>',
              '<p>All fits use 300,000 stored noisy data sets, 63 actual contracts, the same frozen ZCA transform, broad physical priors and inferred Gaussian noise multiplier c. A learned MLP summary layer is added before the same FlowMatching posterior head. The MLP fits use the same declared seed; changing the architecture changes weight shapes and random-number use, so a one-seed screen cannot separate representation effects from optimization variability. The screen order is 32, 16, 8, 4, then 64 dimensions. Only completed endpoints appear below; stopped partial fits are excluded. Mask-aware and moving-grid estimators are not trained by this comparison.</p>',
              '<h2>Training and endpoint accounting</h2>']
    chunks.append(_table(["Endpoint", "Role", "Epoch", "Rows", "Fit minutes", "Optimizer updates", "Population outside prior", "SBC screens"], [
        [entry["label"], "primary" if run["endpoint"] == "primary_final" else "secondary", run["evaluated_epoch"], run["train_rows"],
         run.get("elapsed_seconds", 0) / 60, run.get("optimizer_updates", "unrecorded"),
         f"{100 * evaluation['population']['support']['outside_prior']['sample_fraction']:.3f}%",
         f"{sum(evaluation['population']['rank_dkw']['passed'])}/6"] for entry, run, evaluation, _ in rows]))
    chunks.append('<p>Fit time and optimizer updates describe the complete training trajectory for both endpoint rows, not additional training for the secondary checkpoint. These wall times were measured on shared local CPUs, so they are not a dedicated performance benchmark. Draws outside support are retained; lower violation rates do not imply conditional accuracy.</p>')
    histories = [row for row in primary if row[3] is not None]
    if histories:
        columns = min(2, len(histories))
        grid_rows = (len(histories) + columns - 1) // columns
        fig, axes = plt.subplots(grid_rows, columns, figsize=(6 * columns, 3.6 * grid_rows),
                                 squeeze=False, constrained_layout=True)
        for ax, (entry, run, _, history) in zip(axes.flat, histories):
            ax.plot(history["epoch"], history["loss"], label="training", linewidth=1)
            ax.plot(history["epoch"], history["val_loss"], label="validation", linewidth=1)
            ax.axvline(run["best_epoch"], color="#777", linestyle=":", label="validation minimum")
            ax.set(title=entry["label"], xlabel="epoch", ylabel="native loss")
            ax.title.set_fontsize(11)
            ax.legend(fontsize=8)
        for ax in axes.flat[len(histories):]:
            ax.set_visible(False)
        chunks.append('<h2>Learning curves</h2><figure>' + _svg(fig) + '<figcaption>One trajectory per fit; no posterior-based checkpoint selection.</figcaption></figure>')
    chunks.append('<h2>Population recovery and calibration</h2><p>The same 512 historical truths and observations are paired across endpoints, with 512 draws per observation. Sampling uncertainty and repeated use of this development population limit interpretation.</p>')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), constrained_layout=True)
    x = np.arange(len(NAMES)); width = .8 / len(primary)
    baseline_rmse = np.array([rows[0][2]["parameters"][name]["rmse"] for name in NAMES])
    for index, (entry, _, evaluation, _) in enumerate(primary):
        offset = x + (index - (len(primary) - 1) / 2) * width
        axes[0].bar(offset, [evaluation["parameters"][name]["coverage90"] for name in NAMES], width=width, label=entry["label"], color=primary_colors[index])
        axes[1].plot(offset, np.array([evaluation["parameters"][name]["rmse"] for name in NAMES]) / baseline_rmse,
                     linestyle="none", marker="o", markersize=5, label=entry["label"], color=primary_colors[index])
    axes[0].axhline(.9, color="#222", linestyle="--", linewidth=1)
    axes[0].set(ylim=(0, 1.04), title="Primary finals: nominal 90% coverage", ylabel="fraction of truths covered")
    axes[1].axhline(1, color="#222", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set(title="Primary finals: RMSE relative to direct baseline", ylabel="RMSE / direct baseline RMSE (log scale)")
    for ax in axes:
        ax.set_xticks(x, LABELS); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2, fontsize=8)
    chunks.append('<figure>' + _svg(fig) + '<figcaption>Primary final-300 endpoints only. Coverage close to 90% is desirable; lower RMSE is better. RMSE ratios use a logarithmic axis so the failed small bottleneck does not hide differences among other dimensions. Secondary validation minima remain in the tables below. These summaries do not replace the conditional checks.</figcaption></figure>')
    chunks.append(_table(["Endpoint", "Parameter", "Median-estimate RMSE", "90% coverage", "Median posterior 90% width"], [
        [entry["label"], name, evaluation["parameters"][name]["rmse"], f"{100 * evaluation['parameters'][name]['coverage90']:.3f}%",
         evaluation["parameters"][name]["median_width90"]] for entry, _, evaluation, _ in rows for name in NAMES]))
    chunks.append('<h2>Conditional agreement with matched MCMC</h2><p>Each case uses 4,096 raw network draws and its original matched full reference chain. The mean metric is the saved absolute mean discrepancy in reference-posterior SD units; width is the central-90% interval ratio. Shaded diagnostic bands reproduce the development conventions (mean ≤0.3; width 0.75–1.25), not a new acceptance decision. The market posterior remains sensitive to the declared noise support.</p>')
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for row_index, case_id in enumerate(CASES):
        for index, (entry, _, evaluation, _) in enumerate(primary):
            parameters = evaluation["cases"][case_id]["parameters"]
            for col, metric in enumerate(("mean_error_in_reference_sd", "width90_ratio")):
                axes[row_index, col].plot(x, [parameters[name][metric] for name in NAMES], marker="o", markersize=3,
                                           label=entry["label"], linewidth=1, color=primary_colors[index])
        for col, bounds in enumerate(((0, .3), (.75, 1.25))):
            ax = axes[row_index, col]
            ax.axhspan(*bounds, color="#c6e6c9", alpha=.4)
            ax.set_xticks(x, LABELS)
            ax.set(title=case_id, ylabel="absolute mean discrepancy / reference SD" if col == 0 else "90% width / reference width")
            ax.set_yscale("symlog", linthresh=1 if col == 0 else 2)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, -.01), ncol=2, fontsize=8)
    chunks.append('<figure>' + _svg(fig) + '<figcaption>Primary final-300 endpoints only; secondary validation minima are retained in the table. Vertical axes use a linear region near zero and logarithmic scaling for large errors. Agreement need not improve together across parameters, cases or endpoints.</figcaption></figure>')
    chunks.append(_table(["Case", "Endpoint", "Parameter", "Mean discrepancy / reference SD", "90% width ratio"], [
        [case_id, entry["label"], name, evaluation["cases"][case_id]["parameters"][name]["mean_error_in_reference_sd"],
         evaluation["cases"][case_id]["parameters"][name]["width90_ratio"]]
        for case_id in CASES for entry, _, evaluation, _ in rows for name in NAMES]))
    chunks.append('<h2>Limits and provenance</h2><p>One seed per bottleneck size is an initial screen. Different dimensions and their checkpoint outcomes must all be retained. A smaller summary matching an imperfect baseline does not prove sufficiency. Full-grid MLP summaries do not establish support for masks, token permutations or moved contracts.</p><p>This self-contained report was rendered only from compact aggregate records. No model, clean price, individual parameter draw, raw observation or machine path is included. Payload hashes identify the private sources; no pricing, posterior sampling or reference sampling occurred during reporting.</p>')
    chunks.append('<details><summary>Aggregate source identities</summary>' + _table(["Endpoint", "Evaluation SHA-256", "Private model SHA-256"], [
        [entry["label"], evaluation["source_evaluation_sha256"], run["source_hashes"].get("model_sha256", "historical baseline identity in run.json")]
        for entry, run, evaluation, _ in rows]) + '</details></main></body></html>')
    # Matplotlib SVG path data contains trailing spaces; normalize them for Git
    # while preserving newlines as legal SVG coordinate separators.
    document = '\n'.join(line.rstrip() for line in ''.join(chunks).splitlines()) + '\n'
    (directory / "report.html").write_text(document)
    summary = {"format": "heston_flexible_report_v1", "scientific_acceptance": False,
               "report_sha256": digest(directory / "report.html"), "report_bytes": len(document.encode()),
               "inline_svg_count": document.count("<svg"), "endpoint_count": len(rows),
               "source_study_sha256": digest(directory / "study.json"), "new_sampling": False, "new_pricing": False}
    write(directory / "report-summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[], help="Completed private MLP directory; may be repeated")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-training", type=Path, help="Relocated original baseline training record, checked against its stored hash")
    parser.add_argument("--baseline-bank", type=Path, help="Relocated original baseline bank record, checked against its stored hash")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true", help="Regenerate HTML from an already exported portable study")
    args = parser.parse_args()
    if args.render_only:
        result = render_report(args.out)
    else:
        if not args.run:
            parser.error("at least one --run is required for export")
        result = export_study(args.run, args.baseline, args.out,
                              baseline_training=args.baseline_training, baseline_bank=args.baseline_bank)
    print(json.dumps(result))

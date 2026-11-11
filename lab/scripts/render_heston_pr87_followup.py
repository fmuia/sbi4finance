"""Compact offline300-epoch PR87 report with one predeclared secondary endpoint.

Reads only hash-bound summaries, recovery quantiles and existing compact MCMC
records. No model loading, posterior sampling, pricing or native chain reads.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_pr87_followup as evaluation
from scripts import render_heston_pr87_100k as parent_report

read, binding = evaluation.read, evaluation.binding
LABELS = parent_report.LABELS
LIMIT_BYTES = 1024 * 1024
esc, fmt, percent, table = parent_report.esc, parent_report.fmt, parent_report.percent, parent_report.table
checked, load_recovery = parent_report.checked, parent_report.load_recovery
selected_references = parent_report.selected_references
metrics_table, support_table = parent_report.metrics_table, parent_report.support_table
recovery_plot, sbc_plot = parent_report.recovery_plot, parent_report.sbc_plot
conditional_tables = parent_report.conditional_tables


# Frozen necessary-screen conventions; no estimator acceptance is granted.
readiness = parent_report.readiness
readiness_html = parent_report.readiness_html
selected_plot = parent_report.parent_report.selected_plot


def comparison_row(label, record, rows, presentations):
    summary = record["population"]
    defined = summary["calibration_status"] == "defined"
    cov = summary["coverage"]["0.9"]["coverage"] if defined else None
    return [label, str(rows), f"{presentations:,}", fmt(summary["recovery"]["median_rmse"][-1], 5) if defined else "undefined",
        percent(cov[-1]) if defined else "undefined", f"{percent(min(cov[:5]))}–{percent(max(cov[:5]))}" if defined else "undefined",
        f'{sum(summary["rank_dkw"]["passed"])}/6' if defined else "undefined",
        percent(summary["support"]["outside_prior"]["sample_fraction"]), percent(summary["support"]["nonphysical"]["sample_fraction"])]


def checked_history(curve):
    if (not isinstance(curve, list) or len(curve) != 300
            or [point.get("epoch") for point in curve] != list(range(1, 301))
            or not np.isfinite([[point.get("loss", np.nan), point.get("val_loss", np.nan)] for point in curve]).all()):
        raise ValueError("complete finite300-epoch native history required")
    return curve


def learning_plot(curve, train_rows):
    curve = checked_history(curve)
    helper = parent_report.parent_report
    line, text, path = helper.line, helper.text, helper.path
    ys = [point[key] for point in curve for key in ("loss", "val_loss")]
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin: ymax += 1
    px = lambda x: 65 + (x - 1) / 299 * 885
    py = lambda y: 215 - (y - ymin) / (ymax - ymin) * 175
    pieces = [line(65, 215, 950, 215, "#9da9b2"), line(65, 40, 65, 215, "#9da9b2"),
        text(500, 254, f"Native fit_offline epoch over the same {train_rows:,} noisy datasets", anchor="middle")]
    for key, color, label, y in (("loss", helper.BLUE, "Training", 20), ("val_loss", helper.RED, "Validation", 36)):
        pieces += [path([(px(point["epoch"]), py(point[key])) for point in curve], color), text(700, y, label, color=color)]
    for x in (1, 50, 100, 150, 200, 250, 300): pieces.append(text(px(x), 233, str(x), anchor="middle"))
    for y in np.linspace(ymin, ymax, 4): pieces.append(text(58, py(y) + 3, fmt(y), anchor="end"))
    return helper.svg_panel("Native300-epoch training and validation objective", pieces,
        "FlowMatching objective, not density NLL. Final epoch300 remains primary; minimum-validation selection is secondary only.", height=275)


def conditional_support_table(record):
    rows = []
    for case, entry in record["cases"].items():
        support = entry["support"]
        rows.append([case, str(support["samples"]), str(support["outside_prior"]["sample_count"]),
            f'{100 * support["outside_prior"]["sample_fraction"]:.4f}%',
            str(support["nonphysical"]["sample_count"]), str(support["nonfinite"]["sample_count"])])
    return table(["Conditional case", "Raw draws", "Outside prior count", "Outside prior fraction", "Nonphysical count", "Nonfinite count"], rows)


def validate_record(record, protocol, protocol_path, protected, *, primary):
    if (record.get("format") != "heston_pr87_followup_evaluation_v1" or record.get("status") != "complete"
            or record.get("protocol") != binding(protocol_path) or record.get("task") != protocol["task"]["target"]
            or record.get("new_reference_sampling") is not False or record.get("reserved_arrays_opened") is not False
            or record.get("protected_inputs_unchanged") is not True
            or record.get("parent_10k_evaluation") != protocol["parent_10k_evaluation"]
            or record.get("prior_100k_evaluation") != protocol["prior_100k_evaluation"]
            or record.get("training_rows") not in (100000, 200000)
            or record.get("training_presentations") != record.get("native_epoch", 0) * record.get("training_rows", 0)):
        raise ValueError("completed immutable300-epoch PR87 diagnostics required")
    run = checked(record["training_run"], protected)
    accounting = evaluation.training_accounting(run)
    if record["training_rows"] != accounting["train_rows"] or record.get("training_accounting") != accounting:
        raise ValueError("saved followup training accounting differs")
    checkpoint = evaluation.checked_binding(record["checkpoint"], protected).parent
    endpoint = evaluation.checkpoint_endpoint(checkpoint, record, run)
    if endpoint != record.get("endpoint") or (endpoint == "primary_final") != primary:
        raise ValueError("report primary/secondary endpoint identity differs")
    for label, rows, count in (("native_validation", 100, 300), ("population", 512, 512)):
        summary = record[label]
        if summary["rows"] != rows or summary["draws_per_case"] != count:
            raise ValueError("complete unchanged native and historical panels required")
    return run


def build(protocol_path, evaluation_path, out, *, secondary_path=None):
    protocol_path, evaluation_path, out = map(Path, (protocol_path, evaluation_path, out))
    if out.exists(): raise FileExistsError("followup report output identity must be new")
    protected = {}
    protocol = checked(binding(protocol_path), protected)
    if (protocol.get("evaluation") != evaluation.evaluation_contract(protocol["task"]["target"])
            or protocol.get("sources", {}).get("renderer") != binding(__file__)
            or protocol["sources"].get("parent_renderer") != binding(parent_report.__file__)):
        raise ValueError("renderer or evaluation contract differs from frozen300-epoch declaration")
    for spec in protocol["sources"].values(): evaluation.checked_binding(spec, protected)
    primary = checked(binding(evaluation_path), protected)
    training = validate_record(primary, protocol, protocol_path, protected, primary=True)
    parent = checked(protocol["parent_10k_evaluation"], protected)
    if (parent.get("format") != "heston_pr87_reproduction_evaluation_v1" or parent.get("status") != "complete"
            or parent.get("native_epoch") != 200 or parent.get("task") != primary["task"]):
        raise ValueError("complete frozen10k parent diagnostics required")
    prior = checked(protocol["prior_100k_evaluation"], protected)
    if (prior.get("format") != "heston_pr87_100k_evaluation_v1" or prior.get("status") != "complete"
            or prior.get("native_epoch") != 200 or prior.get("endpoint") != "primary_final"
            or prior.get("task") != primary["task"]):
        raise ValueError("complete frozen100k200 primary diagnostics required")
    rows = primary["training_rows"]; size = f"{rows // 1000}k"
    for panel in ("native_validation", "population"):
        if any(primary[panel][key] != comparison[panel][key] for comparison in (parent, prior) for key in ("observations_sha256", "truth_sha256")):
            raise ValueError("followup panels differ from preserved10k/100k observations or truths")
    records = [(size + " primary — final epoch300", primary, evaluation_path)]
    best = training.get("best_validation_archive", {})
    secondary_required = best.get("epoch") != 300
    if secondary_required != (secondary_path is not None):
        raise ValueError("full secondary diagnostics required exactly when predeclared minimum-validation epoch differs")
    if secondary_path is not None:
        secondary_path = Path(secondary_path)
        secondary = checked(binding(secondary_path), protected)
        validate_record(secondary, protocol, protocol_path, protected, primary=False)
        if any(secondary[panel][key] != primary[panel][key] for panel in ("native_validation", "population")
                for key in ("observations_sha256", "truth_sha256")):
            raise ValueError("secondary panels differ from primary observations or truths")
        if secondary["training_run"] != primary["training_run"] or secondary["checkpoint"] == primary["checkpoint"]:
            raise ValueError("secondary must be a distinct endpoint of the same completed training")
        records.append((f'{size} secondary — minimum-validation epoch{secondary["native_epoch"]}', secondary, secondary_path))
    training_dir = Path(primary["training_run"]["path"]).parent
    curve_path = training_dir / "native-epoch-history.json"
    curve_spec = binding(curve_path)
    if training.get("native_epoch_history_sha256") != curve_spec["sha256"]:
        raise ValueError("native objective history differs from completed training")
    curve = checked(curve_spec, protected)
    if isinstance(curve, dict): curve = curve.get("history", curve.get("epochs", []))
    chunks = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PR87 FlowMatching:300-epoch followup</title>',
        '<style>body{font:15px/1.55 system-ui,sans-serif;color:#20313e;background:#f4f7f8;margin:0}main{max-width:1190px;margin:auto;padding:30px 22px 65px}h1{font-size:30px;line-height:1.2}h2{font-size:24px;margin-top:42px}h3{margin-top:28px}.notice{background:#fff6df;border-left:4px solid #bf8731;padding:14px 18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:12px;margin:14px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #dfe5e9;white-space:nowrap}th{background:#e9eff3}figure{margin:20px 0;padding:12px;background:white;border:1px solid #dfe5e9;border-radius:8px}svg{display:block;width:100%;height:auto}figcaption{font-size:12px;color:#536672;padding:7px}details{margin:18px 0}summary{cursor:pointer;font-weight:650}code{font-size:12px;overflow-wrap:anywhere}footer{margin-top:35px;border-top:1px solid #ccd7df;padding-top:18px;font-size:12px;color:#596a77}</style><main>',
        '<h1>PR87 FlowMatching:300-epoch followup — ' + size + '</h1>',
        '<p>Final epoch 300 is the primary endpoint. Each new fit starts from the declared native initialization with the same seed. The 100k run reuses the exact 100k200 training data; the 200k run appends 100,000 checked-Heston datasets from that parent post-append RNG state. Both preserve the exact 10k prefix, the same 100 validation rows and the frozen ZCA matrix fitted on 10k rows. Architecture 128×128×128, dropout 0.1, raw unbounded targets, native AdamW and batch 256 remain fixed.</p>',
        f'<p class="notice"><strong>{300 * rows / 1000000:g} million primary training presentations.</strong> Compare 2 million for 10k200 and 20 million for 100k200. This run completes 300 epochs and {300 * ((rows + 255) // 256):,} optimizer updates. These comparisons combine data and optimization changes; they do not isolate a data-size effect at equal compute. Validation was used during fitting and all posterior panels are development evidence. No held-out test or reserved pool is opened.</p>',
        '<p>Every raw posterior draw is retained: no clipping, filtering, rejection or resampling. One nonfinite draw makes calibration undefined for its whole panel. The predeclared minimum-validation checkpoint is a secondary endpoint only when its epoch differs from 300; it cannot replace the primary or trigger further checkpoint searches.</p>',
        '<h2>Preserved200-epoch baselines versus this300-epoch run</h2>',
        table(["Endpoint", "Training datasets", "Presentations", "c median RMSE", "c90% coverage", "Physical90% coverage range", "SBC screens", "Raw outside prior", "Raw nonphysical"],
            [comparison_row("10k parent — final epoch200", parent, 10000, 2000000),
             comparison_row("100k parent — final epoch200", prior, 100000, 20000000)] +
            [comparison_row(label, record, rows, record["training_presentations"]) for label, record, _ in records]),
        '<p>Every comparison uses the exact historical512×512 population observations, truths and seeds. The physical coverage range summarizes five targets; every target is reported separately below. Conditional comparisons retain the same three joint-MCMC references and4096 draws per case.</p>',
        learning_plot(curve, rows), '<details><summary>Frozen10k per-target population metrics</summary>', metrics_table(parent["population"]), '</details>',
        '<details><summary>Preserved100k200 per-target population metrics</summary>', metrics_table(prior["population"]), '</details>']
    qualifications = {}
    for label, record, source in records:
        native, population = record["native_validation"], record["population"]
        native_recovery, recovery = load_recovery(native, protected), load_recovery(population, protected)
        selected = selected_references(record["matched_mcmc_report"], recovery, record["task"], protected)
        checks = readiness(record); qualifications[record["endpoint"]] = checks
        chunks += ['<h2>' + esc(label) + '</h2>', '<h3>Official-quality readiness across all six targets</h3>',
            '<p>Coverage checks ask whether90% is inside each marginal95% Wilson interval; SBC checks use the saved six-marginal DKW screen within each panel. Conditional mean and width checks retain the earlier descriptive0.3-reference-SD and0.75–1.25 conventions. These are necessary diagnostic screens, without a new simultaneous acceptance claim. Prior support and physical validity are evaluated separately on every raw draw.</p>', readiness_html(checks),
            '<h3>Native validation:100 datasets ×300 draws</h3>', metrics_table(native)]
        if native_recovery is not None:
            chunks += [recovery_plot(native_recovery, record["task"], label + ': native100-case validation'),
                sbc_plot(native).replace('historical 512-case panel', 'native 100-case validation panel')]
        chunks += ['<h3>Historical full-prior population:512 datasets ×512 draws</h3>', metrics_table(population)]
        if recovery is not None:
            chunks += [recovery_plot(recovery, record["task"], label + ': historical512-case panel', rows=128), sbc_plot(population)]
        if selected is not None:
            chunks += ['<h3>Frozen eight-case joint-MCMC interval comparison</h3>', selected_plot(selected),
                table(["Case", "Population row", "Reference"], [[str(i + 1), str(index), reason] for i, (index, reason) in enumerate(zip(selected["indices"], selected["reasons"]))])]
        chunks += ['<h3>Raw support: native100×300</h3>', support_table(native), '<h3>Raw support: historical512×512</h3>', support_table(population),
            '<p>Nonphysical means a nonpositive variance, reversion, volatility or noise scale; correlation outside[−1,1]; or a nonfinite value. Zero draws are discarded, clipped or resampled.</p>',
            '<h3>Three unchanged conditional cases:4096 raw draws each</h3>',
            '<p>All six physical coordinates are compared to ready joint references. Nonphysical draws remain in these comparisons; derived products are skipped when invalid. The market case assesses inference under the assumed likelihood; market-model adequacy requires predictive or residual checks.</p>', conditional_tables(record), conditional_support_table(record),
            '<details><summary>All coverage levels for this endpoint</summary>']
        for panel_label, summary in (("Native100", native), ("Historical512", population)):
            if summary["calibration_status"] == "defined":
                chunks.append('<h3>' + panel_label + '</h3>' + table(["Target", "50%", "80%", "90%", "95%"], [[name] + [percent(summary["coverage"][level]["coverage"][j]) for level in ("0.5", "0.8", "0.9", "0.95")] for j, name in enumerate(LABELS)]))
        chunks += ['</details><p>Evaluation SHA256: <code>' + binding(source)["sha256"] + '</code></p>']
    chunks += ['<footer>Offline report from saved summaries and compact intervals; no model loading, posterior queries, pricing or new MCMC.<br>Protocol SHA256: <code>' + binding(protocol_path)["sha256"] + '</code><br>Parent10k evaluation SHA256: <code>' + protocol["parent_10k_evaluation"]["sha256"] + '</code></footer></main></html>']
    document = ''.join(chunks)
    if len(document.encode()) >= LIMIT_BYTES: raise ValueError("compact Paseo report must remain below one MiB")
    if any(evaluation.fixed.digest(path) != sha for path, sha in protected.items()):
        raise ValueError("protected report input changed during rendering")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(document)
    summary = {"format": "heston_pr87_followup_report_v1", "protocol": binding(protocol_path),
        "evaluation": binding(evaluation_path), "secondary_evaluation": binding(secondary_path) if secondary_path else None,
        "parent_10k_evaluation": protocol["parent_10k_evaluation"], "prior_100k_evaluation": protocol["prior_100k_evaluation"], "report": binding(out), "report_bytes": len(document.encode()),
        "inline_svg_count": document.count('<svg'), "limit_bytes": LIMIT_BYTES, "protected_inputs_unchanged": True,
        "producer": binding(__file__), "readiness": qualifications, "scientific_acceptance": False, "promotion": False}
    evaluation.fixed.write_json(out.with_name("report-summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "evaluation", "out"): parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--secondary", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.protocol, args.evaluation, args.out, secondary_path=args.secondary)))


if __name__ == "__main__": main()

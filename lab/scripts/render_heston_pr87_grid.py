"""Compact saved-data report for the revised63-quote Heston grid.

The old54 control is contextual evidence for a different observation task.
All new conditional comparisons require the new63 likelihood's own reference.
No model loading, pricing, posterior queries or native chain reads occur.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_pr87_grid as evaluation
from scripts import render_heston_pr87_followup as parent_report

read, binding = evaluation.read, evaluation.binding
checked, load_recovery = parent_report.checked, parent_report.load_recovery
LABELS, LIMIT_BYTES = parent_report.LABELS, 1024 * 1024
esc, fmt, percent, table = parent_report.esc, parent_report.fmt, parent_report.percent, parent_report.table
metrics_table, support_table = parent_report.metrics_table, parent_report.support_table
recovery_plot, sbc_plot = parent_report.recovery_plot, parent_report.sbc_plot
conditional_tables, conditional_support_table = parent_report.conditional_tables, parent_report.conditional_support_table
readiness, readiness_html = parent_report.readiness, parent_report.readiness_html
checked_history = parent_report.checked_history


def validate_record(record, protocol, protocol_path, protected, *, primary):
    if (record.get("format") != "heston_pr87_grid_evaluation_v1" or record.get("status") != "complete"
            or record.get("protocol") != binding(protocol_path) or record.get("task") != protocol["task"]["target"]
            or record.get("task", {}).get("quote_count") != 63 or record.get("training_rows") not in (200000, 300000)
            or record.get("old54_control") != protocol["old54_control"]
            or record.get("reference_protocol") != protocol["reference_protocol"]
            or record.get("conditional_panel") != protocol["conditional_panel"]
            or record.get("protected_inputs_unchanged") is not True
            or record.get("new_reference_sampling") is not False or record.get("reserved_arrays_opened") is not False
            or record.get("training_presentations") != record.get("native_epoch", 0)*record.get("training_rows", 0)
            or set(record.get("cases", {})) != set(evaluation.reference.CASES)
            or any(entry.get("support", {}).get("samples") != 4096 for entry in record.get("cases", {}).values())):
        raise ValueError("completed immutable revised63-grid diagnostics required")
    run = checked(record["training_run"], protected); accounting = evaluation.training_accounting(run)
    if record.get("training_accounting") != accounting or record["training_rows"] != accounting["train_rows"]:
        raise ValueError("saved revised-grid optimizer accounting differs")
    checkpoint = evaluation.checked_binding(record["checkpoint"], protected).parent
    endpoint = evaluation.checkpoint_endpoint(checkpoint, record, run)
    if endpoint != record.get("endpoint") or (endpoint == "primary_final") != primary:
        raise ValueError("revised-grid primary/secondary endpoint identity differs")
    for name, rows, draws in (("native_validation", 100, 300), ("population", 512, 512)):
        if record[name]["rows"] != rows or record[name]["draws_per_case"] != draws:
            raise ValueError("complete native100 and paired historical512 panels required")
    return run


def comparison_row(label, record, quotes, rows, presentations):
    values = parent_report.comparison_row(label, record, rows, presentations)
    return values[:1] + [str(quotes)] + values[1:]


def build(protocol_path, evaluation_path, out, *, secondary_path=None):
    protocol_path, evaluation_path, out = map(Path, (protocol_path, evaluation_path, out))
    if out.exists(): raise FileExistsError("revised-grid presentation identity must be new")
    protected = {}; protocol = checked(binding(protocol_path), protected)
    if (protocol.get("evaluation") != evaluation.evaluation_contract(protocol["task"]["target"])
            or protocol.get("sources", {}).get("renderer") != binding(__file__)
            or protocol["sources"].get("parent_renderer") != binding(parent_report.__file__)):
        raise ValueError("renderer/evaluation contract differs from frozen revised-grid declaration")
    for spec in protocol["sources"].values(): evaluation.checked_binding(spec, protected)
    primary = checked(binding(evaluation_path), protected)
    run = validate_record(primary, protocol, protocol_path, protected, primary=True)
    control = checked(protocol["old54_control"], protected)
    if (control.get("format") != "heston_pr87_followup_evaluation_v1"
            or control.get("status") != "complete" or control.get("native_epoch") != 300 or control.get("training_rows") != 200000
            or control.get("endpoint") != "primary_final" or control.get("task", {}).get("quote_count") != 54
            or control["task"]["parameter_names"] != primary["task"]["parameter_names"]
            or control["task"]["prior_bounds"] != primary["task"]["prior_bounds"]):
        raise ValueError("preserved old54/200k300 control with same six-target priors required")
    for name in ("native_validation", "population"):
        if primary[name]["truth_sha256"] != control[name]["truth_sha256"]:
            raise ValueError("new63 truth rows differ from paired old54 control")
    rows = primary["training_rows"]; size = f"{rows//1000}k"
    records = [(size+" new63 primary — final epoch300", primary, evaluation_path)]
    distinct = run["best_validation_archive"]["epoch"] != 300
    if distinct != (secondary_path is not None): raise ValueError("complete secondary required exactly when minimum-validation epoch differs")
    if secondary_path is not None:
        secondary_path = Path(secondary_path); secondary = checked(binding(secondary_path), protected)
        validate_record(secondary, protocol, protocol_path, protected, primary=False)
        if secondary["training_run"] != primary["training_run"] or secondary["common_bank"] != primary["common_bank"]:
            raise ValueError("secondary must be the same fit and frozen common bank")
        if any(secondary[name][key] != primary[name][key] for name in ("native_validation", "population") for key in ("truth_sha256", "observations_sha256")):
            raise ValueError("secondary observations/truth rows differ")
        records.append((f'{size} new63 secondary — minimum-validation epoch{secondary["native_epoch"]}', secondary, secondary_path))
    curve_path = Path(primary["training_run"]["path"]).parent/"native-epoch-history.json"
    curve_binding = binding(curve_path)
    if run.get("native_epoch_history_sha256") != curve_binding["sha256"]: raise ValueError("native history differs from completed revised-grid fit")
    curve = checked(curve_binding, protected); parent_report.checked_history(curve)
    css = 'body{font:15px/1.55 system-ui,sans-serif;color:#20313e;background:#f4f7f8;margin:0}main{max-width:1190px;margin:auto;padding:30px 22px 65px}h1{font-size:30px;line-height:1.2}h2{font-size:24px;margin-top:42px}h3{margin-top:28px}.notice{background:#fff6df;border-left:4px solid #bf8731;padding:14px 18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:12px;margin:14px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #dfe5e9;white-space:nowrap}th{background:#e9eff3}figure{margin:20px 0;padding:12px;background:white;border:1px solid #dfe5e9;border-radius:8px}svg{display:block;width:100%;height:auto}figcaption{font-size:12px;color:#536672;padding:7px}details{margin:18px 0}summary{cursor:pointer;font-weight:650}code{font-size:12px;overflow-wrap:anywhere}footer{margin-top:35px;border-top:1px solid #ccd7df;padding-top:18px;font-size:12px;color:#596a77}'
    chunks = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Revised63 Heston grid study</title><style>'+css+'</style><main>',
        '<h1>Revised63-quote Heston grid — '+size+' simulations ×300 epochs</h1>',
        '<p>The supported revised grid uses63 actual contracts, the declared final-call +1.5 standardized target in fixed16% volatility coordinates and an approximately10.7-month longest actual tenor. The canonical Heston priors, normalized Gaussian price likelihood, half-spread noise convention, raw six-target adapter,128×128×128 FlowMatching, dropout0.1 and native AdamW remain fixed.</p>',
        '<details><summary>Versioned numerical representation</summary><p>The new-grid checked pricer uses an analytic Black control variate with the unchanged Heston characteristic function, financial priors, quadrature ladder, pricing bounds and measurement-scale precision gates. This addresses cancellation for nearly worthless options without clipping prices or inflating measurement noise. Reference precision checks remain explicit sampled checks, not a prior-wide certificate.</p></details>',
        '<p class="notice"><strong>Different observation tasks and optimization budgets.</strong> The preserved old54/200k300 control and new63/200k300 each use60 million presentations; new63/300k300 uses90 million. New quote geometry, measured half-spreads and the63-feature condition change the likelihood and network input dimension. More quotes are more independent information only under the assumed independent noise model. The200k→300k comparison combines more simulations with more optimization.</p>',
        '<p>Both new fits use the exact same first200k rows of one frozen300k bank. One new63 ZCA matrix is fitted only on the first10k new-grid training rows and frozen across both fits; the old54 matrix is preserved for its own task. Native100 and historical512 six-target truths remain paired with the old control, while the quote vectors belong to the revised grid. All these panels are development evidence; the validation split was used during fitting.</p>',
        '<p>Final epoch300 remains primary. A distinct minimum-validation checkpoint is secondary only. Every raw draw is retained, including values outside the prior; no clipping, rejection, filtering or resampling occurs. Nonfinite draws make whole-panel calibration undefined. Old54 MCMC chains cannot validate the new63 posterior; all conditional comparisons below use new target-matched references.</p>',
        '<h2>Old54 control and this new63 fit</h2>',
        table(["Endpoint", "Quotes", "Training simulations", "Presentations", "c median RMSE", "c90% coverage", "Physical90% coverage range", "SBC screens", "Raw outside prior", "Raw nonphysical"],
            [comparison_row("Old54 control —200k300 primary", control, 54, 200000, 60000000)] + [comparison_row(label, record, 63, rows, record["training_presentations"]) for label, record, _ in records]),
        '<p>Coverage/SBC compare calibration to the same generating truths under each grid’s own observations. These conditional posterior targets differ; a direct comparison against the old54 posterior is not a new63 fidelity test.</p>',
        parent_report.learning_plot(curve, rows), '<details><summary>Preserved old54/200k300 per-target population metrics</summary>', metrics_table(control["population"]), '</details>']
    qualifications = {}
    for label, record, path in records:
        checks = readiness(record); qualifications[record["endpoint"]] = checks
        chunks += ['<h2>'+esc(label)+'</h2>', '<h3>All-six-target diagnostic readiness</h3>',
            '<p>Necessary screens retain the earlier conventions: nominal90% inside each marginal95% Wilson interval; six-marginal DKW within each panel; conditional mean difference≤0.3 reference SD and90% width ratio0.75–1.25. These reused development screens do not establish simultaneous or independent acceptance. Missing, failed or unconverged new63 references remain inconclusive.</p>', readiness_html(checks)]
        for name, title, plot_rows in (("native_validation", "Native validation:100×300", 100), ("population", "Paired historical truths:512×512", 128)):
            summary = record[name]; recovery = load_recovery(summary, protected)
            chunks += ['<h3>'+title+'</h3>', metrics_table(summary)]
            if recovery is not None:
                chunks.append(recovery_plot(recovery, record["task"], label+': '+title, rows=plot_rows))
                chunks.append(sbc_plot(summary).replace('historical 512-case panel', 'native 100-case validation panel') if name == "native_validation" else sbc_plot(summary))
            chunks += ['<h3>'+title+' raw support</h3>', support_table(summary)]
        chunks += ['<h3>Three new63 conditional references:4096 raw neural draws each</h3>',
            '<p>Reference status includes four-chain convergence for all six parameters and variance products, plus canonical sampled price/likelihood refinement. A capped or failed reference supplies no reference interval or agreement claim. Market comparisons assess inference under the assumed likelihood; they do not establish market-model adequacy.</p>', conditional_tables(record), conditional_support_table(record),
            '<details><summary>All saved marginal coverage levels</summary>']
        for panel_name, panel_label in (("native_validation", "Native100"), ("population", "Paired historical512")):
            summary = record[panel_name]
            if summary["calibration_status"] == "defined":
                chunks += ['<h3>'+panel_label+'</h3>', table(["Target", "50%", "80%", "90%", "95%"],
                    [[name]+[percent(summary["coverage"][level]["coverage"][j]) for level in ("0.5", "0.8", "0.9", "0.95")] for j, name in enumerate(LABELS)])]
        chunks += ['</details><p>Evaluation SHA256: <code>'+binding(path)["sha256"]+'</code></p>']
    chunks += ['<footer>Saved summaries and quantiles only; no model loading, posterior queries, pricing or reference sampling during rendering.<br>Protocol SHA256: <code>'+binding(protocol_path)["sha256"]+'</code><br>New63 reference protocol SHA256: <code>'+protocol["reference_protocol"]["sha256"]+'</code></footer></main></html>']
    document = ''.join(chunks)
    if len(document.encode()) >= LIMIT_BYTES: raise ValueError("compact revised-grid report must remain below one MiB")
    if any(evaluation.fixed.digest(path) != sha for path, sha in protected.items()): raise ValueError("protected saved-grid presentation input changed")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(document)
    result = {"format": "heston_pr87_grid_report_v1", "protocol": binding(protocol_path), "evaluation": binding(evaluation_path),
        "secondary_evaluation": binding(secondary_path) if secondary_path else None, "old54_control": protocol["old54_control"],
        "report": binding(out), "report_bytes": len(document.encode()), "inline_svg_count": document.count('<svg'),
        "limit_bytes": LIMIT_BYTES, "readiness": qualifications, "protected_inputs_unchanged": True,
        "producer": binding(__file__), "new_neural_sampling": False, "new_reference_sampling": False,
        "scientific_acceptance": False, "promotion": False}
    evaluation.fixed.write_json(out.with_name("report-summary.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "evaluation", "out"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--secondary", type=Path); args = parser.parse_args()
    print(json.dumps(build(args.protocol,args.evaluation,args.out,secondary_path=args.secondary)))

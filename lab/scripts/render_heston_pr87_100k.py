"""Compact offline 10k/100k PR87 report with one predeclared secondary endpoint.

Reads only hash-bound summaries, recovery quantiles and existing compact MCMC
records. No model loading, posterior sampling, pricing or native chain reads.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_pr87_100k as evaluation
from scripts import render_heston_pr87_reproduction as parent_report

read, binding = evaluation.read, evaluation.binding
LABELS = parent_report.LABELS
LIMIT_BYTES = 1024 * 1024
esc, fmt, percent, table = parent_report.esc, parent_report.fmt, parent_report.percent, parent_report.table
checked, load_recovery = parent_report.checked, parent_report.load_recovery
selected_references = parent_report.selected_references
metrics_table, support_table = parent_report.metrics_table, parent_report.support_table
recovery_plot, sbc_plot = parent_report.recovery_plot, parent_report.sbc_plot
conditional_tables = parent_report.conditional_tables


def readiness(record):
    """Describe necessary diagnostic checks; never grant estimator acceptance."""
    panels = {}
    for name in ("native_validation", "population"):
        summary = record[name]
        finite = summary["calibration_status"] == "defined" and summary["support"]["nonfinite"]["value_count"] == 0
        intervals = summary["coverage"]["0.9"]["wilson"] if finite else []
        panels[name] = {"finite": finite,
            "all_six_nominal90_within_wilson95": finite and all(lo <= .9 <= hi for lo, hi in intervals),
            "all_six_sbc_inside_dkw": finite and all(summary["rank_dkw"]["passed"]),
            "all_raw_draws_inside_prior": summary["support"]["outside_prior"]["sample_count"] == 0,
            "all_raw_draws_physical": summary["support"]["nonphysical"]["sample_count"] == 0}
    cases = {}
    for name in evaluation.previous.CASES:
        entry = record["cases"].get(name, {})
        comparison = entry.get("comparison", {})
        defined = entry.get("reference_ready") is True and comparison.get("status") == "defined"
        mean = comparison.get("standardized_mean_error", [])
        width = comparison.get("width_ratio", [])
        support = entry.get("support", {})
        cases[name] = {"reference_and_comparison_ready": defined,
            "all_six_means_within_point3_reference_sd": defined and len(mean) == 6 and all(x <= .3 for x in mean),
            "all_six_width_ratios_between_point75_and1point25": defined and len(width) == 6 and all(.75 <= x <= 1.25 for x in width),
            "all_raw_draws_inside_prior": support.get("outside_prior", {}).get("sample_count", 1) == 0,
            "all_raw_draws_physical": support.get("nonphysical", {}).get("sample_count", 1) == 0,
            "all_raw_draws_finite": support.get("nonfinite", {}).get("value_count", 1) == 0}
    necessary = all(all(checks.values()) for checks in [*panels.values(), *cases.values()])
    return {"panels": panels, "conditional_cases": cases, "necessary_diagnostics_pass": necessary,
        "official_quality_established": False, "scientific_acceptance": False, "promotion": False,
        "scope": "Descriptive necessary checks using existing marginal and conditional conventions; no new acceptance family or independent held-out confirmation."}


def comparison_row(label, record, rows, presentations):
    summary = record["population"]
    defined = summary["calibration_status"] == "defined"
    cov = summary["coverage"]["0.9"]["coverage"] if defined else None
    return [label, str(rows), f"{presentations:,}", fmt(summary["recovery"]["median_rmse"][-1], 5) if defined else "undefined",
        percent(cov[-1]) if defined else "undefined", f"{percent(min(cov[:5]))}–{percent(max(cov[:5]))}" if defined else "undefined",
        f'{sum(summary["rank_dkw"]["passed"])}/6' if defined else "undefined",
        percent(summary["support"]["outside_prior"]["sample_fraction"]), percent(summary["support"]["nonphysical"]["sample_fraction"])]


def readiness_html(checks):
    rows = []
    for panel, values in checks["panels"].items():
        rows.append([panel, *["yes" if value else "no" for value in values.values()]])
    chunks = [table(["Population panel", "All finite", "All six 90% Wilson intervals include90%", "All six SBC screens", "All draws in prior", "All draws physical"], rows)]
    chunks.append(table(["Conditional case", "Ready joint reference", "All six means ≤0.3 ref SD", "All six width ratios0.75–1.25", "All draws in prior", "All draws physical", "All finite"],
        [[name, *["yes" if value else "no" for value in values.values()]] for name, values in checks["conditional_cases"].items()]))
    if checks["necessary_diagnostics_pass"]:
        chunks.append('<p class="notice">The displayed necessary checks pass, but official quality is still unestablished: these reused development panels have no independent confirmation or predeclared simultaneous acceptance family. No acceptance or promotion is granted.</p>')
    else:
        chunks.append('<p class="notice"><strong>Official-quality readiness is not established.</strong> At least one coverage, SBC, conditional-MCMC agreement or raw-support check fails. A favorable recovery plot cannot override these failures. No acceptance or promotion is granted.</p>')
    return ''.join(chunks)


def learning_plot(curve):
    # The frozen helper is a pure saved-history renderer; only its run label changes.
    return parent_report.learning_plot(curve).replace("same fresh 10,000 noisy datasets", "same 100,000 noisy datasets")


def validate_record(record, protocol, protocol_path, protected, *, primary):
    if (record.get("format") != "heston_pr87_100k_evaluation_v1" or record.get("status") != "complete"
            or record.get("protocol") != binding(protocol_path) or record.get("task") != protocol["task"]["target"]
            or record.get("new_reference_sampling") is not False or record.get("reserved_arrays_opened") is not False
            or record.get("protected_inputs_unchanged") is not True
            or record.get("parent_10k_evaluation") != protocol["parent_10k_evaluation"]
            or record.get("training_rows") != 100000 or record.get("training_presentations") != record.get("native_epoch", 0) * 100000):
        raise ValueError("completed immutable100k PR87 diagnostics required")
    run = checked(record["training_run"], protected)
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
    if out.exists(): raise FileExistsError("100k report output identity must be new")
    protected = {}
    protocol = checked(binding(protocol_path), protected)
    if (protocol.get("evaluation") != evaluation.evaluation_contract(protocol["task"]["target"])
            or protocol.get("sources", {}).get("renderer") != binding(__file__)
            or protocol["sources"].get("parent_renderer") != binding(parent_report.__file__)):
        raise ValueError("renderer or evaluation contract differs from frozen100k declaration")
    for spec in protocol["sources"].values(): evaluation.checked_binding(spec, protected)
    primary = checked(binding(evaluation_path), protected)
    training = validate_record(primary, protocol, protocol_path, protected, primary=True)
    parent = checked(protocol["parent_10k_evaluation"], protected)
    if (parent.get("format") != "heston_pr87_reproduction_evaluation_v1" or parent.get("status") != "complete"
            or parent.get("native_epoch") != 200 or parent.get("task") != primary["task"]):
        raise ValueError("complete frozen10k parent diagnostics required")
    for panel in ("native_validation", "population"):
        if any(primary[panel][key] != parent[panel][key] for key in ("observations_sha256", "truth_sha256")):
            raise ValueError("100k report panels differ from preserved10k observations or truths")
    records = [("100k primary — final epoch200", primary, evaluation_path)]
    best = training.get("best_validation_archive", {})
    secondary_required = best.get("epoch") != 200
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
        records.append((f'100k secondary — minimum-validation epoch{secondary["native_epoch"]}', secondary, secondary_path))
    training_dir = Path(primary["training_run"]["path"]).parent
    curve_path = training_dir / "native-epoch-history.json"
    curve_spec = binding(curve_path)
    if training.get("native_epoch_history_sha256") != curve_spec["sha256"]:
        raise ValueError("native objective history differs from completed training")
    curve = checked(curve_spec, protected)
    if isinstance(curve, dict): curve = curve.get("history", curve.get("epochs", []))
    chunks = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PR87 FlowMatching:100k data extension</title>',
        '<style>body{font:15px/1.55 system-ui,sans-serif;color:#20313e;background:#f4f7f8;margin:0}main{max-width:1190px;margin:auto;padding:30px 22px 65px}h1{font-size:30px;line-height:1.2}h2{font-size:24px;margin-top:42px}h3{margin-top:28px}.notice{background:#fff6df;border-left:4px solid #bf8731;padding:14px 18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:12px;margin:14px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #dfe5e9;white-space:nowrap}th{background:#e9eff3}figure{margin:20px 0;padding:12px;background:white;border:1px solid #dfe5e9;border-radius:8px}svg{display:block;width:100%;height:auto}figcaption{font-size:12px;color:#536672;padding:7px}details{margin:18px 0}summary{cursor:pointer;font-weight:650}code{font-size:12px;overflow-wrap:anywhere}footer{margin-top:35px;border-top:1px solid #ccd7df;padding-top:18px;font-size:12px;color:#596a77}</style><main>',
        '<h1>PR87 FlowMatching:100k data extension</h1>',
        '<p>Final epoch200 is the primary endpoint. Training preserves the exact parent10,000-dataset prefix, the same100 validation datasets and the frozen parent full-covariance ZCA matrix, then appends90,000 checked-Heston datasets from the parent post-validation RNG state. Architecture128×128×128, dropout0.1, raw unbounded targets, native AdamW and batch256 remain fixed.</p>',
        '<p class="notice"><strong>20 million versus2 million training presentations.</strong> Both runs complete200 epochs, but the100k primary has78,200 updates versus8,000 for10k. This comparison combines more data and more optimization; it does not isolate a data-size effect at equal compute. The validation split was used during fitting and all posterior panels are development evidence. No held-out test or reserved pool is opened.</p>',
        '<p>Raw posterior draws are retained in full: no clipping, filtering, rejection or resampling. A single nonfinite draw makes calibration undefined for its whole panel. The predeclared minimum-validation checkpoint is evaluated as a secondary endpoint only when its epoch differs from200; it cannot replace the primary or trigger further checkpoint searches.</p>',
        '<h2>Frozen10k baseline versus100k</h2>',
        table(["Endpoint", "Training datasets", "Presentations", "c median RMSE", "c90% coverage", "Physical90% coverage range", "SBC screens", "Raw outside prior", "Raw nonphysical"],
            [comparison_row("10k parent — final epoch200", parent, 10000, 2000000)] +
            [comparison_row(label, record, 100000, record["training_presentations"]) for label, record, _ in records]),
        '<p>All comparison metrics use the exact historical512×512 population panel, with the same observations, truths and seeds. The physical coverage range summarizes five targets; every target is reported separately below. Conditional comparisons use the same three existing joint-MCMC references and4096 draws per case.</p>',
        learning_plot(curve), '<details><summary>Frozen10k per-target population metrics</summary>', metrics_table(parent["population"]), '</details>']
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
            chunks += ['<h3>Frozen eight-case joint-MCMC interval comparison</h3>', parent_report.selected_plot(selected),
                table(["Case", "Population row", "Reference"], [[str(i + 1), str(index), reason] for i, (index, reason) in enumerate(zip(selected["indices"], selected["reasons"]))])]
        chunks += ['<h3>Raw support: native100×300</h3>', support_table(native), '<h3>Raw support: historical512×512</h3>', support_table(population),
            '<p>Nonphysical means a nonpositive variance, reversion, volatility or noise scale; correlation outside[−1,1]; or a nonfinite value. Zero draws are discarded, clipped or resampled.</p>',
            '<h3>Three unchanged conditional cases:4096 raw draws each</h3>',
            '<p>All six physical coordinates are compared to ready joint references. Nonphysical draws remain in these comparisons; derived products are skipped when invalid. The market case assesses inference under the assumed likelihood; market-model adequacy requires predictive or residual checks.</p>', conditional_tables(record),
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
    summary = {"format": "heston_pr87_100k_report_v1", "protocol": binding(protocol_path),
        "evaluation": binding(evaluation_path), "secondary_evaluation": binding(secondary_path) if secondary_path else None,
        "parent_10k_evaluation": protocol["parent_10k_evaluation"], "report": binding(out), "report_bytes": len(document.encode()),
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

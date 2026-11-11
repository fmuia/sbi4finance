"""Assemble one compact campaign preview from saved completed reports only.

This presentation consumer formats already-saved metrics and embeds the existing
per-run report bodies. It does not load models, query posteriors, price options,
open native chains or recompute scientific diagnostics.
"""
import argparse
import html
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_pr87_followup as saved

LIMIT_BYTES = 1024 * 1024
binding, read = saved.binding, saved.read


def report_body(document, prefix):
    body = re.search(r'<main>([\s\S]*?)</main>', document)
    if body is None or document.count('<main>') != 1:
        raise ValueError("exactly one saved report main body required")
    content = body.group(1)
    ids = re.findall(r'\bid="([^"]+)"', content)
    if len(ids) != len(set(ids)):
        raise ValueError("saved report contains duplicate identifiers")
    for identifier in ids:
        new = prefix + "-" + identifier
        content = content.replace('id="' + identifier + '"', 'id="' + new + '"')
        content = content.replace('href="#' + identifier + '"', 'href="#' + new + '"')
        content = content.replace('url(#' + identifier + ')', 'url(#' + new + ')')
    return '<section id="' + prefix + '">' + content + '</section>'


def build(campaign_path, out):
    campaign_path, out = map(lambda p: Path(p).resolve(), (campaign_path, out))
    summary_path = out.with_name("campaign-report-summary.json")
    if out.exists() or summary_path.exists():
        raise FileExistsError("campaign presentation identity must be new")
    protected = {}
    campaign = saved.checked(binding(campaign_path), protected)
    if (campaign.get("format") != "heston_pr87_followup_campaign_v1" or campaign.get("status") != "complete"
            or campaign.get("scientific_acceptance") is not False):
        raise ValueError("completed bounded followup campaign required")
    declaration = saved.checked(campaign["protocol"], protected)
    if declaration.get("format") != "heston_pr87_followup_campaign_protocol_v1":
        raise ValueError("frozen followup campaign declaration required")
    if list(campaign.get("runs", {})) not in (["100k-300"], ["100k-300", "200k-300"]):
        raise ValueError("only the two predeclared run identities are allowed")
    rows, bodies, reports, endpoints = [], [], [], []
    first_study = None
    style = None
    for run_id, completed in campaign["runs"].items():
        study_spec = declaration["runs"][run_id]["protocol"]
        study_path = Path(study_spec["path"])
        study = saved.checked(study_spec, protected)
        first_study = first_study or study
        report_path = study_path.parent / "report.html"
        report_summary_path = study_path.parent / "report-summary.json"
        report_summary = saved.checked(binding(report_summary_path), protected)
        if (report_summary.get("format") != "heston_pr87_followup_report_v1"
                or report_summary.get("protocol") != study_spec
                or report_summary.get("protected_inputs_unchanged") is not True
                or report_summary.get("scientific_acceptance") is not False
                or report_summary.get("promotion") is not False
                or report_summary["report"] != binding(report_path)):
            raise ValueError("completed hash-bound saved per-run report required")
        saved.evaluation.checked_binding(report_summary["report"], protected)
        document = report_path.read_text()
        if len(document.encode()) >= LIMIT_BYTES or '<script' in document.lower():
            raise ValueError("saved report must be compact and script-free")
        match = re.search(r'<style>([\s\S]*?)</style>', document)
        if match is None: raise ValueError("saved report style required")
        if style is None: style = match.group(1)
        bodies.append(report_body(document, "run-" + run_id))
        reports.append({"run_id": run_id, "report": binding(report_path), "summary": binding(report_summary_path)})
        training = saved.checked(completed["training"], protected)
        if training.get("status") != "complete" or training.get("completed_epochs") != 300:
            raise ValueError("each saved run must complete its declared300 epochs")
        expected_secondary = training["best_validation_archive"]["epoch"] != 300
        if expected_secondary != (report_summary.get("secondary_evaluation") is not None):
            raise ValueError("saved report secondary endpoint policy differs")
        for role, key in (("primary", "evaluation"), ("secondary", "secondary_evaluation")):
            spec = report_summary.get(key)
            if spec is None: continue
            record = saved.checked(spec, protected)
            if (record.get("status") != "complete" or record.get("format") != "heston_pr87_followup_evaluation_v1"
                    or record.get("protocol") != study_spec or record.get("training_rows") != declaration["runs"][run_id]["train_rows"]):
                raise ValueError("saved comparison endpoint identity differs")
            label = run_id + (" primary — final epoch300" if role == "primary" else f' secondary — minimum-validation epoch{record["native_epoch"]}')
            rows.append(saved.comparison_row(label, record, record["training_rows"], record["training_presentations"]))
            endpoints.append({"run_id": run_id, "role": role, "evaluation": spec,
                "readiness": report_summary["readiness"][record["endpoint"]]})
    parent = saved.checked(first_study["parent_10k_evaluation"], protected)
    prior = saved.checked(first_study["prior_100k_evaluation"], protected)
    rows = [saved.comparison_row("10k200 primary", parent, 10000, 2000000),
        saved.comparison_row("100k200 primary", prior, 100000, 20000000)] + rows
    navigation = ' · '.join('<a href="' + html.escape(str(Path(item["report"]["path"]).relative_to(out.parent)), quote=True) + '">' + item["run_id"] + ' individual report</a>' for item in reports)
    introduction = '<h1>PR87 FlowMatching:300-epoch followup campaign</h1><p>' + navigation + '</p>'
    introduction += '<p>Final epoch300 is primary for each new fit. Distinct minimum-validation checkpoints remain secondary. The declared 100k300 and 200k300 primary budgets are 30 million and 60 million training presentations, compared with 20 million for 100k200 and 2 million for 10k200. Only completed runs appear below. The same100 validation cases, historical512-case population, posterior seeds and existing joint-MCMC references are preserved. These are development comparisons with different data and optimization budgets.</p>'
    introduction += saved.table(["Endpoint", "Training datasets", "Presentations", "c median RMSE", "c90% coverage", "Physical90% coverage range", "SBC screens", "Raw outside prior", "Raw nonphysical"], rows)
    introduction += '<p class="notice"><strong>No scientific acceptance or promotion.</strong> Point recovery and c coverage must be read alongside all six marginal coverage/SBC diagnostics, raw support and conditional joint-MCMC comparisons below. Every raw draw is retained. No held-out test pool or new reference sampling was used.</p>'
    def assemble(selected_bodies):
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PR87 FlowMatching followup campaign</title><style>' + style + '</style><main>' + introduction + ''.join(selected_bodies) + '<footer>Aggregate presentation from saved hash-bound metrics and per-run report bodies. No scientific diagnostics were recomputed.<br>Campaign SHA256: <code>' + binding(campaign_path)["sha256"] + '</code></footer></main></html>'
    document = assemble(bodies)
    complete_bodies = len(bodies)
    if len(document.encode()) >= LIMIT_BYTES:
        # Keep the campaign overview and latest full diagnostics if necessary.
        document = assemble(bodies[-1:]); complete_bodies = 1
    if len(document.encode()) >= LIMIT_BYTES:
        raise ValueError("campaign overview plus latest report must remain below one MiB")
    ids = re.findall(r'\bid="([^"]+)"', document)
    if len(ids) != len(set(ids)):
        raise ValueError("aggregate report contains duplicate identifiers")
    if any(saved.evaluation.fixed.digest(path) != sha for path, sha in protected.items()):
        raise ValueError("saved campaign presentation input changed")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(document)
    summary = {"format": "heston_pr87_followup_campaign_report_v1", "campaign": binding(campaign_path),
        "campaign_protocol": campaign["protocol"], "report": binding(out), "report_bytes": len(document.encode()),
        "limit_bytes": LIMIT_BYTES, "inline_svg_count": document.count('<svg'), "complete_run_bodies_embedded": complete_bodies,
        "saved_run_reports": reports, "endpoints": endpoints, "duplicate_ids": False,
        "producer": binding(__file__), "saved_presentation_helper": binding(saved.__file__),
        "protected_inputs": [{"path": p, "sha256": sha} for p, sha in sorted(protected.items())],
        "protected_inputs_unchanged": True, "scientific_diagnostics_recomputed": False,
        "model_loaded": False, "new_neural_sampling": False, "new_pricing": False, "new_reference_sampling": False,
        "scientific_acceptance": False, "promotion": False}
    saved.evaluation.fixed.write_json(summary_path, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.campaign, args.out)
    print(json.dumps({"report": result["report"], "report_bytes": result["report_bytes"],
        "complete_run_bodies_embedded": result["complete_run_bodies_embedded"]}))

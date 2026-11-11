"""Aggregate completed revised-grid reports using saved presentation data only.

Embed both full per-run main bodies and display the old54 control once in the
campaign overview. Never load models, raw draws or chains, price options, query
posteriors or recompute scientific diagnostics. Oversize output fails rather
than silently dropping a run's diagnostics.
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_pr87_grid as saved

LIMIT_BYTES = 1024 * 1024
RUNS = (("200k-300", 200000), ("300k-300", 300000))
PROTOCOL_FORMATS = ("heston_pr87_grid_campaign_protocol_v1", "heston_pr87_grid_recovery_campaign_protocol_v1")
binding, read = saved.binding, saved.read


class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.references, self.resources, self.svg_count = [], [], [], 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" or any(name.lower().startswith("on") for name in attrs):
            raise ValueError("saved campaign presentation must be script-free")
        if "id" in attrs:
            if not attrs["id"]: raise ValueError("nonempty saved identifiers required")
            self.ids.append(attrs["id"])
        if tag == "svg": self.svg_count += 1
        for key in ("href", "xlink:href"):
            value = attrs.get(key, "")
            if value.startswith("#"): self.references.append(value[1:])
            if value.lower().startswith(("javascript:", "vbscript:")):
                raise ValueError("executable links are not allowed")
        for key in ("aria-labelledby", "aria-describedby"):
            self.references.extend(attrs.get(key, "").split())
        for value in attrs.values():
            self.references.extend(re.findall(r'url\([\"\']?#([^\s)\"\']+)[\"\']?\)', value or ""))
        if tag in ("img", "script", "iframe", "source", "video", "audio", "link", "object", "embed"):
            for key in ("src", "href", "data"):
                value = attrs.get(key)
                if value and not value.startswith(("data:", "#")):
                    self.resources.append(value)


def markup(document):
    parsed = Markup(); parsed.feed(document); parsed.close()
    if len(parsed.ids) != len(set(parsed.ids)):
        raise ValueError("duplicate saved presentation identifiers")
    if parsed.resources or re.search(r'url\(\s*[\"\']?(?:https?:)?//', document, re.I):
        raise ValueError("saved report requires an external rendering resource")
    if any(value not in parsed.ids for value in parsed.references):
        raise ValueError("saved presentation has an unresolved local identifier")
    return parsed


def report_body(document, prefix):
    if document.count("<main>") != 1 or document.count("</main>") != 1:
        raise ValueError("exactly one complete saved report main body required")
    content = re.search(r'<main>([\s\S]*?)</main>', document).group(1)
    parsed = markup(content)
    mapping = {value: prefix + "-" + value for value in parsed.ids}
    content = re.sub(r'(?<![\w:-])id\s*=\s*([\"\'])(.*?)\1',
        lambda match: 'id=' + match[1] + mapping[match[2]] + match[1], content)
    content = re.sub(r'(?<![\w:-])((?:xlink:)?href)\s*=\s*([\"\'])#(.*?)\2',
        lambda match: match[1] + '=' + match[2] + '#' + mapping.get(match[3], match[3]) + match[2], content)
    content = re.sub(r'url\(([\"\']?)#([^\s)\"\']+)\1\)',
        lambda match: 'url(' + match[1] + '#' + mapping.get(match[2], match[2]) + match[1] + ')', content)
    content = re.sub(r'(aria-labelledby|aria-describedby)\s*=\s*([\"\'])(.*?)\2',
        lambda match: match[1] + '=' + match[2] + ' '.join(mapping.get(x, x) for x in match[3].split()) + match[2], content)
    result = '<section class="campaign-run" id="' + prefix + '">' + content + '</section>'
    markup(result)
    return result


def _check_sources(sources, protected):
    for spec in sources.values(): saved.evaluation.checked_binding(spec, protected)


def _panels(record, control, common):
    for panel, rows, draws in (("native_validation", 100, 300), ("population", 512, 512)):
        current = record[panel]
        if (current.get("rows") != rows or current.get("draws_per_case") != draws
                or current.get("parameter_names") != record["task"]["parameter_names"]
                or current.get("truth_sha256") != control[panel]["truth_sha256"]):
            raise ValueError("complete paired native100/historical512 six-target panels required")
        if current.get("observations_sha256") == control[panel]["observations_sha256"]:
            raise ValueError("new63 observations must not be labelled as the old54 realization")
        identities = {key: current[key] for key in ("truth_sha256", "observations_sha256", "sbc_seed")}
        if panel in common and common[panel] != identities:
            raise ValueError("new63 endpoint observations/truths/SBC seeds are not the same shared panels")
        common[panel] = identities


def aborted_attempt(campaign, declaration, protected):
    """Keep the original operational failure visible beside recovery results."""
    spec = campaign.get("recovery_of", declaration.get("recovery_of"))
    if campaign.get("recovery_of") and declaration.get("recovery_of") and campaign["recovery_of"] != declaration["recovery_of"]:
        raise ValueError("recovery result and declaration identify different aborted attempts")
    if spec is None:
        if declaration["format"] == PROTOCOL_FORMATS[1]:
            raise ValueError("recovery presentation requires the preserved aborted campaign binding")
        return "", None
    original = saved.checked(spec, protected)
    if original.get("format") != "heston_pr87_grid_campaign_v1" or original.get("status") != "failed":
        raise ValueError("recovery_of must bind the original failed campaign")
    original_protocol = saved.checked(original["protocol"], protected)
    _check_sources(original_protocol["sources"], protected)
    old_bank = saved.checked(original_protocol["bank_protocol"], protected)
    new_bank = saved.checked(declaration["bank_protocol"], protected)
    if (not isinstance(old_bank.get("contract"), dict) or old_bank.get("contract") != new_bank.get("contract")
            or original_protocol["sources"]["producer"] != declaration["sources"]["producer"]):
        raise ValueError("operational recovery must preserve the exact scientific bank contract and producer")
    progress_path = Path(old_bank["output"]) / "generation-progress.json"
    progress = saved.checked(binding(progress_path), protected) if progress_path.is_file() else None
    jobs = [[row.get("id", ""), row.get("kind", ""), row.get("status", ""),
        str(row.get("wall_seconds_limit", "—")), str(row.get("returncode", "—"))]
        for row in original.get("jobs", [])]
    content = '<h2>Initial aborted attempt is preserved</h2><p class="notice">'
    content += saved.esc(original.get("error_type", "Operational failure")) + ': ' + saved.esc(original.get("error", "See the frozen attempt record."))
    content += ' The failed attempt is not resumed, overwritten or relabelled successful. Recovery uses separately declared outputs and reference resources; failed or capped references stay inconclusive.</p>'
    content += saved.table(["Original job", "Kind", "Retained status", "Outer cap (s)", "Exit code"], jobs)
    if progress is not None:
        content += '<p>Original bank progress: ' + f'{progress["completed_pricing_calls"]:,}' + ' / ' + f'{progress["total_pricing_calls"]:,}' + ' checked rows; retained data.npz present: ' + ('yes' if (Path(old_bank["output"])/"data.npz").exists() else 'no') + '.</p>'
    recovery = declaration.get("operational_recovery", {})
    if recovery:
        content += saved.table(["Declared operational recovery", "Scope"], [[key.replace("_", " "), str(value)] for key, value in recovery.items()])
    content += '<p>Original failed campaign SHA256: <code>' + spec["sha256"] + '</code>. The scientific bank contract and producer match exactly; reference resource changes do not establish convergence or posterior acceptance.</p>'
    return content, {"campaign": spec, "protocol": original["protocol"], "status": "failed",
        "error_type": original.get("error_type"), "error": original.get("error"),
        "bank_progress": binding(progress_path) if progress is not None else None,
        "scientific_bank_contract_unchanged": True, "producer_unchanged": True,
        "operational_recovery": recovery}


def build(campaign_path, out):
    campaign_path, out = Path(campaign_path).resolve(), Path(out).resolve()
    summary_path = out.with_name("campaign-report-summary.json")
    if out.suffix.lower() not in (".html", ".htm") or out == summary_path:
        raise ValueError("distinct HTML report and JSON summary destinations required")
    if out.exists() or summary_path.exists():
        raise FileExistsError("new aggregate presentation identity required")
    protected = {}
    campaign = saved.checked(binding(campaign_path), protected)
    if (campaign.get("format") != "heston_pr87_grid_campaign_v1" or campaign.get("status") != "complete"
            or campaign.get("scientific_acceptance") is not False
            or list(campaign.get("runs", {})) != [name for name, _ in RUNS]):
        raise ValueError("completed two-fit revised-grid campaign required before aggregate rendering")
    declaration = saved.checked(campaign["protocol"], protected)
    if (declaration.get("format") not in PROTOCOL_FORMATS
            or declaration.get("predeclared_before_bank") is not True
            or declaration.get("scientific_acceptance") is not False
            or list(declaration.get("runs", {})) != [name for name, _ in RUNS]
            or declaration["sources"]["renderer"] != binding(saved.__file__)):
        raise ValueError("frozen two-fit revised-grid campaign declaration required")
    _check_sources(declaration["sources"], protected)
    for key in ("bank_protocol", "reference_protocol", "old54_control", "preserved_artifacts"):
        saved.evaluation.checked_binding(declaration[key], protected)
    for spec in declaration.get("readiness", {}).values():
        if isinstance(spec, dict) and {"path", "sha256"} <= set(spec):
            saved.evaluation.checked_binding(spec, protected)
    attempt_html, attempt_record = aborted_attempt(campaign, declaration, protected)
    control = saved.checked(declaration["old54_control"], protected)
    if (control.get("format") != "heston_pr87_followup_evaluation_v1" or control.get("status") != "complete"
            or control.get("endpoint") != "primary_final" or control.get("native_epoch") != 300
            or control.get("training_rows") != 200000 or control.get("task", {}).get("quote_count") != 54):
        raise ValueError("preserved old54/200k300 primary control required")
    rows = [saved.comparison_row("Old54 control — 200k300 primary", control, 54, 200000, 60000000)]
    reports, bodies, endpoints, common, styles = [], [], [], {}, []
    common_bank = None
    for run_id, train_rows in RUNS:
        spec = declaration["runs"][run_id]
        study_spec = spec["protocol"]
        study = saved.checked(study_spec, protected)
        if (spec["train_rows"] != train_rows or study.get("format") != "heston_pr87_grid_study_v1"
                or study["contract"].get("train_rows") != train_rows or study["contract"].get("epochs") != 300
                or study["sources"] != declaration["sources"] or study["old54_control"] != declaration["old54_control"]
                or study["bank_protocol"] != declaration["bank_protocol"]
                or study["reference_protocol"] != declaration["reference_protocol"]
                or study["task"]["target"].get("quote_count") != 63
                or study["task"]["target"].get("pricing_controls", {}).get("fourier_representation") != "black_control_variate_v1"):
            raise ValueError("per-run task/source/common-bank/reference declaration differs")
        for key in ("task_file", "conditional_panel"):
            saved.evaluation.checked_binding(study[key], protected)
        report_path = Path(study_spec["path"]).parent / "report.html"
        report_summary_path = report_path.with_name("report-summary.json")
        report_summary = saved.checked(binding(report_summary_path), protected)
        report_spec = binding(report_path)
        if (report_summary.get("format") != "heston_pr87_grid_report_v1"
                or report_summary.get("protocol") != study_spec or report_summary.get("report") != report_spec
                or report_summary.get("old54_control") != declaration["old54_control"]
                or report_summary.get("producer") != declaration["sources"]["renderer"]
                or report_summary.get("protected_inputs_unchanged") is not True
                or report_summary.get("scientific_acceptance") is not False or report_summary.get("promotion") is not False
                or report_summary.get("new_neural_sampling") is not False or report_summary.get("new_reference_sampling") is not False):
            raise ValueError("complete hash-bound revised-grid per-run report required")
        saved.evaluation.checked_binding(report_spec, protected)
        raw = report_path.read_bytes(); document = raw.decode("utf-8"); parsed = markup(document)
        if (len(raw) != report_summary.get("report_bytes") or len(raw) >= LIMIT_BYTES
                or parsed.svg_count != report_summary.get("inline_svg_count")):
            raise ValueError("saved report size or inline SVG count differs")
        style = re.findall(r'<style>([\s\S]*?)</style>', document)
        if len(style) != 1: raise ValueError("one saved presentation style required")
        styles.append(style[0])
        training = saved.checked(campaign["runs"][run_id]["training"], protected)
        accounting = saved.evaluation.training_accounting(training)
        if accounting["train_rows"] != train_rows:
            raise ValueError("completed per-run training accounting differs")
        distinct = training["best_validation_archive"]["epoch"] != 300
        if distinct != (report_summary.get("secondary_evaluation") is not None):
            raise ValueError("saved secondary required exactly when validation minimum differs from300")
        for role, key in (("primary", "evaluation"), ("secondary", "secondary_evaluation")):
            evaluation_spec = report_summary.get(key)
            if evaluation_spec is None: continue
            record = saved.checked(evaluation_spec, protected)
            if (record.get("format") != "heston_pr87_grid_evaluation_v1" or record.get("status") != "complete"
                    or record.get("protocol") != study_spec or record.get("task") != study["task"]["target"]
                    or record.get("training_rows") != train_rows
                    or record.get("training_presentations") != train_rows * record.get("native_epoch", 0)
                    or record.get("training_run") != campaign["runs"][run_id]["training"]
                    or record.get("old54_control") != declaration["old54_control"]
                    or record.get("reference_protocol") != declaration["reference_protocol"]
                    or record.get("conditional_panel") != study["conditional_panel"]
                    or record.get("protected_inputs_unchanged") is not True
                    or record.get("reserved_arrays_opened") is not False or record.get("new_reference_sampling") is not False
                    or record["task"]["parameter_names"] != control["task"]["parameter_names"]
                    or record["task"]["prior_bounds"] != control["task"]["prior_bounds"]):
                raise ValueError("saved revised-grid endpoint identity differs")
            checkpoint = saved.evaluation.checked_binding(record["checkpoint"], protected).parent
            endpoint = saved.evaluation.checkpoint_endpoint(checkpoint, record, training)
            if endpoint != record.get("endpoint") or (endpoint == "primary_final") != (role == "primary"):
                raise ValueError("only final300 primary and distinct predeclared minimum are allowed")
            saved.evaluation.checked_binding(record["common_bank"], protected)
            if record["common_bank"] != {"path": str(Path(training["bank_path"]).resolve() / "bank.json"),
                    "sha256": training["bank_record_sha256"]}:
                raise ValueError("endpoint common bank differs from its completed training binding")
            if common_bank is not None and common_bank != record["common_bank"]:
                raise ValueError("all revised-grid endpoints must share one exact bank")
            common_bank = record["common_bank"]
            _panels(record, control, common)
            label = run_id + (" primary — final epoch300" if role == "primary" else f' secondary — minimum-validation epoch{record["native_epoch"]}')
            rows.append(saved.comparison_row(label, record, 63, train_rows, record["training_presentations"]))
            checks = report_summary["readiness"][endpoint]
            if any(checks.get(k) is not False for k in ("official_quality_established", "scientific_acceptance", "promotion")):
                raise ValueError("saved development readiness must not grant official acceptance")
            endpoints.append({"run_id": run_id, "role": role, "evaluation": evaluation_spec,
                "readiness": checks, "checkpoint": record["checkpoint"]})
        bodies.append(report_body(document, "run-" + run_id))
        reports.append({"run_id": run_id, "report": report_spec, "summary": binding(report_summary_path),
            "inline_svg_count": parsed.svg_count, "full_main_body_embedded": True})
    if styles[0] != styles[1]: raise ValueError("both per-run reports must use the same saved style")
    navigation = ' · '.join('<a href="#run-' + name + '">' + name + ' full report</a>' for name, _ in RUNS)
    introduction = '<h1>Revised Heston grid: paired 200k/300k campaign</h1><p>' + navigation + '</p>'
    introduction += '<p>Old54/200k300 is the preserved control. Both new63 fits complete300 epochs:200k uses60 million presentations and234,600 updates;300k uses90 million and351,600. The larger-data comparison also changes optimization and regularization exposure. Final epoch300 remains primary; only a distinct first minimum native-validation objective is secondary.</p>'
    introduction += saved.table(["Endpoint", "Quotes", "Training simulations", "Presentations", "c median RMSE", "c90% coverage", "Physical90% coverage range", "SBC screens", "Raw outside prior", "Raw nonphysical"], rows)
    introduction += '<p class="notice"><strong>Paired truths, different observation tasks.</strong> Native100 and historical512 six-target truths are paired to the old54 control, but quote geometry, measured half-spreads,63-feature input shape and observation realizations differ. Only one contract overlaps: its stored standardized residual is recomposed on the newly priced clean value for the original200k training rows and native100, after drawing all63 errors. Historical512 errors are fresh. Both new fits share the exact same300k bank/panels and a63-feature ZCA fitted only to its first10k training rows. Old54 MCMC chains cannot validate the new63 posterior.</p>'
    introduction += '<p>The new63 numerical representation uses an analytic Black control variate with the unchanged Heston characteristic function, broad physical/c prior, Gaussian likelihood, quadrature ladder and measurement-scale gates. This is explicitly versioned numerical arithmetic, not price clipping or a new financial error term. Finite pilot checks are sampled numerical evidence.</p>'
    introduction += '<p class="notice"><strong>No official acceptance or promotion.</strong> All raw draws remain in coverage, SBC and support diagnostics without clipping, filtering or resampling. Conditional agreement uses the new63 target-matched references; unready references remain inconclusive. These are reused development panels, with the reserved pool unopened. Point recovery or a lower stochastic validation objective cannot replace all-six calibration, valid support, conditional-reference checks and fresh confirmation.</p>'
    introduction += attempt_html
    footer = '<footer>Aggregate formatting of saved hash-bound metrics and both complete per-run report bodies. No scientific diagnostics were recomputed, and no models, raw draws or native chains were opened.<br>Campaign SHA256: <code>' + binding(campaign_path)["sha256"] + '</code></footer>'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Revised Heston grid campaign</title><style>' + styles[0] + '.campaign-run{border-top:2px solid #ccd7df;margin-top:40px;padding-top:12px}</style><main>' + introduction + ''.join(bodies) + footer + '</main></html>'
    parsed = markup(document); size = len(document.encode("utf-8"))
    if size >= LIMIT_BYTES: raise ValueError("both complete revised-grid report bodies must fit below one MiB")
    if parsed.svg_count != sum(item["inline_svg_count"] for item in reports):
        raise ValueError("aggregate omitted an inline saved figure")
    if any(saved.evaluation.fixed.digest(path) != sha for path, sha in protected.items()):
        raise ValueError("aggregate saved source/input changed during formatting")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(document, encoding="utf-8")
    result = {"format": "heston_pr87_grid_campaign_report_v1", "campaign": binding(campaign_path),
        "campaign_protocol": campaign["protocol"], "report": binding(out), "report_bytes": size,
        "limit_bytes": LIMIT_BYTES, "inline_svg_count": parsed.svg_count,
        "complete_run_bodies_embedded": 2, "saved_run_reports": reports, "endpoints": endpoints,
        "old54_control": declaration["old54_control"], "common_bank": common_bank,
        "preserved_aborted_attempt": attempt_record,
        "paired_panels": common, "duplicate_ids": False, "external_rendering_resources": 0,
        "producer": binding(__file__), "saved_presentation_helper": binding(saved.__file__),
        "declared_sources": declaration["sources"],
        "protected_inputs": [{"path": p, "sha256": sha} for p, sha in sorted(protected.items())],
        "protected_inputs_unchanged": True, "scientific_diagnostics_recomputed": False,
        "model_loaded": False, "raw_draw_or_chain_arrays_opened": False, "new_neural_sampling": False,
        "new_pricing": False, "new_reference_sampling": False, "reserved_arrays_opened": False,
        "scientific_acceptance": False, "promotion": False}
    saved.evaluation.fixed.write_json(summary_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); result = build(args.campaign, args.out)
    print(json.dumps({"report": result["report"], "report_bytes": result["report_bytes"],
        "complete_run_bodies_embedded": result["complete_run_bodies_embedded"]}))

"""Freeze the audited actual-63 design and run a bounded pricing/panel pilot.

Preparation never prices. The separately committed pilot uses the existing
checked Heston implementation, then writes an observation-only reference panel.
Historical tasks, prices, posterior draws and model payloads remain immutable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration.heston_spread import HestonSpreadPilot
from scripts import run_heston_spread_reference as reference

CANDIDATE_SHA = "fb33bd7803c86f8b491cff965fbf64997d5ffffbb74f2a33bbf23de5eea27710"
SUMMARY_SHA = "33735204280e8450197345238845853863730a0a1571e48820f002fb2b2bc838"
PILOT_SECONDS = 300
PILOT_SEED = 261114
CASES = ("synthetic-broad", "synthetic-market-like", "market")


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": reference.digest(path)}


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_bindings():
    return {"preparer": binding(__file__), **reference.source_hashes()}


def prepare(candidate_path, summary_path, old_panel_path, parent_training, out):
    candidate_path, summary_path, old_panel_path, parent_training, out = map(
        lambda p: Path(p).resolve(), (candidate_path, summary_path, old_panel_path, parent_training, out))
    if out.exists():
        raise FileExistsError("new revised-grid study identity required")
    candidate, summary, old_panel = map(read, (candidate_path, summary_path, old_panel_path))
    parent = read(parent_training / "run.json")
    if (binding(candidate_path)["sha256"] != CANDIDATE_SHA
            or binding(summary_path)["sha256"] != SUMMARY_SHA
            or summary["complete_supported_quote_layout"] is not True
            or summary["selected_contract_count"] != 63
            or summary["supported_quote_target_count"] != 63
            or summary["duplicate_selected_symbol_count"] != 0
            or summary["carry_disjoint_from_candidate_old_fitted_and_scored"] is not True
            or summary["candidate_disjoint_from_preserved_scoring_quotes"] is not True):
        raise ValueError("the exact audited, fully supported actual63 revision is required")
    if (parent.get("format") != "heston_pr87_followup_run_v1" or parent.get("status") != "complete"
            or parent["protocol"]["train_rows"] != 200000 or parent["completed_epochs"] != 300
            or parent["actual_optimizer_updates"] != 234600
            or binding(parent_training / "data.npz")["sha256"] != parent["data_payload_sha256"]):
        raise ValueError("completed immutable old54 200k/300 control required")
    if binding(old_panel["design_path"])["sha256"] != old_panel["design_sha256"]:
        raise ValueError("historical observation-panel design changed")
    old_design = read(old_panel["design_path"])
    records = candidate["records"]
    model = HestonSpreadPilot(records, infer_c=True, c_bounds=(.5, 2.))
    fixed = HestonSpreadPilot(records)
    if len(model.records["mid"]) != 63 or model.bounds.tolist() != parent["bounds"]:
        raise ValueError("new design must preserve the complete six-parameter prior")
    symbols = [q["symbol"] for q in candidate["selected_quotes"]]
    if len(set(symbols)) != 63 or not all(q["target_supported"] for q in candidate["selected_quotes"]):
        raise ValueError("all physical quote locations must be distinct and supported")
    out.mkdir(parents=True)
    design = {**candidate, "format": "heston_pr87_actual63_design_v1",
        "products": old_panel["products"], "candidate": binding(candidate_path),
        "coverage_summary": binding(summary_path), "old_design": binding(old_panel["design_path"]),
        "comparison": "54 versus63 actual observations: maturity, wing, spread and quote-count package; common conditioned carry union, not strike placement alone"}
    design_path = out / "design.json"
    write_new(design_path, design)
    task = {"format": "heston_pr87_grid_task_v1", "target": model.task_definition(),
        "physical_clean_price_task": {"format": "heston_pr87_grid_fixed_task_v1",
            "target": fixed.task_definition(), "records": records, "design_sha256": reference.digest(design_path)},
        "records": records, "design_sha256": reference.digest(design_path), "design": binding(design_path),
        "conditioning": "63 ordered noisy prices / fixed observed half-spreads, then full-rank training-only ZCA; no target/clean-price inputs",
        "scale_generation": "independent Uniform(.5,2) per dataset, fixed Gaussian realization per row across epochs",
        "scope": "New fixed-layout six-target development task; no support transform, masking, learned summary or acceptance"}
    task_path = out / "task.json"
    write_new(task_path, task)
    bounds = np.asarray(model.bounds[:5])
    rng = np.random.default_rng(PILOT_SEED)
    random_points = rng.uniform(bounds[:, 0], bounds[:, 1], (64, 5))
    corners = np.asarray(list(itertools.product(*bounds)))
    selected = {case["id"]: case for case in old_panel["cases"]}
    truths = {key: selected[key]["physical_truth"] for key in CASES[:2]}
    points = [*random_points.tolist(), *corners.tolist(), *truths.values()]
    protocol = {"format": "heston_pr87_grid_preparation_v1",
        "declared_utc": datetime.now(timezone.utc).isoformat(), "predeclared_before_pricing": True,
        "candidate": binding(candidate_path), "coverage_summary": binding(summary_path),
        "old_panel": binding(old_panel_path), "old_design": binding(old_panel["design_path"]),
        "parent_training": str(parent_training), "parent_run": binding(parent_training / "run.json"),
        "parent_data": binding(parent_training / "data.npz"), "task": binding(task_path),
        "design": binding(design_path), "sources": source_bindings(), "output": str(out),
        "pilot": {"max_seconds": PILOT_SECONDS, "seed": PILOT_SEED, "physical_points": points,
            "random_points": 64, "corners": 32, "paired_case_points": 2,
            "precision": "canonical checked ladder plus one explicit doubled accepted node/cutoff rung; per-point log-likelihood shift <=0.1; no independent model-implementation claim"},
        "conditional_truths": truths, "conditional_case_ids": list(CASES),
        "conditional_noise_seed": PILOT_SEED + 1000,
        "products": old_panel["products"], "reserved_arrays_opened": False,
        "scientific_acceptance": False}
    write_new(out / "preparation-protocol.json", protocol)
    return protocol


def pilot(protocol_path):
    started = time.monotonic()
    protocol_path = Path(protocol_path).resolve()
    p = read(protocol_path)
    if (p.get("format") != "heston_pr87_grid_preparation_v1" or p["sources"] != source_bindings()
            or p["pilot"]["max_seconds"] != PILOT_SECONDS):
        raise ValueError("unchanged predeclared pricing preparation required")
    for spec in [p[k] for k in ("candidate", "coverage_summary", "old_panel", "old_design", "parent_run", "parent_data", "task", "design")]:
        if binding(spec["path"]) != spec:
            raise ValueError("prepared source/data/task identity changed")
    out = Path(p["output"]) / "numerical-pilot"
    out.mkdir(exist_ok=False)
    task = read(p["task"]["path"])
    model = HestonSpreadPilot(task["records"], infer_c=True)
    model.validate_task(task["target"])
    old_panel, old_design = read(p["old_panel"]["path"]), read(p["old_design"]["path"])
    design = read(p["design"]["path"])
    old_symbols = [q["symbol"] for q in old_design["selected_quotes"]]
    shared = [(j, old_symbols.index(q["symbol"])) for j, q in enumerate(design["selected_quotes"]) if q["symbol"] in old_symbols]
    for new_index, old_index in shared:
        for column in ("forward", "strike", "discount", "maturity", "option_type", "half_spread"):
            if design["records"][column][new_index] != old_design["records"][column][old_index]:
                raise ValueError("shared contract has different physical geometry/carry/spread")
    record = {"format": "heston_pr87_grid_numerical_pilot_v1", "status": "running",
        "protocol": binding(protocol_path), "points": [], "shared_contract_indices": shared,
        "reserved_arrays_opened": False, "scientific_acceptance": False}
    reference.write_json(out / "run.json", record)
    try:
        with reference.Budget(PILOT_SECONDS) as budget:
            for index, physical in enumerate(p["pilot"]["physical_points"]):
                budget.check()
                theta = np.r_[physical, 1.]
                clean = model.prices(theta)
                y = clean + model.records["half_spread"] * np.random.default_rng(PILOT_SEED + index).standard_normal(63)
                _, check = reference.price_audit(model, theta, y)
                if not check["pointwise_gates_passed"]:
                    raise FloatingPointError("new-grid representative precision check failed")
                record["points"].append({"index": index, "physical_parameters": physical, **check})
            cases, truth_audit = [], []
            for index, case_id in enumerate(CASES):
                budget.check()
                if case_id == "market":
                    y = np.asarray(task["records"]["mid"])
                else:
                    theta = np.r_[p["conditional_truths"][case_id], 1.]
                    clean = model.prices(theta)
                    errors = np.random.default_rng(p["conditional_noise_seed"] + index).standard_normal(63)
                    old_case = next(x for x in old_panel["cases"] if x["id"] == case_id)
                    for new_index, old_index in shared:
                        errors[new_index] = old_case["standard_errors"][old_index]
                    y = clean + model.records["half_spread"] * errors
                    truth_audit.append({"id": case_id, "physical_truth": theta[:5].tolist(),
                        "generating_c": 1., "noise_seed": p["conditional_noise_seed"] + index,
                        "standard_errors": errors.tolist(), "clean_prices": clean.tolist(),
                        "shared_errors_from_old_panel": shared})
                cases.append({"id": case_id, "observations": y.tolist(), "observations_sha256": reference.values_digest(y)})
            panel = {"format": "heston_pr87_grid_reference_panel_v1", "design": p["design"],
                "task": task["target"], "products": p["products"], "cases": cases,
                "source": binding(__file__), "preparation_protocol": binding(protocol_path),
                "scope": "Only observations, geometry, prior and product horizons enter reference initialization/sampling; no generating truths or neural starts."}
            write_new(Path(p["output"]) / "conditional-panel.json", panel)
            write_new(Path(p["output"]) / "conditional-truth-audit.json", {
                "format": "heston_pr87_grid_conditional_truth_audit_v1", "old_panel": p["old_panel"],
                "cases": truth_audit, "panel": binding(Path(p["output"]) / "conditional-panel.json"),
                "scope": "Report/audit only; never read by reference initialization or sampling."})
            record.update(status="complete", seconds=time.monotonic() - started,
                panel=binding(Path(p["output"]) / "conditional-panel.json"),
                maximum_price_shift_over_minimum_measurement_sd=max(x["price_shift_over_minimum_measurement_sd"] for x in record["points"]),
                maximum_absolute_log_likelihood_shift=max(abs(x["log_likelihood_shift"]) for x in record["points"]))
        reference.write_json(out / "run.json", record)
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic() - started)
        reference.write_json(out / "run.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("candidate", "summary", "old-panel", "parent-training", "out"):
        prep.add_argument("--" + key, type=Path, required=True)
    run = sub.add_parser("pilot")
    run.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.candidate, args.summary, args.old_panel, args.parent_training, args.out) if args.command == "prepare" else pilot(args.protocol)
    print(json.dumps({"format": result["format"], "status": result.get("status", "prepared")}))

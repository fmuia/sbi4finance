"""Read saved revised-grid numerical evidence; never price, fit or sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def checked(spec):
    assert binding(spec["path"]) == {key: spec[key] for key in ("path", "sha256")}
    return Path(spec["path"])


def audit(evidence, out):
    evidence, out = Path(evidence).resolve(), Path(out).resolve()
    failed_path = evidence / "preparation/numerical-pilot/run.json"
    passed_path = evidence / "cv-preparation/numerical-pilot/run.json"
    failed, passed = read(failed_path), read(passed_path)
    old_protocol = read(checked(failed["protocol"]))
    new_protocol = read(checked(passed["protocol"]))
    assert failed["status"] == "failed" and passed["status"] == "complete"
    assert len(failed["points"]) == 74 and len(passed["points"]) == 98
    assert old_protocol["pilot"]["physical_points"] == new_protocol["pilot"]["physical_points"]
    points = new_protocol["pilot"]["physical_points"]
    assert [row["index"] for row in passed["points"]] == list(range(98))
    assert all(row["pointwise_gates_passed"] is True and row["physical_parameters"] == points[row["index"]]
               for row in passed["points"])
    for spec in new_protocol["sources"].values(): checked(spec)
    old_task, task = read(checked(old_protocol["task"])), read(checked(new_protocol["task"]))
    old_target, target = dict(old_task["target"]), dict(task["target"])
    pricing = dict(target["pricing_controls"])
    assert pricing.pop("fourier_representation") == "black_control_variate_v1"
    target["pricing_controls"] = pricing
    assert target == old_target
    assert checked(old_protocol["design"]).read_bytes() == checked(new_protocol["design"]).read_bytes()
    panel = read(checked(passed["panel"]))
    assert panel["task"] == task["target"] and panel["design"] == new_protocol["design"]
    max_price = max(row["price_shift_over_minimum_measurement_sd"] for row in passed["points"])
    max_logl = max(abs(row["log_likelihood_shift"]) for row in passed["points"])
    assert max_price == passed["maximum_price_shift_over_minimum_measurement_sd"]
    assert max_logl == passed["maximum_absolute_log_likelihood_shift"]
    assert failed["reserved_arrays_opened"] is False and passed["reserved_arrays_opened"] is False
    out.mkdir(parents=True, exist_ok=False)
    archived = out / "consumer.py"
    shutil.copyfile(__file__, archived)
    assert binding(__file__)["sha256"] == binding(archived)["sha256"]
    repo = Path(__file__).resolve().parents[2]
    result = {"format": "heston_pr87_grid_numerical_audit_v1", "status": "complete",
        "consumer": binding(__file__), "archived_consumer": binding(archived),
        "failed_original_pilot": binding(failed_path), "successful_control_variate_pilot": binding(passed_path),
        "unchanged_physical_points": len(points), "completed_original_points": len(failed["points"]),
        "original_failed_point_index": 74, "original_failed_physical_parameters": points[74],
        "unchanged_financial_target_design_prior_and_noise": True,
        "representation_only_change": "analytic Black plus integral(phi_Black - phi_Heston); same Heston CF, no clipping",
        "successful_points": len(passed["points"]),
        "base_nodes_histogram": dict(sorted(Counter(row["base_nodes"] for row in passed["points"]).items())),
        "maximum_price_shift_over_minimum_measurement_sd": max_price,
        "maximum_absolute_log_likelihood_shift": max_logl,
        "sources": {**new_protocol["sources"], "black_formula": binding(repo / "tutorial/src/sbi4f_tutorial/shared/gbm.py")},
        "earlier_independent_exploration": {
            "provenance": "Read-only calculations reported during diagnosis before formal CV module; recorded here without new pricing. These scalar checks are not recomputed by this saved-evidence consumer.",
            "original_point74_8192_U4800_all63_bounds_valid": True,
            "original_point74_16384_U9600_far_calls_53_62_negative_about": -1.05e-5,
            "explanation": "whole-surface selection needs adjacent valid rungs; original logging omitted the valid singleton8192 rung. Analytic Black quadrature bias explains small cancellation errors.",
            "independent_cv_98point_pass_count": 98, "independent_cv_98point_failure_count": 0,
            "adaptive_infinite_integral_point74": {
                "method": "SciPy quad controlled-CF integral to infinity, currency epsabs1e-9, epsrel1e-11, limit4000; no warnings",
                "contract_indices": [8, 53, 62],
                "prices": [2.910789165609662e-6, 3.0296777753235915e-8, 1.385302099699004e-7],
                "estimated_absolute_errors": [9.12470988e-10, 9.09372424e-10, 9.66508188e-10]}},
        "new_pricing_calls": 0, "new_model_queries": 0, "reserved_arrays_opened": False,
        "scope": "Numerical representation/readiness evidence; pointwise refinement does not establish global posterior stability or scientific acceptance.",
        "scientific_acceptance": False}
    (out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.evidence, args.out)
    print(json.dumps({"status": result["status"], "successful_points": result["successful_points"],
        "maximum_price_shift_over_minimum_measurement_sd": result["maximum_price_shift_over_minimum_measurement_sd"]}))

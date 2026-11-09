"""Evaluate one frozen Heston screen candidate on development data only.

Reuses the saved-control evaluator's observations, seeds, native sampler and
uncertainty rules. A completed comparison can fail its checks: this is a result,
not an execution failure. The final panel is never read and no model is published.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import check_heston_budget_readiness as common  # noqa: E402


def candidate_identity(protocol, candidate_id, training):
    """Bind the endpoint and its actual optimizer work to one declared recipe."""
    rows = [row for row in protocol["screen_candidates"] if row["id"] == candidate_id]
    if len(rows) != 1:
        raise ValueError("candidate ID must identify one frozen seed-0 recipe")
    candidate, fixed = rows[0], protocol["fixed_training"]
    network = {key: fixed[key] for key in ("family", "encoder", "summary_dim", "flow_steps", "mean_steps")}
    network["widths"] = candidate["widths"]
    contract = training.get("run_contract", {})
    settings = contract.get("settings", {})
    work = training.get("performance", {}).get("work", {})
    if (training.get("format") != "full_surface_training_v1"
            or training.get("network") != network or contract.get("network") != network
            or contract.get("format") != "full_surface_update_budget_run_v1"
            or training.get("checkpoint_selection") != "endpoint"
            or settings.get("checkpoint_selection") != "endpoint"
            or training.get("same_seed_reload_exact") is not True
            or training.get("reserved_test_evaluated") is not False
            or training.get("task_fingerprint") != protocol["bank"]["task_fingerprint"]
            or contract.get("dataset", {}).get("manifest_sha256") != protocol["bank"]["manifest_sha256"]
            or type(settings.get("seed")) is not int
            or settings.get("seed") != candidate["seed"]
            or settings.get("fresh_quote_noise") is not True
            or settings.get("steps_per_execution") != 1
            or settings.get("batch_size") != fixed["batch_size"]
            or settings.get("updates") != candidate["updates"]
            or training.get("selected_update") != candidate["updates"]
            or work.get("optimizer_updates") != candidate["updates"]):
        raise ValueError("checkpoint is not the exact endpoint of the frozen candidate")
    return candidate


def validate_control(control, protocol, protocol_sha256):
    panel = protocol["data_roles"]["development"]
    expected_ids = [f"validation:{index}" for index in range(panel["start"], panel["stop"])]
    if (control.get("format") != "heston_training_budget_readiness_v1"
            or control.get("protocol_sha256") != protocol_sha256
            or control.get("control_checkpoint_sha256") != protocol["control"]["checkpoint_sha256"]
            or control.get("bank_manifest_sha256") != protocol["bank"]["manifest_sha256"]
            or control.get("reserved_test_evaluated") is not False
            or control.get("status") not in ("pass", "inconclusive")
            or control.get("development", {}).get("example_ids") != expected_ids):
        raise ValueError("a matching completed saved-control readiness report is required")


def compare_control(report, control, protocol):
    """Development coverage differences; no final-panel or replacement verdict."""
    import numpy as np

    validate_control(control, protocol, report["protocol_sha256"])
    if control["development"]["example_ids"] != report["development"]["example_ids"]:
        raise ValueError("candidate and control development IDs/order differ")
    margin = protocol["development_gates"]["coverage_difference_from_control_max"]
    names = report["parameter_names"]
    differences, passes = {}, []
    for level in protocol["development_gates"]["coverage_levels"]:
        key = f"{level:g}"
        actual = np.asarray(report["development"]["coverage"][key]["coverage"], dtype=float)
        previous = np.asarray(control["development"]["coverage"][key]["coverage"], dtype=float)
        if (actual.shape != (len(names),) or previous.shape != actual.shape
                or not np.isfinite(actual).all() or not np.isfinite(previous).all()):
            raise ValueError("finite per-parameter coverage vectors are required")
        difference = actual - previous
        within = np.abs(difference) <= margin
        differences[key] = {"candidate_coverage": actual.tolist(), "control_coverage": previous.tolist(),
                            "difference": difference.tolist(), "within_margin": within.tolist()}
        passes.extend(within.tolist())
    candidate_performance = report.get("performance", {})
    control_performance = control.get("performance", {})
    candidate_hardware = candidate_performance.get("hardware", {})
    control_hardware = control_performance.get("hardware", {})
    same_device = (candidate_hardware.get("devices") == control_hardware.get("devices")
                   and candidate_hardware.get("system") == control_hardware.get("system")
                   and candidate_hardware.get("machine") == control_hardware.get("machine"))
    return {"coverage_margin": margin, "coverage": differences, "coverage_within_margin": all(passes),
            "control_status_retained": control["status"],
            "same_reported_device_platform": same_device,
            "candidate_hardware": candidate_hardware, "control_hardware": control_hardware,
            "candidate_runtime": candidate_performance.get("runtime"),
            "control_runtime": control_performance.get("runtime"),
            "scope": "Same fixed development observations/seeds. Cross-device sampling may differ numerically; this is not a timing speedup comparison or a final accuracy verdict."}


def run(args):
    protocol = json.loads(args.protocol.read_text())
    training = json.loads((args.checkpoint / "training.json").read_text())
    candidate = candidate_identity(protocol, args.candidate, training)
    control = json.loads(args.control_readiness.read_text())
    validate_control(control, protocol, common.digest(args.protocol))
    report = common.run(args, role="candidate")
    report["candidate"] = candidate
    report["training_metadata_sha256"] = common.digest(args.checkpoint / "training.json")
    report["control_readiness_sha256"] = common.digest(args.control_readiness)
    report["control_comparison"] = compare_control(report, control, protocol)
    report["gates"]["coverage_within_control_margin"] = report["control_comparison"]["coverage_within_margin"]
    if not report["gates"]["coverage_within_control_margin"]:
        report["status"] = "fail"
    report["execution_status"] = "completed"
    report["eligible_for_final_confirmation"] = False
    report["model_publication_allowed"] = False
    report["scientific_acceptance"] = False
    report["study_scope"] = "Seed-0 development screen with the original control uncertainty retained; no model-selection, confirmation, final-test or publication action."
    report["source_sha256"]["scripts/evaluate_heston_budget_candidate.py"] = common.digest(__file__)
    common.write(args.out / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/heston_training_budget_study.json")
    for key in ("bank", "checkpoint", "control-readiness", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("positive thread count required")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ.setdefault("KERAS_BACKEND", "jax")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        report = run(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        common.write(args.out / "failure.json", {"format": "heston_budget_candidate_failure_v1",
            "candidate": args.candidate, "error_type": type(error).__name__, "error": str(error),
            "reserved_test_evaluated": False, "model_publication_allowed": False})
        raise
    print(json.dumps({"candidate": args.candidate, "execution_status": "completed", "checks_status": report["status"],
                      "reserved_test_evaluated": False, "model_publication_allowed": False}))


if __name__ == "__main__":
    main()

"""Freeze and supervise the authorized 100k/300 then conditional 200k/300 fits.

Both scientific protocols are frozen before the first fit. Only the completed
first primary endpoint determines the second-fit branch; secondary checkpoints
remain separately reported development evidence.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_capacity as shared

FORMAT = "heston_pr87_followup_study_v1"
CAMPAIGN_FORMAT = "heston_pr87_followup_campaign_protocol_v1"
LIMITS = {"maximum_fits": 2, "generation_and_training_seconds_per_fit": 3600,
    "evaluation_seconds_per_checkpoint": 900, "maximum_posterior_checkpoints_per_fit": 2,
    "maximum_concurrent_workers": 1, "maximum_new_pricing_calls": 100000,
    "automatic_retry": False, "unplanned_extension": False}
ENDPOINT_POLICY = {"primary": "final_epoch300",
    "secondary": "native minimum validation objective only if its epoch differs from300",
    "selection_used_posterior_diagnostics": False}
BRANCH_POLICY = {"first_run": "100k-300", "conditional_run": "200k-300",
    "metric": "first primary necessary_diagnostics_pass",
    "rule": "run200k if and only if completed100k primary necessary diagnostics do not all pass",
    "secondary_influences_branch": False, "official_acceptance_from_development_checks": False}
RUNS = (("100k-300", 100000), ("200k-300", 200000))


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": shared.digest(path)}


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def preserved_inventory(parent100k, parent_study):
    """Snapshot earlier469 plus every completed100k evidence/payload file."""
    prior_path = parent_study.parent / "preserved-artifacts-before.json"
    prior = shared.read(prior_path)
    entries = {}
    for row in prior["files"]:
        path = Path(row["path"]).resolve()
        if shared.digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("previously preserved artifact changed: " + str(path))
        entries[str(path)] = row
    for directory in (parent_study.parent, parent100k.parent):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                entries[str(path.resolve())] = {**identity(path), "bytes": path.stat().st_size}
    rows = [entries[key] for key in sorted(entries)]
    return {"format": "heston_pr87_followup_preserved_before_v1",
        "scope": "previous469 artifacts plus complete100k200 evidence, data, models and raw draws; immutable throughout followup",
        "prior_manifest": identity(prior_path), "files": rows,
        "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows)}


def source_bindings():
    from scripts import reproduce_heston_pr87_followup as trainer
    from scripts import evaluate_heston_pr87_followup as evaluator
    from scripts import render_heston_pr87_followup as renderer
    from scripts import evaluate_heston_pr87_100k as prior_evaluator
    from scripts import render_heston_pr87_100k as prior_renderer
    from scripts import evaluate_heston_pr87_reproduction as original_evaluator
    from scripts import render_heston_pr87_reproduction as original_renderer
    from scripts import export_heston_pr87_followup_figures as exporter
    from scripts import export_heston_pr87_reproduction_figures as plotting
    return {"producer": identity(trainer.__file__), "controller": identity(__file__),
        "evaluator": identity(evaluator.__file__), "renderer": identity(renderer.__file__),
        "parent_evaluator": identity(prior_evaluator.__file__), "parent_renderer": identity(prior_renderer.__file__),
        "parent_original_evaluator": identity(original_evaluator.__file__),
        "parent_original_renderer": identity(original_renderer.__file__),
        "exporter": identity(exporter.__file__), "archived_plotting_helper": identity(plotting.__file__)}


def prepare(parent100k, baseline100k_evaluation, evidence, payload):
    """Freeze both run contracts and their preservation inventory; never fit."""
    from scripts import reproduce_heston_pr87_followup as trainer
    from scripts import evaluate_heston_pr87_followup as evaluator
    parent100k, baseline100k_evaluation, evidence, payload = map(
        lambda p: Path(p).resolve(), (parent100k, baseline100k_evaluation, evidence, payload))
    if evidence.exists() or payload.exists():
        raise FileExistsError("followup evidence and payload identities must be new")
    parent_record = shared.read(parent100k / "training.json")
    parent_study = Path(parent_record["protocol_path"]).resolve()
    previous = shared.read(parent_study)
    baseline = shared.read(baseline100k_evaluation)
    if (parent_record.get("format") != "heston_pr87_100k_run_v1" or parent_record.get("status") != "complete"
            or parent_record.get("completed_epochs") != 200 or parent_record.get("actual_optimizer_updates") != 78200
            or previous.get("format") != "heston_pr87_100k_study_v1"
            or baseline.get("format") != "heston_pr87_100k_evaluation_v1" or baseline.get("status") != "complete"
            or baseline.get("native_epoch") != 200 or baseline.get("endpoint") != "primary_final"
            or baseline.get("task") != previous["task"]["target"]
            or baseline.get("training_run") != identity(parent100k / "run.json")
            or baseline100k_evaluation != (Path(previous["evaluations"]["primary"]["output"]) / "evaluation.json").resolve()):
        raise ValueError("completed fixed100k200 parent and its primary evaluation required")
    for new_root in (evidence, payload):
        for immutable_root in (parent_study.parent, parent100k.parent):
            if new_root == immutable_root or immutable_root in new_root.parents:
                raise ValueError("followup outputs must be outside preserved100k roots")
    # Collect hashes before creating any new study identity.
    inventory = preserved_inventory(parent100k, parent_study)
    sources = source_bindings()
    contracts = {name: trainer.protocol_contract(parent100k, train_rows=rows) for name, rows in RUNS}
    task = previous["task"]
    evaluation = evaluator.evaluation_contract(task["target"])
    evidence.mkdir(parents=True); payload.mkdir(parents=True)
    preservation_path = evidence / "preserved-artifacts-before.json"
    write_new(preservation_path, inventory)
    campaign = {"format": CAMPAIGN_FORMAT, "declared_utc": datetime.now(timezone.utc).isoformat(),
        "predeclared_before_first_fit": True, "limits": LIMITS, "branch_policy": BRANCH_POLICY,
        "sources": sources, "preserved_artifacts": identity(preservation_path),
        "parent100k": str(parent100k), "baseline100k_evaluation": identity(baseline100k_evaluation),
        "runs": {}, "branch_proof_path": str(evidence / "branch-proof.json"),
        "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False}
    for name, rows in RUNS:
        run_evidence, run_payload = evidence / name, payload / name
        run_evidence.mkdir(); run_payload.mkdir()
        task_path = run_evidence / "task.json"
        write_new(task_path, task)
        protocol = {"format": FORMAT, "declared_utc": campaign["declared_utc"],
            "predeclared_before_fit": True, "predeclared_before_first_campaign_fit": True,
            "contract": contracts[name], "campaign": LIMITS, "branch_policy": BRANCH_POLICY,
            "run_id": name, "task": task, "task_path": str(task_path), "task_file": identity(task_path),
            "parent_training": str(parent100k), "parent_study": identity(parent_study),
            "parent_10k_evaluation": previous["parent_10k_evaluation"],
            "prior_100k_evaluation": identity(baseline100k_evaluation),
            "original_whitening_study": previous["original_whitening_study"],
            "historical_population_inputs": previous["historical_population_inputs"],
            "output": str(run_payload / "training"), "training_log": str(run_payload / "training.log"),
            "evaluation": evaluation,
            "evaluations": {"primary": {"checkpoint": "epoch-000300", "output": str(run_evidence / "evaluation"),
                    "payload_output": str(run_payload / "evaluation"), "log": str(run_payload / "evaluation.log")},
                "secondary": {"checkpoint": "best-validation", "output": str(run_evidence / "evaluation-best-validation"),
                    "payload_output": str(run_payload / "evaluation-best-validation"),
                    "log": str(run_payload / "evaluation-best-validation.log")}},
            "endpoint_policy": ENDPOINT_POLICY, "sources": sources,
            "preserved_artifacts": identity(preservation_path),
            "comparison": "Fresh same-seed native300-epoch fit. Keep immutable100k noisy prefix, native100 validation rows and parent10k-only ZCA. The200k run appends100k new fixed noisy simulations from100k parent's after_append RNG state. No parent weights or optimizer are resumed; changed epochs/rows also change optimization budget.",
            "official_readiness": {"automatic_acceptance": False, "reserved_test_pool_opened": False,
                "scope": "Reused development diagnostics decide only the authorized second-fit branch; official confirmation remains separate."},
            "new_reference_sampling": False, "scientific_acceptance": False, "fresh_observation_errors_per_pass": False}
        protocol_path = run_evidence / "protocol.json"
        write_new(protocol_path, protocol)
        campaign["runs"][name] = {"train_rows": rows, "protocol": identity(protocol_path)}
    campaign_path = evidence / "campaign-protocol.json"
    write_new(campaign_path, campaign)
    return campaign_path


def study_output_paths(study):
    return [study["output"], study["training_log"], *[entry[key]
        for entry in study["evaluations"].values() for key in ("output", "payload_output", "log")]]


def validate_study(study, rows):
    contract = study.get("contract", {})
    if (study.get("format") != FORMAT or study.get("campaign") != LIMITS
            or study.get("branch_policy") != BRANCH_POLICY or study.get("endpoint_policy") != ENDPOINT_POLICY
            or study.get("predeclared_before_fit") is not True
            or study.get("predeclared_before_first_campaign_fit") is not True
            or study.get("scientific_acceptance") is not False
            or contract.get("format") != "heston_pr87_followup_protocol_v1"
            or contract.get("train_rows") != rows or contract.get("epochs") != 300
            or contract.get("validation_rows") != 100 or contract.get("batch_size") != 256):
        raise ValueError("source-bound predeclared300-epoch study required")
    protected = {**study["sources"], **{"contract_" + k: v for k, v in contract.get("source", {}).items()},
        "task": study["task_file"], "parent": study["parent_study"],
        "parent10k": study["parent_10k_evaluation"], "baseline100k": study["prior_100k_evaluation"],
        "original_study": study["original_whitening_study"], "historical_inputs": study["historical_population_inputs"],
        "preservation": study["preserved_artifacts"]}
    for name, spec in protected.items():
        if shared.digest(spec["path"]) != spec["sha256"]:
            raise ValueError("source or input changed: " + name)
    if study["sources"]["controller"] != identity(__file__) or shared.read(study["task_path"]) != study["task"]:
        raise ValueError("controller or financial task changed")


def validate(campaign):
    if (campaign.get("format") != CAMPAIGN_FORMAT or campaign.get("limits") != LIMITS
            or campaign.get("branch_policy") != BRANCH_POLICY
            or campaign.get("predeclared_before_first_fit") is not True
            or campaign.get("scientific_acceptance") is not False
            or list(campaign.get("runs", {})) != [name for name, _ in RUNS]):
        raise ValueError("bounded predeclared two-fit followup campaign required")
    if campaign["sources"]["controller"] != identity(__file__):
        raise ValueError("controller changed")
    outputs = []
    for name, rows in RUNS:
        spec = campaign["runs"][name]
        if spec["train_rows"] != rows or shared.digest(spec["protocol"]["path"]) != spec["protocol"]["sha256"]:
            raise ValueError("frozen scientific run protocol changed")
        study = shared.read(spec["protocol"]["path"])
        validate_study(study, rows)
        if study["sources"] != campaign["sources"] or study["parent_training"] != campaign["parent100k"]:
            raise ValueError("runs must retain same frozen sources and fixed100k parent")
        outputs += study_output_paths(study)
    resolved = [str(Path(p).resolve()) for p in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("all run/evaluation output identities must be distinct")
    preservation = campaign["preserved_artifacts"]
    if shared.digest(preservation["path"]) != preservation["sha256"]:
        raise ValueError("preservation catalogue changed")
    for spec in shared.read(preservation["path"])["files"]:
        if shared.digest(spec["path"]) != spec["sha256"] or Path(spec["path"]).stat().st_size != spec["bytes"]:
            raise ValueError("preserved artifact changed: " + spec["path"])
    return outputs


def branch_decision(primary_path):
    from scripts import render_heston_pr87_followup as renderer
    primary = shared.read(primary_path)
    if (primary.get("format") != "heston_pr87_followup_evaluation_v1" or primary.get("status") != "complete"
            or primary.get("endpoint") != "primary_final" or primary.get("native_epoch") != 300
            or primary.get("training_rows") != 100000):
        raise ValueError("branch requires completed100k300 primary evaluation")
    readiness = renderer.readiness(primary)
    passed = readiness.get("necessary_diagnostics_pass")
    if type(passed) is not bool:
        raise ValueError("readiness must explicitly report boolean necessary diagnostics")
    return {"format": "heston_pr87_followup_branch_proof_v1", "decision_utc": datetime.now(timezone.utc).isoformat(),
        "primary_evaluation": identity(primary_path), "readiness_source": identity(renderer.__file__),
        "readiness": readiness, "necessary_diagnostics_pass": passed, "run200k": not passed,
        "branch_policy": BRANCH_POLICY, "secondary_influenced_decision": False,
        "scientific_acceptance": False, "official_confirmation_still_required": True}


def run(protocol_path, out):
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    campaign = shared.read(protocol_path)
    outputs = validate(campaign)
    branch_path = Path(campaign["branch_proof_path"])
    if out.exists() or out.with_suffix(".claim.json").exists() or branch_path.exists() or any(Path(p).exists() for p in outputs):
        raise FileExistsError("followup output identities must be new; no retries")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_new(out.with_suffix(".claim.json"), identity(protocol_path))
    result = {"format": "heston_pr87_followup_campaign_v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "protocol": identity(protocol_path),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "jobs": [], "runs": {}, "automatic_retry": False, "scientific_acceptance": False}
    env = dict(os.environ)
    env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
    process = None
    started = time.monotonic()
    shared.write(out, result)

    def execute(name, command, log_path, cap, expected):
        nonlocal process
        validate(campaign)
        entry = {"id": name, "command": command, "log": log_path, "wall_seconds_limit": cap, "status": "running"}
        result["jobs"].append(entry)
        tick = time.monotonic()
        shared.write(out, result)
        with Path(log_path).open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            entry["pid"] = process.pid
            shared.write(out, result)
            try:
                process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                shared.stop(process)
                entry["status"] = "wall_cap"
                raise TimeoutError("followup job exceeded its wall cap: " + name)
        entry.update(returncode=process.returncode, seconds=time.monotonic() - tick)
        if process.returncode or not expected.is_file() or shared.read(expected).get("status") != "complete":
            entry["status"] = "failed"
            raise RuntimeError("followup campaign job failed: " + name)
        entry.update(status="complete", record=identity(expected))
        shared.write(out, result)

    try:
        for name, _ in RUNS:
            study_path = Path(campaign["runs"][name]["protocol"]["path"])
            study = shared.read(study_path)
            command = [sys.executable, study["sources"]["producer"]["path"], "--parent", study["parent_training"],
                "--protocol", str(study_path), "--out", study["output"], "--worker"]
            execute(name + "_training", command, study["training_log"], 3600, Path(study["output"]) / "training.json")
            training = shared.read(Path(study["output"]) / "training.json")
            best_epoch = int(training["best_validation_archive"]["epoch"])
            result["runs"][name] = {"training": identity(Path(study["output"]) / "training.json"),
                "best_validation_epoch": best_epoch}
            for role, spec in study["evaluations"].items():
                if role == "secondary" and best_epoch == 300:
                    result["runs"][name]["secondary_skipped"] = "native best is the same epoch300 primary"
                    shared.write(out, result)
                    continue
                checkpoint = Path(study["output"]) / "checkpoints" / spec["checkpoint"]
                command = [sys.executable, study["sources"]["evaluator"]["path"], "--checkpoint", str(checkpoint),
                    "--protocol", str(study_path), "--out", spec["output"], "--payload-out", spec["payload_output"], "--worker"]
                execute(name + "_" + role + "_evaluation", command, spec["log"], 900, Path(spec["output"]) / "evaluation.json")
            if name == "100k-300":
                validate(campaign)
                decision = branch_decision(Path(study["evaluations"]["primary"]["output"]) / "evaluation.json")
                write_new(branch_path, decision)
                result["branch_proof"] = identity(branch_path)
                result["run200k"] = decision["run200k"]
                shared.write(out, result)
                if not decision["run200k"]:
                    result["stopped_after_first_fit"] = "first primary necessary diagnostics passed; official confirmation remains separate"
                    break
        validate(campaign)
        if shared.digest(protocol_path) != result["protocol"]["sha256"]:
            raise ValueError("campaign declaration changed during execution")
        result.update(status="complete", seconds=time.monotonic() - started)
        shared.write(out, result)
        return result
    except BaseException as error:
        if process is not None:
            shared.stop(process)
        result.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic() - started)
        shared.write(out, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("parent100k", "baseline100k-evaluation", "evidence", "payload"):
        prep.add_argument("--" + key, type=Path, required=True)
    launch = sub.add_parser("run")
    for key in ("protocol", "out"):
        launch.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.parent100k, args.baseline100k_evaluation, args.evidence, args.payload))
    else:
        print(json.dumps(run(args.protocol, args.out)), flush=True)

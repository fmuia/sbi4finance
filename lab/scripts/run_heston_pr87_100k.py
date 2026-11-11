"""Freeze and supervise one nested 100k-data native FlowMatching campaign."""
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

FORMAT = "heston_pr87_100k_study_v1"
LIMITS = {"generation_and_training_seconds": 3600, "evaluation_seconds_per_checkpoint": 900,
    "maximum_posterior_checkpoints": 2, "maximum_concurrent_workers": 1,
    "automatic_retry": False, "automatic_extension": False}


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": shared.digest(path)}


def prepare(parent_study, evidence, payload):
    from scripts import reproduce_heston_pr87_100k as trainer
    from scripts import evaluate_heston_pr87_100k as evaluator
    from scripts import render_heston_pr87_100k as renderer
    from scripts import evaluate_heston_pr87_reproduction as parent_evaluator
    from scripts import render_heston_pr87_reproduction as parent_renderer
    parent_study, evidence, payload = map(lambda p: Path(p).resolve(), (parent_study, evidence, payload))
    if evidence.exists() or payload.exists():
        raise FileExistsError("100k study identities must be new")
    previous = shared.read(parent_study)
    if previous.get("format") != "heston_pr87_reproduction_study_v1":
        raise ValueError("completed 10k native recipe parent required")
    parent = Path(previous["output"])
    if shared.read(parent/"training.json").get("status") != "complete":
        raise ValueError("10k parent must be complete")
    task = previous["task"]
    contract = trainer.protocol_contract(parent)
    protocol = {"format": FORMAT, "declared_utc": datetime.now(timezone.utc).isoformat(),
        "predeclared_before_fit": True, "contract": contract, "campaign": LIMITS,
        "task": task, "task_path": str(evidence/"task.json"),
        "parent_study": identity(parent_study), "parent_training": str(parent.resolve()),
        "parent_10k_evaluation": identity(Path(previous["evaluation_output"])/"evaluation.json"),
        "original_whitening_study": previous["original_whitening_study"],
        "historical_population_inputs": previous["historical_population_inputs"],
        "output": str(payload/"training"), "training_log": str(payload/"training.log"),
        "evaluation": evaluator.evaluation_contract(task["target"]),
        "evaluations": {"primary": {"checkpoint": "epoch-000200", "output": str(evidence/"evaluation"),
                "payload_output": str(payload/"evaluation"), "log": str(payload/"evaluation.log")},
            "secondary": {"checkpoint": "best-validation", "output": str(evidence/"evaluation-best-validation"),
                "payload_output": str(payload/"evaluation-best-validation"), "log": str(payload/"evaluation-best-validation.log")}},
        "endpoint_policy": {"primary": "final_epoch200", "secondary": "native minimum validation objective only if its epoch differs from200",
            "selection_used_posterior_diagnostics": False},
        "sources": {"producer": identity(trainer.__file__), "controller": identity(__file__),
            "evaluator": identity(evaluator.__file__), "renderer": identity(renderer.__file__),
            "parent_evaluator": identity(parent_evaluator.__file__), "parent_renderer": identity(parent_renderer.__file__)},
        "comparison": "Nested old10k fixed noisy prefix plus90k fresh fixed noisy datasets; same old100 validation observations and training-only10k whitener. Same200 epochs changes both diversity and total optimization:20m presentations and78200 updates versus2m and8000. This is not an equal-compute/data-only comparison.",
        "official_readiness": {"automatic_acceptance": False,
            "required_evidence": ["all six marginal coverage and SBC diagnostics", "raw prior support and nonphysical sample rates",
                "conditional six-target means, widths and dependence against ready MCMC", "numerical sensitivity and fresh confirmation after recipe freeze"],
            "reserved_test_pool_opened": False,
            "scope": "Development comparison. Good recovery plots alone cannot authorize an official posterior claim or asset promotion."},
        "new_reference_sampling": False, "scientific_acceptance": False, "fresh_observation_errors_per_pass": False}
    evidence.mkdir(parents=True); payload.mkdir(parents=True)
    shared.write(evidence/"task.json", task)
    protocol["task_file"] = identity(evidence/"task.json")
    shared.write(evidence/"protocol.json", protocol)
    return evidence/"protocol.json"


def output_paths(protocol):
    return [protocol["output"], protocol["training_log"], *[entry[key]
        for entry in protocol["evaluations"].values() for key in ("output", "payload_output", "log")]]


def validate(protocol):
    if (protocol.get("format") != FORMAT or protocol.get("campaign") != LIMITS
            or protocol.get("predeclared_before_fit") is not True
            or protocol.get("scientific_acceptance") is not False
            or protocol.get("endpoint_policy") != {"primary": "final_epoch200", "secondary": "native minimum validation objective only if its epoch differs from200",
                "selection_used_posterior_diagnostics": False}):
        raise ValueError("bounded predeclared 100k comparison required")
    protected = {**protocol["sources"], "task": protocol["task_file"], "parent": protocol["parent_study"],
        "parent_evaluation": protocol["parent_10k_evaluation"],
        "original_study": protocol["original_whitening_study"], "historical_inputs": protocol["historical_population_inputs"]}
    for name, spec in protected.items():
        if shared.digest(spec["path"]) != spec["sha256"]:
            raise ValueError("source or input changed: "+name)
    if (protocol["sources"]["controller"]["sha256"] != shared.digest(__file__)
            or shared.read(protocol["task_path"]) != protocol["task"]):
        raise ValueError("controller or financial task changed")
    paths = [str(Path(p).resolve()) for p in output_paths(protocol)]
    if len(set(paths)) != len(paths):
        raise ValueError("all output identities must be distinct")


def run(protocol_path, out):
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    protocol = shared.read(protocol_path); validate(protocol)
    if out.exists() or out.with_suffix(".claim.json").exists() or any(Path(p).exists() for p in output_paths(protocol)):
        raise FileExistsError("100k output identities must be new; no retries")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".claim.json").open("x") as stream:
        json.dump(identity(protocol_path), stream)
    result = {"format": "heston_pr87_100k_campaign_v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "protocol": identity(protocol_path),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "jobs": [], "automatic_retry": False, "scientific_acceptance": False}
    env = dict(os.environ)
    env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
    process = None; started = time.monotonic(); shared.write(out, result)

    def execute(name, command, log_path, cap, expected):
        nonlocal process
        validate(protocol)
        entry = {"id": name, "command": command, "log": log_path,
            "wall_seconds_limit": cap, "status": "running"}
        result["jobs"].append(entry); tick = time.monotonic(); shared.write(out, result)
        with Path(log_path).open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            entry["pid"] = process.pid; shared.write(out, result)
            try: process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                shared.stop(process); entry["status"] = "wall_cap"
                raise TimeoutError("100k job exceeded its wall cap: "+name)
        entry.update(returncode=process.returncode, seconds=time.monotonic()-tick)
        if process.returncode or not expected.is_file() or shared.read(expected).get("status") != "complete":
            entry["status"] = "failed"
            raise RuntimeError("100k campaign job failed: "+name)
        entry.update(status="complete", record=identity(expected)); shared.write(out, result)

    try:
        command = [sys.executable, protocol["sources"]["producer"]["path"], "--parent", protocol["parent_training"],
            "--protocol", str(protocol_path), "--out", protocol["output"], "--worker"]
        execute("training", command, protocol["training_log"], 3600, Path(protocol["output"])/"training.json")
        training = shared.read(Path(protocol["output"])/"training.json")
        best_epoch = int(training["best_validation_archive"]["epoch"])
        result["best_validation_epoch"] = best_epoch
        for role, spec in protocol["evaluations"].items():
            if role == "secondary" and best_epoch == 200:
                result["secondary_skipped"] = "best validation is the same epoch200 primary endpoint"
                shared.write(out, result); continue
            checkpoint = Path(protocol["output"])/"checkpoints"/spec["checkpoint"]
            command = [sys.executable, protocol["sources"]["evaluator"]["path"], "--checkpoint", str(checkpoint),
                "--protocol", str(protocol_path), "--out", spec["output"], "--payload-out", spec["payload_output"], "--worker"]
            execute(role+"_evaluation", command, spec["log"], 900, Path(spec["output"])/"evaluation.json")
        validate(protocol)
        if shared.digest(protocol_path) != result["protocol"]["sha256"]:
            raise ValueError("100k declaration changed during execution")
        result.update(status="complete", seconds=time.monotonic()-started)
        shared.write(out, result); return result
    except BaseException as error:
        if process is not None: shared.stop(process)
        result.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic()-started)
        shared.write(out, result); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("parent-study", "evidence", "payload"): prep.add_argument("--"+key, type=Path, required=True)
    launch = sub.add_parser("run")
    for key in ("protocol", "out"): launch.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": print(prepare(args.parent_study, args.evidence, args.payload))
    else: print(json.dumps(run(args.protocol, args.out)), flush=True)

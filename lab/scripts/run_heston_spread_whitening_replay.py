"""Freeze and supervise the existing ZCA trajectory's earlier checkpoints.

One replay stops at epoch100 while preserving the original epoch200 cosine
schedule. Four read-only evaluations reuse development data and saved MCMC.
There are no retries, new pricing labels, or reference sampling jobs.
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

FORMAT = "heston_spread_whitening_checkpoint_replay_study_v1"
STEPS = (2000, 3000, 3320, 4000)
LIMITS = {"training_wall_seconds": 900, "evaluation_wall_seconds": 300,
    "max_concurrent_workers": 1, "automatic_retry": False, "automatic_extension": False}


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": shared.digest(path)}


def prepare(original, study, evidence, payload):
    from scripts import replay_heston_spread_whitening as trainer
    from scripts import evaluate_heston_spread_whitening_replay as evaluator
    from scripts import render_heston_spread_whitening_replay as renderer
    original, study, evidence, payload = map(lambda p: Path(p).resolve(),
        (original, study, evidence, payload))
    if evidence.exists() or payload.exists():
        raise FileExistsError("replay evidence and payload identities must be new")
    previous = shared.read(study)
    if previous.get("format") != "heston_spread_whitening_study_v1":
        raise ValueError("existing paired whitening study required")
    task, bank = previous["task"], previous["bank"]
    contract = trainer.protocol_contract(bank["path"], task, original)
    evaluations = [{"updates": step, "epoch": step//40,
        "output": str(evidence/"evaluations"/f"update-{step:06d}"),
        "payload_output": str(payload/"evaluations"/f"update-{step:06d}"),
        "log": str(payload/f"evaluation-{step:06d}.log")} for step in STEPS]
    final = study.parent/"evaluations"/"zca"
    record = {"format": FORMAT, "declared_utc": datetime.now(timezone.utc).isoformat(),
        "replay_points_declared_before_fit": True, "checkpoint_selection_posthoc": True,
        "selection_scope": "Development diagnosis: epoch83 was selected after inspecting the historical validation curve; no fresh acceptance claim.",
        "task": task, "task_path": str(evidence/"task.json"), "bank": bank,
        "original": {"path": str(original), "run": identity(original/"run.json")},
        "original_study": identity(study), "contract": contract,
        "output": str(payload/"training"), "log": str(payload/"training.log"),
        "evaluations": evaluations, "evaluation": evaluator.evaluation_contract(task["target"]),
        "reused_final_evaluation": identity(final/"evaluation.json"),
        "reused_final_recovery": identity(final/"recovery.npz"),
        "campaign": LIMITS, "sources": {"trainer": identity(trainer.__file__),
            "controller": identity(__file__), "evaluator": identity(evaluator.__file__),
            "renderer": identity(renderer.__file__)},
        "reserved_arrays_opened": False, "new_pricing_labels": False,
        "new_reference_sampling": False, "scientific_acceptance": False}
    evidence.mkdir(parents=True); payload.mkdir(parents=True)
    shared.write(evidence/"task.json", task)
    record["task_file"] = identity(evidence/"task.json")
    shared.write(evidence/"protocol.json", record)
    return evidence/"protocol.json"


def validate(protocol):
    if (protocol.get("format") != FORMAT or protocol.get("campaign") != LIMITS
            or [e.get("updates") for e in protocol.get("evaluations", [])] != list(STEPS)
            or protocol.get("checkpoint_selection_posthoc") is not True
            or protocol.get("scientific_acceptance") is not False):
        raise ValueError("bounded exploratory checkpoint replay required")
    protected = {**protocol["sources"], "task": protocol["task_file"],
        "original_run": protocol["original"]["run"], "original_study": protocol["original_study"],
        "final_evaluation": protocol["reused_final_evaluation"],
        "final_recovery": protocol["reused_final_recovery"]}
    for name, source in protected.items():
        if shared.digest(source["path"]) != source["sha256"]:
            raise ValueError("declared source or historical input changed: "+name)
    if (protocol["sources"]["controller"]["sha256"] != shared.digest(__file__)
            or shared.read(protocol["task_path"]) != protocol["task"]):
        raise ValueError("controller or statistical task differs")
    outputs = [protocol["output"], protocol["log"],
        *[entry[key] for entry in protocol["evaluations"] for key in ("output", "payload_output", "log")]]
    if len(set(map(lambda p: str(Path(p).resolve()), outputs))) != len(outputs):
        raise ValueError("all replay output identities must be distinct")


def run(protocol_path, out):
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    protocol = shared.read(protocol_path); validate(protocol)
    outputs = [protocol["output"], protocol["log"],
        *[entry[key] for entry in protocol["evaluations"] for key in ("output", "payload_output", "log")]]
    if out.exists() or out.with_suffix(".claim.json").exists() or any(Path(p).exists() for p in outputs):
        raise FileExistsError("replay outputs must be new; no retries")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".claim.json").open("x") as stream:
        json.dump(identity(protocol_path), stream)
    result = {"format": "heston_spread_whitening_checkpoint_replay_campaign_v1",
        "status": "running", "protocol": identity(protocol_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "jobs": [], "automatic_retry": False, "scientific_acceptance": False}
    started = time.monotonic(); shared.write(out, result)
    training = [sys.executable, protocol["sources"]["trainer"]["path"],
        "--bank", protocol["bank"]["path"], "--task", protocol["task_path"],
        "--original", protocol["original"]["path"], "--protocol", str(protocol_path),
        "--out", protocol["output"], "--worker"]
    jobs = [("training", training, protocol["log"], 900,
        Path(protocol["output"])/"training.json")]
    for entry in protocol["evaluations"]:
        checkpoint = Path(protocol["output"])/"checkpoints"/f"update-{entry['updates']:06d}"
        command = [sys.executable, protocol["sources"]["evaluator"]["path"],
            "--checkpoint", str(checkpoint), "--study", protocol["original_study"]["path"],
            "--replay-protocol", str(protocol_path), "--out", entry["output"],
            "--payload-out", entry["payload_output"]]
        jobs.append((f"evaluation-{entry['updates']:06d}", command, entry["log"], 300,
            Path(entry["output"])/"evaluation.json"))
    env = dict(os.environ)
    env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
    process = None
    try:
        for name, command, log_path, limit, expected in jobs:
            validate(protocol)
            entry = {"id": name, "command": command, "log": log_path,
                "wall_seconds_limit": limit, "status": "running"}
            result["jobs"].append(entry); shared.write(out, result)
            tick = time.monotonic()
            with Path(log_path).open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                entry["pid"] = process.pid; shared.write(out, result)
                try:
                    process.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    shared.stop(process); entry["status"] = "wall_cap"
                    raise TimeoutError("bounded replay job exceeded wall limit: "+name)
            entry.update(returncode=process.returncode, seconds=time.monotonic()-tick)
            if process.returncode or not expected.is_file():
                entry["status"] = "failed"
                raise RuntimeError("replay job failed: "+name)
            saved = shared.read(expected)
            if saved.get("status") != "complete":
                entry["status"] = "failed"
                raise ValueError("replay job did not record completion: "+name)
            if name == "training" and (saved.get("original_learning_curve_prefix_exact") is not True
                    or saved.get("original_curve_points_compared") != 100
                    or saved.get("all_original_initial_state_hashes_exact") is not True):
                entry["status"] = "failed"
                raise ValueError("replay did not reproduce the full original trajectory prefix")
            entry.update(status="complete", record=identity(expected))
            shared.write(out, result)
        validate(protocol)
        if shared.digest(protocol_path) != result["protocol"]["sha256"]:
            raise ValueError("replay declaration changed during execution")
        result.update(status="complete", seconds=time.monotonic()-started)
        shared.write(out, result); return result
    except BaseException as error:
        if process is not None: shared.stop(process)
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
            seconds=time.monotonic()-started)
        shared.write(out, result); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("original", "study", "evidence", "payload"):
        prep.add_argument("--"+key, type=Path, required=True)
    launch = sub.add_parser("run")
    for key in ("protocol", "out"): launch.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": print(prepare(args.original, args.study, args.evidence, args.payload))
    else: print(json.dumps(run(args.protocol, args.out)), flush=True)

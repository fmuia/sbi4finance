"""Freeze and supervise one faithful statistical-recipe reproduction of PR87.

The collaborator's random seed and full software versions are unavailable.
This study records its added seed/current locked environment and preserves the
notebook's data generation, preprocessing, unconstrained targets and native fit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_capacity as shared

FORMAT = "heston_pr87_reproduction_study_v1"
NOTEBOOK_COMMIT = "ddf3619faf48b8f4b0ee56ee841ef24124a508d1"
NOTEBOOK_SHA256 = "22ddfe2d8b97b80b97ac6d0d92f40c04022c9494bfb6fc1fc829574014372641"
LIMITS = {"training_and_generation_wall_seconds": 1800,
    "evaluation_wall_seconds": 900, "max_concurrent_workers": 1,
    "automatic_retry": False, "automatic_extension": False}


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": shared.digest(path)}


def prepare(inputs, notebook, original_study, evidence, payload):
    from scripts import reproduce_heston_pr87 as trainer
    from scripts import evaluate_heston_pr87_reproduction as evaluator
    from scripts import render_heston_pr87_reproduction as renderer
    inputs, notebook, original_study, evidence, payload = map(lambda x: Path(x).resolve(),
        (inputs, notebook, original_study, evidence, payload))
    if evidence.exists() or payload.exists():
        raise FileExistsError("PR87 reproduction identities must be new")
    if shared.digest(notebook) != NOTEBOOK_SHA256:
        raise ValueError("notebook differs from the reviewed PR87 commit")
    old, source = shared.read(original_study), shared.read(inputs)
    task = source["task"]
    if (old.get("format") != "heston_spread_whitening_study_v1"
            or task != old["task"] or task["target"]["infer_c"] is not True):
        raise ValueError("the identical six-target financial task is required")
    portable = inputs.parent/source["payload"]
    if portable.resolve().parent != inputs.parent or shared.digest(portable) != source["payload_sha256"]:
        raise ValueError("historical development observations changed")
    evidence.mkdir(parents=True); payload.mkdir(parents=True)
    archived_notebook = payload/"upstream-notebook.ipynb"
    shutil.copyfile(notebook, archived_notebook)
    shared.write(evidence/"task.json", task)
    contract = trainer.protocol_contract(task, archived_notebook)
    protocol = {"format": FORMAT, "declared_utc": datetime.now(timezone.utc).isoformat(),
        "predeclared_before_fit": True, "contract": contract,
        "task": task, "task_path": str(evidence/"task.json"),
        "task_file": identity(evidence/"task.json"), "input_manifest": identity(inputs),
        "upstream_notebook": {**identity(archived_notebook),
            "pr": "https://github.com/fmuia/sbi4finance/pull/87", "commit": NOTEBOOK_COMMIT,
            "repository_path": "lab/notebooks/heston_6_parameter_bayesflow.ipynb"},
        "original_whitening_study": identity(original_study),
        "historical_population_inputs": identity(inputs),
        "output": str(payload/"training"), "training_log": str(payload/"training.log"),
        "evaluation": evaluator.evaluation_contract(task["target"]),
        "evaluation_output": str(evidence/"evaluation"),
        "evaluation_payload_output": str(payload/"evaluation"),
        "evaluation_log": str(payload/"evaluation.log"),
        "sources": {"producer": identity(trainer.__file__), "controller": identity(__file__),
            "evaluator": identity(evaluator.__file__), "renderer": identity(renderer.__file__)},
        "campaign": LIMITS,
        "primary_endpoint": "final epoch200 exactly as notebook; no best-validation selection",
        "scope": "One statistical-recipe reproduction. Fresh training simulations, native FlowMatching, dropout, target coordinates and optimizer differ from the controlled spline pilot; this does not isolate network-family effects.",
        "reproduction_limits": ["Original notebook has no seed or retained training data/weights; seed9011 is added for reproducibility.",
            "Original metadata records Python3.12.11 but not full package versions; current locked BF2.1/JAX environment is recorded.",
            "Hardware/runtime and unknown original randomness prevent bitwise reproduction of the notebook's saved output."],
        "fresh_noise_per_pass": False, "reserved_arrays_opened": False,
        "new_reference_sampling": False, "scientific_acceptance": False}
    shared.write(evidence/"protocol.json", protocol)
    return evidence/"protocol.json"


def validate(protocol):
    if (protocol.get("format") != FORMAT or protocol.get("campaign") != LIMITS
            or protocol.get("predeclared_before_fit") is not True
            or protocol.get("scientific_acceptance") is not False
            or protocol.get("fresh_noise_per_pass") is not False):
        raise ValueError("bounded single-fit notebook reproduction required")
    protected = {**protocol["sources"], "task": protocol["task_file"],
        "input_manifest": protocol["input_manifest"], "notebook": protocol["upstream_notebook"],
        "original_study": protocol["original_whitening_study"],
        "historical_inputs": protocol["historical_population_inputs"]}
    for name, source in protected.items():
        if shared.digest(source["path"]) != source["sha256"]:
            raise ValueError("declared source or input changed: "+name)
    if (protocol["sources"]["controller"]["sha256"] != shared.digest(__file__)
            or shared.read(protocol["task_path"]) != protocol["task"]):
        raise ValueError("controller or statistical task differs")
    outputs = [protocol[key] for key in ("output", "training_log", "evaluation_output",
        "evaluation_payload_output", "evaluation_log")]
    if len(set(str(Path(p).resolve()) for p in outputs)) != len(outputs):
        raise ValueError("reproduction output identities must be distinct")


def run(protocol_path, out):
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    protocol = shared.read(protocol_path); validate(protocol)
    outputs = [protocol[key] for key in ("output", "training_log", "evaluation_output",
        "evaluation_payload_output", "evaluation_log")]
    if out.exists() or out.with_suffix(".claim.json").exists() or any(Path(p).exists() for p in outputs):
        raise FileExistsError("single-fit reproduction outputs must be new; no retries")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".claim.json").open("x") as stream:
        json.dump(identity(protocol_path), stream)
    result = {"format": "heston_pr87_reproduction_campaign_v1", "status": "running",
        "protocol": identity(protocol_path), "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "jobs": [], "automatic_retry": False, "scientific_acceptance": False}
    training = [sys.executable, protocol["sources"]["producer"]["path"],
        "--task", protocol["task_path"], "--protocol", str(protocol_path), "--out", protocol["output"], "--worker"]
    checkpoint = Path(protocol["output"])/"checkpoints"/"epoch-000200"
    evaluation = [sys.executable, protocol["sources"]["evaluator"]["path"],
        "--checkpoint", str(checkpoint), "--protocol", str(protocol_path),
        "--out", protocol["evaluation_output"], "--payload-out", protocol["evaluation_payload_output"], "--worker"]
    jobs = [("training", training, protocol["training_log"], 1800,
        Path(protocol["output"])/"training.json"),
        ("evaluation", evaluation, protocol["evaluation_log"], 900,
        Path(protocol["evaluation_output"])/"evaluation.json")]
    env = dict(os.environ)
    env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
    process, started = None, time.monotonic(); shared.write(out, result)
    try:
        for name, command, log_path, cap, expected in jobs:
            validate(protocol)
            entry = {"id": name, "command": command, "wall_seconds_limit": cap,
                "log": log_path, "status": "running"}
            result["jobs"].append(entry); shared.write(out, result)
            tick = time.monotonic()
            with Path(log_path).open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                entry["pid"] = process.pid; shared.write(out, result)
                try:
                    process.wait(timeout=cap)
                except subprocess.TimeoutExpired:
                    shared.stop(process); entry["status"] = "wall_cap"
                    raise TimeoutError("bounded notebook reproduction exceeded wall cap: "+name)
            entry.update(returncode=process.returncode, seconds=time.monotonic()-tick)
            if process.returncode or not expected.is_file() or shared.read(expected).get("status") != "complete":
                entry["status"] = "failed"
                raise RuntimeError("notebook reproduction job failed: "+name)
            entry.update(status="complete", record=identity(expected)); shared.write(out, result)
        validate(protocol)
        if shared.digest(protocol_path) != result["protocol"]["sha256"]:
            raise ValueError("reproduction declaration changed")
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
    p = sub.add_parser("prepare")
    for key in ("inputs", "notebook", "original-study", "evidence", "payload"):
        p.add_argument("--"+key, type=Path, required=True)
    r = sub.add_parser("run")
    for key in ("protocol", "out"): r.add_argument("--"+key, type=Path, required=True)
    a = parser.parse_args()
    if a.command == "prepare": print(prepare(a.inputs, a.notebook, a.original_study, a.evidence, a.payload))
    else: print(json.dumps(run(a.protocol, a.out)), flush=True)

"""Supervise the requested 1024/4098 Heston batch comparison without retries.

This is a new study identity. Existing producers and artifacts stay unchanged.
Only direct training children are launched; evaluation reuses settled outputs.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_capacity as shared

BATCH_SIZES = (1024, 4098)
WORKER_SECONDS, CAMPAIGN_SECONDS = 5400, 5700


def validate(protocol):
    if (protocol.get("format") != "heston_spread_scale_large_batch_study_v1"
            or [a.get("batch_size") for a in protocol.get("arms", [])] != list(BATCH_SIZES)
            or protocol.get("campaign") != {
                "worker_wall_seconds": WORKER_SECONDS,
                "whole_training_wall_seconds": CAMPAIGN_SECONDS,
                "max_concurrent_workers": 2,
                "automatic_retry": False, "automatic_extension": False}):
        raise ValueError("not the declared bounded 1024/4098 study")
    for source in protocol["sources"].values():
        if shared.digest(source["path"]) != source["sha256"]:
            raise ValueError("declared source differs")
    if protocol["sources"]["controller"]["sha256"] != shared.digest(__file__):
        raise ValueError("controller source differs")
    for key in ("parent_checkpoint", "bank"):
        value = protocol[key]
        record = Path(value["path"]) / ("checkpoint.json" if key == "parent_checkpoint" else "bank.json")
        if (Path(value["record"]["path"]).resolve() != record.resolve()
                or shared.digest(record) != value["record"]["sha256"]):
            raise ValueError(key + " record differs")
        for name, binding in value.items():
            if name != "path" and shared.digest(binding["path"]) != binding["sha256"]:
                raise ValueError(key + " payload differs")
    for arm in protocol["arms"]:
        c = arm["contract"]
        batch = arm["batch_size"]
        batches = (100000 + batch - 1) // batch
        if (c.get("batch_size") != batch or c.get("train_rows") != 100000
                or c.get("batches_per_pass") != batches
                or c.get("tail_batch_size") != 100000 - (batches - 1) * batch
                or c.get("additional_bank_passes") != 500
                or c.get("additional_presentations") != 50000000
                or c.get("additional_optimizer_updates") != 500 * batches
                or c.get("checkpoint_bank_passes") != [25, 100, 250, 500]
                or c.get("parent_actual_optimizer_updates") != 60000
                or c.get("max_wall_seconds") != WORKER_SECONDS
                or c.get("smoke_test_only") is not False):
            raise ValueError("large-batch exposure/parent/endpoint contract differs")
    outputs = [str(Path(a[key]).resolve()) for a in protocol["arms"] for key in ("output", "log")]
    if len(set(outputs)) != len(outputs):
        raise ValueError("arm outputs/logs must be distinct")


def run(protocol_path, out):
    started = time.monotonic()
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    protocol = shared.read(protocol_path)
    validate(protocol)
    if out.exists() or out.with_suffix(".claim.json").exists():
        raise FileExistsError("campaign output/claim must be new")
    if any(Path(a[key]).exists() for a in protocol["arms"] for key in ("output", "log")):
        raise FileExistsError("arm output/log must be new; no retries")
    out.parent.mkdir(parents=True, exist_ok=True)
    binding = {"path": str(protocol_path), "sha256": shared.digest(protocol_path)}
    with out.with_suffix(".claim.json").open("x") as stream:
        __import__("json").dump(binding, stream)
    result = {"format": "heston_spread_scale_large_batch_campaign_v1", "status": "running",
        "protocol": binding, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arms": {}, "automatic_retry": False, "automatic_extension": False,
        "scientific_acceptance": False, "reserved_arrays_opened": False}
    workers = {}
    shared.write(out, result)
    try:
        for arm in protocol["arms"]:
            key = str(arm["batch_size"])
            log_path = Path(arm["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("x")
            env = dict(os.environ)
            env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
            command = [sys.executable, protocol["sources"]["trainer"]["path"],
                "--checkpoint", protocol["parent_checkpoint"]["path"], "--bank", protocol["bank"]["path"],
                "--protocol", str(protocol_path), "--batch-size", key, "--out", arm["output"], "--worker"]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            workers[key] = {"process": process, "log": log, "started": time.monotonic(), "arm": arm}
            result["arms"][key] = {"status": "running", "pid": process.pid,
                "output": arm["output"], "log": arm["log"], "command": command}
            shared.write(out, result)
        while any(w["process"].poll() is None for w in workers.values()):
            now = time.monotonic()
            for key, worker in workers.items():
                process = worker["process"]
                if process.poll() is None and (now - worker["started"] >= WORKER_SECONDS
                        or now - started >= CAMPAIGN_SECONDS):
                    shared.stop(process)
                    result["arms"][key]["status"] = "wall_cap"
                if process.poll() is not None and "finished" not in worker:
                    worker["finished"] = time.monotonic()
            time.sleep(1)
        for key, worker in workers.items():
            entry, process = result["arms"][key], worker["process"]
            training = Path(worker["arm"]["output"]) / "training.json"
            entry.update(returncode=process.returncode,
                seconds=worker.get("finished", time.monotonic()) - worker["started"])
            if entry["status"] != "wall_cap":
                entry["status"] = "complete" if process.returncode == 0 and training.is_file() else "failed"
            if training.is_file():
                entry["training"] = {"path": str(training), "sha256": shared.digest(training)}
            worker["log"].close()
        if shared.digest(protocol_path) != binding["sha256"]:
            raise ValueError("protocol changed during training")
        validate(protocol)
        result.update(status="complete" if all(a["status"] == "complete" for a in result["arms"].values())
            else "incomplete", seconds=time.monotonic() - started)
        shared.write(out, result)
        return result
    except BaseException as error:
        for worker in workers.values():
            shared.stop(worker["process"])
            worker["log"].close()
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
            seconds=time.monotonic() - started)
        shared.write(out, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.out)
    print(__import__("json").dumps(result), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)

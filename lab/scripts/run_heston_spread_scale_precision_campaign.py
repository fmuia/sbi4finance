"""Run the numerical repair within the original larger-bank campaign deadline."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_capacity as shared


def deadline_seconds(protocol, *, now=None):
    original = protocol["original_campaign"]
    if shared.digest(original["path"]) != original["sha256"]:
        raise ValueError("original campaign record differs")
    record = shared.read(original["path"])
    expected = datetime.fromisoformat(record["started_utc"]) + timedelta(seconds=5700)
    declared = datetime.fromisoformat(protocol["campaign"]["absolute_deadline_utc"])
    if (record["status"] != "incomplete" or declared != expected
            or protocol["campaign"]["worker_wall_seconds"] != 5400
            or protocol["campaign"]["automatic_retry"] is not False):
        raise ValueError("precision replay cannot extend the original campaign")
    return (expected - (now or datetime.now(timezone.utc))).total_seconds()


def run(protocol_path, out):
    started = time.monotonic()
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    protocol = shared.read(protocol_path)
    if (protocol["format"] != "heston_spread_scale_precision_recovery_study_v1"
            or [arm["batch_size"] for arm in protocol["arms"]] != [128, 512]):
        raise ValueError("unsupported paired precision replay")
    remaining = deadline_seconds(protocol)
    if remaining <= 0:
        raise TimeoutError("original campaign deadline already exhausted")
    deadline = started + remaining
    for source in protocol["sources"].values():
        if shared.digest(source["path"]) != source["sha256"]:
            raise ValueError("precision replay source differs")
    if protocol["sources"]["controller"]["sha256"] != shared.digest(__file__):
        raise ValueError("controller differs")
    if out.exists() or any(Path(arm[key]).exists() for arm in protocol["arms"] for key in ("output", "log")):
        raise FileExistsError("precision replay identities must be new")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".claim.json").open("x") as stream:
        json.dump({"path": str(protocol_path), "sha256": shared.digest(protocol_path)}, stream)
    result = {"format": "heston_spread_scale_precision_campaign_v1", "status": "running",
        "protocol": {"path": str(protocol_path), "sha256": shared.digest(protocol_path)},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "original_campaign": protocol["original_campaign"],
        "absolute_deadline_utc": protocol["campaign"]["absolute_deadline_utc"],
        "remaining_original_wall_seconds_at_launch": remaining,
        "automatic_extension": False, "automatic_retry": False, "arms": {}}
    workers = {}
    shared.write(out, result)
    try:
        for arm in protocol["arms"]:
            key = str(arm["batch_size"])
            log_path = Path(arm["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("x")
            command = [sys.executable, protocol["sources"]["trainer"]["path"],
                "--checkpoint", protocol["parent_checkpoint"]["path"], "--bank", protocol["bank"]["path"],
                "--protocol", str(protocol_path), "--batch-size", key, "--out", arm["output"], "--worker"]
            env = dict(os.environ)
            env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            workers[key] = {"process": process, "log": log, "started": time.monotonic(), "arm": arm}
            result["arms"][key] = {"status": "running", "pid": process.pid,
                "output": arm["output"], "log": arm["log"], "command": command}
            shared.write(out, result)
        while any(worker["process"].poll() is None for worker in workers.values()):
            now = time.monotonic()
            for key, worker in workers.items():
                process = worker["process"]
                if process.poll() is None and (now >= deadline or now-worker["started"] >= 5400):
                    shared.stop(process)
                    result["arms"][key]["status"] = "wall_cap"
                if process.poll() is not None and "finished" not in worker:
                    worker["finished"] = time.monotonic()
            time.sleep(1)
        for key, worker in workers.items():
            entry, process = result["arms"][key], worker["process"]
            training = Path(worker["arm"]["output"]) / "training.json"
            if entry["status"] != "wall_cap":
                entry["status"] = "complete" if process.returncode == 0 and training.is_file() else "failed"
            entry.update(returncode=process.returncode,
                         seconds=worker.get("finished", time.monotonic())-worker["started"])
            if training.is_file():
                entry["training"] = {"path": str(training), "sha256": shared.digest(training)}
            worker["log"].close()
        if shared.digest(protocol_path) != result["protocol"]["sha256"]:
            raise ValueError("precision protocol changed during replay")
        if any(shared.digest(source["path"]) != source["sha256"] for source in protocol["sources"].values()):
            raise ValueError("precision replay source changed")
        result.update(status="complete" if all(a["status"] == "complete" for a in result["arms"].values()) else "incomplete",
                      seconds=time.monotonic()-started)
        shared.write(out, result)
        return result
    except BaseException as error:
        for worker in workers.values():
            shared.stop(worker["process"])
            worker["log"].close()
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        shared.write(out, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.out)
    print(json.dumps(result), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)

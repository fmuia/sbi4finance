"""Supervise exactly two declared continuation workers, without retries.

Only training is supervised here. Posterior evaluation is a separate read-only
step after checkpoints settle. Direct children permit bounded cleanup without
process-group permissions or orphaned nested supervisors.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run(protocol_path, out):
    started = time.monotonic()
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    protocol = read(protocol_path)
    if (protocol["format"] != "heston_spread_scale_capacity_study_v1"
            or [arm["batch_size"] for arm in protocol["arms"]] != [128, 512]
            or protocol["campaign"]["worker_wall_seconds"] != 5400
            or protocol["campaign"]["whole_training_wall_seconds"] != 5700
            or protocol["campaign"]["automatic_retry"] is not False):
        raise ValueError("not the declared bounded two-arm study")
    for value in protocol["sources"].values():
        if digest(value["path"]) != value["sha256"]:
            raise ValueError("declared source differs")
    if protocol["sources"]["controller"]["sha256"] != digest(__file__):
        raise ValueError("controller source differs")
    if out.exists():
        raise FileExistsError("campaign output must be new")
    for arm in protocol["arms"]:
        if Path(arm["output"]).exists() or Path(arm["log"]).exists():
            raise FileExistsError("arm output/log must be new; no retries")
    out.parent.mkdir(parents=True, exist_ok=True)
    claim = out.with_suffix(".claim.json")
    with claim.open("x") as stream:
        json.dump({"protocol_path": str(protocol_path), "protocol_sha256": digest(protocol_path)}, stream)
    result = {"format": "heston_spread_scale_capacity_campaign_v1", "status": "running",
        "protocol_path": str(protocol_path), "protocol_sha256": digest(protocol_path),
        "started_utc": datetime.now(timezone.utc).isoformat(), "arms": {},
        "automatic_retry": False, "reserved_arrays_opened": False,
        "scientific_acceptance": False}
    write(out, result)
    workers = {}
    try:
        for arm in protocol["arms"]:
            key = str(arm["batch_size"])
            log_path = Path(arm["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("x")
            env = dict(os.environ)
            env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
            args = [sys.executable, protocol["sources"]["trainer"]["path"],
                "--checkpoint", protocol["parent_checkpoint"]["path"], "--bank", protocol["bank"]["path"],
                "--protocol", str(protocol_path), "--batch-size", key, "--out", arm["output"], "--worker"]
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
            workers[key] = {"process": process, "log": log, "started": time.monotonic(), "arm": arm}
            result["arms"][key] = {"status": "running", "pid": process.pid, "log": str(log_path),
                "output": arm["output"], "command": args}
            write(out, result)
        while any(value["process"].poll() is None for value in workers.values()):
            now = time.monotonic()
            for key, value in workers.items():
                process = value["process"]
                if process.poll() is None and (now - value["started"] >= 5400 or now - started >= 5700):
                    stop(process)
                    result["arms"][key]["status"] = "wall_cap"
                if process.poll() is not None and "finished" not in value:
                    value["finished"] = time.monotonic()
            time.sleep(1)
        for key, value in workers.items():
            entry = result["arms"][key]
            process, arm = value["process"], value["arm"]
            entry.update(returncode=process.returncode, seconds=value.get("finished", time.monotonic()) - value["started"])
            training = Path(arm["output"]) / "training.json"
            if entry["status"] != "wall_cap":
                entry["status"] = "complete" if process.returncode == 0 and training.is_file() else "failed"
            if training.is_file():
                entry["training"] = {"path": str(training), "sha256": digest(training)}
            value["log"].close()
        if digest(protocol_path) != result["protocol_sha256"]:
            raise ValueError("protocol changed during training")
        for source in protocol["sources"].values():
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("declared source changed during training")
        result.update(status="complete" if all(a["status"] == "complete" for a in result["arms"].values())
                      else "incomplete", seconds=time.monotonic() - started)
        write(out, result)
        return result
    except BaseException as error:
        for value in workers.values():
            stop(value["process"])
            value["log"].close()
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      seconds=time.monotonic() - started)
        write(out, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    outcome = run(args.protocol, args.out)
    print(json.dumps(outcome), flush=True)
    if outcome["status"] != "complete":
        raise SystemExit(1)

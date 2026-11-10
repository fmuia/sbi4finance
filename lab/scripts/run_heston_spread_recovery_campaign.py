"""Supervise eight independent joint MCMC recovery fits, two at a time."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

STARTED = time.monotonic()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def child_environment(case_id):
    if not case_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in case_id):
        raise ValueError("safe case identity required")
    base = "/private/tmp/heston-mcmc-recovery-v1/"+case_id
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
        MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib",
        PYTENSOR_FLAGS=f"base_compiledir={base},compiledir={base}/compiled",
        NUMBA_CACHE_DIR=base+"/numba-native")
    return env


def run(protocol_path, out, logs):
    protocol_path, out, logs = map(lambda p: Path(p).resolve(), (protocol_path, out, logs))
    protocol = read(protocol_path)
    limits = {"max_concurrent": 2, "whole_wall_seconds": 4500,
        "per_case_outer_wall_seconds": 950, "max_sum_inner_cpu_seconds": 7200,
        "automatic_retry": False}
    if (protocol.get("format") != "heston_spread_recovery_reference_protocol_v1"
            or protocol.get("campaign") != limits or len(protocol["jobs"]) != 8
            or len({job["id"] for job in protocol["jobs"]}) != 8):
        raise ValueError("bounded eight-case recovery protocol required")
    if protocol["controller"] != {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)}:
        raise ValueError("campaign source differs")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import run_heston_spread_recovery_reference as runner
    for job in protocol["jobs"]:
        runner.validate_protocol(protocol, protocol["panel_path"], job["case_id"],
                                 job["evidence_out"], job["payload_out"])
        if Path(job["evidence_out"]).exists() or Path(job["payload_out"]).exists():
            raise FileExistsError("reference identities must be new; no automatic retries")
    if out.exists() or logs.exists():
        raise FileExistsError("campaign and log identities must be new")
    out.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True)
    with out.with_suffix(".claim.json").open("x") as stream:
        json.dump({"path": str(protocol_path), "sha256": digest(protocol_path)}, stream)
    result = {"format": "heston_spread_recovery_campaign_v1", "status": "running",
        "protocol": {"path": str(protocol_path), "sha256": digest(protocol_path)},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "limits": limits,
        "jobs": {job["id"]: {"status": "pending"} for job in protocol["jobs"]},
        "automatic_retry": False, "scientific_acceptance": False}
    write(out, result)
    pending, active = list(protocol["jobs"]), {}
    try:
        while pending or active:
            now = time.monotonic()
            if now-STARTED >= 4500:
                for job in pending:
                    result["jobs"][job["id"]] = {"status": "not_started_campaign_cap"}
                pending.clear()
                for entry in active.values():
                    stop(entry["process"])
                    entry["wall_cap"] = "campaign"
            for key, entry in list(active.items()):
                process = entry["process"]
                if process.poll() is None and now-entry["started"] >= 950:
                    stop(process)
                    entry["wall_cap"] = "case"
                if process.poll() is not None:
                    path = Path(entry["job"]["evidence_out"])/"run.json"
                    record = read(path) if path.is_file() else {}
                    status = ("ready" if process.returncode == 0 and record.get("reference_ready") is True
                        else "not_converged" if process.returncode == 0 and record.get("status") == "completed"
                        else "failed")
                    result["jobs"][key].update(status=status, exit_code=process.returncode,
                        wall_seconds=time.monotonic()-entry["started"],
                        reference_ready=record.get("reference_ready", False),
                        usage=record.get("usage", {}), wall_cap=entry.get("wall_cap"))
                    if path.is_file():
                        result["jobs"][key]["run"] = {"path": str(path), "sha256": digest(path)}
                    entry["log"].close()
                    del active[key]
                    write(out, result)
            while pending and len(active) < 2:
                job = pending.pop(0)
                key = job["id"]
                log_path = logs/(key+".log")
                log = log_path.open("x")
                command = [sys.executable, str(Path(runner.__file__).resolve()),
                    "--panel", protocol["panel_path"], "--case-id", job["case_id"],
                    "--protocol", str(protocol_path), "--out", job["evidence_out"],
                    "--payload-out", job["payload_out"], "--worker"]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           env=child_environment(job["case_id"]))
                active[key] = {"process": process, "log": log, "started": time.monotonic(), "job": job}
                result["jobs"][key] = {"status": "running", "pid": process.pid,
                    "log": str(log_path), "evidence_out": job["evidence_out"],
                    "payload_out": job["payload_out"], "command": command}
                write(out, result)
            if active:
                time.sleep(.5)
        if digest(protocol_path) != result["protocol"]["sha256"]:
            raise ValueError("protocol changed during sampling")
        for job in protocol["jobs"]:
            runner.validate_protocol(protocol, protocol["panel_path"], job["case_id"],
                                     job["evidence_out"], job["payload_out"])
        result.update(status="completed", wall_seconds=time.monotonic()-STARTED,
            ready_cases=sum(job["status"] == "ready" for job in result["jobs"].values()),
            summed_inner_cpu_seconds=sum(job.get("usage", {}).get("cpu_seconds", 0.) for job in result["jobs"].values()))
        write(out, result)
        return result
    except BaseException as error:
        for entry in active.values():
            stop(entry["process"])
            entry["log"].close()
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write(out, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "out", "logs"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.out, args.logs)), flush=True)

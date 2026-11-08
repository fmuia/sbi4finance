"""Run one admitted Heston budget-study lane with a durable resource reservation.

This ordinary Python CLI is independent of Slurm/Oscar. Planning lists commands;
execution requires hashed readiness/smoke/admission records. Every lane gets a
new directory and consumes its full reserved allowance even if it fails. There
is no retry, matrix expansion or implicit admission after elapsed time.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/heston_training_budget_study.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def candidates(protocol):
    initial = protocol["screen_candidates"]
    if len(initial) != 14 or len({row["id"] for row in initial}) != 14:
        raise ValueError("the frozen screen must contain exactly14 unique seed0 candidates")
    if any(row["seed"] != 0 or row["widths"] not in ([128, 128], [256] * 4)
           or row["updates"] not in (30, 100, 300, 1000, 3000, 10000, 30000) for row in initial):
        raise ValueError("candidate differs from the declared screen")
    return {row["id"]: row for row in initial}


def command(candidate, bank, out):
    return [sys.executable, str(ROOT / "scripts/train_full_surface.py"),
            "--bank", str(Path(bank).resolve()), "--out", str(Path(out).resolve()),
            "--family", "flow_matching", "--encoder", "vector",
            "--widths", ",".join(map(str, candidate["widths"])),
            "--updates", str(candidate["updates"]), "--batch-size", "256",
            "--seed", str(candidate["seed"]), "--checkpoint-selection", "final"]


def validate_admission(protocol_path, readiness_path, smoke_path, admission_path, source_revision):
    protocol, readiness, smoke, admission = map(read_json, (protocol_path, readiness_path, smoke_path, admission_path))
    if protocol.get("format") != "heston_training_budget_protocol_v1":
        raise ValueError("unknown protocol format")
    candidates(protocol)
    if (readiness.get("protocol_sha256") != digest(protocol_path)
            or readiness.get("control_checkpoint_sha256") != protocol["control"]["checkpoint_sha256"]
            or readiness.get("status") != "pass" or readiness.get("reserved_test_evaluated") is not False):
        raise ValueError("saved-control readiness did not pass this exact protocol")
    if (smoke.get("same_seed_reload_exact") is not True or smoke.get("scientific_validation") is not False
            or smoke.get("run_contract", {}).get("format") != "full_surface_update_budget_run_v1"
            or smoke.get("selected_update") != smoke.get("run_contract", {}).get("settings", {}).get("updates")
            or smoke.get("performance", {}).get("work", {}).get("optimizer_updates") != smoke.get("selected_update")):
        raise ValueError("exact-update training/reload smoke evidence is missing")
    if admission.get("format") != "heston_training_budget_admission_v1":
        raise ValueError("unknown admission format")
    required = {"protocol_sha256": digest(protocol_path), "readiness_sha256": digest(readiness_path),
                "smoke_training_sha256": digest(smoke_path), "source_revision": source_revision}
    if any(admission.get(key) != value for key, value in required.items()):
        raise ValueError("admission evidence/source identity changed")
    ceiling = protocol["resources"]
    cpus = admission.get("host_cpus")
    gpus = admission.get("gpu_count")
    if type(cpus) is not int or not 1 <= cpus <= ceiling["max_host_cpus"] or gpus not in (0, 1):
        raise ValueError("invalid admitted resource counts")
    lanes = admission.get("lane_wall_seconds", {})
    if not isinstance(lanes, dict) or not set(lanes) <= set(candidates(protocol)):
        raise ValueError("admission may only include frozen seed0 candidates")
    if any(type(seconds) not in (int, float) or seconds <= 0 for seconds in lanes.values()):
        raise ValueError("positive full lane allowances are required")
    cleanup = admission.get("cleanup_seconds_per_lane")
    if type(cleanup) not in (int, float) or cleanup <= 0 or any(seconds <= cleanup for seconds in lanes.values()):
        raise ValueError("each lane allowance must include a positive cleanup reserve")
    # Reserve readiness, failed attempts and final confirmation before the screen.
    prior = admission.get("already_used", {})
    final = admission.get("reserved_confirmation", {})
    if any(type(d.get(k)) not in (int, float) or d[k] < 0
           for d in (prior, final) for k in ("allocated_cpu_seconds", "gpu_seconds")):
        raise ValueError("nonnegative spent and confirmation reservations are required")
    total_seconds = sum(lanes.values())
    cpu = prior["allocated_cpu_seconds"] + final["allocated_cpu_seconds"] + cpus * total_seconds
    gpu = prior["gpu_seconds"] + final["gpu_seconds"] + gpus * total_seconds
    if cpu > ceiling["allocated_cpu_hours_total"] * 3600 or gpu > ceiling["gpu_hours_total"] * 3600:
        raise ValueError("forecast including final-confirmation reserve exceeds the total resource ceiling")
    if admission.get("approved_for_execution") is not True:
        raise ValueError("cost forecast has not been explicitly admitted")
    return protocol, admission


@contextmanager
def ledger_lock(directory):
    """Portable exclusive creation; a crash leaves a visible lock for inspection."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".ledger.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("study ledger is locked; inspect the active/stale owner before retrying") from error
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "created_unix": time.time()}).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reserve_lane(directory, candidate_id, admission_path, admission):
    """Consume the whole per-lane reservation before launch; failed IDs stay used."""
    with ledger_lock(directory):
        path = directory / "ledger.json"
        identity = digest(admission_path)
        ledger = read_json(path) if path.exists() else {
            "format": "heston_training_budget_ledger_v1", "admission_sha256": identity,
            "automatic_retry": False, "lanes": {}}
        if ledger["admission_sha256"] != identity:
            raise ValueError("study ledger belongs to another immutable admission")
        if candidate_id in ledger["lanes"]:
            raise ValueError("candidate already reserved; no automatic retry or overwrite")
        seconds = admission["lane_wall_seconds"].get(candidate_id)
        if seconds is None:
            raise ValueError("candidate has no admitted time allowance")
        lane = directory / candidate_id
        lane.mkdir(exist_ok=False)
        ledger["lanes"][candidate_id] = {"status": "reserved", "reserved_wall_seconds": seconds,
                                          "reserved_cpu_seconds": seconds * admission["host_cpus"],
                                          "reserved_gpu_seconds": seconds * admission["gpu_count"],
                                          "out": str(lane.resolve())}
        write_json(path, ledger)
        return lane, seconds


def source_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run(args):
    # Source identity must bind a clean immutable tracked checkout. Downloads and
    # lane outputs belong outside it; no switching code underneath active jobs.
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("budget lanes require a clean tracked source checkout")
    protocol, admission = validate_admission(args.protocol, args.readiness, args.smoke, args.admission, source_revision())
    if digest(args.bank / "manifest.json") != protocol["bank"]["manifest_sha256"]:
        raise ValueError("bank manifest differs from the frozen study")
    candidate = candidates(protocol).get(args.candidate)
    if candidate is None:
        raise ValueError("candidate is not in the frozen seed0 screen")
    lane, seconds = reserve_lane(args.out, args.candidate, args.admission, admission)
    journal = {"format": "heston_training_budget_lane_v1", "candidate": candidate,
               "status": "starting", "source_revision": admission["source_revision"],
               "protocol_sha256": digest(args.protocol), "admission_sha256": digest(args.admission),
               "reserved_wall_seconds": seconds, "automatic_retry": False,
               "reserved_test_evaluated": False, "command": command(candidate, args.bank, lane / "checkpoint")}
    write_json(lane / "lane.json", journal)
    started = time.perf_counter()
    environment = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "OMP_NUM_THREADS": str(admission["host_cpus"]),
                   "OPENBLAS_NUM_THREADS": str(admission["host_cpus"]),
                   "MKL_NUM_THREADS": str(admission["host_cpus"]),
                   "NUMEXPR_NUM_THREADS": str(admission["host_cpus"])}
    try:
        with (lane / "training.log").open("x") as stream:
            result = subprocess.run(journal["command"], cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=seconds - admission["cleanup_seconds_per_lane"], check=False)
        journal.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
    except BaseException as error:
        journal.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        journal["elapsed_wall_seconds"] = time.perf_counter() - started
        journal["allocated_cpu_seconds"] = admission["host_cpus"] * journal["elapsed_wall_seconds"]
        journal["gpu_seconds"] = admission["gpu_count"] * journal["elapsed_wall_seconds"]
        write_json(lane / "lane.json", journal)
        with ledger_lock(args.out):
            ledger_path = args.out / "ledger.json"
            ledger = read_json(ledger_path)
            ledger["lanes"][args.candidate].update(status=journal["status"],
                                                   elapsed_wall_seconds=journal["elapsed_wall_seconds"])
            write_json(ledger_path, ledger)
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", help="One seed0 candidate ID; omit to print the independent commands only.")
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    if args.candidate is None:
        protocol = read_json(args.protocol)
        print(json.dumps({"protocol_sha256": digest(args.protocol), "launches_jobs": False,
                          "commands": {key: command(row, args.bank, args.out / key / "checkpoint")
                                       for key, row in candidates(protocol).items()}}, indent=2))
        return
    if any(getattr(args, field) is None for field in ("readiness", "smoke", "admission")):
        parser.error("execution requires --readiness, --smoke and --admission; planning alone never launches jobs")
    result = run(args)
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

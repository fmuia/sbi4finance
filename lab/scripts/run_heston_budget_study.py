"""Run one admitted Heston budget-study lane with a durable resource reservation.

This ordinary Python CLI is independent of Slurm/Oscar. Planning lists commands;
execution requires hashed readiness/smoke/admission records. Every lane gets a
new directory and consumes its full reserved allowance even if it fails. There
is no retry, matrix expansion or implicit admission after elapsed time.
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import math
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/heston_training_budget_study.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number is not allowed: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def candidates(protocol):
    initial = protocol["screen_candidates"]
    expected = {f"{name}-u{updates:05d}-s0": {"widths": widths, "updates": updates, "seed": 0}
                for name, widths in (("small", [128, 128]), ("full", [256] * 4))
                for updates in (30, 100, 300, 1000, 3000, 10000, 30000)}
    if (not isinstance(initial, list) or len(initial) != len(expected)
            or any(not isinstance(row, dict) or row.get("id") not in expected for row in initial)
            or len({row["id"] for row in initial}) != len(expected)):
        raise ValueError("the frozen screen must contain exactly the14 unique declared seed0 candidates")
    for row in initial:
        if (type(row.get("seed")) is not int or type(row.get("updates")) is not int
                or any(row.get(key) != value for key, value in expected[row["id"]].items())):
            raise ValueError("candidate differs from the declared screen")
    return {row["id"]: row for row in initial}


def command(candidate, bank, out):
    return [sys.executable, str(ROOT / "scripts/train_full_surface.py"),
            "--bank", str(Path(bank).resolve()), "--out", str(Path(out).resolve()),
            "--family", "flow_matching", "--encoder", "vector",
            "--widths", ",".join(map(str, candidate["widths"])),
            "--updates", str(candidate["updates"]), "--batch-size", "256",
            "--seed", str(candidate["seed"]), "--checkpoint-selection", "final"]


def finite_number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def runtime_identity():
    result = {"python": platform.python_version()}
    for name in ("bayesflow", "keras", "jax", "numpy", "scipy"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def executing_sources():
    """Bind the complete tracked trainer/runtime, not only a movable branch name."""
    paths = ["scripts/train_full_surface.py", "scripts/train_full_surface_budget.py",
             "scripts/run_heston_budget_study.py", "studies/train_calibration.py", "src/sbi4finance",
             "pyproject.toml", "pixi.lock", "../tutorial/src/sbi4f_tutorial",
             "../tutorial/pyproject.toml", "../tutorial/pixi.lock"]
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=ROOT).decode().split("\0")
    return {name: digest(ROOT / name) for name in sorted(tracked)
            if name and (name.endswith(".py") or name.endswith("pyproject.toml") or name.endswith("pixi.lock"))}


def executing_sources_digest():
    return hashlib.sha256(json.dumps(executing_sources(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def clean_source():
    # Normal Git ignores still permit downloaded assets and installed envs.
    # Untracked executing Python files must not bypass the immutable checkout.
    return not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                       cwd=ROOT, text=True).strip()


def validate_admission(protocol_path, readiness_path, smoke_path, admission_path, source_revision):
    protocol, readiness, smoke, admission = map(read_json, (protocol_path, readiness_path, smoke_path, admission_path))
    if protocol.get("format") != "heston_training_budget_protocol_v1":
        raise ValueError("unknown protocol format")
    candidates(protocol)
    expected_gates = {"finite_in_support_without_clipping", "development_rank_screen",
                      "conditional_all_cases", "predictive_pricing_completed"}
    if (readiness.get("format") != "heston_training_budget_readiness_v1"
            or readiness.get("protocol_sha256") != digest(protocol_path)
            or readiness.get("control_checkpoint_sha256") != protocol["control"]["checkpoint_sha256"]
            or readiness.get("bank_manifest_sha256") != protocol["bank"]["manifest_sha256"]
            or readiness.get("status") != "pass" or readiness.get("reserved_test_evaluated") is not False
            or readiness.get("training_performed") is not False or readiness.get("new_mcmc_performed") is not False
            or not isinstance(readiness.get("gates"), dict)
            or set(readiness["gates"]) != expected_gates
            or any(value is not True for value in readiness["gates"].values())):
        raise ValueError("saved-control readiness did not pass every gate of this exact protocol")
    contract = smoke.get("run_contract", {})
    settings = contract.get("settings", {})
    work = smoke.get("performance", {}).get("work", {})
    selected = smoke.get("selected_update")
    if (smoke.get("format") != "full_surface_training_v1"
            or smoke.get("same_seed_reload_exact") is not True or smoke.get("scientific_validation") is not False
            or smoke.get("reserved_test_evaluated") is not False
            or contract.get("format") != "full_surface_update_budget_run_v1"
            or smoke.get("checkpoint_selection") != "endpoint" or settings.get("checkpoint_selection") != "endpoint"
            or type(selected) is not int or selected <= 0
            or type(work.get("optimizer_updates")) is not int or work["optimizer_updates"] != selected
            or type(settings.get("updates")) is not int or settings["updates"] != selected
            or contract.get("dataset", {}).get("manifest_sha256") != protocol["bank"]["manifest_sha256"]
            or contract.get("dataset", {}).get("task_fingerprint") != protocol["bank"]["task_fingerprint"]
            or smoke.get("task_fingerprint") != protocol["bank"]["task_fingerprint"]
            or settings.get("batch_size") != protocol["fixed_training"]["batch_size"]
            or settings.get("steps_per_execution") != 1
            or work.get("available_training_rows") != protocol["bank"]["splits"]["train"]
            or type(work.get("example_presentations")) is not int or work["example_presentations"] <= 0
            or type(work.get("unique_rows_seen")) is not int or not 0 < work["unique_rows_seen"] <= work["example_presentations"]
            or work.get("reserved_test_evaluated") is not False):
        raise ValueError("exact-update same-bank training/reload smoke evidence is missing or incomplete")
    if admission.get("format") != "heston_training_budget_admission_v1":
        raise ValueError("unknown admission format")
    required = {"protocol_sha256": digest(protocol_path), "readiness_sha256": digest(readiness_path),
                "smoke_training_sha256": digest(smoke_path), "source_revision": source_revision,
                "source_manifest_sha256": executing_sources_digest()}
    if any(admission.get(key) != value for key, value in required.items()):
        raise ValueError("admission evidence/source identity changed")
    runtime = runtime_identity()
    if (admission.get("runtime") != runtime or smoke.get("performance", {}).get("runtime") != runtime
            or any(value is None for value in runtime.values())):
        raise ValueError("installed runtime differs from the admitted exact-update smoke")
    devices = admission.get("execution_devices")
    if (not isinstance(devices, list) or not devices or any(not isinstance(value, str) or not value for value in devices)
            or devices != smoke.get("performance", {}).get("hardware", {}).get("devices")):
        raise ValueError("execution devices require a matching device-specific smoke before admission")
    ceiling = protocol["resources"]
    cpus = admission.get("host_cpus")
    gpus = admission.get("gpu_count")
    if (type(cpus) is not int or not 1 <= cpus <= ceiling["max_host_cpus"]
            or type(gpus) is not int or gpus not in (0, 1)):
        raise ValueError("invalid admitted resource counts")
    lanes = admission.get("lane_wall_seconds", {})
    if not isinstance(lanes, dict) or not lanes or not set(lanes) <= set(candidates(protocol)):
        raise ValueError("admission may only include frozen seed0 candidates")
    if any(not finite_number(seconds, positive=True) for seconds in lanes.values()):
        raise ValueError("positive finite full lane allowances are required")
    cleanup = admission.get("cleanup_seconds_per_lane")
    if not finite_number(cleanup, positive=True) or any(seconds <= cleanup for seconds in lanes.values()):
        raise ValueError("each lane allowance must include a positive finite cleanup reserve")
    memory = admission.get("memory", {})
    host = memory.get("host_allocation_bytes")
    forecast = memory.get("forecast_peak_host_bytes")
    measured = smoke.get("performance", {}).get("peak_process_rss_bytes")
    if (not finite_number(host, positive=True) or host > ceiling["memory_gib"] * 2**30
            or not finite_number(forecast, positive=True) or not finite_number(measured, positive=True)
            or not measured <= forecast <= host):
        raise ValueError("host memory forecast must cover the measured smoke peak within the allocation/ceiling")
    device = memory.get("forecast_peak_device_bytes")
    available = memory.get("available_device_bytes")
    if (not finite_number(device) or not finite_number(available)
            or (gpus == 1 and not 0 < device <= available)
            or (gpus == 0 and (device != 0 or available != 0))):
        raise ValueError("finite device memory forecast/allocation must fit the declared GPU/CPU mode")
    if gpus == 1:
        measured_device = smoke.get("performance", {}).get("hardware", {}).get("device_memory")
        if not isinstance(measured_device, list) or not measured_device:
            raise ValueError("GPU admission requires measured smoke device peak memory")
        peaks = [entry.get("stats", {}).get("peak_bytes_in_use") for entry in measured_device
                 if isinstance(entry, dict) and isinstance(entry.get("stats"), dict)]
        if (not peaks or any(not finite_number(peak, positive=True) for peak in peaks)
                or max(peaks) > device):
            raise ValueError("device-memory forecast must cover every measured GPU smoke peak")
    # Reserve readiness, failed attempts and final confirmation before the screen.
    prior = admission.get("already_used", {})
    final = admission.get("reserved_confirmation", {})
    if any(not finite_number(d.get(k), positive=(d is final and (k == "allocated_cpu_seconds" or gpus == 1)))
           for d in (prior, final) for k in ("allocated_cpu_seconds", "gpu_seconds")):
        raise ValueError("finite prior usage, positive final CPU reserve and a positive final GPU reserve when used are required")
    total_seconds = sum(lanes.values())
    cpu = prior["allocated_cpu_seconds"] + final["allocated_cpu_seconds"] + cpus * total_seconds
    gpu = prior["gpu_seconds"] + final["gpu_seconds"] + gpus * total_seconds
    if (not math.isfinite(cpu) or not math.isfinite(gpu)
            or cpu > ceiling["allocated_cpu_hours_total"] * 3600
            or gpu > ceiling["gpu_hours_total"] * 3600):
        raise ValueError("forecast including final-confirmation reserve exceeds the total resource ceiling")
    if admission.get("approved_for_execution") is not True:
        raise ValueError("cost forecast has not been explicitly admitted")
    return protocol, admission


def verify_execution_resources(admission):
    """Verify allocation/affinity and visible devices; BLAS threads are not a CPU binding."""
    cpus = admission["host_cpus"]
    allocated = os.environ.get("SLURM_CPUS_PER_TASK", "")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if allocated.isdecimal():
        if int(allocated) != cpus:
            raise ValueError("Slurm CPU allocation differs from admission")
        cpu_binding = {"source": "SLURM_CPUS_PER_TASK", "allocated_cpus": int(allocated), "affinity": affinity}
    elif affinity is not None and 0 < len(affinity) <= cpus:
        cpu_binding = {"source": "Linux process affinity", "allocated_cpus": cpus, "affinity": affinity}
    else:
        raise ValueError("admitted study execution requires a matching Slurm CPU allocation or bounded Linux CPU affinity; thread variables alone do not bind CPUs")
    if admission["gpu_count"] == 0:
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    devices = jax.devices()
    if [str(device) for device in devices] != admission["execution_devices"]:
        raise ValueError("visible execution devices differ from the admitted smoke")
    if sum(device.platform != "cpu" for device in devices) != admission["gpu_count"]:
        raise ValueError("visible GPU count differs from admission")
    requested_host = admission["memory"]["host_allocation_bytes"]
    per_node = os.environ.get("SLURM_MEM_PER_NODE", "")
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "")
    allocated_host = (int(per_node) * 2**20 if per_node.isdecimal() else
                      int(per_cpu) * cpus * 2**20 if per_cpu.isdecimal() else None)
    if allocated_host is not None and allocated_host < requested_host:
        raise ValueError("Slurm memory allocation is smaller than the admitted host-memory request")
    # This is an admission forecast and observed allocation, not an OS memory
    # limit imposed by Python. Slurm/cgroups or the caller own hard limits.
    return {"cpu_binding": cpu_binding, "blas_thread_values_are_not_cpu_binding": True,
            "host_memory_allocation_bytes": allocated_host,
            "python_enforces_memory_limit": False, "devices": admission["execution_devices"]}


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
    started = time.perf_counter()
    # Source identity must bind a clean immutable tracked checkout. Downloads and
    # lane outputs belong outside it; no switching code underneath active jobs.
    if not clean_source():
        raise ValueError("budget lanes require a clean tracked source checkout without untracked source files")
    protocol, admission = validate_admission(args.protocol, args.readiness, args.smoke, args.admission, source_revision())
    if digest(args.bank / "manifest.json") != protocol["bank"]["manifest_sha256"]:
        raise ValueError("bank manifest differs from the frozen study")
    resource_binding = verify_execution_resources(admission)
    source_identity = executing_sources_digest()
    revision = source_revision()
    candidate = candidates(protocol).get(args.candidate)
    if candidate is None:
        raise ValueError("candidate is not in the frozen seed0 screen")
    lane, seconds = reserve_lane(args.out, args.candidate, args.admission, admission)
    journal = {"format": "heston_training_budget_lane_v1", "candidate": candidate,
               "status": "starting", "source_revision": admission["source_revision"],
               "protocol_sha256": digest(args.protocol), "admission_sha256": digest(args.admission),
               "reserved_wall_seconds": seconds, "automatic_retry": False,
               "source_manifest_sha256": source_identity, "resource_binding": resource_binding,
               "reserved_test_evaluated": False, "command": command(candidate, args.bank, lane / "checkpoint")}
    write_json(lane / "lane.json", journal)
    environment = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "OMP_NUM_THREADS": str(admission["host_cpus"]),
                   "OPENBLAS_NUM_THREADS": str(admission["host_cpus"]),
                   "MKL_NUM_THREADS": str(admission["host_cpus"]),
                   "NUMEXPR_NUM_THREADS": str(admission["host_cpus"])}
    try:
        remaining = seconds - admission["cleanup_seconds_per_lane"] - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("lane preparation exhausted the admitted allowance before training")
        with (lane / "training.log").open("x") as stream:
            result = subprocess.run(journal["command"], cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=remaining, check=False)
        journal.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
    except BaseException as error:
        journal.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        intact = clean_source() and source_revision() == revision and executing_sources_digest() == source_identity
        journal["executing_source_unchanged"] = intact
        if not intact:
            journal.update(status="failed", source_failure="executing source changed during the lane; endpoint cannot be admitted")
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

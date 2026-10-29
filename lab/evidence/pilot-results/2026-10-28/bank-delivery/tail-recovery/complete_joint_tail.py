"""One finite metadata bridge: checked native tail -> unchanged full finalizer.

No financial simulation, historical seed scan, retry, training, or acceptance
promotion. The generic campaign and this one finalizer share the original
1,500 allocated CPU-hour ceiling. Preserve every prior receipt.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING",
          "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING", "STAGE_OUT"}
WALL, CLEANUP, CEILING = 3600, 240, 1500 * 3600


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); temporary.replace(path)


def frozen(plan):
    for path, digest in plan["frozen_sha256"].items():
        if sha(path) != digest: raise ValueError("Frozen bridge input changed: " + path)


def task_ids(attempt):
    if attempt["role"] == "workers":
        return [attempt["job_id"] + "_" + str(i - attempt["array_offset"]) for i in attempt["indices"]]
    return [attempt["job_id"]]


def accounting(attempts):
    ids = ",".join(a["job_id"] for a in attempts)
    result = subprocess.run(["sacct", "-S", "2026-10-27", "-j", ids, "-n", "-P",
        "--format=JobID%64,State%40,ElapsedRaw,AllocCPUS,ExitCode"],
        capture_output=True, text=True, timeout=45, check=True)
    rows = {}
    for line in result.stdout.splitlines():
        x = line.split("|")
        if len(x) >= 5 and x[0] and "." not in x[0]:
            rows[x[0]] = {"job_id": x[0], "state": x[1].split()[0], "elapsed": int(x[2]),
                          "cpus": int(x[3]), "exit_code": x[4]}
    return rows


def query_or_wait(query, ledger):
    # Read-only accounting may be delayed. A submission is never wrapped here:
    # its timeout remains ambiguous and stops the bridge without retry.
    try:
        return query()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        ledger.update(last_accounting_error={"type": type(error).__name__, "message": str(error), "utc": now()})
        return None


def settled_cost(attempts, rows):
    if not attempts or any(a.get("state") != "submitted" for a in attempts):
        raise ValueError("Missing or ambiguous allocation receipt")
    ids = [i for a in attempts for i in task_ids(a)]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate allocation identities")
    total = 0
    for identifier in ids:
        row = rows.get(identifier)
        if row is None or row["state"] in ACTIVE: return None
        if row["state"] != "COMPLETED" or row["exit_code"] != "0:0" or row["cpus"] != 1:
            raise ValueError("Allocation did not complete successfully: " + identifier)
        total += row["elapsed"]
    return total


def generic_ready(plan, rows=None):
    path = Path(plan["recovery_execution"])
    if not path.exists(): return None
    run = read(path)
    if run["state"].startswith("stopped_"): raise ValueError("Native recovery stopped: " + run["state"])
    if run["state"] != "completed_checked_snapshot": return None
    config, jobs = read(plan["recovery_config"]), read(plan["recovery_jobs"])
    run_config = {k: v for k, v in run["config"].items() if k != "expected_operator_sha256"}
    if (run_config != config or config["cpu_hours"] != 1500
            or config["prior_cpu_seconds"] != plan["prior_cpu_seconds"]
            or jobs["jobs"] != plan["native_jobs"] or run["planned_workers"] != 2
            or sum(len(a.get("indices", [])) for a in run["attempts"]) != 2
            or len(run["attempts"]) != 2
            or {a["role"] for a in run["attempts"]} != {"workers", "final"}):
        raise ValueError("Native campaign recipe or exact two-job map changed")
    rows = accounting(run["attempts"]) if rows is None else rows
    cost = settled_cost(run["attempts"], rows)
    if cost is None: return None
    manifest_path = Path(config["snapshot"]) / "manifest.json"; manifest = read(manifest_path)
    phase = read(plan["stage_plan"])["phases"]["joint"]
    if (run.get("snapshot") != config["snapshot"] or sha(manifest_path) != run.get("snapshot_manifest_sha256")
            or manifest["collection_id"] != phase["collection_id"]
            or manifest["configuration"] != jobs["configuration"]
            or manifest["configuration"]["root_seed"] != phase["root_seed"]
            or manifest["configuration"]["splits"] != phase["splits"]):
        raise ValueError("Native snapshot differs from frozen task/coverage")
    if config["prior_cpu_seconds"] + cost + WALL + CLEANUP > CEILING:
        raise ValueError("Final integrity exposure exceeds original cumulative ceiling")
    return {"run": run, "cost": cost, "manifest": manifest,
            "execution_sha256": sha(path), "snapshot_sha256": sha(manifest_path)}


def submit_integrity(plan, ledger, root, *, run=subprocess.run):
    if ledger.get("integrity_attempt"): raise ValueError("Finalizer was already attempted; no retry")
    phase = read(plan["stage_plan"])["phases"]["joint"]
    for destination in (Path(phase["snapshot"]), Path(phase["root"]) / "final.json", Path(plan["readiness"])):
        if destination.exists(): raise FileExistsError("Refuse existing final output: " + str(destination))
    command = ["env", "PYTHONPATH=" + phase["repo"] + "/src", "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", plan["python"],
        plan["tested_finalizer"], "final", "--plan", plan["stage_plan"], "--phase", "joint"]
    argv = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=8G",
        "--no-requeue", "--account=carney-mjfrank-condo2", "--partition=batch", "--time=01:00:00",
        "--exclude=node2338,node2339,node2336,node2344,node2346", "--job-name=sbi4f-joint-tail-integrity",
        "--chdir=" + phase["repo"], "--output=" + str(root / "integrity-%A.out"), "--wrap=" + shlex.join(command)]
    attempt = {"role": "final", "state": "submission_attempted", "argv": argv, "attempted_utc": now()}
    ledger["integrity_attempt"] = attempt; save(root / "execution.json", ledger)
    try:
        response = run(argv, capture_output=True, text=True, timeout=45)
        attempt.update(stdout=response.stdout, stderr=response.stderr, returncode=response.returncode)
        identifier = response.stdout.strip().split(";")[0]
        if response.returncode or not identifier.isdigit(): raise RuntimeError("Ambiguous finalizer submission")
        attempt.update(state="submitted", job_id=identifier)
    except BaseException:
        attempt["state"] = "unknown_submission"; save(root / "execution.json", ledger); raise
    save(root / "execution.json", ledger)


def completion_report(plan, generic, attempt, rows, started):
    extra = settled_cost([attempt], rows)
    if extra is None: return None
    phase = read(plan["stage_plan"])["phases"]["joint"]
    report = read(Path(phase["root"]) / "final.json")
    original_path = Path(phase["snapshot"]) / "manifest.json"; original = read(original_path)
    if not _same_checked_snapshot(plan, generic, report, original):
        raise ValueError("Full integrity result/snapshot does not match the reviewed native recovery")
    total = plan["prior_cpu_seconds"] + generic["cost"] + extra
    if total > CEILING: raise ValueError("Actual cumulative allocation exceeds original ceiling")
    report.update(recovery_execution=plan["recovery_execution"], recovery_execution_sha256=generic["execution_sha256"],
        recovery_jobs=plan["recovery_jobs"], recovery_jobs_sha256=sha(plan["recovery_jobs"]),
        recovery_config_sha256=sha(plan["recovery_config"]), recovery_snapshot_manifest_sha256=generic["snapshot_sha256"],
        original_execution_sha256=plan["original_execution_sha256"],
        prior_recovery_execution_sha256=plan["prior_recovery_execution_sha256"],
        preserved_inventory_sha256=plan["preserved_inventory_sha256"],
        original_job_map_sha256=plan["original_job_map_sha256"],
        reviewed_failed_allocations=["8046699_997", "8046699_1000"],
        integrity_attempts=[{k: attempt[k] for k in ("role", "state", "job_id")}], integrity_started_utc=started,
        integrity_execution=str(Path(plan["root"]) / "execution.json"), integrity_accounting=list(rows.values()),
        cumulative_allocated_cpu_seconds=total, bridge_plan_sha256=plan["plan_sha256"],
        scientific_acceptance=False, reserved_test_evaluated=False)
    return report


def _same_checked_snapshot(plan, generic, report, original):
    """Snapshot directories differ; resolve their immutable references."""
    phase = read(plan["stage_plan"])["phases"]["joint"]
    expected = {"status": "completed_checked_snapshot", "plan_sha256": sha(plan["stage_plan"]),
        "driver_sha256": plan["tested_finalizer_sha256"], "preserved_parts_unchanged": 1278, "verified_parts": 1280,
        "exact_prior_context_noise_replay": True, "snapshot": phase["snapshot"],
        "snapshot_manifest_sha256": sha(Path(phase["snapshot"]) / "manifest.json"), "split_counts": phase["splits"],
        "collection_id": phase["collection_id"], "source_revision": phase["revision"],
        "source_sha256": original["source_sha256"], "runtime": original["versions"], "task_fingerprint": original["task_fingerprint"]}
    if any(report.get(k) != v for k, v in expected.items()): return False
    native = generic["manifest"]
    for name in ("configuration", "collection_id", "source_sha256", "versions", "task_fingerprint"):
        if original[name] != native[name]: return False
    def references(manifest, directory):
        return [{**p, "path": str((directory / p["path"]).resolve())} for p in manifest["parts"]]
    return references(original, Path(phase["snapshot"])) == references(native, Path(read(plan["recovery_config"])["snapshot"]))


def publish(plan, report):
    # Each destination is new. Hard links publish complete bytes atomically
    # without replacing a failed, historical, or concurrent readiness record.
    root = Path(plan["root"]); completed = root / "completion.json"; temporary = root / "completion.tmp"
    with temporary.open("x") as f: f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.link(temporary, completed); temporary.unlink()
    destination = Path(plan["readiness"])
    os.link(completed, destination)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--plan", type=Path, required=True); a = p.parse_args()
    plan = read(a.plan); plan["plan_sha256"] = sha(a.plan); root = Path(plan["root"])
    lock = (root / "bridge.lock").open("a+"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "execution.json").exists(): raise FileExistsError("Reconcile prior bridge; no automatic restart")
    ledger = {"state": "waiting_for_native_recovery", "started_utc": now(), "pid": os.getpid(),
        "host": socket.gethostname(), "plan_sha256": plan["plan_sha256"], "automatic_retry": False}
    save(root / "execution.json", ledger); started = time.monotonic()
    try:
        while time.monotonic() - started < 14 * 3600:
            if sha(a.plan) != plan["plan_sha256"]: raise ValueError("Frozen bridge plan changed")
            frozen(plan)
            generic = query_or_wait(lambda: generic_ready(plan), ledger)
            if generic is not None:
                if not ledger.get("integrity_attempt"):
                    ledger["recovery_execution_sha256"] = generic["execution_sha256"]
                    submit_integrity(plan, ledger, root)
                if ledger["recovery_execution_sha256"] != generic["execution_sha256"]:
                    raise ValueError("Completed native recovery ledger changed")
                attempt = ledger["integrity_attempt"]
                rows = query_or_wait(lambda: accounting([attempt]), ledger)
                report = (None if rows is None else completion_report(plan, generic, attempt, rows, ledger["started_utc"]))
                if report is not None:
                    ledger.update(state="completed_checked_snapshot", completed_utc=now(),
                        snapshot=report["snapshot"], snapshot_manifest_sha256=report["snapshot_manifest_sha256"],
                        cumulative_allocated_cpu_seconds=report["cumulative_allocated_cpu_seconds"])
                    save(root / "execution.json", ledger); publish(plan, report); return
                ledger["state"] = "waiting_for_final_integrity"
            ledger["last_poll_utc"] = now(); save(root / "execution.json", ledger); time.sleep(60)
        raise TimeoutError("Finite bridge reached14h wall limit")
    except BaseException as error:
        ledger.update(state="stopped_for_review", error_type=type(error).__name__, error=str(error), stopped_utc=now())
        save(root / "execution.json", ledger); raise


if __name__ == "__main__": main()

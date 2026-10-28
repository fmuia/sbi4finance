"""Arm once-only bank extensions after checked predecessor snapshots settle.

This small waiter submits no Slurm jobs itself. It starts the bounded generic
campaign launcher only after the declared dependencies pass.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    return json.loads(Path(path).read_text())  # Actual frozen campaign paths are absolute.


def task_ids(attempt):
    return ([attempt["job_id"] + "_" + str(i-attempt["array_offset"]) for i in attempt["indices"]]
            if attempt["role"] == "workers" else [attempt["job_id"]])


def scheduler_from_launcher(plan):
    spec = importlib.util.spec_from_file_location("frozen_bank_launcher", plan["launcher"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Slurm()


def now():
    return datetime.now(timezone.utc).isoformat()


def original_task_ids(j):
    return [j["job_id"] + "_" + str(i) for i in j["indices"]] if j["role"] == "workers" else [j["job_id"]]


def settled(jobs, started, scheduler, *, original=False):
    if not jobs or any(j.get("state") != "submitted" or not j.get("job_id") for j in jobs):
        raise ValueError("missing or ambiguous predecessor allocation receipts")
    try:
        rows = scheduler.accounting(jobs, started)
    except Exception as error:
        return False, "waiting_for_accounting:" + type(error).__name__
    identifiers = [i for j in jobs for i in (original_task_ids(j) if original else task_ids(j))]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate predecessor allocation identities")
    for i in identifiers:
        row = rows.get(i)
        if not row:
            return False, "waiting_for_accounting:" + i
        if row["state"] != "COMPLETED" or row["exit_code"] != "0:0":
            return False, "predecessor_not_successfully_settled:" + i + ":" + row["state"]
    return True, "all_predecessor_allocations_completed"


def check_snapshot(route, report):
    p = Path(route["dependency_snapshot"]) / "manifest.json"
    if not p.exists() or sha(p) != report["snapshot_manifest_sha256"]:
        raise ValueError("checked predecessor snapshot missing or changed")
    manifest = json.loads(p.read_text())
    collection = json.loads((Path(route["collection"]) / "collection.json").read_text())
    if (sha(Path(route["collection"]) / "collection.json") != route["collection_sha256"]
            or report["collection_id"] != route["collection_id"]
            or manifest["collection_id"] != route["collection_id"]
            or manifest["configuration"]["splits"] != route["dependency_splits"]
            or manifest["configuration"]["root_seed"] != route["root_seed"]
            or manifest["configuration"]["task"] != collection["configuration"]["task"]
            or report["split_counts"] != route["dependency_splits"]
            or report["source_revision"] != route["producer_revision"]
            or report["source_sha256"] != collection["source_sha256"]
            or report["runtime"] != collection["versions"]
            or report["task_fingerprint"] != collection["task_fingerprint"]):
        raise ValueError("predecessor task/root/source/runtime/count identity differs")
    return sha(p)


def external_campaign_ready(dependency, scheduler):
    """A joint-only waiter can observe the already-launched known campaign."""
    c = read_config(dependency["config"])
    path = Path(c["campaign_root"]) / "execution.json"
    if not path.exists():
        return False, "waiting_for_resource_campaign"
    run = json.loads(path.read_text())
    if run["state"].startswith("stopped_"):
        raise ValueError("resource_campaign_failed:" + run["state"])
    if run["state"] != "completed_checked_snapshot":
        return False, "waiting_for_resource_campaign:" + run["state"]
    p = Path(c["snapshot"]) / "manifest.json"
    if sha(p) != run["snapshot_manifest_sha256"]:
        raise ValueError("resource campaign snapshot changed")
    manifest = json.loads(p.read_text())
    if (run["collection_id"] != dependency["collection_id"]
            or manifest["collection_id"] != dependency["collection_id"]
            or manifest["configuration"]["splits"] != dependency["splits"]):
        raise ValueError("resource campaign identity differs")
    return settled(run["attempts"], run["started_utc"], scheduler)


def readiness(plan, name, states, scheduler):
    route = plan["routes"][name]
    if route.get("after_campaign"):
        ready, reason = external_campaign_ready(route["after_campaign"], scheduler)
        if not ready:
            return False, reason
    if route.get("after_route") and states[route["after_route"]]["state"] != "completed_checked_snapshot":
        prior = states[route["after_route"]]["state"]
        if prior in ("blocked_dependency", "blocked_campaign", "unknown_launch"):
            raise ValueError("resource_dependency_failed:" + route["after_route"] + ":" + prior)
        return False, "waiting_for_" + route["after_route"] + "_resource_dependency:" + prior
    final = Path(route["dependency_final"])
    if not final.exists():
        return False, "waiting_for_predecessor_final_check"
    report = json.loads(final.read_text())
    if report.get("status") != "completed_checked_snapshot":
        if report.get("status") == "running":
            return False, "predecessor_final_check_running"
        raise ValueError("predecessor final check failed or is incompatible")
    manifest_sha = check_snapshot(route, report)
    execution_path = Path(route.get("dependency_execution", plan["original_execution"]))
    original = json.loads(execution_path.read_text())
    if route.get("reviewed_recovery"):
        binding = route["reviewed_recovery"]
        if (report.get("recovery_execution") != str(execution_path)
                or report.get("recovery_execution_sha256") != sha(execution_path)
                or report.get("original_execution_sha256") != binding["original_execution_sha256"]
                or sha(plan["original_execution"]) != binding["original_execution_sha256"]
                or report.get("preserved_inventory_sha256") != binding["preserved_inventory_sha256"]
                or report.get("preserved_parts_unchanged") != binding["preserved_parts"]
                or report.get("original_job_map_sha256") != binding["original_job_map_sha256"]
                or report.get("reviewed_failed_indices") != binding["reviewed_failed_indices"]
                or report.get("new_worker_indices") != binding["new_worker_indices"]
                or report.get("plan_sha256") != sha(route["recovery_plan"])):
            raise ValueError("reviewed recovery completion does not match frozen provenance/coverage")
    phase = original["phases"][name]
    if phase.get("status") != "completed_checked_snapshot":
        return False, "waiting_for_predecessor_completion_ledger"
    if (phase.get("snapshot_manifest_sha256") != manifest_sha
            or phase.get("snapshot") != route["dependency_snapshot"]):
        raise ValueError("predecessor completion ledger is not bound to the checked snapshot")
    indices = [i for j in phase["jobs"] if j["role"] == "workers" for i in j["indices"]]
    if len(indices) != len(set(indices)) or len(indices) != route["expected_original_workers"]:
        raise ValueError("predecessor worker coverage or duplicate identity differs")
    if route.get("reviewed_recovery") and sorted(indices) != route["reviewed_recovery"]["new_worker_indices"]:
        raise ValueError("new recovery workers do not match the reviewed missing identities")
    return settled(phase["jobs"], original.get("started_utc", original.get("prepared_utc", "2026-10-28")),
                   scheduler, original=True)


def launch(plan, name, ledger, root, *, run=subprocess.run):
    route = plan["routes"][name]
    state = ledger["routes"][name]
    if state.get("attempt"):
        raise RuntimeError("extension already attempted; reconcile rather than retry")
    command = [plan["python"], plan["launcher"], "--config", route["campaign_config"], "--detach"]
    state.update(state="launch_attempted", attempt={"utc": now(), "command": command})
    save(root / "execution.json", ledger)  # Before any process which might submit work.
    try:
        result = run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError("campaign launch failed: " + result.stderr.strip())
        receipt_path = Path(read_config(route["campaign_config"])["campaign_root"]) / "launch.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("state") != "launched_not_yet_checked" or receipt["operator_sha256"] != sha(plan["launcher"]):
            raise ValueError("missing or incompatible detached campaign receipt")
        state.update(state="launched", campaign_pid=receipt["pid"], launch_receipt=str(receipt_path),
                     launch_receipt_sha256=sha(receipt_path), launched_utc=now())
    except BaseException as error:
        state.update(state="unknown_launch", error_type=type(error).__name__, error=str(error))
        save(root / "execution.json", ledger)
        raise
    save(root / "execution.json", ledger)


def update_campaign(plan, name, ledger, scheduler):
    state = ledger["routes"][name]
    c = read_config(plan["routes"][name]["campaign_config"])
    path = Path(c["campaign_root"]) / "execution.json"
    if not path.exists():
        state["progress"] = "waiting_for_detached_campaign_preparation"
        return
    run = json.loads(path.read_text())
    state["progress"] = run["state"]
    state["campaign_completed_workers"] = run.get("completed_workers", 0)
    state["campaign_planned_workers"] = run.get("planned_workers")
    if run["state"] == "completed_checked_snapshot":
        ready, why = settled(run["attempts"], run["started_utc"], scheduler)
        if ready:
            manifest = Path(c["snapshot"]) / "manifest.json"
            if sha(manifest) != run["snapshot_manifest_sha256"]:
                raise ValueError("extension snapshot changed")
            state.update(state="completed_checked_snapshot", snapshot=c["snapshot"],
                         snapshot_manifest_sha256=sha(manifest), completed_utc=now())
        else:
            state["progress"] = why
    elif run["state"].startswith("stopped_"):
        state.update(state="blocked_campaign", error=run.get("error", run["state"]))


def run_waiter(plan_path, *, scheduler=None, runner=subprocess.run, sleep=time.sleep, prepared_root=False):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    root = Path(plan["root"])
    if not prepared_root:
        root.mkdir(parents=True, exist_ok=False)
    path = root / "execution.json"
    if path.exists():
        raise FileExistsError("existing follow-on waiter must be reconciled, not relaunched")
    scheduler = scheduler or scheduler_from_launcher(plan)
    frozen = {str(p): sha(p) for p in [plan_path, Path(__file__), Path(plan["launcher"]),
                                      *[Path(r["campaign_config"]) for r in plan["routes"].values()]]}
    for route in plan["routes"].values():
        if route.get("recovery_plan"):
            frozen[route["recovery_plan"]] = sha(route["recovery_plan"])
        if route.get("after_campaign"):
            p = route["after_campaign"]["config"]
            frozen[p] = sha(p)
    ledger = {"format": "bank_extension_waiter_v1", "state": "armed", "pid": os.getpid(),
              "started_utc": now(), "frozen_sha256": frozen,
              "routes": {n: {"state": "waiting"} for n in plan["routes"]},
              "scientific_acceptance": False}
    save(path, ledger)
    started = time.monotonic()
    lock = (root / "waiter.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.monotonic() - started < plan["controller_wall_hours"] * 3600:
            if any(sha(p) != digest for p, digest in frozen.items()):
                raise ValueError("frozen waiter/config source changed")
            for name, state in ledger["routes"].items():
                if state["state"] == "waiting":
                    try:
                        ready, reason = readiness(plan, name, ledger["routes"], scheduler)
                        state["progress"] = reason
                        if ready:
                            launch(plan, name, ledger, root, run=runner)
                    except BaseException as error:
                        if state["state"] != "unknown_launch":
                            state.update(state="blocked_dependency", error_type=type(error).__name__, error=str(error))
                elif state["state"] == "launched":
                    try:
                        update_campaign(plan, name, ledger, scheduler)
                    except BaseException as error:
                        state.update(state="blocked_campaign", error_type=type(error).__name__, error=str(error))
            ledger["last_poll_utc"] = now()
            if all(s["state"] == "completed_checked_snapshot" for s in ledger["routes"].values()):
                ledger["state"] = "completed_all_extensions"
                break
            if all(s["state"] in ("blocked_dependency", "blocked_campaign", "unknown_launch", "completed_checked_snapshot")
                   for s in ledger["routes"].values()):
                ledger["state"] = "stopped_for_review"
                break
            save(path, ledger)
            sleep(plan["poll_seconds"])
        else:
            ledger["state"] = "stopped_at_waiter_wall_limit"
        save(path, ledger)
    except BaseException as error:
        ledger.update(state="stopped_for_review", error_type=type(error).__name__, error=str(error))
        save(path, ledger)
        raise
    finally:
        lock.close()
    return ledger


def detach(plan_path, *, popen=subprocess.Popen):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "plan.json", plan)
    save(root / "launch-attempt.json", {"state": "launch_attempted", "utc": now(),
                                       "plan_sha256": sha(root / "plan.json"), "waiter_sha256": sha(__file__)})
    with (root / "waiter.log").open("ab") as log:
        child = popen([plan["python"], str(Path(__file__).resolve()), "--plan", str(root / "plan.json"), "--run"],
                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"state": "armed_process_launched", "pid": child.pid, "utc": now(),
               "plan_sha256": sha(root / "plan.json"), "waiter_sha256": sha(__file__),
               "execution": str(root / "execution.json"), "log": str(root / "waiter.log")}
    save(root / "launch.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.detach and a.run:
        p.error("choose launch or child")
    result = detach(a.plan) if a.detach else run_waiter(a.plan, prepared_root=a.run)
    print(json.dumps({k: result[k] for k in ("state", "pid") if k in result}))


if __name__ == "__main__":
    main()

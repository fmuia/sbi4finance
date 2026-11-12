"""One bounded Heston fit followed by its fixed development evaluation.

The admission runner owns resource reservations. This worker owns a single
lane's child processes, logs and durable stage journal. Failure in this lane
does not submit, cancel or retry any other lane.
"""

import argparse
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_heston_budget_study as study  # noqa: E402


def commands(args, candidate):
    return [("training", study.command(candidate, args.bank, args.out / "checkpoint")),
            ("evaluation", [sys.executable, str(ROOT / "scripts/evaluate_heston_budget_candidate.py"),
                "--protocol", str(args.protocol.resolve()), "--bank", str(args.bank.resolve()),
                "--checkpoint", str((args.out / "checkpoint").resolve()),
                "--out", str((args.out / "evaluation").resolve()), "--candidate", args.candidate,
                "--control-readiness", str(args.control_readiness.resolve()), "--threads", "1"])]


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def run(args):
    started = time.perf_counter()
    if (type(args.wall_seconds) not in (int, float) or not math.isfinite(args.wall_seconds)
            or args.wall_seconds <= 2):
        raise ValueError("a finite lane allowance greater than the two-second cleanup reserve is required")
    deadline = started + args.wall_seconds
    protocol = study.read_json(args.protocol)
    candidate = study.candidates(protocol).get(args.candidate)
    if candidate is None:
        raise ValueError("candidate must belong to the frozen seed-0 matrix")
    if study.digest(args.bank / "manifest.json") != protocol["bank"]["manifest_sha256"]:
        raise ValueError("bank differs from the frozen matrix")
    control = study.read_json(args.control_readiness)
    from scripts.evaluate_heston_budget_candidate import validate_control
    validate_control(control, protocol, study.digest(args.protocol))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "matrix-worker.json"
    if path.exists() or (args.out / "checkpoint").exists() or (args.out / "evaluation").exists():
        raise ValueError("matrix worker requires a fresh lane; no retry or overwrite")
    journal = {"format": "heston_matrix_worker_v1", "status": "preparing", "candidate": candidate,
        "protocol_sha256": study.digest(args.protocol), "bank_manifest_sha256": study.digest(args.bank / "manifest.json"),
        "control_readiness_sha256": study.digest(args.control_readiness), "control_status_retained": control["status"],
        "wall_allowance_seconds": args.wall_seconds, "cleanup_reserve_seconds": 2,
        "automatic_retry": False, "reserved_test_evaluated": False, "model_publication_allowed": False,
        "scientific_acceptance": False,
        "stages": [{"name": name, "command": command, "status": "pending"}
                   for name, command in commands(args, candidate)]}
    study.write_json(path, journal)
    initial_child_cpu = child_cpu()
    try:
        for stage in journal["stages"]:
            remaining = deadline - time.perf_counter() - journal["cleanup_reserve_seconds"]
            if remaining <= 0:
                raise TimeoutError("lane allowance exhausted before the next stage; no automatic continuation")
            stage.update(status="running", timeout_seconds=remaining)
            journal["status"] = stage["name"]
            study.write_json(path, journal)
            before, before_cpu = time.perf_counter(), child_cpu()
            try:
                with (args.out / f"{stage['name']}-stage.log").open("x") as log:
                    result = subprocess.run(stage["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=remaining, check=False)
                stage.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
                if result.returncode:
                    raise RuntimeError(f"{stage['name']} execution failed; preserve its checkpoint, draws and log")
            except BaseException as error:
                stage.update(status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                             error_type=type(error).__name__, error=str(error))
                raise
            finally:
                stage["wall_seconds"] = time.perf_counter() - before
                stage["child_process_cpu_seconds"] = child_cpu() - before_cpu
                study.write_json(path, journal)
        evaluation = study.read_json(args.out / "evaluation/evaluation.json")
        if (evaluation.get("execution_status") != "completed"
                or evaluation.get("reserved_test_evaluated") is not False
                or evaluation.get("model_publication_allowed") is not False):
            raise ValueError("evaluation did not retain the required development-only scope")
        journal.update(status="completed", development_checks_status=evaluation["status"],
                       evaluation_sha256=study.digest(args.out / "evaluation/evaluation.json"))
        return journal
    except BaseException as error:
        journal.update(status="failed_execution", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        journal["wall_seconds"] = time.perf_counter() - started
        journal["child_process_cpu_seconds"] = child_cpu() - initial_child_cpu
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        journal["child_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        journal["cost_scope"] = "Worker wall includes child startup/imports and both stages; child process CPU includes terminated child processes/threads. Scheduler allocation is recorded by the outer lane runner."
        study.write_json(path, journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=study.DEFAULT_PROTOCOL)
    for key in ("bank", "out", "control-readiness"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "candidate": args.candidate,
                      "development_checks_status": result["development_checks_status"],
                      "reserved_test_evaluated": False, "model_publication_allowed": False}))


if __name__ == "__main__":
    main()

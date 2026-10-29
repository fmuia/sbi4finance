"""One-off bounded post-bank baseline phase; no service and no test evaluation.

Initialize once against a clean frozen checkout, then run ``watch`` under nohup.
Only a generation controller's checked final.json can release its route's job.
Unknown submissions and failures are retained and never retried automatically.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROUTES = {
    "known": {"bank": "/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/snapshot-50k",
              "ready": "/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/known/final.json",
              "collection_id": "da963f7294aee242a80ebf9b47d914294b439aef8b6ee8f7c6d1899766e546f1",
              "producer_revision": "616352035be4b9d271151090aa5d2d9539da8b1c", "root_seed": 271300,
              "task_id": "rough_calibration_v1", "splits": {"train": 50000, "validation": 5000, "test": 7000}},
    "joint": {"bank": "/oscar/scratch/afengler/sbi4finance-native-bank-20261027/snapshot-20k",
              "ready": "/oscar/scratch/afengler/sbi4finance-bank-delivery-20261028/joint/final.json",
              "collection_id": "817de3664968de0c605ae4c73e2c99c26465344294af2fc5412f39fffa4baa50",
              "producer_revision": "c773ceb26e7273612b15c69fd63715d131778083", "root_seed": 271200,
              "task_id": "rough_joint_calibration_v1", "splits": {"train": 20000, "validation": 5000, "test": 7000}},
}
RUNTIME = {"bayesflow": "2.1.0", "keras": "3.15.1", "jax": "0.11.1", "numpy": "2.5.3", "scipy": "1.18.1"}
WALL, CLEANUP, CPUS = 3600, 240, 4
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); temporary.replace(path)


def source_check(plan):
    repo = Path(plan["repo"])
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != plan["source_revision"] or dirty: raise ValueError("Frozen training checkout changed")
    for path, digest in plan["source_sha256"].items():
        if sha(path) != digest: raise ValueError("Frozen phase source changed: " + path)


def readiness(route):
    final = Path(route["ready"])
    if not final.exists(): return None
    record = read(final)
    # The finalizer writes an atomic running receipt before its integrity
    # scan. This is ordinary waiting, not a failed or checked snapshot.
    if record.get("status") == "running": return None
    if record.get("status") != "completed_checked_snapshot":
        raise ValueError("Bank controller has not published a successful checked snapshot")
    bank = Path(route["bank"]); manifest = read(bank / "manifest.json")
    digest = sha(bank / "manifest.json")
    config = manifest["configuration"]
    if (record.get("snapshot_manifest_sha256") != digest or record.get("snapshot") != str(bank)
            or record.get("split_counts") != route["splits"] or record.get("collection_id") != route["collection_id"]
            or record.get("source_revision") != route["producer_revision"]
            or record.get("source_sha256") != manifest.get("source_sha256")
            or record.get("task_fingerprint") != manifest.get("task_fingerprint")
            or manifest.get("collection_id") != route["collection_id"]
            or config["root_seed"] != route["root_seed"] or config["splits"] != route["splits"]
            or config["task"]["task_id"] != route["task_id"]):
        raise ValueError("Checked snapshot identity/counts differ from declared route")
    return {"final_sha256": sha(final), "manifest_sha256": digest, "task_fingerprint": manifest["task_fingerprint"]}


def initialize(args):
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=False); (root / "logs").mkdir()
    repo = args.repo.resolve()
    paths = [Path(__file__).resolve(), repo / "scripts/train_full_surface.py", repo / "scripts/train_full_surface_pricer.py",
             repo / "scripts/evaluate_full_surface.py", repo / "pyproject.toml", repo / "pixi.lock",
             repo / "tutorial_final/studies/train_calibration.py", *sorted((repo / "src/sbi4finance/calibration").glob("*.py"))]
    plan = {"format": "full_surface_post_bank_phase_v1", "prepared_utc": utc(), "repo": str(repo),
            "root": str(root), "python": str(args.python.resolve()), "source_revision": args.revision,
            "source_sha256": {str(p): sha(p) for p in paths}, "routes": ROUTES, "runtime": RUNTIME,
            "recipe": {"posterior_epochs": 500, "checkpoint_selection": "final", "family": "flow_matching",
                       "encoder": "vector", "widths": "256,256,256,256", "batch_size": 256,
                       "chunk_epochs": 50, "seed": 0, "pricer_epochs": 300, "pricer_patience": 30,
                       "pricer_learning_rate": .001, "validation_surfaces": 2000, "validation_draws": 1000},
            "resources": {"cpus_per_job": CPUS, "gpus_per_job": 1, "memory_gib": 32, "wall_seconds": WALL,
                          "cleanup_seconds": CLEANUP, "cpu_hour_ceiling": 9, "gpu_hour_ceiling": 2.25,
                          "full_two_job_cpu_seconds": 2 * CPUS * (WALL + CLEANUP),
                          "full_two_job_gpu_seconds": 2 * (WALL + CLEANUP)},
            "scientific_acceptance": False, "reserved_test_evaluated": False, "automatic_retry": False}
    source_check(plan)
    if 2 * CPUS * (WALL + CLEANUP) > 9 * 3600 or 2 * (WALL + CLEANUP) > 2.25 * 3600:
        raise ValueError("Full phase exposure exceeds ceiling")
    save(root / "plan.json", plan)
    save(root / "execution.json", {"format": "full_surface_post_bank_execution_v1", "state": "waiting_for_banks",
         "plan_sha256": sha(root / "plan.json"), "routes": {n: {"state": "waiting"} for n in ROUTES},
         "scientific_acceptance": False, "reserved_test_evaluated": False})


def submit(plan, ledger, name):
    route = ledger["routes"][name]
    if route["state"] != "waiting": raise ValueError("Route already has a recorded attempt")
    ready = readiness(plan["routes"][name])
    if ready is None: return
    source_check(plan)
    root = Path(plan["root"]); directory = root / name; directory.mkdir(exist_ok=False)
    env = {"PYTHONPATH": str(Path(plan["repo"]) / "src"), "KERAS_BACKEND": "jax", "JAX_PLATFORMS": "cuda,cpu",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "MPLCONFIGDIR": str(directory / "cache/mpl"),
           "XDG_CACHE_HOME": str(directory / "cache")}
    command = shlex.join(["env", *[k + "=" + v for k, v in env.items()], plan["python"], __file__,
                          "worker", "--root", plan["root"], "--route", name])
    argv = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--no-requeue", "--account=default",
            "--partition=gpu-debug", "--qos=gpu-debug", "--cpus-per-task=4", "--gres=gpu:l40s:1", "--mem=32G",
            "--time=01:00:00", "--job-name=sbi4f-" + name + "-baseline", "--chdir=" + plan["repo"],
            "--output=" + str(root / "logs" / (name + "-%A.out")), "--wrap=" + command]
    route.update(state="submission_attempted", attempted_utc=utc(), bank=ready, argv=argv)
    save(root / "execution.json", ledger)  # Before scheduler contact; ambiguity is never retried.
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
        route.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        job_id = result.stdout.strip().split(";")[0]
        if result.returncode != 0 or not job_id.isdigit(): raise RuntimeError("Unconfirmed scheduler response")
        route.update(state="submitted", job_id=job_id)
    except BaseException as error:
        route.update(state="unknown_or_failed_submission", error_type=type(error).__name__, error=str(error))
    save(root / "execution.json", ledger)


def monitor(plan, ledger, name):
    record = ledger["routes"][name]
    try:
        result = subprocess.run(["sacct", "-n", "-P", "-j", record["job_id"],
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES"], capture_output=True, text=True, timeout=25)
        if result.returncode: raise RuntimeError(result.stderr)
        lines = [line.split("|") for line in result.stdout.splitlines() if line.split("|")[0] == record["job_id"]]
        if len(lines) != 1: raise RuntimeError("Missing or ambiguous allocation accounting")
        row = lines[0]; state = row[1].split()[0]
        record.update(last_accounting_utc=utc(), scheduler_state=state)
        if state in TERMINAL:
            elapsed, cpus = int(row[3]), int(row[4])
            if cpus != CPUS or "gres/gpu=1" not in row[5]: raise RuntimeError("Unexpected allocation resources")
            record.update(state="completed" if state == "COMPLETED" and row[2] == "0:0" else "failed",
                          accounting=row, allocated_cpu_seconds=elapsed * cpus, allocated_gpu_seconds=elapsed)
            if record["state"] == "completed":
                run = read(Path(plan["root"]) / name / "run.json")
                if run.get("status") != "completed": record.update(state="failed", error="Worker did not complete")
    except Exception as error:
        record["last_accounting_error"] = str(error)  # Keep full original exposure; never resubmit.


def watch_locked(root, once=False):
    plan = read(root / "plan.json"); source_check(plan); started = time.monotonic()
    while True:
        ledger = read(root / "execution.json")
        if ledger["plan_sha256"] != sha(root / "plan.json"): raise ValueError("Phase declaration changed")
        for name, state in ledger["routes"].items():
            try:
                if state["state"] == "waiting": submit(plan, ledger, name)
                elif state["state"] == "submitted": monitor(plan, ledger, name)
            except Exception as error: state.update(state="blocked", error_type=type(error).__name__, error=str(error))
        terminal = all(v["state"] not in ("waiting", "submitted") for v in ledger["routes"].values())
        ledger.update(state="finished" if terminal else "waiting_or_running", updated_utc=utc())
        save(root / "execution.json", ledger)
        if terminal or once: return
        if time.monotonic() - started > 24 * 3600: raise TimeoutError("24-hour watch limit; no additional submissions")
        time.sleep(60)


def watch(root, once=False):
    with (root / "operator.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError("Another phase operator already holds this lock") from error
        watch_locked(root, once=once)


def worker(root, name):
    plan = read(root / "plan.json"); source_check(plan)
    if {k: metadata.version(k) for k in RUNTIME} != RUNTIME: raise ValueError("Pinned core runtime differs")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != CPUS: raise ValueError("Four allocated CPUs required")
    import jax
    if jax.default_backend() != "gpu": raise ValueError("A native GPU backend is required")
    execution = read(root / "execution.json")
    if execution["plan_sha256"] != sha(root / "plan.json"): raise ValueError("Phase declaration changed")
    declaration = execution["routes"][name]
    if readiness(plan["routes"][name]) != declaration["bank"]: raise ValueError("Checked bank changed after submission")
    directory = root / name; bank = plan["routes"][name]["bank"]; repo = Path(plan["repo"])
    started = time.monotonic(); deadline = started + 3550
    journal = {"format": "full_surface_post_bank_run_v1", "status": "running", "route": name,
               "plan_sha256": sha(root / "plan.json"), "bank": declaration["bank"], "stages": [],
               "scientific_acceptance": False, "reserved_test_evaluated": False}
    save(directory / "run.json", journal)
    def stage(label, arguments, limit):
        journal["active_stage"] = label; save(directory / "run.json", journal)
        remaining = min(limit, deadline - time.monotonic())
        if remaining <= 0: raise TimeoutError("Whole-run budget exhausted")
        before = time.monotonic()
        with (directory / (label + ".log")).open("w") as log:
            subprocess.run([plan["python"], *map(str, arguments)], cwd=repo, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=remaining)
        journal["stages"].append({"name": label, "seconds": time.monotonic()-before})
        save(directory / "run.json", journal)
    try:
        stage("posterior", [repo / "scripts/train_full_surface.py", "--bank", bank, "--out", directory / "posterior",
            "--family", "flow_matching", "--encoder", "vector", "--widths", "256,256,256,256", "--epochs", "500",
            "--batch-size", "256", "--chunk-epochs", "50", "--seed", "0", "--checkpoint-selection", "final"], 1500)
        stage("pricer", [repo / "scripts/train_full_surface_pricer.py", "--bank", bank, "--out", directory / "pricer",
            "--widths", "256,256,256,256", "--epochs", "300", "--batch-size", "256", "--learning-rate", ".001",
            "--patience", str(plan["recipe"]["pricer_patience"]), "--seed", "0"], 1200)
        stage("validation", [Path(__file__), "evaluate", "--root", root, "--route", name], 1700)
        source_check(plan)
        if readiness(plan["routes"][name]) != declaration["bank"]: raise ValueError("Checked bank changed during run")
        journal.update(status="completed", seconds=time.monotonic()-started,
                       posterior_sha256=sha(directory / "posterior/model.keras"),
                       pricer_sha256=sha(directory / "pricer/checkpoint/pricer.npz"), frozen_inputs_verified=True)
    except BaseException as error:
        journal.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic()-started)
        save(directory / "run.json", journal); raise
    save(directory / "run.json", journal)


def evaluate(root, name):
    # Preserve batch payloads while keeping the existing evaluator and its
    # numerical calls/seed policy. No new selection or acceptance criterion.
    plan = read(root / "plan.json"); repo = Path(plan["repo"]); sys.path.insert(0, str(repo / "scripts"))
    import evaluate_full_surface as module
    import numpy as np
    directory = root / name; partial = directory / "validation-partial"; partial.mkdir(exist_ok=False)
    sample = module.full_surface.sample
    def committed_sample(*args, **kwargs):
        draws = sample(*args, **kwargs)
        np.savez_compressed(partial / ("batch-%04d.npz" % kwargs["seed"]), draws=draws, first=kwargs["seed"])
        return draws
    module.full_surface.sample = committed_sample
    module.evaluate(argparse.Namespace(bank=Path(plan["routes"][name]["bank"]), checkpoint=directory / "posterior",
        out=directory / "validation", split="validation", first=0, surfaces=2000, draws=1000, batch_size=100, seed=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initialize", "watch", "once", "worker", "evaluate"))
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--repo", type=Path)
    parser.add_argument("--python", type=Path); parser.add_argument("--revision"); parser.add_argument("--route", choices=ROUTES)
    args = parser.parse_args()
    if args.mode == "initialize":
        if None in (args.repo, args.python, args.revision): parser.error("initialize requires repo/python/revision")
        initialize(args)
    elif args.mode in ("watch", "once"): watch(args.root, once=args.mode == "once")
    elif args.mode == "worker": worker(args.root, args.route)
    else: evaluate(args.root, args.route)


if __name__ == "__main__": main()

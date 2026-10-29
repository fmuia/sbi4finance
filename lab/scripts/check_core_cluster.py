"""Bounded CPU/GPU and shared-filesystem readiness for the complete-surface workflow.

Run in an allocation, with --out on the intended shared run filesystem. The
small Heston bank and estimator establish software execution, not posterior
accuracy. Tests of locks use separate processes on one allocated node; they
do not establish cross-node lock coherence by themselves.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sbi4finance.calibration import bank_collection, tasks  # noqa: E402
from scripts import evaluate_full_surface, train_full_surface  # noqa: E402


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_process_lock(path):
    """A second process must fail a nonblocking flock while the first owns it."""
    code = """import fcntl,sys
with open(sys.argv[1], 'a+') as stream:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
raise SystemExit(1)
"""
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, "-c", code, str(path)], timeout=15, check=False)
        fcntl.flock(stream, fcntl.LOCK_UN)
    if result.returncode != 0:
        raise RuntimeError("separate-process lock exclusion failed on the output filesystem")


def run(args):
    if not args.allow_local and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run inside a Slurm allocation; --allow-local is only for local verification")
    if os.environ.get("KERAS_BACKEND") != "jax":
        raise ValueError("activate the locked core/core-gpu environment with KERAS_BACKEND=jax")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"format": "full_surface_cluster_readiness_v1", "scientific_acceptance": False,
              "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
              "output_path": str(args.out.resolve()), "seed": args.seed,
              "runtime": train_full_surface.recovery.runtime_versions(),
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "limitations": ["tiny execution check, not posterior validation",
                              "same-node lock check, not cross-node filesystem certification"],
              "stage": "runtime"}
    _write(args.out / "readiness.json", report)
    try:
        import bayesflow as bf
        import jax
        import keras

        if bf.__version__ != "2.1.0" or keras.backend.backend() != "jax":
            raise RuntimeError("the readiness check requires BayesFlow 2.1.0 with the JAX backend")
        devices = [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()]
        if args.require_gpu and not any(d["platform"] == "gpu" for d in devices):
            raise RuntimeError("no actual JAX GPU device in the allocated job")
        report["devices"] = devices
        report["stage"] = "shared_filesystem"
        check_process_lock(args.out / "readiness.lock")
        config = {"task": tasks.heston_task(), "root_seed": args.seed,
                  "splits": {"train": 4, "validation": 4, "test": 2}}
        collection = args.out / "collection"
        bank_collection.create_collection(collection, config)
        initial_jobs = bank_collection.plan_extension(collection, 4, 2)["jobs"]
        for job in initial_jobs:
            bank_collection.run_job(collection, job)
        first = args.out / "snapshot-4"
        bank_collection.freeze_snapshot(collection, 4, first)
        before_manifest, before = bank_collection.load_bank(first, splits=("train", "validation"),
                                                          roles=bank_collection.TRAINING_ROLES)
        timestamps = {str(p): p.stat().st_mtime_ns for p in collection.glob("parts/*/arrays.npz")}
        for job in initial_jobs:
            bank_collection.run_job(collection, job)
        if any(Path(p).stat().st_mtime_ns != stamp for p, stamp in timestamps.items()):
            raise RuntimeError("retry rewrote an already published shard")
        extra = bank_collection.plan_extension(collection, 8, 2)["jobs"]
        if (sum(j["stop"] - j["start"] for j in extra) != 4
                or any(j["split"] != "train" or j["start"] < 4 for j in extra)):
            raise RuntimeError("extension must generate exactly four new training IDs")
        for job in extra:
            bank_collection.run_job(collection, job)
        snapshot = args.out / "snapshot-8"
        after_manifest = bank_collection.freeze_snapshot(collection, 8, snapshot)
        _, after = bank_collection.load_bank(snapshot, splits=("train", "validation"),
                                            roles=bank_collection.TRAINING_ROLES)
        for name, value in before["train"].items():
            np.testing.assert_array_equal(value, after["train"][name][:4])
        for name, value in before["validation"].items():
            np.testing.assert_array_equal(value, after["validation"][name])
        if before_manifest["seed_audit"]["duplicate_pricing_seeds"] or after_manifest["seed_audit"]["duplicate_pricing_seeds"]:
            raise RuntimeError("duplicate pricing seed in the small collection")
        report["filesystem"] = {"process_lock_exclusion": True, "retry_rewrites": False,
                                "prefix_and_validation_preserved": True, "train_before": 4, "train_after": 8,
                                "validation": 4, "test": 2, "test_loaded_by_training_or_evaluation": False,
                                "snapshot_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()}
        report["stage"] = "training_recovery"
        _write(args.out / "readiness.json", report)
        train_args = argparse.Namespace(bank=snapshot, out=args.out / "training", family="spline", encoder="vector",
                  widths="16,16", summary_dim=32, flow_steps=4, mean_steps=4,
                  epochs=2, batch_size=4, chunk_epochs=1, seed=args.seed)
        recovery = train_full_surface.recovery
        original_commit = recovery.commit_resume_state

        class ExpectedInterruption(Exception):
            pass

        def interrupt_after_commit(*values, **kwargs):
            original_commit(*values, **kwargs)
            raise ExpectedInterruption

        with recovery.training_lock(train_args.out):
            with patch.object(recovery, "commit_resume_state", side_effect=interrupt_after_commit):
                try:
                    train_full_surface.run_training(train_args)
                except ExpectedInterruption:
                    pass
                else:
                    raise RuntimeError("the planned checkpoint interruption did not occur")
            training = train_full_surface.run_training(train_args)
        if training["resumed_at_epochs"] != [1] or not training["same_seed_reload_exact"]:
            raise RuntimeError("committed-epoch recovery or exact checkpoint reload failed")
        evaluation = evaluate_full_surface.evaluate(argparse.Namespace(
            bank=snapshot, checkpoint=train_args.out, out=args.out / "evaluation", split="validation",
            first=0, surfaces=4, draws=32, batch_size=2, seed=args.seed))
        if evaluation["reserved_test_evaluated"]:
            raise RuntimeError("readiness evaluation must not use the test split")
        report.update(stage="complete", passed=True,
                      training={"resumed_at_epochs": training["resumed_at_epochs"],
                                "same_seed_reload_exact": True, "test_read": False,
                                "checkpoint_sha256": training["checkpoint_sha256"]},
                      evaluated_split="validation", seconds=time.perf_counter() - started)
        _write(args.out / "readiness.json", report)
        return report
    except Exception as error:
        report.update(passed=False, error=str(error), seconds=time.perf_counter() - started)
        _write(args.out / "readiness.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271027)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("passed", "devices", "seconds", "scientific_acceptance")}))


if __name__ == "__main__":
    main()

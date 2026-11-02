"""Submit independent, bounded model lanes for the harmonized path campaign.

Inspect --dry-run first. Source/runtime identities and every attempted sbatch
command are recorded. Failed or ambiguous submissions are never retried, and an
existing receipt prevents duplicate launches. The earlier campaign is separate.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


MODELS = ("heston", "rough-known", "rough-joint")
LENGTHS = (1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756, 1260)
CLEANUP_SECONDS = 240
CPU_LIMIT_HOURS = 48
GPU_LIMIT_HOURS = 8
FORMAT = "harmonized_path_scaleup_submission_v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def job_specs(run, cpu_python, gpu_python):
    """Only necessary artifact dependencies; no cross-model failure cascade."""
    run = Path(run)
    specs = []

    def add(name, cpus, minutes, arguments, dependencies=(), *, gpu=False, after="afterok", model=None):
        specs.append({"name": name, "model": model, "cpus": cpus, "minutes": minutes,
                      "gpus": int(gpu), "dependencies": list(dependencies),
                      "dependency_condition": after,
                      "command": [gpu_python if gpu else cpu_python, "gpu" if gpu else "cpu", *arguments]})

    # Local tests are required before launch. This independent allocated check
    # diagnoses the deployed runtime without canceling useful model lanes.
    add("software", 2, 8, ["-m", "unittest", "discover", "-s", "tests", "-p", "test_path_*.py", "-q"])
    for index, model in enumerate(MODELS):
        prefix = f"{model}-"
        bank = run / "banks" / model
        pilot = prefix + "pilot"
        ready = prefix + "gpu-readiness"
        generated = prefix + "bank"
        integrity = prefix + "bank-check"
        add(pilot, 1, 15, ["scripts/check_path_scaleup.py", "--model", model,
            "--out", str(run / "pilot" / model)], model=model)
        add(ready, 4, 5, ["scripts/check_path_cluster.py", "--model", model, "--teaching",
            "--require-gpu", "--production-shape", "--out", str(run / "gpu-readiness" / model)], gpu=True, model=model)
        add(generated, 1, 75, ["scripts/path_bank.py", "generate", "--model", model, "--teaching",
            "--train", "1000000", "--validation", "4000", "--test", "4000", "--chunk", "128",
            "--seed", str(20261031001 + index), "--out", str(bank)], [pilot], model=model)
        add(integrity, 1, 12, ["scripts/check_path_scaleup.py", "--bank", str(bank),
            "--out", str(run / "bank-checks" / model)], [generated], model=model)
        evaluations = []
        for label, count, epochs in (("100k", 100000, 50), ("1m", 1000000, 5)):
            fit = prefix + "fit-" + label
            evaluation = prefix + "evaluation-" + label
            directory = run / "fits" / model / label
            # Same initialization/prefix seed within a model; the 100k rows are
            # the exact initial prefix of this model's million-row collection.
            add(fit, 4, 70, ["scripts/train_path_posterior.py", "--bank", str(bank),
                "--training-limit", str(count), "--epochs", str(epochs), "--max-seconds", "3300",
                "--streaming", "--cache-shards", "2",
                "--batch-size", "128", "--prefix-policy", "uniform_log_mixture",
                "--checkpoint-selection", "final", "--learning-rate", "0.0005",
                "--seed", str(20261031301 + index), "--out", str(directory)],
                [integrity, ready], gpu=True, model=model)
            add(evaluation, 2, 12, ["scripts/evaluate_path_posterior.py", "--bank", str(bank),
                "--checkpoint", str(directory), "--cases", "1024", "--draws", "512",
                "--seed", "20261031401", "--lengths", *map(str, LENGTHS),
                "--out", str(run / "evaluation" / model / label)], [fit], model=model)
            evaluations.append(evaluation)
        add(prefix + "comparison", 2, 10, ["scripts/compare_path_scaleup.py", "--bank", str(bank),
            "--small", str(run / "fits" / model / "100k"),
            "--large", str(run / "fits" / model / "1m"),
            "--small-evaluation", str(run / "evaluation" / model / "100k"),
            "--large-evaluation", str(run / "evaluation" / model / "1m"),
            "--out", str(run / "comparison" / model)], evaluations, after="afterany", model=model)
    add("report", 1, 5, ["cluster/report_path_scaleup.py", "--run-root", str(run)],
        [spec["name"] for spec in specs], after="afterany")
    return specs


def resource_budget(specs):
    seen = set()
    models = {}
    cpu_seconds = gpu_seconds = 0
    for spec in specs:
        if spec["name"] in seen or not set(spec["dependencies"]) <= seen:
            raise ValueError("job names must be unique and dependencies must refer to earlier jobs in this campaign")
        if spec["dependency_condition"] not in ("afterok", "afterany"):
            raise ValueError("unsupported dependency condition")
        if min(spec["cpus"], spec["minutes"]) < 1 or spec["gpus"] not in (0, 1):
            raise ValueError("invalid CPU/GPU/time reservation")
        if spec.get("model") is not None and any(models[name] != spec["model"] for name in spec["dependencies"]):
            raise ValueError("model lanes must not depend on unrelated lanes or global software diagnostics")
        seen.add(spec["name"])
        models[spec["name"]] = spec.get("model")
        seconds = spec["minutes"] * 60 + CLEANUP_SECONDS
        cpu_seconds += seconds * spec["cpus"]
        gpu_seconds += seconds * spec["gpus"]
    if cpu_seconds > CPU_LIMIT_HOURS * 3600 or gpu_seconds > GPU_LIMIT_HOURS * 3600:
        raise RuntimeError(f"campaign ceiling exceeded: {cpu_seconds / 3600:.6f} CPU-h, {gpu_seconds / 3600:.6f} GPU-h")
    return {"reserved_cpu_seconds_including_cleanup": cpu_seconds,
            "reserved_gpu_seconds_including_cleanup": gpu_seconds,
            "reserved_cpu_hours_including_cleanup": cpu_seconds / 3600,
            "reserved_gpu_hours_including_cleanup": gpu_seconds / 3600,
            "cpu_limit_hours": CPU_LIMIT_HOURS, "gpu_limit_hours": GPU_LIMIT_HOURS,
            "cleanup_seconds_per_job": CLEANUP_SECONDS,
            "scope": "new campaign only, including failures; previous 24 CPU-h / 3 GPU-h tranche remains separate",
            "automatic_retries": False}


def source_inventory(lab, specs):
    paths = {lab / "pyproject.toml", lab / "pixi.lock", lab / "cluster/path_job.sbatch",
             Path(__file__).resolve(), lab / "cluster/report_path_scaleup.py"}
    for spec in specs:
        if spec["command"][2].endswith(".py"):
            paths.add(lab / spec["command"][2])
    paths.update((lab / "src/sbi4finance/path_inference").glob("*.py"))
    paths.update(lab / "src/sbi4finance" / name for name in ("heston.py", "rough_bergomi.py", "bergomi.py"))
    paths.update((lab / "tests").glob("test_path_*.py"))
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("missing frozen source files: " + ", ".join(missing))
    return {str(path.relative_to(lab)): sha256(path) for path in sorted(paths)}


def sbatch_command(spec, args, run, lab, identifiers):
    gpu = bool(spec["gpus"])
    readiness = spec["name"].endswith("gpu-readiness")
    account = args.gpu_account if gpu else args.cpu_account
    partition = args.gpu_partition if gpu else args.cpu_partition
    qos = args.gpu_qos if gpu else args.cpu_qos
    if gpu and readiness:
        partition = args.readiness_gpu_partition or partition
        qos = args.readiness_gpu_qos or qos
    command = ["sbatch", "--parsable", "--no-requeue", "--kill-on-invalid-dep=yes",
               f"--job-name=path-scale-{spec['name']}", f"--account={account}", f"--partition={partition}",
               f"--cpus-per-task={spec['cpus']}", f"--mem={'24G' if gpu else '8G'}",
               f"--time={spec['minutes'] // 60:02d}:{spec['minutes'] % 60:02d}:00", f"--chdir={lab}",
               f"--export=ALL,SBI4FINANCE_SOURCE_LAB={lab},SBI4FINANCE_RUNTIME_LOCK_SHA256={args.runtime_lock_sha256}",
               f"--output={run / 'logs' / (spec['name'] + '-%j.out')}"]
    if gpu:
        gpu_type = getattr(args, "gpu_type", None)
        command.append(f"--gres=gpu:{gpu_type}:1" if gpu_type else "--gres=gpu:1")
    if qos:
        command.append(f"--qos={qos}")
    if spec["dependencies"]:
        command.append("--dependency=" + spec["dependency_condition"] + ":" + ":".join(
            identifiers[name] for name in spec["dependencies"]))
    return [*command, str(lab / "cluster/path_job.sbatch"), *spec["command"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cpu-python", required=True)
    parser.add_argument("--gpu-python", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256")
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--cpu-account", default="carney-mjfrank-condo2")
    parser.add_argument("--cpu-partition", default="batch")
    parser.add_argument("--cpu-qos")
    parser.add_argument("--gpu-account", default="default")
    parser.add_argument("--gpu-partition", default="gpu")
    parser.add_argument("--gpu-qos", default="norm-gpu")
    parser.add_argument("--gpu-type", help="Optional Slurm GPU type, for example l40s; applies to readiness and fits")
    parser.add_argument("--readiness-gpu-partition")
    parser.add_argument("--readiness-gpu-qos")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run, lab = args.run_root.resolve(), Path(__file__).resolve().parents[1]
    existing = run / "submission-receipt.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        raise RuntimeError(f"existing receipt records {len(saved.get('jobs', []))} attempts; no jobs resubmitted. Preserve the recorded scheduler IDs.")
    if run.exists() and not args.dry_run:
        raise RuntimeError("run directory already exists; choose a new campaign directory after reviewing existing state")
    for name in ("cpu_python", "gpu_python"):
        path = Path(getattr(args, name))
        if not path.is_absolute() or (not args.dry_run and (not path.is_file() or not os.access(path, os.X_OK))):
            raise ValueError(f"{name} must identify an executable absolute verified-runtime Python")
    specs = job_specs(run, args.cpu_python, args.gpu_python)
    budget = resource_budget(specs)
    sources = source_inventory(lab, specs)
    receipt = {"format": FORMAT, "created_at": utc(), "source_commit": args.source_commit,
               "source_archive_sha256": args.source_archive_sha256, "source_lab": str(lab),
               "source_sha256": sources, "run_root": str(run), "budget": budget,
               "runtime_lock_sha256": args.runtime_lock_sha256,
               "source_lock_sha256": sources["pixi.lock"], "jobs": [],
               "planned_jobs": specs, "automatic_retries": False,
               "model_roles": {"heston": "core", "rough-known": "core", "rough-joint": "research_extension"},
               "reserved_test_evaluated": False, "baseline_replaced": False,
               "scientific_acceptance": False,
               "training_comparison": {"small": {"trajectories": 100000, "epochs": 50},
                   "large": {"trajectories": 1000000, "epochs": 5}, "presentations_per_full_fit": 5000000,
                   "planned_steps_small": math.ceil(100000 / 128) * 50,
                   "planned_steps_large": math.ceil(1000000 / 128) * 5,
                   "note": "same per-model initialization seed and nested bank rows; actual exposure/soft stops determine comparability"}}
    if not args.dry_run:
        run.mkdir(parents=True, exist_ok=False)
        (run / "logs").mkdir()
        atomic_json(existing, receipt)
    identifiers = {}
    try:
        for spec in specs:
            command = sbatch_command(spec, args, run, lab, identifiers)
            print(shlex.join(command), flush=True)
            if args.dry_run:
                identifiers[spec["name"]] = "DRY_" + spec["name"]
                continue
            record = {**spec, "attempted_at": utc(), "state": "submission_attempt_pending",
                      "sbatch": command}
            receipt["jobs"].append(record)
            atomic_json(existing, receipt)
            result = subprocess.run(command, cwd=lab, text=True, capture_output=True, timeout=60, check=False)
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            identifier = result.stdout.strip().split(";")[0]
            if result.returncode != 0 or not identifier.isdigit():
                record["state"] = "submission_uncertain_no_retry"
                raise RuntimeError(f"sbatch failed or returned an ambiguous response for {spec['name']}: "
                                   f"returncode={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}")
            identifiers[spec["name"]] = identifier
            record.update(job_id=identifier, state="submitted", submitted_at=utc())
            atomic_json(existing, receipt)
    except Exception as error:
        receipt.update(submission_complete=False, submission_error=repr(error), stopped_at=utc())
        if receipt["jobs"] and receipt["jobs"][-1]["state"] == "submission_attempt_pending":
            receipt["jobs"][-1].update(state="submission_uncertain_no_retry", error=repr(error))
            if isinstance(error, subprocess.TimeoutExpired):
                for key in ("stdout", "stderr"):
                    value = getattr(error, key, None)
                    receipt["jobs"][-1][key] = value.decode(errors="replace") if isinstance(value, bytes) else value
        if not args.dry_run:
            atomic_json(existing, receipt)
        raise
    receipt.update(submission_complete=True, submission_finished_at=utc())
    if not args.dry_run:
        atomic_json(existing, receipt)
    print(json.dumps({"jobs": len(specs), "dry_run": args.dry_run, "budget": budget}, indent=2))


if __name__ == "__main__":
    main()

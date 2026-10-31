"""Submit one independent CPU-only frozen confirmation/final-characterization phase.

Inspect the dry run. There are no dependencies, retries or model changes. An
exclusive phase guard and pre-submission attempt records prevent duplicate work
even after a partial/ambiguous Slurm response. Final-test claims occur in workers.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("path_workshop_closeout_guard", ROOT / "scripts/finalize_path_workshop.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)

CLEANUP_SECONDS = 240
CPU_LIMIT_HOURS = 8


def budget():
    seconds = sum(3 * 2 * (minutes * 60 + CLEANUP_SECONDS) for minutes in (30, 40))
    if seconds > CPU_LIMIT_HOURS * 3600:
        raise RuntimeError("both closeout phases exceed the eight CPU-hour ceiling")
    return {"all_six_jobs_cpu_seconds_including_cleanup": seconds,
            "all_six_jobs_cpu_hours_including_cleanup": seconds / 3600,
            "cpu_limit_hours": CPU_LIMIT_HOURS, "gpu_hours": 0,
            "cleanup_seconds_per_job_core": CLEANUP_SECONDS, "automatic_retries": False}


def job_specs(selection, run, python, phase, selection_path, review_path=None):
    run = Path(run)
    design = selection["design"]
    cohort = design["confirmation" if phase == "confirmation" else "final"]
    jobs = []
    for name in guard.MODELS:
        record = selection["models"][name]
        command = [str(python), "cpu", "scripts/evaluate_path_posterior.py",
            "--bank", record["bank"], "--checkpoint", record["checkpoint"],
            "--out", str(run / phase / name), "--phase", "confirmation" if phase == "confirmation" else "final-test",
            "--split", cohort["split"], "--cases", str(cohort["cases"]), "--offset", str(cohort["offset"]),
            "--draws", str(design["draws"]), "--seed", str(design["seed"]),
            "--selection", str(Path(selection_path).resolve()), "--lengths", *map(str, design["lengths"])]
        if phase == "final":
            command += ["--review", str(Path(review_path).resolve())]
        jobs.append({"name": f"{phase}-{name}", "model": name, "phase": phase,
                     "minutes": 30 if phase == "confirmation" else 40,
                     "cpus": 2, "gpus": 0, "memory": "12G", "dependencies": [],
                     "output_directory": str(run / phase / name), "command": command})
    return jobs


def preflight(selection, args, jobs):
    design = selection["design"]
    for job in jobs:
        record = selection["models"][job["model"]]
        output = Path(job["output_directory"])
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        cohort = design["confirmation" if args.phase == "confirmation" else "final"]
        request = {"cases": cohort["cases"], "offset": cohort["offset"],
                   "draws": design["draws"], "lengths": design["lengths"], "seed": design["seed"]}
        if args.phase == "confirmation":
            guard.validate_confirmation_request(args.selection, record["bank"], record["checkpoint"], **request)
        else:
            guard.validate_final_request(args.selection, args.review, record["bank"], record["checkpoint"], output, **request)


def command(job, args, run, source):
    result = ["sbatch", "--parsable", "--no-requeue", "--kill-on-invalid-dep=yes",
        f"--job-name=path-closeout-{job['name']}", f"--account={args.cpu_account}",
        f"--partition={args.cpu_partition}", "--ntasks=1", "--cpus-per-task=2", "--mem=12G",
        f"--time=00:{job['minutes']:02d}:00", f"--chdir={source}",
        f"--export=ALL,SBI4FINANCE_SOURCE_LAB={source},SBI4FINANCE_RUNTIME_LOCK_SHA256={args.runtime_lock_sha256}",
        f"--output={run / 'logs' / (job['name'] + '-%j.out')}"]
    if args.cpu_qos:
        result.append(f"--qos={args.cpu_qos}")
    return [*result, str(source / "cluster/path_job.sbatch"), *job["command"]]


def atomic_receipt(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lab", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--python", required=True, dest="python_path")
    parser.add_argument("--run-root", type=Path, required=True, help="New closeout outputs, separate from completed scale-up data")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--phase", choices=("confirmation", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="New durable submission receipt")
    parser.add_argument("--cpu-account", default="carney-mjfrank-condo2")
    parser.add_argument("--cpu-partition", default="batch")
    parser.add_argument("--cpu-qos")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase == "final" and args.review is None:
        parser.error("final evaluation requires an explicit confirmation review")
    run, source, out = args.run_root.resolve(), args.source_lab.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("submission receipt exists; no resubmissions or automatic retries")
    selection = guard.read_selection(args.selection)
    if args.runtime_lock_sha256 != selection["runtime_lock_sha256"]:
        raise ValueError("runtime lock differs from the frozen selection")
    phase_guard = Path(selection["state_directory"]) / f"submission-{args.phase}.json"
    if phase_guard.exists():
        raise FileExistsError("phase submission was already claimed; preserve partial receipt and scheduler IDs")
    python = Path(args.python_path)
    if not python.is_absolute() or (not args.dry_run and (not python.is_file() or not os.access(python, os.X_OK))):
        raise ValueError("an absolute executable verified-runtime Python is required")
    paths = [source / name for name in ("scripts/evaluate_path_posterior.py", "scripts/finalize_path_workshop.py",
        "cluster/submit_path_closeout.py", "cluster/path_job.sbatch", "pyproject.toml", "pixi.lock")]
    paths += sorted((source / "src/sbi4finance/path_inference").glob("*.py"))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("incomplete immutable closeout source")
    source_hashes = {str(path.relative_to(source)): guard.inference.sha256(path) for path in paths}
    jobs = job_specs(selection, run, python, args.phase, args.selection, args.review)
    admission = budget()
    preflight(selection, args, jobs)
    receipt = {"format": "path_workshop_closeout_submission_v1", "created_at": guard.stamp(),
        "phase": args.phase, "run_root": str(run), "source_lab": str(source), "source_commit": args.source_commit,
        "source_archive_sha256": args.source_archive_sha256, "source_sha256": source_hashes,
        "runtime_lock_sha256": args.runtime_lock_sha256, "selection_sha256": selection["receipt_sha256"],
        "review_sha256": guard.inference.sha256(args.review) if args.review else None,
        "budget": admission, "planned_jobs": jobs, "jobs": [], "submission_complete": False,
        "automatic_retries": False, "scientific_acceptance": False, "gpu_allocations": 0}
    if args.dry_run:
        for job in jobs:
            print(shlex.join(command(job, args, run, source)))
        print(json.dumps({"dry_run": True, "budget": admission}))
        return receipt
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    # The canonical guard prevents bypass with another --out after a partial run.
    guard.exclusive_json(phase_guard, {"receipt": str(out), "created_at": guard.stamp(),
                                      "selection_sha256": selection["receipt_sha256"]})
    guard.exclusive_json(out, receipt)
    attempts = out.with_name(out.name + ".attempts")
    try:
        for index, job in enumerate(jobs):
            argv = command(job, args, run, source)
            record = {**job, "attempted_at": guard.stamp(), "state": "submission_attempt_pending", "sbatch": argv}
            receipt["jobs"].append(record)
            guard.exclusive_json(attempts / f"{index:02d}-{job['model']}.json", record)
            atomic_receipt(out, receipt)
            print(shlex.join(argv), flush=True)
            result = subprocess.run(argv, cwd=source, capture_output=True, text=True, check=False, timeout=60)
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            identifier = result.stdout.strip().split(";")[0]
            if result.returncode or not identifier.isdigit():
                record["state"] = "submission_uncertain_no_retry"
                raise RuntimeError(f"ambiguous/failed sbatch for {job['name']}: {result.stderr!r}; stdout={result.stdout!r}")
            record.update(job_id=identifier, state="submitted", submitted_at=guard.stamp())
            guard.exclusive_json(attempts / f"{index:02d}-{job['model']}.result.json", record)
            atomic_receipt(out, receipt)
    except Exception as error:
        receipt.update(submission_error=repr(error), stopped_at=guard.stamp())
        if receipt["jobs"] and receipt["jobs"][-1]["state"] == "submission_attempt_pending":
            receipt["jobs"][-1].update(state="submission_uncertain_no_retry", error=repr(error))
            if isinstance(error, subprocess.TimeoutExpired):
                for name in ("stdout", "stderr"):
                    value = getattr(error, name, None)
                    receipt["jobs"][-1][name] = value.decode(errors="replace") if isinstance(value, bytes) else value
        atomic_receipt(out, receipt)
        raise
    receipt.update(submission_complete=True, submission_finished_at=guard.stamp())
    atomic_receipt(out, receipt)
    return receipt


if __name__ == "__main__":
    main()

"""Submit one bounded, dependency-checked three-task path-inference pilot.

Run on Oscar from the immutable lab source. --dry-run prints exact sbatch
commands without creating directories or submitting. The receipt is updated
after each submission so a partial submission remains auditable.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def jobs(run, cpu_python, gpu_python):
    specs = []

    def add(name, cpu, minutes, arguments, dependencies=(), gpu=False):
        specs.append(dict(name=name, cpus=cpu, minutes=minutes, gpus=int(gpu),
                          dependencies=list(dependencies),
                          command=[gpu_python if gpu else cpu_python, "gpu" if gpu else "cpu", *arguments]))

    add("cpu-smoke", 2, 15, ["-m", "unittest", "discover", "-s", "tests", "-p", "test_path_*.py", "-q"])
    add("gpu-smoke", 4, 15, ["scripts/check_path_cluster.py", "--require-gpu", "--production-shape", "--out", str(run / "gpu-smoke")], gpu=True)
    add("numerics", 2, 30, ["scripts/check_path_numerics.py", "--cases", "32", "--out", str(run / "numerics")], ["cpu-smoke"])
    add("reference", 2, 60, ["scripts/path_reference.py", "--cases", "4", "--out", str(run / "reference")], ["cpu-smoke"])
    for index, model in enumerate(("heston", "rough-known", "rough-joint"), start=1):
        seed = 2026103000 + index
        add(f"bench-{model}", 1, 15, ["scripts/path_bank.py", "benchmark", "--model", model,
            "--count", "256", "--seed", str(seed), "--out", str(run / f"benchmark-{model}.json")], ["cpu-smoke"])
        add(f"bank-{model}", 1, 30, ["scripts/path_bank.py", "generate", "--model", model,
            "--train", "100000", "--validation", "2000", "--test", "2000", "--chunk", "128",
            "--seed", str(seed), "--out", str(run / "banks" / model)], [f"bench-{model}", "numerics"])
        add(f"fit-{model}", 4, 50, ["scripts/train_path_posterior.py", "--bank", str(run / "banks" / model),
            "--epochs", "60", "--batch-size", "128", "--seed", str(2026103100 + index),
            "--max-seconds", "2700", "--out", str(run / "fits" / model)], [f"bank-{model}", "gpu-smoke"], gpu=True)
        add(f"eval-{model}", 2, 30, ["scripts/evaluate_path_posterior.py", "--bank", str(run / "banks" / model),
            "--checkpoint", str(run / "fits" / model), "--cases", "256", "--draws", "512",
            "--out", str(run / "evaluation" / model)], [f"fit-{model}"])
    add("report", 1, 5, ["cluster/report_path_pilot.py", "--run-root", str(run)],
        [spec["name"] for spec in specs])
    specs[-1]["dependency_condition"] = "afterany"
    return specs


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--cpu-python", required=True)
    parser.add_argument("--gpu-python", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--cpu-partition", required=True)
    parser.add_argument("--gpu-partition", required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run = args.run_root.resolve()
    lab = Path(__file__).resolve().parents[1]
    specs = jobs(run, args.cpu_python, args.gpu_python)
    cpu_hours = sum(s["cpus"] * s["minutes"] / 60 for s in specs)
    gpu_hours = sum(s["gpus"] * s["minutes"] / 60 for s in specs)
    if cpu_hours > 24 or gpu_hours > 3:
        raise ValueError("Proposed jobs exceed the fixed 24 CPU-h / 3 GPU-h ceiling")
    receipt = dict(created_at=datetime.now(timezone.utc).isoformat(), source_commit=args.source_commit,
                   source_lab=str(lab), run_root=str(run), runtime_lock_sha256=args.runtime_lock_sha256,
                   source_lock_sha256=hashlib.sha256((lab / "pixi.lock").read_bytes()).hexdigest(),
                   reserved_cpu_hours=cpu_hours, reserved_gpu_hours=gpu_hours, jobs=[],
                   scope="development pilots only; no reserved-test evaluation or scientific acceptance")
    if not args.dry_run:
        run.mkdir(parents=True, exist_ok=False)
        (run / "logs").mkdir()
        atomic_json(run / "submission-receipt.json", receipt)
    identifiers = {}
    try:
        for spec in specs:
            gpu = bool(spec["gpus"])
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", f"--job-name=path-{spec['name']}",
                       f"--export=ALL,SBI4FINANCE_RUNTIME_LOCK_SHA256={args.runtime_lock_sha256}",
                       f"--account={args.account}", f"--partition={args.gpu_partition if gpu else args.cpu_partition}",
                       f"--cpus-per-task={spec['cpus']}", f"--mem={'24G' if gpu else '8G'}",
                       f"--time=00:{spec['minutes']:02d}:00" if spec["minutes"] < 60 else "--time=01:00:00",
                       f"--chdir={lab}", f"--output={run / 'logs' / (spec['name'] + '-%j.out')}"]
            if gpu:
                command.append("--gres=gpu:1")
            if spec["dependencies"]:
                condition = spec.get("dependency_condition", "afterok")
                command.append(f"--dependency={condition}:" + ":".join(identifiers[d] for d in spec["dependencies"]))
            command.extend([str(lab / "cluster/path_job.sbatch"), *spec["command"]])
            print(shlex.join(command), flush=True)
            if args.dry_run:
                identifier = f"DRY_{spec['name']}"
            else:
                result = subprocess.run(command, text=True, capture_output=True, check=True)
                identifier = result.stdout.strip().split(";")[0]
                if not identifier.isdigit():
                    raise RuntimeError(f"Unexpected sbatch reply: {result.stdout!r}")
            identifiers[spec["name"]] = identifier
            receipt["jobs"].append({**spec, "job_id": identifier, "sbatch": command})
            if not args.dry_run:
                atomic_json(run / "submission-receipt.json", receipt)
    except Exception as error:
        receipt["submission_error"] = repr(error)
        if not args.dry_run:
            atomic_json(run / "submission-receipt.json", receipt)
        raise
    receipt["submission_complete"] = True
    if not args.dry_run:
        atomic_json(run / "submission-receipt.json", receipt)
    print(json.dumps({"jobs": len(specs), "reserved_cpu_hours": cpu_hours,
                      "reserved_gpu_hours": gpu_hours, "dry_run": args.dry_run}), flush=True)


if __name__ == "__main__":
    main()

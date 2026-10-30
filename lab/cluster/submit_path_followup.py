"""One fixed short-history candidate per affected model, plus reference repairs.

Print exact sbatch commands with --dry-run before submission. Admission counts
actual old spending plus conservative outstanding limits and four-minute cleanup
allowances, including earlier failed jobs. No automatic retries are submitted.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


def job_specs(old, run, cpu, gpu, reference):
    specs = []

    def add(name, cores, minutes, command, dependencies=(), *, graphics=False, analysis=False, after="afterok"):
        specs.append({"name": name, "cpus": cores, "minutes": minutes, "gpus": int(graphics),
            "dependencies": list(dependencies), "dependency_condition": after,
            "command": [gpu if graphics else reference if analysis else cpu,
                        "gpu" if graphics else "cpu", *command]})

    add("software", 2, 5, ["-m", "unittest", "discover", "-s", "tests", "-p", "test_path_*.py", "-q"])
    add("proposal", 1, 5, ["scripts/prepare_path_proposal.py", "--chains", str(old / "pmmh/length-252"),
        "--out", str(run / "proposal-252.json")], ["software"])
    for chain in range(4):
        add(f"pmmh-{chain}", 1, 15, ["scripts/path_pmmh.py", "--bank", str(old / "banks/heston"),
            "--case-index", "0", "--length", "252", "--chain", str(chain), "--tune", "500", "--draws", "5000",
            "--particles", "512", "--proposal-covariance", str(run / "proposal-252.json"),
            "--initial-checkpoint", str(old / f"pmmh/length-252/chain-{chain}"), "--phase-seed", "2",
            "--out", str(run / f"pmmh/chain-{chain}")], ["proposal"])
    add("noise", 1, 10, ["scripts/check_path_likelihood_noise.py", "--bank", str(old / "banks/heston"),
        "--chains", str(old / "pmmh/length-1260"), "--out", str(run / "noise")], ["software"])
    for index, model in enumerate(("heston", "rough-joint")):
        maximum = 1260 if model == "heston" else 1008
        lengths = [str(n) for n in sorted({1, 2, 4, 8, 17, 32, 63, 126, 252, 504, maximum - 7, maximum})]
        base, candidate = old / "fits" / model, run / "candidates" / model
        shared = ["--bank", str(old / "banks" / model), "--cases", "512", "--draws", "512", "--seed", "20261030211", "--lengths", *lengths]
        add(f"baseline-{model}", 2, 10, ["scripts/evaluate_path_posterior.py", *shared,
            "--checkpoint", str(base), "--out", str(run / "evaluation" / f"baseline-{model}")], ["software"])
        add(f"fit-{model}", 4, 10, ["scripts/train_path_posterior.py", "--bank", str(old / "banks" / model),
            "--init-checkpoint", str(base), "--prefix-policy", "uniform_log_mixture", "--checkpoint-selection", "final",
            "--learning-rate", "0.0001", "--epochs", "8", "--max-seconds", "420", "--batch-size", "128",
            "--seed", str(20261030201 + index), "--out", str(candidate)], ["software"], graphics=True)
        add(f"candidate-{model}", 2, 10, ["scripts/evaluate_path_posterior.py", *shared,
            "--checkpoint", str(candidate), "--out", str(run / "evaluation" / f"candidate-{model}")], [f"fit-{model}"])
        dependencies = [f"baseline-{model}", f"candidate-{model}"]
        comparison = ["scripts/compare_path_refinement.py", "--baseline", str(run / "evaluation" / f"baseline-{model}"),
            "--candidate", str(run / "evaluation" / f"candidate-{model}"), "--task", str(base / "task.json"),
            "--out", str(run / "comparison" / model)]
        if model == "rough-joint":
            for label, checkpoint, dependency in (("baseline", base, "software"), ("candidate", candidate, f"fit-{model}")):
                add(f"exact-{label}", 2, 10, ["scripts/check_path_one_return.py", "--bank", str(old / "banks" / model),
                    "--checkpoint", str(checkpoint), "--cases", "512", "--draws", "2048",
                    "--seed", "20261030213", "--out", str(run / "exact" / label)], [dependency])
                dependencies.append(f"exact-{label}")
                comparison.extend([f"--{label}-exact", str(run / "exact" / label)])
        add(f"compare-{model}", 1, 5, comparison, dependencies)
    add("pmmh-summary", 1, 5, ["scripts/summarize_path_pmmh.py", "--chains", str(run / "pmmh"),
        "--baseline-chains", str(old / "pmmh/length-252"), "--out", str(run / "pmmh-summary/summary.json")],
        [f"pmmh-{chain}" for chain in range(4)], analysis=True)
    add("conditional-heston", 2, 5, ["scripts/compare_path_reference.py", "--bank", str(old / "banks/heston"),
        "--chains", str(run / "pmmh"), "--reference-summary", str(run / "pmmh-summary/summary.json"),
        "--baseline-checkpoint", str(old / "fits/heston"), "--candidate-checkpoint", str(run / "candidates/heston"),
        "--out", str(run / "conditional-heston")], ["pmmh-summary", "fit-heston"])
    add("report", 1, 5, ["cluster/report_path_followup.py", "--run-root", str(run)],
        [spec["name"] for spec in specs], after="afterany")
    return specs


def reconcile(old, specs):
    main = json.loads((old / "submission-receipt.json").read_text())
    extra = json.loads((old / "pmmh/submission-receipt.json").read_text())
    records = {job["job_id"]: {"cpus": 1, "gpus": 0, **job} for job in main["jobs"] + extra["jobs"]}
    prior = ["8109953", "8109954", "8109969"]
    query = subprocess.run(["sacct", "-nP", "-j", ",".join([*records, *prior]), "--format",
                            "JobIDRaw,State,ElapsedRaw,AllocCPUS,AllocTRES"], text=True, capture_output=True, check=True)
    rows = {parts[0]: parts for line in query.stdout.splitlines()
            if len(parts := line.split("|")) >= 5 and parts[0].isdigit()}
    terminal = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
    cpu, gpu = 0.0, 0.0
    for identifier, job in records.items():
        row = rows.get(identifier)
        ended = row and any(row[1].startswith(state) for state in terminal)
        seconds = int(row[2] or 0) if ended else job["minutes"] * 60
        cores = int(row[3] or 0) if ended else job["cpus"]
        cpu += seconds * cores
        gpu += seconds * job["gpus"]
        if not ended or seconds:
            cpu += 240 * job["cpus"]
            gpu += 240 * job["gpus"]
    for identifier in prior:
        row = rows.get(identifier)
        if not row or not any(row[1].startswith(state) for state in terminal):
            raise RuntimeError(f"earlier job {identifier} lacks terminal accounting")
        seconds, cores = int(row[2] or 0), int(row[3] or 0)
        tres = dict(item.split("=", 1) for item in row[4].split(",") if "=" in item)
        if seconds:
            cpu += (seconds + 240) * cores
            gpu += (seconds + 240) * int(tres.get("gres/gpu", "0"))
    old_cpu, old_gpu = cpu, gpu
    cpu += sum((s["minutes"] * 60 + 240) * s["cpus"] for s in specs)
    gpu += sum((s["minutes"] * 60 + 240) * s["gpus"] for s in specs)
    if cpu > 24 * 3600 or gpu > 3 * 3600:
        raise RuntimeError(f"campaign ceiling exceeded: {cpu / 3600:.5f} CPU-h, {gpu / 3600:.5f} GPU-h")
    return {"existing_cpu_hours_including_cleanup": old_cpu / 3600, "existing_gpu_hours_including_cleanup": old_gpu / 3600,
        "total_cpu_hours_including_cleanup": cpu / 3600, "total_gpu_hours_including_cleanup": gpu / 3600,
        "cpu_limit": 24, "gpu_limit": 3, "cleanup_seconds_per_job": 240,
        "accounting_snapshot": query.stdout}, main["runtime_lock_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-run", "run-root", "cpu-python", "gpu-python", "reference-python", "source-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    old, run, lab = Path(args.old_run).resolve(), Path(args.run_root).resolve(), Path(__file__).resolve().parents[1]
    specs = job_specs(old, run, args.cpu_python, args.gpu_python, args.reference_python)
    budget, runtime_lock = reconcile(old, specs)
    receipt = {"created_at": datetime.now(timezone.utc).isoformat(), "source_commit": args.source_commit,
               "old_run": str(old), "run_root": str(run), "budget": budget, "jobs": [],
               "reserved_test_evaluated": False, "baseline_replaced": False}
    if not args.dry_run:
        run.mkdir(parents=True, exist_ok=False)
        (run / "logs").mkdir()
    def save():
        if not args.dry_run:
            temporary = run / "submission-receipt.json.tmp"
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(run / "submission-receipt.json")
    save()
    ids = {}
    try:
        for spec in specs:
            graphics = bool(spec["gpus"])
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", f"--job-name=path-next-{spec['name']}",
                f"--account={'default' if graphics else 'carney-mjfrank-condo2'}", f"--partition={'gpu' if graphics else 'batch'}",
                f"--cpus-per-task={spec['cpus']}", f"--mem={'24G' if graphics else '8G'}",
                f"--time=00:{spec['minutes']:02d}:00", f"--chdir={lab}",
                f"--export=ALL,SBI4FINANCE_SOURCE_LAB={lab},SBI4FINANCE_RUNTIME_LOCK_SHA256={runtime_lock}",
                f"--output={run / 'logs' / (spec['name'] + '-%j.out')}"]
            if graphics:
                command += ["--gres=gpu:1", "--qos=norm-gpu"]
            if spec["dependencies"]:
                command += ["--dependency=" + spec["dependency_condition"] + ":" + ":".join(ids[n] for n in spec["dependencies"])]
            command += [str(lab / "cluster/path_job.sbatch"), *spec["command"]]
            print(shlex.join(command), flush=True)
            if args.dry_run:
                identifier = f"DRY_{spec['name']}"
            else:
                response = subprocess.run(command, cwd=lab, text=True, capture_output=True, check=True)
                identifier = response.stdout.strip().split(";")[0]
                if not identifier.isdigit():
                    raise RuntimeError(f"unexpected sbatch reply: {response.stdout!r}")
            ids[spec["name"]] = identifier
            receipt["jobs"].append({**spec, "job_id": identifier, "sbatch": command})
            save()
    except Exception as error:
        receipt["submission_error"] = repr(error)
        save()
        raise
    receipt["submission_complete"] = True
    save()
    print(json.dumps({"jobs": len(specs), "budget": {k: v for k, v in budget.items() if k != "accounting_snapshot"},
                      "dry_run": args.dry_run}), flush=True)


if __name__ == "__main__":
    main()

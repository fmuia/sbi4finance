"""Add eight bounded Heston PMMH chains after reconciling campaign exposure."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-receipt", type=Path, required=True)
    parser.add_argument("--cpu-python", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--prior-job-ids", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    original = json.loads(args.main_receipt.read_text())
    run = Path(original["run_root"])
    destination = run / "pmmh"
    lab = Path(__file__).resolve().parents[1]
    records = {job["job_id"]: job for job in original["jobs"]}
    prior = [i for i in args.prior_job_ids.split(",") if i]
    query = subprocess.run(["sacct", "-nP", "-j", ",".join([*records, *prior]), "--format",
                            "JobIDRaw,State,ElapsedRaw,AllocCPUS,AllocTRES"],
                           text=True, capture_output=True, check=True)
    states = {}
    for row in query.stdout.splitlines():
        fields = row.split("|")
        if len(fields) >= 5 and fields[0].isdigit():
            states[fields[0]] = fields
    cpu, gpu = 0.0, 0.0
    terminal = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
    for identifier, job in records.items():
        row = states.get(identifier)
        if row and any(row[1].startswith(value) for value in terminal):
            seconds = int(row[2] or 0)
            actual_cpus = int(row[3] or 0)
            cpu += seconds * actual_cpus / 3600
            gpu += seconds * job["gpus"] / 3600
        else:
            cpu += job["minutes"] * job["cpus"] / 60
            gpu += job["minutes"] * job["gpus"] / 60
        # Conservatively reserve cleanup even for completed allocated jobs.
        if not row or int(row[2] or 0) > 0 or not any(row[1].startswith(value) for value in terminal):
            cpu += 240 * job["cpus"] / 3600
            gpu += 240 * job["gpus"] / 3600
    for identifier in prior:
        row = states.get(identifier)
        if not row:
            raise RuntimeError(f"missing accounting for earlier job {identifier}")
        if not any(row[1].startswith(value) for value in terminal):
            raise RuntimeError(f"earlier job {identifier} is still active; remaining exposure must be reserved")
        seconds, cpus = int(row[2] or 0), int(row[3] or 0)
        tres = dict(item.split("=", 1) for item in row[4].split(",") if "=" in item)
        gpus = int(tres.get("gres/gpu", "0"))
        cpu += (seconds + (240 if seconds else 0)) * cpus / 3600
        gpu += (seconds + (240 if seconds else 0)) * gpus / 3600
    # One five-minute software check and eight twenty-minute one-core chains.
    new_cpu = (5 + 8 * 20 + 9 * 4) / 60
    if cpu + new_cpu > 24 or gpu > 3:
        raise RuntimeError(f"exposure including cleanup exceeds ceiling: CPU {cpu + new_cpu:.3f}, GPU {gpu:.3f}")
    receipt = {"created_at": datetime.now(timezone.utc).isoformat(), "source_commit": args.source_commit,
               "main_receipt": str(args.main_receipt), "source_lab": str(lab),
               "accounting_snapshot": query.stdout, "conservative_total_cpu_hours": cpu + new_cpu,
               "conservative_total_gpu_hours": gpu, "cleanup_seconds_per_job": 240,
               "new_requested_cpu_hours": (5 + 8 * 20) / 60, "jobs": [],
               "scope": "two validation prefixes, four chains each; feasibility only, no accepted-reference claim"}
    if not args.dry_run:
        destination.mkdir(parents=True, exist_ok=False)

    def save():
        if not args.dry_run:
            temporary = destination / "submission-receipt.json.tmp"
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(destination / "submission-receipt.json")

    def submit(name, minutes, arguments, dependencies=()):
        command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", f"--job-name=path-{name}",
            f"--account={args.account}", f"--partition={args.partition}", "--cpus-per-task=1", "--mem=8G",
            f"--time=00:{minutes:02d}:00", f"--chdir={lab}",
            f"--export=ALL,SBI4FINANCE_SOURCE_LAB={lab},SBI4FINANCE_RUNTIME_LOCK_SHA256={original['runtime_lock_sha256']}",
            f"--output={run / 'logs' / (name + '-%j.out')}"]
        if dependencies:
            command.append("--dependency=afterok:" + ":".join(dependencies))
        command.extend([str(lab / "cluster/path_job.sbatch"), args.cpu_python, "cpu", *arguments])
        print(shlex.join(command), flush=True)
        if args.dry_run:
            identifier = f"DRY_{name}"
        else:
            result = subprocess.run(command, text=True, capture_output=True, cwd=lab, check=True)
            identifier = result.stdout.strip().split(";")[0]
            if not identifier.isdigit():
                raise RuntimeError(f"unexpected sbatch reply: {result.stdout!r}")
        receipt["jobs"].append({"name": name, "job_id": identifier, "minutes": minutes, "sbatch": command})
        save()
        return identifier

    save()
    ready = submit("pmmh-software", 5, ["-m", "unittest", "discover", "-s", "tests", "-p", "test_path_pmmh.py", "-q"])
    bank_job = next(j["job_id"] for j in original["jobs"] if j["name"] == "bank-heston")
    bank_state = states.get(bank_job, [None, "UNKNOWN"])[1]
    dependencies = [ready] if bank_state == "COMPLETED" else [ready, bank_job]
    for length, particles in ((252, 512), (1260, 2048)):
        for chain in range(4):
            submit(f"pmmh-{length}-{chain}", 20,
                ["scripts/path_pmmh.py", "--bank", str(run / "banks/heston"), "--case-index", "0",
                 "--length", str(length), "--chain", str(chain), "--tune", "500", "--draws", "1000",
                 "--particles", str(particles), "--out", str(destination / f"length-{length}" / f"chain-{chain}")],
                dependencies)
    receipt["submission_complete"] = True
    save()
    print(json.dumps({"jobs": len(receipt["jobs"]), "total_cpu_exposure_with_cleanup": cpu + new_cpu,
                      "total_gpu_exposure_with_cleanup": gpu, "dry_run": args.dry_run}))


if __name__ == "__main__":
    main()

"""Collect scale-up artifacts and accounting even when individual lanes fail."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


MODELS = ("heston", "rough-known", "rough-joint")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_artifact(path, *, bank=False):
    result = {"path": str(path), "missing": not path.is_file()}
    if result["missing"]:
        return result
    try:
        value = json.loads(path.read_text())
        result["sha256"] = sha256(path)
        if bank:
            value = {key: value.get(key) for key in ("format", "task", "task_sha256", "root_seed", "split_counts", "provenance", "timing_seconds")}
        result["record"] = value
    except Exception as error:
        result.update(read_error=repr(error), usable=False)
    return result


def read_directory(path):
    paths = sorted(path.glob("*.json")) if path.is_dir() else []
    return {"directory": str(path), "missing": not paths,
            "records": {item.name: read_artifact(item) for item in paths}}


def accounting_summary(raw, jobs, cleanup):
    rows = {fields[0]: fields for line in raw.splitlines()
            if len(fields := line.split("|")) >= 7 and fields[0].isdigit()}
    actual_cpu = actual_gpu = 0
    exposure_cpu = exposure_gpu = 0
    statuses = {}
    for job in jobs:
        identifier = str(job.get("job_id", ""))
        if not identifier.isdigit():
            statuses[job["name"]] = {"status": job["state"], "job_id": None,
                                      "submission_error": job.get("error"), "stderr": job.get("stderr")}
            exposure_cpu += (job["minutes"] * 60 + cleanup) * job["cpus"]
            exposure_gpu += (job["minutes"] * 60 + cleanup) * job["gpus"]
            continue
        row = rows.get(identifier)
        if row is None:
            statuses[job["name"]] = {"status": "accounting_unavailable", "job_id": identifier}
            exposure_cpu += (job["minutes"] * 60 + cleanup) * job["cpus"]
            exposure_gpu += (job["minutes"] * 60 + cleanup) * job["gpus"]
            continue
        state = row[1].split()[0].rstrip("+")
        seconds, cores = int(row[3] or 0), int(row[4] or 0)
        tres = dict(item.split("=", 1) for item in row[5].split(",") if "=" in item)
        gpus = int(tres.get("gres/gpu", job["gpus"] if seconds else 0))
        actual_cpu += seconds * cores
        actual_gpu += seconds * gpus
        if state in TERMINAL:
            exposure_cpu += seconds * cores + (cleanup * job["cpus"] if seconds else 0)
            exposure_gpu += seconds * gpus + (cleanup * job["gpus"] if seconds else 0)
        else:
            exposure_cpu += (max(seconds, job["minutes"] * 60) + cleanup) * job["cpus"]
            exposure_gpu += (max(seconds, job["minutes"] * 60) + cleanup) * job["gpus"]
        statuses[job["name"]] = {"job_id": identifier, "status": state, "exit_code": row[2],
            "elapsed_seconds": seconds, "allocated_cpus": cores, "allocated_gpus": gpus,
            "max_rss_top_level": row[6], "requested_cpus": job["cpus"], "requested_gpus": job["gpus"]}
    return {"jobs": statuses, "actual_allocated_cpu_seconds_snapshot": actual_cpu,
            "actual_allocated_gpu_seconds_snapshot": actual_gpu,
            "conservative_exposure_cpu_hours_including_cleanup": exposure_cpu / 3600,
            "conservative_exposure_gpu_hours_including_cleanup": exposure_gpu / 3600,
            "scope": "snapshot while the report job may still be running; terminal accounting may lag; no batch/extern double counting"}


def collect(run):
    receipt_path = run / "submission-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    jobs = receipt["jobs"]
    ids = sorted({str(job["job_id"]) for job in jobs if str(job.get("job_id", "")).isdigit()})
    if ids:
        try:
            query = subprocess.run(["sacct", "-nP", "-j", ",".join(ids), "--format",
                "JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS"],
                text=True, capture_output=True, check=False, timeout=45)
            raw, error, code = query.stdout, query.stderr, query.returncode
        except (OSError, subprocess.TimeoutExpired) as failure:
            raw, error, code = "", repr(failure), None
    else:
        raw, error, code = "", "No known scheduler IDs; inspect the submission receipt.", None
    (run / "accounting.psv").write_text(raw)
    (run / "accounting.stderr.txt").write_text(error)
    result = {"format": "harmonized_path_scaleup_report_v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "source_commit": receipt["source_commit"], "source_archive_sha256": receipt.get("source_archive_sha256"),
              "source_lab": receipt["source_lab"], "runtime_lock_sha256": receipt["runtime_lock_sha256"],
              "submission_receipt_sha256": sha256(receipt_path), "budget_admission": receipt["budget"],
              "submission_complete": receipt.get("submission_complete", False),
              "submission_error": receipt.get("submission_error"),
              "models": {},
              "accounting_query_returncode": code, "accounting_query_stderr": error,
              "accounting": accounting_summary(raw, jobs, receipt["budget"]["cleanup_seconds_per_job"]),
              "reserved_test_evaluated": False, "baseline_replaced": False, "scientific_acceptance": False,
              "scope": "development scale-up; failed/missing lanes remain explicit; no automatic estimator publication"}
    result["software"] = {"execution": result["accounting"]["jobs"].get("software", {"status": "not_submitted"}),
                          "log_pattern": str(run / "logs/software-*.out"), "separate_artifact_expected": False}
    for model in MODELS:
        lane = {"role": receipt.get("model_roles", {}).get(model),
                "pilot": read_directory(run / "pilot" / model),
                "gpu_readiness": read_directory(run / "gpu-readiness" / model),
                "bank": read_artifact(run / "banks" / model / "manifest.json", bank=True),
                "bank_check": read_directory(run / "bank-checks" / model),
                "training": {}, "evaluation": {}, "comparison": read_directory(run / "comparison" / model)}
        for label in ("100k", "1m"):
            lane["training"][label] = read_artifact(run / "fits" / model / label / "training.json")
            lane["evaluation"][label] = read_artifact(run / "evaluation" / model / label / "evaluation.json")
        records = [lane["training"][label].get("record", {}) for label in ("100k", "1m")]
        planned = (50, 5)
        lane["exposure_comparison"] = {
            "planned_epochs": dict(zip(("100k", "1m"), planned)),
            "epochs_completed": {label: record.get("epochs_completed") for label, record in zip(("100k", "1m"), records)},
            "optimizer_steps": {label: record.get("optimizer_steps") for label, record in zip(("100k", "1m"), records)},
            "actual_presentations": {label: record.get("sample_presentations", record.get("training_presentations", record.get("completed_training_views")))
                                      for label, record in zip(("100k", "1m"), records)},
            "stop_status": {label: record.get("status") for label, record in zip(("100k", "1m"), records)},
            "both_planned_epoch_counts_completed": all(record.get("epochs_completed") == epochs for record, epochs in zip(records, planned)),
            "note": "Inspect actual optimizer steps/presentations and stop reasons before attributing differences to training-data count."}
        result["models"][model] = lane
    temporary = run / "scaleup-summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(run / "scaleup-summary.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.run_root.resolve())
    print(json.dumps({"summary": str(args.run_root / "scaleup-summary.json"),
                      "scientific_acceptance": False, "submission_complete": result["submission_complete"]}))


if __name__ == "__main__":
    main()

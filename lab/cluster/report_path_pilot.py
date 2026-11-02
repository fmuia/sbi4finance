"""Collect final pilot outputs and Slurm accounting without opening reserved tests."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.run_root / "submission-receipt.json").read_text())
    identifiers = [job["job_id"] for job in receipt["jobs"]]
    query = subprocess.run(["sacct", "-nP", "-j", ",".join(identifiers), "--format",
        "JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS,TotalCPU"],
        capture_output=True, text=True)
    (args.run_root / "accounting.psv").write_text(query.stdout)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "source_commit": receipt["source_commit"], "jobs": receipt["jobs"],
              "reserved_cpu_hours": receipt["reserved_cpu_hours"],
              "reserved_gpu_hours": receipt["reserved_gpu_hours"],
              "accounting_query_returncode": query.returncode,
              "accounting_query_stderr": query.stderr, "tasks": {},
              "scope": "development results; conditional references for broader tasks remain pending",
              "reserved_test_evaluated": False}
    for model in ("heston", "rough-known", "rough-joint"):
        item = {}
        for label, path in (("bank", args.run_root / "banks" / model / "manifest.json"),
                            ("training", args.run_root / "fits" / model / "training.json"),
                            ("evaluation", args.run_root / "evaluation" / model / "evaluation.json")):
            if path.exists():
                item[label] = {"path": str(path), "record": json.loads(path.read_text())}
            else:
                item[label] = {"path": str(path), "missing": True}
        result["tasks"][model] = item
    (args.run_root / "overnight-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({model: {key: value.get("missing", False) for key, value in data.items()}
                      for model, data in result["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()

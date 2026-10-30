"""Collect follow-up artifacts and accounting after all scheduled jobs terminate."""

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
    ids = [job["job_id"] for job in receipt["jobs"]]
    accounting = subprocess.run(["sacct", "-nP", "-j", ",".join(ids), "--format",
        "JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS"], text=True, capture_output=True)
    (args.run_root / "accounting.psv").write_text(accounting.stdout)
    report = {"audit_utc": datetime.now(timezone.utc).isoformat(), "source_commit": receipt["source_commit"],
              "budget_admission": receipt["budget"], "jobs": receipt["jobs"], "artifacts": {},
              "reserved_test_evaluated": False, "baseline_replaced": False}
    paths = ["comparison/heston/comparison.json", "comparison/rough-joint/comparison.json",
             "pmmh-summary/summary.json", "noise/panel.json", "conditional-heston/comparison.json"]
    paths += [f"candidates/{model}/training.json" for model in ("heston", "rough-joint")]
    for name in paths:
        path = args.run_root / name
        report["artifacts"][name] = json.loads(path.read_text()) if path.exists() else {"missing": True}
    (args.run_root / "followup-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

"""Run one predeclared curve-family price challenge with bounded fitting effort."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from sbi4finance.calibration import curve_adequacy, tasks  # noqa: E402
from sbi4finance.gbm_bank import _sha256, _write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[p["name"] for p in curve_adequacy.panel()], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=271026)
    parser.add_argument("--fit-paths", type=int, default=2048)
    parser.add_argument("--audit-paths", type=int, default=32768)
    parser.add_argument("--fit-evaluations", type=int, default=80)
    parser.add_argument("--audit-replicates", type=int, default=2)
    parser.add_argument("--grid-factor", type=int, default=1)
    args = parser.parse_args()
    task = tasks.joint_rough_task() if args.config is None else json.loads(args.config.read_text())["task"]
    case = next(p for p in curve_adequacy.panel() if p["name"] == args.case)
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out/"request.json", {"case": case, "task": task,
                                          "settings": {k: v for k, v in vars(args).items() if k not in ("out", "config")}})
    report = curve_adequacy.run_case(task, case, seed=args.seed, fit_paths=args.fit_paths,
                                      audit_paths=args.audit_paths, fit_evaluations=args.fit_evaluations,
                                      audit_replicates=args.audit_replicates, grid_factor=args.grid_factor)
    report["source_sha256"]["scripts/check_curve_adequacy.py"] = _sha256(Path(__file__))
    _write_json(args.out/"report.json", report)
    print(json.dumps({"case": args.case, "status": report["status"], "out": str(args.out),
                      "timing_seconds": report["timing_seconds"], "scientific_acceptance": False}))
    if report["status"] != "completed_price_screen":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

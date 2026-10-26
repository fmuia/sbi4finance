"""Run one predeclared curve-family price challenge with bounded fitting effort."""

import argparse
import copy
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
    parser.add_argument("--replay-report", type=Path,
                        help="Reuse the exact observations/design/noise/grid from an earlier development report.")
    parser.add_argument("--initial-parameters", nargs=6, type=float,
                        help="Physical H, eta, rho, short variance, long variance and decay optimizer start.")
    parser.add_argument("--audit-namespace", default="", help="Fresh audit streams; does not change observations or fit stream.")
    parser.add_argument("--decay-upper", type=float, help="Candidate prior-bound sensitivity only; no production-prior approval.")
    args = parser.parse_args()
    source = None if args.replay_report is None else json.loads(args.replay_report.read_text())
    if args.config is not None:
        task = json.loads(args.config.read_text())["task"]
    elif source is not None:
        task = copy.deepcopy(source["task"])
    else:
        task = tasks.joint_rough_task()
    if args.decay_upper is not None:
        task["prior_bounds"][5][1] = args.decay_upper
    case = next(p for p in curve_adequacy.panel() if p["name"] == args.case)
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out/"request.json", {"case": case, "task": task,
                                          "source_report_sha256": None if source is None else _sha256(args.replay_report),
                                          "settings": {k: v for k, v in vars(args).items()
                                                       if k not in ("out", "config", "replay_report")}})
    report = curve_adequacy.run_case(task, case, seed=args.seed, fit_paths=args.fit_paths,
                                      audit_paths=args.audit_paths, fit_evaluations=args.fit_evaluations,
                                      audit_replicates=args.audit_replicates, grid_factor=args.grid_factor,
                                      replay_report=source, initial_parameters=args.initial_parameters,
                                      audit_namespace=args.audit_namespace)
    if source is not None:
        report["replay"]["source_report_file_sha256"] = _sha256(args.replay_report)
    report["source_sha256"]["scripts/check_curve_adequacy.py"] = _sha256(Path(__file__))
    _write_json(args.out/"report.json", report)
    print(json.dumps({"case": args.case, "status": report["status"], "out": str(args.out),
                      "timing_seconds": report["timing_seconds"], "scientific_acceptance": False}))
    if report["status"] != "completed_price_screen":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Declare, execute one component, or reduce the reviewed 44-call rough pilot.

Declare on the execution host, before any pricing. Components are indexed
0..31 (adaptive), 32..35 (anchor 0), 36..39 (anchor 4), 40..43 (anchor 6).
Within each point: working before refined; fixed anchors have rep 0 then 1.
The refined case-4 components 38/39 have the reviewed 90-minute limit.

Optional reducer accounting is a JSON list of records with component_index,
state, elapsed_seconds, total_cpu_seconds and peak_rss_bytes. Missing accounting
is left unknown; a missing report alone does not prove a numerical failure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration import rough_working_grid_pilot as pilot, tasks  # noqa: E402
from sbi4finance.gbm_bank import _write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)
    declare = subparsers.add_parser("declare")
    declare.add_argument("--config", type=Path, default=ROOT / "configs/tasks/calibration_rough_joint_pilot.json")
    declare.add_argument("--revision", required=True, help="Frozen execution source revision.")
    declare.add_argument("--out", type=Path, required=True, help="New declaration directory.")
    run = subparsers.add_parser("run")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--component-index", type=int, required=True)
    run.add_argument("--out", type=Path, required=True, help="New component directory; never resumed.")
    reduce = subparsers.add_parser("reduce")
    reduce.add_argument("--declaration", type=Path, required=True)
    reduce.add_argument("--components", type=Path, required=True, help="Parent of component-00 ... component-43.")
    reduce.add_argument("--accounting", type=Path)
    reduce.add_argument("--out", type=Path, required=True, help="New reduction directory.")
    args = parser.parse_args()
    if args.role == "declare":
        task = json.loads(args.config.read_text())["task"]
        reviewed = json.loads((ROOT / "configs/tasks/calibration_rough_joint_pilot.json").read_text())["task"]
        tasks.validate_task(task)
        if task != reviewed:
            raise ValueError("the reviewed full 63-contract candidate task is required; no silent prior/grid change")
        declaration = pilot.declare(task, source_revision=args.revision)
        args.out.mkdir(parents=True, exist_ok=False)
        _write_json(args.out / "declaration.json", declaration)
        print(json.dumps({"declaration": str(args.out / "declaration.json"),
                          "sha256": declaration["declaration_sha256"],
                          "calls": len(declaration["components"]), "seed_audit": declaration["seed_audit"],
                          "component_wall_allocation_seconds": declaration["component_wall_allocation_seconds"]}))
        return
    declaration = json.loads(args.declaration.read_text())
    if args.role == "run":
        def terminated(number, frame):
            raise SystemExit(f"received signal {number}; no automatic retry")
        previous = signal.signal(signal.SIGTERM, terminated)
        try:
            report = pilot.run_component(declaration, args.component_index, args.out)
        finally:
            signal.signal(signal.SIGTERM, previous)
        print(json.dumps({"component_index": args.component_index, "status": report["status"],
                          "instrumented_cpu_seconds": report["instrumented_cpu_seconds"],
                          "summed_block_cpu_seconds": report["summed_block_cpu_seconds"]}))
        if report["status"] != "priced":
            raise SystemExit(2)
        return
    reports, provenance = [], []
    for component in declaration["components"]:
        index = component["component_index"]
        path = args.components / f"component-{index:02d}" / "report.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
            provenance.append({"component_index": index, "path": str(path),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    accounting = [] if args.accounting is None else json.loads(args.accounting.read_text())
    report = pilot.reduce_components(declaration, reports, accounting=accounting)
    report["input_artifacts"] = provenance
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "reports_found": len(reports),
                      "by_policy": report["by_policy"], "out": str(args.out)}))


if __name__ == "__main__":
    main()

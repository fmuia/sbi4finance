"""Fixed-budget rough grid pilot. Run one predeclared case per CPU array task.

Example: pixi run --locked -e core python scripts/pilot_rough_numerics.py
  --config configs/tasks/calibration_rough_joint_pilot.json --case 0 --out NEW_DIR
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbi4finance.calibration import bank, full_surface, numerical_pilot  # noqa: E402
from sbi4finance.gbm_bank import _json, _write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True, help="Predeclared index 0–7; stable identity uses case contents.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=32768)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=271026)
    parser.add_argument("--grids", default="1,2,4", help="Global increasing grid multipliers, e.g. 4,8.")
    parser.add_argument("--only-grid", type=int, help="Run only this globally declared grid.")
    parser.add_argument("--only-replicate", type=int, help="Run only this global zero-based replicate ID.")
    parser.add_argument("--refine-block", type=int, help="Refine only this zero-based block; requires --fixed-other-grid.")
    parser.add_argument("--fixed-other-grid", type=int, help="Fixed multiplier for blocks other than --refine-block.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    task = full_surface.validate_task(config["task"])
    panel = numerical_pilot.cases(task)
    if not 0 <= args.case < len(panel):
        raise ValueError("case must select a predeclared panel index")
    settings = {"paths": args.paths, "replicates": args.replicates,
                "multipliers": [int(value) for value in args.grids.split(",")], "root_seed": args.seed,
                "refine_block": args.refine_block, "fixed_other_grid": args.fixed_other_grid}
    allocations = {str(grid): numerical_pilot.block_scales(task, grid, refine_block=args.refine_block,
                                                           fixed_other_grid=args.fixed_other_grid)
                   for grid in settings["multipliers"]}
    seed_table = [{"case": case["name"], "grid_multiplier": grid, "replicate": replicate,
                   "block_multipliers": allocations[str(grid)],
                   "pricing_seeds": [numerical_pilot.stream_seed(task, case, args.seed, grid, replicate, b,
                       block_multipliers=allocations[str(grid)] if args.refine_block is not None else None)
                                     for b in range(len(task["numerics"]["blocks"]))]}
                  for case in panel for grid in settings["multipliers"] for replicate in range(args.replicates)]
    reduced_seeds = [seed for row in seed_table for seed in row["pricing_seeds"]]
    if len(set(reduced_seeds)) != len(reduced_seeds):
        raise ValueError("actual reduced pricing-seed collision in the declared panel")
    declaration = {"format": numerical_pilot.FORMAT, "task": task, "cases": panel,
                   "settings": settings, "case_index": args.case,
                   "declared_block_multipliers": allocations,
                   "component_selection": {"grid_multiplier": args.only_grid, "replicate": args.only_replicate},
                   "actual_seed_collision_audit_passed": True, "pricing_seed_table": seed_table,
                   "source_sha256": {**bank.source_hashes(task),
                       "calibration/numerical_pilot.py": hashlib.sha256(Path(numerical_pilot.__file__).read_bytes()).hexdigest(),
                       "scripts/pilot_rough_numerics.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                   "versions": {"python": platform.python_version(),
                                **{key: metadata.version(key) for key in ("numpy", "scipy")}},
                   "target": "grid-development price comparisons, not production bank or posterior acceptance"}
    declaration["declaration_sha256"] = hashlib.sha256(_json(declaration).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    # Persist the entire selected panel and settings before computing a price.
    _write_json(args.out / "declaration.json", declaration)
    def progress(report):
        _write_json(args.out / "partial.json", report)
        row = report["runs"][-1]
        print(json.dumps({key: row.get(key) for key in
                          ("grid_multiplier", "replicate", "status", "cpu_seconds", "max_mc_se_over_quote_sigma")}), flush=True)
    report = numerical_pilot.run_case(task, panel[args.case], progress=progress,
                                      only_grid=args.only_grid, only_replicate=args.only_replicate, **settings)
    report["declaration_sha256"] = declaration["declaration_sha256"]
    _write_json(args.out / "report.json", report)
    print(json.dumps({"case": panel[args.case]["name"], "status": report["status"],
                      "cpu_seconds": report["total_cpu_seconds"], "out": str(args.out)}))


if __name__ == "__main__":
    main()

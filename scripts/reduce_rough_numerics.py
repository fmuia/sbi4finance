"""Reduce exact independent grid/replicate components for one predeclared case."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration import numerical_pilot  # noqa: E402
from sbi4finance.gbm_bank import _json, _write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    declarations, reports, provenance = [], [], []
    for directory in args.components:
        declaration = json.loads((directory / "declaration.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        expected = dict(declaration)
        recorded = expected.pop("declaration_sha256")
        if (hashlib.sha256(_json(expected).encode()).hexdigest() != recorded
                or report.get("declaration_sha256") != recorded):
            raise ValueError("component declaration/report identity mismatch")
        if declarations:
            for key in ("format", "task", "cases", "settings", "case_index", "source_sha256", "versions"):
                if declaration[key] != declarations[0][key]:
                    raise ValueError(f"component declaration {key} differs")
        if declaration.get("component_selection") != report.get("component_selection"):
            raise ValueError("component selection differs from its declaration")
        declarations.append(declaration)
        reports.append(report)
        provenance.append({"directory": str(directory), "declaration_sha256": recorded,
                           "report_sha256": hashlib.sha256((directory / "report.json").read_bytes()).hexdigest()})
    first = declarations[0]
    settings = first["settings"]
    for report, declaration in zip(reports, declarations, strict=True):
        if (report["paths_per_grid_replicate"] != settings["paths"]
                or report["replicates"] != settings["replicates"]
                or report["grid_multipliers"] != settings["multipliers"]
                or report["root_seed"] != settings["root_seed"]
                or report.get("block_refinement", {"refine_block": None, "fixed_other_grid": None})
                   != {key: settings.get(key) for key in ("refine_block", "fixed_other_grid")}
                or ("declared_block_multipliers" in declaration
                    and report.get("declared_block_multipliers") != declaration["declared_block_multipliers"])):
            raise ValueError("component settings differ from their declaration")
    result = numerical_pilot.combine_components(first["task"], first["cases"][first["case_index"]], reports)
    result["component_provenance"] = provenance
    result["reducer_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "report.json", result)
    print(json.dumps({"case": result["case"]["name"], "status": result["status"],
                      "components": len(reports), "cpu_seconds": result["total_cpu_seconds"], "out": str(args.out)}))


if __name__ == "__main__":
    main()

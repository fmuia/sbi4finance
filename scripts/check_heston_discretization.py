"""Check the inference task's daily grid at the endpoints and center of its prior.

Reuses the independent continuous-time references from the kernel increment.
Numerical checks remain separate from same-discrete-model posterior validation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sbi4finance import heston
from sbi4finance.heston_bank import validate_config
import validate_heston


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/heston_inference.json"))
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True, help="new report file")
    args = parser.parse_args()
    if args.paths < 2 or args.out.exists():
        parser.error("paths>=2 and a new report file are required")
    config = json.loads(args.config.read_text())
    validate_config(config)
    task = config["task"]
    low, high = task["theta_bounds"]
    cases = []
    for index, theta in enumerate((low, np.sqrt(low * high), high)):
        result = validate_heston.run_case(
            f"theta_{theta:.6g}", {**task["parameters"], "theta": float(theta)},
            task["initial_variance"], paths=args.paths, seed=76112, stream_id=index,
            backend="numba", observations=task["n_observations"],
            horizon=task["n_observations"] * task["dt_years"], substep_levels=(1, 4, 8))
        result["daily_grid_diagnostic_passed"] = bool(
            result["levels"][0]["reference_checks_passed"]
            and all(row["passed"] for row in result["paired_refinement"])
            and result["ode_reference"]["passed"])
        cases.append(result)
    report = {
        "data_origin": "synthetic", "evidence_scope": "S task-specific numerical diagnostic",
        "task": task, "paths_per_case": args.paths, "substep_levels": [1, 4, 8],
        "absolute_budgets": validate_heston.BUDGETS,
        "mc_standard_error_multiplier": validate_heston.SE_MULTIPLIER,
        "source_sha256": {"heston.py": hashlib.sha256(Path(heston.__file__).read_bytes()).hexdigest(),
                          "validate_heston.py": hashlib.sha256(Path(validate_heston.__file__).read_bytes()).hexdigest(),
                          "check_heston_discretization.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "cases": cases, "passed": all(case["daily_grid_diagnostic_passed"] for case in cases),
        "limits": "Three prior locations, moments and selected transforms only. Budget plus sampling allowance is not a proven discretization-bias bound. Inference reference and network target the daily discrete law regardless of this check.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"passed": report["passed"], "cases": [
        {"name": row["case"], "passed": row["daily_grid_diagnostic_passed"],
         "negative_terminal_auxiliary_fraction": row["levels"][0]["negative_terminal_auxiliary_fraction"]}
        for row in cases]}, indent=2))
    if not report["passed"]:
        raise SystemExit("Daily-grid diagnostic failed; do not claim continuous-time fidelity.")


if __name__ == "__main__":
    main()

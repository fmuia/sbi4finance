"""Run the frozen paired-grid study; finite-grid resolution is not a bias bound."""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from sbi4finance.bergomi_refinement import coupled_option_panels
from validate_bergomi import write_report


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(panels, pairs, budget, multiplier):
    """Select the predeclared differences, preserving contract order."""
    available = {}
    for group in (panels["adjacent"], panels["to_finest"]):
        for index, pair in enumerate(group["step_pairs"]):
            available[tuple(map(int, pair))] = (group, index)
    rows = []
    for coarse, fine in pairs:
        group, index = available[(coarse, fine)]
        difference = group["differences"][index]
        se = group["paired_standard_errors"][index]
        envelope = np.abs(difference) + multiplier * se
        rows.append({"coarse_steps": coarse, "fine_steps": fine,
                     "difference": difference, "paired_standard_errors": se,
                     "independent_standard_errors": group["independent_standard_errors"][index],
                     "variance_reduction": group["variance_reduction"][index],
                     "resolution_envelope": envelope,
                     "resolved_within_budget": envelope <= budget})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/bergomi_refinement_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/bergomi-coupled-refinement.json")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; preserve the recorded run and select a new path")
    protocol = json.loads(args.protocol.read_text())
    if protocol["protocol_id"] != "bergomi_coupled_refinement_v1":
        raise ValueError("unsupported protocol")
    configs = []
    for item in protocol["tasks"]:
        path = ROOT / item["config"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"task changed after protocol freeze: {item['config']}")
        task = json.loads(path.read_text())["task"]
        if (task["measure"] != "Q" or task["model"] not in ("one_factor", "rough")
                or len(task["parameter_names"]) != 2
                or np.asarray(task["prior_bounds"]).shape != (2, 2)):
            raise ValueError("expected restricted two-parameter Q tasks")
        configs.append(task)
    started = time.perf_counter()
    source_files = ["src/sbi4finance/bergomi.py", "src/sbi4finance/rough_bergomi.py",
                    "src/sbi4finance/bergomi_refinement.py",
                    "scripts/check_bergomi_coupled_refinement.py", "scripts/validate_bergomi.py"]
    source_start = {name: digest(ROOT / name) for name in source_files}
    models = []
    for model_index, task in enumerate(configs):
        names = task["parameter_names"]
        bounds = np.asarray(task["prior_bounds"], float)
        points = list(itertools.product(*bounds)) + [tuple(bounds.mean(axis=1))]
        context = task["context"]
        budget = protocol["absolute_budget_over_quote_sigma"] * context["quote_noise_std"]
        cases = []
        for case_index, values in enumerate(points):
            parameters = dict(task["fixed_parameters"], **dict(zip(names, values)))
            seed = protocol["root_seed"] + 100 * model_index + case_index
            tick = time.perf_counter()
            panels = coupled_option_panels(
                task["model"], parameters, task["model_context"], context["contracts"],
                spot=context["spot"], rate=context["rate"], dividend=context["dividend"],
                n_paths=protocol["n_paths"], steps=tuple(protocol["grids"]), seed=seed,
                chunk_size=protocol["chunk_size"])
            comparisons = compare(panels, protocol["comparisons"], budget, protocol["mc_se_multiplier"])
            case = {"case_index": case_index, "role": "center" if case_index == 4 else "prior_corner",
                    "parameters": parameters, "seed": seed, "seconds": time.perf_counter() - tick,
                    "panels": panels, "comparisons": comparisons,
                    "marginal_sampling_within_budget": panels["standard_errors"] <= budget}
            cases.append(case)
            print(json.dumps({"model": task["model"], "case": case_index,
                              "max_base_to_finest_envelope": float(np.max(comparisons[-1]["resolution_envelope"])),
                              "max_last_refinement_envelope": float(np.max(comparisons[-2]["resolution_envelope"])),
                              "seconds": case["seconds"]}), flush=True)
        summary = {
            "all_comparisons_resolved": all(np.all(r["resolved_within_budget"]) for c in cases for r in c["comparisons"]),
            "base_to_finest_resolved": all(np.all(c["comparisons"][-1]["resolved_within_budget"]) for c in cases),
            "last_refinement_resolved": all(np.all(c["comparisons"][-2]["resolved_within_budget"]) for c in cases),
            "all_marginal_sampling_within_budget": all(np.all(c["marginal_sampling_within_budget"]) for c in cases),
            "max_base_to_finest_envelope": max(float(np.max(c["comparisons"][-1]["resolution_envelope"])) for c in cases),
            "max_last_refinement_envelope": max(float(np.max(c["comparisons"][-2]["resolution_envelope"])) for c in cases),
        }
        item = protocol["tasks"][model_index]
        models.append({"model": task["model"], "task": task, "config_file": item["config"],
                       "config_sha256": item["sha256"], "absolute_budget_currency": budget,
                       "cases": cases, "summary": summary})
    if source_start != {name: digest(ROOT / name) for name in source_files}:
        raise RuntimeError("study source changed during execution")
    report = {"protocol": protocol, "protocol_sha256": digest(args.protocol),
              "models": models, "study_completed": True,
              "finite_grid_resolution_passed": all(m["summary"]["all_comparisons_resolved"] for m in models),
              "numerical_bias_certified": False, "posterior_validated": False,
              "elapsed_seconds": time.perf_counter() - started,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": platform.machine(), "scipy": scipy.__version__},
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "limits": protocol["limitations"]}
    write_report(args.output, report, source_files)
    print(json.dumps({"output": str(args.output), "finite_grid_resolution_passed": report["finite_grid_resolution_passed"],
                      "numerical_bias_certified": False, "seconds": report["elapsed_seconds"]}), flush=True)
    # A completed diagnostic with a failed resolution screen is still a saved
    # experiment, not an accepted bank or a reason to discard its result.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

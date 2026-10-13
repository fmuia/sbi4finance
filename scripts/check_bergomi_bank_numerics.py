"""Frozen config-specific corner/center price-refinement screen, not a bias bound."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from sbi4finance.bergomi import option_panel
from validate_bergomi import comparison, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    task = config["task"]
    names, bounds = task["parameter_names"], np.asarray(task["prior_bounds"], float)
    if len(names) != 2 or bounds.shape != (2, 2) or task["model"] not in ("one_factor", "rough"):
        raise ValueError("this frozen study requires a two-target Bergomi task")
    settings, context = task["numerics"], task["context"]
    if settings["n_steps"] != 128 or settings["n_paths"] != 262144:
        raise ValueError("frozen screen expects bank128steps and262144paths")
    points = list(itertools.product(*bounds)) + [tuple(bounds.mean(axis=1))]
    cases = []
    absolute_budget = .1 * context["quote_noise_std"]
    for i, values in enumerate(points):
        parameters = dict(task["fixed_parameters"], **dict(zip(names, values)))
        estimates = []
        for grid in (128, 512):
            tick = time.perf_counter()
            panel = option_panel(task["model"], parameters, task["model_context"], context["contracts"],
                                 spot=context["spot"], rate=context["rate"], dividend=context["dividend"],
                                 n_paths=262144, n_steps=grid, seed=731901+1000*i+grid,
                                 method="hybrid")
            panel["seconds"] = time.perf_counter()-tick
            estimates.append(panel)
        compared = comparison(*estimates, absolute_budget=absolute_budget)
        cases.append({"role": "center" if i == 4 else "prior_corner", "parameters": parameters,
                      "estimates": estimates, "refinement": compared})
        print(json.dumps({"case": i, "model": task["model"], "passed": compared["passed"],
                          "max_difference_currency": float(np.max(abs(compared["price_difference_second_minus_first"]))) }), flush=True)
    report = {"scope": "five config-specific synthetic corner/center refinement screens; no posterior acceptance",
              "task": task, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "config_file": args.config.name,
              "predeclared_design": {"base_grid": 128, "fine_grid": 512, "paths_per_run": 262144,
                                     "seeds": "731901 + 1000*case_index + grid_steps",
                                     "absolute_budget_currency": absolute_budget, "mc_se_multiplier": 4,
                                     "paired_paths": False, "redraws_or_tuning": False},
              "cases": cases, "passed": all(c["refinement"]["passed"] for c in cases),
              "limits": ["Independent run MC SE can mask bias below its resolution; passing does not prove bias below .025 currency.",
                         "512-step prices retain kernel/asset discretization bias; no continuum oracle is claimed.",
                         "Corner checks do not cover every prior draw. Bank labels remain numerical approximations.",
                         "Sampling covariance, quote noise, and discretization diagnostics are separate."],
              "elapsed_seconds": time.perf_counter()-started}
    sources = ["src/sbi4finance/bergomi.py", "scripts/check_bergomi_bank_numerics.py", "scripts/validate_bergomi.py"]
    if task["model"] == "rough":
        sources.append("src/sbi4finance/rough_bergomi.py")
    write_report(args.output, report, sources)
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "seconds": report["elapsed_seconds"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

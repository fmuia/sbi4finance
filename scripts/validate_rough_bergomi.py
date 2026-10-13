"""Rough-driver covariance, normalization and bounded option-price checks."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from sbi4finance.bergomi import option_panel
from sbi4finance.rough_bergomi import (hybrid_covariance, rough_covariance,
                                     rough_driver, simulate_rough)
from validate_bergomi import comparison, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/results/rough-bergomi-validation.json"))
    parser.add_argument("--paths", type=int, default=32768)
    args = parser.parse_args()
    started = time.perf_counter()
    curve = {"times": [0., .25, .5], "values": [.035, .04, .045]}
    contracts = [{"strike": k, "maturity": t, "option_type": "put"} for t in (.25, .5) for k in (.9, 1., 1.1)]
    cases = [{"hurst": .1, "eta": 1., "rho": -.7},
             {"hurst": .07, "eta": 1.5, "rho": -.9},
             {"hurst": .3, "eta": .8, "rho": -.3}]
    results = []
    driver_checks = []
    for case_id, parameters in enumerate(cases):
        h = parameters["hurst"]
        covariance, cross = rough_covariance(h, .5/32, 32)
        hybrid, hybrid_cross = hybrid_covariance(h, .5/32, 32)
        normalized_error = np.max(abs(hybrid-covariance)/np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal())))
        driver_checks.append({"hurst": h, "steps": 32, "horizon": .5,
                              "hybrid_max_standardized_covariance_error": normalized_error,
                              "hybrid_max_asset_cross_covariance_error": np.max(abs(hybrid_cross-cross)),
                              "passed": bool(normalized_error < .02 and np.max(abs(hybrid_cross-cross)) < 1e-12)})
        grids = []
        for steps in (32, 64, 128):
            tick = time.perf_counter()
            panel = option_panel("rough", parameters, curve, contracts, n_paths=args.paths,
                                 n_steps=steps, seed=91901+1000*case_id+steps)
            panel["seconds"] = time.perf_counter()-tick
            grids.append(panel)
        tick = time.perf_counter()
        reference = option_panel("rough", parameters, curve, contracts, n_paths=args.paths,
                                 n_steps=64, seed=92371+1000*case_id, method="covariance")
        reference["seconds"] = time.perf_counter()-tick
        results.append({"parameters": parameters, "hybrid_grids": grids, "covariance_reference": reference,
                        "refinements": [comparison(grids[0], grids[1]), comparison(grids[1], grids[2])],
                        "same_grid_reference_comparison": comparison(grids[1], reference)})
    # Independent Gaussian draws verify the reference's W dependence and both
    # normalizers. Analytic Gaussian/lognormal variances justify these SEs.
    n, steps, dt = 100000, 8, .5/8
    z = np.random.default_rng(93701).standard_normal((n, steps, 3))
    c, cross = rough_covariance(.1, dt, steps)
    y, _ = rough_driver(.1, dt, z, method="covariance")
    gaussian_error = np.max(abs(y.T@y/n-c)/np.sqrt((np.outer(c.diagonal(), c.diagonal())+c*c)/n))
    cross_error = np.max(abs(y.T@(np.sqrt(dt)*z[..., 0])/n-cross)/np.sqrt((c.diagonal()[:, None]*dt+cross*cross)/n))
    means = []
    for method in ("hybrid", "covariance"):
        simulation = simulate_rough(cases[0], curve, dt, z, method=method)
        target = np.interp(simulation["times"][1:], curve["times"], curve["values"])
        q = simulation["driver_variance"][1:]
        se = target*np.sqrt(np.expm1(q)/n)
        z_error = (simulation["variance"][:, 1:].mean(axis=0)-target)/se
        means.append({"method": method, "times": simulation["times"][1:], "target_mean": target,
                      "empirical_mean": simulation["variance"][:, 1:].mean(axis=0),
                      "analytic_mc_standard_error": se, "z_errors": z_error,
                      "passed": bool(np.max(abs(z_error)) < 5)})
    moments = {"paths": n, "seed": 93701, "gaussian_max_covariance_z_error": gaussian_error,
               "gaussian_max_asset_cross_z_error": cross_error, "normalization": means,
               "passed": bool(gaussian_error < 5 and cross_error < 5 and all(c["passed"] for c in means))}
    report = {"scope": "synthetic bounded Q numerical fixtures; no posterior or market evidence",
              "model": "rough_bergomi_volterra", "curve": curve, "contracts": contracts,
              "predeclared_budgets": {"absolute_price_in_spot_units": .002, "mc_se_multiplier": 4,
                                      "standardized_gaussian_covariance_error": .02, "moment_z_limit": 5},
              "driver_checks": driver_checks, "moments": moments, "cases": results,
              "passed": bool(moments["passed"] and all(d["passed"] for d in driver_checks)
                             and all(c["same_grid_reference_comparison"]["passed"] and all(r["passed"] for r in c["refinements"]) for c in results)),
              "limits": ["Covariance reference is exact for Gaussian grid values, not asset option prices.",
                         "Hybrid uses its own covariance normalization; residual kernel and asset-grid bias remain.",
                         "Refinement differences do not bound continuum bias or certify arbitrary parameter draws.",
                         "Bounded conditional puts justify sampling SE; raw call payoff SE is not used.",
                         "No rough-memory continuation from scalar variance, P forecast, or neural posterior is implemented."],
              "elapsed_seconds": time.perf_counter()-started}
    write_report(args.output, report, ["src/sbi4finance/bergomi.py", "src/sbi4finance/rough_bergomi.py",
                                      "scripts/validate_bergomi.py", "scripts/validate_rough_bergomi.py"])
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "seconds": report["elapsed_seconds"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

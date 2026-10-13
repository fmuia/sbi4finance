"""Bounded one-factor numerical fixtures; no posterior or empirical validation."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sbi4finance.bergomi import option_panel, simulate_one_factor


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def comparison(first, second, absolute_budget=.002):
    difference = second["prices"] - first["prices"]
    combined_se = np.hypot(first["standard_errors"], second["standard_errors"])
    return {"price_difference_second_minus_first": difference,
            "mc_standard_error_of_difference": combined_se,
            "absolute_numerical_budget": absolute_budget,
            "allowance": absolute_budget + 4 * combined_se,
            "passed": bool(np.all(abs(difference) <= absolute_budget + 4 * combined_se)),
            "interpretation": "independent runs; refinement screen, not a bound on continuum bias"}


def write_report(path, report, sources):
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    report["numpy_version"] = np.__version__
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/results/bergomi-validation.json"))
    parser.add_argument("--paths", type=int, default=32768)
    args = parser.parse_args()
    started = time.perf_counter()
    curve = {"times": [0., .25, .5], "values": [.035, .04, .045]}
    contracts = [{"strike": k, "maturity": t, "option_type": "put"} for t in (.25, .5) for k in (.9, 1., 1.1)]
    cases = [{"kappa": 2., "eta": 1., "rho": -.6},
             {"kappa": .5, "eta": .5, "rho": -.2},
             {"kappa": 4., "eta": 1.5, "rho": -.9}]
    results = []
    for case_id, parameters in enumerate(cases):
        grids = []
        for steps in (32, 64, 128):
            seed = 81801 + 1000*case_id + steps
            tick = time.perf_counter()
            panel = option_panel("one_factor", parameters, curve, contracts,
                                 n_paths=args.paths, n_steps=steps, seed=seed)
            panel["seconds"] = time.perf_counter() - tick
            grids.append(panel)
        results.append({"parameters": parameters, "grids": grids,
                        "refinements": [comparison(grids[0], grids[1]), comparison(grids[1], grids[2])]})
    # A large one-step independent moment fixture, separate from price draws.
    n, dt, k, eta = 100000, .3, 2., 1.
    z = np.random.default_rng(82719).standard_normal((n, 1, 3))
    paths = simulate_one_factor(cases[0], curve, [dt], z)
    x, dw = paths["factor_state"][:, 1, 0], np.sqrt(dt)*z[:, 0, 0]
    q = -np.expm1(-2*k*dt)/(2*k)
    cross = -np.expm1(-k*dt)/k
    xi = float(np.interp(dt, curve["times"], curve["values"]))
    moments = {
        "paths": n, "seed": 82719, "dt_years": dt,
        "factor_variance": q, "factor_empirical_variance": float(np.mean(x*x)),
        "factor_variance_z_error": (np.mean(x*x)-q)/(q*np.sqrt(2/n)),
        "brownian_cross_covariance": cross, "brownian_empirical_cross_covariance": float(np.mean(x*dw)),
        "cross_covariance_z_error": (np.mean(x*dw)-cross)/np.sqrt((q*dt+cross*cross)/n),
        "forward_variance": xi, "empirical_variance_mean": paths["variance"][:, 1].mean(),
        "variance_mean_z_error": (paths["variance"][:, 1].mean()-xi)/(xi*np.sqrt(np.expm1(eta*eta*q)/n)),
    }
    moments["passed"] = all(abs(moments[key]) < 5 for key in ("factor_variance_z_error", "cross_covariance_z_error", "variance_mean_z_error"))
    report = {"scope": "synthetic bounded Q numerical fixtures; no posterior or market evidence",
              "model": "one_factor_bergomi", "curve": curve, "contracts": contracts,
              "predeclared_budgets": {"absolute_price_in_spot_units": .002, "mc_se_multiplier": 4, "moment_z_limit": 5},
              "moments": moments, "cases": results,
              "passed": bool(moments["passed"] and all(r["passed"] for c in results for r in c["refinements"])),
              "limits": ["Exact OU only; price integration remains approximate.",
                         "Refinement differences do not bound remaining bias and do not validate every prior draw.",
                         "MC covariance is price-estimator covariance, separate from quote noise.",
                         "Bounded conditional puts justify sampling SE; raw call payoff SE is not used.",
                         "Q simulation and continuation are not P historical forecasting."],
              "elapsed_seconds": time.perf_counter()-started}
    write_report(args.output, report, ["src/sbi4finance/bergomi.py", "scripts/validate_bergomi.py"])
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "seconds": report["elapsed_seconds"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

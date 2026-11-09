"""Audit Heston quadrature on the actual raw-quote pilot design.

Run from lab/ with the pilot environment, for example:
  pixi run -e core-reference python scripts/audit_heston_pilot_precision.py \
      --design /path/to/design.json --out /path/to/pricing_precision.json

The panel is fixed before prices are inspected: 32 Sobol prior draws, all
32 prior corners, the box center, and a declared reference parameter vector.
Every failure is retained.  Adaptive quadrature shares the model CF and Lewis
inversion formula, so it is a second integration method, not an independent
financial-model implementation.  A pointwise pass is not posterior stability.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import qmc

from sbi4finance.calibration.heston_pilot import HestonPilot, REQUIRED_COLUMNS


SEED = 20261109
TRUTH = (.04, 2., .04, .6, -.7)


def read_design(path):
    design = json.loads(Path(path).read_text())
    if "records" in design:
        records = design["records"]
    elif "quotes" in design:
        rows = design["quotes"]
        records = {key: [row[key] for row in rows] for key in REQUIRED_COLUMNS}
    else:
        raise ValueError("design needs records arrays or actual quote rows")
    settings = {key: value for key, value in design.get("pilot_settings", {}).items()
                if key in ("bounds", "quadrature_ladder", "numerical_fraction", "spread_multiplier")}
    return design, HestonPilot(records, **settings)


def panel(pilot, sobol_points=32):
    if sobol_points < 1 or sobol_points & (sobol_points - 1):
        raise ValueError("Sobol point count must be a positive power of two")
    bounds = pilot.bounds
    rng = qmc.Sobol(d=5, scramble=True, seed=SEED)
    unit = rng.random_base2(int(np.log2(sobol_points)))
    rows = [{"id": f"sobol:{i}", "parameters": row.tolist()}
            for i, row in enumerate(qmc.scale(unit, bounds[:, 0], bounds[:, 1]))]
    rows.extend({"id": f"corner:{i}", "parameters": list(row)}
                for i, row in enumerate(itertools.product(*bounds.tolist())))
    rows += [{"id": "center", "parameters": bounds.mean(axis=1).tolist()},
             {"id": "reference", "parameters": list(TRUTH)}]
    return rows


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(pilot, *, sobol_points=32, independent_contracts=8):
    rows = panel(pilot, sobol_points)
    report = {
        "schema": "raw_heston_pointwise_precision_v1", "seed": SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameter_names": pilot.parameter_names, "bounds": pilot.bounds.tolist(),
        "number_of_contracts": len(pilot.records["mid"]),
        "measurement_sigma_range": [float(pilot.measurement_sigma.min()), float(pilot.measurement_sigma.max())],
        "spread_multiplier": pilot.spread_multiplier,
        "numerical_fraction": pilot.numerical_fraction,
        "quadrature_ladder": pilot.quadrature_ladder,
        "panel": "seeded scrambled Sobol draws; all physical-prior corners; center; declared reference",
        "versions": {package: metadata.version(package) for package in ("numpy", "scipy")},
        "python": platform.python_version(), "platform": platform.platform(),
        "scope": "pointwise pricing and likelihood sensitivity; neither posterior stability nor model adequacy",
        "posterior_stability_assessed": False,
        "points": [], "independent_quadrature": [],
    }
    started = time.perf_counter()
    for row in rows:
        point = dict(row)
        before = time.perf_counter()
        try:
            result = pilot.refinement_report(row["parameters"])
            # Shifted adaptive ladders can converge on the same accepted rung.
            # An audit must actually go beyond that rung, including at difficult
            # prior corners; repeating its successful comparison adds no evidence.
            base_control = result["base_diagnostics"]
            refined_control = result["refined_diagnostics"]
            if (refined_control["nodes"] <= base_control["nodes"]
                    or refined_control["cutoff"] <= base_control["cutoff"]):
                nodes, cutoff = 2 * base_control["nodes"], 2 * base_control["cutoff"]
                indices = np.arange(len(pilot.records["mid"]))
                refined = pilot._rung_prices(row["parameters"], indices, nodes, cutoff)
                base = result["base_prices"]
                sigma = pilot.observation_sigma(row["parameters"])
                y = pilot.records["mid"]
                shift = refined - base
                ratio = float(np.max(np.abs(shift) / pilot.measurement_sigma))
                result.update(refined_prices=refined, price_shift=shift,
                              max_abs_price_shift_over_measurement_sigma=ratio,
                              log_likelihood_shift=float(-.5 * np.sum(((y - refined) / sigma)**2 - ((y - base) / sigma)**2)),
                              refined_diagnostics={"nodes": nodes, "cutoff": cutoff,
                                  "base_nodes": base_control["nodes"], "base_cutoff": base_control["cutoff"],
                                  "explicit_beyond_accepted_rung": True,
                                  "max_difference_over_measurement_sigma": ratio})
            point.update(
                priced=True,
                max_abs_price_shift_over_measurement_sigma=result["max_abs_price_shift_over_measurement_sigma"],
                max_abs_price_shift=float(np.max(np.abs(result["price_shift"]))),
                worst_contract_index=int(np.argmax(np.abs(result["price_shift"]) / pilot.measurement_sigma)),
                log_likelihood_shift=result["log_likelihood_shift"],
                base_diagnostics={key: value for key, value in result["base_diagnostics"].items() if key != "difference"},
                refined_diagnostics={key: value for key, value in result["refined_diagnostics"].items() if key != "difference"},
                pointwise_price_gate_passed=result["max_abs_price_shift_over_measurement_sigma"] <= pilot.numerical_fraction,
            )
        except Exception as error:
            point.update(priced=False, error_type=type(error).__name__, error=str(error),
                         pointwise_price_gate_passed=False)
        point["wall_seconds"] = time.perf_counter() - before
        report["points"].append(point)
        print(json.dumps({key: point[key] for key in ("id", "priced", "wall_seconds")}), flush=True)

    n = len(pilot.records["mid"])
    chosen = np.unique(np.linspace(0, n - 1, min(independent_contracts, n), dtype=int))
    # Last corner is not the strongest short-maturity case: choose low variance,
    # low kappa, high xi and strong negative correlation explicitly by names.
    stress = [pilot.bounds[0, 0], pilot.bounds[1, 0], pilot.bounds[2, 0],
              pilot.bounds[3, 1], pilot.bounds[4, 0]]
    independent_panel = [dict(id="center", parameters=pilot.bounds.mean(axis=1).tolist()),
                         dict(id="reference", parameters=list(TRUTH)),
                         dict(id="low_variance_high_xi", parameters=stress)]
    for row in independent_panel:
        point = {**row, "contract_indices": chosen.tolist(),
                 "scope": "adaptive scipy.quad on [0,infinity); shared characteristic function and Lewis formula"}
        before = time.perf_counter()
        try:
            actual = pilot.prices(row["parameters"], chosen)
            reference, estimates = pilot.independent_prices(row["parameters"], chosen)
            ratio = np.abs(actual - reference) / pilot.measurement_sigma[chosen]
            point.update(priced=True, prices=actual.tolist(), independent_prices=reference.tolist(),
                         independent_error_estimates=estimates.tolist(),
                         max_abs_difference_over_measurement_sigma=float(ratio.max()),
                         price_gate_passed=bool(np.all(ratio <= pilot.numerical_fraction)))
        except Exception as error:
            point.update(priced=False, error_type=type(error).__name__, error=str(error), price_gate_passed=False)
        point["wall_seconds"] = time.perf_counter() - before
        report["independent_quadrature"].append(point)
        print(json.dumps({key: point[key] for key in ("id", "priced", "wall_seconds")}), flush=True)
    priced = [row for row in report["points"] if row["priced"]]
    report.update(
        wall_seconds=time.perf_counter() - started,
        all_points_priced=len(priced) == len(rows),
        failed_point_ids=[row["id"] for row in report["points"] if not row["priced"]],
        max_abs_price_shift_over_measurement_sigma=max(
            (row["max_abs_price_shift_over_measurement_sigma"] for row in priced), default=None),
        max_abs_log_likelihood_shift=max((abs(row["log_likelihood_shift"]) for row in priced), default=None),
        pointwise_numerical_gates_passed=all(row["pointwise_price_gate_passed"] for row in report["points"])
            and all(row["price_gate_passed"] for row in report["independent_quadrature"]),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sobol-points", type=int, default=32)
    parser.add_argument("--independent-contracts", type=int, default=8)
    args = parser.parse_args()
    if args.independent_contracts < 1:
        parser.error("at least one independent contract is required")
    _, pilot = read_design(args.design)
    report = audit(pilot, sobol_points=args.sobol_points, independent_contracts=args.independent_contracts)
    from sbi4finance.calibration import heston_pilot
    from sbi4f_tutorial.heston_options import model
    sources = [Path(__file__), Path(heston_pilot.__file__), Path(model.__file__)]
    report["source_sha256"] = {str(path): _digest(path) for path in sources}
    report["design_path"] = str(args.design.resolve())
    report["design_sha256"] = _digest(args.design)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "all_points_priced", "max_abs_price_shift_over_measurement_sigma",
        "max_abs_log_likelihood_shift", "pointwise_numerical_gates_passed", "wall_seconds")}), flush=True)
    if not report["pointwise_numerical_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

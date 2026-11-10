"""Audit preserved clean labels for the new one-scale Heston likelihood.

This is a small numerical/engineering probe, not training or posterior acceptance.
Only named validation rows are read; the reserved test arrays are never accessed.
Run from lab/ with an exact historical design and its immutable clean bank.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi4finance.calibration.heston_spread import (
    HestonSpreadPilot, conditional_c_log_density, conditional_c_mode,
)
from sbi4f_tutorial.heston_options import model


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(design_path, bank_path, out, rows=(0, 1, 2), seed=261110):
    rows = list(rows)
    if (not rows or len(rows) > 16 or len(set(rows)) != len(rows)
            or any(type(i) is not int or i < 0 for i in rows)):
        raise ValueError("select 1-16 distinct nonnegative validation-row indices")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    design_path, bank_path, out = map(Path, (design_path, bank_path, out))
    design = json.loads(design_path.read_text())
    bank = json.loads((bank_path / "bank.json").read_text())
    source_task = bank["identity"]["task"]
    if digest(design_path) != source_task["design_sha256"]:
        raise ValueError("historical bank does not bind this exact design")
    payload = bank_path / bank["payload"]
    if digest(payload) != bank["payload_sha256"]:
        raise ValueError("historical bank payload hash differs")
    fixed = HestonSpreadPilot(design["records"], c=1.)
    joint = HestonSpreadPilot(design["records"], infer_c=True)
    if (source_task["parameter_names"][:5] != fixed.names
            or not np.array_equal(source_task["prior_bounds"][:5], fixed.bounds)
            or source_task["pricing_controls"]["measurement_spread_multiplier"] != .5
            or source_task["pricing_controls"]["numerical_fraction"] != fixed.numerical_fraction
            or source_task["pricing_controls"]["quadrature_ladder"]
                != [list(rung) for rung in fixed.quadrature_ladder]):
        raise ValueError("physical prior or pricing gate differs from preserved producer")
    with np.load(payload, allow_pickle=False) as values:
        count = len(values["validation_parameters"])
        if max(rows) >= count:
            raise ValueError("validation row outside preserved pool")
        physical = values["validation_parameters"][rows].copy()
        clean = values["validation_clean"][rows].copy()
    if physical.shape != (len(rows), 5) or clean.shape != (len(rows), len(fixed.records["mid"])):
        raise ValueError("preserved validation arrays have incompatible dimensions")
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"format": "heston_spread_engineering_probe_v1", "seed": seed,
        "validation_rows": rows, "arrays_read": ["validation_parameters", "validation_clean"],
        "reserved_test_arrays_opened": False, "design_sha256": digest(design_path),
        "bank_manifest_sha256": digest(bank_path / "bank.json"),
        "bank_payload_sha256": bank["payload_sha256"], "fixed_task": fixed.task_definition(),
        "joint_task": joint.task_definition(),
        "source_sha256": {"probe": digest(__file__),
            "spread_likelihood": digest(sys.modules[HestonSpreadPilot.__module__].__file__),
            "checked_pricing_support": digest(Path(__file__).resolve().parents[1]
                / "src/sbi4finance/calibration/heston_pilot.py"), "runtime_pricer": digest(model.__file__)},
        "versions": {"python": platform.python_version(),
            **{name: metadata.version(name) for name in ("numpy", "scipy")}},
        "scope": "Sampled label-reuse and likelihood probe; no MCMC, training, SBC or scientific acceptance."}
    write_json(out / "run.json", provenance)
    cases, results = [], []
    try:
        for j, (row, phi, prices) in enumerate(zip(rows, physical, clean)):
            generated = fixed.simulate(phi, seed=seed+j, clean_prices=prices)[0]
            actual, diagnostics = fixed.prices(phi, return_diagnostics=True)
            h = fixed.records["half_spread"]
            discrepancy = float(np.max(np.abs(actual-prices)/(.5*h)))
            refinement = fixed.refinement_report(phi, observations=generated)
            fixed_ll = fixed.log_likelihood(phi, observations=generated)
            joint_ll = joint.log_likelihood(np.r_[phi, 1.], observations=generated)
            residual = (generated-actual)/h
            grid = np.linspace(.5, 2., 5001)
            density = np.exp(conditional_c_log_density(grid, residual))
            integral = float(np.trapezoid(density, grid))
            mode = conditional_c_mode(residual)
            metrics = {"label_shift": discrepancy,
                "refinement_shift": refinement["max_abs_price_shift_over_measurement_sigma"],
                "refinement_log_likelihood_shift": refinement["log_likelihood_shift"],
                "fixed_log_likelihood": fixed_ll, "joint_log_likelihood": joint_ll,
                "conditional_c_integral": integral, "conditional_c_mode": mode}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"nonfinite engineering probe metrics: {metrics!r}")
            passed = bool(discrepancy <= fixed.numerical_fraction
                and refinement["max_abs_price_shift_over_measurement_sigma"] <= fixed.numerical_fraction
                and np.isclose(fixed_ll, joint_ll, rtol=0., atol=1e-10)
                and abs(integral-1.) < 1e-5)
            cases.append({"validation_row": row, "physical_parameters": phi.tolist(),
                "generating_c": 1., "observations": generated.tolist(),
                "scope": "One fixed observation reused for known-c sensitivity and joint-c checks."})
            results.append({"validation_row": row, "checks_passed": passed,
                "bank_price_shift_over_minimum_measurement_sd": discrepancy,
                "refined_price_shift_over_minimum_measurement_sd":
                    refinement["max_abs_price_shift_over_measurement_sigma"],
                "refined_log_likelihood_shift": refinement["log_likelihood_shift"],
                "fixed_c1_log_likelihood": fixed_ll, "joint_at_c1_log_likelihood": joint_ll,
                "conditional_c_mode_at_true_phi": mode,
                "conditional_c_density_grid_integral": integral,
                "pricing_nodes": diagnostics["nodes"], "pricing_cutoff": diagnostics["cutoff"]})
        write_json(out / "cases.json", {"cases": cases})
        summary = {"format": "heston_spread_engineering_probe_results_v1", "cases": results,
            "checks_passed": all(case["checks_passed"] for case in results),
            "reserved_test_arrays_opened": False, "scientific_acceptance": False,
            "case_file_sha256": digest(out / "cases.json"),
            "scope": provenance["scope"]}
        write_json(out / "summary.json", summary)
        if not summary["checks_passed"]:
            raise FloatingPointError("sampled numerical or normalized-likelihood probe failed")
        return summary
    except Exception as error:
        write_json(out / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "completed_cases": results, "scientific_acceptance": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=261110)
    args = parser.parse_args()
    result = run(args.design, args.bank, args.out, args.rows, args.seed)
    print(json.dumps({"out": str(args.out), "cases": len(result["cases"]),
                      "checks_passed": result["checks_passed"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()

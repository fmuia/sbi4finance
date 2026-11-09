"""Pointwise numerical sensitivity on every chain of completed Heston pilots.

Select four deterministic, evenly spaced retained draws per chain by default.
Reprice beyond each accepted rung with twice its nodes and cutoff; assess the
actual conditioning mask and normalized observation law from the run record.
This does not replace an independently rerun refined posterior.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sbi4finance.calibration.heston_pilot import DEFAULT_LADDER, HestonPilot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_reference(directory, design, *, points_per_chain=4, log_likelihood_gate=.1):
    if points_per_chain < 1:
        raise ValueError("at least one point per chain required")
    root = Path(directory)
    config = json.loads((root / "run.json").read_text())
    chain_hash = digest(root / "chains.npz")
    with np.load(root / "chains.npz", allow_pickle=False) as archive:
        draws = archive["parameters"]
    if (draws.shape != (config["settings"]["chains"], config["settings"]["draws"], len(config["names"]))
            or not np.isfinite(draws).all()):
        raise ValueError("draw shape or values disagree with the completed reference record")
    ladder = config.get("quadrature_ladder", DEFAULT_LADDER)
    if config.get("extended_quadrature") and len(ladder) == len(DEFAULT_LADDER):
        ladder = (*ladder, (16384, 9600.))
    pilot = HestonPilot(design["records"], bounds=np.asarray(config["bounds"])[:5],
                        infer_discrepancy=config["infer_discrepancy"],
                        spread_multiplier=config["spread_multiplier"], tau_max=config["tau_max"],
                        quadrature_ladder=ladder)
    if pilot.names != config["names"]:
        raise ValueError("run parameter order differs from numerical audit")
    indices = (np.flatnonzero(design["short_mask"]) if config["arm"] == "short"
               else np.arange(len(pilot.records["mid"])))
    run = {"reference_path": str(root.resolve()), "arm": config["arm"],
           "names": pilot.names, "bounds": pilot.bounds.tolist(), "conditioning_quotes": len(indices),
           "run_sha256": digest(root / "run.json"), "chains_sha256": chain_hash,
           "chain_shape": list(draws.shape), "points_per_chain": points_per_chain,
           "source_mcmc_diagnostics_passed": None, "points": []}
    if (root / "diagnostics.json").exists():
        diagnostic = json.loads((root / "diagnostics.json").read_text())
        run["source_mcmc_diagnostics_passed"] = diagnostic.get("mcmc_diagnostics_passed")
    positions = np.unique(np.linspace(0, draws.shape[1] - 1, min(points_per_chain, draws.shape[1]), dtype=int))
    for chain in range(draws.shape[0]):
        for draw in positions:
            theta = draws[chain, draw]
            point = {"chain": chain, "draw": int(draw), "parameters": theta.tolist()}
            before = time.perf_counter()
            try:
                base, controls = pilot.prices(theta, indices, return_diagnostics=True)
                nodes, cutoff = 2 * controls["nodes"], 2 * controls["cutoff"]
                refined = pilot._rung_prices(theta, indices, nodes, cutoff)
                shift = refined - base
                measurement_sigma = pilot.measurement_sigma[indices]
                sigma = pilot.observation_sigma(theta, indices)
                y = pilot.records["mid"][indices]
                loglaw_shift = float(-.5 * np.sum(((y - refined) / sigma)**2 - ((y - base) / sigma)**2))
                ratio = float(np.max(np.abs(shift) / measurement_sigma))
                point.update(priced=True, base_nodes=controls["nodes"], base_cutoff=controls["cutoff"],
                             refined_nodes=nodes, refined_cutoff=cutoff,
                             max_abs_price_shift_over_measurement_sigma=ratio,
                             log_likelihood_shift=loglaw_shift,
                             pointwise_gates_passed=bool(ratio <= pilot.numerical_fraction and abs(loglaw_shift) <= log_likelihood_gate))
            except Exception as error:
                point.update(priced=False, error_type=type(error).__name__, error=str(error),
                             pointwise_gates_passed=False)
            point["wall_seconds"] = time.perf_counter() - before
            run["points"].append(point)
    priced = [point for point in run["points"] if point["priced"]]
    run.update(all_points_priced=len(priced) == len(run["points"]),
               pointwise_gates_passed=all(point["pointwise_gates_passed"] for point in run["points"]),
               max_abs_price_shift_over_measurement_sigma=max(
                   (point["max_abs_price_shift_over_measurement_sigma"] for point in priced), default=None),
               max_abs_log_likelihood_shift=max((abs(point["log_likelihood_shift"]) for point in priced), default=None))
    if digest(root / "chains.npz") != chain_hash:
        raise RuntimeError("chain payload changed during pointwise numerical audit")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points-per-chain", type=int, default=4)
    parser.add_argument("--log-likelihood-gate", type=float, default=.1)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    design_hash = digest(args.design)
    report = {"schema": "raw_heston_posterior_pointwise_precision_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "design_path": str(args.design.resolve()), "design_sha256": design_hash,
              "script_sha256": digest(__file__), "price_shift_gate_measurement_sigma": .02,
              "log_likelihood_shift_gate": args.log_likelihood_gate,
              "posterior_stability_assessed": False,
              "scope": "pointwise numerical sensitivity on retained chains; does not establish refined posterior stability",
              "references": []}
    for directory in args.reference:
        config = json.loads((directory / "run.json").read_text())
        if config["design_sha256"] != design_hash:
            raise ValueError("reference conditions differ from numerical audit design")
        run = audit_reference(directory, design, points_per_chain=args.points_per_chain,
                              log_likelihood_gate=args.log_likelihood_gate)
        report["references"].append(run)
        print(json.dumps({key: run[key] for key in (
            "reference_path", "all_points_priced", "max_abs_price_shift_over_measurement_sigma",
            "max_abs_log_likelihood_shift", "pointwise_gates_passed")}), flush=True)
    report["pointwise_gates_passed"] = all(run["pointwise_gates_passed"] for run in report["references"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["pointwise_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

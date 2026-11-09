"""Pointwise precision and exact observation binding for synthetic references.

Audit the six initially declared development references once all are complete.
Each reference supplies its own arbitrary finite Gaussian observation vector;
the frozen market mids only define design metadata and observed vegas. Neither
negative synthetic observations nor failed numerical refinements are discarded.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np

from sbi4finance.calibration.heston_pilot import HestonPilot


NAMES = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "tau"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def bind_reference(directory, design, design_hash, panel, panel_hash, index):
    """Bind observations to their declared truth/noise, exact files and chains."""
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text())
    case = json.loads((directory / "case.json").read_text())
    diagnostic = json.loads((directory / "diagnostics.json").read_text())
    declared = [item for item in panel["cases"] if item["index"] == index]
    if len(declared) != 1:
        raise ValueError("index must select exactly one frozen panel case")
    declared = declared[0]
    if (run["format"] != "heston_training_conditional_reference_v1" or run["arm"] != "full"
            or run["index"] != index or case["index"] != index
            or run["design_sha256"] != design_hash or case["design_sha256"] != design_hash
            or run["panel_sha256"] != panel_hash or case["panel_sha256"] != panel_hash
            or panel["design_sha256"] != design_hash
            or run["case_json_sha256"] != digest(directory / "case.json")
            or diagnostic["case_json_sha256"] != digest(directory / "case.json")
            or run["names"] != NAMES or panel["names"] != NAMES
            or run["bounds"] != panel["task_bounds"]
            or run["spread_multiplier"] != 1. or not run["infer_discrepancy"]):
        raise ValueError("synthetic reference design, panel, case or physical-prior binding differs")
    for key in ("index", "id", "category", "parameters", "tau_sampling_bounds",
                "spread_multiplier", "standard_normal_errors"):
        if case[key] != declared[key]:
            raise ValueError(f"synthetic reference case changed declared {key}")
    bounds = np.asarray(run["bounds"], dtype=float)
    if bounds.shape != (6, 2) or run["tau_max"] != bounds[-1, 1] or bounds[-1, 0] != 0:
        raise ValueError("synthetic reference discrepancy support differs")
    pilot = HestonPilot(design["records"], bounds=bounds[:5], infer_discrepancy=True,
                        spread_multiplier=run["spread_multiplier"], tau_max=run["tau_max"],
                        quadrature_ladder=run["quadrature_ladder"])
    if len(pilot.records["mid"]) != 54:
        raise ValueError("the complete frozen 54-quote layout is required")
    if run["source_sha256"]["pilot"] != digest(sys.modules[HestonPilot.__module__].__file__):
        raise ValueError("current pilot source differs from the sampled reference producer")
    observed = np.asarray(run["observations"], dtype=float)
    if (observed.shape != (54,) or not np.isfinite(observed).all()
            or not np.array_equal(observed, np.asarray(case["observations"], dtype=float))
            or values_digest(observed) != run["observation_sha256"]
            or values_digest(observed) != case["observation_sha256"]):
        raise ValueError("synthetic observations or their float64 hashes differ")
    truth = np.asarray(declared["parameters"], dtype=float)
    errors = np.asarray(declared["standard_normal_errors"], dtype=float)
    if (truth.shape != (6,) or errors.shape != (54,) or not np.isfinite(truth).all()
            or not np.isfinite(errors).all() or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1])):
        raise ValueError("declared truth or standard-normal errors disagree with task support")
    clean = pilot.prices(truth)
    recomposed = clean + errors * pilot.observation_sigma(truth)
    if (not np.array_equal(clean, np.asarray(case["clean_prices"], dtype=float))
            or not np.array_equal(recomposed, observed)):
        raise ValueError("synthetic observations do not reproduce the declared pricing/noise experiment")
    chain_hash = digest(directory / "chains.npz")
    if chain_hash != diagnostic["chain_payload_sha256"]:
        raise ValueError("reference draws differ from diagnosed payload")
    with np.load(directory / "chains.npz", allow_pickle=False) as archive:
        draws = archive["parameters"]
    shape = (run["settings"]["chains"], run["settings"]["draws"], len(NAMES))
    if (draws.shape != shape or not np.isfinite(draws).all()
            or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
        raise ValueError("reference chain/draw shape or finite physical support differs")
    identity = {"run_sha256": digest(directory / "run.json"), "case_json_sha256": digest(directory / "case.json"),
                "chains_sha256": chain_hash, "diagnostics_sha256": digest(directory / "diagnostics.json"),
                "observation_sha256": values_digest(observed),
                "declared_observation_recomposition_exact": True,
                "negative_observation_count": int(np.sum(observed < 0)),
                "reference_source_sha256": run["source_sha256"]}
    return pilot, observed, draws, diagnostic, identity


def audit_one(directory, design, design_hash, panel, panel_hash, index, *, points=16, likelihood_gate=.1):
    pilot, observed, draws, diagnostic, identity = bind_reference(
        directory, design, design_hash, panel, panel_hash, index)
    if points < draws.shape[0]:
        raise ValueError("numerical audit must cover every retained chain")
    chosen = []
    for chain in range(draws.shape[0]):
        count = points // draws.shape[0] + (chain < points % draws.shape[0])
        chosen.extend((chain, int(draw)) for draw in np.unique(np.linspace(0, draws.shape[1]-1,
                                                                            min(count, draws.shape[1]), dtype=int)))
    result = {"index": index, "path": str(Path(directory).resolve()), "identity": identity,
              "chain_shape": list(draws.shape), "source_mcmc_diagnostics_passed": diagnostic["mcmc_diagnostics_passed"],
              "selection": "evenly spaced retained draws within every chain; no posterior-point selection by pricing outcome",
              "points": []}
    for chain, draw in chosen:
        theta = draws[chain, draw]
        point = {"chain": chain, "draw": draw, "parameters": theta.tolist()}
        try:
            base, controls = pilot.prices(theta, return_diagnostics=True)
            nodes, cutoff = 2*controls["nodes"], 2*controls["cutoff"]
            refined = pilot._rung_prices(theta, np.arange(54), nodes, cutoff)
            measurement_sigma = pilot.measurement_sigma
            sigma = pilot.observation_sigma(theta)
            normalization = np.log(2*np.pi*sigma**2)
            base_loglik = float(-.5*np.sum(((observed-base)/sigma)**2 + normalization))
            refined_loglik = float(-.5*np.sum(((observed-refined)/sigma)**2 + normalization))
            delta = refined_loglik-base_loglik
            ratio = float(np.max(np.abs(refined-base)/measurement_sigma))
            point.update(priced=True, base_nodes=controls["nodes"], base_cutoff=controls["cutoff"],
                         refined_nodes=nodes, refined_cutoff=cutoff,
                         max_price_shift_measurement_sigma=ratio,
                         base_log_likelihood=base_loglik, refined_log_likelihood=refined_loglik,
                         log_likelihood_shift=delta,
                         pointwise_gates_passed=bool(ratio <= pilot.numerical_fraction and abs(delta) <= likelihood_gate))
        except Exception as error:
            point.update(priced=False, error_type=type(error).__name__, error=str(error), pointwise_gates_passed=False)
        result["points"].append(point)
    priced = [point for point in result["points"] if point["priced"]]
    result.update(all_points_priced=len(priced) == len(chosen),
                  pointwise_gates_passed=all(point["pointwise_gates_passed"] for point in result["points"]),
                  max_price_shift_measurement_sigma=max((point["max_price_shift_measurement_sigma"] for point in priced), default=None),
                  max_abs_log_likelihood_shift=max((abs(point["log_likelihood_shift"]) for point in priced), default=None))
    if digest(Path(directory)/"chains.npz") != identity["chains_sha256"]:
        raise RuntimeError("chain payload changed during numerical audit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--points", type=int, default=16)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    panel = json.loads(args.panel.read_text())
    indices = panel["initial_reference_indices"] if args.indices is None else args.indices
    if len(set(indices)) != len(indices):
        raise ValueError("distinct declared reference indices required")
    required = ("case.json", "run.json", "chains.npz", "diagnostics.json")
    missing = {index: [name for name in required if not (args.references/f"case-{index:03d}"/name).exists()]
               for index in indices}
    missing = {index: names for index, names in missing.items() if names}
    if missing:
        raise RuntimeError(f"selected references are not complete; no partial final audit written: {missing}")
    from sbi4f_tutorial.heston_options import model
    report = {"schema": "heston_training_reference_pointwise_precision_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "design_sha256": digest(args.design), "panel_sha256": digest(args.panel),
              "indices": indices, "requested_points_per_reference": args.points,
              "source_sha256": {"auditor": digest(__file__), "pilot": digest(sys.modules[HestonPilot.__module__].__file__),
                                "heston_model": digest(model.__file__)},
              "versions": {name: metadata.version(name) for name in ("numpy", "scipy")},
              "gates": {"price_shift_measurement_sigma": .02, "absolute_log_likelihood_shift": .1},
              "observation_source": "each run's exactly bound arbitrary finite synthetic observations; not frozen market mids",
              "posterior_stability_assessed": False, "scientific_acceptance": False,
              "scope": "pointwise deterministic likelihood precision, separate from MCMC convergence, neural agreement, posterior stability and market-model adequacy",
              "references": []}
    for index in indices:
        result = audit_one(args.references/f"case-{index:03d}", design, report["design_sha256"],
                           panel, report["panel_sha256"], index, points=args.points)
        report["references"].append(result)
        print(json.dumps({key: result[key] for key in ("index", "all_points_priced", "pointwise_gates_passed",
                                                      "max_price_shift_measurement_sigma", "max_abs_log_likelihood_shift")}), flush=True)
    report["all_pointwise_gates_passed"] = all(row["pointwise_gates_passed"] for row in report["references"])
    report["all_source_mcmc_diagnostics_passed"] = all(row["source_mcmc_diagnostics_passed"] for row in report["references"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if not report["all_pointwise_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

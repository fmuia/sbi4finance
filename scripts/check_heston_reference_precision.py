"""Pointwise Fourier refinement checks on completed PyMC Heston draws.

This reprices a deterministic, bounded selection of chain/draw points. It does
not rerun MCMC or establish posterior stability from small price differences.
An unconverged source chain remains inconclusive even when every point prices.
"""

import argparse
import copy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbi4finance.calibration import bank, full_surface, tasks  # noqa: E402
from sbi4finance.gbm_bank import _json  # noqa: E402


def selected_indices(shape, points=16):
    """Evenly spaced draws within every chain, with no randomized selection."""
    if (len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape)
            or type(points) is not int or not 1 <= points <= 20 or points < shape[0]):
        raise ValueError("chain/draw/parameter axes and 1–20 points covering every chain are required")
    chains, draws, _ = shape
    count = min(points, chains * draws)
    result = []
    for chain in range(chains):
        per_chain = count // chains + (chain < count % chains)
        result.extend((chain, int(draw)) for draw in np.linspace(0, draws - 1, per_chain, dtype=int))
    return result


def refined_task(task):
    """Double both nodes and cutoff at every rung; retain the recorded budget."""
    full_surface.validate_task(task)
    if task["model"] != "hkde":
        raise ValueError("precision check supports only the full-surface Heston task")
    result = copy.deepcopy(task)
    result["numerics"]["quadrature_ladder"] = [
        [2 * nodes, 2.0 * cutoff] for nodes, cutoff in task["numerics"]["quadrature_ladder"]]
    full_surface.validate_task(result)
    return result


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prices(task, parameters):
    price, _, _, difference, _ = tasks.price(task, parameters, np.empty(0), seed=0)
    price, difference = np.asarray(price), np.asarray(difference)
    if (price.shape != (63,) or difference.shape != (63,) or np.iscomplexobj(price)
            or np.iscomplexobj(difference) or not np.isfinite(price).all() or not np.isfinite(difference).all()):
        raise ValueError("invalid Fourier prices or checked-ladder differences")
    return price, difference


def check(reference_directory, points=16):
    """Return a pointwise report, preserving every selected pricing failure."""
    root = Path(reference_directory)
    inputs = {name: _digest(root / name) for name in ("reference.json", "chains.npz")}
    record = json.loads((root / "reference.json").read_text())
    if record.get("format") != "full_surface_pymc_reference_v1":
        raise ValueError("a recorded full-surface PyMC reference is required")
    task = record["task"]
    finer = refined_task(task)
    quotes = np.asarray(record["observed_prices"])
    if quotes.shape != (63,) or np.iscomplexobj(quotes) or not np.isfinite(quotes).all():
        raise ValueError("63 complete finite real observed prices are required")
    with np.load(root / "chains.npz", allow_pickle=False) as archive:
        parameters = archive["parameters"]
    bounds = np.asarray(task["prior_bounds"])
    if (parameters.ndim != 3 or parameters.shape[-1] != len(task["parameter_names"])
            or parameters.shape[:2] != (record["settings"]["chains"], record["settings"]["draws"])
            or np.iscomplexobj(parameters) or not np.isfinite(parameters).all()
            or np.any(parameters < bounds[:, 0]) or np.any(parameters > bounds[:, 1])):
        raise ValueError("finite chain/draw/parameter arrays within the recorded prior and settings are required")
    indices = selected_indices(parameters.shape, points)
    diagnostics_path = root / "diagnostics.json"
    diagnostics = None
    if diagnostics_path.exists():
        inputs["diagnostics.json"] = _digest(diagnostics_path)
        diagnostics = json.loads(diagnostics_path.read_text())
    runtime = {**bank.versions(), "machine": platform.machine(), "platform": platform.platform(),
               "numba": metadata.version("numba")}
    sources = {"scripts/check_heston_reference_precision.py": _digest(Path(__file__)), **bank.source_hashes(task)}
    report = {
        "format": "heston_reference_pointwise_precision_v1", "scientific_acceptance": False,
        "posterior_stability_assessed": False, "mcmc_rerun": False,
        "source_mcmc_diagnostics_passed": None if diagnostics is None else diagnostics.get("mcmc_diagnostics_passed"),
        "source_surface_id": record.get("surface_id"), "parameter_names": task["parameter_names"],
        "selection": "evenly spaced draw indices within every chain; correlated and not comprehensive",
        "chain_shape": list(parameters.shape), "input_sha256": inputs,
        "source_sha256": sources, "source_bundle_sha256": hashlib.sha256(_json(sources).encode()).hexdigest(),
        "runtime": runtime, "runtime_sha256": hashlib.sha256(_json(runtime).encode()).hexdigest(),
        "environment_sha256": {name: _digest(ROOT / name) for name in ("pyproject.toml", "pixi.lock")},
        "pricing_sources_match_reference": record.get("reference_pricing_source_sha256") == bank.source_hashes(task),
        "reference_versions": record.get("versions"),
        "base_numerics": task["numerics"], "refined_numerics": finer["numerics"],
        "scope": "pointwise numerical sensitivity only; inconclusive MCMC remains inconclusive; small price shifts do not establish posterior stability",
        "points": [],
    }
    sigma = tasks.sigma(task)
    normalization = np.log(2 * np.pi * sigma**2).sum()
    started = time.perf_counter()
    for chain, draw in indices:
        point = {"chain": chain, "draw": draw, "parameters": parameters[chain, draw].tolist()}
        try:
            base, base_difference = _prices(task, parameters[chain, draw])
            refined, refined_difference = _prices(finer, parameters[chain, draw])
            price_shift = (refined - base) / sigma
            base_residual, refined_residual = (quotes - base) / sigma, (quotes - refined) / sigma
            base_loglik = float(-0.5 * (base_residual @ base_residual + normalization))
            refined_loglik = float(-0.5 * (refined_residual @ refined_residual + normalization))
            if not np.isfinite([base_loglik, refined_loglik]).all():
                raise ValueError("nonfinite Gaussian log likelihood")
            point.update(priced=True, base_prices=base.tolist(), refined_prices=refined.tolist(),
                         price_shift_over_quote_sigma=price_shift.tolist(),
                         max_abs_price_shift_over_quote_sigma=float(np.abs(price_shift).max()),
                         worst_contract_index=int(np.argmax(np.abs(price_shift))),
                         base_ladder_max_refinement_over_quote_sigma=float(np.max(np.abs(base_difference) / sigma)),
                         refined_ladder_max_refinement_over_quote_sigma=float(np.max(np.abs(refined_difference) / sigma)),
                         base_log_likelihood=base_loglik, refined_log_likelihood=refined_loglik,
                         log_likelihood_shift=refined_loglik - base_loglik)
        except (ValueError, FloatingPointError, OverflowError) as error:
            point.update(priced=False, error=str(error))
        report["points"].append(point)
    priced = [p for p in report["points"] if p["priced"]]
    report.update(all_points_priced=len(priced) == len(indices), unpriced_points=len(indices) - len(priced),
                  max_abs_price_shift_over_quote_sigma=max(
                      (p["max_abs_price_shift_over_quote_sigma"] for p in priced), default=None),
                  max_abs_log_likelihood_shift=max((abs(p["log_likelihood_shift"]) for p in priced), default=None),
                  seconds=time.perf_counter() - started)
    for name, digest in inputs.items():
        if _digest(root / name) != digest:
            raise ValueError("reference inputs changed during the precision check")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        report = check(args.reference, args.points)
    except Exception as error:
        (args.out / "failure.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
        raise
    (args.out / "precision.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "all_points_priced", "unpriced_points", "max_abs_price_shift_over_quote_sigma",
        "max_abs_log_likelihood_shift", "source_mcmc_diagnostics_passed", "scientific_acceptance")}))
    if not report["all_points_priced"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

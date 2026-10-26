"""Compare a core estimator with one completed full-surface PyMC reference.

Failed or incomplete reference diagnostics are inconclusive and never produce
an accuracy verdict for the neural estimator. Passing parameter-level checks
is development evidence, not global scientific acceptance or product evidence.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbi4finance.calibration import full_surface, validation  # noqa: E402
from sbi4finance.gbm_bank import _json, _write_json  # noqa: E402


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_issues(diagnostics, names):
    issues = []
    if diagnostics is None or diagnostics.get("mcmc_diagnostics_passed") is not True:
        issues.append("source MCMC diagnostics did not pass")
    if diagnostics is not None:
        for name in names:
            row = diagnostics.get("parameters", {}).get(f"parameters[{name}]", {})
            for key, bound, upper in (("r_hat", 1.01, True), ("ess_bulk", 400, False), ("ess_tail", 400, False)):
                value = row.get(key)
                if (not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value)
                        or (value > bound if upper else value < bound)):
                    issues.append(f"{name}: missing or failing {key}")
    return issues


def compare(checkpoint_directory, reference_directory, *, precision=None, draws=10000, seed=0):
    """Return report and optional neural draws; reference chains are never rewritten."""
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("at least two neural draws and a nonnegative integer seed required")
    checkpoint, root = Path(checkpoint_directory), Path(reference_directory)
    record_path = root / "reference.json"
    reference_hashes = {"reference.json": _digest(record_path)}
    reference = json.loads(record_path.read_text())
    if (reference.get("format") != "full_surface_pymc_reference_v1"
            or reference.get("target") != "independent Gaussian quote noise around deterministic refined Fourier prices"):
        raise ValueError("the recorded deterministic Fourier/Gaussian full-surface reference is required")
    task, workflow, training = full_surface.load_checkpoint(checkpoint)
    fingerprint = hashlib.sha256(_json(task).encode()).hexdigest()
    if task != reference["task"] or training["task_fingerprint"] != fingerprint:
        raise ValueError("reference/checkpoint tasks differ: exact contracts, noise, prior and numerics must match")
    quotes = np.asarray(reference["observed_prices"], dtype=np.float64)
    if quotes.shape != (63,) or not np.isfinite(quotes).all():
        raise ValueError("reference must record the complete finite observed 63-price vector")
    # Reuse the same strict complete-input contract as sampling.
    full_surface.observations(quotes[None, :], task)
    names = task["parameter_names"]
    diagnostics = None
    for filename in ("chains.npz", "diagnostics.json", "failure.json"):
        if (root / filename).exists():
            reference_hashes[filename] = _digest(root / filename)
    if "diagnostics.json" in reference_hashes:
        diagnostics = json.loads((root / "diagnostics.json").read_text())
    issues = _reference_issues(diagnostics, names)
    if "failure.json" in reference_hashes:
        issues.append("source reference retained a failure record")
    chains = None
    if "chains.npz" not in reference_hashes:
        issues.append("completed reference chains are missing")
    else:
        with np.load(root / "chains.npz", allow_pickle=False) as archive:
            chains = archive["parameters"]
        bounds = np.asarray(task["prior_bounds"])
        expected = (reference["settings"]["chains"], reference["settings"]["draws"], len(names))
        if (chains.shape != expected or len(expected) != 3 or min(expected[:2]) < 2
                or np.iscomplexobj(chains) or not np.isfinite(chains).all()
                or np.any(chains < bounds[:, 0]) or np.any(chains > bounds[:, 1])):
            raise ValueError("finite chain/draw/parameter arrays must match reference settings and prior support")

    precision_report = None
    precision_issues = []
    if precision is not None:
        precision = Path(precision)
        precision_report = json.loads(precision.read_text())
        if (precision_report.get("format") != "heston_reference_pointwise_precision_v1"
                or precision_report.get("source_surface_id") != reference.get("surface_id")
                or precision_report.get("parameter_names") != names
                or precision_report.get("base_numerics") != task["numerics"]
                or any(precision_report.get("input_sha256", {}).get(k) != reference_hashes.get(k)
                       for k in ("reference.json", "chains.npz", "diagnostics.json"))):
            raise ValueError("precision report does not identify these exact reference inputs and numerical task")
        if precision_report.get("all_points_priced") is not True:
            precision_issues.append("precision report contains unpriced points")
        if precision_report.get("pricing_sources_match_reference") is not True:
            precision_issues.append("precision pricing sources differ from the recorded reference")

    report = {
        "format": "full_surface_reference_comparison_v1", "scientific_acceptance": False,
        "status": "inconclusive_reference" if issues else "started",
        "scope": "one development conditional comparison; no global acceptance, coverage or product claim",
        "task_fingerprint": fingerprint, "task": task, "parameter_names": names,
        "surface_id": reference.get("surface_id"), "reference_split": reference.get("split"),
        "observed_prices": quotes.tolist(), "sampling_seed": seed, "neural_draws_requested": draws,
        "checkpoint_sha256": training["checkpoint_sha256"], "network": training["network"],
        "training_dataset": training["run_contract"]["dataset"],
        "checkpoint_metadata_sha256": {n: _digest(checkpoint/n) for n in ("task.json", "training.json", "model.keras")},
        "reference_input_sha256": reference_hashes, "reference_chain_shape": None if chains is None else list(chains.shape),
        "reference_diagnostics": diagnostics, "reference_issues": issues,
        "precision": None if precision_report is None else {
            "sha256": _digest(precision), "all_points_priced": precision_report.get("all_points_priced"),
            "max_abs_price_shift_over_quote_sigma": precision_report.get("max_abs_price_shift_over_quote_sigma"),
            "max_abs_log_likelihood_shift": precision_report.get("max_abs_log_likelihood_shift"),
            "posterior_stability_assessed": precision_report.get("posterior_stability_assessed"),
            "issues": precision_issues, "scope": "pointwise numerical sensitivity only; not posterior stability"},
        "proposed_parameter_agreement": None,
        "source_sha256": {"scripts/compare_full_surface_reference.py": _digest(__file__),
                          "calibration/full_surface.py": _digest(full_surface.__file__),
                          "calibration/validation.py": _digest(validation.__file__)},
        "versions": {n: metadata.version(n) for n in ("bayesflow", "keras", "jax", "numpy", "scipy")},
    }
    neural = None
    started = time.perf_counter()
    if not issues and precision_issues:
        report["status"] = "inconclusive_reference_precision"
    elif not issues:
        # Only flatten after retaining and validating chain/draw axes. These are
        # pooled posterior summaries, not an assertion of independent draws.
        reference_draws = chains.reshape(-1, len(names))
        reference_sd = reference_draws.std(axis=0)
        reference_q05, reference_q95 = np.quantile(reference_draws, [0.05, 0.95], axis=0)
        if np.any(reference_sd <= 0) or np.any(reference_q95 <= reference_q05):
            report["status"] = "inconclusive_reference"
            report["reference_issues"].append("reference posterior has zero spread")
        else:
            neural = full_surface.sample(workflow, task, quotes[None, :], draws,
                                           encoder=training["network"]["encoder"], seed=seed)[0]
            agreement = validation.agreement(neural, reference_draws)
            parameters = {}
            for j, name in enumerate(names):
                error, ratio = agreement["standardized_mean_error"][j], agreement["width_ratio"][j]
                row = diagnostics["parameters"][f"parameters[{name}]"]
                mcse_mean, mcse_sd = row.get("mcse_mean"), row.get("mcse_sd")
                parameters[name] = {
                    "standardized_mean_error": error, "width_ratio_90": ratio,
                    "mean_error_within_0_3_reference_sd": bool(error <= 0.3),
                    "relative_width_error_within_25_percent": bool(abs(ratio-1) <= 0.25),
                    "reference_mean": float(reference_draws[:, j].mean()), "reference_sd": float(reference_sd[j]),
                    "reference_q05": float(reference_q05[j]), "reference_q95": float(reference_q95[j]),
                    "reference_mcse_mean": mcse_mean, "reference_mcse_sd": mcse_sd,
                    "reference_mcse_mean_over_sd": None if mcse_mean is None else float(mcse_mean/reference_sd[j]),
                    "reference_quantile_mcse": None,
                }
            report["status"] = "completed_development_comparison"
            report["proposed_parameter_agreement"] = {
                "parameters": parameters,
                "all_checks_passed": all(p["mean_error_within_0_3_reference_sd"]
                                          and p["relative_width_error_within_25_percent"] for p in parameters.values()),
                "limitations": "proposed parameter-only rules; reference quantile MCSE, numerical posterior stability and products remain unassessed"}
    report["sampling_and_comparison_seconds"] = time.perf_counter()-started
    for filename, digest in reference_hashes.items():
        if _digest(root/filename) != digest:
            raise ValueError("reference inputs changed during comparison")
    return report, neural


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--precision", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        report, neural = compare(args.checkpoint, args.reference, precision=args.precision,
                                  draws=args.draws, seed=args.seed)
        _write_json(args.out/"comparison.json", report)
        if neural is not None:
            np.savez_compressed(args.out/"neural_draws.npz", parameters=neural)
        print(json.dumps({"status": report["status"], "scientific_acceptance": False,
                          "proposed_parameter_agreement": None if report["proposed_parameter_agreement"] is None
                          else report["proposed_parameter_agreement"]["all_checks_passed"]}))
    except Exception as error:
        _write_json(args.out/"failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()

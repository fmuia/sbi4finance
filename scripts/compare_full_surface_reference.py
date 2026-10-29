"""Compare a core estimator with one completed full-surface PyMC reference.

Failed or incomplete reference diagnostics are inconclusive and never produce
an accuracy verdict for the neural estimator. Parameter checks and descriptive
analytic variance products are development evidence, not scientific acceptance.
Rough forward-emulator references remain approximate targets: agreement does
not establish independent pricing accuracy or an exact-model posterior.
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

from sbi4finance.calibration import full_surface, validation, variance_functionals  # noqa: E402
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


def _emulator_binding(reference, task, fingerprint):
    """Verify the fixed emulator bytes, not its financial accuracy."""
    from sbi4finance.calibration import reference as reference_helpers

    expected_target = ("independent Gaussian quote noise around deterministic forward-pricer emulator; "
                       "approximate price target")
    identity = reference.get("emulator", {})
    if (task["model"] != "rough" or reference.get("target") != expected_target
            or reference.get("task_fingerprint") != fingerprint
            or reference.get("scientific_acceptance") is not False
            or identity.get("task_fingerprint") != fingerprint or not identity.get("directory")):
        raise ValueError("a task-bound approximate rough-emulator reference is required")
    _, actual = reference_helpers.load_forward_pricer(identity["directory"], task)
    if any(identity.get(key) != actual[key] for key in ("input_sha256", "activation", "training_record")):
        raise ValueError("fixed emulator input identities differ from the recorded reference")
    return actual


def _rough_derived_comparison(task, neural, chains, diagnostics):
    names = task["parameter_names"]
    eta, rho = names.index("eta"), names.index("rho")
    reference = (chains[..., eta] * chains[..., rho]).reshape(-1)
    estimated = neural[:, eta] * neural[:, rho]
    result = validation.agreement(estimated[:, None], reference[:, None])
    row = diagnostics["derived_quantities"]["parameters"]["parameters[rho_eta]"]
    sd = reference.std()
    return {"target": "rho_eta", "status": "descriptive_approximate_target_comparison",
            "scientific_acceptance": False, "standardized_mean_error": result["standardized_mean_error"][0],
            "width_ratio_90": result["width_ratio"][0], "reference_mean": float(reference.mean()),
            "neural_mean": float(estimated.mean()), "reference_sd": float(sd),
            "reference_q05_q95": np.quantile(reference, [.05, .95]).tolist(),
            "neural_q05_q95": np.quantile(estimated, [.05, .95]).tolist(),
            "reference_mcse_mean": row.get("mcse_mean"), "reference_mcse_sd": row.get("mcse_sd"),
            "reference_mcse_mean_over_sd": (None if row.get("mcse_mean") is None else float(row["mcse_mean"]/sd)),
            "reference_quantile_mcse": None, "reference_diagnostics": row,
            "scope": "Derived parameter agreement with the approximate emulator target; not product pricing, "
                     "an independent model-price check or an acceptance rule."}


def _variance_product_comparison(task, neural, chains, diagnostics):
    """Describe two analytic curve functionals; no unregistered acceptance rule."""
    reference_values = variance_functionals.heston_variance_functionals(task, chains)
    neural_values = variance_functionals.heston_variance_functionals(task, neural)
    product_diagnostics = diagnostics.get("variance_functionals", {})
    rows = {}
    for name, values in reference_values.items():
        reference = values.reshape(-1)
        estimated = neural_values[name]
        q05, q10, q90, q95 = np.quantile(reference, [.05, .10, .90, .95])
        neural_q10, neural_q90 = np.quantile(estimated, [.10, .90])
        row = product_diagnostics.get("names", {}).get(name, {})
        diagnostic_pass = all(
            isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
            and np.isfinite(row[key]) and (row[key] <= bound if upper else row[key] >= bound)
            for key, bound, upper in (("r_hat", 1.01, True), ("ess_bulk", 400, False), ("ess_tail", 400, False)))
        rows[name] = {
            "reference_diagnostics_passed": diagnostic_pass, "reference_diagnostics": row,
            "reference_mean": float(reference.mean()), "neural_mean": float(estimated.mean()),
            "reference_sd": float(reference.std()),
            "reference_q05": float(q05), "reference_q10": float(q10),
            "reference_q90": float(q90), "reference_q95": float(q95),
            "neural_q10": float(neural_q10), "neural_q90": float(neural_q90),
            "absolute_q10_error": float(abs(neural_q10-q10)),
            "absolute_q90_error": float(abs(neural_q90-q90)),
            "reference_90_percent_width": float(q95-q05),
            "reference_quantile_mcse": None,
        }
    return {
        "status": "descriptive_development_comparison" if all(
            row["reference_diagnostics_passed"] for row in rows.values()) else "inconclusive_product_reference",
        "scientific_acceptance": False, "units": "annualized variance, spot-normalized, continuous monitoring",
        "products": rows,
        "limitations": "Analytic mean-curve functionals only; no vol-of-vol/leverage or path-dependent-price check. "
                       "Absolute product tolerances and reference quantile MCSE are unassessed; no acceptance verdict.",
    }


def compare(checkpoint_directory, reference_directory, *, precision=None, draws=10000, seed=0):
    """Return report and optional neural draws; reference chains are never rewritten."""
    if type(draws) is not int or draws < 2 or type(seed) is not int or seed < 0:
        raise ValueError("at least two neural draws and a nonnegative integer seed required")
    checkpoint, root = Path(checkpoint_directory), Path(reference_directory)
    record_path = root / "reference.json"
    reference_hashes = {"reference.json": _digest(record_path)}
    reference = json.loads(record_path.read_text())
    approximate = reference.get("likelihood_kind") == "forward-pricer-emulator"
    if (reference.get("format") != "full_surface_pymc_reference_v1"
            or (not approximate and reference.get("target") !=
                "independent Gaussian quote noise around deterministic refined Fourier prices")):
        raise ValueError("a recorded Fourier/Gaussian or approximate rough-emulator reference is required")
    task, workflow, training = full_surface.load_checkpoint(checkpoint)
    fingerprint = hashlib.sha256(_json(task).encode()).hexdigest()
    if task != reference["task"] or training["task_fingerprint"] != fingerprint:
        raise ValueError("reference/checkpoint tasks differ: exact contracts, noise, prior and numerics must match")
    emulator = _emulator_binding(reference, task, fingerprint) if approximate else None
    if not approximate and (task["model"] != "hkde" or reference.get("likelihood_kind", "fourier") != "fourier"):
        raise ValueError("the deterministic Fourier reference requires the Heston task")
    if approximate and precision is not None:
        raise ValueError("Heston precision reports cannot be applied to rough-emulator references")
    quotes = np.asarray(reference["observed_prices"], dtype=np.float64)
    if quotes.shape != (63,) or not np.isfinite(quotes).all():
        raise ValueError("reference must record the complete finite observed 63-price vector")
    # Reuse the same strict complete-input contract as sampling.
    full_surface.observations(quotes[None, :], task)
    if ("observation_sha256" in reference and reference["observation_sha256"] !=
            hashlib.sha256(_json(reference["observed_prices"]).encode()).hexdigest()):
        raise ValueError("recorded observation checksum differs")
    curve = None
    if approximate:
        if full_surface.uses_known_curve(task):
            curve = np.asarray(reference.get("known_curve_levels"), dtype=np.float64)
            if curve.shape != (7,):
                raise ValueError("known rough reference requires seven recorded curve levels")
            curve = curve[None, :]
            full_surface.conditioning_data(quotes[None, :], task, context_values=curve)
        elif reference.get("known_curve_levels") is not None:
            raise ValueError("joint rough references must not observe auxiliary curve levels")
        expected_curve_hash = (None if curve is None else
            hashlib.sha256(_json(reference["known_curve_levels"]).encode()).hexdigest())
        if "known_curve_sha256" in reference and reference["known_curve_sha256"] != expected_curve_hash:
            raise ValueError("recorded known-curve checksum differs")
    names = task["parameter_names"]
    diagnostics = None
    for filename in ("chains.npz", "diagnostics.json", "failure.json"):
        if (root / filename).exists():
            reference_hashes[filename] = _digest(root / filename)
    if "diagnostics.json" in reference_hashes:
        diagnostics = json.loads((root / "diagnostics.json").read_text())
    issues = _reference_issues(diagnostics, names)
    if approximate:
        if diagnostics is not None and diagnostics.get("parameter_mcmc_diagnostics_passed", True) is not True:
            issues.append("source parameter MCMC diagnostics did not pass")
        derived = None if diagnostics is None else diagnostics.get("derived_quantities")
        issues += ["rho_eta: " + issue for issue in _reference_issues(derived, ["rho_eta"])]
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
        "scope": "one development conditional comparison; no global acceptance or coverage claim",
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
        "variance_product_comparison": None,
        "source_sha256": {"scripts/compare_full_surface_reference.py": _digest(__file__),
                          "calibration/full_surface.py": _digest(full_surface.__file__),
                          "calibration/validation.py": _digest(validation.__file__),
                          "calibration/variance_functionals.py": _digest(variance_functionals.__file__)},
        "versions": {n: metadata.version(n) for n in ("bayesflow", "keras", "jax", "numpy", "scipy")},
    }
    if approximate:
        report.update(reference_target_kind="approximate_forward_pricer_emulator", emulator=emulator,
            known_curve_levels=None if curve is None else curve[0].tolist(),
            observation_sha256=reference.get("observation_sha256"), known_curve_sha256=reference.get("known_curve_sha256"),
            independent_pricing_assessed=False, derived_parameter_comparison=None,
            scope="One development comparison with a fixed approximate emulator posterior; no exact-model accuracy claim.",
            approximation_scope="The reference uses a deterministic emulator mean with Gaussian quote noise only. "
                "The amortized estimator trains on numerical Monte Carlo bank prices plus fresh quote noise. "
                "Emulator bias and bank Monte Carlo randomness/covariance are not propagated by this reference. "
                "Independent pricing/refinement and posterior/product checks remain unassessed; even perfect "
                "emulator agreement need not be like-for-like agreement with the simulator posterior.")
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
        derived_degenerate = (approximate and np.std(chains[..., names.index("eta")] * chains[..., names.index("rho")]) <= 0)
        if np.any(reference_sd <= 0) or np.any(reference_q95 <= reference_q05) or derived_degenerate:
            report["status"] = "inconclusive_reference"
            report["reference_issues"].append("reference posterior has zero spread")
        else:
            sampling_kwargs = {} if curve is None else {"context_values": curve}
            neural = full_surface.sample(workflow, task, quotes[None, :], draws,
                                           encoder=training["network"]["encoder"], seed=seed, **sampling_kwargs)[0]
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
            report["status"] = "completed_approximate_emulator_comparison" if approximate else "completed_development_comparison"
            report["proposed_parameter_agreement"] = {
                "parameters": parameters,
                "all_checks_passed": all(p["mean_error_within_0_3_reference_sd"]
                                          and p["relative_width_error_within_25_percent"] for p in parameters.values()),
                "limitations": "proposed parameter-only rules; reference quantile MCSE, numerical posterior stability and product acceptance remain unassessed"}
            if approximate:
                report["proposed_parameter_agreement"]["limitations"] += (
                    "; these rules describe agreement with an approximate emulator target only, not exact-model accuracy")
                report["derived_parameter_comparison"] = _rough_derived_comparison(task, neural, chains, diagnostics)
            else:
                report["variance_product_comparison"] = _variance_product_comparison(task, neural, chains, diagnostics)
    report["sampling_and_comparison_seconds"] = time.perf_counter()-started
    for filename, digest in reference_hashes.items():
        if _digest(root/filename) != digest:
            raise ValueError("reference inputs changed during comparison")
    if emulator is not None and any(_digest(p) != h for p, h in emulator["input_sha256"].items()):
        raise ValueError("fixed emulator inputs changed during comparison")
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

"""Reproduce the bounded neural/reference comparison from preserved payloads.

The 2k screen and 10k follow-up reuse development cases. Neither is a fresh
acceptance evaluation. Both target the original vol-of-vol upper bound 1.5;
they must not be compared as matched inference for a widened-prior reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from run_heston_observation_pilot import expected_variance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def parameter_comparison(neural, reference, names):
    result = {}
    for i, name in enumerate(names):
        rq = np.quantile(reference[:, i], [.05, .5, .95])
        nq = np.quantile(neural[:, i], [.05, .5, .95])
        error = (neural[:, i].mean() - reference[:, i].mean()) / reference[:, i].std()
        width_error = (nq[-1] - nq[0]) / (rq[-1] - rq[0]) - 1
        result[name] = {
            "reference_q05_q50_q95": rq.tolist(), "neural_q05_q50_q95": nq.tolist(),
            "mean_error_reference_sd": float(error), "interval_width_relative_error": float(width_error),
            "passes_point_comparison": bool(abs(error) <= .3 and abs(width_error) <= .25)}
    return result


def product_comparison(neural, reference, products):
    result = {}
    for name, interval in products.items():
        rp = expected_variance(reference, interval)
        np_ = expected_variance(neural, interval)
        rq = np.quantile(rp, [.05, .1, .5, .9, .95])
        nq = np.quantile(np_, [.05, .1, .5, .9, .95])
        tolerance = max(.1 * (rq[-1] - rq[0]), .0001)
        errors = np.abs(nq[[1, 3]] - rq[[1, 3]])
        result[name] = {"interval": interval,
                        "reference_q05_q10_q50_q90_q95": rq.tolist(),
                        "neural_q05_q10_q50_q90_q95": nq.tolist(),
                        "q10_q90_absolute_error": errors.tolist(), "tolerance": float(tolerance),
                        "passes_point_comparison": bool(np.max(errors) <= tolerance)}
    return result


def synthetic_summary(draws, truth, names):
    low, high = np.quantile(draws, [.05, .95], axis=1)
    ranks = (draws < truth[:, None, :]).sum(axis=1)
    result = {}
    for i, name in enumerate(names):
        covered = (truth[:, i] >= low[:, i]) & (truth[:, i] <= high[:, i])
        test = binomtest(int(covered.sum()), len(truth), .9)
        interval = test.proportion_ci()
        result[name] = {
            "covered": int(covered.sum()), "cases": len(truth), "coverage_90": float(covered.mean()),
            "coverage_exact_ci95": [float(interval.low), float(interval.high)],
            "unadjusted_exact_p_value": float(test.pvalue),
            "rank_histogram_10bins": np.histogram(ranks[:, i], bins=np.linspace(0, draws.shape[1]+1, 11))[0].tolist(),
            "posterior_mean_rmse": float(np.sqrt(np.mean((draws[:, :, i].mean(axis=1)-truth[:, i])**2)))}
    return result


def compare(design_path, root):
    design_path, root = Path(design_path), Path(root)
    design = read_json(design_path)
    design_hash = digest(design_path)
    report = {
        "format": "heston_pilot_neural_reference_development_comparison_v1",
        "scientific_acceptance": False, "design_sha256": design_hash,
        "scope": "The 2k screen and 10k follow-up reuse 128 development cases and the same market reference. Original xi<=1.5 prior only; c=1 conditional market comparison. No fresh acceptance or widened-prior neural result.",
        "criteria": {"parameter_absolute_mean_error_reference_sd": .3,
                     "parameter_absolute_relative_90_interval_width_error": .25,
                     "product_quantile_error": "q10/q90 absolute error <= max(.10*(reference q95-q05),.0001 annualized variance units)"},
        "limitations": ["Point comparisons omit reference and neural-sampling Monte Carlo uncertainty; borderline coordinate decisions are inconclusive.",
                        "Coverage confidence intervals are marginal exact binomial intervals, not simultaneous acceptance bands.",
                        "Ten-bin rank histograms on 128 cases are descriptive development checks, not a broad SBC certification.",
                        "One frozen quote design, vegas and carry anchors; no variable-design, liquidity-selection or market-wide calibration claim."],
        "source_sha256": {"compare_heston_pilot_neural.py": digest(__file__),
                          "run_heston_observation_pilot.py": digest(Path(__file__).with_name("run_heston_observation_pilot.py"))},
        "endpoints": {}}
    for budget in (2000, 10000):
        endpoint = {}
        for arm in ("full", "short"):
            directory = root / (arm + "-neural" + ("-10k" if budget == 10000 else ""))
            reference_directory = root / ("reference-" + arm + "-v2")
            training = read_json(directory / "training.json")
            reference_run = read_json(reference_directory / "run.json")
            reference_diagnostics = read_json(reference_directory / "diagnostics.json")
            names, bounds = training["names"], np.asarray(training["bounds"])
            if (training["design_sha256"] != design_hash or reference_run["design_sha256"] != design_hash
                    or training["layout"] != arm or training["updates"] != budget
                    or reference_run["arm"] != arm
                    or names != ["initial_variance", "kappa", "theta", "vol_of_vol", "rho", "tau"]
                    or names != reference_run["names"] or bounds.tolist() != reference_run["bounds"]
                    or bounds[names.index("vol_of_vol"), 1] != 1.5
                    or reference_run["spread_multiplier"] != 1 or not reference_run["infer_discrepancy"]):
                raise ValueError("neural/reference design, priors, arm, budget or observation model differ")
            if (digest(directory / "model.keras") != training["checkpoint_sha256"]
                    or digest(directory / "posterior-draws.npz") != training["posterior_sha256"]):
                raise ValueError("preserved neural payload hash differs from training record")
            reference_hash = digest(reference_directory / "chains.npz")
            if reference_hash != reference_diagnostics["chain_payload_sha256"]:
                raise ValueError("reference payload differs from its diagnostic binding")
            bank_record = read_json(root/"bank-putheavy"/"bank.json")
            bank_hash = digest(root/"bank-putheavy"/"bank.npz")
            if bank_hash != bank_record["payload_sha256"] or bank_hash != training["bank"]["payload_sha256"]:
                raise ValueError("paired bank payload differs from its training/manifest binding")
            with np.load(reference_directory / "chains.npz") as values:
                raw_reference = values["parameters"]
            settings = reference_run["settings"]
            if (raw_reference.shape != (settings["chains"], settings["draws"], len(names))
                    or not np.isfinite(raw_reference).all()):
                raise ValueError("reference must preserve chain/draw/parameter axes")
            reference = raw_reference.reshape(-1, len(names))
            with np.load(directory / "posterior-draws.npz") as values:
                payload = {key: values[key].copy() for key in values.files}
            neural = payload["market_c1.0"]
            if neural.ndim != 2 or neural.shape[1] != len(names) or not np.isfinite(neural).all():
                raise ValueError("finite market draws in matching physical parameter order required")
            arrays = {key: value for key, value in payload.items() if key == "test" or key.startswith("market_")}
            result = {
                "training": {key: training[key] for key in ("layout", "names", "bounds", "architecture", "seed",
                            "updates", "batch_size", "endpoint_selection", "history", "training_seconds", "same_seed_reload_exact", "versions", "source_sha256")},
                "identity": {"training_record_sha256": digest(directory/"training.json"),
                             "checkpoint_sha256": training["checkpoint_sha256"],
                             "neural_draws_sha256": training["posterior_sha256"],
                             "bank_payload_sha256": training["bank"]["payload_sha256"],
                             "bank_record_sha256": digest(root/"bank-putheavy"/"bank.json"),
                             "bank_split_counts": training["bank"]["identity"]["splits"],
                             "bank_pricing_seconds": training["bank"]["pricing_seconds"],
                             "reference_run_sha256": digest(reference_directory/"run.json"),
                             "reference_chains_sha256": digest(reference_directory/"chains.npz"),
                             "reference_diagnostics_sha256": digest(reference_directory/"diagnostics.json")},
                "reference_mcmc_diagnostics_passed": reference_diagnostics["mcmc_diagnostics_passed"],
                "parameters": parameter_comparison(neural, reference, names),
                "products": product_comparison(neural, reference, design["products"]),
                "synthetic_calibration": synthetic_summary(payload["test"], payload["test_truth"], names),
                "numeric_boundary_draw_counts": {key: int(np.any((value <= bounds[:, 0]) | (value >= bounds[:, 1]), axis=-1).sum()) for key, value in arrays.items()},
                "outside_prior_draw_counts": {key: int(np.any((value < bounds[:, 0]) | (value > bounds[:, 1]), axis=-1).sum()) for key, value in arrays.items()}}
            result["conditional_point_comparisons_passed"] = bool(
                all(value["passes_point_comparison"] for value in result["parameters"].values())
                and all(value["passes_point_comparison"] for value in result["products"].values()))
            result["conclusion"] = (
                "reference convergence failed; conditional comparison inconclusive" if not result["reference_mcmc_diagnostics_passed"] else
                "conditional validation failed; preserve as development evidence" if not result["conditional_point_comparisons_passed"] else
                "point comparisons pass; fresh acceptance and Monte Carlo precision remain unverified")
            endpoint[arm] = result
        report["endpoints"][str(budget)] = endpoint
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Preserved local Heston pilot payload directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.design, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "scientific_acceptance": False,
                      "conditional_point_comparisons_passed": {budget: {arm: value["conditional_point_comparisons_passed"]
                                                                         for arm, value in endpoint.items()}
                                                               for budget, endpoint in result["endpoints"].items()}}))


if __name__ == "__main__":
    main()

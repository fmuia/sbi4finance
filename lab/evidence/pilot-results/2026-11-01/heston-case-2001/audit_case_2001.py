"""Audit archived Heston S3 cases using saved draws and deterministic Fourier prices.

From lab/:
    python evidence/pilot-results/2026-11-01/heston-case-2001/audit_case_2001.py \
        --output evidence/pilot-results/2026-11-01/heston-case-2001/summary.json

Omit --output to print the report. Inputs are read only. This does not load a
neural model, simulate a new bank, optimize a fit, or run a posterior sampler.
The only optional write is the explicitly selected output JSON.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy


REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "lab" / "src"))

from sbi4finance.calibration import bank, correction, pricing, tasks  # noqa: E402
from sbi4finance.svj.pmmh import rhat_ess  # noqa: E402


def binding(path):
    path = Path(path)
    return {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def original_units(draws, truth):
    q = np.quantile(draws, [.025, .05, .5, .95, .975], axis=0)
    mean, sd = draws.mean(axis=0), draws.std(axis=0, ddof=0)
    return {"draws": len(draws), "mean": mean.tolist(), "sd_ddof0": sd.tolist(),
            "q025": q[0].tolist(), "q05": q[1].tolist(), "median": q[2].tolist(),
            "q95": q[3].tolist(), "q975": q[4].tolist(),
            "truth_in_central90": ((q[1] <= truth) & (truth <= q[3])).tolist(),
            "truth_in_central95": ((q[0] <= truth) & (truth <= q[4])).tolist(),
            "truth_minus_mean_over_sd": ((truth - mean) / sd).tolist(),
            "empirical_fraction_at_or_below_truth": np.mean(draws <= truth, axis=0).tolist()}


def fourier(task, theta, nodes, cutoff):
    parameters, initial = tasks._heston_split(tasks.model_parameters(task, theta))
    return pricing.heston_surface(parameters, initial, task["context"]["contracts"],
                                  nodes=nodes, upper_limit=cutoff)


def archived_reference_prices(task, theta):
    """The original S3 exact_prices policy, without importing its model loader."""
    try:
        return fourier(task, theta, 1024, 600.0)
    except ValueError:
        parameters, initial = tasks._heston_split(tasks.model_parameters(task, theta))
        return tasks.heston_prices(task, parameters, initial)[0]


def audit():
    assets = REPO / "tutorial" / "assets"
    source = assets / "results" / "calibration" / "heston" / "s3"
    config_path = REPO / "lab" / "configs" / "tasks" / "calibration_heston_bank.json"
    config = json.loads(config_path.read_text())
    task = config["task"]
    exported_task_path = assets / "checkpoints" / "heston" / "task.json"
    training_path = assets / "checkpoints" / "heston" / "training.json"
    report = json.loads((source / "heston-check.json").read_text())
    training = json.loads(training_path.read_text())
    accepted = json.loads((source / "ACCEPTED.json").read_text())
    assert json.loads(exported_task_path.read_text()) == task
    fingerprint = bank.task_fingerprint(config)
    assert fingerprint == report["task_fingerprint"] == training["task_fingerprint"]
    assert report["checkpoint_sha256"] == training["checkpoint_sha256"]
    assert config["root_seed"] == 2610
    assert task["parameter_names"] == ["initial_variance", "theta", "vol_of_vol", "rho"]
    settings = report["settings"]
    assert settings["first"] == 2000 and settings["chain_surfaces"] == 4
    accepted_matches = {name: binding(source / name)["sha256"] == digest
                        for name, digest in accepted["source_sha256"].items()}
    assert all(accepted_matches.values())

    dependency_paths = [REPO / "lab" / "src" / "sbi4finance" / p
                        for p in bank.source_files(task)]
    dependency_paths += [REPO / "lab" / "src" / "sbi4finance" / p for p in
                         ["calibration/correction.py", "svj/pmmh.py"]]
    input_paths = [Path(__file__).resolve(), config_path, exported_task_path, training_path,
                   *(source / p for p in ["ACCEPTED.json", "heston-check.json", "posterior-draws.npz", "agreement.png"]),
                   REPO / "lab" / "studies" / "heston_check.py",
                   REPO / "lab" / "studies" / "evaluate_calibration.py",
                   REPO / "lab" / "src" / "sbi4finance" / "calibration" / "plots.py",
                   *dependency_paths]
    sigma = tasks.sigma(task)
    bounds = np.asarray(task["prior_bounds"])
    words = [int(fingerprint[i:i + 8], 16) for i in range(0, 64, 8)]
    cases = []
    with np.load(source / "posterior-draws.npz", allow_pickle=False) as saved:
        for row in range(2000, 2004):
            entry = next(s for s in report["surfaces"] if s["test_row"] == row)
            truth, observation = saved[f"truth_{row}"], saved[f"observation_{row}"]
            assert np.array_equal(truth, np.asarray(entry["truth"]))
            identity = [0x4F5054, config["root_seed"], *words, bank.SPLITS["test"], row]
            prior_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 0])))
            quote_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, 2])))
            regenerated_truth = prior_rng.uniform(bounds[:, 0], bounds[:, 1])
            quote_standard_normals = quote_rng.standard_normal(len(sigma))
            parameters, initial = tasks._heston_split(tasks.model_parameters(task, truth))
            generated_prices, refinement = tasks.heston_prices(task, parameters, initial)
            regenerated_observation = generated_prices + sigma * quote_standard_normals
            parameter_error = float(np.max(np.abs(regenerated_truth - truth)))
            observation_error = float(np.max(np.abs(regenerated_observation - observation) / sigma))
            assert parameter_error < 1e-14
            assert observation_error < 1e-5

            reference = saved[f"chain_{row}"]
            chains = reference.reshape(settings["chains"], settings["steps"] - settings["burn"], len(truth))
            assert chains.shape == (4, 5000, 4)
            diagnostics = np.asarray([rhat_ess(chains[:, :, j]) for j in range(len(truth))])
            assert np.allclose(diagnostics[:, 0], entry["chains"]["rhat"], rtol=1e-12, atol=1e-12)
            assert np.allclose(diagnostics[:, 1], entry["chains"]["ess"], rtol=1e-12, atol=1e-12)
            methods = {name: original_units(saved[f"{prefix}_{row}"], truth)
                       for name, prefix in [("fourier_mcmc", "chain"), ("network", "network"),
                                            ("importance_resampled", "reweighted")]}
            noise = (observation - generated_prices) / sigma
            posterior_mean = reference.mean(axis=0)
            at_truth = archived_reference_prices(task, truth)
            at_mean = archived_reference_prices(task, posterior_mean)
            truth_chi2 = float(np.sum(((observation - at_truth) / sigma) ** 2))
            mean_chi2 = float(np.sum(((observation - at_mean) / sigma) ** 2))
            delta_loglik = float(correction.gaussian_loglik(task, observation, at_mean)[0]
                                - correction.gaussian_loglik(task, observation, at_truth)[0])
            assert np.isclose(delta_loglik, (truth_chi2 - mean_chi2) / 2, atol=1e-10)
            fine_truth = fourier(task, truth, 4096, 3000.0)
            fine_mean = fourier(task, posterior_mean, 4096, 3000.0)
            max_price_error = max(float(np.max(np.abs(fine_truth - at_truth) / sigma)),
                                  float(np.max(np.abs(fine_mean - at_mean) / sigma)))
            assert max_price_error < 1e-4
            result = {"test_row": row, "truth": truth.tolist(), "original_units": methods,
                      "reconstruction": {"declared_split": "test", "local_example_id": row,
                                         "prior_purpose": 0, "quote_noise_purpose": 2,
                                         "max_absolute_parameter_difference": parameter_error,
                                         "max_observation_difference_over_quote_sd": observation_error},
                      "noise_and_likelihood": {
                          "residual_definition": "(saved observation - task-refined price at truth) / quote SD",
                          "noise_rmse": float(np.sqrt(np.mean(noise ** 2))),
                          "maximum_absolute_noise": float(np.max(np.abs(noise))),
                          "noise_mean_by_maturity": noise.reshape(7, 9).mean(axis=1).tolist(),
                          "noise_rmse_by_maturity": np.sqrt(np.mean(noise.reshape(7, 9) ** 2, axis=1)).tolist(),
                          "chi_squared_at_truth": truth_chi2,
                          "chi_squared_at_saved_reference_mean": mean_chi2,
                          "loglik_mean_minus_truth": delta_loglik,
                          "likelihood_ratio_mean_over_truth": float(np.exp(delta_loglik)),
                          "maximum_4096_vs_reference_price_difference_over_quote_sd_at_truth_and_mean": max_price_error,
                          "maximum_recorded_generation_refinement_over_quote_sd": float(np.max(np.abs(refinement) / sigma))},
                      "reference_diagnostics": {"rhat": diagnostics[:, 0].tolist(),
                                                "ess_bulk": diagnostics[:, 1].tolist(),
                                                "matches_cached_report": True,
                                                "acceptance": entry["chains"]["acceptance"],
                                                "per_chain_mean": chains.mean(axis=1).tolist()},
                      "sigma_rho_reference_correlation": float(np.corrcoef(reference[:, 2], reference[:, 3])[0, 1]),
                      "importance_diagnostics": entry["audit"],
                      "historical_agreement": {k: entry[k] for k in
                                               ["network_vs_chain", "reweighted_vs_chain", "network_vs_reweighted"]}}
            if row == 2001:
                result["noise_and_likelihood"]["reconstructed_quote_standard_normals"] = quote_standard_normals.tolist()
                result["noise_and_likelihood"]["realized_noise_standard_units"] = noise.tolist()
                indices = [c * 5000 + j for c in range(4) for j in [0, 1666, 3333, 4999]]
                price_differences, loglik_differences = [], []
                for index in indices:
                    base_price = archived_reference_prices(task, reference[index])
                    refined_price = fourier(task, reference[index], 4096, 3000.0)
                    price_differences.append(float(np.max(np.abs(base_price - refined_price) / sigma)))
                    loglik_differences.append(float(abs(correction.gaussian_loglik(task, observation, base_price)[0]
                                                        - correction.gaussian_loglik(task, observation, refined_price)[0])))
                assert max(price_differences) < 1e-4
                result["numerical_reference_spot_check"] = {
                    "saved_pooled_chain_indices": indices,
                    "max_absolute_price_difference_over_quote_sd": max(price_differences),
                    "max_absolute_loglik_difference": max(loglik_differences),
                    "scope": "16 existing saved draws, four per chain; not a uniform quadrature error bound"}
            cases.append(result)

    return {"format": "heston_archived_s3_case_audit_v1", "audit_date": "2026-11-01",
            "scope": {"existing_cached_cases_only": True, "neural_models_loaded": 0,
                      "new_bank_examples_written": 0, "fits_or_posterior_samplers_run": 0,
                      "original_seed2610_raw_bank_available": False,
                      "inputs_modified": False, "scientific_acceptance": False},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "input_bindings": [binding(p) for p in dict.fromkeys(input_paths)],
            "accepted_source_hashes_match": accepted_matches,
            "parameter_order": task["parameter_names"],
            "task": {"task_fingerprint": fingerprint, "root_seed": config["root_seed"],
                     "measure": task["measure"], "fixed_parameters": task["fixed_parameters"],
                     "prior_bounds": task["prior_bounds"], "quote_noise": "independent additive Gaussian, no clipping",
                     "rng": "PCG64; SeedSequence(0x4F5054, root_seed, task digest words, split=2, local ID, purpose)",
                     "generation_quadrature_ladder": task["numerics"]["quadrature_ladder"],
                     "generation_refinement_budget_over_quote_sd": task["numerics"]["max_refinement_over_quote_sigma"],
                     "reference_quadrature": {"nodes": 1024, "cutoff": 600.0, "fallback": "task-refined ladder on ValueError"},
                     "audit_refinement": {"nodes": 4096, "cutoff": 3000.0},
                     "reconstruction_tolerances": {"parameter_absolute": 1e-14, "observation_over_quote_sd": 1e-5}},
            "historical_sampler": {"method": "random-walk Metropolis in logit coordinates; uniform physical prior with logit Jacobian",
                                   "settings": settings, "checkpoint_sha256_as_reported": report["checkpoint_sha256"],
                                   "starts_and_proposal_covariance": "derived from neural posterior draws",
                                   "diagnostics": "rank-normalized split R-hat, max of bulk/folded; bulk ESS"},
            "cases": cases,
            "interpretation": {
                "row_association": "Saved truths/observations match declared deterministic test IDs and quote-noise streams to numerical precision; no row/parameter-label mismatch found.",
                "case2001": "Truth lies in both the sigma_v and rho reference tails. Their strong positive dependence and the observed likelihood shift are consistent with a data-associated noise realization; this does not prove global reference correctness.",
                "retain_case": True,
                "not_a_population_evaluation": True},
            "limitations": [
                "The original seed2610 raw bank was not recovered; this reconstructs the declared rows and checks cached observations, not the missing original bank-file hashes.",
                "S3 records task/checkpoint identities but no source-code or bank-manifest hash. Current source bindings make this audit reproducible; they do not establish the original executable's byte identity.",
                "All four historical chains start within the neural proposal; favorable R-hat/ESS cannot rule out an undiscovered mode.",
                "Case2001 importance k-hat exceeds .7; importance agreement is not an independent reference guarantee.",
                "Historical mean/width agreement flags are not themselves a convergence gate or a truth-coverage test.",
                "Quadruple-node comparisons at truth, mean and 16 saved case2001 draws are numerical spot checks, not a uniform integration error certificate.",
                "No latent path generated these option observations: Fourier expectations plus quote noise did. A future process-path illustration would be separately simulated and labelled."
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        protected = {REPO / entry["path"] for entry in result["input_bindings"]}
        if args.output.resolve() in protected:
            raise ValueError("the output must not overwrite an input")
        args.output.write_text(text)
        print(f"Audited four saved cases; wrote {len(text.encode())} bytes to {args.output}")


if __name__ == "__main__":
    main()

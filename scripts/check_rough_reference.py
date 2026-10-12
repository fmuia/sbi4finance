"""Build the frozen finite-Monte-Carlo rough-Bergomi importance reference.

This command reads stored audit panels and a separate prior bank. It never
regenerates audit cases, uses noisy reference observations as likelihood means,
or adds price-estimator covariance to the Gaussian quote-error likelihood.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from sbi4finance import option_bank, rough_reference as ref
from sbi4finance.bergomi import option_panel
from sbi4finance.gbm_bank import _integer, _sha256, _write_json


TARGET_TAG = 0x52545052
STRESS_TAG = 0x52535452


def price_parameters(task, parameters, contracts, *, n_paths, n_steps, seed, case_id=0):
    """Return prices and marginal SEs [N,C] using fresh per-row target streams.

    Each row uses SeedSequence(TARGET_TAG, seed, case_id, row). Target randomness
    is independent of bank pricing randomness and stress/quote streams. ``case_id``
    distinguishes neural cases; rows must retain their declared order. Means,
    rather than the distribution of these noisy row estimates, are the downstream
    estimand. No numerical bias or per-row precision certificate is implied.
    """
    parameters = np.asarray(parameters, dtype=float)
    if (task["model"] != "rough" or task["parameter_names"] != ["hurst", "eta"]
            or parameters.ndim != 2 or not len(parameters) or parameters.shape[1] != 2
            or not np.isfinite(parameters).all()):
        raise ValueError("rough task and finite ordered [N,2] H/eta parameters required")
    n_paths = _integer(n_paths, "n_paths", minimum=2)
    n_steps = _integer(n_steps, "n_steps", minimum=1)
    seed, case_id = _integer(seed, "seed"), _integer(case_id, "case_id")
    context = task["context"]
    estimates, errors = np.empty((len(parameters), len(contracts))), np.empty((len(parameters), len(contracts)))
    before = time.perf_counter()
    for row, values in enumerate(parameters):
        row_seed = int(np.random.SeedSequence([TARGET_TAG, seed, case_id, row]).generate_state(
            1, dtype=np.uint64)[0])
        model = dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], values)))
        result = option_panel("rough", model, task["model_context"], contracts,
                              spot=context["spot"], rate=context["rate"], dividend=context["dividend"],
                              n_paths=n_paths, n_steps=n_steps, seed=row_seed,
                              method=task["numerics"]["method"])
        estimates[row], errors[row] = result["prices"], result["standard_errors"]
        if (row + 1) % 128 == 0 or row + 1 == len(parameters):
            print(f"Target pricing {row + 1}/{len(parameters)} rows, "
                  f"{time.perf_counter() - before:.1f}s", flush=True)
    return estimates, errors


def accepted_inputs(bank_path, reference_bank_path, protocol_path):
    """Reject incompatible identities before starting expensive target pricing."""
    protocol = json.loads(protocol_path.read_text())
    config = json.loads(Path(protocol["bank_config"]).read_text())
    reference_config = json.loads(Path(protocol["reference_config"]).read_text())
    if (config["task"] != reference_config["task"] or config["task"]["model"] != "rough"
            or config["task"]["parameter_names"] != ["hurst", "eta"]
            or config["root_seed"] == reference_config["root_seed"]):
        raise ValueError("matching rough tasks with independent bank root seeds required")
    settings = protocol["reference"]
    if (settings["split"] != "train" or settings["blocks"] != 2
            or reference_config["splits"]["train"] != settings["rows"]
            or settings["rows"] % 2 != 0
            or protocol["target_pricing"]["n_steps"] != config["task"]["numerics"]["n_steps"]
            or protocol["target_pricing"]["method"] != config["task"]["numerics"]["method"]):
        raise ValueError("reference row/block or target numerical controls differ from the protocol")
    manifest, bank = option_bank.load_bank(bank_path)
    reference_manifest, reference_bank = option_bank.load_bank(reference_bank_path)
    for label, saved, expected in (("bank", manifest, config),
                                   ("reference bank", reference_manifest, reference_config)):
        if saved["configuration"] != expected:
            raise ValueError(f"{label} configuration differs from the frozen protocol")
        if set(saved["source_sha256"]) != set(option_bank.source_files("rough")):
            raise ValueError(f"{label} generator source set differs")
        for name, digest in saved["source_sha256"].items():
            if digest != _sha256(Path(option_bank.__file__).with_name(name)):
                raise ValueError(f"{label} generator source changed: {name}")
    ids = protocol["conditional_audit_ids"]
    lower, upper = protocol["audit_validation_ids"]
    if (not ids or len(set(ids)) != len(ids) or any(isinstance(i, bool) or not isinstance(i, int)
            or not lower <= i < upper for i in ids) or upper > len(bank["validation"]["parameters"])):
        raise ValueError("conditional IDs must be unique members of the audit validation range")
    return protocol, config, reference_config, manifest, bank, reference_manifest, reference_bank


def fixed_cases(config, protocol, bank):
    """Read audit observations verbatim; simulate only the declared stress cases."""
    ids, task = protocol["conditional_audit_ids"], config["task"]
    bounds = np.asarray(task["prior_bounds"])
    fractions = np.asarray(protocol["stress_prior_fractions"], dtype=float)
    if (fractions.ndim != 2 or fractions.shape[1] != 2 or not np.isfinite(fractions).all()
            or np.any((fractions <= 0) | (fractions >= 1))):
        raise ValueError("stress fractions must be finite interior [N,2] values")
    stress = bounds[:, 0] + fractions * np.diff(bounds)[:, 0]
    prices, errors, covariances = [], [], []
    for index, values in enumerate(stress):
        seed = int(np.random.SeedSequence([STRESS_TAG, protocol["stress_pricing_seed"], index]).generate_state(
            1, dtype=np.uint64)[0])
        model = dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], values)))
        estimate, error, covariance, _ = option_bank.price_panel(task, model, seed)
        prices.append(estimate)
        errors.append(error)
        covariances.append(covariance)
    prices, errors, covariances = np.asarray(prices), np.asarray(errors), np.asarray(covariances)
    observed = prices + np.random.default_rng(protocol["stress_quote_seed"]).normal(
        0, task["context"]["quote_noise_std"], prices.shape)
    selected = bank["validation"]
    arrays = {
        "observations": np.concatenate([selected["observations"][ids], observed[..., None]]),
        "truth": np.concatenate([selected["parameters"][ids], stress]),
        "case_ids": np.concatenate([selected["example_id"][ids], [f"stress:{i}" for i in range(len(stress))]]),
        "stress_reference_prices": prices, "stress_pricing_standard_errors": errors,
        "stress_pricing_covariance": covariances,
    }
    return arrays


def case_reference(parameters, prices, targets, observation, sigma, thresholds):
    """Apply the frozen reference gates to one weighted conditional experiment."""
    weights = ref.importance_weights(prices, observation[:, 0], sigma)
    diagnostic = ref.reference_diagnostics(parameters, weights)
    target = ref.weighted_target_mean(targets, weights)
    midpoint = len(parameters) // 2
    blocks = []
    for indices in (slice(0, midpoint), slice(midpoint, len(parameters))):
        try:
            blocks.append(ref.weighted_target_mean(targets[indices], weights[indices]))
        except ValueError:
            blocks.append(None)
    valid = diagnostic["blocks_valid"] and all(block is not None for block in blocks)
    comparison = diagnostic["block_comparison"]
    block_target_difference = (np.asarray(blocks[1]["mean"]) - blocks[0]["mean"]) if valid else None
    metrics = {
        "min_ess": diagnostic["ess"], "max_weight": diagnostic["max_weight"],
        "max_mean_mcse_in_ref_sd": float(np.max(diagnostic["mean_mcse_in_ref_sd"])),
        "max_abs_block_mean_difference_in_ref_sd":
            float(np.max(np.abs(comparison["mean_difference_in_ref_sd"]))) if valid else None,
        "max_abs_block_relative_std_difference":
            float(np.max(np.abs(np.asarray(comparison["std_ratio"]) - 1))) if valid else None,
        "max_abs_block_correlation_difference":
            abs(comparison["correlation_difference"]) if valid else None,
        "max_target_mean_mcse_currency": float(np.max(target["mcse"])),
        "max_abs_block_target_mean_difference_currency":
            float(np.max(np.abs(block_target_difference))) if valid else None,
    }
    if set(metrics) != set(thresholds):
        raise ValueError("reference thresholds differ from the implemented diagnostic metrics")
    gates = {key: (value is not None and (value >= thresholds[key] if key == "min_ess"
                                         else value <= thresholds[key]))
             for key, value in metrics.items()}
    return weights, {"passed": all(gates.values()), "metrics": metrics, "gates": gates,
                     "reference": diagnostic, "target": target, "target_blocks": blocks,
                     "block_target_mean_difference_currency":
                         block_target_difference.tolist() if valid else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/rough_inference_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol, config, reference_config, manifest, bank, reference_manifest, reference_bank = accepted_inputs(
        args.bank, args.reference_bank, args.protocol)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        arrays = fixed_cases(config, protocol, bank)
        cases_ready = time.perf_counter()
        rows = reference_bank[protocol["reference"]["split"]]
        target_spec, task = protocol["target_pricing"], config["task"]
        estimates, errors = price_parameters(task, rows["parameters"], protocol["targets"],
                                             n_paths=target_spec["reference_paths_per_row"],
                                             n_steps=target_spec["n_steps"], seed=target_spec["reference_seed"])
        priced = time.perf_counter()
        thresholds, all_weights, cases = protocol["reference"]["gates"], [], []
        for case_id, observation in zip(arrays["case_ids"], arrays["observations"], strict=True):
            weights, case = case_reference(rows["parameters"], rows["reference_prices"], estimates,
                                          observation, task["context"]["quote_noise_std"], thresholds)
            all_weights.append(weights)
            cases.append({"case_id": str(case_id), **case})
            print(f"{case_id}: ESS {case['metrics']['min_ess']:.1f}, passed {case['passed']}", flush=True)
        arrays.update(parameters=rows["parameters"], reference_prices=rows["reference_prices"],
                      target_estimates=estimates, target_standard_errors=errors, weights=np.asarray(all_weights))
        artifact = args.out / "reference.npz"
        temporary = args.out / "reference.npz.tmp"
        with temporary.open("wb") as file:
            np.savez_compressed(file, **arrays)
        os.replace(temporary, artifact)
        (args.out / "reference-bank-manifest.json").write_bytes((args.reference_bank / "manifest.json").read_bytes())
        metrics = {key: (min(case["metrics"][key] for case in cases) if key == "min_ess" else
                         max(case["metrics"][key] for case in cases))
                   if all(case["metrics"][key] is not None for case in cases) else None for key in thresholds}
        gates = {key: all(case["gates"][key] for case in cases) for key in thresholds}
        stress_limit = task["context"]["quote_noise_std"] * task["numerics"]["max_mc_se_over_quote_sigma"]
        stress_max_se = float(np.max(arrays["stress_pricing_standard_errors"]))
        stress_passed = stress_max_se <= stress_limit
        report = {
            "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q",
            "protocol_sha256": _sha256(args.protocol),
            "bank_manifest_sha256": _sha256(args.bank / "manifest.json"),
            "reference_bank_manifest_sha256": _sha256(args.reference_bank / "manifest.json"),
            "reference_bank_configuration": reference_config, "task": task,
            "observation_law": protocol["observation_law"], "target_contracts": protocol["targets"],
            "target_estimand": target_spec["estimand"], "case_ids": arrays["case_ids"].tolist(),
            "thresholds": thresholds, "metrics": metrics, "gates": gates,
            "cases": cases, "passed": bool(all(gates.values()) and stress_passed),
            "stress_pricing": {"max_standard_error": stress_max_se, "limit": stress_limit,
                               "passed": bool(stress_passed)},
            "artifact_sha256": {"reference.npz": _sha256(artifact)},
            "source_sha256": {"check_rough_reference.py": _sha256(__file__),
                              "rough_reference.py": _sha256(ref.__file__),
                              **{name: _sha256(Path(option_bank.__file__).with_name(name))
                                 for name in option_bank.source_files("rough")}},
            "rng": {"target": "SeedSequence(TARGET_TAG=0x52545052, reference_seed, case_id=0, row)",
                    "stress_pricing": "SeedSequence(STRESS_TAG=0x52535452, stress_pricing_seed, index)",
                    "stress_quotes": "PCG64(stress_quote_seed), one independent matrix draw"},
            "timing_seconds": {"input_and_stress": cases_ready-started,
                               "independent_target_prices": priced-cases_ready,
                               "total": time.perf_counter()-started},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "reserved_test_evaluated": False,
            "limits": ["Finite-Monte-Carlo numerical posterior; no continuum posterior claim.",
                       "ESS, delta-method SE and two independent blocks do not certify support or bound finite-bank bias.",
                       "Target means use fresh randomness; noisy per-row target estimates are not exact-price posterior draws.",
                       "Target sampling SE excludes time-discretization bias; no separate continuum target refinement certificate.",
                       "Audit panels are loaded verbatim; reference-bank validation/test compatibility rows are unused."],
        }
        _write_json(args.out / "reference.json", report)
    except Exception as error:
        _write_json(args.out / "failure.json", {"error": str(error), "protocol_sha256": _sha256(args.protocol),
                                               "elapsed_seconds": time.perf_counter()-started})
        raise
    print(json.dumps({"reference_passed": report["passed"], "metrics": metrics}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("Frozen reference gates unresolved; report retained, no automatic expansion.")


if __name__ == "__main__":
    main()

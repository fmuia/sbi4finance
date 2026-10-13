"""Validation-only audit of the frozen rough-Bergomi posterior pilot.

The final-test split has no execution path here. Scientific acceptance combines
conditional/joint diagnostics, marginal SBC, and posterior MEAN target pricing.
Independent target randomness avoids conditioning on the observation pricer's
latent randomness. No target-price quantile or continuum-price claim is made.
"""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from sbi4finance import rough_inference as inference, rough_reference as reference
from sbi4finance.gbm_bank import _sha256
from train_rough import accepted_inputs


def joint(samples):
    """Preserve the shared sample axis of the two named raw-parameter arrays."""
    arrays = [np.asarray(samples[name], dtype=np.float64) for name in inference.PARAMETER_NAMES]
    if (arrays[0].ndim != 3 or arrays[0].shape[-1] != 1
            or arrays[0].shape != arrays[1].shape
            or not all(np.isfinite(a).all() for a in arrays)):
        raise ValueError("matched finite named samples [batch,draw,1] required")
    return np.concatenate(arrays, axis=-1)


def target_mean_metrics(estimates, reference_estimates, weights, standard_errors=None):
    """Compare independent nested-MC target means; do not double-count inner SE.

    Rows of estimates are independent (theta,V) draws. Their empirical sample
    variance contains both outer parameter and inner pricing variation. Inner
    standard errors are reported descriptively; adding them again to that
    variance would inflate the uncertainty. The independent-reference SNIS SE
    is combined in quadrature. Neither SE includes numerical bias or flow bias.
    """
    values = np.asarray(estimates, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 3
            or not np.isfinite(values).all()):
        raise ValueError("finite target estimates [L,2], L>=3 required")
    ref = reference.weighted_target_mean(reference_estimates, weights)
    if len(ref["mean"]) != 2:
        raise ValueError("exactly two reference target means required")
    mean, mcse = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))
    difference = mean - np.asarray(ref["mean"])
    combined = np.sqrt(mcse**2 + np.asarray(ref["mcse"])**2)
    result = {
        "target_mean": mean.tolist(), "target_mean_mcse": mcse.tolist(),
        "reference_target_mean": ref["mean"], "reference_target_mean_mcse": ref["mcse"],
        "target_mean_error_currency": difference.tolist(),
        "target_mean_combined_mcse_currency": combined.tolist(),
        "target_mean_error_plus_2mcse_currency": (np.abs(difference) + 2 * combined).tolist(),
        "target_mean_uncertainty_scope": "IID neural (theta,V) sample-mean SE plus independent reference SNIS SE; excludes numerical and neural bias",
    }
    if standard_errors is not None:
        se = np.asarray(standard_errors, dtype=float)
        if se.shape != values.shape or not np.isfinite(se).all() or np.any(se < 0):
            raise ValueError("finite nonnegative pricing standard errors matching target estimates required")
        result["target_mean_inner_mcse_component"] = (np.sqrt(np.sum(se**2, axis=0)) / len(se)).tolist()
    return result


def aggregate(cases, *, include_targets=True):
    """Frozen aggregate definitions; report individual cases alongside averages."""
    if not cases:
        raise ValueError("nonempty conditional cases required")
    mean_errors = np.asarray([c["mean_error_in_ref_sd"] for c in cases])
    masses = np.asarray([c["reference_mass_of_90_interval"] for c in cases]).mean(axis=0)
    result = {
        "mean_rmse_in_ref_sd": float(np.sqrt(np.mean(mean_errors**2))),
        "max_abs_mean_error_in_ref_sd": float(np.max(np.abs(mean_errors))),
        "max_abs_relative_std_error": float(np.max(np.abs([c["relative_std_error"] for c in cases]))),
        "max_abs_correlation_error": float(np.max(np.abs([c["correlation_error"] for c in cases]))),
        "mean_reference_mass_of_90_interval_abs_error": float(np.max(np.abs(masses - 0.9))),
    }
    if include_targets:
        result["max_target_mean_error_plus_2mcse_currency"] = float(np.max([
            c["target_mean_error_plus_2mcse_currency"] for c in cases]))
    if not np.isfinite(list(result.values())).all():
        raise ValueError("all aggregate diagnostics must be finite")
    return result


def validation_gates(metrics, sbc, thresholds):
    if set(metrics) != set(thresholds):
        raise ValueError("all frozen conditional and target-mean gates must be present")
    gates = {key: bool(value <= thresholds[key]) for key, value in metrics.items()}
    gates["marginal_sbc"] = bool(sbc["rank_diagnostic_passed"])
    return gates


def checked_inputs(bank_path, reference_path, checkpoint_path, protocol_path):
    """Bind validation to the accepted reference and exact trained candidate."""
    protocol, config, ref_report, manifest, splits = accepted_inputs(bank_path, reference_path, protocol_path)
    training = json.loads((checkpoint_path / "training.json").read_text())
    checkpoint_hash = _sha256(checkpoint_path / "model.keras")
    if (training["protocol_sha256"] != _sha256(protocol_path)
            or training["bank_configuration"] != config
            or training["checkpoint_sha256"] != checkpoint_hash
            or not training["same_seed_reload_exact"]
            or training["reserved_test_evaluated"]
            or training["adapter_id"] != inference.ADAPTER_ID
            or training["training"] != protocol["training"]
            or training["selection_validation_ids"] != protocol["selection_validation_ids"]
            or training["bank_manifest_sha256"] != _sha256(bank_path / "manifest.json")
            or training["reference_report_sha256"] != _sha256(reference_path / "reference.json")
            or training["reference_bank_manifest_sha256"] != ref_report["reference_bank_manifest_sha256"]
            or json.loads((checkpoint_path / "task.json").read_text()) != config["task"]
            or json.loads((checkpoint_path / "protocol.json").read_text()) != protocol
            or json.loads((checkpoint_path / "bank-manifest.json").read_text()) != manifest):
        raise ValueError("matching frozen task, checkpoint, training record and accepted reference required")
    expected_sources = {"train_rough.py": Path(__file__).with_name("train_rough.py"),
                        "rough_inference.py": Path(inference.__file__)}
    if training["source_sha256"] != {name: _sha256(path) for name, path in expected_sources.items()}:
        raise ValueError("checkpoint training or inference source changed")
    base_count = len(splits["train"]["parameters"])
    lo, hi = protocol["selection_validation_ids"]
    if (training["training_base_examples"] != base_count
            or training["training_examples"] != base_count * protocol["training"]["quote_augmentations_per_base"]
            or training["selection_examples"] != hi - lo):
        raise ValueError("training base, augmentation or selection counts changed")
    return protocol, config, ref_report, manifest, splits, training


def checked_reference_arrays(path, protocol, task, validation):
    with np.load(path, allow_pickle=False) as file:
        arrays = dict(file)
    n = protocol["reference"]["rows"]
    c = len(protocol["conditional_audit_ids"]) + len(protocol["stress_prior_fractions"])
    shapes = {"parameters": (n, 2), "reference_prices": (n, 9), "target_estimates": (n, 2),
              "observations": (c, 9, 1), "truth": (c, 2), "case_ids": (c,), "weights": (c, n)}
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"reference artifact has wrong {name} shape")
        if name != "case_ids" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"reference artifact has nonfinite {name}")
    ids = protocol["conditional_audit_ids"]
    if (not np.array_equal(arrays["observations"][:len(ids)], validation["observations"][ids])
            or not np.array_equal(arrays["truth"][:len(ids)], validation["parameters"][ids])
            or not np.array_equal(arrays["case_ids"][:len(ids)], validation["example_id"][ids])
            or len(set(arrays["case_ids"].tolist())) != c):
        raise ValueError("conditional audit identities changed")
    for observation, weights in zip(arrays["observations"], arrays["weights"], strict=True):
        expected = reference.importance_weights(arrays["reference_prices"], observation[:, 0],
                                                 task["context"]["quote_noise_std"])
        if not np.allclose(weights, expected, rtol=1e-12, atol=1e-15):
            raise ValueError("stored weights disagree with the declared finite-MC quote law")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/rough_inference_protocol.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    protocol, config, ref_report, manifest, splits, training = checked_inputs(
        args.bank, args.reference, args.checkpoint, args.protocol)
    task, evaluation = config["task"], protocol["evaluation"]
    arrays = checked_reference_arrays(args.reference / "reference.npz", protocol, task, splits["validation"])
    parameters, weights, observations, truth, case_ids = [arrays[key] for key in
        ("parameters", "weights", "observations", "truth", "case_ids")]
    lo, hi = protocol["audit_validation_ids"]
    audit = {key: value[lo:hi] for key, value in splits["validation"].items()}
    args.out.mkdir(parents=True, exist_ok=False)

    from check_rough_reference import price_parameters

    workflow = inference.build_workflow(task)
    workflow.load_approximator(args.checkpoint / "model.keras")
    loaded = time.perf_counter()
    draws = joint(workflow.sample(num_samples=evaluation["conditional_draws"],
                                  conditions=inference.conditioning_data(observations, task),
                                  batch_size=8, seed=evaluation["sample_seed"]))
    sampled = time.perf_counter()
    bounds = np.asarray(task["prior_bounds"])
    prior_rng = np.random.default_rng(evaluation["prior_control_seed"])
    prior_draws = prior_rng.uniform(bounds[:, 0], bounds[:, 1],
                                   (len(case_ids), evaluation["prior_control_draws"], 2))
    cases, prior_cases, summaries, targets, target_ses = [], [], [], [], []
    for index, (one_draws, one_weights) in enumerate(zip(draws, weights, strict=True)):
        target_spec = protocol["target_pricing"]
        prices, standard_errors = price_parameters(
            task, one_draws, protocol["targets"], n_paths=target_spec["neural_paths_per_draw"],
            n_steps=target_spec["n_steps"], seed=target_spec["neural_seed"], case_id=index)
        metrics = reference.conditional_metrics(one_draws, parameters, one_weights, prior_bounds=bounds)
        metrics.update(target_mean_metrics(prices, arrays["target_estimates"], one_weights, standard_errors))
        metrics["case_id"] = str(case_ids[index])
        summary = reference.weighted_summary(parameters, one_weights)
        metrics["reference_summary"] = summary
        cases.append(metrics)
        summaries.append(summary)
        targets.append(prices)
        target_ses.append(standard_errors)
        prior_case = reference.conditional_metrics(prior_draws[index], parameters, one_weights, prior_bounds=bounds)
        prior_case["case_id"] = str(case_ids[index])
        prior_cases.append(prior_case)
        print(f"Priced posterior mean case {index + 1}/{len(draws)}", flush=True)
    priced = time.perf_counter()
    audit_draws = joint(workflow.sample(num_samples=evaluation["sbc_draws"],
                                        conditions=inference.conditioning_data(audit["observations"], task),
                                        batch_size=16, seed=evaluation["sample_seed"] + 1))
    sbc = reference.randomized_sbc(audit_draws, audit["parameters"],
                                  seed=evaluation["rank_randomization_seed"], alpha=evaluation["sbc_family_alpha"])
    prior_audit_draws = prior_rng.uniform(bounds[:, 0], bounds[:, 1], audit_draws.shape)
    prior_sbc = reference.randomized_sbc(prior_audit_draws, audit["parameters"],
                                        seed=evaluation["rank_randomization_seed"] + 1,
                                        alpha=evaluation["sbc_family_alpha"])
    sbc_finished = time.perf_counter()
    metrics = aggregate(cases)
    gates = validation_gates(metrics, sbc, evaluation["gates"])
    rank_arrays = {key: np.asarray(sbc.pop(key)) for key in ("ranks", "randomized_ranks")}
    rank_arrays.update({"prior_" + key: np.asarray(prior_sbc.pop(key)) for key in ("ranks", "randomized_ranks")})
    checks_path = args.out / "validation-checks.npz"
    np.savez_compressed(checks_path, case_ids=case_ids, observations=observations, truth=truth,
                        conditional_draws=draws, prior_conditional_draws=prior_draws,
                        reference_parameters=parameters, reference_weights=weights,
                        reference_target_estimates=arrays["target_estimates"],
                        target_estimates=np.asarray(targets), target_standard_errors=np.asarray(target_ses),
                        reference_mean=np.asarray([s["mean"] for s in summaries]),
                        reference_std=np.asarray([s["std"] for s in summaries]),
                        reference_correlation=np.asarray([s["correlation"] for s in summaries]),
                        audit_ids=audit["example_id"], audit_truth=audit["parameters"], **rank_arrays)
    report = {
        "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q", "split": "validation",
        "observation_law": protocol["observation_law"], "target_estimand": protocol["target_pricing"]["estimand"],
        "protocol_sha256": _sha256(args.protocol), "checkpoint_sha256": training["checkpoint_sha256"],
        "training_report_sha256": _sha256(args.checkpoint / "training.json"),
        "reference_report_sha256": _sha256(args.reference / "reference.json"),
        "reference_bank_manifest_sha256": ref_report["reference_bank_manifest_sha256"],
        "bank_manifest_sha256": _sha256(args.bank / "manifest.json"), "checks_sha256": _sha256(checks_path),
        "passed": bool(all(gates.values())), "metrics": metrics, "thresholds": evaluation["gates"], "gates": gates,
        "case_ids": case_ids.tolist(), "conditional_draws_per_case": int(draws.shape[1]), "cases": cases,
        "sbc": sbc,
        "prior_baseline": {"cases": prior_cases, "metrics": aggregate(prior_cases, include_targets=False),
                           "sbc": prior_sbc, "draws_per_case": evaluation["prior_control_draws"],
                           "mean": bounds.mean(axis=1).tolist(),
                           "std": (np.diff(bounds, axis=1)[:, 0] / np.sqrt(12)).tolist(),
                           "scope": "Independent prior draws ignore every observation; SBC can pass despite incorrect conditioning."},
        "mean_reference_mass_of_90_interval_by_parameter": np.mean([
            c["reference_mass_of_90_interval"] for c in cases], axis=0).tolist(),
        "timing_seconds": {"loading_including_reference": loaded - started,
                           "conditional_sampling_including_compile": sampled - loaded,
                           "target_mean_pricing_and_conditional_checks": priced - sampled,
                           "sbc_and_prior_control": sbc_finished - priced,
                           "total": time.perf_counter() - started},
        "source_sha256": {"evaluate_rough.py": _sha256(__file__),
                          "train_rough.py": _sha256(Path(__file__).with_name("train_rough.py")),
                          "check_rough_reference.py": _sha256(Path(__file__).with_name("check_rough_reference.py")),
                          "rough_reference.py": _sha256(reference.__file__),
                          "rough_inference.py": _sha256(inference.__file__)},
        "reserved_test_evaluated": False,
        "limits": ["Validation only; the reserved test has no execution path in this script.",
                   "Reference is finite-bank importance sampling under the declared finite-MC observation law.",
                   "Marginal prior-predictive SBC can pass for a network that ignores data; conditional/joint checks remain necessary.",
                   "Target outputs are posterior mean finite-grid expected payoffs with sampling SEs; not price quantiles or continuum certification.",
                   "No real-market evidence, no P risk probabilities and no future option-chain forecast.",
                   "One frozen architecture, seed and budget; failed gates do not trigger automatic tuning."],
    }
    (args.out / "neural-validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "metrics": metrics, "gates": gates,
                      "sbc": sbc, "prior_sbc": prior_sbc}, indent=2))
    if not report["passed"]:
        raise SystemExit("Frozen neural gates failed; checkpoint remains provisional and final test stays reserved.")


if __name__ == "__main__":
    main()

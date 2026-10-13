"""One frozen held-out evaluation of the selected rough-Bergomi checkpoint.

Reference gates run first on test:0..15. Any reference failure stops before
framework loading or neural/SBC evaluation. A passing reference unlocks the
unchanged conditional metrics and marginal SBC on all 256 reserved test cases.
There is no training, retry, case replacement, or automatic budget expansion.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from sbi4finance import rough_inference as inference, rough_reference as reference
from sbi4finance.gbm_bank import _sha256, _write_json
from check_rough_reference import case_reference, price_parameters
from evaluate_rough import (
    aggregate, checked_inputs, checked_reference_arrays, joint,
    target_mean_metrics, validation_gates,
)


def checked_prerequisites(protocol_path):
    """Check frozen files and passing development evidence before array loading."""
    protocol = json.loads(protocol_path.read_text())
    paths = {name: Path(value) for name, value in protocol["paths"].items()}
    development_path = Path(protocol["development_protocol"])
    required = {development_path, paths["bank"] / "manifest.json",
                paths["validation_report"],
                *(paths["checkpoint"] / name for name in
                  ("model.keras", "training.json", "task.json", "protocol.json", "bank-manifest.json")),
                *(paths["reference"] / name for name in
                  ("reference.json", "reference.npz", "reference-bank-manifest.json"))}
    # JSON metadata is safe to inspect before opening any reserved arrays.
    manifest = json.loads((paths["bank"] / "manifest.json").read_text())
    required.update(paths["bank"] / item["file"] for item in manifest["splits"].values())
    development = json.loads(development_path.read_text())
    required.update(Path(development[key]) for key in ("bank_config", "reference_config"))
    validation = json.loads(paths["validation_report"].read_text())
    required.add(paths["validation_report"].parent / "validation-checks.npz")
    if not required <= {Path(key) for key in protocol["input_sha256"]}:
        raise ValueError("frozen input hashes must cover every protocol, bank, checkpoint and validation artifact")
    expected_sources = {
        "scripts/evaluate_rough_test.py", "scripts/evaluate_rough.py", "scripts/train_rough.py",
        "scripts/check_rough_reference.py", "src/sbi4finance/rough_inference.py",
        "src/sbi4finance/rough_reference.py",
    }
    if not expected_sources <= set(protocol["frozen_source_sha256"]):
        raise ValueError("frozen source hashes must cover the held-out runner and scientific helpers")
    for label in ("input_sha256", "frozen_source_sha256"):
        for filename, digest in protocol[label].items():
            if not isinstance(digest, str) or len(digest) != 64 or _sha256(Path(filename)) != digest:
                raise ValueError(f"frozen {label} mismatch: {filename}")
    if (protocol["split"] != "test" or protocol["conditional_ids"] != list(range(16))
            or protocol["sbc_ids"] != [0, 256]
            or any(type(i) is not int for i in protocol["conditional_ids"] + protocol["sbc_ids"])):
        raise ValueError("fixed test:0..15 conditional cases and test:0..255 SBC cases required")
    if (protocol["reference_gates"] != development["reference"]["gates"]
            or protocol["targets"] != development["targets"]
            or protocol["observation_law"] != development["observation_law"]):
        raise ValueError("reference gates, targets and observation law must remain unchanged")
    expected_evaluation = dict(development["evaluation"], sample_seed=84034,
                               rank_randomization_seed=84035, prior_control_seed=84036)
    expected_target = dict(development["target_pricing"], neural_seed=84031)
    if protocol["evaluation"] != expected_evaluation or protocol["target_pricing"] != expected_target:
        raise ValueError("unchanged neural gates/budgets and the declared held-out RNG seeds required")
    expected_validation_gates = set(development["evaluation"]["gates"]) | {"marginal_sbc"}
    if (validation.get("passed") is not True or validation.get("split") != "validation"
            or validation.get("reserved_test_evaluated") is not False
            or validation.get("experiment_id") != development["experiment_id"]
            or set(validation.get("gates", {})) != expected_validation_gates
            or not all(value is True for value in validation["gates"].values())
            or validation.get("thresholds") != development["evaluation"]["gates"]
            or validation.get("protocol_sha256") != _sha256(development_path)
            or validation.get("checkpoint_sha256") != _sha256(paths["checkpoint"] / "model.keras")
            or validation.get("training_report_sha256") != _sha256(paths["checkpoint"] / "training.json")
            or validation.get("reference_report_sha256") != _sha256(paths["reference"] / "reference.json")
            or validation.get("bank_manifest_sha256") != _sha256(paths["bank"] / "manifest.json")
            or validation.get("checks_sha256") != _sha256(paths["validation_report"].parent / "validation-checks.npz")):
        raise ValueError("all matching frozen reference and neural validation prerequisites must pass")
    return protocol, paths


def selected_test_cases(split, protocol):
    """Keep original observations and separate truth; reject reordered base IDs."""
    ids = np.asarray(split["example_id"])
    if ids.shape != (256,) or not np.array_equal(ids, [f"test:{i}" for i in range(256)]):
        raise ValueError("exactly the original ordered test:0..255 base IDs required")
    observations, truth = np.asarray(split["observations"]), np.asarray(split["parameters"])
    if (observations.shape != (256, 9, 1) or truth.shape != (256, 2)
            or not np.isfinite(observations).all() or not np.isfinite(truth).all()):
        raise ValueError("complete finite held-out observations [256,9,1] and separate truth [256,2] required")
    selected = protocol["conditional_ids"]
    return ids[selected], observations[selected], truth[selected]


def evaluate(protocol_path, out):
    """Return the recorded status; never overwrite a previous scientific run."""
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError("held-out output already exists; previous outcomes must be retained")
    protocol, paths = checked_prerequisites(protocol_path)
    out.mkdir(parents=True, exist_ok=False)
    accessed, neural_evaluated = False, False
    reference_ids, neural_ids, sbc_ids = [], [], []
    try:
        # This integrity loader opens the reserved split, after all frozen-file
        # and development-acceptance checks above. No test value chose the run.
        accessed = True
        development, config, development_reference, manifest, splits, training = checked_inputs(
            paths["bank"], paths["reference"], paths["checkpoint"], Path(protocol["development_protocol"]))
        task, evaluation = config["task"], protocol["evaluation"]
        arrays = checked_reference_arrays(paths["reference"] / "reference.npz", development,
                                         task, splits["validation"])
        case_ids, observations, truth = selected_test_cases(splits["test"], protocol)
        parameters = arrays["parameters"]
        thresholds = protocol["reference_gates"]
        cases, weights = [], []
        for case_id, observation in zip(case_ids, observations, strict=True):
            one_weights, case = case_reference(parameters, arrays["reference_prices"],
                                               arrays["target_estimates"], observation,
                                               task["context"]["quote_noise_std"], thresholds)
            cases.append({"case_id": str(case_id), **case})
            reference_ids.append(str(case_id))
            weights.append(one_weights)
            print(f"{case_id}: ESS {case['metrics']['min_ess']:.1f}, reference passed {case['passed']}", flush=True)
        weights = np.asarray(weights)
        gates = {key: all(case["gates"][key] for case in cases) for key in thresholds}
        metrics = {key: (min(case["metrics"][key] for case in cases) if key == "min_ess" else
                         max(case["metrics"][key] for case in cases))
                   if all(case["metrics"][key] is not None for case in cases) else None for key in thresholds}
        reference_artifact = out / "reference-checks.npz"
        np.savez_compressed(reference_artifact, case_ids=case_ids, observations=observations, truth=truth,
                            parameters=parameters, reference_prices=arrays["reference_prices"], weights=weights,
                            target_estimates=arrays["target_estimates"],
                            target_standard_errors=arrays["target_standard_errors"])
        common = {
            "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q", "split": "test",
            "protocol_sha256": _sha256(protocol_path), "input_sha256": protocol["input_sha256"],
            "source_sha256": protocol["frozen_source_sha256"],
            "development_protocol_sha256": _sha256(Path(protocol["development_protocol"])),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "training_report_sha256": _sha256(paths["checkpoint"] / "training.json"),
            "validation_report_sha256": _sha256(paths["validation_report"]),
            "development_reference_report_sha256": _sha256(paths["reference"] / "reference.json"),
            "reference_bank_manifest_sha256": development_reference["reference_bank_manifest_sha256"],
            "bank_manifest_sha256": _sha256(paths["bank"] / "manifest.json"),
            "observation_law": protocol["observation_law"], "target_estimand": protocol["target_pricing"]["estimand"],
            "reserved_test_accessed": True,
        }
        reference_finished = time.perf_counter()
        reference_report = {
            **common, "passed": bool(all(gates.values())), "case_ids": case_ids.tolist(),
            "thresholds": thresholds, "metrics": metrics, "gates": gates, "cases": cases,
            "artifact_sha256": {reference_artifact.name: _sha256(reference_artifact)},
            "neural_test_evaluated": False, "reserved_test_evaluated": True, "full_test_evaluated": False,
            "reference_evaluated_case_ids": case_ids.tolist(),
            "neural_evaluated_case_ids": [], "sbc_evaluated_case_ids": [],
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "timing_seconds": {"input_and_reference_checks": reference_finished - started},
            "limits": ["Same independent 8192-row prior reference; no reference or target repricing.",
                       "All 16 declared cases retained; a failed reference blocks neural evaluation.",
                       "Finite-bank precision checks do not prove posterior support or continuum accuracy."],
        }
        _write_json(out / "reference.json", reference_report)
        if not reference_report["passed"]:
            status = {**common, "status": "blocked_by_reference", "passed": False,
                      "reference_report_sha256": _sha256(out / "reference.json"),
                      "neural_test_evaluated": False, "sbc_test_evaluated": False,
                      "reserved_test_evaluated": True, "full_test_evaluated": False,
                      "reference_evaluated_case_ids": case_ids.tolist(),
                      "neural_evaluated_case_ids": [], "sbc_evaluated_case_ids": [],
                      "test_cohort_status": "First 16 cases consumed; remaining 240 not scientifically evaluated; cohort is no longer wholly untouched.",
                      "decision": "Stop. Preserve all 16 reference outcomes; no neural sampling, case dropping or budget change.",
                      "timing_seconds": {"total": time.perf_counter() - started}}
            _write_json(out / "run-status.json", status)
            return status

        # Framework imports first occur inside build_workflow, after every
        # conditional reference gate has passed. Truth never enters conditions.
        workflow = inference.build_workflow(task)
        workflow.load_approximator(paths["checkpoint"] / "model.keras")
        loaded = time.perf_counter()
        draws = joint(workflow.sample(num_samples=evaluation["conditional_draws"],
                                      conditions=inference.conditioning_data(observations, task),
                                      batch_size=8, seed=evaluation["sample_seed"]))
        neural_evaluated = True
        neural_ids = case_ids.tolist()
        if draws.shape != (16, evaluation["conditional_draws"], 2):
            raise ValueError("complete matched conditional draw budget required")
        sampled = time.perf_counter()
        bounds = np.asarray(task["prior_bounds"])
        prior_rng = np.random.default_rng(evaluation["prior_control_seed"])
        prior_draws = prior_rng.uniform(bounds[:, 0], bounds[:, 1],
                                       (len(case_ids), evaluation["prior_control_draws"], 2))
        neural_cases, prior_cases, summaries, targets, target_ses = [], [], [], [], []
        for index, (one_draws, one_weights) in enumerate(zip(draws, weights, strict=True)):
            target_spec = protocol["target_pricing"]
            prices, errors = price_parameters(task, one_draws, protocol["targets"],
                                              n_paths=target_spec["neural_paths_per_draw"],
                                              n_steps=target_spec["n_steps"], seed=target_spec["neural_seed"],
                                              case_id=index)
            case = reference.conditional_metrics(one_draws, parameters, one_weights, prior_bounds=bounds)
            case.update(target_mean_metrics(prices, arrays["target_estimates"], one_weights, errors))
            summary = reference.weighted_summary(parameters, one_weights)
            case.update(case_id=str(case_ids[index]), reference_summary=summary)
            neural_cases.append(case)
            summaries.append(summary)
            targets.append(prices)
            target_ses.append(errors)
            prior_case = reference.conditional_metrics(prior_draws[index], parameters, one_weights, prior_bounds=bounds)
            prior_case["case_id"] = str(case_ids[index])
            prior_cases.append(prior_case)
            print(f"Priced held-out posterior mean case {index + 1}/{len(draws)}", flush=True)
        priced = time.perf_counter()
        lo, hi = protocol["sbc_ids"]
        audit = {key: splits["test"][key][lo:hi] for key in ("example_id", "observations", "parameters")}
        audit_draws = joint(workflow.sample(num_samples=evaluation["sbc_draws"],
                                          conditions=inference.conditioning_data(audit["observations"], task),
                                          batch_size=16, seed=evaluation["sample_seed"] + 1))
        if audit_draws.shape != (256, evaluation["sbc_draws"], 2):
            raise ValueError("complete matched SBC draw budget required")
        sbc = reference.randomized_sbc(audit_draws, audit["parameters"],
                                      seed=evaluation["rank_randomization_seed"], alpha=evaluation["sbc_family_alpha"])
        prior_audit_draws = prior_rng.uniform(bounds[:, 0], bounds[:, 1], audit_draws.shape)
        prior_sbc = reference.randomized_sbc(prior_audit_draws, audit["parameters"],
                                            seed=evaluation["rank_randomization_seed"] + 1,
                                            alpha=evaluation["sbc_family_alpha"])
        sbc_ids = audit["example_id"].tolist()
        sbc_finished = time.perf_counter()
        metrics = aggregate(neural_cases)
        gates = validation_gates(metrics, sbc, evaluation["gates"])
        rank_arrays = {key: np.asarray(sbc.pop(key)) for key in ("ranks", "randomized_ranks")}
        rank_arrays.update({"prior_" + key: np.asarray(prior_sbc.pop(key))
                            for key in ("ranks", "randomized_ranks")})
        checks_path = out / "test-checks.npz"
        np.savez_compressed(checks_path, case_ids=case_ids, observations=observations, truth=truth,
                            conditional_draws=draws, prior_conditional_draws=prior_draws,
                            reference_parameters=parameters, reference_weights=weights,
                            reference_target_estimates=arrays["target_estimates"],
                            reference_target_standard_errors=arrays["target_standard_errors"],
                            target_estimates=np.asarray(targets), target_standard_errors=np.asarray(target_ses),
                            reference_mean=np.asarray([s["mean"] for s in summaries]),
                            reference_std=np.asarray([s["std"] for s in summaries]),
                            reference_correlation=np.asarray([s["correlation"] for s in summaries]),
                            audit_ids=audit["example_id"], audit_observations=audit["observations"],
                            audit_truth=audit["parameters"], audit_draws=audit_draws,
                            prior_audit_draws=prior_audit_draws, **rank_arrays)
        passed = bool(all(gates.values()))
        report = {
            **common, "status": "accepted" if passed else "provisional", "passed": passed,
            "reference_report_sha256": _sha256(out / "reference.json"), "checks_sha256": _sha256(checks_path),
            "metrics": metrics, "thresholds": evaluation["gates"], "gates": gates,
            "case_ids": case_ids.tolist(), "conditional_draws_per_case": int(draws.shape[1]), "cases": neural_cases,
            "sbc": sbc,
            "prior_baseline": {"cases": prior_cases, "metrics": aggregate(prior_cases, include_targets=False),
                               "sbc": prior_sbc, "draws_per_case": evaluation["prior_control_draws"],
                               "mean": bounds.mean(axis=1).tolist(),
                               "std": (np.diff(bounds, axis=1)[:, 0] / np.sqrt(12)).tolist(),
                               "scope": "Independent prior draws ignore all observations; SBC may pass despite incorrect conditioning."},
            "mean_reference_mass_of_90_interval_by_parameter": np.mean([
                c["reference_mass_of_90_interval"] for c in neural_cases], axis=0).tolist(),
            "neural_test_evaluated": True, "sbc_test_evaluated": True, "reserved_test_evaluated": True,
            "full_test_evaluated": True,
            "reference_evaluated_case_ids": case_ids.tolist(), "neural_evaluated_case_ids": case_ids.tolist(),
            "sbc_evaluated_case_ids": audit["example_id"].tolist(),
            "test_cohort_status": "All 256 test cases consumed; reproduction is not a fresh acceptance attempt.",
            "acceptance_scope": protocol.get("decision_rule", {}).get(
                "all_neural_gates_pass" if passed else "any_neural_gate_failure",
                "Declared finite synthetic task and diagnostics only."),
            "timing_seconds": {"input_and_reference_checks": reference_finished - started,
                               "checkpoint_loading": loaded - reference_finished,
                               "conditional_sampling_including_compile": sampled - loaded,
                               "target_mean_pricing_and_conditional_checks": priced - sampled,
                               "sbc_and_prior_control": sbc_finished - priced,
                               "total": time.perf_counter() - started},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         **{package: version(package) for package in ("bayesflow", "keras", "jax")}},
            "limits": ["Acceptance concerns this frozen synthetic Q task, finite numerical law, candidate and declared diagnostics.",
                       "Reference-bank precision is finite; shared bank error and support limitations remain.",
                       "256 prior-predictive marginal SBC cases have limited power and are not joint or conditional certification.",
                       "Conditional 16-case audits overlap the SBC cohort; these are complementary dependent checks.",
                       "Posterior target means concern the fixed 512-step law, not price quantiles or continuum accuracy.",
                       "No real-market adequacy, P risk probabilities, market-chain forecasts or broad speed superiority established.",
                       "This test cohort is now consumed for this candidate; no tuning, replacement or automatic follow-up."],
        }
        _write_json(out / "neural-test.json", report)
        _write_json(out / "run-status.json", {key: report[key] for key in
                    ("experiment_id", "status", "passed", "protocol_sha256", "reserved_test_accessed",
                     "neural_test_evaluated", "sbc_test_evaluated", "reserved_test_evaluated", "full_test_evaluated",
                     "reference_evaluated_case_ids", "neural_evaluated_case_ids", "sbc_evaluated_case_ids",
                     "test_cohort_status", "timing_seconds")})
        return report
    except Exception as error:
        _write_json(out / "run-status.json", {
            "experiment_id": protocol["experiment_id"], "status": "execution_failed", "passed": False,
            "error": str(error), "protocol_sha256": _sha256(protocol_path),
            "reserved_test_accessed": accessed, "neural_test_evaluated": neural_evaluated,
            "reserved_test_evaluated": bool(reference_ids), "full_test_evaluated": len(sbc_ids) == 256,
            "reference_evaluated_case_ids": reference_ids, "neural_evaluated_case_ids": neural_ids,
            "sbc_evaluated_case_ids": sbc_ids,
            "decision": "Preserve partial artifacts; no automatic scientific rerun or reinterpretation.",
            "timing_seconds": {"total": time.perf_counter() - started},
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/rough_heldout_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.protocol, args.out)
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "metrics": report.get("metrics"), "gates": report.get("gates")}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("Held-out acceptance unresolved; recorded result retained without automatic follow-up.")


if __name__ == "__main__":
    main()

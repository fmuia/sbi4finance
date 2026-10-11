"""Audit one frozen defensive neural importance sampler; preserve old failures."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
from sbi4finance import hkde_correction as correction, hkde_diagnostics as diagnostics
from sbi4finance import hkde_inference as inference, hkde_reference as reference
from sbi4finance import hkde_surrogate as surrogate, option_bank
from check_hkde_reference import compare, load_grid, jsonable, sha
from evaluate_hkde import aggregate


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def case_seed(protocol, split, phase, index):
    return int(np.random.SeedSequence([protocol["root_seed"],
        ("validation", "test").index(split), phase, index]).generate_state(1)[0])


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("check_hkde_reference.py"),
             Path(__file__).with_name("evaluate_hkde.py")]
    paths += [Path(module.__file__) for module in
              (correction, inference, reference, surrogate, diagnostics)]
    return {path.name: sha(path) for path in paths}


def load_inputs(protocol_path=Path("configs/hkde_correction_boundary_protocol.json")):
    """Bind task, frozen checkpoint, numerical reference and accepted surrogate."""
    protocol_path = Path(protocol_path).resolve()
    root = protocol_path.parents[1]
    protocol = json.loads(protocol_path.read_text())
    baseline_path = root / protocol["baseline_protocol"]
    baseline = json.loads(baseline_path.read_text())
    config_path = root / protocol["task_config"]
    configuration = json.loads(config_path.read_text())
    task = configuration["task"]
    old_dir = root / protocol["baseline_results"]
    checkpoint_dir = root / protocol["checkpoint"]
    fast_dir = root / protocol["surrogate_results"]
    paths = [protocol_path, baseline_path, config_path,
             old_dir / "reference.json", old_dir / "reference-grid.npz",
             old_dir / "neural-validation.json", checkpoint_dir / "model.keras",
             checkpoint_dir / "training.json", checkpoint_dir / "task.json",
             fast_dir / "report.json", fast_dir / "model.npz"]
    if "predecessor_results" in protocol:
        paths.append(root / protocol["predecessor_results"] / "validation.json")
    bindings = {str(path.relative_to(root)): sha(path) for path in paths}
    ref_report = json.loads((old_dir / "reference.json").read_text())
    training = json.loads((checkpoint_dir / "training.json").read_text())
    fast_report = json.loads((fast_dir / "report.json").read_text())
    if (not ref_report["passed"] or ref_report["task"] != task
            or ref_report["protocol_sha256"] != sha(baseline_path)
            or ref_report["bank_config_sha256"] != sha(config_path)
            or ref_report["reference_grid_sha256"] != sha(old_dir / "reference-grid.npz")
            or training["protocol_sha256"] != sha(baseline_path)
            or training["checkpoint_sha256"] != sha(checkpoint_dir / "model.keras")
            or training["reference_report_sha256"] != sha(old_dir / "reference.json")
            or training["bank_configuration"] != configuration
            or not training["same_seed_reload_exact"]
            or json.loads((checkpoint_dir / "task.json").read_text()) != task
            or not fast_report["passed"] or fast_report["task"] != task
            or fast_report["targets"] != baseline["targets"]
            or fast_report["artifact_sha256"]["model.npz"] != sha(fast_dir / "model.npz")):
        raise ValueError("matching frozen checkpoint, accepted numerical reference and surrogate required")
    for name in ("hkde_reference.py", *option_bank.source_files("hkde")):
        if ref_report["source_sha256"][name] != sha(Path(reference.__file__).with_name(name)):
            raise ValueError(f"numerical reference source changed: {name}")
    if training["source_sha256"]["hkde_inference.py"] != sha(inference.__file__):
        raise ValueError("frozen inference source changed")
    if fast_report["source_sha256"]["hkde_surrogate.py"] != sha(surrogate.__file__):
        raise ValueError("accepted surrogate source changed")
    if protocol["gates"] != baseline["evaluation"]["gates"]:
        raise ValueError("correction must preserve baseline inference thresholds")
    grid, saved = load_grid(old_dir / "reference-grid.npz", task, baseline["targets"])
    model = surrogate.load_model(fast_dir / "model.npz")
    return protocol, baseline, task, grid, saved, model, bindings


def require_validation(path, protocol, bindings):
    """Guard BEFORE fresh-test observations or truths are generated."""
    if path is None:
        raise ValueError("new test requires a matching passed validation report")
    report = json.loads(Path(path).read_text())
    if (not report["passed"] or report["split"] != "validation"
            or report["experiment_id"] != protocol["experiment_id"]
            or report["input_sha256"] != bindings
            or report["source_sha256"] != source_hashes()
            or not report["calibration"]["rank_diagnostic_passed"]
            or not all(report["gates"].values())
            or not all(report["precision"]["gates"].values())):
        raise ValueError("failed or mismatched validation cannot unlock the new test")


def coarse_grid(fine):
    """Reuse matching 65-point nodes from 129; recompute trapezoid masses."""
    if fine["shape"] != (129, 129):
        raise ValueError("frozen follow-up requires the accepted 129x129 grid")
    axes = tuple(axis[::2] for axis in fine["axes"])
    weights = []
    for axis in axes:
        gaps = np.diff(axis)
        weights.append(np.r_[gaps[0], gaps[:-1] + gaps[1:], gaps[-1]] / (2 * (axis[-1] - axis[0])))
    result = dict(fine, axes=axes, shape=(65, 65),
                  quadrature_weights=np.outer(*weights).ravel())
    for key in ("parameters", "prices", "target_prices"):
        values = fine[key]
        result[key] = values.reshape(129, 129, -1)[::2, ::2].reshape(-1, values.shape[-1])
    return result


def summarize_cases(cases, efficiencies, limits, thresholds):
    metrics = aggregate(cases)
    gates = {key: value <= thresholds[key] for key, value in metrics.items()}
    proposal = {"min_ess_fraction": min(row["ess_fraction"] for row in efficiencies),
                "max_weight": max(row["max_weight"] for row in efficiencies)}
    gates.update(proposal_ess=proposal["min_ess_fraction"] >= limits["min_ess_fraction"],
                 proposal_max_weight=proposal["max_weight"] <= limits["max_weight"])
    return {"metrics": metrics, "gates": gates, "cases": cases, "proposal_diagnostics": proposal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/hkde_correction_boundary_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--validation-report", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol, baseline, task, grid, saved, model, bindings = load_inputs(args.protocol)
    if args.split == "test":
        require_validation(args.validation_report, protocol, bindings)
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.protocol.resolve().parents[1]
    config = json.loads((root / protocol["task_config"]).read_text())
    if args.split == "validation":
        lo, hi = protocol["calibration"]["validation_ids"]
        audit = option_bank.generate_examples(config, "validation", range(lo, hi))
        observations, truth, ids = [saved[key] for key in ("observations", "truth", "case_ids")]
    else:
        config = dict(config, root_seed=protocol["test"]["root_seed"],
                      splits={"train": 1, "validation": 1, "test": protocol["test"]["count"]})
        audit = option_bank.generate_examples(config, "test", range(protocol["test"]["count"]))
        selected = protocol["test"]["conditional_ids"]
        observations, truth, ids = [audit[key][selected] for key in ("observations", "parameters", "example_id")]
        np.savez_compressed(args.out / "new-test-panels.npz", **audit)
    sigma = task["context"]["quote_noise_std"]
    differences, _, weights = compare(coarse_grid(grid), grid, observations, sigma, protocol["quantiles"])
    ref_metrics = {key: max(row[key] for row in differences) for key in differences[0]}
    ref_passed = all(value <= baseline["reference"]["gates"][key] for key, value in ref_metrics.items())
    report = {"experiment_id": protocol["experiment_id"], "split": args.split,
              "evidence_scope": "S", "measure": "Q", "task": task, "targets": baseline["targets"],
              "input_sha256": bindings, "source_sha256": source_hashes(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__},
              "thresholds": protocol["gates"], "passed": False,
              "reference": {"passed": ref_passed, "metrics": ref_metrics},
              "new_test_evaluated": args.split == "test", "old_test_evaluated": False,
              "panel_configuration": config, "participant_samples": protocol["conditional"]["participant_samples"],
              "precision_samples": protocol["conditional"]["samples"],
              "limits": ["Self-normalized finite importance estimates are approximate, not exact posterior samples.",
                         "Positive prior support and high observed ESS cannot certify absence of missed modes.",
                         "SIR SBC audits the finite-pool algorithm; resamples share a proposal cloud.",
                         "The nine numerical likelihood prices are still required; only the two target prices use the surrogate.",
                         "Original neural failure and original test remain unchanged; S evidence is not market validation."]}
    if args.split == "test":
        report["test_panels_sha256"] = sha(args.out / "new-test-panels.npz")
        report["validation_report_sha256"] = sha(args.validation_report)
    if not ref_passed:
        write_json(args.out / f"{args.split}.json", report)
        raise SystemExit("New reference refinement failed; no interpretation or automatic refinement.")
    workflow = inference.build_workflow(task)
    workflow.load_approximator(root / protocol["checkpoint"] / "model.keras")
    timing = {"load_generate_and_reference": time.perf_counter() - started,
              "proposal": 0., "likelihood": 0., "target_surrogate_and_checks": 0.}
    cases, precise, efficiencies, precise_efficiencies, pools = [], [], [], [], []
    for i, (observation, one_weights) in enumerate(zip(observations, weights, strict=True)):
        tick = time.perf_counter()
        proposed = correction.propose(workflow, task, observation,
             num_samples=protocol["conditional"]["samples"], seed=case_seed(protocol, args.split, 0, i),
             prior_fraction=protocol["proposal"]["prior_fraction"],
             boundary_fraction=protocol["proposal"].get("boundary_fraction", 0.0))
        timing["proposal"] += time.perf_counter() - tick
        tick = time.perf_counter()
        fixed = correction.correct(proposed["parameters"], proposed["log_proposal"], observation, task)
        timing["likelihood"] += time.perf_counter() - tick
        tick = time.perf_counter()
        parameters, all_weights = proposed["parameters"], fixed["weights"]
        targets = surrogate.predict(model, parameters, task=task, contracts=baseline["targets"])
        summary = reference.summarize(grid, one_weights, protocol["quantiles"])
        for count, output, eff in ((protocol["conditional"]["participant_samples"], cases, efficiencies),
                                  (len(parameters), precise, precise_efficiencies)):
            w = all_weights[:count] / all_weights[:count].sum()
            row = correction.summarize(parameters[:count], w, targets[:count], reference_summary=summary,
                  probabilities=protocol["quantiles"], blocks=protocol["conditional"]["uncertainty_blocks"])
            row["case_id"] = str(ids[i])
            row["ess_fraction"] = float(1 / np.sum(w*w) / count)
            row["max_weight"] = float(w.max())
            output.append(row)
            eff.append(row)
        pools.append((parameters, all_weights, targets, summary["target_quantiles"],
                      proposed["log_proposal"], fixed["log_likelihood"], proposed["is_prior"], proposed["is_boundary"]))
        timing["target_surrogate_and_checks"] += time.perf_counter() - tick
        print(f"{args.split} conditional {i+1}/{len(ids)}", flush=True)
    report.update(summarize_cases(cases, efficiencies, protocol["conditional"], protocol["gates"]))
    report["precision"] = summarize_cases(precise, precise_efficiencies, protocol["conditional"], protocol["gates"])
    conditional_passed = all(report["gates"].values()) and all(report["precision"]["gates"].values())
    checks = {"case_ids": ids, "observations": observations, "truth": truth,
              "parameters": np.asarray([p[0] for p in pools]), "weights": np.asarray([p[1] for p in pools]),
              "target_prices": np.asarray([p[2] for p in pools]),
              "reference_target_quantiles": np.asarray([p[3] for p in pools]),
              "log_proposal": np.asarray([p[4] for p in pools]),
              "log_likelihood": np.asarray([p[5] for p in pools]),
              "is_prior": np.asarray([p[6] for p in pools]),
              "is_boundary": np.asarray([p[7] for p in pools])}
    if conditional_passed:
        tick = time.perf_counter()
        audit_draws, audit_eff, audit_max = [], [], []
        settings = protocol["calibration"]
        for i, observation in enumerate(audit["observations"]):
            proposal = correction.propose(workflow, task, observation, num_samples=settings["proposal_samples"],
                       seed=case_seed(protocol, args.split, 1, i), prior_fraction=protocol["proposal"]["prior_fraction"],
             boundary_fraction=protocol["proposal"].get("boundary_fraction", 0.0))
            fixed = correction.correct(proposal["parameters"], proposal["log_proposal"], observation, task)
            audit_eff.append(fixed["ess_fraction"])
            audit_max.append(fixed["max_weight"])
            audit_draws.append(correction.resample(proposal["parameters"], fixed["weights"],
                               num_samples=settings["resampled_draws"], seed=case_seed(protocol, args.split, 2, i)))
            if (i+1) % 32 == 0:
                print(f"{args.split} calibration {i+1}/{len(audit['observations'])}", flush=True)
        sbc = diagnostics.randomized_sbc(np.asarray(audit_draws), audit["parameters"],
                    seed=case_seed(protocol, args.split, 3, 0), alpha=settings["family_alpha"])
        checks.update(calibration_ids=audit["example_id"], calibration_truth=audit["parameters"],
                      calibration_ranks=np.asarray(sbc.pop("ranks")),
                      calibration_randomized_ranks=np.asarray(sbc.pop("randomized_ranks")),
                      calibration_ess_fraction=audit_eff, calibration_max_weight=audit_max)
        sbc.update(min_ess_fraction=min(audit_eff), max_weight=max(audit_max),
                   proposal_samples=settings["proposal_samples"], scope=settings["scope"])
        report["calibration"] = sbc
        report["gates"].update(sir_sbc=sbc["rank_diagnostic_passed"],
                 calibration_ess=min(audit_eff) >= settings["min_ess_fraction"],
                 calibration_max_weight=max(audit_max) <= settings["max_weight"])
        timing["calibration_including_proposal_likelihood_resampling"] = time.perf_counter() - tick
    else:
        report["calibration"] = {"skipped": "conditional gates failed", "rank_diagnostic_passed": False}
    destination = args.out / f"{args.split}-checks.npz"
    np.savez_compressed(destination, **checks)
    report.update(checks_sha256=sha(destination), passed=bool(conditional_passed and all(report["gates"].values())))
    timing["total"] = time.perf_counter() - started
    report["timing_seconds"] = timing
    write_json(args.out / f"{args.split}.json", report)
    print(json.dumps(jsonable({key: report[key] for key in ("passed", "metrics", "gates", "proposal_diagnostics")}), indent=2))
    if not report["passed"]:
        raise SystemExit("Frozen correction gates failed; no further stage or automatic tuning.")


if __name__ == "__main__":
    main()

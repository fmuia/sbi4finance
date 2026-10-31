"""Freeze candidate identities, assess untouched validation and guard one test opening.

This records marginal development evidence, never conditional certification or
automatic publication. Statistical screen failures may be characterized on tests
only after an explicit review retaining the identical frozen candidate.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import bank, inference

MODELS = tuple(bank.MODELS)
LENGTHS = (1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756, 1260)
DEFAULT_SEED = 20261101001


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sealed(value):
    value = dict(value)
    value["receipt_sha256"] = inference.fingerprint(value)
    return value


def read_receipt(path):
    value = json.loads(Path(path).read_text())
    digest = value.pop("receipt_sha256")
    if inference.fingerprint(value) != digest:
        raise ValueError("receipt hash differs; frozen decisions cannot be revised")
    return {**value, "receipt_sha256": digest}


def read_selection(path):
    value = read_receipt(path)
    canonical = read_receipt(Path(value["state_directory"]) / "selection.json")
    if value != canonical:
        raise ValueError("selection differs from its canonical frozen receipt")
    return value


def exclusive_json(path, value):
    """Exclusive, durable publication: failures leave a consumed guard, not a retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_once(selection, name, out, value):
    canonical = Path(selection["state_directory"]) / name
    out = Path(out).resolve()
    if out != canonical and out.exists():
        raise FileExistsError(out)
    exclusive_json(canonical, value)
    if out != canonical:
        exclusive_json(out, value)
    return value


def checked_json(path):
    path = Path(path)
    return json.loads(path.read_text()), inference.sha256(path)


def _check_candidate(record, bank_path, checkpoint_path):
    bank_path, checkpoint_path = Path(bank_path).resolve(), Path(checkpoint_path).resolve()
    if bank_path != Path(record["bank"]) or checkpoint_path != Path(record["checkpoint"]):
        raise ValueError("bank/checkpoint paths differ from the frozen candidate")
    for path, digest in ((bank_path / "manifest.json", record["bank_manifest_sha256"]),
                         (checkpoint_path / "model.keras", record["model_sha256"]),
                         (checkpoint_path / "training.json", record["training_sha256"]),
                         (checkpoint_path / "task.json", record["task_file_sha256"])):
        if inference.sha256(path) != digest:
            raise ValueError(f"frozen artifact changed: {path}")
    manifest = bank.load_manifest(bank_path)
    if manifest["task_sha256"] != record["task_sha256"]:
        raise ValueError("frozen task differs from bank")
    return manifest


def freeze_selection(run_root, out, *, seed=DEFAULT_SEED):
    run_root, out = Path(run_root).resolve(), Path(out).resolve()
    inference.integer(seed, "confirmation seed", minimum=0)
    submission, submission_hash = checked_json(run_root / "submission-receipt.json")
    summary, summary_hash = checked_json(run_root / "scaleup-summary.json")
    if (not submission.get("submission_complete") or summary.get("reserved_test_evaluated")
            or summary.get("submission_receipt_sha256") != submission_hash
            or summary.get("source_commit") != submission["source_commit"]):
        raise ValueError("incomplete campaign or inconsistent source/submission identity")
    state = run_root / "workshop-evaluation-guard"
    if out.exists() or (state / "selection.json").exists():
        raise FileExistsError("selection already exists; revisions are forbidden")
    models = {}
    for name in MODELS:
        directory = run_root / "fits" / name / "1m"
        bank_path = run_root / "banks" / name
        manifest = bank.load_manifest(bank_path)
        training, training_hash = checked_json(directory / "training.json")
        task, task_hash = checked_json(directory / "task.json")
        if (task != manifest["task"] or bank.task_fingerprint(task) != bank.task_fingerprint(bank.teaching_task(name))
                or training["task_sha256"] != manifest["task_sha256"]
                or training["bank_manifest_sha256"] != inference.sha256(bank_path / "manifest.json")
                or training["model_sha256"] != inference.sha256(directory / "model.keras")
                or training.get("training_trajectories") != 1_000_000
                or training.get("epochs_completed") != 5 or training.get("checkpoint_selection") != "final"
                or training.get("same_seed_reload_exact") is not True or training.get("status") != "completed"
                or manifest["split_counts"] != {"train": 1_000_000, "validation": 4000, "test": 4000}):
            raise ValueError(f"{name} does not match the completed 1M final-endpoint recipe")
        development = {}
        for label in ("100k", "1m"):
            path = run_root / "evaluation" / name / label / "evaluation.json"
            evaluation, digest = checked_json(path)
            if (evaluation["task_sha256"] != manifest["task_sha256"]
                    or evaluation["bank_manifest_sha256"] != training["bank_manifest_sha256"]
                    or evaluation.get("split") != "validation" or evaluation.get("offset") != 0
                    or evaluation.get("cases") != 1024 or evaluation.get("reserved_test_evaluated")):
                raise ValueError("development cohort/identity differs")
            if label == "1m" and evaluation["checkpoint_sha256"] != training["model_sha256"]:
                raise ValueError("selected model differs from its development evaluation")
            development[label] = {"path": str(path), "sha256": digest,
                "checkpoint_sha256": evaluation["checkpoint_sha256"],
                "coverage90_ranges": [min(min(row["coverage"]["0.9"]["coverage"]) for row in evaluation["lengths"].values()),
                                      max(max(row["coverage"]["0.9"]["coverage"]) for row in evaluation["lengths"].values())]}
        comparison_path = run_root / "comparison" / name / "comparison.json"
        comparison, comparison_hash = checked_json(comparison_path)
        if comparison.get("status") != "development_comparison_complete" or comparison.get("reserved_test_evaluated"):
            raise ValueError("development comparison is missing or incomplete")
        models[name] = {"checkpoint": str(directory), "bank": str(bank_path), "task": task,
            "candidate_scope": ("numerical_development_prototype_with_unrepresented_drift_atoms" if name == "heston"
                                else "approximate_neural_development_candidate"),
            "task_sha256": manifest["task_sha256"], "task_file_sha256": task_hash,
            "bank_manifest_sha256": training["bank_manifest_sha256"], "training_sha256": training_hash,
            "model_sha256": training["model_sha256"], "runtime": training["runtime"],
            "source_sha256": training["source_sha256"], "development": development,
            "comparison": {"path": str(comparison_path), "sha256": comparison_hash,
                           "known_development_screen_failures": "Retain every reported coverage/risk failure; selection does not imply a universal 1M advantage."}}
    cells = len(LENGTHS) * sum(len(record["task"]["parameter_names"]) for record in models.values())
    value = sealed({"format": "path_workshop_selection_v1", "created_at": stamp(),
        "state_directory": str(state), "run_root": str(run_root), "models": models,
        "source_commit": submission["source_commit"], "source_archive_sha256": submission.get("source_archive_sha256"),
        "runtime_lock_sha256": submission["runtime_lock_sha256"],
        "submission_receipt_sha256": submission_hash, "scaleup_summary_sha256": summary_hash,
        "selection": "provisional_common_1m_final_endpoint_recipe",
        "rationale": "Consistent workshop recipe chosen from existing development evidence, retaining both100k/1M comparisons; no universal larger-bank superiority or calibration acceptance is claimed.",
        "design": {"lengths": list(LENGTHS), "draws": 512, "seed": seed,
            "confirmation": {"split": "validation", "offset": 1024, "cases": 2976},
            "final": {"split": "test", "offset": 0, "cases": 4000}},
        "gate_policy": {"family_cells": cells, "coverage_nominal": .9, "coverage_family_alpha": .025,
            "coverage_method": "exact Clopper-Pearson; Bonferroni across every model/length/target",
            "heston_drift_coverage": "one-sided minimum nominal coverage; central intervals for a mixed/atomic posterior can legitimately overcover",
            "risk_family_alpha": .025, "risk_method": "two-sided bounded paired-loss empirical Bernstein; Bonferroni across every model/length/target",
            "risk_guard": "no cell with demonstrated worse risk than analytic prior mean (lower bound>0); require at least one non-drift target informative (upper bound<0) at1260 permodel; unresolved cells explicitly broad/unresolved",
            "execution_guard": "identities/cohorts match; zero nonfinite and support violations",
            "interpretation": "combined family error bound0.05 for the two declared marginal screens under iid trajectories; no conditional/copula certification"},
        "scientific_acceptance": False, "reserved_test_evaluated": False,
        "baseline_replaced": False})
    return publish_once(value, "selection.json", out, value)


def _request(selection, bank_path, checkpoint_path, *, phase, cases, draws, lengths, seed, offset):
    bank_path, checkpoint_path = Path(bank_path).resolve(), Path(checkpoint_path).resolve()
    matches = [(name, record) for name, record in selection["models"].items()
               if Path(record["bank"]) == bank_path and Path(record["checkpoint"]) == checkpoint_path]
    if len(matches) != 1:
        raise ValueError("request does not identify exactly one frozen candidate")
    name, record = matches[0]
    design = selection["design"]
    expected = design[phase]
    if (cases != expected["cases"] or offset != expected["offset"] or draws != design["draws"]
            or list(lengths) != design["lengths"] or seed != design["seed"]):
        raise ValueError("evaluation request differs from the frozen design")
    _check_candidate(record, bank_path, checkpoint_path)
    return name, record


def validate_confirmation_request(selection_path, bank_path, checkpoint_path, *, cases, draws, lengths, seed, offset):
    selection = read_selection(selection_path)
    return _request(selection, bank_path, checkpoint_path, phase="confirmation", cases=cases,
                    draws=draws, lengths=lengths, seed=seed, offset=offset)[0]


def prior_moments(task):
    means, variances = [], []
    for (lo, hi), kind in zip(task["prior_bounds"], task["prior_types"], strict=True):
        if kind == "uniform":
            mean, variance = (lo + hi) / 2, (hi - lo) ** 2 / 12
        else:
            mean = (hi - lo) / np.log(hi / lo)
            variance = (hi ** 2 - lo ** 2) / (2 * np.log(hi / lo)) - mean ** 2
        means.append(mean)
        variances.append(variance)
    return np.asarray(means), np.asarray(variances)


def simultaneous_screens(arrays, report, task, policy):
    truth = arrays["truth"]
    n, dimension = truth.shape
    center, variance = prior_moments(task)
    width2 = np.diff(np.asarray(task["prior_bounds"]), axis=1)[:, 0] ** 2
    cells = policy["family_cells"]
    tail = policy["coverage_family_alpha"] / (2 * cells)
    logarithm = np.log(4 * cells / policy["risk_family_alpha"])
    result, execution, calibration, harm, informative = {}, True, True, False, False
    for length, metric in report["lengths"].items():
        mean, low, high = [arrays[f"{key}_{length}"] for key in ("posterior_mean", "lower90", "upper90")]
        if (any(x.shape != truth.shape or not np.isfinite(x).all() for x in (mean, low, high))
                or np.any(low > high)):
            raise ValueError("invalid case-summary posterior arrays")
        bounds = np.asarray(task["prior_bounds"])
        support = all(np.all((x >= bounds[:, 0]) & (x <= bounds[:, 1])) for x in (mean, low, high))
        execution &= support and metric["nonfinite_count"] == 0 and not any(metric["out_of_prior_fraction"])
        hits = ((truth >= low) & (truth <= high)).sum(axis=0)
        lower = np.where(hits == 0, 0, beta.ppf(tail, hits, n - hits + 1))
        upper = np.where(hits == n, 1, beta.ppf(1 - tail, hits + 1, n - hits))
        calibrated = (lower <= .9) & (upper >= .9)
        criteria = ["two_sided_nominal_compatibility"] * dimension
        if task.get("model") == "heston" and "drift" in task["parameter_names"]:
            drift = task["parameter_names"].index("drift")
            # Correct central quantile intervals can contain >90% mass when
            # the true posterior has atoms. Retain the same conservative
            # Bonferroni upper bound and test undercoverage only for this cell.
            calibrated[drift] = upper[drift] >= .9
            criteria[drift] = "one_sided_minimum_nominal_coverage_for_atomic_target"
        calibration &= bool(calibrated.all())
        delta = ((mean - truth) ** 2 - (center - truth) ** 2) / width2
        radius = np.sqrt(2 * delta.var(axis=0, ddof=1) * logarithm / n) + 14 * logarithm / (3 * (n - 1))
        estimate = delta.mean(axis=0) * width2 / variance
        risk_low, risk_high = (delta.mean(axis=0) - radius) * width2 / variance, (delta.mean(axis=0) + radius) * width2 / variance
        worse, better = risk_low > 0, risk_high < 0
        harm |= bool(worse.any())
        if int(length) == 1260:
            informative = bool(any(flag and name != "drift" for flag, name in zip(better, task["parameter_names"])))
        result[length] = {"coverage90": (hits / n).tolist(), "coverage90_simultaneous_interval": [lower.tolist(), upper.tolist()],
            "nominal_inside_interval": calibrated.tolist(),
            "coverage_criterion": criteria,
            "prior_normalized_paired_risk_difference": estimate.tolist(),
            "risk_difference_simultaneous_interval": [risk_low.tolist(), risk_high.tolist()],
            "risk_labels": ["demonstrated_degradation" if bad else "informative" if good else "broad_or_unresolved" for bad, good in zip(worse, better)]}
    return {"execution_pass": bool(execution), "calibration_pass": bool(calibration),
            "risk_pass": bool(not harm and informative), "long_history_non_drift_informative": informative,
            "lengths": result}


def _assess(selection_path, evaluations, out, *, phase):
    selection = read_selection(selection_path)
    if set(evaluations) != set(MODELS):
        raise ValueError("all three frozen model evaluations are required")
    panels, inputs = {}, {}
    for name, directory in evaluations.items():
        directory = Path(directory).resolve()
        report, digest = checked_json(directory / "evaluation.json")
        expected = selection["models"][name]
        design = selection["design"]
        cohort = design[phase]
        final = phase == "final"
        if (report.get("split") != cohort["split"] or report.get("offset") != cohort["offset"]
                or report.get("cases") != cohort["cases"] or report.get("draws") != design["draws"]
                or report.get("seed") != design["seed"] or report.get("reserved_test_evaluated") is not final
                or sorted(map(int, report["lengths"])) != design["lengths"]
                or report.get("targets") != expected["task"]["parameter_names"]
                or any(report.get(field) != expected[key] for field, key in
                       (("checkpoint_sha256", "model_sha256"), ("task_sha256", "task_sha256"), ("bank_manifest_sha256", "bank_manifest_sha256")))):
            raise ValueError(f"{phase} identity/cohort/design differs from selection")
        if final:
            opening = read_receipt(Path(selection["state_directory"]) / f"test-opening-{name}.json")
            if (opening["selection_sha256"] != selection["receipt_sha256"]
                    or opening["model_sha256"] != expected["model_sha256"]
                    or opening["output"] != str(directory)
                    or report.get("test_opening_receipt_sha256") != opening["receipt_sha256"]):
                raise ValueError("final report differs from its once-only test-opening receipt")
        manifest = _check_candidate(expected, expected["bank"], expected["checkpoint"])
        path = directory / report["case_summary_file"]
        if inference.sha256(path) != report["case_summary_sha256"]:
            raise ValueError("confirmation arrays changed")
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        ids = [f"{manifest['task_sha256'][:16]}:{manifest['root_seed']}:{cohort['split']}:{i}"
               for i in range(cohort["offset"], cohort["offset"] + cohort["cases"])]
        truth = arrays["truth"]
        bounds = np.asarray(expected["task"]["prior_bounds"])
        if (truth.shape != (cohort["cases"], len(bounds)) or not np.isfinite(truth).all()
                or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1])
                or not np.array_equal(arrays["trajectory_id"], ids)
                or report["case_identities_sha256"] != inference.fingerprint(ids)):
            raise ValueError("confirmation truth/identity arrays differ from declared cohort")
        panels[name] = simultaneous_screens(arrays, report, expected["task"], selection["gate_policy"])
        inputs[name] = {"path": str(directory / "evaluation.json"), "sha256": digest,
                        "case_summary_path": str(path), "case_summary_sha256": report["case_summary_sha256"]}
    execution = all(panel["execution_pass"] for panel in panels.values())
    calibration = all(panel["calibration_pass"] for panel in panels.values())
    risk = all(panel["risk_pass"] for panel in panels.values())
    value = sealed({"format": "path_workshop_final_characterization_v1" if phase == "final" else "path_workshop_confirmation_v1", "created_at": stamp(),
        "selection_sha256": selection["receipt_sha256"], "evaluations": inputs, "models": panels,
        "execution_pass": execution, "calibration_pass": calibration, "risk_pass": risk,
        "passed": execution and calibration and risk,
        "status": "execution_failure" if not execution else "failed_calibration_screen" if not calibration else "failed_risk_screen" if not risk else "development_confirmation_passed",
        "scientific_acceptance": False, "reserved_test_evaluated": phase == "final",
        "scope": "predeclared simultaneous marginal screens; no conditional, all-prefix or copula certification"})
    return publish_once(selection, "final-characterization.json" if phase == "final" else "confirmation-assessment.json", out, value)


def assess_confirmation(selection_path, evaluations, out):
    return _assess(selection_path, evaluations, out, phase="confirmation")


def assess_final(selection_path, evaluations, out):
    return _assess(selection_path, evaluations, out, phase="final")


def review_confirmation(selection_path, confirmation_path, out, *, mode, reviewer, reason):
    selection, confirmation = read_selection(selection_path), read_receipt(confirmation_path)
    canonical = read_receipt(Path(selection["state_directory"]) / "confirmation-assessment.json")
    if (confirmation != canonical or confirmation["selection_sha256"] != selection["receipt_sha256"]
            or not confirmation["execution_pass"]):
        raise ValueError("execution/identity failures block reserved-test opening")
    if mode not in ("confirm_for_workshop", "evaluate_frozen_development_candidate") or not reviewer.strip() or not reason.strip():
        raise ValueError("explicit review mode, reviewer and reason are required")
    if mode == "confirm_for_workshop" and not confirmation["passed"]:
        raise ValueError("workshop confirmation requires all frozen statistical screens")
    for item in confirmation["evaluations"].values():
        if (inference.sha256(item["path"]) != item["sha256"]
                or inference.sha256(item["case_summary_path"]) != item["case_summary_sha256"]):
            raise ValueError("reviewed confirmation artifacts changed")
    value = sealed({"format": "path_workshop_confirmation_review_v1", "created_at": stamp(),
        "selection_sha256": selection["receipt_sha256"], "confirmation_sha256": confirmation["receipt_sha256"],
        "mode": mode, "reviewer": reviewer, "reason": reason,
        "confirmation_status": confirmation["status"], "statistical_screens_passed": confirmation["passed"],
        "scientific_acceptance": False, "candidate_status": "frozen_development_candidate",
        "model_changes_after_confirmation_allowed": False})
    return publish_once(selection, "confirmation-review.json", out, value)


def _reviewed_final_request(selection_path, review_path, bank_path, checkpoint_path, output_path, *, cases, draws, lengths, seed, offset):
    selection, review = read_selection(selection_path), read_receipt(review_path)
    canonical = read_receipt(Path(selection["state_directory"]) / "confirmation-review.json")
    confirmation = read_receipt(Path(selection["state_directory"]) / "confirmation-assessment.json")
    if (review != canonical or review["selection_sha256"] != selection["receipt_sha256"]
            or review["confirmation_sha256"] != confirmation["receipt_sha256"]
            or not confirmation["execution_pass"] or review["scientific_acceptance"] is not False):
        raise ValueError("final evaluation requires the unchanged explicit confirmation review")
    if review["mode"] == "confirm_for_workshop" and not confirmation["passed"]:
        raise ValueError("statistical screens do not support workshop confirmation")
    for item in confirmation["evaluations"].values():
        if (inference.sha256(item["path"]) != item["sha256"]
                or inference.sha256(item["case_summary_path"]) != item["case_summary_sha256"]):
            raise ValueError("confirmation evidence changed after review")
    name, record = _request(selection, bank_path, checkpoint_path, phase="final", cases=cases,
                            draws=draws, lengths=lengths, seed=seed, offset=offset)
    if Path(output_path).exists():
        raise FileExistsError("final output directory already exists")
    return selection, review, confirmation, name, record


def validate_final_request(selection_path, review_path, bank_path, checkpoint_path, output_path, *, cases, draws, lengths, seed, offset=0):
    """Scheduler preflight only: validate the review without consuming test access."""
    selection, _, _, name, _ = _reviewed_final_request(selection_path, review_path,
        bank_path, checkpoint_path, output_path, cases=cases, draws=draws, lengths=lengths, seed=seed, offset=offset)
    if (Path(selection["state_directory"]) / f"test-opening-{name}.json").exists():
        raise FileExistsError("reserved-test access was already claimed; no retry allocation")
    return name


def claim_final_test(selection_path, review_path, bank_path, checkpoint_path, output_path, *, cases, draws, lengths, seed, offset=0):
    """Call BEFORE reading reserved trajectories or loading/sampling a model.

    The exclusive marker is never removed, including after an evaluator crash.
    It prevents retries and a copied selection receipt cannot create a new gate.
    """
    selection, review, confirmation, name, record = _reviewed_final_request(selection_path, review_path,
        bank_path, checkpoint_path, output_path, cases=cases, draws=draws, lengths=lengths, seed=seed, offset=offset)
    value = sealed({"format": "path_workshop_test_opening_v1", "claimed_at": stamp(), "model": name,
        "selection_sha256": selection["receipt_sha256"], "review_sha256": review["receipt_sha256"],
        "confirmation_sha256": confirmation["receipt_sha256"], "review_mode": review["mode"],
        "model_sha256": record["model_sha256"], "task_sha256": record["task_sha256"],
        "bank_manifest_sha256": record["bank_manifest_sha256"], "output": str(Path(output_path).resolve()),
        "design": selection["design"], "state": "claimed_once_before_test_access",
        "retry_allowed": False, "scientific_acceptance": False,
        "scope": "descriptive final characterization of an unchanged frozen development candidate"})
    exclusive_json(Path(selection["state_directory"]) / f"test-opening-{name}.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for name in ("assess", "assess-final"):
        assess = commands.add_parser(name)
        assess.add_argument("--selection", type=Path, required=True)
        assess.add_argument("--evaluation", action="append", required=True, help="MODEL=EVALUATION_DIRECTORY")
        assess.add_argument("--out", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--selection", type=Path, required=True)
    review.add_argument("--confirmation", type=Path, required=True)
    review.add_argument("--mode", choices=("confirm_for_workshop", "evaluate_frozen_development_candidate"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze_selection(args.run_root, args.out, seed=args.seed)
    elif args.command in ("assess", "assess-final"):
        pairs = [item.split("=", 1) for item in args.evaluation]
        if any(len(pair) != 2 for pair in pairs) or len(dict(pairs)) != len(pairs):
            parser.error("evaluations must be unique MODEL=PATH entries")
        function = assess_final if args.command == "assess-final" else assess_confirmation
        value = function(args.selection, dict(pairs), args.out)
    else:
        value = review_confirmation(args.selection, args.confirmation, args.out,
                                   mode=args.mode, reviewer=args.reviewer, reason=args.reason)
    print(json.dumps({key: value[key] for key in ("format", "receipt_sha256", "scientific_acceptance")}))


if __name__ == "__main__":
    main()

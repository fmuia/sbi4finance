"""Paired development screens for a fixed short-prefix training candidate.

Rows 0:256 were used for checkpoint development; rows 256:512 are the
predeclared primary validation cohort. Neither cohort is the reserved test.
No baseline file or published estimator is changed by this comparison.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import inference  # noqa: E402


def prior_mean(task):
    return np.asarray([(lo + hi) / 2 if kind == "uniform" else (hi - lo) / np.log(hi / lo)
                       for (lo, hi), kind in zip(task["prior_bounds"], task["prior_types"], strict=True)])


def paired_risk(baseline, candidate, truth, center, *, seed=0, bootstraps=1000):
    baseline, candidate, truth = [np.asarray(x, dtype=float) for x in (baseline, candidate, truth)]
    if baseline.shape != candidate.shape or baseline.shape != truth.shape or baseline.ndim != 2:
        raise ValueError("paired posterior means and truths must have the same [case,target] shape")
    old, new, null = (baseline - truth) ** 2, (candidate - truth) ** 2, (center - truth) ** 2
    denominator = null.mean(axis=0)
    if np.any(denominator <= 0):
        raise ValueError("prior-mean baseline risk must be positive")
    indices = np.random.default_rng(seed).integers(0, len(truth), (bootstraps, len(truth)))
    changes = (new[indices].mean(axis=1) - old[indices].mean(axis=1)) / null[indices].mean(axis=1)
    old_ratio = old.mean(axis=0) / denominator
    margin = np.maximum(.01, .05 * old_ratio)
    low, high = np.quantile(changes, [.025, .975], axis=0)
    return {"baseline_mse_ratio": old_ratio.tolist(),
            "candidate_mse_ratio": (new.mean(axis=0) / denominator).tolist(),
            "paired_change": ((new.mean(axis=0) - old.mean(axis=0)) / denominator).tolist(),
            "paired_change_interval95": [low.tolist(), high.tolist()],
            "nondegradation_margin": margin.tolist(),
            "nondegradation_pass": (high <= margin).tolist(),
            "interval_scope": "pointwise paired trajectory bootstrap; no simultaneous accuracy certificate"}


def load_evaluation(directory):
    report = json.loads((directory / "evaluation.json").read_text())
    path = directory / report["case_summary_file"]
    if inference.sha256(path) != report["case_summary_sha256"]:
        raise ValueError("case-summary checksum differs from the evaluation report")
    if report.get("split") != "validation" or report.get("offset") != 0 or report.get("reserved_test_evaluated"):
        raise ValueError("comparison requires validation rows starting at zero, with tests closed")
    with np.load(path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    if len(arrays["truth"]) != 512:
        raise ValueError("comparison requires exactly the predeclared 512 validation rows")
    return report, arrays


def interval_screen(old_low, old_high, new_low, new_high, truth):
    old = ((truth >= old_low) & (truth <= old_high)).mean(axis=0)
    new = ((truth >= new_low) & (truth <= new_high)).mean(axis=0)
    change = np.abs(new - .9) - np.abs(old - .9)
    return {"baseline_coverage90": old.tolist(), "candidate_coverage90": new.tolist(),
            "absolute_coverage_error_change": change.tolist(), "maximum_allowed_increase": .02,
            "pass": (change <= .02 + 1e-12).tolist(),
            "scope": "descriptive pointwise coverage guard; no simultaneous calibration certificate"}


def exact_screen(directory, ids, rows, evaluation):
    report = json.loads((directory / "one-return.json").read_text())
    for key in ("task_sha256", "checkpoint_sha256"):
        if report.get(key) != evaluation[key]:
            raise ValueError(f"exact-check {key} differs from its evaluated checkpoint")
    path = directory / "one-return.npz"
    if inference.sha256(path) != report["arrays_sha256"]:
        raise ValueError("exact-check arrays have changed")
    if report["offset"] != 0 or report["cases"] != 512 or report["reserved_test_evaluated"]:
        raise ValueError("exact check must use the same 512 validation cases")
    with np.load(path, allow_pickle=False) as values:
        if not np.array_equal(values["trajectory_id"], ids):
            raise ValueError("exact-check case identities differ")
        error = values["neural_mean_errors_reference_sd"][rows]
        mass_error = np.abs(values["reference_mass_in_neural_interval"][rows] - .9)
    if not np.isfinite(error).all() or not np.isfinite(mass_error).all():
        raise ValueError("nonfinite exact-check arrays")
    rmse, mae = np.sqrt(np.mean(error ** 2, axis=0)), mass_error.mean(axis=0)
    refinement = report["quadrature_refinement"]
    return {"mean_rmse_reference_sd": rmse.tolist(), "mean_absolute_interval_mass_error": mae.tolist(),
            "interval_mass_error_p95": np.quantile(mass_error, .95, axis=0).tolist(),
            "mean_error_gate": .10, "interval_mass_error_gate": .04,
            "pass": bool(np.all(rmse <= .10) and np.all(mae <= .04)
                         and max(refinement.values()) <= 1e-4),
            "quadrature_refinement": refinement,
            "scope": "exact one-return marginal reference; no long-history or joint-copula certification"}


def compare(args):
    old_report, old = load_evaluation(args.baseline)
    new_report, new = load_evaluation(args.candidate)
    for field in ("task_sha256", "targets", "case_identities_sha256", "draws"):
        if old_report[field] != new_report[field]:
            raise ValueError(f"baseline/candidate differ in {field}")
    if not np.array_equal(old["truth"], new["truth"]) or not np.array_equal(old["trajectory_id"], new["trajectory_id"]):
        raise ValueError("paired cases or generating truths differ")
    task = inference.validate_task(json.loads(args.task.read_text()))
    if inference.fingerprint(task) != old_report["task_sha256"]:
        raise ValueError("task differs from evaluated task")
    landmarks = task["prefix_lengths"]
    result = {"format": "path_refinement_development_comparison_v1", "task_sha256": old_report["task_sha256"],
              "targets": old_report["targets"], "baseline_checkpoint_sha256": old_report["checkpoint_sha256"],
              "candidate_checkpoint_sha256": new_report["checkpoint_sha256"],
              "reserved_test_evaluated": False, "baseline_replaced": False, "cohorts": {},
              "scope": "fixed-candidate development screen; keep baseline on failed or inconclusive checks"}
    valid = all(x["nonfinite_count"] == 0 and not any(x["out_of_prior_fraction"])
                for x in new_report["lengths"].values())
    for number, (name, rows) in enumerate((("selection_reused", slice(0, 256)), ("new_validation", slice(256, 512)))):
        truth = old["truth"][rows]
        cohort = {"cases": len(truth), "landmarks": {}}
        for length in landmarks:
            cohort["landmarks"][str(length)] = paired_risk(old[f"posterior_mean_{length}"][rows],
                new[f"posterior_mean_{length}"][rows], truth, prior_mean(task),
                seed=args.seed + number * 10007 + length, bootstraps=args.bootstraps)
            cohort["landmarks"][str(length)]["interval_screen"] = interval_screen(
                old[f"lower90_{length}"][rows], old[f"upper90_{length}"][rows],
                new[f"lower90_{length}"][rows], new[f"upper90_{length}"][rows], truth)
        long_pass = all(all(x["nondegradation_pass"]) and all(x["interval_screen"]["pass"])
                        for x in cohort["landmarks"].values())
        if task["model"] == "rough-joint":
            if args.baseline_exact is None or args.candidate_exact is None:
                raise ValueError("joint refinement requires both exact one-return checks")
            cohort["baseline_exact"] = exact_screen(args.baseline_exact, old["trajectory_id"], rows, old_report)
            cohort["candidate_exact"] = exact_screen(args.candidate_exact, old["trajectory_id"], rows, new_report)
            short_pass = cohort["candidate_exact"]["pass"]
        else:
            lengths = [n for n in (1, 2, 4, 8, 17, 32) if str(n) in old_report["lengths"]]
            old_hits, new_hits = [], []
            for n in lengths:
                old_hits.append((truth >= old[f"lower90_{n}"][rows]) & (truth <= old[f"upper90_{n}"][rows]))
                new_hits.append((truth >= new[f"lower90_{n}"][rows]) & (truth <= new[f"upper90_{n}"][rows]))
            old_hits, new_hits = np.stack(old_hits, axis=1), np.stack(new_hits, axis=1)
            old_error = float(np.abs(old_hits.mean(axis=0) - .9).mean())
            new_error = float(np.abs(new_hits.mean(axis=0) - .9).mean())
            first = new_hits[:, 0].mean(axis=0)
            short_pass = bool(new_error < old_error and np.all((first >= .85) & (first <= .95)))
            cohort["short_history"] = {"lengths": lengths, "baseline_mean_absolute_coverage_error": old_error,
                "candidate_mean_absolute_coverage_error": new_error, "candidate_length1_coverage90": first.tolist(),
                "pass": short_pass, "scope": "simulation-based marginal screen; eight-substep Heston has no exact-prior one-return control"}
        cohort["development_screen_pass"] = bool(valid and long_pass and short_pass)
        result["cohorts"][name] = cohort
    result["finite_and_support_pass"] = valid
    result["candidate_status"] = ("passes_primary_development_screen" if result["cohorts"]["new_validation"]["development_screen_pass"]
                                  else "retain_original_baseline_needs_further_work")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"candidate_status": result["candidate_status"], "baseline_replaced": False}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "task", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--baseline-exact", type=Path)
    parser.add_argument("--candidate-exact", type=Path)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261030191)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()

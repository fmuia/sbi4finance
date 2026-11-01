"""Frozen selection, simultaneous marginal screens and irreversible test guards."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sbi4finance.path_inference import bank, inference

LAB = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("path_workshop_finalize_tests", LAB / "scripts/finalize_path_workshop.py")
closeout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closeout)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class WorkshopAcceptanceTests(unittest.TestCase):
    def test_heston_atomic_drift_does_not_require_exact_nominal_coverage(self):
        task = {"model": "heston", "parameter_names": ["kappa", "drift"],
                "prior_bounds": [[.25, 5], [-.1, .2]], "prior_types": ["log_uniform", "uniform"]}
        truth = inference.prior_samples(task | {"measure": "P", "n_observations": 1260,
            "prefix_lengths": [1260], "dt_years": 1 / 252}, 2976, 81)
        low, high = truth.copy(), truth.copy()
        low[-60:] = [.25, -.1]
        high[-60:] = [.25, -.1]
        arrays = {"truth": truth, "posterior_mean_1260": truth, "lower90_1260": low, "upper90_1260": high}
        result = closeout.simultaneous_screens(arrays, {"lengths": {"1260": {
            "nonfinite_count": 0, "out_of_prior_fraction": [0, 0]}}}, task,
            {"family_cells": 192, "coverage_family_alpha": .025, "risk_family_alpha": .025})
        row = result["lengths"]["1260"]
        self.assertEqual(row["nominal_inside_interval"], [False, True])
        self.assertEqual(row["coverage_criterion"][1], "one_sided_minimum_nominal_coverage_for_atomic_target")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        submission = {"submission_complete": True, "source_commit": "frozen-numerics",
                      "source_archive_sha256": "source-archive", "runtime_lock_sha256": "runtime-lock"}
        write(self.run / "submission-receipt.json", submission)
        write(self.run / "scaleup-summary.json", {"reserved_test_evaluated": False,
            "source_commit": "frozen-numerics",
            "submission_receipt_sha256": inference.sha256(self.run / "submission-receipt.json")})
        for name in closeout.MODELS:
            task = bank.teaching_task(name)
            manifest = {"format": bank.FORMAT, "task": task, "task_sha256": bank.task_fingerprint(task),
                "root_seed": 101, "split_counts": {"train": 1_000_000, "validation": 4000, "test": 4000},
                "splits": {split: [{"path": f"{split}/part-000000.npz", "start": 0, "count": count, "sha256": "unused-unopened-shard"}]
                           for split, count in (("train", 1_000_000), ("validation", 4000), ("test", 4000))}}
            write(self.run / "banks" / name / "manifest.json", manifest)
            directory = self.run / "fits" / name / "1m"
            write(directory / "task.json", task)
            (directory / "model.keras").write_bytes(name.encode())
            training = {"task_sha256": manifest["task_sha256"],
                "bank_manifest_sha256": inference.sha256(self.run / "banks" / name / "manifest.json"),
                "model_sha256": inference.sha256(directory / "model.keras"), "training_trajectories": 1_000_000,
                "epochs_completed": 5, "checkpoint_selection": "final", "same_seed_reload_exact": True,
                "status": "completed", "runtime": {"bayesflow": "2.1.0"}, "source_sha256": {"inference.py": "source-hash"}}
            write(directory / "training.json", training)
            for label in ("100k", "1m"):
                write(self.run / "evaluation" / name / label / "evaluation.json", {
                    "task_sha256": manifest["task_sha256"], "bank_manifest_sha256": training["bank_manifest_sha256"],
                    "checkpoint_sha256": training["model_sha256"] if label == "1m" else "other-model",
                    "split": "validation", "offset": 0, "cases": 1024, "reserved_test_evaluated": False,
                    "lengths": {"1260": {"coverage": {"0.9": {"coverage": [.87] * len(task["parameter_names"])}}}}})
            write(self.run / "comparison" / name / "comparison.json", {
                "status": "development_comparison_complete", "reserved_test_evaluated": False})
        self.selection_path = self.run.parent / "selected.json"
        self.selection = closeout.freeze_selection(self.run, self.selection_path)

    def evaluations(self, *, coverage=.9, nonfinite=0, phase="confirmation"):
        result = {}
        for name, candidate in self.selection["models"].items():
            task = candidate["task"]
            cohort = self.selection["design"][phase]
            n = cohort["cases"]
            truth = inference.prior_samples(task, n, 83)
            bounds = np.asarray(task["prior_bounds"])
            ids = np.asarray([f"{candidate['task_sha256'][:16]}:101:{cohort['split']}:{i}"
                              for i in range(cohort["offset"], cohort["offset"] + n)])
            arrays = {"truth": truth, "trajectory_id": ids}
            for length in closeout.LENGTHS:
                low, high = truth.copy(), truth.copy()
                # A fixed fraction of complete trajectories miss across all
                # parameters/lengths; the simultaneous family permits dependence.
                low[round(coverage * n):] = bounds[:, 0]
                high[round(coverage * n):] = bounds[:, 0]
                arrays.update({f"posterior_mean_{length}": truth, f"lower90_{length}": low, f"upper90_{length}": high})
            directory = self.run.parent / phase / name
            directory.mkdir(parents=True)
            np.savez_compressed(directory / "cases.npz", **arrays)
            report = {"task_sha256": candidate["task_sha256"],
                "checkpoint_sha256": candidate["model_sha256"], "bank_manifest_sha256": candidate["bank_manifest_sha256"],
                "targets": task["parameter_names"], "split": cohort["split"], "offset": cohort["offset"],
                "cases": n, "draws": 512, "seed": closeout.DEFAULT_SEED, "reserved_test_evaluated": phase == "final",
                "case_identities_sha256": inference.fingerprint(ids.tolist()), "case_summary_file": "cases.npz",
                "case_summary_sha256": inference.sha256(directory / "cases.npz"),
                "lengths": {str(length): {"nonfinite_count": nonfinite, "out_of_prior_fraction": [0] * len(bounds)} for length in closeout.LENGTHS}}
            if phase == "final":
                report["test_opening_receipt_sha256"] = closeout.read_receipt(
                    Path(self.selection["state_directory"]) / f"test-opening-{name}.json")["receipt_sha256"]
            write(directory / "evaluation.json", report)
            result[name] = directory
        return result

    def assess(self, **kwargs):
        path = self.run.parent / "assessment.json"
        value = closeout.assess_confirmation(self.selection_path, self.evaluations(**kwargs), path)
        return path, value

    def review(self, assessment, mode):
        path = self.run.parent / "review.json"
        value = closeout.review_confirmation(self.selection_path, assessment, path, mode=mode,
            reviewer="independent reviewer", reason="Characterize the identical frozen recipe with the recorded limitations.")
        return path, value

    def claim(self, review, **kwargs):
        candidate = self.selection["models"]["heston"]
        values = {"cases": 4000, "draws": 512, "lengths": list(closeout.LENGTHS), "seed": closeout.DEFAULT_SEED}
        values.update(kwargs)
        return closeout.claim_final_test(self.selection_path, review, candidate["bank"], candidate["checkpoint"],
            self.run.parent / "test-evaluation", **values)

    def test_selection_is_immutable_and_retains_the_known_development_limitations(self):
        self.assertEqual(self.selection["gate_policy"]["family_cells"], 192)
        self.assertFalse(self.selection["scientific_acceptance"])
        self.assertEqual(self.selection["models"]["heston"]["development"]["1m"]["coverage90_ranges"], [.87, .87])
        with self.assertRaises(FileExistsError):
            closeout.freeze_selection(self.run, self.run.parent / "replacement.json")
        altered = dict(self.selection, selection="changed-after-confirmation")
        altered.pop("receipt_sha256")
        write(self.run.parent / "altered.json", closeout.sealed(altered))
        with self.assertRaisesRegex(ValueError, "canonical"):
            closeout.read_selection(self.run.parent / "altered.json")

    def test_valid_confirmation_and_review_allow_only_one_descriptive_test_opening(self):
        assessment, result = self.assess()
        self.assertTrue(result["passed"])
        self.assertFalse(result["scientific_acceptance"])
        review, _ = self.review(assessment, "confirm_for_workshop")
        marker = self.claim(review)
        self.assertFalse(marker["retry_allowed"])
        self.assertFalse(marker["scientific_acceptance"])
        with self.assertRaises(FileExistsError):
            self.claim(review)
        self.assertFalse((self.run.parent / "test-evaluation").exists())

    def test_failed_statistical_screen_needs_explicit_development_characterization_review(self):
        assessment, result = self.assess(coverage=.8)
        self.assertFalse(result["passed"])
        self.assertTrue(result["execution_pass"])
        self.assertEqual(result["status"], "failed_calibration_screen")
        with self.assertRaisesRegex(ValueError, "all frozen statistical"):
            self.review(assessment, "confirm_for_workshop")
        review, decision = self.review(assessment, "evaluate_frozen_development_candidate")
        self.assertFalse(decision["statistical_screens_passed"])
        self.assertEqual(self.claim(review)["review_mode"], "evaluate_frozen_development_candidate")

    def test_execution_failure_cannot_be_overridden_by_review(self):
        assessment, result = self.assess(nonfinite=1)
        self.assertFalse(result["execution_pass"])
        with self.assertRaisesRegex(ValueError, "execution/identity"):
            self.review(assessment, "evaluate_frozen_development_candidate")

    def test_changed_model_and_design_refuse_before_claiming_test_access(self):
        assessment, _ = self.assess()
        review, _ = self.review(assessment, "confirm_for_workshop")
        with self.assertRaisesRegex(ValueError, "frozen design"):
            self.claim(review, seed=42)
        candidate = self.selection["models"]["heston"]
        (Path(candidate["checkpoint"]) / "model.keras").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "frozen artifact"):
            self.claim(review)
        self.assertFalse((Path(self.selection["state_directory"]) / "test-opening-heston.json").exists())

    def test_confirmation_hash_change_and_repeated_review_are_rejected(self):
        assessment, _ = self.assess()
        review, _ = self.review(assessment, "confirm_for_workshop")
        with self.assertRaises(FileExistsError):
            self.review(assessment, "confirm_for_workshop")
        report = self.run.parent / "confirmation/heston/evaluation.json"
        report.write_text(report.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.claim(review)

    def test_confirmation_request_enforces_the_unused_rows_and_frozen_identities(self):
        candidate = self.selection["models"]["rough-joint"]
        parameters = {"cases": 2976, "draws": 512, "lengths": list(closeout.LENGTHS),
                      "seed": closeout.DEFAULT_SEED, "offset": 1024}
        self.assertEqual(closeout.validate_confirmation_request(self.selection_path, candidate["bank"],
            candidate["checkpoint"], **parameters), "rough-joint")
        with self.assertRaisesRegex(ValueError, "frozen design"):
            closeout.validate_confirmation_request(self.selection_path, candidate["bank"], candidate["checkpoint"],
                **dict(parameters, offset=0))

    def test_final_reports_bind_to_once_only_openings_and_keep_development_status(self):
        assessment, _ = self.assess()
        review, _ = self.review(assessment, "confirm_for_workshop")
        for name, candidate in self.selection["models"].items():
            closeout.claim_final_test(self.selection_path, review, candidate["bank"], candidate["checkpoint"],
                self.run.parent / "final" / name, cases=4000, draws=512,
                lengths=list(closeout.LENGTHS), seed=closeout.DEFAULT_SEED)
        evaluations = self.evaluations(phase="final")
        out = self.run.parent / "final-assessment.json"
        result = closeout.assess_final(self.selection_path, evaluations, out)
        self.assertTrue(result["passed"])
        self.assertTrue(result["reserved_test_evaluated"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertEqual(result["format"], "path_workshop_final_characterization_v1")
        with self.assertRaises(FileExistsError):
            closeout.assess_final(self.selection_path, evaluations, out)


if __name__ == "__main__":
    unittest.main()

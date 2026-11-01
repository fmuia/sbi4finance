"""Separate development cohorts preserve native identity and reserved-test guards."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sbi4finance.path_inference import bank, inference


LAB = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("path_fresh_cohort_evaluator_tests", LAB / "scripts/evaluate_path_posterior.py")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture_bank(root, task, seed):
    """Small native-shaped fixtures: no production simulations or reserved reads."""
    root.mkdir()
    digest = bank.task_fingerprint(task)
    manifest = {"format": bank.FORMAT, "task": task, "task_sha256": digest, "root_seed": seed,
                "split_counts": {"train": 8, "validation": 8, "test": 2}, "splits": {}}
    rng = np.random.default_rng(seed)
    for split, count in manifest["split_counts"].items():
        (root / split).mkdir()
        manifest["splits"][split] = []
        for index, start in enumerate(range(0, count, 2)):
            size = min(2, count - start)
            path = root / split / f"part-{index:06d}.npz"
            np.savez(path, observations=rng.normal(0, .01, (size, task["n_observations"], 1)),
                     parameters=inference.prior_samples(task, size, seed + start),
                     trajectory_id=np.asarray([f"{digest[:16]}:{seed}:{split}:{i}" for i in range(start, start + size)]))
            manifest["splits"][split].append({"path": str(path.relative_to(root)), "start": start,
                "count": size, "sha256": inference.sha256(path), "bytes": path.stat().st_size})
    write_json(root / "manifest.json", manifest)
    return manifest


class FreshCohortTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = dict(bank.default_task("rough-known"), n_observations=9,
                         prefix_lengths=[1, 3, 9], history_length_range=[1, 9])
        self.training_bank, self.cohort_bank = self.root / "training-bank", self.root / "cohort-bank"
        fixture_bank(self.training_bank, self.task, 83)
        fixture_bank(self.cohort_bank, self.task, 91)
        self.checkpoint = self.checkpoint_fixture("checkpoint")

    def checkpoint_fixture(self, name):
        directory = self.root / name
        directory.mkdir()
        write_json(directory / "task.json", self.task)
        (directory / "model.keras").write_bytes(("native-fixture-" + name).encode())
        write_json(directory / "training.json", {"format": inference.FORMAT,
            "task_sha256": inference.fingerprint(self.task),
            "model_sha256": inference.sha256(directory / "model.keras"),
            "bank_manifest_sha256": inference.sha256(self.training_bank / "manifest.json"),
            "normalization": "fixed_prior_units_no_fitted_standardizer_v1",
            "prefix_policy": "uniform", "history_training": inference.PREFIX_POLICIES["uniform"],
            "network": inference.DEFAULT_NETWORK})
        return directory

    def arguments(self, **changes):
        values = dict(bank=self.training_bank, cohort_bank=self.cohort_bank, checkpoint=self.checkpoint,
                      out=self.root / "evaluation", cases=3, offset=3, draws=8, batch_size=2,
                      lengths=[1, 9], seed=83, split="validation", phase="development",
                      cohort_purpose="paired development diagnosis", selection=None, review=None)
        return SimpleNamespace(**(values | changes))

    def run_evaluation(self, args, *, load_side_effect=None):
        def load(path):
            if load_side_effect:
                load_side_effect()
            return self.task, object(), json.loads((path / "training.json").read_text())

        def sample(workflow, task, observations, draws, *, seed):
            return inference.prior_samples(task, len(observations) * draws, seed).reshape(
                len(observations), draws, len(task["parameter_names"]))

        with mock.patch.object(evaluator.inference, "load_checkpoint", side_effect=load), \
                mock.patch.object(evaluator.inference, "sample", side_effect=sample):
            return evaluator.evaluate(args)

    def test_distinct_cohort_binds_both_manifests_and_reads_only_intersecting_rows(self):
        original_training = (self.checkpoint / "training.json").read_bytes()
        with mock.patch.object(evaluator.bank, "load_split", side_effect=AssertionError("whole split read")), \
                mock.patch.object(evaluator.bank, "_load_part", wraps=bank._load_part) as parts:
            report = self.run_evaluation(self.arguments())
        self.assertEqual([(call.args[2], call.args[3]["start"]) for call in parts.call_args_list],
                         [("validation", 2), ("validation", 4)])
        self.assertEqual(report["cases"], 3)
        self.assertEqual(report["bank_manifest_sha256"], inference.sha256(self.training_bank / "manifest.json"))
        self.assertEqual(report["training_bank_manifest_sha256"], report["bank_manifest_sha256"])
        self.assertEqual(report["evaluation_bank_manifest_sha256"], inference.sha256(self.cohort_bank / "manifest.json"))
        self.assertNotEqual(report["evaluation_bank_manifest_sha256"], report["bank_manifest_sha256"])
        self.assertFalse(report["reserved_test_evaluated"])
        with np.load(self.root / "evaluation/case-summary.npz") as saved:
            self.assertEqual(saved["truth"].shape, (3, 3))
            self.assertEqual(saved["trajectory_id"].tolist(),
                [f"{inference.fingerprint(self.task)[:16]}:91:validation:{i}" for i in (3, 4, 5)])
        opening = evaluator.workshop_guard().read_receipt(self.root / "evaluation/cohort-opening.json")
        self.assertEqual(opening["checkpoint_sha256"], inference.sha256(self.checkpoint / "model.keras"))
        self.assertEqual(opening["case_identities_sha256"], report["case_identities_sha256"])
        self.assertEqual(opening["receipt_sha256"], report["cohort_opening_receipt_sha256"])
        self.assertEqual(original_training, (self.checkpoint / "training.json").read_bytes())

    def test_development_cohort_is_reusable_for_paired_models_but_outputs_are_exclusive(self):
        first = self.run_evaluation(self.arguments())
        other = self.checkpoint_fixture("candidate")
        second = self.run_evaluation(self.arguments(checkpoint=other, out=self.root / "candidate-evaluation"))
        self.assertEqual(first["case_identities_sha256"], second["case_identities_sha256"])
        self.assertNotEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
        key = "development_exposure_receipt_sha256"
        self.assertEqual(first["cohort_opening_receipt"][key], second["cohort_opening_receipt"][key])
        marker = evaluator.workshop_guard().read_receipt(self.cohort_bank / "evaluation-guard/development-exposure.json")
        self.assertEqual(marker["usage"], "development_only_not_fresh_acceptance")
        self.assertEqual(marker["exposed_split"], "validation")
        with mock.patch.object(evaluator.inference, "load_checkpoint") as load:
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(self.arguments())
            load.assert_not_called()

    def test_default_route_retains_the_exact_training_manifest_guard(self):
        report = self.run_evaluation(self.arguments(cohort_bank=None))
        self.assertEqual(report["evaluation_bank_manifest_sha256"], report["bank_manifest_sha256"])
        self.assertNotIn("cohort_opening_receipt", report)
        self.assertFalse((self.cohort_bank / "evaluation-guard").exists())
        with mock.patch.object(evaluator.bank, "load_split") as rows:
            with self.assertRaisesRegex(ValueError, "bank manifest"):
                self.run_evaluation(self.arguments(cohort_bank=None, bank=self.cohort_bank, out=self.root / "wrong-default"))
            rows.assert_not_called()

    def test_wrong_task_and_same_root_seed_fail_before_backend_or_payload_reads(self):
        for name, task, seed, message in [
            ("wrong-task", self.task | {"fixed": self.task["fixed"] | {"variance_level": .05}}, 92, "task"),
            ("same-seed", self.task, 83, "root seed")]:
            with self.subTest(name=name):
                other = self.root / name
                fixture_bank(other, task, seed)
                with mock.patch.object(evaluator.inference, "load_checkpoint") as load, \
                        mock.patch.object(evaluator.bank, "_load_part") as rows:
                    with self.assertRaisesRegex(ValueError, message):
                        evaluator.evaluate(self.arguments(cohort_bank=other, out=self.root / (name + "-out")))
                    load.assert_not_called()
                    rows.assert_not_called()
                self.assertFalse((other / "evaluation-guard").exists())

    def test_tampered_training_manifest_or_missing_native_identity_fail_closed(self):
        original = (self.training_bank / "manifest.json").read_text()
        manifest = json.loads(original)
        write_json(self.training_bank / "manifest.json", manifest | {"changed": True})
        with mock.patch.object(evaluator.inference, "load_checkpoint") as load:
            with self.assertRaisesRegex(ValueError, "bank manifest"):
                evaluator.evaluate(self.arguments())
            load.assert_not_called()
        (self.training_bank / "manifest.json").write_text(original)
        training_path = self.checkpoint / "training.json"
        training = json.loads(training_path.read_text())
        for key in ("format", "task_sha256", "model_sha256", "bank_manifest_sha256"):
            with self.subTest(key=key), mock.patch.object(evaluator.inference, "load_checkpoint") as load:
                write_json(training_path, {name: value for name, value in training.items() if name != key})
                with self.assertRaisesRegex(ValueError, "native"):
                    evaluator.evaluate(self.arguments())
                load.assert_not_called()
        self.assertFalse((self.cohort_bank / "evaluation-guard").exists())

    def test_overlapping_payload_ids_are_rejected_even_with_new_seed_and_updated_hash(self):
        manifest_path = self.cohort_bank / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = manifest["splits"]["validation"][1]
        path = self.cohort_bank / record["path"]
        with np.load(path) as saved:
            arrays = dict(saved)
        arrays["trajectory_id"] = np.asarray([value.replace(":91:", ":83:") for value in arrays["trajectory_id"]])
        np.savez(path, **arrays)
        record.update(sha256=inference.sha256(path), bytes=path.stat().st_size)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "trajectory IDs"):
            self.run_evaluation(self.arguments())

    def test_fresh_confirmation_and_reserved_test_routes_are_not_implemented_or_bypassed(self):
        for phase, split, selection, review in [("confirmation", "validation", None, None),
                ("final-test", "test", None, None), ("development", "test", None, None),
                ("development", "validation", self.root / "old-selection.json", None),
                ("development", "validation", None, self.root / "old-review.json")]:
            with self.subTest(phase=phase, split=split), \
                    mock.patch.object(evaluator.inference, "load_checkpoint") as load, \
                    mock.patch.object(evaluator.bank, "_load_part") as rows:
                with self.assertRaisesRegex(ValueError, "development/validation only"):
                    evaluator.evaluate(self.arguments(phase=phase, split=split, selection=selection, review=review))
                load.assert_not_called()
                rows.assert_not_called()
        self.assertFalse((self.cohort_bank / "evaluation-guard").exists())

    def test_default_reserved_test_still_requires_the_original_selection_guard(self):
        with mock.patch.object(evaluator.bank, "load_split") as rows:
            with self.assertRaisesRegex(ValueError, "frozen selection"):
                self.run_evaluation(self.arguments(cohort_bank=None, phase="final-test", split="test"))
            rows.assert_not_called()

    def test_incomplete_cohort_and_mutation_during_native_reload_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluator.evaluate(self.arguments(cases=20))
        self.assertFalse((self.cohort_bank / "evaluation-guard").exists())

        def changed():
            manifest = json.loads((self.cohort_bank / "manifest.json").read_text())
            write_json(self.cohort_bank / "manifest.json", manifest | {"changed": True})

        with mock.patch.object(evaluator.bank, "_load_part") as rows:
            with self.assertRaisesRegex(ValueError, "immutable"):
                self.run_evaluation(self.arguments(), load_side_effect=changed)
            rows.assert_not_called()


if __name__ == "__main__":
    unittest.main()

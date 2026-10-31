"""Replay curation preserves source draws and refuses altered payloads/cohorts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sbi4finance.path_inference import bank, inference, teaching


class PathTeachingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fit, self.evaluation = self.root / "fit", self.root / "evaluation"
        self.fit.mkdir()
        self.evaluation.mkdir()
        self.task = bank.teaching_task("rough-joint", observations=21)
        self.truth = inference.prior_samples(self.task, 5, 93)
        self.draws = inference.prior_samples(self.task, 5 * 16, 94).reshape(5, 16, 7)
        self.ids = np.asarray([f"validation:{i}" for i in range(5)])
        (self.fit / "task.json").write_text(json.dumps(self.task))
        (self.fit / "model.keras").write_bytes(b"native model bytes retained unchanged")
        self.training = {"task_sha256": inference.fingerprint(self.task),
                         "model_sha256": inference.sha256(self.fit / "model.keras"),
                         "bank_manifest_sha256": "source-bank"}
        (self.fit / "training.json").write_text(json.dumps(self.training))
        np.savez_compressed(self.evaluation / "teaching-cases.npz", truth=self.truth,
                            observations=np.zeros((5, 21, 1)), trajectory_id=self.ids,
                            draws_1=self.draws, draws_21=self.draws)
        self.report = {"task_sha256": self.training["task_sha256"],
                       "checkpoint_sha256": self.training["model_sha256"],
                       "bank_manifest_sha256": "source-bank", "split": "validation", "offset": 0,
                       "reserved_test_evaluated": False, "draws": 16,
                       "teaching_cases_sha256": inference.sha256(self.evaluation / "teaching-cases.npz")}
        self.write_report()

    def write_report(self):
        (self.evaluation / "evaluation.json").write_text(json.dumps(self.report))

    def test_first_declared_cases_and_joint_draw_rows_are_preserved_exactly(self):
        out = self.root / "curated"
        teaching.prepare(self.fit, self.evaluation, out, cases=4)
        task, bundle, arrays = teaching.load_cache(out)
        self.assertEqual(task, self.task)
        self.assertFalse(bundle["scientific_acceptance"])
        np.testing.assert_array_equal(arrays["truth"], self.truth[:4])
        np.testing.assert_array_equal(arrays["draws_21"], self.draws[:4])
        self.assertEqual(bundle["case_ids"], self.ids[:4].tolist())
        with self.assertRaises(FileExistsError):
            teaching.prepare(self.fit, self.evaluation, out)

    def test_source_hash_mismatch_refuses_before_publication(self):
        (self.fit / "model.keras").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            teaching.prepare(self.fit, self.evaluation, self.root / "curated")
        self.assertFalse((self.root / "curated").exists())

    def test_reserved_or_different_source_cohort_cannot_become_validation_replay(self):
        self.report.update(split="test", reserved_test_evaluated=True)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "source cohort"):
            teaching.prepare(self.fit, self.evaluation, self.root / "curated")
        self.assertFalse((self.root / "curated").exists())

    def test_modified_payload_fails_its_bundle_hash(self):
        out = self.root / "curated"
        teaching.prepare(self.fit, self.evaluation, out)
        with (out / "posterior.npz").open("ab") as stream:
            stream.write(b"unexpected mutation")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            teaching.load_cache(out)


if __name__ == "__main__":
    unittest.main()

"""Real freeze guard around the evaluator, with only neural execution mocked."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import test_path_workshop_acceptance as fixtures

LAB = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("path_evaluation_guard_tests", LAB / "scripts/evaluate_path_posterior.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class EvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.campaign = fixtures.WorkshopAcceptanceTests("test_selection_is_immutable_and_retains_the_known_development_limitations")
        self.campaign.setUp()
        self.addCleanup(self.campaign.doCleanups)
        assessment, _ = self.campaign.assess(coverage=.8)
        self.review, _ = self.campaign.review(assessment, "evaluate_frozen_development_candidate")
        candidate = self.campaign.selection["models"]["heston"]
        self.task = candidate["task"]
        self.training = json.loads((Path(candidate["checkpoint"]) / "training.json").read_text())
        self.args = SimpleNamespace(bank=Path(candidate["bank"]), checkpoint=Path(candidate["checkpoint"]),
            out=self.campaign.run.parent / "test-result", phase="final-test", split="test",
            selection=self.campaign.selection_path, review=self.review, offset=0, cases=4000,
            draws=512, batch_size=4000, lengths=list(fixtures.closeout.LENGTHS), seed=fixtures.closeout.DEFAULT_SEED)
        self.marker = Path(self.campaign.selection["state_directory"]) / "test-opening-heston.json"

    def test_claim_precedes_test_read_and_survives_read_failure_without_retry(self):
        events = []

        def read_test(directory, split):
            self.assertEqual(split, "test")
            self.assertTrue(self.marker.is_file())
            receipt = fixtures.closeout.read_receipt(self.marker)
            self.assertEqual(receipt["state"], "claimed_once_before_test_access")
            self.assertFalse(receipt["retry_allowed"])
            events.append("read_after_durable_claim")
            raise RuntimeError("simulated reserved-shard read failure")

        with patch.object(evaluator.inference, "load_checkpoint", return_value=(self.task, object(), self.training)), \
                patch.object(evaluator.bank, "load_split", side_effect=read_test) as read:
            with self.assertRaisesRegex(RuntimeError, "reserved-shard read failure"):
                evaluator.evaluate(self.args)
            self.assertEqual(events, ["read_after_durable_claim"])
            self.assertEqual(read.call_count, 1)
            self.assertTrue(self.marker.exists())
            self.assertFalse(self.args.out.exists())
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(self.args)
            self.assertEqual(read.call_count, 1)

    def test_invalid_review_refuses_before_reserved_data_is_read(self):
        self.args.review = self.campaign.run.parent / "missing-review.json"
        with patch.object(evaluator.inference, "load_checkpoint", return_value=(self.task, object(), self.training)), \
                patch.object(evaluator.bank, "load_split") as read:
            with self.assertRaises(FileNotFoundError):
                evaluator.evaluate(self.args)
        read.assert_not_called()
        self.assertFalse(self.marker.exists())

    def test_final_report_identifies_test_split_and_the_consumed_opening_receipt(self):
        parameters = evaluator.inference.prior_samples(self.task, 4000, 12)
        candidate = self.campaign.selection["models"]["heston"]
        identities = np.asarray([f"{candidate['task_sha256'][:16]}:101:test:{i}" for i in range(4000)])
        split = {"observations": np.zeros((4000, 1260, 1)), "parameters": parameters,
                 "trajectory_id": identities, "task": self.task}

        def read_test(directory, split_name):
            self.assertEqual(split_name, "test")
            self.assertTrue(self.marker.exists())
            return split

        def sample(workflow, task, observations, draws, *, seed):
            # Broadcast views keep this bookkeeping test independent of BF/JAX
            # and avoid storing a large synthetic posterior panel.
            return np.broadcast_to(parameters[:, None, :], (4000, draws, parameters.shape[1]))

        with patch.object(evaluator.inference, "load_checkpoint", return_value=(self.task, object(), self.training)), \
                patch.object(evaluator.bank, "load_split", side_effect=read_test), \
                patch.object(evaluator.inference, "sample", side_effect=sample), \
                patch.object(evaluator.inference, "diagnostics", return_value={}), \
                patch.object(evaluator.np, "savez_compressed", side_effect=lambda path, **arrays: Path(path).write_bytes(b"mocked-posterior-file")), \
                patch.object(evaluator.np, "quantile", side_effect=lambda values, levels, axis: np.stack([parameters, parameters])), \
                contextlib.redirect_stdout(io.StringIO()):
            result = evaluator.evaluate(self.args)
        self.assertEqual(result["split"], "test")
        self.assertEqual(result["phase"], "final-test")
        self.assertTrue(result["reserved_test_evaluated"])
        self.assertFalse(result["scientific_acceptance"])
        opening = fixtures.closeout.read_receipt(self.marker)
        self.assertEqual(result["test_opening_receipt_sha256"], opening["receipt_sha256"])
        saved = json.loads((self.args.out / "evaluation.json").read_text())
        self.assertEqual(saved["split"], "test")


if __name__ == "__main__":
    unittest.main()

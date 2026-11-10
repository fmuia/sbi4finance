"""Corrected-producer evaluations retain the predeclared posterior diagnostics."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import scripts

from scripts import evaluate_heston_spread_scale_precision as evaluation
from scripts import train_heston_spread_scale as scale
from test_heston_spread_training import fixture


class PrecisionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        design, _, _, _ = fixture(self.root)
        self.task, _ = scale.make_task(design)
        self.target = self.task["target"]
        self.study = {"format": "heston_spread_scale_precision_recovery_study_v1",
            "joint_task": self.target, "evaluation": evaluation.evaluation_contract(self.target)}
        for key in ("original_study", "reference_recovery", "baseline_evaluation", "precision_recovery"):
            path = self.root/(key+".json")
            evaluation.fixed.write_json(path, {})
            self.study[key] = {"path": str(path), "sha256": evaluation.fixed.digest(path)}

    def test_corrected_identity_requires_audit_and_unchanged_evaluation_contract(self):
        evaluation.validate_study(self.study, self.target)
        for key, value in (("family_checks", 96), ("conditional_draws", 1024),
                ("population_rows", 128), ("checkpoint_bank_passes", [25])):
            changed = copy.deepcopy(self.study)
            changed["evaluation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "declaration differs"):
                evaluation.validate_study(changed, self.target)
        old = {**self.study, "format": "heston_spread_scale_capacity_study_v1"}
        with self.assertRaisesRegex(ValueError, "precision recovery evaluation declaration"):
            evaluation.validate_study(old, self.target)
        changed = copy.deepcopy(self.study)
        changed["precision_recovery"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "audit binding"):
            evaluation.validate_study(changed, self.target)

    def test_sources_bind_the_new_consumer_and_shared_frozen_mathematical_helpers(self):
        producer = self.root/"precision-producer.py"
        producer.write_text("# no execution\n")
        sources = evaluation.source_bindings(types.SimpleNamespace(__file__=str(producer)))
        self.assertEqual(sources["evaluator"]["sha256"], evaluation.fixed.digest(evaluation.__file__))
        self.assertEqual(sources["continuation_trainer"]["sha256"], evaluation.fixed.digest(producer))
        self.assertEqual(sources["shared_capacity_evaluator"]["sha256"], evaluation.fixed.digest(evaluation.shared.__file__))
        self.assertEqual(sources["population_exporter"]["sha256"], evaluation.fixed.digest(evaluation.previous.__file__))
        self.assertEqual(sources["comparison_metrics"]["sha256"], evaluation.fixed.digest(evaluation.baseline.heston_budget_diagnostics.__file__))

    def test_new_loader_is_selected_and_changed_training_protocol_rejected_before_sampling(self):
        study_path = self.root/"precision-study.json"
        evaluation.fixed.write_json(study_path, self.study)
        protocol = self.root/"training-protocol.json"
        evaluation.fixed.write_json(protocol, {})
        metadata = {"task": self.task, "batch_size": 128, "completed_bank_passes": 25,
            "train_rows": 100000, "architecture": {"effective_dropout": 0.},
            "protocol_path": str(protocol), "protocol_sha256": "different"}
        producer = types.ModuleType("scripts.continue_heston_spread_scale_precision")
        producer.load_checkpoint = lambda checkpoint: (object(), metadata)
        with patch.dict("sys.modules", {"scripts.continue_heston_spread_scale_precision": producer}), \
                patch.object(scripts, "continue_heston_spread_scale_precision", producer, create=True), \
                patch.object(evaluation.previous.trainer, "load_checkpoint", side_effect=AssertionError("old loader called")), \
                patch.object(evaluation.previous.trainer, "sample", side_effect=AssertionError("sampling before validation")):
            with self.assertRaisesRegex(ValueError, "training protocol differs"):
                evaluation.evaluate(self.root/"checkpoint", study_path, self.root/"out", self.root/"payload")
        self.assertFalse((self.root/"out").exists())
        self.assertFalse((self.root/"payload").exists())


if __name__ == "__main__":
    unittest.main()

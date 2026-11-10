"""Final whitening diagnostics retain raw data, sidecars and joint targets."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_heston_spread_whitening as evaluation
from test_heston_spread_training import fixture


class WhiteningEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        design, _, _, _ = fixture(self.root)
        self.task, _ = evaluation.trainer.scale.make_task(design)
        self.target = self.task["target"]

    def study(self):
        value = {"format": "heston_spread_whitening_study_v1", "joint_task": self.target,
            "evaluation": evaluation.evaluation_contract(self.target), "arms": [{"mode": "diagonal"}, {"mode": "zca"}], "sources": {}}
        for key in ("original_study", "reference_recovery", "baseline_evaluation", "matched_mcmc_report"):
            path = self.root/(key+".json")
            evaluation.fixed.write_json(path, {})
            value[key] = evaluation.binding(path)
        return value

    def test_declaration_keeps_exact_old_population_and_descriptive_scope(self):
        study = self.study()
        evaluation.validate_study(study, self.target)
        self.assertNotIn("family_checks", study["evaluation"])
        for key, value in (("population_rows", 100), ("seed", 7), ("final_updates", 10000),
                ("preprocessing", "refit using validation"), ("modes", ["zca"])):
            changed = copy.deepcopy(study)
            changed["evaluation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "declaration differs"):
                evaluation.validate_study(changed, self.target)

    def test_endpoint_rejects_different_budget_input_standardization_or_sidecar(self):
        checkpoint = self.root/"checkpoint"; checkpoint.mkdir()
        evaluation.fixed.write_json(checkpoint/"preprocessing.json", {"fixture": "loader validates actual fitted state"})
        path = self.root/"study.json"; evaluation.fixed.write_json(path, {})
        contract = {"train_rows": 10000, "batch_size": 256, "bank_passes": 200,
            "planned_presentations": 2000000, "planned_updates": 8000,
            "standardize": ["inference_variables"],
            "architecture": {"family": "spline", "depth": 6, "widths": [128, 128], "dropout": 0.}}
        study = {"format": "heston_spread_whitening_study_v1", "arms": [{"mode": "zca", "contract": contract}]}
        metadata = {"mode": "zca", "format": "heston_spread_whitening_checkpoint_v1",
            "training_protocol": contract, "actual_optimizer_updates": 8000,
            "protocol_path": str(path), "protocol_sha256": evaluation.fixed.digest(path),
            "paired_initialization": {"all_trainable_hashes_exact": True}, "reserved_arrays_opened": False,
            "preprocessing_sha256": evaluation.fixed.digest(checkpoint/"preprocessing.json")}
        evaluation.validate_checkpoint(metadata, study, path, checkpoint)
        for key, value in (("actual_optimizer_updates", 5000), ("preprocessing_sha256", "wrong"),
                ("paired_initialization", {"all_trainable_hashes_exact": False})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "sidecar or training-data boundary"):
                evaluation.validate_checkpoint({**metadata, key: value}, study, path, checkpoint)

    def test_preprocessing_diagnostics_inspect_frozen_matrix_without_refitting(self):
        rng = np.random.default_rng(8101)
        values = rng.standard_normal((128, 1))@np.linspace(1., 3., 54)[None, :]+.2*rng.standard_normal((128, 54))
        binding = {"splits": {"train": {"rows": 128, "conditions": evaluation.fixed.array_summary(values),
            "targets": evaluation.fixed.array_summary(np.ones((128, 6))), "row_id": {"sha256": "fixture-rows"}}},
            "bank_record_sha256": "fixture-bank", "bank_payload_sha256": "fixture-payload"}
        sidecar = evaluation.trainer.fit_preprocessing(self.task, {"observations": values}, binding, "zca")
        with patch.object(evaluation.trainer.whitening, "fit_whitening", side_effect=AssertionError("evaluation must never fit")):
            summary = evaluation.preprocessing_summary(sidecar)
        self.assertEqual(summary["fit_rows"], 128)
        self.assertEqual(summary["fit_split"], "train")
        self.assertFalse(summary["validation_used_for_fit"])
        self.assertEqual(summary["feature_dimension"], 54)
        self.assertGreater(summary["raw_covariance_condition_number"], 1000)
        self.assertAlmostEqual(summary["transformed_covariance_condition_number"], 1., places=7)

    def test_descriptive_joint_comparison_has_no_acceptance_family_or_fixed_parameter(self):
        bounds = np.asarray(self.target["prior_bounds"])
        chains = np.random.default_rng(301).uniform(bounds[:, 0], bounds[:, 1], (4, 64, 6))
        result = evaluation.descriptive_comparison(chains.reshape(-1, 6), chains,
            self.target["parameter_names"], {"control": [0., .1], "affected": [.3, .9]})
        self.assertEqual(result["quantities"], [*self.target["parameter_names"], "control", "affected"])
        np.testing.assert_allclose(result["standardized_mean_error"], 0., atol=1e-12)
        np.testing.assert_allclose(result["width_ratio"], 1., atol=1e-12)
        self.assertNotIn("family_checks", result)
        self.assertNotIn("status", result)


if __name__ == "__main__":
    unittest.main()

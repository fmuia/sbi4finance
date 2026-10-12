"""Cheap estimator, gate and provenance checks; no frozen neural audit is run."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance import rough_inference, rough_reference
from sbi4finance.gbm_bank import _sha256


SCRIPTS = Path(__file__).parents[1] / "scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    spec = importlib.util.spec_from_file_location("evaluate_rough_tested", SCRIPTS / "evaluate_rough.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)


class RoughEvaluationTests(unittest.TestCase):
    def test_joint_preserves_paired_sample_axis(self):
        h = np.arange(12, dtype=float).reshape(2, 6, 1)
        eta = 100 + h
        pairs = audit.joint({"hurst": h, "eta": eta})
        assert_array_equal(pairs[..., 0], h[..., 0])
        assert_array_equal(pairs[..., 1], eta[..., 0])
        with self.assertRaises(ValueError):
            audit.joint({"hurst": h, "eta": eta[:, :-1]})

    def test_nested_target_mean_sampling_error_counts_inner_variation_once(self):
        prices = np.array([[1., 2.], [3., 6.], [5., 10.], [7., 14.]])
        ref_prices = prices + [0.1, -0.2]
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        # Large descriptive inner errors must not be added to the empirical SE.
        result = audit.target_mean_metrics(prices, ref_prices, weights, np.full_like(prices, 9))
        expected_se = prices.std(axis=0, ddof=1) / np.sqrt(4)
        expected_reference = rough_reference.weighted_target_mean(ref_prices, weights)
        combined = np.sqrt(expected_se**2 + np.asarray(expected_reference["mcse"])**2)
        assert_allclose(result["target_mean"], prices.mean(axis=0))
        assert_allclose(result["target_mean_mcse"], expected_se)
        assert_allclose(result["target_mean_combined_mcse_currency"], combined)
        assert_allclose(result["target_mean_error_plus_2mcse_currency"],
                        np.abs(prices.mean(axis=0) - expected_reference["mean"]) + 2 * combined)
        assert_allclose(result["target_mean_inner_mcse_component"], [4.5, 4.5])
        self.assertNotIn("target_quantiles", result)

    def test_invalid_target_shapes_and_errors_fail(self):
        valid = np.arange(8, dtype=float).reshape(4, 2)
        weights = np.ones(4)
        for values in (valid[:, :1], valid[:2], np.full((4, 2), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                audit.target_mean_metrics(values, valid, weights)
        for errors in (np.ones((4, 1)), -np.ones((4, 2)), np.full((4, 2), np.nan)):
            with self.subTest(errors=errors.shape), self.assertRaises(ValueError):
                audit.target_mean_metrics(valid, valid, weights, errors)

    def test_aggregate_and_all_frozen_gates_are_required(self):
        cases = [
            {"mean_error_in_ref_sd": [0.1, -0.2], "relative_std_error": [0.05, -0.1],
             "correlation_error": -0.03, "reference_mass_of_90_interval": [0.87, 0.95],
             "target_mean_error_plus_2mcse_currency": [0.04, 0.03]},
            {"mean_error_in_ref_sd": [0.3, -0.4], "relative_std_error": [0.2, -0.15],
             "correlation_error": 0.02, "reference_mass_of_90_interval": [0.91, 0.91],
             "target_mean_error_plus_2mcse_currency": [0.06, 0.05]},
        ]
        metrics = audit.aggregate(cases)
        self.assertAlmostEqual(metrics["mean_rmse_in_ref_sd"], np.sqrt(0.075))
        self.assertAlmostEqual(metrics["max_abs_relative_std_error"], 0.2)
        self.assertAlmostEqual(metrics["mean_reference_mass_of_90_interval_abs_error"], 0.03)
        self.assertAlmostEqual(metrics["max_target_mean_error_plus_2mcse_currency"], 0.06)
        self.assertTrue(all(audit.validation_gates(metrics, {"rank_diagnostic_passed": True}, metrics).values()))
        self.assertFalse(audit.validation_gates(metrics, {"rank_diagnostic_passed": False}, metrics)["marginal_sbc"])
        smaller = {**metrics, "max_target_mean_error_plus_2mcse_currency": 0.059}
        self.assertFalse(audit.validation_gates(metrics, {"rank_diagnostic_passed": True}, smaller)[
            "max_target_mean_error_plus_2mcse_currency"])
        parameter_only = audit.aggregate(cases, include_targets=False)
        with self.assertRaises(ValueError):
            audit.validation_gates(parameter_only, {"rank_diagnostic_passed": True}, metrics)
        cases[0]["mean_error_in_ref_sd"][0] = np.nan
        with self.assertRaises(ValueError):
            audit.aggregate(cases)

    def test_checkpoint_identity_and_reserved_test_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank, ref, checkpoint = [root / name for name in ("bank", "reference", "checkpoint")]
            for directory in (bank, ref, checkpoint):
                directory.mkdir()
            protocol = {"training": {"quote_augmentations_per_base": 8}, "selection_validation_ids": [0, 1]}
            config = {"task": {"sentinel": "frozen-task"}}
            manifest = {"configuration": config}
            ref_report = {"reference_bank_manifest_sha256": "independent-reference-identity"}
            protocol_path = root / "protocol.json"
            for path, value in ((protocol_path, protocol), (bank / "manifest.json", manifest),
                                (ref / "reference.json", ref_report), (checkpoint / "task.json", config["task"]),
                                (checkpoint / "protocol.json", protocol), (checkpoint / "bank-manifest.json", manifest)):
                path.write_text(json.dumps(value))
            (checkpoint / "model.keras").write_bytes(b"checkpoint-identity-fixture-only")
            training = {
                "protocol_sha256": _sha256(protocol_path), "bank_configuration": config,
                "checkpoint_sha256": _sha256(checkpoint / "model.keras"), "same_seed_reload_exact": True,
                "reserved_test_evaluated": False, "adapter_id": rough_inference.ADAPTER_ID,
                "training": protocol["training"], "selection_validation_ids": [0, 1],
                "bank_manifest_sha256": _sha256(bank / "manifest.json"),
                "reference_report_sha256": _sha256(ref / "reference.json"),
                "reference_bank_manifest_sha256": ref_report["reference_bank_manifest_sha256"],
                "source_sha256": {"train_rough.py": _sha256(SCRIPTS / "train_rough.py"),
                                  "rough_inference.py": _sha256(rough_inference.__file__)},
                "training_base_examples": 2, "training_examples": 16, "selection_examples": 1,
            }
            splits = {"train": {"parameters": np.ones((2, 2))}}
            result = protocol, config, ref_report, manifest, splits
            with patch.object(audit, "accepted_inputs", return_value=result):
                record = checkpoint / "training.json"
                record.write_text(json.dumps(training))
                self.assertEqual(audit.checked_inputs(bank, ref, checkpoint, protocol_path)[-1], training)
                changes = {"checkpoint_sha256": "wrong", "reference_report_sha256": "wrong",
                           "bank_manifest_sha256": "wrong", "protocol_sha256": "wrong",
                           "same_seed_reload_exact": False, "reserved_test_evaluated": True,
                           "source_sha256": {}, "training_examples": 17, "selection_examples": 2}
                for key, value in changes.items():
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        record.write_text(json.dumps({**training, key: value}))
                        audit.checked_inputs(bank, ref, checkpoint, protocol_path)

    def test_reference_arrays_use_clean_weights_and_exact_audit_cases(self):
        rng = np.random.default_rng(76)
        clean = rng.normal(size=(8, 9))
        observations = rng.normal(size=(2, 9, 1))
        truth = rng.uniform(size=(2, 2))
        task = {"context": {"quote_noise_std": 0.25}}
        protocol = {"reference": {"rows": 8}, "conditional_audit_ids": [0], "stress_prior_fractions": [[0.5, 0.5]]}
        validation = {"observations": observations[:1], "parameters": truth[:1], "example_id": np.array(["validation:0"])}
        arrays = {"parameters": rng.uniform(size=(8, 2)), "reference_prices": clean,
                  "target_estimates": rng.uniform(size=(8, 2)), "observations": observations,
                  "truth": truth, "case_ids": np.array(["validation:0", "stress:0"]),
                  "weights": np.asarray([rough_reference.importance_weights(clean, y[:, 0], 0.25) for y in observations])}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.npz"
            np.savez(path, **arrays)
            restored = audit.checked_reference_arrays(path, protocol, task, validation)
            assert_array_equal(restored["weights"], arrays["weights"])
            broken = deepcopy(arrays)
            broken["weights"] = np.full((2, 8), 1 / 8)
            np.savez(path, **broken)
            with self.assertRaisesRegex(ValueError, "quote law"):
                audit.checked_reference_arrays(path, protocol, task, validation)
            broken = deepcopy(arrays)
            broken["observations"][0, 0, 0] += 1
            np.savez(path, **broken)
            with self.assertRaisesRegex(ValueError, "identities"):
                audit.checked_reference_arrays(path, protocol, task, validation)


if __name__ == "__main__":
    unittest.main()

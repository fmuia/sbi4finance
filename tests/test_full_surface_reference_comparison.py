"""Comparison gating and identity checks without training a second estimator."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance.calibration import tasks
from sbi4finance.gbm_bank import _json

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/compare_full_surface_reference.py"
spec = importlib.util.spec_from_file_location("full_surface_comparison_for_test", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root/"reference"
        self.checkpoint = self.root/"checkpoint"
        self.reference.mkdir()
        self.checkpoint.mkdir()
        self.task = tasks.heston_task()
        fingerprint = hashlib.sha256(_json(self.task).encode()).hexdigest()
        self.training = {"task_fingerprint": fingerprint, "checkpoint_sha256": "checkpoint",
                         "network": {"encoder": "vector", "family": "spline"},
                         "run_contract": {"dataset": {"manifest_sha256": "training-data"}}}
        for name in ("task.json", "training.json", "model.keras"):
            (self.checkpoint/name).write_text("test fixture")
        self.record = {"format": "full_surface_pymc_reference_v1", "task": self.task,
                       "target": "independent Gaussian quote noise around deterministic refined Fourier prices",
                       "observed_prices": np.zeros(63).tolist(), "surface_id": "validation:0", "split": "validation",
                       "settings": {"chains": 4, "draws": 100}}
        self.diagnostics = {"mcmc_diagnostics_passed": True, "parameters": {
            f"parameters[{name}]": {"r_hat": 1.001, "ess_bulk": 500, "ess_tail": 450,
                                    "mcse_mean": 1e-5, "mcse_sd": 1e-5} for name in self.task["parameter_names"]}}
        bounds = np.asarray(self.task["prior_bounds"])
        rng = np.random.default_rng(103)
        self.chains = rng.uniform(bounds[:, 0], bounds[:, 1], size=(4, 100, 4))
        self._write_reference()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_reference(self):
        (self.reference/"reference.json").write_text(json.dumps(self.record))
        (self.reference/"diagnostics.json").write_text(json.dumps(self.diagnostics))
        np.savez(self.reference/"chains.npz", parameters=self.chains)

    def _compare(self, **kwargs):
        with patch.object(comparison.full_surface, "load_checkpoint", return_value=(self.task, object(), self.training)):
            return comparison.compare(self.checkpoint, self.reference, **kwargs)

    def test_failed_diagnostics_is_inconclusive_without_neural_verdict_or_sampling(self):
        self.diagnostics["mcmc_diagnostics_passed"] = False
        self._write_reference()
        with patch.object(comparison.full_surface, "sample") as sample:
            report, neural = self._compare()
        self.assertEqual(report["status"], "inconclusive_reference")
        self.assertIsNone(report["proposed_parameter_agreement"])
        self.assertIsNone(neural)
        sample.assert_not_called()

    def test_claimed_pass_with_failing_parameter_diagnostic_remains_inconclusive(self):
        self.diagnostics["parameters"]["parameters[rho]"]["ess_tail"] = 20
        self._write_reference()
        with patch.object(comparison.full_surface, "sample") as sample:
            report, _ = self._compare()
        self.assertEqual(report["status"], "inconclusive_reference")
        sample.assert_not_called()

    def test_success_uses_pooled_reference_only_for_summaries_and_reports_mcse(self):
        neural = self.chains.reshape(-1, 4)
        with patch.object(comparison.full_surface, "sample", return_value=neural[None]) as sample:
            report, actual = self._compare(draws=400, seed=9)
        self.assertEqual(report["reference_chain_shape"], [4, 100, 4])
        self.assertEqual(report["status"], "completed_development_comparison")
        self.assertFalse(report["scientific_acceptance"])
        self.assertTrue(report["proposed_parameter_agreement"]["all_checks_passed"])
        rho = report["proposed_parameter_agreement"]["parameters"]["rho"]
        self.assertEqual(rho["standardized_mean_error"], 0)
        self.assertEqual(rho["width_ratio_90"], 1)
        self.assertEqual(rho["reference_mcse_mean"], 1e-5)
        self.assertEqual(sample.call_args.kwargs["seed"], 9)
        np.testing.assert_array_equal(actual, neural)
        with np.load(self.reference/"chains.npz") as saved:
            self.assertEqual(saved["parameters"].shape, (4, 100, 4))

    def test_noise_prior_or_observation_mismatch_is_rejected(self):
        original = copy.deepcopy(self.record)
        for mutate in (lambda r: r["task"]["context"]["quote_noise_std"].__setitem__(0, 0.01),
                       lambda r: r["task"]["prior_bounds"][0].__setitem__(1, 0.2),
                       lambda r: r.__setitem__("observed_prices", [0]*62)):
            self.record = copy.deepcopy(original)
            mutate(self.record)
            self._write_reference()
            with self.assertRaises(ValueError):
                self._compare()

    def test_missing_final_chains_is_inconclusive(self):
        (self.reference/"chains.npz").unlink()
        with patch.object(comparison.full_surface, "sample") as sample:
            report, _ = self._compare()
        self.assertEqual(report["status"], "inconclusive_reference")
        self.assertIsNone(report["reference_chain_shape"])
        sample.assert_not_called()

    def test_precision_must_identify_same_reference_files(self):
        precision = {"format": "heston_reference_pointwise_precision_v1", "source_surface_id": "validation:0",
                     "parameter_names": self.task["parameter_names"], "base_numerics": self.task["numerics"],
                     "input_sha256": {n: comparison._digest(self.reference/n)
                                      for n in ("reference.json", "chains.npz", "diagnostics.json")},
                     "all_points_priced": True, "pricing_sources_match_reference": True,
                     "posterior_stability_assessed": False}
        path = self.root/"precision.json"
        path.write_text(json.dumps(precision))
        with patch.object(comparison.full_surface, "sample", return_value=self.chains.reshape(1, -1, 4)):
            report, _ = self._compare(precision=path)
        self.assertEqual(report["status"], "completed_development_comparison")
        self.assertFalse(report["precision"]["posterior_stability_assessed"])
        precision["input_sha256"]["chains.npz"] = "different chains"
        path.write_text(json.dumps(precision))
        with self.assertRaisesRegex(ValueError, "precision report"):
            self._compare(precision=path)


if __name__ == "__main__":
    unittest.main()

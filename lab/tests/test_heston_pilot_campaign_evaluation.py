import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_heston_pilot_campaign as evaluation


BOUNDS = [[.01, .16], [.1, 10.], [.01, .16], [.1, 3.], [-.95, -.1], [0., .02]]


class CampaignEvaluationTests(unittest.TestCase):
    def test_documented_direct_cli_help_without_pythonpath(self):
        lab = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, str(lab / "scripts/evaluate_heston_pilot_campaign.py"), "--help"],
                                cwd=lab, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evaluate", result.stdout)
        self.assertIn("panel", result.stdout)

    def test_native_float32_boundary_rounding_is_reported_without_clipping(self):
        values = np.array([[.01, .1, .01, .1, -.95, 0.]], dtype=np.float32)
        original = values.copy()
        report = evaluation.neural_support_report(values, BOUNDS)
        self.assertGreater(report["representational_excursion_draw_count"], 0)
        self.assertEqual(report["material_outside_draw_count"], 0)
        np.testing.assert_array_equal(values, original)
        strict = evaluation.neural_support_report(values.astype(np.float64), BOUNDS)
        self.assertEqual(strict["material_outside_draw_count"], 1)
        bad = values.copy()
        for _ in range(5):
            bad[0, 0] = np.nextafter(bad[0, 0], np.float32(-np.inf))
        self.assertEqual(evaluation.neural_support_report(bad, BOUNDS)["material_outside_draw_count"], 1)

    def test_market_binding_rejects_a_same_design_synthetic_reference(self):
        observed = np.ones(54)
        market = {"format": "heston_spxw_reference_v1"}
        evaluation.validate_reference_observations(market, observed, "market")
        synthetic = {"format": "heston_training_conditional_reference_v1", "index": 0,
                     "observations": observed.tolist(), "observation_sha256": evaluation.values_digest(observed)}
        with self.assertRaisesRegex(ValueError, "original frozen-market"):
            evaluation.validate_reference_observations(synthetic, observed, "market")
        with self.assertRaisesRegex(ValueError, "observation vector"):
            evaluation.validate_reference_observations(synthetic, observed + 1, "synthetic")

    def test_synthetic_case_binds_index_truth_errors_and_both_observation_vectors(self):
        declared = {"index": 2, "id": "conditional:002", "parameters": [.04, 2., .05, .7, -.6, .01],
                    "standard_normal_errors": [.1] * 54}
        records = {"half_spread": [.2] * 54, "market_vega": [1.] * 54}
        observed = np.ones(54) + .1 * np.hypot(.2, .01)
        case = {**declared, "clean_prices": [1.] * 54, "observations": observed.tolist(),
                "observation_sha256": evaluation.values_digest(observed), "design_sha256": "design", "panel_sha256": "panel"}
        run = {"index": 2, "format": "heston_training_conditional_reference_v1", "observations": observed.tolist(),
               "observation_sha256": evaluation.values_digest(observed), "design_sha256": "design", "panel_sha256": "panel"}
        evaluation.validate_synthetic_case(run, case, declared, "panel", "design", records)
        for target, key, value in ((run, "index", 4), (case, "id", "conditional:004"),
                                   (case, "standard_normal_errors", [.2] * 54),
                                   (run, "observations", [2.] * 54)):
            broken_run, broken_case = copy.deepcopy(run), copy.deepcopy(case)
            (broken_run if target is run else broken_case)[key] = value
            with self.assertRaises(ValueError):
                evaluation.validate_synthetic_case(broken_run, broken_case, declared, "panel", "design", records)

    def test_reserved_test_members_are_never_requested_from_the_bank(self):
        accessed = []

        class GuardedArchive:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __getitem__(self, key):
                if not key.startswith("validation_"):
                    raise AssertionError("reserved or training bank member opened")
                accessed.append(key)
                return np.zeros((10, 5)) if key == "validation_parameters" else np.arange(10)

        with patch.object(evaluation.np, "load", return_value=GuardedArchive()):
            arrays = evaluation.load_validation_members("bank.npz", 3)
        self.assertEqual(accessed, ["validation_parameters", "validation_clean", "validation_row_id"])
        self.assertTrue(all(len(value) == 3 for value in arrays))

    def test_fresh_discrepancy_is_not_a_fixed_bank_column_and_noise_is_untruncated(self):
        physical = np.tile([.04, 2., .05, .7, -.6], (128, 1))
        clean = np.zeros((128, 2))
        records = {"mid": [1., 1.], "half_spread": [.1, .1], "market_vega": [10., 10.]}
        truth, observations = evaluation.fresh_development(physical, clean, records, BOUNDS, 123)
        self.assertGreater(np.std(truth[:, -1]), .004)
        self.assertTrue(np.any(observations < 0))
        again = evaluation.fresh_development(physical, clean, records, BOUNDS, 123)
        np.testing.assert_array_equal(truth, again[0])
        np.testing.assert_array_equal(observations, again[1])
        other = evaluation.fresh_development(physical, clean, records, BOUNDS, 124)
        self.assertFalse(np.array_equal(truth[:, -1], other[0][:, -1]))

    def test_finite_one_draw_randomized_ranks_are_continuously_uniform_under_null(self):
        rng = np.random.default_rng(20261112)
        draws = rng.uniform(size=(4000, 1, 2))
        truth = rng.uniform(size=(4000, 2))
        report = evaluation.calibration_metrics(draws, truth, ["a", "b"], 901)
        for result in report["coordinate_ks"].values():
            self.assertGreater(result["p_value"], .001)
            self.assertFalse(result["holm_reject_at_05"])
        np.testing.assert_allclose(evaluation.holm([.01, .04, .03]), [.03, .06, .06])

    def test_reference_panel_is_deterministic_stratified_and_complete(self):
        design = {"records": {"mid": [1.] * 54}}
        panel = evaluation.declare_reference_panel(design, "design", BOUNDS)
        self.assertEqual(panel, evaluation.declare_reference_panel(design, "design", BOUNDS))
        self.assertEqual(panel["initial_reference_indices"], [0, 2, 4, 6, 8, 10])
        categories = [case["category"] for case in panel["cases"]]
        self.assertEqual(categories.count("broad_prior"), 6)
        self.assertEqual(categories.count("market_like"), 4)
        self.assertEqual(categories.count("low_tau_boundary"), 2)
        for case in panel["cases"]:
            self.assertEqual(len(case["standard_normal_errors"]), 54)
            self.assertEqual(len(case["parameters"]), 6)
            self.assertGreater(case["parameters"][-1], 0)

    def test_identity_mismatches_reject_instead_of_comparing_unmatched_priors(self):
        task = {"parameter_names": evaluation.NAMES, "prior_bounds": BOUNDS,
                "design_sha256": "design", "tau_training_bounds": [0., .02]}
        fp = evaluation.fingerprint(task)
        metadata = {"task": task, "design_sha256": "design", "names": evaluation.NAMES, "bounds": BOUNDS,
                    "task_fingerprint": fp, "bank_record_sha256": "record", "bank_payload_sha256": "payload"}
        bank = {"payload_sha256": "payload", "identity": {"task": task, "task_fingerprint": fp}}
        evaluation.validate_identity(metadata, "design", bank, "record", "payload")
        bad = copy.deepcopy(metadata)
        bad["bounds"][1][1] = 20.
        with self.assertRaisesRegex(ValueError, "identities differ"):
            evaluation.validate_identity(bad, "design", bank, "record", "payload")
        with self.assertRaisesRegex(ValueError, "identities differ"):
            evaluation.validate_identity(metadata, "wrong-design", bank, "record", "payload")


if __name__ == "__main__":
    unittest.main()

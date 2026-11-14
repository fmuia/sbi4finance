"""Support violations must be distinguished from passing a necessary envelope."""

import copy
import json
import unittest

import numpy as np

from sbi4f_tutorial.rough_options import context_checks
from sbi4f_tutorial.shared import backbones, tasks
from sbi4f_tutorial.shared.paths import ASSETS


class RoughContextTests(unittest.TestCase):
    def setUp(self):
        self.task, _, self.case = backbones.load_example("rough-known100k-validation0")

    def test_saved_spy_curve_provably_exceeds_current_training_context_support(self):
        saved = json.loads((ASSETS / "results/calibration/spy/surface.json").read_text())
        result = context_checks.context_support(self.task, saved["forward_variance_levels"])
        self.assertEqual(result["status"], "outside_support")
        self.assertTrue(result["violates_necessary_bounds"])
        self.assertIn(7, result["violating_intervals"])
        self.assertAlmostEqual(result["observed_end_ratio"], 2.9335368733726876)
        self.assertAlmostEqual(result["maximum_end_ratio"], 2.1503461322213813)
        self.assertEqual(result["lower_ratios"][0], 1)
        self.assertEqual(result["upper_ratios"][0], 1)

    def test_known_in_prior_curve_passes_but_passing_is_not_a_full_support_claim(self):
        result = context_checks.context_support(self.task, self.case["context_values"])
        self.assertEqual(result["status"], "not_ruled_out")
        self.assertFalse(result["violates_necessary_bounds"])
        # This impossible absolute scale has the same admissible ratios.
        counterexample = context_checks.context_support(self.task, np.full(7, 100.0))
        self.assertEqual(counterexample["status"], "not_ruled_out")
        self.assertIn("passing does not establish full joint", counterexample["scope"])

    def test_ratio_check_is_invariant_to_a_common_positive_scale(self):
        first = context_checks.context_support(self.task, self.case["context_values"])
        second = context_checks.context_support(self.task, 3 * self.case["context_values"])
        np.testing.assert_allclose(first["observed_ratios"], second["observed_ratios"], rtol=2e-16)
        np.testing.assert_array_equal(first["lower_ratios"], second["lower_ratios"])
        np.testing.assert_array_equal(first["upper_ratios"], second["upper_ratios"])

    def test_actual_prior_draws_obey_necessary_bounds(self):
        rng = np.random.default_rng(37)
        for _ in range(50):
            levels, _ = tasks.draw_context(self.task, rng)
            self.assertFalse(context_checks.context_support(self.task, levels)["violates_necessary_bounds"])

    def test_invalid_levels_and_unsupported_context_laws_fail_explicitly(self):
        invalid = [np.zeros(7), np.ones(6), np.full(7, np.nan), np.ones(7, dtype=complex),
                   np.ma.array(np.ones(7), mask=False)]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "curve level"):
                    context_checks.context_support(self.task, values)
        alternate = copy.deepcopy(self.task)
        alternate["context_prior"]["forward_variance"] = {
            "kind": "power_law_walk", "level_bounds": [.1, .4], "exponent_bounds": [-.6, .6],
            "walk_sd": .15, "vol_bounds": [.06, .6]}
        with self.assertRaisesRegex(ValueError, "linear_wobble"):
            context_checks.context_support(alternate, np.full(7, .04))
        joint, _, _ = backbones.load_example("rough-joint100k-validation0")
        with self.assertRaisesRegex(ValueError, "linear_wobble"):
            context_checks.context_support(joint, np.full(7, .04))


if __name__ == "__main__":
    unittest.main()

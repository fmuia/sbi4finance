"""Benchmark protocol checks; elapsed time never determines a test result."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts import benchmark_heston_pricing as benchmark
from sbi4f_tutorial.shared import pricing


ROOT = Path(__file__).resolve().parents[1]


class PricingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.task = json.loads((ROOT / "configs/tasks/calibration_heston_bank.json").read_text())["task"]

    def test_panel_is_fixed_and_preserves_corners_and_order(self):
        panel = benchmark.make_panel(self.task)
        benchmark.validate_panel(panel, self.task)
        self.assertEqual(len(panel["rows"]), 276)
        np.testing.assert_array_equal(panel["rows"][256]["parameters"], np.asarray(self.task["prior_bounds"])[:, 0])
        np.testing.assert_array_equal(panel["rows"][271]["parameters"], np.asarray(self.task["prior_bounds"])[:, 1])
        self.assertEqual(benchmark.identity(panel), benchmark.identity(benchmark.make_panel(self.task)))
        altered = copy.deepcopy(panel)
        altered["rows"][0]["parameters"][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "predeclared"):
            benchmark.validate_panel(altered, self.task)

    def test_real_checked_and_notebook_routes_match_frozen_reference(self):
        row = np.asarray(self.task["prior_bounds"]).mean(1)
        sigma = np.asarray(self.task["context"]["quote_noise_std"])
        for route in ("checked", "notebook"):
            actual = benchmark.evaluate(self.task, row, route)
            reference = benchmark.evaluate(self.task, row, route, frozen=True)
            compared = benchmark.compare_rows([dict(id="midpoint", status="ok", **actual)],
                                             [dict(id="midpoint", status="ok", **reference)], sigma)
            self.assertTrue(compared["passes_price_and_failure_gate"])
            self.assertEqual(compared["max_abs_refinement_difference"], 0.)

    def test_failure_is_retained_and_current_function_is_restored(self):
        original = pricing.heston_surface
        row = {"id": "difficult", "parameters": [0., 0., 100., -2.]}
        result = benchmark.timed_row(self.task, row, "checked", False)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["id"], "difficult")
        self.assertIs(pricing.heston_surface, original)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            with patch.object(benchmark, "evaluate", side_effect=RuntimeError("unexpected")):
                benchmark.timed_row(self.task, row, "checked", False)

    def test_comparison_checks_prices_noise_units_failures_and_ids(self):
        reference = [{"id": "a", "status": "ok", "prices": [1.], "refined_minus_base": [0.]},
                     {"id": "b", "status": "failed", "error": "bounds"}]
        altered = copy.deepcopy(reference)
        altered[0]["prices"][0] += 1e-10
        result = benchmark.compare_rows(altered, reference, np.array([1e-5]))
        self.assertFalse(result["passes_price_and_failure_gate"])
        altered = copy.deepcopy(reference)
        altered[1] = {"id": "b", "status": "ok", "prices": [0.], "refined_minus_base": [0.]}
        self.assertEqual(benchmark.compare_rows(altered, reference, np.array([1.]))["failure_outcome_mismatch_ids"], ["b"])
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            benchmark.compare_rows(reference[::-1], reference, np.array([1.]))


if __name__ == "__main__":
    unittest.main()

"""Chain selection, Gaussian sensitivity and failure preservation for refinement."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import check_heston_reference_precision as precision
from sbi4finance.calibration import tasks


class ReferencePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = tasks.heston_task()
        theta = np.mean(self.task["prior_bounds"], axis=1)
        self.parameters = np.broadcast_to(theta, (2, 4, 4)).copy()
        record = {"format": "full_surface_pymc_reference_v1", "task": self.task,
                  "observed_prices": np.zeros(63).tolist(), "settings": {"chains": 2, "draws": 4},
                  "surface_id": "validation:0"}
        (self.root / "reference.json").write_text(json.dumps(record))
        (self.root / "diagnostics.json").write_text(json.dumps({"mcmc_diagnostics_passed": False}))
        np.savez_compressed(self.root / "chains.npz", parameters=self.parameters)

    def test_selection_preserves_axes_and_all_chains(self):
        expected = [(0, 0), (0, 2), (0, 5), (1, 0), (1, 2), (1, 5), (2, 0), (2, 5)]
        self.assertEqual(precision.selected_indices((3, 6, 4), 8), expected)
        self.assertEqual(precision.selected_indices((2, 2, 4), 20), [(0, 0), (0, 1), (1, 0), (1, 1)])
        for shape, points in (((4, 10, 4), 3), ((2, 10), 16), ((2, 10, 4), 21)):
            with self.assertRaises(ValueError):
                precision.selected_indices(shape, points)

    def test_refinement_is_a_copy_and_doubles_nodes_and_cutoffs(self):
        old = copy.deepcopy(self.task)
        fine = precision.refined_task(self.task)
        self.assertEqual(self.task, old)
        self.assertEqual(fine["numerics"]["quadrature_ladder"],
                         [[2*n, 2*c] for n, c in self.task["numerics"]["quadrature_ladder"]])

    def test_price_and_log_likelihood_shift_keep_inconclusive_chain_status(self):
        original_nodes = self.task["numerics"]["quadrature_ladder"][0][0]
        sigma = tasks.sigma(self.task)

        def prices(task, values, context, seed):
            shift = 0.0 if task["numerics"]["quadrature_ladder"][0][0] == original_nodes else 0.5
            value = shift * sigma
            return value, None, None, np.zeros(63), None

        with patch.object(tasks, "price", side_effect=prices):
            report = precision.check(self.root, 4)
        self.assertEqual([(p["chain"], p["draw"]) for p in report["points"]], [(0, 0), (0, 3), (1, 0), (1, 3)])
        self.assertAlmostEqual(report["max_abs_price_shift_over_quote_sigma"], 0.5)
        self.assertAlmostEqual(report["points"][0]["log_likelihood_shift"], -0.5 * 63 * 0.5**2)
        self.assertFalse(report["source_mcmc_diagnostics_passed"])
        self.assertFalse(report["scientific_acceptance"])
        self.assertFalse(report["posterior_stability_assessed"])
        self.assertTrue(report["all_points_priced"])

    def test_pricing_failures_are_retained_without_replacement(self):
        with patch.object(tasks, "price", side_effect=ValueError("unpriced refinement")) as price:
            report = precision.check(self.root, 4)
        self.assertEqual(price.call_count, 4)
        self.assertEqual(report["unpriced_points"], 4)
        self.assertFalse(report["all_points_priced"])
        self.assertTrue(all(p["error"] == "unpriced refinement" for p in report["points"]))

    def test_invalid_chain_axes_or_support_fail_before_pricing(self):
        for values in (self.parameters.reshape(8, 4), np.full((2, 4, 4), np.nan), self.parameters + 100):
            np.savez_compressed(self.root / "chains.npz", parameters=values)
            with self.assertRaises(ValueError):
                precision.check(self.root, 4)

    def test_actual_checked_fourier_refinement_is_finite(self):
        report = precision.check(self.root, 2)
        self.assertTrue(report["all_points_priced"])
        self.assertTrue(np.isfinite(report["max_abs_log_likelihood_shift"]))
        self.assertLess(report["max_abs_price_shift_over_quote_sigma"], 0.1)


if __name__ == "__main__":
    unittest.main()

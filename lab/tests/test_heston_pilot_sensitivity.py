import copy
import json
from pathlib import Path
import unittest

import numpy as np

from scripts import check_heston_pilot_sensitivity as sensitivity


DESIGN = Path(__file__).resolve().parents[1] / "evidence/pilot-results/2026-11-09/heston-observations/designs/put_heavy.json"


class HestonPilotSensitivityTests(unittest.TestCase):
    def test_carry_axes_have_declared_radius_and_ignore_price_unit_choice(self):
        covariance = np.array([[.04, .00006], [.00006, .000001]])
        axes = sensitivity.standardized_carry_axes(covariance)
        for direction in axes:
            self.assertAlmostEqual(direction @ np.linalg.solve(covariance, direction), 4., places=12)
        unit_change = np.diag([100., 1.])
        changed = sensitivity.standardized_carry_axes(unit_change @ covariance @ unit_change)
        np.testing.assert_allclose(changed, axes @ unit_change, rtol=1e-12, atol=1e-12)

    def test_scenarios_freeze_quotes_and_scales_and_use_positive_actual_contracts(self):
        design = json.loads(DESIGN.read_text())
        original = copy.deepcopy(design)
        scenarios = sensitivity.scenario_definitions()
        self.assertEqual(len(scenarios), 11)
        for case in scenarios:
            varied, details = sensitivity.perturb_design(design, case)
            for key in ("market_vega", "market_iv", "mid", "half_spread", "strike", "option_type"):
                self.assertEqual(varied["records"][key], design["records"][key])
            self.assertTrue(np.all(np.asarray(varied["records"]["maturity"]) > 0))
            if case["kind"] == "carry":
                for expiry in details:
                    self.assertAlmostEqual(details[expiry]["mahalanobis_radius"], 2., places=12)
            if case["kind"] == "clock":
                change = -case["valuation_shift_minutes"] / sensitivity.MINUTES_PER_YEAR
                np.testing.assert_allclose(np.asarray(varied["records"]["maturity"])
                                           - np.asarray(design["records"]["maturity"]), change, atol=1e-15)
                self.assertEqual(varied["products"]["control"][0], 0.)
                self.assertAlmostEqual(varied["products"]["affected"][0] - design["products"]["affected"][0], change)
        self.assertEqual(design, original)

    def test_optimizes_physical_uniform_map_at_boundary_without_jacobian(self):
        class QuadraticPilot:
            bounds = np.array([[.01, .16], [-.95, -.1]])
            names = ["positive", "negative"]

            def log_likelihood(self, point, mask):
                return -100. * (point[0] - .02)**2 - (point[1] + .1)**2

        pilot = QuadraticPilot()
        result = sensitivity.optimize_physical(pilot, None, [np.array([.1, -.5]), np.array([.05, -.3])], 1000)
        np.testing.assert_allclose(result["best"]["parameters"], [.02, -.1], atol=2e-6)
        self.assertIn("negative", result["parameters_near_support_boundary"])
        self.assertTrue(result["all_starts_converged"])


if __name__ == "__main__":
    unittest.main()

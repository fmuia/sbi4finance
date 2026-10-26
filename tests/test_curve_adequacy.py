"""Analytic family, whole-horizon positivity and bounded screen mechanics."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from sbi4finance.calibration import curve_adequacy as ca, pricing, surface, tasks


class CurveFamilyTests(unittest.TestCase):
    def test_exponential_is_no_hump_special_case(self):
        times = np.array([0, 1e-10, 0.1, 1, 2])
        coefficients = [0.025, 0.045, 0, 0.5, 0.3]
        assert_allclose(ca.hump_values(times, coefficients), pricing.heston_variance_values(times, 0.07, 0.025, 2))
        assert_allclose(ca.hump_integrated_variance(times, coefficients),
                        pricing.heston_integrated_variance(times, 0.07, 0.025, 2), atol=1e-16)

    def test_integral_matches_quadrature_including_small_times(self):
        coefficients = [0.035, -0.015, 0.12, 0.25, 0.3]
        for horizon in (1e-10, 1e-4, 0.1, 1, 2):
            exact, _ = quad(lambda t: ca.hump_values(t, coefficients), 0, horizon, epsabs=1e-14)
            assert_allclose(ca.hump_integrated_variance(horizon, coefficients), exact, rtol=1e-12, atol=1e-20)

    def test_interior_negative_dip_rejected_even_with_positive_endpoints(self):
        coefficients = [0.03, 0.0, -0.1, 0.5, 0.1]
        self.assertGreater(ca.hump_values(0, coefficients), 0)
        self.assertGreater(ca.hump_values(1, coefficients), 0)
        with self.assertRaisesRegex(ValueError, "throughout"):
            ca.positive_hump_minimum(coefficients, 1)

    def test_fast_timescale_dip_not_hidden_by_underflow_at_endpoint(self):
        for b1 in (0.0, 0.001, -0.001):
            coefficients = [0.01, b1, -0.1, 0.002, 0.001]
            self.assertGreater(ca.hump_values(0, coefficients), 0)
            self.assertGreater(ca.hump_values(1, coefficients), 0)
            self.assertLess(ca.hump_values(0.001, coefficients), 0)
            with self.subTest(b1=b1), self.assertRaisesRegex(ValueError, "throughout"):
                ca.positive_hump_minimum(coefficients, 1)

    def test_minimum_matches_dense_independent_checks(self):
        rng = np.random.default_rng(140)
        grid = np.linspace(0, 2, 10001)
        for _ in range(30):
            c = [rng.uniform(0.03, 0.1), rng.uniform(-0.02, 0.1), rng.uniform(-0.02, 0.1),
                 rng.uniform(0.05, 1.5), rng.uniform(0.05, 1.5)]
            result = ca.positive_hump_minimum(c, 2)
            numerical = ca.hump_values(grid, c).min()
            self.assertLessEqual(result, numerical+1e-12)
            self.assertLess(numerical-result, 1e-6)

    def test_all_predeclared_cases_positive_and_reproducible_streams(self):
        for case in ca.panel():
            self.assertGreater(ca.positive_hump_minimum(case["curve_coefficients"], 2), 0)
            self.assertEqual(ca.seed_for(12, case["name"], "fit"), ca.seed_for(12, case["name"], "fit"))
            self.assertNotEqual(ca.seed_for(12, case["name"], "fit"), ca.seed_for(12, case["name"], "audit"))

    def test_hump_node_pricing_has_the_discrete_black_scholes_limit(self):
        coefficients = ca.panel()[2]["curve_coefficients"]
        blocks = [{"horizon": 1.0, "n_steps": 64}]
        curve = ca.hump_curve(coefficients, blocks)
        contract = {"strike": 1.0, "maturity": 1.0, "option_type": "put"}
        prices, se, _ = pricing.rough_block({"hurst": 0.12, "eta": 0.0, "rho": 0.0}, curve,
                                             [contract], horizon=1.0, n_steps=64, n_pairs=32, seed=12)
        left_integral = np.mean(ca.hump_values(np.arange(64)/64, coefficients))
        expected = surface.bs_price(0, 1, np.sqrt(left_integral), "put")
        assert_allclose(prices, [expected], atol=1e-13)
        self.assertLess(float(se.max()), 1e-13)

    def test_bounded_screen_audits_on_distinct_streams_and_never_accepts_science(self):
        seeds = []

        def fake_price(task, rough, curve, paths, seed):
            seeds.append(seed)
            c = len(task["context"]["contracts"])
            # A cheap deterministic smooth forward map verifies bounded control flow only.
            value = np.interp(0.1, curve["times"], curve["values"])
            return np.full(c, value), np.zeros(c), np.zeros((c, c)), np.full(3, paths)

        with patch.object(ca, "price_curve", side_effect=fake_price):
            report = ca.run_case(tasks.joint_rough_task(), ca.panel()[0], fit_evaluations=8,
                                  audit_replicates=2, fit_paths=64, audit_paths=128)
        self.assertEqual(report["status"], "completed_price_screen")
        self.assertFalse(report["scientific_acceptance"])
        self.assertLessEqual(report["fit"]["price_evaluations"], 8)
        audit = report["streams"]["audit_fit"]+report["streams"]["audit_truth"]
        self.assertEqual(len(set(audit)), 4)
        self.assertNotIn(report["streams"]["fit_objective"], audit)
        self.assertTrue(all(s in seeds for s in audit))

    def test_pricing_failure_is_recorded_not_redrawn(self):
        with patch.object(ca, "price_curve", side_effect=ValueError("unpriced seed")) as price:
            report = ca.run_case(tasks.joint_rough_task(), ca.panel()[0], fit_evaluations=8)
        self.assertEqual(price.call_count, 1)
        self.assertEqual(report["status"], "inconclusive_failure")
        self.assertEqual(report["error"], "unpriced seed")


if __name__ == "__main__":
    unittest.main()

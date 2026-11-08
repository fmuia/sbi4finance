"""Analytic extensions preserve the observed curve, paired rows and untouched tail prior."""

import unittest

import numpy as np
from scipy.integrate import quad

from sbi4f_tutorial.rough_options import studies
from sbi4f_tutorial.shared import backbones, pricing


class RoughAnalyticStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task, _, arrays = backbones.load_example("rough-joint100k-validation0")
        cls.rows = arrays["neural_draws"][:3].copy()
        cls.known, _, cls.known_arrays = backbones.load_example("rough-known100k-validation0")

    @staticmethod
    def xi(t, row):
        short, long, decay = row[3:]
        return long + (short - long) * np.exp(-decay * t)

    def test_original_functionals_match_quadrature_and_preserve_complete_row_order(self):
        before = self.rows.copy()
        result = studies.curve_functionals(self.task, self.rows)
        for name, maturity in (("variance_swap_3m", .25), ("variance_swap_1y", 1.0)):
            expected = [quad(self.xi, 0, maturity, args=(row,), epsabs=1e-12)[0] / maturity for row in self.rows]
            np.testing.assert_allclose(result[name], expected, rtol=1e-12)
            reversed_result = studies.curve_functionals(self.task, self.rows[::-1])
            np.testing.assert_array_equal(reversed_result[name], result[name][::-1])
        np.testing.assert_array_equal(self.rows, before)

    def test_extended_integrals_match_quadrature_at_and_after_the_quote_boundary(self):
        times, alpha = np.array([0, .25, 1, 1.5, 3, 5]), np.array([-.3, 0, .2])
        result = studies.extended_integrated_variance(self.task, self.rows, times, alpha)
        for i, (row, slope) in enumerate(zip(self.rows, alpha, strict=True)):
            for j, maturity in enumerate(times):
                base = quad(self.xi, 0, min(maturity, 1), args=(row,), epsabs=1e-12)[0]
                tail = quad(lambda t: self.xi(t, row) * np.exp(slope * (t - 1)), 1, max(maturity, 1), epsabs=1e-12)[0]
                self.assertAlmostEqual(result[i, j], base + tail, places=12)
        reversed_result = studies.extended_integrated_variance(self.task, self.rows[::-1], times, alpha[::-1])
        np.testing.assert_array_equal(reversed_result, result[::-1])

    def test_quotes_and_short_functionals_cannot_learn_or_depend_on_the_tail_slope(self):
        times = np.array([.02, .25, .5, 1])
        a = studies.extended_integrated_variance(self.task, self.rows, times, np.array([-.5, 0, .5]))
        b = studies.extended_integrated_variance(self.task, self.rows, times, np.array([2, -2, 1]))
        original = np.asarray([pricing.heston_integrated_variance(times, *row[3:]) for row in self.rows])
        np.testing.assert_array_equal(a, original)
        np.testing.assert_array_equal(b, original)

    def test_zero_slope_recovers_the_original_curve_and_alpha_equal_decay_is_finite(self):
        times = np.array([.25, 1, 3, 5])
        expected = np.asarray([pricing.heston_integrated_variance(times, *row[3:]) for row in self.rows])
        zero = studies.extended_integrated_variance(self.task, self.rows, times, np.zeros(3))
        np.testing.assert_array_equal(zero, expected)
        near = studies.extended_integrated_variance(self.task, self.rows, times, np.array([1e-12, -1e-12, 0]))
        np.testing.assert_allclose(near, zero, rtol=6e-12)
        equal_decay = studies.extended_integrated_variance(self.task, self.rows, [3], self.rows[:, 5])[:, 0]
        for actual, row in zip(equal_decay, self.rows, strict=True):
            expected_tail = quad(lambda t: self.xi(t, row) * np.exp(row[5] * (t - 1)), 1, 3, epsabs=1e-12)[0]
            np.testing.assert_allclose(actual, pricing.heston_integrated_variance(1, *row[3:]) + expected_tail, rtol=2e-13)
        result = studies.tail_functionals(self.task, self.rows, 0)
        np.testing.assert_array_equal(result["variance_swap"], result["base_variance_swap"])
        np.testing.assert_array_equal(result["forward_variance"], result["base_forward_variance"])

    def test_width_reuses_independent_normals_and_same_fitted_rows_without_learning_alpha(self):
        before = self.rows.copy()
        a = studies.tail_functionals(self.task, self.rows, .2, seed=37)
        b = studies.tail_functionals(self.task, self.rows, .4, seed=37)
        np.testing.assert_array_equal(a["alpha"], .2 * np.random.default_rng(37).standard_normal(len(self.rows)))
        np.testing.assert_array_equal(b["alpha"], 2 * a["alpha"])
        np.testing.assert_array_equal(a["base_variance_swap"], b["base_variance_swap"])
        np.testing.assert_array_equal(a["base_forward_variance"], b["base_forward_variance"])
        totals = studies.extended_integrated_variance(self.task, self.rows, [1, 3], a["alpha"])
        np.testing.assert_array_equal(a["variance_swap"], totals[:, 1] / 3)
        np.testing.assert_array_equal(a["forward_variance"], (totals[:, 1] - totals[:, 0]) / 2)
        np.testing.assert_array_equal(self.rows, before)
        self.assertEqual(a["metadata"]["rows"], len(self.rows))

    def test_supplied_curve_rates_are_deterministic_integrals_of_that_supplied_curve(self):
        flat = np.full_like(self.known_arrays["context_values"], .04)
        result = studies.supplied_curve_functionals(self.known, flat)
        self.assertAlmostEqual(result["variance_swap_3m"], .04, places=15)
        self.assertAlmostEqual(result["variance_swap_1y"], .04, places=15)
        with self.assertRaisesRegex(ValueError, "supplied-curve"):
            studies.supplied_curve_functionals(self.task, flat)
        with self.assertRaisesRegex(ValueError, "six-target"):
            studies.curve_functionals(self.known, self.rows)

    def test_marginal_shuffle_keeps_parameter_marginals_but_changes_the_derived_distribution(self):
        shuffled = self.rows.copy()
        shuffled[:, 3] = np.roll(shuffled[:, 3], 1)
        shuffled[:, 4] = np.roll(shuffled[:, 4], 2)
        np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(self.rows, axis=0))
        original = studies.curve_functionals(self.task, self.rows)
        comparison = studies.curve_functionals(self.task, shuffled)
        self.assertFalse(np.array_equal(original["variance_swap_1y"], comparison["variance_swap_1y"]))

    def test_invalid_support_slope_width_and_horizon_stop_without_dropping_rows(self):
        outside = self.rows.copy()
        outside[0, 3] = 0
        for invalid in (outside, self.rows[:, :-1], np.full_like(self.rows, np.nan), self.rows.astype(complex)):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    studies.curve_functionals(self.task, invalid)
        for width in (-.1, .51, np.nan, True):
            with self.assertRaises(ValueError):
                studies.tail_functionals(self.task, self.rows, width)
        for maturity in (1, 0, 5.01, np.inf):
            with self.assertRaises(ValueError):
                studies.tail_functionals(self.task, self.rows, .2, maturity=maturity)
        for alpha in (np.array([0]), np.array([0, np.inf, 0]), np.full(3, 1000.0)):
            with self.assertRaises(ValueError):
                studies.extended_integrated_variance(self.task, self.rows, [5], alpha)


if __name__ == "__main__":
    unittest.main()

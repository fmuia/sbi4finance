"""Analytic extensions preserve the observed curve, paired rows and untouched tail prior."""

import unittest

import numpy as np
from scipy.integrate import quad

from sbi4f_tutorial.rough_options import studies
from sbi4f_tutorial.shared import backbones, pricing
from sbi4f_tutorial.shared import parameter_transforms as transforms


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


class RoughPointCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task, _, arrays = backbones.load_example("rough-joint100k-validation0")
        cls.rows = arrays["neural_draws"][:4].copy()
        cls.sigma = np.asarray(cls.task["context"]["quote_noise_std"])
        cls.bounds = np.asarray(cls.task["prior_bounds"])
        cls.matrix = .05 * np.vstack([np.eye(6) for _ in range(11)])[:63]
        cls.intercept = np.linspace(1., 2., 63)

    def pricer(self, logits):
        return np.asarray(logits) @ self.matrix.T + self.intercept

    def quotes(self, logits):
        return self.pricer(np.asarray(logits)[None, :])[0] * self.sigma

    def test_scaled_least_squares_recovers_known_minimum_without_jacobian_or_prior_penalty(self):
        target = np.array([.8, -.9, .5, -.4, .7, -.6])
        quote = self.quotes(target)
        fit = studies.point_calibration(self.pricer, self.task, quote, starts=3, seed=31)
        np.testing.assert_allclose(fit["logits"], target, atol=1e-7)
        np.testing.assert_allclose(fit["parameters"], transforms.to_bounded(target, self.bounds), atol=2e-8)
        np.testing.assert_allclose(fit["fitted_prices"], quote, atol=1e-13)
        np.testing.assert_array_equal(fit["observation"], quote)
        self.assertLess(fit["chi2"], 1e-16)
        self.assertEqual(fit["chi2"], fit["residuals_noise_units"] @ fit["residuals_noise_units"])
        np.testing.assert_array_equal(fit["runs"][0]["initial_logits"], np.zeros(6))
        for run in fit["runs"]:
            self.assertLessEqual(run["evaluations"], 500)
            self.assertTrue(np.all(np.abs(run["initial_logits"]) <= 12))
        self.assertEqual(fit["settings"]["numerical_logit_bounds"], [-12., 12.])

    def test_changing_observation_changes_the_fit_but_not_random_initializations(self):
        first, second = np.zeros(6), np.array([-.3, .5, .7, .2, -.6, .8])
        a = studies.point_calibration(self.pricer, self.task, self.quotes(first), starts=2, seed=37)
        b = studies.point_calibration(self.pricer, self.task, self.quotes(second), starts=2, seed=37)
        self.assertFalse(np.array_equal(a["parameters"], b["parameters"]))
        np.testing.assert_allclose(b["logits"], second, atol=1e-7)
        for start_a, start_b in zip(a["runs"], b["runs"], strict=True):
            np.testing.assert_array_equal(start_a["initial_parameters"], start_b["initial_parameters"])

    def test_hard_budget_counts_finite_difference_calls_and_returns_no_partial_fit(self):
        calls = []

        def counted(z):
            calls.append(z.copy())
            return self.pricer(z)

        with self.assertRaisesRegex(RuntimeError, "No local start converged.*evaluation_budget"):
            studies.point_calibration(counted, self.task, self.quotes(np.ones(6)),
                                      starts=3, max_evaluations=1)
        self.assertEqual(len(calls), 3, "one residual call per start, with no final hidden re-evaluation")

    def test_native_scaled_price_contract_and_invalid_inputs_fail(self):
        known, _, _ = backbones.load_example("rough-known100k-validation0")
        with self.assertRaisesRegex(ValueError, "joint-curve"):
            studies.point_calibration(self.pricer, known, np.zeros(63))
        for quote in (np.zeros((1, 63)), np.full(63, np.nan), np.zeros(63, complex)):
            with self.assertRaises(ValueError):
                studies.point_calibration(self.pricer, self.task, quote)
        for bad_pricer in (lambda z: np.zeros(63), lambda z: np.full((1, 63), np.nan)):
            with self.assertRaisesRegex(RuntimeError, "No local start converged.*numerical_failure"):
                studies.point_calibration(bad_pricer, self.task, np.zeros(63), starts=1)
        for kwargs in ({"starts": 0}, {"max_evaluations": True}, {"seed": -1}):
            with self.assertRaises(ValueError):
                studies.point_calibration(self.pricer, self.task, np.zeros(63), **kwargs)

    def test_uncertainty_transfer_retains_pairs_and_plugin_strike_matches_quadrature(self):
        from copy import deepcopy

        quote = self.quotes(np.array([.1, -.2, .3, .4, -.5, .6]))
        fit = studies.point_calibration(self.pricer, self.task, quote, starts=1)
        before = self.rows.copy()
        effect = studies.calibration_effects(self.task, self.rows, fit)
        reversed_effect = studies.calibration_effects(self.task, self.rows[::-1], fit)
        np.testing.assert_array_equal(self.rows, before)
        np.testing.assert_array_equal(effect["posterior_curves"][::-1], reversed_effect["posterior_curves"])
        np.testing.assert_array_equal(effect["posterior_variance_swap_1y"][::-1],
                                      reversed_effect["posterior_variance_swap_1y"])
        expected = quad(RoughAnalyticStudyTests.xi, 0, 1, args=(fit["parameters"],), epsabs=1e-12)[0]
        self.assertAlmostEqual(effect["point_variance_swap_1y"], expected, places=12)
        expected_curve = pricing.heston_variance_values(effect["times_years"], *fit["parameters"][3:])
        np.testing.assert_array_equal(effect["point_curve"], expected_curve)
        altered = deepcopy(self.task)
        altered["context"]["quote_noise_std"][0] *= 2
        with self.assertRaisesRegex(ValueError, "exact task"):
            studies.calibration_effects(altered, self.rows, fit)

    def test_posterior_propagation_needs_no_pricer_and_invents_no_point_estimate(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        effect = studies.calibration_effects(self.task, self.rows)
        for field in ("point_parameters", "point_curve", "point_variance_swap_1y"):
            self.assertIsNone(effect[field])
        np.testing.assert_array_equal(effect["posterior_variance_swap_1y"],
                                      studies.curve_functionals(self.task, self.rows)["variance_swap_1y"])
        figure = studies.calibration_effect_figure(self.task, self.rows)
        self.assertEqual(len(figure.axes[0].lines), 1, "only the posterior median curve is drawn")
        self.assertEqual(len(figure.axes[1].lines), 0, "no plug-in strike can be invented")
        self.assertIn("load a pricer", figure._suptitle.get_text())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()

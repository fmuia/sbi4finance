"""The running example of the slot 1 slides: its design, pricing formulas, error law and two routes to a range."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad

from sbi4f_tutorial.heston_options import model as heston
from sbi4f_tutorial.shared import surface

TUTORIAL = Path(__file__).resolve().parents[1]
SOURCE = TUTORIAL / "slides" / "slot1" / "illustration.py"


def load_illustration():
    spec = importlib.util.spec_from_file_location("slot1_illustration", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


il = load_illustration()


class DesignTests(unittest.TestCase):
    def test_the_design_is_the_tutorials_63_spx_quotes(self):
        self.assertEqual(len(il.SPX["strike"]), 63)
        self.assertEqual(il.DAYS, (7, 14, 30, 60, 81, 172, 325))
        z = np.log(il.SPX["strike"]) / (il.SPX_REFERENCE * np.sqrt(il.SPX["maturity"]))
        self.assertLess(np.max(np.abs(z - il.SPX["target_z"])), 0.27)  # every quote within 0.26 of its target
        np.testing.assert_array_equal(il.SPX["put"], z <= 0.0)  # out of the money: puts below the forward
        self.assertTrue(np.all(il.SPX["half_spread"] > 0))
        median = np.median(il.SPX["half_spread"] / surface.bs_vega(np.log(il.SPX["strike"]), il.SPX["maturity"],
                                                                   0.17))
        self.assertTrue(0.0003 < median < 0.002)  # a few hundredths of a volatility point

    def test_readings_are_nested_by_expiry(self):
        readings = il.readings()
        self.assertEqual([readings[reach].n for reach in il.REACH], [63, 45, 27, 18, 9])
        full = readings["325 days"]
        for reading in readings.values():
            np.testing.assert_array_equal(reading.observed, full.observed[:reading.n])
            np.testing.assert_array_equal(reading.half_spread, il.SPX["half_spread"][:reading.n])

    def test_the_generating_parameters_are_consistent_with_the_quotes(self):
        reading = il.readings()["325 days"]
        misses = reading.sum_of_squares(il.TRUTH) / il.C_TRUE**2
        self.assertLess(abs(misses - 63) / np.sqrt(2 * 63), 3.0)


class PricingTests(unittest.TestCase):
    def test_the_control_variate_gives_the_runtimes_fourier_price(self):
        parameters, v0 = il.heston_parameters(il.TRUTH)
        strikes, maturity = np.array([0.9, 1.0, 1.1]), 0.5
        plain = heston.european_price(1.0, strikes, 1.0, maturity, parameters, v0, kind="call", nodes=4096,
                                      upper_limit=2400.0)
        np.testing.assert_allclose(il.prices(il.TRUTH, strikes, maturity, False), plain, atol=1e-10)

    def test_the_quadrature_passes_the_applications_gate_at_the_generating_parameters(self):
        self.assertLess(il.gate(il.TRUTH), il.GATE)

    def test_the_forward_variance_is_the_average_expected_variance(self):
        v0, kappa, long_run = il.TRUTH[:3]
        t1, t2 = il.LATER_WINDOW
        expected = quad(lambda t: long_run + (v0 - long_run) * np.exp(-kappa * t), t1, t2)[0] / (t2 - t1)
        self.assertAlmostEqual(il.forward_variance(il.TRUTH) ** 2, expected, places=12)


class ForwardStartTests(unittest.TestCase):
    def test_a_start_date_of_zero_gives_the_vanilla_call(self):
        strikes = (0.9, 1.0, 1.1)
        vanilla = [il.call(il.TRUTH, strike, 0.5) for strike in strikes]
        np.testing.assert_allclose(il.forward_start_calls(il.TRUTH, strikes, start=1e-10, end=0.5), vanilla, atol=1e-8)

    def test_without_volatility_of_volatility_the_price_is_black_scholes(self):
        theta = il.TRUTH.copy()
        theta[3] = 1e-4
        v0, kappa, long_run = theta[:3]
        start, end = il.START, il.END
        variance = long_run * (end - start) + (v0 - long_run) * (np.exp(-kappa * start) - np.exp(-kappa * end)) / kappa
        black_scholes = float(surface.bs_call(0.0, 1.0, np.sqrt(variance)))  # at the money, total variance `variance`
        self.assertAlmostEqual(il.forward_start_call(theta), black_scholes, places=5)

    def test_forward_start_calls_fall_with_the_strike(self):
        values = il.forward_start_calls(il.TRUTH, (0.9, 1.0, 1.1))
        self.assertTrue(np.all(np.diff(values) < 0))
        smile = il.forward_smile(il.TRUTH, (0.9, 1.0, 1.1))
        self.assertTrue(np.all(np.isfinite(smile)) and smile[0] > smile[2])  # the forward smile keeps a skew


class ErrorScaleTests(unittest.TestCase):
    def test_integrating_c_out_matches_quadrature_up_to_a_constant(self):
        n = 18
        values = (5.0, 18.0, 60.0, 400.0)
        numeric = [np.log(quad(lambda c: c ** (-n) * np.exp(-r2 / (2 * c * c)), il.C_LOWER, il.C_UPPER)[0])
                   for r2 in values]
        closed = il._log_c_integral(np.array(values), n)
        np.testing.assert_allclose(closed - closed[0], np.subtract(numeric, numeric[0]), atol=1e-8)

    def test_c_is_drawn_from_its_conditional(self):
        n, r2 = 18, 30.0
        draws = il.draw_c(np.full(40000, r2), n, np.random.default_rng(0))
        self.assertTrue(np.all((draws >= il.C_LOWER) & (draws <= il.C_UPPER)))
        density = lambda c: c ** (-n) * np.exp(-r2 / (2 * c * c))  # noqa: E731
        mean = quad(lambda c: c * density(c), il.C_LOWER, il.C_UPPER)[0] / quad(density, il.C_LOWER, il.C_UPPER)[0]
        self.assertAlmostEqual(draws.mean(), mean, delta=0.005)


class RangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readings = il.readings()

    def test_least_squares_with_every_quote_brackets_the_true_price(self):
        result = il.fit(self.readings["325 days"], starts=3)
        self.assertEqual(result.on_bound, ())
        self.assertTrue(0.7 < result.c < 1.3)
        point, interval = il.delta_range(result, il.forward_start_call)
        truth = il.forward_start_call(il.TRUTH)
        self.assertTrue(interval[0] < truth < interval[1])
        self.assertLess(interval[1] - interval[0], 0.001)  # under 0.1 per 100 of forward
        self.assertTrue(interval[0] < point < interval[1])

    def test_a_fixed_parameter_keeps_its_value_and_moves_the_range(self):
        reading = self.readings["14 days"]
        free = il.fit(reading, starts=3)
        fixed = il.fit(reading, fixed={1: 2.0}, starts=3)
        self.assertEqual(fixed.theta[1], 2.0)
        self.assertGreaterEqual(fixed.sum_of_squares, free.sum_of_squares - 1e-6)
        centre = lambda result: np.mean(il.delta_range(result, il.forward_start_call)[1])  # noqa: E731
        self.assertGreater(abs(centre(fixed) - centre(free)), 0.001)  # more than 0.1 per 100 of forward

    def test_importance_sampling_returns_draws_inside_the_prior(self):
        posterior = il.posterior(self.readings["30 days"], adapt=(300, 300), batch=400, target=100.0, most=1200,
                                 keep=200, rejuvenate_below=0.0)
        draws = posterior.draws
        self.assertEqual(draws.shape, (200, 6))
        self.assertTrue(np.all((draws[:, :5] >= il.LOWER) & (draws[:, :5] <= il.UPPER)))
        self.assertTrue(np.all((draws[:, 5] >= il.C_LOWER) & (draws[:, 5] <= il.C_UPPER)))
        self.assertGreater(posterior.effective_sample_size, 30.0)
        self.assertLess(posterior.failed, 0.01)
        centre = np.abs(draws[:, :5].mean(axis=0) - il.TRUTH) / (il.UPPER - il.LOWER)
        self.assertLess(centre.max(), 0.1)

    def test_draws_within_a_smaller_box(self):
        draws = np.array([[0.027, 4.3, 0.056, 1.8, -0.67, 1.0], [0.027, 4.3, 0.15, 1.8, -0.67, 1.0]])
        upper = il.UPPER.copy()
        upper[2] = 0.09
        np.testing.assert_array_equal(il.within(draws, upper=upper), draws[:1])


if __name__ == "__main__":
    unittest.main()

"""The running example of the slot 1 slides: its pricing formulas, its designs and its two routes to a range."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

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


class ForwardStartTests(unittest.TestCase):
    def test_a_start_date_of_zero_gives_the_vanilla_call(self):
        parameters, v0 = il.heston_parameters(il.TRUTH)
        vanilla = float(heston.european_price(1.0, 1.0, 1.0, 1.5, parameters, v0, kind="call",
                                              nodes=il.NODES, upper_limit=il.CUTOFF))
        self.assertAlmostEqual(il.forward_start_call(il.TRUTH, start=1e-10, end=1.5), vanilla, places=8)

    def test_without_volatility_of_volatility_the_price_is_black_scholes(self):
        theta = il.TRUTH.copy()
        theta[3] = 1e-4
        v0, long_run, kappa = theta[:3]
        start, end = 1.0, 2.0
        variance = long_run * (end - start) + (v0 - long_run) * (np.exp(-kappa * start) - np.exp(-kappa * end)) / kappa
        black_scholes = float(surface.bs_call(0.0, 1.0, np.sqrt(variance)))  # at the money, total variance `variance`
        self.assertAlmostEqual(il.forward_start_call(theta, start, end), black_scholes, places=5)

    def test_a_higher_variance_gives_a_higher_price(self):
        low = il.forward_start_call(np.array([0.02, 0.02, 2.0, 0.5, -0.6]))
        high = il.forward_start_call(np.array([0.08, 0.08, 2.0, 0.5, -0.6]))
        self.assertLess(low, high)
        self.assertGreater(low, 0.0)
        self.assertLess(high, 1.0)


class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.designs = il.designs()

    def test_designs_are_nested_readings_of_one_surface(self):
        sizes = [len(self.designs[reach].contracts) for reach in il.REACH]
        self.assertEqual(sizes, [7, 14, 21, 28, 35])
        full = self.designs["2 years"]
        for reach in il.REACH:
            design = self.designs[reach]
            n = len(design.contracts)
            np.testing.assert_array_equal(design.observed, full.observed[:n])
            np.testing.assert_array_equal(design.noise, full.noise[:n])

    def test_the_error_band_is_the_tutorial_rule_in_volatility(self):
        design = self.designs["2 years"]
        delta = surface.standardized_moneyness(design.k, design.maturity)
        clean = il.surface_prices(il.TRUTH, design.contracts)
        implied = surface.implied_vol(clean, design.k, design.maturity, design.kind)
        band = design.noise / surface.bs_vega(design.k, design.maturity, implied)
        np.testing.assert_allclose(band, 0.003 + 0.0035 * np.abs(delta), rtol=1e-9)
        self.assertGreater(np.min(clean / design.noise), 3.0)  # every price is several error bands above zero

    def test_the_generating_parameters_are_consistent_with_the_quotes(self):
        design = self.designs["2 years"]
        chi2 = -2.0 * design.log_likelihood(il.TRUTH)
        self.assertLess(abs(chi2 - 35) / np.sqrt(2 * 35), 3.0)


class RangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.designs = il.designs()

    def test_least_squares_with_many_quotes_brackets_the_true_price(self):
        result = il.fit(self.designs["2 years"], starts=3)
        self.assertEqual(result.on_bound, ())
        point, interval = il.delta_range(result, il.put_price)
        self.assertLess(interval[0], il.put_price(il.TRUTH))
        self.assertGreater(interval[1], il.put_price(il.TRUTH))
        self.assertLess(interval[1] - interval[0], 0.01)  # under 1 per 100 of forward
        self.assertTrue(interval[0] < point < interval[1])

    def test_a_fixed_parameter_keeps_its_value_and_narrows_the_range(self):
        design = self.designs["3 months"]
        free = il.fit(design, starts=3)
        fixed = il.fit(design, fixed={2: 2.0}, starts=3)
        self.assertEqual(fixed.theta[2], 2.0)
        self.assertGreaterEqual(fixed.chi2, free.chi2 - 1e-6)
        width = lambda result: np.subtract(*il.delta_range(result, il.put_price)[1][::-1])  # noqa: E731
        self.assertLess(width(fixed), width(free))

    def test_importance_sampling_returns_draws_inside_the_prior_box(self):
        posterior = il.posterior(self.designs["2 years"], adapt=(300, 300), batch=400, target=100.0, most=1200,
                                 keep=200)
        self.assertEqual(posterior.draws.shape, (200, 5))
        self.assertTrue(np.all((posterior.draws >= il.LOWER) & (posterior.draws <= il.UPPER)))
        self.assertGreater(posterior.effective_sample_size, 30.0)
        self.assertLess(posterior.failed, 0.01)
        centre = np.abs(posterior.draws.mean(axis=0) - il.TRUTH) / (il.UPPER - il.LOWER)
        self.assertLess(centre.max(), 0.1)

    def test_draws_within_a_smaller_box(self):
        draws = np.array([[0.02, 0.05, 2.0, 0.9, -0.65], [0.02, 0.15, 2.0, 0.9, -0.65]])
        upper = il.UPPER.copy()
        upper[1] = 0.09
        np.testing.assert_array_equal(il.within(draws, upper=upper), draws[:1])


if __name__ == "__main__":
    unittest.main()

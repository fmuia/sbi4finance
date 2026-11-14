"""Price displays preserve units, quote order and Gaussian predictive tails."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from sbi4f_tutorial.shared import surface


HELPER = Path(__file__).resolve().parents[1] / "notebooks/_option_display_units.py"
SPEC = importlib.util.spec_from_file_location("option_display_units_under_test", HELPER)
UNITS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UNITS)


class OptionDisplayUnitsTests(unittest.TestCase):
    def setUp(self):
        self.forward = np.array([5300., 5400., 5600.])
        self.strike = self.forward * np.exp([-.08, 0., .12])
        self.maturity = np.array([.1, .4, 1.])
        self.discount = np.array([.997, .988, .97])
        self.kind = np.array(["put", "call", "call"])
        self.iv = np.array([.25, .2, .3])
        self.prices = self.discount * self.forward * surface.bs_price(
            np.log(self.strike / self.forward), self.maturity, self.iv, self.kind)
        self.vega = UNITS.black_vega(self.forward, self.strike, self.maturity, self.discount, self.iv)

    def exact_points(self, prices, **kwargs):
        return UNITS.implied_volatility_points(prices, self.forward, self.strike, self.maturity,
                                               self.discount, self.kind, **kwargs)

    def test_exact_black_roundtrip_is_in_percentage_points(self):
        points, valid = self.exact_points(self.prices)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, 100 * self.iv, atol=1e-8)
        rows = np.broadcast_to(self.prices, (2, 3))
        points, valid = self.exact_points(rows)
        self.assertEqual(points.shape, rows.shape)
        self.assertEqual(valid.shape, rows.shape)

    def test_price_vega_uses_each_quotes_own_forward_discount_and_iv(self):
        epsilon = 1e-6
        bump = self.discount * self.forward * surface.bs_price(
            np.log(self.strike / self.forward), self.maturity, self.iv + epsilon, self.kind)
        np.testing.assert_allclose((bump - self.prices) / epsilon, self.vega, rtol=8e-6)
        residual = UNITS.volatility_point_residuals(bump, self.prices, self.vega)
        np.testing.assert_allclose(residual, np.full(3, 100 * epsilon), rtol=8e-6)

    def test_affine_quantile_transform_preserves_paired_predictive_inclusion(self):
        rng = np.random.default_rng(52)
        clean = self.prices + rng.normal(size=(1024, 3)) * [.2, 1., 3.]
        c = rng.uniform(.5, 50., size=(1024, 1))
        noisy = clean + c * [.7, .2, 1.] * rng.standard_normal(clean.shape)
        raw_quantiles = np.quantile(noisy, [.05, .5, .95], axis=0)
        display = UNITS.volatility_point_residuals(noisy, self.prices, self.vega)
        quantiles = UNITS.volatility_point_residuals(raw_quantiles, self.prices, self.vega)
        np.testing.assert_allclose(np.quantile(display, [.05, .5, .95], axis=0), quantiles)
        raw_contains = (raw_quantiles[0] <= self.prices) & (self.prices <= raw_quantiles[2])
        display_contains = (quantiles[0] <= 0) & (0 <= quantiles[2])
        np.testing.assert_array_equal(raw_contains, display_contains)
        self.assertEqual(len(display), len(noisy))

    def test_negative_gaussian_prices_remain_in_equivalent_displays(self):
        prices = np.array([[-10., 0., 10000.], self.prices])
        residual = UNITS.volatility_point_residuals(prices, self.prices, self.vega)
        np.testing.assert_allclose(residual, 100 * (prices - self.prices) / self.vega)
        levels = UNITS.equivalent_volatility_points(prices, self.prices, self.iv, self.vega)
        np.testing.assert_allclose(levels[1], 100 * self.iv)
        np.testing.assert_allclose(levels - 100 * self.iv, residual)
        self.assertEqual(levels.shape, prices.shape)

    def test_exact_inverse_reports_invalid_prices_and_above_bracket_values(self):
        normalized = np.where(self.kind == "put", np.exp(np.log(self.strike / self.forward)), 1.)
        cap = normalized * self.discount * self.forward
        invalid = np.stack([np.full(3, -1.), cap * 1.01, np.full(3, np.nan)])
        points, valid = self.exact_points(invalid)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(points).all())
        above = self.discount * self.forward * surface.bs_price(
            np.log(self.strike / self.forward), self.maturity, 1., self.kind)
        points, valid = self.exact_points(above, upper=.5)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(points).all())
        beyond_default = self.discount * self.forward * surface.bs_price(
            np.log(self.strike / self.forward), self.maturity, 6., self.kind)
        points, valid = self.exact_points(beyond_default)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(points).all())
        at_default = self.discount * self.forward * surface.bs_price(
            np.log(self.strike / self.forward), self.maturity, 5., self.kind)
        points, valid = self.exact_points(at_default)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, 500., atol=1e-8)

    def test_sd_conversion_and_nonfinite_values_do_not_filter_rows(self):
        sd = np.array([[.1, .2, .3], [.4, .5, .6]])
        np.testing.assert_allclose(UNITS.volatility_point_sd(sd, self.vega), 100 * sd / self.vega)
        np.testing.assert_allclose(UNITS.volatility_point_sd(.01 * self.vega, self.vega), 1.)
        data = np.array([[np.nan, 2., 3.], [1., np.inf, 5.]])
        transformed = UNITS.volatility_point_residuals(data, self.prices, self.vega)
        self.assertEqual(transformed.shape, data.shape)
        np.testing.assert_array_equal(np.isnan(transformed), np.isnan(data))
        np.testing.assert_array_equal(np.isinf(transformed), np.isinf(data))

    def test_unconverged_inverse_is_invalid_without_changing_shape(self):
        prices = np.broadcast_to(self.prices, (2, 3))
        points, valid = self.exact_points(prices, max_iter=1)
        self.assertEqual(points.shape, prices.shape)
        self.assertEqual(valid.shape, prices.shape)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(points).all())
        points, valid = self.exact_points(prices)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, np.broadcast_to(100 * self.iv, prices.shape), atol=1e-8)

    def test_forward_start_uses_an_explicit_black_equivalent_benchmark(self):
        reset, expiry = .25, .75
        volatility = np.array([[.15, .2], [.3, .4]])
        prices_percent = 100 * surface.bs_call(0., expiry - reset, volatility)
        points, valid = UNITS.forward_start_volatility_points(prices_percent, reset, expiry)
        self.assertEqual(points.shape, volatility.shape)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, 100 * volatility, atol=1e-8)
        points, valid = UNITS.forward_start_volatility_points([-1., 110.], reset, expiry)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(points).all())
        with self.assertRaises(ValueError):
            UNITS.forward_start_volatility_points(prices_percent, expiry, reset)

    def test_invalid_geometry_vega_and_contract_order_shapes_fail(self):
        for bad in ([0., 1., 2.], [-1., 1., 2.], [np.nan, 1., 2.]):
            with self.assertRaises(ValueError):
                UNITS.volatility_point_residuals(self.prices, self.prices, bad)
        with self.assertRaises(ValueError):
            UNITS.volatility_point_residuals(self.prices[:2], self.prices, self.vega)
        with self.assertRaises(ValueError):
            UNITS.volatility_point_sd([-1., .2, .3], self.vega)
        with self.assertRaises(ValueError):
            UNITS.black_vega(self.forward, self.strike, [-1., .2, .3], self.discount, self.iv)
        with self.assertRaises(ValueError):
            self.exact_points(self.prices, lower=.5, upper=.2)


if __name__ == "__main__":
    unittest.main()

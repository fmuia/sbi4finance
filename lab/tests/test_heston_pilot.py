"""Scientific contracts of the raw-quote Heston observation pilot."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from sbi4finance.calibration.heston_pilot import HestonPilot, black_price


def records():
    f = np.array([100., 102., 105., 105.])
    k = f * np.array([.9, 1., 1.05, .95])
    d = np.array([.99, .98, .96, .96])
    t = np.array([.1, .25, .75, .75])
    iv = np.array([.25, .22, .18, .23])
    kind = np.array(["put", "call", "call", "put"])
    return dict(forward=f, strike=k, discount=d, maturity=t, option_type=kind,
                mid=black_price(f, k, d, t, iv, kind), half_spread=np.array([.03, .02, .04, .03]))


THETA = np.array([.04, 2., .05, .4, -.7])


class HestonPilotTests(unittest.TestCase):
    def test_midpoint_vega_uses_observed_iv_and_frozen_inputs(self):
        original = records()
        pilot = HestonPilot(original)
        assert_allclose(pilot.market_iv, [.25, .22, .18, .23], atol=1e-11)
        original["mid"][:] = 0
        self.assertTrue(np.all(pilot.records["mid"] > 0))
        self.assertFalse(pilot.market_vega.flags.writeable)
        d1 = np.log(pilot.records["forward"] / pilot.records["strike"]) / (pilot.market_iv * np.sqrt(pilot.records["maturity"])) + pilot.market_iv * np.sqrt(pilot.records["maturity"]) / 2
        expected = pilot.records["discount"] * pilot.records["forward"] * np.sqrt(pilot.records["maturity"]) * np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
        assert_allclose(pilot.market_vega, expected, rtol=1e-13)

    def test_gaussian_normalization_and_iv_discrepancy_are_in_quote_units(self):
        pilot = HestonPilot(records(), infer_discrepancy=True, spread_multiplier=.5)
        theta = np.r_[THETA, .003]
        prediction = pilot.records["mid"] + np.array([.01, -.02, .03, -.04])
        expected_sd = np.hypot(.5 * records()["half_spread"], .003 * pilot.market_vega)
        with patch.object(pilot, "prices", return_value=prediction):
            expected = -.5 * np.sum(((pilot.records["mid"] - prediction) / expected_sd)**2 + np.log(2 * np.pi * expected_sd**2))
            self.assertAlmostEqual(pilot.log_likelihood(theta), expected, places=13)
        assert_allclose(pilot.observation_sigma(theta), expected_sd)
        self.assertEqual(pilot.names[-1], "tau")
        assert_allclose(pilot.bounds[-1], [0, .02])

    def test_raw_contract_pricing_preserves_selection_order_and_matches_adaptive_quad(self):
        pilot = HestonPilot(records())
        all_prices = pilot.prices(THETA)
        assert_allclose(pilot.prices(THETA, [3, 0]), all_prices[[3, 0]], atol=1e-10)
        assert_allclose(pilot.prices(THETA, np.array([True, False, False, True])), all_prices[[0, 3]], atol=1e-10)
        independent, errors = pilot.independent_prices(THETA, [0, 2])
        assert_allclose(independent, all_prices[[0, 2]], atol=1e-5)
        self.assertTrue(np.all(errors < .02 * pilot.measurement_sigma[[0, 2]]))
        check = pilot.refinement_report(THETA)
        self.assertLess(check["max_abs_price_shift_over_measurement_sigma"], .02)

    def test_discrepancy_does_not_loosen_numerical_gate_and_failures_propagate(self):
        pilot = HestonPilot(records(), infer_discrepancy=True,
                            quadrature_ladder=((16, 20.), (32, 40.)))
        with patch.object(pilot, "_rung_prices", side_effect=[np.ones(4), np.ones(4) + .01]):
            with self.assertRaisesRegex(FloatingPointError, "refinement gate"):
                pilot.prices(np.r_[THETA, .02])
        with patch.object(pilot, "prices", side_effect=FloatingPointError("pricing failed")):
            with self.assertRaisesRegex(FloatingPointError, "pricing failed"):
                pilot.log_likelihood(np.r_[THETA, .02])
        outside = np.r_[THETA, .021]
        self.assertEqual(pilot.log_likelihood(outside), -np.inf)

    def test_invalid_spreads_masks_and_market_bounds_fail(self):
        for key, value in (("half_spread", 0.), ("mid", 0.), ("forward", np.nan)):
            bad = records()
            bad[key][0] = value
            with self.assertRaises(ValueError):
                HestonPilot(bad)
        pilot = HestonPilot(records())
        for mask in ([0, 0], [], [4], [True, False], [0., 1.]):
            with self.assertRaises(ValueError):
                pilot.prices(THETA, mask)
        with self.assertRaises(ValueError):
            HestonPilot(records(), quadrature_ladder=((16, 20.), (16, 40.)))
        with self.assertRaises(ValueError):
            HestonPilot(records(), bounds=((.01, .1),) * 4 + ((-1.1, .1),))


if __name__ == "__main__":
    unittest.main()

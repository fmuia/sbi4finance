"""The runtime's computations on small inputs: prices, implied volatilities, the long end, the network."""

import json
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sbi4f_tutorial import heston, inference, long_end, surface, tasks, teaching
from sbi4f_tutorial.paths import CHECKPOINTS, RESULTS


class PricingTests(unittest.TestCase):
    def test_black_scholes_parity_and_implied_volatility(self):
        k, maturity, kind = surface.contract_arrays(surface.design_grid())
        prices = surface.bs_price(k, maturity, 0.23, kind)
        calls, puts = surface.bs_call(k, maturity, 0.23), surface.bs_put(k, maturity, 0.23)
        assert_allclose(calls - puts, 1.0 - np.exp(k), atol=1e-14)
        assert_allclose(surface.implied_vol(prices, k, maturity, kind), 0.23, atol=1e-9)

    def test_heston_without_vol_of_vol_is_black_scholes_with_the_mean_variance(self):
        theta, v0, kappa, t = 0.04, 0.09, 2.0, 0.5
        w = theta * t + (v0 - theta) * (1 - np.exp(-kappa * t)) / kappa
        price = heston.european_price(1.0, 1.1, 1.0, t, kappa, theta, 0.0, -0.5, v0, nodes=2048, upper_limit=400.0)
        assert_allclose(price, surface.bs_call(np.log(1.1), t, np.sqrt(w / t)), atol=1e-9)

    def test_heston_smile_of_the_checkpoint_task_lies_inside_the_bounds(self):
        task = json.loads((CHECKPOINTS / "heston" / "task.json").read_text())
        prices = teaching.heston_smile(task, [0.04, 0.05, 0.8, -0.7])
        k, maturity, kind = surface.contract_arrays(task["context"]["contracts"])
        self.assertTrue(np.isfinite(surface.implied_vol(prices, k, maturity, kind)).all())

    def test_one_rough_smile_has_a_negative_skew_falling_with_maturity(self):
        result = teaching.price_rough_smile(0.1, 2.0, -0.8, paths=1024)
        self.assertTrue(np.isfinite(result["implied"]).all())
        self.assertTrue(np.all(np.asarray(result["skew"]) < 0))
        self.assertLess(result["exponent"], 0)

    def test_rough_design_matches_the_design_grid(self):
        design = tasks.rough_design()
        self.assertEqual(len(design["contracts"]), 63)
        assert_allclose(design["knots"][1:] * surface.TRADING_DAYS, surface.DESIGN_DAYS)
        self.assertEqual([b["n_steps"] for b in design["blocks"]], [80, 504, 504])


class LongEndAndNetworkTests(unittest.TestCase):
    def test_long_end_posterior_is_a_covariance_and_ranges_are_positive(self):
        report = json.loads((long_end.QUICK_CHECKS / "long_end" / "long_end.json").read_text())
        mean, cov = long_end.posterior(np.log(np.asarray(report["xi_flat"])))
        self.assertEqual(cov.shape, (14, 14))
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))
        desk, post, shares, _ = long_end.ranges(report, long_end.S_PRIOR, long_end.VS_SD)
        self.assertTrue(np.all(desk >= 0) and np.all(post > 0))  # the variance swap's desk range is zero: it
        self.assertEqual(float(desk[-1]), 0.0)  # depends on the curve alone, which the desk holds fixed

    def test_the_heston_network_loads_and_samples_inside_its_prior(self):
        task, workflow = teaching.load_network(CHECKPOINTS / "heston")
        spy = json.loads((RESULTS / "spy" / "surface.json").read_text())
        draws = inference.sample(workflow, task, np.asarray(spy["prices_forward_units"])[None], None, 200, seed=0)
        self.assertEqual(draws.shape, (1, 200, 4))
        bounds = np.asarray(task["prior_bounds"])
        self.assertTrue(np.all((draws > bounds[:, 0]) & (draws < bounds[:, 1])))


if __name__ == "__main__":
    unittest.main()

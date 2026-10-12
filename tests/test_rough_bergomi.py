"""Independent Gaussian reference, hybrid approximation and rough-Q limits."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from sbi4finance.bergomi import option_panel
from sbi4finance.rough_bergomi import (hybrid_covariance, hybrid_weights,
                                     rough_covariance, rough_driver, simulate_rough)


CURVE = {"times": [0., 1.], "values": [.04, .04]}
PARAMETERS = {"hurst": .1, "eta": 1., "rho": -.7}


class RoughBergomiTests(unittest.TestCase):
    def test_covariance_reference_has_correct_gaussian_and_asset_dependence(self):
        n, steps, dt = 90000, 8, .125
        z = np.random.default_rng(211).standard_normal((n, steps, 3))
        c, cross = rough_covariance(.1, dt, steps)
        driver, variance = rough_driver(.1, dt, z, method="covariance")
        observed = driver.T @ driver / n
        covariance_se = np.sqrt((np.outer(c.diagonal(), c.diagonal()) + c*c)/n)
        self.assertLess(np.max(np.abs(observed-c)/covariance_se), 5.)
        observed_cross = driver.T @ (np.sqrt(dt)*z[..., 0])/n
        cross_se = np.sqrt((c.diagonal()[:, None]*dt+cross*cross)/n)
        self.assertLess(np.max(np.abs(observed_cross-cross)/cross_se), 5.)
        assert_allclose(variance, (dt*np.arange(1, steps+1))**.2)
        self.assertTrue(np.all(np.triu(cross, 1) == 0))

    def test_hybrid_fft_matches_causal_sum_and_joint_covariance(self):
        steps, dt, h = 9, .05, .08
        z = np.random.default_rng(33).standard_normal((5, steps, 3))
        y, variance = rough_driver(h, dt, z)
        weights, bridge = hybrid_weights(h, dt, steps)
        direct = np.array([np.convolve(row[:, 0], weights)[:steps]+bridge*row[:, 1] for row in z])
        assert_allclose(y, direct, atol=1e-14)
        cov, cross = hybrid_covariance(h, dt, steps)
        exact, exact_cross = rough_covariance(h, dt, steps)
        assert_allclose(variance, cov.diagonal(), atol=1e-15)
        assert_allclose(cross, exact_cross, atol=2e-15)
        # The kernel approximation is visible, not disguised as an exact field.
        self.assertGreater(np.max(np.abs(cov-exact)), 1e-4)
        self.assertLess(np.max(np.abs(cov-exact)/np.sqrt(np.outer(exact.diagonal(), exact.diagonal()))), .02)

    def test_near_brownian_limit_and_normalized_variance(self):
        z = np.random.default_rng(773).standard_normal((90000, 8, 3))
        y, variance = rough_driver(.499999, .125, z[:10])
        assert_allclose(y, np.cumsum(np.sqrt(.125)*z[:10, :, 0], axis=1), atol=1e-5)
        for method in ("hybrid", "covariance"):
            result = simulate_rough(PARAMETERS, CURVE, .125, z, method=method)
            q = result["driver_variance"][1:]
            se = .04*np.sqrt(np.expm1(q)/len(z))
            self.assertLess(np.max(abs(result["variance"][:, 1:].mean(axis=0)-.04)/se), 5.)
            self.assertFalse(result["continuation_state_available"])

    def test_zero_eta_and_parity_use_same_bounded_estimator(self):
        contracts = [{"strike": 1.1, "maturity": .5, "option_type": k} for k in ("put", "call")]
        a = option_panel("rough", dict(PARAMETERS, eta=0., rho=0.), CURVE, contracts,
                         n_paths=64, n_steps=16, seed=3)
        b = option_panel("one_factor", {"eta": 0., "rho": 0., "kappa": 2.}, CURVE, contracts,
                         n_paths=64, n_steps=16, seed=3)
        assert_allclose(a["prices"], b["prices"], atol=1e-15)
        assert_allclose(a["prices"][1]-a["prices"][0], -.1, atol=1e-15)
        self.assertLess(a["standard_errors"].max(), 1e-15)

    def test_covariance_and_hybrid_prices_agree_at_bounded_fixture_budget(self):
        contracts = [{"strike": k, "maturity": .5} for k in (.9, 1., 1.1)]
        a = option_panel("rough", PARAMETERS, CURVE, contracts, n_paths=8192, n_steps=32, seed=309)
        b = option_panel("rough", PARAMETERS, CURVE, contracts, n_paths=8192, n_steps=32, seed=819, method="covariance")
        budget = .002 + 4*np.hypot(a["standard_errors"], b["standard_errors"])
        self.assertTrue(np.all(abs(a["prices"]-b["prices"]) < budget))

    def test_rejects_nonrough_domain_positive_leverage_and_large_reference(self):
        z = np.zeros((2, 4, 3))
        for changes in ({"hurst": 0.}, {"hurst": .5}, {"rho": .1}):
            with self.assertRaises(ValueError):
                simulate_rough(dict(PARAMETERS, **changes), CURVE, .1, z)
        with self.assertRaises(ValueError):
            rough_covariance(.1, .01, 129)
        with self.assertRaises(ValueError):
            simulate_rough(PARAMETERS, CURVE, .1, z, method="euler")


if __name__ == "__main__":
    unittest.main()

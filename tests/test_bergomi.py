"""Scientific one-factor, continuation and bounded-pricing checks."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import ndtr

from sbi4finance.bergomi import (conditional_put_payoffs, curve_values,
                               option_panel, simulate_one_factor)


CURVE = {"times": [0., .5, 1.], "values": [.035, .045, .05]}
PARAMETERS = {"kappa": 2., "eta": 1.1, "rho": -.65}


class BergomiTests(unittest.TestCase):
    def test_exact_ou_transition_and_brownian_cross_covariance(self):
        n, dt = 160000, .3
        z = np.random.default_rng(713).standard_normal((n, 1, 3))
        result = simulate_one_factor(PARAMETERS, CURVE, [dt], z)
        x, dw = result["factor_state"][:, 1, 0], np.sqrt(dt) * z[:, 0, 0]
        k = PARAMETERS["kappa"]
        variance = -np.expm1(-2 * k * dt) / (2 * k)
        covariance = -np.expm1(-k * dt) / k
        self.assertLess(abs(x.mean()), 5 * np.sqrt(variance / n))
        self.assertLess(abs(np.mean(x*x) - variance), 5 * np.sqrt(2 / n) * variance)
        self.assertLess(abs(np.mean(x*dw) - covariance), 5 * np.sqrt((variance*dt + covariance**2)/n))
        # Exact Gaussian lognormal normalization; no empirical renormalization.
        expected = curve_values(CURVE, [dt])[0]
        se = expected * np.sqrt(np.expm1(PARAMETERS["eta"] ** 2 * variance) / n)
        self.assertLess(abs(result["variance"][:, 1].mean() - expected), 5 * se)

    def test_complete_state_and_elapsed_curve_preserve_continuation(self):
        dt = np.array([.1, .2, .05, .15])
        z = np.random.default_rng(51).standard_normal((17, 4, 3))
        whole = simulate_one_factor(PARAMETERS, CURVE, dt, z)
        first = simulate_one_factor(PARAMETERS, CURVE, dt[:2], z[:, :2])
        rest = simulate_one_factor(PARAMETERS, CURVE, dt[2:], z[:, 2:],
                                  factor_state=first["factor_state"][:, -1, 0],
                                  elapsed_time=first["elapsed_time"])
        assert_allclose(whole["log_returns"][:, 2:], rest["log_returns"], atol=2e-15)
        assert_allclose(whole["factor_state"][:, 2:], rest["factor_state"], atol=2e-15)
        assert_allclose(whole["variance"][:, 2:], rest["variance"], atol=2e-15)
        self.assertEqual(rest["elapsed_time"], .5)

    def test_zero_eta_returns_and_integrated_curve_are_exact_discrete_gbm(self):
        dt = np.array([.1, .2, .2])
        z = np.random.default_rng(81).standard_normal((5, 3, 3))
        p = dict(PARAMETERS, eta=0.)
        result = simulate_one_factor(p, CURVE, dt, z, rate=.03, dividend=.01)
        variance = curve_values(CURVE, [0., .1, .3])
        expected = (.02-.5*variance)*dt + np.sqrt(variance*dt)*(p["rho"]*z[..., 0]+np.sqrt(1-p["rho"]**2)*z[..., 2])
        assert_allclose(result["log_returns"][..., 0], expected, atol=1e-15)
        assert_allclose(result["integrated_variance"], np.tile(np.cumsum(variance*dt), (5, 1)))

    def test_bounded_conditional_estimator_matches_black_scholes_and_parity(self):
        curve = {"times": [0., 1.], "values": [.04, .04]}
        contracts = [{"strike": 1.05, "maturity": .5, "option_type": kind} for kind in ("put", "call")]
        r, q, t, k = .03, .01, .5, 1.05
        result = option_panel("one_factor", {"kappa": 2., "eta": 0., "rho": 0.}, curve, contracts,
                              rate=r, dividend=q, n_paths=16, n_steps=16, seed=7)
        d1 = (np.log(1/k)+(r-q+.02)*t)/np.sqrt(.04*t)
        d2 = d1 - np.sqrt(.04*t)
        put = k*np.exp(-r*t)*ndtr(-d2)-np.exp(-q*t)*ndtr(-d1)
        assert_allclose(result["prices"], [put, put+np.exp(-q*t)-k*np.exp(-r*t)], atol=2e-15)
        self.assertLess(result["standard_errors"].max(), 1e-15)

    def test_conditional_put_bounds_panel_covariance_and_chunk_invariance(self):
        contracts = [{"strike": 1., "maturity": .5},
                     {"strike": 1., "maturity": .5, "option_type": "call"},
                     {"strike": 1.1, "maturity": 1.}]
        z = np.random.default_rng(49).standard_normal((100, 16, 3))
        paths = simulate_one_factor(PARAMETERS, CURVE, np.full(16, 1/16), z)
        payoffs = conditional_put_payoffs(paths, PARAMETERS, contracts, [8, 8, 16], spot=1., rate=0., dividend=0.)
        self.assertTrue(np.all(payoffs >= 0))
        self.assertTrue(np.all(payoffs <= np.array([1., 1., 1.1])))
        a = option_panel("one_factor", PARAMETERS, CURVE, contracts, n_paths=2000, n_steps=16, seed=49, chunk_size=137)
        b = option_panel("one_factor", PARAMETERS, CURVE, contracts, n_paths=2000, n_steps=16, seed=49, chunk_size=1024)
        assert_allclose(a["prices"], b["prices"], atol=1e-14)
        assert_allclose(a["estimator_covariance"], b["estimator_covariance"], atol=1e-18)
        assert_allclose(a["estimator_covariance"][0], a["estimator_covariance"][1], atol=1e-18)
        self.assertGreater(a["estimator_covariance"][0, 2], 0)
        self.assertGreaterEqual(np.linalg.eigvalsh(a["estimator_covariance"]).min(), -1e-18)
        self.assertFalse(a["metadata"]["mc_error_includes_discretization_bias"])

    def test_perfect_negative_leverage_has_no_independent_asset_noise(self):
        z = np.random.default_rng(12).standard_normal((500, 8, 3))
        p = dict(PARAMETERS, rho=-1.)
        paths = simulate_one_factor(p, CURVE, np.full(8, .5/8), z, rate=.02)
        contracts = [{"strike": 1.1, "maturity": .5}]
        conditional = conditional_put_payoffs(paths, p, contracts, [8], spot=1., rate=.02, dividend=0.)[:, 0]
        exact = np.exp(-.01) * np.maximum(1.1-np.exp(paths["log_spot"][:, -1]), 0)
        assert_allclose(conditional, exact, atol=1e-15)

    def test_rejects_unsupported_measure_domain_context_and_restart(self):
        z = np.zeros((2, 4, 3))
        for changes in ({"rho": .1}, {"kappa": 0.}, {"eta": -1.}):
            with self.assertRaises(ValueError):
                simulate_one_factor(dict(PARAMETERS, **changes), CURVE, [.1]*4, z)
        with self.assertRaises(ValueError):
            simulate_one_factor(PARAMETERS, CURVE, [.1]*4, z, factor_state=.2)
        with self.assertRaises(ValueError):
            curve_values(CURVE, [1.1])
        with self.assertRaises(ValueError):
            option_panel("one_factor", PARAMETERS, CURVE, [{"strike": 1., "maturity": .33}, {"strike": 1., "maturity": 1.}], n_steps=16)


if __name__ == "__main__":
    unittest.main()

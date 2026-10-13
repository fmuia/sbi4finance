"""Tests for the SVJ pipeline: simulator, filters, particle MCMC diagnostics, risk, checks and audit.

    pixi run test-svj
"""

import os
import unittest

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, norm

from sbi4finance.svj import audit, checks, classical, filtering, model, pmmh, risk

EXAMPLE = np.array([8.0, np.log(0.18), np.log(35), np.log(0.55), np.arctanh(-0.6), np.log(2.0), -2.5, np.log(3.5)])


def example_series(seed=3):
    return model.simulate_from_coordinates(EXAMPLE[None], np.random.default_rng(seed))


class TestModel(unittest.TestCase):
    def test_prior_ranges_match_design_document(self):
        table = model.prior_table()
        expected = {
            "Long-run volatility": (0.11, 0.55),
            "Half-life of variance shocks": (7.6, 118.3),
            "CV of variance": (0.30, 1.19),
            "Leverage": (-0.89, 0.43),
            "Jump frequency": (0.57, 10.87),
            "Mean jump size": (-5.9, 3.9),
            "Jump size sd": (1.24, 7.25),
        }
        for quantity, (lo, hi) in expected.items():
            self.assertAlmostEqual(table.loc[quantity, "2.5%"], lo, delta=0.01)
            self.assertAlmostEqual(table.loc[quantity, "97.5%"], hi, delta=0.06)

    def test_log_prior_matches_densities_of_the_quantities(self):
        coords = model.draw_coordinates(50, np.random.default_rng(0))
        # leverage: z = atanh(rho), (rho + 1) / 2 ~ Beta(2, 4), dz -> du Jacobian 2 u (1 - u)
        u = (np.tanh(coords[:, 4]) + 1) / 2
        expected = beta.logpdf(u, 2, 4) + np.log(2 * u * (1 - u))
        for i, mean, sd in model.PRIOR.normal_coordinates():
            expected += norm.logpdf(coords[:, i], mean, sd)
        np.testing.assert_allclose(model.log_prior(coords), expected, rtol=1e-10)

    def test_parameters_and_daily_model(self):
        p = model.model_params(EXAMPLE)
        self.assertAlmostEqual(p[1], (18**2) / 252)  # theta in %² per day
        self.assertAlmostEqual(np.log(2) / p[2], 35)  # half-life
        self.assertAlmostEqual(p[3] / np.sqrt(2 * p[2] * p[1]), 0.55)  # c
        sims = example_series()
        self.assertEqual(sims["returns"].shape, (1, model.WINDOW))
        self.assertTrue((sims["variance"] >= model.V_MIN).all())
        self.assertAlmostEqual(sims["targets"][0, -1], model.log_spot_vol(sims["variance"][0, -1]))

    def test_stationary_start_and_long_run_level(self):
        rng = np.random.default_rng(1)
        coords = np.repeat(EXAMPLE[None], 20_000, 0)
        theta = model.model_params(EXAMPLE)[1]
        v0 = model.initial_variance(coords, rng)
        self.assertAlmostEqual(v0.mean() / theta, 1, delta=0.02)
        self.assertAlmostEqual(v0.std() / v0.mean(), 0.55, delta=0.02)

    def test_network_input_and_forward_paths(self):
        y = np.array([[0.0, 1.0, -2.0]])
        x = model.network_input(y)
        self.assertEqual(x.shape, (1, 3, 2))
        np.testing.assert_allclose(x[0, :, 1], np.log(y[0] ** 2 + 0.01), rtol=1e-6)
        paths = model.forward_returns(np.append(EXAMPLE, np.log(0.2))[None].repeat(3, 0), 10, 7, np.random.default_rng(0))
        self.assertEqual(paths.shape, (3, 7, 10))


class TestFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = example_series()["returns"][0]
        cls.params = model.model_params(EXAMPLE)

    def test_jax_twins(self):
        coords = model.draw_coordinates(20, np.random.default_rng(2))
        np.testing.assert_allclose(
            np.asarray(filtering.model_params_jnp(jnp.asarray(coords, jnp.float32))), model.model_params(coords), rtol=2e-5
        )
        for c in coords[:5]:
            self.assertAlmostEqual(float(pmmh.log_prior_jnp(jnp.asarray(c, jnp.float32))), model.log_prior(c), delta=1e-4)

    def test_particle_filter_is_unbiased_against_the_grid(self):
        grid = float(filtering.grid_loglik(jnp.asarray(self.params), jnp.asarray(self.y), n_cells=200))
        loglik, v_t = filtering.loglik_batch(np.repeat(self.params[None], 128, 0), self.y, seed=0, n_particles=400, chunk=128)
        # log of the mean likelihood estimate, not the mean of the logs
        self.assertAlmostEqual(logsumexp(loglik) - np.log(len(loglik)), grid, delta=0.15)
        self.assertLess(loglik.mean(), grid)  # Jensen: log L-hat is biased low
        self.assertTrue((v_t >= model.V_MIN).all())

    def test_noise_falls_with_particles(self):
        sd_small = filtering.loglik_sd(self.params, self.y, 1, 50)
        sd_large = filtering.loglik_sd(self.params, self.y, 1, 400)
        self.assertLess(sd_large, sd_small / 2)


class TestChainDiagnostics(unittest.TestCase):
    def test_independent_chains(self):
        chains = np.random.default_rng(0).standard_normal((4, 2_000))
        rhat, ess = pmmh.rhat_ess(chains)
        self.assertLess(rhat, 1.01)
        self.assertGreater(ess, 5_000)

    def test_stuck_chains(self):
        rng = np.random.default_rng(1)
        chains = rng.standard_normal((4, 2_000)) + np.array([0, 0, 3, 3])[:, None]
        self.assertGreater(pmmh.rhat_ess(chains)[0], 1.5)
        ar = np.zeros((4, 2_000))
        for t in range(1, 2_000):
            ar[:, t] = 0.99 * ar[:, t - 1] + rng.standard_normal(4)
        self.assertLess(pmmh.rhat_ess(ar)[1], 400)


class TestRisk(unittest.TestCase):
    def test_var_es_of_a_normal_sample(self):
        r = np.random.default_rng(0).standard_normal(2_000_000)
        var, es = risk.var_es(r)
        self.assertAlmostEqual(var, norm.ppf(0.99), delta=0.01)
        self.assertAlmostEqual(es, norm.pdf(norm.ppf(0.975)) / 0.025, delta=0.01)

    def test_conditional_var_without_jumps_is_normal(self):
        coords = EXAMPLE.copy()
        coords[5] = np.log(1e-6)  # almost no jumps
        targets = np.append(coords, np.log(0.2))[None]
        v = model.variance_from_log_spot_vol(np.log(0.2))
        mu = model.model_params(coords)[0]
        self.assertAlmostEqual(risk.conditional_var(targets)[0], -(mu + np.sqrt(v) * norm.ppf(0.01)), delta=1e-4)

    def test_conditional_var_agrees_with_simulation(self):
        targets = np.append(EXAMPLE, np.log(0.25))[None]
        pooled = risk.horizon_returns(targets, horizons=(1,), n_paths=400_000, seed=0)[1]
        self.assertAlmostEqual(risk.var_es(pooled)[0] / risk.conditional_var(targets)[0], 1, delta=0.02)

    def test_cloud_quantile_matches_single_state_quantile(self):
        params = model.model_params(EXAMPLE)
        v = model.variance_from_log_spot_vol(np.log(0.25))
        q_cloud = risk.cloud_quantile(params, np.array([v]), 0.01)
        q_mix = risk.mixture_quantile(params[None], np.array([v]), 0.01)[0]
        self.assertAlmostEqual(q_cloud, q_mix, places=6)
        self.assertAlmostEqual(risk.one_day_cdf(params, v, q_cloud), 0.01, places=6)

    def test_mixture_var_exceeds_average_var(self):
        # Appendix H.2: equal mixture of N(0, 1) and N(0, 2²) has 99% VaR 4.11 against an average of 3.49
        r = np.concatenate([np.random.default_rng(0).standard_normal(1_000_000), 2 * np.random.default_rng(1).standard_normal(1_000_000)])
        self.assertAlmostEqual(risk.var_es(r)[0], 4.11, delta=0.02)


class TestChecks(unittest.TestCase):
    def test_exact_posterior_is_calibrated(self):
        rng = np.random.default_rng(0)
        n, draws_per = 1_000, 200
        prior_mean = rng.standard_normal((n, 2))
        truths = prior_mean + rng.standard_normal((n, 2))
        draws = prior_mean[:, None, :] + rng.standard_normal((n, draws_per, 2))
        result = checks.sbc(draws, truths)
        self.assertTrue(result["inside"].all())
        np.testing.assert_allclose(checks.coverage(draws, truths)[2], 0.9, atol=0.03)
        overconfident = prior_mean[:, None, :] + 0.5 * rng.standard_normal((n, draws_per, 2))
        self.assertFalse(checks.sbc(overconfident, truths)["inside"].any())

    def test_contraction(self):
        rng = np.random.default_rng(0)
        truths = rng.standard_normal((2_000, 1))
        draws = truths[:, None, :] * 0 + rng.standard_normal((2_000, 100, 1))  # posterior = prior
        self.assertAlmostEqual(checks.contraction(draws, truths)[0], 0, delta=0.08)

    def test_misspecification_statistic(self):
        rng = np.random.default_rng(0)
        test = checks.MisspecificationTest(rng.standard_normal((500, 4)), rng.standard_normal((200, 4)), 8.0)
        self.assertGreater(test.pvalue(rng.standard_normal((1, 4)))[0], 0.01)
        self.assertLess(test.pvalue(np.full((1, 4), 6.0))[0], 0.01)


class TestClassical(unittest.TestCase):
    def test_bounds_and_exact_gradient(self):
        lo, hi = classical.prior_bounds()
        self.assertTrue(np.all((lo < classical.prior_mean()) & (classical.prior_mean() < hi)))
        y = example_series()["returns"][0]
        value_and_grad, _ = classical._objective(40)
        with jax.enable_x64(True):
            value, grad = value_and_grad(jnp.asarray(EXAMPLE), jnp.asarray(y))
            for k in (1, 5, 7):
                e = np.zeros(8)
                e[k] = 1e-5
                plus = float(value_and_grad(jnp.asarray(EXAMPLE + e), jnp.asarray(y))[0])
                minus = float(value_and_grad(jnp.asarray(EXAMPLE - e), jnp.asarray(y))[0])
                self.assertAlmostEqual(float(grad[k]), (plus - minus) / 2e-5, delta=1e-3 * max(1, abs(float(grad[k]))))


class TestWaldStandardErrors(unittest.TestCase):
    def test_positive_definite_singular_and_indefinite(self):
        positive, se = classical.wald_standard_errors(np.diag([4.0, 1.0]))
        self.assertTrue(positive)
        np.testing.assert_allclose(se, [0.5, 1.0])
        for hessian in (np.array([[1.0, 1.0], [1.0, 1.0]]), np.diag([1.0, 1e-14]), np.diag([1.0, -1.0])):
            positive, se = classical.wald_standard_errors(hessian)
            self.assertFalse(positive)
            self.assertTrue(np.isnan(se).all())


class TestAudit(unittest.TestCase):
    def test_audit_runs_and_normalizes(self):
        sims = example_series()
        rng = np.random.default_rng(0)
        draws = np.append(EXAMPLE, sims["targets"][0, -1]) + 0.1 * rng.standard_normal((500, 9))
        result = audit.audit(draws, sims["returns"][0], seed=0, n_proposals=200, n_particles=100, n_out=300)
        self.assertAlmostEqual(np.exp(logsumexp(result.log_weights)), 1.0, places=6)
        self.assertTrue(0 < result.efficiency <= 1)
        self.assertEqual(result.posterior.shape, (300, 9))


if __name__ == "__main__":
    unittest.main()

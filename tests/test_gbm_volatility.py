"""Tests for the exact answers of the first BayesFlow example: python -m unittest tests/test_gbm_volatility.py"""

import unittest

import numpy as np
from scipy.stats import norm

from sbi4finance import gbm_volatility as gv


class ExactPosterior(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.returns = gv.simulate_returns(6.0, np.log(0.2), gv.WINDOW, self.rng)

    def test_normalized_and_close_to_the_large_sample_answer(self):
        post = gv.exact_posterior(self.returns)
        self.assertAlmostEqual(post["weights"].sum(), 1.0, places=10)
        n, s = len(self.returns), self.returns.std()
        self.assertAlmostEqual(post["mean"][1], np.log(s * np.sqrt(gv.DAYS_PER_YEAR) / 100), delta=0.003)
        self.assertAlmostEqual(post["sd"][1], 1 / np.sqrt(2 * n), delta=0.002)
        # drift: conjugate normal update with the volatility at its estimate
        like_sd = gv.DAYS_PER_YEAR * s / np.sqrt(n)
        prior_mean, prior_sd = gv.DRIFT_PRIOR
        precision = 1 / prior_sd**2 + 1 / like_sd**2
        conjugate_mean = (prior_mean / prior_sd**2 + gv.DAYS_PER_YEAR * self.returns.mean() / like_sd**2) / precision
        self.assertAlmostEqual(post["mean"][0], conjugate_mean, delta=0.02 * post["sd"][0])
        self.assertAlmostEqual(post["sd"][0], precision**-0.5, delta=0.02 * post["sd"][0])

    def test_grid_draws_have_the_posterior_moments(self):
        post = gv.exact_posterior(self.returns)
        draws = gv.sample_posterior(post, 200_000, self.rng)
        np.testing.assert_allclose(draws.mean(0), post["mean"], atol=0.02 * post["sd"].max())
        np.testing.assert_allclose(draws.std(0), post["sd"], rtol=0.02)

    def test_short_windows_leave_the_drift_to_the_prior(self):
        post = gv.exact_posterior(self.returns[-20:])
        self.assertGreater(post["sd"][0], 0.9 * gv.DRIFT_PRIOR[1])


class RiskNumbers(unittest.TestCase):
    def test_single_normal_matches_closed_form(self):
        mean, sd = 0.3, 4.0
        var, es = gv.mixture_var_es([mean], [sd])
        self.assertAlmostEqual(var, -(mean + sd * norm.ppf(0.01)), places=8)
        alpha = 0.025
        self.assertAlmostEqual(es, -mean + sd * norm.pdf(norm.ppf(alpha)) / alpha, places=8)

    def test_mixture_against_simulation(self):
        rng = np.random.default_rng(1)
        means, sds = np.array([0.0, -0.5]), np.array([3.0, 6.0])
        var, es = gv.mixture_var_es(means, sds, [0.7, 0.3])
        pick = rng.random(2_000_000) < 0.3
        sample = np.where(pick, rng.normal(means[1], sds[1], pick.size), rng.normal(means[0], sds[0], pick.size))
        self.assertAlmostEqual(var, -np.quantile(sample, 0.01), delta=0.05)
        tail = sample[sample <= np.quantile(sample, 0.025)]
        self.assertAlmostEqual(es, -tail.mean(), delta=0.05)

    def test_parameter_uncertainty_raises_the_es(self):
        rng = np.random.default_rng(2)
        returns = gv.simulate_returns(8.0, np.log(0.25), 60, rng)
        post = gv.exact_posterior(returns)
        _, es_predictive = gv.exact_predictive_var_es(post, horizon=10)
        _, es_plugin = gv.predictive_var_es(post["mean"][0], post["mean"][1], horizon=10)
        self.assertGreater(es_predictive, es_plugin)


if __name__ == "__main__":
    unittest.main()

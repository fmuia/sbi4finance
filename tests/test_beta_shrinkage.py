"""Checks of the slot-1 beta estimators against direct formulas and simulation."""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from sbi4finance import beta_shrinkage as bs


class EstimatorTests(unittest.TestCase):
    def test_least_squares_matches_linregress(self):
        rng = np.random.default_rng(1)
        market = rng.normal(0, 0.02, 104)
        returns = 0.001 + 1.3 * market + rng.normal(0, 0.03, 104)
        b, se = bs.least_squares(returns, market)
        ref = stats.linregress(market, returns)
        self.assertAlmostEqual(float(b[0]), ref.slope, places=10)
        self.assertAlmostEqual(float(se[0]), ref.stderr, places=10)

    def test_posterior_matches_the_normal_normal_update_on_a_grid(self):
        b, se, m, tau = 1.6, 0.2, 1.0, 0.3
        grid = np.linspace(-1, 3, 40001)
        density = stats.norm.pdf(grid, m, tau) * stats.norm.pdf(b, grid, se)
        density /= density.sum()
        mean, sd, _ = bs.posterior(np.array([b]), np.array([se]), m, tau)
        self.assertAlmostEqual(float(mean[0]), float(np.sum(grid * density)), places=6)
        self.assertAlmostEqual(float(sd[0]), float(np.sqrt(np.sum((grid - mean[0]) ** 2 * density))), places=6)

    def test_empirical_bayes_recovers_the_population_net_of_noise(self):
        rng = np.random.default_rng(2)
        universe = bs.simulate(rng, bs.SETTINGS["monthly"], n_stocks=5000)
        b, se = bs.least_squares(universe["returns"], universe["market"])
        eb = bs.empirical_bayes(b, se)
        self.assertLess(abs(eb["prior_mean"] - 1.0), 0.02)
        self.assertLess(abs(eb["prior_sd"] - 0.3), 0.02)
        self.assertGreater(np.std(b), 0.33)  # the raw estimates are more dispersed than the true betas

    def test_bloomberg_rule(self):
        np.testing.assert_allclose(bs.bloomberg(np.array([1.0, 2.0, 0.4])), [1.0, 1.67, 0.598])


class StudyTests(unittest.TestCase):
    def test_bayes_beats_least_squares_and_removes_the_winners_curse(self):
        result = bs.study(bs.SETTINGS["weekly"], n_universes=20, seed=3)
        self.assertEqual(result["universes_beating_least_squares"]["bayes"], 20)
        self.assertGreater(result["top_bias"]["least_squares"], 0.1)
        self.assertLess(abs(result["top_bias"]["bayes"]), 0.03)
        self.assertLess(result["hedge_exposure"]["least_squares"]["mean"], -5)
        for method in bs.INTERVAL_METHODS:
            self.assertLess(abs(result["coverage"][method] - bs.LEVEL), 0.02)

    def test_window_betas_and_blume_test_on_a_synthetic_panel(self):
        rng = np.random.default_rng(4)
        weeks, stocks = 52 * 6 + 1, 40
        market = rng.normal(0.001, 0.022, weeks)
        beta = rng.normal(1.0, 0.3, stocks)
        log_returns = beta[None, :] * market[:, None] + rng.normal(0, 0.035, (weeks, stocks))
        prices = pd.DataFrame(np.exp(np.cumsum(np.column_stack([log_returns, market]), axis=0)),
                              columns=[f"S{i}" for i in range(stocks)] + ["MKT"])
        rows = bs.window_betas(prices, "MKT", weeks=104)
        self.assertEqual(len(rows), 3 * stocks)
        result = bs.blume_test(rows)
        self.assertEqual(result["pairs"], 2 * stocks)
        self.assertLess(result["mse"]["bayes"], result["mse"]["least_squares"])
        self.assertLess(result["slope_next_on_current"], 1.0)


if __name__ == "__main__":
    unittest.main()

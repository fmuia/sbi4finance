"""Checks of the slot-1 beta estimators against direct formulas and simulation."""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from sbi4finance import beta_shrinkage as bs


def weekly_panel(rng, n_weeks, beta, noise=0.035, market_sd=0.022):
    """Weekly prices (dated Fridays) for stocks with the given betas, plus the market as MKT."""
    market = rng.normal(0.001, market_sd, n_weeks)
    log_returns = np.asarray(beta)[None, :] * market[:, None] + rng.normal(0, noise, (n_weeks, len(beta)))
    prices = np.exp(np.cumsum(np.vstack([np.zeros((1, len(beta) + 1)), np.column_stack([log_returns, market])]), axis=0))
    dates = pd.date_range("2000-01-07", periods=n_weeks + 1, freq="W-FRI")
    return pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(len(beta))] + ["MKT"])


class EstimatorTests(unittest.TestCase):
    def test_least_squares_matches_linregress_for_every_row(self):
        rng = np.random.default_rng(1)
        market = rng.normal(0, 0.02, 104)
        alphas, betas = np.array([0.01, -0.02, 0.0]), np.array([1.3, 0.4, 2.0])
        returns = alphas[:, None] + betas[:, None] * market[None, :] + rng.normal(0, 0.03, (3, 104))
        b, se = bs.least_squares(returns, market)
        for i in range(3):
            ref = stats.linregress(market, returns[i])
            self.assertAlmostEqual(float(b[i]), ref.slope, places=10)
            self.assertAlmostEqual(float(se[i]), ref.stderr, places=10)

    def test_least_squares_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bs.least_squares(np.ones((2, 10)), np.ones(10))  # constant market
        with self.assertRaises(ValueError):
            bs.least_squares(np.ones((2, 10)), np.arange(9.0))  # periods do not match
        b, _ = bs.least_squares(np.arange(10.0) * 2, pd.Series(np.arange(10.0)))
        self.assertAlmostEqual(float(b[0]), 2.0)

    def test_posterior_matches_the_normal_normal_update_on_a_grid(self):
        b, se, m, tau = 1.6, 0.2, 1.0, 0.3
        grid = np.linspace(-1, 3, 40001)
        density = stats.norm.pdf(grid, m, tau) * stats.norm.pdf(b, grid, se)
        density /= density.sum()
        mean, sd, _ = bs.posterior(np.array([b]), np.array([se]), m, tau)
        self.assertAlmostEqual(float(mean[0]), float(np.sum(grid * density)), places=6)
        self.assertAlmostEqual(float(sd[0]), float(np.sqrt(np.sum((grid - mean[0]) ** 2 * density))), places=6)

    def test_empirical_bayes_recovers_the_population_net_of_noise(self):
        for population in (bs.POPULATION, bs.UTILITIES):
            universe = bs.simulate(np.random.default_rng(2), bs.SETTINGS["monthly"], n_stocks=5000, population=population)
            b, se = bs.least_squares(universe["returns"], universe["market"])
            eb = bs.empirical_bayes(b, se)
            self.assertLess(abs(eb["prior_mean"] - population.beta_mean), 0.02)
            self.assertLess(abs(eb["prior_sd"] - population.beta_sd), 0.02)
            self.assertGreater(np.std(b), population.beta_sd + 0.02)  # the raw estimates also carry their noise
            self.assertFalse(eb["at_floor"])

    def test_empirical_bayes_pools_everything_when_the_spread_is_all_noise(self):
        b, se = np.array([0.9, 1.0, 1.1]), np.full(3, 0.5)
        eb = bs.empirical_bayes(b, se)
        self.assertTrue(eb["at_floor"])
        self.assertAlmostEqual(eb["prior_sd"], np.sqrt(bs.TAU2_FLOOR))
        np.testing.assert_allclose(eb["mean"], 1.0, atol=1e-4)

    def test_one_prior_per_volatility_group(self):
        rng = np.random.default_rng(3)
        se = np.concatenate([np.full(250, 0.05), np.full(250, 0.4)])
        beta = np.concatenate([rng.normal(1.0, 0.1, 250), rng.normal(1.0, 0.6, 250)])
        b = beta + se * rng.standard_normal(500)
        grouped = bs.empirical_bayes_by_group(b, se, n_groups=2)
        self.assertEqual(set(grouped["group"][:250]), {0})
        pooled = bs.empirical_bayes(b, se)
        self.assertLess(np.mean((grouped["mean"] - beta) ** 2), np.mean((pooled["mean"] - beta) ** 2))
        few = bs.empirical_bayes_by_group(b[:15], se[:15])  # too few stocks for groups: plain empirical Bayes
        np.testing.assert_allclose(few["mean"], bs.empirical_bayes(b[:15], se[:15])["mean"])

    def test_bloomberg_rule(self):
        np.testing.assert_allclose(bs.bloomberg(np.array([1.0, 2.0, 0.4])), [1.0, 1.67, 0.598])

    def test_population_moments(self):
        for population in (bs.POPULATION, *bs.MISSPECIFIED.values()):
            beta, idio_vol = population.draw(np.random.default_rng(4), 200_000)
            self.assertLess(abs(beta.mean() - population.mean), 0.01, population.label)
            self.assertLess(abs(beta.std() - population.sd), 0.01, population.label)
            self.assertTrue(np.all(idio_vol > 0))
        beta, idio_vol = bs.MISSPECIFIED["vol_on_distance"].draw(np.random.default_rng(5), 100_000)
        self.assertGreater(np.corrcoef(np.abs(beta - 1), np.log(idio_vol))[0, 1], 0.4)
        self.assertGreater(stats.skew(bs.MISSPECIFIED["skewed"].draw(np.random.default_rng(6), 100_000)[0]), 0.6)


class StudyTests(unittest.TestCase):
    def test_bayes_removes_the_winners_curse_in_the_normal_population(self):
        result = bs.study(bs.SETTINGS["weekly"], n_universes=20, seed=3)
        self.assertEqual(result["universes_beating_least_squares"]["bayes"], 20)
        self.assertGreater(result["top_bias"]["least_squares"], 0.1)
        self.assertLess(abs(result["top_bias"]["bayes"]), 0.03)
        self.assertLess(result["hedge_exposure"]["least_squares"]["mean"], -5)
        self.assertEqual(result["hedge_exposure"]["least_squares"]["universes_short"], 20)
        self.assertLess(abs(result["random_hedge_exposure"]["least_squares"]["mean"]), 1.5)  # no selection, no bias
        self.assertAlmostEqual(result["best_fixed_weight"], 0.74, delta=0.02)
        t = stats.t.ppf(0.5 + bs.LEVEL / 2, bs.SETTINGS["weekly"].n_obs - 2)
        self.assertAlmostEqual(result["median_width"]["least_squares"], 2 * t * result["median_se"])
        for method in bs.INTERVAL_METHODS:
            self.assertLess(abs(result["coverage"][method]["all"] - bs.LEVEL), 0.02)
        self.assertLess(result["coverage"]["least_squares"]["top"], 0.86)  # exact per stock, not per selected stock
        self.assertGreater(result["coverage"]["bayes"]["top"], 0.86)
        self.assertLess(result["coverage"]["bayes"]["extreme"], 0.85)

    def test_the_pooled_prior_fails_with_a_noisy_fringe_and_the_grouped_prior_repairs_it(self):
        result = bs.study(bs.SETTINGS["weekly"], bs.MISSPECIFIED["fringe"], n_universes=20, seed=3)
        self.assertGreater(result["mse_vs_least_squares"]["bayes"], 1.0)
        self.assertLess(result["mse_vs_least_squares"]["bayes_by_volatility"], 0.85)

    def test_study_checks_top(self):
        with self.assertRaises(ValueError):
            bs.study(bs.SETTINGS["weekly"], n_universes=1, top=0)


class RealDataTests(unittest.TestCase):
    def test_window_betas_recover_the_betas_in_adjacent_windows(self):
        rng = np.random.default_rng(4)
        beta = rng.normal(1.0, 0.3, 40)
        prices = weekly_panel(rng, 52 * 6 + 10, beta)
        rows = bs.window_betas(prices, "MKT", weeks=104)
        self.assertEqual(len(rows), 3 * 40)
        self.assertEqual(sorted({r["window"] for r in rows}), [0, 1, 2])
        self.assertEqual(max(r["window_end"] for r in rows), prices.index[-1])  # anchored at the latest week
        estimate = np.array([r["beta"] for r in rows if r["window"] == 2])
        self.assertGreater(np.corrcoef(estimate, beta)[0, 1], 0.8)

    def test_window_betas_drop_bad_data(self):
        rng = np.random.default_rng(5)
        prices = weekly_panel(rng, 52 * 4, rng.normal(1.0, 0.3, 10))
        with self.assertRaises(ValueError):
            bs.window_betas(prices, "SPY", weeks=52)
        with self.assertRaises(TypeError):
            bs.window_betas(prices.reset_index(drop=True), "MKT", weeks=52)
        clean = prices.copy()
        prices.iloc[30, 0] = np.nan  # a missing price drops that stock from its window
        prices.iloc[120, 1] = 0.0  # so does a zero price
        prices.iloc[170, -1] = np.nan  # a missing market price drops the whole window
        rows = bs.window_betas(prices, "MKT", weeks=52)
        by_window = {k: {r["stock"] for r in rows if r["window"] == k} for k in range(4)}
        self.assertNotIn("S0", by_window[0])
        self.assertNotIn("S1", by_window[2])
        self.assertEqual(by_window[3], set())
        self.assertTrue(all(np.isfinite(r["beta"]) for r in rows))
        gappy = clean.drop(clean.index[60])  # a missing week: 207 returns, three windows, the oldest spans the gap
        self.assertEqual({r["window"] for r in bs.window_betas(gappy, "MKT", weeks=52)}, {1, 2})

    def test_blume_test_pairs_only_adjacent_windows(self):
        rows = [{"window": k, "stock": f"S{i}", "beta": 1.0 + 0.1 * i, "se": 0.1} for k in (0, 1, 3) for i in range(5)]
        self.assertEqual(bs.blume_test(rows)["window_pairs"], 1)
        with self.assertRaises(ValueError):
            bs.blume_test([r for r in rows if r["window"] != 1])

    def test_blume_test_with_constant_and_drifting_betas(self):
        rng = np.random.default_rng(6)
        beta = rng.normal(1.0, 0.3, 60)
        prices = weekly_panel(rng, 52 * 20, beta)
        constant = bs.blume_test(bs.window_betas(prices, "MKT", weeks=52), n_boot=200, seed=1)
        self.assertEqual(constant["window_pairs"], 19)
        self.assertLess(abs(constant["slope_next_on_current"] - constant["constant_beta_slope"]), 0.08)
        self.assertLess(abs(constant["change_over_noise"] - 1), 0.15)
        self.assertLess(constant["mse"]["bayes"], constant["mse"]["least_squares"])
        self.assertEqual(constant["relative_change_interval"]["least_squares"], [0.0, 0.0])
        lo, hi = constant["relative_change_interval"]["bayes"]
        self.assertLess(lo, constant["mse"]["bayes"] / constant["mse"]["least_squares"] - 1)
        self.assertLess(constant["mse"]["bayes"] / constant["mse"]["least_squares"] - 1, hi)
        rows = bs.window_betas(prices, "MKT", weeks=52)
        for r in rows:  # a level that moves between windows is not drift in any one stock's beta
            r["beta"] += 0.3 * (r["window"] % 2)
        shifted = bs.blume_test(rows)
        self.assertAlmostEqual(shifted["slope_next_on_current"], constant["slope_next_on_current"], places=10)
        drift = rng.normal(0, 0.3, (20, 60))  # every window's betas move away from the last
        for r in rows:
            r["beta"] += drift[r["window"], int(r["stock"][1:])]
        drifting = bs.blume_test(rows)
        self.assertLess(drifting["slope_next_on_current"], drifting["constant_beta_slope"] - 0.2)
        self.assertGreater(drifting["change_over_noise"], 2)
        with self.assertRaises(ValueError):
            bs.blume_test([r for r in rows if r["window"] < 3], n_boot=10)  # two pairs cannot fill blocks of three


if __name__ == "__main__":
    unittest.main()

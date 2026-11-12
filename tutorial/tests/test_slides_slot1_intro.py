"""The opening example of the slot 1 slides: its inputs, the portfolio under a prior, and the share in stocks."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy import stats

TUTORIAL = Path(__file__).resolve().parents[1]
FOLDER = TUTORIAL / "slides" / "slot1"


def load_example():
    spec = importlib.util.spec_from_file_location("slot1_intro_example", FOLDER / "intro_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = load_example()
INPUTS = ex.load()


class InputsTests(unittest.TestCase):
    def test_the_window_is_twenty_years_of_monthly_returns_on_ten_industries(self):
        self.assertEqual((INPUTS.months, INPUTS.first_month, INPUTS.last_month), (240, 200610, 202609))
        self.assertEqual(INPUTS.codes, tuple(ex.INDUSTRIES))
        self.assertEqual(INPUTS.covariance.shape, (10, 10))
        np.testing.assert_allclose(INPUTS.covariance, INPUTS.covariance.T)
        self.assertGreater(np.linalg.eigvalsh(INPUTS.covariance).min(), 0.0)

    def test_the_source_files_are_named_with_their_identity(self):
        source = json.loads(ex.INPUTS.read_text())["source"]
        self.assertEqual(len(source["files"]), 2)
        for name, entry in source["files"].items():
            self.assertTrue(entry["url"].endswith(name))
            self.assertEqual(len(entry["sha256"]), 64)

    def test_the_industry_that_is_raised_has_the_highest_average(self):
        self.assertEqual(INPUTS.changed, int(np.argmax(INPUTS.mean)))
        self.assertEqual(INPUTS.names[INPUTS.changed], "Technology")
        ratio = ex.CHANGE * INPUTS.mean[INPUTS.changed] / INPUTS.standard_error[INPUTS.changed]
        self.assertTrue(0.3 < ratio < 0.4)  # the change is about a third of the average's standard error


class PortfolioTests(unittest.TestCase):
    def test_no_prior_gives_the_plug_in_weights(self):
        plug_in = ex.max_sharpe(INPUTS.mean, INPUTS.covariance)
        np.testing.assert_allclose(ex.weights(INPUTS, 0.0), plug_in, atol=1e-12)
        self.assertAlmostEqual(plug_in.sum(), 1.0, places=12)

    def test_a_certain_prior_gives_the_minimum_variance_portfolio_whatever_the_averages(self):
        minimum = ex.minimum_variance(INPUTS.covariance)
        np.testing.assert_allclose(ex.weights(INPUTS, 1.0), minimum, atol=1e-12)
        np.testing.assert_allclose(ex.weights(INPUTS, 1.0, raised=INPUTS.changed), minimum, atol=1e-12)

    def test_the_predictive_moments_are_those_of_the_posterior_computed_directly(self):
        months, n, weight = INPUTS.months, len(INPUTS.mean), 0.4
        tau = ex.prior_months(weight, months)
        self.assertAlmostEqual(tau / (months + tau), weight, places=12)
        sigma = INPUTS.covariance * (months - 1) / (months - n - 2)
        inverse, one = np.linalg.inv(sigma), np.ones(n)
        # eta diffuse: the prior precision of mu is tau (Sigma^-1 - Sigma^-1 1 1' Sigma^-1 / 1' Sigma^-1 1)
        precision = months * inverse + tau * (inverse - np.outer(inverse @ one, inverse @ one) / (one @ inverse @ one))
        covariance = np.linalg.inv(precision)
        mean, predictive = ex.predictive(INPUTS.mean, INPUTS.covariance, months, weight)
        np.testing.assert_allclose(mean, covariance @ (months * inverse @ INPUTS.mean), rtol=1e-9)
        np.testing.assert_allclose(predictive, sigma + covariance, rtol=1e-9)

    def test_the_move_of_the_raised_industry_shrinks_as_the_prior_gains_weight(self):
        k = INPUTS.changed
        moves = [ex.weights(INPUTS, w, raised=k)[k] - ex.weights(INPUTS, w)[k] for w in np.linspace(0.0, 1.0, 21)]
        self.assertTrue(np.all(np.diff(moves) < 0))
        self.assertAlmostEqual(moves[-1], 0.0, places=12)

    def test_the_weight_the_data_suggest_is_jorions_estimate(self):
        months, n = INPUTS.months, len(INPUTS.mean)
        sigma = INPUTS.covariance * (months - 1) / (months - n - 2)
        gap = INPUTS.mean - ex.minimum_variance(sigma) @ INPUTS.mean
        expected = (n + 2) / ((n + 2) + months * gap @ np.linalg.solve(sigma, gap))
        self.assertAlmostEqual(ex.suggested_weight(INPUTS.mean, INPUTS.covariance, months), expected, places=12)

    def test_one_parameter_is_the_product_of_the_prior_and_the_data(self):
        one = ex.one_parameter(INPUTS, 0.5)  # a prior worth as much as the data
        (prior, prior_sd), (data, data_sd), (posterior, posterior_sd) = one["prior"], one["data"], one["posterior"]
        self.assertAlmostEqual(prior_sd, data_sd, places=12)
        self.assertAlmostEqual(posterior, 0.5 * (prior + data), places=12)
        self.assertAlmostEqual(posterior_sd, data_sd / np.sqrt(2.0), places=12)
        precision = 1.0 / prior_sd**2 + 1.0 / data_sd**2
        self.assertAlmostEqual(posterior, (prior / prior_sd**2 + data / data_sd**2) / precision, places=12)

    def test_the_portfolio_without_short_positions_is_the_best_on_a_grid(self):
        rng = np.random.default_rng(5)
        factor = rng.normal(size=(3, 3))
        covariance = factor @ factor.T + 0.5 * np.eye(3)
        mean = np.array([0.8, 0.3, 0.5])
        held = ex.max_sharpe_long_only(mean, covariance)
        self.assertTrue(np.all(held >= 0) and abs(held.sum() - 1.0) < 1e-12)
        grid = np.array([(a, b, 1.0 - a - b) for a in np.linspace(0, 1, 201) for b in np.linspace(0, 1, 201)
                         if a + b <= 1.0 + 1e-12])
        sharpe = grid @ mean / np.sqrt(np.einsum("ij,jk,ik->i", grid, covariance, grid))
        best = held @ mean / np.sqrt(held @ covariance @ held)
        self.assertGreaterEqual(best, sharpe.max() - 1e-9)
        self.assertLess(np.abs(grid[np.argmax(sharpe)] - held).max(), 0.01)

    def test_the_weights_the_slides_quote(self):
        k, retail = INPUTS.changed, INPUTS.codes.index("Shops")
        before, after = ex.weights(INPUTS, 0.0), ex.weights(INPUTS, 0.0, raised=k)
        self.assertEqual([round(100 * x) for x in (before[k], after[k], before[retail], after[retail])],
                         [74, 95, 31, 18])
        suggested = ex.suggested_weight(INPUTS.mean, INPUTS.covariance, INPUTS.months)
        self.assertEqual(round(100 * suggested), 56)
        self.assertEqual(round(ex.prior_months(suggested, INPUTS.months) / 12), 26)
        held = [round(100 * ex.weights(INPUTS, suggested, raised=j)[k]) for j in (None, k)]
        self.assertEqual(held, [39, 48])
        self.assertEqual(round(100 * ex.weights(INPUTS, 1.0)[k]), 12)


class HorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.years = np.array([1, 10, 20])
        cls.exact, cls.uncertain = ex.shares(INPUTS, cls.years)

    def test_estimates_taken_as_exact_give_about_the_same_share_at_every_horizon(self):
        m, s = INPUTS.market_mean, INPUTS.market_volatility
        continuous = (m + 0.5 * s * s) / (ex.RISK_AVERSION * s * s)  # the share of an investor who rebalances
        self.assertLess(np.ptp(self.exact), 0.02)
        self.assertLess(np.abs(self.exact - continuous).max(), 0.02)

    def test_their_uncertainty_lowers_the_share_and_more_so_at_long_horizons(self):
        self.assertTrue(np.all(self.uncertain < self.exact))
        self.assertTrue(np.all(np.diff(self.uncertain) < 0))
        # the variance of the cumulative return is inflated by 1 + horizon / sample, and the share falls about as much
        inflated = self.exact / (1.0 + 12.0 * self.years / INPUTS.months)
        self.assertLess(np.abs(self.uncertain - inflated).max(), 0.05)

    def test_the_best_share_agrees_with_a_simulation(self):
        horizon = 120.0
        location, scale = horizon * INPUTS.market_mean, INPUTS.market_volatility * np.sqrt(horizon)
        growth = np.exp(location + scale * np.random.default_rng(11).standard_normal(400_000))
        shares = np.linspace(0.0, 0.99, 100)
        wealth = (1.0 - shares[:, None]) + shares[:, None] * growth[None, :]
        simulated = shares[np.argmax((wealth ** (1.0 - ex.RISK_AVERSION) / (1.0 - ex.RISK_AVERSION)).mean(axis=1))]
        self.assertLess(abs(ex.best_share(location, scale, stats.norm.pdf) - simulated), 0.03)

    def test_the_shares_the_slides_quote(self):
        self.assertEqual([round(100 * x) for x in self.uncertain[1:]], [58, 45])
        self.assertTrue(np.all((self.exact > 0.825) & (self.exact < 0.845)))  # 83 to 84%, at 1, 10 and 20 years


if __name__ == "__main__":
    unittest.main()

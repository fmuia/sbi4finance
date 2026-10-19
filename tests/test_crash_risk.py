"""Checks of the slot-1 crash-risk models and estimators against direct formulas and simulation."""

import unittest

import numpy as np
from scipy import stats

from sbi4finance import crash_risk as cr


class JumpModelTests(unittest.TestCase):
    def setUp(self):
        self.model = cr.JumpModel()
        self.x = np.array([np.log(2.5), np.log(3.0), -1.0, np.log(1.0)])
        self.mu = 0.03
        self.y = self.model.simulate(self.x, self.mu, 1000, np.random.default_rng(1))

    def test_likelihood_is_the_daily_two_component_mixture(self):
        p = 2.5 / cr.YEAR
        direct = np.log((1 - p) * stats.norm.pdf(self.y, self.mu, 1.0)
                        + p * stats.norm.pdf(self.y, self.mu - 1.0, np.sqrt(1.0 + 9.0))).sum()
        self.assertAlmostEqual(self.model.loglik(self.x, self.y, self.mu), direct, places=8)

    def test_gradient_matches_finite_differences(self):
        _, grad = self.model.loglik_and_grad(self.x, self.y, self.mu)
        numeric = cr._numeric_grad(lambda z: self.model.loglik(z, self.y, self.mu), self.x)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-5)

    def test_crash_probability_matches_simulation(self):
        x = np.array([np.log(20.0), np.log(3.0), -1.0, np.log(1.0)])
        days = self.model.simulate(x, self.mu, 1_000_000, np.random.default_rng(2))
        daily = self.model.daily_crash_probability(4.0, self.mu, 20.0, 3.0, -1.0, 1.0)
        self.assertLess(abs(np.mean(-days >= 4.0) - daily), 4 * np.sqrt(daily / 1_000_000))
        self.assertAlmostEqual(float(self.model.crash_probability(x, 4.0, self.mu)),
                               1 - (1 - daily) ** cr.YEAR, places=12)

    def test_grid_posterior_is_proper_and_concentrates_with_a_long_history(self):
        x = np.array([np.log(25.0), np.log(3.0), -1.0, np.log(1.0)])
        y = self.model.simulate(x, self.mu, 8_000, np.random.default_rng(3))
        grid = self.model.grid(y, float(y.mean()), n_lam=25, n_jsd=17, n_mj=13, n_var=7)
        log_post = grid["ll"] + grid["lp"]
        w = np.exp(log_post - log_post[np.isfinite(log_post)].max())
        w = np.where(np.isfinite(w), w, 0.0)
        w /= w.sum()
        log_lam = np.sum(w * np.nan_to_num(grid["x"][..., 0]))
        self.assertLess(abs(log_lam - np.log(25.0)), 0.25)

    def test_plane_profile_improves_on_the_grid_and_stays_below_the_maximum(self):
        grid = self.model.grid(self.y, self.mu, n_lam=7, n_jsd=5, n_mj=13, n_var=7)
        plane = cr.jump_plane(self.model, self.y, self.mu, grid, 7.0)
        grid_profile = np.where(np.isfinite(grid["ll"]), grid["ll"], -np.inf).max(axis=(2, 3))
        self.assertTrue(np.all(plane["profile"] >= grid_profile - 1e-6))
        self.assertLessEqual(plane["profile"].max(), cr.fit(self.model, self.y, self.mu)["value"] + 1e-6)
        ends = cr.ridge_ends(plane)
        self.assertLessEqual(ends["low"]["crash"], ends["high"]["crash"])

    def test_threshold_count_interval_is_the_exact_poisson_one(self):
        y = np.zeros(cr.YEAR * 4)
        y[:8] = 50.0  # eight flagged days in four years
        rate, (lo, hi) = cr.threshold_jump_count(y)
        self.assertAlmostEqual(rate, 2.0)
        self.assertAlmostEqual(lo, stats.chi2.ppf(0.025, 16) / 2 / 4)
        self.assertAlmostEqual(hi, stats.chi2.ppf(0.975, 18) / 2 / 4)


class StudentModelTests(unittest.TestCase):
    def test_likelihood_and_gradient(self):
        model = cr.StudentModel()
        x = np.array([np.log(4.0), np.log(0.7)])
        y = model.simulate(x, 0.03, 1000, np.random.default_rng(4))
        ll, grad = model.loglik_and_grad(x, y, 0.03)
        self.assertAlmostEqual(ll, stats.t.logpdf(y, 4.0, loc=0.03, scale=0.7).sum(), places=8)
        numeric = cr._numeric_grad(lambda z: model.loglik(z, y, 0.03), x)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-5)


class EstimatorTests(unittest.TestCase):
    def test_peaks_over_threshold_recovers_a_pareto_tail(self):
        rng = np.random.default_rng(5)
        losses = stats.genpareto.rvs(0.2, scale=1.0, size=200_000, random_state=rng)
        pot = cr.pot_fit(-losses)
        daily = 1 - (1 - cr.pot_crash_probability(pot, 8.0)) ** (1 / cr.YEAR)
        self.assertLess(abs(daily / stats.genpareto.sf(8.0, 0.2, scale=1.0) - 1), 0.1)

    def test_weighted_quantiles_of_equal_atoms(self):
        values = np.arange(1.0, 101.0)
        q = cr._weighted_quantiles(values, np.ones(100), [0.5])
        self.assertAlmostEqual(float(q[0]), 50.5)

    def test_year_probability_is_stable_for_tiny_daily_probabilities(self):
        self.assertLess(abs(float(cr.year_probability(1e-12)) / 252e-12 - 1), 1e-9)
        self.assertEqual(float(cr.year_probability(0.0)), 0.0)

    def test_the_prior_washes_out_with_a_long_history(self):
        model = cr.StudentModel()
        y = model.simulate(np.array([np.log(4.0), np.log(1.2)]), 0.03, 20_000, np.random.default_rng(7))
        mu = float(y.mean())
        summ = cr.grid_summaries(model, model.grid(y, mu), 7.0, mu, -np.inf)
        self.assertLess(abs(summ["bayes"]["predictive"] / summ["bayes_flat"]["predictive"] - 1), 0.05)

    def test_analyse_returns_every_method_for_every_threshold(self):
        model = cr.StudentModel()
        rng = np.random.default_rng(6)
        y = model.simulate(np.array([np.log(5.0), np.log(0.8)]), 0.03, 500, rng)
        out = cr.analyse(model, y, (5.0, 7.0), n_boot=5, rng=rng)
        for threshold in ("5", "7"):
            self.assertEqual(set(out["by_threshold"][threshold]["point"]), set(cr.POINT_METHODS))
            self.assertEqual(set(out["by_threshold"][threshold]["interval"]), set(cr.INTERVAL_METHODS))
            for value in out["by_threshold"][threshold]["point"].values():
                self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()

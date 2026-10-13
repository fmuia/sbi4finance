"""Finite-price randomness, weighted precision and conditional diagnostics."""

import json
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import kstest

from sbi4finance.rough_reference import (
    conditional_metrics, importance_weights, randomized_sbc,
    reference_diagnostics, weighted_summary, weighted_target_mean,
)


class RoughReferenceTests(unittest.TestCase):
    def test_weights_follow_quote_law_and_are_unit_translation_invariant(self):
        prices = np.array([[1., 2.], [2., 4.], [3., 1.]])
        y, sigma = np.array([1.5, 2.5]), np.array([.5, 1.])
        expected = np.exp(-.5 * np.sum(((prices-y)/sigma)**2, axis=1))
        expected /= expected.sum()
        assert_allclose(importance_weights(prices, y, sigma), expected, rtol=1e-14)
        assert_allclose(importance_weights(10*prices+100, 10*y+100, 10*sigma), expected)
        assert_allclose(importance_weights(prices, y, .5),
                        importance_weights(prices, y, [.5, .5]))
        # All unshifted exponentials would underflow. Relative log weights survive.
        stable = importance_weights([[1000.], [1000.001]], [0.], 1.)
        assert_allclose(stable[1]/stable[0], np.exp(-1.0000005), rtol=1e-10)

    def test_finite_price_noise_is_integrated_by_clean_reference_rows(self):
        rng = np.random.default_rng(240)
        theta = rng.normal(size=(131072, 2))
        price_noise = rng.normal(scale=.8, size=theta.shape)
        clean_prices = theta + price_noise
        y, sigma = np.array([.7, -.4]), .6
        weights = importance_weights(clean_prices, y, sigma)
        summary = weighted_summary(theta, weights)
        # Gaussian conjugacy: total likelihood variance is .8^2+.6^2 = 1.
        # No covariance is added in the weights: the independent U rows integrate it.
        assert_allclose(summary["mean"], y/2, atol=.013)
        assert_allclose(summary["std"], np.sqrt(.5), atol=.012)
        self.assertLess(abs(summary["correlation"]), .02)
        diagnostics = reference_diagnostics(theta, weights)
        error = np.abs(np.asarray(summary["mean"])-y/2)
        self.assertTrue(np.all(error < 5*np.asarray(diagnostics["mean_mcse"])))

    def test_weighted_moments_quantiles_and_rescaling(self):
        theta = np.array([[0., 3.], [1., 0.], [3., 1.], [100., 100.]])
        weights = np.array([1., 2., 1., 0.])
        result = weighted_summary(theta, weights)
        mean = np.array([1.25, 1.])
        covariance = ((theta-mean)*np.array([.25, .5, .25, 0.])[:, None]).T @ (theta-mean)
        assert_allclose(result["mean"], mean)
        assert_allclose(result["covariance"], covariance)
        assert_allclose(result["quantiles"], [[0., 1., 3.], [0., 0., 3.]])
        self.assertEqual(result, weighted_summary(theta, weights*1e307))
        json.dumps(result, allow_nan=False)

    def test_uniform_weight_mcse_is_ordinary_sample_mean_se(self):
        rng = np.random.default_rng(241)
        theta = rng.multivariate_normal([1., -2.], [[2., .6], [.6, 1.]], size=4096)
        diagnostics = reference_diagnostics(theta, np.ones(len(theta)))
        assert_allclose(diagnostics["mean_mcse"], theta.std(axis=0, ddof=1)/np.sqrt(len(theta)))
        self.assertEqual(diagnostics["ess"], 4096.)
        self.assertEqual(diagnostics["max_weight"], 1/4096)
        self.assertTrue(diagnostics["blocks_valid"])
        comparison = diagnostics["block_comparison"]
        assert_allclose(comparison["mean_difference_in_ref_sd"],
                        (theta[2048:].mean(axis=0)-theta[:2048].mean(axis=0))/theta.std(axis=0))
        assert_allclose(comparison["std_ratio"], theta[2048:].std(axis=0)/theta[:2048].std(axis=0))
        target = weighted_target_mean(theta, np.ones(len(theta)))
        assert_allclose(target["mean"], theta.mean(axis=0))
        assert_allclose(target["mcse"], theta.std(axis=0, ddof=1)/np.sqrt(len(theta)))
        json.dumps(diagnostics, allow_nan=False)
        json.dumps(target, allow_nan=False)

    def test_snis_target_precision_includes_independent_target_noise(self):
        rng = np.random.default_rng(242)
        theta = rng.normal(size=(65536, 2))
        weights = importance_weights(theta, [.5, -.5], 1.)
        # Independent V makes estimates unbiased for each theta's target price.
        targets = theta + rng.normal(scale=3., size=theta.shape)
        quiet = weighted_target_mean(theta, weights)
        noisy = weighted_target_mean(targets, weights)
        assert_allclose(noisy["mean"], [.25, -.25], atol=.05)
        self.assertTrue(np.all(np.asarray(noisy["mcse"]) > 3*np.asarray(quiet["mcse"])))
        expected = np.sqrt(len(theta)/(len(theta)-1)*np.sum(
            (weights[:, None]*(targets-weights@targets))**2, axis=0))
        assert_allclose(noisy["mcse"], expected)

    def test_conditional_metrics_distinguish_joint_dependence_from_marginals(self):
        rng = np.random.default_rng(243)
        theta = rng.multivariate_normal([.5, -.5], [[.5, .4], [.4, .5]], size=16384)
        weights = np.ones(len(theta))
        matched = conditional_metrics(theta, theta, weights, prior_bounds=[[-2, 2], [-2, 2]])
        broken = theta.copy()
        rng.shuffle(broken[:, 1])
        shuffled = conditional_metrics(broken, theta, weights)
        assert_allclose(matched["mean_error_in_ref_sd"], 0, atol=1e-13)
        assert_allclose(matched["reference_mass_of_90_interval"], [.9, .9], atol=2/len(theta))
        assert_allclose(matched["neural_mean"], shuffled["neural_mean"], atol=1e-13)
        assert_allclose(matched["reference_mass_of_90_interval"], shuffled["reference_mass_of_90_interval"])
        self.assertLess(abs(matched["correlation_error"]), 1e-12)
        self.assertGreater(abs(shuffled["correlation_error"]), .75)
        assert_allclose(matched["reference_contraction"],
                        1-theta.std(axis=0)/(4/np.sqrt(12)))
        assert_allclose(matched["reference_mean_update_in_prior_sd"],
                        theta.mean(axis=0)/(4/np.sqrt(12)))
        json.dumps(matched, allow_nan=False)

    def test_empty_block_support_is_explicit(self):
        theta = np.array([[0., 0.], [1., 1.], [2., 1.], [3., 2.]])
        result = reference_diagnostics(theta, [1., 1., 0., 0.])
        self.assertFalse(result["blocks_valid"])
        self.assertIsNone(result["block_comparison"])
        self.assertFalse(result["blocks"][1]["valid"])
        json.dumps(result, allow_nan=False)

    def test_sbc_statistics_exchangeability_ties_and_conditional_limit(self):
        rng = np.random.default_rng(244)
        draws, truth = rng.normal(size=(1024, 128, 2)), rng.normal(size=(1024, 2))
        result = randomized_sbc(draws, truth, seed=245)
        self.assertEqual(result, randomized_sbc(draws, truth, seed=245))
        assert_allclose(result["rank_ks"], [kstest(column, "uniform").statistic
                                           for column in np.asarray(result["randomized_ranks"]).T])
        self.assertAlmostEqual(result["rank_dkw_bound"], np.sqrt(np.log(4/.05)/(2*1024)))
        assert_allclose(result["coverage_90"], [.9, .9], atol=.035)
        tied = randomized_sbc(np.zeros((1024, 32, 2)), np.zeros((1024, 2)), seed=246)
        self.assertEqual(np.min(tied["ranks"]), 0)
        self.assertEqual(np.max(tied["ranks"]), 32)
        self.assertEqual(tied["coverage_90"], [1., 1.])
        # The same prior sampler can have reasonable ranks but ignore informative y.
        prior = draws[:32].reshape(-1, 2)
        reference = .2*rng.normal(size=(8192, 2))+[2., -2.]
        metrics = conditional_metrics(prior, reference, np.ones(len(reference)))
        self.assertGreater(np.min(np.abs(metrics["mean_error_in_ref_sd"])), 9.)
        json.dumps(result, allow_nan=False)

    def test_invalid_shapes_weights_and_degenerate_support(self):
        theta = np.arange(12.).reshape(6, 2)
        invalid_weights = ([1., 1.], np.zeros(6), [-1., 1., 1., 1., 1., 1.], [np.nan]*6)
        for weights in invalid_weights:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                weighted_summary(theta, weights)
        for sigma in (0., -1., np.inf, [1.], [[1., 1.]]):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                importance_weights(theta, [1., 1.], sigma)
        with self.assertRaises(ValueError):
            importance_weights(theta, [[1., 1.]], 1.)
        with self.assertRaises(ValueError):
            importance_weights(np.full((3, 2), np.nan), [1., 1.], 1.)
        with self.assertRaises(ValueError):
            weighted_summary(np.ones((4, 2)), np.ones(4))
        with self.assertRaises(ValueError):
            conditional_metrics(np.ones((4, 2)), theta, np.ones(6))
        with self.assertRaises(ValueError):
            conditional_metrics(theta, theta, np.ones(6), prior_bounds=[[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            weighted_target_mean(np.ones((1, 2)), [1.])
        with self.assertRaises(ValueError):
            randomized_sbc(np.zeros((2, 4, 3)), np.zeros((2, 3)), seed=1)
        with self.assertRaises(ValueError):
            randomized_sbc(np.full((2, 4, 2), np.nan), np.zeros((2, 2)), seed=1)


if __name__ == "__main__":
    unittest.main()

"""Known distributions distinguish marginal ranks from conditional accuracy."""

import json
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import binom, kstest, norm

from sbi4finance.hkde_diagnostics import conditional_metrics, randomized_sbc


def gaussian_reference(mean=(0., 0.), std=(1., 1.), correlation=0.):
    mean, std = np.asarray(mean), np.asarray(std)
    axes = tuple(np.linspace(m-8*s, m+8*s, 2049) for m, s in zip(mean, std, strict=True))
    probs = np.array([.05, .5, .95])
    return {"mean": mean, "std": std, "correlation": correlation,
            "probabilities": probs, "marginal_axes": axes,
            "marginal_cdfs": tuple(norm.cdf((axis-m)/s) for axis, m, s in zip(axes, mean, std, strict=True)),
            "target_quantiles": mean[:, None]+std[:, None]*norm.ppf(probs)}


class HKDEDiagnosticsTests(unittest.TestCase):
    def test_exchangeable_gaussian_ranks_coverage_and_replay(self):
        rng = np.random.default_rng(160)
        draws, truth = rng.normal(size=(1024, 256, 2)), rng.normal(size=(1024, 2))
        result = randomized_sbc(draws, truth, seed=161)
        self.assertEqual(result, randomized_sbc(draws, truth, seed=161))
        # A calibrated finite experiment can reject by chance. Check the
        # statistic independently, rather than choosing a seed that must pass.
        ranks = np.asarray(result["randomized_ranks"])
        assert_allclose(result["rank_ks"], [kstest(column, "uniform").statistic for column in ranks.T])
        self.assertEqual(result["rank_diagnostic_passed"],
                         max(result["rank_ks"]) <= result["rank_dkw_bound"])
        self.assertAlmostEqual(result["rank_dkw_bound"], np.sqrt(np.log(4/.05)/(2*1024)))
        for coverage, (low, high) in zip(result["coverage_90"], result["coverage_90_wilson_interval"], strict=True):
            self.assertLess(abs(coverage-.9), .03)
            self.assertLessEqual(low, coverage)
            self.assertGreaterEqual(high, coverage)
        json.dumps(result, allow_nan=False)

    def test_ties_are_randomized_across_the_complete_rank_range(self):
        result = randomized_sbc(np.zeros((2048, 32, 2)), np.zeros((2048, 2)), seed=163)
        self.assertTrue(result["rank_diagnostic_passed"])
        ranks = np.asarray(result["ranks"])
        self.assertEqual(ranks.min(), 0)
        self.assertEqual(ranks.max(), 32)
        self.assertEqual(result["coverage_90"], [1., 1.])

    def test_prior_copy_can_pass_sbc_and_fail_conditional_comparison(self):
        rng = np.random.default_rng(164)
        prior = rng.normal(size=(1024, 128, 2))
        truth = rng.normal(size=(1024, 2))
        self.assertTrue(randomized_sbc(prior, truth, seed=165)["rank_diagnostic_passed"])
        samples = prior[:32].reshape(-1, 2)
        metrics = conditional_metrics(samples, gaussian_reference(mean=(2., -2.), std=(.2, .2)), samples)
        self.assertGreater(np.min(np.abs(metrics["mean_error_in_ref_sd"])), 9.)
        self.assertGreater(np.min(metrics["relative_std_error"]), 3.)

    def test_gaussian_conditional_moments_mass_and_quantile_uncertainty(self):
        rng = np.random.default_rng(166)
        draws = rng.normal(size=(32768, 2))
        result = conditional_metrics(draws, gaussian_reference(), draws)
        assert_allclose(result["mean_error_in_ref_sd"], [0, 0], atol=.02)
        assert_allclose(result["relative_std_error"], [0, 0], atol=.02)
        assert_allclose(result["reference_mass_of_90_interval"], [.9, .9], atol=.012)
        assert_allclose(result["mean_mcse_in_ref_sd"], np.std(draws, axis=0, ddof=1)/np.sqrt(len(draws)))
        assert_allclose(result["target_quantile_error_currency"], 0, atol=.035)
        for probability, (lower, upper) in zip([.05, .5, .95], result["target_quantile_mc95_order_indices"], strict=True):
            attained = binom.cdf(upper-1, len(draws), probability)-binom.cdf(lower-1, len(draws), probability)
            self.assertGreaterEqual(attained, .95)
        intervals = np.asarray(result["target_quantile_mc95_interval"])
        quantiles = np.asarray(result["target_quantiles"])
        self.assertTrue(np.all(intervals[..., 0] <= quantiles))
        self.assertTrue(np.all(intervals[..., 1] >= quantiles))
        json.dumps(result, allow_nan=False)

    def test_independent_marginals_do_not_recover_joint_dependence(self):
        rng = np.random.default_rng(167)
        draws = rng.multivariate_normal([0, 0], [[1, .85], [.85, 1]], size=8192)
        reference = gaussian_reference(correlation=.85)
        matched = conditional_metrics(draws, reference, draws)
        shuffled = draws.copy()
        rng.shuffle(shuffled[:, 1])
        broken = conditional_metrics(shuffled, reference, shuffled)
        assert_allclose(matched["neural_mean"], broken["neural_mean"], atol=1e-14)
        assert_allclose(matched["neural_std"], broken["neural_std"], atol=1e-14)
        assert_allclose(matched["reference_mass_of_90_interval"], broken["reference_mass_of_90_interval"])
        self.assertLess(abs(matched["correlation_error"]), .02)
        self.assertGreater(abs(broken["correlation_error"]), .8)

    def test_interval_mass_integrates_cdf_with_clamped_outer_endpoints(self):
        axis = np.linspace(0, 1, 129)
        reference = {"mean": [.5, .5], "std": [1/np.sqrt(12)]*2, "correlation": 0.,
                     "probabilities": [.05, .5, .95], "marginal_axes": (axis, axis),
                     "marginal_cdfs": (axis, axis), "target_quantiles": [[.05, .5, .95]]*2}
        draws = np.stack([np.linspace(-1, 2, 101), np.linspace(0, 1, 101)], axis=1)
        result = conditional_metrics(draws, reference, draws)
        assert_allclose(result["reference_mass_of_90_interval"], [1., .9], atol=1e-14)
        assert_allclose(result["parameter_intervals_90"], [[-.85, 1.85], [.05, .95]], atol=1e-14)

    def test_invalid_inputs_and_unbounded_small_sample_quantile_intervals(self):
        draws = np.arange(6, dtype=float).reshape(3, 2)
        result = conditional_metrics(draws, gaussian_reference(), draws)
        self.assertIsNone(result["target_quantile_mc95_interval"][0][0][0])
        self.assertIsNone(result["target_quantile_mc95_interval"][0][-1][1])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            randomized_sbc(np.zeros((2, 4, 3)), np.zeros((2, 3)), seed=1)
        with self.assertRaises(ValueError):
            randomized_sbc(np.full((2, 4, 2), np.nan), np.zeros((2, 2)), seed=1)
        with self.assertRaises(ValueError):
            conditional_metrics(np.zeros((4, 2)), gaussian_reference(), np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            conditional_metrics(draws, gaussian_reference(), draws, probabilities=(.1, .5, .9))


if __name__ == "__main__":
    unittest.main()

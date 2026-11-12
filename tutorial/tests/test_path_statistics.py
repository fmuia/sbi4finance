"""Ordering and loss statistics must describe paths, including undefined cases."""

import unittest

import numpy as np

from sbi4f_tutorial.shared import path_statistics


class PathStatisticsTests(unittest.TestCase):
    def test_same_returns_and_old_summaries_can_have_different_drawdown(self):
        clustered = np.array([.02, .02, .02, -.03, -.03, -.03])
        alternating = np.array([.02, -.03, .02, -.03, .02, -.03])
        for summary in (np.sum, lambda x: np.sqrt(np.mean(x*x)), lambda x: np.max(np.abs(x))):
            self.assertAlmostEqual(summary(clustered), summary(alternating))
        actual = path_statistics.maximum_drawdown(np.stack([clustered, alternating]))
        np.testing.assert_allclose(actual, [1 - np.exp(-.09), 1 - np.exp(-.05)])

    def test_drawdown_includes_initial_wealth_and_agrees_with_direct_moderate_paths(self):
        self.assertAlmostEqual(path_statistics.maximum_drawdown([-.1]), -np.expm1(-.1))
        self.assertEqual(path_statistics.maximum_drawdown([.1, .2]), 0.)
        self.assertEqual(path_statistics.maximum_drawdown([0., 0.]), 0.)
        paths = np.random.default_rng(3).normal(0, .1, (12, 100))
        wealth = np.exp(np.column_stack([np.zeros(12), np.cumsum(paths, axis=1)]))
        direct = np.max(1 - wealth / np.maximum.accumulate(wealth, axis=1), axis=1)
        np.testing.assert_allclose(path_statistics.maximum_drawdown(paths), direct, atol=2e-16)

    def test_extreme_log_paths_do_not_require_finite_exponentiated_wealth(self):
        self.assertEqual(path_statistics.maximum_drawdown([1000., -1000.]), 1.)
        self.assertEqual(path_statistics.maximum_drawdown([-1000., 1000.]), 1.)
        self.assertEqual(path_statistics.maximum_drawdown([1000., 1000.]), 0.)
        with self.assertRaisesRegex(FloatingPointError, "cumulative log wealth overflowed"):
            path_statistics.maximum_drawdown([1e308, 1e308])

    def test_acf_uses_whole_prefix_mean_and_denominator(self):
        values = np.arange(21.)
        centered = values - np.mean(values)
        expected = [np.sum(centered[:-k] * centered[k:]) / np.sum(centered**2) for k in range(1, 5)]
        result = path_statistics.sample_acf(values)
        np.testing.assert_array_equal(result["lags"], [1, 2, 3, 4])
        np.testing.assert_allclose(result["values"][0], expected)
        self.assertEqual(result["available_count"], 1)
        large = path_statistics.sample_acf((values - 10) * 1e307)
        np.testing.assert_allclose(large["values"], result["values"], rtol=1e-14)

    def test_shuffling_clusters_changes_dependence_but_not_the_marginal_values(self):
        values = np.repeat([1., -1., 1., -1.], 25)
        shuffled = np.random.default_rng(9).permutation(values)
        np.testing.assert_array_equal(np.sort(values), np.sort(shuffled))
        a = path_statistics.sample_acf(values)
        b = path_statistics.sample_acf(shuffled)
        self.assertGreater(a["values"][0, 0], .9)
        self.assertLess(abs(b["values"][0, 0]), .3)

    def test_short_and_constant_statistics_are_unavailable_not_zero(self):
        short = path_statistics.sample_acf(np.arange(20.))
        self.assertEqual(short["values"].shape, (1, 0))
        self.assertFalse(short["available"][0])
        alternating = np.tile([-1., 1.], 15)
        rows = np.stack([np.zeros(30), alternating])
        signed = path_statistics.sample_acf(rows)
        absolute = path_statistics.sample_acf(rows, absolute=True)
        np.testing.assert_array_equal(signed["available"], [False, True])
        np.testing.assert_array_equal(absolute["available"], [False, False])
        self.assertTrue(np.isnan(signed["values"][0]).all())
        self.assertTrue(np.isnan(absolute["values"]).all())
        self.assertEqual(signed["count"], 2)

    def test_invalid_inputs_do_not_get_silently_filtered(self):
        for bad in ([], [np.nan], [np.inf], np.ones((2, 3, 1)), [1j],
                    np.ma.array([.1], mask=[True]), [".1"]):
            for function in (path_statistics.maximum_drawdown, path_statistics.sample_acf):
                with self.subTest(function=function.__name__, value=repr(bad)), self.assertRaises(ValueError):
                    function(bad)


if __name__ == "__main__":
    unittest.main()

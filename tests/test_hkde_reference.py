"""Independent probability examples and bank-pricer parity for the Q reference."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import truncnorm

from sbi4finance import hkde_reference as reference, option_bank


ROOT = Path(__file__).resolve().parents[1]


def task():
    return json.loads((ROOT / "configs/tasks/hkde_option_bank.json").read_text())["task"]


def toy_grid(bounds, n=257, price_map=lambda x: x):
    """Independent test fixture with explicit tensor trapezoid prior masses."""
    axes = tuple(np.linspace(low, high, n) for low, high in bounds)
    parameters = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    one_d = np.ones(n)
    one_d[[0, -1]] = 0.5
    one_d /= n - 1
    return {"axes": axes, "parameters": parameters, "prices": price_map(parameters),
            "quadrature_weights": np.outer(one_d, one_d).ravel(),
            "target_prices": np.column_stack((parameters.sum(axis=1), parameters[:, 0] - parameters[:, 1]))}


class HKDEReferenceTests(unittest.TestCase):
    def test_broadcast_prices_preserve_bank_refined_labels_and_contract_order(self):
        specification = task()
        rows = np.array([[.025, .25], [.045, .625], [.065, 1.]])
        actual = reference.price_parameters(specification, rows)
        expected = []
        for row in rows:
            parameters = dict(specification["fixed_parameters"], **dict(zip(specification["parameter_names"], row)))
            expected.append(option_bank.price_panel(specification, parameters, seed=0)[0])
        assert_allclose(actual, expected, rtol=0, atol=2e-13)
        mixed = copy.deepcopy(specification["context"]["contracts"][:3])
        mixed[1]["option_type"] = "call"
        result = reference.price_parameters(specification, rows, mixed)
        assert_allclose(result[:, [0, 2]], actual[:, [0, 2]], atol=2e-13)
        maturity = mixed[1]["maturity"]
        parity = specification["context"]["spot"] - mixed[1]["strike"] * np.exp(-.02 * maturity)
        assert_allclose(result[:, 1] - actual[:, 1], parity, atol=2e-13)

    def test_grid_prior_weights_endpoints_and_targets(self):
        specification = task()
        # A tiny real grid exercises the same public boundary used by larger runs.
        grid = reference.build_grid(specification, 3)
        self.assertEqual(grid["prices"].shape, (9, 9))
        self.assertEqual(grid["target_prices"].shape, (9, 2))
        self.assertEqual(grid["target_contracts"], list(reference.DEFAULT_TARGET_CONTRACTS))
        assert_allclose(grid["quadrature_weights"].sum(), 1, atol=1e-15)
        assert_allclose(grid["quadrature_weights"][[0, 4]], [1/16, 1/4])
        assert_array_equal(grid["parameters"][[0, -1]], np.asarray(specification["prior_bounds"]).T)

    def test_constant_likelihood_recovers_continuous_uniform_cdf(self):
        grid = toy_grid([[-2, 4], [1, 5]], n=17, price_map=lambda x: np.zeros((len(x), 2)))
        weights = reference.posterior(grid, [.1, -.2], .3)
        assert_allclose(weights, grid["quadrature_weights"], atol=1e-16)
        summary = reference.summarize(grid, weights, probs=[0, .05, .5, .95, 1])
        assert_allclose(summary["mean"], [1, 3], atol=1e-14)
        assert_allclose(summary["marginal_quantiles"], [[-2, -1.7, 1, 3.7, 4], [1, 1.2, 3, 4.8, 5]], atol=1e-14)
        for axis, density, cdf in zip(summary["marginal_axes"], summary["marginal_densities"], summary["marginal_cdfs"]):
            assert_allclose(density, 1 / (axis[-1] - axis[0]), atol=1e-14)
            assert_allclose(cdf, (axis - axis[0]) / (axis[-1] - axis[0]), atol=1e-14)
        self.assertAlmostEqual(summary["correlation"], 0)

    def test_rectangular_truncated_gaussian_matches_analytic_marginals(self):
        bounds = np.array([[-1., 2.], [-2., 1.]])
        location, sigma = np.array([.15, -.35]), .6
        grid = toy_grid(bounds)
        weights = reference.posterior(grid, location[:, None], sigma)
        assert_allclose(weights.sum(), 1, atol=1e-14)
        summary = reference.summarize(grid, weights)
        analytic = truncnorm((bounds[:, 0] - location) / sigma,
                             (bounds[:, 1] - location) / sigma, loc=location, scale=sigma)
        assert_allclose(summary["mean"], analytic.mean(), atol=8e-6)
        assert_allclose(summary["std"], analytic.std(), atol=8e-6)
        expected = analytic.ppf(np.array([.05, .5, .95])[:, None]).T
        assert_allclose(summary["marginal_quantiles"], expected, atol=8e-5)
        self.assertAlmostEqual(summary["correlation"], 0, places=12)

    def test_joint_dependence_is_retained_in_target_prices(self):
        # One observed sum creates a negative-correlation ridge. Independent
        # marginals would give the wrong uncertainty for that same sum target.
        grid = toy_grid([[-2, 2], [-2, 2]], n=129,
                        price_map=lambda x: x.sum(axis=1, keepdims=True))
        weights = reference.posterior(grid, [0], .1)
        summary = reference.summarize(grid, weights)
        self.assertLess(summary["correlation"], -.99)
        self.assertLess(summary["target_std"][0], .11)
        independent_sum_std = np.sqrt(np.sum(summary["std"] ** 2))
        self.assertGreater(independent_sum_std, 1.5)
        assert_allclose(summary["target_std"][0] ** 2,
                        summary["covariance"].sum(), atol=2e-14)

    def test_empirical_target_quantile_convention_and_zero_weights(self):
        assert_array_equal(reference.weighted_quantiles([1, 10, 20, 30], [0, .2, .3, .5],
                                                       [0, .2, .21, .5, .99, 1]), [10, 10, 20, 20, 30, 30])
        result = reference.weighted_quantiles([[1, 3], [2, 2], [3, 1]], [1, 2, 1], [.25, .5, .75])
        assert_array_equal(result, [[1, 2, 2], [1, 2, 2]])

    def test_likelihood_normalization_survives_tiny_absolute_density(self):
        grid = toy_grid([[-1, 1], [-1, 1]], n=9,
                        price_map=lambda x: np.zeros((len(x), 2)))
        # exp(-1e6) underflows, but a common log likelihood cancels exactly.
        weights = reference.posterior(grid, [1000, -1000], 1)
        assert_allclose(weights, grid["quadrature_weights"], rtol=1e-9, atol=1e-12)
        assert_allclose(weights.sum(), 1, atol=1e-9)
        with self.assertRaisesRegex(ValueError, "normalization"):
            reference.posterior(grid, [1e300, 1e300], 1)

    def test_invalid_task_rows_grid_and_observations_fail_explicitly(self):
        specification = task()
        for rows in ([.04, .5], [[.04, .5, 2]], [[.01, .5]], [[np.nan, .5]], []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                reference.price_parameters(specification, rows)
        wrong = copy.deepcopy(specification)
        wrong["parameter_names"].reverse()
        wrong["prior_bounds"].reverse()
        with self.assertRaisesRegex(ValueError, "parameter order"):
            reference.build_grid(wrong, 3)
        for n in (True, 2, 3.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                reference.build_grid(specification, n)
        grid = toy_grid([[-1, 1], [-1, 1]], n=9)
        for observation in ([0], [[0, 0]], [0, np.inf]):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                reference.posterior(grid, observation, .2)
        for sigma in (0, -1, True, [1], np.inf):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                reference.posterior(grid, [0, 0], sigma)
        weights = reference.posterior(grid, [0, 0], .2)
        invalid = dict(grid, parameters=grid["parameters"][::-1])
        with self.assertRaisesRegex(ValueError, "C-order"):
            reference.summarize(invalid, weights)
        for probs in ([], [-.1], [1.1], [np.nan]):
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                reference.summarize(grid, weights, probs)
        with self.assertRaises(ValueError):
            reference.weighted_quantiles([1, 2], [0, 0])


if __name__ == "__main__":
    unittest.main()

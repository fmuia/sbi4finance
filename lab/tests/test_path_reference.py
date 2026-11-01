"""Exact limiting-model tests for the specialized path reference."""

import unittest
import importlib.util
from pathlib import Path

import numpy as np

from sbi4finance.path_inference.reference import (
    _filter, compare_posteriors, posterior_summary, reference_posterior, restricted_task,
)


class PathReferenceTests(unittest.TestCase):
    def test_single_return_has_no_information_about_long_run_variance(self):
        task = restricted_task(1)
        result = reference_posterior([0.01], 1 / 252, task, grid_size=24, particles=8, seed=71)
        _, weights = np.polynomial.legendre.leggauss(24)
        np.testing.assert_allclose(result["grid_probabilities"], weights / 2, rtol=1e-13)
        analytic_mean = (0.08 - 0.02) / np.log(4)
        self.assertAlmostEqual(posterior_summary(result)["mean"], analytic_mean, places=14)

    def test_deterministic_variance_likelihood_matches_gaussian_product(self):
        task = restricted_task(4)
        task["parameters"]["vol_of_vol"] = 0.0
        returns = np.array([0.001, -0.003, 0.015, -0.008])
        result = reference_posterior(returns, 1 / 252, task, grid_size=8, particles=5, seed=72)
        dt = 1 / 252
        for i, theta in enumerate(result["theta_grid"]):
            state, log_likelihood = 0.04, 0.0
            for r in returns:
                var = max(state, 0)
                residual = r - (0.05 - 0.5 * var) * dt
                log_likelihood += -0.5 * (np.log(2 * np.pi * var * dt) + residual ** 2 / (var * dt))
                state += 2 * (theta - var) * dt
            self.assertAlmostEqual(result["log_likelihood"][i], log_likelihood, places=12)
            np.testing.assert_allclose(result["terminal_particles"][i], state, rtol=1e-13)

    def test_zero_variance_return_atom_has_mass_one(self):
        dt = np.full(2, 1 / 252)
        # First state is zero; second is kappa*theta*dt, so its return is Gaussian.
        returns = np.array([0.05 * dt[0], 0.001])
        result = _filter(returns, dt, 0.04, 0.0, 0.05, 2.0, 0.0, -0.6,
                         np.full(2, 0.5), np.zeros((2, 4)))
        var = 2 * 0.04 * dt[0]
        residual = returns[1] - (0.05 - var / 2) * dt[1]
        expected = -0.5 * (np.log(2 * np.pi * var * dt[1]) + residual ** 2 / (var * dt[1]))
        self.assertEqual(result[-1], 0)
        self.assertAlmostEqual(result[0], expected, places=12)

    def test_incompatible_atom_does_not_get_variance_floor(self):
        result = _filter(np.array([0.0]), np.array([1 / 252]), 0.04, 0.0,
                         0.05, 2.0, 0.2, -0.6, np.array([0.5]), np.zeros((1, 4)))
        self.assertEqual(result[-1], 1)
        self.assertEqual(result[0], -np.inf)

    def test_target_scope_is_checked(self):
        task = restricted_task(2)
        task["substeps"] = 4
        with self.assertRaisesRegex(ValueError, "one-substep"):
            reference_posterior([0.0, 0.01], 1 / 252, task, particles=4, grid_size=4)
        task = restricted_task(2)
        task["prior"] = "uniform"
        with self.assertRaisesRegex(ValueError, "log_uniform"):
            reference_posterior([0.0, 0.01], 1 / 252, task, particles=4, grid_size=4)

    def test_comparison_uses_physical_discrete_wasserstein(self):
        first = {"theta_grid": np.array([0.02, 0.04]), "grid_probabilities": np.array([0.5, 0.5])}
        second = {"theta_grid": np.array([0.03, 0.05]), "grid_probabilities": np.array([0.5, 0.5])}
        comparison = compare_posteriors(first, second)
        self.assertAlmostEqual(comparison["wasserstein1_discrete"], 0.01)
        self.assertAlmostEqual(comparison["absolute_mean_difference"], 0.01)

    def test_repeat_seed_reproduces_full_reference(self):
        task = restricted_task(4)
        kwargs = {"grid_size": 4, "particles": 16, "seed": 73}
        a = reference_posterior([0.001, -0.003, 0.015, -0.008], 1 / 252, task, **kwargs)
        b = reference_posterior([0.001, -0.003, 0.015, -0.008], 1 / 252, task, **kwargs)
        np.testing.assert_array_equal(a["log_likelihood"], b["log_likelihood"])
        np.testing.assert_array_equal(a["terminal_particles"], b["terminal_particles"])

    def test_refinement_coupling_preserves_brownian_sum(self):
        source = Path(__file__).resolve().parents[1] / "scripts" / "check_path_numerics.py"
        spec = importlib.util.spec_from_file_location("path_numerics_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(78)
        fine = rng.normal(size=(3, 7, 8, 2))
        for steps in (1, 4, 8):
            coarse = module.coarsen_brownian(fine, steps)
            np.testing.assert_allclose(coarse.sum(axis=2) / np.sqrt(steps),
                                       fine.sum(axis=2) / np.sqrt(8), rtol=1e-13, atol=1e-13)

    def test_rough_discrete_compensator_is_own_covariance(self):
        from sbi4finance.rough_bergomi import hybrid_covariance, rough_driver, rough_covariance
        innovations = np.zeros((1, 12, 3))
        _, discrete_variance = rough_driver(0.1, 1 / 1008, innovations)
        hybrid, hybrid_cross = hybrid_covariance(0.1, 1 / 1008, 12)
        continuum, continuum_cross = rough_covariance(0.1, 1 / 1008, 12)
        np.testing.assert_allclose(discrete_variance, hybrid.diagonal(), rtol=1e-13)
        np.testing.assert_allclose(hybrid_cross, continuum_cross, rtol=1e-13, atol=1e-13)
        self.assertGreater(np.max(np.abs(discrete_variance - continuum.diagonal())), 1e-5)


if __name__ == "__main__":
    unittest.main()

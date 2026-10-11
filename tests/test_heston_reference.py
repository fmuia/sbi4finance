"""Analytic limiting cases for the same-discrete-model particle reference."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance import heston
from sbi4finance.heston_forecast import simulate_futures
from sbi4finance.heston_reference import reference_posterior


class HestonReferenceTests(unittest.TestCase):
    def setUp(self):
        self.task = json.loads((Path(__file__).resolve().parents[1]
                               / "configs/tasks/heston_inference.json").read_text())["task"]

    def test_first_observation_preserves_theta_prior_and_conditions_signed_state(self):
        task = deepcopy(self.task)
        task["n_observations"] = 1
        observed = 0.012
        ref = reference_posterior([observed], task["dt_years"], task,
                                  grid_size=24, particles=8192, seed=921)
        theta, probabilities = ref["theta_grid"], ref["grid_probabilities"]
        low, high = task["theta_bounds"]
        prior_mean = (high - low) / np.log(high / low)
        prior_second = (high**2 - low**2) / (2 * np.log(high / low))
        assert_allclose(probabilities @ theta, prior_mean, atol=1e-14)
        assert_allclose(probabilities @ theta**2, prior_second, atol=1e-14)
        assert_allclose(ref["log_likelihood"], ref["log_likelihood"][0], atol=1e-13)

        p = task["parameters"]
        initial, dt = task["initial_variance"], task["dt_years"]
        residual = observed - (p["drift"] - initial / 2) * dt
        expected_mean = initial + p["kappa"] * (theta - initial) * dt + p["vol_of_vol"] * p["rho"] * residual
        expected_std = p["vol_of_vol"] * np.sqrt(initial * dt * (1 - p["rho"]**2))
        particles = ref["terminal_particles"]
        assert_allclose(particles.mean(axis=1), expected_mean,
                        atol=5.5 * expected_std / np.sqrt(particles.shape[1]), rtol=0)
        assert_allclose(particles.std(axis=1, ddof=1), expected_std,
                        atol=5.5 * expected_std / np.sqrt(2 * (particles.shape[1] - 1)), rtol=0)

    def test_deterministic_variance_reference_matches_direct_likelihood(self):
        task = deepcopy(self.task)
        task["n_observations"] = 12
        task["parameters"]["vol_of_vol"] = 0.0
        returns = np.linspace(-0.017, 0.021, task["n_observations"])
        ref = reference_posterior(returns, task["dt_years"], task, grid_size=32, particles=8)
        p, dt = task["parameters"], task["dt_years"]
        variance = np.full(len(ref["theta_grid"]), task["initial_variance"])
        log_likelihood = np.zeros(len(variance))
        for observed in returns:
            mean = (p["drift"] - variance / 2) * dt
            log_likelihood += -0.5 * (np.log(2 * np.pi * variance * dt)
                                     + (observed - mean)**2 / (variance * dt))
            variance += p["kappa"] * (ref["theta_grid"] - variance) * dt
        assert_allclose(ref["log_likelihood"], log_likelihood, atol=1e-12)
        assert_allclose(ref["terminal_particles"], np.broadcast_to(variance[:, None], (32, 8)), atol=1e-14)
        _, weights = np.polynomial.legendre.leggauss(32)
        expected = weights * np.exp(log_likelihood - log_likelihood.max())
        assert_allclose(ref["grid_probabilities"], expected / expected.sum(), atol=1e-14)

    def test_zero_variance_atom_is_not_a_floored_gaussian(self):
        task = deepcopy(self.task)
        task["n_observations"] = 1
        task["initial_variance"] = 0.0
        atom = task["parameters"]["drift"] * task["dt_years"]
        ref = reference_posterior([atom], task["dt_years"], task, grid_size=8, particles=8)
        assert_array_equal(ref["log_likelihood"], np.zeros(8))
        expected = task["parameters"]["kappa"] * ref["theta_grid"] * task["dt_years"]
        assert_allclose(ref["terminal_particles"], np.broadcast_to(expected[:, None], (8, 8)))
        self.assertEqual(ref["diagnostics"]["atom_observations"], 1)
        with self.assertRaisesRegex(RuntimeError, "no compatible"):
            reference_posterior([np.nextafter(atom, np.inf)], task["dt_years"], task,
                                grid_size=8, particles=8)
        task["initial_variance"] = 0.04
        with self.assertRaisesRegex(RuntimeError, "no compatible"):
            reference_posterior([atom], task["dt_years"], task, grid_size=8, particles=8)

    def test_reference_rejects_other_models_and_time_designs(self):
        returns = np.zeros(self.task["n_observations"])
        for name, value in [("substeps", 8), ("measure", "Q"), ("prior", "uniform")]:
            task = {**self.task, name: value}
            with self.subTest(name=name), self.assertRaises(ValueError):
                reference_posterior(returns, task["dt_years"], task, grid_size=4, particles=8)
        with self.assertRaises(ValueError):
            reference_posterior(returns, self.task["dt_years"] * 2, self.task, grid_size=4, particles=8)
        task = deepcopy(self.task)
        task["parameters"]["jump_intensity"] = 0.1
        with self.assertRaises(ValueError):
            reference_posterior(returns, task["dt_years"], task, grid_size=4, particles=8)

    def test_forecast_preserves_pairs_and_signed_continuation(self):
        pairs = np.array([[0.025, -0.001], [0.06, 0.07]])
        forecast = simulate_futures(pairs, self.task, steps=3, seed=321)
        dt = np.full(3, self.task["dt_years"])
        inputs = heston.draw_innovations(dt, 1, seed=321, example_ids=[0, 1], stream_id=91)
        expected = heston.simulate({**self.task["parameters"], "theta": pairs[:, 0]},
                                   dt, pairs[:, 1], inputs)
        for name in ("log_returns", "variance_state"):
            assert_array_equal(forecast[name], expected[name])
        assert_array_equal(forecast["variance_state"][:, 0, 0], pairs[:, 1])
        assert_allclose(forecast["relative_loss"], 1 - np.exp(expected["log_returns"].sum(axis=1)[:, 0]))
        broken_pairs = pairs.copy()
        broken_pairs[:, 1] = pairs[::-1, 1]
        shuffled = simulate_futures(broken_pairs, self.task, steps=3, seed=321)
        self.assertFalse(np.array_equal(shuffled["log_returns"], forecast["log_returns"]))
        wrong_task = deepcopy(self.task)
        wrong_task["parameters"]["jump_intensity"] = 1.0
        with self.assertRaisesRegex(ValueError, "no-jump"):
            simulate_futures(pairs, wrong_task)


if __name__ == "__main__":
    unittest.main()

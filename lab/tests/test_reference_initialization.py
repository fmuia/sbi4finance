"""Transformed target, curvature, finite budget and supported-start coverage."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expit, log_expit, logsumexp

from sbi4finance.calibration import reference, reference_initialization as initialization


class InitializationMathTests(unittest.TestCase):
    def test_uniform_density_and_logit_jacobian_are_included_once(self):
        bounds = np.array([[-3.0, 5.0], [2.0, 8.0]])
        objective = initialization.transformed_objective(lambda _: 0.0, bounds)
        for z in (np.array([0.0, 0.0]), np.array([-2.0, 1.5])):
            jacobian = np.sum(np.log(bounds[:, 1] - bounds[:, 0]) + log_expit(z) + log_expit(-z))
            uniform = -np.log(bounds[:, 1] - bounds[:, 0]).sum()
            self.assertAlmostEqual(objective(z), -(uniform + jacobian))

    def test_gaussian_logit_curvature_matches_analytic_hessian(self):
        bounds = np.array([[-4.0, 4.0], [-4.0, 4.0]])
        physical_covariance = np.array([[0.25, 0.18], [0.18, 0.36]])
        precision = np.linalg.inv(physical_covariance)
        objective = initialization.transformed_objective(lambda x: -0.5 * x @ precision @ x, bounds)
        curvature = initialization.local_curvature(objective, np.zeros(2))
        expected = 4 * precision + 0.5 * np.eye(2)  # dtheta/dz=2 at z=0, plus Uniform transform curvature.
        assert_allclose(curvature["hessian"], expected, rtol=3e-5, atol=1e-6)
        assert_allclose(curvature["covariance"], np.linalg.inv(expected), rtol=4e-5)
        self.assertFalse(curvature["regularization_applied"])

    def test_indefinite_curvature_regularizes_only_the_proposal(self):
        curvature = initialization.local_curvature(lambda z: z[0]**2 - z[1]**2, np.zeros(2))
        self.assertTrue(curvature["regularization_applied"])
        eigenvalues = np.linalg.eigvalsh(curvature["covariance"])
        self.assertTrue(np.all(eigenvalues > 0))
        self.assertTrue(np.all(eigenvalues <= 4))
        assert_allclose(curvature["raw_hessian_eigenvalues"], [-2, 2])

    def test_multistart_budget_is_enforced_and_starts_are_deterministic(self):
        log_likelihood = lambda x: -0.5 * np.sum((x / 0.5)**2)
        kwargs = dict(fit_starts=2, fit_budget=160)
        first = reference.data_fit_initialization(log_likelihood, [[-4, 4]], 4, 57, **kwargs)
        second = reference.data_fit_initialization(log_likelihood, [[-4, 4]], 4, 57, **kwargs)
        self.assertLessEqual(first["optimizer_evaluations"], 160)
        self.assertTrue(all(f["function_evaluations"] <= f["max_function_evaluations"] for f in first["fits"]))
        assert_allclose(first["chain_starts"], second["chain_starts"], rtol=0, atol=0)
        self.assertEqual(len({tuple(x) for x in first["chain_starts"]}), 4)
        self.assertFalse(first["optimizer_box_limits_sampling_support"])
        self.assertEqual(first["proposal"]["distribution"], "PyMC MultivariateNormalProposal")

    def test_separated_supported_fit_locations_are_kept_in_chain_starts(self):
        bounds = np.array([[-4.0, 4.0]])

        def likelihood(theta):
            return float(logsumexp(-0.5 * ((theta[0] - np.array([-2.0, 2.0])) / .2)**2))

        objective = initialization.transformed_objective(likelihood, bounds)
        locations = [-np.log(3), np.log(3)]
        fit = iter(locations)

        def fitted(fn, x0, **kwargs):
            z = np.array([next(fit)])
            return SimpleNamespace(x=z, fun=objective(z), success=True, message="declared separated test fits")

        with patch.object(initialization, "minimize", side_effect=fitted):
            result = initialization.data_fit_initialization(likelihood, bounds, 4, 71, fit_starts=2, fit_budget=100)
        self.assertEqual(len(result["separated_supported_fit_locations"]), 2)
        self.assertEqual(set(d["fit"] for d in result["chain_start_details"]), {0, 1})
        starts = np.asarray(result["chain_starts"])[:, 0]
        self.assertTrue(np.any(starts < 0) and np.any(starts > 0))

    def test_random_or_unpriced_target_is_not_hidden(self):
        sequence = iter(range(100))
        with self.assertRaisesRegex(ValueError, "deterministic"):
            initialization.data_fit_initialization(lambda _: float(next(sequence)), [[-4, 4]], 2, 7,
                                                     fit_starts=2, fit_budget=100)
        with self.assertRaisesRegex(ValueError, "pricing failed"):
            initialization.data_fit_initialization(lambda _: (_ for _ in ()).throw(ValueError("pricing failed")),
                                                     [[-4, 4]], 2, 7, fit_starts=2, fit_budget=100)


if __name__ == "__main__":
    unittest.main()

"""Target/Jacobian and pseudo-marginal state-retention tests."""

import unittest

import numpy as np

from sbi4finance.path_inference.bank import default_task
from sbi4finance.path_inference.pmmh import (
    _bootstrap_filter, LikelihoodEstimate, log_prior_in_logits, particle_log_likelihood,
    physical_to_prior_logit, prior_logit_to_physical, run_chain, validate_reference_task,
)


class PathPMMHTests(unittest.TestCase):
    def test_prior_cdf_logit_roundtrip_and_density_jacobian(self):
        task = default_task("heston")
        z = np.array([-2.0, 0.7, -0.3, 1.9])
        values = prior_logit_to_physical(z, task)
        np.testing.assert_allclose(physical_to_prior_logit(values, task), z, atol=1e-13)
        log_density_and_jacobian = 0.0
        for j, ((low, high), kind) in enumerate(zip(task["prior_bounds"], task["prior_types"])):
            delta = np.eye(4)[j] * 1e-5
            derivative = (prior_logit_to_physical(z + delta, task)[j]
                          - prior_logit_to_physical(z - delta, task)[j]) / 2e-5
            density = (1 / (values[j] * np.log(high / low)) if kind == "log_uniform"
                       else 1 / (high - low))
            log_density_and_jacobian += np.log(density) + np.log(derivative)
        self.assertAlmostEqual(log_density_and_jacobian, log_prior_in_logits(z), places=9)

    def test_rejection_retains_estimate_and_accepted_seed_without_refresh(self):
        calls = []

        def estimator(z, seed):
            calls.append((z.copy(), seed))
            return LikelihoodEstimate(0.0 if len(calls) == 1 else -np.inf, seed,
                                      minimum_ess=3.0, minimum_compatible=4,
                                      collapse=len(calls) > 1)

        result = run_chain(estimator, np.zeros(4), tune=3, draws=5, seed=42)
        self.assertEqual(len(calls), 1 + 3 + 5)
        for phase in ("tune", "draws"):
            np.testing.assert_array_equal(result[phase]["z"], 0.0)
            np.testing.assert_array_equal(result[phase]["log_likelihood_estimate"], 0.0)
            np.testing.assert_array_equal(result[phase]["likelihood_seed"], calls[0][1])
            self.assertFalse(result[phase]["accepted"].any())
            self.assertTrue(result[phase]["proposal_collapsed"].all())

    def test_adaptation_stops_before_retained_samples(self):
        result = run_chain(lambda z, seed: LikelihoodEstimate(-np.sum(z ** 2) / 2, seed),
                           np.zeros(4), tune=20, draws=30, seed=43)
        scales = result["draws"]["proposal_scale"]
        np.testing.assert_array_equal(scales, scales[0])
        self.assertEqual(scales[0], result["final_proposal_scale"])

    def test_seed_reproduces_chain_and_accepted_stochastic_states(self):
        def estimator(z, seed):
            rng = np.random.default_rng(seed)
            noise = rng.normal(scale=0.2)
            return LikelihoodEstimate(-np.sum(z ** 2) / 2 + noise - 0.02, seed)

        a = run_chain(estimator, np.zeros(4), tune=5, draws=20, seed=44)
        b = run_chain(estimator, np.zeros(4), tune=5, draws=20, seed=44)
        for key in a["draws"]:
            np.testing.assert_array_equal(a["draws"][key], b["draws"][key])
        self.assertEqual(a["final_rng_state"], b["final_rng_state"])

    def test_first_observation_likelihood_is_exact_gaussian(self):
        task = default_task("heston")
        task["numerics"]["substeps"] = 1
        dt, variance, observed = task["dt_years"], task["fixed"]["initial_variance"], 0.001
        expected = -0.5 * (np.log(2 * np.pi * variance * dt)
                          + (observed - (0.05 - variance / 2) * dt) ** 2 / (variance * dt))
        for z in (np.zeros(4), np.array([-1.5, 1.2, -0.7, 1.6])):
            estimate = particle_log_likelihood(np.array([observed]), task,
                                               prior_logit_to_physical(z, task), particles=16, seed=45)
            self.assertAlmostEqual(estimate.log_value, expected, places=12)

    def test_rejects_unsupported_reference_law_and_nonfinite_initialization(self):
        task = default_task("rough-known")
        with self.assertRaisesRegex(ValueError, "four-parameter Heston"):
            validate_reference_task(task)
        with self.assertRaisesRegex(RuntimeError, "initial likelihood"):
            run_chain(lambda z, seed: LikelihoodEstimate(-np.inf, seed), np.zeros(4), draws=1)

    def test_eight_substep_filter_matches_constant_variance_gaussian_limit(self):
        task = default_task("heston")
        task["numerics"]["substeps"] = 8
        returns = np.array([0.001, -0.003, 0.007, -0.012])
        dt, variance = task["dt_years"], task["fixed"]["initial_variance"]
        physical = np.array([2.0, variance, 0.0, 0.0])
        expected = np.sum(-0.5 * (np.log(2 * np.pi * variance * dt)
                                 + (returns - (0.05 - variance / 2) * dt) ** 2 / (variance * dt)))
        estimate = particle_log_likelihood(returns, task, physical, particles=16, seed=47)
        self.assertFalse(estimate.collapse)
        self.assertAlmostEqual(estimate.log_value, expected, places=12)

    def test_one_step_bootstrap_and_adapted_likelihood_expectations_agree(self):
        task = default_task("heston")
        task["numerics"]["substeps"] = 1
        returns = np.array([0.001, -0.003, 0.007, -0.012])
        physical = np.array([2.0, 0.04, 0.2, -0.6])
        estimates = []
        for method in ("adapted", "bootstrap"):
            estimates.append(np.array([particle_log_likelihood(returns, task, physical,
                particles=128, seed=1000 + seed, method=method).log_value for seed in range(128)]))
        origin = max(x.max() for x in estimates)
        likelihoods = [np.exp(x - origin) for x in estimates]
        difference = abs(likelihoods[0].mean() - likelihoods[1].mean())
        standard_error = np.sqrt(sum(x.var(ddof=1) / len(x) for x in likelihoods))
        self.assertLess(difference, 5 * standard_error)

    def test_substep_return_atom_uses_exact_discrete_sum_and_signed_state(self):
        dt, substeps, drift = 1 / 252, 8, 0.05
        atom = 0.0
        for step in range(substeps):
            atom += drift * (dt / substeps)
        result = _bootstrap_filter(np.array([atom, atom]), dt, substeps,
                                   0.04, -1.0, drift, 2.0, 0.2, -0.6, 8, 48)
        self.assertEqual(result[-1], 0)
        self.assertEqual(result[0], 0.0)
        np.testing.assert_allclose(result[1], -1 + 2 * 2 * 0.04 * dt, rtol=1e-13)
        incompatible = _bootstrap_filter(np.array([atom]), dt, substeps,
                                         0.04, 0.04, drift, 2.0, 0.2, -0.6, 8, 49)
        self.assertEqual(incompatible[-1], 1)
        self.assertEqual(incompatible[0], -np.inf)


if __name__ == "__main__":
    unittest.main()

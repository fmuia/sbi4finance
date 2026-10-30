"""Target/Jacobian and pseudo-marginal state-retention tests."""

import unittest
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from sbi4finance.path_inference.bank import default_task, task_fingerprint, write_bank
from sbi4finance.path_inference.pmmh import (
    _bootstrap_filter, LikelihoodEstimate, load_initial_checkpoint, log_prior_in_logits,
    particle_log_likelihood, physical_to_prior_logit, prior_logit_to_physical,
    proposal_cholesky, regularized_covariance, run_chain, validate_reference_task,
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

    def test_preconditioned_symmetric_proposal_recovers_gaussian_ridge(self):
        covariance = np.array([[1.0, 0.95], [0.95, 1.0]])
        precision = np.linalg.inv(covariance)

        def estimator(z, seed):
            # Cancel the transformed logistic prior to give a known Gaussian
            # target, testing the full MH kernel with a correlated proposal.
            return LikelihoodEstimate(-0.5 * z @ precision @ z - log_prior_in_logits(z), seed)

        result = run_chain(estimator, np.zeros(2), tune=500, draws=6000, seed=73,
                           proposal_scale=1.0, adapt=False, proposal_covariance=covariance)
        z = result["draws"]["z"]
        np.testing.assert_allclose(z.mean(axis=0), np.zeros(2), atol=0.12)
        np.testing.assert_allclose(np.cov(z, rowvar=False), covariance, atol=0.12)
        np.testing.assert_allclose(result["proposal_covariance"], covariance, atol=1e-14)
        delta = np.array([0.2, -0.4])
        self.assertEqual(float(delta @ precision @ delta), float((-delta) @ precision @ (-delta)))

    def test_proposal_rejects_invalid_matrices_and_regularizes_singular_pool(self):
        for matrix in (np.array([[1, 2], [2, 1]]), np.array([[1, 0], [1, 1]]), np.eye(3)):
            with self.assertRaises(ValueError):
                proposal_cholesky(matrix, 2)
        covariance, info = regularized_covariance(np.zeros((10, 4)))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        self.assertEqual(info["pooled_samples"], 10)

    def test_supplied_accepted_estimate_is_not_refreshed_with_new_geometry(self):
        accepted = LikelihoodEstimate(1.25, 98765, minimum_ess=10.0, minimum_compatible=11)
        calls = []

        def estimator(z, seed):
            calls.append(seed)
            return LikelihoodEstimate(-np.inf, seed, collapse=True)

        covariance = np.array([[1, 0.8], [0.8, 1]])
        result = run_chain(estimator, np.array([0.3, -0.4]), tune=2, draws=3, seed=74,
                           initial_estimate=accepted, proposal_covariance=covariance)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["initial"]["likelihood_seed"], 98765)
        self.assertEqual(result["initial"]["log_likelihood_estimate"], 1.25)
        for phase in ("tune", "draws"):
            np.testing.assert_array_equal(result[phase]["likelihood_seed"], 98765)
            np.testing.assert_array_equal(result[phase]["log_likelihood_estimate"], 1.25)

    def test_warm_start_requires_task_observation_count_and_consistent_state(self):
        task = default_task("heston")
        task.update(n_observations=4, history_length_range=[1, 4], prefix_lengths=[1, 4])
        returns = np.array([0.001, -0.003, 0.007, -0.012])
        observation = {"task_sha256": task_fingerprint(task), "trajectory_id": "example:validation:0",
                       "length": 4, "source_shard_sha256": "ab" * 32, "split": "validation"}
        z = np.array([0.3, -0.4, 0.7, -1.2])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            np.savez(path / "observations.npz", log_returns=returns, dt_years=np.full(4, task["dt_years"]))
            np.savez(path / "draws.npz", z=z[None, None], log_likelihood_estimate=np.array([[1.25]]),
                     likelihood_seed=np.array([[98765]], dtype=np.uint64),
                     minimum_ess=np.array([[10.0]]), minimum_compatible=np.array([[11]]))
            report = {"task": task, "observation": observation, "particles": 512,
                      "chain_id": 0, "chain_seed": 8888,
                      "sampler_state": {"current_z": z.tolist(), "current_likelihood_seed": 98765,
                                        "current_log_likelihood_estimate": 1.25}}
            (path / "pmmh-summary.json").write_text(json.dumps(report))
            loaded_z, estimate, provenance = load_initial_checkpoint(path, task, returns, observation, particles=512)
            np.testing.assert_array_equal(loaded_z, z)
            self.assertEqual(estimate.seed, 98765)
            self.assertEqual(provenance["source_particles"], 512)
            with self.assertRaisesRegex(ValueError, "particle count"):
                load_initial_checkpoint(path, task, returns, observation, particles=1024)
            changed = copy.deepcopy(task)
            changed["fixed"]["drift"] += 0.01
            with self.assertRaisesRegex(ValueError, "task identity"):
                load_initial_checkpoint(path, changed, returns, observation, particles=512)
            with self.assertRaisesRegex(ValueError, "observation identity"):
                load_initial_checkpoint(path, task, returns, {**observation, "length": 3}, particles=512)
            with self.assertRaisesRegex(ValueError, "actual observations"):
                load_initial_checkpoint(path, task, returns + 0.001, observation, particles=512)
            report["sampler_state"]["current_likelihood_seed"] += 1
            (path / "pmmh-summary.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "states differ"):
                load_initial_checkpoint(path, task, returns, observation, particles=512)

    def test_eight_substep_cli_new_phase_preserves_accepted_estimate(self):
        lab = Path(__file__).resolve().parents[1]
        script = lab / "scripts/path_pmmh.py"
        environment = {**os.environ, "PYTHONPATH": str(lab / "src")}
        task = default_task("heston")
        task.update(n_observations=4, history_length_range=[1, 4], prefix_lengths=[1, 4])
        task["numerics"]["substeps"] = 8
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_bank(task, root / "bank", 0, 1, 0, chunk=1, seed=91)
            common = [sys.executable, str(script), "--bank", str(root / "bank"), "--length", "4",
                      "--particles", "16", "--tune", "1", "--draws", "2", "--noise-repeats", "2",
                      "--initial-attempts", "2", "--checkpoint-every", "1"]
            old = root / "old/chain-0"
            subprocess.run([*common, "--out", str(old)], cwd=lab, env=environment,
                           capture_output=True, text=True, check=True)
            proposal = root / "proposal.json"
            subprocess.run([sys.executable, str(lab / "scripts/prepare_path_proposal.py"),
                            "--chains", str(root / "old"), "--out", str(proposal)],
                           cwd=lab, env=environment, capture_output=True, text=True, check=True)
            new = root / "new"
            subprocess.run([*common, "--out", str(new), "--initial-checkpoint", str(old),
                            "--proposal-covariance", str(proposal), "--phase-seed", "2"],
                           cwd=lab, env=environment, capture_output=True, text=True, check=True)
            report = json.loads((new / "pmmh-summary.json").read_text())
            previous = json.loads((old / "pmmh-summary.json").read_text())
            self.assertFalse(report["exact_resume"])
            self.assertEqual(report["execution_mode"], "new_phase_from_accepted_extended_state")
            self.assertNotEqual(report["chain_seed"], previous["chain_seed"])
            self.assertEqual(report["initial"]["likelihood_seed"], previous["sampler_state"]["current_likelihood_seed"])
            self.assertEqual(report["initial"]["log_likelihood_estimate"], previous["sampler_state"]["current_log_likelihood_estimate"])


if __name__ == "__main__":
    unittest.main()

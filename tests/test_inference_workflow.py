"""Shared lifecycle decisions and real application bridge contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.inference_workflow import (
    Diagnostic, Estimate, FlaggedInferenceError, JointDraws, PreparedObservation,
    WorkflowHooks, WorkflowPolicy, run_workflow,
)
from sbi4finance import hkde_correction, hkde_inference, hkde_reference
from sbi4finance.hkde_workflow import HKDEWorkflowConfig, run_panel
from sbi4finance.svj import model, pipeline

ROOT = Path(__file__).resolve().parents[1]
TASK = json.loads((ROOT / "configs/tasks/hkde_inference_bank.json").read_text())["task"]


class CoreWorkflowTests(unittest.TestCase):
    def make_hooks(self, initial=(), correction=None, fallback=None):
        draws = JointDraws([[1, 10], [2, 20]], ("parameter", "state"))
        return WorkflowHooks(lambda value: PreparedObservation(value, {"case": "synthetic"}),
            lambda obs: Estimate(draws, "network", initial), lambda obs, estimate: (),
            lambda obs, estimate: {"mean": estimate.draws.summary()["mean"]}, correction, fallback)

    def test_joint_weights_remain_paired_and_replay(self):
        draws = JointDraws([[1, 10], [2, 20], [3, 30]], ("a", "state"), [0, 2, 8])
        assert_allclose(draws.summary()["mean"], [2.8, 28])
        sampled = draws.resample(100, 87)
        assert_array_equal(sampled.values, draws.resample(100, 87).values)
        assert_array_equal(sampled.values[:, 1], 10 * sampled.values[:, 0])
        self.assertTrue(np.all(sampled.values[:, 0] >= 2))
        self.assertIsNone(sampled.weights)
        with self.assertRaises(ValueError):
            draws.values[0, 0] = 5
        for values, names, weights in [([[1, np.nan]], ("a", "b"), None),
                                      ([[1, 2]], ("a", "a"), None),
                                      ([[1]], ("a",), [0]), ([[1]], ("a",), [-1])]:
            with self.assertRaises(ValueError):
                JointDraws(values, names, weights)

    def test_unchecked_network_is_never_accepted(self):
        result = run_workflow(0, self.make_hooks(), application="test")
        self.assertEqual(result.report["reliability"], "unassessed")
        self.assertEqual(result.report["scientific_acceptance"], "not_assessed_by_this_run")
        self.assertEqual(result.report["downstream"]["reliability"], "unassessed")

    def test_correction_fallback_order_and_observation_flags_persist(self):
        calls = []
        def correction(obs, estimate):
            calls.append("correction")
            return Estimate(estimate.draws, "importance", (Diagnostic("ess", "flagged"),))
        def fallback(obs, estimate):
            calls.append("fallback")
            return Estimate(estimate.draws, "mcmc", (Diagnostic("mixing", "passed"),))
        hooks = self.make_hooks((Diagnostic("misspecification", "flagged", scope="observation"),), correction, fallback)
        result = run_workflow(0, hooks, application="test", policy=WorkflowPolicy("fallback"))
        self.assertEqual(calls, ["correction", "fallback"])
        self.assertEqual([s["stage"] for s in result.report["stages"]], ["amortized", "correction", "fallback"])
        self.assertEqual(result.report["method"], "mcmc")
        self.assertEqual(result.report["diagnostic_reasons"], ["misspecification"])
        self.assertEqual(result.report["reliability"], "flagged")
        self.assertEqual(result.report["scientific_acceptance"], "not_assessed_by_this_run")
        # Historical importance failure does not itself invalidate a checked fallback.
        result = run_workflow(0, self.make_hooks(correction=correction, fallback=fallback),
                              application="test", policy=WorkflowPolicy("fallback"))
        self.assertEqual(result.report["reliability"], "locally_checked")
        self.assertEqual(result.report["stages"][1]["diagnostics"][0]["status"], "flagged")

    def test_fail_closed_stops_downstream_and_missing_fallback_stays_flagged(self):
        hooks = self.make_hooks((Diagnostic("accuracy", "flagged"),))
        downstream = Mock(side_effect=AssertionError("must not run"))
        with self.assertRaises(FlaggedInferenceError) as caught:
            run_workflow(0, WorkflowHooks(hooks.prepare, hooks.infer, hooks.diagnose, downstream),
                         application="test", policy=WorkflowPolicy("raise"))
        self.assertEqual(caught.exception.report["decision"], "downstream_declined")
        downstream.assert_not_called()
        result = run_workflow(0, hooks, application="test", policy=WorkflowPolicy("fallback"))
        self.assertEqual(result.report["decision"], "fallback_unavailable")
        self.assertEqual(result.report["downstream"]["reliability"], "flagged")

    def test_correction_cannot_reorder_coordinates(self):
        hooks = self.make_hooks(correction=lambda obs, est: Estimate(JointDraws([[10, 1]], ("state", "parameter")), "wrong"))
        with self.assertRaisesRegex(ValueError, "coordinate names"):
            run_workflow(0, hooks, application="test")


class UniformHKDEPosterior:
    """Exact uniform proposal to exercise real pricing and likelihood correction."""
    def sample(self, *, num_samples, conditions, seed):
        if set(conditions) != {"observations"} or conditions["observations"].shape != (1, 9):
            raise AssertionError("only the observed panel may enter conditioning")
        bounds = np.asarray(TASK["prior_bounds"])
        values = np.random.default_rng(seed).uniform(bounds[:, 0], bounds[:, 1], (num_samples, 2))
        return {name: values[None, :, index:index+1] for index, name in enumerate(hkde_inference.PARAMETER_NAMES)}

    def log_prob(self, data):
        widths = np.diff(np.asarray(TASK["prior_bounds"]), axis=1)[:, 0]
        return np.full(len(data["theta"]), -np.log(widths).sum())


class HKDEWorkflowTests(unittest.TestCase):
    def test_real_synthetic_panel_correction_and_joint_price_propagation(self):
        truth = np.asarray(TASK["prior_bounds"]).mean(axis=1)[None]
        observation = hkde_reference.price_parameters(TASK, truth)[0, :, None]
        observation += np.random.default_rng(57).normal(0, TASK["context"]["quote_noise_std"], observation.shape)
        config = HKDEWorkflowConfig(n_draws=16, n_proposals=32, seed=52, boundary_fraction=0,
                                    min_ess_fraction=.01, max_weight=1.)
        posterior = UniformHKDEPosterior()
        result = run_panel(observation, posterior, TASK, checkpoint_task=TASK, config=config)
        # Compare the bridge to the pre-existing correction, with the same seed/law.
        proposals = hkde_correction.propose(posterior, TASK, observation, 32, 52, .1, 0.)
        expected = hkde_correction.correct(proposals["parameters"], proposals["log_proposal"], observation, TASK)
        assert_array_equal(result.posterior.values, proposals["parameters"])
        assert_allclose(result.posterior.weights, expected["weights"], rtol=1e-13)
        targets = hkde_reference.price_parameters(TASK, result.posterior.values, hkde_reference.DEFAULT_TARGET_CONTRACTS)
        assert_allclose(result.report["downstream"]["target_mean"], expected["weights"] @ targets)
        assert_allclose(result.report["downstream"]["target_quantiles"],
                        hkde_reference.weighted_quantiles(targets, expected["weights"]))
        self.assertEqual(result.report["method"], "defensive_importance_sampling")
        self.assertEqual(result.report["reliability"], "locally_checked")
        self.assertEqual(result.report["scientific_acceptance"], "not_assessed_by_this_run")
        json.dumps(result.report, allow_nan=False)

    def test_task_mismatch_and_reference_failure_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "checkpoint task"):
            run_panel(np.zeros((9, 1)), UniformHKDEPosterior(), TASK,
                      checkpoint_task={**TASK, "measure": "P"})
        result = run_panel(np.zeros((9, 1)), UniformHKDEPosterior(), TASK, checkpoint_task=TASK,
            config=HKDEWorkflowConfig(n_draws=4, run_correction=False),
            diagnostic_fn=lambda obs, est: (Diagnostic("independent_reference", "flagged", explanation="conditional mismatch"),),
            target_price_fn=lambda rows: np.column_stack([rows[:, 0] + rows[:, 1], rows[:, 0] * rows[:, 1]]))
        self.assertEqual(result.report["reliability"], "flagged")
        self.assertEqual(result.report["downstream"]["diagnostic_reasons"], ["independent_reference"])


class SVJWorkflowTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        coords = model.draw_coordinates(8, rng)
        self.values = np.column_stack([coords, np.full(8, np.log(.25))])
        self.posterior = SimpleNamespace(sample=lambda returns, num_samples, seed: self.values[None],
                                         summaries=lambda returns: np.ones((1, 3)))
        self.returns = np.zeros(model.WINDOW)
        self.config = pipeline.PipelineConfig(n_draws=8, n_proposals=8, horizons=(1,), n_paths=2, variance_split=False)

    def audit_result(self, efficiency=.5):
        return SimpleNamespace(efficiency=efficiency, ess=8 * efficiency, n_particles=32, loglik_sd=1.,
                               posterior=self.values.copy(), proposals=self.values.copy(),
                               log_weights=np.full(8, -np.log(8)), seconds=.1,
                               flagged=lambda threshold: efficiency < threshold)

    def pmmh_result(self, rhat=1.005, ess=500, count=16):
        values = np.tile(self.values, (2, 1))[:count]
        return SimpleNamespace(seconds=.1, rhat=np.full(9, rhat), ess_bulk=np.full(9, ess),
                               n_particles=64, loglik_sd=1., accept_rate=.25, pooled=lambda: values)

    def run_light(self, audit_result=None, chains=None, misspecification=None):
        with patch.object(pipeline, "audit", return_value=audit_result), \
             patch.object(pipeline, "run_pmmh", return_value=chains), \
             patch.object(pipeline.risk, "risk_numbers", return_value={"1": {"var99": 1.}}), \
             patch.object(pipeline.risk, "forecast", return_value={"definition": "negative_cumulative_percent_log_return_v1"}):
            return pipeline.run_window(self.returns, self.posterior, misspecification, self.config)

    def test_routes_and_status_reach_every_final_risk(self):
        self.config.run_audit = False
        report, final = self.run_light()
        self.assertEqual(report["status"], "unassessed")
        assert_array_equal(final, self.values)
        self.config.run_audit = True
        for efficiency, status in ((.5, "locally_checked"), (.001, "flagged")):
            report, final = self.run_light(self.audit_result(efficiency))
            self.assertEqual(report["route"], "network + importance sampling")
            self.assertEqual(report["status"], status)
            self.assertEqual(report["risk"]["reliability"], status)
            self.assertEqual(report["scientific_acceptance"], "not_assessed_by_this_run")
            assert_array_equal(final, self.values)
            json.dumps(report, allow_nan=False)

    def test_pmmh_is_evaluated_and_small_or_nonfinite_chains_stay_flagged(self):
        self.config.pmmh_on_flag = True
        for chains, status in ((self.pmmh_result(), "locally_checked"),
                               (self.pmmh_result(rhat=1.2), "flagged"),
                               (self.pmmh_result(rhat=np.nan), "flagged"),
                               (self.pmmh_result(ess=20), "flagged"),
                               (self.pmmh_result(count=3), "flagged")):
            report, final = self.run_light(self.audit_result(.001), chains)
            self.assertEqual(report["route"], "particle MCMC")
            self.assertEqual(report["status"], status)
            self.assertEqual(len(final), min(8, len(chains.pooled())))
            json.dumps(report, allow_nan=False)

    def test_failed_audit_retains_raw_estimator_with_explicit_failure(self):
        with patch.object(pipeline, "audit", side_effect=ValueError("importance audit failed: all-zero weights")), \
             patch.object(pipeline.risk, "risk_numbers", return_value={}), \
             patch.object(pipeline.risk, "forecast", return_value={}):
            report, final = pipeline.run_window(self.returns, self.posterior, config=self.config)
        self.assertEqual(report["status"], "flagged")
        self.assertEqual(report["route"], "network")
        self.assertEqual(report["workflow"]["method"], "amortized_flow_after_failed_audit")
        self.assertIn("particle_importance_audit_failed", report["diagnostic_reasons"])
        assert_array_equal(final, self.values)

    def test_misspecification_flag_survives_a_successful_fallback(self):
        self.config.pmmh_on_flag = True
        report, _ = self.run_light(self.audit_result(.001), self.pmmh_result(),
                                  SimpleNamespace(pvalue=lambda summaries: [.001]))
        self.assertEqual(report["status"], "flagged")
        self.assertIn("summary_space_misspecification", report["diagnostic_reasons"])

    def test_reference_binding_mismatch_stops_before_neural_sampling(self):
        from sbi4finance.svj.checks import MisspecificationTest
        posterior = SimpleNamespace(sample=Mock(), artifact={
            "files": {"model.keras": "checkpoint-A"}, "task_sha256": "task-A"})
        for binding in (None, {"checkpoint_sha256": "checkpoint-B", "task_sha256": "task-A"},
                        {"checkpoint_sha256": "checkpoint-A", "task_sha256": "task-B"}):
            reference = MisspecificationTest(np.zeros((2, 32)), np.ones((2, 32)), 1., binding)
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                pipeline.run_window(self.returns, posterior, reference, self.config)
            posterior.sample.assert_not_called()

    def test_historical_misspecification_screen_remains_five_percent(self):
        self.assertEqual(self.config.misspecification_threshold, .05)
        for pvalue, expected in ((.02, "flagged"), (.05, "locally_checked")):
            report, _ = self.run_light(self.audit_result(), misspecification=
                SimpleNamespace(pvalue=lambda summaries: [pvalue]))
            self.assertEqual(report["status"], expected)
            diagnostic = report["workflow"]["stages"][0]["diagnostics"][0]
            self.assertEqual(diagnostic["thresholds"], {"min_pvalue": .05})

    def test_nonfinite_observations_fail_before_network(self):
        values = self.returns.copy()
        values[-1] = np.nan
        with self.assertRaises(ValueError):
            pipeline.run_window(values, self.posterior, config=self.config)

    def test_real_small_forecast_preserves_legacy_risk_definition(self):
        self.config.run_audit = False
        report, final = pipeline.run_window(self.returns, self.posterior, config=self.config)
        self.assertEqual(report["risk"]["definition"], "negative_cumulative_percent_log_return_v1")
        self.assertEqual(report["risk"]["unit"], "percent_log_loss")
        self.assertEqual(report["risk"]["reliability"], "unassessed")
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

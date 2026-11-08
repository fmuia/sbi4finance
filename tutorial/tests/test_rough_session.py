"""Bounded repricing preserves joint draws, context roles and numerical budgets."""

import unittest
from unittest.mock import patch

import numpy as np

from sbi4f_tutorial.rough_options import lesson
from sbi4f_tutorial.shared import backbones, pricing, tasks


class RoughSessionTests(unittest.TestCase):
    def setUp(self):
        self.known, _, self.a = backbones.load_example("rough-known100k-validation0")
        self.joint, _, self.b = backbones.load_example("rough-joint100k-validation0")

    @staticmethod
    def fake_price(parameters, curve, contracts, blocks, ladder, sigma, budget, seed, cap=None):
        prices = np.full(len(contracts), 0.01 + parameters["hurst"] / 100)
        errors = sigma / 10
        return prices, errors, np.diag(errors**2), np.array([0, ladder[0], 0])

    def test_known_context_and_joint_curves_use_the_same_task_grids_and_no_truth(self):
        for task, arrays in ((self.known, self.a), (self.joint, self.b)):
            with self.subTest(task=task["task_id"]), patch.object(
                    pricing, "rough_surface", side_effect=self.fake_price) as pricer:
                result = lesson.rough_posterior_predictive(
                    task, arrays["neural_draws"], arrays["observations"],
                    context_values=arrays.get("context_values"), draws=3, seed=37)
                selected = result["metadata"]["posterior_indices"]
                np.testing.assert_array_equal(result["selected_parameters"], arrays["neural_draws"][selected])
                self.assertEqual(result["reference_prices"].shape, (3, 9))
                self.assertEqual(result["replicated_quotes"].shape, (24, 9))
                np.testing.assert_array_equal(result["contract_indices"], np.arange(36, 45))
                for row, call in zip(result["selected_parameters"], pricer.call_args_list, strict=True):
                    params, curve, contracts, blocks, ladder, sigma, budget, _ = call.args
                    self.assertIs(blocks, task["numerics"]["blocks"])
                    self.assertEqual(budget, task["numerics"]["max_mc_se_over_quote_sigma"])
                    self.assertEqual(ladder, [32768, 65536, 131072])
                    self.assertIsNone(call.kwargs["cap"], "the looser final cap cannot move to an earlier rung")
                    self.assertTrue(all(c["maturity"] == .25 for c in contracts))
                    expected = (tasks.joint_curve(task, row) if tasks.is_joint_curve_task(task)
                                else tasks.curve(task, arrays["context_values"]))
                    np.testing.assert_array_equal(curve["times"], expected["times"])
                    np.testing.assert_array_equal(curve["values"], expected["values"])
                    self.assertEqual(set(params), {"hurst", "eta", "rho"})

    def test_fresh_noise_is_reproducible_separate_from_monte_carlo_and_does_not_change_prices(self):
        with patch.object(pricing, "rough_surface", side_effect=self.fake_price) as pricer:
            kwargs = dict(context_values=self.a["context_values"], draws=3)
            a = lesson.rough_posterior_predictive(self.known, self.a["neural_draws"], self.a["observations"], **kwargs)
            b = lesson.rough_posterior_predictive(self.known, self.a["neural_draws"], self.a["observations"], **kwargs)
            c = lesson.rough_posterior_predictive(self.known, self.a["neural_draws"], self.a["observations"],
                                                  **kwargs, quote_replicates=2)
        np.testing.assert_array_equal(a["reference_prices"], b["reference_prices"])
        np.testing.assert_array_equal(a["replicated_quotes"], b["replicated_quotes"])
        np.testing.assert_array_equal(a["reference_prices"], c["reference_prices"])
        self.assertFalse(np.array_equal(a["replicated_quotes"], np.repeat(a["reference_prices"], 8, axis=0)))
        self.assertEqual(pricer.call_count, 9)
        self.assertEqual(a["metadata"]["max_mc_se_over_sigma"], .1)

    def test_failed_pricing_does_not_discard_a_posterior_row_or_return_a_partial_check(self):
        with patch.object(pricing, "rough_surface", side_effect=ValueError("Monte Carlo standard-error budget failed")) as p:
            with self.assertRaisesRegex(ValueError, "posterior row 0.*budget failed"):
                lesson.rough_posterior_predictive(self.known, self.a["neural_draws"], self.a["observations"],
                                                   context_values=self.a["context_values"], draws=3)
        self.assertEqual(p.call_count, 1)

    def test_hidden_joint_curve_cannot_be_passed_as_observed_context(self):
        with self.assertRaisesRegex(ValueError, "supply no curve context"):
            lesson.rough_posterior_predictive(self.joint, self.b["neural_draws"], self.b["observations"],
                                               context_values=np.full(7, .04))
        with self.assertRaisesRegex(ValueError, "observed positive curve"):
            lesson.rough_posterior_predictive(self.known, self.a["neural_draws"], self.a["observations"])

    def test_time_guard_stops_before_another_surface_and_never_returns_a_partial_result(self):
        with patch("time.perf_counter", side_effect=[0, 3]), patch.object(pricing, "rough_surface") as pricer:
            with self.assertRaisesRegex(RuntimeError, "no partial result"):
                lesson.rough_posterior_predictive(
                    self.known, self.a["neural_draws"], self.a["observations"],
                    context_values=self.a["context_values"], draws=3, max_seconds=1)
        pricer.assert_not_called()

    def test_fresh_training_pairs_are_prior_draws_not_selected_cached_targets(self):
        with patch.object(pricing, "rough_surface", side_effect=self.fake_price):
            a, metadata = lesson.generate_training_pairs(self.known, count=3, seed=37)
            b, _ = lesson.generate_training_pairs(self.known, count=3, seed=37)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(a["parameters"].shape, (3, 3))
        self.assertEqual(a["observations"].shape, (3, 63))
        self.assertEqual(a["context_values"].shape, (3, 7))
        self.assertFalse(np.any(np.all(a["parameters"] == self.a["truth"], axis=1)))
        self.assertFalse(np.array_equal(a["context_values"][0], a["context_values"][1]))
        self.assertEqual(metadata["examples"], 3)

    def test_analytic_joint_curve_is_sampled_on_all_original_simulator_nodes(self):
        row = self.b["neural_draws"][0]
        curve = tasks.joint_curve(self.joint, row)
        t = np.unique(np.concatenate([b["horizon"] / b["n_steps"] * np.arange(b["n_steps"] + 1)
                                      for b in self.joint["numerics"]["blocks"]]))
        np.testing.assert_array_equal(curve["times"], t)
        short, long, decay = row[3:]
        np.testing.assert_allclose(curve["values"], long + (short - long) * np.exp(-decay * t), rtol=1e-15)


if __name__ == "__main__":
    unittest.main()

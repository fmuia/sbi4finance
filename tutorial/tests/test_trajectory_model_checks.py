"""Return-data checks preserve the full canonical law and compare only the revealed prefix."""

import copy
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import diagnostics, model_checks, simulation


class TrajectoryModelCheckTests(unittest.TestCase):
    def setUp(self):
        self.task = simulation.teaching_task("rough-known", observations=29)
        self.panel = diagnostics.generate_panel(self.task, count=4, seed=73)
        self.observed = self.panel["data"]["observations"][0].copy()

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_prior_comparison_uses_all_paired_panel_rows_and_only_the_same_prefix(self):
        result = model_checks.compare_prior(self.task, self.panel, self.observed, length=7)
        values = self.panel["data"]["observations"][:, :7, 0]
        expected = 100 * np.column_stack([
            values.sum(axis=1), np.sqrt(np.mean(values**2, axis=1) / self.task["dt_years"]),
            np.max(np.abs(values), axis=1)])
        np.testing.assert_allclose(result["summaries"], expected, rtol=3e-16)
        np.testing.assert_array_equal(result["selected_parameters"], self.panel["data"]["parameters"])
        np.testing.assert_array_equal(result["replicated_returns"], self.panel["data"]["observations"])
        self.assertEqual(result["replicated_prefix"].shape, (4, 7, 1))
        self.assertEqual(result["metadata"]["count"], 4)
        self.assertEqual(result["kind"], "prior_predictive")
        self.assertEqual(result["task_fingerprint"], simulation.task_fingerprint(self.task))
        self.assertEqual(self.panel["data"]["observations"].shape, (4, 29, 1))

    def test_every_observation_shape_ignores_unrevealed_nonfinite_future_values(self):
        expected = model_checks.compare_prior(self.task, self.panel, self.observed, 7)
        changed = self.observed.copy()
        changed[7:] = np.nan
        for values in (changed[:, 0], changed, changed[None]):
            with self.subTest(shape=values.shape):
                result = model_checks.compare_prior(self.task, self.panel, values, 7)
                np.testing.assert_array_equal(result["observed_summary"], expected["observed_summary"])
                np.testing.assert_array_equal(result["observed_returns"], self.observed[:7])
        with self.assertRaisesRegex(ValueError, "one observed history"):
            model_checks.compare_prior(self.task, self.panel, np.zeros((2, 29, 1)), 7)
        with self.assertRaisesRegex(ValueError, "observed prefix"):
            model_checks.compare_prior(self.task, self.panel, changed, 29)

    def test_posterior_replication_replays_canonical_full_task_and_whole_parameter_rows(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task = simulation.teaching_task(model, observations=29)
                pairs = simulation.generate(task, "validation", 0, 6, seed=73, backend="numpy")
                values = pairs["parameters"]
                before = values.copy()
                result = model_checks.replicate_check(task, pairs["observations"][0], values,
                                                       length=7, count=4, seed=91)
                indices = result["metadata"]["posterior_indices"]
                self.assertEqual(len(indices), len(set(indices)))
                np.testing.assert_array_equal(result["selected_parameters"], values[indices])
                expected = simulation.simulate_parameters(task, values[indices], seed=91, backend="numpy")
                np.testing.assert_array_equal(result["replicated_returns"], expected)
                np.testing.assert_array_equal(result["replicated_prefix"], expected[:, :7])
                np.testing.assert_array_equal(values, before)
                self.assertEqual(result["metadata"]["source_row_count"], 6)
                self.assertEqual(result["kind"], "posterior_predictive")

    def test_replication_never_shortens_252_or_1260_task_or_adds_observation_noise(self):
        for maximum in (252, 1260):
            task = simulation.teaching_task("rough-known", observations=maximum)
            parameters = np.tile(np.mean(task["prior_bounds"], axis=1), (3, 1))
            generated = np.arange(3 * maximum, dtype=float).reshape(3, maximum, 1) * .00001
            with mock.patch.object(simulation, "simulate_parameters", return_value=generated) as simulate:
                result = model_checks.replicate_check(task, np.zeros(maximum), parameters, 13, seed=91)
            self.assertEqual(simulate.call_args.args[0], task)
            self.assertEqual(simulate.call_args.args[0]["n_observations"], maximum)
            self.assertEqual(simulate.call_args.kwargs, {"seed": 91, "backend": "numpy"})
            np.testing.assert_array_equal(result["replicated_returns"], generated)
            np.testing.assert_array_equal(result["replicated_prefix"], generated[:, :13])
            self.assertEqual(result["metadata"]["count"], 3)
            self.assertEqual(result["metadata"]["requested_count"], 32)
        other = simulation.teaching_task("rough-known", observations=252)
        with self.assertRaisesRegex(ValueError, "tasks differ"):
            model_checks.compare_prior(other, self.panel, np.zeros(252), 13)

    def test_one_day_constant_negative_and_zero_return_statistics_are_finite(self):
        for length in (1, 7, 29):
            for level in (0., -.01, .01):
                with self.subTest(length=length, level=level):
                    changed = copy.deepcopy(self.panel)
                    changed["data"]["observations"].fill(level)
                    result = model_checks.compare_prior(self.task, changed, np.full(29, level), length)
                    expected = [100 * length * level,
                                100 * abs(level) / np.sqrt(self.task["dt_years"]), 100 * abs(level)]
                    np.testing.assert_allclose(result["observed_summary"], expected, rtol=3e-16)
                    self.assertTrue(np.isfinite(result["summaries"]).all())
                    np.testing.assert_array_equal(result["empirical_percentiles"], [50., 50., 50.])

    def test_only_observed_comparison_changes_when_observed_history_changes(self):
        values = self.panel["data"]["parameters"]
        a = model_checks.replicate_check(self.task, np.zeros(29), values, 7, seed=91)
        b = model_checks.replicate_check(self.task, np.ones(29) * .01, values, 7, seed=91)
        np.testing.assert_array_equal(a["replicated_returns"], b["replicated_returns"])
        np.testing.assert_array_equal(a["selected_parameters"], b["selected_parameters"])
        self.assertFalse(np.array_equal(a["observed_summary"], b["observed_summary"]))

    def test_bad_parameters_shapes_and_work_limits_fail_without_simulation(self):
        values = self.panel["data"]["parameters"]
        for invalid in (values[:, :-1], values[:1], values.astype(complex),
                        np.ma.array(values, mask=True), np.zeros_like(values), np.full_like(values, np.nan)):
            with self.subTest(shape=invalid.shape), mock.patch.object(simulation, "simulate_parameters") as simulate:
                with self.assertRaises(ValueError):
                    model_checks.replicate_check(self.task, self.observed, invalid, 7)
                simulate.assert_not_called()
        for count in (1, 33, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                model_checks.replicate_check(self.task, self.observed, values, 7, count=count)
        for length in (0, 30, 7., True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                model_checks.compare_prior(self.task, self.panel, self.observed, length)

    def test_posterior_boundary_rounding_is_retained_and_failures_never_redraw_or_return_partial(self):
        bounds = np.asarray(self.task["prior_bounds"])
        parameters = np.stack([bounds[:, 0], bounds[:, 1]])
        with mock.patch.object(simulation, "simulate_parameters", return_value=np.zeros((2, 29, 1))) as simulate:
            result = model_checks.replicate_check(self.task, self.observed, parameters, 7)
        np.testing.assert_array_equal(result["selected_parameters"], parameters[result["metadata"]["posterior_indices"]])
        self.assertEqual(simulate.call_count, 1)
        with mock.patch.object(simulation, "simulate_parameters", side_effect=FloatingPointError("failed row")) as simulate:
            with self.assertRaisesRegex(FloatingPointError, "failed row"):
                model_checks.replicate_check(self.task, self.observed, parameters, 7)
            self.assertEqual(simulate.call_count, 1)
        with mock.patch.object(simulation, "simulate_parameters", return_value=np.zeros((1, 29, 1))) as simulate:
            with self.assertRaisesRegex(ValueError, "replicated histories"):
                model_checks.replicate_check(self.task, self.observed, parameters, 7)
            self.assertEqual(simulate.call_count, 1)

    def test_histograms_include_every_simulated_summary_and_observation_without_clipping(self):
        result = model_checks.compare_prior(self.task, self.panel, self.observed, 7)
        result["observed_summary"] += 1000.
        figure = model_checks.plot_check(result)
        for index, ax in enumerate(figure.axes):
            self.assertEqual(sum(p.get_height() for p in ax.patches), 4)
            low, high = ax.get_xlim()
            self.assertLess(low, min(result["summaries"][:, index].min(), result["observed_summary"][index]))
            self.assertGreater(high, max(result["summaries"][:, index].max(), result["observed_summary"][index]))
            self.assertEqual(ax.lines[0].get_xdata()[0], result["observed_summary"][index])


if __name__ == "__main__":
    unittest.main()

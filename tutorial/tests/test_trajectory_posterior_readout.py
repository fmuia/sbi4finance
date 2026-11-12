"""Computed interpretation uses paired physical draws and the actual declared priors."""

import unittest

import numpy as np

from sbi4f_tutorial.trajectories import lesson, simulation


class TrajectoryPosteriorReadoutTests(unittest.TestCase):
    @staticmethod
    def prior_quantiles(task, probabilities):
        probabilities = np.asarray(probabilities)
        columns = []
        for (low, high), kind in zip(task["prior_bounds"], task["prior_types"], strict=True):
            columns.append(np.exp(np.log(low) + probabilities * np.log(high / low))
                           if kind == "log_uniform" else low + probabilities * (high - low))
        return np.column_stack(columns)

    def test_uniform_and_log_uniform_prior_central_intervals_use_their_correct_cdfs(self):
        task = simulation.teaching_task("heston")
        values = self.prior_quantiles(task, np.linspace(0, 1, 101))
        # Floating exponentiation can exceed an endpoint by a final bit; replace
        # these generated fixture endpoints with the exact declared bounds.
        values[0] = np.asarray(task["prior_bounds"])[:, 0]
        values[-1] = np.asarray(task["prior_bounds"])[:, 1]
        result = lesson.posterior_readout(task, values)
        for i, name in enumerate(task["parameter_names"]):
            record = result["parameters"][name]
            expected = self.prior_quantiles(task, [.05, .5, .95])[:, i]
            np.testing.assert_allclose([record["prior_q05"], record["prior_q50"], record["prior_q95"]], expected,
                                       rtol=2e-15)
            np.testing.assert_allclose([record["q05"], record["q50"], record["q95"]], expected, rtol=2e-15)
            self.assertAlmostEqual(record["width_ratio90"], 1., places=14)
        low, high = task["prior_bounds"][0]
        kappa = result["parameters"]["kappa"]
        self.assertAlmostEqual(kappa["prior_q50"], np.sqrt(low * high))
        self.assertNotAlmostEqual(kappa["prior_q05"], low + .05 * (high - low))
        self.assertIsNone(result["truth_excluded_parameters"])

    def test_narrowest_and_widest_compare_relative_prior_widths_not_physical_units(self):
        task = simulation.teaching_task("heston")
        fractions = [.8, .4, .6, .3, .2]
        columns = []
        for i, fraction in enumerate(fractions):
            columns.append(self.prior_quantiles(task, .5 + fraction * np.linspace(-.5, .5, 101))[:, i])
        values = np.column_stack(columns)
        result = lesson.posterior_readout(task, values)
        posterior = np.quantile(values, [.05, .95], axis=0)
        prior = self.prior_quantiles(task, [.05, .95])
        ratios = (posterior[1] - posterior[0]) / (prior[1] - prior[0])
        np.testing.assert_allclose([result["parameters"][name]["width_ratio90"] for name in task["parameter_names"]], ratios)
        self.assertEqual(result["narrowest_relative_to_prior"], task["parameter_names"][np.argmin(ratios)])
        self.assertEqual(result["widest_relative_to_prior"], task["parameter_names"][np.argmax(ratios)])

    def test_strongest_negative_pair_preserves_rows_and_excludes_constant_columns(self):
        task = simulation.teaching_task("rough-known")
        x = np.linspace(-1, 1, 9)
        values = np.column_stack([.2 + .02*x, 1.4 - .1*x, np.full(9, -.7), np.zeros(9)])
        before = values.copy()
        result = lesson.posterior_readout(task, values)
        self.assertEqual(result["strongest_pair"]["names"], ["hurst", "eta"])
        self.assertAlmostEqual(result["strongest_pair"]["correlation"], -1.)
        np.testing.assert_array_equal(values, before)
        reordered = lesson.posterior_readout(task, values[::-1])
        self.assertEqual(reordered["strongest_pair"]["names"], result["strongest_pair"]["names"])
        self.assertAlmostEqual(reordered["strongest_pair"]["correlation"], -1.)
        shuffled = values.copy()
        shuffled[:, 1] = np.roll(shuffled[:, 1], 1)
        changed = lesson.posterior_readout(task, shuffled)
        self.assertLess(abs(changed["strongest_pair"]["correlation"]), .9)
        for name in task["parameter_names"]:
            for key in ("q05", "q50", "q95"):
                self.assertEqual(changed["parameters"][name][key], result["parameters"][name][key])

    def test_constant_columns_and_only_one_varying_column_have_no_correlation_claim(self):
        task = simulation.teaching_task("rough-known")
        values = np.tile(np.mean(task["prior_bounds"], axis=1), (9, 1))
        result = lesson.posterior_readout(task, values)
        self.assertIsNone(result["strongest_pair"])
        self.assertTrue(all(row["width_ratio90"] == 0 for row in result["parameters"].values()))
        values[:, -1] = np.linspace(-.01, .01, 9)
        result = lesson.posterior_readout(task, values)
        self.assertIsNone(result["strongest_pair"])

    def test_near_zero_drift_needs_no_mean_normalization_and_correlation_does_not_underflow(self):
        task = simulation.teaching_task("rough-known")
        x = np.linspace(-1, 1, 9)
        values = np.column_stack([.2 + .02*x, np.full(9, 1.4), np.full(9, -.7), 1e-310*x])
        result = lesson.posterior_readout(task, values)
        self.assertEqual(result["strongest_pair"]["names"], ["hurst", "drift"])
        self.assertAlmostEqual(result["strongest_pair"]["correlation"], 1.)
        self.assertTrue(np.isfinite(result["parameters"]["drift"]["width_ratio90"]))

    def test_truth_adds_inside_excluded_flags_without_changing_any_posterior_statistics(self):
        task = simulation.teaching_task("rough-known")
        values = np.column_stack([np.linspace(.15, .25, 21), np.linspace(1., 2., 21),
                                  np.linspace(-.8, -.6, 21), np.linspace(-.02, .02, 21)])
        truth = np.array([.2, 2.3, -.7, 0.])
        result = lesson.posterior_readout(task, values, truth)
        without = lesson.posterior_readout(task, values)
        self.assertEqual(result["truth_excluded_parameters"], ["eta"])
        self.assertTrue(result["parameters"]["drift"]["truth_inside90"])
        self.assertFalse(result["parameters"]["eta"]["truth_inside90"])
        for name in task["parameter_names"]:
            for key, value in result["parameters"][name].items():
                if key != "truth_inside90":
                    self.assertEqual(value, without["parameters"][name][key])
        for key in ("strongest_pair", "narrowest_relative_to_prior", "widest_relative_to_prior"):
            self.assertEqual(result[key], without[key])

    def test_bad_rows_bounds_and_truth_are_rejected_instead_of_filtered_or_clipped(self):
        task = simulation.teaching_task("rough-known")
        values = np.tile(np.mean(task["prior_bounds"], axis=1), (3, 1))
        bad = [values[:1], values[:, :-1], values.astype(complex),
               np.ma.array(values, mask=True), np.full_like(values, np.nan)]
        outside = values.copy()
        outside[0, 0] = task["prior_bounds"][0][0] - .01
        bad.append(outside)
        for rows in bad:
            with self.subTest(shape=rows.shape), self.assertRaises(ValueError):
                lesson.posterior_readout(task, rows)
        bounds = np.asarray(task["prior_bounds"])
        boundary_rows = np.stack([bounds[:, 0], bounds[:, 1]])
        result = lesson.posterior_readout(task, boundary_rows)
        self.assertEqual(result["draw_count"], 2)
        for truth in (bounds[:, 0], np.zeros(3), np.full(4, np.nan),
                      values[0].astype(complex), np.ma.array(values[0], mask=True)):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                lesson.posterior_readout(task, values, truth)


if __name__ == "__main__":
    unittest.main()

"""The interactive forward model uses the same P law and fixed innovations as generation."""

import unittest

import numpy as np

from sbi4f_tutorial.trajectories import lesson, simulation


class TrajectoryForwardTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_chosen_parameter_simulation_replays_the_canonical_training_innovations(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task = simulation.teaching_task(model, observations=29)
                pairs = simulation.generate(task, "train", 0, 3, seed=37, backend="numpy")
                returns = simulation.simulate_parameters(task, pairs["parameters"], seed=37, backend="numpy")
                np.testing.assert_array_equal(returns, pairs["observations"])
                self.assertEqual(returns.shape, (3, 29, 1))
                self.assertFalse(np.array_equal(returns[0], returns[1]))

    def test_fixed_seed_makes_a_drift_change_exactly_a_change_in_the_return_drift(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task = simulation.teaching_task(model, observations=29)
                values = np.mean(task["prior_bounds"], axis=1)[None, :]
                drift = task["parameter_names"].index("drift")
                values[0, drift] = .03
                a = simulation.simulate_parameters(task, values, seed=37)
                values[0, drift] = .08
                b = simulation.simulate_parameters(task, values, seed=37)
                np.testing.assert_allclose(b - a, .05 * task["dt_years"], rtol=0, atol=2e-17)

    def test_current_published_histories_retain_their_generation_law(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task, _, arrays = lesson.load_example(model)
                seed = int(arrays["trajectory_id"][0].split(":")[1])
                pairs = simulation.generate(task, "validation", 0, 1, seed=seed, backend="numpy")
                np.testing.assert_array_equal(pairs["parameters"][0], arrays["truth"][0])
                # The frozen rough histories used a different FFT environment;
                # its last-bit differences are tiny relative to a daily return.
                np.testing.assert_allclose(pairs["observations"][0], arrays["observations"][0], rtol=1e-12, atol=1e-13)

    def test_preview_respects_parameter_names_support_and_bounded_work(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task, _, _ = lesson.load_example(model)
                values = np.mean(task["prior_bounds"], axis=1)
                chosen = dict(reversed(list(zip(task["parameter_names"], values, strict=True))))
                original_task = simulation.task_fingerprint(task)
                a = lesson.simulator_preview(task, chosen, paths=3, days=29, seed=37)
                b = lesson.simulator_preview(task, chosen, paths=3, days=29, seed=37)
                np.testing.assert_array_equal(a["parameters"], values)
                np.testing.assert_array_equal(a["observations"], b["observations"])
                self.assertEqual(a["observations"].shape, (3, 29, 1))
                self.assertEqual(original_task, simulation.task_fingerprint(task))
                wrong = {**chosen, "latent_variance": .04}
                with self.assertRaisesRegex(ValueError, "exactly"):
                    lesson.simulator_preview(task, wrong)
                outside = values.copy()
                outside[0] = task["prior_bounds"][0][1] + 1
                with self.assertRaisesRegex(ValueError, "prior support"):
                    simulation.simulate_parameters(task, outside[None, :])
                for invalid in (np.full((2, len(values)), np.nan), values[None, :].astype(complex)):
                    with self.assertRaises(ValueError):
                        simulation.simulate_parameters(task, invalid)
                with self.assertRaisesRegex(ValueError, "16 paths"):
                    lesson.simulator_preview(task, chosen, paths=17)

    def test_simulator_limits_stay_fixed_when_parameters_change_and_only_presets_change_them(self):
        task, _, _ = lesson.load_example("rough-known")
        parameters = {"hurst": .1, "eta": .5, "rho": -.7, "drift": .05}
        low = lesson.simulator_preview(task, parameters, paths=2, days=29)
        parameters["eta"] = 2.5
        high = lesson.simulator_preview(task, parameters, paths=2, days=29)
        self.assertFalse(np.array_equal(low["observations"], high["observations"]))
        for scale in ("Standard", "Wide", "Very wide"):
            a = lesson.forward_simulator_figure(low, scale=scale)
            b = lesson.forward_simulator_figure(high, scale=scale)
            for ax_a, ax_b, name in zip(a.axes, b.axes, ("prices", "returns", "realized"), strict=True):
                self.assertEqual(ax_a.get_ylim(), lesson.SIMULATOR_SCALES[scale][name])
                self.assertEqual(ax_b.get_ylim(), ax_a.get_ylim())
        with self.assertRaisesRegex(ValueError, "fixed simulator scale"):
            lesson.forward_simulator_figure(high, scale="automatic")

    def test_fixed_views_report_outside_values_without_deleting_or_changing_them(self):
        task = simulation.teaching_task("rough-known", observations=29)
        returns = np.stack([np.full((29, 1), .2), np.full((29, 1), -.2)])
        original = returns.copy()
        result = {"task": task, "observations": returns}
        rows = lesson.simulator_axis_summary(result, "Standard")
        expected_prices = 100 * np.exp(np.column_stack([np.zeros(2), np.cumsum(returns[:, :, 0], axis=1)]))
        self.assertEqual(rows[0]["points above view"], np.count_nonzero(expected_prices > 200))
        self.assertEqual(rows[0]["total points"], 60)
        self.assertEqual(rows[0]["data maximum"], expected_prices.max())
        self.assertEqual(rows[1]["points above view"], 29)
        self.assertEqual(rows[1]["total points"], 29)
        figure = lesson.forward_simulator_figure(result, scale="Standard")
        for i in range(2):
            np.testing.assert_array_equal(figure.axes[0].lines[i].get_ydata(), expected_prices[i])
        np.testing.assert_array_equal([bar.get_height() for bar in figure.axes[1].patches], np.full(29, 20.))
        self.assertGreater(figure.axes[0].lines[0].get_ydata().max(), figure.axes[0].get_ylim()[1])
        np.testing.assert_array_equal(returns, original)


if __name__ == "__main__":
    unittest.main()

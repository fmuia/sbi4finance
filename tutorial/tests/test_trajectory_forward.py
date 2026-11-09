"""The interactive forward model uses the same P law and fixed innovations as generation."""

import unittest

import numpy as np

from sbi4f_tutorial.trajectories import lesson, simulation


class TrajectoryForwardTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

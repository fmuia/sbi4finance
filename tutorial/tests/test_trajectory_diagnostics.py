"""Diagnostic histories keep their exact P law, prefix mask and generating labels separate."""

import copy
import types
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import diagnostics, inference, simulation


class TrajectoryDiagnosticTests(unittest.TestCase):
    @staticmethod
    def pairs(task, count, seed):
        digest = simulation.task_fingerprint(task)
        bounds = np.asarray(task["prior_bounds"])
        return {
            "parameters": np.tile(bounds.mean(axis=1), (count, 1)),
            "observations": np.arange(count * task["n_observations"], dtype=float).reshape(
                count, task["n_observations"], 1) * .00001,
            "trajectory_id": np.array([f"{digest[:16]}:{seed}:validation:{i}" for i in range(count)]),
        }

    def panel(self, model="rough-known", maximum=29, count=3, seed=73):
        task = simulation.teaching_task(model, observations=maximum)
        data = self.pairs(task, count, seed)
        with mock.patch.object(simulation, "generate", return_value=data):
            panel = diagnostics.generate_panel(task, count=count, seed=seed)
        return task, panel

    def test_generation_is_the_canonical_validation_law_for_each_model_without_redraws(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task = simulation.teaching_task(model, observations=29)
                panel = diagnostics.generate_panel(task, count=3, seed=73)
                direct = simulation.generate(task, "validation", 0, 3, seed=73, backend="numpy")
                for key in direct:
                    np.testing.assert_array_equal(panel["data"][key], direct[key])
                self.assertEqual(panel["metadata"]["count"], 3)
                self.assertEqual(panel["metadata"]["source_split"], "validation")
                self.assertGreaterEqual(panel["metadata"]["generation_seconds"], 0)
                self.assertEqual(panel["task_fingerprint"], simulation.task_fingerprint(task))
                self.assertIsNot(panel["task"], task)

    def test_full_252_and_1260_tasks_are_generated_directly_and_never_substituted(self):
        for model, maximum in [("rough-known", 252), *[(name, 1260) for name in simulation.MODELS]]:
            with self.subTest(model=model, maximum=maximum):
                task = simulation.teaching_task(model, observations=maximum)
                pairs = self.pairs(task, 200, 73)
                with mock.patch.object(simulation, "generate", return_value=pairs) as generate:
                    panel = diagnostics.generate_panel(task, seed=73)
                generate.assert_called_once_with(task, "validation", 0, 200, seed=73, backend="numpy")
                self.assertEqual(panel["data"]["observations"].shape, (200, maximum, 1))
                self.assertEqual(len(panel["data"]["trajectory_id"]), 200)
                self.assertEqual(panel["metadata"]["max_history_length"], maximum)
                other = simulation.teaching_task(model, observations=29)
                with self.assertRaisesRegex(ValueError, "tasks differ"):
                    diagnostics.evaluate(object(), other, panel, 21)

    def test_task_changes_in_prior_grid_and_numerical_law_cannot_reuse_a_panel(self):
        task, panel = self.panel()
        changes = []
        prior = copy.deepcopy(task)
        prior["prior_bounds"][0][0] += .001
        changes.append(prior)
        grid = simulation.teaching_task("rough-known", observations=30)
        changes.append(grid)
        numerics = copy.deepcopy(task)
        numerics["numerics"]["observation_substeps"] = 8
        changes.append(numerics)
        for changed in changes:
            with self.subTest(task=changed), mock.patch.object(inference, "sample") as sample:
                with self.assertRaises(ValueError):
                    diagnostics.evaluate(object(), changed, panel, 21)
                sample.assert_not_called()

    def test_failure_and_bad_generation_return_no_partial_or_selected_panel(self):
        task = simulation.teaching_task("rough-known", observations=29)
        with mock.patch.object(simulation, "generate", side_effect=FloatingPointError("failed prior row")) as generate:
            with self.assertRaisesRegex(FloatingPointError, "failed prior row"):
                diagnostics.generate_panel(task, count=3, seed=73)
            self.assertEqual(generate.call_count, 1)
        for field, value in (("observations", np.zeros((2, 29, 1))),
                             ("parameters", np.zeros((3, 4))),
                             ("trajectory_id", np.array(["wrong"] * 3))):
            data = self.pairs(task, 3, 73)
            data[field] = value
            with self.subTest(field=field), mock.patch.object(simulation, "generate", return_value=data) as generate:
                with self.assertRaises(ValueError):
                    diagnostics.generate_panel(task, count=3, seed=73)
                self.assertEqual(generate.call_count, 1)

    def test_prefix_mask_excludes_future_returns_and_never_conditions_on_parameter_labels(self):
        task, panel = self.panel()
        dimensions = len(task["parameter_names"])
        recorded = []

        class Workflow:
            def sample(self, **kwargs):
                conditions = kwargs["conditions"]
                recorded.append(conditions)
                n = len(conditions["observations"])
                state = conditions["observations"][:, :, 0].sum(axis=1) * .001
                return {"parameters": np.broadcast_to(state[:, None, None],
                        (n, kwargs["num_samples"], dimensions)).copy()}

        a = diagnostics.evaluate(Workflow(), task, panel, length=7, draws=5, seed=73)
        changed = copy.deepcopy(panel)
        changed["data"]["observations"][:, 7:] = 100000.
        changed["data"]["parameters"][:, 0] += .001
        b = diagnostics.evaluate(Workflow(), task, changed, length=7, draws=5, seed=73)
        np.testing.assert_array_equal(a, b)
        for name in recorded[0]:
            np.testing.assert_array_equal(recorded[0][name], recorded[1][name])
        self.assertEqual(set(recorded[0]), {"observations", "time_context", "summary_mask"})
        np.testing.assert_array_equal(recorded[0]["observations"][:, 7:], 0)
        np.testing.assert_array_equal(recorded[0]["summary_mask"][:, :7], True)
        np.testing.assert_array_equal(recorded[0]["summary_mask"][:, 7:], False)
        self.assertEqual(a.shape, (3, 5, dimensions))
        # The panel itself remains complete rather than being truncated/masked.
        self.assertEqual(panel["data"]["observations"].shape, (3, 29, 1))
        self.assertTrue(np.any(panel["data"]["observations"][:, 7:] != 0))

    def test_evaluation_passes_full_returns_with_explicit_lengths_to_canonical_sampler(self):
        task, panel = self.panel()
        sentinel = np.broadcast_to(panel["data"]["parameters"][:, None], (3, 5, 4)).copy()
        workflow = object()
        with mock.patch.object(inference, "sample", return_value=sentinel) as sample:
            result = diagnostics.evaluate(workflow, task, panel, length=13, draws=5, seed=19)
        np.testing.assert_array_equal(result, sentinel)
        args, kwargs = sample.call_args
        self.assertIs(args[0], workflow)
        self.assertEqual(args[1], task)
        np.testing.assert_array_equal(args[2], panel["data"]["observations"])
        np.testing.assert_array_equal(kwargs["lengths"], [13, 13, 13])
        self.assertEqual(set(kwargs), {"draws", "lengths", "seed"})
        self.assertEqual(kwargs["seed"], 19)

    def test_prefix_and_work_limits_fail_before_calling_the_network_or_generator(self):
        task, panel = self.panel()
        with mock.patch.object(inference, "sample") as sample:
            for length in (0, 30, 7., True):
                with self.subTest(length=length), self.assertRaises(ValueError):
                    diagnostics.evaluate(object(), task, panel, length)
            sample.assert_not_called()
        with mock.patch.object(simulation, "generate") as generate:
            for count in (0, 1, 201, True):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    diagnostics.generate_panel(task, count=count)
            generate.assert_not_called()

    def test_metadata_ids_latent_fields_nonfinite_values_and_masks_are_rejected(self):
        task, panel = self.panel()
        invalid = []
        seed = copy.deepcopy(panel)
        seed["metadata"]["seed"] += 1
        invalid.append(seed)
        order = copy.deepcopy(panel)
        order["data"]["trajectory_id"] = order["data"]["trajectory_id"][::-1]
        invalid.append(order)
        latent = copy.deepcopy(panel)
        latent["data"]["latent_variance"] = np.ones((3, 29))
        invalid.append(latent)
        for field in ("observations", "parameters"):
            masked = copy.deepcopy(panel)
            masked["data"][field] = np.ma.array(masked["data"][field], mask=True)
            invalid.append(masked)
            nonfinite = copy.deepcopy(panel)
            nonfinite["data"][field].flat[0] = np.nan
            invalid.append(nonfinite)
        for value in invalid:
            with self.subTest(fields=list(value["data"])), self.assertRaises(ValueError):
                diagnostics.evaluate(object(), task, value, length=13)

    def test_named_diagnostics_preserve_paired_physical_rows_and_allow_rounded_posterior_bounds(self):
        for model in simulation.MODELS:
            task, panel = self.panel(model=model)
            values = np.repeat(panel["data"]["parameters"][:, None], 5, axis=1)
            bounds = np.asarray(task["prior_bounds"])
            values[0, 0], values[0, 1] = bounds[:, 0], bounds[:, 1]
            result = diagnostics.diagnostic_data(task, panel, values)
            for i, name in enumerate(task["parameter_names"]):
                np.testing.assert_array_equal(result["estimates"][name], values[:, :, i:i + 1])
                np.testing.assert_array_equal(result["targets"][name], panel["data"]["parameters"][:, i:i + 1])
            values[0, 0, 0] -= 1
            with self.assertRaisesRegex(ValueError, "physical prior"):
                diagnostics.diagnostic_data(task, panel, values)

    def test_native_plot_calls_use_all_named_physical_targets_and_ninety_percent_intervals(self):
        task, panel = self.panel(model="rough-joint")
        values = np.repeat(panel["data"]["parameters"][:, None], 5, axis=1)
        recovery, ranks = mock.Mock(return_value="recovery"), mock.Mock(return_value="ranks")
        backend = types.SimpleNamespace(diagnostics=types.SimpleNamespace(
            recovery=recovery, calibration_ecdf=ranks))
        with mock.patch.dict("sys.modules", {"bayesflow": backend}):
            self.assertEqual(diagnostics.plot_diagnostics(task, panel, values), ("recovery", "ranks"))
        self.assertEqual(recovery.call_args.kwargs["variable_keys"], task["parameter_names"])
        self.assertEqual(recovery.call_args.kwargs["uncertainty_agg_kwargs"], {"prob": .90})
        self.assertEqual(recovery.call_args.kwargs["num_col"], 3)
        self.assertTrue(ranks.call_args.kwargs["difference"])
        self.assertEqual(set(ranks.call_args.kwargs["targets"]), set(task["parameter_names"]))


if __name__ == "__main__":
    unittest.main()

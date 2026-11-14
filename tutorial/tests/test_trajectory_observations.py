"""Fresh observations cannot borrow a preserved case's posterior or hidden state."""

import copy
import importlib.abc
import sys
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import evidence, inference, lesson, observations, simulation


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"bayesflow", "keras", "jax"}:
            raise AssertionError("observation construction initialized a neural backend: " + fullname)
        return None


class ObservationTests(unittest.TestCase):
    def test_preserved_histories_keep_original_bytes_truth_identity_and_paired_states(self):
        for name in simulation.MODELS:
            task, _, arrays = lesson.load_example(name)
            for case in range(4):
                with self.subTest(model=name, case=case):
                    history = observations.saved_history(task, arrays, case)
                    self.assertEqual(history["source"], "saved")
                    self.assertEqual(history["case"], case)
                    self.assertEqual(history["id"], arrays["trajectory_id"][case])
                    np.testing.assert_array_equal(history["returns"], arrays["observations"][case])
                    np.testing.assert_array_equal(history["truth"], arrays["truth"][case])
                    np.testing.assert_array_equal(history["paired_history"]["returns"], history["returns"][:, 0])
                    self.assertEqual(history["paired_history"]["variance"].shape, (1261,))
                    self.assertFalse(history["returns"].flags.writeable)
                    self.assertFalse(history["truth"].flags.writeable)
                    self.assertFalse(history["paired_history"]["variance"].flags.writeable)

    def test_custom_all_models_use_the_unchanged_full_task_and_actual_same_simulation_states(self):
        finder = NoNeuralImports()
        sys.meta_path.insert(0, finder)
        try:
            for name in simulation.MODELS:
                with self.subTest(model=name):
                    task = simulation.teaching_task(name)
                    original = copy.deepcopy(task)
                    theta = np.mean(task["prior_bounds"], axis=1)
                    history = observations.custom_history(task, dict(zip(task["parameter_names"], theta)), 73)
                    expected = simulation.simulate_parameters(task, theta[None], seed=73, backend="numpy",
                                                              return_details=True)
                    self.assertEqual(task, original)
                    self.assertEqual(history["source"], "custom")
                    self.assertIsNone(history["case"])
                    self.assertEqual(history["seed"], 73)
                    self.assertTrue(history["id"].startswith("custom:"))
                    self.assertEqual(history["returns"].shape, (1260, 1))
                    np.testing.assert_array_equal(history["returns"], expected["observations"][0])
                    for key, values in expected["latent"].items():
                        np.testing.assert_array_equal(history["paired_history"][key], values[0])
                    self.assertEqual(history["paired_history"]["days"].dtype.kind, "i")
                    self.assertFalse(history["paired_history"]["metadata"]["inference_input"])
                    self.assertFalse(history["paired_history"]["metadata"]["state_estimate"])
        finally:
            sys.meta_path.remove(finder)

    def test_reproducible_identity_changes_with_parameters_seed_source_and_exact_prefix(self):
        task, _, arrays = lesson.load_example("rough-known")
        theta = np.mean(task["prior_bounds"], axis=1)
        first = observations.custom_history(task, theta, 73)
        repeated = observations.custom_history(task, theta.copy(), 73)
        self.assertEqual(first["id"], repeated["id"])
        self.assertEqual(observations.selection_key(task, first, 137),
                         observations.selection_key(task, repeated, 137))
        changed = theta.copy()
        changed[1] += .1
        candidates = [observations.custom_history(task, changed, 73),
                      observations.custom_history(task, theta, 74), observations.saved_history(task, arrays, 0)]
        self.assertEqual(len({first["id"], *(row["id"] for row in candidates)}), 4)
        for history in candidates:
            self.assertNotEqual(observations.selection_key(task, first, 137),
                                observations.selection_key(task, history, 137))
        self.assertNotEqual(observations.selection_key(task, first, 137),
                            observations.selection_key(task, first, 138))
        for length in (0, 1261, True, 1.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                observations.selection_key(task, first, length)

    def test_chosen_parameters_and_task_are_validated_without_clipping_or_redrawing(self):
        task = simulation.teaching_task("rough-known")
        theta = np.mean(task["prior_bounds"], axis=1)
        invalid = [theta[:-1], theta[None], theta.astype(complex), np.ma.array(theta, mask=True),
                   np.full_like(theta, np.nan), np.array(["bad"]*len(theta)), {"eta": 1.}]
        outside = theta.copy()
        outside[0] = task["prior_bounds"][0][0] - 1e-6
        invalid.append(outside)
        with mock.patch.object(simulation, "simulate_parameters", side_effect=AssertionError("must validate first")):
            for values in invalid:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    observations.custom_history(task, values, 73)
            for seed in (-1, True, 1.5, 2**64):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    observations.custom_history(task, theta, seed)
            for changed in (simulation.teaching_task("rough-known", observations=252), copy.deepcopy(task)):
                if changed == task:
                    changed["prior_bounds"][0][1] = .45
                with self.subTest(task=changed), self.assertRaisesRegex(ValueError, "unchanged five-year"):
                    observations.custom_history(changed, theta, 73)

    def test_saved_fallback_uses_original_paired_draws_and_matching_live_translation(self):
        task, _, arrays = lesson.load_example("rough-known")
        history = observations.saved_history(task, arrays, 2)
        saved = observations.posterior_view(task, arrays, history, 137)
        self.assertEqual(saved["source"], "saved")
        self.assertEqual(saved["length"], 126)
        np.testing.assert_array_equal(saved["draws"], arrays["draws_126"][2])
        key = observations.selection_key(task, history, 137)
        candidate = {"selection": key, "draws": arrays["draws_126"][1]}
        fresh = observations.posterior_view(task, arrays, history, 137, candidate)
        self.assertEqual(fresh["source"], "live")
        self.assertEqual(fresh["length"], 137)
        self.assertEqual(fresh["selection"], key)
        np.testing.assert_array_equal(fresh["draws"], candidate["draws"])
        stale = observations.posterior_view(task, arrays, history, 138, candidate)
        self.assertEqual(stale["source"], "saved")
        self.assertEqual(stale["length"], 126)

    def test_custom_has_no_fallback_even_when_saved_data_or_stale_native_draws_are_available(self):
        task, _, arrays = lesson.load_example("rough-known")
        history = observations.custom_history(task, np.mean(task["prior_bounds"], axis=1), 73)
        with mock.patch.object(lesson, "posterior_view", side_effect=AssertionError("no saved fallback")):
            self.assertIsNone(observations.posterior_view(task, arrays, history, 137))
            candidate = {"selection": observations.selection_key(task, history, 137),
                         "draws": arrays["draws_126"][0]}
            self.assertIsNone(observations.posterior_view(task, arrays, history, 138, candidate))
            other = observations.custom_history(task, history["truth"], 74)
            self.assertIsNone(observations.posterior_view(task, arrays, other, 137, candidate))
            fresh = observations.posterior_view(task, arrays, history, 137, candidate)
            self.assertEqual(fresh["source"], "live")
            self.assertEqual(fresh["length"], 137)
            np.testing.assert_array_equal(fresh["draws"], candidate["draws"])

    def test_matching_fresh_rows_require_complete_finite_real_values_in_support(self):
        task = simulation.teaching_task("rough-known")
        theta = np.mean(task["prior_bounds"], axis=1)
        history = observations.custom_history(task, theta, 73)
        rows = np.tile(theta, (4, 1))
        invalid = [rows[:1], rows[:, :-1], rows.astype(complex), np.ma.array(rows, mask=True),
                   np.full_like(rows, np.nan), np.full(rows.shape, "bad")]
        outside = rows.copy()
        outside[1, 0] = task["prior_bounds"][0][1] + 1e-6
        invalid.append(outside)
        for draws in invalid:
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                observations.posterior_view(task, None, history, 137,
                    {"selection": observations.selection_key(task, history, 137), "draws": draws})

    def test_history_bytes_truth_and_saved_case_cannot_be_relabelled(self):
        task, _, arrays = lesson.load_example("rough-known")
        history = observations.saved_history(task, arrays, 0)
        wrong = copy.deepcopy(arrays)
        wrong["observations"][0, 0, 0] += 1e-6
        with self.assertRaisesRegex(ValueError, "exact preserved history"):
            observations.posterior_view(task, wrong, history, 137)
        for field in ("returns", "truth"):
            changed = dict(history)
            changed[field] = changed[field].copy()
            changed[field].flat[-1] += 1e-6
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "bytes or generating parameters"):
                observations.selection_key(task, changed, 137)

    def test_custom_endpoint_truth_is_preserved_but_training_label_transform_stays_strict(self):
        for model in simulation.MODELS:
            task = simulation.teaching_task(model)
            bounds = np.asarray(task["prior_bounds"])
            draws = np.tile(bounds.mean(axis=1), (4, 1))
            for side in (0, 1):
                with self.subTest(model=model, side=side):
                    truth = bounds[:, side]
                    history = observations.custom_history(task, truth, 73)
                    np.testing.assert_array_equal(history["truth"], truth)
                    reading = lesson.posterior_readout(task, draws, truth=history["truth"])
                    self.assertEqual(reading["truth_excluded_parameters"], task["parameter_names"])
                    with self.assertRaisesRegex(ValueError, "strictly inside"):
                        inference.transform_parameters(truth, task)

    def test_custom_evidence_and_conditioning_use_this_history_without_truth_or_future_inputs(self):
        task = simulation.teaching_task("rough-known")
        history = observations.custom_history(task, np.mean(task["prior_bounds"], axis=1), 73)
        manager = evidence.native_manager(task, history["returns"], "matching-checkpoint", history["id"])
        self.assertEqual(manager["metadata"]["history_sha256"], history["history_sha256"])
        self.assertEqual(evidence.frame_list(manager, 137), [])
        conditions = inference.conditioning_data(history["returns"][None], task, lengths=np.array([137]))
        self.assertEqual(set(conditions), {"observations", "time_context", "summary_mask"})
        self.assertTrue(conditions["summary_mask"][0, :137].all())
        self.assertFalse(conditions["summary_mask"][0, 137:].any())
        np.testing.assert_array_equal(conditions["observations"][0, 137:], 0)
        changed = history["returns"].copy()
        changed[137:] = np.nan
        other = inference.conditioning_data(changed[None], task, lengths=np.array([137]))
        for key in conditions:
            np.testing.assert_array_equal(conditions[key], other[key])


if __name__ == "__main__":
    unittest.main()

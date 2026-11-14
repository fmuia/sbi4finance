"""Evidence playback uses actual retained posterior frames and exactly revealed data."""

import copy
import unittest
from unittest import mock

import numpy as np

from sbi4f_tutorial.trajectories import evidence, inference, simulation


class EvidenceFrameTests(unittest.TestCase):
    def setUp(self):
        self.task = simulation.teaching_task("rough-known", observations=63)
        pairs = simulation.generate(self.task, "validation", 0, 4, seed=73, backend="numpy")
        self.arrays = {"observations": pairs["observations"], "trajectory_id": pairs["trajectory_id"],
                       "truth": pairs["parameters"]}
        for day in (1, 4, 21, 63):
            self.arrays[f"draws_{day}"] = np.repeat(pairs["parameters"][:, None, :], 8, axis=1)

    def fake_sample(self, workflow, task, values, draws, *, lengths, seed):
        # These conditions are what the real sampling helper supplies downstream.
        prepared = inference.conditioning_data(values, task, lengths=lengths)
        for row, length in enumerate(lengths):
            self.assertTrue(np.all(values[row, length:] == 0))
            self.assertTrue(np.all(prepared["observations"][row, length:] == 0))
            self.assertTrue(np.all(prepared["summary_mask"][row, :length]))
            self.assertFalse(np.any(prepared["summary_mask"][row, length:]))
        midpoint = np.mean(task["prior_bounds"], axis=1)
        return np.broadcast_to(midpoint, (len(values), draws, len(midpoint))).copy()

    def test_saved_playback_retains_actual_snapshots_without_interpolation(self):
        manager = evidence.saved_manager(self.task, self.arrays, 2, "checkpoint-A")
        frames = evidence.frame_list(manager, 20)
        self.assertEqual([frame["length"] for frame in frames], [1, 4])
        np.testing.assert_array_equal(frames[1]["draws"], self.arrays["draws_4"][2])
        selected = evidence.select_frame(manager, 20)
        self.assertEqual(selected["length"], 4)
        self.assertEqual(selected["requested_length"], 20)
        self.assertFalse(selected["exact"])
        self.assertIn("nearest actual past", selected["label"])
        np.testing.assert_array_equal(selected["observed_prefix"], self.arrays["observations"][2, :4])
        self.assertTrue(evidence.select_frame(manager, 21)["exact"])
        # A rendering payload cannot alter the retained parameter pairs.
        frames[1]["draws"].fill(0.)
        np.testing.assert_array_equal(evidence.select_frame(manager, 4)["draws"], self.arrays["draws_4"][2])

    def test_137_day_request_uses_only_actual_126_day_saved_frame_and_history(self):
        task = simulation.teaching_task("rough-known", observations=252)
        midpoint = np.mean(task["prior_bounds"], axis=1)
        arrays = {"observations": np.zeros((1, 252, 1)), "trajectory_id": np.array(["one-history"]),
                  "draws_1": np.broadcast_to(midpoint, (1, 8, 4)),
                  "draws_126": np.broadcast_to(midpoint, (1, 8, 4)),
                  "draws_252": np.broadcast_to(midpoint, (1, 8, 4))}
        frame = evidence.select_frame(evidence.saved_manager(task, arrays, 0, "checkpoint-A"), 137)
        self.assertEqual(frame["length"], 126)
        self.assertEqual(frame["observed_prefix"].shape, (126, 1))
        self.assertNotIn("137", str(frame["metadata"]["actual_length"]))

    def test_native_queries_mask_future_before_sampling_and_never_update_original_prior(self):
        observed = self.arrays["observations"][0].copy()
        observed[7:] = np.nan  # unavailable future never reaches the estimator
        original_task = copy.deepcopy(self.task)
        manager = evidence.native_manager(self.task, observed, "checkpoint-A", "history-A", seed=17)
        with mock.patch.object(inference, "sample", side_effect=self.fake_sample) as sample:
            result = evidence.prepare_frames(manager, object(), [7, 1], batch_size=8)
        self.assertEqual(result["new_lengths"], [1, 7])
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(sample.call_args.args[1], original_task)
        self.assertEqual(sample.call_args.args[2].shape, (2, 63, 1))
        np.testing.assert_array_equal(sample.call_args.kwargs["lengths"], [1, 7])
        self.assertEqual(self.task, original_task)
        self.assertEqual(evidence.select_frame(manager, 7)["observed_prefix"].shape, (7, 1))
        self.assertEqual([frame["length"] for frame in evidence.frame_list(manager, 63)], [1, 7])
        self.assertEqual(evidence.select_frame(manager, 63)["length"], 7)
        with mock.patch.object(inference, "sample") as sample:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                evidence.prepare_frames(manager, object(), [8])
            sample.assert_not_called()

    def test_existing_samples_are_reused_and_a_failed_query_never_publishes_partial_frames(self):
        manager = evidence.native_manager(self.task, self.arrays["observations"][0], "checkpoint-A", "history-A")
        with mock.patch.object(inference, "sample", side_effect=self.fake_sample):
            evidence.prepare_frames(manager, object(), [1, 3])
        before = evidence.select_frame(manager, 1)["draws"].copy()
        with mock.patch.object(inference, "sample") as sample:
            result = evidence.prepare_frames(manager, object(), [3, 1])
            sample.assert_not_called()
        self.assertEqual(result["reused_lengths"], [1, 3])
        self.assertEqual(result["new_lengths"], [])
        good = np.broadcast_to(np.mean(self.task["prior_bounds"], axis=1), (1, 128, 4)).copy()
        with mock.patch.object(inference, "sample", side_effect=[good, RuntimeError("backend failed")]):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                evidence.prepare_frames(manager, object(), [5, 7], batch_size=1)
        self.assertEqual([f["length"] for f in evidence.frame_list(manager, 63)], [1, 3])
        np.testing.assert_array_equal(evidence.select_frame(manager, 1)["draws"], before)

    def test_context_keys_bind_task_checkpoint_history_source_draw_count_and_seed(self):
        observed = self.arrays["observations"][0].copy()
        base = evidence.native_manager(self.task, observed, "checkpoint-A", "history-A")
        changes = [evidence.native_manager(self.task, observed, "checkpoint-B", "history-A"),
                   evidence.native_manager(self.task, observed, "checkpoint-A", "history-B"),
                   evidence.native_manager(self.task, observed, "checkpoint-A", "history-A", draws=256),
                   evidence.native_manager(self.task, observed, "checkpoint-A", "history-A", seed=8),
                   evidence.saved_manager(self.task, self.arrays, 0, "checkpoint-A")]
        observed[0] += .001
        changes.append(evidence.native_manager(self.task, observed, "checkpoint-A", "history-A"))
        other = simulation.teaching_task("heston", observations=63)
        changes.append(evidence.native_manager(other, observed, "checkpoint-A", "history-A"))
        self.assertTrue(all(item["context_id"] != base["context_id"] for item in changes))
        # Changing external inputs cannot mutate the frozen history or task.
        self.task["prior_bounds"][0][0] = .04
        evidence.frame_list(base, 63)
        base["metadata"]["checkpoint_id"] = "checkpoint-B"
        with self.assertRaisesRegex(ValueError, "context changed"):
            evidence.frame_list(base, 63)

    def test_schedule_is_bounded_exact_and_does_not_silently_coarsen(self):
        self.assertEqual(evidence.exact_schedule(10, step=5), [1, 6, 10])
        self.assertEqual(evidence.exact_schedule(21), list(range(1, 22)))
        with self.assertRaisesRegex(ValueError, "needs 63 frames"):
            evidence.exact_schedule(63)
        for target, step in ((0, 1), (7.5, 1), (21, 0), (21, True)):
            with self.subTest(target=target, step=step), self.assertRaises(ValueError):
                evidence.exact_schedule(target, step)

    def test_dense_saved_frames_require_same_history_and_do_not_replace_existing_samples(self):
        manager = evidence.saved_manager(self.task, self.arrays, 0, "checkpoint-A")
        original = evidence.select_frame(manager, 4)["draws"].copy()
        metadata = manager["metadata"]
        identity = {key: metadata[key] for key in ("task_sha256", "checkpoint_id", "history_id", "history_sha256")}
        extra = {2: self.arrays["draws_4"][0], 4: self.arrays["draws_21"][0]}
        with self.assertRaisesRegex(ValueError, "differs"):
            evidence.add_saved_frames(manager, extra, **dict(identity, checkpoint_id="checkpoint-B"))
        result = evidence.add_saved_frames(manager, extra, **identity)
        self.assertEqual(result, {"new_lengths": [2], "reused_lengths": [4]})
        np.testing.assert_array_equal(evidence.select_frame(manager, 4)["draws"], original)
        self.assertEqual([f["length"] for f in evidence.frame_list(manager, 20)], [1, 2, 4])
        with self.assertRaises(ValueError):
            evidence.add_saved_frames(manager, {2.5: extra[2]}, **identity)

    def test_cross_context_frame_or_invalid_physical_draws_are_not_rendered(self):
        manager = evidence.saved_manager(self.task, self.arrays, 0, "checkpoint-A")
        other = evidence.saved_manager(self.task, self.arrays, 1, "checkpoint-A")
        manager["frames"][1] = other["frames"][1]
        with self.assertRaisesRegex(ValueError, "another task"):
            evidence.frame_list(manager, 1)
        native = evidence.native_manager(self.task, self.arrays["observations"][0], "checkpoint-A", "history-A")
        with mock.patch.object(inference, "sample", return_value=np.zeros((1, 128, 4))):
            with self.assertRaisesRegex(ValueError, "prior support"):
                evidence.prepare_frames(native, object(), [1])
        self.assertEqual(native["frames"], {})


if __name__ == "__main__":
    unittest.main()

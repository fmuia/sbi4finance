"""Visual controls preserve real data, masks and generating-value annotations."""

import copy
import unittest

import numpy as np

from sbi4f_tutorial.trajectories import (
    evidence_widget, history_widget, inference, latent_assets, lesson, simulation, training_widget,
)


class HistoryWidgetTests(unittest.TestCase):
    def test_all_models_show_original_returns_and_verified_effective_boundary_variance(self):
        for model in simulation.MODELS:
            with self.subTest(model=model):
                task, _, arrays = lesson.load_example(model)
                paired = latent_assets.load_history(task, arrays, 0)
                payload = history_widget.build_payload(paired, end=1260)
                np.testing.assert_array_equal(payload["returns"], 100 * arrays["observations"][0, :, 0])
                np.testing.assert_array_equal(payload["variance"], paired["variance"])
                self.assertEqual(len(payload["variance"]), 1261)
                self.assertTrue(np.all(np.asarray(payload["variance"]) >= 0))
                self.assertIn("never supplied", payload["scope"])

    def test_revealed_display_ignores_unselected_suffix_and_does_not_change_pair(self):
        values = {"returns": np.linspace(-.01, .01, 29), "variance": np.linspace(.04, .02, 30)}
        before = copy.deepcopy(values)
        expected = history_widget.build_payload(values, end=21)
        values["returns"][21:] = np.nan
        values["variance"][22:] = np.nan
        self.assertEqual(history_widget.build_payload(values, end=21), expected)
        np.testing.assert_array_equal(values["returns"][:21], before["returns"][:21])
        for bad in (-1, 0, True):
            with self.subTest(end=bad), self.assertRaises(ValueError):
                history_widget.build_payload(before, end=bad)
        wrong = {**before, "variance": -before["variance"]}
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            history_widget.build_payload(wrong, end=21)


class TrainingWidgetTests(unittest.TestCase):
    def setUp(self):
        self.task = simulation.teaching_task("rough-known", observations=252)
        self.pair = simulation.generate(self.task, "train", 0, 1, seed=20261107, backend="numpy")

    def test_animation_uses_the_actual_preprocessing_tokens_masks_and_known_times(self):
        observations = self.pair["observations"][0].copy()
        observations[0] = 0.
        payload = training_widget.build_payload(self.task, observations)
        self.assertEqual([example["length"] for example in payload["examples"]], [1, 21, 63, 126, 252])
        for example in payload["examples"]:
            n = example["length"]
            actual = inference.conditioning_data(observations[None, :n], self.task)
            np.testing.assert_array_equal(example["tokens"], actual["observations"][0])
            np.testing.assert_array_equal(example["mask"], actual["summary_mask"][0])
            np.testing.assert_array_equal(example["time_context"], actual["time_context"][0])
            self.assertTrue(example["mask"][0])
            self.assertEqual(example["tokens"][0][0], 0.)
            self.assertEqual(sum(example["mask"]), n)

    def test_later_returns_never_enter_shorter_prefix_tokens(self):
        observations = self.pair["observations"][0].copy()
        original = training_widget.build_payload(self.task, observations)
        observations[21:] += .3
        changed = training_widget.build_payload(self.task, observations)
        self.assertEqual(original["examples"][:2], changed["examples"][:2])
        self.assertNotEqual(original["examples"][-1]["tokens"], changed["examples"][-1]["tokens"])
        html = training_widget.render_html(self.task, observations)
        self.assertIn("not measured neural activations", html)
        self.assertIn('id="play"', html)
        self.assertIn("not an autoregressive posterior decoder", training_widget.encoder_comparison_html())


class EvidenceTruthMarkerTests(unittest.TestCase):
    def test_generating_markers_annotate_without_changing_any_posterior_frame(self):
        task = simulation.teaching_task("rough-known", observations=29)
        bounds = np.asarray(task["prior_bounds"])
        rows = bounds[:, 0] + np.linspace(.2, .8, 10)[:, None] * (bounds[:, 1]-bounds[:, 0])
        frame = {"length": 21, "draws": rows, "source": "saved", "frame_id": "actual21"}
        plain = evidence_widget.build_payload(task, np.zeros(29), [frame], end=29)
        truth = rows[3].copy()
        marked = evidence_widget.build_payload(task, np.zeros(29), [frame], end=29, truth=truth)
        self.assertEqual(plain["frames"], marked["frames"])
        np.testing.assert_array_equal([coordinate["truth"] for coordinate in marked["coordinates"]], truth)
        for wrong in (truth[:-1], np.full(len(truth), np.nan), np.zeros(len(truth)), truth.astype(complex)):
            with self.subTest(shape=wrong.shape), self.assertRaises(ValueError):
                evidence_widget.build_payload(task, np.zeros(29), [frame], end=29, truth=wrong)


if __name__ == "__main__":
    unittest.main()

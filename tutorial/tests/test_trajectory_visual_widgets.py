"""Visual controls preserve real data, masks and generating-value annotations."""

import copy
from html.parser import HTMLParser
import json
import unittest

import numpy as np

from sbi4f_tutorial.trajectories import (
    evidence_widget, history_widget, inference, latent_assets, lesson, simulation, training_widget,
)


class HistoryWidgetTests(unittest.TestCase):
    def test_presentation_options_omit_annotations_without_altering_history_values(self):
        from itertools import product

        class Elements(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids, self.tags = set(), []

            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)
                self.ids.add(dict(attrs).get("id"))

        history = {"returns": np.array([.01, -.01, 0.]),
                   "variance": np.array([.04, 1e-8, 0., 1e-15]),
                   "mean_variance": np.full(4, .04),
                   "median_variance": np.array([.04, .02, .01, .005])}
        expected = history_widget.build_payload(history, end=3)
        for header, footer, references in product((True, False), repeat=3):
            with self.subTest(header=header, footer=footer, references=references):
                rendered = history_widget.render_html(history, end=3, show_header=header,
                    show_footer=footer, show_references=references)
                payload, _ = json.JSONDecoder().raw_decode(rendered.split("const data=", 1)[1])
                self.assertEqual({key: payload[key] for key in expected}, expected)
                self.assertEqual(payload["presentation"]["show_references"], references)
                elements = Elements()
                elements.feed(rendered)
                self.assertEqual("title" in elements.ids, header)
                self.assertEqual("description" in elements.ids, header)
                for name in ("variance-legend", "variance-readout", "counts"):
                    self.assertEqual(name in elements.ids, footer)
                self.assertEqual(elements.tags.count("svg"), 4)
                for name in ("returns", "variance", "returnhist", "variancehist", "scrub", "play"):
                    self.assertIn(name, elements.ids)
        default = history_widget.render_html(history, end=3)
        payload, _ = json.JSONDecoder().raw_decode(default.split("const data=", 1)[1])
        self.assertTrue(all(payload["presentation"][key]
                            for key in ("show_header", "show_footer", "show_references")))

    def test_notebook_labels_are_text_data_and_cannot_inject_html(self):
        history = {"returns": np.array([0., -.01]), "variance": np.full(3, .04)}
        literal = '</script><img src="invalid" onerror="alert(1)">'
        labels = {"returns": literal, "variance": "Annual variance V",
                  "returnhist": "Return distribution", "variancehist": "Variance distribution",
                  "description": "Notebook-owned explanation"}
        rendered = history_widget.render_html(history, end=2, labels=labels)
        payload, _ = json.JSONDecoder().raw_decode(rendered.split("const data=", 1)[1])
        self.assertEqual(payload["presentation"]["labels"], labels)
        self.assertNotIn('</script><img', rendered)
        self.assertNotIn('<img src="invalid"', rendered)
        self.assertIn('<\\/script>', rendered)

    def test_presentation_rejects_ambiguous_flags_and_nontext_labels(self):
        history = {"returns": np.zeros(2), "variance": np.full(3, .04)}
        for flag in ("show_header", "show_footer", "show_references"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "booleans"):
                history_widget.render_html(history, end=2, **{flag: "false"})
        for labels in ([], {"unknown": "text"}, {"returns": 1}, {"description": None}):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "labels"):
                history_widget.render_html(history, end=2, labels=labels)

    def test_mean_and_median_references_remain_paired_and_tiny_values_are_not_floored(self):
        history = {"returns": np.array([.01, -.01, 0.]),
                   "variance": np.array([.04, 1e-8, 0., 1e-15]),
                   "mean_variance": np.full(4, .04),
                   "median_variance": np.array([.04, .02, .01, .005])}
        payload = history_widget.build_payload(history, end=3)
        np.testing.assert_array_equal(payload["variance"], history["variance"])
        np.testing.assert_array_equal(payload["mean_variance"], history["mean_variance"])
        np.testing.assert_array_equal(payload["median_variance"], history["median_variance"])
        short = history_widget.build_payload(history, end=1)
        history["mean_variance"][2:] = np.nan
        history["median_variance"][2:] = np.nan
        history["variance"][2:] = np.nan
        self.assertEqual(history_widget.build_payload(history, end=1), short)
        with self.assertRaisesRegex(ValueError, "references"):
            history_widget.build_payload({**history, "mean_variance": np.array([.04])}, end=1)

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
        self.task = simulation.teaching_task("rough-known", observations=1260)
        self.pair = simulation.generate(self.task, "train", 0, 1, seed=20261107, backend="numpy")

    def test_animation_uses_the_actual_preprocessing_tokens_masks_and_known_times(self):
        observations = self.pair["observations"][0].copy()
        observations[0] = 0.
        parameters = self.pair["parameters"][0]
        payload = training_widget.build_payload(self.task, observations, parameters=parameters)
        self.assertEqual([example["length"] for example in payload["examples"]], [1, 21, 63, 126, 252, 1260])
        for example in payload["examples"]:
            n = example["length"]
            actual = inference.conditioning_data(observations[None, :n], self.task)
            np.testing.assert_array_equal(example["tokens"], actual["observations"][0])
            np.testing.assert_array_equal(example["mask"], actual["summary_mask"][0])
            np.testing.assert_array_equal(example["time_context"], actual["time_context"][0])
            self.assertTrue(example["mask"][0])
            self.assertEqual(example["tokens"][0][0], 0.)
            self.assertEqual(sum(example["mask"]), n)
            np.testing.assert_array_equal(example["first"]["token"], actual["observations"][0, 0])
            np.testing.assert_array_equal(example["last"]["token"], actual["observations"][0, n-1])
            self.assertEqual(example["first"]["raw_return"], observations[0, 0])
            self.assertEqual(example["last"]["raw_return"], observations[n-1, 0])
            self.assertEqual(example["last"]["day"], n)
            if n < len(observations):
                self.assertIsNone(example["first_padding"]["raw_return"])
                self.assertFalse(example["first_padding"]["mask"])
                self.assertEqual(example["first_padding"]["day"], n+1)
                np.testing.assert_array_equal(example["first_padding"]["token"], [0., 0.])
            else:
                self.assertIsNone(example["first_padding"])

    def test_later_returns_never_enter_shorter_prefix_tokens(self):
        observations = self.pair["observations"][0].copy()
        parameters = self.pair["parameters"][0]
        original = training_widget.build_payload(self.task, observations, parameters=parameters)
        observations[21:] += .3
        changed = training_widget.build_payload(self.task, observations, parameters=parameters)
        self.assertEqual(original["examples"][:2], changed["examples"][:2])
        self.assertNotEqual(original["examples"][-1]["tokens"], changed["examples"][-1]["tokens"])
        self.assertEqual(original["target"], changed["target"])

    def test_generating_joint_label_and_float32_transform_stay_fixed_across_every_prefix(self):
        for name in simulation.MODELS:
            with self.subTest(model=name):
                task = simulation.teaching_task(name, observations=1260)
                pair = simulation.generate(task, "train", 0, 1, seed=20261107, backend="numpy")
                observations, parameters = pair["observations"][0], pair["parameters"][0]
                observations_before, parameters_before = observations.copy(), parameters.copy()
                payload = training_widget.build_payload(task, observations, parameters=parameters)
                self.assertEqual(payload["targets"], task["parameter_names"])
                self.assertEqual(payload["target"]["shape"], [1, len(parameters)])
                self.assertEqual(payload["target"]["route"], "training_label_only")
                np.testing.assert_array_equal(payload["target"]["physical"], parameters)
                transformed = None
                for example in payload["examples"]:
                    actual = inference.training_data(pair, task, lengths=np.array([example["length"]]))
                    self.assertEqual(actual["parameters"].dtype, np.float32)
                    np.testing.assert_array_equal(payload["target"]["transformed"], actual["parameters"][0])
                    if transformed is not None:
                        np.testing.assert_array_equal(actual["parameters"][0], transformed)
                    transformed = actual["parameters"][0]
                np.testing.assert_array_equal(observations, observations_before)
                np.testing.assert_array_equal(parameters, parameters_before)

    def test_training_label_is_optional_and_never_changes_neural_condition_payloads(self):
        observations, parameters = self.pair["observations"][0], self.pair["parameters"][0]
        inference_display = training_widget.build_payload(self.task, observations)
        training_display = training_widget.build_payload(self.task, observations, parameters=parameters)
        self.assertIsNone(inference_display["target"])
        self.assertEqual(inference_display["examples"], training_display["examples"])
        self.assertEqual(inference_display["raw_returns_percent"], training_display["raw_returns_percent"])
        for example in training_display["examples"]:
            self.assertTrue(set(example).isdisjoint({"parameters", "target", "truth", "physical", "transformed", "variance"}))
            actual = inference.conditioning_data(observations[None, :example["length"]], self.task)
            self.assertEqual(set(actual), {"observations", "summary_mask", "time_context"})
            np.testing.assert_array_equal(example["tokens"], actual["observations"][0])

    def test_invalid_generating_labels_cannot_be_silently_truncated_or_transformed(self):
        parameters = self.pair["parameters"][0]
        bounds = np.asarray(self.task["prior_bounds"])
        at_boundary = parameters.copy()
        at_boundary[0] = bounds[0, 0]
        outside = parameters.copy()
        outside[1] = bounds[1, 1]+.01
        for invalid in (parameters[:-1], parameters[None], np.r_[parameters, .1],
                        np.full_like(parameters, np.nan), np.full_like(parameters, np.inf),
                        parameters.astype(complex), np.ma.array(parameters, mask=False),
                        at_boundary, outside):
            with self.subTest(shape=np.shape(invalid)), self.assertRaises(ValueError):
                training_widget.build_payload(self.task, self.pair["observations"][0], parameters=invalid)


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

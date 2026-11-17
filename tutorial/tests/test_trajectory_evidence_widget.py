"""Evidence displays retain actual frames and the exact declared prior."""

import copy
import unittest

import numpy as np

from sbi4f_tutorial.trajectories import evidence, evidence_widget, simulation


class EvidenceWidgetTests(unittest.TestCase):
    def setUp(self):
        self.task = simulation.teaching_task("rough-joint", observations=29)
        self.history = np.linspace(-.02, .03, 29)
        bounds = np.asarray(self.task["prior_bounds"])
        self.draws = bounds[:, 0] + np.linspace(.15, .85, 11)[:, None] * (bounds[:, 1] - bounds[:, 0])
        self.frames = [{"length": day, "draws": self.draws.copy(), "source": "saved", "frame_id": f"actual-{day}"}
                       for day in (1, 8, 21)]

    def test_payload_has_analytic_prior_then_only_actual_frame_days_and_paired_draw_summaries(self):
        result = evidence_widget.build_payload(self.task, self.history[:, None], self.frames, end=21)
        self.assertEqual([frame["length"] for frame in result["frames"]], [0, 1, 8, 21])
        self.assertEqual(result["frames"][0]["count"], 0)
        self.assertEqual([frame["frame_id"] for frame in result["frames"]][1:], ["actual-1", "actual-8", "actual-21"])
        expected = np.quantile(self.draws, [.05, .5, .95], axis=0).T
        np.testing.assert_array_equal(result["frames"][-1]["quantiles"], expected)
        for index, coordinate in enumerate(result["coordinates"]):
            widths = np.diff(coordinate["edges"])
            for frame in result["frames"]:
                self.assertAlmostEqual(np.sum(np.asarray(frame["densities"][index]) * widths), 1.)
            self.assertGreater(coordinate["density_max"], max(frame["densities"][index][j]
                                                           for frame in result["frames"] for j in range(24)))
        decay = self.task["parameter_names"].index("decay")
        low, high = self.task["prior_bounds"][decay]
        np.testing.assert_allclose(result["frames"][0]["quantiles"][decay],
                                   np.exp(np.log(low) + np.array([.05, .5, .95]) * np.log(high / low)))

    def test_hidden_suffix_never_changes_path_payload_or_prior_and_actual_frames(self):
        a = evidence_widget.build_payload(self.task, self.history, self.frames, end=21)
        changed = self.history.copy()
        changed[21:] = np.nan
        b = evidence_widget.build_payload(self.task, changed, self.frames, end=21)
        self.assertEqual(a, b)
        self.assertEqual(len(a["returns"]), 21)
        with self.assertRaisesRegex(ValueError, "finite observed prefix"):
            evidence_widget.build_payload(self.task, changed, self.frames, end=29)

    def test_bad_or_relabelled_frame_axes_and_support_are_rejected(self):
        for bad in (self.draws[:, :-1], self.draws.astype(complex), np.ma.array(self.draws, mask=False),
                    np.full_like(self.draws, np.nan), np.zeros_like(self.draws)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                evidence_widget.build_payload(self.task, self.history, [{**self.frames[0], "draws": bad}], end=21)
        for day in (True, 0, 22):
            with self.subTest(day=day), self.assertRaises(ValueError):
                evidence_widget.build_payload(self.task, self.history, [{**self.frames[0], "length": day}], end=21)
        with self.assertRaisesRegex(ValueError, "unique exact"):
            evidence_widget.build_payload(self.task, self.history, [self.frames[0], self.frames[0]], end=21)

    def test_html_is_self_contained_and_escapes_document_closing_text(self):
        value = evidence_widget.render_html(self.task, self.history, self.frames, end=21,
                                            history_label="case </script><script>bad()</script>")
        self.assertNotIn("case </script>", value)
        self.assertIn("case <\\/script>", value)
        self.assertIn('aria-label="Actual posterior frame"', value)
        self.assertIn('id="play"', value)
        self.assertIn('id="advance"', value)
        self.assertIn('id="reset"', value)
        self.assertNotIn('src="http', value)
        self.assertNotIn('fetch(', value)

    def test_real_manager_frames_display_without_creating_intermediate_posteriors(self):
        pairs = simulation.generate(self.task, "validation", 0, 1, seed=73, backend="numpy")
        arrays = {"observations": pairs["observations"], "trajectory_id": pairs["trajectory_id"],
                  **{f"draws_{day}": self.draws[None].copy() for day in (1, 8, 21)}}
        manager = evidence.saved_manager(self.task, arrays, 0, "checkpoint-identity")
        before = copy.deepcopy(manager["frames"])
        payload = evidence_widget.build_payload(self.task, arrays["observations"][0], evidence.frame_list(manager, 17), end=17)
        self.assertEqual([frame["length"] for frame in payload["frames"]], [0, 1, 8])
        self.assertEqual(set(manager["frames"]), set(before))
        for day in before:
            np.testing.assert_array_equal(manager["frames"][day]["draws"], before[day]["draws"])


if __name__ == "__main__":
    unittest.main()

"""Recovery clicks map original data-set IDs across all parameter panels."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


HELPER = Path(__file__).resolve().parents[1] / "notebooks/_heston_recovery_picker.py"
SPEC = importlib.util.spec_from_file_location("heston_recovery_picker_under_test", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(importlib.util.find_spec("plotly"), "requires the notebook's Plotly dependency")
class HestonRecoveryPickerTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.random.default_rng(12).uniform(size=(64, 6))
        self.q = np.stack([self.truth - .1, self.truth + .02, self.truth + .15])
        self.bounds = np.tile([0., 1.], (6, 1))
        self.figure = MODULE.recovery_figure(self.truth, self.q, self.bounds, list("abcdef"))

    def test_plain_clicks_use_stable_dataset_ids_in_every_panel(self):
        self.assertEqual(self.figure.layout.clickmode, "event+select")
        self.assertFalse(self.figure.layout.dragmode)
        self.assertEqual(len(self.figure.data), 12)
        for parameter in range(6):
            points = [{"curveNumber": 2 * parameter + 1, "pointIndex": 47, "x": .2, "y": .3}]
            self.assertEqual(MODULE.selected_case(points, len(self.truth), self.figure), 47)
            trace = self.figure.data[2 * parameter + 1]
            self.assertEqual(list(trace.customdata[47][:2]), [47, parameter])

    def test_identity_does_not_depend_on_coordinate_rounding_or_trace_order(self):
        # Browser events are not required to preserve x/y or customdata.
        points = [{"curveNumber": 7, "pointNumber": 23}]
        self.assertEqual(MODULE.selected_case(points, 64, self.figure), 23)
        self.assertEqual(MODULE.selected_case([{"customdata": [23, 3, 0., 1.]}], 64), 23)
        self.assertEqual(MODULE.selected_case([], 64, self.figure), 0)
        self.assertEqual(MODULE.selected_case([{"curveNumber": 0, "pointIndex": 0}], 64, self.figure), 0)

    def test_the_plot_preserves_values_outside_prior_bounds(self):
        trace = self.figure.data[1]
        np.testing.assert_array_equal(trace.x, self.truth[:, 0])
        np.testing.assert_array_equal(trace.y, self.q[1, :, 0])
        self.assertLess(self.figure.layout.yaxis.range[0], self.q[0, :, 0].min())
        self.assertGreater(self.figure.layout.yaxis.range[1], self.q[2, :, 0].max())
        self.assertEqual(np.count_nonzero(trace.error_y.array), 32)

    def test_invalid_selection_ids_cannot_index_another_dataset(self):
        for points in ([{"customdata": [64, 0, 0., 1.]}],
                       [{"customdata": [-1, 0, 0., 1.]}],
                       [{"customdata": [True, 0, 0., 1.]}],
                       [{"customdata": [0, 6, 0., 1.]}],
                       [{"curveNumber": 1, "pointIndex": 64}],
                       [{"curveNumber": 99, "pointIndex": 0}]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                MODULE.selected_case(points, 64, self.figure)

    def test_wrong_target_shapes_or_inverted_intervals_are_rejected(self):
        for q in (self.q[:, :, :5], self.q[::-1], self.q * np.nan):
            with self.subTest(shape=q.shape), self.assertRaises(ValueError):
                MODULE.recovery_figure(self.truth, q, self.bounds, list("abcdef"))


if __name__ == "__main__":
    unittest.main()

"""The didactic input vector is exactly the adapter's declared market condition."""
import unittest
import numpy as np
from sbi4f_tutorial.shared import backbones, full_surface
from sbi4f_tutorial.rough_options import lesson


class RoughInputTests(unittest.TestCase):
    def test_supplied_curve_is_appended_after_all_scaled_quotes(self):
        task, _, case = backbones.load_example("rough-known100k-validation0")
        result = lesson.conditioning_example(task, case["observations"], case["context_values"])
        data = full_surface.conditioning_data(case["observations"][None], task,
                                              context_values=case["context_values"][None])
        np.testing.assert_array_equal(result["condition_vector"],
                                      np.r_[data["observations"][0], data["curve_context"][0]])
        self.assertEqual(result["condition_vector"].shape, (70,))
        np.testing.assert_array_equal(result["log_curve_levels"], np.log(case["context_values"]))

    def test_joint_targets_do_not_become_supplied_curve_features(self):
        task, _, case = backbones.load_example("rough-joint100k-validation0")
        result = lesson.conditioning_example(task, case["observations"])
        self.assertEqual(result["condition_vector"].shape, (63,))
        self.assertEqual(result["log_curve_levels"].shape, (0,))
        self.assertNotIn("truth", result)
        np.testing.assert_array_equal(result["condition_vector"], result["scaled_quotes"])


if __name__ == "__main__":
    unittest.main()

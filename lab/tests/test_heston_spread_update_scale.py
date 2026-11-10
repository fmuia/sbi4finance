"""Focused algebra checks for discarded native-update counterfactuals."""
import unittest

import numpy as np

from scripts.diagnose_heston_spread_update_scale import counterfactual_states, shift_summary


class UpdateScaleTests(unittest.TestCase):
    def test_weight_scaling_keeps_candidate_other_state_without_changing_saved_state(self):
        old = ([np.array([1., 3.])], [np.array([7.])], [np.array([50000])], [])
        new = ([np.array([2., 1.])], [np.array([8.])], [np.array([50001])], [])
        variants = counterfactual_states(old, new)
        np.testing.assert_array_equal(variants["scaled_weight_delta"][0][0], [1.1, 2.8])
        np.testing.assert_array_equal(old[0][0], [1., 3.])
        self.assertIs(variants["native"], new)
        for variant in ("scaled_weight_delta", "candidate_other_state_only"):
            for group in range(1, 4):
                self.assertIs(variants[variant][group], new[group])
        self.assertIs(variants["candidate_other_state_only"][0], old[0])

    def test_rms_and_max_are_over_independent_batches_per_parameter(self):
        report = shift_summary([[3., -2.], [-3., 0.]])
        np.testing.assert_allclose(report["rms"], [3., np.sqrt(2)])
        np.testing.assert_allclose(report["max_abs"], [3., 2.])
        np.testing.assert_allclose(report["signed_mean"], [0., -1.])
        with self.assertRaises(ValueError):
            shift_summary([[np.nan]])


if __name__ == "__main__":
    unittest.main()

"""Core observation contract, target isolation and architecture identity."""

import copy
import unittest

import numpy as np

from sbi4finance.calibration import full_surface, inference, tasks


class FullSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.task = tasks.heston_task()
        bounds = np.asarray(self.task["prior_bounds"])
        self.split = {"observations": np.zeros((3, 63, 1)), "reference_prices": np.zeros((3, 63)),
                      "parameters": np.tile(bounds.mean(axis=1), (3, 1))}

    def test_complete_only_and_no_extra_channels(self):
        for invalid in (np.zeros((3, 62)), np.zeros((3, 63, 2)), np.full((3, 63), np.nan),
                        np.full((3, 63), np.inf), np.zeros((0, 63)), np.zeros(63)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                full_surface.observations(invalid, self.task)
        self.assertEqual(full_surface.observations(self.split["observations"], self.task).shape, (3, 63))

    def test_finite_but_masked_prices_and_targets_rejected(self):
        prices = np.ma.array(np.zeros((3, 63)), mask=False)
        prices.mask[0, 0] = True
        with self.assertRaises(ValueError):
            full_surface.observations(prices, self.task)
        parameters = np.ma.array(self.split["parameters"], mask=False)
        parameters.mask[0, 0] = True
        with self.assertRaises(ValueError):
            full_surface.training_data({**self.split, "parameters": parameters}, self.task)

    def test_changed_design_and_wrong_heston_targets_rejected(self):
        changed = copy.deepcopy(self.task)
        changed["context"]["contracts"][0]["strike"] *= 1.01
        with self.assertRaises(ValueError):
            full_surface.validate_task(changed)
        changed = copy.deepcopy(self.task)
        changed["context"]["contracts"][0], changed["context"]["contracts"][1] = (
            changed["context"]["contracts"][1], changed["context"]["contracts"][0])
        with self.assertRaises(ValueError):
            full_surface.validate_task(changed)

    def test_canonical_design_accepts_cross_platform_roundoff_only(self):
        rounded = copy.deepcopy(self.task)
        rounded["context"]["contracts"][0]["strike"] = float(np.nextafter(
            rounded["context"]["contracts"][0]["strike"], np.inf))
        self.assertIs(full_surface.validate_task(rounded), rounded)
        # Membership tolerance does not alter the task or its content identity.
        self.assertNotEqual(rounded, self.task)
        changed = copy.deepcopy(self.task)
        changed["parameter_names"] = list(reversed(changed["parameter_names"]))
        changed["prior_bounds"] = list(reversed(changed["prior_bounds"]))
        with self.assertRaises(ValueError):
            full_surface.validate_task(changed)

    def test_supplied_curve_task_not_core(self):
        with self.assertRaises(ValueError):
            full_surface.validate_task(tasks.rough_task())

    def test_target_or_auxiliary_changes_do_not_change_conditions(self):
        data = full_surface.training_data(self.split, self.task)
        altered = {**self.split, "context_values": np.full((3, 7), np.nan),
                   "context_parameters": np.full((3, 3), 100.0),
                   "parameters": self.split["parameters"] * 1.001}
        other = full_surface.training_data(altered, self.task)
        self.assertEqual(set(data), {"observations", *self.task["parameter_names"]})
        np.testing.assert_array_equal(data["observations"], other["observations"])

    def test_joint_curve_is_target_not_condition(self):
        if not hasattr(tasks, "joint_rough_task"):
            self.skipTest("joint generator not yet available")
        task = tasks.joint_rough_task()
        bounds = np.asarray(task["prior_bounds"])
        split = {**self.split, "parameters": np.tile(bounds.mean(axis=1), (3, 1)),
                 "context_values": np.full((3, 7), np.nan)}
        data = full_surface.training_data(split, task)
        self.assertEqual(len(task["parameter_names"]), 6)
        self.assertEqual(set(data), {"observations", *task["parameter_names"]})
        with self.assertRaisesRegex(ValueError, "generating curves are not conditions"):
            inference.conditioning_data(np.zeros((3, 63)), np.ones((3, 7)), task)

    def test_set_noise_changes_only_price_channel(self):
        for encoder in ("deepset", "set_transformer"):
            data = full_surface.training_data(self.split, self.task, encoder)
            self.assertEqual(data["observations"].shape, (3, 63, 5))
            augment = full_surface.fresh_noise(2, encoder)
            first = augment(data)
            second = augment(data)
            np.testing.assert_array_equal(first["observations"][..., 1:], data["observations"][..., 1:])
            np.testing.assert_array_equal(data["observations"][..., 0], 0)
            self.assertFalse(np.array_equal(first["observations"][..., 0], second["observations"][..., 0]))
            for name in self.task["parameter_names"]:
                np.testing.assert_array_equal(first[name], data[name])

    def test_family_settings_are_explicit_and_validated(self):
        configs = [full_surface.configuration(family=f) for f in full_surface.FAMILIES]
        self.assertEqual(len({c["family"] for c in configs}), 4)
        for kwargs in ({"family": "unknown"}, {"encoder": "masked"}, {"widths": [16]},
                       {"mean_steps": 0}, {"flow_steps": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                full_surface.configuration(**kwargs)


if __name__ == "__main__":
    unittest.main()

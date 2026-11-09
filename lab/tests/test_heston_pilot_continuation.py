"""Continuation retains absolute data steps and rejects optimizer/state resets."""
import copy
import unittest

import numpy as np

from scripts import continue_heston_pilot_campaign as continuation
from scripts import train_heston_pilot_campaign as producer
from scripts.train_full_surface_budget import BatchPlan


class ContinuationTests(unittest.TestCase):
    def test_absolute_batch_matches_uninterrupted_data_and_augmentation(self):
        rows = 23
        physical = np.tile([.03, 2., .04, .6, -.7], (rows, 1))
        physical[:, 0] += np.arange(rows) / 10000
        train = {"parameters": physical, "clean": np.arange(rows * 3).reshape(rows, 3)}
        plan = BatchPlan(rows, 8, 100, 9011)
        indices, quotes, c, theta = continuation.absolute_batch(train, plan, 50, np.ones(3), np.arange(1,4), 9011)
        expected_indices = plan.indices(50)
        expected = producer.augmented_batch(train["clean"][expected_indices], physical[expected_indices],
                                             np.ones(3), np.arange(1,4), 50, 9011)
        np.testing.assert_array_equal(indices, expected_indices)
        for actual, wanted in zip((quotes, c, theta), expected):
            np.testing.assert_array_equal(actual, wanted)
        reset = continuation.absolute_batch(train, plan, 0, np.ones(3), np.arange(1,4), 9011)
        self.assertFalse(np.array_equal(quotes, reset[1]))

    def test_iteration_guard_uses_absolute_optimizer_count(self):
        self.assertEqual(continuation.expected_iteration(50000, 0, 50001), 50001)
        self.assertEqual(continuation.expected_iteration(50000, 24999, 75000), 75000)
        for actual in (1, 50000, 50002):
            with self.assertRaises(RuntimeError):
                continuation.expected_iteration(50000, 0, actual)

    def test_initial_state_guard_detects_slot_or_standardization_reset(self):
        initial = {key: {"sha256": key} for key in ("trainable", "non_trainable", "standardization", "optimizer", "estimator_seed")}
        initial.update(iterations=50000, optimizer_config_sha256="fixed-schedule", learning_rate=5e-5)
        continuation.assert_same_state(initial, copy.deepcopy(initial))
        for key in ("optimizer", "standardization", "estimator_seed"):
            changed = copy.deepcopy(initial)
            changed[key]["sha256"] = "reset"
            with self.assertRaisesRegex(ValueError, key):
                continuation.assert_same_state(initial, changed)
        changed = copy.deepcopy(initial)
        changed["optimizer_config_sha256"] = "new-longer-schedule"
        with self.assertRaisesRegex(ValueError, "optimizer"):
            continuation.assert_same_state(initial, changed)


if __name__ == "__main__":
    unittest.main()

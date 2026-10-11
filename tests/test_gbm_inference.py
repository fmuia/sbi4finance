"""Observation boundaries and an offline saved-bank/checkpoint round trip."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.gbm_bank import generate_examples, load_bank, write_bank
from sbi4finance.gbm_inference import (
    build_workflow,
    conditioning_data,
    training_data,
)


class GBMInferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "task": {
                "task_id": "gbm_drift_known_vol_bank_v1",
                "measure": "P",
                "volatility": 0.2,
                "prior_mean": 0.0,
                "prior_std": 0.1,
                "n_observations": 8,
                "dt_years": 1 / 252,
            },
            "root_seed": 719,
            "splits": {"train": 512, "validation": 64, "test": 8},
        }
        self.task = self.config["task"]

    def test_only_observations_condition_and_truth_is_a_separate_target(self):
        original = generate_examples(self.config, "train", range(4))
        selected = training_data(original, self.task)
        self.assertEqual(set(selected), {"mu", "sum_log_returns"})
        self.assertEqual(selected["sum_log_returns"].shape, (4, 1))

        changed = deepcopy(original)
        changed["parameters"] += 100.0
        changed["example_id"] = np.array(["other"] * 4)
        changed["target_state"] = np.full((4, 1), np.nan)
        changed["future_outcomes"] = np.full((4, 20, 1), 1e6)
        changed["diagnostics"] = np.full((4, 3), -1e6)
        selected_changed = training_data(changed, self.task)
        assert_array_equal(
            selected_changed["sum_log_returns"], selected["sum_log_returns"]
        )
        assert_array_equal(selected_changed["mu"], changed["parameters"])
        self.assertFalse(np.array_equal(selected_changed["mu"], selected["mu"]))

        # A conditioning record has no parameter/state truth or future labels.
        unlabelled = {key: original[key] for key in ("observations", "dt_years")}
        conditions = conditioning_data(
            unlabelled["observations"], unlabelled["dt_years"], self.task
        )
        self.assertEqual(set(conditions), {"sum_log_returns"})
        assert_array_equal(conditions["sum_log_returns"], selected["sum_log_returns"])

    def test_checkpoint_time_design_must_match_exactly(self):
        example = generate_examples(self.config, "test", [0])
        grid = example["dt_years"]
        same_duration = grid.copy()
        same_duration[:2] += [0.001, -0.001]
        invalid_grids = [grid[:-1], grid[:, None], grid * 2, same_duration]
        for invalid in invalid_grids:
            with self.subTest(grid=invalid):
                with self.assertRaisesRegex(ValueError, "time grid"):
                    conditioning_data(example["observations"], invalid, self.task)
        with self.assertRaisesRegex(ValueError, "complete finite"):
            conditioning_data(example["observations"][:, :-1], grid, self.task)

    def test_saved_bank_training_normalization_and_checkpoint_work_offline(self):
        # Block outbound Python socket connections throughout framework import,
        # training, checkpoint loading and inference; no market files are used.
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("socket.socket.connect", side_effect=AssertionError("network disabled")),
            patch("socket.create_connection", side_effect=AssertionError("network disabled")),
            patch.dict(os.environ, {"KERAS_BACKEND": "jax"}),
        ):
            import keras

            keras.utils.set_random_seed(815)
            directory = Path(temporary)
            write_bank(directory / "bank", self.config)
            manifest, splits = load_bank(directory / "bank")
            task = manifest["configuration"]["task"]
            train = training_data(splits["train"], task)
            validation = training_data(splits["validation"], task)
            workflow = build_workflow(directory / "checkpoint")
            history = workflow.fit_offline(
                data=train, validation_data=validation,
                epochs=2, batch_size=64, verbose=0,
            )
            self.assertTrue(np.isfinite(history.history["loss"]).all())
            self.assertTrue(np.isfinite(history.history["val_loss"]).all())
            checkpoint = directory / "checkpoint" / "model.keras"
            self.assertTrue(checkpoint.is_file())
            workflow.load_approximator()  # Compare the saved best epoch.

            layers = workflow.approximator.standardizer.standardize_layers
            self.assertEqual(set(layers), {"inference_variables", "inference_conditions"})
            for name, key in (
                ("inference_variables", "mu"),
                ("inference_conditions", "sum_log_returns"),
            ):
                layer = layers[name]
                expected = train[key].astype("float32")
                mean = layer.moving_mean[0].numpy()
                std = np.asarray(layer.moving_std(0))
                assert_allclose(mean, expected.mean(axis=0), rtol=2e-5, atol=1e-7)
                assert_allclose(std, expected.std(axis=0), rtol=2e-5, atol=1e-7)
                # Both saved epochs see whole training splits. Validation has
                # a different mean and must not add rows to normalization.
                count = float(layer.count[0].numpy())
                self.assertIn(count, (512.0, 1024.0))
                self.assertFalse(np.allclose(
                    mean, validation[key].mean(axis=0), rtol=2e-5, atol=1e-7
                ))

            before_sampling = {
                name: [weight.numpy().copy() for weight in layer.weights]
                for name, layer in layers.items()
            }
            conditions = conditioning_data(
                splits["test"]["observations"], splits["test"]["dt_years"], task
            )
            samples = workflow.sample(num_samples=32, conditions=conditions, seed=317)
            self.assertEqual(set(samples), {"mu"})
            self.assertEqual(samples["mu"].shape, (8, 32, 1))
            self.assertTrue(np.isfinite(samples["mu"]).all())
            for name, layer in layers.items():
                for before, after in zip(before_sampling[name], layer.weights):
                    assert_array_equal(before, after.numpy())

            reloaded = build_workflow()
            reloaded.load_approximator(checkpoint)
            repeated = reloaded.sample(num_samples=32, conditions=conditions, seed=317)
            assert_array_equal(repeated["mu"], samples["mu"])


if __name__ == "__main__":
    unittest.main()

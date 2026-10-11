"""Stable synthetic Heston examples and joint inference without latent leakage."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.heston_bank import generate_examples, load_bank, write_bank
from sbi4finance.heston_inference import (
    ADAPTER_ID, TIME_ADAPTER_ID, build_workflow, conditioning_data, training_data,
)


class HestonBankInferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "task": {
                "task_id": "heston_theta_state_v1", "measure": "P",
                "parameters": {"drift": 0.05, "kappa": 2.0, "vol_of_vol": 0.2, "rho": -0.6},
                "theta_bounds": [0.02, 0.08], "prior": "log_uniform",
                "initial_variance": 0.04, "n_observations": 12,
                "dt_years": 1 / 252, "substeps": 1,
            },
            "root_seed": 43021,
            "splits": {"train": 128, "validation": 32, "test": 8},
        }
        self.task = self.config["task"]

    def test_example_values_survive_reordering_chunking_and_count_changes(self):
        original = generate_examples(self.config, "train", range(9))
        requested = [8, 0, 3, 1]
        reordered = generate_examples(self.config, "train", requested)
        with patch("sbi4finance.heston_bank.GENERATION_BATCH_SIZE", 3):
            rechunked = generate_examples(self.config, "train", range(9))
        enlarged = deepcopy(self.config)
        enlarged["splits"]["train"] = 1024
        resized = generate_examples(enlarged, "train", range(9))
        for key, values in original.items():
            assert_array_equal(rechunked[key], values)
            assert_array_equal(resized[key], values)
            assert_array_equal(reordered[key], values if key == "dt_years" else values[requested])
        assert_array_equal(original["example_id"], [f"train:{i}" for i in range(9)])

    def test_split_isolation_and_verified_bank_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bank"
            written = write_bank(path, self.config)
            manifest, splits = load_bank(path)
            self.assertEqual(manifest, written)
            self.assertEqual(manifest["data_origin"], "synthetic")
            self.assertEqual(manifest["evidence_scope"], "S")
            self.assertFalse(manifest["future_outcomes_in_conditioning"])
            all_ids = [set(split["example_id"]) for split in splits.values()]
            self.assertEqual(len(set.union(*all_ids)), sum(map(len, all_ids)))
            for name, count in self.config["splits"].items():
                expected = generate_examples(self.config, name, range(count))
                for key, values in expected.items():
                    assert_array_equal(splits[name][key], values)
                self.assertEqual(splits[name]["terminal_state"].shape, (count, 1))
            self.assertFalse(np.array_equal(
                splits["train"]["observations"][:8], splits["test"]["observations"]
            ))
            self.assertFalse(np.array_equal(
                splits["train"]["parameters"][:8], splits["test"]["parameters"]
            ))
            corrupted = path / "train.npz"
            contents = bytearray(corrupted.read_bytes())
            contents[-1] ^= 1
            corrupted.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_bank(path)

    def test_ordered_conditioning_excludes_both_targets_and_future_metadata(self):
        split = generate_examples(self.config, "train", range(4))
        selected = training_data(split, self.task)
        self.assertEqual(set(selected), {"observations", "history_features", "theta", "terminal_state"})
        assert_array_equal(selected["observations"], split["observations"])
        # Independent scalar suffix calculations retain both the level and
        # timing of observations; these features contain no latent state.
        expected_features = []
        for history in split["observations"][:, :, 0]:
            features = []
            for requested_width in (8, 32, 128, len(history)):
                window = history[-min(requested_width, len(history)):].tolist()
                features.extend([sum(window) / len(window),
                                 sum(value * value for value in window)
                                 / (len(window) * self.task["dt_years"])])
            expected_features.append(features)
        self.assertEqual(selected["history_features"].shape, (4, 8))
        assert_allclose(selected["history_features"], expected_features, rtol=2e-15, atol=1e-16)
        changed = deepcopy(split)
        changed["parameters"][:] = 0.05
        changed["terminal_state"][:] = -0.02  # Signed states must remain valid targets.
        changed["example_id"] = np.array(["different"] * 4)
        changed["future_returns"] = np.full((4, 10, 1), 1e6)
        changed["diagnostics"] = np.full((4, 3), np.nan)
        after = training_data(changed, self.task)
        assert_array_equal(after["observations"], selected["observations"])
        assert_array_equal(after["history_features"], selected["history_features"])
        assert_array_equal(after["theta"], changed["parameters"])
        assert_array_equal(after["terminal_state"], changed["terminal_state"])

        unlabelled = {key: split[key] for key in ("observations", "dt_years")}
        conditions = conditioning_data(unlabelled["observations"], unlabelled["dt_years"], self.task)
        self.assertEqual(set(conditions), {"observations", "history_features"})
        assert_array_equal(conditions["observations"], selected["observations"])
        assert_array_equal(conditions["history_features"], selected["history_features"])
        reversed_history = split["observations"][:, ::-1]
        reversed_conditions = conditioning_data(reversed_history, split["dt_years"], self.task)
        assert_array_equal(reversed_conditions["observations"], reversed_history)
        self.assertFalse(np.array_equal(reversed_conditions["observations"], conditions["observations"]))
        self.assertFalse(np.array_equal(reversed_conditions["history_features"][:, :2],
                                       conditions["history_features"][:, :2]))
        scaled = conditioning_data(2 * split["observations"], split["dt_years"], self.task)
        assert_array_equal(scaled["observations"], 2 * split["observations"])
        assert_allclose(scaled["history_features"][:, ::2], 2 * conditions["history_features"][:, ::2])
        assert_allclose(scaled["history_features"][:, 1::2], 4 * conditions["history_features"][:, 1::2])

    def test_unsupported_tasks_time_grids_and_target_support_fail(self):
        split = generate_examples(self.config, "test", [0])
        for key, value in (("task_id", "other"), ("measure", "Q"), ("prior", "normal")):
            task = {**self.task, key: value}
            with self.subTest(task_field=key):
                with self.assertRaises(ValueError):
                    conditioning_data(split["observations"], split["dt_years"], task)
        irregular = split["dt_years"].copy()
        irregular[:2] += [0.001, -0.001]  # Same horizon still violates the fixed design.
        for grid in (irregular, split["dt_years"][:-1], split["dt_years"] * 2):
            with self.subTest(grid=grid):
                with self.assertRaisesRegex(ValueError, "time grid"):
                    conditioning_data(split["observations"], grid, self.task)
        for theta in self.task["theta_bounds"]:
            invalid = {**split, "parameters": np.array([[theta]])}
            with self.assertRaisesRegex(ValueError, "theta targets"):
                training_data(invalid, self.task)
        invalid = {**split, "terminal_state": np.array([[np.nan]])}
        with self.assertRaisesRegex(ValueError, "terminal_state targets"):
            training_data(invalid, self.task)

    def test_explicit_time_conditioning_uses_only_the_known_observation_grid(self):
        split = generate_examples(self.config, "train", range(4))
        original_observations = split["observations"].copy()
        selected = training_data(split, self.task, adapter_id=TIME_ADAPTER_ID)
        legacy = training_data(split, self.task)
        self.assertEqual(selected["observations"].shape, (4, 12, 2))
        assert_array_equal(selected["observations"][..., :1], original_observations)
        # Independently accumulate the supplied intervals, rather than deriving
        # time from return values or any simulator-only metadata.
        elapsed = 0.0
        end_times = []
        for interval in split["dt_years"]:
            elapsed += float(interval)
            end_times.append(elapsed)
        expected_time = np.broadcast_to(end_times, (4, 12))
        assert_array_equal(selected["observations"][..., 1], expected_time)
        assert_array_equal(selected["history_features"], legacy["history_features"])
        self.assertEqual(split["observations"].shape, (4, 12, 1))
        assert_array_equal(split["observations"], original_observations)

        changed = deepcopy(split)
        changed["parameters"][:] = 0.06
        changed["terminal_state"][:] = -0.02
        changed["future_returns"] = np.full((4, 10, 1), 1e6)
        changed["diagnostics"] = np.full((4, 3), np.nan)
        changed["example_id"] = np.array(["different"] * 4)
        after = training_data(changed, self.task, adapter_id=TIME_ADAPTER_ID)
        for key in ("observations", "history_features"):
            assert_array_equal(after[key], selected[key])
        assert_array_equal(after["theta"], changed["parameters"])
        assert_array_equal(after["terminal_state"], changed["terminal_state"])

        for history in (original_observations, 2 * original_observations,
                        original_observations[:, ::-1]):
            with self.subTest(history=history):
                conditions = conditioning_data(history, split["dt_years"], self.task,
                                               adapter_id=TIME_ADAPTER_ID)
                self.assertEqual(set(conditions), {"observations", "history_features"})
                assert_array_equal(conditions["observations"][..., :1], history)
                assert_array_equal(conditions["observations"][..., 1], expected_time)

        # Changing the declared design and its matching context changes only
        # the time channel and the dt-dependent second-moment features.
        slower_task = {**self.task, "dt_years": 2 * self.task["dt_years"]}
        slower = conditioning_data(original_observations, 2 * split["dt_years"],
                                   slower_task, adapter_id=TIME_ADAPTER_ID)
        assert_array_equal(slower["observations"][..., :1], original_observations)
        assert_allclose(slower["observations"][..., 1], 2 * expected_time)
        with self.assertRaisesRegex(ValueError, "time grid"):
            conditioning_data(original_observations, 2 * split["dt_years"], self.task,
                              adapter_id=TIME_ADAPTER_ID)

    def test_unknown_adapter_is_rejected_at_every_public_entry_point(self):
        split = generate_examples(self.config, "test", [0])
        with self.assertRaisesRegex(ValueError, "unsupported Heston adapter_id"):
            conditioning_data(split["observations"], split["dt_years"], self.task,
                              adapter_id="unknown")
        with self.assertRaisesRegex(ValueError, "unsupported Heston adapter_id"):
            training_data(split, self.task, adapter_id="unknown")
        with self.assertRaisesRegex(ValueError, "unsupported Heston adapter_id"):
            build_workflow(self.task, adapter_id="unknown")

    def test_offline_joint_training_normalization_sampling_and_reload(self):
        self._check_offline_roundtrip("flow_matching")

    def test_spline_joint_training_normalization_sampling_and_reload(self):
        self._check_offline_roundtrip("spline")

    def test_explicit_time_joint_training_normalization_sampling_and_reload(self):
        self._check_offline_roundtrip("spline", adapter_id=TIME_ADAPTER_ID)

    def _check_offline_roundtrip(self, density_estimator, *, adapter_id=ADAPTER_ID):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("socket.socket.connect", side_effect=AssertionError("network disabled")),
            patch("socket.create_connection", side_effect=AssertionError("network disabled")),
            patch.dict(os.environ, {"KERAS_BACKEND": "jax"}),
        ):
            import keras

            keras.utils.set_random_seed(581)
            directory = Path(temporary)
            write_bank(directory / "bank", self.config)
            _, splits = load_bank(directory / "bank")
            train, validation = [training_data(splits[name], self.task, adapter_id=adapter_id)
                                 for name in ("train", "validation")]
            workflow = build_workflow(self.task, directory / "checkpoint",
                                      density_estimator=density_estimator,
                                      adapter_id=adapter_id)
            history = workflow.fit_offline(data=train, validation_data=validation,
                                           epochs=2, batch_size=32, verbose=0)
            self.assertTrue(all(np.isfinite(values).all() for values in history.history.values()))
            checkpoint = directory / "checkpoint" / "model.keras"
            self.assertTrue(checkpoint.is_file())
            workflow.load_approximator()
            expected_time_axis = 1 if adapter_id == TIME_ADAPTER_ID else None
            self.assertEqual(workflow.approximator.summary_network.time_axis, expected_time_axis)

            # Compare saved statistics to independent transformed training-array
            # moments, including global raw-return scaling across all windows.
            low, high = self.task["theta_bounds"]
            transformed_theta = np.log((train["theta"] - low) / (high - train["theta"]))
            expected_targets = np.concatenate([transformed_theta, train["terminal_state"]], axis=-1)
            channel_count = 2 if adapter_id == TIME_ADAPTER_ID else 1
            expected = {"inference_variables": expected_targets.astype("float32"),
                        "summary_variables": train["observations"].reshape(-1, channel_count).astype("float32"),
                        "inference_conditions": train["history_features"].astype("float32")}
            layers = workflow.approximator.standardizer.standardize_layers
            self.assertEqual(set(layers), set(expected))
            for key, values in expected.items():
                layer = layers[key]
                assert_allclose(layer.moving_mean[0].numpy(), values.mean(axis=0), rtol=2e-5, atol=1e-7)
                assert_allclose(np.asarray(layer.moving_std(0)), values.std(axis=0), rtol=2e-5, atol=1e-7)
                self.assertIn(float(layer.count[0].numpy()), (float(len(values)), float(2 * len(values))))
            # Held-out values are different; equality above cannot accidentally
            # pass through a normalization fit on the validation split.
            self.assertFalse(np.allclose(
                expected["summary_variables"].mean(axis=0),
                validation["observations"].mean(axis=(0, 1)), rtol=2e-5, atol=1e-7,
            ))
            saved_statistics = {name: [weight.numpy().copy() for weight in layer.weights]
                                for name, layer in layers.items()}
            conditions = conditioning_data(splits["test"]["observations"],
                                           splits["test"]["dt_years"], self.task,
                                           adapter_id=adapter_id)
            samples = workflow.sample(num_samples=32, conditions=conditions, seed=983)
            self.assertEqual(set(samples), {"theta", "terminal_state"})
            for values in samples.values():
                self.assertEqual(values.shape, (8, 32, 1))
                self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.all((samples["theta"] > low) & (samples["theta"] < high)))
            for name, layer in layers.items():
                for before, after in zip(saved_statistics[name], layer.weights):
                    assert_array_equal(before, after.numpy())
            restored = build_workflow(self.task)
            restored.load_approximator(checkpoint)
            # The saved network, not the fresh workflow's default adapter,
            # determines whether explicit time is split from the input.
            self.assertEqual(restored.approximator.summary_network.time_axis, expected_time_axis)
            repeated = restored.sample(num_samples=32, conditions=conditions, seed=983)
            for key in samples:
                assert_array_equal(repeated[key], samples[key])


if __name__ == "__main__":
    unittest.main()

"""Observation boundaries and a small offline HKDE checkpoint round trip."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sbi4finance.hkde_inference import (
    PARAMETER_NAMES, build_workflow, conditioning_data, training_data,
)


class HKDEInferenceTests(unittest.TestCase):
    def setUp(self):
        config = Path(__file__).parents[1] / "configs/tasks/hkde_option_bank.json"
        self.task = json.loads(config.read_text())["task"]
        self.rng = np.random.default_rng(813)

    def _split(self, count):
        # Array-boundary tests need no Fourier evaluation or inference study.
        bounds = np.asarray(self.task["prior_bounds"])
        return {"observations": self.rng.normal(5, 2, (count, 9, 1)),
                "parameters": self.rng.uniform(bounds[:, 0], bounds[:, 1], (count, 2))}

    def test_truth_diagnostics_and_future_outcomes_do_not_condition(self):
        split = self._split(4)
        selected = training_data(split, self.task)
        self.assertEqual(set(selected), {"observations", *PARAMETER_NAMES})
        assert_array_equal(selected["observations"], split["observations"][..., 0])
        for index, name in enumerate(PARAMETER_NAMES):
            assert_array_equal(selected[name], split["parameters"][:, index:index + 1])

        changed = deepcopy(split)
        changed["parameters"] = changed["parameters"][::-1].copy()
        for key in ("reference_prices", "pricing_standard_errors", "pricing_covariance",
                    "refinement_difference", "future_outcomes", "terminal_state"):
            changed[key] = np.full((4, 9), np.nan)
        changed["example_id"] = ["other"] * 4
        after = training_data(changed, self.task)
        unlabelled = conditioning_data(split["observations"], self.task)
        self.assertEqual(set(unlabelled), {"observations"})
        assert_array_equal(after["observations"], selected["observations"])
        assert_array_equal(unlabelled["observations"], selected["observations"])
        for index, name in enumerate(PARAMETER_NAMES):
            assert_array_equal(after[name], changed["parameters"][:, index:index + 1])
            self.assertFalse(np.array_equal(after[name], selected[name]))

    def test_prices_keep_order_scale_and_negative_gaussian_quotes(self):
        original = np.arange(-9, 9, dtype=float).reshape(2, 9, 1)
        for prices in (original, 2 * original, original[:, ::-1], original[::-1]):
            with self.subTest(prices=prices):
                assert_array_equal(conditioning_data(prices, self.task)["observations"],
                                   prices[..., 0])
        self.assertFalse(np.array_equal(conditioning_data(original, self.task)["observations"],
                                        conditioning_data(original[:, ::-1], self.task)["observations"]))

    def test_invalid_observations_and_targets_are_rejected(self):
        split = self._split(2)
        invalid = [split["observations"][..., 0], split["observations"][:, :-1],
                   np.empty((0, 9, 1)), np.full((2, 9, 1), np.nan)]
        for observations in invalid:
            with self.subTest(shape=observations.shape):
                with self.assertRaisesRegex(ValueError, "complete finite"):
                    conditioning_data(observations, self.task)
        for targets in (np.ones((2, 1)), np.full((2, 2), np.inf),
                        np.asarray(self.task["prior_bounds"]).T):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    training_data({**split, "parameters": targets}, self.task)

    def test_task_model_order_context_and_prior_are_validated(self):
        variants = []
        for key, value in (("measure", "P"), ("model", "rough"),
                           ("parameter_names", ["jump_intensity", "theta"]),
                           ("prior_bounds", [[0.065, 0.025], [0.25, 1.0]])):
            variants.append({**self.task, key: value})
        context = deepcopy(self.task)
        context["context"]["quote_noise_std"] = 0.0
        variants.append(context)
        contracts = deepcopy(self.task)
        contracts["context"]["contracts"].pop()
        variants.append(contracts)
        missing_context = deepcopy(self.task)
        missing_context["model_context"] = {}
        variants.append(missing_context)
        for task in variants:
            with self.subTest(task=task):
                with self.assertRaises(ValueError):
                    conditioning_data(np.ones((1, 9, 1)), task)
                with self.assertRaises(ValueError):
                    build_workflow(task)

    def test_offline_train_only_scaling_joint_support_and_exact_reload(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("socket.socket.connect", side_effect=AssertionError("network disabled")),
            patch("socket.create_connection", side_effect=AssertionError("network disabled")),
            patch.dict(os.environ, {"KERAS_BACKEND": "jax"}),
        ):
            import keras

            keras.utils.set_random_seed(815)
            train = training_data(self._split(64), self.task)
            validation_split = self._split(16)
            validation_split["observations"] += 30  # Detect accidental validation normalization.
            validation = training_data(validation_split, self.task)
            directory = Path(temporary) / "checkpoint"
            workflow = build_workflow(self.task, directory)
            # These targets round to bounds in float32. Transforming first
            # must preserve finite logits rather than manufacture infinities.
            edge_split = self._split(2)
            bounds = np.asarray(self.task["prior_bounds"])
            edge_split["parameters"] = np.stack([
                np.nextafter(bounds[:, 0], bounds[:, 1]),
                np.nextafter(bounds[:, 1], bounds[:, 0]),
            ])
            adapted_edges = workflow.adapter(training_data(edge_split, self.task))
            self.assertTrue(np.isfinite(adapted_edges["inference_variables"]).all())
            history = workflow.fit_offline(data=train, validation_data=validation,
                                           epochs=1, batch_size=32, verbose=0)
            self.assertTrue(all(np.isfinite(v).all() for v in history.history.values()))
            checkpoint = directory / "model.keras"
            self.assertTrue(checkpoint.is_file())
            workflow.load_approximator()

            transformed = []
            for name, (lower, upper) in zip(PARAMETER_NAMES, self.task["prior_bounds"], strict=True):
                transformed.append(np.log((train[name] - lower) / (upper - train[name])))
            expected = {"inference_variables": np.concatenate(transformed, axis=1).astype("float32"),
                        "inference_conditions": train["observations"].astype("float32")}
            layers = workflow.approximator.standardizer.standardize_layers
            self.assertEqual(set(layers), set(expected))
            for key, values in expected.items():
                layer = layers[key]
                assert_allclose(layer.moving_mean[0].numpy(), values.mean(axis=0), rtol=2e-5, atol=1e-7)
                assert_allclose(np.asarray(layer.moving_std(0)), values.std(axis=0), rtol=2e-5, atol=1e-7)
                self.assertEqual(float(layer.count[0].numpy()), 64.0)

            before = {name: [weight.numpy().copy() for weight in layer.weights]
                      for name, layer in layers.items()}
            conditions = conditioning_data(self._split(2)["observations"], self.task)
            draws = workflow.sample(num_samples=16, conditions=conditions, seed=317)
            self.assertEqual(set(draws), set(PARAMETER_NAMES))
            for name, (lower, upper) in zip(PARAMETER_NAMES, self.task["prior_bounds"], strict=True):
                self.assertEqual(draws[name].shape, (2, 16, 1))
                self.assertTrue(np.isfinite(draws[name]).all())
                self.assertTrue(np.all((draws[name] > lower) & (draws[name] < upper)))
            for name, layer in layers.items():
                for old, new in zip(before[name], layer.weights, strict=True):
                    assert_array_equal(old, new.numpy())

            restored = build_workflow(self.task)
            restored.load_approximator(checkpoint)
            repeated = restored.sample(num_samples=16, conditions=conditions, seed=317)
            for name in PARAMETER_NAMES:
                assert_array_equal(repeated[name], draws[name])


if __name__ == "__main__":
    unittest.main()

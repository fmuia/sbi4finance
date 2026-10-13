"""Both inference families execute the same fit/selection/reload lifecycle."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sbi4finance import hkde_inference, training_workflow

ROOT = Path(__file__).resolve().parents[1]


class TestTrainingWorkflow(unittest.TestCase):
    def test_online_selection_and_exposure_accounting(self):
        calls = []
        class Workflow:
            checkpoint_filepath = "unused-fake-directory"
            save_weights_only = False
            save_best_only = False
            def fit_online(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(history={"val_loss": [3.0, 2.0]})
            def load_approximator(self):
                calls.append("selected-checkpoint-loaded")
        workflow = Workflow()
        _, record = training_workflow.fit_and_select(
            workflow, training_data=None, validation_data={"x": np.ones((2, 3))},
            epochs=2, batch_size=4, seed=17, num_batches_per_epoch=3)
        self.assertTrue(workflow.save_best_only)
        self.assertEqual(calls[-1], "selected-checkpoint-loaded")
        self.assertEqual(record["selected_epoch"], 2)
        self.assertEqual(record["training_exposures"], 24)
        self.assertIsNone(record["unique_training_examples"])
        self.assertFalse(record["reserved_test_evaluated"])
        shared = {"x": np.ones((2, 3))}
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            training_workflow.fit_and_select(workflow, training_data=shared, validation_data=shared,
                                             epochs=1, batch_size=2, seed=17)

    def test_hkde_genuine_fit_through_shared_lifecycle(self):
        import keras
        task = json.loads((ROOT / "configs/tasks/hkde_inference_bank.json").read_text())["task"]
        rng = np.random.default_rng(72)
        def split(n):
            bounds = np.asarray(task["prior_bounds"])
            return {"parameters": rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, 2)),
                    "observations": rng.uniform(2, 20, size=(n, 9, 1))}
        training = hkde_inference.training_data(split(8), task)
        validation = hkde_inference.training_data(split(4), task)
        validation["observations"] += 100
        with tempfile.TemporaryDirectory() as temporary:
            path = training_workflow.reserve_output_dir(Path(temporary) / "checkpoint")
            keras.utils.set_random_seed(872)
            workflow = hkde_inference.build_workflow(task, path)
            epoch_weights = []
            class ForceSelectionCase(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs["val_loss"] = float(epoch + 1)  # first epoch wins, final epoch loses
                    epoch_weights.append([value.numpy().copy() for value in self.model.weights])
            fit = workflow.fit_offline
            def instrumented_fit(**kwargs):
                return fit(**kwargs, callbacks=[ForceSelectionCase()])
            with patch.object(workflow, "fit_offline", side_effect=instrumented_fit):
                _, record = training_workflow.fit_and_select(
                    workflow, training_data=training, validation_data=validation,
                    epochs=2, batch_size=4, seed=872)
            self.assertEqual(record["selected_epoch"], 1)
            self.assertTrue(any(not np.array_equal(first, final)
                                for first, final in zip(*epoch_weights, strict=True)))
            for saved, selected in zip(epoch_weights[0], workflow.approximator.weights, strict=True):
                np.testing.assert_array_equal(saved, selected.numpy())
            layer = workflow.approximator.standardizer.standardize_layers["inference_conditions"]
            np.testing.assert_allclose(layer.moving_mean[0].numpy(), training["observations"].mean(axis=0), rtol=2e-5)
            self.assertEqual(float(layer.count[0].numpy()), 8)
            def restore():
                restored = hkde_inference.build_workflow(task)
                restored.load_approximator(path / "model.keras")
                return restored
            conditions = {"observations": validation["observations"][:1]}
            self.assertTrue(training_workflow.verify_reload(
                workflow, restore=restore,
                sample=lambda candidate: candidate.sample(num_samples=8, conditions=conditions, seed=129)))
            self.assertEqual(record["unique_training_examples"], 8)
            self.assertEqual(record["training_exposures"], 16)
            self.assertFalse(record["reserved_test_evaluated"])


if __name__ == "__main__":
    unittest.main()

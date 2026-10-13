"""Tiny genuine SVJ saved-bank fit/reload, with split and normalization checks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from sbi4finance import training_workflow
from sbi4finance.svj import adapters, artifacts, bank


class TestSVJTraining(unittest.TestCase):
    def test_offline_train_selected_reload_and_train_only_normalization(self):
        from sbi4finance.svj import amortized
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = {"task": bank.default_task(), "root_seed": 827,
                   "splits": {"train": 4, "validation": 2, "test": 1}}
            bank.write_bank(root / "bank", cfg)
            _, splits = bank.load_bank(root / "bank")
            trained, record = amortized.train(root / "checkpoint", epochs=1, batches_per_epoch=1,
                                              batch_size=2, seed=837, bank=root / "bank")
            lifecycle = record["lifecycle"]
            self.assertEqual(lifecycle["unique_training_examples"], 4)
            self.assertEqual(lifecycle["training_exposures"], 4)
            self.assertEqual(lifecycle["validation_examples"], 2)
            self.assertEqual(lifecycle["selected_epoch"], 1)
            self.assertFalse(lifecycle["reserved_test_evaluated"])
            self.assertTrue(lifecycle["same_seed_reload_exact"])
            self.assertEqual(record["simulated_series"], 0)
            layers = trained.standardizer.standardize_layers
            self.assertEqual(set(layers), {"inference_variables"})
            layer = layers["inference_variables"]
            expected = adapters.bank_batch(splits["train"])["targets"].astype("float32")
            np.testing.assert_allclose(layer.moving_mean[0].numpy(), expected.mean(axis=0), rtol=2e-5, atol=1e-6)
            self.assertEqual(float(layer.count[0].numpy()), 4)
            posterior = amortized.AmortizedPosterior.load(root / "checkpoint")
            layer = posterior.approximator.standardizer.standardize_layers["inference_variables"]
            weights = [weight.numpy().copy() for weight in layer.weights]
            posterior.sample(adapters.bank_batch(splits["validation"])["features"][:1], num_samples=8, seed=31)
            for old, new in zip(weights, layer.weights, strict=True):
                np.testing.assert_array_equal(old, new.numpy())
            self.assertEqual(float(layer.count[0].numpy()), 4)
            artifacts.validate_checkpoint(root / "checkpoint")
            digest = artifacts.sha256(root / "checkpoint/model.keras")
            with self.assertRaises(FileExistsError):
                amortized.train(root / "checkpoint", epochs=1, batches_per_epoch=1, batch_size=2)
            self.assertEqual(artifacts.sha256(root / "checkpoint/model.keras"), digest)

    def test_duplicate_split_identities_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            training_workflow.validate_training_splits({"train": {"example_id": ["a"]},
                                                        "validation": {"example_id": ["a"]}})


if __name__ == "__main__":
    unittest.main()

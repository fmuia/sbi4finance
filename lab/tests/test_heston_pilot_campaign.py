"""Fresh nuisance augmentation preserves the declared engineering target."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from scripts import train_heston_pilot_campaign as trainer


class CampaignDataTests(unittest.TestCase):
    def test_tau_is_resampled_while_repeated_dataset_probes_are_exact(self):
        clean = np.zeros((10000, 2))
        physical = np.tile([.03, 2., .04, .6, -.7], (len(clean), 1))
        args = (clean, physical, np.array([.2, .4]), np.array([20., 40.]), 7, 9011)
        first = trainer.augmented_batch(*args, multiplier=1.)
        repeated = trainer.augmented_batch(*args, multiplier=1.)
        for actual, expected in zip(first, repeated):
            np.testing.assert_array_equal(actual, expected)
        other = trainer.augmented_batch(*args[:4], 8, 9011, multiplier=1.)
        self.assertFalse(np.array_equal(first[2][:, -1], other[2][:, -1]))
        np.testing.assert_array_equal(first[2][:, :5], physical)
        self.assertTrue(np.all((first[2][:, -1] > 0) & (first[2][:, -1] < .02)))
        self.assertAlmostEqual(first[2][:, -1].mean(), .01, delta=.0002)
        # Integrate tau^2 under Uniform(0,.02) independently of the simulator.
        expected_variance = np.array([.2, .4])**2 + np.array([20., 40.])**2 * .02**2 / 3
        np.testing.assert_allclose(first[0].var(axis=0), expected_variance, rtol=.04)

    def test_small_training_arm_is_exact_prefix_and_validation_is_independent(self):
        bounds = np.array([[.01,.16],[.1,10],[.01,.16],[.1,3],[-.95,-.1]])
        large = trainer.physical_rows(bounds, 100, 9011, 0)
        small = trainer.physical_rows(bounds, 20, 9011, 0)
        validation = trainer.physical_rows(bounds, 20, 9011, 1)
        reserved = trainer.physical_rows(bounds, 20, 9011, 2)
        np.testing.assert_array_equal(large[:20], small)
        self.assertFalse(np.array_equal(small, validation))
        self.assertFalse(np.array_equal(small, reserved))
        self.assertFalse(np.array_equal(validation, reserved))
        payload = {"train_parameters": large, "train_clean": np.arange(300).reshape(100,3),
                   "train_row_id": np.arange(100), "validation_clean": np.ones((20,3))}
        subset = trainer.paired_subset(payload, 20)
        self.assertEqual(set(subset), {"parameters", "clean", "row_id"})
        np.testing.assert_array_equal(subset["parameters"], small)
        np.testing.assert_array_equal(subset["clean"], payload["train_clean"][:20])
        with self.assertRaises(ValueError):
            trainer.paired_subset(payload, 101)

    def test_existing_bank_loader_never_reads_reserved_test_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez_compressed(root / "bank.npz", train_clean=np.ones((2, 3)),
                                validation_clean=np.ones((2, 3)), test_clean=np.zeros((2, 3)))
            identity = {"test": "loader isolation"}
            record = {"identity": identity, "payload_sha256": trainer.digest(root / "bank.npz")}
            (root / "bank.json").write_text(json.dumps(record))
            real_load = np.load

            class GuardedArchive:
                def __init__(self, *args, **kwargs):
                    self.archive = real_load(*args, **kwargs)
                    self.files = self.archive.files

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.archive.close()

                def __getitem__(self, key):
                    if key.startswith("test_"):
                        raise AssertionError("reserved test payload was read")
                    return self.archive[key]

            with patch.object(trainer.np, "load", side_effect=GuardedArchive):
                payload, _ = trainer.load_or_generate_bank(root, identity, None)
            self.assertEqual(set(payload), {"train_clean", "validation_clean"})


if __name__ == "__main__":
    unittest.main()

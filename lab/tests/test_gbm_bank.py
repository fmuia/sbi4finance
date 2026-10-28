"""Small synthetic-bank checks independent of training or market fixtures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from sbi4finance.gbm_bank import generate_examples, load_bank, write_bank


class GBMBankTests(unittest.TestCase):
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
            "root_seed": 2026,
            "splits": {"train": 16, "validation": 8, "test": 8},
        }
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_example_values_survive_reordering_rebatching_and_count_changes(self):
        ids = [0, 3, 7]
        original = generate_examples(self.config, "train", ids)
        reordered = generate_examples(self.config, "train", [7, 0, 3])
        for name in ("observations", "parameters", "example_id"):
            assert_array_equal(reordered[name], original[name][[2, 0, 1]])
            separate = [generate_examples(self.config, "train", [i])[name] for i in ids]
            assert_array_equal(np.concatenate(separate), original[name])
        resized = deepcopy(self.config)
        resized["splits"] = {"train": 32, "validation": 16, "test": 16}
        enlarged = generate_examples(resized, "train", ids)
        for name, value in original.items():
            assert_array_equal(enlarged[name], value)

    def test_splits_have_distinct_examples_and_explicit_truth_arrays(self):
        train = generate_examples(self.config, "train", [0, 1, 2, 3])
        test = generate_examples(self.config, "test", [0, 1, 2, 3])
        self.assertEqual(
            set(train), {"observations", "parameters", "dt_years", "example_id"}
        )
        self.assertEqual(train["observations"].shape, (4, 8, 1))
        self.assertEqual(train["parameters"].shape, (4, 1))
        self.assertEqual(train["dt_years"].shape, (8,))
        for name in ("observations", "parameters", "dt_years"):
            self.assertEqual(train[name].dtype, np.dtype("float64"))
            self.assertTrue(np.isfinite(train[name]).all())
        assert_array_equal(train["example_id"], ["train:0", "train:1", "train:2", "train:3"])
        assert_array_equal(test["example_id"], ["test:0", "test:1", "test:2", "test:3"])
        self.assertFalse(np.array_equal(train["parameters"], test["parameters"]))
        self.assertFalse(np.array_equal(train["observations"], test["observations"]))
        assert_array_equal(train["dt_years"], np.full(8, 1 / 252))

    def test_saved_bank_round_trip_and_existing_directory_protection(self):
        path = self.directory / "bank"
        written = write_bank(path, self.config)
        manifest, splits = load_bank(path)
        self.assertEqual(manifest, written)
        self.assertEqual(manifest["data_origin"], "synthetic")
        self.assertEqual(manifest["evidence_scope"], "S")
        self.assertEqual(manifest["measure"], "P")
        self.assertEqual(manifest["truth_available"], ["drift"])
        self.assertFalse(manifest["future_outcomes_in_conditioning"])
        self.assertEqual(set(splits), {"train", "validation", "test"})
        for split, count in self.config["splits"].items():
            expected = generate_examples(self.config, split, list(range(count)))
            for name, values in expected.items():
                assert_array_equal(splits[split][name], values)
        with self.assertRaises(FileExistsError):
            write_bank(path, self.config)
        # Refusing a second writer must leave the original bank usable.
        reloaded, _ = load_bank(path)
        self.assertEqual(reloaded, manifest)

    def test_corruption_missing_files_and_semantic_tampering_are_rejected(self):
        for mode in ("bytes", "missing", "split_ids", "configuration"):
            with self.subTest(mode=mode):
                path = self.directory / mode
                manifest = write_bank(path, self.config)
                manifest_path = path / "manifest.json"
                record = manifest["splits"]["train"]
                source = path / record["file"]
                if mode == "bytes":
                    contents = bytearray(source.read_bytes())
                    contents[-1] ^= 1
                    source.write_bytes(contents)
                elif mode == "missing":
                    source.unlink()
                elif mode == "split_ids":
                    with np.load(source, allow_pickle=False) as saved:
                        arrays = dict(saved)
                    arrays["example_id"][0] = "test:0"
                    np.savez(source, **arrays)
                    # A valid checksum cannot excuse an invalid scientific ID.
                    record["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                    manifest_path.write_text(json.dumps(manifest))
                else:
                    manifest["configuration"]["task"]["prior_mean"] = 0.5
                    manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    load_bank(path)

    def test_invalid_configuration_and_example_requests_fail(self):
        changes = [
            ("task", "measure", "Q"),
            ("task", "task_id", "another_task"),
            ("task", "prior_std", 0.0),
            ("task", "volatility", 0.0),
            ("task", "n_observations", 0),
            ("splits", "train", -1),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                invalid = deepcopy(self.config)
                invalid[section][key] = value
                with self.assertRaises((ValueError, TypeError)):
                    generate_examples(invalid, "train", [0])
        for split, ids in (("unknown", [0]), ("train", [-1]), ("train", [1.5])):
            with self.subTest(split=split, ids=ids):
                with self.assertRaises((ValueError, TypeError)):
                    generate_examples(self.config, split, ids)
        invalid_path = self.directory / "invalid"
        with self.assertRaises((ValueError, TypeError)):
            write_bank(invalid_path, invalid)
        self.assertFalse(invalid_path.exists())

    def test_incomplete_or_failed_generation_is_not_a_loadable_bank(self):
        incomplete = self.directory / "incomplete"
        incomplete.mkdir()
        np.save(incomplete / "observations.npy", np.zeros((1, 8, 1)))
        with self.assertRaises((ValueError, FileNotFoundError)):
            load_bank(incomplete)

        failed = self.directory / "failed"
        with patch(
            "sbi4finance.gbm_bank.simulate_log_returns",
            side_effect=FloatingPointError("injected generation failure"),
        ) as simulator:
            with self.assertRaisesRegex(ValueError, "failed example train:0"):
                write_bank(failed, self.config)
            self.assertEqual(simulator.call_count, 1)  # No replacement draw.
        self.assertTrue((failed / "failure.json").is_file())
        self.assertFalse((failed / "manifest.json").exists())
        failure = json.loads((failed / "failure.json").read_text())
        self.assertIn("injected generation failure", json.dumps(failure))
        self.assertIn("train:0", json.dumps(failure))
        with self.assertRaises((ValueError, FileNotFoundError)):
            load_bank(failed)


if __name__ == "__main__":
    unittest.main()

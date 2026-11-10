import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_heston_spread_bank_resolution.py"
spec = importlib.util.spec_from_file_location("spread_bank_resolution", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BankResolutionTests(unittest.TestCase):
    def test_exact_neighbors_match_brute_force_whitened_distances(self):
        rng = np.random.default_rng(61)
        train, queries, h = rng.normal(size=(40, 5)), rng.normal(size=(7, 5)), np.array([.1, .2, .5, 1., 2.])
        preserved = train.copy()
        result = audit.exact_neighbors(train, queries, h, prefixes=(10, 40))
        for n in (10, 40):
            distances = np.linalg.norm((queries[:, None] - train[None, :n]) / h, axis=-1)
            np.testing.assert_allclose(result[str(n)]["distance"], distances.min(axis=1), rtol=1e-13, atol=1e-13)
            np.testing.assert_array_equal(result[str(n)]["nearest_training_index"], distances.argmin(axis=1))
        self.assertTrue(np.all(result["40"]["distance"] <= result["10"]["distance"]))
        np.testing.assert_array_equal(train, preserved)

    def test_invalid_scaling_and_shapes_fail_before_tree_build(self):
        for h in ([1., 0.], [1., np.nan], [1.]):
            with self.assertRaisesRegex(ValueError, "positive spreads"):
                audit.exact_neighbors(np.ones((4, 2)), np.ones((2, 2)), h, prefixes=(2, 4))
        with self.assertRaises(ValueError):
            audit.exact_neighbors(np.ones((4, 2)), np.ones((2, 2)), [1., 1.], prefixes=(4, 2))

    def test_named_archive_reads_never_touch_reserved_or_parameter_members(self):
        accessed = []
        class GuardedArchive:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __getitem__(self, key):
                accessed.append(key)
                if key == "train_clean": return np.ones((4, 2))
                if key == "validation_clean": return np.ones((3, 2))
                raise AssertionError("unexpected NPZ member: " + key)
        with patch.object(audit.np, "load", return_value=GuardedArchive()):
            train, validation = audit.read_clean_arrays("unused", 4, 3, 2, query_rows=2)
        self.assertEqual(accessed, ["train_clean", "validation_clean"])
        self.assertEqual(train.shape, (4, 2))
        self.assertEqual(validation.shape, (2, 2))

    def test_cli_help_runs_without_payloads_or_model_backend(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--binding-record", result.stdout)


if __name__ == "__main__":
    unittest.main()

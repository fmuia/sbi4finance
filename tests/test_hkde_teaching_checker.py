"""Regression checks for isolation when the combined stack contains raw banks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "teaching_checker", Path(__file__).parents[1] / "scripts/check_hkde_teaching_integration.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class TeachingCheckerIsolationTests(unittest.TestCase):
    def test_combined_results_copy_excludes_unneeded_data_and_models(self):
        required = (
            "src/sbi4finance/__init__.py", "scripts/helper.py", "configs/task.json",
            "docs/results/hkde-correction/validation-checks.npz",
            "checkpoints/hkde-theta-intensity-provisional/model.keras",
            "data/synthetic/hkde-walkthrough/panel/observations.csv",
        )
        unrelated = (
            "docs/results/rough-reference8192/banks/reference/test.npz",
            "docs/results/hkde-surrogate/bank-test.npz",
            "tests/fixtures/local/observations.csv", "data/market/quotes.csv",
            "checkpoints/rough/model.keras",
        )
        with tempfile.TemporaryDirectory() as directory:
            root, isolated = Path(directory) / "source", Path(directory) / "isolated"
            for name in (*required, *unrelated):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            hashes, status = checker.copy_inputs(root, isolated, required[3:])
            self.assertEqual(set(hashes), set(required))
            self.assertFalse(any(status.values()))
            for name in required:
                self.assertEqual((isolated / name).read_bytes(), (root / name).read_bytes())
            for name in unrelated:
                self.assertFalse((isolated / name).exists())

    def test_accidentally_bound_raw_bank_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, isolated = Path(directory) / "source", Path(directory) / "isolated"
            name = "docs/results/rough-reference8192/banks/reference/test.npz"
            path = root / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b"raw bank")
            with self.assertRaisesRegex(RuntimeError, "unexpected inputs"):
                checker.copy_inputs(root, isolated, {name})


if __name__ == "__main__":
    unittest.main()

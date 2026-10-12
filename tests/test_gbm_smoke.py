"""Checks specific to the cluster smoke verifier, without a scheduler dependency."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sbi4finance.gbm_bank import write_bank


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_gbm_bank", ROOT / "scripts/check_gbm_bank.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class GBMSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bank = Path(temporary.name) / "bank"
        config = json.loads((ROOT / "configs/tasks/gbm_bank.json").read_text())
        config["splits"] = {"train": 4, "validation": 4, "test": 4}
        self.manifest = write_bank(self.bank, config)

    def save_manifest(self):
        (self.bank / "manifest.json").write_text(json.dumps(self.manifest))

    def test_regenerates_selected_ids_and_declares_version_differences(self):
        report = CHECKER.check_bank(self.bank)
        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_examples"]["train"]["regenerated_ids"],
                         ["train:3", "train:2", "train:0"])
        self.manifest["versions"]["python"] = "another-build"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "environment versions"):
            CHECKER.check_bank(self.bank)
        portable = CHECKER.check_bank(self.bank, portable=True)
        self.assertEqual(portable["mode"], "portable")
        self.assertNotEqual(portable["generating_versions"], portable["checking_versions"])

    def test_source_identity_cannot_be_skipped_or_changed(self):
        original = self.manifest["source_sha256"].copy()
        for source in ({}, {**original, "gbm.py": "0" * 64}):
            with self.subTest(source=source):
                self.manifest["source_sha256"] = source
                self.save_manifest()
                with self.assertRaisesRegex(ValueError, "source"):
                    CHECKER.check_bank(self.bank, portable=True)

    def test_regeneration_rejects_wrong_values_even_with_a_valid_file_hash(self):
        file = self.bank / "train.npz"
        with np.load(file, allow_pickle=False) as saved:
            arrays = dict(saved)
        arrays["observations"][0, 0, 0] += 0.001
        np.savez(file, **arrays)
        self.manifest["splits"]["train"]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        self.save_manifest()
        for portable in (False, True):
            with self.subTest(portable=portable), self.assertRaises(AssertionError):
                CHECKER.check_bank(self.bank, portable=portable)


if __name__ == "__main__":
    unittest.main()

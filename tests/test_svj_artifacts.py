"""Artifact binding rejects incompatible tasks and altered reference/checkpoint bytes."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from sbi4finance.svj import artifacts, bank

ROOT = Path(__file__).resolve().parents[1]


class TestSVJArtifacts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "checkpoint"
        shutil.copytree(ROOT / "checkpoints/svj", self.checkpoint)

    def test_original_bytes_and_complete_binding(self):
        record = artifacts.validate_checkpoint(self.checkpoint)
        self.assertEqual(record["task"], bank.default_task())
        self.assertEqual(record["files"], {
            "model.keras": "02fd80d07babc73f7f454cf43c6390fe9dc0ad77b9f50af042264c378d6b81cf",
            "training.json": "5e2379ac7a11885065b09355918c4a3a5e3033e08594ce8ef8d76a383f43f850",
            "misspecification.npz": "d2740b384da70f99555bc6133abd43b57695f159e2e5a7861fb90464ea4155ef",
        })
        self.assertFalse(record["scientific_acceptance"])
        self.assertTrue(record["provenance"]["historical_unknowns"])

    def test_hashes_task_features_solver_and_versions_rejected(self):
        original = json.loads((self.checkpoint / artifacts.SIDECAR).read_text())
        for change in (lambda r: r["features"].update(squared_return_offset=1),
                       lambda r: r["sampler"].update(steps=20),
                       lambda r: r["task"]["constants"].update(variance_floor=0),
                       lambda r: r["standardization"].update(conditioning="per_window"),
                       lambda r: r["runtime_versions"].update(jax="unknown")):
            record = copy.deepcopy(original)
            change(record)
            (self.checkpoint / artifacts.SIDECAR).write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                artifacts.validate_checkpoint(self.checkpoint)
        (self.checkpoint / artifacts.SIDECAR).write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "50 Euler"):
            artifacts.validate_checkpoint(self.checkpoint, steps=49)
        for name in ("model.keras", "training.json", "misspecification.npz"):
            path = self.checkpoint / name
            raw = path.read_bytes()
            path.write_bytes(raw + b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                artifacts.validate_checkpoint(self.checkpoint)
            path.write_bytes(raw)

    def test_new_reference_binding_never_replaces_original(self):
        old_hash = artifacts.sha256(self.checkpoint / "misspecification.npz")
        reference = Path(self.temporary.name) / "new-reference.npz"
        reference.write_bytes(b"reference fixture")
        artifacts.bind_reference(reference, checkpoint=self.checkpoint, seed=73)
        self.assertEqual(artifacts.validate_reference(reference, checkpoint=self.checkpoint)["seed"], 73)
        self.assertEqual(artifacts.sha256(self.checkpoint / "misspecification.npz"), old_hash)
        with self.assertRaises(FileExistsError):
            artifacts.bind_reference(reference, checkpoint=self.checkpoint, seed=74)
        reference.write_bytes(b"altered")
        with self.assertRaises(ValueError):
            artifacts.validate_reference(reference, checkpoint=self.checkpoint)

    def test_missing_sidecar_stops_before_model_deserialization(self):
        from unittest.mock import patch
        from sbi4finance.svj import amortized
        (self.checkpoint / artifacts.SIDECAR).unlink()
        with patch.object(amortized.keras.saving, "load_model") as load:
            with self.assertRaises(ValueError):
                amortized.AmortizedPosterior.load(self.checkpoint)
            load.assert_not_called()

    def test_reference_object_cannot_be_applied_to_another_checkpoint(self):
        from sbi4finance.svj.checks import MisspecificationTest
        posterior = SimpleNamespace(artifact=artifacts.validate_checkpoint(self.checkpoint))
        reference = MisspecificationTest.load(self.checkpoint / "misspecification.npz")
        reference.assert_compatible(posterior)
        other = Path(self.temporary.name) / "other-checkpoint"
        shutil.copytree(self.checkpoint, other)
        (other / "model.keras").write_bytes(b"distinct-model-fixture")
        binding = json.loads((other / artifacts.SIDECAR).read_text())
        binding["files"]["model.keras"] = artifacts.sha256(other / "model.keras")
        (other / artifacts.SIDECAR).write_text(json.dumps(binding))
        mismatched = MisspecificationTest.load(other / "misspecification.npz")
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            mismatched.assert_compatible(posterior)


if __name__ == "__main__":
    unittest.main()

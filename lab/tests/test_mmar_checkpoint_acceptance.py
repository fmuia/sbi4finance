"""Imported MMAR checkpoints retain exact bytes and an honest, bound wrapper."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from sbi4f_tutorial.mmar_takehome import model


spec = importlib.util.spec_from_file_location("mmar_checkpoint_accept", Path(__file__).resolve().parents[1] / "studies" / "accept.py")
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class MMARCheckpointAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "import-wrapper"
        self.source.mkdir()
        self.assets = self.root / "tutorial" / "assets"
        self.assets.mkdir(parents=True)
        self.task = json.loads(json.dumps(model.DESIGN))
        # Intentionally not a Keras archive: acceptance must hash/copy the
        # artifact, without importing a neural framework or inspecting weights.
        self.payload = b"original imported model bytes\x00\xff"
        (self.source / "model.keras").write_bytes(self.payload)
        self.metadata = {"format": "mmar_posterior_v1",
                         "task_sha256": fingerprint(self.task),
                         "model_sha256": hashlib.sha256(self.payload).hexdigest(),
                         "origin": "https://example.org/pinned-source/checkpoints/sensitivity/model.keras",
                         "scope": "Imported original source checkpoint with preserved bytes; historical training bank and seeds not independently reconstructed."}
        self.write_wrapper()
        self.destination = self.assets / "local-checkpoints" / "mmar-import"

    def tearDown(self):
        self.temporary.cleanup()

    def write_wrapper(self):
        (self.source / "task.json").write_text(json.dumps(self.task, indent=6) + "\n")
        (self.source / "training.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def run_accept(self):
        arguments = ["accept.py", "--checkpoint", str(self.source), "--to", "local-checkpoints/mmar-import"]
        with mock.patch.object(accept, "ASSETS", self.assets), mock.patch.object(accept, "REPOSITORY", self.root), \
                mock.patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            accept.main()

    def test_valid_import_is_copied_unchanged_without_loading_or_stripping_model(self):
        with mock.patch.object(accept, "strip_optimizer", side_effect=AssertionError("must preserve bytes")), \
                mock.patch.object(accept, "same_draws", side_effect=AssertionError("must not load models")), \
                mock.patch.dict(sys.modules, {"bayesflow": None, "keras": None}):
            self.run_accept()
        for name in ("model.keras", "task.json", "training.json"):
            self.assertEqual((self.destination / name).read_bytes(), (self.source / name).read_bytes())
        receipt = json.loads((self.destination / "ACCEPTED.json").read_text())
        self.assertFalse(receipt["scientific_acceptance"])
        self.assertEqual(receipt["checkpoint_origin"], self.metadata["origin"])
        self.assertEqual(receipt["scope"], self.metadata["scope"])
        self.assertIn("not a newly trained", receipt["checkpoint_provenance"])
        self.assertEqual(receipt["source_sha256"]["model.keras"], self.metadata["model_sha256"])

    def test_bad_task_or_model_hash_rejects_without_publishing_or_staging(self):
        original = dict(self.metadata)
        for key in ("task_sha256", "model_sha256"):
            with self.subTest(key=key):
                self.metadata = dict(original, **{key: "0"*64})
                self.write_wrapper()
                with self.assertRaisesRegex(ValueError, "hash|bytes"):
                    self.run_accept()
                self.assertFalse(self.destination.exists())
                self.assertFalse(self.destination.with_name(self.destination.name+".staging").exists())

    def test_self_consistent_changed_task_does_not_bypass_runtime_design_binding(self):
        self.task["prior"]["q"]["bounds"] = [.5, .99]
        self.metadata["task_sha256"] = fingerprint(self.task)
        self.write_wrapper()
        with self.assertRaisesRegex(ValueError, "complete model design"):
            self.run_accept()
        self.assertFalse(self.destination.exists())

    def test_import_requires_recorded_source_and_provenance_limits(self):
        original = dict(self.metadata)
        for key in ("origin", "scope"):
            with self.subTest(key=key):
                self.metadata = dict(original, **{key: ""})
                self.write_wrapper()
                with self.assertRaisesRegex(ValueError, "provenance"):
                    self.run_accept()
                self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()

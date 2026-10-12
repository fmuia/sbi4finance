"""Storage and conditioning boundaries for small local observation artifacts."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sbi4finance.datasets.local import ArtifactError, load_artifact, write_artifact


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "local"
METADATA = {
    "schema_version": 1,
    "artifact_id": "parser-example",
    "task_id": "parser-only",
    "artifact_type": "return_window",
    "representation": "simple_returns",
    "measure": "P",
    "data_origin": "synthetic",
    "purpose": "parser_test",
    "conventions": {"return_unit": "decimal", "normalization_version": 1},
    "provenance": {"source": "hand-constructed; not simulator truth"},
    "columns": [
        {"name": "id", "dtype": "str", "role": "id"},
        {"name": "return", "dtype": "float", "role": "observation",
         "unit": "decimal", "nullable": True},
        {"name": "dt", "dtype": "float", "role": "context", "unit": "year"},
        {"name": "quote_time", "dtype": "str", "role": "auxiliary", "nullable": True},
        {"name": "future_return", "dtype": "float", "role": "auxiliary", "unit": "decimal"},
    ],
}
ROWS = [
    {"id": "second", "return": 0.0, "dt": 0.01, "quote_time": None, "future_return": 0.1},
    {"id": "first", "return": None, "dt": 0.02, "quote_time": "known", "future_return": -0.1},
]


class LocalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def artifact(self, name="artifact", metadata=None, rows=None):
        path = self.root / name
        write_artifact(path, metadata if metadata is not None else METADATA,
                       rows if rows is not None else ROWS)
        return path

    def modify_manifest(self, path, change):
        manifest = path / "manifest.json"
        meta = json.loads(manifest.read_text())
        change(meta)
        manifest.write_text(json.dumps(meta))

    def replace_table(self, path, text):
        payload = text.encode()
        (path / "observations.csv").write_bytes(payload)
        self.modify_manifest(path, lambda m: m["table"].update(
            sha256=hashlib.sha256(payload).hexdigest()))

    def test_all_documented_fixtures_roundtrip(self):
        for name in ("returns", "options_iv", "options_raw"):
            with self.subTest(name=name):
                original = load_artifact(FIXTURES / name)
                self.assertEqual(original.metadata["purpose"], "parser_test")
                destination = self.root / name
                write_artifact(destination, original.metadata, original.rows)
                restored = load_artifact(destination)
                self.assertEqual(restored.rows, original.rows)
                self.assertEqual(restored.metadata, original.metadata)

    def test_order_zero_missing_and_unlabelled_are_preserved(self):
        loaded = load_artifact(self.artifact())
        self.assertEqual(tuple(row["id"] for row in loaded.rows), ("second", "first"))
        self.assertEqual(loaded.observations, {"return": (0.0, None)})
        self.assertIsNone(loaded.rows[0]["quote_time"])
        self.assertIsNone(loaded.truth)
        self.assertNotIn("truth", loaded.metadata)
        self.assertEqual(loaded.context, {"dt": (0.01, 0.02)})

    def test_labels_and_auxiliary_values_cannot_enter_conditioning(self):
        original = load_artifact(self.artifact())
        metadata, rows = copy.deepcopy(METADATA), copy.deepcopy(ROWS)
        metadata["truth"] = {"parameters": {"sigma": 0.2}, "state": [0.04]}
        rows[0]["future_return"] = 999.0
        rows[0]["quote_time"] = "different metadata"
        changed = load_artifact(self.artifact("changed", metadata, rows))
        self.assertEqual(original.observations, changed.observations)
        self.assertEqual(original.context, changed.context)
        self.assertEqual(changed.truth, metadata["truth"])
        metadata["data_origin"] = "market"
        with self.assertRaisesRegex(ArtifactError, "market observations"):
            self.artifact("invalid-market-truth", metadata, rows)

    def test_integrity_and_incomplete_artifacts(self):
        corrupt = self.artifact("corrupt")
        with (corrupt / "observations.csv").open("a") as stream:
            stream.write("unexpected modification")
        with self.assertRaisesRegex(ArtifactError, "checksum"):
            load_artifact(corrupt)
        count = self.artifact("count")
        self.modify_manifest(count, lambda m: m["table"].update(n_rows=10))
        with self.assertRaisesRegex(ArtifactError, "row count"):
            load_artifact(count)
        for missing in ("manifest.json", "observations.csv"):
            with self.subTest(missing=missing):
                path = self.artifact(missing)
                (path / missing).unlink()
                with self.assertRaisesRegex(ArtifactError, "cannot load"):
                    load_artifact(path)

    def test_versions_column_roles_and_header_are_checked(self):
        changes = [
            (lambda m: m.update(schema_version=2), "schema_version"),
            (lambda m: m["columns"][1].update(role="target"), "role"),
            (lambda m: m["columns"][1].update(name="id"), "unique"),
            (lambda m: m["columns"][1].pop("unit"), "unit"),
        ]
        for i, (change, message) in enumerate(changes):
            with self.subTest(message=message):
                path = self.artifact(str(i))
                self.modify_manifest(path, change)
                with self.assertRaisesRegex(ArtifactError, message):
                    load_artifact(path)
        path = self.artifact("header")
        self.replace_table(path, (path / "observations.csv").read_text().replace(
            "id,return,dt", "id,dt,return", 1))
        with self.assertRaisesRegex(ArtifactError, "header/order"):
            load_artifact(path)

    def test_duplicate_identity_and_bad_numbers_fail_after_hash_verification(self):
        for i, (row, message) in enumerate([
            ("second,0,0.02,,0", "duplicate row ID"),
            ("first,bad,0.02,,0", "invalid number"),
            ("first,nan,0.02,,0", "non-finite number"),
            ("first,inf,0.02,,0", "non-finite number"),
            ("first,0,,,0", "missing required value"),
        ]):
            with self.subTest(row=row):
                path = self.artifact(str(i))
                lines = (path / "observations.csv").read_text().splitlines()
                self.replace_table(path, "\n".join(lines[:2] + [row]) + "\n")
                with self.assertRaisesRegex(ArtifactError, message):
                    load_artifact(path)

    def test_writer_rejects_overwrite_and_invalid_rows_before_publication(self):
        path = self.artifact()
        before = (path / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            write_artifact(path, METADATA, ROWS)
        self.assertEqual((path / "manifest.json").read_bytes(), before)
        rows = copy.deepcopy(ROWS)
        rows[0]["undeclared_numeric"] = 1.0
        with self.assertRaisesRegex(ArtifactError, "row keys"):
            self.artifact("invalid", rows=rows)
        self.assertFalse((self.root / "invalid").exists())

    def test_import_and_loading_need_no_site_packages_or_network(self):
        path = self.artifact()
        script = """
import socket, sys
sys.path.insert(0, sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError('local loading attempted network access')
socket.socket = socket.create_connection = forbidden
from sbi4finance.datasets.local import ArtifactError, load_artifact
assert load_artifact(sys.argv[2]).observations['return'] == (0.0, None)
try:
    load_artifact(sys.argv[2] + '-missing')
except ArtifactError:
    pass
else:
    raise AssertionError('missing local data did not fail')
assert not {'numpy', 'pandas', 'bayesflow', 'yfinance'} & sys.modules.keys()
"""
        subprocess.run([sys.executable, "-S", "-c", script, str(ROOT / "src"), str(path)],
                       check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()

"""Portable acquisition preserves the original bundle and explicit model choice."""

from contextlib import contextmanager
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import heston_notebook_bundle as bundle


class HestonNotebookBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.lab = Path(self.temporary.name) / "lab"
        self.lab.mkdir()
        self.payload = b"unchanged-native-payload"
        self.manifest = json.dumps({
            "format": "heston_actual63_collaborator_bundle_v1",
            "files": {"checkpoints/native.bin": {
                "bytes": len(self.payload), "sha256": sha256(self.payload).hexdigest()}},
        }).encode()
        self.manifest_hash = sha256(self.manifest).hexdigest()

    def make_tree(self, path):
        (path / "checkpoints").mkdir(parents=True)
        (path / "bundle.json").write_bytes(self.manifest)
        (path / "checkpoints/native.bin").write_bytes(self.payload)
        return path

    def make_archive(self, extras=()):
        archive = self.lab / "download.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for name, value in (("bundle.json", self.manifest), ("checkpoints/native.bin", self.payload)):
                member = tarfile.TarInfo(bundle.BUNDLE_DIRECTORY + "/" + name)
                member.size = len(value)
                output.addfile(member, io.BytesIO(value))
            for member in extras:
                output.addfile(member, io.BytesIO(b"") if member.isfile() else None)
        return archive

    @contextmanager
    def trusted_fixture(self, archive=None):
        values = {"MANIFEST_SHA256": self.manifest_hash}
        if archive:
            values.update(ARCHIVE_BYTES=archive.stat().st_size,
                          ARCHIVE_SHA256=sha256(archive.read_bytes()).hexdigest())
        with patch.multiple(bundle, **values):
            yield

    def test_absent_empty_and_explicit_choices_stay_distinct(self):
        self.assertIsNone(bundle.resolve_bundle(None, lab=self.lab))
        cached = self.make_tree(bundle.default_bundle_path(self.lab))
        with self.trusted_fixture():
            self.assertEqual(bundle.resolve_bundle(None, lab=self.lab), cached)
            self.assertIsNone(bundle.resolve_bundle("", lab=self.lab))
            external = self.make_tree(self.lab.parent / "external")
            self.assertEqual(bundle.resolve_bundle(str(external), lab=self.lab), external.resolve())
            with self.assertRaisesRegex(ValueError, "No bundle.json"):
                bundle.resolve_bundle(self.lab / "wrong", lab=self.lab)

    def test_wrong_bundle_is_not_substituted_or_relabelled(self):
        path = self.make_tree(self.lab / "different")
        with self.assertRaisesRegex(ValueError, "hash-pinned"):
            bundle.resolve_bundle(path, lab=self.lab)
        self.assertEqual((path / "bundle.json").read_bytes(), self.manifest)

    def test_install_validates_and_preserves_payloads_and_is_idempotent(self):
        archive = self.make_archive()
        with self.trusted_fixture(archive):
            installed = bundle.install_archive(archive, lab=self.lab)
            self.assertEqual(installed, bundle.default_bundle_path(self.lab))
            self.assertEqual((installed / "checkpoints/native.bin").read_bytes(), self.payload)
            self.assertEqual(bundle.install_archive(archive, lab=self.lab), installed)
            self.assertEqual(list(installed.parent.glob(".install-*")), [])

    def test_wrong_download_or_corrupt_member_is_rejected_without_replacement(self):
        archive = self.make_archive()
        with self.assertRaisesRegex(ValueError, "Drive sign-in page"):
            bundle.install_archive(archive, lab=self.lab)
        self.assertFalse(bundle.default_bundle_path(self.lab).exists())
        existing = self.make_tree(bundle.default_bundle_path(self.lab))
        (existing / "checkpoints/native.bin").write_bytes(b"corrupted")
        with self.trusted_fixture(archive), self.assertRaisesRegex(ValueError, "corrupt"):
            bundle.install_archive(archive, lab=self.lab)
        self.assertEqual((existing / "checkpoints/native.bin").read_bytes(), b"corrupted")

    def test_archive_path_escapes_links_and_duplicate_names_are_rejected(self):
        members = [tarfile.TarInfo("../outside"), tarfile.TarInfo("/absolute"),
                   tarfile.TarInfo("wrong-root/inside"),
                   tarfile.TarInfo(bundle.BUNDLE_DIRECTORY + "/bundle.json")]
        symlink = tarfile.TarInfo(bundle.BUNDLE_DIRECTORY + "/link")
        symlink.type, symlink.linkname = tarfile.SYMTYPE, "../../outside"
        members.append(symlink)
        for member in members:
            with self.subTest(member=member.name):
                archive = self.make_archive([member])
                with self.trusted_fixture(archive), self.assertRaisesRegex(ValueError, "Unsafe"):
                    bundle.install_archive(archive, lab=self.lab)
                self.assertFalse(bundle.default_bundle_path(self.lab).exists())
        self.assertFalse((self.lab.parent / "outside").exists())

    def test_reviewed_ui_lock_reconstructs_the_original_numerical_lock(self):
        current = (bundle.LAB / "pixi.lock").read_bytes()
        self.assertEqual(sha256(current).hexdigest(), bundle.NOTEBOOK_UI_LOCK_SHA256)
        url = (b"https://files.pythonhosted.org/packages/e7/c3/"
               b"3031c931098de393393e1f93a38dc9ed6805d86bb801acc3cf2d5bd1e6b7/plotly-6.5.0-py3-none-any.whl")
        reference = b"      - pypi: " + url + b"\n"
        self.assertEqual(current.count(reference), 5)
        lines = current.splitlines(keepends=True)
        start = lines.index(b"- pypi: " + url + b"\n")
        end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith(b"- "))
        self.assertEqual(end - start, 40)
        self.assertEqual(sha256(b"".join(lines[start:end])).hexdigest(),
                         "6a38009cedca9fc2bf395dd676ea8302ce2b3b33be248b7529a239e0a0d27845")
        original = b"".join(lines[:start] + lines[end:]).replace(reference, b"")
        self.assertEqual(sha256(original).hexdigest(), bundle.ORIGINAL_LOCK_SHA256)

    def test_ui_exception_is_exact_lock_only_and_rejects_numerical_changes(self):
        current = (bundle.LAB / "pixi.lock").read_bytes()
        lock = self.lab / "pixi.lock"
        lock.write_bytes(current)
        self.assertEqual(bundle._notebook_source_digest(lock, lock_path=lock), bundle.ORIGINAL_LOCK_SHA256)
        self.assertEqual(bundle._notebook_source_digest(lock, lock_path=self.lab / "another.lock"),
                         bundle.NOTEBOOK_UI_LOCK_SHA256)
        for old, new in ((b"jax-0.11.1", b"jax-0.11.2"), (b"plotly-6.5.0", b"plotly-6.5.1")):
            with self.subTest(change=old):
                self.assertIn(old, current)
                changed = current.replace(old, new)
                lock.write_bytes(changed)
                self.assertEqual(bundle._notebook_source_digest(lock, lock_path=lock), sha256(changed).hexdigest())
                self.assertNotEqual(bundle._notebook_source_digest(lock, lock_path=lock),
                                    bundle.ORIGINAL_LOCK_SHA256)

    def test_adapter_keeps_canonical_loader_globals_and_export_unchanged(self):
        from scripts import heston_grid_collaborator as canonical
        original_digest = canonical.digest
        adapted = bundle.load_delivery_module()
        self.assertIs(canonical.digest, original_digest)
        self.assertIsNot(adapted.load_checkpoint.__globals__, canonical.load_checkpoint.__globals__)
        self.assertEqual(canonical.digest(bundle.LAB / "pixi.lock"), bundle.NOTEBOOK_UI_LOCK_SHA256)
        self.assertTrue(callable(canonical.export))
        self.assertFalse(hasattr(adapted, "export"))
        self.assertEqual(adapted.load_checkpoint.__globals__["digest"](bundle.LAB / "pixi.lock"),
                         bundle.ORIGINAL_LOCK_SHA256)


if __name__ == "__main__":
    unittest.main()

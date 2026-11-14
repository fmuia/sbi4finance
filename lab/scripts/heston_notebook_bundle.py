"""Install and locate the immutable Heston SPX notebook bundle.

Download the archive in a browser signed into the project's Google Drive first.
Importing this module performs no network requests, training or native model
imports. The explicit read-only delivery adapter below loads the original
native loader when the notebook needs inference or its preserved bank.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from types import SimpleNamespace


ARCHIVE_FILENAME = "heston-actual63-development-20261111.tar.gz"
ARCHIVE_BYTES = 464907858
ARCHIVE_SHA256 = "a2c23ab962d534e8807de0488c37e93581062bac8d87e81dc32a2f06c3380b3a"
MANIFEST_SHA256 = "3ec94acb5728efd5f2dab65cecd299345ea1191f282b439ee8b4008848dc5343"
BUNDLE_DIRECTORY = "heston-actual63-development"
DRIVE_FILE_URL = "https://drive.google.com/file/d/1ukmoPqlEED4fLbVPX0f8W5JaIPWrLsyI/view"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1UYBredhF2hrxoZI-sP7sAawC3V6XQpB8"
LAB = Path(__file__).resolve().parents[1]
# The archived bundle pins the complete numerical environment. The notebook
# adds only the Plotly 6.5.0 UI wheel (five environment refs + one package record).
# Review of that exact additive diff reconstructs ORIGINAL_LOCK_SHA256 byte for
# byte. This allowlist accepts no other lock update, including numerical pins.
ORIGINAL_LOCK_SHA256 = "994fa6075abf60c5e2045a6e2d73302a0a84af9b37e311e8ac2cdfcd8854a871"
NOTEBOOK_UI_LOCK_SHA256 = "61faf7922eee794f5ad277f2ac4e07437ab39772a0921df0ddf9cc160d3e0676"


def digest(path):
    result = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def default_bundle_path(lab=LAB):
    return Path(lab).resolve() / ".cache" / "heston-actual63" / BUNDLE_DIRECTORY


def _notebook_source_digest(path, *, lock_path, native_digest=digest):
    """Permit the reviewed UI-only lock in read-only notebook loading."""
    actual = native_digest(path)
    if Path(path).resolve() == Path(lock_path).resolve() and actual == NOTEBOOK_UI_LOCK_SHA256:
        return ORIGINAL_LOCK_SHA256
    return actual


@lru_cache(maxsize=1)
def load_delivery_module():
    """Return read-only APIs of an isolated original loader, with one UI exception.

    The original delivery module, producer globals, archive, recorded source
    hashes and network payloads are never changed. Only the archived lock-source
    check recognizes the exact reviewed Plotly-only notebook lock. Every other
    source/version and native task/model/bank check remains byte-exact. This is
    inference/review compatibility, not provenance for generation or resumption.
    No export or generation API is exposed by this facade.
    """
    source = LAB / "scripts" / "heston_grid_collaborator.py"
    spec = importlib.util.spec_from_file_location("_heston_actual63_notebook_delivery", source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original_digest = native.digest
    lock_path = LAB / "pixi.lock"

    def notebook_digest(path):
        return _notebook_source_digest(path, lock_path=lock_path, native_digest=original_digest)

    native.digest = notebook_digest  # private module only; the canonical module is untouched
    return SimpleNamespace(load_bank=native.load_bank, load_checkpoint=native.load_checkpoint,
                           training_inputs=native.training_inputs, verify=native.verify,
                           compatibility_scope="read-only exact Plotly-6.5.0 UI lock addition")


def validate_bundle(directory, *, payloads=False):
    """Check this delivery's identity; native loading additionally checks sources."""
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "bundle.json"
    if not manifest_path.is_file():
        raise ValueError(f"No bundle.json at {directory}. Select the extracted {BUNDLE_DIRECTORY} folder, "
                         "not its archive or parent folder.")
    if digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError("This is not the notebook's hash-pinned Heston actual63 bundle. "
                         f"Install {ARCHIVE_FILENAME}; older tutorial bundles have a different task.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "heston_actual63_collaborator_bundle_v1":
        raise ValueError("The Heston actual63 bundle format differs.")
    if payloads:
        for name, expected in manifest["files"].items():
            path = (directory / name).resolve()
            if (not path.is_relative_to(directory) or not path.is_file()
                    or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]):
                raise ValueError("Missing or corrupt Heston bundle member: " + name)
    return directory


def resolve_bundle(explicit_env=None, *, lab=LAB):
    """Return the selected path or None; never download or load a model.

    An absent environment value (None) enables the documented local cache.
    An empty string disables the optional model, including cache discovery.
    A supplied path must be the correct extracted bundle; mistakes are reported
    instead of silently substituting some other cached model.
    """
    if explicit_env == "":
        return None
    if explicit_env is not None:
        return validate_bundle(explicit_env)
    cached = default_bundle_path(lab)
    return validate_bundle(cached) if cached.exists() else None


def _safe_members(archive):
    names = set()
    members = archive.getmembers()
    for member in members:
        name = PurePosixPath(member.name)
        if (name.is_absolute() or ".." in name.parts or not name.parts
                or name.parts[0] != BUNDLE_DIRECTORY or "\\" in member.name
                or ":" in member.name or member.name in names
                or not (member.isfile() or member.isdir())):
            raise ValueError("Unsafe or unexpected archive member: " + member.name)
        names.add(member.name)
    return members


def install_archive(archive_path, *, lab=LAB):
    """Verify, securely extract and atomically install the unchanged delivery."""
    archive_path = Path(archive_path).expanduser().resolve()
    if (not archive_path.is_file() or archive_path.stat().st_size != ARCHIVE_BYTES
            or digest(archive_path) != ARCHIVE_SHA256):
        raise ValueError(f"The archive must be {ARCHIVE_FILENAME} ({ARCHIVE_BYTES:,} bytes), "
                         f"SHA-256 {ARCHIVE_SHA256}. Download the actual file, not a Drive sign-in page.")
    target = default_bundle_path(lab)
    if target.exists():
        return validate_bundle(target, payloads=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=target.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as source:
            source.extractall(staging, members=_safe_members(source), filter="data")
        candidate = validate_bundle(staging / BUNDLE_DIRECTORY, payloads=True)
        candidate.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="downloaded Heston actual63 .tar.gz to verify and install")
    parser.add_argument("--check", action="store_true", help="verify every member of the installed cache")
    args = parser.parse_args()
    if args.archive:
        path = install_archive(args.archive)
    elif args.check:
        path = validate_bundle(default_bundle_path(), payloads=True)
    else:
        parser.error("use --archive PATH to install, or --check to verify the local cache")
    print(f"Verified Heston bundle: {path}")
    print("The notebook discovers this cache automatically. Restart marimo if it is already running.")


if __name__ == "__main__":
    main()

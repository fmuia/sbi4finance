"""Append-only calibration shards and immutable, lightweight prefix snapshots.

The legacy generator remains unchanged: counts and scheduling never enter an
example's random identity. Collection identity binds that generator, task, root
seed and runtime. A snapshot binds exact immutable parts and split counts.
"""

from contextlib import contextmanager
import copy
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import bank


COLLECTION_FORMAT = "calibration_bank_collection_v1"
SNAPSHOT_FORMAT = "calibration_bank_snapshot_v1"
RNG_PROTOCOL = "calibration_legacy_seedsequence_uint64_pricing_v1"
TRAINING_ROLES = ("observations", "reference_prices", "parameters", "context_values", "example_id")
_MAX = 2**32 - 1
_SPLITS = tuple(bank.SPLITS)


def _digest(value):
    return hashlib.sha256(bank._json(value).encode()).hexdigest()


def _integer(value, name, minimum=0):
    value = bank._integer(value, name, minimum=minimum)
    if value > _MAX:
        raise ValueError(f"{name} must fit in an unsigned 32-bit integer")
    return value


def _config(config):
    bank.validate_config(config)
    _integer(config["root_seed"], "root seed")
    for split, count in config["splits"].items():
        _integer(count, f"{split} count", 1)
    return copy.deepcopy(config)


def _json_new(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Publish a complete file without overwriting an existing manifest.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _identity(config, sources, versions):
    return {"task": config["task"], "root_seed": config["root_seed"],
            "rng_protocol": RNG_PROTOCOL, "source_sha256": sources, "versions": versions}


def _read_collection(path, *, current=False):
    root = Path(path).resolve()
    record = json.loads((root / "collection.json").read_text())
    config = _config(record["configuration"])
    expected = _identity(config, record["source_sha256"], record["versions"])
    if (record.get("format") != COLLECTION_FORMAT or record.get("rng_protocol") != RNG_PROTOCOL
            or record.get("collection_id") != _digest(expected)
            or record.get("configuration_sha256") != _digest(config)
            or record.get("task_fingerprint") != bank.task_fingerprint(config)
            or record.get("fixed_splits") != {s: config["splits"][s] for s in ("validation", "test")}):
        raise ValueError("incompatible or changed collection manifest")
    if current and (record["source_sha256"] != bank.source_hashes(config["task"])
                    or record["versions"] != bank.versions()):
        raise ValueError("collection generator sources or runtime differ from this checkout")
    return root, record


def _target(record, train_count):
    config = copy.deepcopy(record["configuration"])
    config["splits"]["train"] = _integer(train_count, "training count", 1)
    return config


def _seed_prefix(config):
    root = _integer(config["root_seed"], "root seed")
    digest = bank.task_fingerprint(config)
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    return [0x4F5054, root, *words]


def _pricing_seed_from_prefix(prefix, split_code, example_id):
    """Low-level legacy derivation; the expensive task digest is already computed."""
    sequence = np.random.SeedSequence([*prefix, split_code, example_id, 1])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def pricing_seed(config, split, example_id):
    """Legacy pricing seed, exposed for exact duplicate audits without pricing."""
    if split not in bank.SPLITS:
        raise ValueError("unknown split")
    example_id = _integer(example_id, "example ID")
    return _pricing_seed_from_prefix(_seed_prefix(config), bank.SPLITS[split], example_id)


def _audit_seeds(config):
    seen, prefix = {}, _seed_prefix(config)
    split_names = {code: name for name, code in bank.SPLITS.items()}
    for split in _SPLITS:
        code = bank.SPLITS[split]
        for example_id in range(config["splits"][split]):
            seed = _pricing_seed_from_prefix(prefix, code, example_id)
            if seed in seen:
                previous = seen[seed]
                raise ValueError(f"pricing seed collision: {split_names[previous >> 32]}:{previous & _MAX} "
                                 f"and {split}:{example_id}")
            seen[seed] = (code << 32) | example_id
    return {"protocol": RNG_PROTOCOL, "configuration_sha256": _digest(config),
            "examples_checked": len(seen), "duplicate_pricing_seeds": 0}


def create_collection(path, config):
    config = _config(config)
    sources, versions = bank.source_hashes(config["task"]), bank.versions()
    record = {"format": COLLECTION_FORMAT, "configuration": config,
              "configuration_sha256": _digest(config), "task_fingerprint": bank.task_fingerprint(config),
              "rng_protocol": RNG_PROTOCOL, "source_sha256": sources, "versions": versions,
              "fixed_splits": {s: config["splits"][s] for s in ("validation", "test")},
              "collection_id": _digest(_identity(config, sources, versions))}
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("parts", ".locks", ".tmp"):
        (root / name).mkdir()
    _json_new(root / "collection.json", record)
    return record


def _name(split, start, stop):
    return f"{split}-{start:010d}-{stop:010d}"


def _range(config, split, start, stop):
    start, stop = _integer(start, "start"), _integer(stop, "stop", 1)
    if split not in bank.SPLITS or not start < stop <= config["splits"][split]:
        raise ValueError("part must be a nonempty range within the target split")
    return start, stop


def _compatible(config, record):
    config = _config(config)
    original = record["configuration"]
    if (config["task"] != original["task"] or config["root_seed"] != original["root_seed"]
            or any(config["splits"][s] != record["fixed_splits"][s] for s in record["fixed_splits"])):
        raise ValueError("task, root seed or fixed validation/test allocation differs")
    return config


def _part(path, record, *, full=True):
    path = Path(path)
    manifest = path / "manifest.json"
    part = json.loads(manifest.read_text())
    config = _compatible(part["configuration"], record)
    split, start, stop = part["split"], part["start"], part["stop"]
    _range(config, split, start, stop)
    if (part.get("format") != bank.SHARD_FORMAT or part.get("file") != "arrays.npz"
            or part.get("examples") != stop - start or part.get("configuration_sha256") != _digest(config)
            or part.get("task_fingerprint") != record["task_fingerprint"]
            or part.get("source_sha256") != record["source_sha256"]
            or part.get("versions") != record["versions"]):
        raise ValueError(f"incompatible or changed shard: {path}")
    payload = path / "arrays.npz"
    if bank._sha256(payload) != part.get("sha256"):
        raise ValueError(f"shard checksum mismatch: {path}")
    if full:
        with np.load(payload, allow_pickle=False) as saved:
            arrays = dict(saved)
        bank.check_arrays(arrays, config, split, start, stop)
    return {"path": path.resolve(), "split": split, "start": start, "stop": stop,
            "manifest_sha256": bank._sha256(manifest), "array_sha256": part["sha256"], "record": part}


def _inventory(root, record, *, full=True):
    parts = []
    for path in sorted((root / "parts").iterdir()):
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"unexpected collection part: {path}")
        part = _part(path, record, full=full)
        if path.name != _name(part["split"], part["start"], part["stop"]):
            raise ValueError("part path disagrees with its manifest range")
        parts.append(part)
    _no_overlap(parts)
    return parts


def _no_overlap(parts):
    for split in _SPLITS:
        previous = None
        for part in sorted((p for p in parts if p["split"] == split), key=lambda p: p["start"]):
            if previous is not None and part["start"] < previous["stop"]:
                raise ValueError(f"overlapping {split} ranges")
            previous = part


@contextmanager
def _lock(path, *, nonblocking=False):
    with Path(path).open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _active_overlap(root, split, start, stop, *, own=None):
    for path in (root / ".locks").glob(f"{split}-*.lock"):
        if path == own:
            continue
        _, left, right = path.stem.rsplit("-", 2)
        if int(left) >= stop or int(right) <= start:
            continue
        try:
            with _lock(path, nonblocking=True):
                pass
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise ValueError("an overlapping generation job is active") from error


def _existing(parts, split, start, stop):
    overlaps = [p for p in parts if p["split"] == split and p["start"] < stop and p["stop"] > start]
    if not overlaps:
        return None
    if len(overlaps) == 1 and overlaps[0]["start"] == start and overlaps[0]["stop"] == stop:
        return overlaps[0]
    raise ValueError("requested range overlaps existing immutable parts")


def _range_conflicts(root, split, start, stop):
    """Check only range names under the mutation lock, never unrelated payloads.

    Parts are published atomically under that lock. Full content validation is
    done at adoption/planning/freezing, and on the exact part reused by a worker.
    A worker must not decompress all earlier parts on every publication.
    """
    overlaps = []
    with os.scandir(root / "parts") as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError(f"unexpected collection part: {entry.path}")
            try:
                name, left, right = entry.name.rsplit("-", 2)
                left, right = int(left), int(right)
            except ValueError as error:
                raise ValueError(f"invalid collection part name: {entry.name}") from error
            if (name not in bank.SPLITS or not 0 <= left < right <= _MAX
                    or entry.name != _name(name, left, right)):
                raise ValueError(f"invalid collection part name: {entry.name}")
            if name == split and left < stop and right > start:
                overlaps.append({"path": Path(entry.path), "split": name, "start": left, "stop": right})
    return overlaps


def adopt_shards(path, shard_paths):
    root, record = _read_collection(path, current=True)
    # Validate all source parts before publishing any imported part.
    sources = [_part(p, record) for p in shard_paths]
    _no_overlap(sources)
    adopted = []
    with _lock(root / ".locks" / "mutation.lock"):
        existing = _inventory(root, record)
        for source in sources:
            match = _existing(existing, source["split"], source["start"], source["stop"])
            if match and (match["array_sha256"] != source["array_sha256"]
                          or match["manifest_sha256"] != source["manifest_sha256"]):
                raise ValueError("existing part differs from the source being adopted")
            _active_overlap(root, source["split"], source["start"], source["stop"])
        for source in sources:
            destination = root / "parts" / _name(source["split"], source["start"], source["stop"])
            if not destination.exists():
                temporary = Path(tempfile.mkdtemp(prefix="adopt-", dir=root / ".tmp"))
                try:
                    staged = temporary / "part"
                    shutil.copytree(source["path"], staged)
                    copied = _part(staged, record)
                    if copied["manifest_sha256"] != source["manifest_sha256"]:
                        raise ValueError("source manifest changed while copying")
                    os.rename(staged, destination)
                finally:
                    shutil.rmtree(temporary)
            adopted.append(str(destination))
    return {"collection_id": record["collection_id"], "adopted": adopted}


def _prefix(parts, config, *, complete=False):
    selected, missing = [], []
    for split in _SPLITS:
        limit, cursor = config["splits"][split], 0
        for part in sorted((p for p in parts if p["split"] == split), key=lambda p: p["start"]):
            if part["start"] >= limit:
                continue
            if part["stop"] > limit:
                raise ValueError(f"{split} prefix cuts through a finished shard; use a shard boundary")
            if part["start"] > cursor:
                missing.append((split, cursor, part["start"]))
            selected.append(part)
            cursor = part["stop"]
        if cursor < limit:
            missing.append((split, cursor, limit))
    if complete and missing:
        raise ValueError(f"snapshot has missing ID ranges: {missing}")
    return selected, missing


def _job_hash(job):
    return _digest({k: job[k] for k in ("collection_id", "collection_sha256", "configuration", "split", "start", "stop")})


def plan_extension(path, train_count, chunk_size):
    root, record = _read_collection(path, current=True)
    chunk_size = _integer(chunk_size, "chunk size", 1)
    config = _target(record, train_count)
    seed_audit = _audit_seeds(config)
    with _lock(root / ".locks" / "mutation.lock"):
        parts = _inventory(root, record)
        _, missing = _prefix(parts, config)
    jobs = []
    for split, start, stop in missing:
        for first in range(start, stop, chunk_size):
            job = {"collection_id": record["collection_id"],
                   "collection_sha256": bank._sha256(root / "collection.json"), "configuration": config,
                   "split": split, "start": first, "stop": min(first + chunk_size, stop)}
            job["job_sha256"] = _job_hash(job)
            jobs.append(job)
    return {"collection_id": record["collection_id"], "configuration": config,
            "seed_audit": seed_audit, "jobs": jobs}


def run_job(path, job):
    root, record = _read_collection(path, current=True)
    if (job.get("collection_id") != record["collection_id"]
            or job.get("collection_sha256") != bank._sha256(root / "collection.json")
            or job.get("job_sha256") != _job_hash(job)):
        raise ValueError("changed job or different collection")
    config = _compatible(job["configuration"], record)
    split, start, stop = job["split"], job["start"], job["stop"]
    _range(config, split, start, stop)
    name = _name(split, start, stop)
    destination, lock = root / "parts" / name, root / ".locks" / (name + ".lock")
    with _lock(lock):
        with _lock(root / ".locks" / "mutation.lock"):
            match = _existing(_range_conflicts(root, split, start, stop), split, start, stop)
            if match is None:
                _active_overlap(root, split, start, stop, own=lock)
        if match is not None:
            actual = _part(match["path"], record)
            if (actual["split"], actual["start"], actual["stop"]) != (split, start, stop):
                raise ValueError("existing part manifest disagrees with the requested range")
            return actual["record"]
        temporary = Path(tempfile.mkdtemp(prefix=name + "-", dir=root / ".tmp"))
        try:
            staged = temporary / "part"
            bank.write_shard(staged, config, split, start, stop)
            part = _part(staged, record)
            if (part["split"], part["start"], part["stop"]) != (split, start, stop):
                raise ValueError("generated part disagrees with the requested range")
            with _lock(root / ".locks" / "mutation.lock"):
                if _existing(_range_conflicts(root, split, start, stop), split, start, stop) is not None:
                    raise ValueError("another immutable part appeared while this job ran")
                os.rename(staged, destination)
        except BaseException as error:
            # Keep the original failure.json (or interrupted partial work) apart
            # from immutable finished parts. A retry gets a new staging name;
            # failed work is never eligible for adoption or snapshot coverage.
            try:
                failures = root / "failures"
                failures.mkdir(exist_ok=True)
                os.rename(temporary, failures / temporary.name)
            except OSError as preservation_error:
                error.add_note(f"Could not preserve staging at {temporary}: {preservation_error}")
            raise
        else:
            # Successful publication moved the part; remove its empty staging.
            shutil.rmtree(temporary)
        return part["record"]


def freeze_snapshot(path, train_count, out):
    root, record = _read_collection(path)
    destination = Path(out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    config = _target(record, train_count)
    seed_audit = _audit_seeds(config)
    with _lock(root / ".locks" / "mutation.lock"):
        parts, _ = _prefix(_inventory(root, record), config, complete=True)
        identity = {"configuration_sha256": _digest(config), "task_fingerprint": record["task_fingerprint"],
                    "source_sha256": record["source_sha256"]}
        manifest = bank._manifest(config, identity, record["versions"])
        manifest.update(format=SNAPSHOT_FORMAT, collection_id=record["collection_id"], seed_audit=seed_audit,
                        collection={"path": os.path.relpath(root, destination),
                                    "manifest_sha256": bank._sha256(root / "collection.json")}, parts=[])
        for part in parts:
            manifest["parts"].append({k: part[k] for k in
                                      ("split", "start", "stop", "manifest_sha256", "array_sha256")}
                                     | {"path": os.path.relpath(part["path"], destination)})
        manifest["splits"] = {s: {"examples": n} for s, n in config["splits"].items()}
        destination.mkdir(parents=True, exist_ok=False)
        _json_new(destination / "manifest.json", manifest)
    return manifest


def _roles(config, roles):
    expected = bank.array_shapes(config, 1)
    chosen = tuple(expected) if roles is None else tuple(roles)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(expected):
        raise ValueError("unknown, duplicate or empty array roles")
    return chosen


def _read_roles(payload, config, split, start, stop, roles):
    shapes = bank.array_shapes(config, stop - start)
    with np.load(payload, allow_pickle=False) as saved:
        if set(saved.files) != set(shapes):
            raise ValueError("unexpected shard arrays")
        arrays = {key: saved[key] for key in roles}
    for key, value in arrays.items():
        if value.shape != shapes[key] or (key != "example_id"
                and (value.dtype != np.float64 or not np.isfinite(value).all())):
            raise ValueError(f"invalid projected array: {key}")
    if "example_id" in arrays and not np.array_equal(arrays["example_id"],
                                                     [f"{split}:{i}" for i in range(start, stop)]):
        raise ValueError("changed example identities")
    return arrays


def _read_snapshot(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    config = _config(manifest["configuration"])
    collection_root = (path / manifest["collection"]["path"]).resolve()
    root, record = _read_collection(collection_root)
    if (manifest.get("format") != SNAPSHOT_FORMAT or manifest.get("collection_id") != record["collection_id"]
            or manifest.get("measure") != "Q" or manifest.get("evidence_scope") != "S"
            or manifest["collection"]["manifest_sha256"] != bank._sha256(root / "collection.json")
            or manifest.get("configuration_sha256") != _digest(config)
            or manifest.get("task_fingerprint") != record["task_fingerprint"]
            or manifest.get("source_sha256") != record["source_sha256"]
            or manifest.get("versions") != record["versions"]
            or manifest.get("splits") != {s: {"examples": n} for s, n in config["splits"].items()}
            or manifest.get("seed_audit") != {"protocol": RNG_PROTOCOL, "configuration_sha256": _digest(config),
                                               "examples_checked": sum(config["splits"].values()),
                                               "duplicate_pricing_seeds": 0}):
        raise ValueError("incompatible or changed snapshot manifest")
    _compatible(config, record)
    parts = []
    for reference in manifest["parts"]:
        _range(config, reference["split"], reference["start"], reference["stop"])
        part_path = (path / reference["path"]).resolve()
        if part_path != root / "parts" / _name(reference["split"], reference["start"], reference["stop"]):
            raise ValueError("snapshot reference escapes its canonical collection part")
        # Payload verification occurs only for requested splits, avoiding test-data reads during training.
        metadata = {k: reference[k] for k in ("split", "start", "stop", "manifest_sha256", "array_sha256")}
        metadata["path"] = part_path
        parts.append(metadata)
    _no_overlap(parts)
    selected, _ = _prefix(parts, config, complete=True)
    if len(selected) != len(parts):
        raise ValueError("snapshot contains ranges beyond its declared coverage")
    return path, manifest, record, parts


def iter_snapshot(path, split, roles=None):
    """Yield requested array members one verified shard at a time, in ID order."""
    _, manifest, record, parts = _read_snapshot(path)
    if split not in bank.SPLITS:
        raise ValueError("unknown split")
    config = manifest["configuration"]
    chosen = _roles(config, roles)
    for reference in sorted((p for p in parts if p["split"] == split), key=lambda p: p["start"]):
        actual = _part(reference["path"], record, full=roles is None)
        if any(actual[k] != reference[k] for k in ("manifest_sha256", "array_sha256", "split", "start", "stop")):
            raise ValueError("immutable snapshot part changed")
        yield _read_roles(reference["path"] / "arrays.npz", config, split,
                          reference["start"], reference["stop"], chosen)


def load_bank(path, splits=None, roles=None):
    """Read legacy banks or snapshots; projected reads never decode unrequested members."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") == bank.FORMAT and roles is None:
        return bank.load_bank(path, splits=splits)
    if manifest.get("format") == bank.FORMAT:
        config = _config(manifest["configuration"])
        if (manifest.get("configuration_sha256") != _digest(config)
                or manifest.get("task_fingerprint") != bank.task_fingerprint(config)
                or manifest.get("measure") != "Q" or manifest.get("evidence_scope") != "S"
                or set(manifest.get("splits", {})) != set(_SPLITS)):
            raise ValueError("incompatible legacy bank manifest")
        selected = _SPLITS if splits is None else tuple(splits)
        chosen, result = _roles(config, roles), {}
        if not set(selected) <= set(_SPLITS) or len(set(selected)) != len(selected):
            raise ValueError("unknown or duplicate split")
        for split in selected:
            item = manifest["splits"][split]
            if item.get("file") != f"{split}.npz" or item.get("examples") != config["splits"][split]:
                raise ValueError("inconsistent legacy split record")
            payload = path / item["file"]
            if bank._sha256(payload) != item["sha256"]:
                raise ValueError("legacy bank checksum mismatch")
            result[split] = _read_roles(payload, config, split, 0, config["splits"][split], chosen)
        return manifest, result
    if manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("unknown calibration bank format")
    _, manifest, _, _ = _read_snapshot(path)
    config = manifest["configuration"]
    selected = _SPLITS if splits is None else tuple(splits)
    if not set(selected) <= set(_SPLITS) or len(set(selected)) != len(selected):
        raise ValueError("unknown or duplicate split")
    chosen, result = _roles(config, roles), {}
    for split in selected:
        count = config["splits"][split]
        shapes = bank.array_shapes(config, count)
        arrays = {key: np.empty(shapes[key], dtype=(f"U{len(split) + 1 + len(str(count - 1))}"
                                                   if key == "example_id" else np.float64)) for key in chosen}
        cursor = 0
        for part in iter_snapshot(path, split, roles=roles):
            size = len(part[chosen[0]])
            for key in chosen:
                arrays[key][cursor:cursor + size] = part[key]
            cursor += size
        result[split] = arrays
    return manifest, result

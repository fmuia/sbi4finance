"""Versioned return-path tasks and deterministic, sharded synthetic banks.

Latent variance is simulated but is never part of the saved conditioning data.
The rough tasks define a synthetic P law explicitly; they do not infer a change
of measure from the existing option-pricing model. Every trajectory starts at
the declared origin. Prefixes are nested observations of that same trajectory.
"""

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from sbi4finance import heston
from sbi4f_tutorial.trajectories.simulation import (
    MODELS,
    SPLITS,
    TEACHING_LANDMARKS,
    TEACHING_TASK_IDS,
    _integer,
    _json,
    _rng,
    default_task,
    generate,
    task_fingerprint,
    teaching_task,
    validate_task,
)
from sbi4finance.runtime_sources import trajectory_runtime_hashes


FORMAT = "physical_return_path_bank_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_bank(task, out, train, validation, test, chunk=128, seed=30102026):
    """Write new shards, verify them, then publish the completion manifest last."""
    digest = task_fingerprint(task)
    counts = {k: _integer(v, k) for k, v in zip(SPLITS, (train, validation, test), strict=True)}
    chunk, seed = _integer(chunk, "chunk", 1), _integer(seed, "seed")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    manifest = {"format": FORMAT, "task": task, "task_sha256": digest, "root_seed": seed,
                "split_counts": counts, "splits": {k: [] for k in SPLITS},
                "roles": {"observations": "observed decimal log returns [trajectory,time,1]",
                          "parameters": "synthetic target labels only [trajectory,parameter]",
                          "trajectory_id": "split-qualified identity; never conditioning"},
                "provenance": {"data_origin": "synthetic", "measure": "P",
                    "prior_independence": True, "future_outcomes_in_conditioning": False,
                    "latent_states_in_conditioning": False,
                    "versions": {"python": platform.python_version(), **{
                        k: metadata.version(k) for k in ("numpy", "scipy", "numba")}},
                    "source_sha256": {**{str(p.relative_to(package)): _sha256(p) for p in
                        (Path(__file__), package / "heston.py", package / "rough_bergomi.py",
                         package / "bergomi.py", package / "runtime_sources.py")},
                        **trajectory_runtime_hashes(task)}},
                "timing_seconds": {}, "payload_bytes": 0,
                "payload_bytes_by_split": {split: 0 for split in SPLITS}}
    started = perf_counter()
    try:
        for split, count in counts.items():
            (out / split).mkdir()
            split_started = perf_counter()
            for index, start in enumerate(range(0, count, chunk)):
                arrays = generate(task, split, start, min(chunk, count - start), seed)
                path = out / split / f"part-{index:06d}.npz"
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez(stream, **arrays)
                os.replace(temporary, path)
                payload_bytes = path.stat().st_size
                manifest["splits"][split].append({"path": str(path.relative_to(out)),
                    "sha256": _sha256(path), "count": len(arrays["parameters"]), "start": start,
                    "bytes": payload_bytes})
                manifest["payload_bytes"] += payload_bytes
                manifest["payload_bytes_by_split"][split] += payload_bytes
            manifest["timing_seconds"][split] = perf_counter() - split_started
        manifest["timing_seconds"]["total"] = perf_counter() - started
        verification_started = perf_counter()
        for split in SPLITS:
            # Chunk verification avoids materializing the whole bank here.
            for record in manifest["splits"][split]:
                _load_part(out, manifest, split, record)
        manifest["timing_seconds"]["verification"] = perf_counter() - verification_started
        manifest["timing_seconds"]["total_including_verification"] = perf_counter() - started
        _write_json(out / "manifest.json", manifest)
    except Exception as error:
        (out / "manifest.json").unlink(missing_ok=True)
        _write_json(out / "failure.json", {"task_sha256": digest, "error": repr(error)})
        raise
    return manifest


def load_manifest(bank):
    bank = Path(bank)
    manifest = json.loads((bank / "manifest.json").read_text())
    if (manifest.get("format") != FORMAT or manifest.get("task_sha256") != task_fingerprint(manifest["task"])
            or set(manifest.get("splits", {})) != set(SPLITS)
            or set(manifest.get("split_counts", {})) != set(SPLITS)):
        raise ValueError("invalid path-bank manifest/task fingerprint")
    _integer(manifest["root_seed"], "root_seed")
    for split in SPLITS:
        expected = 0
        for index, record in enumerate(manifest["splits"][split]):
            if record.get("path") != f"{split}/part-{index:06d}.npz" or record.get("start") != expected:
                raise ValueError("noncanonical shard path or noncontiguous trajectory IDs")
            expected += _integer(record["count"], "part count", 1)
        if expected != _integer(manifest["split_counts"][split], "split count"):
            raise ValueError("split count does not match shard coverage")
    return manifest


def _load_part(bank, manifest, split, record):
    path = Path(bank) / record["path"]
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"{path}: checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    task, count = manifest["task"], record["count"]
    shapes = {"observations": (count, task["n_observations"], 1),
              "parameters": (count, len(task["parameter_names"])), "trajectory_id": (count,)}
    if set(arrays) != set(shapes) or any(arrays[k].shape != v for k, v in shapes.items()):
        raise ValueError("shard arrays/shape differ from manifest")
    for key in ("observations", "parameters"):
        if arrays[key].dtype != np.float64 or not np.isfinite(arrays[key]).all():
            raise ValueError("path arrays must be finite float64")
    bounds = np.asarray(task["prior_bounds"])
    if np.any(arrays["parameters"] < bounds[:, 0]) or np.any(arrays["parameters"] > bounds[:, 1]):
        raise ValueError("parameter labels lie outside the declared prior")
    expected = np.asarray([f"{manifest['task_sha256'][:16]}:{manifest['root_seed']}:{split}:{i}"
                           for i in range(record["start"], record["start"] + count)])
    if arrays["trajectory_id"].dtype.kind not in "US" or not np.array_equal(arrays["trajectory_id"], expected):
        raise ValueError("trajectory IDs differ from split-qualified manifest identities")
    return arrays


def load_split(bank, split):
    """Integrity-check and materialize only one requested split; return its task."""
    if split not in SPLITS:
        raise ValueError("unknown split")
    manifest = load_manifest(bank)
    parts = [_load_part(bank, manifest, split, record) for record in manifest["splits"][split]]
    task = manifest["task"]
    if parts:
        result = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    else:
        result = {"observations": np.empty((0, task["n_observations"], 1)),
                  "parameters": np.empty((0, len(task["parameter_names"]))),
                  "trajectory_id": np.asarray([], dtype="U1")}
    return {**result, "task": task}


def verify_bank(bank):
    """Check all native payloads while retaining at most one decoded shard.

    This is an integrity check, not an evaluation of posterior accuracy. In
    particular, it may inspect reserved-split identities and bytes without
    running inference or reporting statistical outcomes from those histories.
    """
    bank = Path(bank)
    manifest = load_manifest(bank)
    counts = {}
    checked_shards = 0
    largest_decoded_shard_bytes = 0
    expected_paths = set()
    for split in SPLITS:
        count = 0
        for record in manifest["splits"][split]:
            path = bank / record["path"]
            if "bytes" in record and path.stat().st_size != record["bytes"]:
                raise ValueError(f"{path}: payload size differs from manifest")
            arrays = _load_part(bank, manifest, split, record)
            count += len(arrays["parameters"])
            largest_decoded_shard_bytes = max(
                largest_decoded_shard_bytes, sum(value.nbytes for value in arrays.values()))
            del arrays
            checked_shards += 1
            expected_paths.add(record["path"])
        counts[split] = count
    actual_paths = {path.relative_to(bank).as_posix() for path in bank.rglob("*.npz")}
    if actual_paths != expected_paths:
        raise ValueError("native shard inventory differs from manifest")
    return {"passed": True, "task_sha256": manifest["task_sha256"], "counts": counts,
            "checked_shards": checked_shards,
            "largest_decoded_shard_bytes": largest_decoded_shard_bytes,
            "scope": "byte/array/identity integrity; not scientific acceptance"}

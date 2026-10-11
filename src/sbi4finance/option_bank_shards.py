"""Fixed-ID option-bank chunks for a CPU array, assembled into option_bank_v1.

The existing generator owns every random stream and numerical check. Chunks
only divide its IDs; they do not redraw failed examples or create a new bank
schema. A final manifest marks a complete chunk or assembled bank.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from . import option_bank
from .gbm_bank import _integer, _json, _sha256, _write_json


def _identity(config):
    return {
        "configuration_sha256": hashlib.sha256(_json(config).encode()).hexdigest(),
        "task_fingerprint": option_bank.task_fingerprint(config),
        "source_sha256": {name: _sha256(Path(option_bank.__file__).with_name(name))
                          for name in option_bank.source_files(config["task"]["model"])},
        "shard_writer_sha256": _sha256(Path(__file__)),
    }


def _versions():
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version()}


def _range(config, split, start, stop):
    option_bank.validate_config(config)
    start = _integer(start, "start")
    stop = _integer(stop, "stop", minimum=1)
    if split not in option_bank.SPLITS or not start < stop <= config["splits"][split]:
        raise ValueError("chunk must be a nonempty range within a configured split")
    return start, stop


def _save_arrays(path, arrays):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
    os.replace(temporary, path)


def write_shard(path, config, split, start, stop):
    """Generate [start, stop) in a new directory; failures preserve their ID."""
    start, stop = _range(config, split, start, stop)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    record = {"format": "option_bank_shard_v1", "configuration": config,
              **_identity(config), "versions": _versions(), "split": split,
              "start": start, "stop": stop, "file": "arrays.npz"}
    begun = time.perf_counter()
    try:
        arrays = option_bank.generate_examples(config, split, range(start, stop))
        destination = path / record["file"]
        _save_arrays(destination, arrays)
        record.update(sha256=_sha256(destination), examples=stop - start,
                      generation_seconds=time.perf_counter() - begun)
        _write_json(path / "manifest.json", record)
    except Exception as error:
        _write_json(path / "failure.json", {**record, "error": str(error),
                    "generation_seconds": time.perf_counter() - begun})
        raise
    return record


def _check_arrays(arrays, config, split, start, stop):
    """Check shard contents before a final bank manifest can mark success."""
    count = stop - start
    task = config["task"]
    c, p = len(task["context"]["contracts"]), len(task["parameter_names"])
    shapes = {"observations": (count, c, 1), "parameters": (count, p),
              "reference_prices": (count, c), "pricing_standard_errors": (count, c),
              "pricing_covariance": (count, c, c), "refinement_difference": (count, c),
              "example_id": (count,)}
    if set(arrays) != set(shapes):
        raise ValueError("unexpected shard arrays")
    for key, shape in shapes.items():
        a = arrays[key]
        if a.shape != shape or (key != "example_id" and
                               (a.dtype != np.float64 or not np.isfinite(a).all())):
            raise ValueError(f"invalid shard {key}")
    if not np.array_equal(arrays["example_id"], [f"{split}:{i}" for i in range(start, stop)]):
        raise ValueError("changed shard base-panel identity")
    bounds = np.asarray(task["prior_bounds"])
    if np.any(arrays["parameters"] < bounds[:, 0]) or np.any(arrays["parameters"] > bounds[:, 1]):
        raise ValueError("shard parameters outside prior")
    covariance, se = arrays["pricing_covariance"], arrays["pricing_standard_errors"]
    if (np.any(se < 0) or not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-12)
            or not np.allclose(se ** 2, np.diagonal(covariance, axis1=-2, axis2=-1), rtol=1e-10, atol=1e-14)
            or np.linalg.eigvalsh(covariance).min() < -1e-10):
        raise ValueError("invalid shard MC covariance/standard errors")
    numerics = task["numerics"]
    sigma = task["context"]["quote_noise_std"]
    if task["model"] == "hkde":
        if np.max(np.abs(arrays["refinement_difference"])) > sigma * numerics["max_refinement_over_quote_sigma"]:
            raise ValueError("shard Fourier refinement budget failed")
    elif np.max(se) > sigma * numerics["max_mc_se_over_quote_sigma"]:
        raise ValueError("shard Monte Carlo standard-error budget failed")


def _bank_manifest(config, identity, versions):
    # Keep the existing bank's schema and roles. The separate shard provenance
    # below records execution; it does not alter task identity or conditioning.
    model = config["task"]["model"]
    return {
        "format": "option_bank_v1", "configuration": config,
        "configuration_sha256": identity["configuration_sha256"],
        "task_fingerprint": identity["task_fingerprint"], "data_origin": "synthetic",
        "evidence_scope": "S", "measure": "Q", "purpose": "synthetic fixed-panel inference training/evaluation",
        "conditioning_cutoff_years": 0.0, "contract_selection": "fixed ordered synthetic contracts in task context",
        "roles": {"observations": "conditioning: noisy currency prices [batch,contract,1]",
                  "context": "known synthetic contracts/spot/rates/quote sigma; model_context also known; stored in configuration",
                  "parameters": "ordered synthetic training targets only; never observed conditioning",
                  "reference_prices": "auxiliary numerical expectation estimates, not exact market truth",
                  "pricing_standard_errors": "auxiliary MC standard errors (zero for deterministic Fourier)",
                  "pricing_covariance": "auxiliary covariance of MC price MEANS; zero for Fourier",
                  "refinement_difference": "auxiliary refined-minus-base Fourier estimate; MC zeros mean not calculated",
                  "example_id": "auxiliary split-qualified base-panel identity; future augmentations must keep base grouping"},
        "truth_available": config["task"]["parameter_names"], "future_outcomes_in_conditioning": False,
        "prior": "independent uniforms in explicit parameter_names order",
        "noise": "independent additive Gaussian currency quote errors, no clipping",
        "numerical_scope": {"mc_se_excludes_bias": True, "fourier_difference_is_error_bound": False,
                            "refinement_difference_available": model == "hkde", "posterior_validated": False},
        "rng": "PCG64; SeedSequence(option tag, root, task digest words, split, ID, purpose); purposes 0 prior,1 pricing,2 quotes",
        "versions": versions, "source_sha256": identity["source_sha256"], "splits": {},
    }


def assemble_bank(path, config, shard_paths):
    """Validate exact split coverage and write an ordinary option_bank_v1 bank.

    All chunks must use identical source and generation versions. Assembly may
    run on another machine; its versions are separately recorded. No output
    directory is created until every input chunk has passed validation.
    """
    option_bank.validate_config(config)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    identity = _identity(config)
    parts = {split: [] for split in option_bank.SPLITS}
    provenance, versions = [], None
    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        manifest_path = shard_path / "manifest.json"
        record = json.loads(manifest_path.read_text())
        if record.get("format") != "option_bank_shard_v1" or record.get("configuration") != config:
            raise ValueError("incompatible shard configuration")
        for key, value in identity.items():
            if record.get(key) != value:
                raise ValueError(f"shard {key} mismatch")
        if not isinstance(record.get("versions"), dict) or set(record["versions"]) != set(_versions()):
            raise ValueError("missing generation versions")
        if versions is None:
            versions = record["versions"]
        elif record["versions"] != versions:
            raise ValueError("shards use different generation versions")
        split, start, stop = record["split"], record["start"], record["stop"]
        _range(config, split, start, stop)
        if record.get("file") != "arrays.npz" or record.get("examples") != stop - start:
            raise ValueError("inconsistent shard range/file")
        source = shard_path / record["file"]
        if _sha256(source) != record.get("sha256"):
            raise ValueError("shard checksum mismatch")
        with np.load(source, allow_pickle=False) as saved:
            arrays = dict(saved)
        _check_arrays(arrays, config, split, start, stop)
        parts[split].append((start, stop, arrays))
        provenance.append({"split": split, "start": start, "stop": stop,
                           "manifest_sha256": _sha256(manifest_path),
                           "array_sha256": record["sha256"],
                           "generation_seconds": record["generation_seconds"]})
    for split, chunks in parts.items():
        cursor = 0
        chunks.sort(key=lambda item: item[0])
        for start, stop, _ in chunks:
            if start != cursor:
                raise ValueError(f"{split}: overlapping, duplicate or missing shard IDs")
            cursor = stop
        if cursor != config["splits"][split]:
            raise ValueError(f"{split}: missing shard IDs")
    manifest = _bank_manifest(config, identity, versions)
    manifest["shard_assembly"] = {
        "shard_writer_sha256": identity["shard_writer_sha256"],
        "assembly_versions": _versions(),
        "chunks": sorted(provenance, key=lambda p: (option_bank.SPLITS[p["split"]], p["start"])),
    }
    path.mkdir(parents=True, exist_ok=False)
    try:
        for split, chunks in parts.items():
            arrays = {key: np.concatenate([chunk[2][key] for chunk in chunks], axis=0)
                      for key in chunks[0][2]}
            destination = path / f"{split}.npz"
            _save_arrays(destination, arrays)
            manifest["splits"][split] = {"file": destination.name,
                                          "examples": config["splits"][split],
                                          "sha256": _sha256(destination)}
        _write_json(path / "manifest.json", manifest)
    except Exception as error:
        _write_json(path / "failure.json", {"configuration": config, "error": str(error)})
        raise
    return manifest

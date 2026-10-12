"""Small, independently reproducible GBM banks; NumPy arrays plus a manifest.

This is a fixed known-volatility P task, not a general storage or model API.
The neural adapter lives separately and only receives explicitly selected keys.
"""

import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

from .gbm import simulate_log_returns


TASK_ID = "gbm_drift_known_vol_bank_v1"
SPLITS = {"train": 0, "validation": 1, "test": 2}


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_config(config):
    """Reject unsupported task changes instead of silently ignoring them."""
    if set(config) != {"task", "root_seed", "splits"}:
        raise ValueError("config needs exactly task, root_seed and splits")
    task = config["task"]
    required = {"task_id", "measure", "volatility", "prior_mean", "prior_std",
                "n_observations", "dt_years"}
    if set(task) != required or task["task_id"] != TASK_ID or task["measure"] != "P":
        raise ValueError("unsupported task: expected the fixed known-volatility P GBM task")
    _integer(config["root_seed"], "root_seed")
    _integer(task["n_observations"], "n_observations", minimum=1)
    for key in ("volatility", "prior_mean", "prior_std", "dt_years"):
        value = task[key]
        if isinstance(value, bool) or np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key != "prior_mean" and value <= 0:
            raise ValueError(f"{key} must be positive")
    if set(config["splits"]) != set(SPLITS):
        raise ValueError("exactly train, validation and test counts are required")
    for split, count in config["splits"].items():
        _integer(count, split, minimum=1)
    _json(config)


def task_fingerprint(config):
    return hashlib.sha256(_json(config["task"]).encode()).hexdigest()


def generate_examples(config, split, example_ids):
    """Regenerate arbitrary local IDs, independently of ordering or bank sizes.

    An example identity is (task fingerprint, root seed, split, local ID).
    Prior and process shocks have separate PCG64/SeedSequence streams. IDs are
    split-qualified strings in saved arrays; each history is its own base group.
    Reproducibility is for the recorded NumPy/backend, not across all versions.
    """
    validate_config(config)
    if split not in SPLITS:
        raise ValueError("unknown split")
    ids = [_integer(i, "example ID") for i in example_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("example IDs must be nonempty and unique")
    task = config["task"]
    dt = np.full(task["n_observations"], task["dt_years"], dtype=np.float64)
    fingerprint = task_fingerprint(config)
    task_words = [int(fingerprint[i:i + 8], 16) for i in range(0, 64, 8)]
    observations = np.empty((len(ids), len(dt), 1), dtype=np.float64)
    parameters = np.empty((len(ids), 1), dtype=np.float64)
    for row, example_id in enumerate(ids):
        identity = [config["root_seed"], *task_words, SPLITS[split], example_id]
        prior_rng, shock_rng = [np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([*identity, purpose]))) for purpose in (0, 1)]
        try:
            with np.errstate(over="raise", invalid="raise"):
                mu = prior_rng.normal(task["prior_mean"], task["prior_std"])
                parameters[row, 0] = mu
                observations[row] = simulate_log_returns(
                    mu, task["volatility"], dt, rng=shock_rng)[0]
            if not np.isfinite(observations[row]).all() or not np.isfinite(mu):
                raise ValueError("nonfinite generated value")
        except (ValueError, FloatingPointError, OverflowError) as error:
            raise ValueError(f"failed example {split}:{example_id}: {error}") from error
    return {"observations": observations, "parameters": parameters,
            "dt_years": dt, "example_id": np.asarray([f"{split}:{i}" for i in ids])}


def _sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value))
    os.replace(temporary, path)


def write_bank(path, config):
    """Write a new bank; manifest.json is the last, atomic completion marker.

    Failed generation leaves failure.json with the original ID/error and config.
    Nothing is redrawn. Regenerate that ID to diagnose, then write a new bank.
    Existing directories are never overwritten or automatically resumed.
    """
    validate_config(config)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": "gbm_bank_v1", "configuration": config,
        "configuration_sha256": hashlib.sha256(_json(config).encode()).hexdigest(),
        "task_fingerprint": task_fingerprint(config),
        "data_origin": "synthetic", "evidence_scope": "S", "measure": "P",
        "purpose": "independent synthetic posterior training and reference evaluation",
        "conditioning_cutoff_years": config["task"]["n_observations"] * config["task"]["dt_years"],
        "roles": {"observations": "observed decimal log returns [batch,time,1]",
                  "dt_years": "known context [time]; other fixed context in task configuration",
                  "parameters": "synthetic drift truth [batch,1]; supervised target only",
                  "example_id": "auxiliary independent base/group identity; never conditioning"},
        "truth_available": ["drift"], "future_outcomes_in_conditioning": False,
        "rng": "PCG64; SeedSequence(root_seed, task SHA256 words, split code, local ID, purpose)",
        "split_codes": SPLITS, "versions": {"numpy": np.__version__, "python": platform.python_version()},
        "source_sha256": {name: _sha256(Path(__file__).with_name(name))
                          for name in ("gbm.py", "gbm_bank.py")},
        "splits": {},
    }
    try:
        for split, count in config["splits"].items():
            arrays = generate_examples(config, split, range(count))
            destination = path / f"{split}.npz"
            temporary = path / f"{split}.npz.tmp"
            with temporary.open("wb") as file:
                np.savez(file, **arrays)
            os.replace(temporary, destination)
            manifest["splits"][split] = {"file": destination.name, "examples": count,
                                         "sha256": _sha256(destination)}
        _write_json(path / "manifest.json", manifest)
    except Exception as error:
        _write_json(path / "failure.json", {"configuration": config, "error": str(error)})
        raise
    return manifest


def load_bank(path):
    """Validate a complete bank, then load its three small splits into memory."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    config = manifest["configuration"]
    validate_config(config)
    if (manifest["format"] != "gbm_bank_v1" or manifest["data_origin"] != "synthetic"
            or manifest["evidence_scope"] != "S" or manifest["measure"] != "P"
            or manifest["configuration_sha256"] != hashlib.sha256(_json(config).encode()).hexdigest()
            or manifest["task_fingerprint"] != task_fingerprint(config)
            or set(manifest["splits"]) != set(SPLITS)):
        raise ValueError("incompatible bank manifest")
    task = config["task"]
    result = {}
    all_ids = set()
    for split, count in config["splits"].items():
        record = manifest["splits"][split]
        if record["file"] != f"{split}.npz" or record["examples"] != count:
            raise ValueError(f"{split}: inconsistent manifest")
        source = path / record["file"]
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"{split}: checksum mismatch")
        with np.load(source, allow_pickle=False) as saved:
            arrays = dict(saved)
        shapes = {"observations": (count, task["n_observations"], 1),
                  "parameters": (count, 1), "dt_years": (task["n_observations"],),
                  "example_id": (count,)}
        if set(arrays) != set(shapes):
            raise ValueError(f"{split}: unexpected arrays")
        for key, shape in shapes.items():
            values = arrays[key]
            if values.shape != shape:
                raise ValueError(f"{split}: invalid {key} shape")
            if key != "example_id" and (values.dtype != np.float64 or not np.isfinite(values).all()):
                raise ValueError(f"{split}: invalid {key} values/dtype")
        expected_ids = np.asarray([f"{split}:{i}" for i in range(count)])
        if not np.array_equal(arrays["example_id"], expected_ids):
            raise ValueError(f"{split}: duplicate, missing or misassigned example IDs")
        ids = set(arrays["example_id"].tolist())
        if all_ids.intersection(ids):
            raise ValueError("base examples overlap across splits")
        all_ids.update(ids)
        if not np.array_equal(arrays["dt_years"], np.full(task["n_observations"], task["dt_years"])):
            raise ValueError(f"{split}: time grid differs from task")
        result[split] = arrays
    return manifest, result

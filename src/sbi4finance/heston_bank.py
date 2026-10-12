"""Small synthetic banks for one fixed-design Heston theta/state task.

The signed terminal auxiliary state is a supervised target, never an observed
input. Generation uses the same daily discrete law as the inference reference.
"""

import hashlib
import json
import os
from pathlib import Path
import platform

import numba
import numpy as np

from . import heston
from .gbm_bank import _integer, _json, _sha256, _write_json


TASK_ID = "heston_theta_state_v1"
SPLITS = {"train": 0, "validation": 1, "test": 2}
GENERATION_BATCH_SIZE = 256


def validate_config(config):
    """Accept the declared P, log-uniform-theta, one-step-per-day task only."""
    if set(config) != {"task", "root_seed", "splits"}:
        raise ValueError("config needs exactly task, root_seed and splits")
    task = config["task"]
    required = {"task_id", "measure", "parameters", "theta_bounds", "prior",
                "initial_variance", "n_observations", "dt_years", "substeps"}
    if (set(task) != required or task["task_id"] != TASK_ID
            or task["measure"] != "P" or task["prior"] != "log_uniform"):
        raise ValueError("unsupported Heston theta/state task")
    seed = _integer(config["root_seed"], "root_seed")
    if seed >= 2**64:
        raise ValueError("root_seed must fit an unsigned 64-bit integer")
    _integer(task["n_observations"], "n_observations", minimum=1)
    if _integer(task["substeps"], "substeps", minimum=1) != 1:
        raise ValueError("this reference task requires one update per observation")
    parameters = task["parameters"]
    if set(parameters) != {"drift", "kappa", "vol_of_vol", "rho"}:
        raise ValueError("fixed parameters must be drift, kappa, vol_of_vol and rho")
    for key, value in {**parameters, "initial_variance": task["initial_variance"],
                       "dt_years": task["dt_years"]}.items():
        if isinstance(value, bool) or np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite scalar")
        if key not in ("drift", "rho") and value <= 0:
            raise ValueError(f"{key} must be positive")
    if abs(parameters["rho"]) >= 1:
        raise ValueError("the conditional reference requires |rho| < 1")
    bounds = np.asarray(task["theta_bounds"], dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError("theta_bounds must be two increasing positive values")
    if set(config["splits"]) != set(SPLITS):
        raise ValueError("exactly train, validation and test counts are required")
    for split, count in config["splits"].items():
        _integer(count, split, minimum=1)
    _json(config)


def task_fingerprint(config):
    return hashlib.sha256(_json(config["task"]).encode()).hexdigest()


def generate_examples(config, split, example_ids):
    """Return requested examples independently of request order and chunk size."""
    validate_config(config)
    if split not in SPLITS:
        raise ValueError("unknown split")
    ids = [_integer(i, "example ID") for i in example_ids]
    if not ids or len(ids) != len(set(ids)) or max(ids) >= 2**64:
        raise ValueError("example IDs must be nonempty, unique unsigned 64-bit integers")
    task = config["task"]
    dt = np.full(task["n_observations"], task["dt_years"], dtype=np.float64)
    digest = task_fingerprint(config)
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    seed = config["root_seed"]
    identity = [0x48535431, seed & 0xffffffff, seed >> 32, *words, SPLITS[split]]
    kernel_seed = int(np.random.SeedSequence([*identity, 1]).generate_state(1, dtype=np.uint64)[0])
    bounds = np.log(task["theta_bounds"])
    theta = np.empty((len(ids), 1), dtype=np.float64)
    for row, example_id in enumerate(ids):
        prior_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [*identity, 0, example_id & 0xffffffff, example_id >> 32])))
        theta[row, 0] = np.exp(prior_rng.uniform(*bounds))
        if not task["theta_bounds"][0] < theta[row, 0] < task["theta_bounds"][1]:
            raise ValueError(f"failed example {split}:{example_id}: theta draw outside open bounds")
    observations = np.empty((len(ids), len(dt), 1), dtype=np.float64)
    terminal = np.empty((len(ids), 1), dtype=np.float64)
    for start in range(0, len(ids), GENERATION_BATCH_SIZE):
        stop = min(start + GENERATION_BATCH_SIZE, len(ids))
        try:
            noise = heston.draw_innovations(dt, 1, seed=kernel_seed,
                                            example_ids=ids[start:stop], stream_id=0)
            result = heston.simulate({**task["parameters"], "theta": theta[start:stop, 0]},
                                     dt, task["initial_variance"], noise, backend="numba")
            observations[start:stop] = result["log_returns"]
            terminal[start:stop] = result["variance_state"][:, -1, :]
        except (ValueError, FloatingPointError, OverflowError) as error:
            failed = [f"{split}:{i}" for i in ids[start:stop]]
            raise ValueError(f"failed example group {failed}: {error}") from error
    return {"observations": observations, "parameters": theta, "terminal_state": terminal,
            "dt_years": dt, "example_id": np.asarray([f"{split}:{i}" for i in ids])}


def write_bank(path, config):
    """Write a new bank with manifest-last completion; failures are not redrawn."""
    validate_config(config)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": "heston_bank_v1", "configuration": config,
        "configuration_sha256": hashlib.sha256(_json(config).encode()).hexdigest(),
        "task_fingerprint": task_fingerprint(config),
        "data_origin": "synthetic", "evidence_scope": "S", "measure": "P",
        "purpose": "joint theta/signed-state inference for the declared daily discrete model",
        "scheme": "full-truncation-euler-signed-auxiliary-v1; one update per observation",
        "conditioning_cutoff_years": config["task"]["n_observations"] * config["task"]["dt_years"],
        "roles": {"observations": "observed decimal log returns [batch,time,1]",
                  "parameters": "synthetic theta truth [batch,1]; target only",
                  "terminal_state": "signed terminal auxiliary truth [batch,1]; target only",
                  "dt_years": "known context [time]; remaining fixed context in task",
                  "example_id": "independent base/group identity; never conditioning"},
        "truth_available": ["theta", "signed_terminal_auxiliary_variance"],
        "future_outcomes_in_conditioning": False,
        "rng": "PCG64; task/root/split SeedSequence identity; separate prior/kernel domains and per-example IDs",
        "split_codes": SPLITS, "generation_batch_size": GENERATION_BATCH_SIZE,
        "versions": {"numpy": np.__version__, "numba": numba.__version__, "python": platform.python_version()},
        "source_sha256": {name: _sha256(Path(__file__).with_name(name))
                          for name in ("heston.py", "heston_bank.py", "gbm_bank.py")},
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
    """Verify the completed bank, task, split identities, arrays and checksums."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    config = manifest["configuration"]
    validate_config(config)
    if (manifest["format"] != "heston_bank_v1" or manifest["data_origin"] != "synthetic"
            or manifest["evidence_scope"] != "S" or manifest["measure"] != "P"
            or manifest["configuration_sha256"] != hashlib.sha256(_json(config).encode()).hexdigest()
            or manifest["task_fingerprint"] != task_fingerprint(config)
            or set(manifest["splits"]) != set(SPLITS)):
        raise ValueError("incompatible bank manifest")
    task, result = config["task"], {}
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
                  "parameters": (count, 1), "terminal_state": (count, 1),
                  "dt_years": (task["n_observations"],), "example_id": (count,)}
        if set(arrays) != set(shapes):
            raise ValueError(f"{split}: unexpected arrays")
        for key, shape in shapes.items():
            values = arrays[key]
            if values.shape != shape:
                raise ValueError(f"{split}: invalid {key} shape")
            if key != "example_id" and (values.dtype != np.float64 or not np.isfinite(values).all()):
                raise ValueError(f"{split}: invalid {key} values/dtype")
        if not np.array_equal(arrays["example_id"], [f"{split}:{i}" for i in range(count)]):
            raise ValueError(f"{split}: duplicate, missing or misassigned example IDs")
        low, high = task["theta_bounds"]
        if np.any((arrays["parameters"] <= low) | (arrays["parameters"] >= high)):
            raise ValueError(f"{split}: theta labels outside the open prior bounds")
        if not np.array_equal(arrays["dt_years"], np.full(task["n_observations"], task["dt_years"])):
            raise ValueError(f"{split}: time grid differs from task")
        result[split] = arrays
    return manifest, result

"""Bind SVJ checkpoints and diagnostic references to their complete task."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .adapters import ADAPTER_ID
from .bank import default_task, task_fingerprint, validate_task

SIDECAR = "artifact.json"
FORMAT = "svj_checkpoint_v1"
STANDARDIZATION = {
    "variables": "inference_variables", "conditioning": "unstandardized_native_features",
    "fit": "training_only", "inference": "frozen_saved_statistics",
    "state": "embedded_in_model.keras",
}
SAMPLER = {"method": "euler", "steps": 50, "noise": "numpy_default_rng_float32_normal"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_dir(path):
    path = Path(path)
    if path.is_file():
        if path.name != "model.keras":
            raise ValueError("a bound SVJ checkpoint must be named model.keras")
        return path.parent
    return path


def runtime_versions():
    return {name: version(name) for name in ("bayesflow", "keras", "jax")}


def write_checkpoint_sidecar(path, *, task=None, provenance=None):
    """Add a new sidecar; original model, training and reference bytes stay intact."""
    path = checkpoint_dir(path)
    task = default_task() if task is None else task
    validate_task(task)
    training = json.loads((path / "training.json").read_text())
    files = {name: sha256(path / name) for name in ("model.keras", "training.json")}
    files["misspecification.npz"] = sha256(path / "misspecification.npz") if (path / "misspecification.npz").exists() else None
    record = {
        "format": FORMAT, "task": task, "task_sha256": task_fingerprint({"task": task}),
        "adapter_id": ADAPTER_ID, "features": task["features"],
        "target_coordinates": task["target_coordinates"],
        "terminal_target": "log(sqrt(252 * terminal_variance) / 100)",
        "sampler": SAMPLER, "standardization": STANDARDIZATION,
        "files": files, "runtime_versions": training["versions"],
        "scientific_acceptance": False,
        "status": "provisional; artifact compatibility is not scientific acceptance",
        "provenance": provenance or {
            "training_lifecycle": training.get("lifecycle"),
            "source_sha256": training.get("source_sha256"),
            "runtime_environment": training.get("runtime_environment"),
            "environment_lock_sha256": training.get("environment_lock_sha256"),
            "source_revision": None,
            "historical_unknowns": ["git revision not captured; exact source file hashes recorded instead"],
        },
    }
    with (path / SIDECAR).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return record


def validate_checkpoint(path, *, steps=50, expected_task=None, check_versions=True):
    """Fail before deserialization if task, transforms, solver or bytes differ."""
    path = checkpoint_dir(path)
    try:
        record = json.loads((path / SIDECAR).read_text())
        task = record["task"]
        validate_task(task)
        if expected_task is not None and task != expected_task:
            raise ValueError("checkpoint task differs from requested task")
        expected = {
            "format": FORMAT, "task_sha256": task_fingerprint({"task": task}),
            "adapter_id": ADAPTER_ID, "features": task["features"],
            "target_coordinates": task["target_coordinates"],
            "terminal_target": "log(sqrt(252 * terminal_variance) / 100)",
            "sampler": SAMPLER, "standardization": STANDARDIZATION,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"incompatible checkpoint {key}")
        if type(steps) is not int or steps != SAMPLER["steps"]:
            raise ValueError("checkpoint evidence binds exactly 50 Euler steps")
        if set(record["files"]) != {"model.keras", "training.json", "misspecification.npz"}:
            raise ValueError("incomplete checkpoint file bindings")
        for name, digest in record["files"].items():
            file = path / name
            if digest is None and name == "misspecification.npz" and not file.exists():
                continue
            if not isinstance(digest, str) or not file.is_file() or sha256(file) != digest:
                raise ValueError(f"checkpoint hash mismatch: {name}")
        training = json.loads((path / "training.json").read_text())
        if (training.get("window") != task["n_observations"]
                or training.get("targets") != task["target_coordinates"]):
            raise ValueError("training metadata does not match checkpoint task")
        if record["runtime_versions"] != training["versions"]:
            raise ValueError("checkpoint versions differ from training metadata")
        if check_versions and record["runtime_versions"] != runtime_versions():
            raise ValueError("checkpoint requires its recorded BayesFlow/Keras/JAX versions")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"incomplete SVJ checkpoint binding at {path}: {exc}") from exc
    return record


def bind_reference(path, *, checkpoint, seed):
    """Bind a newly written misspecification reference without editing checkpoint."""
    path = Path(path)
    record = validate_checkpoint(checkpoint)
    binding = {"format": "svj_misspecification_reference_v1", "reference_sha256": sha256(path),
               "checkpoint_sha256": record["files"]["model.keras"],
               "task_sha256": record["task_sha256"], "seed": seed,
               "scientific_acceptance": False}
    with path.with_suffix(".json").open("x") as stream:
        json.dump(binding, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return binding


def validate_reference(path, *, checkpoint):
    path = Path(path)
    checkpoint = checkpoint_dir(checkpoint)
    record = validate_checkpoint(checkpoint)
    if path.resolve() == (checkpoint / "misspecification.npz").resolve():
        if record["files"]["misspecification.npz"] is None:
            raise ValueError("checkpoint has no bound misspecification reference")
        return {"reference_sha256": record["files"]["misspecification.npz"],
                "checkpoint_sha256": record["files"]["model.keras"], "task_sha256": record["task_sha256"]}
    binding = json.loads(path.with_suffix(".json").read_text())
    for key, expected in (("format", "svj_misspecification_reference_v1"),
                          ("reference_sha256", sha256(path)),
                          ("checkpoint_sha256", record["files"]["model.keras"]),
                          ("task_sha256", record["task_sha256"])):
        if binding.get(key) != expected:
            raise ValueError(f"incompatible diagnostic reference {key}")
    return binding

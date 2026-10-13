"""Reproducible banks for Francesco's daily SVJ parameter/state task.

Stored returns use the repository's decimal convention; model coordinates and
variance retain their native units. Only returns and the declared time design
are observations. Parameter/state truth is supervised output, never conditioning.
The bank's RNG protocol is distinct from the retained historical online stream.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from ..gbm_bank import _integer, _json, _sha256, _write_json
from . import model

TASK_ID = "svj_daily_parameter_state_v1"
RNG_VERSION = "svj_bank_pcg64_per_example_v1"
SPLITS = {"train": 0, "validation": 1, "test": 2}
STREAMS = {"prior": 0, "initialization": 1, "diffusion": 2, "jump_indicator": 3, "jump_size": 4}
GENERATION_BATCH_SIZE = 256
# Chosen before cluster comparisons, rather than fitted to their differences.
PORTABLE_TOLERANCE = {"rtol": 1e-10, "atol": 1e-12}


def default_task():
    """A fresh, complete description of the supported daily inference task."""
    return {
        "task_id": TASK_ID,
        "model_id": "daily_bernoulli_gaussian_jump_floored_euler_v1",
        "measure": "P",
        "n_observations": model.WINDOW,
        "dt_years": 1 / model.DAYS_PER_YEAR,
        "substeps": 1,
        "observation_unit": "decimal_log_return",
        "parameter_coordinates": list(model.COORDINATE_NAMES),
        "target_coordinates": list(model.TARGET_NAMES),
        "terminal_state": {"name": "terminal_variance", "unit": "percent_squared_per_trading_day"},
        "prior": asdict(model.PRIOR),
        "constants": {"days_per_year": model.DAYS_PER_YEAR, "variance_floor": model.V_MIN,
                      "jump_probability_cap": model.LAMBDA_MAX, "rho_prior_probability_clip": 1e-9},
        "initialization": {"family": "floored_gamma", "shape": "exp(-2*log_cv)",
                           "scale": "theta*exp(2*log_cv)", "floor": model.V_MIN},
        "dynamics": {"return_drift": "additive_daily_log_return_intercept",
                     "variance": "previous_variance_euler_with_positive_floor",
                     "jumps": "one_bernoulli_gaussian_opportunity_per_day"},
        "features": {"channels": ["percent_log_return", "log_squared_percent_return_plus_offset"],
                     "squared_return_offset": 0.01, "dtype": "float32"},
        "reference_sampler": {"method": "euler", "steps": 50},
    }


def validate_task(task):
    """Validate the fixed design and a complete, positive-scale native prior."""
    if not isinstance(task, dict) or set(task) != set(default_task()):
        raise ValueError("unsupported SVJ task fields")
    expected = default_task()
    # Prior parameters can change for an explicitly identified new task; every
    # other convention is fixed. Checkpoint compatibility is stricter still.
    for key in set(expected) - {"prior"}:
        if _json(task[key]) != _json(expected[key]):
            raise ValueError(f"unsupported SVJ task field: {key}")
    prior = task["prior"]
    if not isinstance(prior, dict) or set(prior) != set(expected["prior"]):
        raise ValueError("SVJ prior needs every named Prior field")
    for key, value in prior.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"prior {key} must be a finite scalar")
        if key not in {"drift_loc", "mean_jump_loc"} and value <= 0:
            raise ValueError(f"prior {key} must be positive")
    _json(task)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"task", "root_seed", "splits"}:
        raise ValueError("config needs exactly task, root_seed and splits")
    validate_task(config["task"])
    if _integer(config["root_seed"], "root_seed") >= 2**64:
        raise ValueError("root_seed must fit an unsigned 64-bit integer")
    if not isinstance(config["splits"], dict) or set(config["splits"]) != set(SPLITS):
        raise ValueError("exactly train, validation and test counts are required")
    for split, count in config["splits"].items():
        if _integer(count, split, minimum=1) >= 2**64:
            raise ValueError("split counts must fit unsigned 64-bit integers")
    _json(config)


def task_fingerprint(config):
    """Statistical task identity; root seed, split sizes and RNG are separate."""
    return hashlib.sha256(_json(config["task"]).encode()).hexdigest()


def _rngs(config, split, example_id):
    digest = task_fingerprint(config)
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    seed = int(config["root_seed"])
    # The fixed domain tag versions this protocol independently of legacy RNG.
    identity = [0x53564A31, seed & 0xffffffff, seed >> 32, *words, SPLITS[split],
                example_id & 0xffffffff, example_id >> 32]
    return {name: np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, code])))
            for name, code in STREAMS.items()}


def generate_examples(config, split, example_ids):
    """Produce split-qualified IDs stably under order, count and chunk changes.

    Prior, Gamma initialization, diffusion, jump indicators and sizes have
    separate per-example streams. The time loop is batched and retains no
    variance trajectory in the bank. Failures name their IDs and never redraw.
    """
    validate_config(config)
    if split not in SPLITS:
        raise ValueError("unknown split")
    ids = [_integer(value, "example ID") for value in example_ids]
    if not ids or len(ids) != len(set(ids)) or max(ids) >= 2**64:
        raise ValueError("example IDs must be nonempty, unique unsigned 64-bit integers")
    task = config["task"]
    prior = model.Prior(**task["prior"])
    n_days = task["n_observations"]
    observations = np.empty((len(ids), n_days, 1), dtype=np.float64)
    parameters = np.empty((len(ids), model.N_PARAMS), dtype=np.float64)
    terminal = np.empty((len(ids), 1), dtype=np.float64)
    for start in range(0, len(ids), GENERATION_BATCH_SIZE):
        stop = min(start + GENERATION_BATCH_SIZE, len(ids))
        initial = np.empty(stop - start, dtype=np.float64)
        eps1, eps2, uniforms, normals = [np.empty((n_days, stop - start)) for _ in range(4)]
        for column, example_id in enumerate(ids[start:stop]):
            try:
                streams = _rngs(config, split, example_id)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    coordinates = model.draw_coordinates(1, streams["prior"], prior)
                    parameters[start + column] = coordinates[0]
                    initial[column] = model.initial_variance(coordinates, streams["initialization"])[0]
                    eps1[:, column] = streams["diffusion"].standard_normal(n_days)
                    eps2[:, column] = streams["diffusion"].standard_normal(n_days)
                    uniforms[:, column] = streams["jump_indicator"].random(n_days)
                    normals[:, column] = streams["jump_size"].standard_normal(n_days)
            except (ValueError, FloatingPointError, OverflowError) as error:
                raise ValueError(f"failed example {split}:{example_id}: {error}") from error
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = model.daily_model_from_innovations(
                    model.model_params(parameters[start:stop]), initial, eps1, eps2, uniforms, normals
                )
            observations[start:stop, :, 0] = result["returns"] / 100
            terminal[start:stop] = result["terminal_state"]
        except (ValueError, FloatingPointError, OverflowError) as error:
            failed = [f"{split}:{value}" for value in ids[start:stop]]
            raise ValueError(f"failed example group {failed}: {error}") from error
    arrays = {"observations": observations, "parameters": parameters, "terminal_state": terminal,
              "dt_years": np.full(n_days, task["dt_years"], dtype=np.float64),
              "example_id": np.asarray([f"{split}:{value}" for value in ids])}
    _validate_arrays(arrays, task, split, ids)
    return arrays


def _validate_arrays(arrays, task, split, ids):
    shapes = {"observations": (len(ids), task["n_observations"], 1),
              "parameters": (len(ids), model.N_PARAMS), "terminal_state": (len(ids), 1),
              "dt_years": (task["n_observations"],), "example_id": (len(ids),)}
    if set(arrays) != set(shapes):
        raise ValueError(f"{split}: unexpected arrays; latent or future truth is not an observation")
    for key, shape in shapes.items():
        values = arrays[key]
        if values.shape != shape:
            raise ValueError(f"{split}: invalid {key} shape")
        if key == "example_id":
            if values.dtype.kind != "U":
                raise ValueError(f"{split}: example IDs must be Unicode strings")
        elif values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError(f"{split}: invalid {key} values/dtype")
    if not np.array_equal(arrays["example_id"], [f"{split}:{value}" for value in ids]):
        raise ValueError(f"{split}: duplicate, missing or misassigned example IDs")
    if np.any(arrays["terminal_state"] < model.V_MIN):
        raise ValueError(f"{split}: terminal variance below model floor")
    if not np.array_equal(arrays["dt_years"], np.full(task["n_observations"], task["dt_years"])):
        raise ValueError(f"{split}: time grid differs from task")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        params = model.model_params(arrays["parameters"])
    if (not np.isfinite(params).all() or np.any(params[:, [1, 2, 3, 7]] <= 0)
            or np.any(np.abs(params[:, 4]) >= 1)
            or np.any((params[:, 5] < 0) | (params[:, 5] > model.LAMBDA_MAX))):
        raise ValueError(f"{split}: parameter coordinates outside numerically supported model domain")


def _semantics(config):
    """Fields whose roles/units are validated, rather than trusting free text."""
    return {
        "format": "svj_bank_v1", "data_origin": "synthetic", "evidence_scope": "S", "measure": "P",
        "task_fingerprint": task_fingerprint(config),
        "configuration_sha256": hashlib.sha256(_json(config).encode()).hexdigest(),
        "conditioning_cutoff_years": config["task"]["n_observations"] * config["task"]["dt_years"],
        "roles": {"observations": "observed decimal log returns [batch,time,1]",
                  "parameters": "native transformed parameter truth [batch,8]; supervised target only",
                  "terminal_state": "terminal variance truth [batch,1], percent squared per trading day; target only",
                  "dt_years": "known observation context [time]; other fixed context in task",
                  "example_id": "independent base/group identity; never conditioning"},
        "truth_available": [*model.COORDINATE_NAMES, "terminal_variance"],
        "future_outcomes_in_conditioning": False,
        "rng": {"version": RNG_VERSION, "bit_generator": "PCG64", "streams": STREAMS,
                "identity": "protocol/task_fingerprint/root_seed/split/local_example_id/purpose"},
        "split_codes": SPLITS,
        "portable_tolerance": PORTABLE_TOLERANCE,
    }


def source_hashes():
    paths = {"svj/model.py": Path(model.__file__), "svj/bank.py": Path(__file__),
             "gbm_bank.py": Path(__file__).parents[1] / "gbm_bank.py"}
    return {name: _sha256(path) for name, path in paths.items()}


def environment_versions():
    return {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()}


def write_bank(path, config):
    """Write a new directory, atomic split files and the manifest last.

    Failure leaves failure.json with the original IDs/error. Existing banks are
    never overwritten and failed examples are never silently dropped/redrawn.
    """
    validate_config(config)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    manifest = {**_semantics(config), "configuration": config,
                "purpose": "joint parameter/terminal-state inference for the daily SVJ P model",
                "generation_batch_size": GENERATION_BATCH_SIZE,
                "versions": environment_versions(), "source_sha256": source_hashes(), "splits": {}}
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
    """Verify completion, hashes, roles, task, shapes, support and split IDs."""
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except FileNotFoundError as error:
        raise ValueError("incomplete bank: no completion manifest") from error
    if (path / "failure.json").exists():
        raise ValueError("incomplete bank: failure record present")
    config = manifest.get("configuration")
    validate_config(config)
    for key, expected in _semantics(config).items():
        if manifest.get(key) != expected:
            raise ValueError(f"incompatible bank manifest field: {key}")
    if (set(manifest.get("splits", {})) != set(SPLITS)
            or set(manifest.get("source_sha256", {})) != set(source_hashes())
            or set(manifest.get("versions", {})) != set(environment_versions())):
        raise ValueError("incomplete bank identity or provenance")
    for digest in manifest["source_sha256"].values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source SHA256")
    task, result = config["task"], {}
    for split, count in config["splits"].items():
        record = manifest["splits"][split]
        if (set(record) != {"file", "examples", "sha256"}
                or record["file"] != f"{split}.npz" or record["examples"] != count):
            raise ValueError(f"{split}: inconsistent manifest")
        source = path / record["file"]
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"{split}: checksum mismatch")
        with np.load(source, allow_pickle=False) as saved:
            arrays = dict(saved)
        _validate_arrays(arrays, task, split, range(count))
        result[split] = arrays
    return manifest, result

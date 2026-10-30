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
from sbi4finance.rough_bergomi import rough_driver


SPLITS = {"train": 0, "validation": 1, "test": 2}
MODELS = ("heston", "rough-known", "rough-joint")
FORMAT = "physical_return_path_bank_v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) < 2**64):
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**64)")
    return int(value)


def default_task(name):
    """Return a fresh fully specified P task; landmarks do not restrict inputs."""
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    rough = name != "heston"
    task = {
        "task_id": f"physical_{name.replace('-', '_')}_paths_v1",
        "model": name,
        "measure": "P",
        "observation": "decimal_log_returns_only",
        "n_observations": 1008 if rough else 1260,
        "dt_years": 1 / (252 * 4) if rough else 1 / 252,
        "min_history": 1,
        "history_length_range": [1, 1008 if rough else 1260],
        "prefix_lengths": [63, 126, 252, 504, 1008 if rough else 1260],
        "prefix_lengths_role": "validation_landmarks",
        "fixed": {"drift": 0.05},
    }
    if not rough:
        task.update(parameter_names=["kappa", "long_variance", "vol_of_vol", "rho"],
                    prior_bounds=[[0.25, 5.0], [0.01, 0.09], [0.05, 0.6], [-0.95, -0.05]],
                    prior_types=["log_uniform", "log_uniform", "uniform", "uniform"],
                    numerics={"scheme": "full_truncation_euler_signed", "substeps": 1})
        task["fixed"]["initial_variance"] = 0.04
    else:
        task.update(parameter_names=["hurst", "eta", "rho"],
                    prior_bounds=[[0.03, 0.4], [0.2, 2.5], [-0.95, -0.05]],
                    prior_types=["uniform", "uniform", "uniform"],
                    numerics={"scheme": "log_euler", "driver_method": "hybrid",
                              "variance_compensator": "discrete_hybrid_variance",
                              "initialization": "zero_volterra_driver_at_origin"})
        if name == "rough-known":
            task["fixed"]["variance_level"] = 0.04
        else:
            task["parameter_names"] += ["short_variance", "long_variance", "decay"]
            task["prior_bounds"] += [[0.01, 0.09], [0.01, 0.09], [0.25, 8.0]]
            task["prior_types"] += ["log_uniform"] * 3
    validate_task(task)
    return task


def validate_task(task):
    """Validate the supported discrete laws without changing their priors."""
    if not isinstance(task, dict) or task.get("model") not in MODELS:
        raise ValueError("unsupported path model")
    if task.get("measure") != "P" or task.get("observation") != "decimal_log_returns_only":
        raise ValueError("path tasks require P decimal log-return observations")
    n = _integer(task.get("n_observations"), "n_observations", 1)
    minimum = _integer(task.get("min_history"), "min_history", 1)
    if minimum > n or task.get("history_length_range") != [minimum, n]:
        raise ValueError("history length range must match min_history and n_observations")
    landmarks = task.get("prefix_lengths")
    if (not isinstance(landmarks, list) or not landmarks
            or any(_integer(x, "prefix length", 1) < minimum or x > n for x in landmarks)
            or landmarks != sorted(set(landmarks))
            or task.get("prefix_lengths_role") != "validation_landmarks"):
        raise ValueError("prefix_lengths must be ordered unique validation landmarks in the supported range")
    dt = task.get("dt_years")
    if isinstance(dt, bool) or not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_years must be finite and positive")
    rough = task["model"] != "heston"
    expected = (["hurst", "eta", "rho"] if rough
                else ["kappa", "long_variance", "vol_of_vol", "rho"])
    if task["model"] == "rough-joint":
        expected += ["short_variance", "long_variance", "decay"]
    if task.get("parameter_names") != expected:
        raise ValueError("parameter names/order differs from the path task contract")
    bounds = np.asarray(task.get("prior_bounds"), dtype=np.float64)
    types = task.get("prior_types")
    if (bounds.shape != (len(expected), 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1]) or not isinstance(types, list)
            or len(types) != len(expected) or set(types) - {"uniform", "log_uniform"}):
        raise ValueError("invalid independent prior bounds/types")
    for name, (low, high), kind in zip(expected, bounds, types, strict=True):
        if ((name == "rho" and not -1 < low < high < 1)
                or (name != "rho" and low <= 0)
                or (name == "hurst" and high >= 0.5)
                or (kind == "log_uniform" and low <= 0)):
            raise ValueError(f"invalid prior support for {name}")
    fixed = task.get("fixed")
    fixed_keys = ({"drift", "initial_variance"} if not rough else
                  {"drift", "variance_level"} if task["model"] == "rough-known" else {"drift"})
    if not isinstance(fixed, dict) or set(fixed) != fixed_keys:
        raise ValueError("fixed parameter contract differs")
    if any(isinstance(x, bool) or not np.isscalar(x) or not np.isfinite(x) for x in fixed.values()):
        raise ValueError("fixed parameters must be finite scalars")
    if any(fixed[k] <= 0 for k in fixed_keys - {"drift"}):
        raise ValueError("fixed variance must be positive")
    numerics = task.get("numerics")
    if not rough:
        if (not isinstance(numerics, dict) or set(numerics) != {"scheme", "substeps"}
                or numerics["scheme"] != "full_truncation_euler_signed"):
            raise ValueError("unsupported Heston numerical law")
        _integer(numerics["substeps"], "substeps", 1)
    elif numerics != {"scheme": "log_euler", "driver_method": "hybrid",
                      "variance_compensator": "discrete_hybrid_variance",
                      "initialization": "zero_volterra_driver_at_origin"}:
        raise ValueError("unsupported rough numerical law/compensator/initialization")
    _json(task)


def task_fingerprint(task):
    validate_task(task)
    return hashlib.sha256(_json(task).encode()).hexdigest()


def _rng(task_digest, split, identifier, seed, domain):
    words = [int(task_digest[i:i + 8], 16) for i in range(0, 64, 8)]
    identity = [0x50415448, *words, seed & 0xffffffff, seed >> 32,
                SPLITS[split], identifier & 0xffffffff, identifier >> 32, domain]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(identity)))


def generate(task, split, start, count, seed=30102026):
    """Generate a chunk independently of chunk size, order and other splits.

    Parameters and innovations have separate random domains. No failed path is
    redrawn. Only returns, parameter labels and trajectory identities leave this
    boundary: latent simulator states cannot become inference conditions.
    """
    digest = task_fingerprint(task)
    if split not in SPLITS:
        raise ValueError("unknown split")
    start, count, seed = (_integer(start, "start"), _integer(count, "count", 1),
                          _integer(seed, "seed"))
    if start + count > 2**64:
        raise ValueError("trajectory IDs exceed uint64 range")
    n = task["n_observations"]
    bounds = np.asarray(task["prior_bounds"])
    parameters = np.empty((count, len(bounds)), dtype=np.float64)
    for row, identifier in enumerate(range(start, start + count)):
        uniform = _rng(digest, split, identifier, seed, 0).random(len(bounds))
        for col, ((low, high), kind) in enumerate(zip(bounds, task["prior_types"], strict=True)):
            parameters[row, col] = (np.exp(np.log(low) + uniform[col] * np.log(high / low))
                                    if kind == "log_uniform" else low + uniform[col] * (high - low))
    if task["model"] == "heston":
        substeps = task["numerics"]["substeps"]
        noise = np.empty((count, n, substeps, 2), dtype=np.float64)
        for row, identifier in enumerate(range(start, start + count)):
            noise[row] = _rng(digest, split, identifier, seed, 1).standard_normal((n, substeps, 2))
        inputs = {"brownian": noise, "jump_counts": np.zeros((count, n), dtype=np.int64),
                  "jump_normals": np.zeros((count, n))}
        p = dict(zip(task["parameter_names"], parameters.T, strict=True))
        law = {"drift": task["fixed"]["drift"], "kappa": p["kappa"],
               "theta": p["long_variance"], "vol_of_vol": p["vol_of_vol"], "rho": p["rho"]}
        observations = heston.simulate(law, np.full(n, task["dt_years"]),
                                       task["fixed"]["initial_variance"], inputs)["log_returns"]
    else:
        observations = np.empty((count, n, 1), dtype=np.float64)
        dt = task["dt_years"]
        times = dt * np.arange(n + 1)
        for row, identifier in enumerate(range(start, start + count)):
            normals = _rng(digest, split, identifier, seed, 1).standard_normal((1, n, 3))
            p = dict(zip(task["parameter_names"], parameters[row], strict=True))
            driver, discrete_variance = rough_driver(p["hurst"], dt, normals, method="hybrid")
            driver = np.r_[0.0, driver[0]]
            discrete_variance = np.r_[0.0, discrete_variance]
            xi = (np.full(n + 1, task["fixed"]["variance_level"])
                  if task["model"] == "rough-known" else
                  p["long_variance"] + (p["short_variance"] - p["long_variance"]) * np.exp(-p["decay"] * times))
            # Explicit synthetic P law: E_P[v_t]=xi(t), with declared hybrid
            # Gaussian compensator; mu is physical proportional price drift.
            with np.errstate(over="raise", invalid="raise"):
                variance = xi * np.exp(p["eta"] * driver - 0.5 * p["eta"] ** 2 * discrete_variance)
                left = variance[:-1]
                observations[row, :, 0] = ((task["fixed"]["drift"] - 0.5 * left) * dt
                    + np.sqrt(left * dt) * (p["rho"] * normals[0, :, 0]
                                             + np.sqrt(1 - p["rho"] ** 2) * normals[0, :, 2]))
    if not np.isfinite(observations).all() or not np.isfinite(parameters).all():
        raise FloatingPointError(f"nonfinite path in {split}:{start}-{start + count}; no redraw")
    identities = np.asarray([f"{digest[:16]}:{seed}:{split}:{i}" for i in range(start, start + count)])
    return {"observations": observations, "parameters": parameters, "trajectory_id": identities}


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
                    "source_sha256": {str(p.relative_to(package)): _sha256(p) for p in
                        (Path(__file__), package / "heston.py", package / "rough_bergomi.py", package / "bergomi.py")}},
                "timing_seconds": {}}
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
                manifest["splits"][split].append({"path": str(path.relative_to(out)),
                    "sha256": _sha256(path), "count": len(arrays["parameters"]), "start": start})
            manifest["timing_seconds"][split] = perf_counter() - split_started
        manifest["timing_seconds"]["total"] = perf_counter() - started
        for split in SPLITS:
            # Chunk verification avoids materializing the whole bank here.
            for record in manifest["splits"][split]:
                _load_part(out, manifest, split, record)
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

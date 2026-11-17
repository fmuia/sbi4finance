"""Physical-measure task definitions and small reproducible training-pair simulations.

This is the generative law used by the lab's sharded bank writer. Each pair
contains parameters and observed returns only; latent variance stays hidden.
Use backend="numpy" for a small tutorial example; the lab uses optional Numba.
"""

import hashlib
import json

import numpy as np

from ..heston_options import simulator as heston
from ..rough_options.simulator import rough_driver


SPLITS = {"train": 0, "validation": 1, "test": 2}


MODELS = ("heston", "rough-known", "rough-joint")


TEACHING_TASK_IDS = {name: f"physical_{name.replace('-', '_')}_daily_drift_paths_v2"
                     for name in MODELS}


TEACHING_LANDMARKS = (21, 63, 126, 252, 504, 756, 1260)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
                    numerics={"scheme": "full_truncation_euler_signed", "substeps": 8})
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


def teaching_task(name, *, observations=1260):
    """Harmonized daily, five-year teaching task with inferred physical drift.

    These are new task identities. Legacy defaults retain their existing time
    designs, fixed drifts, priors and random streams. Rough paths use four
    internal steps per observed day, with daily log returns summed only after
    simulating the entire zero-origin Volterra path.
    """
    observations = _integer(observations, "observations", 1)
    task = default_task(name)
    task.update(task_id=TEACHING_TASK_IDS[name], task_version=2,
                n_observations=observations, dt_years=1 / 252,
                history_length_range=[1, observations],
                prefix_lengths=sorted(set([n for n in TEACHING_LANDMARKS if n <= observations]
                                          + [observations])))
    task["parameter_names"].append("drift")
    task["prior_bounds"].append([-.10, .20])
    task["prior_types"].append("uniform")
    task["fixed"].pop("drift")
    if name != "heston":
        task["numerics"]["observation_substeps"] = 4
    validate_task(task)
    return task


def validate_task(task):
    """Validate the supported discrete laws without changing their priors."""
    if not isinstance(task, dict) or task.get("model") not in MODELS:
        raise ValueError("unsupported path model")
    if task.get("measure") != "P" or task.get("observation") != "decimal_log_returns_only":
        raise ValueError("path tasks require P decimal log-return observations")
    teaching = task.get("task_id") == TEACHING_TASK_IDS[task["model"]]
    if teaching and task.get("task_version") != 2:
        raise ValueError("daily drift teaching tasks require explicit task_version 2")
    if task.get("task_version") == 2 and not teaching:
        raise ValueError("task_version 2 requires its named daily drift teaching identity")
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
    if teaching and dt != 1 / 252:
        raise ValueError("harmonized teaching observations must use daily dt_years=1/252")
    rough = task["model"] != "heston"
    expected = (["hurst", "eta", "rho"] if rough
                else ["kappa", "long_variance", "vol_of_vol", "rho"])
    if task["model"] == "rough-joint":
        expected += ["short_variance", "long_variance", "decay"]
    if teaching:
        expected += ["drift"]
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
                or (name not in ("rho", "drift") and low <= 0)
                or (name == "drift" and kind != "uniform")
                or (name == "hurst" and high >= 0.5)
                or (kind == "log_uniform" and low <= 0)):
            raise ValueError(f"invalid prior support for {name}")
    fixed = task.get("fixed")
    fixed_keys = ({"drift", "initial_variance"} if not rough else
                  {"drift", "variance_level"} if task["model"] == "rough-known" else {"drift"})
    if teaching:
        fixed_keys.remove("drift")
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
        if teaching and numerics["substeps"] != 8:
            raise ValueError("daily teaching Heston requires eight internal steps per day")
    else:
        expected_numerics = {"scheme": "log_euler", "driver_method": "hybrid",
                             "variance_compensator": "discrete_hybrid_variance",
                             "initialization": "zero_volterra_driver_at_origin"}
        if teaching:
            expected_numerics["observation_substeps"] = 4
        if numerics != expected_numerics:
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


def generate(task, split, start, count, seed=30102026, *, backend="numba"):
    """Generate a chunk independently of chunk size, order and other splits.

    Parameters and innovations have separate random domains. No failed path is
    redrawn. Only returns, parameter labels and trajectory identities leave this
    boundary: latent simulator states cannot become inference conditions.
    """
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be numpy or numba")
    digest = task_fingerprint(task)
    if split not in SPLITS:
        raise ValueError("unknown split")
    start, count, seed = (_integer(start, "start"), _integer(count, "count", 1),
                          _integer(seed, "seed"))
    if start + count > 2**64:
        raise ValueError("trajectory IDs exceed uint64 range")
    bounds = np.asarray(task["prior_bounds"])
    parameters = np.empty((count, len(bounds)), dtype=np.float64)
    for row, identifier in enumerate(range(start, start + count)):
        uniform = _rng(digest, split, identifier, seed, 0).random(len(bounds))
        for col, ((low, high), kind) in enumerate(zip(bounds, task["prior_types"], strict=True)):
            parameters[row, col] = (np.exp(np.log(low) + uniform[col] * np.log(high / low))
                                    if kind == "log_uniform" else low + uniform[col] * (high - low))
    observations = _simulate_observations(task, parameters, digest=digest, split=split,
                                           start=start, seed=seed, backend=backend)
    if not np.isfinite(observations).all() or not np.isfinite(parameters).all():
        raise FloatingPointError(f"nonfinite path in {split}:{start}-{start + count}; no redraw")
    identities = np.asarray([f"{digest[:16]}:{seed}:{split}:{i}" for i in range(start, start + count)])
    return {"observations": observations, "parameters": parameters, "trajectory_id": identities}


def _simulate_observations(task, parameters, *, digest, split, start, seed, backend, return_details=False):
    """One numerical implementation for prior-pair generation and fixed-parameter illustrations."""
    count, n = len(parameters), task["n_observations"]
    if task["model"] == "heston":
        substeps = task["numerics"]["substeps"]
        noise = np.empty((count, n, substeps, 2), dtype=np.float64)
        for row, identifier in enumerate(range(start, start + count)):
            noise[row] = _rng(digest, split, identifier, seed, 1).standard_normal((n, substeps, 2))
        inputs = {"brownian": noise, "jump_counts": np.zeros((count, n), dtype=np.int64),
                  "jump_normals": np.zeros((count, n))}
        p = dict(zip(task["parameter_names"], parameters.T, strict=True))
        law = {"drift": p["drift"] if "drift" in p else task["fixed"]["drift"], "kappa": p["kappa"],
               "theta": p["long_variance"], "vol_of_vol": p["vol_of_vol"], "rho": p["rho"]}
        result = heston.simulate(law, np.full(n, task["dt_years"]),
                                task["fixed"]["initial_variance"], inputs, backend=backend)
        observations = result["log_returns"]
        if return_details:
            signed = result["variance_state"][:, :, 0].copy()
            latent = {"variance_state": signed, "variance": np.maximum(signed, 0.)}
    else:
        observations = np.empty((count, n, 1), dtype=np.float64)
        observation_substeps = task["numerics"].get("observation_substeps", 1)
        internal_count = n * observation_substeps
        dt = task["dt_years"] / observation_substeps
        times = dt * np.arange(internal_count + 1)
        if return_details:
            latent = {name: np.empty((count, n+1)) for name in
                      ("variance", "driver", "driver_variance", "mean_variance", "median_variance")}
        for row, identifier in enumerate(range(start, start + count)):
            normals = _rng(digest, split, identifier, seed, 1).standard_normal((1, internal_count, 3))
            p = dict(zip(task["parameter_names"], parameters[row], strict=True))
            driver, discrete_variance = rough_driver(p["hurst"], dt, normals, method="hybrid")
            driver = np.r_[0.0, driver[0]]
            discrete_variance = np.r_[0.0, discrete_variance]
            xi = (np.full(internal_count + 1, task["fixed"]["variance_level"])
                  if task["model"] == "rough-known" else
                  p["long_variance"] + (p["short_variance"] - p["long_variance"]) * np.exp(-p["decay"] * times))
            # Explicit synthetic P law: E_P[v_t]=xi(t), with declared hybrid
            # Gaussian compensator; mu is physical proportional price drift.
            with np.errstate(over="raise", invalid="raise"):
                variance = xi * np.exp(p["eta"] * driver - 0.5 * p["eta"] ** 2 * discrete_variance)
                left = variance[:-1]
                drift = p["drift"] if "drift" in p else task["fixed"]["drift"]
                fine_returns = ((drift - 0.5 * left) * dt
                    + np.sqrt(left * dt) * (p["rho"] * normals[0, :, 0]
                                             + np.sqrt(1 - p["rho"] ** 2) * normals[0, :, 2]))
                observations[row, :, 0] = (fine_returns if observation_substeps == 1 else
                                           fine_returns.reshape(n, observation_substeps).sum(axis=1))
                if return_details:
                    # These are the actual states at observed-day boundaries,
                    # including the original starting point. No interpolation,
                    # new innovations or volatility reconstruction is involved.
                    daily = slice(None, None, observation_substeps)
                    latent["variance"][row] = variance[daily]
                    latent["driver"][row] = driver[daily]
                    latent["driver_variance"][row] = discrete_variance[daily]
                    latent["mean_variance"][row] = xi[daily]
                    latent["median_variance"][row] = xi[daily] * np.exp(-.5*p["eta"]**2*discrete_variance[daily])
    if not np.isfinite(observations).all():
        raise FloatingPointError("nonfinite simulated returns; no redraw")
    if not return_details:
        return observations
    if not all(np.isfinite(values).all() for values in latent.values()):
        raise FloatingPointError("nonfinite simulated latent states; no redraw")
    return {"observations": observations, "latent": latent}


def simulate_parameters(task, parameters, seed=20261108, *, backend="numpy", return_details=False):
    """Simulate chosen parameter rows with fixed innovations from the canonical P law.

    By default returns only daily returns, with shape [rows, observations, 1].
    ``return_details=True`` additionally exposes boundary latent truth for a
    simulator illustration, never as observed inference conditions. This
    session-only forward illustration assigns no training-bank trajectory IDs.
    Reusing a task and seed reuses innovations when its chosen parameters change.
    """
    digest = task_fingerprint(task)
    seed = _integer(seed, "seed")
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be numpy or numba")
    if not isinstance(return_details, (bool, np.bool_)):
        raise ValueError("return_details must be a boolean")
    if np.iscomplexobj(parameters) or np.ma.isMaskedArray(parameters):
        raise ValueError("real unmasked parameter rows are required")
    parameters = np.asarray(parameters, dtype=np.float64)
    bounds = np.asarray(task["prior_bounds"], dtype=float)
    if (parameters.ndim != 2 or not 1 <= len(parameters) <= 32 or parameters.shape[1] != len(bounds)
            or not np.isfinite(parameters).all() or np.any(parameters < bounds[:, 0])
            or np.any(parameters > bounds[:, 1])):
        raise ValueError("one to 32 finite complete parameter rows within declared prior support are required")
    return _simulate_observations(task, parameters, digest=digest, split="train",
                                   start=0, seed=seed, backend=backend, return_details=return_details)


def replay_details(task, split, start, count, seed=30102026, *, backend="numpy"):
    """Recover the actual latent truth of the same canonical identified histories.

    Parameters and innovations use their original task/split/case RNG domains.
    Capturing details must preserve observations bitwise on this runtime. Saved
    historical arrays may differ by numerical-library rounding; their separate
    asset producer verifies equivalence before publishing a paired latent view.
    """
    pairs = generate(task, split, start, count, seed=seed, backend=backend)
    details = _simulate_observations(task, pairs["parameters"], digest=task_fingerprint(task), split=split,
                                      start=start, seed=seed, backend=backend, return_details=True)
    if not np.array_equal(details["observations"], pairs["observations"]):
        raise RuntimeError("latent capture changed the canonical returns; no paired states can be published")
    return {**pairs, "latent": details["latent"]}

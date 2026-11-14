"""One frozen synthetic observation shared by every posterior and check view.

Preserved examples keep their original cached posteriors. A newly simulated
history has no cached posterior: only a matching explicit native query can
provide one. Parameters and latent paths are display/evaluation truth, never
neural conditions. No model is loaded or sampled by this module.
"""

from collections.abc import Mapping
import hashlib

import numpy as np

from . import inference, latent_assets, lesson, simulation


FORMAT = "trajectory_observation_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _task(task):
    simulation.validate_task(task)
    _require(task == simulation.teaching_task(task["model"]),
             "observations require the unchanged five-year teaching task and prior")


def _array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _real(values, shape, name):
    _require(not np.iscomplexobj(values) and not np.ma.isMaskedArray(values),
             name + " must be real unmasked values")
    values = np.asarray(values)
    _require(values.shape == shape and values.dtype.kind in "fiu" and np.isfinite(values).all(),
             name + " must have finite shape " + str(shape))
    result = np.asarray(values, dtype="<f8").copy()
    result.setflags(write=False)
    return result


def _record(task, returns, truth, identifier, source, case, seed, paired):
    length, dimensions = task["n_observations"], len(task["parameter_names"])
    returns = _real(returns, (length, 1), "observed history")
    truth = _real(truth, (dimensions,), "generating parameters")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(truth >= bounds[:, 0]) and np.all(truth <= bounds[:, 1]),
             "generating parameters must lie within the declared training bounds")
    # Own independent read-only arrays; an unrelated simulator display cannot
    # mutate the selected observation or its paired truth through aliases.
    paired = dict(paired)
    for key, value in paired.items():
        if isinstance(value, np.ndarray):
            _require(not np.iscomplexobj(value) and not np.ma.isMaskedArray(value)
                     and value.dtype.kind in "fiu" and np.isfinite(value).all(),
                     "paired " + key + " must contain finite real values")
            paired[key] = value.copy()
            paired[key].setflags(write=False)
    result = {"format": FORMAT, "task_sha256": simulation.task_fingerprint(task),
              "returns": returns, "truth": truth, "id": identifier, "source": source,
              "case": case, "seed": seed, "paired_history": paired,
              "history_sha256": _array_sha(returns), "parameter_sha256": _array_sha(truth)}
    _validate(task, result)
    return result


def _validate(task, history):
    _task(task)
    _require(isinstance(history, Mapping) and history.get("format") == FORMAT
             and history.get("task_sha256") == simulation.task_fingerprint(task)
             and history.get("source") in ("saved", "custom")
             and isinstance(history.get("id"), str) and bool(history["id"]),
             "selected history belongs to a different task or observation source")
    length, dimensions = task["n_observations"], len(task["parameter_names"])
    returns = _real(history.get("returns"), (length, 1), "observed history")
    truth = _real(history.get("truth"), (dimensions,), "generating parameters")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(truth >= bounds[:, 0]) and np.all(truth <= bounds[:, 1])
             and history.get("history_sha256") == _array_sha(returns)
             and history.get("parameter_sha256") == _array_sha(truth),
             "selected history bytes or generating parameters changed")
    paired = history.get("paired_history", {})
    _require(np.array_equal(paired.get("returns"), returns[:, 0])
             and np.array_equal(paired.get("parameters"), truth)
             and paired.get("parameter_names") == task["parameter_names"],
             "displayed latent truth must stay paired with the selected observation")
    variance = _real(paired.get("variance"), (length+1,), "paired boundary variance")
    _require(np.all(variance >= 0), "effective paired variance must be nonnegative")
    if history["source"] == "saved":
        simulation._integer(history.get("case"), "saved history case")
    else:
        _require(history.get("case") is None, "a custom history cannot be relabelled as a saved case")
        simulation._integer(history.get("seed"), "custom history seed")
        expected = _custom_id(task, history["seed"], history["history_sha256"], history["parameter_sha256"])
        _require(history["id"] == expected, "custom history identity differs from its parameters, seed or bytes")


def _custom_id(task, seed, observed_sha, parameter_sha):
    identity = {"task_sha256": simulation.task_fingerprint(task), "seed": seed,
                "history_sha256": observed_sha, "parameter_sha256": parameter_sha}
    return "custom:" + inference.fingerprint(identity)


def saved_history(task, arrays, case):
    """Bind one original asset-validated history and its actual paired latent path."""
    _task(task)
    case = simulation._integer(case, "saved history case")
    paired = latent_assets.load_history(task, arrays, case)
    return _record(task, arrays["observations"][case], arrays["truth"][case],
                   str(arrays["trajectory_id"][case]), "saved", case, None, paired)


def custom_history(task, theta, seed):
    """Generate one full-horizon history at chosen supported parameters, explicitly.

    ``theta`` is one physical row in task order, or a mapping with exactly the
    declared parameter names. The unchanged full task is simulated before any
    prefix is revealed. Reusing parameters and seed reproduces the same history.
    """
    _task(task)
    seed = simulation._integer(seed, "custom history seed")
    if isinstance(theta, Mapping):
        _require(set(theta) == set(task["parameter_names"]), "provide exactly the model's parameters")
        theta = [theta[name] for name in task["parameter_names"]]
    theta = _real(theta, (len(task["parameter_names"]),), "chosen parameters")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]),
             "chosen parameters must lie within the declared training bounds")
    generated = simulation.simulate_parameters(task, theta[None], seed=seed,
                                                backend="numpy", return_details=True)
    returns = generated["observations"][0]
    latent = {name: values[0] for name, values in generated["latent"].items()}
    days = np.arange(task["n_observations"]+1)
    paired = {"returns": returns[:, 0], "parameters": theta, "parameter_names": list(task["parameter_names"]),
              "days": days, "time_years": days*task["dt_years"], **latent,
              "variance_state": latent.get("variance_state", latent["variance"]),
              **{name: latent.get(name) for name in ("driver", "driver_variance", "mean_variance", "median_variance")},
              "metadata": {"source": "actual chosen-parameter simulation truth, not a state estimate",
                           "inference_input": False, "state_estimate": False, "seed": seed}}
    identifier = _custom_id(task, seed, _array_sha(returns), _array_sha(theta))
    return _record(task, returns, theta, identifier, "custom", None, seed, paired)


def selection_key(task, history, length):
    """Bind a query to the task, source, frozen history bytes and exact prefix."""
    _validate(task, history)
    length = simulation._integer(length, "observed days", minimum=1)
    _require(length <= task["n_observations"], "observed prefix exceeds the selected history")
    return (history["task_sha256"], history["source"], history["id"], history["history_sha256"], length)


def _draws(task, draws):
    _require(not np.iscomplexobj(draws) and not np.ma.isMaskedArray(draws),
             "fresh posterior draws must be real unmasked paired rows")
    values = np.asarray(draws)
    _require(values.ndim == 2 and len(values) >= 2 and values.shape[1] == len(task["parameter_names"])
             and values.dtype.kind in "fiu" and np.isfinite(values).all(), "invalid fresh posterior draw axes or values")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
             "fresh posterior draws lie outside the declared prior support")
    return np.asarray(values, dtype=np.float64)


def posterior_view(task, arrays, history, length, live=None):
    """Use matching native draws, or a saved fallback only for an original case.

    Native checkpoint/task verification remains the responsibility of
    ``lesson.load_live`` before creating ``live``. Custom observations return
    ``None`` until that explicit query succeeds; they never borrow cached draws.
    """
    key = selection_key(task, history, length)
    _require(live is None or isinstance(live, Mapping), "fresh posterior result must be a mapping or None")
    matching = live is not None and live.get("selection") == key
    draws = _draws(task, live.get("draws")) if matching else None
    if history["source"] == "custom":
        if draws is None:
            return None
        return {"requested_length": length, "length": length, "exact": True, "draws": draws,
                "source": "live", "selection": key,
                "label": f"Fresh approximate posterior for the chosen-parameter history at exactly {length} trading days"}
    case = history["case"]
    _require(case < len(arrays["trajectory_id"])
             and str(arrays["trajectory_id"][case]) == history["id"]
             and np.array_equal(arrays["observations"][case], history["returns"])
             and np.array_equal(arrays["truth"][case], history["truth"]),
             "saved fallback must belong to this exact preserved history")
    translated = None if draws is None else {"selection": lesson.selection_key(task, arrays, case, length),
                                            "draws": draws}
    result = lesson.posterior_view(task, arrays, case, length, live=translated)
    return {**result, "selection": key}

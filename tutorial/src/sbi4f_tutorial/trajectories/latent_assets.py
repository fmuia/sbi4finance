"""Actual hidden simulator paths paired with the preserved synthetic histories.

These are revealed simulation truth, never observations or neural state estimates.
The frozen original returns remain unchanged; the replay's library-level rounding
agreement is verified before its hidden paths can be displayed.
"""

import hashlib
import json

import numpy as np

from ..shared.paths import ASSETS
from . import lesson, simulation


ASSET_FOLDER = "latent-history-v1"
BUNDLE_SHA256 = "7beb5b9647fdf2fdec819f5bf0c7f7c5170841572490412643ddebb24a74436c"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _asset_file(path):
    result = path.resolve()
    _require(result.is_relative_to(ASSETS.resolve()), "paired latent inputs must remain under tutorial/assets")
    return result


def load_history(task, arrays, case):
    """Return the complete original history and its actual boundary latent truth."""
    simulation.validate_task(task)
    case = simulation._integer(case, "latent history case", minimum=0)
    folder = ASSETS / "trajectories" / ASSET_FOLDER
    source = _asset_file(folder / "bundle.json")
    _require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == BUNDLE_SHA256,
             "paired latent bundle is missing or differs from its pinned identity")
    bundle = json.loads(source.read_text())
    _require(bundle.get("format") == "trajectory_paired_latent_v1"
             and bundle.get("scientific_acceptance") is False and bundle.get("inference_input") is False
             and bundle.get("state_estimate") is False and bundle.get("preserves_observed_history") is True
             and bundle.get("observed_day_boundaries") is True
             and set(bundle.get("models", {})) == set(simulation.MODELS),
             "paired latent format or simulation-truth scope differs")
    name = task["model"]
    record = bundle["models"][name]
    catalogue = lesson.catalogue()["artifacts"][name]
    _require(record.get("task") == task and record.get("task_sha256") == simulation.task_fingerprint(task)
             and record.get("source_replay_bundle_sha256") == catalogue["bundle_sha256"],
             "paired latent task or original replay identity differs")
    filename = name + ".npz"
    _require(set(bundle.get("files", {})) == {model+".npz" for model in simulation.MODELS},
             "paired latent file identities differ")
    payload = _asset_file(folder / filename)
    _require(payload.is_file() and hashlib.sha256(payload.read_bytes()).hexdigest() == bundle["files"][filename],
             "paired latent arrays differ from their bound file hash")
    with np.load(payload, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    common = {"trajectory_id", "parameters", "replayed_observations", "variance"}
    extra = ({"variance_state"} if name == "heston" else
             {"driver", "driver_variance", "mean_variance", "median_variance"})
    _require(set(data) == common | extra, "unexpected paired latent fields")
    count, days = 4, task["n_observations"]
    _require(case < count and np.shape(arrays["observations"]) == (count, days, 1)
             and np.shape(arrays["truth"]) == (count, len(task["parameter_names"]))
             and data["trajectory_id"].shape == (count,)
             and data["trajectory_id"].dtype.kind in "US"
             and np.array_equal(data["trajectory_id"], arrays["trajectory_id"])
             and data["trajectory_id"].tolist() == record.get("case_ids"),
             "paired latent cases must be the same four preserved histories in order")
    _require(data["parameters"].shape == arrays["truth"].shape
             and np.array_equal(data["parameters"], arrays["truth"]),
             "paired latent generating parameters differ from the original truth")
    _require(len(record.get("histories", [])) == count, "paired latent history identities are incomplete")
    for index, history in enumerate(record["histories"]):
        _require(history.get("case") == index
                 and history.get("observation_sha256") == _array_sha(arrays["observations"][index])
                 and history.get("parameter_sha256") == _array_sha(arrays["truth"][index]),
                 "paired latent observed-history bytes or parameters differ")
    replay = data["replayed_observations"]
    _require(replay.shape == arrays["observations"].shape and np.isfinite(replay).all()
             and np.allclose(replay, arrays["observations"], rtol=2e-12, atol=2e-14),
             "latent replay does not reproduce the preserved observed history")
    for key in extra | {"variance"}:
        _require(data[key].shape == (count, days+1) and data[key].dtype.kind == "f"
                 and np.isfinite(data[key]).all(), "paired latent boundary axes or finite values differ: "+key)
    _require(np.all(data["variance"] >= 0), "effective latent variance cannot be negative")
    if name == "heston":
        _require(np.array_equal(data["variance"], np.maximum(data["variance_state"], 0.))
                 and np.all(data["variance_state"][:, 0] == task["fixed"]["initial_variance"]),
                 "Heston latent variance must preserve its original signed numerical state")
    else:
        _require(np.all(data["driver_variance"] >= 0) and np.all(data["mean_variance"] > 0)
                 and np.all(data["median_variance"] >= 0)
                 and np.all(data["driver"][:, 0] == 0) and np.all(data["driver_variance"][:, 0] == 0)
                 and np.array_equal(data["variance"][:, 0], data["mean_variance"][:, 0]),
                 "rough latent paths must preserve the original zero-driver initialization")
        eta = data["parameters"][:, task["parameter_names"].index("eta"), None]
        expected = data["mean_variance"]*np.exp(eta*data["driver"]-.5*eta**2*data["driver_variance"])
        expected_median = data["mean_variance"]*np.exp(-.5*eta**2*data["driver_variance"])
        _require(np.allclose(data["variance"], expected, rtol=2e-13, atol=1e-16),
                 "rough latent variance and its paired driver disagree")
        _require(np.allclose(data["median_variance"], expected_median, rtol=2e-13, atol=1e-16),
                 "rough marginal median and the fixed-mean compensator disagree")
    selected = {key: data[key][case].copy() for key in extra | {"variance"}}
    return {"returns": np.asarray(arrays["observations"][case, :, 0]).copy(),
            "days": np.arange(days+1), "time_years": np.arange(days+1)*task["dt_years"],
            "variance": selected["variance"],
            "variance_state": selected.get("variance_state", selected["variance"].copy()),
            **{key: selected.get(key) for key in ("driver", "driver_variance", "mean_variance", "median_variance")},
            "parameters": data["parameters"][case].copy(), "parameter_names": list(task["parameter_names"]),
            "metadata": {"source": "actual synthetic generator truth, not a state estimate",
                         "task_sha256": record["task_sha256"], "case_id": str(data["trajectory_id"][case]),
                         "boundary_units": "annualized variance at observed trading-day boundaries, including day zero",
                         "original_returns_unchanged": True,
                         "replay_max_abs_return_error": float(np.max(np.abs(replay[case]-arrays["observations"][case]))),
                         "inference_input": False, "state_estimate": False}}

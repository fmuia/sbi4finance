"""Asset-local trajectory replays and explicit CPU inference for participants.

Native inference is defined in trajectories.inference and reused by the lab.
This module owns participant catalogue validation, case selection and figures.
Cached loading imports NumPy, never a neural backend, and never downloads data.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..shared.paths import ASSETS
from .inference import (
    FORMAT, DEFAULT_NETWORK, LOGISTIC_SD, PREFIX_POLICIES, BACKEND_SEED_MAPPING,
    fingerprint, sha256, integer, backend_seed, validate_task, _bounds,
    inverse_parameters, conditioning_data, build_workflow, sample, load_checkpoint,
)


CATALOGUE = ASSETS / "trajectories" / "artifacts.json"
MODEL_LABELS = {"heston": "Heston", "rough-known": "Rough Bergomi · known variance curve",
                "rough-joint": "Rough Bergomi · inferred variance curve"}
TRAJECTORY_FILES = {"task.json", "posterior.npz", "training.json", "evaluation.json"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _asset_path(path):
    result = Path(path).resolve()
    _require(result.is_relative_to(ASSETS.resolve()), "trajectory inputs must remain under tutorial/assets/")
    return result


def catalogue():
    path = _asset_path(CATALOGUE)
    if not path.exists():
        raise FileNotFoundError("Trajectory examples are not packaged yet. Obtain the tutorial trajectory assets first.")
    value = json.loads(path.read_text())
    _require(value.get("format") == "trajectory_tutorial_artifacts_v1"
             and value.get("scientific_acceptance") is False, "invalid trajectory catalogue")
    _require(isinstance(value.get("artifacts"), dict) and value["artifacts"]
             and set(value["artifacts"]) <= set(MODEL_LABELS), "invalid trajectory model entries")
    return value


def _teaching_task(task):
    validate_task(task)
    model = task.get("model")
    names = {"heston": ["kappa", "long_variance", "vol_of_vol", "rho", "drift"],
             "rough-known": ["hurst", "eta", "rho", "drift"],
             "rough-joint": ["hurst", "eta", "rho", "short_variance", "long_variance", "decay", "drift"]}
    _require(model in names and task.get("task_version") == 2
             and task.get("task_id") == f"physical_{model.replace('-', '_')}_daily_drift_paths_v2"
             and task.get("parameter_names") == names[model], "expected a harmonized daily drift teaching task")
    _require(task["n_observations"] == 1260 and task["dt_years"] == 1 / 252
             and task.get("history_length_range") == [1, 1260]
             and task["prior_bounds"][-1] == [-.10, .20] and task["prior_types"][-1] == "uniform"
             and "drift" not in task.get("fixed", {}), "daily grid, history range or inferred drift differs")
    _require((model == "heston" and task.get("numerics", {}).get("substeps") == 8)
             or (model != "heston" and task.get("numerics", {}).get("observation_substeps") == 4),
             "trajectory numerical observation law differs")
    return task


def load_example(name):
    """Validate all cached bytes, identities and daily return/draw axes offline."""
    entry = catalogue()["artifacts"][name]
    cache = _asset_path(CATALOGUE.parent / entry["cache"])
    bundle_path = _asset_path(cache / "bundle.json")
    _require(sha256(bundle_path) == entry["bundle_sha256"], "trajectory catalogue/bundle hash differs")
    report = json.loads(bundle_path.read_text())
    _require(report.get("format") == "trajectory_teaching_replay_v1"
             and report.get("role") == "cached_development_trajectory_replay"
             and report.get("scientific_acceptance") is False
             and report.get("reserved_test_evaluated") is False
             and report.get("source_split") == "validation" and report.get("source_offset") == 0
             and report.get("model") == name and set(report.get("files", {})) == TRAJECTORY_FILES,
             "trajectory replay format or development scope differs")
    for filename, expected in report["files"].items():
        _require(sha256(_asset_path(cache / filename)) == expected, "trajectory asset checksum differs: " + filename)
    task = _teaching_task(json.loads((cache / "task.json").read_text()))
    training = json.loads((cache / "training.json").read_text())
    evaluation = json.loads((cache / "evaluation.json").read_text())
    task_hash, model_hash = fingerprint(task), report["checkpoint_sha256"]
    _require(task["model"] == name and task_hash == entry["task_sha256"] == report["task_sha256"]
             == training["task_sha256"] == evaluation["task_sha256"], "trajectory task identity differs")
    _require(model_hash == entry["checkpoint_sha256"] == training["model_sha256"] == evaluation["checkpoint_sha256"]
             and training["bank_manifest_sha256"] == evaluation["bank_manifest_sha256"],
             "trajectory checkpoint or bank identity differs")
    _require(training.get("format") == FORMAT
             and training.get("normalization") == "fixed_prior_units_no_fitted_standardizer_v1"
             and training.get("supported_history_lengths") == [1, 1260]
             and training.get("prefix_policy", "uniform") in PREFIX_POLICIES
             and training.get("history_training") == PREFIX_POLICIES[training.get("prefix_policy", "uniform")],
             "trajectory checkpoint preprocessing differs")
    with np.load(cache / "posterior.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    lengths = report.get("cached_lengths", [])
    _require(isinstance(lengths, list) and lengths and lengths == sorted(set(lengths))
             and lengths[0] == 1 and lengths[-1] == 1260
             and all(type(n) is int and 1 <= n <= 1260 for n in lengths),
             "invalid cached trajectory lengths")
    _require(set(arrays) == {"observations", "truth", "trajectory_id", *[f"draws_{n}" for n in lengths]},
             "unexpected replay arrays; latent states must not be conditions")
    ids = arrays["trajectory_id"]
    count, dimensions = len(ids), len(task["parameter_names"])
    _require(ids.ndim == 1 and ids.dtype.kind in "US" and 1 <= count <= 4
             and ids.tolist() == report.get("case_ids"), "trajectory case identity differs")
    _require(all(identifier.split(":")[-2:] == ["validation", str(i)]
                 and identifier.split(":")[0] == task_hash[:16]
                 for i, identifier in enumerate(ids.tolist())), "cache must retain the predeclared first validation cases")
    _require(arrays["observations"].shape == (count, 1260, 1)
             and arrays["truth"].shape == (count, dimensions), "trajectory observation/truth shape differs")
    bounds = np.asarray(task["prior_bounds"])
    for key, values in arrays.items():
        if key == "trajectory_id":
            continue
        _require(np.isfinite(values).all(), "nonfinite cached trajectory array: " + key)
        if key.startswith("draws_"):
            _require(values.ndim == 3 and values.shape[0] == count and values.shape[1] >= 2
                     and values.shape[2] == dimensions, "cached posterior axes differ: " + key)
        if key == "truth" or key.startswith("draws_"):
            _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
                     "cached parameters outside declared prior: " + key)
    return task, {**report, "training_record_sha256": fingerprint(training),
                  "model_source": entry.get("model_source", "matching native checkpoint"),
                  "limitations": report.get("limitations", "Development neural examples; conditional accuracy is separate.")}, arrays


def cached_posterior(task, arrays, case, length):
    """Use the nearest cached past prefix, never future observations or interpolation."""
    case = integer(case, "case", minimum=0)
    length = integer(length, "observed days")
    _require(case < len(arrays["trajectory_id"]) and length <= task["n_observations"], "case or history outside replay")
    lengths = sorted(int(key.removeprefix("draws_")) for key in arrays if key.startswith("draws_"))
    selected = max(n for n in lengths if n <= length)
    return {"requested_length": length, "length": selected, "exact": selected == length,
            "draws": arrays[f"draws_{selected}"][case],
            "label": (f"Cached posterior at exactly {length} trading days" if selected == length else
                      f"Requested {length} days; cached posterior uses the nearest past prefix, {selected} days")}


def saved_lengths(arrays):
    """The participant's saved-mode control uses only actual posterior snapshots."""
    return sorted(int(key.removeprefix("draws_")) for key in arrays if key.startswith("draws_"))


def selection_key(task, arrays, case, length):
    """Bind session-only draws to the task, observed history and revealed prefix."""
    case = integer(case, "case", minimum=0)
    length = integer(length, "observed days")
    _require(case < len(arrays["trajectory_id"]) and length <= task["n_observations"],
             "case or history outside replay")
    return fingerprint(task), str(arrays["trajectory_id"][case]), length


def posterior_view(task, arrays, case, length, *, live=None):
    """Use matching fresh draws; stale actions fall back to an explicitly labelled cache.

    Every posterior display consumes this single result. In particular, moving
    to a different history, model or day cannot relabel earlier fresh draws.
    """
    key = selection_key(task, arrays, case, length)
    cached = {**cached_posterior(task, arrays, case, length), "source": "saved", "selection": key}
    if live is None or live.get("selection") != key:
        return cached
    draws = np.asarray(live["draws"])
    bounds = np.asarray(task["prior_bounds"])
    _require(draws.ndim == 2 and draws.shape[0] >= 2 and draws.shape[1] == len(bounds)
             and np.isfinite(draws).all() and np.all(draws >= bounds[:, 0])
             and np.all(draws <= bounds[:, 1]), "invalid fresh posterior draws")
    return {"requested_length": length, "length": length, "exact": True,
            "draws": draws, "source": "live", "selection": key,
            "label": f"Fresh approximate posterior at exactly {length} trading days"}


def posterior_rows(task, draws):
    quantiles = np.quantile(draws, [.05, .5, .95], axis=0)
    return [{"parameter": name, "posterior median": float(quantiles[1, index]),
             "central 90%": f"{quantiles[0, index]:.4g} to {quantiles[2, index]:.4g}",
             "prior bounds": f"{task['prior_bounds'][index][0]:.4g} to {task['prior_bounds'][index][1]:.4g}"}
            for index, name in enumerate(task["parameter_names"])]


def plot_returns(task, arrays, case, length):
    import matplotlib.pyplot as plt
    values = arrays["observations"][case, :length, 0]
    time = np.arange(1, length + 1) * task["dt_years"]
    figure, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    axes[0].plot(time, 100 * values, linewidth=.7, color="#2563eb")
    axes[0].set_ylabel("Daily log return (%)")
    axes[1].plot(time, np.cumsum(values), linewidth=1, color="#0f766e")
    axes[1].set_ylabel("Cumulative log return")
    axes[1].set_xlabel("Observed trading years (252 days/year)")
    figure.suptitle(f"{MODEL_LABELS[task['model']]} · first {length} observed days")
    figure.tight_layout()
    return figure


def plot_evolution(task, arrays, case, length, *, posterior=None):
    import matplotlib.pyplot as plt
    lengths = sorted(int(key.removeprefix("draws_")) for key in arrays
                     if key.startswith("draws_") and int(key.removeprefix("draws_")) <= length)
    time = np.asarray(lengths) * task["dt_years"]
    quantiles = np.asarray([np.quantile(arrays[f"draws_{n}"][case], [.05, .5, .95], axis=0) for n in lengths])
    snapshot_time, snapshot_quantiles = time.copy(), quantiles.copy()
    if lengths[-1] < length:
        time = np.r_[time, length * task["dt_years"]]
        quantiles = np.concatenate([quantiles, quantiles[-1:]], axis=0)
    names = task["parameter_names"]
    figure, axes = plt.subplots((len(names) + 1) // 2, 2, figsize=(10, 2.35 * ((len(names) + 1) // 2)), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(names):
            ax.set_visible(False)
            continue
        low, high = task["prior_bounds"][index]
        ax.axhspan(low, high, color="#64748b", alpha=.09, label="prior support")
        ax.fill_between(time, quantiles[:, 0, index], quantiles[:, 2, index], step="post", alpha=.22, color="#2563eb")
        ax.step(time, quantiles[:, 1, index], where="post", color="#2563eb", linewidth=1,
                label="cached median and 90% interval")
        ax.errorbar(snapshot_time, snapshot_quantiles[:, 1, index],
                    yerr=[snapshot_quantiles[:, 1, index] - snapshot_quantiles[:, 0, index],
                          snapshot_quantiles[:, 2, index] - snapshot_quantiles[:, 1, index]],
                    fmt="o", markersize=2.5, capsize=2, elinewidth=.7, color="#2563eb")
        if posterior is not None and posterior["source"] == "live":
            fresh = np.quantile(posterior["draws"][:, index], [.05, .5, .95])
            ax.errorbar(length * task["dt_years"], fresh[1],
                        yerr=[[fresh[1] - fresh[0]], [fresh[2] - fresh[1]]],
                        fmt="D", markersize=4, capsize=3, color="#0f766e",
                        label="fresh median and 90% interval")
        ax.axhline(arrays["truth"][case, index], color="#111827", linestyle=":", linewidth=1, label="synthetic truth")
        ax.set_title(names[index].replace("_", " "), fontsize=10)
        ax.set_xlim(0, max(task["dt_years"] * length, .02))
        ax.set_xlabel("Observed years", fontsize=8)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=7, loc="best")
    figure.suptitle("Approximate posterior evolution: saved snapshots shown as steps", fontsize=11)
    figure.tight_layout()
    return figure


def load_live(name, directory=None):
    """Explicit CPU role, restricted to matching native bytes in local assets."""
    directory = Path(directory) if directory else ASSETS / "local-checkpoints" / f"path-{name}"
    if not directory.is_absolute():
        directory = ASSETS / directory
    directory = _asset_path(directory)
    _require(directory.is_relative_to((ASSETS / "local-checkpoints").resolve()),
             "place the matching native model under tutorial/assets/local-checkpoints/ first")
    for filename in ("task.json", "training.json", "model.keras"):
        _asset_path(directory / filename)
    if not all((directory / filename).is_file() for filename in ("task.json", "training.json", "model.keras")):
        raise FileNotFoundError(f"Matching native checkpoint missing. Place task.json, training.json and model.keras in assets/local-checkpoints/path-{name}/.")
    task, report, _ = load_example(name)
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["JAX_PLATFORMS"] = "cpu"
    live_task, workflow, record = load_checkpoint(directory)
    _require(live_task == task and record["model_sha256"] == report["checkpoint_sha256"]
             and fingerprint(record) == report["training_record_sha256"],
             "native checkpoint differs from this cached trajectory example")
    return live_task, workflow, record

"""Asset-local trajectory replays and explicit CPU inference for participants.

The native inference definitions below are copied unchanged from the lab.
Catalogue validation, case selection and figures are participant-only helpers.
Cached loading imports NumPy, never a neural backend, and never downloads data.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .paths import ASSETS


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


def plot_evolution(task, arrays, case, length):
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
        ax.axhline(arrays["truth"][case, index], color="#111827", linestyle=":", linewidth=1, label="synthetic truth")
        ax.set_title(names[index].replace("_", " "), fontsize=10)
        ax.set_xlim(0, max(task["dt_years"] * length, .02))
        ax.set_xlabel("Observed years", fontsize=8)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=7, loc="best")
    figure.suptitle("Approximate posterior evolution: cached snapshots shown as steps", fontsize=11)
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


FORMAT = "path_posterior_v1"


DEFAULT_NETWORK = {"hidden_dim": 64, "summary_dim": 32, "depth": 6,
                   "bins": 8, "widths": [64, 64]}


LOGISTIC_SD = np.pi / np.sqrt(3.0)


PREFIX_POLICIES = {
    "uniform": "uniform_integer_per_trajectory_1_to_max_with_mask",
    "uniform_log_mixture": "half_uniform_half_log_uniform_integer_per_trajectory_1_to_max_with_mask",
}


BACKEND_SEED_MAPPING = "nonnegative integer modulo 2**32; existing uint32 seeds unchanged"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def backend_seed(seed):
    """Fold a recorded experiment seed to Keras' uint32 seed domain.

    Physical simulation and prefix RNGs retain their original full integer
    seeds. Only backend initialization/posterior sampling use this explicit,
    reproducible modulo mapping. Every previously valid uint32 seed is unchanged.
    """
    return integer(seed, "backend seed", minimum=0) % (2 ** 32)


def validate_task(task):
    names = task.get("parameter_names", [])
    bounds = np.asarray(task.get("prior_bounds", []), dtype=np.float64)
    kinds = task.get("prior_types", [])
    if (not names or len(set(names)) != len(names) or bounds.shape != (len(names), 2)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0])
            or len(kinds) != len(names) or any(k not in ("uniform", "log_uniform") for k in kinds)):
        raise ValueError("invalid path-task parameter/prior definition")
    for kind, (lower, _) in zip(kinds, bounds, strict=True):
        if kind == "log_uniform" and lower <= 0:
            raise ValueError("log-uniform prior must be positive")
    integer(task["n_observations"], "n_observations")
    if not np.isfinite(task["dt_years"]) or task["dt_years"] <= 0 or task.get("measure") != "P":
        raise ValueError("positive time step and an explicit P task required")
    if any(integer(n, "prefix length") > task["n_observations"] for n in task["prefix_lengths"]):
        raise ValueError("validation landmark exceeds maximum history")
    return task


def _bounds(task):
    validate_task(task)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64).copy()
    logged = np.asarray([kind == "log_uniform" for kind in task["prior_types"]])
    bounds[logged] = np.log(bounds[logged])
    return bounds, logged


def inverse_parameters(values, task):
    values = np.asarray(values, dtype=np.float64)
    bounds, logged = _bounds(task)
    if values.shape[-1:] != (len(bounds),) or not np.isfinite(values).all():
        raise ValueError("invalid unconstrained posterior array")
    # Stable sigmoid. Do not clip samples, reject them, or redraw them.
    z = values * LOGISTIC_SD
    u = np.exp(-np.logaddexp(0.0, -z))
    result = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
    result[..., logged] = np.exp(result[..., logged])
    return result


def conditioning_data(values, task, *, lengths=None):
    """Pad returns and known time coordinates, exposing neither latent states nor labels.

    ``values`` may contain any complete prefix, or a full history with explicit
    per-row lengths. Values after those lengths are ignored, even if nonfinite.
    """
    validate_task(task)
    if np.iscomplexobj(values) or np.ma.isMaskedArray(values):
        raise ValueError("real unmasked return arrays required")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    maximum = task["n_observations"]
    if values.ndim != 3 or values.shape[2] != 1 or not len(values) or not 1 <= values.shape[1] <= maximum:
        raise ValueError("returns must be [batch,time,1] with 1 <= time <= task maximum")
    if lengths is None:
        lengths = np.full(len(values), values.shape[1], dtype=np.int64)
    lengths = np.asarray(lengths)
    if (lengths.shape != (len(values),) or lengths.dtype.kind not in "iu"
            or np.any(lengths < 1) or np.any(lengths > values.shape[1])):
        raise ValueError("each integer length must lie within its supplied history")
    mask = np.arange(maximum)[None, :] < lengths[:, None]
    observed_mask = mask[:, :values.shape[1]]
    # Keep the full bank in float64, but materialize neural inputs directly in
    # float32. Chunked float64 scaling avoids whole-bank temporary arrays.
    tokens = np.zeros((len(values), maximum, 2), dtype=np.float32)
    scale = np.sqrt(0.04 * task["dt_years"])
    times = (np.arange(maximum) + 1) / maximum
    for start in range(0, len(values), 1024):
        stop = start + 1024
        present = observed_mask[start:stop]
        raw = values[start:stop, :, 0]
        if not np.isfinite(raw[present]).all():
            raise ValueError("observed returns must be finite")
        tokens[start:stop, :values.shape[1], 0] = np.where(present, raw / scale, 0.0)
        tokens[start:stop, :, 1] = np.where(mask[start:stop], times[None, :], 0.0)
    if not np.isfinite(tokens).all():
        raise ValueError("return scaling overflowed neural float32 storage")
    context = np.column_stack([lengths / maximum,
                               np.full(len(values), task["dt_years"] * 252.0)]).astype(np.float32)
    return {"observations": tokens, "time_context": context, "summary_mask": mask}


def build_workflow(task, *, network=None, optimizer=None):
    validate_task(task)
    import bayesflow as bf
    if bf.__version__ != "2.1.0":
        raise RuntimeError("path inference requires the locked BayesFlow 2.1.0 core environment")
    settings = dict(DEFAULT_NETWORK if network is None else network)
    adapter = (bf.Adapter().convert_dtype("float64", "float32")
               .rename("parameters", "inference_variables")
               .rename("observations", "summary_variables")
               .rename("time_context", "inference_conditions"))
    summary = bf.networks.RecurrentNetwork(
        hidden_dim=settings["hidden_dim"], summary_dim=settings["summary_dim"],
        recurrent_type="gru", bidirectional=False, time_axis=1,
        time_embed_dim=4, dropout=0.0, return_sequences=False)
    density = bf.networks.CouplingFlow(
        depth=settings["depth"], transform="spline",
        subnet_kwargs={"widths": settings["widths"]},
        transform_kwargs={"bins": settings["bins"]})
    return bf.BasicWorkflow(adapter=adapter, summary_network=summary, inference_network=density,
                            optimizer=optimizer, standardize=None)


def sample(workflow, task, values, draws, *, lengths=None, seed=0):
    conditions = conditioning_data(values, task, lengths=lengths)
    output = workflow.sample(num_samples=integer(draws, "draws"), conditions=conditions,
                             sample_shape=(), seed=backend_seed(seed))
    result = inverse_parameters(np.asarray(output["parameters"], dtype=np.float64), task)
    expected = (len(conditions["observations"]), draws, len(task["parameter_names"]))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"invalid posterior sample shape/value: expected {expected}, got {result.shape}")
    return result


def load_checkpoint(directory):
    directory = Path(directory)
    task = validate_task(json.loads((directory / "task.json").read_text()))
    report = json.loads((directory / "training.json").read_text())
    policy = report.get("prefix_policy", "uniform")
    if (report.get("format") != FORMAT or report.get("task_sha256") != fingerprint(task)
            or report.get("model_sha256") != sha256(directory / "model.keras")
            or report.get("normalization") != "fixed_prior_units_no_fitted_standardizer_v1"
            or policy not in PREFIX_POLICIES
            or report.get("history_training") != PREFIX_POLICIES[policy]):
        raise ValueError("checkpoint task/model/normalization contract mismatch")
    workflow = build_workflow(task, network=report["network"])
    workflow.load_estimator(directory / "model.keras")
    return task, workflow, report

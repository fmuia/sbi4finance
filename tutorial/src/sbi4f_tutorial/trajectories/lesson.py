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
from ..shared.full_surface_teaching import digest as asset_digest
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
    """Validate cached identities offline; only JSON CRLF/LF is equivalent.

    This also accepts existing Windows checkouts whose JSON line endings were
    converted before the asset-preserving Git attributes were introduced.
    Binary arrays and model files retain exact byte-level validation.
    """
    entry = catalogue()["artifacts"][name]
    cache = _asset_path(CATALOGUE.parent / entry["cache"])
    bundle_path = _asset_path(cache / "bundle.json")
    _require(asset_digest(bundle_path) == entry["bundle_sha256"], "trajectory catalogue/bundle hash differs")
    report = json.loads(bundle_path.read_text())
    _require(report.get("format") == "trajectory_teaching_replay_v1"
             and report.get("role") == "cached_development_trajectory_replay"
             and report.get("scientific_acceptance") is False
             and report.get("reserved_test_evaluated") is False
             and report.get("source_split") == "validation" and report.get("source_offset") == 0
             and report.get("model") == name and set(report.get("files", {})) == TRAJECTORY_FILES,
             "trajectory replay format or development scope differs")
    for filename, expected in report["files"].items():
        _require(asset_digest(_asset_path(cache / filename)) == expected, "trajectory asset checksum differs: " + filename)
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


def posterior_readout(task, draws, truth=None):
    """Describe paired physical draws; interval contraction is not accuracy.

    Widths are compared with the central 90% of each declared prior, including
    log-uniform priors. Pearson correlation describes linear dependence only.
    Optional generating truth adds a per-example check, never a calibration
    conclusion or a change to the posterior statistics.
    """
    validate_task(task)
    _require(not np.iscomplexobj(draws) and not np.ma.isMaskedArray(draws),
             "posterior readout requires real unmasked parameter rows")
    values = np.asarray(draws)
    names = task["parameter_names"]
    dimensions = len(names)
    _require(values.ndim == 2 and len(values) >= 2 and values.shape[1] == dimensions
             and values.dtype.kind in "fiu" and np.isfinite(values).all(),
             "posterior readout requires finite complete rows [draws>=2, parameters]")
    values = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
             "posterior readout rows are outside physical prior bounds")
    if truth is not None:
        _require(not np.iscomplexobj(truth) and not np.ma.isMaskedArray(truth),
                 "generating truth must be real unmasked physical values")
        truth = np.asarray(truth)
        _require(truth.shape == (dimensions,) and truth.dtype.kind in "fiu"
                 and np.isfinite(truth).all() and np.all(truth > bounds[:, 0])
                 and np.all(truth < bounds[:, 1]),
                 "generating truth must be one finite row strictly inside prior bounds")
    probabilities = np.array([.05, .5, .95])
    posterior = np.quantile(values, probabilities, axis=0)
    prior = bounds[:, 0] + probabilities[:, None] * (bounds[:, 1] - bounds[:, 0])
    logged = np.asarray([kind == "log_uniform" for kind in task["prior_types"]])
    prior[:, logged] = np.exp(np.log(bounds[logged, 0]) + probabilities[:, None]
                              * (np.log(bounds[logged, 1]) - np.log(bounds[logged, 0])))
    prior_widths = prior[2] - prior[0]
    _require(np.all(prior_widths > 0), "prior central 90% widths must be numerically positive")
    ratios = (posterior[2] - posterior[0]) / prior_widths
    parameters = {}
    excluded = None if truth is None else []
    for index, name in enumerate(names):
        inside = None if truth is None else bool(posterior[0, index] <= truth[index] <= posterior[2, index])
        parameters[name] = {
            "q05": float(posterior[0, index]), "q50": float(posterior[1, index]),
            "q95": float(posterior[2, index]), "prior_q05": float(prior[0, index]),
            "prior_q50": float(prior[1, index]), "prior_q95": float(prior[2, index]),
            "width_ratio90": float(ratios[index]), "truth_inside90": inside,
        }
        if inside is False:
            excluded.append(name)
    varying = np.flatnonzero(np.any(values != values[0], axis=0))
    strongest = None
    if len(varying) >= 2:
        centered = values[:, varying] - values[:, varying].mean(axis=0)
        # Correlation is scale invariant. Scaling deviations avoids underflow
        # for a very concentrated drift posterior whose mean may be near zero.
        centered /= np.max(np.abs(centered), axis=0)
        correlations = np.corrcoef(centered, rowvar=False)
        first, second = np.triu_indices(len(varying), k=1)
        selected = int(np.argmax(np.abs(correlations[first, second])))
        strongest = {"names": [names[varying[first[selected]]], names[varying[second[selected]]]],
                     "correlation": float(correlations[first[selected], second[selected]])}
    return {
        "draw_count": len(values), "parameters": parameters,
        "narrowest_relative_to_prior": names[int(np.argmin(ratios))],
        "widest_relative_to_prior": names[int(np.argmax(ratios))],
        "strongest_pair": strongest, "truth_excluded_parameters": excluded,
    }


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


TUTORIAL_NETWORK = {"hidden_dim": 16, "summary_dim": 16, "depth": 2,
                    "bins": 8, "widths": [32, 32]}


def simulator_preview(task, parameters, *, paths=8, days=1260, seed=20261108):
    """A small given-parameter forward example, separate from every fitted history."""
    from copy import deepcopy
    from . import simulation

    _teaching_task(task)
    days, paths = integer(days, "days"), integer(paths, "paths")
    _require(days <= 1260 and 2 <= paths <= 16, "preview supports up to five years and two to 16 paths")
    _require(isinstance(parameters, dict) and set(parameters) == set(task["parameter_names"]),
             "supply exactly the model's declared parameters")
    preview_task = deepcopy(task)
    preview_task.update(n_observations=days, history_length_range=[1, days],
                        prefix_lengths=sorted({n for n in task["prefix_lengths"] if n <= days} | {days}))
    values = np.asarray([parameters[name] for name in task["parameter_names"]])
    observations = simulation.simulate_parameters(
        preview_task, np.repeat(values[None, :], paths, axis=0), seed=seed, backend="numpy")
    return {"task": preview_task, "parameters": values, "observations": observations, "seed": seed}


SIMULATOR_SCALES = {
    "Standard": {"prices": (0.0, 200.0), "returns": (-8.0, 8.0), "realized": (0.0, 80.0)},
    "Wide": {"prices": (0.0, 400.0), "returns": (-20.0, 20.0), "realized": (0.0, 200.0)},
    "Very wide": {"prices": (0.0, 1000.0), "returns": (-50.0, 50.0), "realized": (0.0, 500.0)},
}


def _forward_series(result):
    """Return every simulated point and every selected return-derived summary."""
    task, returns = result["task"], result["observations"][:, :, 0]
    times = np.arange(returns.shape[1] + 1) * task["dt_years"]
    with np.errstate(over="raise", invalid="raise"):
        prices = 100 * np.exp(np.column_stack([np.zeros(len(returns)), np.cumsum(returns, axis=1)]))
    window = min(21, returns.shape[1])
    realized = np.std(np.lib.stride_tricks.sliding_window_view(returns[0], window), axis=1) / np.sqrt(task["dt_years"])
    return {"prices": prices, "returns": 100 * returns[0], "realized": 100 * realized,
            "times": times, "window": window}


def simulator_axis_summary(result, scale="Standard"):
    """Report view clipping explicitly; changing a scale never alters the data."""
    _require(scale in SIMULATOR_SCALES, "choose a declared fixed simulator scale")
    series = _forward_series(result)
    labels = {"prices": "Asset price: all paths", "returns": "Daily return: highlighted path (%)",
              "realized": "Realized volatility: highlighted path (%)"}
    rows = []
    for name, label in labels.items():
        values = series[name]
        low, high = SIMULATOR_SCALES[scale][name]
        rows.append({"panel": label, "fixed lower limit": low, "fixed upper limit": high,
                     "points below view": int(np.count_nonzero(values < low)),
                     "points above view": int(np.count_nonzero(values > high)),
                     "total points": int(values.size), "data minimum": float(values.min()),
                     "data maximum": float(values.max())})
    return rows


def forward_simulator_figure(result, *, scale="Standard"):
    """Fixed-axis path/return plots retain all points and report any outside view."""
    import matplotlib.pyplot as plt

    _require(scale in SIMULATOR_SCALES, "choose a declared fixed simulator scale")
    task, series = result["task"], _forward_series(result)
    times, prices = series["times"], series["prices"]
    figure = plt.figure(figsize=(10, 5), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=[1.3, 1])
    path_ax = figure.add_subplot(grid[:, 0])
    return_ax = figure.add_subplot(grid[0, 1])
    vol_ax = figure.add_subplot(grid[1, 1], sharex=return_ax)
    shades = plt.get_cmap("PuBuGn")(np.linspace(.35, .8, len(prices)))
    for i in range(len(prices)):
        path_ax.plot(times, prices[i], color="#7c3aed" if i == 0 else shades[i],
                     linewidth=1.7 if i == 0 else .85, alpha=1 if i == 0 else .75,
                     label="Highlighted path" if i == 0 else None)
    path_ax.axhline(100, color="#94a3b8", linestyle=":", linewidth=.8)
    path_ax.set(xlabel="Years", ylabel="Asset price (starts at 100)",
                title=f"Same parameters · {len(prices)} independent paths")
    path_ax.legend(fontsize=8)
    selected = series["returns"]
    return_ax.bar(times[1:], selected, width=task["dt_years"] * .8,
                  color=np.where(selected >= 0, "#1baf7a", "#eb6834"), alpha=.8)
    return_ax.set(ylabel="Daily log return (%)", title="The highlighted path, day by day")
    vol_ax.plot(times[series["window"]:], series["realized"], color="#0f766e", linewidth=1.2)
    vol_ax.fill_between(times[series["window"]:], series["realized"], color="#1baf7a", alpha=.15)
    vol_ax.set(xlabel="Years", ylabel=f"{series['window']}-day realized vol. (%)",
               title="A summary computed from returns")
    for ax, name, summary in zip((path_ax, return_ax, vol_ax), ("prices", "returns", "realized"),
                                  simulator_axis_summary(result, scale), strict=True):
        ax.set(xlim=(0, times[-1]), ylim=SIMULATOR_SCALES[scale][name])
        outside = summary["points below view"] + summary["points above view"]
        ax.text(.02, .02, f"{outside}/{summary['total points']} points outside view",
                transform=ax.transAxes, va="bottom", fontsize=7, color="#475569",
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
    figure.suptitle(MODEL_LABELS[task["model"]] + f" · fixed {scale.lower()} axes", fontsize=12)
    return figure


def load_market_example(name="voo-covid-256"):
    """Read the fixed motivation window offline; never send it to an estimator."""
    _require(name in ("voo-covid-256", "voo-2020-20261113"), "choose a packaged dated market example")
    folder = _asset_path(ASSETS / "market-examples" / name)
    accepted = json.loads(_asset_path(folder / "ACCEPTED.json").read_text())
    for filename in ("market.json", "returns.npz"):
        _require(asset_digest(_asset_path(folder / filename)) == accepted["source_sha256"][filename],
                 "market motivation checksum differs: " + filename)
    metadata = json.loads((folder / "market.json").read_text())
    with np.load(folder / "returns.npz", allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in ("dates", "simple_returns", "log_returns")}
        prices = {key: saved[key].copy() for key in ("price_dates", "adjusted_close") if key in saved}
    expected = 256 if name == "voo-covid-256" else 1726
    _require(metadata.get("ticker") == "VOO" and metadata.get("n") == expected,
             "expected the selected fixed VOO motivation window")
    _require(all(values[key].shape == (expected,) for key in values), "market window axes differ")
    _require(values["dates"].dtype.kind in "US" and np.all(values["dates"][1:] > values["dates"][:-1]),
             "market dates must be ordered ISO strings")
    for key in ("simple_returns", "log_returns"):
        _require(values[key].dtype.kind in "fiu" and np.isfinite(values[key]).all(),
                 "market returns must be finite real values")
    _require(np.all(values["simple_returns"] > -1), "simple returns must exceed -100%")
    _require(np.allclose(np.log1p(values["simple_returns"]), values["log_returns"], rtol=1e-12, atol=1e-15),
             "market log returns differ from the documented log1p conversion")
    if name != "voo-covid-256":
        _require(set(prices) == {"price_dates", "adjusted_close"}
                 and prices["adjusted_close"].shape == prices["price_dates"].shape == (expected + 1,)
                 and np.isfinite(prices["adjusted_close"]).all() and np.all(prices["adjusted_close"] > 0)
                 and np.array_equal(prices["price_dates"][1:], values["dates"]),
                 "market adjusted-price axes or values differ")
        _require(np.allclose(prices["adjusted_close"][1:] / prices["adjusted_close"][:-1] - 1,
                            values["simple_returns"], rtol=0, atol=1e-15),
                 "market returns do not reproduce the preserved adjusted prices")
    return {"metadata": metadata, **values, **prices}


def market_motivation_figure(example):
    """Observed ordering and a Gaussian tail benchmark, with no neural inference."""
    import matplotlib.pyplot as plt
    from scipy.special import ndtr

    returns = example["log_returns"]
    dates = example["dates"].astype("datetime64[D]")
    mean, scale = float(returns.mean()), float(returns.std())
    _require(scale > 0, "Gaussian motivation benchmark needs nonconstant returns")
    magnitudes = np.sort(np.abs(returns))
    thresholds, first_indices = np.unique(magnitudes, return_index=True)
    grid = np.linspace(0, float(magnitudes[-1]) * 1.05, 300)
    gaussian_tail = ndtr((-grid - mean) / scale) + ndtr((mean - grid) / scale)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.1), layout="constrained")
    axes[0].plot(dates, 100 * returns, color="#334155", linewidth=.8)
    axes[0].axhline(0, color="#94a3b8", linewidth=.7)
    axes[0].set(title="Quiet periods and clusters of large moves", ylabel="Daily log return (%)", xlabel="Trading date")
    # Plot strictly positive empirical probabilities; no zero gets a log transform.
    survival = (len(returns) - first_indices) / len(returns)
    axes[1].step(100 * thresholds, survival, where="post", color="#7c3aed", label="Observed, 256 days")
    axes[1].plot(100 * grid, gaussian_tail, color="#0f766e", label="Fitted Gaussian benchmark")
    axes[1].set(yscale="log", ylim=(.5 / len(returns), 1.1), xlim=(0, 100 * grid[-1]),
                xlabel="Absolute daily log return (%)", ylabel="Probability of at least this size",
                title="Compare large-return frequencies")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)
    figure.suptitle(f"VOO · {example['dates'][0]} to {example['dates'][-1]} · 256 observations", fontsize=11)
    return figure


SIMULATOR_COMPARISONS = {
    "hurst": (.05, .15, .35), "eta": (.5, 1.5, 2.5),
    "vol_of_vol": (.10, .30, .50), "kappa": (.5, 2., 5.),
}


def simulator_comparison(task, parameters, parameter, *, paths=4, days=1260, seed=20261108):
    """Compare three settings with the same per-row innovations in every column."""
    _teaching_task(task)
    choices = ("vol_of_vol", "kappa") if task["model"] == "heston" else ("hurst", "eta")
    _require(parameter in choices, "choose a comparison parameter for this model")
    bounds = np.asarray(task["prior_bounds"])[task["parameter_names"].index(parameter)]
    settings = np.asarray(SIMULATOR_COMPARISONS[parameter])
    if np.any(settings < bounds[0]) or np.any(settings > bounds[1]):
        settings = np.linspace(*bounds, 5)[1:-1]
    previews = []
    for value in settings:
        chosen = {**parameters, parameter: float(value)}
        # Separate calls deliberately reuse row IDs and task-bound random inputs.
        previews.append(simulator_preview(task, chosen, paths=paths, days=days, seed=seed))
    return {"parameter": parameter, "settings": settings, "previews": previews,
            "fixed_parameters": {key: value for key, value in parameters.items() if key != parameter}, "seed": seed}


def simulator_comparison_figure(comparison, *, scale="Standard"):
    """Aligned return/realized-volatility views, retaining every out-of-view value."""
    import matplotlib.pyplot as plt

    _require(scale in SIMULATOR_SCALES, "choose a declared fixed simulator scale")
    figure, axes = plt.subplots(2, 3, figsize=(10, 4.4), sharex=True, sharey="row", layout="constrained")
    label = {"hurst": "H", "eta": "η", "vol_of_vol": "σ", "kappa": "κ"}[comparison["parameter"]]
    for column, (value, preview) in enumerate(zip(comparison["settings"], comparison["previews"], strict=True)):
        returns = 100 * preview["observations"][:, :, 0]
        dt = preview["task"]["dt_years"]
        days = np.arange(1, returns.shape[1] + 1)
        window = min(21, returns.shape[1])
        realized = np.std(np.lib.stride_tricks.sliding_window_view(returns, window, axis=1), axis=-1) / np.sqrt(dt)
        for row in range(len(returns)):
            color, alpha = ("#7c3aed", 1.) if row == 0 else ("#94a3b8", .5)
            axes[0, column].plot(days, returns[row], color=color, alpha=alpha, linewidth=1. if row == 0 else .55)
            axes[1, column].plot(days[window - 1:], realized[row], color=color, alpha=alpha, linewidth=1. if row == 0 else .55)
        axes[0, column].set_title(f"{label} = {value:g}", fontsize=10)
        for row, (name, data) in enumerate((("returns", returns), ("realized", realized))):
            low, high = SIMULATOR_SCALES[scale][name]
            outside = np.count_nonzero((data < low) | (data > high))
            axes[row, column].set(ylim=(low, high), xlim=(1, len(days)))
            axes[row, column].text(.02, .02, f"{outside}/{data.size} outside view", transform=axes[row, column].transAxes,
                                   fontsize=6.5, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
            axes[row, column].tick_params(labelsize=8)
        axes[1, column].set_xlabel("Trading day", fontsize=8)
    axes[0, 0].set_ylabel("Daily log return (%)", fontsize=9)
    axes[1, 0].set_ylabel("21-day realized vol. (%)", fontsize=9)
    figure.suptitle("Same random inputs across settings · purple = the same realization · fixed axes", fontsize=10)
    return figure


def architecture_figure(task, *, network=None):
    """Decompose the actual recurrent encoder and spline density estimator."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    validate_task(task)
    settings = DEFAULT_NETWORK if network is None else network
    figure, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [(.015, .22, "Observed input", f"Returns + time\nMask: 1 to {task['n_observations']} days", "#eff6ff"),
             (.27, .24, "Encoder · causal GRU", f"{settings['hidden_dim']} hidden units\n{settings['summary_dim']} summary values", "#f3e8ff"),
             (.55, .26, "Density · coupling flow", f"{settings['depth']} spline blocks\nSubnets {settings['widths']} · {settings['bins']} bins", "#dcfce7"),
             (.85, .135, "Joint draws", f"{len(task['parameter_names'])} parameters\nOriginal units", "#eff6ff")]
    for x, width, title, body, color in boxes:
        ax.add_patch(FancyBboxPatch((x, .35), width, .4, boxstyle="round,pad=.008,rounding_size=.02",
                                   facecolor=color, edgecolor="#94a3b8", linewidth=.9))
        ax.text(x + width / 2, .66, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + width / 2, .49, body, ha="center", va="center", fontsize=8)
    for start, end in ((.24, .26), (.515, .54), (.815, .84)):
        ax.add_patch(FancyArrowPatch((start, .55), (end, .55), arrowstyle="-|>",
                                     mutation_scale=12, color="#64748b", linewidth=1.2))
    ax.text(.53, .25, "Known time context also conditions the flow", ha="center", fontsize=8, color="#64748b")
    ax.text(.5, .14, "Training: generating parameters score the density; both networks learn together.",
             ha="center", fontsize=9, color="#7c3aed")
    ax.text(.5, .04, "Inference: only observed returns, their mask and the known time design enter.",
             ha="center", fontsize=9, color="#0f766e")
    return figure


def input_concept_figure(task):
    """Explain the actual token/condition contract, with labels on a separate training route."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    validate_task(task)
    maximum = task["n_observations"]
    figure, ax = plt.subplots(figsize=(12, 3.15), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [(.01, .16, "One observed history", r"$r_1,\ldots,r_T$" + "\nKeep the time order", "#eff6ff"),
             (.205, .25, "Preprocess return tokens", r"$[r_j/\sqrt{0.04\Delta t},\ j/T_{\max}]$" + "\nTrue mask: observed; false: padding", "#eff6ff"),
             (.49, .15, "GRU encoder", "Reads tokens\nwith their mask", "#f3e8ff"),
             (.675, .31, "Coupling-flow conditions", "GRU summary + known time context\n" + r"$[T/T_{\max},\ \Delta t/(1/252)]$", "#dcfce7")]
    for x, width, title, body, color in boxes:
        ax.add_patch(FancyBboxPatch((x, .56), width, .35, boxstyle="round,pad=.007,rounding_size=.02",
                                   facecolor=color, edgecolor="#94a3b8", linewidth=.8))
        ax.text(x + width / 2, .83, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + width / 2, .70, body, ha="center", va="center", fontsize=8)
    for start, end in ((.17, .20), (.455, .485), (.64, .67)):
        ax.add_patch(FancyArrowPatch((start, .735), (end, .735), arrowstyle="-|>",
                                     mutation_scale=12, color="#64748b", linewidth=1.1))
    training_boxes = [(.08, "Generating parameters Θ", "Training labels only; absent at query"),
                      (.59, "Training objective", r"$-\log q_\phi(\Theta\mid r_{1:T},T)$")]
    for x, title, body in training_boxes:
        ax.add_patch(FancyBboxPatch((x, .19), .38, .21,
                                   boxstyle="round,pad=.009,rounding_size=.02",
                                   facecolor="#f3e8ff", edgecolor="#a78bfa", linewidth=.9))
        ax.text(x + .19, .335, title, ha="center", va="center", fontsize=9,
                fontweight="bold", color="#6d28d9")
        ax.text(x + .19, .255, body, ha="center", va="center", fontsize=8.5, color="#6d28d9")
    ax.add_patch(FancyArrowPatch((.47, .295), (.58, .295), arrowstyle="-|>",
                                 mutation_scale=12, color="#7c3aed", linewidth=1.1, linestyle="--"))
    ax.add_patch(FancyArrowPatch((.83, .55), (.83, .415), arrowstyle="-|>",
                                 mutation_scale=12, color="#7c3aed", linewidth=1.1, linestyle="--"))
    ax.text(.5, .10, "Dashed arrows: training only", ha="center", va="center", fontsize=8, color="#7c3aed")
    ax.text(.5, .025, f"Query: upper route only · maximum {maximum} days · no generating values or latent variance enter conditions.",
             ha="center", va="center", fontsize=8.5, color="#0f766e")
    return figure

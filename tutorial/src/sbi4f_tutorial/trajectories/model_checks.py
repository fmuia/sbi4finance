"""Prior and posterior-predictive checks of return histories under the stated P law.

Replicated histories start at the simulator's origin. They are neither future
continuations of the observation nor latent-volatility forecasts.
"""

import copy
import time

import numpy as np

from . import diagnostics, simulation


METRIC_LABELS = (
    "Cumulative log return (log %)",
    "Annualized return RMS (%)",
    "Largest absolute daily log return (log %)",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _length(task, length):
    simulation.validate_task(task)
    length = simulation._integer(length, "checked history length", minimum=1)
    _require(length <= task["n_observations"], "checked prefix exceeds this task's maximum")
    return length


def _real(values, shape, name):
    _require(not np.iscomplexobj(values) and not np.ma.isMaskedArray(values),
             f"{name} must be real unmasked values")
    values = np.asarray(values)
    _require(values.shape == shape and values.dtype.kind in "fiu" and np.isfinite(values).all(),
             f"{name} must have finite shape {shape}")
    return np.asarray(values, dtype=np.float64)


def _observed_prefix(task, observed, length):
    values = np.asanyarray(observed)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    _require(values.ndim == 2 and values.shape[1] == 1
             and length <= len(values) <= task["n_observations"],
             "provide one observed history [T], [T,1] or [1,T,1] covering the checked prefix")
    # The unused suffix is never summarized, including nonfinite future values.
    return _real(values[:length], (length, 1), "observed prefix").copy()


def _summaries(returns, dt):
    """Three finite return statistics; RMS does not estimate hidden V directly."""
    values = returns[:, :, 0]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            largest = np.max(np.abs(values), axis=1)
            divisor = np.where(largest > 0, largest, 1.)
            # Scale before squaring so a finite large return is not squared into infinity.
            rms = largest * np.sqrt(np.mean((values / divisor[:, None])**2, axis=1) / dt)
            result = 100 * np.column_stack([np.sum(values, axis=1), rms, largest])
    except FloatingPointError as error:
        raise FloatingPointError("return summaries overflowed; no histories were dropped") from error
    _require(np.isfinite(result).all(), "return summaries must be finite; no histories were dropped")
    return result


def _result(task, observed, length, replicated, parameters, metadata, kind):
    n = len(parameters)
    replicated = _real(replicated, (n, task["n_observations"], 1), "replicated histories")
    prefix = replicated[:, :length, :]
    summary = _summaries(prefix, task["dt_years"])
    observed_summary = _summaries(observed[None, :, :], task["dt_years"])[0]
    percentiles = 100 * np.mean((summary < observed_summary)
                                + .5 * (summary == observed_summary), axis=0)
    return {
        "kind": kind, "task": copy.deepcopy(task),
        "task_fingerprint": simulation.task_fingerprint(task), "length": length,
        "replicated_returns": replicated, "replicated_prefix": prefix,
        "observed_returns": observed, "selected_parameters": parameters,
        "summaries": summary, "observed_summary": observed_summary,
        "metric_labels": METRIC_LABELS, "empirical_percentiles": percentiles,
        "metadata": {
            **metadata, "count": n, "full_history_length": task["n_observations"],
            "measure": "P", "percentile_rule": "fraction below + half the fraction tied",
            "scope": "finite return-summary comparison; not calibration, joint support, model validation or forecasting",
        },
    }


def compare_prior(task, panel, observed, length):
    """Compare the observed prefix with unchanged, task-bound prior simulations."""
    length = _length(task, length)
    data = diagnostics._panel(task, panel)
    observed = _observed_prefix(task, observed, length)
    return _result(task, observed, length, data["observations"], data["parameters"], {
        "seed": panel["metadata"]["seed"], "source": "prior_predictive_panel",
        "source_row_count": len(data["parameters"]), "posterior_indices": None,
        "simulation_seconds": panel["metadata"]["generation_seconds"],
    }, "prior_predictive")


def replicate_check(task, observed, posterior_draws, length, count=32, seed=20261113):
    """Generate fresh complete histories at selected paired physical posterior rows.

    Parameter uncertainty and process randomness are both retained. No quote
    noise is added. The supplied observation is used only for the comparison.
    """
    length = _length(task, length)
    count = simulation._integer(count, "replicate count", minimum=2)
    seed = simulation._integer(seed, "replication seed")
    _require(count <= 32, "posterior predictive illustration is capped at 32 histories")
    observed = _observed_prefix(task, observed, length)
    values = np.asanyarray(posterior_draws)
    dimensions = len(task["parameter_names"])
    _require(values.ndim == 2 and values.shape[1] == dimensions and len(values) >= 2,
             "provide at least two complete paired physical posterior parameter rows")
    values = _real(values, values.shape, "posterior parameter rows")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
             "posterior parameter rows are outside physical prior bounds")
    rng = np.random.default_rng(np.random.SeedSequence([0x50504352, seed]))
    selected = rng.choice(len(values), size=min(count, len(values)), replace=False)
    parameters = values[selected].copy()
    started = time.perf_counter()
    # Keep the full original task: shortening a rough simulation changes its
    # task-keyed random stream and would cease to be this task's replicated data.
    replicated = simulation.simulate_parameters(task, parameters, seed=seed, backend="numpy")
    seconds = time.perf_counter() - started
    return _result(task, observed, length, replicated, parameters, {
        "seed": seed, "source": "posterior_draws_argument", "requested_count": count,
        "source_row_count": len(values), "posterior_indices": selected.tolist(),
        "selection": "uniform complete rows without replacement", "simulation_seconds": seconds,
        "initialization": "fresh full history from the task's origin; no conditioning on observed latent state",
    }, "posterior_predictive")


def plot_check(result):
    """Histograms of all simulated summaries and the observed prefix's values."""
    import matplotlib.pyplot as plt

    values = _real(result["summaries"], (result["metadata"]["count"], 3), "summary samples")
    observed = _real(result["observed_summary"], (3,), "observed summary")
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained")
    for index, (ax, label) in enumerate(zip(axes, result["metric_labels"], strict=True)):
        all_values = np.r_[values[:, index], observed[index]]
        low, high = all_values.min(), all_values.max()
        pad = max((high - low) * .05, .01)
        low, high = low - pad, high + pad
        ax.hist(values[:, index], bins=np.linspace(low, high, 17), color="#2563eb", alpha=.7,
                label="Simulated histories")
        ax.axvline(observed[index], color="#111827", lw=2, label="Observed prefix")
        ax.set(xlim=(low, high), xlabel=label, ylabel="Number of histories" if index == 0 else "")
        ax.set_title(f"Observed position: {result['empirical_percentiles'][index]:.0f}%", fontsize=9)
        ax.text(.02, .96, f"All {len(values)} histories retained", transform=ax.transAxes,
                va="top", fontsize=7, color="#475569")
    axes[0].legend(fontsize=7, loc="lower right")
    title = "Before observing data: the prior predictive" if result["kind"] == "prior_predictive" else "Replicated histories from the approximate posterior"
    figure.suptitle(f"{title} · first {result['length']} days", fontsize=11)
    return figure

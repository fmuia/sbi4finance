"""Prior and posterior-predictive checks of return histories under the stated P law.

Replicated histories start at the simulator's origin. They are neither future
continuations of the observation nor latent-volatility forecasts.
"""

import copy
import time

import numpy as np

from . import diagnostics, simulation
from ..shared import path_statistics


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
        "ordering": _ordering(prefix, observed),
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


def _parameter_rows(task, posterior_draws):
    values = np.asanyarray(posterior_draws)
    dimensions = len(task["parameter_names"])
    _require(values.ndim == 2 and values.shape[1] == dimensions and len(values) >= 2,
             "provide at least two complete paired physical posterior parameter rows")
    values = _real(values, values.shape, "posterior parameter rows")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
             "posterior parameter rows are outside physical prior bounds")
    return values


def _chunk_seeds(seed, count):
    # Every chunk receives distinct process innovations. The first chunk retains
    # the original small-check seed; arithmetic wraps within the accepted uint64 domain.
    return [(seed + index) % (2 ** 64) for index in range((count + 31) // 32)]


def _simulate_chunked(task, parameters, seed):
    seeds = _chunk_seeds(seed, len(parameters))
    chunks = []
    for index, start in enumerate(range(0, len(parameters), 32)):
        rows = parameters[start:start + 32]
        values = simulation.simulate_parameters(task, rows, seed=seeds[index], backend="numpy")
        chunks.append(_real(values, (len(rows), task["n_observations"], 1), "replicated histories"))
    return np.concatenate(chunks), seeds


def _ordering(replicated, observed, shuffle_seed=20261112):
    """Reuse exactly the revealed returns; preserve unavailable rows explicitly."""
    length = len(observed)
    permutation = np.random.default_rng(shuffle_seed).permutation(length)
    shuffled = observed[permutation].copy()
    result = {"length": length, "shuffle_seed": shuffle_seed,
              "shuffle_indices": permutation, "shuffled_returns": shuffled}
    for key, absolute in (("signed", False), ("absolute", True)):
        simulated = path_statistics.sample_acf(replicated[:, :, 0], absolute=absolute)
        actual = path_statistics.sample_acf(observed[:, 0], absolute=absolute)
        intervention = path_statistics.sample_acf(shuffled[:, 0], absolute=absolute)
        result[key] = {"replicated": simulated["values"], "available": simulated["available"],
                       "available_count": simulated["available_count"], "count": simulated["count"],
                       "observed": actual["values"][0], "observed_available": bool(actual["available"][0]),
                       "shuffled": intervention["values"][0],
                       "shuffled_available": bool(intervention["available"][0])}
        result["lags"] = simulated["lags"]
    return result


def replicate_check(task, observed, posterior_draws, length, count=32, seed=20261113):
    """Generate fresh complete histories at selected paired physical posterior rows.

    Parameter uncertainty and process randomness are both retained. No quote
    noise is added. The supplied observation is used only for the comparison.
    """
    length = _length(task, length)
    count = simulation._integer(count, "replicate count", minimum=2)
    seed = simulation._integer(seed, "replication seed")
    _require(count <= 32 or count == 256,
             "choose at most 32 illustrative replicas, or the explicit 256-history precision option")
    observed = _observed_prefix(task, observed, length)
    values = _parameter_rows(task, posterior_draws)
    rng = np.random.default_rng(np.random.SeedSequence([0x50504352, seed]))
    size = count if count == 256 else min(count, len(values))
    replace = size > len(values)
    selected = rng.choice(len(values), size=size, replace=replace)
    parameters = values[selected].copy()
    started = time.perf_counter()
    # Keep the full original task: shortening a rough simulation changes its
    # task-keyed random stream and would cease to be this task's replicated data.
    replicated, chunk_seeds = _simulate_chunked(task, parameters, seed)
    seconds = time.perf_counter() - started
    return _result(task, observed, length, replicated, parameters, {
        "seed": seed, "source": "posterior_draws_argument", "requested_count": count,
        "source_row_count": len(values), "posterior_indices": selected.tolist(),
        "selection": "uniform complete rows " + ("with replacement" if replace else "without replacement"),
        "chunk_seeds": chunk_seeds, "simulation_seconds": seconds,
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


def plot_ordering(result, *, show_shuffle=False):
    """Compare temporal dependence, retaining undefined rows in the readout.

    Small posterior panels show their individual curves. Prior panels and the
    explicit 256-history posterior panel show available-only pointwise bands.
    These are descriptive comparisons, not simultaneous coverage or calibration.
    """
    import matplotlib.pyplot as plt

    ordering = result["ordering"]
    lags = ordering["lags"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), layout="constrained")
    for ax, key, title in zip(axes, ("signed", "absolute"),
                              ("Signed-return dependence", "Absolute-return dependence"), strict=True):
        data = ordering[key]
        ax.set(xlim=(.5, max(4.5, len(lags) + .5)), ylim=(-1, 1),
               xlabel="Lag (trading days)", ylabel="Sample autocorrelation", title=title)
        ax.axhline(0, color="#94a3b8", lw=.8)
        if not len(lags):
            ax.text(.5, .55, "At least 21 observed days are needed\nfor this dependence illustration.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
        elif data["available_count"]:
            curves = data["replicated"][data["available"]]
            if result["kind"] == "posterior_predictive" and data["count"] <= 32:
                for curve in curves:
                    ax.plot(lags, curve, color="#2563eb", alpha=.18, lw=.8)
            else:
                low, high = np.quantile(curves, [.05, .95], axis=0)
                ax.fill_between(lags, low, high, color="#2563eb", alpha=.18,
                                label="Pointwise central 90% of available replicas")
            ax.plot(lags, np.median(curves, axis=0), color="#2563eb", lw=1.7,
                    label="Simulation median")
        if data["observed_available"]:
            ax.plot(lags, data["observed"], color="#111827", lw=2, label="Observed prefix")
        if show_shuffle and data["shuffled_available"]:
            ax.plot(lags, data["shuffled"], color="#e77927", lw=1.4, linestyle="--",
                    label="Shuffled prefix (teaching intervention)")
        status = (f"Available replicas: {data['available_count']}/{data['count']}\n"
                  f"Observed ACF: {'available' if data['observed_available'] else 'undefined / insufficient days'}")
        ax.text(.02, .03, status, transform=ax.transAxes, fontsize=7, va="bottom")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6.8, loc="upper right")
    figure.suptitle(f"Ordering check · first {result['length']} days · all histories retained", fontsize=11)
    return figure


def generate_drawdown(task, posterior_draws, length, count=128, seed=20261115):
    """Point-parameter and posterior-mixture scenarios with matched process inputs.

    Complete posterior rows are selected with replacement by a separate RNG.
    Both groups retain process randomness. Each distinct <=32-row chunk uses
    the canonical full task, with its process seed matched between groups.
    This is fresh-origin replication, never a state-conditioned continuation.
    """
    length = _length(task, length)
    count = simulation._integer(count, "drawdown scenario count", minimum=2)
    _require(count in (128, 512), "choose 128 drawdown scenarios, or the explicit 512-history precision option")
    seed = simulation._integer(seed, "drawdown seed")
    values = _parameter_rows(task, posterior_draws)
    rng = np.random.default_rng(np.random.SeedSequence([0x44524457, seed]))
    indices = rng.choice(len(values), size=count, replace=True)
    point = np.mean(values, axis=0)
    groups = {}
    started = time.perf_counter()
    for name, parameters in (("point", np.repeat(point[None, :], count, axis=0)),
                             ("posterior", values[indices].copy())):
        full, chunk_seeds = _simulate_chunked(task, parameters, seed)
        prefix = full[:, :length].copy()
        groups[name] = {"parameters": parameters, "returns": prefix,
                        "maximum_drawdown": path_statistics.maximum_drawdown(prefix[:, :, 0])}
    return {"task": copy.deepcopy(task), "task_fingerprint": simulation.task_fingerprint(task),
            "length": length, "groups": groups, "point_parameters": point,
            "posterior_indices": indices, "metadata": {
                "count": count, "seed": seed, "chunk_seeds": chunk_seeds,
                "full_history_length": task["n_observations"], "source_row_count": len(values),
                "parameter_selection": "uniform complete posterior rows with replacement",
                "point_definition": "arithmetic posterior mean in physical parameter units",
                "simulation_seconds": time.perf_counter() - started,
                "process_matching": "same task, row positions and distinct chunk seeds in both groups",
                "scope": "independent replicated-history drawdown; approximate posterior and finite simulation; not forecasting",
                "initialization": "full task from its declared origin, then comparison prefix",
            }}


def drawdown_readout(result, threshold):
    """Read an event threshold from retained outcomes without new simulations."""
    _require(not isinstance(threshold, (bool, np.bool_))
             and isinstance(threshold, (int, float, np.integer, np.floating))
             and np.isfinite(threshold) and 0 <= threshold <= 1,
             "drawdown threshold must be a fraction between zero and one")
    rows = []
    for group in ("point", "posterior"):
        values = result["groups"][group]["maximum_drawdown"]
        count = int(np.count_nonzero(values > threshold))
        rows.append({"group": group, "count": len(values), "exceedances": count,
                     "proportion": count / len(values), "threshold": float(threshold),
                     "event": "maximum_drawdown > threshold"})
    return rows


def plot_drawdown(result):
    """A few bounded wealth traces and ECDFs of every retained drawdown outcome."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    limit = 3.
    outside = []
    for group, label, color in (("point", "Posterior-mean parameters", "#ea7c32"),
                                ("posterior", "Paired posterior parameters", "#2563eb")):
        data = result["groups"][group]
        log_wealth = path_statistics.cumulative_log_wealth(data["returns"][:, :, 0])
        beyond = log_wealth > np.log(limit)
        outside.append(f"{label}: {np.count_nonzero(np.any(beyond, axis=1))}/{len(log_wealth)} above {limit:g}")
        # Clip for rendering only. All returns and all drawdown outcomes remain
        # unchanged; boundary triangles identify out-of-view displayed values.
        with np.errstate(under="ignore"):
            display = np.exp(np.minimum(log_wealth[:6], np.log(limit)))
        days = np.arange(result["length"] + 1)
        for index, values in enumerate(display):
            axes[0].plot(days, values, color=color, alpha=.6, lw=.9,
                         label=label if index == 0 else None)
            axes[0].plot(days[beyond[index]], np.full(np.count_nonzero(beyond[index]), limit),
                         "^", color=color, markersize=3, alpha=.4)
        mdd = np.sort(data["maximum_drawdown"])
        axes[1].step(np.r_[0., 100 * mdd], np.r_[0., np.arange(1, len(mdd) + 1) / len(mdd)],
                     where="post", color=color, lw=1.8, label=f"{label} (all {len(mdd)})")
    axes[0].set(xlim=(0, result["length"]), ylim=(0, limit), xlabel="Trading day",
                ylabel="Wealth from one", title="Six fresh histories per group; fixed wealth view")
    axes[0].text(.02, .03, "All scenarios, including traces not shown:\n" + "\n".join(outside),
                 transform=axes[0].transAxes, fontsize=6.8, va="bottom")
    axes[1].set(xlim=(0, 100), ylim=(0, 1), xlabel="Maximum drawdown (%)",
                ylabel="Fraction at or below", title="Drawdown ECDF: every scenario retained")
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[1].legend(fontsize=6.5, loc="lower right")
    figure.suptitle(f"Independent replicated-history drawdown · first {result['length']} days", fontsize=11)
    return figure

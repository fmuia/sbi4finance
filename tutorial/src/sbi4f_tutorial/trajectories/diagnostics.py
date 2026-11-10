"""Explicit held-out trajectory checks for the fitted model and its exact P task.

Full validation histories are generated once in memory and reused at different
prefix lengths. Generating parameters are labels, never conditioning inputs.
"""

import copy
import time

import numpy as np

from . import inference, simulation


_FIELDS = {"observations", "parameters", "trajectory_id"}
_LABELS = {
    "kappa": "Mean reversion κ", "long_variance": "Long-run variance",
    "vol_of_vol": "Volatility of variance", "rho": "Correlation ρ",
    "hurst": "Roughness H", "eta": "Variance amplitude η",
    "short_variance": "Initial variance level", "decay": "Curve decay",
    "drift": "Physical drift μ",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real(values, shape, name):
    _require(not np.iscomplexobj(values) and not np.ma.isMaskedArray(values),
             f"{name} must be real unmasked arrays")
    values = np.asarray(values)
    _require(values.shape == shape and values.dtype.kind in "fiu" and np.isfinite(values).all(),
             f"{name} must have finite physical-unit shape {shape}")
    return values


def _panel(task, panel):
    """Validate a complete task-bound panel; never slice one task into another."""
    simulation.validate_task(task)
    _require(isinstance(panel, dict)
             and set(panel) == {"task", "task_fingerprint", "data", "metadata"},
             "expected a task-bound trajectory diagnostic panel")
    simulation.validate_task(panel["task"])
    digest = simulation.task_fingerprint(task)
    _require(panel["task"] == task and panel["task_fingerprint"] == digest,
             "diagnostic panel and selected model tasks differ; regenerate for this complete task")
    metadata = panel["metadata"]
    _require(isinstance(metadata, dict) and metadata.get("format") == "trajectory_diagnostic_panel_v1"
             and metadata.get("task_fingerprint") == digest
             and metadata.get("source_split") == "validation" and metadata.get("start") == 0
             and metadata.get("backend") == "numpy",
             "trajectory diagnostic generation identity differs")
    n = simulation._integer(metadata.get("count"), "diagnostic count", minimum=2)
    seed = simulation._integer(metadata.get("seed"), "diagnostic seed")
    _require(n <= 200, "session diagnostics are capped at 200 complete histories")
    data = panel["data"]
    _require(isinstance(data, dict) and set(data) == _FIELDS,
             "diagnostic data must contain only returns, parameter labels and trajectory IDs")
    maximum, dimensions = task["n_observations"], len(task["parameter_names"])
    observations = _real(data["observations"], (n, maximum, 1), "diagnostic returns")
    parameters = _real(data["parameters"], (n, dimensions), "generating parameters")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(parameters > bounds[:, 0]) and np.all(parameters < bounds[:, 1]),
             "generating parameters must remain strictly inside the task's prior bounds")
    identities = np.asarray(data["trajectory_id"])
    expected_ids = [f"{digest[:16]}:{seed}:validation:{i}" for i in range(n)]
    _require(identities.shape == (n,) and identities.dtype.kind == "U"
             and identities.tolist() == expected_ids and len(np.unique(identities)) == n,
             "diagnostic histories must retain all canonical validation IDs in generation order")
    return {"observations": observations, "parameters": parameters, "trajectory_id": identities}


def generate_panel(task, count=200, seed=20261111):
    """Generate all independent prior-predictive histories, without saving assets.

    Every call uses the declared full task and the validation RNG domain. A
    numerical failure raises; there is no redraw, filtering or partial fallback.
    """
    simulation.validate_task(task)
    count = simulation._integer(count, "diagnostic count", minimum=2)
    seed = simulation._integer(seed, "diagnostic seed")
    _require(count <= 200, "session diagnostics are capped at 200 complete histories")
    snapshot = copy.deepcopy(task)
    digest = simulation.task_fingerprint(snapshot)
    started = time.perf_counter()
    data = simulation.generate(snapshot, "validation", 0, count, seed=seed, backend="numpy")
    metadata = {
        "format": "trajectory_diagnostic_panel_v1", "task_fingerprint": digest,
        "count": count, "seed": seed, "source_split": "validation", "start": 0,
        "backend": "numpy", "generation_seconds": time.perf_counter() - started,
        "max_history_length": snapshot["n_observations"], "measure": "P",
        "scope": "session-only independent validation simulations; no accuracy or calibration claim",
    }
    panel = {"task": snapshot, "task_fingerprint": digest, "data": data, "metadata": metadata}
    _panel(snapshot, panel)
    return panel


def evaluate(workflow, task, panel, length, draws=200, seed=20261111):
    """Infer from exactly the chosen prefix; retain all paired physical draws."""
    data = _panel(task, panel)
    length = simulation._integer(length, "diagnostic history length", minimum=task["min_history"])
    _require(length <= task["n_observations"], "diagnostic history exceeds this task's maximum")
    draws = simulation._integer(draws, "posterior draws", minimum=2)
    seed = simulation._integer(seed, "sampling seed")
    lengths = np.full(len(data["observations"]), length, dtype=np.int64)
    return inference.sample(workflow, task, data["observations"], draws=draws,
                            lengths=lengths, seed=seed)


def diagnostic_data(task, panel, draws):
    """Named physical estimates/targets for native BayesFlow diagnostics."""
    data = _panel(task, panel)
    values = np.asarray(draws)
    n, p = data["parameters"].shape
    _require(values.ndim == 3 and values.shape[0] == n and values.shape[1] >= 2 and values.shape[2] == p,
             "posterior draws must have shape [histories, draws>=2, parameters]")
    values = _real(draws, values.shape, "posterior draws")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
             "posterior draws must be in physical prior bounds")
    return {
        "estimates": {name: values[:, :, i:i + 1] for i, name in enumerate(task["parameter_names"])},
        "targets": {name: data["parameters"][:, i:i + 1] for i, name in enumerate(task["parameter_names"])},
    }


def plot_diagnostics(task, panel, draws):
    """Recovery with 90% intervals and rank ECDFs; no pass/fail certificate."""
    data = diagnostic_data(task, panel, draws)
    import bayesflow as bf

    from ..shared import plots

    names = task["parameter_names"]
    labels = [_LABELS.get(name, name) for name in names]
    columns = 2 if len(names) == 4 else min(3, len(names))
    recovery = bf.diagnostics.recovery(
        **data, variable_keys=names, variable_names=labels,
        uncertainty_agg_kwargs={"prob": .90}, num_col=columns, color=plots.SERIES[0])
    calibration = bf.diagnostics.calibration_ecdf(
        **data, variable_keys=names, variable_names=labels, difference=True,
        num_col=columns, legend_location="upper right",
        rank_ecdf_color=plots.SERIES[0], fill_color=plots.BASELINE)
    return recovery, calibration

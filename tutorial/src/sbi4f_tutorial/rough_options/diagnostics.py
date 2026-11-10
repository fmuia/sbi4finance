"""Compact held-out markets, reusable across estimators of the same rough task.

Reading inputs and preparing diagnostics require no neural backend. Only the
explicit sampling and plotting actions use a fitted workflow or BayesFlow.
"""

import json
from pathlib import Path

import numpy as np

from ..shared import full_surface, full_surface_teaching, paths, tasks


_FIELDS = {
    "parameters", "observations", "reference_prices", "pricing_standard_errors",
    "context_values", "example_ids", "pricing_paths",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _known_task(task):
    full_surface.validate_task(task)
    _require(full_surface.uses_known_curve(task)
             and task["parameter_names"] == ["hurst", "eta", "rho"],
             "diagnostic inputs require the three-parameter supplied-curve rough task")
    return task


def _integer(value, name, *, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _real(values, shape, name):
    _require(not np.iscomplexobj(values)
             and not (np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values))),
             f"{name} must be unmasked real values")
    array = np.asarray(values)
    _require(array.dtype.kind in "fiu" and array.shape == shape and np.isfinite(array).all(),
             f"{name} must have finite real shape {shape}")
    return np.asarray(array, dtype=np.float64)


def _split(task, split, *, count=None):
    """Validate physical rows without using labels as network conditions."""
    _known_task(task)
    _require(set(split) == _FIELDS, "diagnostic input fields differ from the panel schema")
    observed = full_surface.observations(split["observations"], task)
    n = len(observed)
    _require(n >= 2 and (count is None or n == count), "diagnostic panel row count differs")
    result = {"observations": observed}
    for name, shape in (("parameters", (n, 3)), ("reference_prices", (n, 63)),
                        ("pricing_standard_errors", (n, 63)), ("context_values", (n, 7))):
        result[name] = _real(split[name], shape, name)
    _require(np.all(result["pricing_standard_errors"] >= 0), "Monte Carlo standard errors must be nonnegative")
    # This canonical path validates the positive curve and strictly bounded
    # generating labels. It is used for validation, never passed to sample().
    full_surface.training_data(result, task)
    ids = np.asarray(split["example_ids"])
    _require(ids.shape == (n,) and ids.dtype.kind in "iu" and np.all(ids >= 0)
             and len(np.unique(ids)) == n, "example_ids must be unique nonnegative integer row identities")
    used = np.asarray(split["pricing_paths"])
    blocks = len(task["numerics"]["blocks"])
    _require(used.shape == (n, blocks) and used.dtype.kind in "iu" and np.all(used >= 0),
             "pricing_paths must be nonnegative integer counts per row and block")
    result["example_ids"], result["pricing_paths"] = ids, used
    return result


def _precision(task, metadata, split):
    """Check the declared policy against the task and each block's actual rung."""
    try:
        settings = metadata["generator"]["settings"]
        ladder = settings["paths_ladder"]
        target = settings["max_mc_se_over_quote_sigma"]
        cap = settings["max_mc_se_over_quote_sigma_last_rung"]
    except (KeyError, TypeError) as error:
        raise ValueError("diagnostic generator precision policy is incomplete") from error
    numerics = task["numerics"]
    declared = numerics["paths_ladder"]
    _require(isinstance(ladder, list) and ladder and all(type(n) is int for n in ladder)
             and ladder == declared[:len(ladder)], "diagnostic paths ladder must be a prefix of the task ladder")
    _require(type(target) in (int, float) and target == numerics["max_mc_se_over_quote_sigma"],
             "diagnostic Monte Carlo target differs from the task")
    allowed_cap = (numerics.get("max_mc_se_over_quote_sigma_last_rung")
                   if ladder[-1] == declared[-1] else None)
    _require(cap == allowed_cap and (cap is None or type(cap) in (int, float)),
             "a relaxed precision cap is allowed only on the task's actual final rung")
    contracts = task["context"]["contracts"]
    maturities = np.asarray([c["maturity"] for c in contracts])
    assignment = np.full(len(contracts), -1)
    for block, spec in reversed(list(enumerate(numerics["blocks"]))):
        assignment[maturities <= spec["horizon"] + 1e-12] = block
    _require(np.all(assignment >= 0), "every diagnostic contract needs its declared pricing block")
    ratios = split["pricing_standard_errors"] / tasks.sigma(task)
    for block in range(len(numerics["blocks"])):
        chosen = assignment == block
        used = split["pricing_paths"][:, block]
        if not np.any(chosen):
            _require(np.all(used == 0), "unused pricing blocks must have zero paths")
            continue
        _require(np.isin(used, ladder).all(), "pricing paths are outside the recorded task ladder")
        limit = np.full(len(used), float(target))
        if cap is not None:
            limit[used == declared[-1]] = cap
        _require(np.all(ratios[:, chosen] <= limit[:, None] + 8 * np.finfo(float).eps),
                 "a diagnostic market exceeds its block's Monte Carlo precision budget")


def load_panel(directory, expected_task):
    """Read verified held-out inputs within tutorial/assets, without model weights.

    A task fixes priors, grid, curve conditions, noise and numerical settings.
    Input identity therefore does not depend on a fitted architecture/checkpoint.
    """
    directory = Path(directory).expanduser().resolve()
    assets = paths.ASSETS.resolve()
    _require(directory != assets and directory.is_relative_to(assets),
             "diagnostic panels must remain under tutorial/assets/")
    for name in ("panel.json", "task.json", "inputs.npz"):
        file = directory / name
        _require(file.is_file() and file.resolve().is_relative_to(directory),
                 f"diagnostic file must remain inside its panel: {name}")
    metadata = json.loads((directory / "panel.json").read_text())
    _require(isinstance(metadata, dict) and metadata.get("format") == "rough_diagnostic_panel_v1",
             "expected a rough diagnostic panel")
    n = _integer(metadata.get("count"), "panel count", minimum=2)
    _require(metadata.get("scientific_acceptance") is False
             and metadata.get("reserved_test_evaluated") is False,
             "diagnostic inputs do not establish scientific acceptance or a reserved-test result")
    files = metadata.get("files", {})
    _require(isinstance(files, dict) and set(files) == {"task.json", "inputs.npz"},
             "diagnostic payload file inventory differs")
    for name, digest in files.items():
        _require(isinstance(digest, str) and full_surface_teaching.digest(directory / name) == digest,
                 f"diagnostic checksum mismatch: {name}")
    task = _known_task(json.loads((directory / "task.json").read_text()))
    _known_task(expected_task)
    _require(task == expected_task
             and metadata.get("task_fingerprint") == full_surface_teaching.fingerprint(task)
             and metadata.get("parameter_names") == task["parameter_names"],
             "diagnostic task, parameter order or expected task differs")
    generator = metadata.get("generator")
    _require(isinstance(generator, dict), "diagnostic generator record is missing")
    _integer(generator.get("seed"), "generator seed")
    with np.load(directory / "inputs.npz", allow_pickle=False) as saved:
        split = _split(task, {name: saved[name] for name in saved.files}, count=n)
    _precision(task, metadata, split)
    return task, metadata, split


def evaluate(workflow, task, split, draws=200, seed=20261110):
    """Sample this workflow on the fixed inputs; generating truths stay separate."""
    data = _split(task, split)
    _integer(draws, "posterior draw count", minimum=2)
    _integer(seed, "sampling seed")
    return full_surface.sample(workflow, task, data["observations"], num_samples=draws,
                               context_values=data["context_values"], seed=seed)


def diagnostic_data(task, split, draws):
    """Prepare physical-unit targets and paired estimates for native diagnostics."""
    data = _split(task, split)
    values = np.asarray(draws)
    _require(values.ndim == 3 and values.shape[0] == len(data["parameters"])
             and values.shape[2] == 3 and values.shape[1] >= 2,
             "posterior draws must have shape [markets, draws>=2, 3]")
    values = _real(draws, values.shape, "posterior draws")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
             "posterior draws are outside physical prior bounds")
    return {
        "estimates": {name: values[:, :, i:i + 1] for i, name in enumerate(task["parameter_names"])},
        "targets": {name: data["parameters"][:, i:i + 1] for i, name in enumerate(task["parameter_names"])},
    }


def plot_diagnostics(task, split, draws):
    """Native recovery and rank ECDF figures; neither is a model-quality verdict."""
    data = diagnostic_data(task, split, draws)
    import bayesflow as bf

    from ..shared import plots

    names = task["parameter_names"]
    labels = ["Roughness H", "Variance amplitude η", "Price–variance correlation ρ"]
    recovery = bf.diagnostics.recovery(
        **data, variable_keys=names, variable_names=labels,
        uncertainty_agg_kwargs={"prob": .90}, color=plots.SERIES[0])
    calibration = bf.diagnostics.calibration_ecdf(
        **data, variable_keys=names, variable_names=labels, difference=True,
        legend_location="upper right", rank_ecdf_color=plots.SERIES[0], fill_color=plots.BASELINE)
    return recovery, calibration

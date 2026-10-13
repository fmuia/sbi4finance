"""Restricted joint HKDE inference from a fixed, ordered currency-price panel.

The two targets are theta and jump intensity under Q, with the independent
uniform prior declared by the option-bank task. Contracts, rates, spot, quote
noise, initial variance, other model parameters and pricing settings are fixed
checkpoint context, not additional network inputs. The caller must compare the
complete saved task with the application task before loading a checkpoint:
matching the array shape or adapter ID alone is insufficient.

Framework imports are deferred until workflow construction. No market data,
pricing labels, numerical diagnostics or simulator truth enter conditioning.
"""

import numpy as np

from .option_bank import validate_config


ADAPTER_ID = "hkde_ordered_currency_panel_v1"
PARAMETER_NAMES = ("theta", "jump_intensity")


def _task_design(task):
    if (task.get("task_id") != "hkde_option_panel_v1"
            or task.get("model") != "hkde" or task.get("measure") != "Q"
            or task.get("parameter_names") != list(PARAMETER_NAMES)):
        raise ValueError("expected the HKDE Q task with ordered theta, jump_intensity targets")
    # The bank schema defines independent uniforms and validates the complete
    # known context and pricing design. Counts and seed are irrelevant here.
    validate_config({"task": task, "root_seed": 0,
                     "splits": {"train": 1, "validation": 1, "test": 1}})
    if len(task["context"]["contracts"]) != 9:
        raise ValueError("the restricted HKDE adapter requires nine ordered contracts")
    return np.asarray(task["prior_bounds"], dtype=np.float64)


def conditioning_data(observations, task):
    """Return only ordered noisy prices [B,9], without per-panel scaling.

    Input is the bank's [B,9,1] currency-price array. Contract order must agree
    with the saved task; reordering prices alone changes their interpretation.
    Negative noisy quotes are valid under the declared additive Gaussian model.
    Unlabelled observations use this same boundary without parameter targets.
    """
    _task_design(task)
    prices = np.asarray(observations, dtype=np.float64)
    if (prices.ndim != 3 or prices.shape[0] == 0 or prices.shape[1:] != (9, 1)
            or not np.isfinite(prices).all()):
        raise ValueError("complete finite [batch,9,1] currency prices are required")
    return {"observations": prices[..., 0]}


def training_data(split, task):
    """Add synthetic parameters as separate targets for one joint distribution.

    The remaining bank arrays are intentionally ignored. Open bounds are
    required by the logit transform; do not clip or redraw boundary targets.
    """
    result = conditioning_data(split["observations"], task)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    parameters = np.asarray(split["parameters"], dtype=np.float64)
    if (parameters.shape != (len(result["observations"]), 2)
            or not np.isfinite(parameters).all()):
        raise ValueError("synthetic parameter targets must be finite [batch,2]")
    if np.any(parameters <= bounds[:, 0]) or np.any(parameters >= bounds[:, 1]):
        raise ValueError("parameter targets must lie strictly inside prior_bounds")
    return {**result, **{name: parameters[:, index:index + 1]
                        for index, name in enumerate(PARAMETER_NAMES)}}


def build_workflow(task, checkpoint_dir=None):
    """Small spline coupling flow with train-only global normalization.

    Built-in logit constraints act on float64 before neural float32 casting.
    The adapter reverses both constraints during sampling, returning named
    raw-unit arrays [batch,draw,1]. Keep the common draw index when jointly
    propagating theta and jump intensity into downstream model prices.

    All nine prices condition the flow directly; there is no learned encoder.
    Normalization is global across training panels and freezes at inference.
    """
    bounds = _task_design(task)
    import bayesflow as bf

    adapter = bf.Adapter()
    for name, (lower, upper) in zip(PARAMETER_NAMES, bounds, strict=True):
        adapter = adapter.constrain(name, lower=float(lower), upper=float(upper),
                                    inclusive="none")
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(list(PARAMETER_NAMES), into="inference_variables")
               .rename("observations", "inference_conditions"))
    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.CouplingFlow(
            depth=4, transform="spline", subnet_kwargs={"widths": [64, 64]},
            transform_kwargs={"bins": 8},
        ),
        standardize="all",
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )

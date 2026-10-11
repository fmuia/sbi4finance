"""Ordered-return inference for theta and the signed Heston continuation state.

This is the restricted ``heston_theta_state_v1`` task. Fixed model parameters,
initialization, prior and discretization belong to its saved task configuration;
a checkpoint may only be applied with that same configuration. Framework
imports are deferred so conditioning/target preparation needs only NumPy.
"""

import numpy as np


ADAPTER_ID = "heston_raw_and_window_moments_v1"


def _task_design(task):
    if (task.get("task_id") != "heston_theta_state_v1" or task.get("measure") != "P"
            or task.get("prior") != "log_uniform"):
        raise ValueError("expected the heston_theta_state_v1 P task with log_uniform theta prior")
    substeps = task.get("substeps")
    if (isinstance(substeps, (bool, np.bool_))
            or not isinstance(substeps, (int, np.integer)) or substeps != 1):
        raise ValueError("heston_theta_state_v1 requires substeps == 1")
    bounds = np.asarray(task["theta_bounds"], dtype=np.float64)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] < bounds[1]):
        raise ValueError("theta_bounds must be two finite, strictly increasing positive values")
    count = task["n_observations"]
    if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
            or count < 1):
        raise ValueError("n_observations must be a positive integer")
    step = task["dt_years"]
    if isinstance(step, (bool, np.bool_)) or np.ndim(step) != 0 or not np.isfinite(step) or step <= 0:
        raise ValueError("dt_years must be a positive finite scalar")
    return bounds, int(count), float(step)


def conditioning_data(log_returns, dt_years, task):
    """Select the ordered history and observation-derived trailing moments.

    Returns unchanged ``observations [B,T,1]`` plus ``history_features [B,8]``.
    For trailing lengths ``min(8,T), min(32,T), min(128,T), T``, the features
    interleave mean log return and mean squared log return divided by dt.
    Duplicate short-history windows keep the feature count fixed. These are
    not sufficient statistics: the GRU also receives the complete raw history.
    They do not standardize individual windows or use targets/future outcomes.

    The fixed time design must match. The caller must additionally match the
    checkpoint's full saved task and adapter ID, including fixed parameters,
    theta prior, initialization and integration substeps.
    """
    _, count, step = _task_design(task)
    observations = np.asarray(log_returns, dtype=np.float64)
    if (observations.ndim != 3 or observations.shape[0] == 0
            or observations.shape[1:] != (count, 1)
            or not np.isfinite(observations).all()):
        raise ValueError("complete finite [batch,time,1] log returns are required")
    if not np.array_equal(np.asarray(dt_years, dtype=np.float64), np.full(count, step)):
        raise ValueError("time grid does not match the fixed checkpoint task")
    features = []
    for width in (min(8, count), min(32, count), min(128, count), count):
        window = observations[:, -width:, 0]
        features.extend((window.mean(axis=1), np.square(window).mean(axis=1) / step))
    return {"observations": observations,
            "history_features": np.stack(features, axis=1)}


def training_data(split, task):
    """Add theta and SIGNED terminal auxiliary variance as separate targets.

    The network subsequently learns their joint distribution. No logarithm or
    positivity constraint is applied to the state: clipping it would change the
    full-truncation transition used by conditional forecasts.
    """
    result = conditioning_data(split["observations"], split["dt_years"], task)
    bounds, _, _ = _task_design(task)
    shape = (len(result["observations"]), 1)
    theta = np.asarray(split["parameters"], dtype=np.float64)
    state = np.asarray(split["terminal_state"], dtype=np.float64)
    for name, values in (("theta", theta), ("terminal_state", state)):
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{name} targets must be finite [batch,1]")
    if np.any(theta <= bounds[0]) or np.any(theta >= bounds[1]):
        raise ValueError("theta targets must lie strictly inside theta_bounds")
    return {**result, "theta": theta, "terminal_state": state}


def build_workflow(task, checkpoint_dir=None):
    """Joint flow conditioned on a GRU summary and observed trailing moments.

    The built-in logit constraint is applied in float64 before neural casting.
    The state is an identity transform. Sampling reverses the saved adapter and
    returns matched raw-unit ``theta`` and ``terminal_state`` arrays: keep their
    common draw index when forecasting. Training-only global normalization of
    targets, raw returns and moment features preserves between-window scale;
    there is no per-window standardization or conditioning on future outcomes.
    """
    bounds, _, _ = _task_design(task)
    import bayesflow as bf

    adapter = (bf.Adapter()
               .constrain("theta", lower=float(bounds[0]), upper=float(bounds[1]), inclusive="none")
               .convert_dtype("float64", "float32")
               .concatenate(["theta", "terminal_state"], into="inference_variables")
               .rename("observations", "summary_variables")
               .rename("history_features", "inference_conditions"))
    summary = bf.networks.RecurrentNetwork(
        summary_dim=16, hidden_dim=32, recurrent_type="gru", bidirectional=False,
        time_embed_dim=4, dropout=0.0, return_sequences=False,
    )
    return bf.BasicWorkflow(
        adapter=adapter, summary_network=summary,
        inference_network=bf.networks.FlowMatching(
            subnet_kwargs={"widths": [32, 32]}, integrate_kwargs={"steps": 16}),
        standardize="all",
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )

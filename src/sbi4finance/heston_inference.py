"""Ordered-return inference for theta and the signed Heston continuation state.

This is the restricted ``heston_theta_state_v1`` task. Fixed model parameters,
initialization, prior and discretization belong to its saved task configuration;
a checkpoint may only be applied with that same configuration. Framework
imports are deferred so conditioning/target preparation needs only NumPy.
"""

import numpy as np


ADAPTER_ID = "heston_raw_and_window_moments_v1"
TIME_ADAPTER_ID = "heston_returns_time_and_window_moments_v2"


def _check_adapter(adapter_id):
    if adapter_id not in (ADAPTER_ID, TIME_ADAPTER_ID):
        raise ValueError("unsupported Heston adapter_id")


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


def conditioning_data(log_returns, dt_years, task, *, adapter_id=ADAPTER_ID):
    """Select the ordered history and observation-derived trailing moments.

    V1 returns unchanged ``observations [B,T,1]``. V2 appends known observation
    end times in years as channel 1, giving ``observations [B,T,2]`` for the
    neural adapter only; the bank still stores raw returns [B,T,1]. Training
    standardization puts this fixed time channel on an O(1) scale before the
    recurrent network's Time2Vec layer. Both return ``history_features [B,8]``.
    For trailing lengths ``min(8,T), min(32,T), min(128,T), T``, the features
    interleave mean log return and mean squared log return divided by dt.
    Duplicate short-history windows keep the feature count fixed. These are
    not sufficient statistics: the GRU also receives the complete raw history.
    They do not standardize individual windows or use targets/future outcomes.

    The fixed time design must match. The caller must additionally match the
    checkpoint's full saved task and adapter ID, including fixed parameters,
    theta prior, initialization and integration substeps.
    """
    _check_adapter(adapter_id)
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
    if adapter_id == TIME_ADAPTER_ID:
        end_times = np.cumsum(np.asarray(dt_years, dtype=np.float64))
        time_channel = np.broadcast_to(end_times[None, :, None], observations.shape)
        observations = np.concatenate([observations, time_channel], axis=-1)
    return {"observations": observations,
            "history_features": np.stack(features, axis=1)}


def training_data(split, task, *, adapter_id=ADAPTER_ID):
    """Add theta and SIGNED terminal auxiliary variance as separate targets.

    The network subsequently learns their joint distribution. No logarithm or
    positivity constraint is applied to the state: clipping it would change the
    full-truncation transition used by conditional forecasts.
    """
    result = conditioning_data(split["observations"], split["dt_years"], task,
                               adapter_id=adapter_id)
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


def build_workflow(task, checkpoint_dir=None, *, density_estimator="flow_matching",
                   adapter_id=ADAPTER_ID):
    """Joint flow conditioned on a GRU summary and observed trailing moments.

    The built-in logit constraint is applied in float64 before neural casting.
    The state is an identity transform. Sampling reverses the saved adapter and
    returns matched raw-unit ``theta`` and ``terminal_state`` arrays: keep their
    common draw index when forecasting. Training-only global normalization of
    targets, raw returns and moment features preserves between-window scale;
    there is no per-window standardization or conditioning on future outcomes.
    """
    _check_adapter(adapter_id)
    bounds, _, _ = _task_design(task)
    if density_estimator not in ("flow_matching", "spline"):
        raise ValueError("density_estimator must be flow_matching or spline")
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
        time_axis=1 if adapter_id == TIME_ADAPTER_ID else None,
    )
    if density_estimator == "spline":
        inference = bf.networks.CouplingFlow(
            depth=4, transform="spline", subnet_kwargs={"widths": [32, 32]},
            transform_kwargs={"bins": 8},
        )
    else:
        inference = bf.networks.FlowMatching(
            subnet_kwargs={"widths": [32, 32]}, integrate_kwargs={"steps": 16})
    return bf.BasicWorkflow(
        adapter=adapter, summary_network=summary,
        inference_network=inference,
        standardize="all",
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )

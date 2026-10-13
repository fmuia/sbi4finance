"""Explicit observation selection and a tiny BayesFlow GBM workflow.

No hidden state or raw-history encoder is needed for this one-parameter task.
Imports of the neural framework are deferred so array preparation stays light.
"""

import numpy as np


def conditioning_data(log_returns, dt_years, task):
    """Select observations only; require the checkpoint's fixed time design.

    For known constant volatility and fixed elapsed time, sum(log returns) is
    sufficient for drift. This reduction is special to GBM, not SVJ/Bergomi.
    Truth, IDs, prices, future outcomes and other metadata are not arguments.
    The caller must also match the declared P task, prior and known volatility.
    """
    values = np.asarray(log_returns, dtype=np.float64)
    dt = np.asarray(dt_years, dtype=np.float64)
    expected_dt = np.full(task["n_observations"], task["dt_years"])
    if (values.ndim != 3 or values.shape[0] == 0
            or values.shape[1:] != (task["n_observations"], 1)
            or not np.isfinite(values).all()):
        raise ValueError("complete finite [batch,time,1] log returns are required")
    if not np.array_equal(dt, expected_dt):
        raise ValueError("time grid does not match the fixed checkpoint task")
    return {"sum_log_returns": values.sum(axis=1)}


def training_data(split, task):
    """Add synthetic drift as the supervised target, never as a condition."""
    result = conditioning_data(split["observations"], split["dt_years"], task)
    parameters = np.asarray(split["parameters"], dtype=np.float64)
    if parameters.shape != result["sum_log_returns"].shape or not np.isfinite(parameters).all():
        raise ValueError("synthetic drift targets must be finite [batch,1]")
    return {**result, "mu": parameters}


def build_workflow(checkpoint_dir=None):
    """Pinned BayesFlow 2.0.14 API; normalization sees training batches only."""
    import bayesflow as bf

    adapter = (bf.Adapter().convert_dtype("float64", "float32")
               .rename("mu", "inference_variables")
               .rename("sum_log_returns", "inference_conditions"))
    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(
            subnet_kwargs={"widths": [32, 32]}, integrate_kwargs={"steps": 16}),
        standardize="all", checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )

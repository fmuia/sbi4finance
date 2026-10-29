"""Analytic mean-variance products for the fixed-kappa Heston teaching task.

These functionals test inferred curve uncertainty. They do not test leverage,
vol-of-vol or path-dependent prices, and are not volatility-swap prices.
"""

import numpy as np

from . import tasks


def heston_variance_functionals(task, draws):
    """Return annualized variance-swap rates, preserving all draw axes.

    With continuous monitoring and the declared zero-jump Heston model,
    E[V(t)] = theta + (v0-theta)*exp(-kappa*t). Integrating this expectation
    prices the realized-variance rate without Monte Carlo path error. A
    forward rate averages it over years 0.5 to 1.0. Units are variance/year;
    no square root, discount factor, notional or percentage-point scaling.
    """
    tasks.validate_task(task)
    if (task["model"] != "hkde" or "kappa" not in task["fixed_parameters"]
            or not {"initial_variance", "theta"} <= set(task["parameter_names"])):
        raise ValueError("the zero-jump Heston task with inferred variance levels and fixed kappa is required")
    if np.ma.isMaskedArray(draws) and np.ma.getmaskarray(draws).any():
        raise ValueError("complete unmasked parameter draws required")
    values = np.asarray(draws)
    names = task["parameter_names"]
    if (values.ndim < 1 or values.shape[-1] != len(names)
            or np.iscomplexobj(values) or not np.isfinite(values).all()):
        raise ValueError("finite real draws in the task's parameter order required")
    bounds = np.asarray(task["prior_bounds"])
    if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
        raise ValueError("draws must lie within the task's prior support")
    v0, theta = values[..., names.index("initial_variance")], values[..., names.index("theta")]
    kappa = task["fixed_parameters"]["kappa"]

    def average(start, end):
        short_weight = np.exp(-kappa*start) * (-np.expm1(-kappa*(end-start))) / (kappa*(end-start))
        return short_weight*v0 + (1-short_weight)*theta

    return {"variance_swap_1y": average(0.0, 1.0),
            "forward_variance_0_5_to_1y": average(0.5, 1.0)}

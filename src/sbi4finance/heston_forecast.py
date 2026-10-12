"""Continue paired theta/auxiliary-state draws for the fixed Heston task."""

import numpy as np

from . import heston


def simulate_futures(joint_draws, task, *, steps=10, seed=912):
    """Return future paths and losses, keeping each parameter/state pair intact.

    Columns are theta and SIGNED terminal auxiliary variance. These are joint
    posterior draws, not two independently sampled marginals. Future inputs
    are independent of history-generation inputs. Outputs are predictions,
    never additional conditioning for the posterior that supplied the draws.
    """
    draws = np.asarray(joint_draws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] == 0 or draws.shape[1] != 2 or not np.isfinite(draws).all():
        raise ValueError("joint_draws must be finite [draw,2] theta/state pairs")
    if np.any(draws[:, 0] <= 0):
        raise ValueError("theta must be positive; signed continuation state is allowed")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if task["measure"] != "P" or task["substeps"] != 1 or task["task_id"] != "heston_theta_state_v1":
        raise ValueError("forecast requires the fixed one-step P Heston task")
    if set(task["parameters"]) != {"drift", "kappa", "vol_of_vol", "rho"}:
        raise ValueError("forecast supports only the declared no-jump fixed parameters")
    parameters = {**task["parameters"], "theta": draws[:, 0]}
    dt = np.full(steps, task["dt_years"])
    innovations = heston.draw_innovations(dt, 1, seed=seed,
                                         example_ids=np.arange(len(draws)), stream_id=91)
    result = heston.simulate(parameters, dt, draws[:, 1], innovations)
    result["relative_loss"] = -np.expm1(result["log_returns"].sum(axis=1)[:, 0])
    return result

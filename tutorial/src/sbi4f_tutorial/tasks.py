"""Reading a calibration task: the dictionary in a checkpoint's task.json that fixes contracts, noise and priors.

A task lists the ordered design contracts, one quote-noise standard deviation per contract, the inferred
parameters with their uniform prior box, and the fixed ones. Rough Bergomi adds the knots of its piecewise-
constant forward-variance curve, whose levels are inputs of each smile; the Heston check has none.
The lab's sbi4finance.calibration.tasks builds and validates these tasks; here they are only read.
"""

import numpy as np

from . import surface


def sigma(task):
    """Quote-noise standard deviation of each contract, in forward units."""
    return np.asarray(task["context"]["quote_noise_std"], dtype=float)


def knots(task):
    return np.asarray(task["model_context"].get("forward_variance_knots", []), dtype=float)


def context_size(task):
    """Number of forward-variance levels per smile: one per interval between knots for rough, none for Heston."""
    return max(len(knots(task)) - 1, 0)


def model_parameters(task, values):
    """The full model dictionary for one row of inferred parameter values."""
    return dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], (float(v) for v in values))))


def rough_design(days=surface.DESIGN_DAYS):
    """Contracts, quote noise, curve knots and simulation blocks of the rough-Bergomi task on the design grid.

    The blocks are the bank's: one grid per maturity group, each starting at zero, with 8 steps a day up to
    two weeks, 8 a day up to three months and 2 a day beyond.
    """
    contracts = surface.design_grid(days)
    blocks = []
    for limit, per_day in ((10, 8), (63, 8), (max(days), 2)):
        inside = [d for d in days if d <= limit and (not blocks or d > blocks[-1]["days"])]
        if inside:
            blocks.append({"days": max(inside), "n_steps": per_day * max(inside)})
    return {"contracts": contracts,
            "sigma": np.asarray(surface.quote_noise(contracts), dtype=float),
            "knots": np.array([0.0] + [d / surface.TRADING_DAYS for d in days]),
            "blocks": [{"horizon": b["days"] / surface.TRADING_DAYS, "n_steps": b["n_steps"]} for b in blocks]}

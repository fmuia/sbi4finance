"""Reading a calibration task: the dictionary that fixes contracts, quote noise and priors.

A task lists the ordered design contracts, one quote-noise standard deviation per contract, the inferred
parameters with their uniform prior box, and the fixed ones. Rough Bergomi adds the knots of its piecewise-
constant forward-variance curve, whose levels are inputs of each smile; the Heston check has none.
The lab builds and validates tasks (sbi4finance/calibration/tasks.py). The tutorial reads the ones the lab
stored: the Heston network's task.json, and the rough-Bergomi task in assets/tasks/. The functions are the
lab's, copied unchanged, except rough_task, which reads the stored task instead of building it;
lab/tests/test_tutorial_parity.py checks them.
"""

import json

import numpy as np

from . import pricing
from .paths import TASKS


def knots(task):
    return np.asarray(task["model_context"].get("forward_variance_knots", []), dtype=float)


def context_size(task):
    """Number of per-example context values: forward-variance levels for rough, none for Heston."""
    return max(len(knots(task)) - 1, 0)


def sigma(task):
    return np.asarray(task["context"]["quote_noise_std"], dtype=float)


def model_parameters(task, values):
    """The full model dictionary for one row of inferred parameter values."""
    return dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], (float(v) for v in values))))


def _heston_split(parameters):
    parameters = dict(parameters)
    return parameters, float(parameters.pop("initial_variance"))


def heston_prices(task, parameters, v0, contracts=None):
    """Refined Fourier prices and refined-minus-base differences, escalating the quadrature.

    Rung r is accepted when rungs r and r + 1 both price inside the no-arbitrage
    bounds and agree within the budget; the refined rung's prices are returned.
    Low variance at one week with a steep skew needs the higher rungs.
    """
    contracts = task["context"]["contracts"] if contracts is None else contracts
    numerics = task["numerics"]
    limit = numerics["max_refinement_over_quote_sigma"] * sigma(task)
    ladder = numerics["quadrature_ladder"]
    failure = None
    for (nodes, cutoff), (next_nodes, next_cutoff) in zip(ladder, ladder[1:]):
        try:
            base = pricing.heston_surface(parameters, v0, contracts, nodes=nodes, upper_limit=cutoff)
            refined = pricing.heston_surface(parameters, v0, contracts, nodes=next_nodes, upper_limit=next_cutoff)
        except ValueError as error:
            failure = error
            continue
        if len(contracts) != len(limit) or np.all(np.abs(refined - base) <= limit):
            return refined, refined - base
        failure = ValueError("Fourier refinement budget failed")
    raise ValueError(f"{failure} on every quadrature rung")


def curve(task, context_values):
    return pricing.step_curve(knots(task), context_values)


def rough_task():
    """The rough-Bergomi task of the pilot and of slot 5's live smile, as the lab stored it.

    The lab's tasks.rough_task() builds and validates it; the tutorial reads the lab's copy,
    configs/tasks/calibration_rough_pilot.json, from assets/tasks/.
    """
    return json.loads((TASKS / "calibration_rough_pilot.json").read_text())["task"]

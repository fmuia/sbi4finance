"""Reading a calibration task: the dictionary that fixes contracts, quote noise and priors.

A task lists the ordered design contracts, one quote-noise standard deviation per contract, the inferred
parameters with their uniform prior box, and the fixed ones. Rough Bergomi adds the knots of its piecewise-
constant forward-variance curve, whose levels are inputs of each smile; the Heston check has none.
These validation and pricing functions are shared by the lessons and lab.
The participant rough_task reader loads the published task from assets/tasks/;
the lab additionally owns task builders and disk-bank generation.
"""

import json

import numpy as np

from . import pricing, surface
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
    parameters = dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], (float(v) for v in values))))
    if is_joint_curve_task(task):
        return {name: parameters[name] for name in ("hurst", "eta", "rho")}
    return parameters


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
    if is_joint_curve_task(task):
        raise ValueError("joint rough pricing requires joint_curve(task, parameter_values), not curve context")
    return pricing.step_curve(knots(task), context_values)


def rough_task():
    """The rough-Bergomi task of the pilot and of slot 5's live smile, as the lab stored it.

    The lab's tasks.rough_task() builds and validates it; the tutorial reads the lab's copy,
    configs/tasks/calibration_rough_pilot.json, from assets/tasks/.
    """
    return json.loads((TASKS / "calibration_rough_pilot.json").read_text())["task"]


FAMILY = "_calibration_v1"


JOINT_ROUGH_ID = "rough_joint_calibration_v1"


JOINT_ROUGH_PARAMETERS = ["hurst", "eta", "rho", "short_variance", "long_variance", "decay"]


HKDE_PARAMETERS = {"kappa", "theta", "vol_of_vol", "rho", "jump_intensity",
                   "jump_up_probability", "jump_up_rate", "jump_down_rate"}


UNIVERSE = {"rough": {"hurst", "eta", "rho"}, "hkde": HKDE_PARAMETERS | {"initial_variance"}}


TASK_KEYS = {"task_id", "model", "measure", "context", "model_context", "parameter_names",
             "prior_bounds", "fixed_parameters", "numerics", "context_prior"}


CURVE_KINDS = {"linear_wobble": {"level_bounds", "slope_bounds", "log_perturbation"},
               "power_law_walk": {"kind", "level_bounds", "exponent_bounds", "walk_sd", "vol_bounds"}}


def is_calibration_task(task):
    return isinstance(task, dict) and str(task.get("task_id", "")).endswith(FAMILY)


def is_joint_curve_task(task):
    """The additive six-target task; stored derived curve arrays are auxiliary, never conditions."""
    return isinstance(task, dict) and task.get("task_id") == JOINT_ROUGH_ID


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _interval(value, name):
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
        raise ValueError(f"{name} must be an increasing finite pair")
    return pair


def validate_context(context):
    """Forward units and one positive quote-noise standard deviation per ordered contract."""
    if set(context) != {"spot", "rate", "dividend", "contracts", "quote_noise_std"}:
        raise ValueError("context needs spot, rate, dividend, contracts, quote_noise_std")
    if (context["spot"], context["rate"], context["dividend"]) != (1.0, 0.0, 0.0):
        raise ValueError("calibration tasks are in forward units: spot 1, rate 0, dividend 0")
    contracts = context["contracts"]
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("contracts must be a nonempty ordered list")
    for contract in contracts:
        if not isinstance(contract, dict) or set(contract) != {"strike", "maturity", "option_type"}:
            raise ValueError("each contract needs strike, maturity, option_type")
        for name in ("strike", "maturity"):
            x = contract[name]
            if isinstance(x, bool) or np.ndim(x) != 0 or not np.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be a positive finite scalar")
        if contract["option_type"] not in ("put", "call"):
            raise ValueError("only European puts and calls are supported")
    noise = context["quote_noise_std"]
    if (not isinstance(noise, list) or len(noise) != len(contracts)
            or any(isinstance(x, bool) or np.ndim(x) != 0 or not np.isfinite(x) or x <= 0 for x in noise)):
        raise ValueError("quote_noise_std needs one positive finite value per contract")


def validate_task(task):
    if is_joint_curve_task(task):
        _validate_joint_task(task)
        return
    if not isinstance(task, dict) or set(task) != TASK_KEYS or task["model"] not in UNIVERSE \
            or task["measure"] != "Q" or task["task_id"] != task["model"] + FAMILY:
        raise ValueError("unsupported calibration task")
    context = task["context"]
    validate_context(context)
    names, fixed = task["parameter_names"], task["fixed_parameters"]
    bounds = np.asarray(task["prior_bounds"], dtype=float)
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not isinstance(fixed, dict) or set(names) & set(fixed)
            or set(names) | set(fixed) != UNIVERSE[task["model"]]
            or bounds.shape != (len(names), 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("explicit parameter order and finite independent-uniform bounds required")
    for corner in bounds.T:
        parameters = model_parameters(task, corner)
        if task["model"] == "hkde":
            from ..heston_options.model import validate_parameters
            parameters, v0 = _heston_split(parameters)
            validate_parameters(parameters)
            if v0 <= 0:
                raise ValueError("initial variance must be positive on the prior box")
        else:
            from ..rough_options.model import _model_parameters
            _model_parameters(parameters, "rough")
    if task["model"] == "hkde" and (fixed.get("jump_intensity") != 0.0):
        raise ValueError("the Heston check fixes jump_intensity at zero")
    maturities = np.array([c["maturity"] for c in context["contracts"]])
    numerics, prior = task["numerics"], task["context_prior"]
    if task["model"] == "hkde":
        if task["model_context"] != {} or prior != {}:
            raise ValueError("the Heston check has no curve context")
        if set(numerics) != {"quadrature_ladder", "max_refinement_over_quote_sigma"}:
            raise ValueError("Heston requires a quadrature ladder and a refinement budget")
        ladder = numerics["quadrature_ladder"]
        if not isinstance(ladder, list) or len(ladder) < 2:
            raise ValueError("the quadrature ladder needs at least a base and a refined rung")
        for (nodes, cutoff), (next_nodes, next_cutoff) in zip(ladder, ladder[1:]):
            _positive_int(nodes, "nodes")
            _positive_int(next_nodes, "nodes")
            if not (next_nodes > nodes and next_cutoff > cutoff > 0):
                raise ValueError("each rung needs more nodes and a larger upper limit than the last")
        budget = numerics["max_refinement_over_quote_sigma"]
    else:
        grid = knots(task)
        if set(task["model_context"]) != {"forward_variance_knots"} or grid.ndim != 1 or len(grid) < 2 \
                or grid[0] != 0 or np.any(np.diff(grid) <= 0) \
                or not np.all(np.isclose(maturities[:, None], grid[None, 1:], rtol=0, atol=1e-12).any(axis=1)):
            raise ValueError("forward-variance knots start at zero, increase, and include every maturity")
        if set(prior) != {"forward_variance"} or not isinstance(prior["forward_variance"], dict):
            raise ValueError("rough calibration needs a forward-variance context prior")
        validate_curve_prior(prior["forward_variance"], grid)
        required = {"method", "blocks", "paths_ladder", "max_mc_se_over_quote_sigma"}
        if not required <= set(numerics) <= required | {"max_mc_se_over_quote_sigma_last_rung"} \
                or numerics["method"] != "hybrid":
            raise ValueError("rough calibration requires hybrid blocks and a paths ladder")
        ladder = [_positive_int(x, "paths_ladder entry") for x in numerics["paths_ladder"]]
        if not ladder or any(x % 2 for x in ladder) or ladder != sorted(set(ladder)):
            raise ValueError("paths ladder must be increasing even path counts")
        covered = np.zeros(len(maturities), dtype=bool)
        previous = 0.0
        for block in numerics["blocks"]:
            if set(block) != {"horizon", "n_steps"} or not block["horizon"] > previous:
                raise ValueError("blocks need increasing horizons and step counts")
            steps = _positive_int(block["n_steps"], "n_steps")
            inside = (maturities > previous + 1e-12) & (maturities <= block["horizon"] + 1e-12)
            raw = maturities[inside] * steps / block["horizon"]
            if not np.allclose(raw, np.rint(raw), rtol=0, atol=1e-9):
                raise ValueError("contracts must land on their block's uniform grid")
            grid_raw = grid[(grid > 0) & (grid <= block["horizon"] + 1e-12)] * steps / block["horizon"]
            if not np.allclose(grid_raw, np.rint(grid_raw), rtol=0, atol=1e-9):
                raise ValueError("forward-variance knots must land on every block grid they fall in")
            covered |= inside
            previous = block["horizon"]
        if not covered.all() or previous < grid[-1] - 1e-12:
            raise ValueError("blocks must cover every contract and the last knot")
        budget = numerics["max_mc_se_over_quote_sigma"]
        cap = numerics.get("max_mc_se_over_quote_sigma_last_rung", budget)
        if isinstance(cap, bool) or np.ndim(cap) != 0 or not np.isfinite(cap) or cap < budget:
            raise ValueError("the last rung's cap must be a number at least the budget")
    if isinstance(budget, bool) or np.ndim(budget) != 0 or not np.isfinite(budget) or budget <= 0:
        raise ValueError("numerical precision budget must be positive")


def _validate_joint_task(task):
    """Reuse the legacy numerical-domain validation without changing legacy task semantics."""
    if (set(task) != TASK_KEYS or task["model"] != "rough" or task["measure"] != "Q"
            or task["parameter_names"] != JOINT_ROUGH_PARAMETERS or task["fixed_parameters"] != {}
            or task["context_prior"] != {} or not isinstance(task["model_context"], dict)
            or set(task["model_context"]) != {"forward_variance_knots", "curve_family"}
            or task["model_context"]["curve_family"] != "heston_exponential_v1"):
        raise ValueError("joint rough task requires six ordered targets and its explicit analytic curve family")
    bounds = np.asarray(task["prior_bounds"], dtype=float)
    if (bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(bounds[3:, 0] <= 0)):
        raise ValueError("six finite increasing uniform bounds with positive curve levels/decay required")
    legacy = {**task, "task_id": "rough" + FAMILY, "parameter_names": JOINT_ROUGH_PARAMETERS[:3],
              "prior_bounds": bounds[:3].tolist(),
              "model_context": {"forward_variance_knots": task["model_context"]["forward_variance_knots"]},
              "context_prior": {"forward_variance": {"level_bounds": [0.1, 0.2],
                                                        "slope_bounds": [0.0, 0.1], "log_perturbation": 0.0}}}
    validate_task(legacy)


def curve_kind(prior):
    return prior.get("kind", "linear_wobble")


def validate_curve_prior(prior, grid):
    """The forward-variance prior's keys and values; grid is the knot vector starting at zero."""
    kind = curve_kind(prior)
    if kind not in CURVE_KINDS or set(prior) != CURVE_KINDS[kind]:
        raise ValueError(f"forward-variance prior needs exactly {sorted(CURVE_KINDS.get(kind, ()))}")
    level = _interval(prior["level_bounds"], "level_bounds")
    if level[0] <= 0:
        raise ValueError("the level bounds must be positive")
    if kind == "linear_wobble":
        slope = _interval(prior["slope_bounds"], "slope_bounds")
        wobble = prior["log_perturbation"]
        if 1 + slope[0] * grid[-1] <= 0 or not np.isfinite(wobble) or wobble < 0:
            raise ValueError("slopes that keep every level positive and a non-negative perturbation required")
        return
    _interval(prior["exponent_bounds"], "exponent_bounds")
    vol = _interval(prior["vol_bounds"], "vol_bounds")
    walk = prior["walk_sd"]
    if isinstance(walk, bool) or np.ndim(walk) != 0 or not np.isfinite(walk) or walk < 0:
        raise ValueError("walk_sd must be a non-negative number")
    if vol[0] <= 0 or not (vol[0] <= level[0] and level[1] <= vol[1]):
        raise ValueError("vol_bounds must be positive and contain the level bounds")

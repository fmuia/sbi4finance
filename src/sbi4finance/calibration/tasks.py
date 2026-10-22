"""The calibration task family: rough Bergomi and the Heston check.

The tasks keep the layout of the fixed-panel option_bank_v1 tasks, so the same
pricers, priors and seed streams apply, and differ from them in four ways:

* its task_id ends in "_calibration_v1" and it carries a "context_prior" key;
* quote noise is one standard deviation per contract;
* rough Bergomi draws its forward-variance curve per example: n piecewise-
  constant levels on the intervals between the design maturities, built as
  sigma0^2 (1 + s * midpoint) * exp(u_j) with sigma0, s and u_j uniform;
  the levels are stored per example and conditioned on, never inferred;
* the Heston check is the HKDE model with jump intensity fixed at zero and
  the initial variance as an inferred parameter.

Prices are in forward units: spot 1, rate 0 and dividend 0 are required.
The archived fixed-panel modules (option_panel, option_bank) are reused by
import only: archived evidence pins their source hashes.
"""

import numpy as np

from . import pricing, surface

FAMILY = "_calibration_v1"
HKDE_PARAMETERS = {"kappa", "theta", "vol_of_vol", "rho", "jump_intensity",
                   "jump_up_probability", "jump_up_rate", "jump_down_rate"}
UNIVERSE = {"rough": {"hurst", "eta", "rho"}, "hkde": HKDE_PARAMETERS | {"initial_variance"}}
TASK_KEYS = {"task_id", "model", "measure", "context", "model_context", "parameter_names",
             "prior_bounds", "fixed_parameters", "numerics", "context_prior"}


def is_calibration_task(task):
    return isinstance(task, dict) and str(task.get("task_id", "")).endswith(FAMILY)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _interval(value, name):
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
        raise ValueError(f"{name} must be an increasing finite pair")
    return pair


def knots(task):
    return np.asarray(task["model_context"].get("forward_variance_knots", []), dtype=float)


def context_size(task):
    """Number of per-example context values: forward-variance levels for rough, none for Heston."""
    return max(len(knots(task)) - 1, 0)


def context_parameter_size(task):
    return 2 if task["model"] == "rough" else 0


def block_count(task):
    return len(task["numerics"]["blocks"]) if task["model"] == "rough" else 0


def sigma(task):
    return np.asarray(task["context"]["quote_noise_std"], dtype=float)


def model_parameters(task, values):
    """The full model dictionary for one row of inferred parameter values."""
    return dict(task["fixed_parameters"], **dict(zip(task["parameter_names"], (float(v) for v in values))))


def _heston_split(parameters):
    parameters = dict(parameters)
    return parameters, float(parameters.pop("initial_variance"))


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
            from ..hkde import validate_parameters
            parameters, v0 = _heston_split(parameters)
            validate_parameters(parameters)
            if v0 <= 0:
                raise ValueError("initial variance must be positive on the prior box")
        else:
            from ..bergomi import _model_parameters
            _model_parameters(parameters, "rough")
    if task["model"] == "hkde" and (fixed.get("jump_intensity") != 0.0):
        raise ValueError("the Heston check fixes jump_intensity at zero")
    maturities = np.array([c["maturity"] for c in context["contracts"]])
    numerics, prior = task["numerics"], task["context_prior"]
    if task["model"] == "hkde":
        if task["model_context"] != {} or prior != {}:
            raise ValueError("the Heston check has no curve context")
        if set(numerics) != {"nodes", "upper_limit", "refined_nodes", "refined_upper_limit",
                             "max_refinement_over_quote_sigma"}:
            raise ValueError("Heston requires base and refined Fourier settings")
        for key in ("nodes", "refined_nodes"):
            _positive_int(numerics[key], key)
        if numerics["refined_nodes"] <= numerics["nodes"] or numerics["refined_upper_limit"] <= numerics["upper_limit"] \
                or numerics["upper_limit"] <= 0:
            raise ValueError("refinement requires more nodes and a larger upper limit")
        budget = numerics["max_refinement_over_quote_sigma"]
    else:
        grid = knots(task)
        if set(task["model_context"]) != {"forward_variance_knots"} or grid.ndim != 1 or len(grid) < 2 \
                or grid[0] != 0 or np.any(np.diff(grid) <= 0) \
                or not np.all(np.isclose(maturities[:, None], grid[None, 1:], rtol=0, atol=1e-12).any(axis=1)):
            raise ValueError("forward-variance knots start at zero, increase, and include every maturity")
        if set(prior) != {"forward_variance"} or set(prior["forward_variance"]) != {
                "level_bounds", "slope_bounds", "log_perturbation"}:
            raise ValueError("rough calibration needs a forward-variance context prior")
        level = _interval(prior["forward_variance"]["level_bounds"], "level_bounds")
        slope = _interval(prior["forward_variance"]["slope_bounds"], "slope_bounds")
        wobble = prior["forward_variance"]["log_perturbation"]
        if level[0] <= 0 or 1 + slope[0] * grid[-1] <= 0 or not np.isfinite(wobble) or wobble < 0:
            raise ValueError("positive level and slopes that keep every level positive required")
        if set(numerics) != {"method", "blocks", "paths_ladder", "max_mc_se_over_quote_sigma"} \
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
    if isinstance(budget, bool) or np.ndim(budget) != 0 or not np.isfinite(budget) or budget <= 0:
        raise ValueError("numerical precision budget must be positive")


def draw_context(task, rng):
    """Per-example forward-variance levels and their (sigma0, slope) for rough; empty for Heston."""
    if task["model"] != "rough":
        return np.empty(0), np.empty(0)
    prior = task["context_prior"]["forward_variance"]
    grid = knots(task)
    level = rng.uniform(*prior["level_bounds"])
    slope = rng.uniform(*prior["slope_bounds"])
    wobble = prior["log_perturbation"]
    midpoints = 0.5 * (grid[1:] + grid[:-1])
    log_noise = rng.uniform(-wobble, wobble, len(midpoints)) if wobble > 0 else np.zeros(len(midpoints))
    levels = level**2 * (1.0 + slope * midpoints) * np.exp(log_noise)
    return levels, np.array([level, slope])


def curve(task, context_values):
    return pricing.step_curve(knots(task), context_values)


def price(task, parameter_values, context_values, seed):
    """(prices, standard errors, covariance, refinement difference, paths per block) for one example."""
    parameters = model_parameters(task, parameter_values)
    contracts = task["context"]["contracts"]
    numerics = task["numerics"]
    n = len(contracts)
    if task["model"] == "hkde":
        parameters, v0 = _heston_split(parameters)
        base = pricing.heston_surface(parameters, v0, contracts, nodes=numerics["nodes"],
                                      upper_limit=numerics["upper_limit"])
        refined = pricing.heston_surface(parameters, v0, contracts, nodes=numerics["refined_nodes"],
                                         upper_limit=numerics["refined_upper_limit"])
        if np.any(np.abs(refined - base) > numerics["max_refinement_over_quote_sigma"] * sigma(task)):
            raise ValueError("Fourier refinement budget failed")
        return refined, np.zeros(n), np.zeros((n, n)), refined - base, np.empty(0)
    prices, errors, covariance, used = pricing.rough_surface(
        parameters, curve(task, context_values), contracts, numerics["blocks"], numerics["paths_ladder"],
        sigma(task), numerics["max_mc_se_over_quote_sigma"], seed)
    return prices, errors, covariance, np.zeros(n), used


def _base_task(contracts, sigmas):
    return {"spot": 1.0, "rate": 0.0, "dividend": 0.0, "contracts": contracts, "quote_noise_std": sigmas}


def rough_task(*, days=surface.DESIGN_DAYS, deltas=surface.DESIGN_DELTAS, sigma_ref=surface.SIGMA_REF,
               noise_atm=surface.NOISE_ATM, noise_wing=surface.NOISE_WING,
               prior_bounds=((0.02, 0.35), (0.5, 3.5), (-0.99, -0.2)),
               level_bounds=(0.10, 0.40), slope_bounds=(-0.3, 0.6), log_perturbation=0.2,
               blocks=None, paths_ladder=(32768, 65536, 131072, 262144), budget=0.25):
    contracts = surface.design_grid(days, deltas, sigma_ref)
    if blocks is None:
        short = [d for d in days if d <= 63]
        blocks = [{"horizon": max(short) / surface.TRADING_DAYS, "n_steps": 8 * max(short)},
                  {"horizon": max(days) / surface.TRADING_DAYS, "n_steps": 2 * max(days)}]
    task = {
        "task_id": "rough" + FAMILY, "model": "rough", "measure": "Q",
        "context": _base_task(contracts, surface.quote_noise(contracts, sigma_ref, noise_atm, noise_wing)),
        "model_context": {"forward_variance_knots": [0.0] + [d / surface.TRADING_DAYS for d in days]},
        "context_prior": {"forward_variance": {"level_bounds": list(level_bounds),
                                               "slope_bounds": list(slope_bounds),
                                               "log_perturbation": log_perturbation}},
        "parameter_names": ["hurst", "eta", "rho"],
        "prior_bounds": [list(b) for b in prior_bounds],
        "fixed_parameters": {},
        "numerics": {"method": "hybrid", "blocks": blocks, "paths_ladder": list(paths_ladder),
                     "max_mc_se_over_quote_sigma": budget},
    }
    validate_task(task)
    return task


def heston_task(*, days=surface.DESIGN_DAYS, deltas=surface.DESIGN_DELTAS, sigma_ref=surface.SIGMA_REF,
                noise_atm=surface.NOISE_ATM, noise_wing=surface.NOISE_WING,
                prior_bounds=((0.01, 0.16), (0.01, 0.16), (0.1, 1.5), (-0.95, -0.1)), kappa=2.0,
                nodes=512, upper_limit=400.0, refined_nodes=1024, refined_upper_limit=600.0, budget=0.1):
    contracts = surface.design_grid(days, deltas, sigma_ref)
    task = {
        "task_id": "hkde" + FAMILY, "model": "hkde", "measure": "Q",
        "context": _base_task(contracts, surface.quote_noise(contracts, sigma_ref, noise_atm, noise_wing)),
        "model_context": {}, "context_prior": {},
        "parameter_names": ["initial_variance", "theta", "vol_of_vol", "rho"],
        "prior_bounds": [list(b) for b in prior_bounds],
        "fixed_parameters": {"kappa": kappa, "jump_intensity": 0.0, "jump_up_probability": 0.5,
                             "jump_up_rate": 10.0, "jump_down_rate": 10.0},
        "numerics": {"nodes": nodes, "upper_limit": upper_limit, "refined_nodes": refined_nodes,
                     "refined_upper_limit": refined_upper_limit, "max_refinement_over_quote_sigma": budget},
    }
    validate_task(task)
    return task

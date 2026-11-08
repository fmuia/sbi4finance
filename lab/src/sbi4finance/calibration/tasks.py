"""The calibration task family: rough Bergomi and the Heston check.

The tasks keep the layout of the fixed-panel option_bank_v1 tasks, so the same
pricers, priors and seed streams apply, and differ from them in four ways:

* its task_id ends in "_calibration_v1" and it carries a "context_prior" key;
* quote noise is one standard deviation per contract;
* rough Bergomi draws its forward-variance curve per example: n piecewise-
  constant levels on the intervals between the design maturities, from one
  of two context priors (CURVE_KINDS): the pilot's "linear_wobble",
  sigma0^2 (1 + s * midpoint) * exp(u_j) with sigma0, s and u_j uniform, and
  the full bank's "power_law_walk", a power law in time times a Gaussian
  random walk in log time (draw_context); the levels are stored per example;
* the Heston check is the HKDE model with jump intensity fixed at zero and
  the initial variance as an inferred parameter.

The additive rough_joint_calibration_v1 task instead infers three analytic
Heston-shaped curve coefficients alongside H, eta and rho. Its stored curve
averages are derived auxiliary labels, never known inference conditions.

Prices are in forward units: spot 1, rate 0 and dividend 0 are required.
The archived fixed-panel modules (option_panel, option_bank) are reused by
import only: archived evidence pins their source hashes.
"""

import numpy as np

from . import pricing, surface

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.shared.tasks import (
    MAX_CURVE_DRAWS,
    interval_midpoints,
    draw_context,
    _joint_coefficients,
    derived_context,
    joint_curve,
    knots,
    context_size,
    sigma,
    model_parameters,
    _heston_split,
    heston_prices,
    curve,
    FAMILY,
    JOINT_ROUGH_ID,
    JOINT_ROUGH_PARAMETERS,
    HKDE_PARAMETERS,
    UNIVERSE,
    TASK_KEYS,
    CURVE_KINDS,
    is_calibration_task,
    is_joint_curve_task,
    _positive_int,
    _interval,
    validate_context,
    validate_task,
    _validate_joint_task,
    curve_kind,
    validate_curve_prior,
)

# The keys of each forward-variance prior. A prior without "kind" is the pilot's linear_wobble, so the pilot's
# configuration, fingerprint and draws are unchanged.

# The full bank's forward-variance prior (configs/tasks/calibration_rough_bank.json; docs/course/v2-plan.md, G0').
FULL_BANK_CURVE_PRIOR = {"kind": "power_law_walk", "level_bounds": [0.08, 0.60], "exponent_bounds": [-0.6, 0.6],
                         "walk_sd": 0.15, "vol_bounds": [0.06, 0.60]}


def context_parameter_size(task):
    return 3 if is_joint_curve_task(task) else (2 if task["model"] == "rough" else 0)


def block_count(task):
    return len(task["numerics"]["blocks"]) if task["model"] == "rough" else 0


def price(task, parameter_values, context_values, seed, diagnostics=None):
    """(prices, standard errors, covariance, refinement difference, paths per block) for one example.

    diagnostics, a list, receives the rough pricer's per-block control-variate variances.
    """
    parameters = model_parameters(task, parameter_values)
    contracts = task["context"]["contracts"]
    numerics = task["numerics"]
    n = len(contracts)
    if task["model"] == "hkde":
        parameters, v0 = _heston_split(parameters)
        prices, difference = heston_prices(task, parameters, v0)
        return prices, np.zeros(n), np.zeros((n, n)), difference, np.empty(0)
    initial_curve = joint_curve(task, parameter_values) if is_joint_curve_task(task) else curve(task, context_values)
    prices, errors, covariance, used = pricing.rough_surface(
        parameters, initial_curve, contracts, numerics["blocks"], numerics["paths_ladder"],
        sigma(task), numerics["max_mc_se_over_quote_sigma"], seed,
        cap=numerics.get("max_mc_se_over_quote_sigma_last_rung"), diagnostics=diagnostics)
    return prices, errors, covariance, np.zeros(n), used


def _base_task(contracts, sigmas):
    return {"spot": 1.0, "rate": 0.0, "dividend": 0.0, "contracts": contracts, "quote_noise_std": sigmas}


def rough_task(*, days=surface.DESIGN_DAYS, deltas=surface.DESIGN_DELTAS, sigma_ref=surface.SIGMA_REF,
               noise_atm=surface.NOISE_ATM, noise_wing=surface.NOISE_WING,
               prior_bounds=((0.02, 0.35), (0.5, 3.5), (-0.99, -0.2)),
               level_bounds=(0.10, 0.40), slope_bounds=(-0.3, 0.6), log_perturbation=0.2, curve_prior=None,
               blocks=None, paths_ladder=tuple(2**k for k in range(15, 22)), budget=0.25, last_rung_cap=0.5):
    """The rough-Bergomi calibration task; curve_prior, if given, replaces the pilot's linear_wobble prior."""
    contracts = surface.design_grid(days, deltas, sigma_ref)
    if curve_prior is None:
        curve_prior = {"level_bounds": list(level_bounds), "slope_bounds": list(slope_bounds),
                       "log_perturbation": log_perturbation}
    if blocks is None:
        # One grid per maturity group, each starting at zero: the one- and two-week quotes (8 steps a day,
        # where the far puts need the most paths and a path costs least), up to three months (8 a day),
        # and the rest (2 a day).
        blocks = []
        for limit, per_day in ((10, 8), (63, 8), (max(days), 2)):
            inside = [d for d in days if d <= limit and (not blocks or d > blocks[-1]["days"])]
            if inside:
                blocks.append({"days": max(inside), "n_steps": per_day * max(inside)})
        blocks = [{"horizon": b["days"] / surface.TRADING_DAYS, "n_steps": b["n_steps"]} for b in blocks]
    task = {
        "task_id": "rough" + FAMILY, "model": "rough", "measure": "Q",
        "context": _base_task(contracts, surface.quote_noise(contracts, sigma_ref, noise_atm, noise_wing)),
        "model_context": {"forward_variance_knots": [0.0] + [d / surface.TRADING_DAYS for d in days]},
        "context_prior": {"forward_variance": dict(curve_prior)},
        "parameter_names": ["hurst", "eta", "rho"],
        "prior_bounds": [list(b) for b in prior_bounds],
        "fixed_parameters": {},
        "numerics": {"method": "hybrid", "blocks": blocks, "paths_ladder": list(paths_ladder),
                     "max_mc_se_over_quote_sigma": budget, "max_mc_se_over_quote_sigma_last_rung": last_rung_cap},
    }
    validate_task(task)
    return task


def heston_task(*, days=surface.DESIGN_DAYS, deltas=surface.DESIGN_DELTAS, sigma_ref=surface.SIGMA_REF,
                noise_atm=surface.NOISE_ATM, noise_wing=surface.NOISE_WING,
                prior_bounds=((0.01, 0.16), (0.01, 0.16), (0.1, 1.5), (-0.95, -0.1)), kappa=2.0,
                quadrature_ladder=((512, 400.0), (1024, 600.0), (2048, 1500.0), (4096, 3000.0), (8192, 6000.0)),
                budget=0.1):
    contracts = surface.design_grid(days, deltas, sigma_ref)
    task = {
        "task_id": "hkde" + FAMILY, "model": "hkde", "measure": "Q",
        "context": _base_task(contracts, surface.quote_noise(contracts, sigma_ref, noise_atm, noise_wing)),
        "model_context": {}, "context_prior": {},
        "parameter_names": ["initial_variance", "theta", "vol_of_vol", "rho"],
        "prior_bounds": [list(b) for b in prior_bounds],
        "fixed_parameters": {"kappa": kappa, "jump_intensity": 0.0, "jump_up_probability": 0.5,
                             "jump_up_rate": 10.0, "jump_down_rate": 10.0},
        "numerics": {"quadrature_ladder": [list(rung) for rung in quadrature_ladder],
                     "max_refinement_over_quote_sigma": budget},
    }
    validate_task(task)
    return task


def joint_rough_task(*, prior_bounds=((0.02, 0.35), (0.5, 3.5), (-0.99, -0.2),
                                     (0.01, 0.16), (0.01, 0.16), (0.25, 8.0)), **kwargs):
    """Candidate six-target task for bounded development, not an approved production prior.

    Levels are independent uniforms in annualized variance units, decay in
    inverse years. The illustrative positive bounds and numerical ladder need
    the declared family/prior/precision pilot before substantial generation.
    kwargs supplies rough_task's design/numerical settings; curve-prior kwargs
    are not part of this jointly inferred family.
    """
    forbidden = {"curve_prior", "level_bounds", "slope_bounds", "log_perturbation"} & set(kwargs)
    if forbidden:
        raise ValueError(f"joint curve priors belong in the six prior_bounds, not {sorted(forbidden)}")
    task = rough_task(**kwargs)
    task.update(task_id=JOINT_ROUGH_ID, parameter_names=list(JOINT_ROUGH_PARAMETERS),
                prior_bounds=[list(b) for b in prior_bounds], context_prior={})
    task["model_context"]["curve_family"] = "heston_exponential_v1"
    validate_task(task)
    return task

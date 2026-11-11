"""Versioned actual-layout contracts for a future rough numerical pilot.

No simulation occurs here. Legacy validators and numerical kernels stay intact.
The original supplied piecewise curve is evaluated at declared simulation
nodes; its financial knots and prior are not moved to observation expiries.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np

from ..shared import tasks as legacy_tasks
from .model import curve_values

FORMAT = "rough_actual_layout_task_v1"
COLUMNS = ("forward", "discount", "strike", "maturity", "option_type", "mid", "half_spread")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalize_design(records):
    if not set(COLUMNS) <= set(records): raise ValueError("complete actual quote geometry and half-spreads required")
    arrays = {key: np.asarray(records[key], dtype=str if key == "option_type" else float) for key in COLUMNS}
    n = len(arrays["mid"])
    if n != 63 or any(value.shape != (n,) for value in arrays.values()): raise ValueError("ordered complete actual63 quote vectors required")
    if any(np.iscomplexobj(records[key]) or not np.isfinite(arrays[key]).all() for key in COLUMNS if key != "option_type"):
        raise ValueError("finite real actual quotes required")
    if any(np.any(arrays[key] <= 0) for key in ("forward", "discount", "strike", "maturity", "half_spread")):
        raise ValueError("positive F,D,K,T and half-spreads required")
    if not set(arrays["option_type"]) <= {"put", "call"}: raise ValueError("European puts/calls only")
    times, counts = np.unique(arrays["maturity"], return_counts=True)
    if len(times) != 7 or not np.array_equal(counts, np.full(7, 9)): raise ValueError("seven actual expiries with nine quotes each required")
    factor = arrays["discount"]*arrays["forward"]
    return {"format": "rough_actual_quote_design_v1", "raw_records": {key: value.tolist() for key, value in arrays.items()},
        "contracts": [{"strike": float(k/f), "maturity": float(t), "option_type": str(kind)}
            for k, f, t, kind in zip(arrays["strike"], arrays["forward"], arrays["maturity"], arrays["option_type"], strict=True)],
        "price_factor_DF": factor.tolist(), "observations_forward": (arrays["mid"]/factor).tolist(),
        "half_spread_forward": (arrays["half_spread"]/factor).tolist(), "actual_maturities": times.tolist(),
        "normalization": "p=y_price/(D*F), h=h_price/(D*F), k=K/F; raw covariance divided by DF on both axes"}


def expiry_blocks(design, step_counts):
    times = design["actual_maturities"]
    if len(step_counts) != len(times) or any(type(value) is not int or value < 1 for value in step_counts):
        raise ValueError("one explicit positive integer step count per actual expiry required")
    return [{"horizon": time, "n_steps": steps} for time, steps in zip(times, step_counts, strict=True)]


def make_task(base_task, records, step_counts):
    legacy_tasks.validate_task(base_task)
    if base_task["model"] != "rough": raise ValueError("original rough financial-law template required")
    design = normalize_design(records)
    joint = legacy_tasks.is_joint_curve_task(base_task)
    return {"format": FORMAT, "base_task": deepcopy(base_task), "base_task_fingerprint": fingerprint(base_task),
        "curve_role": "joint_unknown" if joint else "supplied_known", "parameter_names": list(base_task["parameter_names"]),
        "prior_bounds": deepcopy(base_task["prior_bounds"]), "design": design, "blocks": expiry_blocks(design, step_counts),
        "observation_model": {"kind": "independent Gaussian price errors", "known_c": 1., "strict_supported_c_min": .5},
        "curve_discretization": "existing analytic family evaluated on every node" if joint else "original piecewise-constant curve evaluated at left simulation nodes",
        "numerical_gate": {"mcse_over_strict_sd": .25, "last_rung_exception": False, "quote_noise_includes_MC": False},
        "scientific_acceptance": False}


def validate_task(task):
    if task.get("format") != FORMAT: raise ValueError("new actual-layout rough task identity required")
    base = task["base_task"]; legacy_tasks.validate_task(base)
    if (base["model"] != "rough" or fingerprint(base) != task["base_task_fingerprint"]
            or task["parameter_names"] != base["parameter_names"] or task["prior_bounds"] != base["prior_bounds"]):
        raise ValueError("original rough prior, curve law and target order must be preserved")
    joint = legacy_tasks.is_joint_curve_task(base)
    if task["curve_role"] != ("joint_unknown" if joint else "supplied_known"):
        raise ValueError("explicit supplied/joint curve role differs from original task")
    design = normalize_design(task["design"]["raw_records"])
    if design != task["design"] or task["blocks"] != expiry_blocks(design, [b["n_steps"] for b in task["blocks"]]):
        raise ValueError("actual expiries, carry normalization or block ordering changed")
    expected_curve_rule = "existing analytic family evaluated on every node" if joint else "original piecewise-constant curve evaluated at left simulation nodes"
    if task.get("curve_discretization") != expected_curve_rule or task.get("scientific_acceptance") is not False:
        raise ValueError("curve discretization must preserve the declared financial family")
    if not joint and design["actual_maturities"][-1] > base["model_context"]["forward_variance_knots"][-1]:
        raise ValueError("original supplied curve does not cover the actual expiry; no extrapolation")
    if task["observation_model"] != {"kind": "independent Gaussian price errors", "known_c": 1., "strict_supported_c_min": .5}:
        raise ValueError("initial task has known c=1 and unchanged half-spread convention")
    if task["numerical_gate"] != {"mcse_over_strict_sd": .25, "last_rung_exception": False, "quote_noise_includes_MC": False}:
        raise ValueError("strict measurement-only numerical gate required")
    return task


def simulation_nodes(task, multipliers=(1, 2, 4)):
    validate_task(task)
    if any(type(g) is not int or g < 1 for g in multipliers): raise ValueError("positive integer grid multipliers required")
    return np.unique(np.concatenate([b["horizon"]/(g*b["n_steps"])*np.arange(g*b["n_steps"]+1)
        for b in task["blocks"] for g in multipliers]))


def piecewise_node_curve(knots, levels, nodes):
    knots, levels, nodes = np.asarray(knots, float), np.asarray(levels, float), np.asarray(nodes, float)
    if (knots.ndim != 1 or levels.shape != (len(knots)-1,) or nodes.ndim != 1 or not len(nodes)
            or knots[0] != 0 or nodes[0] != 0 or np.any(np.diff(knots) <= 0) or np.any(np.diff(nodes) <= 0)
            or not all(np.isfinite(value).all() for value in (knots, levels, nodes)) or np.any(levels <= 0)
            or nodes[-1] > knots[-1]): raise ValueError("original positive piecewise curve must cover every node")
    indices = np.minimum(np.searchsorted(knots, nodes, side="right")-1, len(levels)-1)
    values = levels[indices]
    return {"times": nodes.tolist(), "values": values.tolist()}


def known_curve(task, levels, multipliers=(1, 2, 4)):
    validate_task(task)
    if task["curve_role"] != "supplied_known": raise ValueError("supplied-curve task required")
    return piecewise_node_curve(task["base_task"]["model_context"]["forward_variance_knots"], levels, simulation_nodes(task, multipliers))


def discrete_curve_audit(task, curve, multiplier):
    validate_task(task)
    results = []
    for block in task["blocks"]:
        n = block["n_steps"]*multiplier; dt = block["horizon"]/n; left = dt*np.arange(n)
        values = curve_values(curve, left)
        results.append({"horizon": block["horizon"], "steps": n, "dt": dt,
            "discrete_expected_integrated_variance": float(values.sum()*dt),
            "control_centering": "canonical rough_block uses this same left-node sum"})
    return results


def conditioning_inputs(task, observations_forward, *, supplied_levels=None):
    """Pure future-adapter boundary; never use joint generating curves as inputs."""
    validate_task(task); values = np.asarray(observations_forward, float)
    if values.ndim != 2 or values.shape[1] != 63 or not np.isfinite(values).all(): raise ValueError("finite full63 observations required")
    result = {"observations": values/np.asarray(task["design"]["half_spread_forward"]),
        "known_c": np.ones((len(values), 1))}
    if task["curve_role"] == "joint_unknown":
        if supplied_levels is not None: raise ValueError("joint generating forward curve must never enter inference conditions")
    else:
        levels = np.asarray(supplied_levels, float)
        expected = len(task["base_task"]["model_context"]["forward_variance_knots"])-1
        if levels.shape != (len(values), expected) or not np.isfinite(levels).all() or np.any(levels <= 0):
            raise ValueError("explicit positive supplied original curve levels required")
        result["curve_context"] = np.log(levels)
    return result


def covariance_to_forward(task, covariance):
    validate_task(task); covariance = np.asarray(covariance, float); factors = np.asarray(task["design"]["price_factor_DF"])
    if covariance.shape[-2:] != (63, 63) or not np.isfinite(covariance).all(): raise ValueError("finite price covariance with last axes63×63 required")
    return covariance/factors[:, None]/factors[None, :]


def covariance_to_raw(task, covariance):
    validate_task(task); covariance = np.asarray(covariance, float); factors = np.asarray(task["design"]["price_factor_DF"])
    if covariance.shape[-2:] != (63, 63) or not np.isfinite(covariance).all(): raise ValueError("finite price covariance with last axes63×63 required")
    return covariance*factors[:, None]*factors[None, :]

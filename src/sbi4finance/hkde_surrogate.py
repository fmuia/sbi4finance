"""Small fixed-contract HKDE price surrogate, separate from posterior inference.

The inputs are (annualized variance theta, jump intensity per year), with all
other risk-neutral model and market context fixed by the supplied task. Outputs
are two European put expectations in the same currency units as spot/strikes.
Training labels must already be accurate expectation estimates; this module
cannot infer their units or accuracy from their numerical values.

A tensor Chebyshev polynomial is fitted once by float64 least squares on the
declared parameter rectangle. Prediction refuses extrapolation and changed
context. No clipping, posterior network, price-noise model, or tuning is hidden
inside the regression. Numerical and downstream errors require separate checks.
"""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander2d

from .hkde_reference import DEFAULT_TARGET_CONTRACTS
from .option_bank import validate_config


_FORMAT = "hkde_fixed_target_chebyshev_v1"
_BASIS = "tensor Chebyshev first kind on bounds mapped to [-1,1]"
_ORDER = "C order: column i*(degree+1)+j is T_i(theta_scaled)*T_j(jump_intensity_scaled)"
_PARAMETER_UNITS = ["annualized variance", "jumps per year"]
_PRICE_UNITS = "currency in the units of task.context.spot and contract strikes"
_FIELDS = {"format", "task", "contracts", "degree", "coefficients", "rank",
           "condition_number", "training_rows", "basis", "coefficient_order",
           "parameter_units", "price_units"}


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _specification(task, contracts):
    # Copy nested caller-owned dictionaries, so later edits cannot silently
    # alter the context stored alongside fitted coefficients.
    try:
        task = json.loads(json.dumps(task, allow_nan=False))
        contracts = json.loads(json.dumps(list(contracts), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("JSON-compatible finite task and ordered contracts required") from error
    if not isinstance(task, dict):
        raise ValueError("a declared HKDE task dictionary is required")
    validate_config({"task": task, "root_seed": 0,
                     "splits": {"train": 1, "validation": 1, "test": 1}})
    if task["model"] != "hkde" or task["parameter_names"] != ["theta", "jump_intensity"]:
        raise ValueError("surrogate requires HKDE parameter order theta, jump_intensity")
    if contracts != list(DEFAULT_TARGET_CONTRACTS):
        raise ValueError("surrogate requires the fixed DEFAULT_TARGET_CONTRACTS in their declared order")
    return task, contracts


def _parameters(parameters, task):
    rows = np.asarray(parameters, dtype=np.float64)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    if (rows.ndim != 2 or rows.shape[1] != 2 or not len(rows)
            or not np.isfinite(rows).all() or np.any(rows < bounds[:, 0])
            or np.any(rows > bounds[:, 1])):
        raise ValueError("finite [N,2] parameters inside the declared prior bounds required; no extrapolation")
    return rows


def _design(parameters, task, degree):
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    scaled = 2 * (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1
    return chebvander2d(scaled[:, 0], scaled[:, 1], [degree, degree])


def _validate_model(model):
    if not isinstance(model, dict) or set(model) != _FIELDS:
        raise ValueError("model fields do not match the fixed HKDE surrogate")
    task, contracts = _specification(model["task"], model["contracts"])
    degree = _integer(model["degree"], "degree")
    count = (degree + 1) ** 2
    rank = _integer(model["rank"], "rank", minimum=1)
    _integer(model["training_rows"], "training_rows", minimum=count)
    coefficients = np.asarray(model["coefficients"])
    condition = model["condition_number"]
    if (model["format"] != _FORMAT or model["basis"] != _BASIS
            or model["coefficient_order"] != _ORDER
            or model["parameter_units"] != _PARAMETER_UNITS or model["price_units"] != _PRICE_UNITS
            or rank != count or coefficients.shape != (count, 2)
            or coefficients.dtype != np.dtype("float64") or not np.isfinite(coefficients).all()
            or isinstance(condition, (bool, np.bool_)) or np.ndim(condition) != 0
            or not np.isfinite(condition) or condition < 1):
        raise ValueError("invalid surrogate coefficients, full-rank fit metadata, basis, or units")
    return task, contracts, degree, coefficients


def fit(task, contracts, parameters, prices, degree=6):
    """Fit a full-rank tensor polynomial to finite currency-price labels [N,2].

    The two columns retain the fixed target-contract order. Parameters remain
    in their raw units until the deterministic bounds transformation; training
    data do not estimate scaling. A deficient design raises instead of silently
    selecting a lower degree. ``condition_number`` describes this design matrix,
    not model accuracy. Constant output columns are valid regression targets.
    """
    task, contracts = _specification(task, contracts)
    degree = _integer(degree, "degree")
    rows = _parameters(parameters, task)
    labels = np.asarray(prices, dtype=np.float64)
    if labels.shape != (len(rows), 2) or not np.isfinite(labels).all():
        raise ValueError("finite currency-price labels [N,2] matching parameter rows required")
    design = _design(rows, task, degree)
    count = (degree + 1) ** 2
    if len(rows) < count:
        raise ValueError("too few parameter rows for the requested full-rank tensor degree")
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, labels, rcond=None)
    if rank != count:
        raise ValueError("rank-deficient parameter design for the requested tensor degree")
    condition = float(singular_values[0] / singular_values[-1])
    model = {"format": _FORMAT, "task": task, "contracts": contracts, "degree": degree,
             "coefficients": coefficients, "rank": int(rank), "condition_number": condition,
             "training_rows": len(rows), "basis": _BASIS, "coefficient_order": _ORDER,
             "parameter_units": list(_PARAMETER_UNITS), "price_units": _PRICE_UNITS}
    _validate_model(model)
    return model


def predict(model, parameters, *, task=None, contracts=None):
    """Predict fixed-target currency prices [N,2], without clipping/extrapolation.

    Optional task and contract arguments assert compatibility with a caller's
    intended calculation. Every part of the task is bound, including market
    context, parameter order, numerical label controls and the prior rectangle.
    Omitting them uses the context explicitly stored in the model.
    """
    saved_task, saved_contracts, degree, coefficients = _validate_model(model)
    if task is not None or contracts is not None:
        supplied_task, supplied_contracts = _specification(
            saved_task if task is None else task, saved_contracts if contracts is None else contracts)
        if supplied_task != saved_task or supplied_contracts != saved_contracts:
            raise ValueError("prediction task or contracts differ from the fitted context")
    rows = _parameters(parameters, saved_task)
    result = _design(rows, saved_task, degree) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("nonfinite surrogate prediction")
    return result


def save_model(path, model):
    """Write a new NPZ file containing float64 coefficients and JSON metadata."""
    _, _, _, coefficients = _validate_model(model)
    metadata = {key: value for key, value in model.items() if key != "coefficients"}
    encoded = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    with Path(path).open("xb") as file:
        np.savez(file, coefficients=coefficients, metadata=encoded)


def load_model(path):
    """Load and validate the plain NPZ artifact without enabling pickle."""
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"coefficients", "metadata"}:
            raise ValueError("surrogate archive requires coefficients and metadata only")
        metadata = saved["metadata"]
        if metadata.shape != () or metadata.dtype.kind != "U":
            raise ValueError("surrogate metadata must be a scalar JSON string")
        model = json.loads(str(metadata.item()))
        if not isinstance(model, dict) or "coefficients" in model:
            raise ValueError("surrogate metadata must be an object without embedded coefficients")
        model["coefficients"] = saved["coefficients"].copy()
    _validate_model(model)
    return model

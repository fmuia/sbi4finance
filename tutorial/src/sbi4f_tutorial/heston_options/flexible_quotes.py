"""Pure quote-set inputs for experimental Heston summaries and missing quotes.

Prices enter in currency units. Geometry and spreads are observed conditions;
neither generating parameters nor the inferred multiplier ``c`` enter tokens.
This module does not price options, train estimators or use a saved dense-grid
whitening transform. All operations touching values select valid records first.
"""
from hashlib import sha256
import json

import numpy as np


GEOMETRY_KEYS = ("forward", "strike", "discount", "maturity", "option_type", "half_spread")
FEATURE_NAMES = ("price_forward", "log_moneyness", "maturity", "put_flag", "half_spread_forward")
SCALER_FORMAT = "heston_quote_token_scaler_v1"


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _mask(value, shape):
    result = np.ones(shape, dtype=bool) if value is None else np.asarray(value)
    if result.dtype != np.dtype(bool) or result.shape != shape:
        raise ValueError(f"availability mask must have boolean dtype and shape {shape}")
    if np.any(result.sum(axis=1) == 0):
        raise ValueError("each quote set requires at least one available record")
    return result.copy()


def _columns(records, shape, mask):
    if not all(key in records for key in GEOMETRY_KEYS):
        raise ValueError(f"geometry requires columns {GEOMETRY_KEYS}")
    columns = {}
    for key in GEOMETRY_KEYS:
        array = np.asarray(records[key], dtype=str) if key == "option_type" else _real(records[key], key)
        if array.ndim not in (1, 2) or array.shape[-1] != shape[1]:
            raise ValueError(f"{key} must contain one value per quote")
        try:
            array = np.broadcast_to(array, shape)
        except ValueError as error:
            raise ValueError(f"{key} cannot broadcast to quote batch {shape}") from error
        valid = array[mask]
        if key == "option_type":
            if not np.isin(valid, ("call", "put")).all():
                raise ValueError("available option_type must be call or put")
        elif not np.isfinite(valid).all() or np.any(valid <= 0):
            raise ValueError(f"available {key} must be finite and positive")
        columns[key] = array
    return columns


def validate_records(records, *, expected_quotes=None):
    """Validate complete geometry and return copied batched columns.

    ``mid`` is optional and is not an inference input until explicitly passed
    to :func:`encode_quotes`. Generic sets may have arbitrary positive lengths.
    """
    if "forward" not in records:
        raise ValueError("geometry requires forward")
    forward = np.asarray(records["forward"])
    if forward.ndim not in (1, 2) or not forward.size:
        raise ValueError("geometry must be a nonempty quote vector or batch")
    shape = (1, forward.shape[0]) if forward.ndim == 1 else forward.shape
    if expected_quotes is not None and shape[1] != expected_quotes:
        raise ValueError(f"expected {expected_quotes} quotes")
    return {key: value.copy() for key, value in _columns(records, shape, _mask(None, shape)).items()}


def validate_actual63_records(records):
    """Validate the actual63 column schema; do not impose an array order."""
    return validate_records(records, expected_quotes=63)


def geometry_fingerprint(records):
    """Bind price provenance to complete coordinates, option types and spreads."""
    columns = validate_records(records)
    body = {key: columns[key].tolist() for key in GEOMETRY_KEYS}
    return sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def encode_quotes(records, observations, mask=None, *, priced_geometry=None):
    """Return tokens ``[B,N,5]``, boolean mask ``[B,N]`` and count ``[B,1]``.

    ``observations`` are noisy raw prices, not ``Y/h`` or whitened values.
    Masked payloads (including NaNs) are never evaluated. Jittered geometry is
    deliberately marked as requiring repricing; its producer must supply the
    matching geometry fingerprint only after actually pricing those locations.
    """
    prices = _real(observations, "observations")
    if prices.ndim == 1:
        prices = prices[None, :]
    if prices.ndim != 2 or not prices.size:
        raise ValueError("observations must be a nonempty [batch, quotes] array")
    available = _mask(mask, prices.shape)
    columns = _columns(records, prices.shape, available)
    if not np.isfinite(prices[available]).all():
        raise ValueError("available observations must be finite")
    if records.get("_requires_repricing", False):
        if priced_geometry != geometry_fingerprint(records):
            raise ValueError("jittered coordinates require matching newly priced geometry; old prices cannot be relabelled")
    elif priced_geometry is not None and priced_geometry != geometry_fingerprint(records):
        raise ValueError("price geometry differs from supplied quote geometry")
    factor = columns["forward"][available] * columns["discount"][available]
    if not np.isfinite(factor).all() or np.any(factor <= 0):
        raise ValueError("available discount times forward must be finite and positive")
    tokens = np.zeros((*prices.shape, len(FEATURE_NAMES)), dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        tokens[available] = np.column_stack((
            prices[available] / factor,
            np.log(columns["strike"][available]) - np.log(columns["forward"][available]),
            columns["maturity"][available],
            (columns["option_type"][available] == "put").astype(float),
            columns["half_spread"][available] / factor,
        ))
    if not np.isfinite(tokens[available]).all():
        raise ValueError("available normalized token values must be finite")
    return {"tokens": tokens, "mask": available, "count": available.sum(axis=1, keepdims=True)}


def _token_batch(tokens, mask):
    values = _real(tokens, "tokens")
    if values.ndim != 3 or values.shape[-1] != len(FEATURE_NAMES) or not values.size:
        raise ValueError("tokens require shape [batch, quotes, 5]")
    available = _mask(mask, values.shape[:2])
    if not np.isfinite(values[available]).all():
        raise ValueError("available tokens must be finite")
    return values, available


def fit_token_scaler(tokens, mask, *, split="train"):
    """Fit shared feature moments on valid training tokens only.

    Features never mix quote locations. The type indicator remains unchanged;
    all other scales are shared over all valid records, retaining coordinates.
    The caller owns the train/development split and must label it honestly.
    """
    if split != "train":
        raise ValueError("token scaler may be fitted on the training split only")
    values, available = _token_batch(tokens, mask)
    selected = values[available]
    mean, scale = selected.mean(axis=0), selected.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    mean[3], scale[3] = 0.0, 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("training token moments must remain finite")
    return {"format": SCALER_FORMAT, "feature_names": list(FEATURE_NAMES), "fitted_split": "train",
            "valid_training_tokens": int(available.sum()), "mean": mean.tolist(), "scale": scale.tolist()}


def transform_tokens(tokens, mask, scaler):
    """Apply persisted moments exactly once; padding is always zero."""
    values, available = _token_batch(tokens, mask)
    if (scaler.get("format") != SCALER_FORMAT or scaler.get("feature_names") != list(FEATURE_NAMES)
            or scaler.get("fitted_split") != "train"):
        raise ValueError("a compatible training-only token scaler is required")
    mean, scale = _real(scaler.get("mean"), "scaler mean"), _real(scaler.get("scale"), "scaler scale")
    if (mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0) or mean[3] != 0 or scale[3] != 1):
        raise ValueError("invalid shared token scaler moments")
    result = np.zeros_like(values)
    result[available] = (values[available] - mean) / scale
    if not np.isfinite(result[available]).all():
        raise ValueError("scaled token values must remain finite")
    return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rng(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(seed)


def make_masks(records, batch_size, *, mode="full", seed=0, retained_counts=(16, 32, 47, 63)):
    """Draw reproducible geometry-only availability; never inspect prices.

    ``expiry`` removes one whole expiry selected independently for each row.
    Wing modes remove the outer half of the chosen option type in each expiry.
    Random thinning uses a uniformly chosen exact count from retained_counts.
    """
    columns = validate_records(records)
    if columns["forward"].shape[0] != 1:
        raise ValueError("mask generation requires one parent geometry")
    batch_size = _positive_integer(batch_size, "batch_size")
    n = columns["forward"].shape[1]
    rng = _rng(seed)
    masks = np.ones((batch_size, n), dtype=bool)
    if mode == "full":
        return masks
    if mode == "random":
        candidates = np.asarray(retained_counts)
        if (candidates.ndim != 1 or not candidates.size or candidates.dtype.kind not in "iu"
                or np.any(candidates < 1) or np.any(candidates > n)):
            raise ValueError("retained_counts must be nonempty integer counts inside [1, quotes]")
        for row in masks:
            row[:] = False
            row[rng.choice(n, size=int(rng.choice(candidates)), replace=False)] = True
        return masks
    expiry = columns["maturity"][0]
    unique = np.unique(expiry)
    if mode == "expiry":
        if len(unique) < 2:
            raise ValueError("expiry masking needs at least two maturity blocks")
        for row in masks:
            row[expiry == rng.choice(unique)] = False
    elif mode in ("put_wing", "call_wing"):
        kind = "put" if mode == "put_wing" else "call"
        for time in unique:
            indices = np.flatnonzero((expiry == time) & (columns["option_type"][0] == kind))
            indices = indices[np.argsort(columns["strike"][0, indices], kind="stable")]
            count = (len(indices) + 1) // 2
            dropped = indices[:count] if kind == "put" else indices[len(indices) - count:]
            masks[:, dropped] = False
    else:
        raise ValueError("unsupported mask family")
    return _mask(masks, masks.shape)


def interpolate_carry(records, maturities):
    """Log-linear forward/discount interpolation inside observed time support.

    This is an explicitly conditioned synthetic carry convention, not an
    estimate of new market carry. No extrapolation is permitted.
    """
    columns = validate_records(records)
    if columns["forward"].shape[0] != 1:
        raise ValueError("carry interpolation requires one parent geometry")
    time = columns["maturity"][0]
    knots = np.unique(time)
    curves = {}
    for key in ("forward", "discount"):
        curve = []
        for point in knots:
            values = columns[key][0, time == point]
            if not np.allclose(values, values[0], rtol=1e-12, atol=0):
                raise ValueError("carry must be shared within each maturity block")
            curve.append(values[0])
        curves[key] = np.asarray(curve)
    requested = _real(maturities, "maturities")
    if (not np.isfinite(requested).all() or np.any(requested < knots[0]) or np.any(requested > knots[-1])):
        raise ValueError("carry interpolation cannot extrapolate beyond observed maturity support")
    return {key: np.exp(np.interp(requested, knots, np.log(value))) for key, value in curves.items()}


def jitter_geometry(records, batch_size, *, seed=0, strike_log_width=0.01, maturity_relative_width=0.05):
    """Create bounded synthetic geometries, never prices at new coordinates.

    Expiry shifts are shared by each original expiry block and kept within
    observed carry support and adjacent block midpoints. Strike shifts preserve
    slot order and option type. Each slot retains its original ``h/(D*F)``;
    these are controlled synthetic spreads, not observed market spreads.
    The returned records contain no ``mid`` and require actual repricing before
    they can enter :func:`encode_quotes`.
    """
    columns = validate_records(records)
    if columns["forward"].shape[0] != 1:
        raise ValueError("geometry jitter requires one parent geometry")
    batch_size = _positive_integer(batch_size, "batch_size")
    if (not np.isfinite(strike_log_width) or not 0 <= strike_log_width <= .1
            or not np.isfinite(maturity_relative_width) or not 0 <= maturity_relative_width <= .25):
        raise ValueError("jitter widths require log-strike [0,.1] and relative maturity [0,.25]")
    rng = _rng(seed)
    base = {key: value[0] for key, value in columns.items()}
    n = len(base["forward"])
    result = {key: np.repeat(value[None, :], batch_size, axis=0) for key, value in base.items()}
    times = np.unique(base["maturity"])
    for block, time in enumerate(times):
        selected = base["maturity"] == time
        left = times[0] if block == 0 else (times[block - 1] + time) / 2
        right = times[-1] if block == len(times) - 1 else (time + times[block + 1]) / 2
        lower, upper = max(left, time * (1 - maturity_relative_width)), min(right, time * (1 + maturity_relative_width))
        moved = rng.uniform(lower, upper, size=batch_size)
        result["maturity"][:, selected] = moved[:, None]
        logk = np.log(base["strike"][selected] / base["forward"][selected])
        order = np.argsort(logk, kind="stable")
        distances = np.diff(logk[order])
        if np.any(distances <= 0):
            raise ValueError("geometry jitter requires distinct strikes within each expiry")
        widths = np.full(len(logk), strike_log_width)
        if len(logk) > 1:
            widths[order[:-1]] = np.minimum(widths[order[:-1]], .49 * distances)
            widths[order[1:]] = np.minimum(widths[order[1:]], .49 * distances)
        shifted = logk[None, :] + rng.uniform(-widths, widths, size=(batch_size, len(logk)))
        result["strike"][:, selected] = np.exp(shifted)  # temporarily K/F
    carry = interpolate_carry(records, result["maturity"])
    result.update(carry)
    result["strike"] *= result["forward"]
    normalized_spread = base["half_spread"] / (base["forward"] * base["discount"])
    result["half_spread"] = normalized_spread[None, :] * result["forward"] * result["discount"]
    if result["strike"].shape != (batch_size, n):
        raise ValueError("geometry batch shape differs")
    validate_records(result)
    result["_requires_repricing"] = True
    result["_base_geometry_sha256"] = geometry_fingerprint(records)
    result["_spread_policy"] = "fixed normalized h/(D*F) per original slot; synthetic scenario"
    return result

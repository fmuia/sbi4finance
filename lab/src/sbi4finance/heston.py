"""Full-truncation Heston and independent Gaussian log-price jumps.

All inputs are annualized; observations are decimal log returns. The numerical
state is SIGNED auxiliary variance, not its nonnegative reported value. Initial
continuous-time variance is nonnegative; restarting this discrete scheme may
require a negative state. Parameters belong to the caller's declared P/Q task.
"""

import math

import numpy as np
from numba import njit


_PARAMETERS = ("drift", "kappa", "theta", "vol_of_vol", "rho",
               "jump_intensity", "jump_mean", "jump_std")


def _dt(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("dt must be a nonempty positive finite vector of year fractions")
    return np.ascontiguousarray(result)


def _batch(values, batch, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim > 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or batch vector")
    try:
        return np.broadcast_to(array, (batch,)).copy()
    except ValueError as error:
        raise ValueError(f"{name} must be a scalar or vector of length {batch}") from error


def _parameters(values, batch):
    if not set(_PARAMETERS[:5]).issubset(values) or set(values) - set(_PARAMETERS):
        raise ValueError(f"parameters require {_PARAMETERS[:5]}; only {_PARAMETERS} are supported")
    result = np.column_stack([_batch(values.get(name, 0.0), batch, name) for name in _PARAMETERS])
    if (np.any(result[:, 1] <= 0) or np.any(result[:, [2, 3, 5, 7]] < 0)
            or np.any(np.abs(result[:, 4]) > 1)):
        raise ValueError("require kappa>0; theta, vol_of_vol, jump_intensity, jump_std>=0; |rho|<=1")
    with np.errstate(over="raise", invalid="raise"):
        try:
            compensation = result[:, 5] * np.expm1(result[:, 6] + 0.5 * result[:, 7] ** 2)
        except FloatingPointError as error:
            raise ValueError("jump compensation overflows; unsupported parameter magnitudes") from error
    if not np.isfinite(compensation).all():
        raise ValueError("jump compensation must be finite")
    return np.ascontiguousarray(result)


def _unsigned(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) < 2**64):
        raise ValueError(f"{name} must be an integer in [0, 2**64)")
    return int(value)


def draw_innovations(dt, substeps, *, seed, example_ids, jump_intensity=0.0, stream_id=0):
    """Draw explicit inputs with separate PCG64 streams per example and purpose.

    Shapes: brownian [B,T,L,2], jump_counts/jump_normals [B,T]. Brownian
    channels are independent standard normals; simulate applies rho. Conditional
    Gaussian jump totals are exact over observation intervals, independently of
    diffusion substeps. Individual jump times/path extrema are not represented.

    IDs and streams are unsigned 64-bit integers. Changing batch order/size
    does not change an example. Use different stream_id values for history and
    future draws; slice prescribed inputs to test uninterrupted continuation.
    Call in modest batches: inputs use O(B*T*L) memory, not a whole-bank buffer.
    """
    steps = _dt(dt)
    seed, stream = _unsigned(seed, "seed"), _unsigned(stream_id, "stream_id")
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    ids = [_unsigned(i, "example ID") for i in example_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("example IDs must be nonempty and unique")
    intensity = _batch(jump_intensity, len(ids), "jump_intensity")
    if np.any(intensity < 0):
        raise ValueError("jump_intensity must be nonnegative")
    normals = np.empty((len(ids), len(steps), substeps, 2))
    counts = np.empty((len(ids), len(steps)), dtype=np.int64)
    jump_normals = np.empty((len(ids), len(steps)))
    for row, example_id in enumerate(ids):
        # Fixed-width words prevent ambiguity when joining large integer IDs.
        identity = [0x48535631, seed & 0xffffffff, seed >> 32,
                    example_id & 0xffffffff, example_id >> 32,
                    stream & 0xffffffff, stream >> 32]
        rngs = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, p])))
                for p in range(3)]
        normals[row] = rngs[0].standard_normal((len(steps), substeps, 2))
        counts[row] = rngs[1].poisson(intensity[row] * steps)
        jump_normals[row] = rngs[2].standard_normal(len(steps))
    return {"brownian": normals, "jump_counts": counts, "jump_normals": jump_normals}


def _numpy_steps(parameters, dt, initial, normals, counts, jump_normals):
    batch, times, substeps, _ = normals.shape
    returns = np.empty((batch, times, 1))
    states = np.empty((batch, times + 1, 1))
    states[:, 0, 0] = initial
    variance = initial.copy()
    drift, kappa, theta, xi, rho, intensity, jm, js = parameters.T
    independent_weight = np.sqrt(1 - rho * rho)
    compensation = intensity * np.expm1(jm + 0.5 * js * js)
    for t in range(times):
        h = dt[t] / substeps
        total = np.zeros(batch)
        for step in range(substeps):
            positive = np.maximum(variance, 0.0)
            scale = np.sqrt(positive * h)
            zv, independent = normals[:, t, step, :].T
            total += (drift - 0.5 * positive) * h + scale * (rho * zv + independent_weight * independent)
            variance += kappa * (theta - positive) * h + xi * scale * zv
        total += counts[:, t] * jm + js * np.sqrt(counts[:, t]) * jump_normals[:, t] - compensation * dt[t]
        returns[:, t, 0] = total
        states[:, t + 1, 0] = variance
    return returns, states


@njit(cache=True, fastmath=False)
def _numba_steps(parameters, dt, initial, normals, counts, jump_normals):
    batch, times, substeps, _ = normals.shape
    returns = np.empty((batch, times, 1))
    states = np.empty((batch, times + 1, 1))
    for b in range(batch):
        drift, kappa, theta, xi, rho, intensity, jm, js = parameters[b]
        independent_weight = math.sqrt(1 - rho * rho)
        compensation = intensity * math.expm1(jm + 0.5 * js * js)
        variance = initial[b]
        states[b, 0, 0] = variance
        for t in range(times):
            h = dt[t] / substeps
            total = 0.0
            for step in range(substeps):
                positive = max(variance, 0.0)
                scale = math.sqrt(positive * h)
                zv, independent = normals[b, t, step]
                total += (drift - 0.5 * positive) * h + scale * (rho * zv + independent_weight * independent)
                variance += kappa * (theta - positive) * h + xi * scale * zv
            total += counts[b, t] * jm + js * math.sqrt(counts[b, t]) * jump_normals[b, t] - compensation * dt[t]
            returns[b, t, 0] = total
            states[b, t + 1, 0] = variance
    return returns, states


def simulate(parameters, dt, variance_state, innovations, *, backend="numba"):
    """Advance the declared discrete model from its complete variance state.

    Required parameter keys: drift, kappa, theta, vol_of_vol, rho. Optional
    jump_intensity, jump_mean and jump_std default to zero. Scalars and batch
    vectors are accepted. Drift is TOTAL expected proportional price growth:
    supply mu under P, or r-q under Q with separately chosen Q parameters.

    Output: log_returns [B,T,1] and signed variance_state [B,T+1,1], including
    the starting state. max(variance_state,0) is effective variance, NOT the
    state to use on restart. The latter array is latent truth/auxiliary output,
    not an observed conditioning variable. No measure change is inferred here.

    innovations must follow draw_innovations' contract. Supplying prescribed
    inputs is also supported for matched tests; their distribution is the
    caller's responsibility. The kernel validates shapes/values, not randomness.
    """
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be numpy or numba")
    steps = _dt(dt)
    if set(innovations) != {"brownian", "jump_counts", "jump_normals"}:
        raise ValueError("innovations need brownian, jump_counts and jump_normals")
    normals = np.asarray(innovations["brownian"], dtype=np.float64)
    if (normals.ndim != 4 or normals.shape[0] < 1 or normals.shape[1] != len(steps)
            or normals.shape[2] < 1 or normals.shape[3] != 2 or not np.isfinite(normals).all()):
        raise ValueError("brownian must be finite [batch,time,substeps,2]")
    batch = normals.shape[0]
    counts = np.asarray(innovations["jump_counts"])
    jumps = np.asarray(innovations["jump_normals"], dtype=np.float64)
    if (counts.shape != (batch, len(steps)) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or np.any(counts > np.iinfo(np.int64).max)):
        raise ValueError("jump_counts must be nonnegative int64-compatible [batch,time]")
    if jumps.shape != counts.shape or not np.isfinite(jumps).all():
        raise ValueError("jump_normals must be finite [batch,time]")
    params = _parameters(parameters, batch)
    initial = _batch(variance_state, batch, "variance_state")
    kernel = _numba_steps if backend == "numba" else _numpy_steps
    returns, states = kernel(params, steps, initial, np.ascontiguousarray(normals),
                             np.ascontiguousarray(counts, dtype=np.int64), np.ascontiguousarray(jumps))
    if not np.isfinite(returns).all() or not np.isfinite(states).all():
        raise ValueError("nonfinite trajectory; reject this parameter/grid choice without redrawing")
    return {"log_returns": returns, "variance_state": states}

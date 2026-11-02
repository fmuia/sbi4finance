"""Heston–Kou double-exponential jumps and European Fourier prices.

Paths reuse Heston's signed full-truncation state. The characteristic function
and pricer describe the continuous-time model instead. These are deliberately
separate numerical approximations; path/payoff and quadrature refinement remain
necessary. No physical/risk-neutral parameter conversion is supplied.
"""

from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

from . import gbm, heston

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.heston_options.model import (
    PARAMETERS,
    validate_parameters,
    _compensation,
    characteristic_function,
    _quadrature,
    european_price,
)


MODEL_ID = "heston_kou_double_exponential_v1"


def jump_compensation(parameters):
    """Return lambda*(E[exp(J)]-1), in inverse years."""
    return _compensation(validate_parameters(parameters))


def draw_innovations(dt, substeps, *, seed, example_ids, parameters, stream_id=0):
    """Explicit per-example Brownian and exact interval jump totals.

    Up/down counts are independent Poisson(lambda*p*dt), Poisson(lambda*(1-p)*dt).
    Their positive magnitude sums are Gamma(count, scale=1/rate), with zero
    for a zero count. Brownian shape is [B,T,L,2]; other arrays are [B,T].
    Observation-interval jump totals cannot resolve barriers or jump times.
    """
    steps = heston._dt(dt)
    seed, stream = heston._unsigned(seed, "seed"), heston._unsigned(stream_id, "stream_id")
    if isinstance(substeps, (bool, np.bool_)) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    ids = [heston._unsigned(value, "example ID") for value in example_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("example IDs must be nonempty and unique")
    values = {key: heston._batch(value, len(ids), key)
              for key, value in validate_parameters(parameters).items()}
    shape = (len(ids), len(steps))
    result = {"brownian": np.empty((*shape, substeps, 2)),
              "jump_up_counts": np.empty(shape, dtype=np.int64),
              "jump_down_counts": np.empty(shape, dtype=np.int64),
              "jump_up_sums": np.empty(shape), "jump_down_sums": np.empty(shape)}
    for row, example_id in enumerate(ids):
        identity = [0x484B4431, seed & 0xffffffff, seed >> 32, example_id & 0xffffffff,
                    example_id >> 32, stream & 0xffffffff, stream >> 32]
        rngs = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([*identity, purpose])))
                for purpose in range(5)]
        result["brownian"][row] = rngs[0].standard_normal((len(steps), substeps, 2))
        lam, p = values["jump_intensity"][row], values["jump_up_probability"][row]
        for side, intensity, count_rng, gamma_rng in (("up", lam*p, rngs[1], rngs[3]),
                                                    ("down", lam*(1-p), rngs[2], rngs[4])):
            counts = count_rng.poisson(intensity * steps)
            result[f"jump_{side}_counts"][row] = counts
            result[f"jump_{side}_sums"][row] = gamma_rng.gamma(counts, 1 / values[f"jump_{side}_rate"][row])
    return result


def simulate(parameters, dt, variance_state, innovations, *, backend="numba"):
    """Return log_returns [B,T,1], signed variance_state [B,T+1,1].

    Under Q set drift=r-q, or drift=0 to simulate S_T/F_T from initial 1.
    The complete signed auxiliary variance is necessary for continuation.
    Brownian and jump sums can be sliced to check segmented continuation.
    """
    values = validate_parameters(parameters)
    steps = heston._dt(dt)
    required = {"brownian", "jump_up_counts", "jump_down_counts", "jump_up_sums", "jump_down_sums"}
    if set(innovations) != required:
        raise ValueError(f"innovations require {sorted(required)}")
    brownian = np.asarray(innovations["brownian"], dtype=np.float64)
    if brownian.ndim != 4:
        raise ValueError("brownian must have shape [B,T,L,2]")
    shape = brownian.shape[:2]
    totals = []
    for side in ("up", "down"):
        counts = np.asarray(innovations[f"jump_{side}_counts"])
        sums = np.asarray(innovations[f"jump_{side}_sums"], dtype=np.float64)
        if (counts.shape != shape or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 0) or np.any(counts > np.iinfo(np.int64).max)):
            raise ValueError("jump counts must be nonnegative int64-compatible [B,T]")
        if (sums.shape != shape or not np.isfinite(sums).all() or np.any(sums < 0)
                or np.any((counts == 0) & (sums != 0))):
            raise ValueError("jump sums must be finite nonnegative [B,T], zero when counts are zero")
        totals.append(sums)
    diffusion = {key: values[key] for key in ("kappa", "theta", "vol_of_vol", "rho")}
    diffusion["drift"] = values["drift"] - _compensation(values)
    result = heston.simulate(diffusion, steps, variance_state,
                            {"brownian": brownian, "jump_counts": np.zeros(shape, dtype=np.int64),
                             "jump_normals": np.zeros(shape)}, backend=backend)
    result["log_returns"][:, :, 0] += totals[0] - totals[1]
    if not np.isfinite(result["log_returns"]).all():
        raise ValueError("nonfinite trajectory; reject parameter/grid without redrawing")
    return result

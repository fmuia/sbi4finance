"""Finite-window cascade model, with explicit prior and accepted-path law.

Design credit: https://github.com/stefanradev93/amortized-fractals
Pinned design: 069d5dc5943dbe85505849f2fa8bfb563617ff97.
Returns are simple daily returns; all 256 days are accepted together before any
shorter segment is selected. This is a scenario model, not a latent-state filter.
"""

import hashlib
import json

import numpy as np

WINDOW = 256
PARAMETERS = ("mu", "sigma_bar", "q", "nu")
LABELS = ("Daily location μ", "Baseline scale σ̄", "Cascade contrast q", "Tail degrees ν")
# The descending order is part of the pinned simulation random-stream contract.
SUPPORTED_D = (128, 64, 32, 16, 8, 4)
CAPSTONE_D = (4, 8, 128)
BOUNDS = ((None, None), (.005, .025), (.52, .90), (2.05, 12.))
DESIGN = {
    "format": "mmar_takehome_v1", "window": WINDOW, "parameter_order": PARAMETERS,
    "return_type": "daily_simple", "supported_d": SUPPORTED_D,
    "prior": {"mu": {"family": "normal", "mean": 0., "sd": .0015},
              "sigma_bar": {"family": "uniform", "bounds": [.005, .025]},
              "q": {"family": "uniform", "bounds": [.52, .90]},
              "nu_minus_two": {"family": "log_uniform", "bounds": [.05, 10.]}},
    "acceptance": "redraw the complete 256-day path when nonfinite or any simple return <= -1",
    "cascade": "binary contrasts 2q/2(1-q), normalized mean one, circular random phase",
    "upstream_revision": "069d5dc5943dbe85505849f2fa8bfb563617ff97",
}


def fingerprint(design=None):
    """Canonical model/prior identity; tuples and decoded JSON lists agree."""
    return hashlib.sha256(json.dumps(DESIGN if design is None else design,
                                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _count(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def resolutions(n, d, rng):
    """Validate one known block count per path, or draw the source's design prior."""
    if d is None:
        return rng.choice(SUPPORTED_D, size=n)
    counts = np.full(n, d) if np.ndim(d) == 0 else np.asarray(d)
    if counts.shape != (n,) or not np.isin(counts, SUPPORTED_D).all():
        raise ValueError(f"d must supply one of {SUPPORTED_D} for each of {n} paths")
    return counts.astype(int)


def prior(n, rng):
    """Independent parameters; ν−2 is log-uniform, not ν-uniform."""
    n = _count(n, "number of parameter draws")
    scale = rng.uniform(.005, .025, n)
    contrast = rng.uniform(.52, .90, n)
    tails = 2 + np.exp(rng.uniform(np.log(2.05 - 2), np.log(12. - 2), n))
    location = rng.normal(0., .0015, n)
    return np.column_stack((location, scale, contrast, tails))


def parameter_array(values):
    """Preserve the named target order; do not rely on dictionary insertion order."""
    result = (np.concatenate([np.asarray(values[k]) for k in PARAMETERS], axis=-1)
              if isinstance(values, dict) else np.asarray(values, dtype=float))
    if result.ndim < 2 or result.shape[-1] != 4 or not np.isfinite(result).all():
        raise ValueError("provide finite complete parameter rows in μ, σ̄, q, ν order")
    return result


def cascade(contrast, counts, rng):
    """Mean-one intensities in d blocks, with a uniformly random circular phase."""
    contrast = np.asarray(contrast, dtype=float)
    counts = resolutions(len(contrast), counts, rng)
    intensities = np.empty((len(contrast), WINDOW))
    for blocks in np.unique(counts):
        rows = np.flatnonzero(counts == blocks)
        mass = np.ones((len(rows), 1))
        hi, lo = 2 * contrast[rows, None], 2 * (1 - contrast[rows, None])
        for _level in range(int(np.log2(blocks))):
            orientation = rng.random(mass.shape) < .5
            child_factors = np.stack((np.where(orientation, hi, lo),
                                      np.where(orientation, lo, hi)), axis=-1)
            mass = (mass[..., None] * child_factors).reshape(len(rows), -1)
        mass = mass / mass.mean(axis=1, keepdims=True)
        daily = np.repeat(mass, WINDOW // blocks, axis=1)
        phase = rng.integers(0, WINDOW, size=(len(rows), 1))
        intensities[rows] = np.take_along_axis(daily, (np.arange(WINDOW)[None] + phase) % WINDOW, axis=1)
    return intensities


def simulate_parameters(parameters, d=8, *, seed=None, rng=None):
    """One accepted complete path per parameter row; never discard hard parameters."""
    theta = parameter_array(parameters)
    if theta.ndim != 2 or np.any(theta[:, 1] <= 0) or np.any((theta[:, 2] < .5) | (theta[:, 2] >= 1)) or np.any(theta[:, 3] <= 2):
        raise ValueError("require σ̄ > 0, .5 <= q < 1 and ν > 2")
    if rng is not None and seed is not None:
        raise ValueError("provide rng or seed, not both")
    rng = np.random.default_rng(seed) if rng is None else rng
    counts = resolutions(len(theta), d, rng)
    returns, weights = np.empty((2, len(theta), WINDOW))
    pending = np.arange(len(theta))
    for _attempt in range(100):
        weights[pending] = cascade(theta[pending, 2], counts[pending], rng)
        nu = theta[pending, 3, None]
        innovation = rng.standard_t(nu, size=(len(pending), WINDOW)) * np.sqrt((nu - 2) / nu)
        returns[pending] = theta[pending, 0, None] + theta[pending, 1, None] * np.sqrt(weights[pending]) * innovation
        valid = np.isfinite(returns[pending]).all(axis=1) & (returns[pending] > -1).all(axis=1)
        pending = pending[~valid]
        if not len(pending):
            break
    else:
        raise RuntimeError("256-day acceptance failed after 100 redraws; no parameter rows were dropped")
    result = {name: theta[:, j:j+1] for j, name in enumerate(PARAMETERS)}
    result.update(returns=returns, trading_time_increments=weights,
                  d=counts[:, None], log_d=np.log(counts)[:, None])
    return result


def simulate(n, d=None, *, seed=20261112):
    rng = np.random.default_rng(seed)
    theta = prior(n, rng)
    counts = resolutions(len(theta), d, rng)
    return simulate_parameters(theta, counts, rng=rng)


def conditions(returns, d=8):
    """Only observations and the supplied assumption enter conditional inference."""
    if np.iscomplexobj(returns) or np.ma.isMaskedArray(returns) or d is None:
        raise ValueError("provide real unmasked returns and a supplied known d")
    values = np.asarray(returns, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :, None]
    elif values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3 or values.shape[1:] != (WINDOW, 1) or not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("provide accepted simple-return windows [batch,256,1]")
    counts = resolutions(len(values), d, np.random.default_rng(0))
    return {"returns": values, "log_d": np.log(counts).astype("float32")[:, None]}

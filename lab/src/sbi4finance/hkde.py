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


PARAMETERS = ("kappa", "theta", "vol_of_vol", "rho", "jump_intensity",
              "jump_up_probability", "jump_up_rate", "jump_down_rate")
MODEL_ID = "heston_kou_double_exponential_v1"


def validate_parameters(parameters):
    """Validate mathematical support, returning broadcast scalar/1D arrays.

    ``drift`` is optional (default zero) and affects paths only. It is total
    proportional expected growth before subtracting jump compensation. Rates
    are inverse decimal log-jump sizes. If lambda*p>0, eta_up must exceed one.
    Zero intensity and one-sided jumps are supported without evaluating an
    absent component's nonexistent exponential moment. This is mathematical
    support, not a pricing-accuracy certificate for the entire parameter space.
    """
    if not set(PARAMETERS).issubset(parameters) or set(parameters) - set(PARAMETERS) - {"drift"}:
        raise ValueError(f"require {PARAMETERS}; only optional drift is supported")
    keys = (*PARAMETERS, "drift")
    values = [np.asarray(parameters.get(key, 0.0), dtype=np.float64) for key in keys]
    if any(value.ndim > 1 or not value.size or not np.isfinite(value).all() for value in values):
        raise ValueError("parameters must be finite scalars or nonempty batch vectors")
    try:
        result = dict(zip(keys, np.broadcast_arrays(*values), strict=True))
    except ValueError as error:
        raise ValueError("parameter batch vectors must broadcast") from error
    if (np.any(result["kappa"] <= 0) or np.any(result["theta"] < 0)
            or np.any(result["vol_of_vol"] < 0) or np.any(np.abs(result["rho"]) > 1)
            or np.any(result["jump_intensity"] < 0)
            or np.any((result["jump_up_probability"] < 0) | (result["jump_up_probability"] > 1))
            or np.any(result["jump_up_rate"] <= 0) or np.any(result["jump_down_rate"] <= 0)):
        raise ValueError("require kappa, jump rates>0; theta, xi, lambda>=0; |rho|<=1; 0<=p<=1")
    upward_mass = (result["jump_intensity"] > 0) & (result["jump_up_probability"] > 0)
    if np.any(upward_mass & (result["jump_up_rate"] <= 1)):
        raise ValueError("positive upward jump intensity requires jump_up_rate>1 for finite E[S]")
    return result


def _compensation(values):
    lam, p = values["jump_intensity"], values["jump_up_probability"]
    up = np.divide(p, values["jump_up_rate"] - 1, out=np.zeros_like(p), where=(lam > 0) & (p > 0))
    # Algebraically E[e^J]-1 = p/(eta_up-1) - (1-p)/(eta_down+1).
    result = lam * (up - (1 - p) / (values["jump_down_rate"] + 1))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite jump compensation; unsupported parameter magnitude")
    return result


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


def characteristic_function(u, maturity, parameters, initial_variance):
    """E[exp(i*u*log(S_T/F_T))] for scalar model parameters and v0>=0.

    Frequencies and maturities broadcast. The supported complex strip is
    -1 <= Im(u) <= 0, which includes the half-shift European inversion contour.
    Uses the decaying-exponential Heston formula and explicit xi=0 limit.
    Caller supplies Q parameters when using this for option prices; drift is
    deliberately omitted because the forward normalization removes carry.
    """
    values = validate_parameters(parameters)
    if any(value.ndim != 0 for value in values.values()):
        raise ValueError("characteristic_function requires scalar model parameters")
    v0 = float(gbm._scalar(initial_variance, "initial_variance", nonnegative=True))
    frequency, time = np.broadcast_arrays(np.asarray(u, dtype=np.complex128),
                                          np.asarray(maturity, dtype=np.float64))
    if (not np.isfinite(frequency).all() or not np.isfinite(time).all() or np.any(time < 0)
            or np.any(frequency.imag < -1) or np.any(frequency.imag > 0)):
        raise ValueError("require finite T>=0 and frequencies in -1<=Im(u)<=0")
    kappa, theta, xi, rho, lam, p, up, down = (float(values[key]) for key in PARAMETERS)
    # Exact removable limits, also avoiding the alternate sqrt branch at -i.
    identity = (frequency == 0) | (frequency == -1j) | (time == 0)
    frequency = np.where(identity, 0j, frequency)
    iu = 1j * frequency
    if xi == 0:
        integrated = theta*time + (v0-theta)*(-np.expm1(-kappa*time))/kappa
        exponent = -0.5*(frequency**2 + iu)*integrated
    else:
        b = kappa - rho*xi*iu
        d = np.sqrt(b*b + xi*xi*(frequency**2 + iu))
        a = -(frequency**2 + iu)/(b+d)  # (b-d)/xi**2 without cancellation
        g = xi*xi*a/(b+d)
        decay = np.exp(-d*time)
        log_ratio = np.log1p(-g*decay) - np.log1p(-g)
        exponent = (kappa*theta*(a*time - 2*log_ratio/(xi*xi))
                    + v0*a*(-np.expm1(-d*time))/(1-g*decay))
    if lam > 0:
        jump_cf = np.zeros_like(frequency)
        if p > 0:
            jump_cf += p*up/(up-iu)
        if p < 1:
            jump_cf += (1-p)*down/(down+iu)
        exponent += time*(lam*(jump_cf-1) - iu*float(_compensation(values)))
    result = np.where(identity, 1+0j, np.exp(exponent))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite characteristic function; unsupported numerical parameters")
    return result


@lru_cache(maxsize=8)
def _quadrature(nodes, upper_limit):
    roots, weights = roots_legendre(nodes)
    return 0.5*upper_limit*(roots+1), 0.5*upper_limit*weights


def european_price(forward, strike, discount, maturity, parameters, initial_variance,
                   *, kind="call", upper_limit=200.0, nodes=512):
    """Broadcast European Q prices via a Lewis half-shift Fourier integral.

    F,K,D>0, T>=0; model parameters and v0 are scalar. Numerical controls are
    the frequency truncation and Gauss–Legendre node count, NOT an error bound.
    Compare both enlarged cutoff and node count before accepting a new region.
    The bank stores the actual refinement discrepancy separately from noise.
    No clipping hides quadrature errors: material price-bound violations fail.
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be call or put")
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)) or nodes < 16:
        raise ValueError("nodes must be an integer >=16")
    cutoff = gbm._scalar(upper_limit, "upper_limit", positive=True)
    fwd, k, disc, time = np.broadcast_arrays(*[np.asarray(value, dtype=np.float64)
                                              for value in (forward, strike, discount, maturity)])
    if any(not value.size or not np.isfinite(value).all() or np.any(value <= 0)
           for value in (fwd, k, disc)) or not np.isfinite(time).all() or np.any(time < 0):
        raise ValueError("require finite F,K,D>0 and T>=0")
    # Validate even at expiry, where numerical integration is unnecessary.
    characteristic_function(0.0, 0.0, parameters, initial_variance)
    values = validate_parameters(parameters)
    if float(values["vol_of_vol"]) == 0 and float(values["jump_intensity"]) == 0:
        kappa, theta = float(values["kappa"]), float(values["theta"])
        integrated = theta*time + (initial_variance-theta)*(-np.expm1(-kappa*time))/kappa
        return gbm.european_price(fwd, k, disc, integrated, option=kind)
    u, weights = _quadrature(int(nodes), cutoff)
    # A surface repeats each maturity across strikes. The characteristic
    # function depends on maturity, not strike; retain exact floating values.
    unique_time, inverse = np.unique(time.reshape(-1), return_inverse=True)
    if len(unique_time) < time.size:
        phi = characteristic_function(u-0.5j, unique_time[:, None], parameters, initial_variance)
        phi = phi[inverse].reshape((*time.shape, len(u)))
    else:
        phi = characteristic_function(u-0.5j, time[..., None], parameters, initial_variance)
    integrand = np.real(np.exp(1j*u*np.log(fwd/k)[..., None])*phi)/(u*u+0.25)
    call = disc*(fwd - np.sqrt(fwd*k)*np.sum(integrand*weights, axis=-1)/np.pi)
    call = np.where(time == 0, disc*np.maximum(fwd-k, 0), call)
    price = call if kind == "call" else call-disc*(fwd-k)
    lower = disc*np.maximum(fwd-k if kind == "call" else k-fwd, 0)
    upper = disc*(fwd if kind == "call" else k)
    tolerance = 1e-9*disc*np.maximum(fwd, k)
    if (not np.isfinite(price).all() or np.any(price < lower-tolerance)
            or np.any(price > upper+tolerance)):
        raise ValueError("Fourier price violates bounds; refine controls or reject this domain")
    return price

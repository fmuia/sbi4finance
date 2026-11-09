"""Heston prices by one Fourier integral, in forward units: the exact likelihood of slot 4.

The variance v follows dv = kappa (theta - v) dt + vol_of_vol sqrt(v) dW, correlated with the price by rho,
from v0 today. The characteristic function of log(S_T / F_T) is closed form, and a European price is one
integral of it (Lewis's half-shift contour), computed by Gauss-Legendre quadrature on [0, upper_limit].
The quadrature is a numerical control, not an error bound: the task's ladder (tasks.heston_prices) refines it
until two rungs agree within a stated share of the quote noise.

The generalized model also supports Kou double-exponential jumps
(Heston-Kou, "hkde"). The Heston task fixes their intensity at zero, so the
jump terms below never act. Lab option workers reuse these pricing functions.
"""

from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

from ..shared import gbm

PARAMETERS = ("kappa", "theta", "vol_of_vol", "rho", "jump_intensity",
              "jump_up_probability", "jump_up_rate", "jump_down_rate")


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
    return (_read_only(0.5*upper_limit*(roots+1)),
            _read_only(0.5*upper_limit*weights))


def _read_only(array):
    """A byte-backed array cannot be made writable by a cache consumer."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


# Four grid plans of at most 16 MiB each. Oversized/non-repeating grids still
# use the same Fourier formula without retaining their phase arrays. The
# separately cached quadrature vectors are shared, not copied into each plan.
_MAX_PLAN_BYTES = 16 * 1024 * 1024


@lru_cache(maxsize=4)
def _fourier_plan(shape, forward_bytes, strike_bytes, maturity_bytes, nodes, upper_limit):
    """Immutable strike phases and exact maturity maps, keyed by grid content.

    Keys retain the full float64 values and order, so mutating a caller's grid
    or changing even one maturity creates a different plan. No model parameter
    or price is cached. Arithmetic order matches the uncached formula.
    """
    forward = np.frombuffer(forward_bytes, dtype=np.float64).reshape(shape)
    strike = np.frombuffer(strike_bytes, dtype=np.float64).reshape(shape)
    maturity = np.frombuffer(maturity_bytes, dtype=np.float64).reshape(shape)
    u, _ = _quadrature(nodes, upper_limit)
    phase = _read_only(np.exp(1j*u*np.log(forward/strike)[..., None]))
    unique_time, inverse = np.unique(maturity.reshape(-1), return_inverse=True)
    return phase, _read_only(unique_time), _read_only(inverse)


def _grid_plan(forward, strike, maturity, nodes, upper_limit):
    count = maturity.size
    # The phase array dominates; include grid keys and index/time arrays in
    # the admission bound so unusual large caller grids cannot fill the cache.
    estimate = count * (16*nodes + 40)
    if estimate > _MAX_PLAN_BYTES:
        return None
    return _fourier_plan(maturity.shape, forward.tobytes(), strike.tobytes(),
                         maturity.tobytes(), nodes, upper_limit)


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
    return _european_price(forward, strike, discount, maturity, parameters, initial_variance,
                           put=kind == "put", upper_limit=upper_limit, nodes=nodes)


def _european_price(forward, strike, discount, maturity, parameters, initial_variance,
                    *, put, upper_limit, nodes):
    """One integral for a scalar type or a mixed call/put surface.

    ``put`` is an internal boolean scalar or a broadcast contract mask. Price
    bounds apply to each requested option, rather than to an intermediate call
    that a put-only caller did not request.
    """
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
        if np.ndim(put) == 0:
            return gbm.european_price(fwd, k, disc, integrated, option="put" if put else "call")
        # Keep Black's direct put formula in this exact limit. Subtracting
        # parity from a call would introduce unnecessary cancellation.
        price = np.empty_like(time)
        for is_put in (False, True):
            chosen = put == is_put
            if np.any(chosen):
                price[chosen] = gbm.european_price(fwd[chosen], k[chosen], disc[chosen],
                                                  integrated[chosen],
                                                  option="put" if is_put else "call")
        return price
    u, weights = _quadrature(int(nodes), cutoff)
    # A surface repeats each maturity across strikes. The characteristic
    # function depends on maturity, not strike; retain exact floating values.
    plan = _grid_plan(fwd, k, time, int(nodes), cutoff)
    if plan is None:
        phase = np.exp(1j*u*np.log(fwd/k)[..., None])
        unique_time, inverse = np.unique(time.reshape(-1), return_inverse=True)
    else:
        phase, unique_time, inverse = plan
    if len(unique_time) < time.size:
        phi = characteristic_function(u-0.5j, unique_time[:, None], parameters, initial_variance)
        phi = phi[inverse].reshape((*time.shape, len(u)))
    else:
        phi = characteristic_function(u-0.5j, time[..., None], parameters, initial_variance)
    integrand = np.real(phase*phi)/(u*u+0.25)
    call = disc*(fwd - np.sqrt(fwd*k)*np.sum(integrand*weights, axis=-1)/np.pi)
    call = np.where(time == 0, disc*np.maximum(fwd-k, 0), call)
    price = np.where(put, call-disc*(fwd-k), call)
    lower = disc*np.maximum(np.where(put, k-fwd, fwd-k), 0)
    upper = disc*np.where(put, k, fwd)
    tolerance = 1e-9*disc*np.maximum(fwd, k)
    if (not np.isfinite(price).all() or np.any(price < lower-tolerance)
            or np.any(price > upper+tolerance)):
        raise ValueError("Fourier price violates bounds; refine controls or reject this domain")
    return price

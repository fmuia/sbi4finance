"""Frozen pre-maturity-reuse Heston–Kou pricer for regression and timing only.

Production callers use the tutorial-owned model. Keep this deliberately simple
reference unchanged when optimizing that implementation.
"""

import numpy as np

from sbi4finance import gbm, hkde


def original_price(forward, strike, discount, maturity, parameters, initial_variance,
                   *, kind="call", upper_limit=200.0, nodes=512):
    """Frozen pre-optimization reference, with one CF row per contract."""
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
    hkde.characteristic_function(0.0, 0.0, parameters, initial_variance)
    values = hkde.validate_parameters(parameters)
    if float(values["vol_of_vol"]) == 0 and float(values["jump_intensity"]) == 0:
        kappa, theta = float(values["kappa"]), float(values["theta"])
        integrated = theta*time + (initial_variance-theta)*(-np.expm1(-kappa*time))/kappa
        return gbm.european_price(fwd, k, disc, integrated, option=kind)
    u, weights = hkde._quadrature(int(nodes), cutoff)
    phi = hkde.characteristic_function(u-0.5j, time[..., None], parameters, initial_variance)
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


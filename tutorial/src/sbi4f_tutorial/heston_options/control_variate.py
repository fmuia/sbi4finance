"""Lewis Heston inversion with an analytic Black control variate.

The characteristic function, financial model and quadrature plans are owned
by ``model``. This equivalent representation integrates phi_Black - phi_Heston
and adds the analytic Black price. It avoids subtracting two spot-sized terms
when the requested option is nearly worthless. No price clipping is applied.

The control's integrated variance is E[integral v dt]. Its precise choice
affects numerical efficiency, not the infinite-integral Heston price.
"""
import numpy as np

from . import model
from ..shared import gbm


def european_price(forward, strike, discount, maturity, parameters, initial_variance,
                   *, put=False, upper_limit=200., nodes=512):
    """Mixed European prices with unchanged CF, bounds and numerical controls.

    For either option type, price_H = price_Black + D sqrt(FK)/pi times the
    integral of Re[exp(iu log(F/K)) (phi_Black - phi_H)]/(u^2+1/4).
    Numerical truncation/refinement remains the caller's responsibility.
    """
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer)) or nodes < 16:
        raise ValueError("nodes must be an integer >=16")
    cutoff = gbm._scalar(upper_limit, "upper_limit", positive=True)
    if any(np.iscomplexobj(value) for value in (forward, strike, discount, maturity, initial_variance)):
        raise ValueError("real pricing geometry and variance required")
    f, k, d, t = np.broadcast_arrays(*[np.asarray(value, dtype=np.float64)
        for value in (forward, strike, discount, maturity)])
    mask = np.broadcast_to(np.asarray(put, dtype=bool), t.shape)
    if (any(not value.size or not np.isfinite(value).all() or np.any(value <= 0) for value in (f, k, d))
            or not np.isfinite(t).all() or np.any(t < 0)):
        raise ValueError("require finite F,K,D>0 and T>=0")
    model.characteristic_function(0., 0., parameters, initial_variance)
    pars = model.validate_parameters(parameters)
    if any(value.ndim != 0 for value in pars.values()):
        raise ValueError("scalar model parameters required")
    kappa, theta = float(pars["kappa"]), float(pars["theta"])
    v0 = float(initial_variance)
    integrated = theta*t + (v0-theta)*(-np.expm1(-kappa*t))/kappa
    black = np.where(mask,
        gbm.european_price(f, k, d, integrated, option="put"),
        gbm.european_price(f, k, d, integrated, option="call"))
    if float(pars["vol_of_vol"]) == 0 and float(pars["jump_intensity"]) == 0:
        return black
    u, weights = model._quadrature(int(nodes), cutoff)
    plan = model._grid_plan(f, k, t, int(nodes), cutoff)
    if plan is None:
        phase = np.exp(1j*u*np.log(f/k)[..., None])
        unique_t, inverse = np.unique(t.reshape(-1), return_inverse=True)
    else:
        phase, unique_t, inverse = plan
    unique_variance = theta*unique_t + (v0-theta)*(-np.expm1(-kappa*unique_t))/kappa
    phi_h = model.characteristic_function(u-.5j, unique_t[:, None], parameters, v0)
    phi_b = np.exp(-.5*unique_variance[:, None]*(u*u+.25))
    correction = (phi_b-phi_h)[inverse].reshape((*t.shape, len(u)))
    integrand = np.real(phase*correction)/(u*u+.25)
    price = black + d*np.sqrt(f*k)*np.sum(integrand*weights, axis=-1)/np.pi
    price = np.where(t == 0, d*np.maximum(np.where(mask, k-f, f-k), 0), price)
    lower = d*np.maximum(np.where(mask, k-f, f-k), 0)
    upper = d*np.where(mask, k, f)
    tolerance = 1e-9*d*np.maximum(f, k)
    if (not np.isfinite(price).all() or np.any(price < lower-tolerance)
            or np.any(price > upper+tolerance)):
        raise ValueError("controlled Fourier price violates bounds; refine controls or reject this domain")
    return price

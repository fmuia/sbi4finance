"""Heston prices by one Fourier integral, in forward units: the exact likelihood of slot 4.

The variance v follows dv = kappa (theta - v) dt + vol_of_vol sqrt(v) dW, correlated with the price by rho,
from v0 today. The characteristic function of log(S_T / F_T) is closed form, and a European price is one
integral of it (Lewis's half-shift contour), computed by Gauss-Legendre quadrature on [0, upper_limit].
The quadrature is a numerical control, not an error bound: the task's ladder refines it until two rungs
agree within a stated share of the quote noise.
This is the lab's hkde.py with the jumps switched off, as in the Heston check task.
"""

from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

from . import tasks


def characteristic_function(u, maturity, kappa, theta, vol_of_vol, rho, v0):
    """E[exp(i u log(S_T / F_T))] for -1 <= Im(u) <= 0; frequencies and maturities broadcast."""
    frequency, time = np.broadcast_arrays(np.asarray(u, dtype=np.complex128), np.asarray(maturity, dtype=np.float64))
    # Exact removable limits, also avoiding the alternate square-root branch at -i.
    identity = (frequency == 0) | (frequency == -1j) | (time == 0)
    frequency = np.where(identity, 0j, frequency)
    iu = 1j * frequency
    xi = vol_of_vol
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
    result = np.where(identity, 1+0j, np.exp(exponent))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite characteristic function; unsupported numerical parameters")
    return result


@lru_cache(maxsize=8)
def _quadrature(nodes, upper_limit):
    roots, weights = roots_legendre(nodes)
    return 0.5*upper_limit*(roots+1), 0.5*upper_limit*weights


def european_price(forward, strike, discount, maturity, kappa, theta, vol_of_vol, rho, v0,
                   *, kind="call", upper_limit=200.0, nodes=512):
    """European prices; F, K, D > 0 and T >= 0 broadcast, the model parameters are scalars.

    A price outside the no-arbitrage bounds raises: the quadrature is too coarse there.
    """
    if not (kappa > 0 and theta >= 0 and vol_of_vol >= 0 and abs(rho) <= 1 and v0 >= 0):
        raise ValueError("require kappa > 0, theta >= 0, vol_of_vol >= 0, |rho| <= 1 and v0 >= 0")
    fwd, k, disc, time = np.broadcast_arrays(*[np.asarray(value, dtype=np.float64)
                                              for value in (forward, strike, discount, maturity)])
    u, weights = _quadrature(int(nodes), float(upper_limit))
    phi = characteristic_function(u-0.5j, time[..., None], kappa, theta, vol_of_vol, rho, v0)
    integrand = np.real(np.exp(1j*u*np.log(fwd/k)[..., None])*phi)/(u*u+0.25)
    call = disc*(fwd - np.sqrt(fwd*k)*np.sum(integrand*weights, axis=-1)/np.pi)
    call = np.where(time == 0, disc*np.maximum(fwd-k, 0), call)
    price = call if kind == "call" else call-disc*(fwd-k)
    lower = disc*np.maximum(fwd-k if kind == "call" else k-fwd, 0)
    upper = disc*(fwd if kind == "call" else k)
    tolerance = 1e-9*disc*np.maximum(fwd, k)
    if not np.isfinite(price).all() or np.any(price < lower-tolerance) or np.any(price > upper+tolerance):
        raise ValueError("Fourier price violates bounds; refine controls or reject this domain")
    return price


def surface_prices(parameters, contracts, *, nodes, upper_limit):
    """Prices of the ordered contracts at spot 1 and zero carry; parameters holds kappa, theta, vol_of_vol,
    rho and initial_variance."""
    p = parameters
    k = np.log(np.array([c["strike"] for c in contracts], dtype=float))
    maturity = np.array([c["maturity"] for c in contracts], dtype=float)
    kind = np.array([c["option_type"] for c in contracts])
    prices = np.empty(len(contracts))
    for value in np.unique(kind):
        chosen = kind == value
        prices[chosen] = european_price(1.0, np.exp(k[chosen]), 1.0, maturity[chosen], float(p["kappa"]),
                                        float(p["theta"]), float(p["vol_of_vol"]), float(p["rho"]),
                                        float(p["initial_variance"]), kind=str(value), nodes=nodes,
                                        upper_limit=upper_limit)
    return prices


def refined_prices(task, parameters):
    """The task's quadrature ladder: the first rung whose refinement moves no price by more than the budget."""
    contracts = task["context"]["contracts"]
    numerics = task["numerics"]
    limit = numerics["max_refinement_over_quote_sigma"] * tasks.sigma(task)
    ladder = numerics["quadrature_ladder"]
    failure = None
    for (nodes, cutoff), (next_nodes, next_cutoff) in zip(ladder, ladder[1:]):
        try:
            base = surface_prices(parameters, contracts, nodes=nodes, upper_limit=cutoff)
            refined = surface_prices(parameters, contracts, nodes=next_nodes, upper_limit=next_cutoff)
        except ValueError as error:
            failure = error
            continue
        if np.all(np.abs(refined - base) <= limit):
            return refined
        failure = ValueError("Fourier refinement budget failed")
    raise ValueError(f"{failure} on every quadrature rung")


def smile(task, row):
    """The 63 prices of one parameter row (v0, theta, vol_of_vol, rho), kappa from the task.

    One quadrature rung of 1,024 nodes; the task's full ladder where that rung fails.
    """
    parameters = tasks.model_parameters(task, np.asarray(row, dtype=float))
    try:
        return surface_prices(parameters, task["context"]["contracts"], nodes=1024, upper_limit=600.0)
    except ValueError:
        return refined_prices(task, parameters)

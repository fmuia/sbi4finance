"""Surface pricers for the calibration tasks, in forward units (spot 1, zero carry).

rough_surface prices under rough Bergomi by Monte Carlo. It reuses the hybrid
driver (rough_bergomi.simulate_rough) and the conditional Black–Scholes put
estimator (bergomi.conditional_put_payoffs), and adds three things:

* antithetic paths: every normal draw z is paired with -z, and pairs are
  averaged before any statistic is computed, so pairs are the iid units;
* two control variates per maturity with known mean zero: the effective
  forward exp(rho*int sqrt(V) dW - rho^2/2 int V dt) - 1, a discrete martingale,
  and int V dt minus its mean, which is exact because the hybrid scheme keeps
  E[v_i] = xi0(t_i); one regression per block removes their linear part;
* maturity blocks, each simulated on its own uniform grid with its own stream,
  so short maturities get fine steps; blocks are independent.

Calls follow from puts by parity, which is exact in forward units.

heston_surface prices with the Fourier integral of hkde.py with jumps off,
vectorized across contracts, at base and refined quadrature settings.
"""

import numpy as np

from ..bergomi import conditional_put_payoffs, curve_values
from ..rough_bergomi import simulate_rough

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.shared.pricing import (
    _PAIR_CHUNK,
    step_curve,
    _block_contracts,
    rough_block,
    rough_surface,
    heston_surface,
)


def heston_variance_values(times, short_variance, long_variance, decay):
    """Positive Heston-shaped initial mean curve in variance units, not Heston dynamics."""
    t = np.asarray(times, dtype=float)
    coefficients = np.asarray([short_variance, long_variance, decay], dtype=float)
    if (coefficients.shape != (3,) or not np.isfinite(coefficients).all() or np.any(coefficients <= 0)
            or not np.isfinite(t).all() or np.any(t < 0)):
        raise ValueError("positive finite curve coefficients and nonnegative finite times required")
    # Convex weights avoid cancellation when the short level is far below the long level.
    weight = -np.expm1(-decay * t)
    return short_variance * (1.0 - weight) + long_variance * weight


def heston_integrated_variance(times, short_variance, long_variance, decay):
    """Analytic W(T)=integral_0^T xi(s)ds; expm1 resolves small decay*T."""
    t = np.asarray(times, dtype=float)
    heston_variance_values(t, short_variance, long_variance, decay)  # shared validation
    average_short_weight = np.ones_like(t)
    np.divide(-np.expm1(-decay * t), decay * t, out=average_short_weight, where=t != 0)
    return t * (short_variance * average_short_weight + long_variance * (1.0 - average_short_weight))


def heston_variance_curve(times, short_variance, long_variance, decay):
    """Sample the analytic curve at all simulator grid nodes, including time zero.

    The simulator consumes a piecewise-linear table. When every block grid is
    included, its node/left-cell evaluations equal the analytic curve. Its
    expected integrated-variance control remains the same discrete left sum,
    not the continuous W(T). Changing the grid changes that numerical model.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("increasing curve grid from zero with at least two nodes required")
    return {"times": t.tolist(),
            "values": heston_variance_values(t, short_variance, long_variance, decay).tolist()}

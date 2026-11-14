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
    heston_variance_values,
    heston_integrated_variance,
    heston_variance_curve,
    _PAIR_CHUNK,
    step_curve,
    _block_contracts,
    rough_block,
    rough_surface,
    heston_surface,
)

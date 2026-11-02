"""Riemann--Liouville Volterra rough Bergomi, not fractional Brownian motion.

Y_t=sqrt(2H) integral_0^t (t-s)^(H-1/2)dW_s, Var(Y_t)=t^(2H).
The hybrid kappa=1 scheme treats the singular last cell exactly and older cells
with their average kernel. Its own covariance normalizes lognormal variance.
A small-grid independent Gaussian-covariance reference includes the same asset W.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz

from sbi4finance.bergomi import (_asset_paths, _model_parameters, _normals,
                               _positive_integer, _q_inputs, _scalar, curve_values)

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.rough_options.simulator import (
    hybrid_weights,
    _reference_factors,
    rough_covariance,
    rough_driver,
    simulate_rough,
)


def hybrid_covariance(hurst, dt_years, n_steps):
    """Exact covariance of the *discrete approximation*, for numerical checks.

    Returns C_YY and C_Y,dW on positive grid times; this is not the continuum
    covariance used by rough_covariance. The independent cell residual is reused
    only at its own endpoint in this one-cell hybrid approximation.
    """
    weights, bridge = hybrid_weights(hurst, dt_years, n_steps)
    matrix = toeplitz(weights, np.r_[weights[0], np.zeros(n_steps - 1)])
    return matrix @ matrix.T + bridge ** 2 * np.eye(n_steps), matrix * np.sqrt(dt_years)

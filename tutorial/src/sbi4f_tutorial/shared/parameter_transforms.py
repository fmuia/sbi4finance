"""Uniform-box logit coordinates shared by price emulators and lab samplers.

These transforms preserve the established numerical convention and Jacobian.
"""

import numpy as np
from scipy.special import expit, logit


def bounds_of(task):
    return np.asarray(task["prior_bounds"], dtype=np.float64)



def to_unbounded(parameters, bounds):
    unit = (np.asarray(parameters, dtype=float) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return logit(unit)



def to_bounded(z, bounds):
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * expit(np.asarray(z, dtype=float))



def log_jacobian(z, bounds):
    """log |d theta / d z| for theta = lower + width * expit(z), summed over coordinates."""
    z = np.asarray(z, dtype=float)
    return np.sum(np.log(bounds[:, 1] - bounds[:, 0]) - np.logaddexp(0.0, z) - np.logaddexp(0.0, -z), axis=-1)

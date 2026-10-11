"""One reversible coordinate change for the signed Heston inference target.

Imported by workflow construction, not by the NumPy-only data preparation.
The simulator, bank labels and returned posterior draws stay in variance units.
"""

import numpy as np

from bayesflow.adapters.transforms import ElementwiseTransform
from bayesflow.utils.serialization import serializable


@serializable("sbi4finance")
class SignedStateAsinh(ElementwiseTransform):
    """Map a signed variance state a to asinh(a / scale), without clipping.

    The inverse is scale * sinh(z). The positive scale has the same variance
    units as a; it controls coordinates, not model support or the simulator law.
    Jacobian reductions follow BayesFlow's per-batch elementwise-transform contract.
    """

    def __init__(self, scale=0.01):
        if isinstance(scale, (bool, np.bool_)) or np.ndim(scale) != 0:
            raise ValueError("scale must be a finite positive scalar")
        try:
            self.scale = float(scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("scale must be a finite positive scalar") from error
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be a finite positive scalar")

    def get_config(self):
        return {"scale": self.scale}

    def forward(self, data, **kwargs):
        with np.errstate(over="raise", invalid="raise"):
            return np.arcsinh(np.asarray(data) / self.scale)

    def inverse(self, data, **kwargs):
        # Extreme neural draws must fail visibly, never be clipped or redrawn.
        with np.errstate(over="raise", invalid="raise"):
            return self.scale * np.sinh(np.asarray(data))

    def log_det_jac(self, data, inverse=False, **kwargs):
        values = np.asarray(data)
        if inverse:
            # log(scale * cosh(z)), evaluated without overflowing cosh.
            elementwise = np.log(self.scale) + np.logaddexp(values, -values) - np.log(2.0)
        else:
            # d asinh(a/scale) / da = 1 / sqrt(scale**2 + a**2).
            elementwise = -np.log(np.hypot(self.scale, values))
        return np.sum(elementwise, axis=tuple(range(1, elementwise.ndim)))

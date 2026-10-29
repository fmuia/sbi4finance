"""Optional PyTensor likelihood Op, imported only by the PyMC reference route.

A standard Op avoids wrap_py's module-global registration requirement when
PyMC cloudpickles a black-box likelihood for spawn/forkserver workers.
"""

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.op import Op


class DeterministicLikelihoodOp(Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood

    def perform(self, node, inputs, outputs):
        value = float(self.log_likelihood(inputs[0]))
        if np.isnan(value) or value == np.inf:
            raise FloatingPointError("nonfinite reference log likelihood")
        outputs[0][0] = np.asarray(value, dtype=np.float64)

    def infer_shape(self, fgraph, node, shapes):
        return [()]

"""Versioned numerical representation for revised-grid Heston experiments.

The financial law, priors, normalized Gaussian likelihood, characteristic
function, pricing bounds and measurement-scale precision gates are inherited.
Only the Fourier integral's algebraic representation is changed.
"""
from sbi4f_tutorial.heston_options import control_variate

from .heston_spread import HestonSpreadPilot


class HestonGridPilot(HestonSpreadPilot):
    """Checked Heston spread task using an analytic Black control variate."""

    def _rung_prices(self, parameters, indices, nodes, cutoff):
        _, pars, v0 = self._parameters(parameters)
        records = self.records
        return control_variate.european_price(
            records["forward"][indices], records["strike"][indices], records["discount"][indices],
            records["maturity"][indices], pars, v0,
            put=records["option_type"][indices] == "put", nodes=nodes, upper_limit=cutoff)

    def task_definition(self):
        task = super().task_definition()
        task["pricing_controls"]["fourier_representation"] = "black_control_variate_v1"
        return task

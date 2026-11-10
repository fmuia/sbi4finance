"""Raw European quote likelihood for the bounded Heston observation pilot.

Prices, mids and half-spreads retain the quote's currency units.  A scalar
discrepancy ``tau`` is an IV-decimal scale multiplied by each observed quote's
frozen Black vega.  It describes independent extra scatter, not a correction
for systematic smile bias.  Numerical refinement is always measured against
measurement uncertainty alone; increasing tau cannot excuse a loose pricer.

This experiment has its own identity.  It does not reinterpret the existing
63-contract tasks, their banks, or their observation-noise conventions.
"""

from collections.abc import Mapping

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import ndtr
import warnings

from sbi4f_tutorial.heston_options import model


PARAMETER_NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho")
DEFAULT_BOUNDS = ((.01, .16), (.1, 10.), (.01, .16), (.1, 1.5), (-.95, -.1))
DEFAULT_LADDER = ((256, 150.), (512, 300.), (1024, 600.),
                  (2048, 1200.), (4096, 2400.), (8192, 4800.))
REQUIRED_COLUMNS = ("forward", "strike", "discount", "maturity",
                    "option_type", "mid", "half_spread")


def black_price(forward, strike, discount, maturity, volatility, option_type):
    """Black price in quote units, using the direct formula for each type."""
    f, k, d, t, vol = np.broadcast_arrays(forward, strike, discount, maturity, volatility)
    sd = vol * np.sqrt(t)
    if np.any(sd <= 0):
        raise ValueError("positive volatility and maturity are required")
    d1 = np.log(f / k) / sd + sd / 2
    d2 = d1 - sd
    return np.where(np.asarray(option_type) == "call",
                    d * (f * ndtr(d1) - k * ndtr(d2)),
                    d * (k * ndtr(-d2) - f * ndtr(-d1)))


def observed_black_vega(records):
    """Observed IV and price vega per unit decimal IV; never a fixed 20% vega.

    The midpoint must be strictly inside European price bounds to obtain a
    finite, positive IV and vega.  An explicitly supplied ``market_iv`` avoids
    inversion.  A supplied ``market_vega`` is accepted only as a positive
    observed quantity; its data provenance belongs in the experiment record.
    """
    f, k, d, t = (records[key] for key in ("forward", "strike", "discount", "maturity"))
    if "market_iv" in records:
        iv = np.asarray(records["market_iv"], dtype=float)
        if iv.shape != f.shape or not np.isfinite(iv).all() or np.any(iv <= 0):
            raise ValueError("market_iv requires one finite positive decimal volatility per quote")
    else:
        iv = np.empty_like(f)
        for i in range(len(iv)):
            args = [f[i], k[i], d[i], t[i]]
            kind, mid = records["option_type"][i], records["mid"][i]
            upper = 2.
            while black_price(*args, upper, kind) < mid and upper < 128.:
                upper *= 2
            iv[i] = brentq(lambda vol: float(black_price(*args, vol, kind) - mid),
                           1e-10, upper, xtol=1e-13)
    sd = iv * np.sqrt(t)
    d1 = np.log(f / k) / sd + sd / 2
    vega = d * f * np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi) * np.sqrt(t)
    if "market_vega" in records:
        vega = np.asarray(records["market_vega"], dtype=float).copy()
    if vega.shape != f.shape or not np.isfinite(vega).all() or np.any(vega <= 0):
        raise ValueError("market vega must be positive, finite and in price per decimal-IV units")
    return iv, vega


class HestonPilot:
    """Checked deterministic pricing and a normalized independent Gaussian law.

    ``records`` is a mapping of one-dimensional arrays in actual quote units.
    Boolean masks or integer selections preserve contract order.  Bounds are
    independent physical Uniform priors; optional tau is Uniform(0, tau_max).
    The Gaussian-SD-equals-half-spread convention is explicit and can be varied
    with ``spread_multiplier``.  No synthetic noise floor is applied.
    """

    def __init__(self, records, *, spread_multiplier=1., infer_discrepancy=False,
                 tau_max=.02, bounds=None, quadrature_ladder=DEFAULT_LADDER,
                 numerical_fraction=.02):
        if not isinstance(records, Mapping) or not set(REQUIRED_COLUMNS) <= records.keys():
            raise ValueError(f"records require {REQUIRED_COLUMNS}")
        self.records = {key: np.asarray(records[key], dtype=str if key == "option_type" else float).copy()
                        for key in (*REQUIRED_COLUMNS, "market_iv", "market_vega") if key in records}
        shape = self.records["mid"].shape
        if len(shape) != 1 or not shape[0] or any(a.shape != shape for a in self.records.values()):
            raise ValueError("one nonempty one-dimensional array per quote column is required")
        for key, a in self.records.items():
            if key != "option_type" and (not np.isfinite(a).all() or np.any(a <= 0)):
                raise ValueError(f"{key} must contain positive finite values")
        if np.any(~np.isin(self.records["option_type"], ["call", "put"])):
            raise ValueError("only European call and put quote types are supported")
        f, k, d = (self.records[key] for key in ("forward", "strike", "discount"))
        put = self.records["option_type"] == "put"
        lower = d * np.maximum(np.where(put, k - f, f - k), 0)
        upper = d * np.where(put, k, f)
        if np.any(self.records["mid"] <= lower) or np.any(self.records["mid"] >= upper):
            raise ValueError("midpoints must lie strictly inside European price bounds")
        if not np.isfinite(spread_multiplier) or spread_multiplier <= 0:
            raise ValueError("positive finite spread_multiplier is required")
        if not np.isfinite(numerical_fraction) or not 0 < numerical_fraction < 1:
            raise ValueError("numerical_fraction must lie strictly between zero and one")
        if type(infer_discrepancy) is not bool or not np.isfinite(tau_max) or tau_max <= 0:
            raise ValueError("boolean infer_discrepancy and positive finite tau_max are required")
        ladder = tuple(tuple(rung) for rung in quadrature_ladder)
        if len(ladder) < 2 or any(len(rung) != 2 for rung in ladder):
            raise ValueError("at least two node/cutoff pairs are required")
        for nodes, cutoff in ladder:
            if (isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer))
                    or nodes < 16 or not np.isfinite(cutoff) or cutoff <= 0):
                raise ValueError("each rung requires integer nodes >=16 and positive finite cutoff")
        if any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(ladder, ladder[1:])):
            raise ValueError("both nodes and cutoff must increase strictly on each rung")
        physical_bounds = np.asarray(DEFAULT_BOUNDS if bounds is None else bounds, dtype=float)
        if (physical_bounds.shape != (5, 2) or not np.isfinite(physical_bounds).all()
                or np.any(physical_bounds[:, 1] <= physical_bounds[:, 0])):
            raise ValueError("bounds require five finite increasing physical parameter intervals")
        if (np.any(physical_bounds[:4, 0] <= 0)
                or physical_bounds[4, 0] < -1 or physical_bounds[4, 1] > 1):
            raise ValueError("positive variance/kappa/vol-of-vol support and rho within [-1,1] required")
        self.infer_discrepancy = infer_discrepancy
        self.parameter_names = list(PARAMETER_NAMES) + (["tau"] if infer_discrepancy else [])
        self.names = self.parameter_names
        self.bounds = np.vstack([physical_bounds, [0., tau_max]]) if infer_discrepancy else physical_bounds
        self.spread_multiplier = float(spread_multiplier)
        self.measurement_sigma = self.spread_multiplier * self.records["half_spread"]
        self.numerical_fraction = float(numerical_fraction)
        self.quadrature_ladder = ladder
        self.market_iv, self.market_vega = observed_black_vega(self.records)
        for array in (*self.records.values(), self.bounds, self.measurement_sigma,
                      self.market_iv, self.market_vega):
            array.setflags(write=False)

    def _indices(self, mask):
        if mask is None:
            return np.arange(len(self.records["mid"]))
        values = np.asarray(mask)
        if values.dtype.kind == "b":
            if values.shape != self.records["mid"].shape:
                raise ValueError("boolean mask must match all quote rows")
            indices = np.flatnonzero(values)
        elif values.dtype.kind in "iu" and values.ndim == 1:
            indices = values
        else:
            raise ValueError("one-dimensional boolean mask or integer row indices required")
        if (not len(indices) or np.any(indices < 0) or np.any(indices >= len(self.records["mid"]))
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("a nonempty unique in-range contract selection is required")
        return indices

    def _parameters(self, parameters):
        values = np.asarray(parameters, dtype=float)
        if values.shape != (len(self.parameter_names),) or not np.isfinite(values).all():
            raise ValueError("one finite parameter vector in pilot parameter order is required")
        v0, kappa, theta, xi, rho = values[:5]
        result = dict(kappa=kappa, theta=theta, vol_of_vol=xi, rho=rho,
                      jump_intensity=0., jump_up_probability=.5, jump_up_rate=10., jump_down_rate=10.)
        model.characteristic_function(0., 0., result, v0)
        if self.infer_discrepancy and values[-1] < 0:
            raise ValueError("tau must be nonnegative")
        return values, result, v0

    def _rung_prices(self, parameters, indices, nodes, cutoff):
        _, pars, v0 = self._parameters(parameters)
        r = self.records
        return model._european_price(r["forward"][indices], r["strike"][indices],
                                     r["discount"][indices], r["maturity"][indices], pars, v0,
                                     put=r["option_type"][indices] == "put", nodes=nodes,
                                     upper_limit=cutoff)

    def _checked_prices(self, parameters, indices, ladder):
        limit = self.numerical_fraction * self.measurement_sigma[indices]
        previous, previous_rung = None, None
        attempts = []
        for nodes, cutoff in ladder:
            try:
                price = self._rung_prices(parameters, indices, nodes, cutoff)
            except (ValueError, FloatingPointError) as error:
                attempts.append({"nodes": int(nodes), "cutoff": float(cutoff), "error": str(error)})
                previous, previous_rung = None, None
                continue
            if previous is not None:
                difference = price - previous
                ratio = float(np.max(np.abs(difference) / self.measurement_sigma[indices]))
                attempts.append({"nodes": int(nodes), "cutoff": float(cutoff),
                                 "max_difference_over_measurement_sigma": ratio})
                if np.all(np.abs(difference) <= limit):
                    return price, {"nodes": int(nodes), "cutoff": float(cutoff),
                                   "base_nodes": int(previous_rung[0]), "base_cutoff": float(previous_rung[1]),
                                   "difference": difference, "max_difference_over_measurement_sigma": ratio,
                                   "numerical_fraction": self.numerical_fraction, "attempts": attempts}
            previous, previous_rung = price, (nodes, cutoff)
        raise FloatingPointError(f"Heston quadrature failed measurement-error refinement gate: {attempts}")

    def prices(self, parameters, mask=None, *, return_diagnostics=False):
        """Checked prices on the selected actual contracts; failures propagate."""
        indices = self._indices(mask)
        result = self._checked_prices(parameters, indices, self.quadrature_ladder)
        return result if return_diagnostics else result[0]

    def observation_sigma(self, parameters, mask=None):
        """Measurement plus independent discrepancy SD, in quote units."""
        values, _, _ = self._parameters(parameters)
        indices = self._indices(mask)
        tau = values[-1] if self.infer_discrepancy else 0.
        return np.hypot(self.measurement_sigma[indices], self.market_vega[indices] * tau)

    def log_likelihood(self, parameters, mask=None):
        """Normalized Gaussian density; physical Uniform priors belong to sampler."""
        values = np.asarray(parameters, dtype=float)
        if values.shape != (len(self.parameter_names),) or not np.isfinite(values).all():
            raise ValueError("one finite parameter vector in pilot parameter order is required")
        if np.any(values < self.bounds[:, 0]) or np.any(values > self.bounds[:, 1]):
            return -np.inf
        indices = self._indices(mask)
        price = self.prices(values, indices)
        sigma = self.observation_sigma(values, indices)
        residual = (self.records["mid"][indices] - price) / sigma
        return float(-.5 * np.sum(residual**2 + np.log(2 * np.pi * sigma**2)))

    def refinement_report(self, parameters, mask=None):
        """Double every node count and cutoff; report prices and log-law changes.

        This is pointwise sensitivity, not a proof of posterior stability.
        """
        indices = self._indices(mask)
        base, base_diagnostics = self._checked_prices(parameters, indices, self.quadrature_ladder)
        finer = tuple((2 * n, 2 * c) for n, c in self.quadrature_ladder)
        refined, refined_diagnostics = self._checked_prices(parameters, indices, finer)
        sigma = self.observation_sigma(parameters, indices)
        y = self.records["mid"][indices]
        shift = refined - base
        return {"base_prices": base, "refined_prices": refined,
                "price_shift": shift, "base_diagnostics": base_diagnostics,
                "refined_diagnostics": refined_diagnostics,
                "max_abs_price_shift_over_measurement_sigma": float(np.max(np.abs(shift) / self.measurement_sigma[indices])),
                "log_likelihood_shift": float(-.5 * np.sum(((y - refined) / sigma)**2 - ((y - base) / sigma)**2)),
                "scope": "pointwise quadrature sensitivity; does not establish posterior stability"}

    def independent_prices(self, parameters, mask=None):
        """Adaptive infinite-interval quadrature as a second integration method.

        Shares the characteristic function and Lewis formula with production;
        this checks quadrature, not independent financial-model correctness.
        Integration warnings and error-budget failures propagate.
        """
        indices = self._indices(mask)
        _, pars, v0 = self._parameters(parameters)
        r = self.records
        prices, errors = [], []
        for i in indices:
            f, k, d, t = (r[key][i] for key in ("forward", "strike", "discount", "maturity"))
            factor = d * np.sqrt(f * k) / np.pi
            tolerance = self.numerical_fraction * self.measurement_sigma[i] / 5

            def integrand(u):
                phi = model.characteristic_function(u - .5j, t, pars, v0)
                return float(np.real(np.exp(1j * u * np.log(f / k)) * phi) / (u*u + .25))

            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                integral, error = quad(integrand, 0., np.inf, epsabs=tolerance / factor,
                                       epsrel=1e-11, limit=1000)
            if factor * error > tolerance:
                raise FloatingPointError("adaptive quadrature exceeded its measurement-error budget")
            call = d * f - factor * integral
            price = call - d * (f - k) if r["option_type"][i] == "put" else call
            put = r["option_type"][i] == "put"
            lower = d * max(k - f if put else f - k, 0)
            upper = d * (k if put else f)
            if not np.isfinite(price) or not lower - tolerance <= price <= upper + tolerance:
                raise FloatingPointError("adaptive quadrature violated European price bounds")
            prices.append(price)
            errors.append(factor * error)
        return np.asarray(prices), np.asarray(errors)

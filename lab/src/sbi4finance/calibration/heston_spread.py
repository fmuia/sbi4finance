"""One-component Gaussian observations for a fixed European Heston design.

The physical price is computed by the existing checked Heston pilot.  This
likelihood uses only ``c * half_spread`` as its price standard deviation; no
implied volatility, observed vega or additional discrepancy is required.
``c`` is either supplied as fixed or has a bounded Uniform prior in its own
physical coordinate.  Half-spread-as-SD remains a modelling convention.
"""

from collections.abc import Mapping
import hashlib
import json

import numpy as np
from scipy.integrate import quad

from .heston_pilot import DEFAULT_LADDER, PARAMETER_NAMES, REQUIRED_COLUMNS, HestonPilot


DEFAULT_SPREAD_BOUNDS = ((.01, .16), (.1, 10.), (.01, .16), (.1, 3.), (-.95, -.1))
DEFAULT_SPREAD_LADDER = (*DEFAULT_LADDER, (16384, 9600.))


def _c_bounds(bounds):
    if np.iscomplexobj(bounds):
        raise ValueError("c_bounds must be real")
    values = np.asarray(bounds, dtype=float)
    if (values.shape != (2,) or not np.isfinite(values).all()
            or not 0 < values[0] < values[1]):
        raise ValueError("c_bounds require two positive finite increasing endpoints")
    return tuple(float(value) for value in values)


def _residual_statistics(residual_over_spread):
    if np.iscomplexobj(residual_over_spread):
        raise ValueError("standardized residuals must be real")
    residual = np.asarray(residual_over_spread, dtype=float)
    if residual.ndim != 1 or not residual.size or not np.isfinite(residual).all():
        raise ValueError("one nonempty finite standardized residual vector is required")
    with np.errstate(over="ignore"):
        square = float(residual @ residual)
    if not np.isfinite(square):
        raise ValueError("standardized residual sum of squares must be finite")
    return len(residual), square


def conditional_c_mode(residual_over_spread, c_bounds=(.5, 2.)):
    """Mode in physical c under Uniform(c_lo,c_hi), at fixed Heston prices."""
    lo, hi = _c_bounds(c_bounds)
    count, square = _residual_statistics(residual_over_spread)
    return float(np.clip(np.sqrt(square / count), lo, hi))


def conditional_c_log_density(c, residual_over_spread, c_bounds=(.5, 2.)):
    """Normalized one-dimensional oracle for p(c | physical parameters, y).

    Its unnormalized density is c**(-n) exp(-R/(2*c**2)) on the stated
    interval, for R=sum(((y-price)/half_spread)**2).  Quadrature is rescaled
    at the physical-coordinate mode; numerical normalization failures raise.
    The Uniform-prior constant cancels in the conditional normalization.
    """
    lo, hi = _c_bounds(c_bounds)
    count, square = _residual_statistics(residual_over_spread)
    if np.iscomplexobj(c):
        raise ValueError("c evaluation points must be real")
    values = np.asarray(c, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("c evaluation points must be finite")
    mode = float(np.clip(np.sqrt(square / count), lo, hi))

    def kernel(value):
        return -count * np.log(value) - square / (2 * value**2)

    peak = kernel(mode)
    integral, error = quad(lambda value: np.exp(kernel(value) - peak), lo, hi,
                           points=[mode] if lo < mode < hi else None,
                           epsabs=1e-12, epsrel=1e-11, limit=200)
    if not np.isfinite(integral) or integral <= 0 or error > 1e-9 * integral:
        raise FloatingPointError("conditional c density normalization failed")
    result = np.full(values.shape, -np.inf)
    inside = (values >= lo) & (values <= hi)
    result[inside] = kernel(values[inside]) - peak - np.log(integral)
    return float(result) if result.ndim == 0 else result


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class HestonSpreadPilot(HestonPilot):
    """Checked Heston pricing with fixed c or one inferred spread multiplier.

    The checked pricing methods and integrator are inherited, without calling
    the discrepancy pilot's IV/vega-dependent initializer.  Every pricing
    geometry array and half-spread is frozen.  ``mid`` is an observation and
    may contain any finite values, including negative Gaussian realizations.

    ``bounds`` contains five physical Uniform intervals.  If ``infer_c`` is
    true, ``c`` is appended as a sixth physical Uniform target on ``c_bounds``;
    otherwise ``c`` is fixed.  The numerical gate uses the *lower endpoint*
    of c_bounds for both modes, never the candidate inferred c.  The default
    gate is therefore .02 * (.5 * half_spread) = .01 * half_spread.
    """

    def __init__(self, records, *, c=1., infer_c=False, c_bounds=(.5, 2.),
                 bounds=None, quadrature_ladder=DEFAULT_SPREAD_LADDER,
                 numerical_fraction=.02):
        if not isinstance(records, Mapping) or not set(REQUIRED_COLUMNS) <= records.keys():
            raise ValueError(f"records require {REQUIRED_COLUMNS}")
        if any(np.iscomplexobj(records[key]) for key in REQUIRED_COLUMNS if key != "option_type"):
            raise ValueError("quote columns must be real")
        self.records = {
            key: np.asarray(records[key], dtype=str if key == "option_type" else float).copy()
            for key in REQUIRED_COLUMNS
        }
        shape = self.records["mid"].shape
        if len(shape) != 1 or not shape[0] or any(a.shape != shape for a in self.records.values()):
            raise ValueError("one nonempty one-dimensional array per quote column is required")
        for key, values in self.records.items():
            if key == "option_type":
                if np.any(~np.isin(values, ["call", "put"])):
                    raise ValueError("only European call and put quote types are supported")
            elif not np.isfinite(values).all() or (key != "mid" and np.any(values <= 0)):
                raise ValueError(f"{key} requires finite values and positive pricing geometry/spreads")
        if type(infer_c) is not bool:
            raise ValueError("infer_c must be boolean")
        self.c_bounds = _c_bounds(c_bounds)
        if (np.ndim(c) != 0 or isinstance(c, (bool, np.bool_)) or not np.isfinite(c)
                or not self.c_bounds[0] <= c <= self.c_bounds[1]):
            raise ValueError("fixed c must be finite and inside c_bounds")
        if not np.isfinite(numerical_fraction) or not 0 < numerical_fraction < 1:
            raise ValueError("numerical_fraction must lie strictly between zero and one")
        ladder = tuple(tuple(rung) for rung in quadrature_ladder)
        if len(ladder) < 2 or any(len(rung) != 2 for rung in ladder):
            raise ValueError("at least two node/cutoff pairs are required")
        for nodes, cutoff in ladder:
            if (isinstance(nodes, (bool, np.bool_)) or not isinstance(nodes, (int, np.integer))
                    or nodes < 16 or not np.isfinite(cutoff) or cutoff <= 0):
                raise ValueError("each rung requires integer nodes >=16 and positive finite cutoff")
        if any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(ladder, ladder[1:])):
            raise ValueError("both nodes and cutoff must increase strictly on each rung")
        ladder = tuple((int(nodes), float(cutoff)) for nodes, cutoff in ladder)
        if np.iscomplexobj(bounds):
            raise ValueError("physical bounds must be real")
        physical = np.asarray(DEFAULT_SPREAD_BOUNDS if bounds is None else bounds, dtype=float).copy()
        if (physical.shape != (5, 2) or not np.isfinite(physical).all()
                or np.any(physical[:, 1] <= physical[:, 0])):
            raise ValueError("bounds require five finite increasing physical parameter intervals")
        if np.any(physical[:4, 0] <= 0) or physical[4, 0] < -1 or physical[4, 1] > 1:
            raise ValueError("positive variance/kappa/vol-of-vol support and rho within [-1,1] required")
        self.infer_c, self.c = infer_c, float(c)
        self.infer_discrepancy = False
        self.parameter_names = list(PARAMETER_NAMES) + (["c"] if infer_c else [])
        self.names = self.parameter_names
        self.bounds = np.vstack([physical, self.c_bounds]) if infer_c else physical
        self.pricing_c_min = self.spread_multiplier = self.c_bounds[0]
        with np.errstate(over="ignore", under="ignore"):
            supported_scales = np.asarray(self.c_bounds)[:, None] * self.records["half_spread"]
        if not np.isfinite(supported_scales).all() or np.any(supported_scales <= 0):
            raise ValueError("supported c times half-spread must remain positive and finite")
        self.measurement_sigma = supported_scales[0].copy()
        self.numerical_fraction = float(numerical_fraction)
        self.quadrature_ladder = ladder
        for values in (*self.records.values(), self.bounds, self.measurement_sigma):
            values.setflags(write=False)

    def _parameters(self, parameters):
        if np.iscomplexobj(parameters):
            raise ValueError("physical parameters and c must be real")
        values, pars, v0 = super()._parameters(parameters)
        if self.infer_c and values[-1] <= 0:
            raise ValueError("c must be positive")
        return values, pars, v0

    def _observations(self, observations):
        if np.iscomplexobj(observations):
            raise ValueError("observations must be real")
        values = self.records["mid"] if observations is None else np.asarray(observations, dtype=float)
        if values.shape != self.records["mid"].shape or not np.isfinite(values).all():
            raise ValueError("one finite observation per full-design quote is required")
        return values

    def observation_sigma(self, parameters, mask=None):
        """Exactly c * half-spread; no vega or additional discrepancy term."""
        values, _, _ = self._parameters(parameters)
        indices = self._indices(mask)
        c = values[-1] if self.infer_c else self.c
        return c * self.records["half_spread"][indices]

    def log_likelihood(self, parameters, mask=None, *, observations=None):
        """Normalized Gaussian; the sampler supplies physical Uniform priors."""
        if np.iscomplexobj(parameters):
            raise ValueError("physical parameters and c must be real")
        values = np.asarray(parameters, dtype=float)
        if values.shape != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError("one finite parameter vector in pilot parameter order is required")
        observed = self._observations(observations)
        indices = self._indices(mask)
        if np.any(values < self.bounds[:, 0]) or np.any(values > self.bounds[:, 1]):
            return -np.inf
        price = self.prices(values, indices)
        sigma = self.observation_sigma(values, indices)
        residual = (observed[indices] - price) / sigma
        return float(-.5 * np.sum(residual**2) - np.sum(np.log(sigma))
                     - .5 * len(indices) * np.log(2 * np.pi))

    def simulate(self, parameters, *, seed, size=1, mask=None, clean_prices=None):
        """Fresh Gaussian errors from an explicit seed; return [size, quotes].

        Optional finite full-design clean prices permit immutable bank reuse.
        Their provenance and numerical accuracy must be checked by the caller.
        Quote errors are drawn on the full design before selection, preserving
        surviving errors in paired full/restricted panels with the same seed.
        """
        if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0
                or isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1):
            raise ValueError("nonnegative integer seed and positive integer size required")
        values, _, _ = self._parameters(parameters)
        if np.any(values < self.bounds[:, 0]) or np.any(values > self.bounds[:, 1]):
            raise ValueError("simulation parameters must lie inside the declared prior bounds")
        indices = self._indices(mask)
        if clean_prices is None:
            clean = self.prices(values, indices)
        else:
            clean = np.asarray(clean_prices, dtype=float)
            if clean.shape != self.records["mid"].shape or not np.isfinite(clean).all():
                raise ValueError("bank reuse requires one finite clean price per full-design quote")
            clean = clean[indices]
        sigma = self.observation_sigma(values, indices)
        error = np.random.default_rng(seed).standard_normal((size, len(self.records["mid"])))
        return clean + error[:, indices] * sigma

    def conditional_c_mode(self, physical_parameters, mask=None, *, observations=None):
        """Physical-coordinate conditional mode under this object's c prior."""
        residual = self._conditional_c_residual(physical_parameters, mask, observations)
        return conditional_c_mode(residual, self.c_bounds)

    def conditional_c_log_density(self, c, physical_parameters, mask=None, *, observations=None):
        """Normalized conditional-c oracle, including its bounded Uniform prior."""
        residual = self._conditional_c_residual(physical_parameters, mask, observations)
        return conditional_c_log_density(c, residual, self.c_bounds)

    def _conditional_c_residual(self, physical_parameters, mask, observations):
        if not self.infer_c:
            raise ValueError("conditional c density requires infer_c=True")
        if np.iscomplexobj(physical_parameters):
            raise ValueError("physical parameters must be real")
        physical = np.asarray(physical_parameters, dtype=float)
        if physical.shape != (5,) or not np.isfinite(physical).all():
            raise ValueError("five finite physical parameters are required")
        if np.any(physical < self.bounds[:5, 0]) or np.any(physical > self.bounds[:5, 1]):
            raise ValueError("physical parameters must lie inside the declared prior bounds")
        indices = self._indices(mask)
        y = self._observations(observations)[indices]
        price = self.prices(np.r_[physical, self.c_bounds[0]], indices)
        return (y - price) / self.records["half_spread"][indices]

    def refinement_report(self, parameters, mask=None, *, observations=None):
        """Inherited pointwise pricing check, evaluated on supplied observations."""
        report = super().refinement_report(parameters, mask)
        indices = self._indices(mask)
        y = self._observations(observations)[indices]
        sigma = self.observation_sigma(parameters, indices)
        report["log_likelihood_shift"] = float(-.5 * np.sum(
            ((y - report["refined_prices"]) / sigma)**2 - ((y - report["base_prices"]) / sigma)**2))
        return report

    def task_definition(self):
        """Observation-independent target identity; bind actual data separately.

        External runners additionally bind source files, design provenance and
        observation bytes.  Supplied IV/vega fields are deliberately irrelevant.
        """
        geometry = {key: values.tolist() for key, values in self.records.items() if key != "mid"}
        return {"format": "heston_spread_task_v1", "model": "heston", "measure": "Q",
                "parameter_names": list(self.names), "prior_bounds": self.bounds.tolist(),
                "prior": "independent Uniform in physical coordinates",
                "infer_c": self.infer_c, "fixed_c": None if self.infer_c else self.c,
                "c_bounds": list(self.c_bounds), "quote_count": len(self.records["mid"]),
                "geometry_and_spreads_sha256": _fingerprint(geometry),
                "observation_model": "independent Gaussian with price SD c * half_spread",
                "pricing_controls": {"c_min": self.pricing_c_min,
                    "numerical_fraction": self.numerical_fraction,
                    "quadrature_ladder": [list(rung) for rung in self.quadrature_ladder]}}

    def validate_task(self, task):
        """Reject task reuse after a target, geometry, spread or gate change."""
        if not isinstance(task, Mapping) or dict(task) != self.task_definition():
            raise ValueError("spread task identity differs in target, geometry or numerical controls")
        return task

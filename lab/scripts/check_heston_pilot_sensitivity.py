"""Bounded physical-MAP sensitivity to conditional carry and valuation clock.

Run from lab/: python -m scripts.check_heston_pilot_sensitivity --help.
This optimizes the normalized likelihood under the declared physical Uniform
prior. It does not reuse transformed-coordinate MAP objectives or integrate
carry uncertainty into the posterior. All price/spread observations, selected
contracts and frozen market vegas stay fixed across the sensitivity cases.
"""

import argparse
import copy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from sbi4finance.calibration.heston_pilot import HestonPilot
from scripts.run_heston_observation_pilot import expected_variance, write_json


MINUTES_PER_YEAR = 365. * 24. * 60.


def standardized_carry_axes(covariance, radius=2.):
    """Unit-invariant two-sigma ellipse axes in physical (a, D) units.

    Eigendecompose the correlation matrix, then return to physical units.
    Raw covariance eigenvectors would change when prices change currency
    units. Each returned direction has Mahalanobis norm ``radius``.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T):
        raise ValueError("finite symmetric two-dimensional covariance required")
    if np.min(np.linalg.eigvalsh(cov)) <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("positive definite covariance and positive radius required")
    sd = np.sqrt(np.diag(cov))
    eigenvalues, eigenvectors = np.linalg.eigh(cov / np.outer(sd, sd))
    axes = []
    for index in np.argsort(eigenvalues)[::-1]:
        direction = eigenvectors[:, index]
        # Resolve arbitrary eigenvector sign reproducibly.
        if direction[0] < -1e-14 or (abs(direction[0]) <= 1e-14 and direction[1] < 0):
            direction = -direction
        axes.append(radius * sd * direction * np.sqrt(eigenvalues[index]))
    return np.asarray(axes)


def scenario_definitions():
    cases = [{"name": "baseline", "kind": "baseline"}]
    for pattern in ("level", "maturity_tilt"):
        for axis in range(2):
            for sign in (-1, 1):
                cases.append({"name": f"carry_{pattern}_axis{axis}_{'plus' if sign > 0 else 'minus'}2se",
                              "kind": "carry", "pattern": pattern, "axis": axis, "sign": sign,
                              "mahalanobis_radius_per_expiry": 2.})
    for minutes in (-15., 15.):
        cases.append({"name": f"clock_valuation_{'plus' if minutes > 0 else 'minus'}15min",
                      "kind": "clock", "valuation_shift_minutes": minutes})
    return cases


def perturb_design(design, scenario):
    """Apply a predeclared scenario while preserving observed quotes/scales."""
    varied = copy.deepcopy(design)
    spot = float(design["snapshot_metadata"]["underlying_price"])
    details = {}
    shift = (-scenario.get("valuation_shift_minutes", 0.) / MINUTES_PER_YEAR
             if scenario["kind"] == "clock" else 0.)
    expiries = design["expiries"]
    for index, expiry in enumerate(expiries):
        baseline = design["carry"][expiry]
        coef = np.array([baseline["parity_intercept"], baseline["discount"]])
        cov = np.asarray(baseline["intercept_discount_covariance"])
        delta = np.zeros(2)
        if scenario["kind"] == "carry":
            pattern = 1. if scenario["pattern"] == "level" or index >= len(expiries) / 2 else -1.
            delta = pattern * scenario["sign"] * standardized_carry_axes(
                cov, scenario["mahalanobis_radius_per_expiry"])[scenario["axis"]]
        a, discount = coef + delta
        maturity = baseline["maturity"] + shift
        forward = spot + a / discount
        if discount <= 0 or forward <= 0 or maturity <= 0:
            raise ValueError("carry/clock perturbation invalidated positive F, D or maturity")
        varied["carry"][expiry].update(forward=float(forward), discount=float(discount),
                                       parity_intercept=float(a), maturity=float(maturity),
                                       rate=float(-np.log(discount) / maturity))
        details[expiry] = {"delta_intercept": float(delta[0]), "delta_discount": float(delta[1]),
                           "delta_forward": float(forward - baseline["forward"]),
                           "forward": float(forward), "discount": float(discount),
                           "rate": float(-np.log(discount) / maturity),
                           "mahalanobis_radius": float(np.sqrt(max(0., delta @ np.linalg.solve(cov, delta))))}
    for index, quote in enumerate(varied["selected_quotes"]):
        carry = varied["carry"][quote["expiration"]]
        for key in ("forward", "discount", "maturity"):
            quote[key] = carry[key]
            varied["records"][key][index] = carry[key]
    for key, interval in varied["products"].items():
        # A forward variance interval starts at a fixed expiry, while the
        # control starts at valuation time itself.
        varied["products"][key] = [0. if interval[0] == 0. else interval[0] + shift, interval[1] + shift]
    return varied, details


def initialization_starts(path, bounds):
    """Convert initializer locations, ignoring its transformed objective."""
    record = json.loads(Path(path).read_text())
    if not np.allclose(record["bounds"], bounds):
        raise ValueError("initializer uses a different prior support")
    lower, width = bounds[:, 0], np.diff(bounds, axis=1)[:, 0]
    return [lower + width * expit(np.asarray(fit["logit"])) for fit in record["fits"]]


def choose_starts(bounds, candidates, count, seed):
    lower, width = bounds[:, 0], np.diff(bounds, axis=1)[:, 0]
    selected = []
    for candidate in candidates:
        u = np.clip((np.asarray(candidate) - lower) / width, 0., 1.)
        if not selected or all(np.linalg.norm(u - existing) > .02 for existing in selected):
            selected.append(u)
        if len(selected) == count:
            return [lower + width * u for u in selected]
    rng = np.random.default_rng(seed)
    while len(selected) < count:
        selected.append(np.clip(selected[0] + rng.normal(0., .1, len(lower)), .001, .999)
                        if selected else rng.uniform(.1, .9, len(lower)))
    return [lower + width * u for u in selected]


def optimize_physical(pilot, mask, starts, maxfun):
    """Multistart bounded optimization in a linear physical unit box."""
    lower, width = pilot.bounds[:, 0], np.diff(pilot.bounds, axis=1)[:, 0]
    attempts = []
    best = None
    for index, start in enumerate(starts):
        seen = {"value": np.inf, "u": None, "evaluations": 0}

        def objective(u):
            point = np.clip(lower + np.asarray(u) * width, pilot.bounds[:, 0], pilot.bounds[:, 1])
            value = -pilot.log_likelihood(point, mask)
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite likelihood inside bounded optimization")
            seen["evaluations"] += 1
            if value < seen["value"]:
                seen.update(value=value, u=np.asarray(u).copy())
            return value

        result = minimize(objective, np.clip((start - lower) / width, 0., 1.),
                          method="L-BFGS-B", jac="3-point", bounds=[(0., 1.)] * len(lower),
                          options={"maxfun": maxfun, "maxiter": 1000, "ftol": 1e-11,
                                   "gtol": 1e-5, "finite_diff_rel_step": 1e-5})
        point = np.clip(lower + seen["u"] * width, pilot.bounds[:, 0], pilot.bounds[:, 1])
        attempt = {"start": index, "parameters": point.tolist(),
                   "log_likelihood": float(-seen["value"]),
                   "optimizer_success": bool(result.success), "message": str(result.message),
                   "evaluations": seen["evaluations"]}
        attempts.append(attempt)
        if best is None or attempt["log_likelihood"] > best["log_likelihood"]:
            best = attempt
    units = (np.asarray(best["parameters"]) - lower) / width
    return {"best": best, "attempts": attempts,
            "all_starts_converged": all(a["optimizer_success"] for a in attempts),
            "max_bestfit_loglike_gap_across_starts": float(max(a["log_likelihood"] for a in attempts)
                                                         - min(a["log_likelihood"] for a in attempts)),
            "parameters_near_support_boundary": [name for name, u in zip(pilot.names, units)
                                                   if min(u, 1 - u) < 1e-4]}


def fitted_products(point, design):
    return {key: float(expected_variance(np.asarray(point), interval)) for key, interval in design["products"].items()}


def run(args):
    design = json.loads(args.design.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    config = {"format": "heston_carry_clock_physical_map_sensitivity_v1",
              "scope": "Conditional finite-scenario optimization sensitivity; no carry-integrated posterior and no posterior sensitivity claim.",
              "target": "Normalized independent Gaussian likelihood with physical independent Uniform priors; no coordinate Jacobian.",
              "optimizer": "scipy L-BFGS-B on linear unit-box physical coordinates, numerical 3-point gradients, finite multistart search",
              "carry_scenarios": "Eight signed coherent curves from both eigenaxes of standardized (parity-intercept, discount) correlation matrices. Each expiry has conditional-WLS Mahalanobis radius2; level and first-half/last-half sign-tilt patterns. These are stress directions, not joint 95% confidence regions or independent draws.",
              "clock_scenarios": "Valuation timestamp +/-15min around the frozen baseline; selected physical quotes and carry F,D fixed.",
              "frozen_quantities": ["actual symbols, strikes, option types", "observed mids and halfspreads", "market vegas", "prior support"],
              "limitations": ["WLS carry covariance inherits the assumed spread-to-Gaussian-SD convention.",
                              "Carry anchors and smile quotes can share unmodeled quote errors.",
                              "Finite coherent carry curves do not cover every combination across expiries.",
                              "A physical-density MAP can sit on a support boundary and differs from a transformed-coordinate density MAP.",
                              "Repeated optimizer convergence does not establish global mode coverage.",
                              "The scalar discrepancy cannot remove systematic smile bias."],
              "design_sha256": hashlib.sha256(args.design.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in ("numpy", "scipy")}},
              "settings": {"starts": args.starts, "maxfun_per_start": args.maxfun, "seed": args.seed},
              "results": {}, "scientific_acceptance": False}
    begun = time.perf_counter()
    for arm in ("full", "short"):
        pilot = HestonPilot(design["records"], infer_discrepancy=True)
        mask = np.asarray(design["short_mask"]) if arm == "short" else None
        initializer = getattr(args, "initializer_" + arm)
        source_starts = initialization_starts(initializer, pilot.bounds)
        config.setdefault("initializer_sha256", {})[arm] = hashlib.sha256(initializer.read_bytes()).hexdigest()
        baseline_best = baseline_prices = baseline_products = baseline_loglike = None
        config["results"][arm] = []
        for scenario in scenario_definitions():
            varied, carry_details = perturb_design(design, scenario)
            active = HestonPilot(varied["records"], infer_discrepancy=True)
            candidates = ([] if baseline_best is None else [baseline_best]) + source_starts
            starts = choose_starts(active.bounds, candidates, args.starts, args.seed)
            started = time.perf_counter()
            fit = optimize_physical(active, mask, starts, args.maxfun)
            point = np.asarray(fit["best"]["parameters"])
            products = fitted_products(point, varied)
            if baseline_best is None:
                baseline_best = point
                baseline_loglike = fit["best"]["log_likelihood"]
                baseline_products = products
                baseline_prices = pilot.prices(point)
            price_shift_sigma = (active.prices(baseline_best) - baseline_prices) / pilot.measurement_sigma
            use = np.ones(len(price_shift_sigma), dtype=bool) if mask is None else mask
            result = {"scenario": scenario, "carry": carry_details, "fit": fit,
                      "fit_loglike_delta_from_baseline": fit["best"]["log_likelihood"] - baseline_loglike,
                      "baseline_map_loglike_delta_under_perturbed_design": active.log_likelihood(baseline_best, mask) - baseline_loglike,
                      "physical_map_parameter_delta": (point - baseline_best).tolist(),
                      "fitted_expected_variance_strikes": products,
                      "fitted_expected_variance_strike_delta": {key: products[key] - baseline_products[key] for key in products},
                      "baseline_map_price_shift_over_measurement_sigma": {
                          "max_abs_all": float(np.max(abs(price_shift_sigma))),
                          "max_abs_fitted_rows": float(np.max(abs(price_shift_sigma[use]))),
                          "rms_fitted_rows": float(np.sqrt(np.mean(price_shift_sigma[use]**2)))},
                      "wall_seconds": time.perf_counter() - started}
            config["results"][arm].append(result)
            config["wall_seconds"] = time.perf_counter() - begun
            write_json(args.out / "sensitivity.json", config)
            print(json.dumps({"arm": arm, "case": scenario["name"], "loglike": fit["best"]["log_likelihood"],
                              "delta_loglike": result["fit_loglike_delta_from_baseline"],
                              "all_starts_converged": fit["all_starts_converged"], "seconds": result["wall_seconds"]}), flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--initializer-full", type=Path, required=True)
    parser.add_argument("--initializer-short", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxfun", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=261110)
    args = parser.parse_args()
    if args.starts < 2 or args.maxfun < 100:
        parser.error("at least two starts and 100 function evaluations per start required")
    run(args)


if __name__ == "__main__":
    main()

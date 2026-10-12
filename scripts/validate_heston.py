"""Small Heston/SVJ refinement pilot against continuous-time references.

Run with ``--out /tmp/heston-validation.json`` to save a NEW JSON report.
The default uses 8,192 paths, 64 observations over one year, and 1/4/16
internal steps per observation. No market data, posterior or pricer is used.
"""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from sbi4finance import heston


FREQUENCIES = np.array([(0.0, 20.0), (3.0, 0.0), (3.0, 20.0), (-3.0, 20.0)])
SUBSTEPS = (1, 4, 16)
OBSERVATIONS = 64
HORIZON = 1.0
SE_MULTIPLIER = 5.0
# Absolute budgets in raw units, fixed before running this bounded pilot.
# The test separately reports sampling error; these are not confidence bounds
# on the unknown discretization bias, or accuracy claims for other settings.
BUDGETS = {
    "variance_mean": 2e-4,
    "variance_second_moment": 1e-4,
    "variance_variance": 1e-4,
    "log_return_mean": 5e-4,
    "characteristic_component": 3e-3,
}
ODE_STEPS = (2048, 4096)
ODE_TOLERANCE = 1e-9


def joint_transform(parameters, initial_variance, horizon, frequencies, steps):
    """Independent affine-generator reference, integrated by complex RK4.

    Computes E[exp(i*u*log(S_T/S_0) + i*w*V_T)] for the continuous model.
    D(0)=i*w, C(0)=0; phi=exp(C+D*v0). This ODE uses neither simulated
    variance paths nor the full-truncation transition. Refinement is checked.
    """
    u, w = np.asarray(frequencies, dtype=np.float64).T
    kappa, theta = parameters["kappa"], parameters["theta"]
    xi, rho = parameters["vol_of_vol"], parameters["rho"]
    intensity = parameters.get("jump_intensity", 0.0)
    jump_mean, jump_std = parameters.get("jump_mean", 0.0), parameters.get("jump_std", 0.0)
    compensator = np.expm1(jump_mean + 0.5 * jump_std**2)
    constant = (1j * u * parameters["drift"]
                + intensity * (np.exp(1j * u * jump_mean - 0.5 * u**2 * jump_std**2)
                               - 1.0 - 1j * u * compensator))

    def derivative(state):
        d = state[1]
        return np.stack((constant + kappa * theta * d,
                         0.5 * xi**2 * d**2 + (1j * u * rho * xi - kappa) * d
                         - 0.5 * (u**2 + 1j * u)))

    state = np.zeros((2, len(u)), dtype=np.complex128)
    state[1] = 1j * w
    step = horizon / steps
    for _ in range(steps):
        a = derivative(state)
        b = derivative(state + 0.5 * step * a)
        c = derivative(state + 0.5 * step * b)
        d = derivative(state + step * c)
        state += step * (a + 2 * b + 2 * c + d) / 6
    result = np.exp(state[0] + state[1] * initial_variance)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite affine-transform reference")
    return result


def moment_references(parameters, initial_variance, horizon):
    """CIR moments and compensated log-return mean, for kappa > 0."""
    kappa, theta, xi = parameters["kappa"], parameters["theta"], parameters["vol_of_vol"]
    one_minus_decay = -np.expm1(-kappa * horizon)
    decay = np.exp(-kappa * horizon)
    mean = theta + (initial_variance - theta) * decay
    variance = (initial_variance * xi**2 * decay * one_minus_decay / kappa
                + theta * xi**2 * one_minus_decay**2 / (2 * kappa))
    integrated_mean = theta * horizon + (initial_variance - theta) * one_minus_decay / kappa
    jump_mean, jump_std = parameters.get("jump_mean", 0.0), parameters.get("jump_std", 0.0)
    jump_correction = parameters.get("jump_intensity", 0.0) * horizon * (
        jump_mean - np.expm1(jump_mean + 0.5 * jump_std**2))
    return {"variance_mean": float(mean), "variance_second_moment": float(variance + mean**2),
            "variance_variance": float(variance),
            "log_return_mean": float(parameters["drift"] * horizon - 0.5 * integrated_mean
                                     + jump_correction)}


def compare(samples, reference, budget):
    """Compare a sample mean with an independent reference and fixed budget."""
    estimate = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    discrepancy = estimate - reference
    return {"estimate": estimate, "reference": float(reference),
            "signed_error": float(discrepancy), "mc_standard_error": se,
            "absolute_discretization_budget": budget,
            "sampling_allowance": SE_MULTIPLIER * se,
            "passed": bool(abs(discrepancy) <= budget + SE_MULTIPLIER * se)}


def summarize(log_returns, terminal_variance, references, transform):
    """Observable arrays also allow lower-noise paired refinement checks."""
    observables = {"variance_mean": terminal_variance,
                   "variance_second_moment": terminal_variance**2,
                   "log_return_mean": log_returns}
    # Centering at the fixed analytic mean gives an unbiased sample-mean
    # estimator of the continuous variance under the reference law and makes
    # the standard-error and coarse/fine comparisons explicit.
    observables["variance_variance"] = (terminal_variance - references["variance_mean"])**2
    comparisons = {key: compare(values, references[key], BUDGETS[key])
                   for key, values in observables.items()}
    characteristic = []
    for index, (u, w) in enumerate(FREQUENCIES):
        values = np.exp(1j * (u * log_returns + w * terminal_variance))
        row = {"u": float(u), "w": float(w)}
        for component, sample, exact in (("real", values.real, transform[index].real),
                                         ("imag", values.imag, transform[index].imag)):
            row[component] = compare(sample, exact, BUDGETS["characteristic_component"])
            observables[f"cf_{index}_{component}"] = sample
        characteristic.append(row)
    passed = all(item["passed"] for item in comparisons.values()) and all(
        row[component]["passed"] for row in characteristic for component in ("real", "imag"))
    return {"moments": comparisons, "joint_characteristic_function": characteristic,
            "reference_checks_passed": bool(passed),
            "sample_terminal_variance_variance": float(np.var(terminal_variance, ddof=1))}, observables


def coarsen(innovations, substeps):
    """Sum the fine independent Brownian channels; reuse observation jumps."""
    fine = innovations["brownian"]
    if fine.shape[2] % substeps:
        raise ValueError("substeps must divide the finest grid")
    ratio = fine.shape[2] // substeps
    brownian = fine if ratio == 1 else fine.reshape(
        fine.shape[0], fine.shape[1], substeps, ratio, 2).sum(axis=3) / np.sqrt(ratio)
    return {"brownian": brownian, "jump_counts": innovations["jump_counts"],
            "jump_normals": innovations["jump_normals"]}


def run_case(name, parameters, initial_variance, *, paths, seed, stream_id, backend,
             observations=OBSERVATIONS, horizon=HORIZON, substep_levels=SUBSTEPS):
    dt = np.full(observations, horizon / observations)
    references = moment_references(parameters, initial_variance, horizon)
    ode = [joint_transform(parameters, initial_variance, horizon, FREQUENCIES, steps)
           for steps in ODE_STEPS]
    ode_error = float(np.max(np.abs(ode[1] - ode[0])))
    started = perf_counter()
    finest = heston.draw_innovations(dt, max(substep_levels), seed=seed,
                                    example_ids=np.arange(paths), stream_id=stream_id,
                                    jump_intensity=parameters.get("jump_intensity", 0.0))
    generated = perf_counter() - started
    levels, paired_checks = [], []
    previous_observables = None
    for substeps in substep_levels:
        innovations = coarsen(finest, substeps)
        started = perf_counter()
        output = heston.simulate(parameters, dt, initial_variance, innovations, backend=backend)
        elapsed = perf_counter() - started
        raw_variance = output["variance_state"][:, :, 0]
        log_returns = output["log_returns"][:, :, 0].sum(axis=1)
        terminal = np.maximum(raw_variance[:, -1], 0.0)
        summary, observables = summarize(log_returns, terminal, references, ode[1])
        summary.update({"substeps_per_observation": substeps,
                        "internal_dt_years": horizon / (observations * substeps),
                        "simulation_seconds_including_compile_if_needed": elapsed,
                        "negative_auxiliary_fraction_at_observation_times": float(np.mean(raw_variance[:, 1:] < 0)),
                        "negative_terminal_auxiliary_fraction": float(np.mean(raw_variance[:, -1] < 0)),
                        "minimum_auxiliary_variance": float(np.min(raw_variance)),
                        "terminal_auxiliary_mean": float(np.mean(raw_variance[:, -1]))})
        levels.append(summary)
        if previous_observables is not None:
            comparisons = {key: compare(values - previous_observables[key], 0.0,
                                         BUDGETS["characteristic_component"] if key.startswith("cf_") else BUDGETS[key])
                           for key, values in observables.items()}
            paired_checks.append({"coarse_substeps": substep_levels[len(levels) - 2],
                                  "fine_substeps": substeps, "differences": comparisons,
                                  "passed": all(item["passed"] for item in comparisons.values())})
        previous_observables = observables
    passed = (ode_error <= ODE_TOLERANCE and levels[-1]["reference_checks_passed"]
              and paired_checks[-1]["passed"])
    return {"case": name, "parameters": parameters, "initial_variance": initial_variance,
            "feller_ratio_2_kappa_theta_over_xi_squared": 2 * parameters["kappa"] * parameters["theta"] / parameters["vol_of_vol"]**2,
            "paths": paths, "seed": seed, "stream_id": stream_id,
            "innovation_generation_seconds": generated,
            "reference_moments": references,
            "ode_reference": {"steps": list(ODE_STEPS), "max_refinement_difference": ode_error,
                              "tolerance": ODE_TOLERANCE, "passed": ode_error <= ODE_TOLERANCE},
            "levels": levels, "paired_refinement": paired_checks,
            "finest_grid_pilot_passed": bool(passed)}


def run(*, paths=8192, seed=6301, backend="numba", include_svj=False):
    if isinstance(paths, bool) or not isinstance(paths, int) or paths < 2:
        raise ValueError("paths must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    regular = {"drift": 0.05, "kappa": 2.0, "theta": 0.04, "vol_of_vol": 0.3, "rho": -0.6}
    boundary = {"drift": 0.05, "kappa": 1.0, "theta": 0.04, "vol_of_vol": 0.6, "rho": -0.7}
    cases = [("regular_heston", regular, 0.06), ("near_zero_feller_violation", boundary, 0.005)]
    if include_svj:
        cases.append(("modest_gaussian_svj", {**regular, "jump_intensity": 0.7,
                                              "jump_mean": -0.07, "jump_std": 0.1}, 0.06))
    results = [run_case(name, parameters, initial_variance, paths=paths, seed=seed,
                        stream_id=index, backend=backend)
               for index, (name, parameters, initial_variance) in enumerate(cases)]
    try:
        numba_version = version("numba")
    except PackageNotFoundError:
        numba_version = None
    return {
        "data_origin": "synthetic", "evidence_scope": "S numerical kernel pilot only",
        "backend": backend, "versions": {"numpy": np.__version__, "python": platform.python_version(),
                                          "numba": numba_version},
        "source_sha256": {"heston.py": hashlib.sha256(Path(heston.__file__).read_bytes()).hexdigest(),
                          "validate_heston.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "design": {"horizon_years": HORIZON, "observations": OBSERVATIONS,
                   "substeps_per_observation": list(SUBSTEPS), "frequencies_u_w": FREQUENCIES.tolist(),
                   "reference": "independent continuous-time affine-generator RK4 and analytic CIR/log-return moments",
                   "variance_comparison": "max(auxiliary terminal variance, 0); raw auxiliary state retained for continuation",
                   "coupling": "sum fine Brownian increments; identical observation-level Poisson counts and jump normals",
                   "absolute_budgets": BUDGETS, "mc_se_multiplier": SE_MULTIPLIER,
                   "acceptance": "finest-grid errors <= budget + 5 MC SE; last paired refinement changes satisfy same rule; ODE refinement passes",
                   "limits": "diagnostic pilot, not simultaneous confidence bounds or a proof of a discretization-bias bound; no posterior, pricing, market or arbitrary-domain acceptance",
                   "variance_variance_estimator": "mean squared deviation from analytic continuous variance mean; sample variance also reported",
                   "refinement_note": "paired changes measure sensitivity, not the remaining bias; monotone noisy estimates are not required"},
        "reference_sources": ["https://repub.eur.nl/pub/18571",
                              "https://www.federalreserve.gov/econres/feds/files/2023053pap.pdf"],
        "cases": results, "passed": all(case["finest_grid_pilot_passed"] for case in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--include-svj", action="store_true", help="also check a modest Gaussian-jump case")
    parser.add_argument("--out", type=Path, help="optional NEW JSON report file")
    args = parser.parse_args()
    if args.paths < 2 or args.seed < 0:
        parser.error("paths must be >= 2 and seed must be nonnegative")
    if args.out is not None and args.out.exists():
        parser.error("--out must be a new file; existing reports are not overwritten")
    report = run(paths=args.paths, seed=args.seed, backend=args.backend, include_svj=args.include_svj)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as file:
            file.write(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit("Heston refinement pilot failed; inspect the reported numerical and sampling errors.")


if __name__ == "__main__":
    main()

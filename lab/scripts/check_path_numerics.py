"""Bounded numerical diagnostics for synthetic path-inference pilots.

These comparisons record discretization differences; they do not impose an
unjustified acceptance threshold or validate a continuous-time posterior.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from sbi4finance.heston import draw_innovations, simulate
from sbi4finance.rough_bergomi import hybrid_covariance, rough_covariance


def coarsen_brownian(fine, substeps):
    """Group independent standard normals while preserving Brownian sums."""
    fine = np.asarray(fine, dtype=float)
    if (fine.ndim != 4 or fine.shape[-1] != 2 or substeps < 1
            or fine.shape[2] % substeps):
        raise ValueError("fine must be [case,time,fine_steps,2] divisible by substeps")
    groups = fine.shape[2] // substeps
    return fine.reshape(*fine.shape[:2], substeps, groups, 2).sum(axis=3) / np.sqrt(groups)


def heston_refinement(cases, *, seed, observations=1260):
    """Paired one/four/eight-substep diagnostics over a declared fixed panel."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0])))
    parameters = {"drift": np.full(cases, 0.05),
                  "kappa": np.exp(rng.uniform(np.log(0.25), np.log(5.0), cases)),
                  "theta": np.exp(rng.uniform(np.log(0.01), np.log(0.09), cases)),
                  "vol_of_vol": rng.uniform(0.05, 0.6, cases),
                  "rho": rng.uniform(-0.95, -0.05, cases)}
    dt = np.full(observations, 1.0 / 252)
    fine = draw_innovations(dt, 8, seed=seed, example_ids=np.arange(cases), stream_id=21)
    arrays, reports, outputs = {}, [], {}
    for steps in (1, 4, 8):
        inputs = {**fine, "brownian": coarsen_brownian(fine["brownian"], steps)}
        result = simulate(parameters, dt, 0.04, inputs)
        returns = result["log_returns"][:, :, 0]
        states = result["variance_state"][:, :, 0]
        outputs[steps] = returns
        arrays[f"returns_substeps_{steps}"] = returns
        arrays[f"terminal_signed_variance_substeps_{steps}"] = states[:, -1]
        reports.append({"substeps": steps,
                        "mean_realized_variance": float(np.mean(np.sum(returns ** 2, axis=1))),
                        "mean_terminal_log_return": float(np.mean(returns.sum(axis=1))),
                        "mean_terminal_effective_variance": float(np.maximum(states[:, -1], 0).mean()),
                        "nonpositive_observation_states": int(np.count_nonzero(states[:, :-1] <= 0)),
                        "negative_observation_state_fraction": float(np.mean(states[:, :-1] < 0)),
                        "exact_drift_atoms": int(np.count_nonzero(returns == parameters["drift"][:, None] * dt)),
                        "finite": bool(np.isfinite(returns).all() and np.isfinite(states).all())})
    paired = []
    for low, high in ((1, 4), (1, 8), (4, 8)):
        delta = outputs[low] - outputs[high]
        rms = np.sqrt(np.mean(delta ** 2, axis=1))
        scale = np.sqrt(np.mean(outputs[high] ** 2, axis=1))
        terminal_delta = delta.sum(axis=1)
        arrays[f"daily_rms_difference_{low}_vs_{high}"] = rms
        arrays[f"terminal_log_return_difference_{low}_vs_{high}"] = terminal_delta
        paired.append({"substeps": [low, high], "daily_rms_difference_mean": float(rms.mean()),
                       "relative_daily_rms_difference_mean": float(np.mean(rms / scale)),
                       "terminal_log_return_difference_mean": float(terminal_delta.mean()),
                       "terminal_log_return_difference_rms": float(np.sqrt(np.mean(terminal_delta ** 2))),
                       "terminal_log_return_absolute_difference_max": float(np.max(np.abs(terminal_delta)))})
    arrays.update({f"parameter_{k}": v for k, v in parameters.items()})
    return {"cases": cases, "observations": observations, "dt_years": 1.0 / 252,
            "panel": "fixed-seed independent prior-domain diagnostic panel; same bounds/types as physical_heston_paths_v1 but independent seeds",
            "parameter_ranges": {"kappa": [0.25, 5.0], "theta": [0.01, 0.09],
                                 "vol_of_vol": [0.05, 0.6], "rho": [-0.95, -0.05]},
            "prior_types": {"kappa": "log_uniform", "theta": "log_uniform",
                            "vol_of_vol": "uniform", "rho": "uniform"},
            "fixed": {"drift": 0.05, "initial_variance": 0.04},
            "feller_satisfied_cases": int(np.count_nonzero(2 * parameters["kappa"] * parameters["theta"] >= parameters["vol_of_vol"] ** 2)),
            "coupling": "same eight fine independent Brownian increments, grouped/sqrt(group_size); no jumps",
            "runs": reports, "paired_comparisons": paired,
            "interpretation": "pathwise discretization sensitivity on this bounded panel; changing substeps changes the discrete inference target"}, arrays


def rough_covariance_check(cases, *, seed, steps=128):
    """Compare actual hybrid covariance and its normalizer with continuum values.

    This deterministic Gaussian calculation applies to both known and inferred
    positive curve levels. A deterministic curve multiplies the variance; it
    cancels from the expected-variance/curve ratio. It does not change C_YY.
    """
    if not 1 <= steps <= 128:
        raise ValueError("continuum covariance reference is bounded to <=128 steps")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
    # Repeated H values limit adaptive-quadrature work while covering roughness.
    hurst_panel = np.array([0.03, 0.1, 0.2, 0.4])
    hursts = np.resize(hurst_panel, cases)
    etas = rng.uniform(0.2, 2.5, cases)
    short = np.exp(rng.uniform(np.log(0.01), np.log(0.09), cases))
    long = np.exp(rng.uniform(np.log(0.01), np.log(0.09), cases))
    decay = np.exp(rng.uniform(np.log(0.25), np.log(8.0), cases))
    dt = 1.0 / 1008
    times = dt * np.arange(1, steps + 1)
    cache, details = {}, []
    hybrid_diagonals, continuum_diagonals = [], []
    for i, h in enumerate(hursts):
        if h not in cache:
            hybrid, cross_hybrid = hybrid_covariance(float(h), dt, steps)
            continuum, cross_continuum = rough_covariance(float(h), dt, steps)
            cache[h] = hybrid, continuum, cross_hybrid, cross_continuum
        hybrid, continuum, cross_hybrid, cross_continuum = cache[h]
        difference = hybrid - continuum
        diagonal = hybrid.diagonal()
        continuous_diagonal = continuum.diagonal()
        hybrid_diagonals.append(diagonal)
        continuum_diagonals.append(continuous_diagonal)
        # For Y~N(0,C), E exp(eta*Y - eta²*C_ii/2)=1 exactly.
        own_normalizer_ratio = np.exp(0.5 * etas[i] ** 2 * (diagonal - diagonal))
        wrong_normalizer_ratio = np.exp(0.5 * etas[i] ** 2 * (diagonal - continuous_diagonal))
        joint_curve = long[i] + (short[i] - long[i]) * np.exp(-decay[i] * times)
        details.append({"case": i, "hurst": float(h), "eta": float(etas[i]),
                        "known_curve": {"constant_variance": 0.04},
                        "joint_curve": {"short": float(short[i]), "long": float(long[i]), "decay": float(decay[i])},
                        "covariance_relative_frobenius_error": float(np.linalg.norm(difference) / np.linalg.norm(continuum)),
                        "covariance_max_absolute_error": float(np.max(np.abs(difference))),
                        "terminal_driver_variance_relative_error": float(diagonal[-1] / continuous_diagonal[-1] - 1),
                        "driver_brownian_cross_covariance_max_absolute_error": float(np.max(np.abs(cross_hybrid - cross_continuum))),
                        "own_discrete_normalizer_expected_variance_ratio_max_error": float(np.max(np.abs(own_normalizer_ratio - 1))),
                        "continuum_normalizer_on_hybrid_expected_variance_ratio_min": float(wrong_normalizer_ratio.min()),
                        "known_expected_terminal_variance": float(0.04 * own_normalizer_ratio[-1]),
                        "joint_expected_terminal_variance": float(joint_curve[-1] * own_normalizer_ratio[-1]),
                        "joint_curve_terminal_variance": float(joint_curve[-1])})
    return {"cases": cases, "steps": steps, "dt_years": dt, "hurst_panel": hurst_panel.tolist(),
            "method": "analytic hybrid covariance versus independent continuum endpoint quadrature",
            "normalizer": "subtract eta²*Var_hybrid(Y_t)/2 for hybrid paths; continuum reference uses eta²*t^(2H)/2",
            "scope": "both known-flat and joint exponential deterministic curves; covariance reference is a small-grid diagnostic, not a continuum asset likelihood",
            "initialization": "Riemann-Liouville driver starts at zero at original time zero; no stationary restart or hidden prehistory is introduced",
            "runs": details,
            "interpretation": "nonzero covariance differences quantify the approximation; preserving E[v_t]=curve(t) does not make path laws identical"}, {
                "hurst": hursts, "eta": etas, "hybrid_driver_diagonal": np.asarray(hybrid_diagonals),
                "continuum_driver_diagonal": np.asarray(continuum_diagonals)}


def run(out, *, cases=32, seed=20261030071, model="all", observations=1260, rough_steps=128):
    if cases < 1 or observations < 1 or model not in ("all", "heston", "rough"):
        raise ValueError("positive cases/observations and model all/heston/rough required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"schema": "path-numerics-v1", "seed": seed, "status": "diagnostic_only",
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "files": []}
    for name in ("heston", "rough"):
        if model not in ("all", name):
            continue
        tick = time.perf_counter()
        summary, arrays = (heston_refinement(cases, seed=seed, observations=observations)
                           if name == "heston" else rough_covariance_check(cases, seed=seed, steps=rough_steps))
        summary["seconds"] = time.perf_counter() - tick
        report[name] = summary
        target = out / f"{name}-numerics.npz"
        np.savez_compressed(target, **arrays)
        report["files"].append({"file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    report["seconds"] = time.perf_counter() - started
    (out / "numerics-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261030071)
    parser.add_argument("--model", choices=["all", "heston", "rough"], default="all")
    parser.add_argument("--observations", type=int, default=1260)
    parser.add_argument("--rough-steps", type=int, default=128)
    args = parser.parse_args()
    report = run(args.out, cases=args.cases, seed=args.seed, model=args.model,
                 observations=args.observations, rough_steps=args.rough_steps)
    print(json.dumps({"out": str(args.out), "status": report["status"], "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()

"""Run repeated restricted Heston particle/quadrature precision controls.

Run from lab/: python scripts/path_reference.py --out /path/to/control
This does not validate the full multivariate neural task or rough Bergomi.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numba
import numpy as np

from sbi4finance.heston import draw_innovations, simulate
from sbi4finance.path_inference.reference import (
    compare_posteriors, posterior_summary, reference_posterior, restricted_task,
)


def run(out, *, cases=4, particles=(512, 2048), grids=(24, 48),
        repeats=2, seed=20261030051, observations=252):
    if cases < 1 or repeats < 2 or min(particles) < 2 or min(grids) < 2:
        raise ValueError("positive cases, at least two repeats, and grids/particles >=2 required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    task = restricted_task(observations)
    dt = np.full(observations, task["dt_years"])
    rng = np.random.Generator(np.random.PCG64(seed))
    truth = np.exp(rng.uniform(*np.log(task["theta_bounds"]), cases))
    innovations = draw_innovations(dt, 1, seed=seed, example_ids=np.arange(cases), stream_id=1)
    histories = simulate({**task["parameters"], "theta": truth}, dt,
                         task["initial_variance"], innovations)["log_returns"][:, :, 0]
    source_file = out / "observations.npz"
    np.savez_compressed(source_file, log_returns=histories, theta_truth=truth, dt_years=dt)
    # Compile the same kernel before measuring completed reference runs.
    warm_count = min(2, observations)
    warm_task = restricted_task(warm_count)
    reference_posterior(histories[0, :warm_count], dt[:warm_count], warm_task,
                        grid_size=2, particles=2, seed=seed)
    runs, comparisons = [], []
    started = time.perf_counter()
    for case in range(cases):
        results = {}
        for grid in sorted(set(grids)):
            for count in sorted(set(particles)):
                for repeat in range(repeats):
                    run_seed = int(np.random.SeedSequence([seed, case, grid, count, repeat])
                                   .generate_state(1, dtype=np.uint64)[0])
                    tick = time.perf_counter()
                    result = reference_posterior(histories[case], dt, task, grid_size=grid,
                                                 particles=count, seed=run_seed)
                    elapsed = time.perf_counter() - tick
                    filename = f"case-{case:03d}-grid-{grid}-particles-{count}-rep-{repeat}.npz"
                    np.savez_compressed(out / filename,
                                        **{k: v for k, v in result.items() if k != "diagnostics"})
                    runs.append({"case": case, "grid_size": grid, "particles": count,
                                 "repeat": repeat, "seed": run_seed, "seconds": elapsed,
                                 "file": filename, "sha256": hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                                 "theta_truth": float(truth[case]),
                                 "summary": posterior_summary(result), "diagnostics": result["diagnostics"]})
                    results[(grid, count, repeat)] = result
        largest_grid, largest_particles = max(grids), max(particles)
        baseline = results[(largest_grid, largest_particles, 0)]
        for (grid, count, repeat), result in results.items():
            if (grid, count, repeat) == (largest_grid, largest_particles, 0):
                continue
            comparisons.append({"case": case, "grid_size": grid, "particles": count,
                                "repeat": repeat,
                                "against": {"grid_size": largest_grid, "particles": largest_particles, "repeat": 0},
                                **compare_posteriors(result, baseline)})
    report = {
        "schema": "path-reference-control-v1", "task": task,
        "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "status": "precision_check_only_not_accepted_reference",
        "method": "fully adapted particle filter plus Gauss-Legendre quadrature in log(theta)",
        "reason_for_specialized_method": "discrete zero-variance return atoms and exact conditional transition; PyMC NUTS requires a different differentiable target",
        "scope": "restricted single-long-run-variance control for the one-step discrete P Heston model; no full-task or rough posterior validation",
        "observations": {"file": source_file.name, "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                         "seed": seed, "case_count": cases, "latent_truth_used_for_inference": False},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__},
        "seconds_excluding_compilation": time.perf_counter() - started,
        "runs": runs, "comparisons": comparisons,
        "interpretation": "Review independent-repeat, particle-count and grid-size differences before accepting precision. Even small differences do not validate multivariate SBI or a continuous-time model.",
    }
    (out / "reference-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--particles", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[24, 48])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261030051)
    parser.add_argument("--observations", type=int, default=252)
    args = parser.parse_args()
    report = run(args.out, cases=args.cases, particles=args.particles, grids=args.grid_sizes,
                 repeats=args.repeats, seed=args.seed, observations=args.observations)
    print(json.dumps({"out": str(args.out), "status": report["status"],
                      "runs": len(report["runs"]), "seconds": report["seconds_excluding_compilation"]}))


if __name__ == "__main__":
    main()

"""Isolated, order-balanced full-versus-lean conditional-pricing benchmark.

Both routes call the actual rough_block estimator with identical innovations,
antithetics, curve, contracts and regression. Only unused simulator outputs
differ. Each timed observation has its own process for a comparable RSS
high-water mark. This measures a long-maturity block, not mean bank throughput.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration import bank, pricing, tasks  # noqa: E402


def worker(args):
    task = tasks.joint_rough_task()
    contracts = [c for c in task["context"]["contracts"] if c["maturity"] > .25]
    curve = pricing.heston_variance_curve(np.linspace(0., 1., args.steps[0] + 1), .04, .09, 2.)
    parameters = {"hurst": .1, "eta": 1.5, "rho": -.7}
    simulator = pricing.simulate_rough

    def full(*values, **settings):
        settings["conditional_only"] = False
        return simulator(*values, **settings)

    context = patch.object(pricing, "simulate_rough", side_effect=full) if args.worker == "full" else nullcontext()
    with context:
        kwargs = dict(horizon=1., n_steps=args.steps[0], n_pairs=args.pairs, seed=args.seed)
        pricing.rough_block(parameters, curve, contracts, **kwargs)  # Untimed warmup, same for each route.
        cpu, wall = time.process_time(), time.perf_counter()
        prices, errors, covariance = pricing.rough_block(parameters, curve, contracts, **kwargs)
        cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    np.savez_compressed(args.out.with_suffix(".npz"), prices=prices, standard_errors=errors, covariance=covariance)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {"mode": args.worker, "steps": args.steps[0], "pairs": args.pairs, "paths": 2 * args.pairs,
              "seed": args.seed, "contracts": len(contracts), "parameters": parameters,
              "curve_coefficients": {"short_variance": .04, "long_variance": .09, "decay": 2.},
              "cpu_seconds": cpu, "wall_seconds": wall,
              "process_peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024),
              "runtime": bank.versions(), "pricing_source_sha256": bank.source_hashes(task),
              "thread_environment": {name: os.environ[name] for name in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def benchmark(args):
    if (args.repeats < 2 or args.repeats % 2 or args.pairs < 1
            or any(n < 2 or n % 2 for n in args.steps) or args.seed < 0):
        raise ValueError("positive pairs, even steps/repeats (at least two) and nonnegative seed required")
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for steps in args.steps:
        runs = {"full": [], "lean": []}
        for repeat in range(args.repeats):
            order = ("full", "lean") if repeat % 2 == 0 else ("lean", "full")
            files = {}
            for mode in order:
                prefix = args.out / f"steps-{steps}-repeat-{repeat}-{mode}"
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                                "--steps", str(steps), "--pairs", str(args.pairs), "--seed", str(args.seed + repeat),
                                "--out", str(prefix)], check=True, cwd=ROOT)
                files[mode] = prefix
                runs[mode].append(json.loads(prefix.with_suffix(".json").read_text()))
            with np.load(files["full"].with_suffix(".npz")) as full, np.load(files["lean"].with_suffix(".npz")) as lean:
                for key in full.files:
                    if not np.array_equal(full[key], lean[key]):
                        raise RuntimeError(f"non-bit-exact {key} at steps={steps}, seed={args.seed + repeat}; stop optimization")
        full_cpu = statistics.median(r["cpu_seconds"] for r in runs["full"])
        lean_cpu = statistics.median(r["cpu_seconds"] for r in runs["lean"])
        reports.append({"steps": steps, "pairs": args.pairs, "repeats": args.repeats,
                        "all_prices_errors_covariance_bit_exact": True,
                        "full_median_cpu_seconds": full_cpu, "lean_median_cpu_seconds": lean_cpu,
                        "measured_cpu_speedup": full_cpu / lean_cpu,
                        "cpu_ranges": {mode: [min(r["cpu_seconds"] for r in rows), max(r["cpu_seconds"] for r in rows)]
                                       for mode, rows in runs.items()},
                        "median_process_peak_rss_bytes": {mode: statistics.median(r["process_peak_rss_bytes"] for r in rows)
                                                           for mode, rows in runs.items()}, "runs": runs})
    report = {"format": "rough_conditional_output_benchmark_v1", "results": reports,
              "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "actual long-maturity rough_block estimator; separate order-balanced processes, one warmup and one timed call each; not mean adaptive bank throughput",
              "rss_scope": "process high-water mark including imports and warmup; separate process per mode/observation"}
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{key: r[key] for key in ("steps", "measured_cpu_speedup", "full_median_cpu_seconds",
          "lean_median_cpu_seconds", "median_process_peak_rss_bytes", "all_prices_errors_covariance_bit_exact")}
                      for r in reports], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[504, 2016])
    parser.add_argument("--pairs", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=271029)
    parser.add_argument("--worker", choices=("full", "lean"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()

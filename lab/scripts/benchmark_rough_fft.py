"""Benchmark research FFT adapters against the current opt-in Numba backend.

Run from lab/ with PYTHONPATH=.:src:../tutorial/src. This reads the committed
synthetic actual63 panel, never production journals, and writes a new research
record. Every path, grid, RNG channel and canonical estimator is preserved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
from unittest.mock import patch

import numba
import numpy as np
import scipy
from scipy.fft import next_fast_len

from sbi4f_tutorial.rough_options import _pricing_acceleration as acceleration
from sbi4f_tutorial.rough_options import _pricing_fft_experiments as experiments
from sbi4f_tutorial.rough_options import actual_layout, model, simulator
from sbi4f_tutorial.shared import pricing
from scripts import benchmark_rough_pricing_speed as benchmark
from sbi4finance.calibration import pricing as lab_pricing

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("numba", "cached_exact", "fast_length")


def run_case(task, case, index, *, paths, repeats):
    timings = {variant: [] for variant in VARIANTS}
    block_timings = {variant: [] for variant in VARIANTS}
    maximum = {variant: {} for variant in VARIANTS if variant != "numba"}
    calls = {variant: [] for variant in maximum}
    for repeat in range(repeats):
        found = {}
        order = VARIANTS[repeat % len(VARIANTS):]+VARIANTS[:repeat % len(VARIANTS)]
        for variant in order:
            driver = (simulator.rough_driver if variant == "numba"
                      else experiments.PreparedHybridDriver(variant))
            with patch.object(acceleration, "rough_driver", driver):
                result = benchmark.surface(task, case, index, paths, "numba")
            found[variant] = result
            timings[variant].append({name: result[name] for name in ("seconds", "cpu_seconds")})
            block_timings[variant].append([block["seconds"] for block in result["blocks"]])
            if variant != "numba":
                calls[variant].append({"setup_calls": driver.setup_calls, "driver_calls": driver.driver_calls})
        for variant in maximum:
            for block_index, (reference, candidate) in enumerate(zip(
                    found["numba"]["blocks"], found[variant]["blocks"], strict=True)):
                for name, value in benchmark.differences(reference["values"], candidate["values"]).items():
                    label = f"block{block_index}.{name}"
                    maximum[variant][label] = max(maximum[variant].get(label, 0), value)
    medians = {variant: statistics.median(t["seconds"] for t in values)
               for variant, values in timings.items()}
    return {"case_id": case["id"], "timings": timings, "block_timings": block_timings,
            "median_seconds": medians,
            "speedup_vs_current_numba": {variant: medians["numba"]/medians[variant] for variant in maximum},
            "maximum_absolute_differences": maximum,
            "bitwise_equal": {variant: all(value == 0 for value in values.values())
                              for variant, values in maximum.items()},
            "setup_call_counts": calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", type=Path, default=benchmark.WORK)
    parser.add_argument("--cases", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--paths", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.paths < 8 or args.paths % 2 or args.repeats < 1 or args.out.exists()
            or any(index < 0 or index >= 12 for index in args.cases)):
        parser.error("even path count>=8, positive repeats, joint case indices0..11 and new output path required")
    work = json.loads(args.work_order.read_text())
    task = work["tasks"]["joint_unknown"]
    # Compile outside the measured samples. All variants use the same kernel.
    benchmark.surface(task, work["cases"][0], 0, 8, "numba")
    cases = []
    for index in args.cases:
        result = run_case(task, work["cases"][index], index, paths=args.paths, repeats=args.repeats)
        cases.append(result)
        print(json.dumps({name: result[name] for name in (
            "case_id", "median_seconds", "speedup_vs_current_numba", "bitwise_equal")}), flush=True)
    files = [Path(__file__), Path(benchmark.__file__), Path(acceleration.__file__),
             Path(experiments.__file__), Path(actual_layout.__file__), Path(model.__file__),
             Path(simulator.__file__), Path(pricing.__file__), Path(lab_pricing.__file__)]
    record = {"format": "rough_actual63_fft_speed_experiment_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "numba": numba.__version__},
              "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
              "work_order_sha256": hashlib.sha256(args.work_order.read_bytes()).hexdigest(),
              "paths_per_block": args.paths, "repeats": args.repeats,
              "antithetic_pairs_per_chunk": pricing._PAIR_CHUNK,
              "steps": [block["n_steps"] for block in task["blocks"]],
              "power2_lengths": [1 << (2*block["n_steps"]-1).bit_length() for block in task["blocks"]],
              "fast_lengths": [next_fast_len(2*block["n_steps"]-1, real=True) for block in task["blocks"]],
              "cases": cases, "production_journals_opened": False, "remote_jobs_changed": False,
              "scope": "Local warmed serial comparison at fixed paths; same canonical estimator, RNG, grid and support checks. Fast lengths alter roundoff. No current-campaign or Oscar timing claim."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

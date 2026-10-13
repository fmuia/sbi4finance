"""Time the same Heston/SVJ task with NumPy and serial Numba, without market data.

The first Numba call includes compilation OR cache loading, depending on the
cache. For a cold measurement set NUMBA_CACHE_DIR to a new empty directory
before starting Python. Imports are excluded; full simulation includes input
generation, public validation, stepping and output allocation, but not disk IO.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numba
import numpy as np

from sbi4finance import heston


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/tasks/heston_svj_pilot.json"


def timed(call, repeats):
    seconds = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        seconds.append(perf_counter() - start)
    return {"seconds": seconds, "median_seconds": float(np.median(seconds))}


def run(config_path, *, batch_size=256, batches=4, repeats=3, seed=7412):
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    parameters = config["parameters"]
    dt = np.full(config["observations"], config["dt_years"])
    initial = config["initial_variance"]
    if initial < 0:
        raise ValueError("physical initial variance must be nonnegative")
    ids = np.arange(batch_size)

    def draw(example_ids):
        return heston.draw_innovations(dt, config["substeps"], seed=seed,
                                       example_ids=example_ids,
                                       jump_intensity=parameters.get("jump_intensity", 0.0))

    innovations = draw(ids)
    start = perf_counter()
    compiled = heston.simulate(parameters, dt, initial, innovations, backend="numba")
    first_call = perf_counter() - start
    reference = heston.simulate(parameters, dt, initial, innovations, backend="numpy")
    maximum_errors = {}
    for name in reference:
        np.testing.assert_allclose(compiled[name], reference[name], rtol=0, atol=1e-13)
        maximum_errors[name] = float(np.max(np.abs(compiled[name] - reference[name])))

    rng = timed(lambda: draw(ids), repeats)
    warm = {backend: timed(lambda: heston.simulate(parameters, dt, initial, innovations,
                                                 backend=backend), repeats)
            for backend in ("numpy", "numba")}

    def full(backend):
        for batch in range(batches):
            inputs = draw(ids + batch * batch_size)
            heston.simulate(parameters, dt, initial, inputs, backend=backend)

    complete = {backend: timed(lambda: full(backend), repeats) for backend in ("numpy", "numba")}
    # Demonstrate continuation with new independent future draws. The starting
    # state is the signed auxiliary value, not its clipped reported variance.
    future_dt = np.full(10, config["dt_years"])
    future_inputs = heston.draw_innovations(future_dt, config["substeps"], seed=seed,
                                           example_ids=ids, stream_id=1,
                                           jump_intensity=parameters.get("jump_intensity", 0.0))
    future = heston.simulate(parameters, future_dt, compiled["variance_state"][:, -1, 0],
                             future_inputs)
    np.testing.assert_array_equal(future["variance_state"][:, 0], compiled["variance_state"][:, -1])

    return {
        "task": config, "batch_size": batch_size, "batches": batches, "repeats": repeats,
        "seed": seed, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                       "numba": numba.__version__, "platform": platform.platform()},
        "source_sha256": {"heston.py": hashlib.sha256(Path(heston.__file__).read_bytes()).hexdigest(),
                          "benchmark_heston.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "config": hashlib.sha256(config_bytes).hexdigest()},
        "kernel": "serial float64, fastmath=False; both backends include public validation and output allocation",
        "first_numba_call_seconds": first_call,
        "numba_cache_directory": os.environ.get("NUMBA_CACHE_DIR", "default"),
        "first_call_scope": "compilation or cache loading plus one simulation; imports excluded",
        "innovation_array_bytes_per_batch": sum(value.nbytes for value in innovations.values()),
        "output_array_bytes_per_batch": sum(value.nbytes for value in compiled.values()),
        "matched_input_max_absolute_error": maximum_errors,
        "rng_one_batch": rng, "warm_simulation_one_batch": warm, "complete_simulation": complete,
        "warm_speedup_numpy_over_numba": warm["numpy"]["median_seconds"] / warm["numba"]["median_seconds"],
        "complete_speedup_numpy_over_numba": complete["numpy"]["median_seconds"] / complete["numba"]["median_seconds"],
        "complete_paths_per_second": {backend: batch_size * batches / value["median_seconds"]
                                      for backend, value in complete.items()},
        "continuation": {"history_shape": list(compiled["log_returns"].shape),
                         "future_shape": list(future["log_returns"].shape),
                         "matched_start_state": True,
                         "interpretation": "known-parameter synthetic continuation, not posterior forecasting"},
        "limits": "fixed-parameter CPU pilot; no bank IO, neural training, inference, pricing or broad-domain accuracy claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7412)
    parser.add_argument("--out", type=Path, help="optional NEW JSON report")
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.repeats) < 1 or not 0 <= args.seed < 2**64:
        parser.error("sizes/repeats must be positive and seed an unsigned 64-bit integer")
    if args.out is not None and args.out.exists():
        parser.error("--out must be a new file")
    report = run(args.config, batch_size=args.batch_size, batches=args.batches,
                 repeats=args.repeats, seed=args.seed)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as file:
            file.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

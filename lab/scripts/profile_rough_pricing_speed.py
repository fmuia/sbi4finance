"""Profile the warmed opt-in rough pricer without touching production journals.

Run from lab/ with PYTHONPATH=.:src:../tutorial/src. Explicit timers separate
native RNG, FFT/driver, compiled integration, guards, payoffs and other work.
The profile uses only the committed synthetic challenge panel.
"""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import statistics
import time
from unittest.mock import patch

import numba
import numpy as np

from scripts.benchmark_rough_pricing_speed import ROOT, WORK
from sbi4f_tutorial.rough_options import _pricing_acceleration as acceleration, actual_layout
from sbi4f_tutorial.shared import pricing
from sbi4finance.calibration.pricing import heston_variance_curve


def profile(task, case, *, pairs, repeats):
    curve = heston_variance_curve(actual_layout.simulation_nodes(task), *case["parameters"][3:])
    parameters = dict(zip(("hurst", "eta", "rho"), case["parameters"][:3], strict=True))

    def run(count=pairs, diagnostics=True, blocks=range(7)):
        for index in blocks:
            block = task["blocks"][index]
            contracts = [c for c in task["design"]["contracts"] if c["maturity"] == block["horizon"]]
            pricing.rough_block(parameters, curve, contracts, horizon=block["horizon"],
                n_steps=block["n_steps"], n_pairs=count, seed=np.random.SeedSequence([20261111, index]),
                diagnostics={"capture_raw_control": True} if diagnostics else None, path_backend="numba")

    run(count=8)
    profiles = []
    for _ in range(repeats):
        timers, counts = {}, {}

        def timed(name, function):
            def wrapper(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    timers[name] = timers.get(name, 0) + time.perf_counter()-started
                    counts[name] = counts.get(name, 0)+1
            return wrapper

        original_rng = np.random.default_rng

        class RandomTimer:
            def __init__(self, *args, **kwargs):
                self.generator = original_rng(*args, **kwargs)

            def standard_normal(self, *args, **kwargs):
                return timed("rng", self.generator.standard_normal)(*args, **kwargs)

        kernel = acceleration._compiled_kernel()
        with ExitStack() as stack:
            stack.enter_context(patch.object(pricing.np.random, "default_rng", RandomTimer))
            stack.enter_context(patch.object(acceleration, "rough_driver", timed("driver_including_fft", acceleration.rough_driver)))
            stack.enter_context(patch.object(acceleration, "_compiled_kernel", lambda: timed("compiled_reducer", kernel)))
            stack.enter_context(patch.object(acceleration, "_certifies_omitted_paths", timed("overflow_certificate", acceleration._certifies_omitted_paths)))
            stack.enter_context(patch.object(acceleration, "_normals", timed("helper_normal_validation", acceleration._normals)))
            stack.enter_context(patch.object(acceleration, "curve_values", timed("helper_curve", acceleration.curve_values)))
            stack.enter_context(patch.object(pricing, "conditional_put_payoffs", timed("payoffs", pricing.conditional_put_payoffs)))
            started = time.perf_counter()
            run()
            total = time.perf_counter()-started
        timers["other"] = total-sum(timers.values())
        profiles.append({"total_seconds": total, "components_seconds": timers, "component_counts": counts})

    profiler = cProfile.Profile()
    profiler.enable()
    run(count=4096, blocks=[6])
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(30)
    diagnostics = []
    for flag in (True, False, False, True, True, False):
        started = time.perf_counter()
        run(count=4096, diagnostics=flag, blocks=[6])
        diagnostics.append({"diagnostics": flag, "seconds": time.perf_counter()-started})
    median = statistics.median(p["total_seconds"] for p in profiles)
    components = {name: statistics.median(p["components_seconds"][name] for p in profiles)
                  for name in profiles[0]["components_seconds"]}
    result = {"repeats": profiles, "median_total_seconds": median,
        "median_component_seconds": components, "component_fractions": {k: v/median for k, v in components.items()},
        "reducer_amdahl_2x_speedup": 1/(1-components["compiled_reducer"]/median/2),
        "reducer_amdahl_infinite_speedup": 1/(1-components["compiled_reducer"]/median),
        "diagnostics_toggle_longest_block4096pairs": diagnostics}
    return result, stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", type=Path, default=WORK)
    parser.add_argument("--case", type=int, default=0, choices=range(12))
    parser.add_argument("--pairs", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile_path = args.out.with_suffix(".profile.txt")
    if args.pairs < 8 or args.repeats < 1 or args.out.exists() or profile_path.exists():
        parser.error("pairs>=8, positive repeats and new output paths required")
    work = json.loads(args.work_order.read_text())
    result, profile_text = profile(work["tasks"]["joint_unknown"], work["cases"][args.case],
                                   pairs=args.pairs, repeats=args.repeats)
    files = [Path(__file__), Path(pricing.__file__), Path(acceleration.__file__)]
    result.update(platform=platform.platform(), numpy=np.__version__, numba=numba.__version__,
        paths_per_block=2*args.pairs, case=work["cases"][args.case]["id"],
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        work_order_sha256=hashlib.sha256(args.work_order.read_bytes()).hexdigest(),
        scope="Warmed local serial NumPyFFT+Numba actual63 fixed-path surface; nested components separated; driver includes its own validation. Other includes dispatch, controls, regressions and summaries. No production writes.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    with profile_path.open("x") as output:
        output.write(profile_text)
    print(json.dumps({"median_seconds": result["median_total_seconds"], "component_fractions": result["component_fractions"]}))


if __name__ == "__main__":
    main()

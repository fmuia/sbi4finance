"""Build a temporary Cython kernel and compare it with warmed Numba and reference.

Run from lab/ with PYTHONPATH=.:src:../tutorial/src plus an isolated Cython
installation. Generated C, objects and extension binaries stay in --build-dir.
Only committed synthetic cases are read; production journals are never opened.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from unittest.mock import patch

import Cython
from Cython.Build import cythonize
import numba
import numpy as np
import scipy
from setuptools import Distribution, Extension

from sbi4f_tutorial.rough_options import _pricing_acceleration as acceleration, model, simulator
from sbi4f_tutorial.shared import pricing
from scripts.benchmark_rough_pricing_speed import ROOT, WORK, differences, surface


FLAGS = ["-O3", "-fno-fast-math", "-ffp-contract=off"]


def build_kernel(directory):
    directory = directory.resolve()
    if directory.exists():
        raise FileExistsError("new isolated build directory required")
    if directory.is_relative_to(ROOT):
        raise ValueError("generated build outputs must stay outside the checkout")
    directory.mkdir(parents=True)
    source = ROOT/"tutorial/src/sbi4f_tutorial/rough_options/_pricing_cython.pyx"
    started = time.perf_counter()
    extensions = cythonize([Extension("_bergomi_pricing_cython", [str(source)],
                                     extra_compile_args=FLAGS)],
                          build_dir=str(directory/"generated"), quiet=True,
                          compiler_directives={"language_level": 3})
    distribution = Distribution({"name": "bergomi-cython-research", "ext_modules": extensions})
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(directory/"lib")
    command.build_temp = str(directory/"objects")
    command.ensure_finalized()
    command.run()
    binary = next((directory/"lib").glob("_bergomi_pricing_cython*"))
    spec = importlib.util.spec_from_file_location("_bergomi_pricing_cython", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiler_command = list(getattr(command.compiler, "compiler_so", None)
                            or getattr(command.compiler, "compiler", None) or [])
    compiler_version = "not available"
    if compiler_command:
        try:
            compiler_version = subprocess.check_output(
                [*compiler_command, "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            pass  # Preserve a successful build when compiler version reporting is unavailable.
    return module.terminal_moments, {"seconds": time.perf_counter()-started,
        "flags": FLAGS, "cython": Cython.__version__, "binary_path": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "compiler_command": compiler_command, "compiler": compiler_version}


def compare_kernel(first, second):
    maxima = []
    equal = True
    for reference, candidate in zip(first, second, strict=True):
        if isinstance(reference, (bool, np.bool_)):
            if reference != candidate:
                raise AssertionError("finite-status changed")
            continue
        np.testing.assert_allclose(candidate, reference, rtol=5e-13, atol=5e-14)
        maxima.append(float(np.max(np.abs(candidate-reference))))
        equal = equal and np.array_equal(reference, candidate)
    return {"maximum_absolute_difference": max(maxima), "bitwise_equal": bool(equal)}


def microbenchmarks(compiled, task):
    results = []
    functions = {"numba": acceleration._compiled_kernel(), "cython": compiled}
    for steps, horizon in ((task["blocks"][0]["n_steps"], task["blocks"][0]["horizon"]),
                           (task["blocks"][-1]["n_steps"], task["blocks"][-1]["horizon"])):
        z = np.random.default_rng(20261111+steps).standard_normal((2048, steps, 3))
        driver, variance = simulator.rough_driver(.12, horizon/steps, z)
        arguments = (driver, variance, z, np.full(steps+1, .04), horizon/steps,
                     1.5, np.array([steps], dtype=np.int64))
        expected = functions["numba"](*arguments)
        comparison = compare_kernel(expected, compiled(*arguments))
        timing = {name: [] for name in functions}
        for repeat in range(7):
            for name in (tuple(functions) if repeat%2 == 0 else tuple(reversed(functions))):
                started = time.perf_counter(); functions[name](*arguments)
                timing[name].append(time.perf_counter()-started)
        medians = {name: statistics.median(values) for name, values in timing.items()}
        results.append({"steps": steps, "pairs": 2048, "timings": timing,
            "median_seconds": medians, "cython_speedup_over_numba": medians["numba"]/medians["cython"],
            "comparison": comparison})
    return results


def check_edges(compiled):
    result = []
    for hurst, eta in ((.025, 3.4), (.12, 1.5), (.49, 0.)):
        z = np.random.default_rng(904).standard_normal((13, 17, 3))
        driver, variance = simulator.rough_driver(hurst, .01, z)
        arguments = (driver, variance, z, np.linspace(.02, .08, 18), .01, eta,
                     np.array([1, 3, 17], dtype=np.int64))
        result.append({"hurst": hurst, "eta": eta,
            **compare_kernel(acceleration._compiled_kernel()(*arguments), compiled(*arguments))})
    z = np.zeros((2, 1, 3)); z[:, 0, 1] = 1e6
    with patch.object(acceleration, "_compiled_kernel", return_value=compiled):
        try:
            acceleration.antithetic_terminal_paths({"hurst": .1, "eta": 100., "rho": -.7},
                {"times": [0., .01], "values": [.04, .04]}, .01, z, [1])
        except ValueError as error:
            if "variance overflow" not in str(error):
                raise
        else:
            raise AssertionError("unused terminal-variance overflow was accepted")
    return {"cases": result, "terminal_variance_overflow_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--paths", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists() or args.paths < 8 or args.paths%2 or args.repeats < 1:
        parser.error("new output, even paths>=8 and positive repeats required")
    work = json.loads(WORK.read_text()); task = work["tasks"]["joint_unknown"]
    if any(index not in range(12) for index in args.cases):
        parser.error("joint synthetic cases 0..11 required")
    compiled, build = build_kernel(args.build_dir)
    micro = microbenchmarks(compiled, task)
    edges = check_edges(compiled)
    print(json.dumps({"microbenchmarks": [{"steps": r["steps"], "cython_speedup_over_numba": r["cython_speedup_over_numba"]} for r in micro], "edge_checks_passed": True}), flush=True)
    cases = []
    for index in args.cases:
        case = work["cases"][index]
        times = {name: [] for name in ("reference", "numba", "cython")}
        block_times = {name: [] for name in times}
        maximum = {}; finite = True
        for repeat in range(args.repeats):
            found = {}
            order = tuple(times) if repeat%2 == 0 else tuple(reversed(times))
            for name in order:
                if name == "cython":
                    with patch.object(acceleration, "_compiled_kernel", return_value=compiled):
                        value = surface(task, case, index, args.paths, "numba")
                else:
                    value = surface(task, case, index, args.paths, name)
                times[name].append({k: value[k] for k in ("seconds", "cpu_seconds")})
                block_times[name].append([b["seconds"] for b in value["blocks"]])
                found[name] = value
            for name in ("numba", "cython"):
                for block_index, (reference, candidate) in enumerate(zip(found["reference"]["blocks"], found[name]["blocks"], strict=True)):
                    for key, value in differences(reference["values"], candidate["values"]).items():
                        label = f"{name}.block{block_index}.{key}"
                        maximum[label] = max(maximum.get(label, 0), value)
        medians = {name: statistics.median(t["seconds"] for t in values) for name, values in times.items()}
        entry = {"case_id": case["id"], "timings": times, "block_timings": block_times,
            "median_seconds": medians, "numba_speedup_over_reference": medians["reference"]/medians["numba"],
            "cython_speedup_over_reference": medians["reference"]/medians["cython"],
            "cython_speedup_over_numba": medians["numba"]/medians["cython"],
            "all_outputs_bitwise_equal": all(value == 0 for value in maximum.values()),
            "maximum_absolute_differences": maximum}
        cases.append(entry)
        print(json.dumps({key: entry[key] for key in ("case_id", "median_seconds", "cython_speedup_over_numba", "all_outputs_bitwise_equal")}), flush=True)
    files = [Path(__file__), Path(pricing.__file__), Path(acceleration.__file__), Path(simulator.__file__),
             Path(model.__file__), ROOT/"tutorial/src/sbi4f_tutorial/rough_options/_pricing_cython.pyx"]
    record = {"format": "rough_cython_comparison_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "versions": {"python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "numba": numba.__version__, "cython": Cython.__version__},
        "build": build, "paths_per_block": args.paths, "repeats": args.repeats, "microbenchmarks": micro,
        "correctness": edges, "cases": cases,
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "work_order_sha256": hashlib.sha256(WORK.read_bytes()).hexdigest(),
        "scope": "Serial warmed local CPU; identical terminal loops with finite checks, same RNG/FFT/grid/controls. Full surface fixed paths excludes adaptive escalation/SQLite/Slurm. Cython experimental override only; no installed runtime extension or production change."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    main()

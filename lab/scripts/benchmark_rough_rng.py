"""Compare RNG buffering and future bit generators without changing bank sources.

The buffer experiment preserves PCG64 draws and compares every fitted output.
Other bit generators yield different streams and are research options requiring
a new generator identity; they are never installed into a production worker.
Run from lab/ with PYTHONPATH=.:src:../tutorial/src.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch

import numpy as np

from scripts.benchmark_rough_pricing_speed import ROOT, WORK, differences, surface


def buffer_micro(steps, *, pairs, chunks, repeats):
    rows = []
    for count in steps:
        shape = (pairs, count, 3)
        timings = {"allocate": [], "reuse_out": []}
        last = {}
        for repeat in range(repeats):
            order = tuple(timings) if repeat % 2 == 0 else tuple(reversed(timings))
            for mode in order:
                rng = np.random.default_rng(20261111)
                buffer = np.empty(shape) if mode == "reuse_out" else None
                started = time.perf_counter()
                for _ in range(chunks):
                    if mode == "reuse_out":
                        rng.standard_normal(out=buffer)
                        value = buffer
                    else:
                        value = rng.standard_normal(shape)
                timings[mode].append(time.perf_counter()-started)
                last[mode] = value.copy()
        np.testing.assert_array_equal(last["allocate"], last["reuse_out"])
        medians = {k: statistics.median(v) for k, v in timings.items()}
        rows.append({"steps": count, "shape": shape, "chunks": chunks, "times": timings,
            "median_seconds": medians, "speedup": medians["allocate"]/medians["reuse_out"],
            "last_chunk_bitwise_equal": True})
    return rows


def buffer_surface(task, case, *, case_index, paths, repeats):
    original = np.random.default_rng

    class BufferGenerator:
        def __init__(self, *args, **kwargs):
            self.inner, self.buffers = original(*args, **kwargs), {}

        def standard_normal(self, shape):
            if shape not in self.buffers:
                self.buffers[shape] = np.empty(shape, dtype=np.float64)
            self.inner.standard_normal(out=self.buffers[shape])
            return self.buffers[shape]

    surface(task, case, case_index, 8, "numba")
    durations, checks = {"allocate": [], "reuse_out": []}, []
    for repeat in range(repeats):
        found = {}
        order = tuple(durations) if repeat % 2 == 0 else tuple(reversed(durations))
        for name in order:
            context = patch.object(np.random, "default_rng", BufferGenerator) if name == "reuse_out" else nullcontext()
            with context:
                result = surface(task, case, case_index, paths, "numba")
            durations[name].append(result["seconds"])
            found[name] = result
        maximum = {}
        for block_index, (a, b) in enumerate(zip(found["allocate"]["blocks"], found["reuse_out"]["blocks"], strict=True)):
            for key, value in differences(a["values"], b["values"]).items():
                maximum[f"block{block_index}.{key}"] = value
        checks.append({"all_numeric_outputs_bitwise_equal": all(v == 0 for v in maximum.values()),
                       "maximum_absolute_difference": max(maximum.values())})
    medians = {k: statistics.median(v) for k, v in durations.items()}
    return {"case_id": case["id"], "times": durations, "median_seconds": medians,
            "speedup": medians["allocate"]/medians["reuse_out"], "checks": checks}


def bit_generators(steps, *, pairs, chunks, repeats):
    names = ("PCG64", "PCG64DXSM", "SFC64", "Philox")
    rows = []
    for count in steps:
        shape = (pairs, count, 3)
        buffer = np.empty(shape, dtype=np.float64)
        timings = {name: [] for name in names}
        for repeat in range(repeats):
            offset = repeat % len(names)
            for name in names[offset:]+names[:offset]:
                rng = np.random.Generator(getattr(np.random, name)(20261111+repeat))
                started = time.perf_counter()
                for _ in range(chunks):
                    rng.standard_normal(out=buffer)
                timings[name].append(time.perf_counter()-started)
        medians = {k: statistics.median(v) for k, v in timings.items()}
        rows.append({"steps": count, "shape": shape, "chunks": chunks, "times": timings,
            "median_seconds": medians, "speedup_vs_PCG64": {k: medians["PCG64"]/v for k, v in medians.items()}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", type=Path, default=WORK)
    parser.add_argument("--mode", choices=("all", "micro", "surface", "bitgenerators"), default="all")
    parser.add_argument("--steps", type=int, nargs="+", default=[622, 1796])
    parser.add_argument("--pairs-per-chunk", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--paths", type=int, default=32768)
    parser.add_argument("--case", type=int, default=0, choices=range(12))
    parser.add_argument("--micro-repeats", type=int, default=4)
    parser.add_argument("--surface-repeats", type=int, default=3)
    parser.add_argument("--bitgen-repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bounds = [args.pairs_per_chunk, args.chunks, args.micro_repeats, args.surface_repeats, args.bitgen_repeats, *args.steps]
    if any(v < 1 for v in bounds) or args.paths < 8 or args.paths % 2 or args.out.exists():
        parser.error("positive sizes/repeats, even paths>=8 and a new output path required")
    result = {"platform": platform.platform(), "numpy": np.__version__, "mode": args.mode,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Research only; PCG64 buffer modes preserve realized draws. Alternate bit generators change all realized streams and require a new source/RNG identity. No production writes or deployment."}
    options = {"pairs": args.pairs_per_chunk, "chunks": args.chunks}
    if args.mode in ("all", "micro"):
        result["buffer_micro"] = buffer_micro(args.steps, **options, repeats=args.micro_repeats)
    if args.mode in ("all", "surface"):
        work = json.loads(args.work_order.read_text())
        result["buffer_surface"] = buffer_surface(work["tasks"]["joint_unknown"], work["cases"][args.case],
            case_index=args.case, paths=args.paths, repeats=args.surface_repeats)
        result["work_order_sha256"] = hashlib.sha256(args.work_order.read_bytes()).hexdigest()
    if args.mode in ("all", "bitgenerators"):
        result["bitgenerators"] = bit_generators(args.steps, **options, repeats=args.bitgen_repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"mode": args.mode, "out": str(args.out)}))


if __name__ == "__main__":
    main()

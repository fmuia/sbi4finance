"""Compare opt-in rough pricing kernels on the committed synthetic actual63 panel.

Run from lab/ with PYTHONPATH=.:src:../tutorial/src. This benchmark writes a new
research record only: it never opens a production journal or changes a bank.
Fixed first-rung path counts isolate kernel cost from adaptive stopping.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import numpy as np
import scipy

from sbi4f_tutorial.rough_options import actual_layout, simulator
from sbi4f_tutorial.shared import pricing
from sbi4finance.calibration.pricing import heston_variance_curve

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "lab/evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation-portable/work-order.json"


def differences(reference, candidate):
    """Check every fitted-control diagnostic as well as all three public outputs."""
    maxima = {}

    def compare(a, b, name):
        if isinstance(a, dict):
            if set(a) != set(b):
                raise AssertionError(f"diagnostic keys changed at {name}")
            for key in a:
                compare(a[key], b[key], f"{name}.{key}")
        elif isinstance(a, (np.ndarray, list, float, int)) and not isinstance(a, bool):
            a, b = np.asarray(a), np.asarray(b)
            np.testing.assert_allclose(b, a, rtol=5e-12, atol=5e-14, err_msg=name)
            maxima[name] = float(np.max(np.abs(a-b))) if a.size else 0.0
        elif a != b:
            raise AssertionError(f"diagnostic value changed at {name}")

    for name, a, b in zip(("prices", "standard_errors", "covariance", "diagnostics"), reference, candidate, strict=True):
        compare(a, b, name)
    return maxima


def surface(task, case, case_index, paths, backend):
    parameters = dict(zip(("hurst", "eta", "rho"), case["parameters"][:3], strict=True))
    curve = heston_variance_curve(actual_layout.simulation_nodes(task), *case["parameters"][3:])
    results = []
    started, cpu = time.perf_counter(), time.process_time()
    for block_index, block in enumerate(task["blocks"]):
        contracts = [c for c in task["design"]["contracts"] if c["maturity"] == block["horizon"]]
        diagnostic = {"capture_raw_control": True}
        block_started = time.perf_counter()
        output = pricing.rough_block(parameters, curve, contracts,
            horizon=block["horizon"], n_steps=block["n_steps"], n_pairs=paths//2,
            seed=np.random.SeedSequence([0x52425350, 20261111, case_index, block_index]),
            diagnostics=diagnostic, path_backend=backend)
        results.append({"seconds": time.perf_counter()-block_started,
            "values": (*output, diagnostic)})
    return {"seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
        "blocks": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", type=Path, default=WORK)
    parser.add_argument("--cases", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--paths", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.paths < 8 or args.paths % 2 or args.repeats < 1 or args.out.exists():
        parser.error("even path count >=8, positive repeats and a new output path required")
    work = json.loads(args.work_order.read_text())
    task = work["tasks"]["joint_unknown"]
    if any(i < 0 or i >= 12 for i in args.cases):
        parser.error("use joint synthetic case indices 0 through 11")
    import numba
    import sbi4f_tutorial.rough_options._pricing_acceleration as acceleration

    started = time.perf_counter()
    surface(task, work["cases"][0], 0, 8, "numba")
    warmup = time.perf_counter()-started
    cases = []
    for index in args.cases:
        case = work["cases"][index]
        timings = {name: [] for name in ("reference", "numba")}
        block_timings = {name: [] for name in timings}
        maximum = {}
        for repeat in range(args.repeats):
            found = {}
            # Alternate order to reduce systematic warm-cache/order advantage.
            order = tuple(timings) if repeat % 2 == 0 else tuple(reversed(timings))
            for backend in order:
                result = surface(task, case, index, args.paths, backend)
                timings[backend].append({k: result[k] for k in ("seconds", "cpu_seconds")})
                block_timings[backend].append([b["seconds"] for b in result["blocks"]])
                found[backend] = result
            for block_index, (ref, fast) in enumerate(zip(found["reference"]["blocks"], found["numba"]["blocks"], strict=True)):
                for key, value in differences(ref["values"], fast["values"]).items():
                    label = f"block{block_index}.{key}"
                    maximum[label] = max(maximum.get(label, 0), value)
        medians = {name: statistics.median(t["seconds"] for t in values) for name, values in timings.items()}
        entry = {"case_id": case["id"], "timings": timings, "block_timings": block_timings,
            "median_seconds": medians, "speedup": medians["reference"]/medians["numba"],
            "all_outputs_bitwise_equal": all(value == 0 for value in maximum.values()),
            "maximum_absolute_differences": maximum}
        cases.append(entry)
        print(json.dumps({k: entry[k] for k in ("case_id", "median_seconds", "speedup", "all_outputs_bitwise_equal")}), flush=True)
    files = [Path(__file__), Path(pricing.__file__), Path(simulator.__file__), Path(acceleration.__file__)]
    record = {"format": "rough_actual63_pricing_speed_experiment_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__},
        "source_commit": "record separately with git diff; this record binds exact source bytes",
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "work_order_sha256": hashlib.sha256(args.work_order.read_bytes()).hexdigest(),
        "paths_per_block": args.paths, "antithetic_pairs_per_chunk": pricing._PAIR_CHUNK,
        "steps_per_block": [b["n_steps"] for b in task["blocks"]], "numba_warmup_seconds": warmup,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "cases": cases, "production_journals_opened": False, "remote_jobs_changed": False,
        "scope": "Fixed-path seven-expiry 63-quote surface on local CPU; warmed serial execution, same RNG/chunk/controls/grid. Excludes adaptive escalation, SQLite, Slurm and training. Local equality does not certify cross-platform identity or continuous-model accuracy."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

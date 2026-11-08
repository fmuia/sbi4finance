"""Reproducible Heston pricing comparison, without training or bank generation.

Run from lab/, e.g. ``pixi run --locked -e core python
scripts/benchmark_heston_pricing.py --out /path/to/benchmark``. Reuse panel.json
with --panel to compare checkouts; --reference accepts an earlier report.json.
The frozen reference, current checked bank route and notebook route all retain
their price bounds. Failures are results: no row is discarded or redrawn.
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SEED = 20261107


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_panel(task):
    import numpy as np

    bounds = np.asarray(task["prior_bounds"], dtype=float)
    if task["parameter_names"] != ["initial_variance", "theta", "vol_of_vol", "rho"] or bounds.shape != (4, 2):
        raise ValueError("the four-parameter Heston option task is required")
    random = np.random.default_rng(SEED).uniform(bounds[:, 0], bounds[:, 1], (256, 4))
    rows = [{"id": f"prior:{i}", "parameters": row.tolist()} for i, row in enumerate(random)]
    rows += [{"id": f"corner:{i}", "parameters": list(row)}
             for i, row in enumerate(itertools.product(*bounds.tolist()))]
    # Low variance, strong vol-of-vol and correlation stress the one-week quotes.
    hard = [[.01001, .01001, 1.4999, -.9499], [.01001, .15999, 1.4999, -.9499],
            [.015, .025, 1.3, -.9], [.01001, .01001, 1.4999, -.1001]]
    if np.any(np.asarray(hard) < bounds[:, 0]) or np.any(np.asarray(hard) > bounds[:, 1]):
        raise ValueError("the targeted hard cases must lie inside the task prior")
    rows += [{"id": f"short-maturity:{i}", "parameters": row} for i, row in enumerate(hard)]
    return {"format": "heston_pricing_panel_v1", "seed": SEED, "task": task, "rows": rows,
            "scope": "256 fixed prior draws, all 16 prior-box corners and four short-maturity stress cases; no redraws"}


def validate_panel(panel, task):
    import numpy as np
    from sbi4finance.calibration import tasks

    tasks.validate_task(task)
    if panel.get("format") != "heston_pricing_panel_v1" or panel.get("task") != task:
        raise ValueError("panel format/task mismatch")
    rows = panel["rows"]
    if len(rows) != 276 or len({row["id"] for row in rows}) != 276:
        raise ValueError("the complete 276-row panel with distinct IDs is required")
    if panel != make_panel(task):
        raise ValueError("panel must match the predeclared seed, corners and hard cases")
    if len(task["context"]["contracts"]) != 63:
        raise ValueError("the ordered 63-contract task is required")
    if not np.isfinite(np.asarray([row["parameters"] for row in rows])).all():
        raise ValueError("nonfinite panel parameters")


def clear_caches():
    from sbi4f_tutorial.heston_options import model
    from sbi4f_tutorial.shared import pricing

    model._quadrature.cache_clear()
    # Optimization-specific caches are optional so this runner works on baseline.
    for module in (model, pricing):
        for name in ("_fourier_plan", "_surface_plan"):
            cache = getattr(module, name, None)
            if hasattr(cache, "cache_clear"):
                cache.cache_clear()


@contextmanager
def reference_pricer(enabled):
    from sbi4f_tutorial.heston_options import model
    from sbi4f_tutorial.shared import pricing
    from sbi4finance.calibration.heston_pricing_reference import original_price

    original = pricing.heston_surface
    original_european = model.european_price

    def surface(parameters, v0, contracts, *, nodes, upper_limit):
        import numpy as np

        # Freeze the pre-optimization surface dispatch as well as its CF path.
        strike = np.exp(np.log(np.asarray([c["strike"] for c in contracts])))
        maturity = np.asarray([c["maturity"] for c in contracts])
        kind = np.asarray([c["option_type"] for c in contracts])
        prices = np.empty(len(contracts))
        for option in np.unique(kind):
            chosen = kind == option
            prices[chosen] = original_price(1., strike[chosen], 1., maturity[chosen], parameters, v0,
                                            kind=str(option), nodes=nodes, upper_limit=upper_limit)
        return prices

    if enabled:
        pricing.heston_surface = surface
        model.european_price = original_price
    try:
        yield
    finally:
        pricing.heston_surface = original
        model.european_price = original_european


def frozen_checked(task, parameters, v0):
    """Original adjacent-rung retry policy, including repeated successful rungs."""
    import numpy as np
    from sbi4f_tutorial.shared import pricing
    from sbi4finance.calibration import tasks

    limit = task["numerics"]["max_refinement_over_quote_sigma"] * tasks.sigma(task)
    failure = None
    ladder = task["numerics"]["quadrature_ladder"]
    for (nodes, cutoff), (next_nodes, next_cutoff) in zip(ladder, ladder[1:]):
        try:
            base = pricing.heston_surface(parameters, v0, task["context"]["contracts"], nodes=nodes, upper_limit=cutoff)
            refined = pricing.heston_surface(parameters, v0, task["context"]["contracts"], nodes=next_nodes, upper_limit=next_cutoff)
        except ValueError as error:
            failure = error
            continue
        if np.all(np.abs(refined - base) <= limit):
            return refined, refined - base
        failure = ValueError("Fourier refinement budget failed")
    raise ValueError(f"{failure} on every quadrature rung")


def evaluate(task, row, route, *, frozen=False):
    from sbi4f_tutorial.heston_options.lesson import heston_smile
    from sbi4f_tutorial.shared import pricing
    from sbi4finance.calibration import tasks

    with reference_pricer(frozen):
        parameters, v0 = tasks._heston_split(tasks.model_parameters(task, row))
        if route == "notebook":
            if not frozen:
                prices = heston_smile(task, row)
            else:
                try:
                    prices = pricing.heston_surface(parameters, v0, task["context"]["contracts"], nodes=1024, upper_limit=600.)
                except ValueError:
                    prices = frozen_checked(task, parameters, v0)[0]
            return {"prices": prices.tolist(), "refined_minus_base": None}
        prices, difference = (frozen_checked(task, parameters, v0) if frozen else tasks.heston_prices(task, parameters, v0))
        return {"prices": prices.tolist(), "refined_minus_base": difference.tolist()}


def timed_row(task, row, route, frozen):
    before_wall, before_cpu = time.perf_counter(), time.process_time()
    try:
        outcome = {"status": "ok", **evaluate(task, row["parameters"], route, frozen=frozen)}
    except ValueError as error:
        outcome = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    return {"id": row["id"], **outcome, "wall_seconds": time.perf_counter() - before_wall,
            "process_cpu_seconds": time.process_time() - before_cpu}


def summary(rows):
    import numpy as np

    result = {"evaluations": len(rows), "failures": sum(row["status"] != "ok" for row in rows)}
    for clock in ("wall_seconds", "process_cpu_seconds"):
        values = np.asarray([row[clock] for row in rows])
        result[clock] = {"total": float(values.sum()), "median": float(np.median(values)),
                         "p95": float(np.quantile(values, .95)), "max": float(values.max())}
    return result


def compare_rows(actual, reference, sigma):
    import numpy as np

    if [row["id"] for row in actual] != [row["id"] for row in reference]:
        raise ValueError("comparison IDs/order differ")
    failures, price_diff, noise_diff, diagnostic_diff = [], 0., 0., 0.
    for row, other in zip(actual, reference, strict=True):
        if row["status"] != other["status"] or (row["status"] == "failed" and row["error"] != other["error"]):
            failures.append(row["id"])
        if row["status"] == other["status"] == "ok":
            difference = np.abs(np.asarray(row["prices"]) - other["prices"])
            price_diff = max(price_diff, float(difference.max()))
            noise_diff = max(noise_diff, float((difference / sigma).max()))
            if row["refined_minus_base"] is not None and other["refined_minus_base"] is not None:
                diagnostic_diff = max(diagnostic_diff, float(np.max(np.abs(
                    np.asarray(row["refined_minus_base"]) - other["refined_minus_base"]))))
    return {"failure_outcome_mismatch_ids": failures, "max_abs_price_difference": price_diff,
            "max_price_difference_over_quote_sigma": noise_diff,
            "max_abs_refinement_difference": diagnostic_diff,
            "passes_price_and_failure_gate": not failures and price_diff <= 1e-12 and noise_diff <= 1e-6}


def run(task, panel, repeats):
    import numpy as np

    routes = [("reference_checked", "checked", True), ("current_checked", "checked", False),
              ("reference_notebook", "notebook", True), ("current_notebook", "notebook", False)]
    outputs = {name: {"cold": [], "warm": []} for name, _, _ in routes}
    # Cold means empty caches at the start of the panel, not before every row.
    for name, route, frozen in routes:
        clear_caches()
        outputs[name]["cold"] = [timed_row(task, row, route, frozen) for row in panel["rows"]]
    for repeat in range(repeats):
        for row_index, row in enumerate(panel["rows"]):
            for name, route, frozen in routes[::1 if (row_index + repeat) % 2 == 0 else -1]:
                outputs[name]["warm"].append(timed_row(task, row, route, frozen))
    sigma = np.asarray(task["context"]["quote_noise_std"])
    comparisons = {}
    for route in ("checked", "notebook"):
        for cache in ("cold", "warm"):
            comparisons[f"{route}_{cache}"] = compare_rows(outputs[f"current_{route}"][cache],
                                                          outputs[f"reference_{route}"][cache], sigma)
    return {"results": outputs, "summaries": {name: {cache: summary(rows) for cache, rows in result.items()}
                                               for name, result in outputs.items()}, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=ROOT / "configs/tasks/calibration_heston_bank.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--reference", type=Path, help="Report from the preceding production checkout")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("positive thread and repeat counts required")
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(args.threads)
    # Set threads before importing numpy/BLAS or the numerical runtime.
    task_file = json.loads(args.task.read_text())
    task = task_file.get("task", task_file)
    panel = json.loads(args.panel.read_text()) if args.panel else make_panel(task)
    validate_panel(panel, task)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "panel.json").write_text(json.dumps(panel, indent=2, allow_nan=False) + "\n")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    report = run(task, panel, args.repeats)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sources = [Path(__file__), ROOT / "src/sbi4finance/calibration/heston_pricing_reference.py",
               ROOT.parent / "tutorial/src/sbi4f_tutorial/heston_options/model.py",
               ROOT.parent / "tutorial/src/sbi4f_tutorial/heston_options/lesson.py",
               ROOT.parent / "tutorial/src/sbi4f_tutorial/shared/pricing.py",
               ROOT.parent / "tutorial/src/sbi4f_tutorial/shared/tasks.py"]
    report.update(format="heston_pricing_benchmark_v1", panel_sha256=identity(panel), task_sha256=identity(task),
                  task_file_sha256=digest(args.task), repeats=args.repeats, threads=args.threads,
                  source_sha256={str(path.relative_to(ROOT.parent)): digest(path) for path in sources},
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                  python=platform.python_version(), versions={name: metadata.version(name) for name in ("numpy", "scipy")},
                  total_wall_seconds=time.perf_counter() - started_wall, total_process_cpu_seconds=time.process_time() - started_cpu,
                  process_peak_rss_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
                  memory_scope="Process lifetime RSS high-water mark, including imports and both implementations; not incremental allocation.",
                  cache_scope="Cold panel clears caches once before its first row; warm rows alternate implementation order and retain caches.",
                  resource_allocation_seconds=None, resource_scope="Local process benchmark; no scheduler allocation or queue time.")
    if args.reference:
        previous = json.loads(args.reference.read_text())
        if previous["panel_sha256"] != report["panel_sha256"]:
            raise ValueError("previous report used a different panel")
        import numpy as np
        report["previous_report_sha256"] = digest(args.reference)
        report["previous_checkout_comparisons"] = {route: compare_rows(report["results"][route]["cold"],
            previous["results"][route]["cold"], np.asarray(task["context"]["quote_noise_std"]))
            for route in ("current_checked", "current_notebook")}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "summaries": report["summaries"], "comparisons": report["comparisons"]}, indent=2))
    if any(not result["passes_price_and_failure_gate"] for result in report["comparisons"].values()):
        raise SystemExit("pricing/reference discrepancy; inspect retained rows, including all failures")
    if any(not result["passes_price_and_failure_gate"] for result in report.get("previous_checkout_comparisons", {}).values()):
        raise SystemExit("previous-checkout discrepancy; inspect retained report")


if __name__ == "__main__":
    main()

"""The declared 44-call working-grid pilot; no bank generation or retries.

The two policies share 16 parameter draws, not pricing randomness. Every
possible adaptive block/rung stream is audited before a price is computed.
One component runs per process so an observer can journal the unchanged
rough_block calls made by the actual tasks.price adaptive implementation.
"""

from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from ..gbm_bank import _integer, _json, _write_json
from . import bank, numerical_pilot, pricing, tasks
from sbi4f_tutorial.shared import pricing as core_pricing

FORMAT = "rough_working_grid_pilot_v1"
ROOT_SEED = 271100
POLICIES = (("working", (16, 4, 4)), ("refined", (32, 8, 8)))
ANCHOR_PATHS = {0: 131072, 4: 1048576, 6: 524288}  # Total paths, not pairs.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PACKAGE = Path(__file__).resolve().parents[1]


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _seed(task, root, purpose, identity):
    digest = _digest({"format": FORMAT, "task": task, "identity": identity})
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    return int(np.random.SeedSequence([0x52574750, root, purpose, *words])
               .generate_state(1, dtype=np.uint64)[0])


def _rng_digest(seed):
    # default_rng uses the entire SeedSequence to initialize PCG64, not just
    # generate_state(1). Audit its actual initialized state as well.
    return _digest(np.random.PCG64(seed).state)


def _stream(seed, block, rung, paths):
    entropy = [seed, block, rung]
    sequence = np.random.SeedSequence(entropy)
    return {"block": block, "rung": rung, "paths": paths,
            "seed_sequence_entropy": entropy,
            "seed_sequence_state_uint64": sequence.generate_state(4, dtype=np.uint64).tolist(),
            "pcg64_initial_state_sha256": _rng_digest(sequence)}


def _design(task, root, *, frozen_points=None):
    tasks.validate_task(task)
    if not tasks.is_joint_curve_task(task) or len(task["numerics"]["blocks"]) != 3:
        raise ValueError("the three-block joint rough task is required")
    root = _integer(root, "root seed")
    bounds = np.asarray(task["prior_bounds"])
    if frozen_points is not None and len(frozen_points) != 16:
        raise ValueError("exactly sixteen frozen prior points required")
    points, reduced, states = [], [], []
    for index in range(16):
        seed = _seed(task, root, 0, {"prior_point": index})
        values = (np.random.default_rng(seed).uniform(bounds[:, 0], bounds[:, 1]) if frozen_points is None
                  else np.asarray(frozen_points[index]["parameters"], dtype=float))
        if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < bounds[:, 0]) \
                or np.any(values > bounds[:, 1]):
            raise ValueError("frozen prior point is not a finite supported six-parameter vector")
        points.append({"point_id": index, "parameter_seed": seed,
                       "pcg64_initial_state_sha256": _rng_digest(seed),
                       "parameters": values.tolist()})
        reduced.append(seed)
        states.append(points[-1]["pcg64_initial_state_sha256"])
    policies = []
    for name, scales in POLICIES:
        policy_task = numerical_pilot.refined_task(task, 1, block_multipliers=scales)
        policies.append({"name": name, "block_multipliers": list(scales), "task": policy_task,
                         "task_sha256": _digest(policy_task)})
    components = []

    def add(mode, point, policy, replicate, paths):
        identity = {"mode": mode, "point_id": point["point_id"],
                    "policy": policy["name"], "replicate": replicate}
        seed = _seed(task, root, 1 if mode == "adaptive" else 2, identity)
        ladder = policy["task"]["numerics"]["paths_ladder"] if mode == "adaptive" else [paths]
        streams = [_stream(seed, block, rung, count) for block in range(3)
                   for rung, count in enumerate(ladder)]
        limit = 1200 if mode == "adaptive" else (5400 if point["point_id"] == 4
                  and policy["name"] == "refined" else 1800)
        components.append({"component_index": len(components), **identity,
                           "parameters": point["parameters"], "pricing_seed": seed,
                           "fixed_total_paths": paths, "streams": streams,
                           "resources": {"cpus": 1, "memory_gib": 8, "wall_limit_seconds": limit}})
        reduced.append(seed)
        for stream in streams:
            reduced.append(stream["seed_sequence_state_uint64"][0])
            states.append(stream["pcg64_initial_state_sha256"])

    for point in points:
        for policy in policies:
            add("adaptive", point, policy, None, None)
    stress = numerical_pilot.cases(task)
    for index, paths in ANCHOR_PATHS.items():
        point = {"point_id": index, "parameters": stress[index]["parameters"]}
        for policy in policies:
            for replicate in range(2):
                add("fixed_anchor", point, policy, replicate, paths)
    if len(set(reduced)) != len(reduced) or len(set(states)) != len(states):
        raise ValueError("actual reduced seed or initialized PCG64 state collision; no redraw")
    return {"format": FORMAT, "root_seed": root, "task": task, "task_sha256": _digest(task),
            "prior_sampling": "16 independent uniform six-parameter draws; same parameters at both policies",
            "prior_points": points, "policies": policies, "components": components,
            "seed_protocol": "Separate prior/adaptive/fixed purposes; exact task/point/policy/rep identity. "
                             "rough_surface uses SeedSequence([pricing_seed, block, rung]); independent streams.",
            "seed_audit": {"passed": True, "unique_reduced_uint64": len(reduced),
                           "unique_initialized_pcg64_states": len(states),
                           "all_possible_block_rung_streams": sum(len(c["streams"]) for c in components)},
            "component_wall_allocation_seconds": sum(c["resources"]["wall_limit_seconds"] for c in components),
            "whole_pilot_cpu_ceiling_seconds": 72000,
            "scientific_scope": "working-grid development, not posterior/product or whole-prior numerical acceptance"}


def _sources(task):
    script = PACKAGE.parents[1] / "scripts/pilot_rough_working_grid.py"
    config = PACKAGE.parents[1] / "configs/tasks/calibration_rough_joint_pilot.json"
    return {**bank.source_hashes(task),
            "calibration/numerical_pilot.py": hashlib.sha256(Path(numerical_pilot.__file__).read_bytes()).hexdigest(),
            "calibration/rough_working_grid_pilot.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scripts/pilot_rough_working_grid.py": hashlib.sha256(script.read_bytes()).hexdigest(),
            "configs/tasks/calibration_rough_joint_pilot.json": hashlib.sha256(config.read_bytes()).hexdigest()}


def _runtime():
    return {**bank.versions(), "numba": metadata.version("numba"),
            "system": platform.system(), "machine": platform.machine(),
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES}}


def declare(task, *, root_seed=ROOT_SEED, source_revision):
    """Freeze every parameter, task, possible stream and resource limit before pricing."""
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("source revision is required")
    declaration = _design(copy.deepcopy(task), root_seed)
    declaration.update(source_sha256=_sources(task), runtime=_runtime(), source_revision=source_revision)
    declaration["declaration_sha256"] = _digest(declaration)
    return declaration


def validate_declaration(declaration, *, execution=False):
    body = dict(declaration)
    recorded = body.pop("declaration_sha256", None)
    if _digest(body) != recorded:
        raise ValueError("declaration digest mismatch")
    # Execution replays the recorded generator exactly. A reducer on another
    # platform uses the original, digest-bound parameter table unchanged;
    # cross-platform float regeneration never replaces those values.
    expected = _design(body["task"], body["root_seed"],
                       frozen_points=None if execution else body["prior_points"])
    if any(body.get(key) != value for key, value in expected.items()):
        raise ValueError("declaration design, parameter or seed identity mismatch")
    if execution and (body["source_sha256"] != _sources(body["task"]) or body["runtime"] != _runtime()):
        raise ValueError("execution source/runtime differs from frozen declaration")
    return declaration


def _policy(declaration, component):
    return next(p for p in declaration["policies"] if p["name"] == component["policy"])


def _arrays(prices, errors, covariance):
    return {"prices": np.asarray(prices).tolist(), "standard_errors": np.asarray(errors).tolist(),
            "covariance": np.asarray(covariance).tolist()}


@contextmanager
def _observe(report, component, sigma, progress):
    """Process-local journal; the original estimator and RNG receive unchanged arguments."""
    original = core_pricing.rough_block
    declared = {s["pcg64_initial_state_sha256"]: s for s in component["streams"]}

    def observed(*args, **kwargs):
        identity = _rng_digest(kwargs["seed"])
        if identity not in declared or kwargs["n_pairs"] * 2 != declared[identity]["paths"]:
            raise ValueError("actual block/rung stream is not in the frozen declaration")
        if any(a["stream"]["pcg64_initial_state_sha256"] == identity for a in report["attempts"]):
            raise ValueError("a declared stream was repeated")
        row = {"stream": declared[identity], "status": "running", "started_utc": _utc()}
        report["attempts"].append(row)
        progress(report)  # Persist before the expensive call, including censored attempts.
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            prices, errors, covariance = original(*args, **kwargs)
        except BaseException as error:
            elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
            row.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                       error_type=type(error).__name__, error=str(error))
            raise
        else:
            elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
            if not all(np.isfinite(a).all() for a in (prices, errors, covariance)) or np.any(errors < 0):
                row.update(status="failed", error="nonfinite prices or covariance, or negative SE")
                raise FloatingPointError(row["error"])
            # Resolve this block's global quote sigmas through its exact contracts.
            chosen = [report["contract_keys"].index((c["maturity"], c["strike"], c["option_type"]))
                      for c in args[2]]
            ratio = np.asarray(errors) / sigma[chosen]
            row.update(status="priced", **_arrays(prices, errors, covariance),
                       max_mc_se_over_quote_sigma=float(np.max(ratio)))
            return prices, errors, covariance
        finally:
            # Take endpoints before JSON serialization; block totals exclude journaling.
            row.update(cpu_seconds=elapsed_cpu, wall_seconds=elapsed_wall,
                       finished_utc=_utc())
            progress(report)

    core_pricing.rough_block = observed
    try:
        yield
    finally:
        core_pricing.rough_block = original


def run_component(declaration, component_index, output):
    """Run one immutable component once, with an atomic journal and no output reuse."""
    validate_declaration(declaration, execution=True)
    index = _integer(component_index, "component index")
    if index >= len(declaration["components"]):
        raise ValueError("component index is outside the declaration")
    component = declaration["components"][index]
    task = _policy(declaration, component)["task"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sigma = tasks.sigma(task)
    report = {"format": FORMAT, "declaration_sha256": declaration["declaration_sha256"],
              "component": component, "source_sha256": declaration["source_sha256"],
              "runtime": declaration["runtime"], "status": "running", "started_utc": _utc(),
              "attempts": [], "scientific_acceptance": "not_assessed",
              "timing_scope": "actual tasks.price for adaptive; instrumented total includes journal/setup; "
                              "summed block times exclude JSON journaling",
              "contract_keys": [(c["maturity"], c["strike"], c["option_type"])
                                for c in task["context"]["contracts"]]}

    def progress(value):
        _write_json(output / "report.json", value)

    progress(report)
    wall, cpu = time.perf_counter(), time.process_time()
    diagnostics = []
    try:
        with _observe(report, component, sigma, progress):
            if component["mode"] == "adaptive":
                prices, errors, covariance, _, used = tasks.price(
                    task, component["parameters"], [], component["pricing_seed"], diagnostics=diagnostics)
            else:
                # Infinite selection budget is a fixed-mode control only. Task
                # budgets remain frozen and the measured .25/.5 flags are below.
                prices, errors, covariance, used = pricing.rough_surface(
                    tasks.model_parameters(task, component["parameters"]),
                    tasks.joint_curve(task, component["parameters"]), task["context"]["contracts"],
                    task["numerics"]["blocks"], [component["fixed_total_paths"]], sigma, np.inf,
                    component["pricing_seed"], cap=None, diagnostics=diagnostics)
        ratio = errors / sigma
        numerics = task["numerics"]
        report.update(status="priced", **_arrays(prices, errors, covariance),
                      paths_per_block=np.asarray(used, dtype=int).tolist(),
                      max_mc_se_over_quote_sigma=float(np.max(ratio)),
                      mc_target_met=bool(np.all(ratio <= numerics["max_mc_se_over_quote_sigma"])),
                      within_exception_cap=bool(np.all(ratio <= numerics["max_mc_se_over_quote_sigma_last_rung"])))
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(error).__name__, error=str(error))
    finally:
        report.update(instrumented_cpu_seconds=time.process_time() - cpu,
                      instrumented_wall_seconds=time.perf_counter() - wall, finished_utc=_utc())
        report["summed_block_cpu_seconds"] = sum(a.get("cpu_seconds", 0.) for a in report["attempts"])
        report["summed_block_wall_seconds"] = sum(a.get("wall_seconds", 0.) for a in report["attempts"])
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = int(rss if platform.system() == "Darwin" else rss * 1024)
        report.pop("contract_keys")
        progress(report)
    return report


def reduce_components(declaration, reports, *, accounting=()):
    """Validate observed streams; retain missing, interrupted and scheduler-censored calls."""
    validate_declaration(declaration)
    found, scheduler, observed = {}, {}, set()
    for entry in accounting:
        index = _integer(entry["component_index"], "accounting component index")
        if index >= 44 or index in scheduler:
            raise ValueError("duplicate or undeclared scheduler component")
        for field in ("elapsed_seconds", "total_cpu_seconds", "peak_rss_bytes"):
            if field in entry and (not np.isfinite(entry[field]) or entry[field] < 0):
                raise ValueError("invalid scheduler timing/memory")
        scheduler[index] = entry
    for report in reports:
        component = report["component"]
        index = component["component_index"]
        if (index in found or not 0 <= index < 44 or component != declaration["components"][index]
                or report.get("declaration_sha256") != declaration["declaration_sha256"]
                or report.get("source_sha256") != declaration["source_sha256"]
                or report.get("runtime") != declaration["runtime"]):
            raise ValueError("duplicate component or task/source/runtime/declaration identity mismatch")
        allowed = {s["pcg64_initial_state_sha256"]: s for s in component["streams"]}
        for attempt in report["attempts"]:
            stream = attempt["stream"]
            identity = stream["pcg64_initial_state_sha256"]
            if identity in observed or allowed.get(identity) != stream:
                raise ValueError("repeated or undeclared actual block/rung stream")
            observed.add(identity)
        if report["status"] == "priced":
            count = len(declaration["task"]["context"]["contracts"])
            p, s, c = (np.asarray(report[k]) for k in ("prices", "standard_errors", "covariance"))
            if (p.shape != (count,) or s.shape != p.shape or c.shape != (count, count)
                    or not all(np.isfinite(a).all() for a in (p, s, c)) or np.any(s < 0)
                    or not np.allclose(np.diag(c), s*s, rtol=1e-12, atol=1e-20)):
                raise ValueError("invalid priced component arrays/covariance")
        elif report["status"] not in ("failed", "running", "interrupted"):
            raise ValueError("unknown component status")
        found[index] = report
    rows = []
    for component in declaration["components"]:
        index = component["component_index"]
        record = found.get(index)
        status = record["status"] if record else "missing"
        account = scheduler.get(index)
        # Missing/running alone cannot prove a timeout or a numerical failure.
        if status in ("missing", "running", "interrupted"):
            status = "censored" if account and str(account.get("state", "")).startswith("TIMEOUT") else "incomplete"
        row = {"component_index": index, "mode": component["mode"], "point_id": component["point_id"],
               "policy": component["policy"], "replicate": component["replicate"], "status": status,
               "raw_status": record["status"] if record else "missing", "scheduler": account,
               "wall_limit_seconds": component["resources"]["wall_limit_seconds"]}
        if record:
            for key in ("instrumented_cpu_seconds", "instrumented_wall_seconds", "summed_block_cpu_seconds",
                        "summed_block_wall_seconds", "process_peak_rss_bytes", "max_mc_se_over_quote_sigma",
                        "mc_target_met", "within_exception_cap", "paths_per_block", "error_type", "error"):
                if key in record:
                    row[key] = record[key]
            row["completed_attempt_cpu_seconds"] = sum(a.get("cpu_seconds", 0.) for a in record["attempts"])
        rows.append(row)
    sigma = tasks.sigma(declaration["task"])
    comparisons = []
    for mode, points, repetitions in (("adaptive", range(16), 1), ("fixed_anchor", ANCHOR_PATHS, 2)):
        for point in points:
            groups = [[found[c["component_index"]] for c in declaration["components"]
                       if c["mode"] == mode and c["point_id"] == point and c["policy"] == name
                       and c["component_index"] in found and found[c["component_index"]]["status"] == "priced"]
                      for name, _ in POLICIES]
            item = {"mode": mode, "point_id": point, "status": "incomplete"}
            if all(len(g) == repetitions for g in groups):
                item.update(status="reported", **numerical_pilot.comparison(*groups, sigma))
            comparisons.append(item)
    by_policy = {}
    for name, _ in POLICIES:
        by_policy[name] = {}
        for mode in ("adaptive", "fixed_anchor"):
            selected = [r for r in rows if r["policy"] == name and r["mode"] == mode]
            by_policy[name][mode] = {
                "declared_components": len(selected),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "priced_components_meeting_mc_target": sum(r.get("mc_target_met", False) for r in selected),
                "priced_components_above_target_within_exception": sum(
                    not r.get("mc_target_met", True) and r.get("within_exception_cap", False) for r in selected),
                "priced_components_above_exception_cap": sum(
                    r["status"] == "priced" and not r.get("within_exception_cap", True) for r in selected)}
    return {"format": FORMAT, "declaration_sha256": declaration["declaration_sha256"],
            "status": "completed" if all(r["status"] == "priced" for r in rows) else "incomplete",
            "declared_components": 44, "declared_independent_prior_points": 16,
            "components": rows, "comparisons": comparisons, "by_policy": by_policy,
            "seed_audit": {**declaration["seed_audit"], "observed_unique_block_rung_streams": len(observed)},
            "observed_instrumented_cpu_seconds": sum(r.get("instrumented_cpu_seconds", 0.) for r in rows),
            "scheduler_cpu_seconds_where_available": sum(r.get("total_cpu_seconds", 0.) for r in scheduler.values()),
            "cost_scope": "Completed timings are conditional on completion; missing/censored cost is not zero. "
                          "Scheduler accounting is separate, not added twice. No unconditional bank average inferred.",
            "scientific_acceptance": "not_assessed",
            "interpretation": "16 paired parameter points and 3 fixed anchors; independent price streams; "
                              "pointwise price intervals, no global or posterior/product numerical certificate."}

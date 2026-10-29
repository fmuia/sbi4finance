"""A frozen 52-call check of fixed-batch uncertainty and same-allocation timing.

Pooled estimates average independently fitted batches. They are not an OLS
refit at the pooled path count. No simulator, estimator or seed is replaced.
"""

import copy
import hashlib
import os
from pathlib import Path
import resource
import platform
import time

import numpy as np
from scipy.stats import t as student_t

from ..gbm_bank import _json, _write_json
from . import pricing, tasks
from . import rough_working_grid_pilot as base

FORMAT = "rough_control_diagnostic_v1"
ROOT_SEED = 271101
PATHS = 32768
BATCHES = 16
ANCHORS = (0, 4, 6)
REFERENCE = base.PACKAGE.parents[1] / "evidence/pilot-results/2026-10-27/rough-reviewed/declaration.json"


def _native(value):
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _design(reference, root):
    base.validate_declaration(reference)
    policies = reference["policies"]  # Authoritative Linux floats; no reconstruction.
    groups = [{"group_index": i, "role": "anchor_batches", "case_index": case,
               "resources": {"cpus": 1, "memory_gib": 8, "wall_limit_seconds": 1500}}
              for i, case in enumerate(ANCHORS)]
    groups.append({"group_index": 3, "role": "matched_timing", "case_index": None,
                   "resources": {"cpus": 1, "memory_gib": 8, "wall_limit_seconds": 1500}})
    calls = []

    def add(group, policy, parameters, batch):
        identity = {"format": FORMAT, "reference": reference["declaration_sha256"],
                    "group": group, "policy": policy, "batch": batch}
        digest = base._digest(identity)
        words = [int(digest[i:i+8], 16) for i in range(0, 64, 8)]
        seed = int(np.random.SeedSequence([0x52434447, root, *words])
                   .generate_state(1, dtype=np.uint64)[0])
        calls.append({"call_index": len(calls), "group_index": group, "policy": policy,
                      "batch": batch, "parameters": parameters, "total_paths": PATHS,
                      "pricing_seed": seed, "streams": [base._stream(seed, b, 0, PATHS) for b in range(3)],
                      "capture_raw_control": group != 3})

    for group, case in enumerate(ANCHORS):
        parameters = next(c["parameters"] for c in reference["components"]
                          if c["mode"] == "fixed_anchor" and c["point_id"] == case)
        for batch in range(BATCHES):
            add(group, "working", parameters, batch)
    slow = reference["prior_points"][6]["parameters"]
    for policy, batch in (("working", 0), ("refined", 0), ("refined", 1), ("working", 1)):
        add(3, policy, slow, batch)
    previous_states = {p["pcg64_initial_state_sha256"] for p in reference["prior_points"]}
    previous_seeds = {p["parameter_seed"] for p in reference["prior_points"]}
    for component in reference["components"]:
        previous_seeds.add(component["pricing_seed"])
        for stream in component["streams"]:
            previous_states.add(stream["pcg64_initial_state_sha256"])
            previous_seeds.add(stream["seed_sequence_state_uint64"][0])
    states = [s["pcg64_initial_state_sha256"] for c in calls for s in c["streams"]]
    seeds = [c["pricing_seed"] for c in calls] + [s["seed_sequence_state_uint64"][0] for c in calls for s in c["streams"]]
    if len(set(states)) != len(states) or set(states) & previous_states \
            or len(set(seeds)) != len(seeds) or set(seeds) & previous_seeds:
        raise ValueError("new or historical actual stream/seed collision; no redraw")
    return {"format": FORMAT, "root_seed": root, "reference": reference,
            "policies": policies, "groups": groups, "calls": calls,
            "seed_audit": {"passed": True, "new_pricing_streams": len(states),
                           "previous_initialized_pcg64_states": len(previous_states),
                           "new_reduced_uint64": len(seeds), "historical_reduced_uint64": len(previous_seeds),
                           "scope": "All possible block/rung streams and parameter RNG states of reviewed root271100."},
            "allocated_cpu_ceiling_seconds": 7200,
            "group_allocation_seconds": sum(g["resources"]["wall_limit_seconds"] for g in groups),
            "declaration_allocation_seconds": 300,
            "pooling": "Mean independent fixed-32768-path batch estimates; no pooled-path OLS refit.",
            "timing_order": ["working", "refined", "refined", "working"],
            "paths_unit": "Individual paths, not antithetic pairs, per independent maturity block.",
            "no_extra_pricing_warmup": True}


def _sources(reference):
    root = base.PACKAGE.parents[1]
    paths = {"calibration/rough_control_diagnostic.py": Path(__file__),
             "scripts/diagnose_rough_controls.py": root / "scripts/diagnose_rough_controls.py",
             "evidence/pilot-results/2026-10-27/rough-reviewed/declaration.json": REFERENCE}
    return {**base._sources(reference["task"]),
            **{k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}}


def declare(reference, *, source_revision, root_seed=ROOT_SEED):
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("frozen source revision is required")
    declaration = _design(copy.deepcopy(reference), root_seed)
    declaration.update(source_revision=source_revision, source_sha256=_sources(reference), runtime=base._runtime())
    declaration["declaration_sha256"] = base._digest(declaration)
    return declaration


def validate(declaration, *, execution=False):
    body = dict(declaration)
    if body.pop("declaration_sha256", None) != base._digest(body):
        raise ValueError("diagnostic declaration digest mismatch")
    expected = _design(body["reference"], body["root_seed"])
    if any(body.get(k) != v for k, v in expected.items()):
        raise ValueError("diagnostic design/stream identity mismatch")
    if execution and (body["source_sha256"] != _sources(body["reference"]) or body["runtime"] != base._runtime()):
        raise ValueError("diagnostic source/runtime differs from declaration")


def _fixed_call(task, call, row, progress):
    """The original block estimator once per block; no paths or controls reused."""
    curve = tasks.joint_curve(task, call["parameters"])
    parameters = tasks.model_parameters(task, call["parameters"])
    contracts = task["context"]["contracts"]
    maturities = np.array([c["maturity"] for c in contracts])
    assignment = np.full(len(contracts), -1)
    for b, block in reversed(list(enumerate(task["numerics"]["blocks"]))):
        assignment[maturities <= block["horizon"] + 1e-12] = b
    count = len(contracts)
    output = {"controlled_prices": np.empty(count), "controlled_covariance": np.zeros((count, count))}
    if call["capture_raw_control"]:
        output.update(raw_prices=np.empty(count), raw_covariance=np.zeros((count, count)),
                      paired_difference=np.empty(count), paired_covariance=np.zeros((count, count)))
    for b, block in enumerate(task["numerics"]["blocks"]):
        chosen = np.flatnonzero(assignment == b)
        stream = call["streams"][b]
        seed = np.random.SeedSequence(stream["seed_sequence_entropy"])
        if base._rng_digest(seed) != stream["pcg64_initial_state_sha256"]:
            raise ValueError("actual initialized block stream differs")
        attempt = {"block": b, "contract_indices": chosen.tolist(), "stream": stream, "status": "running"}
        row["blocks"].append(attempt)
        progress()
        details = {"capture_raw_control": True} if call["capture_raw_control"] else None
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            prices, errors, covariance = pricing.rough_block(parameters, curve, [contracts[i] for i in chosen],
                horizon=block["horizon"], n_steps=block["n_steps"], n_pairs=call["total_paths"] // 2,
                seed=seed, diagnostics=details)
        except BaseException as error:
            elapsed_cpu, elapsed_wall = time.process_time()-cpu, time.perf_counter()-wall
            attempt.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                           error_type=type(error).__name__, error=str(error))
            raise
        else:
            elapsed_cpu, elapsed_wall = time.process_time()-cpu, time.perf_counter()-wall
            if not all(np.isfinite(a).all() for a in (prices, errors, covariance)):
                attempt.update(status="failed", error="nonfinite default estimator result")
                raise FloatingPointError(attempt["error"])
            attempt.update(status="priced", prices=prices.tolist(), standard_errors=errors.tolist(), covariance=covariance.tolist())
            output["controlled_prices"][chosen] = prices
            output["controlled_covariance"][np.ix_(chosen, chosen)] = covariance
            if details is not None:
                diagnostic = details["raw_control"]
                native_diagnostic = _native(diagnostic)
                try:
                    _json(native_diagnostic)
                except ValueError as error:
                    attempt.update(status="failed", error="nonfinite optional raw/control diagnostic; "
                                   "default estimator return retained above")
                    raise FloatingPointError(attempt["error"]) from error
                parity = np.array([1.-contracts[i]["strike"] if contracts[i]["option_type"] == "call" else 0. for i in chosen])
                output["raw_prices"][chosen] = diagnostic["raw_put_mean"] + parity
                output["raw_covariance"][np.ix_(chosen, chosen)] = diagnostic["raw_put_covariance_of_mean"]
                output["paired_difference"][chosen] = diagnostic["paired_controlled_minus_raw_mean"]
                output["paired_covariance"][np.ix_(chosen, chosen)] = diagnostic["paired_controlled_minus_raw_covariance_of_mean"]
                attempt["diagnostics"] = native_diagnostic
        finally:
            attempt.update(cpu_seconds=elapsed_cpu, wall_seconds=elapsed_wall)
            progress()
    if not all(np.isfinite(a).all() for a in output.values()):
        raise FloatingPointError("nonfinite fixed-batch price or diagnostic covariance")
    row.update(_native(output))


def run_group(declaration, group_index, output):
    validate(declaration, execution=True)
    if type(group_index) is not int or not 0 <= group_index < len(declaration["groups"]):
        raise ValueError("declared group index required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    group = declaration["groups"][group_index]
    report = {"format": FORMAT, "declaration_sha256": declaration["declaration_sha256"], "group": group,
              "source_sha256": declaration["source_sha256"], "runtime": declaration["runtime"],
              "started_utc": base._utc(), "status": "running", "runs": [],
              "timing_scope": "Call timers include curve/setup/journaling; block timers end on estimator return. "
                              "Raw/control diagnostics enabled for anchor batches only, disabled for ABBA timing.",
              "scheduler_environment": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST")}}
    def progress():
        _write_json(output / "report.json", report)
    progress()
    for call in (c for c in declaration["calls"] if c["group_index"] == group_index):
        row = {"call": call, "status": "running", "blocks": [], "started_utc": base._utc(),
               "first_call_in_allocation": not report["runs"]}
        report["runs"].append(row)
        progress()
        wall, cpu = time.perf_counter(), time.process_time()
        interrupted = False
        try:
            task = next(p["task"] for p in declaration["policies"] if p["name"] == call["policy"])
            _fixed_call(task, call, row, progress)
            row["status"] = "priced"
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            row.update(status="interrupted" if interrupted else "failed", error_type=type(error).__name__, error=str(error))
        finally:
            row.update(cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall, finished_utc=base._utc())
            progress()
        if interrupted:
            break
    expected = sum(c["group_index"] == group_index for c in declaration["calls"])
    report["status"] = "completed" if len(report["runs"]) == expected and all(r["status"] == "priced" for r in report["runs"]) else "incomplete"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_peak_rss_bytes"] = int(rss if platform.system() == "Darwin" else rss*1024)
    report["finished_utc"] = base._utc()
    progress()
    return report


def batch_summary(rows):
    """Pure analysis of ordered independent batches; no simulation or OLS refit."""
    count = len(rows)
    if count < 2:
        raise ValueError("at least two independent priced batches required")
    result = {}
    for name, price_key, cov_key in (("controlled", "controlled_prices", "controlled_covariance"),
                                     ("raw", "raw_prices", "raw_covariance"),
                                     ("paired_controlled_minus_raw", "paired_difference", "paired_covariance")):
        values = np.array([r[price_key] for r in rows]);covariances = np.array([r[cov_key] for r in rows])
        mean = values.mean(axis=0)
        between = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        nominal = covariances.sum(axis=0) / count**2
        empirical = between / count
        nominal_se = np.sqrt(np.maximum(np.diag(nominal), 0.))
        empirical_se = np.sqrt(np.maximum(np.diag(empirical), 0.))
        ratio = np.divide(empirical_se, nominal_se, out=np.full_like(empirical_se, np.nan), where=nominal_se > 0)
        # Undefined ratios remain explicit null, never a fabricated zero/pass.
        ratio = [float(x) if np.isfinite(x) else None for x in ratio]
        half = student_t.ppf(.975, count-1)*empirical_se
        item = {"mean": mean.tolist(), "nominal_covariance_of_pooled_mean": nominal.tolist(),
                "between_batch_covariance_of_pooled_mean": empirical.tolist(),
                "between_to_nominal_se_ratio": ratio, "descriptive_t95_lower": (mean-half).tolist(),
                "descriptive_t95_upper": (mean+half).tolist(), "pooling": []}
        for size in (1,2,4,8,16):
            if count % size:
                continue
            pools = values.reshape(count//size, size, -1).mean(axis=1)
            pooled_cov = covariances.reshape(count//size, size, *covariances.shape[1:]).sum(axis=1)/size**2
            item["pooling"].append({"batches_per_pool": size, "independent_pools": len(pools),
                "paths_per_block_per_pool": size*PATHS,
                "rms_nominal_se": np.sqrt(np.mean(np.diagonal(pooled_cov,axis1=1,axis2=2),axis=0)).tolist(),
                "empirical_sd_across_pools": None if len(pools)<2 else pools.std(axis=0,ddof=1).tolist()})
        result[name] = item
    result["scope"] = "Heuristic descriptive t intervals from independent batches; pointwise coverage is not verified. Sixteen batches do not certify a CLT or global coverage, continuum truth, or a pooled-path OLS refit. Paired differences compare the fitted-control estimator to bounded raw puts on identical paths."
    return result


def reduce_groups(declaration, reports, *, accounting=()):
    validate(declaration)
    found, calls, streams = set(), {}, set()
    for report in reports:
        index = report["group"]["group_index"]
        if index in found or not 0 <= index < 4 or report["group"] != declaration["groups"][index] \
                or any(report.get(k) != declaration[k] for k in ("declaration_sha256", "source_sha256", "runtime")):
            raise ValueError("duplicate group or source/runtime/declaration identity mismatch")
        found.add(index)
        for row in report["runs"]:
            call = row["call"];i = call["call_index"]
            if i in calls or call != declaration["calls"][i] or call["group_index"] != index:
                raise ValueError("duplicate or undeclared call")
            for block in row["blocks"]:
                stream = block["stream"];fingerprint = stream["pcg64_initial_state_sha256"]
                if fingerprint in streams or stream not in call["streams"]:
                    raise ValueError("duplicate or undeclared actual stream")
                streams.add(fingerprint)
            if row["status"]=="priced":
                count=len(declaration["reference"]["task"]["context"]["contracts"])
                keys=[("controlled_prices",(count,)),("controlled_covariance",(count,count))]
                if call["capture_raw_control"]:
                    keys += [("raw_prices",(count,)),("raw_covariance",(count,count)),
                             ("paired_difference",(count,)),("paired_covariance",(count,count))]
                if len(row["blocks"])!=3 or any(b["status"]!="priced" for b in row["blocks"]) \
                        or any(np.shape(row.get(k))!=shape or not np.isfinite(row[k]).all() for k,shape in keys):
                    raise ValueError("incomplete or nonfinite priced diagnostic arrays")
            calls[i] = row
    anchors=[]
    for group,case in enumerate(ANCHORS):
        expected=[c for c in declaration["calls"] if c["group_index"]==group]
        rows=[calls.get(c["call_index"]) for c in expected]
        item={"case_index":case,"declared_batches":BATCHES,"priced_batches":sum(r is not None and r["status"]=="priced" for r in rows)}
        item["status"]="completed" if item["priced_batches"]==BATCHES else "incomplete"
        if item["status"]=="completed":item["summary"]=batch_summary(rows)
        anchors.append(item)
    timing=[calls.get(c["call_index"]) for c in declaration["calls"] if c["group_index"]==3]
    group_statuses=[];account={}
    for record in accounting:
        index=record["group_index"]
        if index in account or type(index) is not int or not 0<=index<4:
            raise ValueError("duplicate or undeclared accounting group")
        account[index]=record
    for group in declaration["groups"]:
        index=group["group_index"]
        selected=[c for c in declaration["calls"] if c["group_index"]==index]
        complete=all(c["call_index"] in calls and calls[c["call_index"]]["status"]=="priced" for c in selected)
        status="completed" if complete else "censored" if str(account.get(index,{}).get("state","")).startswith("TIMEOUT") else "incomplete"
        group_statuses.append({"group_index":index,"status":status,"scheduler_accounting":account.get(index)})
    return {"format":FORMAT,"declaration_sha256":declaration["declaration_sha256"],"anchors":anchors,
            "declared_calls":len(declaration["calls"]),"reported_calls":len(calls),
            "call_statuses":[{"call_index":c["call_index"],"status":calls.get(c["call_index"],{}).get("status","missing")} for c in declaration["calls"]],
            "seed_audit":{**declaration["seed_audit"],"observed_unique_streams":len(streams)},
            "timing_runs":timing,"group_statuses":group_statuses,"scientific_acceptance":"not_assessed",
            "status":"completed" if len(calls)==len(declaration["calls"]) and all(r["status"]=="priced" for r in calls.values()) else "incomplete"}

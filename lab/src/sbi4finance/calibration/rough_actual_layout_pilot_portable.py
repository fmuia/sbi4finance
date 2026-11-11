"""Portable orchestration for the immutable actual-layout pilot panel.

Only derived known-curve validation permits declared floating-point roundoff.
Every price uses the original stored case values, and all numerical kernels,
financial priors and task helpers remain owned by their canonical sources.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import time
import numpy as np
from . import rough_actual_layout_pilot as original

FORMAT = "rough_actual_layout_portable_pilot_work_order_v1"
CONDITION_FORMAT = "rough_actual_layout_portable_heston_condition_v1"
AUTHORITATIVE_PANEL_SHA256 = 'cbb2c2c31890c6a6ffb45eedf255cb0d7e1ff223c6801c918d11eff7d8fe19c6'
AUTHORITATIVE_TASKS_SHA256 = '480047c1cc89dd1666c31a9cbe76f070cfdce02f0d681ad29717aa30919e8057'
AUTHORITATIVE_HESTON_TARGET_SHA256 = '6e2c4154e4d2003e9d9dc386543020048af330455e4db1be15d3de1571f886d9'
ROUNDING_POLICY = {"known_derived_relative_tolerance": 32*np.finfo(float).eps,
    "known_derived_absolute_tolerance": 0., "fields": ["supplied_levels", "original_curve_draw_context"],
    "execution_values": "original authoritative stored panel, never recomputed replacements"}
ROOT, LIMITS = original.ROOT, original.LIMITS
actual_layout, runtime_pricing = original.actual_layout, original.runtime_pricing
read, binding, digest, write_new = original.read, original.binding, original.digest, original.write_new
curve_for_case, checked_covariance = original.curve_for_case, original.checked_covariance
load_controlled_helper, heston_readiness = original.load_controlled_helper, original.heston_readiness


def source_bindings():
    result = original.source_bindings()
    for name, path in {"portable_pilot": Path(__file__),
            "portable_prepare_cli": ROOT/"lab/scripts/prepare_rough_actual_layout_pilot_portable.py",
            "portable_worker_cli": ROOT/"lab/scripts/run_rough_actual_layout_pilot_portable.py"}.items():
        result[name] = {"relative_path": str(path.resolve().relative_to(ROOT)), "sha256": digest(path)}
    return result


def panel_fingerprint(cases): return actual_layout.fingerprint(cases)


def check_panel(cases, expected):
    if len(cases) != len(expected): raise ValueError("authoritative panel case count changed")
    for stored, regenerated in zip(cases, expected, strict=True):
        allowed = set(ROUNDING_POLICY["fields"]) if stored.get("curve_role") == "supplied_known" else set()
        if {k:v for k,v in stored.items() if k not in allowed} != {k:v for k,v in regenerated.items() if k not in allowed}:
            raise ValueError("exact non-derived physical case, prior role or seed changed")
        for field in allowed:
            a,b = np.asarray(stored.get(field), float), np.asarray(regenerated[field], float)
            if (a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
                    or (field == "supplied_levels" and np.any(a <= 0))
                    or not np.allclose(a,b,rtol=ROUNDING_POLICY["known_derived_relative_tolerance"],atol=0)):
                raise ValueError("known derived curve values exceed declared32eps relative roundoff")


def prepare(parent_work_order, out):
    """Version metadata only; the authoritative existing panel is copied exactly."""
    out = Path(out)
    if out.exists(): raise FileExistsError("new portable preparation identity required")
    parent = read(parent_work_order); original.validate(parent)
    result = deepcopy(parent)
    result.update(format=FORMAT,created_utc=datetime.now(timezone.utc).isoformat(),
        parent_work_order=binding(parent_work_order),authoritative_panel_sha256=panel_fingerprint(parent["cases"]),
        validation_roundoff_policy=deepcopy(ROUNDING_POLICY),source_bindings=source_bindings(),
        sbatch_sha256=hashlib.sha256(sbatch_text().encode()).hexdigest(),
        status="prepared_only_no_MC_executed")
    if result["cases"] != parent["cases"]: raise AssertionError("stored authoritative panel changed")
    validate(result)
    out.mkdir(parents=True); write_new(out/"work-order.json",result)
    (out/"pilot.sbatch").write_text(sbatch_text())
    return result


def validate(work, repo_root=None):
    if (work.get("format") != FORMAT or work.get("limits") != LIMITS
            or work.get("validation_roundoff_policy") != ROUNDING_POLICY
            or work.get("MC_executed") is not False or work.get("reserved_arrays_opened") is not False
            or work.get("grid_estimator_evaluations") != 80
            or work.get("parent_work_order",{}).get("sha256") != "b9bf6e3fa2453b808589cd6e3ad6549789106fbae3bd57f6a80def5d13ead830"
            or work.get("authoritative_panel_sha256") != AUTHORITATIVE_PANEL_SHA256
            or panel_fingerprint(work.get("cases",[])) != AUTHORITATIVE_PANEL_SHA256
            or actual_layout.fingerprint(work.get("tasks",{})) != AUTHORITATIVE_TASKS_SHA256
            or actual_layout.fingerprint(work.get("heston_target",{})) != AUTHORITATIVE_HESTON_TARGET_SHA256):
        raise ValueError("source-bound original-panel portable work order required")
    repo_root = Path(repo_root or ROOT).resolve()
    if set(work["source_bindings"]) != set(source_bindings()): raise ValueError("complete original and portable source bindings required")
    for spec in work["source_bindings"].values():
        if digest(repo_root/spec["relative_path"]) != spec["sha256"]: raise ValueError("actual execution source differs: "+spec["relative_path"])
    if work.get("sbatch_sha256") != hashlib.sha256(sbatch_text().encode()).hexdigest(): raise ValueError("portable submission changed")
    if {name:metadata.version(name) for name in ("numpy","scipy")} != work["execution_versions"]: raise ValueError("execution numerical versions differ")
    for task in work["tasks"].values():
        actual_layout.validate_task(task)
        if [b["n_steps"] for b in task["blocks"]] != original.measured_base_steps(task["base_task"],task["design"]):
            raise ValueError("declared initial time-step recipe changed")
    if (work.get("components") != original.declared_components() or work.get("initial_case_indices") != [0,1,2,12]
            or work.get("static_memory_audit") != original.static_memory_audit(work["tasks"])):
        raise ValueError("declared paths, grids, streams or memory recipe changed")
    expected = original.panel_cases(work["tasks"]["supplied_known"]["base_task"],work["tasks"]["joint_unknown"]["base_task"])
    check_panel(work["cases"],expected)
    return repo_root


def sbatch_text():
    return original.sbatch_text().replace("run_rough_actual_layout_pilot.py", "run_rough_actual_layout_pilot_portable.py")


def bind_heston(work_order_path, evaluation_path, out):
    work = read(work_order_path); validate(work); result = read(evaluation_path)
    checks = heston_readiness(result, work["heston_target"])
    record = {"format": CONDITION_FORMAT, "work_order_sha256": digest(work_order_path),
        "heston_evaluation": binding(evaluation_path), "heston_target": result["task"], "working_grid_condition_met": True,
        "necessary_diagnostics": checks, "scientific_acceptance": False,
        "scope": "bounded rough numerical initiation only; no estimator promotion or production-bank allocation"}
    write_new(out, record); return record


def validate_condition(work, work_order_path, condition, evaluation_path=None):
    if (condition.get("format") != CONDITION_FORMAT or condition.get("working_grid_condition_met") is not True
            or condition.get("work_order_sha256") != digest(work_order_path) or condition.get("heston_target") != work["heston_target"]):
        raise ValueError("completed matching Heston initiation condition required")
    spec = condition.get("heston_evaluation", {})
    path = Path(evaluation_path or spec.get("path", ""))
    if not path.is_file() or digest(path) != spec.get("sha256"):
        raise ValueError("bound Heston evaluation changed")
    checks = heston_readiness(read(path), work["heston_target"])
    if checks != condition.get("necessary_diagnostics"):
        raise ValueError("Heston initiation diagnostics changed")


def run_component(work_order_path, condition_path, index, out, *, process_started, heston_evaluation_path=None):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("one allocated Slurm CPU required; preparation is not permission for local MC")
    work = read(work_order_path); repo_root = validate(work); condition = read(condition_path)
    validate_condition(work, work_order_path, condition, heston_evaluation_path)
    if type(index) is not int or not 0 <= index < 48: raise ValueError("one declared component index required")
    component = work["components"][index]; case = work["cases"][component["case_index"]]; task = work["tasks"][case["curve_role"]]
    out = Path(out)
    if out.exists(): raise FileExistsError("component identity is immutable; no retry")
    out.mkdir(parents=True)
    from scripts.run_heston_spread_reference import Budget
    helper = load_controlled_helper(repo_root)
    curve = curve_for_case(work, case); parameters = dict(zip(["hurst", "eta", "rho"], case["parameters"][:3], strict=True))
    record = {"format": "rough_actual_layout_portable_component_v1", "status": "running", "component": component,
        "case": case, "work_order_sha256": digest(work_order_path), "condition": binding(condition_path),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "scientific_acceptance": False, "reserved_arrays_opened": False,
        "curve_role": case["curve_role"], "blocks_completed": 0, "block_payloads": [],
        "covariance_policy": "finite;32eps relative symmetry; strict nonnegative diagonal; no negative-variance clipping",
        "discrete_curve_controls": {str(g): actual_layout.discrete_curve_audit(task, curve, g) for g in component["multipliers"]}}
    write_new(out/"run.json", record); n = 63; paired = len(component["multipliers"]) == 2
    controlled = np.zeros((2 if paired else 1, n)); covariance = np.zeros((4 if paired else 1, n, 4 if paired else 1, n))
    bank_covariance = np.zeros((len(controlled), n, n))
    raw = np.zeros((2, n)) if paired else None; difference = np.zeros(n) if paired else None; difference_covariance = np.zeros((n, n)) if paired else None
    block_diagnostics = []; started_cpu = time.process_time()
    remaining = LIMITS["component_total_wall_seconds"]-(time.monotonic()-process_started)-2
    try:
        if remaining <= 0: raise TimeoutError("component setup exhausted its total wall cap")
        budget = Budget(remaining)
        with budget:
            for block_index, block in enumerate(task["blocks"]):
                budget.check(); chosen = [i for i, c in enumerate(task["design"]["contracts"]) if c["maturity"] == block["horizon"]]
                contracts = [task["design"]["contracts"][i] for i in chosen]; seed = component["streams"][block_index]["seed"]
                if paired:
                    result = helper.coupled_controlled_block(parameters, curve, contracts, horizon=block["horizon"],
                        steps=tuple(block["n_steps"]*g for g in component["multipliers"]), n_pairs=component["paths"]//2, seed=seed)
                    for moment in result["moments"]: checked_covariance(moment["bank_covariance"], len(chosen))
                    checked_covariance(np.asarray(result["joint_covariance"]).reshape(4*len(chosen), 4*len(chosen)), 4*len(chosen))
                    checked_covariance(result["difference_covariance"], len(chosen))
                    if not all(np.isfinite(result[key]).all() for key in ("controlled_prices", "raw_prices", "difference")):
                        raise ValueError("nonfinite block prices")
                    block_payload = {key: np.asarray(value) for key, value in result.items() if key != "moments"}
                    block_payload["bank_covariance"] = np.asarray([moment["bank_covariance"] for moment in result["moments"]])
                    controlled[:, chosen] = result["controlled_prices"]; raw[:, chosen] = result["raw_prices"]
                    difference[chosen] = result["difference"]; difference_covariance[np.ix_(chosen, chosen)] = result["difference_covariance"]
                    for g in range(2): bank_covariance[g][np.ix_(chosen, chosen)] = np.asarray(result["moments"][g]["bank_covariance"])
                    for a in range(4):
                        for b in range(4): covariance[a, np.asarray(chosen)[:, None], b, np.asarray(chosen)[None, :]] = result["joint_covariance"][a, :, b, :]
                    block_diagnostics.append({"block": block_index, "moments": result["moments"]})
                else:
                    diagnostic = {"capture_raw_control": True}
                    prices, errors, cov = runtime_pricing.rough_block(parameters, curve, contracts, horizon=block["horizon"],
                        n_steps=block["n_steps"]*4, n_pairs=component["paths"]//2, seed=seed, diagnostics=diagnostic)
                    checked_covariance(cov, len(chosen))
                    if not np.isfinite(prices).all() or not np.isfinite(errors).all(): raise ValueError("nonfinite replay prices or errors")
                    block_payload = {"controlled_prices": prices, "standard_errors": errors, "bank_covariance": cov}
                    bank_covariance[0][np.ix_(chosen, chosen)] = cov
                    controlled[0, chosen] = prices; covariance[0, np.asarray(chosen)[:, None], 0, np.asarray(chosen)[None, :]] = cov
                    block_diagnostics.append({"block": block_index, "standard_errors": errors.tolist()})
                block_path = out/f"block-{block_index:02d}.npz"
                with block_path.open("xb") as stream: np.savez_compressed(stream, **block_payload)
                record["block_payloads"].append({"block": block_index, "contracts": chosen, **binding(block_path)})
                record["blocks_completed"] += 1
                (out/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
                (out/"progress.json").write_text(json.dumps({"blocks_completed": record["blocks_completed"], **budget.usage()})+"\n")
            budget.check()
        factors = np.asarray(task["design"]["price_factor_DF"]); strict = .5*np.asarray(task["design"]["half_spread_forward"])
        payload = {"controlled_forward": controlled, "controlled_raw_units": controlled*factors,
            "bank_covariance_nominal_forward": bank_covariance, "bank_covariance_nominal_raw_units": actual_layout.covariance_to_raw(task, bank_covariance),
            "joint_covariance_forward": covariance, "joint_covariance_raw_units": covariance*factors[None, :, None, None]*factors[None, None, None, :],
            "curve_nodes": np.asarray(curve["times"]), "curve_values": np.asarray(curve["values"])}
        if paired: payload.update(raw_forward=raw, difference_forward=difference, difference_covariance_forward=difference_covariance)
        with (out/"results.npz").open("xb") as stream: np.savez_compressed(stream, **payload)
        ratios = [float(np.max(np.sqrt(np.diag(bank_covariance[g]))/strict)) for g in range(len(controlled))]
        record.update(status="complete", block_diagnostics=block_diagnostics, max_mcse_over_strict_sd=ratios,
            strict_mc_target_met=all(value <= .25 for value in ratios), numerical_target_established=False,
            payload=binding(out/"results.npz"), total_wall_seconds=time.monotonic()-process_started,
            pricing_cpu_seconds=time.process_time()-started_cpu, process_cpu_seconds=time.process_time(),
            allocated_cpu_budget_seconds=600, source_bindings=work["source_bindings"])
        (out/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), total_wall_seconds=time.monotonic()-process_started)
        (out/"failure.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        (out/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n"); raise

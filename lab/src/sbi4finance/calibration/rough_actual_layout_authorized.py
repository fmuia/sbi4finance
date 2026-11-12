"""Explicitly authorized rough pilot execution with unchanged financial kernels.

The user's11November instruction permits parallel rough simulations despite
unmet Heston calibration screens. This execution policy supersedes only that
administrative initiation condition and the old two-worker scheduling limit;
it never declares Heston or rough numerical/scientific acceptance.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from . import rough_actual_layout_pilot_portable as prepared

ROOT, LIMITS = prepared.ROOT, prepared.LIMITS
read, binding, digest, write_new = prepared.read, prepared.binding, prepared.digest, prepared.write_new
validate, curve_for_case = prepared.validate, prepared.curve_for_case
actual_layout, runtime_pricing = prepared.actual_layout, prepared.runtime_pricing
checked_covariance, load_controlled_helper = prepared.checked_covariance, prepared.load_controlled_helper
FORMAT = "rough_actual_layout_direct_authorization_v1"
USER_INSTRUCTION = "Ok let us start the Bergomi simulation runs here. Let's parallelize this properly on oscar so we can make progress on this. I think the Bergomi simulations are the most expensive piece, so we should not delay starting those."
POLICY = {"components":48,"maximum_concurrent_workers":16,"cpus_per_component":1,
    "memory_gib_per_component":32,"component_total_wall_seconds":600,
    "maximum_allocated_cpu_seconds":28800,"maximum_peak_allocated_memory_gib":512,
    "automatic_retry":False,"financial_or_numerical_recipe_changes":False}
WORK_SHA = "38455e9565681ea502cac33116f92c27f2d3396eb3ff06c0248b251e9f9edb45"


def source_bindings():
    result=prepared.source_bindings()
    for name,path in {"authorized_pilot":Path(__file__),"authorized_cli":ROOT/"lab/scripts/run_rough_actual_layout_authorized.py"}.items():
        result[name]={"relative_path":str(path.resolve().relative_to(ROOT)),"sha256":digest(path)}
    return result


def validate_authorization(work,work_order_path,authorization,repo_root=None):
    validate(work,repo_root)
    if (authorization.get("format")!=FORMAT or authorization.get("user_instruction")!=USER_INSTRUCTION
            or authorization.get("instruction_sha256")!=hashlib.sha256(USER_INSTRUCTION.encode()).hexdigest()
            or authorization.get("work_order_sha256")!=digest(work_order_path)
            or authorization.get("work_order_sha256")!=WORK_SHA
            or authorization.get("execution_policy")!=POLICY
            or authorization.get("explicit_user_authorization") is not True
            or authorization.get("heston_diagnostics_passed") is not False
            or authorization.get("supersedes_heston_success_condition") is not True
            or authorization.get("scientific_acceptance") is not False
            or authorization.get("authoritative_panel_sha256")!=prepared.AUTHORITATIVE_PANEL_SHA256
            or authorization.get("numerical_target_established") is not False):
        raise ValueError("explicit source-bound rough execution authorization required")
    expected=source_bindings()
    if authorization.get("source_bindings")!=expected:raise ValueError("authorized execution sources differ")
    root=Path(repo_root or ROOT)
    for spec in expected.values():
        if digest(root/spec["relative_path"])!=spec["sha256"]:raise ValueError("execution source identity differs")
    if authorization.get("sbatch_sha256")!=hashlib.sha256(sbatch_text().encode()).hexdigest():
        raise ValueError("authorized scheduler declaration differs")
    return root


def prepare(work_order_path,out):
    work=read(work_order_path);validate(work)
    if digest(work_order_path)!=WORK_SHA:raise ValueError("unchanged authoritative portable work order required")
    out=Path(out)
    if out.exists():raise FileExistsError("new authorization identity required")
    record={"format":FORMAT,"created_utc":datetime.now(timezone.utc).isoformat(),
        "user_instruction":USER_INSTRUCTION,"instruction_sha256":hashlib.sha256(USER_INSTRUCTION.encode()).hexdigest(),
        "explicit_user_authorization":True,"work_order_sha256":digest(work_order_path),
        "work_order":binding(work_order_path),"authoritative_panel_sha256":work["authoritative_panel_sha256"],
        "execution_policy":POLICY,"source_bindings":source_bindings(),
        "supersedes_heston_success_condition":True,"heston_diagnostics_passed":False,
        "heston_status":"completed Heston runs retain calibration errors; permission to compute is separate from accuracy",
        "scheduling_change":{"original_maximum_concurrent_workers":2,"authorized_maximum_concurrent_workers":16,
            "total_allocated_CPU_ceiling_unchanged":True,"financial_cases_grids_paths_streams_unchanged":True},
        "sbatch_sha256":hashlib.sha256(sbatch_text().encode()).hexdigest(),
        "numerical_target_established":False,"scientific_acceptance":False,"pricing_executed_during_preparation":False}
    validate_authorization(work,work_order_path,record)
    out.mkdir(parents=True);write_new(out/"authorization.json",record)
    (out/"pilot.sbatch").write_text(sbatch_text())
    return record


def sbatch_text():
    return r'''#!/bin/bash
#SBATCH --job-name=rough-actual-A1
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --array=0-47%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
set -euo pipefail
: "${ROUGH_REPO_ROOT:?stage frozen complete source checkout}"
: "${ROUGH_PYTHON:?set verified execution-host Python}"
: "${ROUGH_WORK_ORDER:?set unchanged portable work-order.json}"
: "${ROUGH_AUTHORIZATION:?set explicit execution authorization.json}"
: "${ROUGH_PAYLOAD_ROOT:?set NEW output root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rough-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/run_rough_actual_layout_authorized.py" run --work-order "$ROUGH_WORK_ORDER" --authorization "$ROUGH_AUTHORIZATION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT/component-$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
'''


def run_component(work_order_path, authorization_path, index, out, *, process_started, heston_evaluation_path=None):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("one allocated Slurm CPU required; preparation is not permission for local MC")
    work = read(work_order_path); repo_root = validate(work); authorization = read(authorization_path)
    validate_authorization(work, work_order_path, authorization)
    if type(index) is not int or not 0 <= index < 48: raise ValueError("one declared component index required")
    component = work["components"][index]; case = work["cases"][component["case_index"]]; task = work["tasks"][case["curve_role"]]
    out = Path(out)
    if out.exists(): raise FileExistsError("component identity is immutable; no retry")
    out.mkdir(parents=True)
    from scripts.run_heston_spread_reference import Budget
    helper = load_controlled_helper(repo_root)
    curve = curve_for_case(work, case); parameters = dict(zip(["hurst", "eta", "rho"], case["parameters"][:3], strict=True))
    record = {"format": "rough_actual_layout_authorized_component_v1", "status": "running", "component": component,
        "case": case, "work_order_sha256": digest(work_order_path), "execution_authorization": binding(authorization_path),
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
            allocated_cpu_budget_seconds=600, source_bindings=authorization["source_bindings"])
        (out/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), total_wall_seconds=time.monotonic()-process_started)
        (out/"failure.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        (out/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n"); raise

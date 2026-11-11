"""Preparation and execution interface for a conditional rough actual-layout pilot.

Preparation is metadata/curve construction only. Execution is separately gated
by a completed source-bound Heston grid result and a Slurm CPU allocation.
The old tested controlled-CRN helper and canonical hybrid/put estimator supply
all Monte Carlo pricing. No existing task, curve prior or worker is modified.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

from sbi4f_tutorial.rough_options import actual_layout, model, simulator
from sbi4f_tutorial.shared import pricing as runtime_pricing, tasks as runtime_tasks
from . import bank, pricing, tasks

FORMAT = "rough_actual_layout_pilot_work_order_v1"
ROOT = Path(__file__).resolve().parents[4]
HELPER = Path("lab/evidence/pilot-results/2026-10-28/known-pricing-validation/known_rough_pricing_check_20261028.py")
NAMESPACE, ROOT_SEED = 0x52414C31, 20261110
LIMITS = {"allocated_cpu_seconds": 28800, "maximum_concurrent_workers": 2,
    "components": 48, "component_total_wall_seconds": 600, "cpus_per_component": 1,
    "memory_gib_per_component": 32, "gpus": 0, "automatic_retry": False, "automatic_path_escalation": False}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def binding(path): return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def write_new(path, value):
    with Path(path).open("x") as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def source_bindings():
    files = {"actual_layout": Path(actual_layout.__file__), "rough_model": Path(model.__file__),
        "rough_simulator": Path(simulator.__file__), "runtime_pricing": Path(runtime_pricing.__file__),
        "runtime_tasks": Path(runtime_tasks.__file__), "lab_curve_helpers": Path(pricing.__file__),
        "lab_task_helpers": Path(tasks.__file__), "pilot": Path(__file__), "controlled_crn_helper": ROOT/HELPER,
        "rough_compatibility": ROOT/"lab/src/sbi4finance/rough_bergomi.py",
        "bergomi_compatibility": ROOT/"lab/src/sbi4finance/bergomi.py",
        "grid_coupling": ROOT/"lab/src/sbi4finance/bergomi_refinement.py",
        "budget_helper": ROOT/"lab/scripts/run_heston_spread_reference.py",
        "prepare_cli": ROOT/"lab/scripts/prepare_rough_actual_layout_pilot.py",
        "worker_cli": ROOT/"lab/scripts/run_rough_actual_layout_pilot.py",
        "legacy_bank_identity": ROOT/"lab/src/sbi4finance/calibration/bank.py",
        "legacy_json_identity": ROOT/"tutorial/src/sbi4f_tutorial/shared/bank_helpers.py",
        "heston_readiness": ROOT/"lab/scripts/render_heston_pr87_100k.py",
        "heston_grid_report": ROOT/"lab/scripts/render_heston_pr87_grid.py",
        "heston_followup_report": ROOT/"lab/scripts/render_heston_pr87_followup.py"}
    return {name: {"relative_path": str(path.resolve().relative_to(ROOT)), "sha256": digest(path)} for name, path in files.items()}


def measured_base_steps(base, design):
    """Initial work grid keeps or improves the old dt; this is not accuracy approval."""
    found = []
    for maturity in design["actual_maturities"]:
        block = next((b for b in base["numerics"]["blocks"] if maturity <= b["horizon"]), None)
        if block is None: raise ValueError("actual expiry exceeds the preserved curve/grid domain")
        found.append(math.ceil(maturity*block["n_steps"]/block["horizon"]))
    return found


def panel_cases(known_base, joint_base):
    points = [
        [ .12, 1.5, -.70, .04, .04, 2.], [ .025, 3.45, -.98, .04, .04, 2.],
        [ .10, 2.5, -.80, .15, .15, 2.], [ .08, 2., -.80, .015, .15, 3.],
        [ .08, 2., -.80, .15, .015, 3.], [ .20, 1., -.50, .08, .08, 1.],
        [ .12, 1.5, -.70, .02, .14, .26], [ .12, 2.5, -.80, .14, .02, 7.9],
        [ .03, 3.2, -.985, .15, .02, 4.], [ .34, .51, -.21, .02, .15, .3],
        [ .025, .6, -.5, .015, .015, 7.5], [ .30, 3.4, -.95, .15, .15, .3]]
    names = ["typical", "low_h_high_eta", "high_variance", "rising", "falling", "flat",
        "slow_decay", "fast_decay", "leverage_boundary", "high_h_low_eta", "low_level", "high_h_high_eta"]
    bounds = np.asarray(joint_base["prior_bounds"])
    if any(np.any(np.asarray(point) <= bounds[:, 0]) or np.any(np.asarray(point) >= bounds[:, 1]) for point in points):
        raise ValueError("predeclared joint challenge point outside preserved broad prior")
    cases = [{"id": f"joint-{i:02d}-{name}", "curve_role": "joint_unknown", "parameters": point}
        for i, (name, point) in enumerate(zip(names, points, strict=True))]
    known_points = [[.12, 1.5, -.7], [.025, 3.45, -.98], [.10, 2.5, -.8], [.34, .6, -.25]]
    for i, point in enumerate(known_points):
        seed = ROOT_SEED+100+i
        levels, coefficients = tasks.draw_context(known_base, np.random.default_rng(seed))
        cases.append({"id": f"known-{i:02d}", "curve_role": "supplied_known", "parameters": point,
            "supplied_levels": levels.tolist(), "curve_draw_seed": seed, "original_curve_draw_context": coefficients.tolist()})
    return cases


def declared_components():
    selected = [0, 1, 2, 12]; components = []
    for case_index in selected:
        for replicate in range(4):
            for role, pair in (("coupled_1_2", [1, 2]), ("coupled_2_4", [2, 4]), ("finest_replay", [4])):
                seed_group = 0 if role == "coupled_1_2" else 1
                streams = [{"block": block, "entropy": [NAMESPACE, ROOT_SEED, case_index, replicate, block, seed_group],
                    "seed": int(np.random.SeedSequence([NAMESPACE, ROOT_SEED, case_index, replicate, block, seed_group]).generate_state(1)[0])} for block in range(7)]
                components.append({"index": len(components), "case_index": case_index, "replicate": replicate,
                    "role": role, "multipliers": pair, "streams": streams, "paths": 32768,
                    "replay_seed_alias": "coupled_2_4 finest marginal" if role == "finest_replay" else None})
    return components


def prepare(design_path, heston_task_path, known_manifest, joint_manifest, out):
    out = Path(out)
    if out.exists(): raise FileExistsError("new rough preparation identity required")
    design_record, heston_task = read(design_path), read(heston_task_path)
    if (heston_task.get("format") != "heston_pr87_grid_task_v1"
            or digest(design_path) != heston_task["design_sha256"] or heston_task["target"]["quote_count"] != 63
            or heston_task["target"].get("pricing_controls", {}).get("fourier_representation") != "black_control_variate_v1"):
        raise ValueError("frozen supported Heston actual63 design/task pair required")
    known_record, joint_record = read(known_manifest), read(joint_manifest)
    known, joint = known_record["configuration"]["task"], joint_record["configuration"]["task"]
    for record, base in ((known_record, known), (joint_record, joint)):
        if bank.task_fingerprint(record["configuration"]) != record["task_fingerprint"]: raise ValueError("preserved rough task fingerprint differs")
    if runtime_tasks.is_joint_curve_task(known) or not runtime_tasks.is_joint_curve_task(joint):
        raise ValueError("distinct original supplied/joint curve laws required")
    normalized = actual_layout.normalize_design(design_record["records"])
    task_map = {"supplied_known": actual_layout.make_task(known, design_record["records"], measured_base_steps(known, normalized)),
        "joint_unknown": actual_layout.make_task(joint, design_record["records"], measured_base_steps(joint, normalized))}
    for task in task_map.values(): actual_layout.validate_task(task)
    cases = panel_cases(known, joint)
    selected = [0, 1, 2, 12]; components = declared_components()
    result = {"format": FORMAT, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_only_no_MC_executed",
        "design": binding(design_path), "heston_task": binding(heston_task_path), "heston_target": heston_task["target"],
        "preserved_known_snapshot": binding(known_manifest), "preserved_joint_snapshot": binding(joint_manifest),
        "tasks": task_map, "cases": cases, "initial_case_indices": selected, "components": components, "limits": LIMITS,
        "static_memory_audit": static_memory_audit(task_map),
        "grid_estimator_evaluations": 80, "independent_physical_rows": 4, "independent_batches_per_comparison": 4,
        "source_bindings": source_bindings(), "sbatch_sha256": hashlib.sha256(sbatch_text().encode()).hexdigest(), "execution_versions": {name: metadata.version(name) for name in ("numpy", "scipy")},
        "execution_condition": "completed source-bound new63 final300 Heston evaluation must be bound before any Slurm worker prices",
        "prices_generated": 0, "MC_executed": False, "reserved_arrays_opened": False, "scientific_acceptance": False,
        "scope": "first-stage numerical feasibility only; no bank generation, whitening fit, references or neural training"}
    out.mkdir(parents=True); write_new(out/"work-order.json", result)
    (out/"pilot.sbatch").write_text(sbatch_text())
    return result


def validate(work_order, repo_root=None):
    if (work_order.get("format") != FORMAT or work_order.get("limits") != LIMITS
            or len(work_order.get("components", [])) != 48 or work_order.get("grid_estimator_evaluations") != 80
            or work_order.get("MC_executed") is not False or work_order.get("reserved_arrays_opened") is not False):
        raise ValueError("source-bound prepared first-stage work order required")
    repo_root = Path(repo_root or ROOT).resolve()
    for spec in work_order["source_bindings"].values():
        if digest(repo_root/spec["relative_path"]) != spec["sha256"]: raise ValueError("actual execution source differs: "+spec["relative_path"])
    if hashlib.sha256(sbatch_text().encode()).hexdigest() != work_order.get("sbatch_sha256"):
        raise ValueError("prepared submission resources or command changed")
    if {name: metadata.version(name) for name in ("numpy", "scipy")} != work_order["execution_versions"]:
        raise ValueError("execution-host NumPy/SciPy versions differ")
    for task in work_order["tasks"].values():
        actual_layout.validate_task(task)
        if [block["n_steps"] for block in task["blocks"]] != measured_base_steps(task["base_task"], task["design"]):
            raise ValueError("declared initial time-step recipe changed")
    if (work_order.get("components") != declared_components() or work_order.get("initial_case_indices") != [0, 1, 2, 12]
            or work_order.get("cases") != panel_cases(work_order["tasks"]["supplied_known"]["base_task"],work_order["tasks"]["joint_unknown"]["base_task"])
            or work_order.get("static_memory_audit") != static_memory_audit(work_order["tasks"])):
        raise ValueError("declared pilot paths, grids, cases, streams or memory recipe changed")
    return repo_root


def heston_readiness(result, target):
    """Fail closed on incomplete panels, then reuse the frozen necessary screens."""
    if (result.get("format") != "heston_pr87_grid_evaluation_v1" or result.get("status") != "complete"
            or result.get("endpoint") != "primary_final" or result.get("native_epoch") != 300
            or result.get("task") != target or result.get("protected_inputs_unchanged") is not True):
        raise ValueError("user's successful Heston working-grid condition is not yet recorded")
    for name, rows, draws in (("native_validation", 100, 300), ("population", 512, 512)):
        summary = result.get(name, {})
        intervals = np.asarray(summary.get("coverage", {}).get("0.9", {}).get("wilson", []), float)
        ranks = summary.get("rank_dkw", {}).get("passed", [])
        if (summary.get("rows") != rows or summary.get("draws_per_case") != draws
                or intervals.shape != (6, 2) or not np.isfinite(intervals).all()
                or len(ranks) != 6 or any(type(flag) is not bool for flag in ranks)):
            raise ValueError("complete six-parameter Heston panels required for initiation")
    cases = result.get("cases", {})
    if set(cases) != {"synthetic-broad", "synthetic-market-like", "market"}:
        raise ValueError("all three target-matched Heston references required")
    for entry in cases.values():
        comparison = entry.get("comparison", {})
        values = np.asarray([comparison.get("standardized_mean_error", []), comparison.get("width_ratio", [])], float)
        if values.shape != (2, 6) or not np.isfinite(values).all() or entry.get("support", {}).get("samples") != 4096:
            raise ValueError("complete six-parameter conditional comparisons required")
    from scripts.render_heston_pr87_grid import readiness
    checks = readiness(result)
    if checks.get("necessary_diagnostics_pass") is not True:
        raise ValueError("Heston necessary diagnostics fail; completion alone cannot initiate rough pricing")
    return checks


def bind_heston(work_order_path, evaluation_path, out):
    work = read(work_order_path); validate(work); result = read(evaluation_path)
    checks = heston_readiness(result, work["heston_target"])
    record = {"format": "rough_actual_layout_heston_condition_v1", "work_order_sha256": digest(work_order_path),
        "heston_evaluation": binding(evaluation_path), "heston_target": result["task"], "working_grid_condition_met": True,
        "necessary_diagnostics": checks, "scientific_acceptance": False,
        "scope": "bounded rough numerical initiation only; no estimator promotion or production-bank allocation"}
    write_new(out, record); return record


def validate_condition(work, work_order_path, condition, evaluation_path=None):
    if (condition.get("format") != "rough_actual_layout_heston_condition_v1" or condition.get("working_grid_condition_met") is not True
            or condition.get("work_order_sha256") != digest(work_order_path) or condition.get("heston_target") != work["heston_target"]):
        raise ValueError("completed matching Heston initiation condition required")
    spec = condition.get("heston_evaluation", {})
    path = Path(evaluation_path or spec.get("path", ""))
    if not path.is_file() or digest(path) != spec.get("sha256"):
        raise ValueError("bound Heston evaluation changed")
    checks = heston_readiness(read(path), work["heston_target"])
    if checks != condition.get("necessary_diagnostics"):
        raise ValueError("Heston initiation diagnostics changed")


def static_memory_audit(task_map):
    """Conservative shape accounting, not an empirical peak-RSS measurement."""
    chunk = runtime_pricing._PAIR_CHUNK
    steps = max(block["n_steps"]*4 for task in task_map.values() for block in task["blocks"])
    fft_size = 1 << (2*steps-1).bit_length()
    # Existing helper loops chunks, grids and antithetic signs; it retains only
    # 9 payoffs + 2 controls per pair, never all 16,384 full driver paths.
    allowance = chunk*(steps+1)*8*64 + chunk*fft_size*16*16 + 16384*11*8*16
    if chunk != 2048 or allowance >= 24*1024**3: raise ValueError("static memory assumptions changed; review allocation")
    return {"pair_chunk": chunk, "maximum_grid_steps": steps, "maximum_innovation_shape": [chunk, steps, 3],
        "maximum_fft_size": fft_size, "conservative_array_allowance_bytes": allowance,
        "allocation_bytes": LIMITS["memory_gib_per_component"]*1024**3,
        "scope": "64 float64 path arrays plus16 complex128 FFT arrays plus16 retained payoff/control copies; not measured peak RSS"}


def load_controlled_helper(repo_root):
    spec = importlib.util.spec_from_file_location("frozen_rough_controlled_crn", repo_root/HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper


def checked_covariance(value, size):
    """Estimator sums of squares guarantee nonnegative variances; reject violations."""
    covariance = np.asarray(value, float)
    if covariance.shape != (size, size) or not np.isfinite(covariance).all():
        raise ValueError("finite correctly shaped covariance required")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if np.max(np.abs(covariance-covariance.T)) > 32*np.finfo(float).eps*scale:
        raise ValueError("asymmetric covariance exceeds declared32eps relative roundoff tolerance")
    if np.any(np.diag(covariance) < 0):
        raise ValueError("negative covariance variance violates estimator sum-of-squares guarantee")
    return covariance


def sbatch_text():
    return '''#!/bin/bash
#SBATCH --job-name=rough-actual-R1
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --array=0-47%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
set -euo pipefail
: "${ROUGH_REPO_ROOT:?stage full lab and sibling tutorial checkout first}"
: "${ROUGH_PYTHON:?set verified execution-host Python}"
: "${ROUGH_WORK_ORDER:?set frozen work-order.json}"
: "${ROUGH_CONDITION:?bind completed Heston result first}"
: "${ROUGH_HESTON_EVALUATION:?stage exact hash-bound Heston evaluation JSON}"
: "${ROUGH_PAYLOAD_ROOT:?set NEW output root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROUGH_REPO_ROOT/lab/src:$ROUGH_REPO_ROOT/tutorial/src"
exec "$ROUGH_PYTHON" "$ROUGH_REPO_ROOT/lab/scripts/run_rough_actual_layout_pilot.py" --work-order "$ROUGH_WORK_ORDER" --condition "$ROUGH_CONDITION" --heston-evaluation "$ROUGH_HESTON_EVALUATION" --index "$SLURM_ARRAY_TASK_ID" --out "$ROUGH_PAYLOAD_ROOT/component-$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
'''


def curve_for_case(work, case):
    task = work["tasks"][case["curve_role"]]
    nodes = actual_layout.simulation_nodes(task)
    if case["curve_role"] == "supplied_known": return actual_layout.known_curve(task, case["supplied_levels"])
    return pricing.heston_variance_curve(nodes, *case["parameters"][3:])


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
    record = {"format": "rough_actual_layout_component_v1", "status": "running", "component": component,
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

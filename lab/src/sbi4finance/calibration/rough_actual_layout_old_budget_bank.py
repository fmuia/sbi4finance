"""Historical-cost-budget actual-layout candidate banks with durable per-shard journals.

Only orchestration lives here. The original hybrid model, curve prior,
conditional estimator, fitted controls and coupled-grid helper remain canonical.
Generation needs an explicit numerical admission and a bounded active tranche;
the eventual 300k training target does not itself allocate that work.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import sqlite3
import time

import numpy as np
import scipy

from . import rough_actual_layout_pilot_portable as pilot

FORMAT = "rough_actual_layout_legacy_budget_bank_plan_v1"
ADMISSION = "rough_actual_layout_legacy_budget_execution_authorization_v1"
LANES = ("supplied_known", "joint_unknown")
SPLITS = {"train": 0, "validation": 1, "test": 2}
RELAXED_POLICY = "rough_actual_layout_historical_sigma_numerical_budget_v1"
NAMESPACE = 0x52424131
ROOT = pilot.ROOT


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


def sources():
    result = pilot.source_bindings()
    for name, path in {"actual_bank": Path(__file__),
            "actual_bank_cli": ROOT/"lab/scripts/rough_actual_layout_old_budget_bank.py"}.items():
        result[name] = {"relative_path": str(path.resolve().relative_to(ROOT)), "sha256": pilot.digest(path)}
    return result


def checked_policy(policy, candidate_only=False):
    if policy.get("format") != RELAXED_POLICY: raise ValueError("explicit legacy numerical budget version required")
    if policy.get("format") == RELAXED_POLICY:
        if (set(policy) != {"format","grid_multiplier","paths_ladder","numerical_reference_forward","numerical_reference_scope","historical_sigma_vector_sha256","preferred_mcse_over_reference_sigma","hardcap_mcse_over_reference_sigma","refinement_requirement","price_target"}
                or policy["grid_multiplier"] != 1 or policy["preferred_mcse_over_reference_sigma"] != .25
                or policy["hardcap_mcse_over_reference_sigma"] != .5 or policy["refinement_requirement"] != "record_only_not_certified"
                or policy["price_target"] != "finite_hybrid_base_grid_expectation"
                or policy["numerical_reference_scope"] != "historical_median_not_observation_noise"
                or not isinstance(policy["numerical_reference_forward"],(int,float))
                or not np.isfinite(policy["numerical_reference_forward"]) or policy["numerical_reference_forward"]<=0):
            raise ValueError("explicit historical-numerical-reference preferred.25/final.5 finite-base-grid policy required")
        ladder=policy["paths_ladder"]
        if not ladder or any(type(n) is not int or n < 32768 or n % 2 for n in ladder) or any(b<=a for a,b in zip(ladder,ladder[1:])):
            raise ValueError("finite increasing path ladder required")
        return policy
    expected = {"grid_pair", "paths_ladder", "mcse_over_strict_sd", "refinement_over_strict_sd", "refinement_z"}
    if set(policy) != expected or policy["grid_pair"] != [2, 4]:
        raise ValueError("explicit original-pilot grid pair2/4 and complete numerical policy required")
    ladder = policy["paths_ladder"]
    if (not ladder or any(type(n) is not int or n < 32768 or n % 2 for n in ladder)
            or any(b <= a for a, b in zip(ladder, ladder[1:]))):
        raise ValueError("finite increasing even path ladder required; no implicit escalation")
    for key in ("mcse_over_strict_sd", "refinement_over_strict_sd", "refinement_z"):
        if key == "refinement_over_strict_sd" and candidate_only and policy[key] is None: continue
        if not isinstance(policy[key], (int, float)) or not np.isfinite(policy[key]):
            raise ValueError("finite explicit numerical tolerances required")
    if not 0 < policy["mcse_over_strict_sd"] <= .25 or (policy["refinement_over_strict_sd"] is not None and policy["refinement_over_strict_sd"] <= 0) or policy["refinement_z"] < 0:
        raise ValueError("strict measurement-only MC gate and positive declared refinement allowance required")
    return policy


def job_map(active_ranges, target_splits, rows_per_shard):
    if type(rows_per_shard) is not int or rows_per_shard < 1:
        raise ValueError("positive integer shard size required")
    if set(active_ranges) != set(LANES) or set(target_splits) != set(SPLITS):
        raise ValueError("two distinct lanes and train/validation/test target sizes required")
    if any(type(n) is not int or n < 1 for n in target_splits.values()):
        raise ValueError("positive target sizes required")
    jobs = []
    for lane in LANES:
        if not set(active_ranges[lane]) <= set(SPLITS): raise ValueError("unknown active split")
        for split in SPLITS:
            if split not in active_ranges[lane]: continue
            start, stop = active_ranges[lane][split]
            if any(type(n) is not int for n in (start, stop)) or not 0 <= start < stop <= target_splits[split]:
                raise ValueError("active rows must be a nonempty bounded target subset")
            for first in range(start, stop, rows_per_shard):
                jobs.append({"index": len(jobs), "lane": lane, "split": split,
                    "start": first, "stop": min(first+rows_per_shard, stop)})
    if not jobs: raise ValueError("an explicit nonempty active tranche is required")
    return jobs


def prepare(parent_path, out, *, root_seed, target_splits, active_ranges, rows_per_shard,
            numerical_policy, row_wall_seconds, shard_wall_seconds, concurrent_workers, total_cpu_seconds, candidate_only=False, memory_gib_per_worker=32, clean_only=False, inferred_c_blueprint=False):
    """Metadata only. No prices, paths, whitening or neural training are computed."""
    parent = pilot.read(parent_path); pilot.validate(parent)
    if type(candidate_only) is not bool: raise ValueError("explicit candidate/qualified mode required")
    checked_policy(numerical_policy, candidate_only)
    historical_sigma=parent["tasks"]["joint_unknown"]["base_task"]["context"]["quote_noise_std"]
    if numerical_policy["numerical_reference_forward"] != float(np.median(historical_sigma)) or numerical_policy["historical_sigma_vector_sha256"] != fingerprint(historical_sigma):
        raise ValueError("numerical reference must equal the declared historical median; observation h is unchanged")
    if numerical_policy["paths_ladder"] != parent["tasks"]["joint_unknown"]["base_task"]["numerics"]["paths_ladder"]:
        raise ValueError("original finite adaptive path ladder required")
    if type(clean_only) is not bool or type(inferred_c_blueprint) is not bool: raise ValueError("explicit clean-bank and inference-blueprint roles required")
    if inferred_c_blueprint and (not clean_only or active_ranges.get("supplied_known")): raise ValueError("joint-c blueprint requires clean-only joint lane")
    jobs = job_map(active_ranges, target_splits, rows_per_shard)
    bounds = (row_wall_seconds, shard_wall_seconds, concurrent_workers, total_cpu_seconds)
    if any(type(n) is not int or n < 1 for n in (*bounds, root_seed)):
        raise ValueError("explicit positive integer seeds/resource bounds required")
    if concurrent_workers > 208 or memory_gib_per_worker not in (8,32) or concurrent_workers*memory_gib_per_worker > 2300 or shard_wall_seconds < rows_per_shard*row_wall_seconds+30:
        raise ValueError("208CPU/2300GiB safety envelope and complete row/setup wall allowances required")
    worst = len(jobs)*shard_wall_seconds
    if total_cpu_seconds < worst:
        raise ValueError("active-tranche allocated CPU envelope is below its declared worker caps")
    if Path(out).exists(): raise FileExistsError("new immutable bank plan required")
    result = {"format": FORMAT, "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_work_order": pilot.binding(parent_path), "parent_work_order_value": deepcopy(parent),
        "tasks": deepcopy(parent["tasks"]), "root_seed": root_seed,
        "target_splits": target_splits, "active_ranges": active_ranges, "rows_per_shard": rows_per_shard,
        "jobs": jobs, "numerical_policy": numerical_policy, "candidate_only": candidate_only,
        "clean_only":clean_only,"inferred_c_blueprint":inferred_c_blueprint,
        "inference_blueprint":{"parameter_names":[*parent["tasks"]["joint_unknown"]["parameter_names"],"c"],
            "prior_bounds":[*parent["tasks"]["joint_unknown"]["prior_bounds"],[.5,2.]],
            "c_role":"independent posterior target, never conditioning; no c-driven repricing",
            "augmentation_performed":True} if inferred_c_blueprint else None,
        "SBI_training_readiness":False,"prior_filtering_permitted":False, "source_bindings": sources(), "versions": versions(),
        "limits": {"row_wall_seconds": row_wall_seconds, "shard_wall_seconds": shard_wall_seconds,
            "concurrent_workers": concurrent_workers, "cpu_seconds_ceiling": total_cpu_seconds,
            "active_worst_case_allocated_cpu_seconds": worst, "cpus_per_worker": 1, "memory_gib_per_worker": memory_gib_per_worker,
            "automatic_retry_failed_rows": False, "automatic_paths_extension": False},
        "rng": "PCG64/SeedSequence; per split+row shared dynamics, lane-specific pricing/quotes/curve streams; counts/scheduling absent from row identity",
        "observation_model": ("six-input clean price means plus independent U(.5,2) c/Gaussian observation augmentation; c is a target, never conditioning" if inferred_c_blueprint else "clean prices only; no observations generated" if clean_only
            else "known c=1; independent Gaussian errors with forward-unit observed half-spreads; no clipping"),
        "numerical_scope": "nominal fitted-control covariance; refinement where evaluated is auxiliary; no continuous-model bias certificate",
        "pricing_target": numerical_policy.get("price_target", "empirically_refined_grid4_expectation"),
        "parent_numerical_policy_scope": "parent gate is reference provenance; this new bank uses its own explicitly versioned numerical_policy",
        "scientific_acceptance": False, "reserved_test_evaluated": False, "prices_generated": 0}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pilot.write_new(out, result)
    return result


def validate(plan):
    if (plan.get("format") != FORMAT or plan.get("scientific_acceptance") is not False
            or plan.get("prices_generated") != 0 or plan.get("reserved_test_evaluated") is not False):
        raise ValueError("source-bound candidate bank declaration required")
    if type(plan.get("clean_only")) is not bool or type(plan.get("inferred_c_blueprint")) is not bool or plan.get("SBI_training_readiness") is not False or plan.get("prior_filtering_permitted") is not False:
        raise ValueError("explicit clean-label scope and no prior filtering required")
    if plan["inferred_c_blueprint"] and (not plan["clean_only"] or plan["active_ranges"].get("supplied_known")):
        raise ValueError("joint-c blueprint requires clean-only joint lane")
    expected_blueprint={"parameter_names":[*plan["tasks"]["joint_unknown"]["parameter_names"],"c"],
        "prior_bounds":[*plan["tasks"]["joint_unknown"]["prior_bounds"],[.5,2.]],
        "c_role":"independent posterior target, never conditioning; no c-driven repricing","augmentation_performed":True}
    if plan["inference_blueprint"] != (expected_blueprint if plan["inferred_c_blueprint"] else None):
        raise ValueError("inferred-c target/conditioning blueprint changed")
    pilot.validate(plan["parent_work_order_value"])
    if plan["tasks"] != plan["parent_work_order_value"]["tasks"]:
        raise ValueError("financial tasks or curve priors changed")
    if type(plan.get("candidate_only")) is not bool: raise ValueError("explicit candidate/qualified mode required")
    checked_policy(plan["numerical_policy"], plan["candidate_only"])
    historical_sigma=plan["tasks"]["joint_unknown"]["base_task"]["context"]["quote_noise_std"]
    if plan["numerical_policy"]["numerical_reference_forward"] != float(np.median(historical_sigma)) or plan["numerical_policy"]["historical_sigma_vector_sha256"] != fingerprint(historical_sigma) or plan["numerical_policy"]["paths_ladder"] != plan["tasks"]["joint_unknown"]["base_task"]["numerics"]["paths_ladder"]:
        raise ValueError("historical numerical reference or path ladder changed")
    if plan["jobs"] != job_map(plan["active_ranges"], plan["target_splits"], plan["rows_per_shard"]):
        raise ValueError("declared active row/shard map changed")
    if plan["versions"] != versions() or plan["source_bindings"] != sources():
        raise ValueError("producer source or execution environment changed; no continuation")
    limits = plan["limits"]
    if (any(type(limits.get(key)) is not int or limits[key] < 1 for key in
            ("row_wall_seconds", "shard_wall_seconds", "concurrent_workers", "cpu_seconds_ceiling"))
            or limits["concurrent_workers"] > 208
            or limits["concurrent_workers"]*limits["memory_gib_per_worker"] > 2300
            or limits["shard_wall_seconds"] < plan["rows_per_shard"]*limits["row_wall_seconds"]+30):
        raise ValueError("positive explicit resource caps and full row/setup allowances required")
    if (limits["active_worst_case_allocated_cpu_seconds"] != len(plan["jobs"])*limits["shard_wall_seconds"]
            or limits["cpu_seconds_ceiling"] < limits["active_worst_case_allocated_cpu_seconds"]
            or limits["cpus_per_worker"] != 1 or limits["memory_gib_per_worker"] not in (8,32)
            or limits["automatic_retry_failed_rows"] is not False or limits["automatic_paths_extension"] is not False):
        raise ValueError("explicit bounded allocation and no automatic failed-row retries required")


def validate_admission(plan_path, plan, admission_path):
    admission = pilot.read(admission_path)
    spec = admission.get("numerical_review", {})
    path = Path(spec.get("path", ""))
    if (admission.get("format") != ADMISSION or admission.get("bank_plan_sha256") != pilot.digest(plan_path)
            or admission.get("generation_authorized") is not True
            or admission.get("mode") != ("candidate_only" if plan["candidate_only"] else "qualified_rows")
            or (not plan["candidate_only"] and plan["numerical_policy"].get("format") != RELAXED_POLICY and admission.get("pilot_numerical_review_passed") is not True)
            or (plan["numerical_policy"].get("format") == RELAXED_POLICY and (admission.get("label_policy_authorized") is not True or admission.get("pilot_execution_reviewed") is not True))
            or admission.get("numerical_policy") != plan["numerical_policy"]
            or admission.get("scientific_acceptance") is not False
            or not path.is_file() or pilot.digest(path) != spec.get("sha256")):
        raise ValueError("matching explicit execution authorization and numerical policy required before bank pricing")
    return admission


def row_spec(plan, lane, split, row_id):
    if lane not in LANES or split not in SPLITS or type(row_id) is not int or not 0 <= row_id < plan["target_splits"][split]:
        raise ValueError("declared lane/split/physical row required")
    task = plan["tasks"][lane]; bounds = np.asarray(task["prior_bounds"])
    identity = [NAMESPACE, plan["root_seed"], SPLITS[split], row_id]
    prior = np.random.default_rng(np.random.SeedSequence([*identity, 0]))
    values = prior.uniform(bounds[:, 0], bounds[:, 1])
    level, context = (pilot.original.tasks.draw_context(task["base_task"], np.random.default_rng(np.random.SeedSequence([*identity, 3])))
        if lane == "supplied_known" else (np.empty(0), values[3:].copy()))
    return {"lane": lane, "split": split, "row_id": row_id, "parameters": values.tolist(),
        "supplied_curve_levels": level.tolist(), "context_parameters": context.tolist(),
        "identity_entropy": identity, "curve_role": task["curve_role"],
        "noise_scale":float(np.random.default_rng(np.random.SeedSequence([*identity,4])).uniform(.5,2.)) if plan["inferred_c_blueprint"] else 1.}


def row_curve(task, spec):
    if spec["lane"] == "supplied_known": return pilot.actual_layout.known_curve(task, spec["supplied_curve_levels"])
    return pilot.original.pricing.heston_variance_curve(pilot.actual_layout.simulation_nodes(task), *spec["parameters"][3:])


def quality(result, strict_sd, policy):
    if policy.get("format") == RELAXED_POLICY:
        n=len(strict_sd);cov=pilot.checked_covariance(result["moments"][1]["bank_covariance"],n)
        if not np.isfinite(result["controlled_prices"]).all(): raise ValueError("nonfinite candidate block")
        ratio=np.sqrt(np.diag(cov))/strict_sd
        preferred=bool(np.all(ratio <= policy["preferred_mcse_over_reference_sigma"]))
        return {"max_mcse_over_numerical_reference":float(ratio.max()),"preferred_passed":preferred,
            "hardcap_passed":bool(np.all(ratio <= policy["hardcap_mcse_over_reference_sigma"])),
            "refinement_evaluated":False,"continuous_bias_certified":False,"passed":preferred}
    width = len(strict_sd)
    fine = pilot.checked_covariance(result["moments"][1]["bank_covariance"], width)
    difference = pilot.checked_covariance(result["difference_covariance"], width)
    for value in (fine, difference):
        allowance = 128*np.finfo(float).eps*width*max(float(np.max(np.abs(value))), np.finfo(float).tiny)
        if np.linalg.eigvalsh(value).min() < -allowance: raise ValueError("non-PSD candidate covariance")
    if not all(np.isfinite(result[key]).all() for key in ("controlled_prices", "raw_prices", "difference")):
        raise ValueError("nonfinite candidate block")
    mc = np.sqrt(np.diag(fine))/strict_sd
    shift = (np.abs(result["difference"])+policy["refinement_z"]*np.sqrt(np.diag(difference)))/strict_sd
    mc_passed = bool(np.all(mc <= policy["mcse_over_strict_sd"]))
    refinement_passed = None if policy["refinement_over_strict_sd"] is None else bool(np.all(shift <= policy["refinement_over_strict_sd"]))
    return {"max_mcse_over_strict_sd": float(mc.max()), "max_refinement_envelope_over_strict_sd": float(shift.max()),
        "mc_passed": mc_passed, "refinement_passed": refinement_passed,
        "passed": bool(mc_passed and refinement_passed is True)}


def pack(arrays):
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays); return stream.getvalue()


def unpack(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive: return {k: archive[k].copy() for k in archive.files}


def journal(path, identity):
    """One writer per Slurm shard; FULL synchronous transactions retain blocks."""
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE"); connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS draws (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER, block INTEGER, rung INTEGER, status TEXT, summary TEXT, payload BLOB, PRIMARY KEY(id,block,rung))")
    connection.execute("CREATE TABLE IF NOT EXISTS rows (id INTEGER PRIMARY KEY, status TEXT, summary TEXT, payload BLOB)")
    found = connection.execute("SELECT value FROM identity").fetchall()
    if not found:
        connection.execute("INSERT INTO identity VALUES (?)", (canonical(identity),)); connection.commit()
    elif found != [(canonical(identity),)]:
        connection.close(); raise ValueError("journal source/task/job identity differs; no continuation")
    return connection


def _run_shard(plan_path, admission_path, index, out, *, process_started):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("one allocated Slurm CPU required")
    plan = pilot.read(plan_path); validate(plan); validate_admission(plan_path, plan, admission_path)
    if type(index) is not int or not 0 <= index < len(plan["jobs"]): raise ValueError("declared shard index required")
    job = plan["jobs"][index]; task = plan["tasks"][job["lane"]]; policy = plan["numerical_policy"]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    identity = {"plan_sha256": pilot.digest(plan_path), "admission_sha256": pilot.digest(admission_path), "job": job,
        "sources": plan["source_bindings"], "versions": versions()}
    db = journal(out/"journal.sqlite", identity)
    from scripts.run_heston_spread_reference import Budget
    helper = pilot.load_controlled_helper(ROOT)
    try:
        if (out/"manifest.json").exists():
            record = pilot.read(out/"manifest.json")
            if (any(record.get(k) != v for k,v in identity.items())
                    or record.get("payload") != pilot.binding(out/"data.npz")
                    or record.get("row_journal") != pilot.binding(out/"journal.sqlite")):
                raise ValueError("completed shard identity or payload changed")
            return record
        for row_id in range(job["start"], job["stop"]):
            found = db.execute("SELECT status FROM rows WHERE id=?", (row_id,)).fetchone()
            if found:
                if found[0] not in ("complete", "complete_candidate"): raise ValueError("failed row retained; no automatic retry or filtering")
                continue
            saved = db.execute("SELECT value FROM draws WHERE id=?", (row_id,)).fetchone()
            spec = json.loads(saved[0]) if saved else row_spec(plan, job["lane"], job["split"], row_id)
            if not saved:
                db.execute("INSERT INTO draws VALUES (?,?)", (row_id, canonical(spec))); db.commit()
            relaxed = policy.get("format") == RELAXED_POLICY
            strict = np.full(63,policy["numerical_reference_forward"])
            curve = row_curve(task, spec); parameters = dict(zip(("hurst", "eta", "rho"), spec["parameters"][:3], strict=True))
            chosen_results = []; row_started = time.monotonic()
            remaining = min(plan["limits"]["row_wall_seconds"], plan["limits"]["shard_wall_seconds"]-(time.monotonic()-process_started)-5)
            try:
                if remaining <= 0: raise TimeoutError("declared shard total wall budget exhausted")
                with Budget(remaining) as budget:
                    for block_id, block in enumerate(task["blocks"]):
                        columns = [i for i,c in enumerate(task["design"]["contracts"]) if c["maturity"] == block["horizon"]]
                        contracts = [task["design"]["contracts"][i] for i in columns]
                        accepted, last = None, None
                        for rung, paths in enumerate(policy["paths_ladder"]):
                            budget.check()
                            prior = db.execute("SELECT status,summary,payload FROM attempts WHERE id=? AND block=? AND rung=?", (row_id,block_id,rung)).fetchone()
                            if prior:
                                if prior[0] == "accepted": accepted = (columns, paths, json.loads(prior[1]), unpack(prior[2])); break
                                if prior[0] != "rejected_precision": raise ValueError("failed attempt retained; no retry")
                                last = (columns, paths, json.loads(prior[1]), unpack(prior[2]))
                                continue
                            entropy = [*spec["identity_entropy"], 1, LANES.index(job["lane"]), block_id, rung]
                            if relaxed:
                                diagnostic={"capture_raw_control":True}
                                prices,errors,cov=pilot.runtime_pricing.rough_block(parameters,curve,contracts,horizon=block["horizon"],
                                    n_steps=block["n_steps"],n_pairs=paths//2,seed=np.random.SeedSequence(entropy),diagnostics=diagnostic)
                                parity=np.asarray([1-c["strike"] if c["option_type"]=="call" else 0. for c in contracts])
                                raw=np.asarray(diagnostic["raw_control"]["raw_put_mean"])+parity;n=len(columns)
                                controls=diagnostic["raw_control"]
                                moment={"bank_covariance":cov.tolist(),
                                    "coefficient_matrix":np.asarray(controls.get("fitted_coefficients",[])).tolist(),
                                    "control_mean":np.asarray(controls.get("control_mean",[])).tolist(),
                                    "variance_scope":"nominal same-sample fitted controls; independent-batch calibration is separate"}
                                result={"controlled_prices":np.stack([prices,prices]),"raw_prices":np.stack([raw,raw]),
                                    "difference":np.full(n,np.nan),"difference_covariance":np.full((n,n),np.nan),
                                    "moments":[moment,moment]}
                            else:
                                result = helper.coupled_controlled_block(parameters, curve, contracts, horizon=block["horizon"],
                                    steps=tuple(block["n_steps"]*g for g in policy["grid_pair"]), n_pairs=paths//2, seed=np.random.SeedSequence(entropy))
                            metrics = quality(result, strict[columns], policy)
                            if relaxed and rung == len(policy["paths_ladder"])-1:
                                metrics["passed"]=metrics["hardcap_passed"]
                                metrics["used_final_hardcap"]=bool(metrics["hardcap_passed"] and not metrics["preferred_passed"])
                            arrays = {k: np.asarray(v) for k,v in result.items() if k != "moments"}
                            arrays["bank_covariance"] = np.asarray([m["bank_covariance"] for m in result["moments"]])
                            summary = {**metrics, "paths": paths, "entropy": entropy, "moments": result["moments"]}
                            status = "accepted" if metrics["passed"] else "rejected_precision"
                            db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?)", (row_id,block_id,rung,status,canonical(summary),sqlite3.Binary(pack(arrays)))); db.commit()
                            last = (columns, paths, summary, arrays)
                            if metrics["passed"]: accepted = last; break
                        if accepted is None and plan["candidate_only"]: accepted = last
                        if accepted is None: raise ValueError("finite path ladder exhausted without strict precision; row retained")
                        chosen_results.append(accepted)
                prices = np.zeros(63); covariance = np.zeros((63,63)); delta = np.zeros(63); dcov = np.zeros((63,63)); used=[]
                for columns, paths, summary, arrays in chosen_results:
                    prices[columns] = arrays["controlled_prices"][1]; covariance[np.ix_(columns,columns)] = arrays["bank_covariance"][1]
                    delta[columns] = arrays["difference"]; dcov[np.ix_(columns,columns)] = arrays["difference_covariance"]; used.append(paths)
                quote_rng = np.random.default_rng(np.random.SeedSequence([*spec["identity_entropy"],2,LANES.index(job["lane"])]))
                noise = quote_rng.standard_normal(63); h = np.asarray(task["design"]["half_spread_forward"]); factors=np.asarray(task["design"]["price_factor_DF"])
                qualified = all(part[2]["passed"] for part in chosen_results)
                arrays = {"numerical_qualified":np.asarray(qualified), "parameters":np.asarray(spec["parameters"]), "reference_prices_forward":prices, "reference_prices_raw":prices*factors,
                    "observations_forward":prices+h*noise, "observations_raw":(prices+h*noise)*factors,
                    "quote_standard_normals":noise, "pricing_covariance_forward":covariance,
                    "pricing_standard_errors_forward":np.sqrt(np.diag(covariance)), "refinement_difference_forward":delta,
                    "refinement_covariance_forward":dcov, "refinement_evaluated":np.asarray(not relaxed), "continuous_bias_certified":np.asarray(False), "pricing_paths":np.asarray(used,dtype=np.int64),
                    "supplied_curve_levels":np.asarray(spec["supplied_curve_levels"]), "context_parameters":np.asarray(spec["context_parameters"])}
                arrays["physical_parameters"]=arrays["parameters"].copy()
                if plan["inferred_c_blueprint"]:
                    c=spec["noise_scale"]
                    arrays["noise_scale"]=np.asarray(c)
                    arrays["inference_parameters"]=np.concatenate([arrays["physical_parameters"],[c]])
                    arrays["parameters"]=arrays["inference_parameters"].copy()
                    arrays["observations_forward"]=prices+c*h*noise
                    arrays["observations_raw"]=(prices+c*h*noise)*factors
                    arrays["pricing_mcse_over_h"]=np.sqrt(np.diag(covariance))/h
                    arrays["pricing_mcse_over_observation_sd"]=np.sqrt(np.diag(covariance))/(c*h)
                elif plan["clean_only"]:
                    for key in ("observations_forward","observations_raw","quote_standard_normals"): arrays.pop(key)
                db.execute("INSERT INTO rows VALUES (?,?,?,?)", (row_id,"complete" if qualified else "complete_candidate",canonical({"seconds":time.monotonic()-row_started,"spec":spec,"numerical_qualified":qualified}),sqlite3.Binary(pack(arrays)))); db.commit()
            except BaseException as error:
                db.execute("INSERT OR REPLACE INTO rows VALUES (?,?,?,NULL)", (row_id,"failed",canonical({"error_type":type(error).__name__,"error":str(error),"spec":spec}))); db.commit(); raise
        values = db.execute("SELECT id,payload FROM rows WHERE status IN ('complete','complete_candidate') ORDER BY id").fetchall()
        if [v[0] for v in values] != list(range(job["start"],job["stop"])): raise ValueError("shard contains missing physical rows")
        rows = [unpack(value) for _,value in values]; arrays={k:np.stack([v[k] for v in rows]) for k in rows[0]}
        arrays["example_id"] = np.asarray([f"{job['lane']}:{job['split']}:{i}" for i,_ in values])
        payload=out/"data.npz"
        if payload.exists():
            if payload.read_bytes() != pack(arrays): raise ValueError("existing complete shard payload differs")
        else: payload.write_bytes(pack(arrays))
        result={"format":"rough_actual_layout_legacy_budget_bank_shard_v1","status":"complete",**identity,
            "payload":pilot.binding(payload),"physical_rows":len(rows),"scientific_acceptance":False,
            "price_target":policy.get("price_target","empirically_refined_grid4_expectation"),"continuous_bias_certified":False,
            "SBI_training_readiness":False,"prior_filtering_permitted":False,"clean_only":plan["clean_only"],
            "candidate_only":plan["candidate_only"],"training_usable":bool(not plan["clean_only"] and not plan["candidate_only"] and all(bool(v["numerical_qualified"]) for v in rows)),
            "numerical_qualified_rows":sum(bool(v["numerical_qualified"]) for v in rows),
            "seconds":time.monotonic()-process_started,"row_journal":pilot.binding(out/"journal.sqlite"),
            "generation_architecture":platform.machine()}
        if not (out/"manifest.json").exists(): pilot.write_new(out/"manifest.json",result)
        return result
    finally:
        db.close()


def run_shard(plan_path, admission_path, index, out, *, process_started):
    """A lease prevents two scheduler attempts from writing one shard concurrently."""
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("one allocated Slurm CPU required")
    plan = pilot.read(plan_path); validate(plan); validate_admission(plan_path, plan, admission_path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with (out/"worker.lease").open("a") as lease:
        try: fcntl.flock(lease.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise ValueError("another worker owns this shard") from error
        try: return _run_shard(plan_path, admission_path, index, out, process_started=process_started)
        finally: fcntl.flock(lease.fileno(), fcntl.LOCK_UN)

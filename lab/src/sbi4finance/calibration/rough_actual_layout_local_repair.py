"""Explicit local repair of a frozen failed-row transport, never a Slurm resume.

Financial kernels and pure task/RNG helpers are reused unchanged. Original
Oscar attempts remain immutable; a new journal records which blocks were
reused and which were evaluated locally. Cross-platform results have new
provenance even when their financial task and random streams agree.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import sqlite3
import time

import numpy as np

from . import rough_actual_layout_overnight_bank as bank

ROOT = bank.ROOT
FORMAT = "rough_actual_layout_local_failed_repair_plan_v1"
BUNDLE_FORMAT = "rough_failed_rows_repair_bundle_v1"


def runtime():
    return {**bank.versions(), "system": platform.system(), "machine": platform.machine(),
            "platform": platform.platform(), "execution_mode": "local_no_slurm"}


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": bank.pilot.digest(path)}


def checked_file(root, spec):
    path = (Path(root) / spec["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or bank.pilot.digest(path) != spec["sha256"]:
        raise ValueError("transport file absent, outside bundle or changed")
    return path


def source_bindings(producers):
    result = {}
    for producer in producers.values():
        for item in producer["source_bindings"].values():
            relative = item["relative_path"]
            if relative in result and result[relative] != item["sha256"]:
                raise ValueError("incompatible original producer sources")
            result[relative] = item["sha256"]
    for path in (Path(__file__), ROOT / "lab/scripts/rough_actual_layout_local_repair.py"):
        result[str(path.resolve().relative_to(ROOT))] = bank.pilot.digest(path)
    for relative, digest in result.items():
        if bank.pilot.digest(ROOT / relative) != digest:
            raise ValueError("original or local repair source changed")
    return result


def verify_stored_spec(spec, expected, failure):
    """Validate host replay without replacing immutable Oscar physical truths.

    NumPy uniform's affine arithmetic can differ by one ULP on x86/ARM.
    Only derived physical/context values admit a two-ULP replay allowance;
    c, RNG entropy, identities and all other fields remain exact. The archived
    draw must also exactly equal the independently retained failure record.
    """
    if failure.get("spec") != spec or set(spec) != set(expected):
        raise ValueError("stored draw differs from original failure identity")
    replay = {}
    for key in spec:
        if key in ("parameters", "context_parameters"):
            actual, regenerated = np.asarray(spec[key],dtype=np.float64), np.asarray(expected[key],dtype=np.float64)
            if actual.shape != regenerated.shape or not np.isfinite(actual).all() or not np.isfinite(regenerated).all():
                raise ValueError("physical/context replay shape or finiteness changed")
            unit = np.maximum(np.spacing(np.abs(actual)),np.spacing(np.abs(regenerated)))
            distance = np.abs(actual-regenerated)/unit
            if np.any(distance>2):
                raise ValueError("physical/context host replay exceeds two ULP")
            replay[key+"_max_ulp"] = float(distance.max()) if distance.size else 0.
        elif spec[key] != expected[key]:
            raise ValueError("physical prior, c or RNG identity changed")
    return {**replay,"allowance_ulp":2,"scope":"host regeneration check only; exact stored Oscar truths used for execution"}


def selected_row(manifest_path, manifest, item):
    """Read a selected row from the explicitly partial transport schema."""
    root = Path(manifest_path).parent
    producer = manifest["producers"][item["producer_kind"]]
    plan_path = checked_file(root, producer["plan"])
    admission_path = checked_file(root, producer["admission"])
    plan = bank.pilot.read(plan_path)
    if (bank.pilot.digest(plan_path) != item["plan_sha256"]
            or bank.pilot.digest(admission_path) != item["admission_sha256"]):
        raise ValueError("selected row plan/admission binding differs")
    bank.checked_policy(plan["numerical_policy"], True)
    if not plan["inferred_c_blueprint"] or not plan["clean_only"]:
        raise ValueError("joint seven-target financial task required")
    bundle_path = checked_file(root, item["bundle"])
    db = sqlite3.connect("file:" + str(bundle_path) + "?mode=ro", uri=True)
    try:
        key, row = item["source_key"], item["id"]
        identities = db.execute("SELECT value FROM identity WHERE source_key=?", (key,)).fetchall()
        if len(identities) != 1:
            raise ValueError("selected original identity missing or duplicated")
        identity = json.loads(identities[0][0])
        expected_identity = item["identity"]
        if identity != expected_identity or identity["plan_sha256"] != item["plan_sha256"] or identity["admission_sha256"] != item["admission_sha256"] or identity["sources"] != producer["source_bindings"]:
            raise ValueError("original journal identity differs from transport")
        job = identity["job"]
        if job["lane"] != "joint_unknown" or job["split"] != item["split"] or not job["start"] <= row < job["stop"]:
            raise ValueError("selected row is outside original financial job")
        draw = db.execute("SELECT value FROM draws WHERE source_key=? AND id=?", (key, row)).fetchall()
        failed = db.execute("SELECT status,summary,payload FROM rows WHERE source_key=? AND id=?", (key, row)).fetchall()
        if len(draw) != 1 or len(failed) != 1 or failed[0][0] != "failed" or failed[0][2] is not None:
            raise ValueError("exactly one retained failed row required")
        spec = json.loads(draw[0][0])
        failure = json.loads(failed[0][1])
        replay = verify_stored_spec(spec,bank.row_spec(plan,job["lane"],job["split"],row),failure)
        attempts = db.execute("SELECT block,rung,status,summary,payload FROM attempts WHERE source_key=? AND id=? ORDER BY block,rung", (key, row)).fetchall()
        for block, rung, status, summary, payload in attempts:
            value = json.loads(summary)
            entropy = [*spec["identity_entropy"], 1, bank.LANES.index(job["lane"]), block, rung]
            if (status not in ("accepted", "rejected_precision") or value["entropy"] != entropy
                    or value["paths"] != plan["numerical_policy"]["paths_ladder"][rung] or payload is None):
                raise ValueError("stored attempt RNG/path identity changed")
        return plan, identity, spec, failure, attempts, replay
    finally:
        db.close()


def prepare(manifest_path, out, *, workers=6, row_seconds=2700, expected_rows=87, nice=10,
            transport_archive=None, transport_sha256=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = bank.pilot.read(manifest_path)
    if manifest.get("format") != BUNDLE_FORMAT or len(manifest["rows"]) != expected_rows:
        raise ValueError("frozen failed-row bundle count/format required")
    if type(workers) is not int or not 1 <= workers <= 6 or type(row_seconds) is not int or not 1 <= row_seconds <= 2700 or nice != 10:
        raise ValueError("bounded six-worker/2700s/nice10 local authorization required")
    keys = [(r["producer_kind"], r["source_key"], r["id"]) for r in manifest["rows"]]
    joins = [(r["split"], r["id"]) for r in manifest["rows"]]
    if len(set(keys)) != len(keys) or len(set(joins)) != len(joins):
        raise ValueError("duplicate physical bank rows in repair request")
    sources = source_bindings(manifest["producers"])
    transport_files = {str(manifest_path): bank.pilot.digest(manifest_path)}
    replay_checks = []
    for item in manifest["rows"]:
        replay_checks.append(selected_row(manifest_path, manifest, item)[-1])
        path = checked_file(manifest_path.parent, item["bundle"])
        transport_files[str(path)] = item["bundle"]["sha256"]
    for producer in manifest["producers"].values():
        for field in ("plan", "admission"):
            path = checked_file(manifest_path.parent, producer[field])
            transport_files[str(path)] = producer[field]["sha256"]
    archive = None
    if transport_archive is not None:
        archive = binding(transport_archive)
        if archive["sha256"] != transport_sha256:
            raise ValueError("transport archive digest differs")
        transport_files[archive["path"]] = archive["sha256"]
    plan = {"format": FORMAT, "created_utc": datetime.now(timezone.utc).isoformat(),
        "transport_manifest": binding(manifest_path), "transport_archive": archive,
        "transport_files": transport_files, "rows": manifest["rows"], "source_bindings": sources,
        "runtime": runtime(), "limits": {"workers": workers, "row_seconds": row_seconds,
            "total_worker_seconds_cap": len(keys)*row_seconds, "nice": nice, "automatic_retries": False},
        "authorization": "explicit user instruction: repair frozen failed rows locally and upload separately",
        "financial_target_changed": False, "source_execution_identity_changed": True,
        "cross_platform_prior_replay":{"row_checks":replay_checks,"max_ulp":max(max(v["parameters_max_ulp"],v["context_parameters_max_ulp"]) for v in replay_checks),
            "stored_truths_preserved_exactly":True,"c_ids_entropy_exact":True},
        "join_rule": "replace only matching original plan SHA256 + bank_lane + split + id failure; never append a duplicate physical row",
        "cross_platform_scope": "Oscar blocks reused verbatim; new attempts use recorded local platform; seeds agree, floating point need not be bitwise identical",
        "SBI_training_readiness": False, "scientific_acceptance": False}
    bank.pilot.write_new(out, plan)
    return plan


def validate(plan):
    if plan.get("format") != FORMAT or plan.get("runtime") != runtime():
        raise ValueError("local producer/environment identity changed")
    limits = plan["limits"]
    if (not 1 <= limits["workers"] <= 6 or not 1 <= limits["row_seconds"] <= 2700
            or limits["nice"] != 10 or limits["automatic_retries"] is not False
            or limits["total_worker_seconds_cap"] != len(plan["rows"])*limits["row_seconds"]
            or plan["rows"] != bank.pilot.read(plan["transport_manifest"]["path"])["rows"]):
        raise ValueError("frozen local repair rows or resource limits changed")
    for path, digest in plan["transport_files"].items():
        if bank.pilot.digest(path) != digest:
            raise ValueError("frozen failed-row transport changed")
    for relative, digest in plan["source_bindings"].items():
        if bank.pilot.digest(ROOT / relative) != digest:
            raise ValueError("original/local source changed; new repair identity required")
    if plan["scientific_acceptance"] is not False or plan["SBI_training_readiness"] is not False or plan["financial_target_changed"] is not False:
        raise ValueError("development repair scope changed")


def run_row(plan_path, index, out, *, process_started=None):
    """No Slurm variables are invented. Only this newly admitted local path runs."""
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("local repair must not masquerade as a Slurm allocation")
    plan = bank.pilot.read(plan_path); validate(plan)
    if type(index) is not int or not 0 <= index < len(plan["rows"]):
        raise ValueError("frozen repair row index required")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    process_started = time.monotonic() if process_started is None else process_started
    if os.getpriority(os.PRIO_PROCESS, 0) < plan["limits"]["nice"]:
        os.nice(plan["limits"]["nice"] - os.getpriority(os.PRIO_PROCESS, 0))
    with (out / "worker.lease").open("a") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another local repair owns this output") from error
        try:
            return _run_row(plan_path, plan, index, out, process_started)
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def _run_row(plan_path, plan, index, out, started):
    manifest_path = Path(plan["transport_manifest"]["path"])
    manifest = bank.pilot.read(manifest_path); item = plan["rows"][index]
    original, original_identity, spec, failure, stored, replay = selected_row(manifest_path, manifest, item)
    identity = {"repair_plan_sha256": bank.pilot.digest(plan_path), "row_index": index,
        "original_identity": original_identity, "original_failure": failure,
        "original_split": item["split"], "original_row_id": item["id"], "producer_kind": item["producer_kind"],
        "cross_platform_prior_replay": replay,
        "runtime": plan["runtime"], "source_bindings": plan["source_bindings"]}
    db = sqlite3.connect(out / "journal.sqlite")
    db.execute("PRAGMA journal_mode=DELETE"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS identity(value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS attempts(block INTEGER,rung INTEGER,status TEXT,summary TEXT,payload BLOB,origin TEXT,PRIMARY KEY(block,rung))")
    db.execute("CREATE TABLE IF NOT EXISTS outcome(status TEXT,summary TEXT,payload BLOB)")
    found = db.execute("SELECT value FROM identity").fetchall()
    try:
        if not found:
            db.execute("INSERT INTO identity VALUES(?)", (bank.canonical(identity),))
            db.executemany("INSERT INTO attempts VALUES(?,?,?,?,?,?)", [(*r, "oscar_reused") for r in stored]); db.commit()
        elif found != [(bank.canonical(identity),)]:
            raise ValueError("repair journal row/source/configuration changed")
        if (out / "manifest.json").exists():
            result = bank.pilot.read(out / "manifest.json")
            if result["identity"] != identity or result["journal"] != binding(out / "journal.sqlite") or (result.get("payload") and result["payload"] != binding(out / "data.npz")):
                raise ValueError("completed repair payload or identity changed")
            return result
        task = original["tasks"]["joint_unknown"]; policy = original["numerical_policy"]
        curve = bank.row_curve(task, spec); parameters = dict(zip(("hurst", "eta", "rho"), spec["parameters"][:3], strict=True))
        selected = []
        from scripts.run_heston_spread_reference import Budget
        try:
            with Budget(plan["limits"]["row_seconds"]) as budget:
                for block_id, block in enumerate(task["blocks"]):
                    columns = [i for i,c in enumerate(task["design"]["contracts"]) if c["maturity"] == block["horizon"]]
                    contracts = [task["design"]["contracts"][i] for i in columns]
                    accepted = last = None
                    for rung, paths in enumerate(policy["paths_ladder"]):
                        budget.check()
                        prior = db.execute("SELECT status,summary,payload,origin FROM attempts WHERE block=? AND rung=?", (block_id,rung)).fetchone()
                        if prior:
                            if prior[0] == "accepted":
                                accepted = (columns,paths,json.loads(prior[1]),bank.unpack(prior[2]),prior[3]); break
                            if prior[0] != "rejected_precision": raise ValueError("unknown cached rung status")
                            last = (columns,paths,json.loads(prior[1]),bank.unpack(prior[2]),prior[3]); continue
                        entropy = [*spec["identity_entropy"],1,bank.LANES.index("joint_unknown"),block_id,rung]
                        diagnostic = {"capture_raw_control":True}
                        prices,errors,cov = bank.pilot.runtime_pricing.rough_block(parameters,curve,contracts,
                            horizon=block["horizon"],n_steps=block["n_steps"],n_pairs=paths//2,
                            seed=np.random.SeedSequence(entropy),diagnostics=diagnostic)
                        controls = diagnostic["raw_control"]; n = len(columns)
                        raw = np.asarray(controls["raw_put_mean"]) + np.asarray([1-c["strike"] if c["option_type"]=="call" else 0. for c in contracts])
                        moment = {"bank_covariance":np.asarray(cov).tolist(),
                            "coefficient_matrix":np.asarray(controls.get("fitted_coefficients",[])).tolist(),
                            "control_mean":np.asarray(controls.get("control_mean",[])).tolist(),
                            "variance_scope":"nominal same-sample fitted controls; independent-batch calibration is separate"}
                        result = {"controlled_prices":np.stack([prices,prices]),"raw_prices":np.stack([raw,raw]),
                            "difference":np.full(n,np.nan),"difference_covariance":np.full((n,n),np.nan),"moments":[moment,moment]}
                        metrics = bank.quality(result,np.full(n,policy["numerical_reference_forward"]),policy)
                        if rung == len(policy["paths_ladder"])-1:
                            metrics["passed"] = metrics["hardcap_passed"]
                            metrics["used_final_hardcap"] = bool(metrics["hardcap_passed"] and not metrics["preferred_passed"])
                        arrays = {k:np.asarray(v) for k,v in result.items() if k!="moments"}
                        arrays["bank_covariance"] = np.asarray([cov,cov])
                        summary = {**metrics,"paths":paths,"entropy":entropy,"moments":[moment,moment]}
                        status = "accepted" if metrics["passed"] else "rejected_precision"
                        db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?)",(block_id,rung,status,bank.canonical(summary),sqlite3.Binary(bank.pack(arrays)),"local_recomputed"));db.commit()
                        last = (columns,paths,summary,arrays,"local_recomputed")
                        if metrics["passed"]: accepted=last;break
                    selected.append(accepted if accepted is not None else last)
            prices=np.zeros(63);covariance=np.zeros((63,63));paths_used=[];origins=[]
            for columns,paths,summary,arrays,origin in selected:
                prices[columns]=arrays["controlled_prices"][1];covariance[np.ix_(columns,columns)]=arrays["bank_covariance"][1]
                paths_used.append(paths);origins.append(origin)
            h=np.asarray(task["design"]["half_spread_forward"]);factors=np.asarray(task["design"]["price_factor_DF"])
            z=np.random.default_rng(np.random.SeedSequence([*spec["identity_entropy"],2,bank.LANES.index("joint_unknown")])).standard_normal(63)
            c=spec["noise_scale"];physical=np.asarray(spec["parameters"]);qualified=all(v[2]["passed"] for v in selected)
            arrays={"physical_parameters":physical,"parameters":np.concatenate([physical,[c]]),"inference_parameters":np.concatenate([physical,[c]]),
                "noise_scale":np.asarray(c),"reference_prices_forward":prices,"reference_prices_raw":prices*factors,
                "observations_forward":prices+c*h*z,"observations_raw":(prices+c*h*z)*factors,"quote_standard_normals":z,
                "pricing_covariance_forward":covariance,"pricing_standard_errors_forward":np.sqrt(np.diag(covariance)),
                "pricing_mcse_over_h":np.sqrt(np.diag(covariance))/h,"pricing_mcse_over_observation_sd":np.sqrt(np.diag(covariance))/(c*h),
                "pricing_paths":np.asarray(paths_used,dtype=np.int64),"block_origin":np.asarray(origins),"numerical_qualified":np.asarray(qualified),
                "refinement_difference_forward":np.full(63,np.nan),"refinement_covariance_forward":np.full((63,63),np.nan),
                "refinement_evaluated":np.asarray(False),"continuous_bias_certified":np.asarray(False),
                "context_parameters":np.asarray(spec["context_parameters"]),"supplied_curve_levels":np.asarray(spec["supplied_curve_levels"])}
            status="complete" if qualified else "complete_candidate"
            db.execute("INSERT INTO outcome VALUES(?,?,?)",(status,bank.canonical({"original_failure":failure}),sqlite3.Binary(bank.pack(arrays))));db.commit()
            (out/"data.npz").write_bytes(bank.pack(arrays))
            payload=binding(out/"data.npz")
        except BaseException as error:
            status="failed";payload=None
            db.execute("INSERT INTO outcome VALUES(?,?,NULL)",(status,bank.canonical({"error_type":type(error).__name__,"error":str(error)})));db.commit()
            if not isinstance(error,Exception) or (isinstance(error,TimeoutError) and "signal 15" in str(error)):raise
        result={"format":"rough_actual_layout_local_failed_repair_row_v1","status":status,"identity":identity,
            "payload":payload,"journal":binding(out/"journal.sqlite"),"seconds":time.monotonic()-started,
            "SBI_training_readiness":False,"scientific_acceptance":False,"join_rule":plan["join_rule"]}
        bank.pilot.write_new(out/"manifest.json",result)
        return result
    finally:
        db.close()

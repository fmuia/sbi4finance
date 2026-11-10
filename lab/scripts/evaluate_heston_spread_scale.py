"""Matched joint-c posterior checks and prior-predictive development diagnostics.

Use the predeclared observation vectors for conditional checks and independently
sample c from its actual prior for the population. No fitting or pricing occurs.
Full conditional draws stay in an ignored payload; only compact population
quantiles/ranks are committed. A fixed-physical c oracle is not a joint reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_spread_baseline as baseline
from scripts import train_heston_spread_baseline as fixed
from scripts import train_heston_spread_scale as trainer
from scripts import run_heston_spread_reference as reference
from scripts import run_heston_spread_scale_reference as scale_reference
from sbi4finance.calibration import validation


SEED, ROWS, DRAWS, CONDITIONAL_DRAWS = 261112, 512, 512, 4096
FAMILY_CHECKS, WALL_SECONDS = 144, 300
CASES = ("synthetic-broad","synthetic-market-like","market")


def read(path):
    return json.loads(Path(path).read_text())


def validate_study(study, target):
    expected = study["joint_evaluation"]
    if (study["format"] != "heston_spread_noise_scale_study_v1"
            or study["joint_task"] != target or target["infer_c"] is not True
            or target["parameter_names"][-1] != "c" or target["prior_bounds"][-1] != [.5,2.]
            or expected["seed"] != SEED or expected["population_rows"] != ROWS
            or expected["posterior_draws_per_population_case"] != DRAWS
            or expected["conditional_draws"] != CONDITIONAL_DRAWS
            or expected["family_checks"] != FAMILY_CHECKS
            or expected["mean_tolerance_reference_sd"] != .3
            or expected["width_ratio90_limits"] != [.75,1.25]
            or expected["block_bootstrap_replicates"] != 400
            or expected["population_c"] != "independent Uniform(.5,2) per validation surface: draw_c(512,0,261112,namespace=3)"
            or expected["population_gaussian"] != "reuse PCG64 SeedSequence([0x53504556,261112]) standard errors; apply generated c*h"
            or expected["checkpoints"] != [10000,50000,60000]
            or expected["conditional_cases"] != list(CASES)):
        raise ValueError("joint evaluation task/population/conditional protocol differs")


def validate_training_record(metadata, study_path):
    path=Path(study_path).parent/"training-protocol.json"
    contract=trainer.protocol_contract(metadata["task"],metadata)
    declared=read(path)
    if (Path(metadata["protocol_path"]).resolve()!=path.resolve()
            or metadata["protocol_sha256"]!=fixed.digest(path)
            or metadata["recipe"]!=contract
            or any(declared.get(key)!=value for key,value in contract.items())
            or metadata["train_rows"]!=contract["train_rows"]
            or metadata["seed"]!=contract["seed"]
            or metadata["batch_size"]!=contract["batch_size"]
            or metadata["planned_updates"]!=contract["planned_updates"]):
        raise ValueError("joint checkpoint training recipe/protocol differs")
    return path


def population_observations(task, physical, clean):
    """The generating c is a target only, independent of physical truths and Z."""
    h = np.asarray(task["records"]["half_spread"])
    baseline.validate_development_arrays(physical,clean,np.asarray(task["target"]["prior_bounds"][:5]),ROWS,len(h))
    c = trainer.draw_c(ROWS,0,SEED,namespace=3)
    z = np.random.default_rng(np.random.SeedSequence([0x53504556,SEED])).standard_normal(clean.shape)
    return clean+c[:,None]*h*z, np.column_stack((physical,c))


def selected_reference_protocol(study, study_path, recovery_path=None):
    """An explicit startup-only addendum may replace output/cache identities."""
    original=study["reference_protocol"]
    original_path=Path(original["path"])
    if fixed.digest(original_path)!=original["sha256"]:
        raise ValueError("original reference protocol differs")
    if recovery_path is None:
        return original_path,{}
    recovery_path=Path(recovery_path)
    recovery=read(recovery_path)
    if (recovery["format"]!="heston_spread_scale_reference_recovery_v1"
            or Path(recovery["original_study"]["path"]).resolve()!=Path(study_path).resolve()
            or recovery["original_study"]["sha256"]!=fixed.digest(study_path)
            or recovery["original_reference_protocol"]!={"path":str(original_path.resolve()),"sha256":original["sha256"]}):
        raise ValueError("recovery addendum does not bind the original study/reference")
    bindings=[recovery["original_study"],recovery["original_reference_protocol"],recovery["original_campaign"],
        recovery["startup_failure_audit"],recovery["recovery_reference_protocol"],*recovery["sources"].values()]
    protected={str(recovery_path.resolve()):fixed.digest(recovery_path)}
    for value in bindings:
        if fixed.digest(value["path"])!=value["sha256"]:
            raise ValueError("recovery source/audit/protocol differs")
        protected[str(Path(value["path"]).resolve())]=value["sha256"]
    audit=read(recovery["startup_failure_audit"]["path"])
    if (audit.get("all_seven_zero_sampling") is not True or len(audit["jobs"])!=7
            or any(job["sampling_started"] or job["completed_sampler_attempts"]!=0
                   or job["chain_payloads"]!=0 for job in audit["jobs"])):
        raise ValueError("recovery does not establish startup-only failures")
    for job in audit["jobs"]:
        for value in job["files"].values():
            if fixed.digest(value["path"])!=value["sha256"]:
                raise ValueError("startup failure evidence changed")
            protected[str(Path(value["path"]).resolve())]=value["sha256"]
    original_record=read(original_path)
    selected_path=Path(recovery["recovery_reference_protocol"]["path"])
    selected=read(selected_path)
    excluded={"evidence_root","payload_root","jobs","environment_bootstrap_recovery"}
    if ({key:value for key,value in selected.items() if key not in excluded}
            !={key:value for key,value in original_record.items() if key not in excluded}
            or len(selected["jobs"])!=len(original_record["jobs"])):
        raise ValueError("recovery changed reference scientific settings")
    for old,new in zip(original_record["jobs"],selected["jobs"],strict=True):
        omit={"evidence_out","payload_out"}
        if ({key:value for key,value in old.items() if key not in omit}
                !={key:value for key,value in new.items() if key not in omit}):
            raise ValueError("recovery changed a reference target or observations")
    return selected_path,protected


def joint_reference(protocol, case_id, panel_path, target, products, observations):
    spec = next(item for item in protocol["jobs"] if item["case_id"] == case_id and item["mode"] == "joint")
    directory = Path(spec["evidence_out"])
    required = (directory/"run.json",directory/"diagnostics.json")
    if not all(path.is_file() for path in required):
        return None,{"status":"inconclusive","reason":"matched completed joint-c reference unavailable"}
    record, diagnostics = [read(path) for path in required]
    if record.get("status") != "completed" or not record.get("chain_payload_path"):
        return None,{"status":"inconclusive","reason":"matched joint-c reference failed or did not complete"}
    chain_path = Path(record["chain_payload_path"])
    baseline.check_reference_binding(record,diagnostics,chain_path,target=target,
        design_sha256=protocol["design_sha256"],case_id=case_id,observations=observations,
        panel_sha256=fixed.digest(panel_path),products=products,diagnostics_path=required[1])
    if record["names"] != target["parameter_names"] or record["bounds"] != target["prior_bounds"]:
        raise ValueError("joint reference names/bounds differ from neural target")
    binding={"reference_run_path":str(required[0].resolve()),
        "reference_diagnostics_path":str(required[1].resolve()),
        "reference_chain_payload_path":str(chain_path.resolve()),
        "reference_run_sha256":fixed.digest(required[0]),
        "reference_diagnostics_sha256":fixed.digest(required[1]),
        "reference_chain_payload_sha256":fixed.digest(chain_path),
        "reference_ready":diagnostics.get("reference_ready") is True}
    if not binding["reference_ready"]:
        return None,{**binding,"status":"inconclusive","reason":"reference convergence or sampled numerical precision unresolved"}
    with np.load(chain_path,allow_pickle=False) as archive:
        chains = archive["parameters"].copy()
    retained = record["attempts"][record["selected_attempt"]]["draws"]
    bounds=np.asarray(target["prior_bounds"])
    if (chains.shape != (record["settings"]["chains"],retained,6) or not np.isfinite(chains).all()
            or np.any(chains<bounds[:,0]) or np.any(chains>bounds[:,1])):
        raise ValueError("joint reference chain dimensions, support or values differ")
    return (chains,diagnostics),binding


def evaluate(checkpoint, study_path, out, payload_out, *, reference_recovery=None):
    started=time.monotonic()
    checkpoint,study_path,out,payload_out=map(Path,(checkpoint,study_path,out,payload_out))
    if out.exists() or payload_out.exists():
        raise FileExistsError("joint evaluation output identities must be new")
    study=read(study_path)
    workflow,metadata=trainer.load_checkpoint(checkpoint)
    task,target=metadata["task"],metadata["task"]["target"]
    validate_study(study,target)
    training_protocol_path=validate_training_record(metadata,study_path)
    updates=metadata["actual_optimizer_updates"]
    if (updates not in study["joint_evaluation"]["checkpoints"]
            or metadata["source_sha256"]["trainer"] != fixed.digest(trainer.__file__)
            or metadata["architecture"]["effective_dropout"] != 0.):
        raise ValueError("joint checkpoint endpoint, source or effective architecture differs")
    panel_path=Path(study["panel"]["path"])
    ref_path,recovery_protected=selected_reference_protocol(study,study_path,reference_recovery)
    if fixed.digest(panel_path)!=study["panel"]["sha256"]:
        raise ValueError("scale-study panel binding differs")
    panel,ref_protocol=read(panel_path),read(ref_path)
    if panel["joint_task"]!=target or ref_protocol["panel_sha256"]!=fixed.digest(panel_path):
        raise ValueError("joint reference panel targets a different posterior")
    # Recheck all declared source/data bindings before using any ready trace.
    scale_reference.validate_protocol(ref_protocol,ref_protocol["jobs"][0]["id"])
    bank=Path(metadata["bank_path"]);bank_record=read(bank/"bank.json")
    physical_target=task["physical_clean_price_task"]["target"]
    bank_identity=bank_record["identity"]
    payload=bank/bank_record["payload"]
    if (payload.resolve().parent!=bank.resolve() or fixed.digest(bank/"bank.json")!=metadata["bank_record_sha256"]
            or bank_identity!=metadata["bank_identity"] or bank_record["payload_sha256"]!=metadata["bank_payload_sha256"]
            or fixed.digest(payload)!=metadata["bank_payload_sha256"]
            or physical_target["prior_bounds"]!=target["prior_bounds"][:5]):
        raise ValueError("joint evaluation bank/physical projection identity differs")
    sources={"evaluator":Path(__file__),"trainer":Path(trainer.__file__),"fixed_helper":Path(fixed.__file__),
        "comparison_helper":Path(baseline.__file__),"validation":Path(validation.__file__),
        "reference_producer":Path(reference.__file__),"reference_consumer":Path(scale_reference.__file__),
        "comparison_metrics":Path(baseline.heston_budget_diagnostics.__file__)}
    sources.update({"reference_"+name:Path(value["path"]) for name,value in reference.source_hashes().items()})
    source_hashes={name:{"path":str(path.resolve()),"sha256":fixed.digest(path)} for name,path in sources.items()}
    protected={str(path.resolve()):fixed.digest(path) for path in
        [study_path,training_protocol_path,panel_path,ref_path,checkpoint/"checkpoint.json",checkpoint/"model.keras",checkpoint/"state.npz",bank/"bank.json",payload,*sources.values()]}
    protected.update(recovery_protected)
    out.mkdir(parents=True);payload_out.mkdir(parents=True)
    record={"format":"heston_spread_joint_scale_evaluation_v1","status":"started","checkpoint":str(checkpoint.resolve()),
        "checkpoint_record_sha256":fixed.digest(checkpoint/"checkpoint.json"),"model_sha256":metadata["model_sha256"],
        "state_sha256":metadata["state_sha256"],"actual_optimizer_updates":updates,"task":target,
        "study_protocol_sha256":fixed.digest(study_path),"reference_protocol_sha256":fixed.digest(ref_path),
        "reference_protocol_path":str(ref_path.resolve()),
        "reference_recovery":None if reference_recovery is None else {"path":str(Path(reference_recovery).resolve()),"sha256":fixed.digest(reference_recovery)},
        "panel_sha256":fixed.digest(panel_path),"source_sha256":source_hashes,
        "seed":SEED,"family_checks":FAMILY_CHECKS,"cases":{},"scientific_acceptance":False,"reserved_arrays_opened":False}
    fixed.write_json(out/"run.json",record)
    def deadline():
        if time.monotonic()-started>=WALL_SECONDS:
            raise TimeoutError("joint evaluation wall budget exhausted")
    conditional_draws={}
    try:
        for index,case_id in enumerate(CASES):
            deadline()
            _,_,_,y=reference.load_case(panel_path,case_id,infer_c=True)
            neural=trainer.sample(workflow,task,y[None],CONDITIONAL_DRAWS,seed=SEED+index)[0]
            support=baseline.support_report(neural,np.asarray(target["prior_bounds"]))
            if support["material_outside_value_count"]:
                raise ValueError("joint conditional neural support violation")
            conditional_draws[case_id]=neural
            entry={"role":"synthetic development fidelity" if case_id!="market" else "separate market application diagnostic",
                "observation_sha256":baseline.observations_digest(y),"neural_support":support,
                "neural_quantiles_05_50_95":np.quantile(neural,[.05,.5,.95],axis=0).tolist(),
                "c_mean":float(neural[:,-1].mean()),"c_sd":float(neural[:,-1].std(ddof=1))}
            selected,binding=joint_reference(ref_protocol,case_id,panel_path,target,panel["products"],y)
            entry.update(binding)
            for name in ("reference_run","reference_diagnostics","reference_chain_payload"):
                if name+"_path" in binding:
                    protected[binding[name+"_path"]]=binding[name+"_sha256"]
            if selected is not None:
                chains,diagnostics=selected
                entry["comparison"]=baseline.compare_draws(neural,chains,diagnostics,target["parameter_names"],panel["products"],
                    seed=SEED+100+index,replicates=400,family_checks=FAMILY_CHECKS)
                entry["status"]=entry["comparison"]["status"]
            record["cases"][case_id]=entry
        fixed.write_npz(payload_out/"conditional-draws.npz",conditional_draws)
        record["conditional_draws"]={"path":str((payload_out/"conditional-draws.npz").resolve()),
            "sha256":fixed.digest(payload_out/"conditional-draws.npz"),"draws_per_case":CONDITIONAL_DRAWS}
        with np.load(payload,allow_pickle=False) as archive:
            truth=archive["validation_parameters"][:ROWS].copy()
            clean=archive["validation_clean"][:ROWS].copy()
        noisy,truth6=population_observations(task,truth,clean)
        draws=np.concatenate([trainer.sample(workflow,task,noisy[start:start+16],DRAWS,seed=SEED+1000+start)
            for start in range(0,ROWS,16)])
        deadline()
        support=baseline.support_report(draws,np.asarray(target["prior_bounds"]))
        if support["material_outside_value_count"]:
            raise ValueError("joint population neural support violation")
        ranks=validation.sbc(draws,truth6,seed=SEED+2000)
        fractional=ranks.pop("fractional_ranks")
        quantiles=np.quantile(draws,[.05,.5,.95],axis=1)
        fixed.write_npz(out/"recovery.npz",{"truth":truth6,"quantiles":quantiles.astype(np.float32),
            "row_id":np.array([f"validation:{i:08d}" for i in range(ROWS)])})
        record["population"]={"rows":ROWS,"draws_per_case":DRAWS,"parameter_names":target["parameter_names"],
            "support":support,"rank_dkw":ranks,"fractional_ranks":fractional,
            "coverage":validation.coverage(draws,truth6),"recovery_sha256":fixed.digest(out/"recovery.npz"),
            "observations_sha256":baseline.observations_digest(noisy),"truth_sha256":baseline.observations_digest(truth6),
            "generating_c":{"prior":"independent Uniform(.5,2)","namespace":3,"count":ROWS},
            "scope":"Reused development physical rows; generating c is never a condition. Whole-prior SBC only, no rank-uniformity assertion conditional on a c-truth bin."}
        statuses=[entry["status"] for entry in record["cases"].values()]
        synthetic=[record["cases"][case]["status"] for case in CASES[:2]]
        combine=lambda values:"fail" if "fail" in values else "pass" if all(v=="pass" for v in values) else "inconclusive"
        record.update(status="complete",conditional_status=combine(statuses),synthetic_conditional_status=combine(synthetic),seconds=time.monotonic()-started)
        if any(fixed.digest(path)!=value for path,value in protected.items()):
            raise ValueError("a protected input/source changed during joint evaluation")
        deadline();record["protected_inputs_unchanged"]=True
        fixed.write_json(out/"evaluation.json",record)
        fixed.write_json(out/"run.json",record)
        return record
    except BaseException as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
        fixed.write_json(out/"failure.json",record);fixed.write_json(out/"run.json",record)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","study","out","payload-out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--reference-recovery",type=Path)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        result=evaluate(args.checkpoint,args.study,args.out,args.payload_out,reference_recovery=args.reference_recovery)
        print(json.dumps({"out":str(args.out),"synthetic_conditional_status":result["synthetic_conditional_status"]}))
        return
    if args.out.exists() or args.payload_out.exists():
        raise FileExistsError("evaluation output identities must be new")
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],
        env=dict(os.environ,KERAS_BACKEND="jax",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
    try:
        code=process.wait(timeout=WALL_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill();process.wait();args.out.mkdir(parents=True,exist_ok=True)
        fixed.write_json(args.out/"hard-timeout.json",{"status":"killed_at_wall_limit","seconds":WALL_SECONDS,"automatic_retry":False})
        raise TimeoutError("joint evaluation supervisor cap reached")
    if code:raise SystemExit(code)


if __name__=="__main__":main()

"""Precision-safe replay of the paired 100k-bank Heston continuations.

The two batch sizes see exactly the same per-epoch row order, c and Gaussian
errors. Epochs mean complete bank passes, including an unpadded tail batch.
All parent arrays and Adam slots are restored before an explicit LR restart.
Only the forward bounded-target logit is evaluated on the host in float64,
before casting its finite result to float32. Prior draws and inverse sampling
are unchanged. The original failed producer and artifacts remain immutable.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_heston_spread_scale as scale
from scripts import continue_heston_spread_baseline as full_state
from scripts import continue_heston_spread_scale as original

fixed = scale.fixed
FROZEN_SCALE_SHA256 = "93714493768c1db7ce792b931f1dd994e2a5936bafd5da6b1062b666ac4b9068"
FROZEN_EXPANSION_SHA256 = "5d712d9b37dba7a5662c67ccc84e4148b58a45e328815a07815c40dda7109892"
MAX_WALL_SECONDS = 5400
ENDPOINTS = (25,100,250,500)
EPOCH_NAMESPACE, PERMUTATION_NAMESPACE = 501, 0x45504f43
PROCESS_STARTED = time.monotonic()


def schedule_class():
    return original.schedule_class()


def epoch_order(rows,epoch,seed=9011):
    return original.epoch_order(rows,epoch,seed)


def epoch_data(train,halfspread,epoch,seed=9011):
    return original.epoch_data(train,halfspread,epoch,seed)


def validate_parent(record,*,smoke_test_only=False):
    if fixed.digest(original.__file__)!=FROZEN_EXPANSION_SHA256:
        raise ValueError("original expansion helper differs")
    return original.validate_parent(record,smoke_test_only=smoke_test_only)


def load_bank(bank,task,*,smoke_test_only=False):
    return original.load_bank(bank,task,smoke_test_only=smoke_test_only)


def protocol_contract(checkpoint,bank,batch_size,*,smoke_test_only=False):
    parent=json.loads((Path(checkpoint)/"checkpoint.json").read_text())
    validate_parent(parent,smoke_test_only=smoke_test_only)
    contract=original.protocol_contract(checkpoint,bank,batch_size,smoke_test_only=smoke_test_only)
    return {**contract,"format":"heston_spread_scale_precision_expansion_protocol_v1",
        "source_sha256":{**contract["source_sha256"],"original_expansion":FROZEN_EXPANSION_SHA256,
                         "continuation":fixed.digest(__file__)},
        "precision_recipe":{"forward_targets":"NumPy float64 log(x-lower)-log(upper-x), then float32",
            "conditions":"frozen Y/half_spread, float32; same as original adapter",
            "inverse_targets":"unchanged original BayesFlow bounded sigmoid adapter",
            "clipping":False,"resampling":False,"target_prior_changed":False,
            "replay_from_same_original_parent":True,"paired_epoch_stream_unchanged":True}}


def host_adapt_training(task,raw):
    """Preserve near-boundary target distances until after taking logarithms."""
    bounds=np.asarray(task["target"]["prior_bounds"],dtype=np.float64)
    names=task["target"]["parameter_names"]
    if set(raw)!={"observations",*names}:
        raise ValueError("complete six-target training data required")
    values=np.concatenate([np.asarray(raw[name],dtype=np.float64) for name in names],axis=-1)
    conditions=np.asarray(raw["observations"],dtype=np.float64)
    if (values.ndim!=2 or values.shape[1]!=6 or conditions.shape!=(len(values),54)
            or not np.isfinite(values).all() or not np.isfinite(conditions).all()
            or np.any(values<=bounds[:,0]) or np.any(values>=bounds[:,1])):
        raise ValueError("finite targets strictly inside their unchanged priors required")
    targets=(np.log(values-bounds[:,0])-np.log(bounds[:,1]-values)).astype(np.float32)
    result={"inference_variables":targets,"inference_conditions":conditions.astype(np.float32)}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("precision-safe adapted tensors must be finite")
    return result


def save_checkpoint(directory,workflow,state,run,updates,probes):
    directory.mkdir(parents=True,exist_ok=False)
    fixed.assign_state(workflow.estimator,state)
    workflow.estimator.save(directory/"temporary.keras")
    (directory/"temporary.keras").replace(directory/"model.keras")
    fixed.save_state(directory/"state.npz",state)
    record={**run,"format":"heston_spread_scale_precision_checkpoint_v1","actual_optimizer_updates":updates,
        "optimizer_schedule":workflow.estimator.optimizer.get_config()["learning_rate"],
        "model_sha256":fixed.digest(directory/"model.keras"),"state_sha256":fixed.digest(directory/"state.npz"),
        "state":fixed.state_summary(workflow.estimator,state),"probe_metrics":probes,"resume_supported":False}
    fixed.write_json(directory/"checkpoint.json",record)


def load_checkpoint(directory):
    directory=Path(directory)
    record=json.loads((directory/"checkpoint.json").read_text())
    continuation=record.get("continuation_protocol",{})
    scale.validate_task(record["task"])
    if (record.get("format")!="heston_spread_scale_precision_checkpoint_v1"
            or fixed.fingerprint(record["task"])!=record["task_fingerprint"]
            or fixed.digest(directory/"model.keras")!=record["model_sha256"]
            or fixed.digest(directory/"state.npz")!=record["state_sha256"]
            or record["names"]!=record["task"]["target"]["parameter_names"]
            or record["bounds"]!=record["task"]["target"]["prior_bounds"]
            or continuation.get("format")!="heston_spread_scale_precision_expansion_protocol_v1"
            or continuation["source_sha256"]["continuation"]!=fixed.digest(__file__)
            or continuation["source_sha256"]["original_expansion"]!=fixed.digest(original.__file__)
            or continuation["source_sha256"]["frozen_scale"]!=fixed.digest(scale.__file__)
            or continuation["source_sha256"]["frozen_fixed"]!=fixed.digest(fixed.__file__)
            or continuation["source_sha256"]["full_state_helper"]!=fixed.digest(full_state.__file__)     
            or record["protocol_sha256"]!=fixed.digest(record["protocol_path"])
            or record["task_fingerprint"]!=continuation["task_fingerprint"]
            or record["batch_size"]!=continuation["batch_size"]
            or record["train_rows"]!=continuation["train_rows"]
            or record["bank_path"]!=continuation["bank_path"]
            or record["completed_bank_passes"] not in continuation["checkpoint_bank_passes"]
            or record["continuation_source_sha256"]!=continuation["source_sha256"]["continuation"]
            or record["actual_optimizer_updates"]!=continuation["parent_actual_optimizer_updates"]
                +record["completed_bank_passes"]*continuation["batches_per_pass"]):
        raise ValueError("expanded checkpoint source/protocol/endpoint differs")
    full_state.validate_bank_binding(record,continuation)
    protocol=selected_contract(json.loads(Path(record["protocol_path"]).read_text()),record["batch_size"])
    if any(protocol.get(key)!=value for key,value in continuation.items()):
        raise ValueError("expanded checkpoint no longer matches its persisted protocol")
    schedule_class()
    workflow=scale.build_workflow(record["task"])
    workflow.load_estimator(directory/"model.keras")
    fixed.switch_dropout_zero(workflow.estimator,reloading=True)
    full_state.restore_full_state(workflow,record,directory/"state.npz")
    if (int(workflow.estimator.optimizer.iterations.numpy())!=record["actual_optimizer_updates"]
            or workflow.estimator.optimizer.get_config()["learning_rate"]!=record["optimizer_schedule"]):
        raise ValueError("precision checkpoint optimizer iteration/schedule differs")
    return workflow,record


def selected_contract(protocol,batch_size):
    if protocol.get("format")!="heston_spread_scale_precision_recovery_study_v1":
        raise ValueError("the combined predeclared capacity-study protocol is required")
    arms=[arm for arm in protocol.get("arms",[]) if arm.get("batch_size")==batch_size]
    if len(arms)!=1:
        raise ValueError("protocol must contain exactly one selected batch-size arm")
    return arms[0]["contract"]


def install_schedule(model,state,contract):
    fixed.assign_state(model,state)
    before=fixed.state_summary(model,fixed.current_state(model))
    old=model.optimizer.get_config()
    optimizer=model.optimizer
    model.optimizer.learning_rate=schedule_class()(contract["train_rows"],contract["batch_size"],
        contract["parent_actual_optimizer_updates"],contract["schedule"]["decay_bank_passes"])
    after=fixed.state_summary(model,fixed.current_state(model))
    new=model.optimizer.get_config()
    if model.optimizer is not optimizer or before!=after or {k:v for k,v in old.items() if k!="learning_rate"}!={k:v for k,v in new.items() if k!="learning_rate"}:
        raise RuntimeError("LR restart changed optimizer or parent arrays")
    return {"all_array_hashes_unchanged":True,"optimizer_object_preserved":True,
            "old_schedule":old["learning_rate"],"new_schedule":new["learning_rate"]}


def run_continuation(checkpoint,bank,out,*,protocol,batch_size,smoke_test_only=False):
    started=time.monotonic()
    out,checkpoint=Path(out),Path(checkpoint)
    contract=protocol_contract(checkpoint,bank,batch_size,smoke_test_only=smoke_test_only)
    combined=json.loads(Path(protocol).read_text())
    declared=selected_contract(combined,batch_size)
    if any(declared.get(key)!=value for key,value in contract.items()):
        raise ValueError("persist the exact expansion protocol before training")
    arm=next(arm for arm in combined["arms"] if arm["batch_size"]==batch_size)
    if Path(arm["output"]).resolve()!=out.resolve():
        raise ValueError("arm output differs from the predeclared combined protocol")
    if out.exists():
        raise FileExistsError(out)
    os.environ.setdefault("KERAS_BACKEND","jax")
    import bayesflow
    import jax
    import keras
    keras.utils.set_random_seed(contract["seed"])
    workflow,parent=scale.load_checkpoint(checkpoint)
    validate_parent(parent,smoke_test_only=smoke_test_only)
    model=workflow.estimator
    state,restoration=full_state.restore_full_state(workflow,parent,checkpoint/"state.npz")
    if (bool(jax.config.x64_enabled)!=parent["numeric_configuration"]["jax_enable_x64"]
            or keras.backend.floatx()!=parent["numeric_configuration"]["keras_floatx"]):
        raise ValueError("parent float configuration differs")
    task=parent["task"]
    data,binding=load_bank(bank,task,smoke_test_only=smoke_test_only)
    full_state.validate_bank_binding(parent,binding)
    probes,probe_raw={},{}
    for name,namespace in (("train",1),("validation",2)):
        count=parent["probe_rows"][name]
        quotes,targets=scale.augmented_batch(data[name]["clean"][:count],data[name]["parameters"][:count],
            task["records"]["half_spread"],0,9011,namespace=namespace)
        probe_raw[name]=scale.training_data(task,quotes,targets)
        probes[name]=(quotes,workflow.adapter(probe_raw[name]))
    def probe_values():
        fixed.assign_state(model,state)
        return {name+"_probe_loss":float(np.asarray(model.compute_metrics(**adapted,stage="validation")["loss"]))
                for name,(_,adapted) in probes.items()}
    legacy_initial=probe_values()
    if any(legacy_initial[key]!=parent["probe_metrics"][key] for key in legacy_initial):
        raise ValueError("restored parent deterministic probes differ")
    for group,values in zip(fixed.GROUPS,fixed.current_state(model),strict=True):
        if [fixed.array_summary(v)["sha256"] for v in values]!=[r["sha256"] for r in parent["state"][group]]:
            raise ValueError("validation probe changed restored state")
    probes={name:(quotes,host_adapt_training(task,probe_raw[name])) for name,(quotes,_) in probes.items()}
    initial=probe_values()
    if not all(np.isfinite(value) for value in initial.values()):
        raise FloatingPointError("precision-safe initial probe must be finite")
    schedule_proof=install_schedule(model,state,contract)
    out.mkdir(parents=True)
    run={**parent,"format":"heston_spread_scale_precision_run_v1",**binding,"train_rows":contract["train_rows"],
        "batch_size":batch_size,"planned_updates":parent["actual_optimizer_updates"]+contract["additional_optimizer_updates"],
        "continuation_protocol":contract,"protocol_path":str(Path(protocol).resolve()),"protocol_sha256":fixed.digest(protocol),
        "continuation_source_sha256":fixed.digest(__file__),"restoration":restoration,"schedule_restart":schedule_proof,
        "source_sha256":contract["source_sha256"],"precision_recipe":contract["precision_recipe"],
        "parent_recipe":parent.get("recipe"),"recipe":contract,"max_wall_seconds":MAX_WALL_SECONDS,
        "parent_probe_restoration":{"original_adapter_probe_losses_exact":True,"original_adapter_initial":legacy_initial,
            "precision_safe_initial":initial,"same_full_parent_arrays":True},
        "parent_actual_optimizer_updates":parent["actual_optimizer_updates"],
        "continuation_git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        "checkpoint_bank_passes":contract["checkpoint_bank_passes"],
        "checkpoint_updates":[parent["actual_optimizer_updates"]+p*contract["batches_per_pass"] for p in contract["checkpoint_bank_passes"]],
        "scientific_acceptance":False,"resume_supported":False}
    fixed.write_json(out/"run.json",run)
    fixed.write_json(out/"initial-probes.json",initial)
    fixed.write_json(out/"parent-probes.json",legacy_initial)
    step_fn=fixed.compiled_step(model)
    curve,visits,last_checkpoint=[],np.zeros(contract["train_rows"],dtype=np.int64),None
    step=parent["actual_optimizer_updates"]
    phase,fields,indices,quotes,raw,adapted,metrics,candidate,gradients="before_update",[],[],[],{},{},{},None,None
    try:
        for epoch in range(contract["additional_bank_passes"]):
            phase="epoch_augmentation"
            order=epoch_order(contract["train_rows"],epoch,contract["seed"])
            all_quotes,all_targets=epoch_data(data["train"],task["records"]["half_spread"],epoch,contract["seed"])
            for offset in range(0,contract["train_rows"],batch_size):
                if time.monotonic()-started>=MAX_WALL_SECONDS:
                    raise TimeoutError("expanded training wall limit exhausted")
                phase,fields="batch_adaptation",[]
                indices=order[offset:offset+batch_size]
                quotes=all_quotes[indices]
                raw=scale.training_data(task,quotes,all_targets[indices])
                adapted=host_adapt_training(task,raw)
                fields=["adapted_input:"+key for key,value in adapted.items() if not np.isfinite(value).all()]
                if fields:
                    raise FloatingPointError("nonfinite expanded adapted batch")
                phase="optimizer_update"
                metrics,candidate,gradients,flags=step_fn(state,adapted)
                fields=[key for key,value in jax.device_get(flags).items() if not bool(value)]
                if fields:
                    raise FloatingPointError("nonfinite expanded update: "+", ".join(fields))
                if int(np.asarray(candidate[2][0]))!=step+1:
                    raise RuntimeError("expanded optimizer count differs")
                state=candidate; step+=1; visits[indices]+=1
            phase="fixed_probe"
            point={"updates":step,"completed_bank_passes":epoch+1,"additional_presentations":int(visits.sum()),
                "seconds":time.monotonic()-started,"batch_training_loss":float(np.asarray(metrics["loss"])),**probe_values()}
            if not all(np.isfinite(v) for v in point.values()):
                raise FloatingPointError("nonfinite expanded fixed probe")
            curve.append(point)
            fixed.write_json(out/"learning-curve.json",curve)
            if (epoch+1)%5==0 or epoch+1 in contract["checkpoint_bank_passes"]:
                print(json.dumps(point),flush=True)
            if epoch+1 in contract["checkpoint_bank_passes"]:
                phase="checkpoint_and_reload"
                destination=out/"checkpoints"/f"pass-{epoch+1:04d}"
                save_checkpoint(destination,workflow,state,{**run,"completed_bank_passes":epoch+1},step,point)
                before=scale.sample(workflow,task,probes["validation"][0][:2],16,9022)
                restored,_=load_checkpoint(destination)
                after=scale.sample(restored,task,probes["validation"][0][:2],16,9022)
                if not np.array_equal(before,after):
                    raise RuntimeError("expanded checkpoint changed seeded posterior draws")
                last_checkpoint=str(destination.resolve())
            fixed.write_json(out/"progress.json",{"status":"training","completed_updates":step,"completed_bank_passes":epoch+1,
                "additional_presentations":int(visits.sum()),"seconds":time.monotonic()-started,"last_checkpoint":last_checkpoint})
        if not np.array_equal(visits,np.full(contract["train_rows"],contract["additional_bank_passes"])):
            raise RuntimeError("expanded row exposure accounting differs")
        fixed.write_npz(out/"exposure.npz",{"row_id":data["train"]["row_id"],"continuation_presentations":visits})
        result={"status":"complete","completed_updates":step,"completed_bank_passes":contract["additional_bank_passes"],
            "continued_updates":step-parent["actual_optimizer_updates"],"continued_row_presentations":int(visits.sum()),
            "seconds":time.monotonic()-started,"same_seed_reload_exact":True,"all_parent_array_hashes_restored":True,
            "last_checkpoint":last_checkpoint,"reserved_arrays_opened":False,"scientific_acceptance":False}
        fixed.write_json(out/"training.json",result); fixed.write_json(out/"progress.json",result)
        return result
    except BaseException as error:
        if phase!="optimizer_update":
            candidate,gradients=None,None
        failure=fixed.capture_failure(out,model,state,candidate,gradients,step=step,indices=indices,quotes=quotes,
            raw=raw,adapted=adapted,fields=fields,metrics=metrics,error=error,last_checkpoint=last_checkpoint,phase=phase)
        fixed.write_json(out/"progress.json",{"status":"failed","completed_updates":failure["completed_finite_updates"],
            "failure":"failure/failure.json","automatic_retry":False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","bank","out","protocol"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--batch-size",type=int,required=True)
    parser.add_argument("--smoke-test-only",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(run_continuation(args.checkpoint,args.bank,args.out,protocol=args.protocol,
            batch_size=args.batch_size,smoke_test_only=args.smoke_test_only)))
    else:
        if args.out.exists():
            raise FileExistsError(args.out)
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"])
        try:
            code=process.wait(timeout=max(.001,MAX_WALL_SECONDS-(time.monotonic()-PROCESS_STARTED)))
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); args.out.mkdir(parents=True,exist_ok=True)
            fixed.write_json(args.out/"hard-timeout.json",{"status":"killed_at_5400s_limit","automatic_retry":False})
            raise TimeoutError("expanded training supervisor wall limit exhausted")
        if code:
            raise SystemExit(code)


if __name__=="__main__":
    main()

"""Bounded paired 100k-bank continuations of the frozen six-target Heston fit.

The two batch sizes see exactly the same per-epoch row order, c and Gaussian
errors. Epochs mean complete bank passes, including an unpadded tail batch.
All parent arrays and Adam slots are restored before an explicit LR restart.
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

fixed = scale.fixed
FROZEN_SCALE_SHA256 = "93714493768c1db7ce792b931f1dd994e2a5936bafd5da6b1062b666ac4b9068"
MAX_WALL_SECONDS = 5400
ENDPOINTS = (25,100,250,500)
EPOCH_NAMESPACE, PERMUTATION_NAMESPACE = 501, 0x45504f43
PROCESS_STARTED = time.monotonic()


def schedule_class():
    """Register before deserializing native Keras archives; no new optimizer."""
    import keras
    registered = keras.saving.get_registered_object("sbi4finance>SpreadExposureCosine")
    if registered is not None:
        return registered

    @keras.saving.register_keras_serializable(package="sbi4finance")
    class SpreadExposureCosine(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self,train_rows,batch_size,start_updates,decay_passes=400,
                     maximum=5e-5,minimum=5e-6):
            self.train_rows,self.batch_size = int(train_rows),int(batch_size)
            self.start_updates,self.decay_passes = int(start_updates),int(decay_passes)
            self.maximum,self.minimum = float(maximum),float(minimum)
            if min(self.train_rows,self.batch_size,self.decay_passes)<=0 or not 0<self.minimum<=self.maximum:
                raise ValueError("invalid exposure schedule")
        def __call__(self,step):
            ops = keras.ops
            relative = ops.maximum(ops.cast(step,"int32")-self.start_updates,0)
            batches = (self.train_rows+self.batch_size-1)//self.batch_size
            examples = (relative//batches)*self.train_rows+ops.minimum((relative%batches)*self.batch_size,self.train_rows)
            fraction = ops.minimum(ops.cast(examples,"float32")/(self.decay_passes*self.train_rows),1.)
            return self.minimum+(self.maximum-self.minimum)*.5*(1.+ops.cos(np.pi*fraction))
        def get_config(self):
            return {key:getattr(self,key) for key in ("train_rows","batch_size","start_updates","decay_passes","maximum","minimum")}
    return SpreadExposureCosine


def epoch_order(rows,epoch,seed=9011):
    if min(rows,epoch+1)<=0:
        raise ValueError("positive rows and nonnegative epoch required")
    return np.random.default_rng(np.random.SeedSequence([PERMUTATION_NAMESPACE,seed,epoch])).permutation(rows)


def epoch_data(train,halfspread,epoch,seed=9011):
    # Generate once per complete pass, then slice by the shared permutation.
    return scale.augmented_batch(train["clean"],train["parameters"],halfspread,epoch,seed,namespace=EPOCH_NAMESPACE)


def validate_parent(record,*,smoke_test_only=False):
    scale.validate_task(record["task"])
    if (fixed.digest(scale.__file__)!=FROZEN_SCALE_SHA256
            or fixed.digest(fixed.__file__)!=scale.FROZEN_FIXED_SHA256
            or record.get("format")!="heston_spread_scale_checkpoint_v1"
            or record["source_sha256"]["trainer"]!=FROZEN_SCALE_SHA256
            or record["task_fingerprint"]!=fixed.fingerprint(record["task"])
            or record["architecture"]!={"family":"spline","depth":6,"widths":[128,128],"construction_dropout":.05,"effective_dropout":0.}
            or record["seed"]!=9011 or record["batch_size"]!=128
            or (not smoke_test_only and (record["actual_optimizer_updates"]!=60000 or record["train_rows"]!=20000))):
        raise ValueError("continuation requires the unchanged frozen joint-scale parent")
    if smoke_test_only and not 1<=record["actual_optimizer_updates"]<=5:
        raise ValueError("software smoke parent must have at most five updates")
    if not smoke_test_only:
        config=record["optimizer_schedule"]
        if (config["class_name"]!="PiecewiseConstantDecay" or config["config"]["boundaries"]!=[54999]
                or config["config"]["values"]!=[5e-6,5e-7]):
            raise ValueError("production parent must be the completed annealed endpoint")
    lab=Path(__file__).resolve().parents[1]
    spread_path=Path(sys.modules[scale.HestonSpreadPilot.__module__].__file__)
    source_paths={"spread_pilot":spread_path,"heston_pilot":spread_path.with_name("heston_pilot.py"),
                  "batch_plan":Path(sys.modules[fixed.BatchPlan.__module__].__file__)}
    if any(fixed.digest(path)!=record["source_sha256"][name] for name,path in source_paths.items()):
        raise ValueError("parent mathematical/data-order source differs")
    if any(fixed.digest(lab/name)!=value for name,value in record["environment_sha256"].items()):
        raise ValueError("parent locked environment files differ")
    for name,version in record["versions"].items():
        actual=platform.python_version() if name=="python" else metadata.version(name)
        if actual!=version:
            raise ValueError("parent numerical environment version differs: "+name)


def load_bank(bank,task,*,smoke_test_only=False):
    """Production validation remains active; expand its already audited prefix."""
    directory=Path(bank)
    data,binding=scale.load_clean_bank(directory,task,128 if smoke_test_only else 20000,smoke_test_only=smoke_test_only)
    record=json.loads((directory/"bank.json").read_text())
    if (fixed.digest(directory/"bank.json")!=binding["bank_record_sha256"]
            or fixed.digest(directory/record["payload"])!=binding["bank_payload_sha256"]):
        raise ValueError("bank changed after frozen production validation")
    with np.load(directory/record["payload"],allow_pickle=False) as archive:
        full={key:archive["train_"+key].copy() for key in ("parameters","clean","row_id")}
    count=binding["bank_identity"]["train_rows"]
    if (full["parameters"].shape!=(count,5) or full["clean"].shape!=(count,54)
            or full["row_id"].shape!=(count,) or not np.isfinite(full["clean"]).all()
            or not np.array_equal(full["row_id"],np.array([f"train:{i:08d}" for i in range(count)]))):
        raise ValueError("full training-bank arrays or row identities differ")
    fixed.training_data(task["physical_clean_price_task"],full["clean"],full["parameters"])
    for key in full:
        if not np.array_equal(full[key][:len(data["train"][key])],data["train"][key]):
            raise ValueError("expanded bank does not preserve the validated prefix")
    if not smoke_test_only and count!=100000:
        raise ValueError("production expansion requires exactly 100k train rows")
    data["train"]=full
    return data,binding


def protocol_contract(checkpoint,bank,batch_size,*,smoke_test_only=False):
    checkpoint=Path(checkpoint)
    parent=json.loads((checkpoint/"checkpoint.json").read_text())
    validate_parent(parent,smoke_test_only=smoke_test_only)
    if batch_size not in ((32,64) if smoke_test_only else (128,512)):
        raise ValueError("only the declared two batch sizes are supported")
    data,binding=load_bank(bank,parent["task"],smoke_test_only=smoke_test_only)
    full_state.validate_bank_binding(parent,binding)
    rows=len(data["train"]["clean"])
    endpoints=(1,2) if smoke_test_only else ENDPOINTS
    batches=(rows+batch_size-1)//batch_size
    return {"format":"heston_spread_scale_expansion_protocol_v1","predeclared_before_fit":True,
        "task_fingerprint":parent["task_fingerprint"],"design_sha256":parent["design_sha256"],
        "parent_checkpoint_path":str(checkpoint.resolve()),"parent_checkpoint_sha256":fixed.digest(checkpoint/"checkpoint.json"),
        "parent_model_sha256":parent["model_sha256"],"parent_state_sha256":parent["state_sha256"],
        "parent_actual_optimizer_updates":parent["actual_optimizer_updates"],
        **binding,"train_rows":rows,"batch_size":batch_size,"batches_per_pass":batches,
        "tail_batch_size":rows-(batches-1)*batch_size,"additional_bank_passes":endpoints[-1],
        "checkpoint_bank_passes":list(endpoints),"additional_optimizer_updates":batches*endpoints[-1],
        "additional_presentations":rows*endpoints[-1],"seed":9011,
        "augmentation_namespace":EPOCH_NAMESPACE,"permutation_namespace":PERMUTATION_NAMESPACE,
        "paired_examples_across_batch_sizes":True,"row_order":"same full per-pass permutation; unpadded final tail",
        "schedule":{"class_name":"SpreadExposureCosine","maximum":5e-5,"minimum":5e-6,
                    "decay_bank_passes":1 if smoke_test_only else 400,"budget_axis":"exact cumulative additional presentations"},
        "source_sha256":{"continuation":fixed.digest(__file__),"frozen_scale":FROZEN_SCALE_SHA256,
            "frozen_fixed":scale.FROZEN_FIXED_SHA256,"full_state_helper":fixed.digest(full_state.__file__)},
        "versions":parent["versions"],"environment_sha256":parent["environment_sha256"],
        "max_wall_seconds":MAX_WALL_SECONDS,"reserved_arrays_opened":False,"new_pricing_labels":False,
        "automatic_extension":False,"scientific_acceptance":False,"smoke_test_only":smoke_test_only}


def load_checkpoint(directory):
    directory=Path(directory)
    record=json.loads((directory/"checkpoint.json").read_text())
    continuation=record.get("continuation_protocol",{})
    if (continuation.get("format")!="heston_spread_scale_expansion_protocol_v1"
            or continuation["source_sha256"]["continuation"]!=fixed.digest(__file__)
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
    workflow,loaded=scale.load_checkpoint(directory)
    full_state.restore_full_state(workflow,loaded,directory/"state.npz")
    return workflow,loaded


def selected_contract(protocol,batch_size):
    if protocol.get("format")!="heston_spread_scale_capacity_study_v1":
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
    probes={}
    for name,namespace in (("train",1),("validation",2)):
        count=parent["probe_rows"][name]
        quotes,targets=scale.augmented_batch(data[name]["clean"][:count],data[name]["parameters"][:count],
            task["records"]["half_spread"],0,9011,namespace=namespace)
        probes[name]=(quotes,workflow.adapter(scale.training_data(task,quotes,targets)))
    def probe_values():
        fixed.assign_state(model,state)
        return {name+"_probe_loss":float(np.asarray(model.compute_metrics(**adapted,stage="validation")["loss"]))
                for name,(_,adapted) in probes.items()}
    initial=probe_values()
    if any(initial[key]!=parent["probe_metrics"][key] for key in initial):
        raise ValueError("restored parent deterministic probes differ")
    for group,values in zip(fixed.GROUPS,fixed.current_state(model),strict=True):
        if [fixed.array_summary(v)["sha256"] for v in values]!=[r["sha256"] for r in parent["state"][group]]:
            raise ValueError("validation probe changed restored state")
    schedule_proof=install_schedule(model,state,contract)
    out.mkdir(parents=True)
    run={**parent,"format":"heston_spread_scale_run_v1",**binding,"train_rows":contract["train_rows"],
        "batch_size":batch_size,"planned_updates":parent["actual_optimizer_updates"]+contract["additional_optimizer_updates"],
        "continuation_protocol":contract,"protocol_path":str(Path(protocol).resolve()),"protocol_sha256":fixed.digest(protocol),
        "continuation_source_sha256":fixed.digest(__file__),"restoration":restoration,"schedule_restart":schedule_proof,
        "parent_actual_optimizer_updates":parent["actual_optimizer_updates"],
        "continuation_git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        "checkpoint_bank_passes":contract["checkpoint_bank_passes"],
        "checkpoint_updates":[parent["actual_optimizer_updates"]+p*contract["batches_per_pass"] for p in contract["checkpoint_bank_passes"]],
        "scientific_acceptance":False,"resume_supported":False}
    fixed.write_json(out/"run.json",run)
    fixed.write_json(out/"initial-probes.json",initial)
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
                adapted=workflow.adapter(raw)
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
                scale.save_checkpoint(destination,workflow,state,{**run,"completed_bank_passes":epoch+1},step,point)
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

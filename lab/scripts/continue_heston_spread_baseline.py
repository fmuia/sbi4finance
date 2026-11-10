"""One bounded continuation of the fixed-c, zero-dropout Heston control.

Only the declared 10k->50k continuation is supported, with 25k/50k endpoints.
Full saved array state is required; a native .keras archive alone is insufficient.
The CLI supervises a separate worker for a hard 900-second total wall limit.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROCESS_STARTED = time.monotonic()

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from scripts import train_heston_spread_baseline as trainer


FROZEN_TRAINER_SHA256 = "3a8747332a3493adcd9298d701d0d6aab087dada35d8648827d7790330931f38"
ENDPOINTS = (25000, 50000)
MAX_WALL_SECONDS = 900


def validate_parent_record(record, *, smoke_test_only=False):
    """Restrict this continuation to the frozen zero-control recipe."""
    trainer.validate_task(record["task"])
    if (record.get("format") != "heston_spread_baseline_checkpoint_v1"
            or record.get("dropout_zero_control") is not True
            or record["architecture"] != {"family":"spline","widths":[128,128],"subnet_dropout":0.}
            or record["schedule_updates"] != 10000 or record["seed"] != 9011
            or record["batch_size"] != 128 or record["source_sha256"]["trainer"] != FROZEN_TRAINER_SHA256
            or record["source_sha256"]["trainer"] != trainer.digest(trainer.__file__)
            or record["task_fingerprint"] != trainer.fingerprint(record["task"])
            or record["names"] != record["task"]["target"]["parameter_names"]
            or record["bounds"] != record["task"]["target"]["prior_bounds"]
            or record["design_sha256"] != record["task"]["design_sha256"]
            or not record.get("paired_default_reference",{}).get("initial_deterministic_probes_exact")):
        raise ValueError("parent is not the frozen fixed-c zero-dropout control")
    start = record["actual_optimizer_updates"]
    if type(start) is not int or (not smoke_test_only and (start != 10000 or record["planned_updates"] != 10000
            or record["train_rows"] != 20000 or record["probe_rows"] != {"train":512,"validation":512})):
        raise ValueError("production continuation starts only from the declared 10k checkpoint")
    if smoke_test_only and not 1 <= start < 5:
        raise ValueError("toy continuation requires a checkpoint before update five")
    schedule = record["optimizer_schedule"]["serialized"]
    config = schedule["config"]
    if (schedule["class_name"] != "CosineDecay" or config["decay_steps"] != 10000
            or config["initial_learning_rate"] != 5e-4 or config["alpha"] != .1):
        raise ValueError("the original 10k cosine schedule must not be reset or stretched")
    from sbi4f_tutorial.heston_options import model
    spread_path = Path(sys.modules[trainer.HestonSpreadPilot.__module__].__file__)
    source_paths = {"batch_plan":Path(sys.modules[trainer.BatchPlan.__module__].__file__),
        "spread_pilot":spread_path,"heston_pilot":spread_path.with_name("heston_pilot.py"),"heston_model":Path(model.__file__)}
    if any(trainer.digest(path) != record["source_sha256"][name] for name,path in source_paths.items()):
        raise ValueError("parent mathematical/data-order source differs")
    return start


def restore_full_state(workflow, record, state_path):
    """Restore all arrays, including auxiliary RNG/metrics omitted by .keras."""
    state_path = Path(state_path)
    if trainer.digest(state_path) != record["state_sha256"]:
        raise ValueError("parent full-state payload hash differs")
    variables = trainer.model_variables(workflow.estimator)
    expected_keys = {f"{group}_{i:04d}" for group in trainer.GROUPS for i in range(len(record["state"][group]))}
    state = []
    with np.load(state_path,allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("full-state variable counts or groups differ")
        for group, current in zip(trainer.GROUPS,variables,strict=True):
            declared = record["state"][group]
            if len(current) != len(declared):
                raise ValueError("model variable count differs from saved full state")
            arrays = []
            for index,(variable,identity) in enumerate(zip(current,declared,strict=True)):
                value = archive[f"{group}_{index:04d}"].copy()
                summary = trainer.array_summary(value)
                if (any(summary[key] != identity[key] for key in ("shape","dtype","sha256"))
                        or not summary["all_finite"] or tuple(variable.shape) != value.shape
                        or str(variable.dtype) != str(value.dtype)):
                    raise ValueError(f"saved {group} variable {index} shape/dtype/hash differs")
                arrays.append(value)
            state.append(arrays)
    trainer.assign_state(workflow.estimator,tuple(state))
    actual = trainer.current_state(workflow.estimator)
    summary = trainer.state_summary(workflow.estimator,actual)
    for group in trainer.GROUPS:
        for restored,saved in zip(summary[group],record["state"][group],strict=True):
            if any(restored[key] != saved[key] for key in ("shape","dtype","sha256")):
                raise ValueError("full-state assignment changed saved arrays")
    return actual, {"all_array_hashes_exact":True,"variable_counts":{key:len(value) for key,value in summary.items()},
        "variable_mapping":"saved group/index/shape/dtype/hash; Keras constructor name counters and model-root prefixes can change on archive reload",
        "archive_path_changes":{group:sum(a["path"] != b["path"] for a,b in zip(summary[group],record["state"][group],strict=True))
                                for group in trainer.GROUPS},
        "state_sha256":trainer.digest(state_path)}


def visit_counts(plan,start,end):
    visits = np.zeros(plan.rows,dtype=np.int64)
    for step in range(start,end):
        visits[plan.indices(step)] += 1
    return visits


def validate_bank_binding(parent, bank_record):
    for key in ("bank_record_sha256","bank_payload_sha256","bank_identity"):
        if bank_record[key] != parent[key]:
            raise ValueError("parent bank/task binding differs")


def run_continuation(checkpoint,out,*,wall_seconds=MAX_WALL_SECONDS,smoke_test_only=False):
    """Run the one authorized continuation; toy mode stops at update five."""
    started = time.monotonic()
    if not 0 < wall_seconds <= MAX_WALL_SECONDS or type(smoke_test_only) is not bool:
        raise ValueError("continuation wall limit must be positive and at most 900 seconds")
    checkpoint,out = Path(checkpoint),Path(out)
    parent = json.loads((checkpoint/"checkpoint.json").read_text())
    start = validate_parent_record(parent,smoke_test_only=smoke_test_only)
    if out.exists():
        raise FileExistsError(out)
    total = 5 if smoke_test_only else 50000
    endpoints = (5,) if smoke_test_only else ENDPOINTS
    os.environ.setdefault("KERAS_BACKEND","jax")
    import bayesflow  # import-time seed consumption precedes the fixed reconstruction seed
    import jax
    import keras
    keras.utils.set_random_seed(parent["seed"])
    workflow, loaded = trainer.load_checkpoint(checkpoint)
    if loaded != parent:
        raise ValueError("checkpoint changed during loading")
    model = workflow.estimator
    if (model.optimizer.get_config()["learning_rate"] != parent["optimizer_schedule"]["serialized"]
            or bool(jax.config.x64_enabled) != parent["numeric_configuration"]["jax_enable_x64"]
            or keras.backend.floatx() != parent["numeric_configuration"]["keras_floatx"]
            or any(metadata.version(name) != version for name,version in parent["versions"].items() if name != "python")):
        raise ValueError("optimizer schedule or numerical environment differs from parent")
    lab = Path(__file__).resolve().parents[1]
    if any(trainer.digest(lab/name) != value for name,value in parent["environment_sha256"].items()):
        raise ValueError("parent locked environment files differ")
    state,restoration = restore_full_state(workflow,parent,checkpoint/"state.npz")
    if int(model.optimizer.iterations.numpy()) != start:
        raise ValueError("restored optimizer iteration differs from checkpoint endpoint")
    schedule = keras.optimizers.schedules.deserialize(parent["optimizer_schedule"]["serialized"])
    rates = {str(step):float(schedule(step)) for step in (10000,25000,50000)}
    if not np.allclose(list(rates.values()),5e-5,rtol=1e-7,atol=0):
        raise ValueError("continuation must retain the original 5e-5 learning-rate floor")
    task = parent["task"]
    if trainer.digest(parent["design_path"]) != parent["design_sha256"]:
        raise ValueError("parent design file differs")
    data,bank_record = trainer.load_clean_bank(parent["bank_path"],task,parent["train_rows"],smoke_test_only=smoke_test_only)
    validate_bank_binding(parent,bank_record)
    probes = {name:trainer.fixed_probe(workflow,task,data[name],parent["probe_rows"][name],parent["seed"],namespace)
              for name,namespace in (("train",1),("validation",2))}
    def probe_values():
        trainer.assign_state(model,state)
        return {name+"_probe_loss":float(np.asarray(model.compute_metrics(**batch,stage="validation")["loss"]))
                for name,(_,batch) in probes.items()}
    initial = probe_values()
    if any(initial[name] != parent["probe_metrics"][name] for name in initial):
        raise ValueError("resumed deterministic probes differ before any optimizer update")
    # Validation-stage probes must not mutate canonical/model arrays.
    for group,values in zip(trainer.GROUPS,trainer.current_state(model),strict=True):
        if [trainer.array_summary(v)["sha256"] for v in values] != [row["sha256"] for row in parent["state"][group]]:
            raise ValueError("pre-update probe changed full restored state")
    out.mkdir(parents=True)
    protocol = {"format":"heston_spread_zero_continuation_protocol_v1","start_updates":start,
        "target_updates":total,"checkpoint_updates":list(endpoints),"schedule_updates":10000,
        "learning_rates":rates,"wall_seconds":wall_seconds,"smoke_test_only":smoke_test_only,
        "parent_checkpoint_path":str(checkpoint.resolve()),"parent_checkpoint_sha256":trainer.digest(checkpoint/"checkpoint.json"),
        "parent_model_sha256":parent["model_sha256"],"parent_state_sha256":parent["state_sha256"],
        "frozen_trainer_sha256":FROZEN_TRAINER_SHA256,"continuation_source_sha256":trainer.digest(__file__),
        "restoration":restoration,"absolute_data_and_noise_steps":True,"reserved_arrays_opened":False,
        "scope":"one zero-dropout continuation only; no generic checkpoint-resume or scientific acceptance claim"}
    trainer.write_json(out/"resume-protocol.json",protocol)
    run = {**parent,"format":"heston_spread_baseline_run_v1","planned_updates":total,
        "checkpoint_updates":list(endpoints),"wall_seconds":wall_seconds,
        "continuation":protocol,"resume_protocol_sha256":trainer.digest(out/"resume-protocol.json"),
        "parent_actual_optimizer_updates":start,"scientific_acceptance":False,"resume_supported":False}
    run["source_sha256"] = {**parent["source_sha256"],"continuation":trainer.digest(__file__)}
    trainer.write_json(out/"run.json",run)
    trainer.write_json(out/"initial-probes.json",initial)
    plan = trainer.BatchPlan(parent["train_rows"],parent["batch_size"],total,parent["seed"])
    prefix_visits = visit_counts(plan,0,start)
    visits = np.zeros(parent["train_rows"],dtype=np.int64)
    step_fn = trainer.compiled_step(model)
    curve,last_checkpoint = [],None
    step,phase,fields = start,"before_update",[]
    candidate,gradients,metrics,indices,quotes,raw,adapted = None,None,{},[],[],{},{}
    def deadline():
        if time.monotonic()-started >= wall_seconds:
            raise TimeoutError("zero-dropout continuation wall budget exhausted")
    try:
        for step in range(start,total):
            phase,fields = "before_update",[]
            candidate,gradients,metrics,indices,quotes,raw,adapted = None,None,{},[],[],{},{}
            deadline()
            indices,quotes,raw,adapted = trainer.adapted_batch(workflow,task,data["train"],plan,step,parent["seed"])
            fields = ["adapted_input:"+key for key,value in adapted.items() if not np.isfinite(value).all()]
            if fields:
                raise FloatingPointError("nonfinite continuation adapted input")
            if step == start:
                trainer.write_json(out/"first-batch.json",{"absolute_step":step,
                    "indices_sha256":trainer.array_summary(indices)["sha256"],
                    "quotes_sha256":trainer.array_summary(quotes)["sha256"]})
            phase = "optimizer_update"
            metrics,candidate,gradients,flags = step_fn(state,adapted)
            fields = [key for key,value in jax.device_get(flags).items() if not bool(value)]
            if fields:
                raise FloatingPointError("nonfinite continuation update: "+", ".join(fields))
            if int(np.asarray(candidate[2][0])) != step+1:
                raise RuntimeError("continuation optimizer iteration differs from absolute step")
            state = candidate
            visits[indices] += 1
            if (step+1)%1000 == 0 or step+1 in endpoints:
                phase = "fixed_probe"
                point = {"updates":step+1,"seconds":time.monotonic()-started,
                    "batch_training_loss":float(np.asarray(metrics["loss"])),**probe_values()}
                fields = [key for key,value in point.items() if not np.isfinite(value)]
                if fields:
                    metrics,candidate,gradients = point,None,None
                    raise FloatingPointError("nonfinite continuation fixed probe")
                curve.append(point)
                trainer.write_json(out/"learning-curve.json",curve)
                print(json.dumps(point),flush=True)
            if step+1 in endpoints:
                phase = "checkpoint_and_reload"
                destination = out/"checkpoints"/f"update-{step+1:06d}"
                trainer.checkpoint(destination,workflow,state,run,step+1,curve[-1])
                before = trainer.sample(workflow,task,probes["validation"][0][:2],16,parent["seed"]+11)
                restored,record = trainer.load_checkpoint(destination)
                full,proof = restore_full_state(restored,record,destination/"state.npz")
                after = trainer.sample(restored,task,probes["validation"][0][:2],16,parent["seed"]+11)
                if not np.array_equal(before,after) or not proof["all_array_hashes_exact"]:
                    raise RuntimeError("continuation checkpoint changed saved state or seeded inference")
                last_checkpoint = str(destination.resolve())
            trainer.write_json(out/"progress.json",{"status":"training","completed_updates":step+1,
                "continued_updates":step+1-start,"seconds":time.monotonic()-started,"last_checkpoint":last_checkpoint})
        deadline()
        trainer.write_npz(out/"exposure.npz",{"row_id":data["train"]["row_id"],"presentations_before_resume":prefix_visits,
            "continuation_presentations":visits,"presentations":prefix_visits+visits})
        result = {"status":"complete","completed_updates":total,"continued_updates":total-start,
            "seconds":time.monotonic()-started,"last_checkpoint":last_checkpoint,
            "continued_row_presentations":int(visits.sum()),"total_row_presentations":int((prefix_visits+visits).sum()),
            "all_parent_array_hashes_restored":True,"same_seed_reload_exact":True,
            "reserved_arrays_opened":False,"scientific_acceptance":False}
        trainer.write_json(out/"training.json",result)
        trainer.write_json(out/"progress.json",result)
        return result
    except BaseException as error:
        if phase != "optimizer_update":
            candidate,gradients = None,None
        if phase == "fixed_probe":
            adapted = {**adapted,**{f"probe_{name}_{key}":value for name,(_,batch) in probes.items() for key,value in batch.items()}}
        failure = trainer.capture_failure(out,model,state,candidate,gradients,step=step,indices=indices,quotes=quotes,
            raw=raw,adapted=adapted,fields=fields,metrics=metrics,error=error,last_checkpoint=last_checkpoint,phase=phase)
        trainer.write_json(out/"progress.json",{"status":"failed","completed_updates":failure["completed_finite_updates"],
            "failure":"failure/failure.json","seconds":time.monotonic()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,default=MAX_WALL_SECONDS)
    parser.add_argument("--smoke-test-only",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= MAX_WALL_SECONDS:
        parser.error("wall-seconds must be positive and at most900")
    if not args.worker:
        if args.out.exists():
            raise FileExistsError(args.out)
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"])
        try:
            code = process.wait(timeout=max(.001,args.wall_seconds-(time.monotonic()-PROCESS_STARTED)))
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            args.out.mkdir(parents=True,exist_ok=True)
            trainer.write_json(args.out/"hard-timeout.json",{"status":"killed_at_wall_limit","wall_seconds":args.wall_seconds,
                "scientific_acceptance":False,"automatic_retry":False})
            raise TimeoutError("continuation supervisor wall limit exhausted")
        if code:
            raise SystemExit(code)
        return
    print(json.dumps(run_continuation(args.checkpoint,args.out,wall_seconds=args.wall_seconds,smoke_test_only=args.smoke_test_only)))


if __name__ == "__main__":
    main()

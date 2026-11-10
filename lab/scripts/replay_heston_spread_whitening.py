"""Exact bounded prefix replay of the frozen ZCA pilot, with earlier checkpoints.

The original 8,000-update learning-rate schedule is retained while stopping at
4,000 updates. Original noisy rows, preprocessing, initialization and every
fixed-probe learning-curve point are checked against the preserved trajectory.
Historical producers and checkpoints are never modified.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_heston_spread_whitening as frozen

fixed, scale = frozen.fixed, frozen.scale
precision, expansion, full_state = frozen.precision, frozen.expansion, frozen.full_state
SEED, MAX_WALL_SECONDS = frozen.SEED, frozen.MAX_WALL_SECONDS
read_task, prepare_data = frozen.read_task, frozen.prepare_data
fit_preprocessing, preprocessing_transform = frozen.fit_preprocessing, frozen.preprocessing_transform
adapted_data, build_workflow = frozen.adapted_data, frozen.build_workflow
WhitenedWorkflow, sample = frozen.WhitenedWorkflow, frozen.sample
FROZEN_PRODUCER_SHA256 = "6a7b35afc9c5beadcccb8e83cf73603cca2790a5e94b4818aed53f3e29523c9c"
PROTOCOL_FORMAT = "heston_spread_whitening_checkpoint_replay_study_v1"
CHECKPOINT_FORMAT = "heston_spread_whitening_checkpoint_replay_checkpoint_v1"
CHECKPOINT_UPDATES = (2000, 3000, 3320, 4000)


def sources():
    original = frozen.sources()
    if original["trainer"]["sha256"] != FROZEN_PRODUCER_SHA256:
        raise ValueError("original whitening producer changed")
    return {"replay": {"path":str(Path(__file__).resolve()), "sha256":fixed.digest(__file__)},
            "original": original}


def assert_initial_state(actual, original):
    for group in fixed.GROUPS:
        left = [{key:row[key] for key in ("shape","dtype","sha256")} for row in actual[group]]
        right = [{key:row[key] for key in ("shape","dtype","sha256")} for row in original[group]]
        if left != right:
            raise ValueError("replay original initial state differs in " + group)


def assert_curve_point(actual, original):
    if {key:value for key,value in actual.items() if key != "seconds"} != {
            key:value for key,value in original.items() if key != "seconds"}:
        raise ValueError("replay trajectory differs at update " + str(actual.get("updates")))


def protocol_contract(bank, task, original, *, smoke_test_only=False):
    task, original = read_task(task), Path(original).resolve()
    run = json.loads((original/"run.json").read_text())
    expected = frozen.protocol_contract(bank,task,mode="zca",smoke_test_only=smoke_test_only)
    if (run.get("format") != "heston_spread_whitening_run_v1" or run["mode"] != "zca"
            or run["task"] != task or run["training_protocol"] != expected
            or run["source"] != frozen.sources()
            or run["smoke_test_only"] != smoke_test_only):
        raise ValueError("original ZCA training identity, task or fixed data differs")
    preprocessing = json.loads((original/"preprocessing.json").read_text())
    preprocessing_transform(preprocessing)
    if (fixed.fingerprint(preprocessing) != expected["preprocessing_fingerprint"]
            or fixed.digest(original/"preprocessing.json") != run["preprocessing_sha256"]):
        raise ValueError("original frozen preprocessing differs")
    training = json.loads((original/"training.json").read_text())
    if training.get("status") != "complete" or training["completed_updates"] != expected["planned_updates"]:
        raise ValueError("only a completed preserved original trajectory can be replayed")
    final = original/"checkpoints"/f"update-{expected['planned_updates']:06d}"
    checkpoint = json.loads((final/"checkpoint.json").read_text())
    if (checkpoint["format"] != "heston_spread_whitening_checkpoint_v1"
            or checkpoint["actual_optimizer_updates"] != expected["planned_updates"]
            or checkpoint["training_protocol"] != expected
            or checkpoint["task"] != task or checkpoint["source"] != frozen.sources()
            or checkpoint["protocol_sha256"] != fixed.digest(checkpoint["protocol_path"])):
        raise ValueError("original final checkpoint binding differs")
    for filename,field in (("model.keras","model_sha256"),("state.npz","state_sha256"),
                           ("preprocessing.json","preprocessing_sha256")):
        if fixed.digest(final/filename) != checkpoint[field]:
            raise ValueError("original final checkpoint payload differs")
    curve = json.loads((original/"learning-curve.json").read_text())
    if curve[-1]["updates"] != expected["planned_updates"]:
        raise ValueError("original trajectory is incomplete")
    files = {name:{"path":str((original/name).resolve()), "sha256":fixed.digest(original/name)}
             for name in ("run.json","preprocessing.json","initial-probes.json","learning-curve.json","training.json")}
    files["final_checkpoint"] = {"path":str((final/"checkpoint.json").resolve()),
                                 "sha256":fixed.digest(final/"checkpoint.json")}
    updates, checkpoints = (4,[2,3,4]) if smoke_test_only else (4000,list(CHECKPOINT_UPDATES))
    return {**expected,"format":"heston_spread_whitening_checkpoint_replay_protocol_v1",
        "planned_updates":updates,"schedule_updates":expected["planned_updates"],
        "checkpoint_updates":checkpoints,"bank_passes":None if smoke_test_only else 100,
        "planned_presentations":128 if smoke_test_only else 1000000,
        "original":{"directory":str(original),"files":files},"source":sources(),
        "checkpoint_selection":"post-hoc development replay; not a fresh acceptance test",
        "scope":"exact original ZCA trajectory prefix; no fresh noise, pricing, new reference inference or reserved arrays"}


def selected_contract(protocol):
    if protocol.get("format") != PROTOCOL_FORMAT or "contract" not in protocol or "output" not in protocol:
        raise ValueError("persisted single-arm checkpoint replay protocol required")
    return protocol


def save_checkpoint(directory, wrapper, state, run, updates, probes):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    fixed.assign_state(wrapper.estimator,state)
    wrapper.estimator.save(directory/"temporary.keras")
    (directory/"temporary.keras").replace(directory/"model.keras")
    fixed.save_state(directory/"state.npz",state)
    fixed.write_json(directory/"preprocessing.json",wrapper.preprocessing)
    record = {**run,"format":CHECKPOINT_FORMAT,"actual_optimizer_updates":updates,
        "model_sha256":fixed.digest(directory/"model.keras"),"state_sha256":fixed.digest(directory/"state.npz"),
        "preprocessing_sha256":fixed.digest(directory/"preprocessing.json"),
        "state":fixed.state_summary(wrapper.estimator,state),
        "optimizer_schedule":wrapper.estimator.optimizer.get_config()["learning_rate"],
        "probe_metrics":probes,"resume_supported":False}
    fixed.write_json(directory/"checkpoint.json",record)
    return record


def load_checkpoint(directory):
    directory = Path(directory)
    record = json.loads((directory/"checkpoint.json").read_text())
    task, contract = read_task(record["task"]), record["training_protocol"]
    if (record.get("format") != CHECKPOINT_FORMAT or record["task_fingerprint"] != fixed.fingerprint(task)
            or record["source"] != sources() or contract["source"] != sources()
            or record["protocol_sha256"] != fixed.digest(record["protocol_path"])
            or selected_contract(json.loads(Path(record["protocol_path"]).read_text()))["contract"] != contract
            or record["names"] != task["target"]["parameter_names"] or record["bounds"] != task["target"]["prior_bounds"]
            or record["actual_optimizer_updates"] not in contract["checkpoint_updates"]
            or record["mode"] != "zca" or contract["mode"] != "zca"
            or contract["schedule_updates"] != (5 if record["smoke_test_only"] else 8000)
            or contract["planned_updates"] != (4 if record["smoke_test_only"] else 4000)):
        raise ValueError("replay checkpoint source, protocol, step or task differs")
    for value in contract["original"]["files"].values():
        if fixed.digest(value["path"]) != value["sha256"]:
            raise ValueError("preserved original trajectory binding changed")
    for filename,field in (("model.keras","model_sha256"),("state.npz","state_sha256"),
                           ("preprocessing.json","preprocessing_sha256")):
        if fixed.digest(directory/filename) != record[field]:
            raise ValueError("replay checkpoint payload differs")
    preprocessing = json.loads((directory/"preprocessing.json").read_text())
    if (fixed.fingerprint(preprocessing) != contract["preprocessing_fingerprint"]
            or preprocessing["task_fingerprint"] != record["task_fingerprint"]
            or preprocessing["fit"]["conditions"] != contract["splits"]["train"]["conditions"]
            or preprocessing["feature_order_sha256"] != fixed.fingerprint(task["records"])
            or preprocessing["half_spread"] != fixed.array_summary(np.asarray(task["records"]["half_spread"],dtype=np.float64))):
        raise ValueError("replay frozen preprocessing binding differs")
    preprocessing_transform(preprocessing)
    workflow = build_workflow(task,contract["schedule_updates"])
    workflow.load_estimator(directory/"model.keras")
    if any(layer.rate != 0. for layer in fixed.dropout_layers(workflow.estimator)):
        raise ValueError("replay checkpoint changed zero dropout")
    full_state.restore_full_state(workflow,record,directory/"state.npz")
    if (workflow.estimator.optimizer.get_config()["learning_rate"] != record["optimizer_schedule"]
            or int(np.asarray(workflow.estimator.optimizer.iterations)) != record["actual_optimizer_updates"]):
        raise ValueError("replay checkpoint schedule or optimizer step differs")
    return WhitenedWorkflow(workflow,preprocessing),record


def run_training(bank, task, out, *, original, protocol, smoke_test_only=False, wall_seconds=MAX_WALL_SECONDS):
    started = time.monotonic()
    if not 0 < wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("pilot wall limit must be positive and at most900 seconds")
    task, out, protocol = read_task(task), Path(out), Path(protocol)
    contract = protocol_contract(bank,task,original,smoke_test_only=smoke_test_only)
    mode = "zca"
    original = Path(original)
    original_run = json.loads((original/"run.json").read_text())
    original_curve = json.loads((original/"learning-curve.json").read_text())
    declared = selected_contract(json.loads(protocol.read_text()))
    if declared["contract"] != contract or Path(declared["output"]).resolve() != out.resolve():
        raise ValueError("persist the exact source-bound whitening protocol before fitting")
    if out.exists():
        raise FileExistsError(out)
    data,binding = prepare_data(bank,task,smoke_test_only=smoke_test_only)
    preprocessing = json.loads((original/"preprocessing.json").read_text())
    preprocessing_transform(preprocessing)
    transformed = {name:adapted_data(task,raw,preprocessing) for name,raw in data.items()}
    os.environ.setdefault("KERAS_BACKEND","jax")
    import bayesflow
    import jax
    import keras
    batch,updates = contract["batch_size"],contract["planned_updates"]
    first = {key:value[:batch] for key,value in transformed["train"].items()}

    def initialized(arm_mode):
        keras.utils.set_random_seed(SEED)
        arm_preprocessing = preprocessing if arm_mode == mode else fit_preprocessing(task,data["train"],binding,arm_mode)
        workflow = build_workflow(task,contract["schedule_updates"])
        initial = adapted_data(task,{key:value[:batch] for key,value in data["train"].items()},arm_preprocessing)
        workflow.estimator.build_from_data(initial)
        workflow.estimator.optimizer.build(workflow.estimator.trainable_variables)
        return workflow

    workflow = initialized(mode)
    initial_state = fixed.state_summary(workflow.estimator,fixed.current_state(workflow.estimator))
    reference = initialized("zca" if mode == "diagonal" else "diagonal")
    reference_state = fixed.state_summary(reference.estimator,fixed.current_state(reference.estimator))
    initial_hashes = [row["sha256"] for row in initial_state["trainable"]]
    if initial_hashes != [row["sha256"] for row in reference_state["trainable"]]:
        raise ValueError("paired transforms changed fresh trainable initialization")
    del reference
    assert_initial_state(initial_state, original_run["initial_state"])
    wrapper = WhitenedWorkflow(workflow,preprocessing)
    model, state = workflow.estimator, fixed.current_state(workflow.estimator)
    out.mkdir(parents=True)
    fixed.write_json(out/"preprocessing.json",preprocessing)
    lab = Path(__file__).resolve().parents[1]
    run = {"format":"heston_spread_whitening_checkpoint_replay_run_v1","task":task,
        "task_fingerprint":fixed.fingerprint(task),"names":task["target"]["parameter_names"],
        "bounds":task["target"]["prior_bounds"],"mode":mode,"training_protocol":contract,
        "protocol_path":str(protocol.resolve()),"protocol_sha256":fixed.digest(protocol),
        "bank_path":str(Path(bank).resolve()),**binding,"train_rows":contract["train_rows"],
        "batch_size":batch,"planned_updates":updates,"schedule_updates":contract["schedule_updates"],"seed":SEED,
        "architecture":contract["architecture"],"standardize":contract["standardize"],
        "preprocessing_path":str((out/"preprocessing.json").resolve()),
        "preprocessing_sha256":fixed.digest(out/"preprocessing.json"),
        "initial_trainable_hashes":initial_hashes,"initial_state":initial_state,
        "paired_initialization":{"fresh":True,"same_seed":SEED,"all_trainable_hashes_exact":True,
            "comparison":"build both modes on the same targets/first rows before any update; no historical warm start"},
        "versions":{"python":platform.python_version(),**{key:metadata.version(key) for key in ("bayesflow","keras","jax","numpy","scipy")}},
        "numeric_configuration":{"jax_enable_x64":bool(jax.config.x64_enabled),"keras_floatx":keras.backend.floatx()},
        "environment_sha256":{name:fixed.digest(lab/name) for name in ("pyproject.toml","pixi.lock")},
        "original":contract["original"],"source":sources(),"smoke_test_only":smoke_test_only,"max_wall_seconds":wall_seconds,
        "reserved_arrays_opened":False,"new_pricing_labels":False,"scientific_acceptance":False}
    fixed.write_json(out/"run.json",run)
    probes = {name:{key:value[:min(len(value),512)] for key,value in adapted.items()}
        for name,adapted in transformed.items()}

    def probe_values():
        fixed.assign_state(model,state)
        result = {name+"_probe_loss":float(np.asarray(model.compute_metrics(**adapted,stage="validation")["loss"]))
            for name,adapted in probes.items()}
        if not all(np.isfinite(x) for x in result.values()):
            raise FloatingPointError("nonfinite fixed pilot probe")
        return result

    initial_probes = probe_values()
    if initial_probes != json.loads((original/"initial-probes.json").read_text()):
        raise ValueError("replay initial fixed probes differ from original")
    fixed.write_json(out/"initial-probes.json",initial_probes)
    step_fn, visits = fixed.compiled_step(model), np.zeros(contract["train_rows"],dtype=np.int64)
    curve, step, last_checkpoint = [], 0, None
    phase,indices,raw,adapted,candidate,gradients,metrics,fields = "before_update",[],{}, {},None,None,{},[]
    try:
        while step < updates:
            order = expansion.epoch_order(contract["train_rows"],step//contract["batches_per_pass"],SEED)
            for offset in range(0,contract["train_rows"],batch):
                if step >= updates:
                    break
                if time.monotonic()-started >= wall_seconds:
                    raise TimeoutError("bounded whitening pilot wall budget exhausted")
                phase,fields,candidate,gradients = "before_update",[],None,None
                indices = order[offset:offset+batch]
                raw = {key:value[indices] for key,value in data["train"].items()}
                adapted = {key:value[indices] for key,value in transformed["train"].items()}
                phase = "optimizer_update"
                metrics,candidate,gradients,flags = step_fn(state,adapted)
                fields = [name for name,value in jax.device_get(flags).items() if not bool(value)]
                if fields:
                    raise FloatingPointError("nonfinite whitening pilot update: "+", ".join(fields))
                if int(np.asarray(candidate[2][0])) != step+1:
                    raise RuntimeError("pilot optimizer iteration differs")
                state = candidate
                visits[indices] += 1
                step += 1
                if step % contract["batches_per_pass"] == 0 or step == updates:
                    phase = "fixed_probe"
                    point = {"updates":step,"completed_bank_passes":step//contract["batches_per_pass"],
                        "presentations":int(visits.sum()),"seconds":time.monotonic()-started,
                        "batch_training_loss":float(np.asarray(metrics["loss"])),**probe_values()}
                    assert_curve_point(point, original_curve[len(curve)])
                    curve.append(point)
                    fixed.write_json(out/"learning-curve.json",curve)
                    fixed.write_json(out/"progress.json",{"status":"running",**point})
                    print(json.dumps(point),flush=True)
                if step in contract["checkpoint_updates"]:
                    phase = "checkpoint_and_reload"
                    destination = out/"checkpoints"/f"update-{step:06d}"
                    fixed.assign_state(model,state)
                    quotes = data["validation"]["observations"][:2]*np.asarray(task["records"]["half_spread"])
                    before = sample(wrapper,task,quotes,16,SEED+11)
                    checkpoint_probes = curve[-1] if curve and curve[-1]["updates"] == step else {
                        "updates":step,"completed_bank_passes":step//contract["batches_per_pass"],
                        "presentations":int(visits.sum()),"batch_training_loss":float(np.asarray(metrics["loss"])),
                        **probe_values()}
                    save_checkpoint(destination,wrapper,state,run,step,checkpoint_probes)
                    restored,_ = load_checkpoint(destination)
                    after = sample(restored,task,quotes,16,SEED+11)
                    if not np.array_equal(before,after):
                        raise RuntimeError("saved preprocessing/checkpoint changed seeded draws")
                    fixed.assign_state(model,state)
                    last_checkpoint = str(destination.resolve())
        if int(visits.sum()) != contract["planned_presentations"]:
            raise RuntimeError("pilot example budget differs")
        if not smoke_test_only and not np.all(visits==100):
            raise RuntimeError("each fixed training row must be presented100 times")
        fixed.write_npz(out/"exposure.npz",{"row_id":np.array([f"train:{i:08d}" for i in range(len(visits))]),"presentations":visits})
        result = {"status":"complete","completed_updates":step,"completed_bank_passes":step//contract["batches_per_pass"],
            "presentations":int(visits.sum()),"seconds":time.monotonic()-started,"same_seed_reload_exact":True,
            "paired_initial_trainable_hashes_exact":True,"all_original_initial_state_hashes_exact":True,
            "original_learning_curve_prefix_exact":True,"original_curve_points_compared":len(curve),
            "schedule_updates":contract["schedule_updates"],"last_checkpoint":last_checkpoint,
            "reserved_arrays_opened":False,"scientific_acceptance":False}
        fixed.write_json(out/"training.json",result)
        fixed.write_json(out/"progress.json",result)
        return result
    except Exception as error:
        quotes = np.asarray(raw.get("observations",[]))*np.asarray(task["records"]["half_spread"]) if raw else []
        fixed.capture_failure(out,model,state,candidate,gradients,step=step,indices=indices,
            quotes=quotes,raw=raw,adapted=adapted,fields=fields,metrics=metrics,error=error,
            last_checkpoint=last_checkpoint,phase=phase)
        fixed.write_json(out/"progress.json",{"status":"failed","completed_updates":step,
            "phase":phase,"error_type":type(error).__name__,"error":str(error),"scientific_acceptance":False})
        raise



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank","task","out","protocol","original"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--smoke-test-only",action="store_true")
    parser.add_argument("--worker",action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_training(args.bank,args.task,args.out,original=args.original,protocol=args.protocol,
            smoke_test_only=args.smoke_test_only)
    else:
        command = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
        completed = subprocess.run(command,timeout=MAX_WALL_SECONDS)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

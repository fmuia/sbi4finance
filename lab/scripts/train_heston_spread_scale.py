"""One bounded six-target Heston fit with inferred Gaussian spread scale c.

The five physical priors and complete 54-quote design remain frozen. Clean
physical labels are reused; c~Uniform(.5,2) and independent Gaussian errors are
fresh per exposure. No c, sigma, clean price or noise realization is a condition.
Production requires a separately persisted protocol and a 900-second supervisor.
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

PROCESS_STARTED = time.monotonic()
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from scripts import train_heston_spread_baseline as fixed
from sbi4finance.calibration.heston_spread import HestonSpreadPilot


FROZEN_FIXED_SHA256 = "3a8747332a3493adcd9298d701d0d6aab087dada35d8648827d7790330931f38"
SCALE_NAMESPACE, NOISE_NAMESPACE = 0x5343414c, 0x53505244
ENDPOINTS = (10000,50000,60000)


def make_task(design):
    physical_task,physical = fixed.make_task(design)
    pilot = HestonSpreadPilot(physical.records,infer_c=True,c_bounds=(.5,2.))
    task = {"format":"heston_spread_scale_task_v1","target":pilot.task_definition(),
        "physical_clean_price_task":physical_task,"design_sha256":physical_task["design_sha256"],
        "records":physical_task["records"],"conditioning":"54 ordered noisy prices / frozen half_spread only",
        "scale_generation":"independent fresh Uniform(.5,2) per example exposure; sixth inferred target",
        "scope":"six-target fixed-design development experiment; no discrepancy parameter or release acceptance"}
    validate_task(task)
    return task,pilot


def validate_task(task):
    if task.get("format") != "heston_spread_scale_task_v1":
        raise ValueError("unsupported scale-inference task")
    physical = fixed.validate_task(task["physical_clean_price_task"])
    pilot = HestonSpreadPilot(task["records"],infer_c=True,c_bounds=(.5,2.))
    pilot.validate_task(task["target"])
    if (len(pilot.records["mid"]) != 54 or pilot.names != [*physical.names,"c"]
            or not np.array_equal(pilot.bounds[:5],physical.bounds)
            or task["records"] != task["physical_clean_price_task"]["records"]
            or task["design_sha256"] != task["physical_clean_price_task"]["design_sha256"]):
        raise ValueError("joint target changes the frozen physical prior/design/spreads")
    return pilot


def load_clean_bank(bank,task,train_rows=20000,*,smoke_test_only=False):
    """Explicit five-target projection for unchanged physical clean prices."""
    validate_task(task)
    return fixed.load_clean_bank(bank,task["physical_clean_price_task"],train_rows,
                                 smoke_test_only=smoke_test_only)


def draw_c(count,step,seed,*,namespace=0):
    return np.random.default_rng(np.random.SeedSequence([SCALE_NAMESPACE,seed,namespace,step])).uniform(.5,2.,size=count)


def augmented_batch(clean,physical,halfspread,step,seed,*,namespace=0):
    clean,physical,h = np.asarray(clean),np.asarray(physical),np.asarray(halfspread)
    if clean.ndim != 2 or physical.shape != (len(clean),5) or h.shape != (clean.shape[1],):
        raise ValueError("matching clean prices, five physical labels and frozen quote spreads required")
    c = draw_c(len(clean),step,seed,namespace=namespace)
    # Identical Gaussian namespace to the fixed-c recipe, independent of c's stream.
    error = np.random.default_rng(np.random.SeedSequence([NOISE_NAMESPACE,seed,namespace,step])).standard_normal(clean.shape)
    quotes = clean+c[:,None]*h*error
    return quotes,np.column_stack((physical,c))


def condition_data(task,quotes):
    return fixed.condition_data(task,quotes)


def training_data(task,quotes,parameters):
    if np.iscomplexobj(parameters):
        raise ValueError("six real physical/scale targets required")
    values = np.asarray(parameters,dtype=float)
    bounds = np.asarray(task["target"]["prior_bounds"])
    if (values.shape != (len(quotes),6) or not np.isfinite(values).all()
            or np.any(values<=bounds[:,0]) or np.any(values>=bounds[:,1])):
        raise ValueError("six finite targets strictly inside their declared Uniform priors required")
    return {**condition_data(task,quotes),**{name:values[:,i:i+1] for i,name in enumerate(task["target"]["parameter_names"])}}


def build_workflow(task):
    validate_task(task)
    import bayesflow as bf
    import keras
    if bf.__version__ != "2.1.0" or keras.backend.backend() != "jax":
        raise RuntimeError("locked BF2.1 JAX environment required")
    adapter = bf.Adapter()
    for name,(lo,hi) in zip(task["target"]["parameter_names"],task["target"]["prior_bounds"],strict=True):
        adapter = adapter.constrain(name,lower=float(lo),upper=float(hi),inclusive="none")
    adapter = (adapter.convert_dtype("float64","float32")
        .concatenate(task["target"]["parameter_names"],into="inference_variables")
        .concatenate(["observations"],into="inference_conditions"))
    optimizer = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(5e-4,10000,alpha=.1))
    flow = bf.networks.CouplingFlow(transform="spline",depth=6,subnet_kwargs={"widths":[128,128],"dropout":.05})
    workflow = bf.BasicWorkflow(adapter=adapter,inference_network=flow,
        standardize=["inference_variables","inference_conditions"],optimizer=optimizer)
    workflow.estimator.compile(optimizer=optimizer,steps_per_execution=1)
    return workflow


def sample(workflow,task,quotes,count,seed=0):
    values = workflow.sample(num_samples=count,conditions=condition_data(task,quotes),sample_shape=(),seed=seed)
    result = np.concatenate([np.asarray(values[name]) for name in task["target"]["parameter_names"]],axis=-1)
    if result.shape != (len(quotes),count,6) or not np.isfinite(result).all():
        raise FloatingPointError("invalid joint physical/scale posterior samples")
    return result


def install_annealing(model,state):
    """Only replace a built-in schedule; preserve all Adam/model arrays."""
    import keras
    fixed.assign_state(model,state)
    before = fixed.state_summary(model,fixed.current_state(model))
    old = model.optimizer.get_config()
    if (int(model.optimizer.iterations.numpy())!=50000
            or old["learning_rate"]["class_name"]!="CosineDecay"
            or old["learning_rate"]["config"]["decay_steps"]!=10000):
        raise ValueError("joint annealing must begin at absolute50000 with the original cosine schedule")
    model.optimizer.learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
        [54999],[5e-6,5e-7],name="heston_spread_two_stage_annealing")
    after = fixed.state_summary(model,fixed.current_state(model))
    new = model.optimizer.get_config()
    if before != after or {k:v for k,v in old.items() if k!="learning_rate"}!={k:v for k,v in new.items() if k!="learning_rate"}:
        raise RuntimeError("annealing reset saved arrays or other Adam configuration")
    return {"absolute_step":50000,"old_schedule":old["learning_rate"],"new_schedule":new["learning_rate"],
            "all_array_hashes_unchanged":True}


def protocol_contract(task,bank_record):
    return {"format":"heston_spread_scale_protocol_v1","predeclared_before_fit":True,
        "task_fingerprint":fixed.fingerprint(task),"design_sha256":task["design_sha256"],
        "bank_record_sha256":bank_record["bank_record_sha256"],"bank_payload_sha256":bank_record["bank_payload_sha256"],
        "trainer_sha256":fixed.digest(__file__),"frozen_fixed_helper_sha256":FROZEN_FIXED_SHA256,
        "seed":9011,"train_rows":20000,"batch_size":128,"planned_updates":60000,
        "checkpoint_updates":list(ENDPOINTS),"max_wall_seconds":900,
        "scale_prior":[.5,2.],"input_channels":54,
        "schedule":"CosineDecay(5e-4,10000,alpha=.1) through absolute step49999; PiecewiseConstantDecay([54999],[5e-6,5e-7]) from50000",
        "scale_randomness":"PCG64 SeedSequence([0x5343414c,seed,namespace,step]); independent of Gaussian stream",
        "gaussian_randomness":"PCG64 SeedSequence([0x53505244,seed,namespace,step])",
        "reserved_arrays_opened":False,"new_pricing_labels":False,"automatic_extension":False}


def save_checkpoint(directory,workflow,state,run,updates,probes):
    directory.mkdir(parents=True,exist_ok=False)
    fixed.assign_state(workflow.estimator,state)
    workflow.estimator.save(directory/"temporary.keras")
    (directory/"temporary.keras").replace(directory/"model.keras")
    fixed.save_state(directory/"state.npz",state)
    record = {**run,"format":"heston_spread_scale_checkpoint_v1","actual_optimizer_updates":updates,
        "optimizer_schedule":workflow.estimator.optimizer.get_config()["learning_rate"],
        "model_sha256":fixed.digest(directory/"model.keras"),"state_sha256":fixed.digest(directory/"state.npz"),
        "state":fixed.state_summary(workflow.estimator,state),"probe_metrics":probes,"resume_supported":False}
    fixed.write_json(directory/"checkpoint.json",record)


def load_checkpoint(directory):
    directory = Path(directory)
    record = json.loads((directory/"checkpoint.json").read_text())
    validate_task(record["task"])
    if (record["format"]!="heston_spread_scale_checkpoint_v1" or fixed.fingerprint(record["task"])!=record["task_fingerprint"]
            or record["names"]!=record["task"]["target"]["parameter_names"]
            or record["bounds"]!=record["task"]["target"]["prior_bounds"]
            or not 0<record["actual_optimizer_updates"]<=record["planned_updates"]
            or fixed.digest(directory/"model.keras")!=record["model_sha256"]
            or fixed.digest(directory/"state.npz")!=record["state_sha256"]):
        raise ValueError("joint checkpoint identity/payload differs")
    workflow = build_workflow(record["task"])
    workflow.load_estimator(directory/"model.keras")
    fixed.switch_dropout_zero(workflow.estimator,reloading=True)
    actual = fixed.state_summary(workflow.estimator,fixed.current_state(workflow.estimator))
    for group in ("trainable","optimizer","non_trainable"):
        original = [row["sha256"] for row in record["state"][group] if "seed_generator_state" not in row["path"]]
        restored = [row["sha256"] for row in actual[group] if "seed_generator_state" not in row["path"]]
        if original!=restored:
            raise ValueError("joint checkpoint learned/Adam/normalization arrays differ")
    if (int(workflow.estimator.optimizer.iterations.numpy())!=record["actual_optimizer_updates"]
            or workflow.estimator.optimizer.get_config()["learning_rate"]!=record["optimizer_schedule"]):
        raise ValueError("joint checkpoint iteration/schedule differs")
    return workflow,record


def run_training(design,bank,out,*,protocol=None,smoke_test_only=False):
    started = time.monotonic()
    if fixed.digest(fixed.__file__)!=FROZEN_FIXED_SHA256:
        raise ValueError("frozen numerical/data/finite-step helper differs")
    task,pilot = make_task(design)
    train_rows,updates,probe_rows,endpoints = (128,5,32,(2,5)) if smoke_test_only else (20000,60000,512,ENDPOINTS)
    data,bank_record = load_clean_bank(bank,task,train_rows,smoke_test_only=smoke_test_only)
    contract = protocol_contract(task,bank_record)
    if not smoke_test_only:
        if protocol is None or any(json.loads(Path(protocol).read_text()).get(k)!=v for k,v in contract.items()):
            raise ValueError("persist the exact pre-run joint-scale protocol before production")
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("KERAS_BACKEND","jax")
    import bayesflow
    import jax
    import keras
    keras.utils.set_random_seed(9011)
    workflow = build_workflow(task)
    model = workflow.estimator
    plan = fixed.BatchPlan(train_rows,128,updates,9011)
    def batch(step):
        indices = plan.indices(step)
        quotes,targets = augmented_batch(data["train"]["clean"][indices],data["train"]["parameters"][indices],
            task["records"]["half_spread"],step,9011)
        raw = training_data(task,quotes,targets)
        return indices,quotes,raw,workflow.adapter(raw)
    initial_batch = batch(0)[-1]
    model.build_from_data(keras.tree.map_structure(keras.ops.convert_to_tensor,initial_batch))
    model.optimizer.build(model.trainable_variables)
    toggle = fixed.switch_dropout_zero(model)
    state,step_fn = fixed.current_state(model),fixed.compiled_step(model)
    probes = {}
    for name,namespace in (("train",1),("validation",2)):
        count = min(probe_rows,len(data[name]["clean"]))
        quotes,targets = augmented_batch(data[name]["clean"][:count],data[name]["parameters"][:count],
            task["records"]["half_spread"],0,9011,namespace=namespace)
        probes[name] = (quotes,targets,workflow.adapter(training_data(task,quotes,targets)))
    run = {"format":"heston_spread_scale_run_v1","task":task,"task_fingerprint":fixed.fingerprint(task),
        "names":pilot.names,"bounds":pilot.bounds.tolist(),"design_sha256":task["design_sha256"],
        "design_path":str(Path(design).resolve()),**bank_record,"train_rows":train_rows,"planned_updates":updates,
        "seed":9011,"batch_size":128,"checkpoint_updates":list(endpoints),"max_wall_seconds":900,
        "architecture":{"family":"spline","depth":6,"widths":[128,128],"construction_dropout":.05,"effective_dropout":0.},
        "initialization_control":toggle,"initialization_claim":"within6D replay only; dimension/splits differ from5D and new weights are necessary",
        "source_sha256":{"trainer":fixed.digest(__file__),"fixed_helper":fixed.digest(fixed.__file__),
            "spread_pilot":fixed.digest(sys.modules[HestonSpreadPilot.__module__].__file__),
            "heston_pilot":fixed.digest(Path(sys.modules[HestonSpreadPilot.__module__].__file__).with_name("heston_pilot.py")),
            "batch_plan":fixed.digest(sys.modules[fixed.BatchPlan.__module__].__file__)},
        "environment_sha256":{name:fixed.digest(Path(__file__).resolve().parents[1]/name)
                              for name in ("pyproject.toml","pixi.lock")},
        "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        "versions":{"python":platform.python_version(),**{n:metadata.version(n) for n in ("bayesflow","keras","jax","numpy","scipy")}},
        "numeric_configuration":{"jax_enable_x64":bool(jax.config.x64_enabled),"keras_floatx":keras.backend.floatx()},
        "protocol_path":None if protocol is None else str(Path(protocol).resolve()),
        "protocol_sha256":None if protocol is None else fixed.digest(protocol),"recipe":contract,
        "probe_rows":{k:len(v[0]) for k,v in probes.items()},"schedule_changes":[],
        "scientific_acceptance":False,"resume_supported":False}
    fixed.write_json(out/"run.json",run)
    fixed.write_npz(out/"fixed-probes.npz",{f"{name}_{key}":value for name,(quotes,targets,_) in probes.items()
        for key,value in (("quotes",quotes),("targets",targets))})
    curve,visits,last_checkpoint = [],np.zeros(train_rows,dtype=np.int64),None
    indices,quotes,raw,adapted,metrics,candidate,gradients = [],[],{},{},{},None,None
    step,phase,fields = 0,"initial_probe",[]
    def probe_values():
        fixed.assign_state(model,state)
        return {name+"_probe_loss":float(np.asarray(model.compute_metrics(**value[2],stage="validation")["loss"]))
                for name,value in probes.items()}
    try:
        initial_metrics = probe_values()
        fields = [key for key,value in initial_metrics.items() if not np.isfinite(value)]
        if fields:
            metrics = initial_metrics
            raise FloatingPointError("nonfinite joint initial probe")
        fixed.write_json(out/"initial-probes.json",initial_metrics)
        for step in range(updates):
            if time.monotonic()-started>=900:
                raise TimeoutError("joint-scale training wall budget exhausted")
            indices,quotes,raw,adapted,metrics,candidate,gradients,fields = [],[],{},{},{},None,None,[]
            phase = "schedule_change" if step==50000 else "batch_adaptation"
            if step==50000:
                run["schedule_changes"].append(install_annealing(model,state))
                fixed.write_json(out/"run.json",run)
                step_fn = fixed.compiled_step(model)
            indices,quotes,raw,adapted = batch(step)
            fields = ["adapted_input:"+key for key,value in adapted.items() if not np.isfinite(value).all()]
            if fields:
                raise FloatingPointError("nonfinite six-target adapted batch")
            phase = "optimizer_update"
            metrics,candidate,gradients,flags = step_fn(state,adapted)
            fields = [key for key,value in jax.device_get(flags).items() if not bool(value)]
            if fields:
                raise FloatingPointError("nonfinite joint update: "+", ".join(fields))
            if int(np.asarray(candidate[2][0]))!=step+1:
                raise RuntimeError("joint optimizer count differs from absolute batch step")
            state = candidate
            visits[indices]+=1
            if (step+1)%1000==0 or step+1 in endpoints:
                phase = "fixed_probe"
                point = {"updates":step+1,"seconds":time.monotonic()-started,
                    "batch_training_loss":float(np.asarray(metrics["loss"])),**probe_values()}
                fields = [key for key,value in point.items() if not np.isfinite(value)]
                if fields:
                    metrics,candidate,gradients = point,None,None
                    raise FloatingPointError("nonfinite joint fixed probe")
                curve.append(point); fixed.write_json(out/"learning-curve.json",curve)
                print(json.dumps(point),flush=True)
            if step+1 in endpoints:
                phase = "checkpoint_and_reload"
                directory = out/"checkpoints"/f"update-{step+1:06d}"
                save_checkpoint(directory,workflow,state,run,step+1,curve[-1])
                before = sample(workflow,task,probes["validation"][0][:2],16,9022)
                restored,_ = load_checkpoint(directory)
                after = sample(restored,task,probes["validation"][0][:2],16,9022)
                if not np.array_equal(before,after):
                    raise RuntimeError("joint checkpoint changed same-seed posterior draws")
                last_checkpoint = str(directory.resolve())
            fixed.write_json(out/"progress.json",{"status":"training","completed_updates":step+1,"last_checkpoint":last_checkpoint})
        fixed.write_npz(out/"exposure.npz",{"row_id":data["train"]["row_id"],"presentations":visits})
        result = {"status":"complete","completed_updates":updates,"seconds":time.monotonic()-started,
            "same_seed_reload_exact":True,"last_checkpoint":last_checkpoint,"row_presentations":int(visits.sum()),
            "unique_rows_seen":int(np.count_nonzero(visits)),"reserved_arrays_opened":False,"scientific_acceptance":False}
        fixed.write_json(out/"training.json",result); fixed.write_json(out/"progress.json",result)
        return result
    except BaseException as error:
        if phase!="optimizer_update":
            candidate,gradients = None,None
        if phase in ("initial_probe","fixed_probe"):
            adapted = {**adapted,**{f"probe_{name}_{key}":value for name,(_,_,probe) in probes.items()
                                    for key,value in probe.items()}}
        failure = fixed.capture_failure(out,model,state,candidate,gradients,step=step,indices=indices,quotes=quotes,
            raw=raw,adapted=adapted,fields=fields,metrics=metrics,error=error,last_checkpoint=last_checkpoint,phase=phase)
        fixed.write_json(out/"progress.json",{"status":"failed","completed_updates":failure["completed_finite_updates"],
            "failure":"failure/failure.json","automatic_retry":False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("design","bank","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--smoke-test-only",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        if args.out.exists():
            raise FileExistsError(args.out)
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"])
        try:
            code = process.wait(timeout=max(.001,900-(time.monotonic()-PROCESS_STARTED)))
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); args.out.mkdir(parents=True,exist_ok=True)
            fixed.write_json(args.out/"hard-timeout.json",{"status":"killed_at_900s_limit","automatic_retry":False})
            raise TimeoutError("joint training supervisor wall limit exhausted")
        if code:
            raise SystemExit(code)
    else:
        print(json.dumps(run_training(args.design,args.bank,args.out,protocol=args.protocol,smoke_test_only=args.smoke_test_only)))


if __name__=="__main__":
    main()

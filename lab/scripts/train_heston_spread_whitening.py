"""Bounded paired pilot of frozen diagonal versus full-covariance inputs.

Both fresh bounded-target CouplingFlows see the same 10,000 fixed noisy Heston
training rows, repeated 200 times. Only a frozen, invertible input transform
differs. No pricing, adaptive whitening, or historical checkpoint mutation.
"""
import argparse
import hashlib
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
from scripts import train_heston_spread_scale as scale
from scripts import continue_heston_spread_scale_precision as precision
from scripts import continue_heston_spread_scale as expansion
from scripts import continue_heston_spread_baseline as full_state
from sbi4f_tutorial.shared import whitening

fixed = scale.fixed
MODES = ("diagonal", "zca")
MAX_WALL_SECONDS = 900
SEED = 9011
TRAIN_NAMESPACE, VALIDATION_NAMESPACE = 501, 502
FROZEN_SOURCES = {
    "fixed": "3a8747332a3493adcd9298d701d0d6aab087dada35d8648827d7790330931f38",
    "scale": "93714493768c1db7ce792b931f1dd994e2a5936bafd5da6b1062b666ac4b9068",
    "precision": "767a759a79d764c067d7c702ed30272965eb62a58400208bf58bab7331c9db47",
    "expansion": "5d712d9b37dba7a5662c67ccc84e4148b58a45e328815a07815c40dda7109892",
    "full_state": "3c14cb7461d4c0ab741f0a9d279ff99fad75fa20f58f0f1185f265e72d84fc9b",
}


def sources():
    paths = {"trainer": Path(__file__), "whitening": Path(whitening.__file__),
        **{name: Path(module.__file__) for name, module in (
            ("fixed", fixed), ("scale", scale), ("precision", precision),
            ("expansion", expansion), ("full_state", full_state))}}
    result = {name: {"path": str(path.resolve()), "sha256": fixed.digest(path)}
        for name, path in paths.items()}
    if any(result[name]["sha256"] != digest for name, digest in FROZEN_SOURCES.items()):
        raise ValueError("frozen financial, augmentation, state or precision helper differs")
    return result


def read_task(task):
    value = json.loads(Path(task).read_text()) if isinstance(task, (str, Path)) else task
    scale.validate_task(value)
    return value


def prepare_data(bank, task, *, smoke_test_only=False):
    """Decompress only named train/development members of the preserved bank."""
    task = read_task(task)
    if type(smoke_test_only) is not bool:
        raise ValueError("explicit boolean smoke-test identity required")
    data, bank_binding = scale.load_clean_bank(bank, task,
        128 if smoke_test_only else 20000, smoke_test_only=smoke_test_only)
    counts = {"train": 128 if smoke_test_only else 10000,
        "validation": 32 if smoke_test_only else 100}
    raw, splits = {}, {}
    for split, namespace in (("train", TRAIN_NAMESPACE), ("validation", VALIDATION_NAMESPACE)):
        count = counts[split]
        if len(data[split]["clean"]) < count:
            raise ValueError("fixed pilot prefix is larger than the available bank split")
        selected = {key: value[:count] for key, value in data[split].items()}
        quotes, targets = scale.augmented_batch(selected["clean"], selected["parameters"],
            task["records"]["half_spread"], 0, SEED, namespace=namespace)
        raw[split] = scale.training_data(task, quotes, targets)
        splits[split] = {"rows": count, "namespace": namespace, "augmentation_step": 0,
            "row_id": {"shape":list(selected["row_id"].shape),"dtype":str(selected["row_id"].dtype),
                "sha256":hashlib.sha256(np.ascontiguousarray(selected["row_id"]).tobytes()).hexdigest()},
            "first_row_id": str(selected["row_id"][0]), "last_row_id": str(selected["row_id"][-1]),
            "conditions": fixed.array_summary(raw[split]["observations"]),
            "targets": fixed.array_summary(targets), "quotes": fixed.array_summary(quotes)}
    return raw, {**bank_binding, "splits": splits,
        "noise_recipe": "one independent c~Uniform(.5,2) and Gaussian vector per row; frozen before training; repeat identical noisy rows",
        "reserved_arrays_opened": False, "new_pricing_labels": False}


class FrozenDiagonal:
    """Fixed sample mean/SD baseline; all arithmetic before casting is float64."""
    def __init__(self, mean, standard_deviation, training_rows):
        self.mean = np.array(mean, dtype=np.float64, copy=True)
        self.standard_deviation = np.array(standard_deviation, dtype=np.float64, copy=True)
        if (self.mean.shape != (54,) or self.standard_deviation.shape != (54,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.standard_deviation).all()
                or np.any(self.standard_deviation <= 0) or type(training_rows) is not int or training_rows < 2):
            raise ValueError("finite full-rank fixed diagonal transform required")
        self.training_rows = training_rows
        self.mean.setflags(write=False)
        self.standard_deviation.setflags(write=False)

    def values(self, values):
        if np.iscomplexobj(values):
            raise ValueError("real raw conditions required")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 54 or not values.size or not np.isfinite(values).all():
            raise ValueError("finite ordered 54-feature conditions required")
        return values

    def transform(self, values):
        return (self.values(values)-self.mean)/self.standard_deviation

    def inverse(self, values):
        return self.values(values)*self.standard_deviation+self.mean

    def to_dict(self):
        return {"format": "frozen_diagonal_standardization_v1", "dimension": 54,
            "training_rows": self.training_rows, "ddof": 1, "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist()}

    @classmethod
    def from_dict(cls, value):
        if (set(value) != {"format", "dimension", "training_rows", "ddof", "mean", "standard_deviation"}
                or value["format"] != "frozen_diagonal_standardization_v1"
                or value["dimension"] != 54 or value["ddof"] != 1):
            raise ValueError("invalid fixed diagonal transform record")
        return cls(value["mean"], value["standard_deviation"], value["training_rows"])


def fit_preprocessing(task, train, binding, mode):
    task = read_task(task)
    if mode not in MODES:
        raise ValueError("only fixed diagonal and full ZCA modes are supported")
    values = np.asarray(train["observations"], dtype=np.float64)
    if (values.shape != (binding["splits"]["train"]["rows"], 54)
            or fixed.array_summary(values) != binding["splits"]["train"]["conditions"]):
        raise ValueError("preprocessing requires the exact noisy training prefix")
    covariance = np.cov(values, rowvar=False, ddof=1)
    transform = (whitening.fit_whitening(values, relative_eigenvalue_floor=1e-8) if mode == "zca"
        else FrozenDiagonal(values.mean(axis=0, dtype=np.float64), np.sqrt(np.diag(covariance)), len(values)))
    return {"format": "heston_spread_whitening_preprocessing_v1", "mode": mode,
        "task_fingerprint": fixed.fingerprint(task), "design_sha256": task["design_sha256"],
        "feature_order_sha256": fixed.fingerprint(task["records"]),
        "half_spread": fixed.array_summary(np.asarray(task["records"]["half_spread"], dtype=np.float64)),
        "fit": {"split": "train", **binding["splits"]["train"],
            "bank_record_sha256": binding["bank_record_sha256"], "bank_payload_sha256": binding["bank_payload_sha256"]},
        "noisy_training_covariance": covariance.tolist(), "covariance_sha256": fixed.array_summary(covariance)["sha256"],
        "transform": transform.to_dict(), "source": sources()["whitening"],
        "arithmetic": "raw noisy Y/half_spread, mean, covariance and matrix multiply in NumPy float64; final conditions cast to float32",
        "noise_scope": "transform the entire noisy vector; no additional noise after preprocessing",
        "validation_used_for_fit": False, "reserved_arrays_opened": False}


def preprocessing_transform(record):
    if record.get("format") != "heston_spread_whitening_preprocessing_v1" or record.get("mode") not in MODES:
        raise ValueError("unsupported frozen preprocessing identity")
    if record.get("validation_used_for_fit") is not False or record.get("reserved_arrays_opened") is not False:
        raise ValueError("preprocessing must be fitted from training only")
    covariance = np.asarray(record["noisy_training_covariance"], dtype=np.float64)
    if (covariance.shape != (54,54) or not np.isfinite(covariance).all()
            or fixed.array_summary(covariance)["sha256"] != record["covariance_sha256"]
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12)):
        raise ValueError("saved noisy training covariance differs")
    transform = (whitening.FullCovarianceWhitening.from_dict(record["transform"]) if record["mode"] == "zca"
        else FrozenDiagonal.from_dict(record["transform"]))
    if transform.training_rows != record["fit"]["rows"] or len(transform.mean) != 54:
        raise ValueError("fitted transform training rows or dimensions differ")
    if record["mode"] == "diagonal":
        if not np.array_equal(transform.standard_deviation, np.sqrt(np.diag(covariance))):
            raise ValueError("diagonal scales differ from the saved covariance")
    else:
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.allclose(eigenvalues, transform.eigenvalues, rtol=2e-8, atol=1e-9):
            raise ValueError("ZCA eigenvalues differ from the saved covariance")
        reconstructed = transform.inverse_matrix @ transform.inverse_matrix
        floor = transform.eigenvalue_floor
        # With flooring, the saved inverse is the positive definite floored
        # covariance. Compare its action in the original covariance basis.
        vectors = np.linalg.eigh(covariance)[1]
        expected = (vectors*np.maximum(eigenvalues,floor)) @ vectors.T
        if not np.allclose(reconstructed, expected, rtol=2e-8, atol=2e-8*max(1.,float(np.max(np.abs(covariance))))):
            raise ValueError("ZCA directions differ from the saved noisy covariance")
    return transform


def adapted_data(task, raw, preprocessing):
    # Reuse the repaired bounded logit without passing its float32 conditions
    # into whitening: the full input transform starts from untouched float64.
    result = precision.host_adapt_training(task, raw)
    result["inference_conditions"] = preprocessing_transform(preprocessing).transform(raw["observations"]).astype(np.float32)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("nonfinite transformed training tensors")
    return result


def protocol_contract(bank, task, *, mode, smoke_test_only=False):
    task = read_task(task)
    raw, binding = prepare_data(bank, task, smoke_test_only=smoke_test_only)
    preprocessing = fit_preprocessing(task, raw["train"], binding, mode)
    rows, batch, updates = (128,32,5) if smoke_test_only else (10000,256,8000)
    per_pass = (rows+batch-1)//batch
    return {"format": "heston_spread_whitening_protocol_v1", "predeclared_before_fit": True,
        "mode": mode, "task_fingerprint": fixed.fingerprint(task), "design_sha256": task["design_sha256"],
        "bank_path": str(Path(bank).resolve()), **binding, "train_rows": rows, "validation_rows": binding["splits"]["validation"]["rows"],
        "seed": SEED, "batch_size": batch, "batches_per_pass": per_pass,
        "tail_batch_size": rows-(per_pass-1)*batch, "planned_updates": updates,
        "bank_passes": 200 if not smoke_test_only else None,
        "planned_presentations": 2000000 if not smoke_test_only else 160,
        "checkpoint_updates": [updates], "preprocessing_fingerprint": fixed.fingerprint(preprocessing),
        "architecture": {"family":"spline", "depth":6, "widths":[128,128], "dropout":0.},
        "standardize": ["inference_variables"],
        "optimizer": {"class_name":"Adam", "schedule":{"class_name":"CosineDecay", "initial_learning_rate":5e-4,"decay_steps":updates,"alpha":.1}},
        "data_order": "shared complete per-pass seeded permutation, unpadded tail; fixed noisy training rows repeated",
        "source": sources(), "max_wall_seconds": MAX_WALL_SECONDS, "smoke_test_only": smoke_test_only,
        "target_prior_changed": False, "new_pricing_labels": False, "scientific_acceptance": False,
        "automatic_extension": False}


def selected_contract(protocol, mode):
    if protocol.get("format") != "heston_spread_whitening_study_v1":
        raise ValueError("combined predeclared whitening-study protocol required")
    arms = [arm for arm in protocol.get("arms",[]) if arm.get("mode") == mode]
    if len(arms) != 1:
        raise ValueError("exactly one declared whitening arm required")
    return arms[0]


def build_workflow(task, updates=8000):
    task = read_task(task)
    if type(updates) is not int or updates not in (5,8000):
        raise ValueError("only bounded production or explicit smoke budget supported")
    import bayesflow as bf
    import keras
    if bf.__version__ != "2.1.0" or keras.backend.backend() != "jax":
        raise RuntimeError("locked BayesFlow2.1 JAX environment required")
    adapter = bf.Adapter()
    for name,(lo,hi) in zip(task["target"]["parameter_names"],task["target"]["prior_bounds"],strict=True):
        adapter = adapter.constrain(name,lower=float(lo),upper=float(hi),inclusive="none")
    adapter = (adapter.convert_dtype("float64","float32")
        .concatenate(task["target"]["parameter_names"],into="inference_variables")
        .concatenate(["observations"],into="inference_conditions"))
    optimizer = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(5e-4,updates,alpha=.1))
    flow = bf.networks.CouplingFlow(transform="spline",depth=6,
        subnet_kwargs={"widths":[128,128],"dropout":0.})
    workflow = bf.BasicWorkflow(adapter=adapter,inference_network=flow,
        standardize=["inference_variables"],optimizer=optimizer)
    workflow.estimator.compile(optimizer=optimizer,steps_per_execution=1)
    # Host-adapted training bypasses Adapter.forward. Prime only its structural
    # concatenate metadata, needed by condition-only sampling and inverse
    # target splitting. Midpoints avoid the original near-boundary logit issue;
    # these dummy values never initialize, standardize or train the estimator.
    structural = {name:np.full((1,1),(lo+hi)/2,dtype=np.float64)
        for name,(lo,hi) in zip(task["target"]["parameter_names"],task["target"]["prior_bounds"],strict=True)}
    workflow.adapter({**structural,"observations":np.zeros((1,54),dtype=np.float64)},strict=True)
    return workflow


class WhitenedWorkflow:
    """Only public sampling accepts raw Y/h and applies the fitted state once."""
    def __init__(self, workflow, preprocessing):
        self.workflow, self.preprocessing = workflow, preprocessing
        self.transform = preprocessing_transform(preprocessing)
        self.estimator, self.adapter = workflow.estimator, workflow.adapter

    def sample(self, *, conditions, **kwargs):
        if set(conditions) != {"observations"}:
            raise ValueError("sampling requires raw ordered observations=Y/half_spread only")
        values = self.transform.transform(conditions["observations"]).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("transformed sampling conditions must remain finite in float32")
        return self.workflow.sample(conditions={"observations":values},**kwargs)


def sample(workflow, task, quotes, count, seed=0):
    if not isinstance(workflow,WhitenedWorkflow):
        raise ValueError("the raw-condition preprocessing wrapper is required")
    return scale.sample(workflow, task, quotes, count, seed)


def save_checkpoint(directory, wrapper, state, run, updates, probes):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    fixed.assign_state(wrapper.estimator,state)
    wrapper.estimator.save(directory/"temporary.keras")
    (directory/"temporary.keras").replace(directory/"model.keras")
    fixed.save_state(directory/"state.npz",state)
    fixed.write_json(directory/"preprocessing.json",wrapper.preprocessing)
    record = {**run,"format":"heston_spread_whitening_checkpoint_v1",
        "actual_optimizer_updates":updates,"model_sha256":fixed.digest(directory/"model.keras"),
        "state_sha256":fixed.digest(directory/"state.npz"),
        "preprocessing_sha256":fixed.digest(directory/"preprocessing.json"),
        "state":fixed.state_summary(wrapper.estimator,state),
        "optimizer_schedule":wrapper.estimator.optimizer.get_config()["learning_rate"],
        "probe_metrics":probes,"resume_supported":False}
    fixed.write_json(directory/"checkpoint.json",record)
    return record


def load_checkpoint(directory):
    directory = Path(directory)
    record = json.loads((directory/"checkpoint.json").read_text())
    task = read_task(record["task"])
    contract = record["training_protocol"]
    if (record.get("format") != "heston_spread_whitening_checkpoint_v1"
            or record["task_fingerprint"] != fixed.fingerprint(task)
            or contract["source"] != sources()
            or record["protocol_sha256"] != fixed.digest(record["protocol_path"])
            or selected_contract(json.loads(Path(record["protocol_path"]).read_text()),record["mode"])["contract"] != contract
            or record["names"] != task["target"]["parameter_names"] or record["bounds"] != task["target"]["prior_bounds"]
            or record["actual_optimizer_updates"] not in contract["checkpoint_updates"]
            or record["mode"] != contract["mode"]
            or fixed.digest(directory/"model.keras") != record["model_sha256"]
            or fixed.digest(directory/"state.npz") != record["state_sha256"]
            or fixed.digest(directory/"preprocessing.json") != record["preprocessing_sha256"]):
        raise ValueError("whitening checkpoint source, protocol, task or payload differs")
    preprocessing = json.loads((directory/"preprocessing.json").read_text())
    if (fixed.fingerprint(preprocessing) != contract["preprocessing_fingerprint"]
            or preprocessing["task_fingerprint"] != record["task_fingerprint"]
            or preprocessing["fit"]["conditions"] != contract["splits"]["train"]["conditions"]
            or preprocessing["feature_order_sha256"] != fixed.fingerprint(task["records"])
            or preprocessing["half_spread"] != fixed.array_summary(np.asarray(task["records"]["half_spread"],dtype=np.float64))):
        raise ValueError("saved preprocessing fit/task identity differs")
    preprocessing_transform(preprocessing)
    workflow = build_workflow(task,contract["planned_updates"])
    workflow.load_estimator(directory/"model.keras")
    if any(layer.rate != 0. for layer in fixed.dropout_layers(workflow.estimator)):
        raise ValueError("checkpoint changed zero-dropout architecture")
    full_state.restore_full_state(workflow,record,directory/"state.npz")
    if workflow.estimator.optimizer.get_config()["learning_rate"] != record["optimizer_schedule"]:
        raise ValueError("checkpoint learning-rate schedule differs")
    return WhitenedWorkflow(workflow,preprocessing),record


def run_training(bank, task, out, *, mode, protocol, smoke_test_only=False, wall_seconds=MAX_WALL_SECONDS):
    started = time.monotonic()
    if not 0 < wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("pilot wall limit must be positive and at most900 seconds")
    task, out, protocol = read_task(task), Path(out), Path(protocol)
    contract = protocol_contract(bank,task,mode=mode,smoke_test_only=smoke_test_only)
    declared = selected_contract(json.loads(protocol.read_text()),mode)
    if declared["contract"] != contract or Path(declared["output"]).resolve() != out.resolve():
        raise ValueError("persist the exact source-bound whitening protocol before fitting")
    if out.exists():
        raise FileExistsError(out)
    data,binding = prepare_data(bank,task,smoke_test_only=smoke_test_only)
    preprocessing = fit_preprocessing(task,data["train"],binding,mode)
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
        workflow = build_workflow(task,updates)
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
    wrapper = WhitenedWorkflow(workflow,preprocessing)
    model, state = workflow.estimator, fixed.current_state(workflow.estimator)
    out.mkdir(parents=True)
    fixed.write_json(out/"preprocessing.json",preprocessing)
    lab = Path(__file__).resolve().parents[1]
    run = {"format":"heston_spread_whitening_run_v1","task":task,
        "task_fingerprint":fixed.fingerprint(task),"names":task["target"]["parameter_names"],
        "bounds":task["target"]["prior_bounds"],"mode":mode,"training_protocol":contract,
        "protocol_path":str(protocol.resolve()),"protocol_sha256":fixed.digest(protocol),
        "bank_path":str(Path(bank).resolve()),**binding,"train_rows":contract["train_rows"],
        "batch_size":batch,"planned_updates":updates,"seed":SEED,
        "architecture":contract["architecture"],"standardize":contract["standardize"],
        "preprocessing_path":str((out/"preprocessing.json").resolve()),
        "preprocessing_sha256":fixed.digest(out/"preprocessing.json"),
        "initial_trainable_hashes":initial_hashes,"initial_state":initial_state,
        "paired_initialization":{"fresh":True,"same_seed":SEED,"all_trainable_hashes_exact":True,
            "comparison":"build both modes on the same targets/first rows before any update; no historical warm start"},
        "versions":{"python":platform.python_version(),**{key:metadata.version(key) for key in ("bayesflow","keras","jax","numpy","scipy")}},
        "numeric_configuration":{"jax_enable_x64":bool(jax.config.x64_enabled),"keras_floatx":keras.backend.floatx()},
        "environment_sha256":{name:fixed.digest(lab/name) for name in ("pyproject.toml","pixi.lock")},
        "source":sources(),"smoke_test_only":smoke_test_only,"max_wall_seconds":wall_seconds,
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

    fixed.write_json(out/"initial-probes.json",probe_values())
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
                    save_checkpoint(destination,wrapper,state,run,step,curve[-1])
                    restored,_ = load_checkpoint(destination)
                    after = sample(restored,task,quotes,16,SEED+11)
                    if not np.array_equal(before,after):
                        raise RuntimeError("saved preprocessing/checkpoint changed seeded draws")
                    fixed.assign_state(model,state)
                    last_checkpoint = str(destination.resolve())
        if int(visits.sum()) != contract["planned_presentations"]:
            raise RuntimeError("pilot example budget differs")
        if not smoke_test_only and not np.all(visits==200):
            raise RuntimeError("each fixed training row must be presented200 times")
        fixed.write_npz(out/"exposure.npz",{"row_id":np.array([f"train:{i:08d}" for i in range(len(visits))]),"presentations":visits})
        result = {"status":"complete","completed_updates":step,"completed_bank_passes":step//contract["batches_per_pass"],
            "presentations":int(visits.sum()),"seconds":time.monotonic()-started,"same_seed_reload_exact":True,
            "paired_initial_trainable_hashes_exact":True,"last_checkpoint":last_checkpoint,
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
    for name in ("bank","task","out","protocol"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--mode",choices=MODES,required=True)
    parser.add_argument("--smoke-test-only",action="store_true")
    parser.add_argument("--worker",action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_training(args.bank,args.task,args.out,mode=args.mode,protocol=args.protocol,
            smoke_test_only=args.smoke_test_only)
    else:
        command = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
        completed = subprocess.run(command,timeout=MAX_WALL_SECONDS)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

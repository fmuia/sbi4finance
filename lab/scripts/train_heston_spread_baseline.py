"""Bounded five-target Heston training with fixed c=1 and observed spreads.

Reads only training/development members of an immutable historical clean bank.
There is no pricing, tau, inferred noise scale, mask training or asset publication.
The CLI supervises its worker with a hard wall-clock timeout, including imports,
compilation, probes and serialization. Checkpoints are diagnostic endpoints.
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

PROCESS_STARTED = time.monotonic()

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_full_surface_budget import BatchPlan, schedule_record
from sbi4finance.calibration.heston_spread import HestonSpreadPilot


GROUPS = ("trainable", "non_trainable", "optimizer", "metrics")
PRESERVED_DESIGN_SHA256 = "4b84185e7a072844fd6a3ce0f79aed983d39502607a7a76c79dbf7196f9ed547"
PRESERVED_BANK_SHA256 = "35117eb977867ffb5b73e95cc2da6327f74820b2a74be96d223c85dfc8400fbb"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if hasattr(value, "__array__"):
        return _jsonable(np.asarray(value))
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def make_task(design_path):
    design = json.loads(Path(design_path).read_text())
    pilot = HestonSpreadPilot(design.get("records", design), c=1., infer_c=False)
    if len(pilot.records["mid"]) != 54:
        raise ValueError("this development run requires the preserved complete 54-quote design")
    task = {"format": "heston_spread_baseline_task_v1", "target": pilot.task_definition(),
            "design_sha256": digest(design_path),
            "records": {key: value.tolist() for key, value in pilot.records.items()},
            "conditioning": "complete ordered noisy prices / fixed half_spread; no additional channels",
            "scope": "fixed-design development experiment; no scientific acceptance"}
    return task, pilot


def validate_task(task):
    if task.get("format") != "heston_spread_baseline_task_v1":
        raise ValueError("unsupported spread training task")
    pilot = HestonSpreadPilot(task["records"], c=1., infer_c=False)
    pilot.validate_task(task["target"])
    if pilot.infer_c or pilot.c != 1. or len(pilot.records["mid"]) != 54:
        raise ValueError("only fixed c=1 five-target complete 54-quote inference is supported")
    return pilot


def condition_data(task, quotes):
    h = np.asarray(task["records"]["half_spread"], dtype=float)
    values = np.asarray(quotes, dtype=float)
    if (np.iscomplexobj(quotes) or values.ndim != 2 or values.shape[1] != len(h)
            or not len(values) or not np.isfinite(values).all()):
        raise ValueError("complete finite real noisy quote rows required; negative prices are permitted")
    return {"observations": values / h}


def training_data(task, quotes, parameters):
    data = condition_data(task, quotes)
    values = np.asarray(parameters, dtype=float)
    bounds = np.asarray(task["target"]["prior_bounds"])
    if (np.iscomplexobj(parameters) or values.shape != (len(quotes), 5)
            or not np.isfinite(values).all() or np.any(values <= bounds[:, 0])
            or np.any(values >= bounds[:, 1])):
        raise ValueError("five finite physical targets strictly inside the frozen prior required")
    return {**data, **{name: values[:, index:index+1]
                      for index, name in enumerate(task["target"]["parameter_names"])}}


def build_workflow(task, schedule_updates=10000):
    """Public BF2.1 spline workflow; no generating c or tau conditioning."""
    validate_task(task)
    import bayesflow as bf
    import keras
    if bf.__version__ != "2.1.0" or keras.backend.backend() != "jax":
        raise RuntimeError("use the locked BayesFlow 2.1.0 JAX core environment")
    adapter = bf.Adapter()
    names = task["target"]["parameter_names"]
    for name, (lo, hi) in zip(names, task["target"]["prior_bounds"], strict=True):
        adapter = adapter.constrain(name, lower=float(lo), upper=float(hi), inclusive="none")
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(names, into="inference_variables")
               .concatenate(["observations"], into="inference_conditions"))
    optimizer = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-4, decay_steps=schedule_updates, alpha=.1))
    flow = bf.networks.CouplingFlow(transform="spline", subnet_kwargs={"widths": [128, 128], "dropout": .05})
    workflow = bf.BasicWorkflow(adapter=adapter, inference_network=flow,
                               standardize=["inference_variables", "inference_conditions"], optimizer=optimizer)
    workflow.estimator.compile(optimizer=optimizer, steps_per_execution=1)
    return workflow


def sample(workflow, task, quotes, count, seed=0):
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive integer draw count and nonnegative integer seed required")
    values = workflow.sample(num_samples=count, conditions=condition_data(task, quotes),
                             sample_shape=(), seed=seed)
    result = np.concatenate([np.asarray(values[name]) for name in task["target"]["parameter_names"]], axis=-1)
    if result.shape != (len(quotes), count, 5) or not np.isfinite(result).all():
        raise FloatingPointError("invalid five-target posterior draws")
    return result


def load_clean_bank(directory, task, train_rows=20000, *, smoke_test_only=False):
    """Hash the payload, but decompress ONLY explicitly named nonreserved arrays."""
    directory = Path(directory)
    record = json.loads((directory / "bank.json").read_text())
    identity, target = record["identity"], task["target"]
    original = identity["task"]
    controls = original["pricing_controls"]
    if (record.get("format") != "heston_pilot_campaign_clean_bank_v1"
            or original["design_sha256"] != task["design_sha256"]
            or identity["task_fingerprint"] != fingerprint(original)
            or original["parameter_names"][:5] != target["parameter_names"]
            or original["prior_bounds"][:5] != target["prior_bounds"]
            or controls["measurement_spread_multiplier"] != target["pricing_controls"]["c_min"]
            or controls["numerical_fraction"] != target["pricing_controls"]["numerical_fraction"]
            or controls["quadrature_ladder"] != target["pricing_controls"]["quadrature_ladder"]):
        raise ValueError("historical clean bank differs in design, physical prior or numerical gate")
    payload_path = directory / record["payload"]
    if payload_path.resolve().parent != directory.resolve() or digest(payload_path) != record["payload_sha256"]:
        raise ValueError("historical clean bank payload path or hash differs")
    if not smoke_test_only and (task["design_sha256"] != PRESERVED_DESIGN_SHA256
                                or record["payload_sha256"] != PRESERVED_BANK_SHA256):
        raise ValueError("production requires the exact previously audited design and clean bank")
    if (type(train_rows) is not int or not 1 <= train_rows <= identity["train_rows"]
            or (not smoke_test_only and (identity["train_rows"] != 100000
                                         or identity["validation_rows"] != 5000 or train_rows != 20000))):
        raise ValueError("production uses the 20k prefix of the preserved 100k/5k bank")
    with np.load(payload_path, allow_pickle=False) as archive:
        data = {split: {key: archive[f"{split}_{key}"].copy()
                        for key in ("parameters", "clean", "row_id")}
                for split in ("train", "validation")}
    for split, values in data.items():
        count = identity[split + "_rows"]
        if (values["parameters"].shape != (count, 5) or values["clean"].shape != (count, 54)
                or values["row_id"].shape != (count,) or len(np.unique(values["row_id"])) != count
                or not np.isfinite(values["clean"]).all()):
            raise ValueError("invalid preserved bank shapes, IDs or clean values")
        if not np.array_equal(values["row_id"], np.array([f"{split}:{index:08d}" for index in range(count)])):
            raise ValueError("preserved row IDs differ from their declared split/prefix protocol")
        training_data(task, values["clean"], values["parameters"])
    data["train"] = {key: value[:train_rows] for key, value in data["train"].items()}
    return data, {"bank_path": str(directory.resolve()), "bank_record_sha256": digest(directory / "bank.json"),
                  "bank_payload_sha256": record["payload_sha256"], "bank_identity": identity,
                  "reserved_arrays_opened": False, "simulation_calls": 0}


def noisy_quotes(clean, halfspread, step, seed, *, namespace=0):
    """Explicit independent streams; repeated dataset/probe requests are exact."""
    rng = np.random.default_rng(np.random.SeedSequence([0x53505244, seed, namespace, step]))
    return np.asarray(clean) + np.asarray(halfspread) * rng.standard_normal(np.shape(clean))


def adapted_batch(workflow, task, train, plan, step, seed):
    indices = plan.indices(step)
    quotes = noisy_quotes(train["clean"][indices], task["records"]["half_spread"], step, seed)
    # Bank targets were validated once. Preserve even an invalid simulated
    # payload here so the update guard can save its exact adapted tensors.
    raw = {"observations": quotes / np.asarray(task["records"]["half_spread"]),
           **{name: train["parameters"][indices, index:index+1]
              for index, name in enumerate(task["target"]["parameter_names"])}}
    adapted = workflow.adapter(raw)
    return indices, quotes, raw, adapted


def model_variables(model):
    return (model.trainable_variables, model.non_trainable_variables,
            model.optimizer.variables, model.metrics_variables)


def current_state(model):
    return tuple([variable.value for variable in group] for group in model_variables(model))


def assign_state(model, state):
    for variables, arrays in zip(model_variables(model), state, strict=True):
        if len(variables) != len(arrays):
            raise ValueError("model state variable count differs")
        for variable, value in zip(variables, arrays, strict=True):
            variable.assign(value)


def array_summary(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    return {"shape": list(values.shape), "dtype": str(values.dtype), "all_finite": bool(finite.all()),
            "nonfinite_count": int(values.size - finite.sum()),
            "sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}


def state_summary(model, state):
    return {name: [{"path": variable.path, **array_summary(array)}
                   for variable, array in zip(variables, arrays, strict=True)]
            for name, variables, arrays in zip(GROUPS, model_variables(model), state, strict=True)}


def save_state(path, state):
    write_npz(path, {f"{name}_{index:04d}": np.asarray(value)
                     for name, arrays in zip(GROUPS, state, strict=True) for index, value in enumerate(arrays)})


def compiled_step(model):
    """Native BF update plus gradient diagnostics from the identical input state.

    Both functional passes use identical dropout counters and standardization.
    JAX may merge common computation. No state from the diagnostic pass commits.
    """
    import jax
    import jax.numpy as jnp
    def finite(tree):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(tree)]))

    @jax.jit
    def step(state, data):
        (_, _), gradients = jax.value_and_grad(model.stateless_compute_metrics, has_aux=True)(
            state[0], state[1], state[3], data=data, stage="training")
        metrics, candidate = model.stateless_train_step(state, data)
        flags = {"batch_metrics": finite(metrics), "gradients": finite(gradients),
                 "post_update_state": finite(candidate)}
        return metrics, candidate, gradients, flags
    return step


def fixed_probe(workflow, task, split, rows, seed, namespace):
    count = min(rows, len(split["clean"]))
    quotes = noisy_quotes(split["clean"][:count], task["records"]["half_spread"], 0, seed, namespace=namespace)
    return quotes, workflow.adapter(training_data(task, quotes, split["parameters"][:count]))


def checkpoint(directory, workflow, state, run, updates, probes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    assign_state(workflow.estimator, state)
    temporary = directory / "temporary.keras"
    workflow.estimator.save(temporary)
    temporary.replace(directory / "model.keras")
    save_state(directory / "state.npz", state)
    record = {**run, "format": "heston_spread_baseline_checkpoint_v1", "actual_optimizer_updates": updates,
              "model_sha256": digest(directory / "model.keras"), "state_sha256": digest(directory / "state.npz"),
              "state": state_summary(workflow.estimator, state), "probe_metrics": probes,
              "resume_supported": False, "resume_scope": "state sidecar retained; no exact continuation proof supplied"}
    write_json(directory / "checkpoint.json", record)
    return record


def load_checkpoint(directory):
    """Strict inference reload; does not advertise exact training continuation."""
    directory = Path(directory)
    record = json.loads((directory / "checkpoint.json").read_text())
    if (record.get("format") != "heston_spread_baseline_checkpoint_v1"
            or fingerprint(record["task"]) != record["task_fingerprint"]
            or digest(directory / "model.keras") != record["model_sha256"]
            or digest(directory / "state.npz") != record["state_sha256"]
            or record["names"] != record["task"]["target"]["parameter_names"]
            or record["bounds"] != record["task"]["target"]["prior_bounds"]
            or record["design_sha256"] != record["task"]["design_sha256"]
            or not 0 <= record["actual_optimizer_updates"] <= record["planned_updates"]):
        raise ValueError("spread checkpoint target, payload or update identity differs")
    workflow = build_workflow(record["task"], record["schedule_updates"])
    workflow.load_estimator(directory / "model.keras")
    model = workflow.estimator
    actual = state_summary(model, current_state(model))
    # .keras omits auxiliary SeedGenerator state; inference uses explicit seeds.
    # Do not claim its native archive alone can reproduce future dropout draws.
    for group in ("trainable", "optimizer"):
        if ([row["sha256"] for row in actual[group]] != [row["sha256"] for row in record["state"][group]]):
            raise ValueError(f"checkpoint {group} state differs after archive reload")
    old = [row["sha256"] for row in record["state"]["non_trainable"] if "seed_generator_state" not in row["path"]]
    new = [row["sha256"] for row in actual["non_trainable"] if "seed_generator_state" not in row["path"]]
    if old != new or int(model.optimizer.iterations.numpy()) != record["actual_optimizer_updates"]:
        raise ValueError("checkpoint standardization or optimizer iteration differs")
    return workflow, record


def capture_failure(out, model, state, candidate, gradients, *, step, indices, quotes, raw, adapted,
                    fields, metrics, error, last_checkpoint, phase="optimizer_update"):
    """Atomic manifest is written last; preserve all available arrays verbatim."""
    destination = Path(out) / "failure"
    destination.mkdir(exist_ok=False)
    save_state(destination / "pre-update-state.npz", state)
    arrays = {"indices": np.asarray(indices), "quotes": np.asarray(quotes),
              **{"raw_" + key: np.asarray(value) for key, value in raw.items()},
              **{"adapted_" + key: np.asarray(value) for key, value in adapted.items()}}
    if gradients is not None:
        arrays.update({f"gradient_{index:04d}": np.asarray(value) for index, value in enumerate(gradients)})
    write_npz(destination / "offending-batch-and-gradients.npz", arrays)
    if candidate is not None:
        save_state(destination / "post-update-state.npz", candidate)
    assign_state(model, state)
    archive_error = None
    try:
        model.save(destination / "last-valid-pre-update.keras")
    except Exception as saving_error:
        archive_error = str(saving_error)
    record = {"format": "heston_spread_first_failure_v1", "absolute_batch_step": step,
              "phase": phase, "attempted_optimizer_update": step + 1,
              "completed_finite_updates": int(np.asarray(state[2][0])),
              "saved_state_scope": "last finite committed state; pre-update for optimizer failures, post-update for probe/checkpoint failures",
              "failed_fields": fields, "metrics": metrics, "error_type": type(error).__name__, "error": str(error),
              "gradients_available": gradients is not None,
              "gradient_scope": "same pre-update state/data/dropout stream; no diagnostic state committed" if gradients is not None else "no gradient result available before this failure",
              "pre_update_state": state_summary(model, state),
              "post_update_state": None if candidate is None else state_summary(model, candidate),
              "adapted_batch": {key: array_summary(value) for key, value in adapted.items()},
              "batch_tensors_available": bool(adapted),
              "batch_capture_scope": ("exact tensors returned for the indicated batch/probe phase" if adapted
                  else "batch construction did not return; tensors unavailable, never borrowed from a previous batch"),
              "last_scheduled_checkpoint": last_checkpoint, "last_valid_archive_error": archive_error,
              "reserved_arrays_opened": False, "resume_supported": False,
              "files_sha256": {path.name: digest(path) for path in destination.iterdir() if path.is_file()}}
    write_json(destination / "failure.json", record)
    return record


def run_training(design, bank, out, *, train_rows=20000, updates=10000, batch_size=128,
                 seed=9011, probe_rows=512, log_every=1000, wall_seconds=3600,
                 smoke_test_only=False, checkpoint_updates=None):
    """Fresh finite run. The CLI additionally enforces the process wall limit."""
    started = time.monotonic()
    if (not 0 < wall_seconds <= 3600 or seed < 0 or batch_size < 1 or probe_rows < 1 or log_every < 1
            or (not smoke_test_only and (updates != 10000 or batch_size != 128 or seed != 9011))
            or (smoke_test_only and not 1 <= updates <= 5)):
        raise ValueError("production settings are fixed; smoke permits only 1–5 updates")
    checkpoints = tuple(checkpoint_updates or ((updates,) if smoke_test_only else (2000, 10000)))
    if (any(type(value) is not int or not 1 <= value <= updates for value in checkpoints)
            or (not smoke_test_only and checkpoints != (2000, 10000))):
        raise ValueError("checkpoint endpoints must match the predeclared run")
    out, design, bank = map(Path, (out, design, bank))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "progress.json", {"status": "preparing", "completed_updates": 0})
    task, pilot = make_task(design)
    data, bank_record = load_clean_bank(bank, task, train_rows, smoke_test_only=smoke_test_only)
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import bayesflow as bf  # finish import-time seed consumption before setting seed
    import jax
    import keras
    from sbi4f_tutorial.heston_options import model as heston_model
    keras.utils.set_random_seed(seed)
    workflow = build_workflow(task)
    model = workflow.estimator
    plan = BatchPlan(train_rows, batch_size, updates, seed)
    initial_batch = adapted_batch(workflow, task, data["train"], plan, 0, seed)[-1]
    model.build_from_data(keras.tree.map_structure(keras.ops.convert_to_tensor, initial_batch))
    model.optimizer.build(model.trainable_variables)
    state = current_state(model)
    step_fn = compiled_step(model)
    probes = {name: fixed_probe(workflow, task, data[name], probe_rows, seed, namespace)
              for name, namespace in (("train", 1), ("validation", 2))}
    run = {"format": "heston_spread_baseline_run_v1", "task": task, "task_fingerprint": fingerprint(task),
           "names": pilot.names, "bounds": pilot.bounds.tolist(), "design_sha256": digest(design),
           "design_path": str(design.resolve()), **bank_record, "train_rows": train_rows,
           "planned_updates": updates, "schedule_updates": 10000, "seed": seed, "batch_size": batch_size,
           "validation_rows_available": len(data["validation"]["clean"]),
           "seconds_scope": "run_training invocation including lazy neural imports; CLI supervisor additionally includes process/script setup",
           "checkpoint_updates": list(checkpoints), "wall_seconds": wall_seconds,
           "architecture": {"family": "spline", "widths": [128, 128], "subnet_dropout": .05},
           "source_sha256": {"trainer": digest(__file__), "batch_plan": digest(sys.modules[BatchPlan.__module__].__file__),
                             "spread_pilot": digest(sys.modules[HestonSpreadPilot.__module__].__file__),
                             "heston_pilot": digest(Path(sys.modules[HestonSpreadPilot.__module__].__file__).with_name("heston_pilot.py")),
                             "heston_model": digest(heston_model.__file__)},
           "environment_sha256": {name: digest(Path(__file__).resolve().parents[1] / name)
                                   for name in ("pyproject.toml", "pixi.lock")},
           "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
           "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}},
           "devices": [str(device) for device in jax.devices()], "optimizer_schedule": schedule_record(model.optimizer, 10000),
           "numeric_configuration": {"jax_enable_x64": bool(jax.config.x64_enabled), "keras_floatx": keras.backend.floatx()},
           "augmentation": "PCG64 SeedSequence([0x53505244, seed, namespace, step]); namespace0 train,1 fixed train probe,2 fixed validation probe; independent Gaussian SD exactly halfspread",
           "batch_plan": "BatchPlan seeded pass permutations; probes never advance training streams",
           "probe_rows": {name: len(values[0]) for name, values in probes.items()},
           "scientific_acceptance": False, "checkpoint_selection": "predeclared endpoints; validation never selects",
           "training_semantics": "BF2.1 stateless_train_step; same-input diagnostic gradients do not commit state",
           "resume_supported": False}
    write_json(out / "run.json", run)
    curve, last_checkpoint, visits = [], None, np.zeros(train_rows, dtype=np.int64)
    row_order, batch_sizes = hashlib.sha256(), {}
    metrics, gradients, candidate, raw, adapted, indices, quotes = {}, None, None, {}, {}, [], []
    step, phase, fields = 0, "initial_probe", []

    def check_deadline():
        if time.monotonic() - started >= wall_seconds:
            raise TimeoutError("bounded training wall-clock allocation exhausted")

    def probe_values():
        assign_state(model, state)
        return {name + "_probe_loss": float(np.asarray(model.compute_metrics(**values[1], stage="validation")["loss"]))
                for name, values in probes.items()}

    try:
        check_deadline()
        initial_metrics = probe_values()
        fields = [key for key, value in initial_metrics.items() if not np.isfinite(value)]
        if fields:
            metrics = initial_metrics
            raise FloatingPointError("nonfinite initial validation-stage probe")
        write_json(out / "initial-probes.json", initial_metrics)
        for step in range(updates):
            phase = "before_update"
            fields = []
            check_deadline()
            candidate, gradients, metrics = None, None, {}
            # A constructor/adapter exception cannot return its partial tensors.
            # Clear the previous batch before calling it, preserving only the
            # authoritative committed state and exact attempted batch number.
            indices, quotes, raw, adapted, fields = [], [], {}, {}, []
            phase = "batch_adaptation"
            indices, quotes, raw, adapted = adapted_batch(workflow, task, data["train"], plan, step, seed)
            phase = "before_update"
            fields = ["adapted_input:" + key for key, value in adapted.items() if not np.isfinite(value).all()]
            if fields:
                raise FloatingPointError("nonfinite adapted input before optimizer update")
            phase = "optimizer_update"
            metrics, candidate, gradients, flags = step_fn(state, adapted)
            flags = jax.device_get(flags)
            fields = [key for key, value in flags.items() if not bool(value)]
            if fields:
                raise FloatingPointError("nonfinite native training result: " + ", ".join(fields))
            actual = int(np.asarray(candidate[2][0]))
            if actual != step + 1:
                raise RuntimeError("native optimizer iteration differs from absolute batch step")
            state = candidate
            visits[indices] += 1
            row_order.update(np.asarray(indices, dtype="<i8").tobytes())
            batch_sizes[str(len(indices))] = batch_sizes.get(str(len(indices)), 0) + 1
            if (step + 1) % log_every == 0 or step + 1 in checkpoints:
                phase = "fixed_probe"
                point = {"updates": step + 1, "seconds": time.monotonic()-started,
                         "batch_training_loss": float(np.asarray(metrics["loss"])), **probe_values()}
                fields = [key for key, value in point.items() if not np.isfinite(value)]
                if fields:
                    # This update is finite; preserve it, and label the probe failure separately.
                    metrics = point
                    candidate, gradients = None, None
                    raise FloatingPointError("nonfinite validation-stage probe: " + ", ".join(fields))
                curve.append(point)
                write_json(out / "learning-curve.json", curve)
                print(json.dumps(point), flush=True)
            if step + 1 in checkpoints:
                phase = "checkpoint_and_reload"
                directory = out / "checkpoints" / f"update-{step+1:06d}"
                checkpoint(directory, workflow, state, run, step+1, curve[-1])
                before = sample(workflow, task, probes["validation"][0][:2], 16, seed+11)
                restored, _ = load_checkpoint(directory)
                after = sample(restored, task, probes["validation"][0][:2], 16, seed+11)
                if not np.array_equal(before, after):
                    raise RuntimeError("checkpoint changed exact same-seed posterior draws")
                last_checkpoint = str(directory.resolve())
            write_json(out / "progress.json", {"status": "training", "completed_updates": step+1,
                "seconds": time.monotonic()-started, "last_checkpoint": last_checkpoint,
                "first_nonfinite_checked_every_update": True})
        check_deadline()
        write_npz(out / "exposure.npz", {"row_id": data["train"]["row_id"], "presentations": visits})
        result = {"status": "complete", "completed_updates": updates, "seconds": time.monotonic()-started,
                  "last_checkpoint": last_checkpoint, "same_seed_reload_exact": True,
                  "unique_rows_seen": int(np.count_nonzero(visits)), "row_presentations": int(visits.sum()),
                  "batch_size_counts": batch_sizes, "ordered_row_indices_sha256": row_order.hexdigest(),
                  "reserved_arrays_opened": False, "scientific_acceptance": False}
        write_json(out / "training.json", result)
        write_json(out / "progress.json", result)
        return result
    except BaseException as error:
        if phase != "optimizer_update":
            candidate, gradients = None, None
        if phase in ("initial_probe", "fixed_probe"):
            adapted = {**adapted, **{f"probe_{name}_{key}": value
                       for name, (_, data) in probes.items() for key, value in data.items()}}
        failure = capture_failure(out, model, state, candidate, gradients, step=step, indices=indices,
            quotes=quotes, raw=raw, adapted=adapted, fields=fields, metrics=metrics, error=error,
            last_checkpoint=last_checkpoint, phase=phase)
        write_json(out / "progress.json", {"status": "failed", "completed_updates": failure["completed_finite_updates"],
            "seconds": time.monotonic()-started, "failure": "failure/failure.json", "last_checkpoint": last_checkpoint})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design", "bank", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=3600.)
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 3600:
        parser.error("wall-seconds must be positive and at most 3600")
    if not args.worker:
        if args.out.exists():
            raise FileExistsError(args.out)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        process = subprocess.Popen(command)
        try:
            code = process.wait(timeout=max(.001, args.wall_seconds - (time.monotonic() - PROCESS_STARTED)))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(args.out / "hard-timeout.json", {"status": "killed_at_wall_limit", "wall_seconds": args.wall_seconds,
                "scope": "supervisor stopped entire worker; only already committed checkpoints/progress recoverable",
                "automatic_retry": False, "scientific_acceptance": False})
            raise TimeoutError("hard process wall-clock allocation exhausted")
        if code:
            raise SystemExit(code)
        return
    result = run_training(args.design, args.bank, args.out, wall_seconds=args.wall_seconds,
        **({"smoke_test_only": True, "updates": 5, "train_rows": 128, "probe_rows": 32,
            "log_every": 5, "checkpoint_updates": (2, 5)} if args.smoke_test_only else {}))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

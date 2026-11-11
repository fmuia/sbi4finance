"""One source-bound 100k extension of the immutable PR87 FlowMatching run.

Keep its original fixed noisy 10k rows, validation rows and whitening; append
90k fresh simulator draws from its retained post-validation NumPy state.
The model and native offline optimizer are rebuilt, never resumed.
"""

import argparse
from copy import deepcopy
from importlib import metadata
import inspect
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import reproduce_heston_pr87 as parent_recipe

PARENT_PRODUCER_SHA256 = "f80246ceae9f3b0a7a1a2d4dcdf0da109311dd942bb2fd8f42d1e2fe0205ca0a"
PARENT_DATA_SHA256 = "eff82fa69dfdb7786d57c24af1d056f90318d053f2511897ce4ef45df24ced8e"
MAX_WALL_SECONDS = 3600
SEED = parent_recipe.SEED
GROUPS = parent_recipe.GROUPS
# These helpers implement the immutable original recipe, not a second model.
digest = parent_recipe.digest
fingerprint = parent_recipe.fingerprint
write_json = parent_recipe.write_json
write_npz = parent_recipe.write_npz
array_summary = parent_recipe.array_summary
read_task = parent_recipe.read_task
build_workflow = parent_recipe.build_workflow
NotebookWhitening = parent_recipe.NotebookWhitening
RawQuoteWorkflow = parent_recipe.RawQuoteWorkflow
sample = parent_recipe.sample
_runtime = parent_recipe._runtime
_stable_config = parent_recipe._stable_config
_state = parent_recipe._state
_state_summary = parent_recipe._state_summary
_assign_state = parent_recipe._assign_state
HestonSpreadPilot = parent_recipe.HestonSpreadPilot


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def source_bindings():
    sources = parent_recipe.source_bindings()
    sources["parent_producer"] = sources.pop("producer")
    sources["producer"] = _identity(__file__)
    return sources


def _read_parent(directory, *, smoke_test_only=False):
    """Verify all parent inputs without opening model weights or querying it."""
    if type(smoke_test_only) is not bool:
        raise ValueError("explicit boolean smoke identity required")
    directory = Path(directory).resolve()
    record = json.loads((directory / "run.json").read_text())
    task = read_task(record["task"])
    contract = record["protocol"]
    rows, val_rows, epochs = (128, 32, 2) if smoke_test_only else (10000, 100, 200)
    protocol_path = Path(record["protocol_path"])
    declaration = json.loads(protocol_path.read_text())
    declared = declaration.get("contract", declaration)
    final = directory / "checkpoints" / f"epoch-{epochs:06d}"
    checkpoint = json.loads((final / "checkpoint.json").read_text())
    checkpoint_entry = [item for item in record["checkpoints"] if item["epoch"] == epochs]
    if (record.get("format") != "heston_pr87_reproduction_run_v1"
            or record.get("status") != "complete"
            or record.get("smoke_test_only") is not smoke_test_only
            or contract.get("format") != "heston_pr87_reproduction_protocol_v1"
            or contract.get("smoke_test_only") is not smoke_test_only
            or (contract["train_rows"], contract["validation_rows"], contract["epochs"], contract["batch_size"])
                != (rows, val_rows, epochs, 256)
            or record["task_fingerprint"] != fingerprint(task)
            or contract["task_fingerprint"] != fingerprint(task)
            or record["parameter_names"] != task["target"]["parameter_names"]
            or record["bounds"] != task["target"]["prior_bounds"]
            or record["completed_epochs"] != epochs
            or record["actual_optimizer_updates"] != epochs * ((rows + 255) // 256)
            or record["data_payload"] != "data.npz"
            or digest(directory / "run.json") != digest(directory / "training.json")
            or digest(protocol_path) != record["protocol_sha256"] or declared != contract
            or digest(parent_recipe.__file__) != PARENT_PRODUCER_SHA256
            or contract["source"]["producer"]["sha256"] != PARENT_PRODUCER_SHA256
            or parent_recipe.source_bindings() != contract["source"]
            or digest(contract["notebook"]["path"]) != parent_recipe.NOTEBOOK_SHA256
            or len(checkpoint_entry) != 1
            or Path(checkpoint_entry[0]["path"]).resolve() != final
            or digest(final / "checkpoint.json") != checkpoint_entry[0]["checkpoint_sha256"]
            or checkpoint.get("format") != "heston_pr87_reproduction_checkpoint_v1"
            or checkpoint["task"] != task or checkpoint["protocol"] != contract
            or checkpoint["protocol_sha256"] != record["protocol_sha256"]
            or checkpoint["native_epoch"] != epochs
            or checkpoint["data_payload_sha256"] != record["data_payload_sha256"]
            or checkpoint["data_arrays"] != record["data_arrays"]
            or digest(directory / "data.npz") != record["data_payload_sha256"]
            or (not smoke_test_only and record["data_payload_sha256"] != PARENT_DATA_SHA256)
            or digest(directory / "whitening.json") != checkpoint["whitening_sha256"]
            or digest(final / "whitening.json") != checkpoint["whitening_sha256"]):
        raise ValueError("immutable parent task, source, data, whitening or completed-run identity differs")
    if "contract" in declaration and (declaration["task"] != task
            or digest(declaration["task_file"]["path"]) != declaration["task_file"]["sha256"]):
        raise ValueError("parent declared task differs")
    with np.load(directory / "data.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    _validate_shapes(arrays, task, rows, val_rows)
    if {key: array_summary(value) for key, value in arrays.items()} != record["data_arrays"]:
        raise ValueError("parent arrays differ from their retained hashes")
    whitening = NotebookWhitening.from_dict(json.loads((directory / "whitening.json").read_text()))
    if whitening.to_dict() != NotebookWhitening(arrays["train_observations"]).to_dict():
        raise ValueError("parent whitening differs from the exact original training-only fit")
    rng_state = deepcopy(record["numpy_generator_states"]["after_validation"])
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    identity = {"directory": str(directory), "run": _identity(directory / "run.json"),
        "training": _identity(directory / "training.json"), "data": _identity(directory / "data.npz"),
        "whitening": _identity(directory / "whitening.json"), "protocol": _identity(protocol_path),
        "final_checkpoint": _identity(final / "checkpoint.json"),
        "train_rows": rows, "validation_rows": val_rows, "after_validation_rng_state": rng_state,
        "producer_sha256": PARENT_PRODUCER_SHA256,
        "source_commit": "8bc3721", "results_commit": "6559492"}
    return {"directory": directory, "record": record, "task": task, "arrays": arrays,
            "whitening": whitening, "identity": identity}


def _validate_shapes(arrays, task, train_rows, validation_rows):
    expected = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        expected.update({f"{split}_{name}": (rows, 1) for name in task["target"]["parameter_names"]})
        expected.update({f"{split}_{name}": (rows, 54) for name in ("observations", "clean_prices", "quotes")})
    if set(arrays) != set(expected) or any(
            np.iscomplexobj(arrays[key]) or arrays[key].shape != shape
            or not np.isfinite(arrays[key]).all() for key, shape in expected.items()):
        raise ValueError("finite native raw-target arrays with the exact declared shapes required")
    half_spread = np.asarray(task["records"]["half_spread"])
    for split in ("train", "validation"):
        if not np.array_equal(arrays[split + "_observations"], arrays[split + "_quotes"] / half_spread):
            raise ValueError("raw quotes and spread-scaled native observations differ")


def protocol_contract(parentdir, *, smoke_test_only=False):
    """Resolve the new experiment before generation or fitting (no pricing)."""
    parent = _read_parent(parentdir, smoke_test_only=smoke_test_only)
    bf, keras = _runtime()
    keras.utils.set_random_seed(SEED)
    rows, epochs = (256, 2) if smoke_test_only else (100000, 200)
    batches = (rows + 255) // 256
    optimizer = build_workflow(parent["task"]).build_optimizer(epochs, batches, strategy="offline")
    workflow = build_workflow(parent["task"])
    resolved = {"optimizer": _stable_config(keras.optimizers.serialize(optimizer)),
        "network": _stable_config(keras.saving.serialize_keras_object(workflow.inference_network)),
        "fit_offline_signature": str(inspect.signature(bf.BasicWorkflow.fit_offline)),
        "keras_fit_signature": str(inspect.signature(keras.Model.fit))}
    lr = resolved["optimizer"]["config"]["learning_rate"]["config"]
    return {"format": "heston_pr87_100k_protocol_v1", "predeclared_before_fit": True,
        "parent": parent["identity"], "task_fingerprint": fingerprint(parent["task"]),
        "source": source_bindings(), "train_rows": rows,
        "parent_train_rows": parent["identity"]["train_rows"],
        "appended_rows": rows - parent["identity"]["train_rows"],
        "validation_rows": parent["identity"]["validation_rows"], "epochs": epochs, "batch_size": 256,
        "batches_per_epoch": batches, "tail_batch_rows": rows % 256 or 256,
        "example_presentations": rows * epochs, "expected_optimizer_updates": epochs * batches,
        "checkpoint_epochs": [1, 2] if smoke_test_only else [20, 50, 75, 83, 100, 150, 200],
        "primary_checkpoint": "final_epoch_only",
        "archival_selection": "first strictly smallest finite native val_loss; never restore during fit",
        "seed": SEED, "initialization": "fresh same-seed native workflow; never load or resume parent weights",
        "simulation": "restore parent numpy_generator_states.after_validation in np.default_rng(); bf.make_simulator([prior,observation_model]); sample(appended_rows), all new priors then all new noises",
        "dataset_policy": "original fixed noisy training rows remain exact prefix; append fresh iid draws once; all native validation rows unchanged; no noise refresh per epoch",
        "whitening": "exact immutable parent whitening.json bytes; fitted only on original parent training rows; never refit on appended or validation rows",
        "training": "one native fit_offline(data,validation_data,epochs,batch_size); all other training, shuffle, loss and optimizer defaults retained",
        "target_transform": "raw physical coordinates; concatenate then rename only; no constrain, clipping, rejection or target Jacobian",
        "standardize": "inference_variables", "resolved_defaults": resolved,
        "native_optimizer_budget": {"updates": epochs * batches, "warmup_steps": lr["warmup_steps"],
            "decay_steps": lr["decay_steps"], "initial_learning_rate": lr["initial_learning_rate"],
            "warmup_target": lr["warmup_target"], "alpha": lr["alpha"]},
        "versions": {name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "jaxlib", "numpy")},
        "python": platform.python_version(), "platform": platform.platform(), "backend": keras.backend.backend(),
        "devices": [str(device) for device in __import__("jax").devices()],
        "max_wall_seconds": MAX_WALL_SECONDS, "smoke_test_only": smoke_test_only,
        "automatic_extension": False, "retries": 0, "new_pricing_labels": True,
        "fresh_noise_per_pass": False, "reserved_arrays_opened": False, "scientific_acceptance": False}


def append_data(parent, appended_rows, *, check_budget=lambda: None, progress=lambda count: None):
    """Native sequential simulator sampling, continuing the immutable parent RNG."""
    if type(appended_rows) is not int or appended_rows < 1:
        raise ValueError("positive fresh appended row count required")
    bf, _ = _runtime()
    task = parent["task"]
    model = HestonSpreadPilot(task["records"], infer_c=True, c_bounds=(.5, 2.))
    model.validate_task(task["target"])
    bounds, names = np.asarray(model.bounds), model.parameter_names
    half_spread = np.asarray(model.records["half_spread"])
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(parent["identity"]["after_validation_rng_state"])
    clean_rows, quote_rows = [], []

    def prior():
        check_budget()
        return dict(zip(names, rng.uniform(bounds[:, 0], bounds[:, 1]), strict=True))

    def observation_model(initial_variance, kappa, theta, vol_of_vol, rho, c):
        check_budget()
        parameters = np.array([initial_variance, kappa, theta, vol_of_vol, rho, c])
        clean = model.prices(parameters)
        quotes = clean + c * half_spread * rng.standard_normal(half_spread.size)
        clean_rows.append(clean.copy())
        quote_rows.append(quotes.copy())
        if len(clean_rows) % 1000 == 0 or len(clean_rows) == appended_rows:
            progress(len(clean_rows))
        return {"observations": quotes / half_spread}

    before = deepcopy(rng.bit_generator.state)
    appended = bf.make_simulator([prior, observation_model]).sample(appended_rows)
    fresh = {"train_" + key: np.asarray(value) for key, value in appended.items()}
    fresh.update(train_clean_prices=np.asarray(clean_rows), train_quotes=np.asarray(quote_rows))
    arrays = {key: np.concatenate((value, fresh[key]), axis=0) if key.startswith("train_")
              else value.copy() for key, value in parent["arrays"].items()}
    total = parent["identity"]["train_rows"] + appended_rows
    _validate_shapes(arrays, task, total, parent["identity"]["validation_rows"])
    _validate_parent_prefix(arrays, parent)
    return arrays, {"before_append": before, "after_append": deepcopy(rng.bit_generator.state)}


def _validate_parent_prefix(arrays, parent):
    rows = parent["identity"]["train_rows"]
    for key, original in parent["arrays"].items():
        candidate = arrays[key][:rows] if key.startswith("train_") else arrays[key]
        if not np.array_equal(candidate, original) or candidate.dtype != original.dtype:
            raise ValueError("original parent training prefix or native validation rows changed")


def _save_checkpoint(directory, workflow, run_record, whitening_path, epoch, logs):
    directory.mkdir(parents=True, exist_ok=False)
    model = workflow.estimator
    state = _state(model)
    summary = _state_summary(model, state)
    model.save(directory / "model.keras")
    if summary != _state_summary(model, _state(model)):
        raise RuntimeError("native archive serialization changed training state")
    write_npz(directory / "state.npz", {f"{group}_{index:04d}": value
        for group, values in zip(GROUPS, state, strict=True) for index, value in enumerate(values)})
    shutil.copyfile(whitening_path, directory / "whitening.json")
    record = {**run_record, "format": "heston_pr87_100k_checkpoint_v1", "native_epoch": epoch,
        "native_logs": logs, "actual_optimizer_updates": int(model.optimizer.iterations.numpy()),
        "model_sha256": digest(directory / "model.keras"), "state_sha256": digest(directory / "state.npz"),
        "whitening_sha256": digest(directory / "whitening.json"), "state": summary,
        "resolved_optimizer": _stable_config(__import__("keras").optimizers.serialize(model.optimizer)),
        "resume_supported": False,
        "raw_target_support_policy": "retain every draw, including out-of-prior-support draws"}
    write_json(directory / "checkpoint.json", record)
    return record


def _validate_declaration(declaration, contract, parent, out):
    if "contract" in declaration:
        if (declaration.get("format") != "heston_pr87_100k_study_v1"
                or Path(declaration["output"]).resolve() != Path(out).resolve()
                or declaration["task"] != parent["task"]):
            raise ValueError("100k study declaration task or output differs")
        if "task_file" in declaration and (
                digest(declaration["task_file"]["path"]) != declaration["task_file"]["sha256"]
                or read_task(declaration["task_file"]["path"]) != parent["task"]):
            raise ValueError("100k study task file differs")
    if contract["parent"] != parent["identity"]:
        raise ValueError("declared immutable parent differs")


def run(parent, protocol, out):
    """Run one complete new fit; an enclosing wall supervisor is expected."""
    started = time.monotonic()
    protocol_path = Path(protocol).resolve()
    declaration = json.loads(protocol_path.read_text())
    contract = declaration.get("contract", declaration)
    expected = protocol_contract(parent, smoke_test_only=contract.get("smoke_test_only"))
    if contract != expected:
        raise ValueError("persist the exact source-bound 100k contract before generation or fitting")
    parent = _read_parent(parent, smoke_test_only=contract["smoke_test_only"])
    _validate_declaration(declaration, contract, parent, out)
    out = Path(out).resolve()
    repo = Path(__file__).resolve().parents[2]
    if out == repo or (repo in out.parents and out.relative_to(repo).parts[0] != "local-artifacts"):
        raise ValueError("model/data payloads require an outside-Git or untracked local-artifacts destination")
    out.mkdir(parents=True, exist_ok=False)

    def check_budget():
        if time.monotonic() - started >= contract["max_wall_seconds"]:
            raise TimeoutError("predeclared generation+fit wall budget exhausted; no retries or extension")

    task = parent["task"]
    record = {"format": "heston_pr87_100k_run_v1", "status": "started", "task": task,
        "task_fingerprint": fingerprint(task), "parameter_names": task["target"]["parameter_names"],
        "bounds": task["target"]["prior_bounds"], "protocol_path": str(protocol_path),
        "protocol_sha256": digest(protocol_path), "protocol": contract, "parent": contract["parent"],
        "data_payload": "data.npz", "smoke_test_only": contract["smoke_test_only"],
        "primary_epoch": contract["epochs"], "checkpoints": []}
    write_json(out / "run.json", record)
    try:
        bf, keras = _runtime()
        keras.utils.set_random_seed(SEED)
        check_budget()
        generation_started = time.monotonic()

        def generation_progress(count):
            write_json(out / "generation-progress.json", {
                "format": "heston_pr87_100k_generation_progress_v1",
                "completed_fresh_pricing_calls": count, "total_fresh_pricing_calls": contract["appended_rows"],
                "generation_elapsed_seconds": time.monotonic() - generation_started,
                "whole_run_elapsed_seconds": time.monotonic() - started,
                "status": "complete" if count == contract["appended_rows"] else "running"})
            print(f"Fresh simulator labels: {count}/{contract['appended_rows']}", flush=True)
            check_budget()

        generation_progress(0)
        arrays, states = append_data(parent, contract["appended_rows"], check_budget=check_budget,
                                    progress=generation_progress)
        write_npz(out / "data.npz", arrays)
        record.update(data_payload_sha256=digest(out / "data.npz"),
            data_arrays={key: array_summary(value) for key, value in arrays.items()},
            numpy_generator_states=states, pricing_calls=contract["appended_rows"],
            retained_parent_rows=contract["parent_train_rows"],
            retained_validation_rows=contract["validation_rows"])
        whitening = parent["whitening"]
        shutil.copyfile(parent["directory"] / "whitening.json", out / "whitening.json")
        record["whitening_sha256"] = digest(out / "whitening.json")
        data = {split: {name: arrays[f"{split}_{name}"] for name in
                [*task["target"]["parameter_names"], "observations"]}
                for split in ("train", "validation")}
        for split in data:
            data[split]["observations"] = whitening.transform(data[split]["observations"])
        workflow = build_workflow(task)
        history_rows, best = [], {"loss": float("inf"), "state": None, "epoch": None, "logs": None}

        class ArchiveCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                actual = _stable_config(keras.optimizers.serialize(self.model.optimizer))
                if actual != contract["resolved_defaults"]["optimizer"]:
                    raise ValueError("native offline optimizer differs from predeclared resolution")
                record["resolved_optimizer"] = actual
                record["resolved_compile_config"] = _stable_config(self.model.get_compile_config())
                record["resolved_estimator_config"] = _stable_config(self.model.get_config())
                write_json(out / "run.json", record)
                check_budget()

            def on_train_batch_begin(self, batch, logs=None):
                check_budget()

            def on_train_batch_end(self, batch, logs=None):
                check_budget()

            def on_epoch_end(self, epoch, logs=None):
                check_budget()
                logs = {key: float(value) for key, value in (logs or {}).items()}
                if not all(np.isfinite(value) for value in logs.values()):
                    raise FloatingPointError("nonfinite native epoch history")
                epoch += 1
                history_rows.append({"epoch": epoch, **logs})
                write_json(out / "native-epoch-history.json", history_rows)
                if "val_loss" in logs and logs["val_loss"] < best["loss"]:
                    best.update(loss=logs["val_loss"], state=_state(self.model), epoch=epoch, logs=logs)
                if epoch in contract["checkpoint_epochs"]:
                    path = out / "checkpoints" / f"epoch-{epoch:06d}"
                    _save_checkpoint(path, workflow, record, out / "whitening.json", epoch, logs)
                    record["checkpoints"].append({"epoch": epoch, "path": str(path),
                        "checkpoint_sha256": digest(path / "checkpoint.json")})
                write_json(out / "run.json", record)
                check_budget()

        history = workflow.fit_offline(data=data["train"], validation_data=data["validation"],
            epochs=contract["epochs"], batch_size=contract["batch_size"], callbacks=[ArchiveCallback()])
        check_budget()
        write_json(out / "native-history.json", {key: [float(item) for item in values]
                                                  for key, values in history.history.items()})
        if len(history.epoch) != contract["epochs"]:
            raise RuntimeError("native fit did not complete the declared epoch count")
        if int(workflow.estimator.optimizer.iterations.numpy()) != contract["expected_optimizer_updates"]:
            raise RuntimeError("native optimizer updates differ from complete epochs including the tail batch")
        final_state = _state(workflow.estimator)
        if best["state"] is not None:
            try:
                _assign_state(workflow.estimator, best["state"])
                path = out / "checkpoints" / "best-validation"
                _save_checkpoint(path, workflow, record, out / "whitening.json", best["epoch"], best["logs"])
                record["best_validation_archive"] = {"epoch": best["epoch"], "val_loss": best["loss"],
                    "path": str(path), "checkpoint_sha256": digest(path / "checkpoint.json")}
            finally:
                _assign_state(workflow.estimator, final_state)
        record.update(status="complete", completed_epochs=len(history.epoch),
            actual_optimizer_updates=int(workflow.estimator.optimizer.iterations.numpy()),
            primary_checkpoint=str(out / "checkpoints" / f"epoch-{contract['epochs']:06d}"),
            native_history_sha256=digest(out / "native-history.json"),
            native_epoch_history_sha256=digest(out / "native-epoch-history.json"),
            elapsed_seconds=time.monotonic() - started)
        check_budget()
        write_json(out / "run.json", record)
        write_json(out / "training.json", record)
        return record
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - started)
        write_json(out / "run.json", record)
        write_json(out / "training.json", record)
        raise


def load_checkpoint(directory):
    """Check source/parent/data/whitening/full state before exposing raw sampling."""
    directory = Path(directory).resolve()
    record = json.loads((directory / "checkpoint.json").read_text())
    if record.get("format") != "heston_pr87_100k_checkpoint_v1":
        raise ValueError("new 100k checkpoint identity required")
    task = read_task(record["task"])
    contract = record["protocol"]
    protocol_path = Path(record["protocol_path"])
    declaration = json.loads(protocol_path.read_text())
    training = directory.parent.parent
    run_record = json.loads((training / "run.json").read_text())
    parent = _read_parent(contract["parent"]["directory"], smoke_test_only=contract.get("smoke_test_only"))
    _validate_declaration(declaration, contract, parent, training)
    expected = protocol_contract(parent["directory"], smoke_test_only=contract["smoke_test_only"])
    data_path = training / record["data_payload"]
    checkpoint_entries = run_record["checkpoints"]
    if directory.name == "best-validation":
        checkpoint_entries = [run_record["best_validation_archive"]]
    matched = [item for item in checkpoint_entries if Path(item["path"]).resolve() == directory]
    if (run_record.get("format") != "heston_pr87_100k_run_v1" or run_record.get("status") != "complete"
            or contract != expected or declaration.get("contract", declaration) != contract
            or record["task_fingerprint"] != fingerprint(task) or task != parent["task"]
            or record["parameter_names"] != task["target"]["parameter_names"]
            or record["bounds"] != task["target"]["prior_bounds"]
            or digest(protocol_path) != record["protocol_sha256"]
            or run_record["protocol_sha256"] != record["protocol_sha256"]
            or run_record["protocol"] != contract or run_record["task"] != task
            or record["parent"] != contract["parent"] or run_record["parent"] != contract["parent"]
            or record["smoke_test_only"] is not contract["smoke_test_only"]
            or record["primary_epoch"] != contract["epochs"]
            or record["native_epoch"] not in range(1, contract["epochs"] + 1)
            or record["actual_optimizer_updates"] != record["native_epoch"] * contract["batches_per_epoch"]
            or run_record["completed_epochs"] != contract["epochs"]
            or run_record["actual_optimizer_updates"] != contract["expected_optimizer_updates"]
            or digest(training / "run.json") != digest(training / "training.json")
            or record["data_payload"] != "data.npz" or data_path.resolve().parent != training
            or digest(data_path) != record["data_payload_sha256"]
            or run_record["data_payload_sha256"] != record["data_payload_sha256"]
            or run_record["data_arrays"] != record["data_arrays"]
            or digest(training / "whitening.json") != contract["parent"]["whitening"]["sha256"]
            or record["whitening_sha256"] != contract["parent"]["whitening"]["sha256"]
            or len(matched) != 1 or matched[0]["epoch"] != record["native_epoch"]
            or digest(directory / "checkpoint.json") != matched[0]["checkpoint_sha256"]
            or any(digest(directory / name) != record[key] for name, key in
                (("model.keras", "model_sha256"), ("state.npz", "state_sha256"), ("whitening.json", "whitening_sha256")))):
        raise ValueError("100k checkpoint source, parent, task, data, whitening or payload identity differs")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    _validate_shapes(arrays, task, contract["train_rows"], contract["validation_rows"])
    _validate_parent_prefix(arrays, parent)
    if {key: array_summary(value) for key, value in arrays.items()} != record["data_arrays"]:
        raise ValueError("100k data arrays differ from all bound hashes")
    whitening = NotebookWhitening.from_dict(json.loads((directory / "whitening.json").read_text()))
    if whitening.to_dict() != parent["whitening"].to_dict():
        raise ValueError("100k checkpoint whitening differs from immutable original-only fit")
    workflow = build_workflow(task)
    workflow.load_estimator(directory / "model.keras")
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        keys = {f"{group}_{index:04d}" for group in GROUPS for index in range(len(record["state"][group]))}
        if set(archive.files) != keys:
            raise ValueError("checkpoint full-state array inventory differs")
        state = tuple([archive[f"{group}_{index:04d}"].copy() for index in range(len(record["state"][group]))]
                      for group in GROUPS)
    _assign_state(workflow.estimator, state)
    actual = _state_summary(workflow.estimator, _state(workflow.estimator))
    if ({key: [{"shape": item["shape"], "dtype": item["dtype"], "sha256": item["sha256"]}
               for item in values] for key, values in actual.items()}
            != {key: [{"shape": item["shape"], "dtype": item["dtype"], "sha256": item["sha256"]}
                     for item in values] for key, values in record["state"].items()}
            or int(workflow.estimator.optimizer.iterations.numpy()) != record["actual_optimizer_updates"]
            or _stable_config(__import__("keras").optimizers.serialize(workflow.estimator.optimizer))
                != contract["resolved_defaults"]["optimizer"]):
        raise ValueError("100k checkpoint full state or optimizer differs after reload")
    return RawQuoteWorkflow(workflow, task, whitening), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker", action="store_true", help="direct worker under controller wall supervisor")
    args = parser.parse_args()
    run(args.parent, args.protocol, args.out)


if __name__ == "__main__":
    main()

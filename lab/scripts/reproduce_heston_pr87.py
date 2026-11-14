"""One bounded statistical-recipe reproduction of collaborator PR87.

The unseeded notebook gains NumPy/Keras seeds and archival callbacks only.
Its raw physical targets, fresh simulator, whitening and native offline fit
are otherwise literal. Payloads remain untracked under new run identities.
"""

import argparse
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration.heston_spread import HestonSpreadPilot
from sbi4f_tutorial.heston_options import model as financial_model


SEED = 9011
MAX_WALL_SECONDS = 1800
NOTEBOOK_COMMIT = "ddf3619faf48b8f4b0ee56ee841ef24124a508d1"
NOTEBOOK_SHA256 = "22ddfe2d8b97b80b97ac6d0d92f40c04022c9494bfb6fc1fc829574014372641"
GROUPS = ("trainable", "non_trainable", "optimizer", "metrics")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, payload):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def array_summary(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def read_task(value):
    value = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    task = value.get("task", value)
    # Exactly the notebook constructor; validation binds all financial controls.
    model = HestonSpreadPilot(task["records"], infer_c=True, c_bounds=(.5, 2.))
    model.validate_task(task["target"])
    if (task.get("format") != "heston_spread_scale_task_v1"
            or len(model.records["mid"]) != 54 or len(model.names) != 6):
        raise ValueError("the preserved six-target, 54-quote PR87 task is required")
    return task


def _runtime():
    import bayesflow as bf
    import keras
    if bf.__version__ != "2.1.0" or keras.backend.backend() != "jax":
        raise RuntimeError("locked BayesFlow 2.1.0 JAX core-reference runtime required")
    return bf, keras


def build_workflow(task):
    """Literal cells 6-7: intentionally no bound transform or optimizer override."""
    task = read_task(task)
    bf, _ = _runtime()
    adapter = (bf.Adapter()
               .concatenate(task["target"]["parameter_names"], into="inference_variables")
               .rename("observations", "inference_conditions"))
    return bf.BasicWorkflow(adapter=adapter,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (128,) * 3, "dropout": .1}),
        standardize="inference_variables")


def _stable_config(value):
    if isinstance(value, dict):
        return {key: _stable_config(item) for key, item in value.items()
                if key != "name" or set(value) == {"name"}}
    if isinstance(value, (list, tuple)):
        return [_stable_config(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def source_bindings():
    import sbi4finance.calibration.heston_pilot as pilot
    import sbi4finance.calibration.heston_spread as spread
    bf, _ = _runtime()
    paths = {"producer": Path(__file__), "spread_pilot": Path(spread.__file__),
             "checked_pilot": Path(pilot.__file__), "financial_model": Path(financial_model.__file__),
             "bf_workflow": Path(inspect.getfile(bf.BasicWorkflow)),
             "bf_offline_dataset": Path(inspect.getfile(bf.datasets.OfflineDataset)),
             "bf_flow_matching": Path(inspect.getfile(bf.networks.FlowMatching)),
             "bf_make_simulator": Path(inspect.getfile(inspect.unwrap(bf.make_simulator))),
             "bf_sequential_simulator": Path(inspect.getfile(bf.simulators.SequentialSimulator)),
             "bf_lambda_simulator": Path(inspect.getfile(bf.simulators.LambdaSimulator)),
             "lab_lock": Path(__file__).resolve().parents[1] / "pixi.lock"}
    return {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in paths.items()}


def protocol_contract(task, notebook, *, smoke_test_only=False):
    """Resolve, bind and return the pre-fit contract; this never prices or fits."""
    if type(smoke_test_only) is not bool:
        raise ValueError("explicit boolean smoke identity required")
    task = read_task(task)
    notebook = Path(notebook)
    if digest(notebook) != NOTEBOOK_SHA256:
        raise ValueError("notebook bytes differ from the pinned collaborator PR87 commit")
    bf, keras = _runtime()
    keras.utils.set_random_seed(SEED)
    rows, validation_rows, epochs = (128, 32, 2) if smoke_test_only else (10000, 100, 200)
    workflow = build_workflow(task)
    optimizer = workflow.build_optimizer(epochs, (rows + 255) // 256, strategy="offline")
    resolved = {"optimizer": _stable_config(keras.optimizers.serialize(optimizer)),
                "network": _stable_config(keras.saving.serialize_keras_object(workflow.inference_network)),
                "fit_offline_signature": str(inspect.signature(bf.BasicWorkflow.fit_offline)),
                "keras_fit_signature": str(inspect.signature(keras.Model.fit))}
    return {"format": "heston_pr87_reproduction_protocol_v1", "predeclared_before_fit": True,
        "notebook": {"path": str(notebook.resolve()), "sha256": NOTEBOOK_SHA256, "commit": NOTEBOOK_COMMIT},
        "task_fingerprint": fingerprint(task), "source": source_bindings(),
        "train_rows": rows, "validation_rows": validation_rows, "epochs": epochs, "batch_size": 256,
        "checkpoint_epochs": [1, 2] if smoke_test_only else [50, 75, 83, 100, 200],
        "primary_checkpoint": "final_epoch_only", "archival_selection": "first strictly smallest finite native val_loss; never restore during fit",
        "seed": SEED, "seed_addition": "original np.default_rng() and Keras initialization were unseeded; both receive 9011",
        "simulation": "bf.make_simulator([prior, observation_model]); one shared np.default_rng(9011); sample(train_rows) then sample(validation_rows)",
        "training": "one native fit_offline(data,validation_data,epochs,batch_size); all other training, shuffle, loss and optimizer defaults retained",
        "whitening": "mean(axis=0); np.cov(rowvar=False); eigh; floor=eigenvalues[-1]*1e-8; W=(eigenvectors/sqrt(max(eigenvalues,floor)))@eigenvectors.T",
        "target_transform": "raw physical coordinates; concatenate then rename only; no constrain, clipping, rejection or target Jacobian",
        "standardize": "inference_variables", "resolved_defaults": resolved,
        "versions": {name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "jaxlib", "numpy")},
        "python": platform.python_version(), "platform": platform.platform(), "backend": keras.backend.backend(),
        "devices": [str(device) for device in __import__("jax").devices()],
        "reproduction_scope": "statistical recipe reproduction; original platform and complete random state are unknown; no bitwise original-weights claim",
        "max_wall_seconds": MAX_WALL_SECONDS, "smoke_test_only": smoke_test_only,
        "automatic_extension": False, "retries": 0, "new_pricing_labels": True,
        "historical_payloads_opened": False, "scientific_acceptance": False}


class NotebookWhitening:
    """Native notebook float64 arithmetic and train-only frozen state."""
    def __init__(self, observations):
        values = np.asarray(observations)
        if values.ndim != 2 or values.shape[1] != 54 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("finite training matrix with54 features required")
        self.mean = values.mean(axis=0)
        self.covariance = np.cov(values, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)
        self.floor = self.eigenvalues[-1] * 1e-8
        if self.floor <= 0:
            raise ValueError("positive covariance scale required")
        safe = np.maximum(self.eigenvalues, self.floor)
        self.matrix = (self.eigenvectors / np.sqrt(safe)) @ self.eigenvectors.T
        self.training_rows = len(values)

    def transform(self, values):
        values = np.atleast_2d(values)
        if np.iscomplexobj(values) or values.shape[-1] != 54 or not np.isfinite(values).all():
            raise ValueError("finite real rows with54 features required")
        return (values - self.mean) @ self.matrix

    def to_dict(self):
        return {"format": "heston_pr87_notebook_whitening_v1", "training_rows": self.training_rows,
            "mean": self.mean.tolist(), "covariance": self.covariance.tolist(),
            "eigenvalues": self.eigenvalues.tolist(), "eigenvectors": self.eigenvectors.tolist(),
            "eigenvalue_floor": float(self.floor), "whitening_matrix": self.matrix.tolist(),
            "validation_used_for_fit": False, "relative_eigenvalue_floor": 1e-8}

    @classmethod
    def from_dict(cls, record):
        if (record.get("format") != "heston_pr87_notebook_whitening_v1"
                or record.get("validation_used_for_fit") is not False
                or record.get("relative_eigenvalue_floor") != 1e-8):
            raise ValueError("invalid native whitening identity")
        self = cls.__new__(cls)
        self.mean, self.covariance, self.eigenvalues, self.eigenvectors, self.matrix = (
            np.asarray(record[key], dtype=float) for key in
            ("mean", "covariance", "eigenvalues", "eigenvectors", "whitening_matrix"))
        self.floor, self.training_rows = record["eigenvalue_floor"], record["training_rows"]
        if (self.mean.shape != (54,) or self.covariance.shape != (54, 54)
                or self.eigenvalues.shape != (54,) or self.eigenvectors.shape != (54, 54)
                or self.matrix.shape != (54, 54) or self.training_rows < 2
                or not all(np.isfinite(value).all() for value in
                           (self.mean, self.covariance, self.eigenvalues, self.eigenvectors, self.matrix))
                or self.floor != self.eigenvalues[-1] * 1e-8 or self.floor <= 0
                or not np.allclose(self.eigenvectors @ np.diag(self.eigenvalues) @ self.eigenvectors.T,
                                   self.covariance, rtol=1e-10, atol=1e-7)
                or not np.array_equal(self.matrix, (self.eigenvectors / np.sqrt(
                    np.maximum(self.eigenvalues, self.floor))) @ self.eigenvectors.T)):
            raise ValueError("saved native whitening state differs")
        return self


def generate_data(task, train_rows, validation_rows, *, check_budget=lambda: None):
    """Use the notebook simulator with its native sequential sampling order."""
    bf, _ = _runtime()
    task = read_task(task)
    model = HestonSpreadPilot(task["records"], infer_c=True, c_bounds=(.5, 2.))
    model.validate_task(task["target"])
    parameter_names, bounds = model.parameter_names, np.asarray(model.bounds)
    half_spread = np.asarray(model.records["half_spread"])
    rng = np.random.default_rng(SEED)
    clean_rows, quote_rows = [], []

    def prior():
        check_budget()
        values = rng.uniform(bounds[:, 0], bounds[:, 1])
        return dict(zip(parameter_names, values, strict=True))

    def observation_model(initial_variance, kappa, theta, vol_of_vol, rho, c):
        check_budget()
        parameters = np.array([initial_variance, kappa, theta, vol_of_vol, rho, c])
        clean_prices = model.prices(parameters)
        quotes = clean_prices + c * half_spread * rng.standard_normal(half_spread.size)
        clean_rows.append(clean_prices.copy())
        quote_rows.append(quotes.copy())
        return {"observations": quotes / half_spread}

    simulator = bf.make_simulator([prior, observation_model])
    data, extra, states = {}, {}, {"before_train": rng.bit_generator.state}
    for split, count in (("train", train_rows), ("validation", validation_rows)):
        clean_rows.clear()
        quote_rows.clear()
        data[split] = simulator.sample(count)
        extra[split + "_clean_prices"] = np.asarray(clean_rows)
        extra[split + "_quotes"] = np.asarray(quote_rows)
        states["after_" + split] = rng.bit_generator.state
        if (data[split]["observations"].shape != (count, 54)
                or any(data[split][name].shape != (count, 1) for name in parameter_names)
                or not all(np.isfinite(value).all() for value in data[split].values())):
            raise FloatingPointError("native simulator returned invalid data")
    return data, extra, states


def _variable_groups(model):
    return (model.trainable_variables, model.non_trainable_variables,
            model.optimizer.variables, model.metrics_variables)


def _state(model):
    return tuple([np.asarray(variable).copy() for variable in group] for group in _variable_groups(model))


def _state_summary(model, state):
    return {group: [{"path": variable.path, **array_summary(value)}
                   for variable, value in zip(variables, values, strict=True)]
            for group, variables, values in zip(GROUPS, _variable_groups(model), state, strict=True)}


def _assign_state(model, state):
    for variables, values in zip(_variable_groups(model), state, strict=True):
        for variable, value in zip(variables, values, strict=True):
            if tuple(variable.shape) != value.shape:
                raise ValueError("full-state shape differs")
            variable.assign(value)


def _save_checkpoint(directory, workflow, run_record, whitening, epoch, logs):
    directory.mkdir(parents=True, exist_ok=False)
    model = workflow.estimator
    state = _state(model)
    summary = _state_summary(model, state)
    model.save(directory / "model.keras")
    if summary != _state_summary(model, _state(model)):
        raise RuntimeError("native archive serialization changed training state")
    write_npz(directory / "state.npz", {f"{group}_{index:04d}": value
        for group, values in zip(GROUPS, state, strict=True) for index, value in enumerate(values)})
    write_json(directory / "whitening.json", whitening.to_dict())
    record = {**run_record, "format": "heston_pr87_reproduction_checkpoint_v1", "native_epoch": epoch,
        "native_logs": logs, "actual_optimizer_updates": int(model.optimizer.iterations.numpy()),
        "model_sha256": digest(directory / "model.keras"), "state_sha256": digest(directory / "state.npz"),
        "whitening_sha256": digest(directory / "whitening.json"), "state": summary,
        "resolved_optimizer": _stable_config(__import__("keras").optimizers.serialize(model.optimizer)),
        "resume_supported": False, "raw_target_support_policy": "retain every draw, including out-of-prior-support draws"}
    write_json(directory / "checkpoint.json", record)
    return record


def run(task, protocol, out):
    """Run exactly the persisted contract, with an enclosing wall supervisor expected."""
    started = time.monotonic()
    protocol_path = Path(protocol)
    declaration = json.loads(protocol_path.read_text())
    contract = declaration.get("contract", declaration)
    if "contract" in declaration:
        if (declaration.get("format") != "heston_pr87_reproduction_study_v1"
                or Path(declaration["output"]).resolve() != Path(out).resolve()
                or read_task(task) != declaration["task"]
                or digest(declaration["task_file"]["path"]) != declaration["task_file"]["sha256"]):
            raise ValueError("study declaration task or output differs")
    expected = protocol_contract(task, contract["notebook"]["path"],
                                 smoke_test_only=contract.get("smoke_test_only"))
    if contract != expected:
        raise ValueError("persist the exact source-bound PR87 contract before fitting")
    task = read_task(task)
    out = Path(out).resolve()
    # Full banks and models use an untracked payload destination.
    repo = Path(__file__).resolve().parents[2]
    if out == repo or (repo in out.parents and out.relative_to(repo).parts[0] != "local-artifacts"):
        raise ValueError("model/data payloads require an outside-Git or untracked local-artifacts destination")
    out.mkdir(parents=True, exist_ok=False)

    def check_budget():
        if time.monotonic() - started >= contract["max_wall_seconds"]:
            raise TimeoutError("predeclared whole-run wall budget exhausted; no retries or extension")

    record = {"format": "heston_pr87_reproduction_run_v1", "status": "started", "task": task,
        "task_fingerprint": fingerprint(task), "parameter_names": task["target"]["parameter_names"],
        "bounds": task["target"]["prior_bounds"], "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": digest(protocol_path), "protocol": contract, "data_payload": "data.npz",
        "smoke_test_only": contract["smoke_test_only"], "primary_epoch": contract["epochs"], "checkpoints": []}
    write_json(out / "run.json", record)
    try:
        bf, keras = _runtime()
        keras.utils.set_random_seed(SEED)
        data, extra, states = generate_data(task, contract["train_rows"], contract["validation_rows"], check_budget=check_budget)
        payload = {f"{split}_{key}": value for split, values in data.items() for key, value in values.items()} | extra
        write_npz(out / "data.npz", payload)
        record.update(data_payload_sha256=digest(out / "data.npz"),
            data_arrays={key: array_summary(value) for key, value in payload.items()},
            numpy_generator_states=states, pricing_calls=contract["train_rows"] + contract["validation_rows"])
        whitening = NotebookWhitening(data["train"]["observations"])
        write_json(out / "whitening.json", whitening.to_dict())
        for split in data:
            data[split]["observations"] = whitening.transform(data[split]["observations"])
        workflow = build_workflow(task)
        owner = workflow
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
                    _save_checkpoint(path, owner, record, whitening, epoch, logs)
                    record["checkpoints"].append({"epoch": epoch, "path": str(path),
                                                  "checkpoint_sha256": digest(path / "checkpoint.json")})
                write_json(out / "run.json", record)
                check_budget()

        # This is the notebook call with passive archival/budget callbacks only.
        history = workflow.fit_offline(data=data["train"], validation_data=data["validation"],
            epochs=contract["epochs"], batch_size=contract["batch_size"], callbacks=[ArchiveCallback()])
        check_budget()
        write_json(out / "native-history.json", {key: [float(item) for item in values]
                                                  for key, values in history.history.items()})
        if len(history.epoch) != contract["epochs"]:
            raise RuntimeError("native fit did not complete the declared epoch count")
        expected_updates = contract["epochs"] * ((contract["train_rows"] + contract["batch_size"] - 1) // contract["batch_size"])
        if int(workflow.estimator.optimizer.iterations.numpy()) != expected_updates:
            raise RuntimeError("native fit optimizer updates differ from complete epochs including the tail batch")
        final_state = _state(workflow.estimator)
        if best["state"] is not None:
            try:
                _assign_state(workflow.estimator, best["state"])
                path = out / "checkpoints" / "best-validation"
                _save_checkpoint(path, workflow, record, whitening, best["epoch"], best["logs"])
                record["best_validation_archive"] = {"epoch": best["epoch"], "val_loss": best["loss"], "path": str(path)}
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


class RawQuoteWorkflow:
    """Inference boundary consumes actual quote units; internal workflow is native."""
    def __init__(self, workflow, task, whitening):
        self.workflow, self.task, self.whitening = workflow, task, whitening
        self.estimator = workflow.estimator

    def sample(self, raw_quotes, count, seed=0):
        if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
            raise ValueError("positive count and nonnegative integer sampling seed required")
        if np.iscomplexobj(raw_quotes):
            raise ValueError("real raw quote rows required")
        raw_quotes = np.atleast_2d(np.asarray(raw_quotes, dtype=float))
        if raw_quotes.ndim != 2 or raw_quotes.shape[1] != 54 or not len(raw_quotes) or not np.isfinite(raw_quotes).all():
            raise ValueError("complete finite raw quote rows required; negative quotes are permitted")
        conditions = self.whitening.transform(raw_quotes / np.asarray(self.task["records"]["half_spread"]))
        values = self.workflow.sample(conditions={"observations": conditions}, num_samples=count,
                                     sample_shape=(), seed=seed)
        result = np.concatenate([np.asarray(values[name]) for name in self.task["target"]["parameter_names"]], axis=-1)
        if result.shape != (len(raw_quotes), count, 6):
            raise FloatingPointError("invalid native physical-coordinate posterior samples")
        return result


def load_checkpoint(directory):
    directory = Path(directory)
    record = json.loads((directory / "checkpoint.json").read_text())
    task = read_task(record["task"])
    protocol_path = Path(record["protocol_path"])
    declaration = json.loads(protocol_path.read_text())
    contract = declaration.get("contract", declaration)
    training = directory.parent.parent
    run_record = json.loads((training / "run.json").read_text())
    data_path = training / run_record["data_payload"]
    if (record.get("format") != "heston_pr87_reproduction_checkpoint_v1"
            or fingerprint(task) != record["task_fingerprint"]
            or record["parameter_names"] != task["target"]["parameter_names"]
            or record["bounds"] != task["target"]["prior_bounds"]
            or digest(protocol_path) != record["protocol_sha256"]
            or contract != record["protocol"]
            or source_bindings() != contract["source"]
            or digest(contract["notebook"]["path"]) != contract["notebook"]["sha256"]
            or ("contract" in declaration and (declaration["task"] != task
                or digest(declaration["task_file"]["path"]) != declaration["task_file"]["sha256"]))
            or run_record["task"] != task or run_record["protocol"] != contract
            or run_record["protocol_sha256"] != record["protocol_sha256"]
            or data_path.resolve().parent != training.resolve()
            or digest(data_path) != record["data_payload_sha256"]
            or run_record["data_payload_sha256"] != record["data_payload_sha256"]
            or digest(training / "whitening.json") != record["whitening_sha256"]
            or any(digest(directory / name) != record[key] for name, key in
                (("model.keras", "model_sha256"), ("state.npz", "state_sha256"), ("whitening.json", "whitening_sha256")))):
        raise ValueError("checkpoint task or payload identity differs")
    whitening = NotebookWhitening.from_dict(json.loads((directory / "whitening.json").read_text()))
    with np.load(data_path, allow_pickle=False) as archive:
        observations = archive["train_observations"]
        if (array_summary(observations) != record["data_arrays"]["train_observations"]
                or whitening.to_dict() != NotebookWhitening(observations).to_dict()):
            raise ValueError("whitening is not the exact native fit to the bound generated training rows")
    workflow = build_workflow(task)
    workflow.load_estimator(directory / "model.keras")
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        state = tuple([archive[f"{group}_{index:04d}"].copy() for index in range(len(record["state"][group]))]
                      for group in GROUPS)
    # Native .keras may omit auxiliary RNG/metric state. Restore the bound sidecar.
    _assign_state(workflow.estimator, state)
    actual = _state_summary(workflow.estimator, _state(workflow.estimator))
    if ({key: [item["sha256"] for item in value] for key, value in actual.items()}
            != {key: [item["sha256"] for item in value] for key, value in record["state"].items()}
            or int(workflow.estimator.optimizer.iterations.numpy()) != record["actual_optimizer_updates"]):
        raise ValueError("checkpoint full state differs after reload")
    return RawQuoteWorkflow(workflow, task, whitening), record


def sample(workflow, task, raw_quotes, count, seed=0):
    if fingerprint(read_task(task)) != fingerprint(workflow.task):
        raise ValueError("sampling task differs from loaded checkpoint")
    return workflow.sample(raw_quotes, count, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker", action="store_true", help="direct worker under the controller wall supervisor")
    args = parser.parse_args()
    run(args.task, args.protocol, args.out)


if __name__ == "__main__":
    main()

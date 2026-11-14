"""Source-bound revised 63-quote Heston bank and paired native FM fits.

Reprice every new-grid row. Keep the old200k physical/c prefix, native100 and
historical512 truths paired, fit ZCA only on the first10k new-grid train rows,
and freeze one300k fixed noisy bank for both fresh300-epoch fits.
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
from scripts import reproduce_heston_pr87 as native
from scripts import reproduce_heston_pr87_followup as prior
from sbi4finance.calibration.heston_grid import HestonGridPilot

SEED, GROUPS = native.SEED, native.GROUPS
MAX_BANK_SECONDS = MAX_TRAIN_SECONDS = 3600
PARENT_PRODUCER_SHA256 = "151c03e9e337f1fd4b97ee6d5dd2cffe4ac9f79372fd666275015ddd69fc7561"
PARENT_DATA_SHA256 = "563c628c998e7e8f322c9e584eb6b69047ece87e099a3de5b0286835da1310b6"
digest, fingerprint, write_json, write_npz = native.digest, native.fingerprint, native.write_json, native.write_npz
array_summary, HestonSpreadPilot = native.array_summary, HestonGridPilot
_runtime, _stable_config = native._runtime, native._stable_config
_state, _state_summary, _assign_state = native._state, native._state_summary, native._assign_state
GEOMETRY_KEYS = ("forward", "strike", "discount", "maturity", "option_type", "half_spread")


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def read_task(value):
    value = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    task = value.get("task", value)
    model = HestonSpreadPilot(task["records"], infer_c=True, c_bounds=(.5, 2.))
    model.validate_task(task["target"])
    if task.get("format") != "heston_pr87_grid_task_v1" or len(model.records["mid"]) != 63 or len(model.names) != 6:
        raise ValueError("canonical revised63-quote six-target grid task required")
    return task


def build_workflow(task):
    task = read_task(task)
    bf, _ = _runtime()
    adapter = (bf.Adapter().concatenate(task["target"]["parameter_names"], into="inference_variables")
               .rename("observations", "inference_conditions"))
    return bf.BasicWorkflow(adapter=adapter,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (128,) * 3, "dropout": .1}),
        standardize="inference_variables")


def source_bindings():
    sources = prior.source_bindings()
    sources["parent_followup_producer"] = sources.pop("producer")
    sources["producer"] = identity(__file__)
    sources["grid_pilot"] = identity(inspect.getfile(HestonSpreadPilot))
    from sbi4f_tutorial.heston_options import control_variate
    sources["black_control_variate"] = identity(control_variate.__file__)
    return sources


def _read_parent(directory, *, smoke_test_only=False):
    if type(smoke_test_only) is not bool:
        raise ValueError("explicit boolean smoke identity required")
    directory = Path(directory).resolve()
    record = json.loads((directory / "run.json").read_text())
    contract = record["protocol"]
    rows, epochs = (512, 2) if smoke_test_only else (200000, 300)
    ancestor = prior._read_parent(contract["parent"]["directory"], smoke_test_only=smoke_test_only)
    expected = prior.protocol_contract(ancestor["directory"], train_rows=rows, smoke_test_only=smoke_test_only)
    declaration = json.loads(Path(record["protocol_path"]).read_text())
    prior._validate_declaration(declaration, contract, ancestor, directory)
    final = directory / "checkpoints" / f"epoch-{epochs:06d}"
    checkpoint = json.loads((final / "checkpoint.json").read_text())
    entries = [q for q in record["checkpoints"] if q["epoch"] == epochs]
    if (record.get("format") != "heston_pr87_followup_run_v1" or record.get("status") != "complete"
            or record.get("smoke_test_only") is not smoke_test_only or contract != expected
            or declaration.get("contract", declaration) != contract
            or digest(prior.__file__) != PARENT_PRODUCER_SHA256
            or record["completed_epochs"] != epochs or record["actual_optimizer_updates"] != epochs * ((rows + 255) // 256)
            or digest(directory / "run.json") != digest(directory / "training.json")
            or digest(record["protocol_path"]) != record["protocol_sha256"]
            or record["data_payload"] != "data.npz" or digest(directory / "data.npz") != record["data_payload_sha256"]
            or (not smoke_test_only and record["data_payload_sha256"] != PARENT_DATA_SHA256)
            or len(entries) != 1 or digest(final / "checkpoint.json") != entries[0]["checkpoint_sha256"]
            or Path(entries[0]["path"]).resolve() != final
            or checkpoint["protocol"] != contract or checkpoint["task"] != record["task"]
            or checkpoint["data_payload_sha256"] != record["data_payload_sha256"]
            or digest(directory / "whitening.json") != checkpoint["whitening_sha256"]):
        raise ValueError("immutable old54-grid completed200k parent identity differs")
    with np.load(directory / "data.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    prior._validate_shapes(arrays, record["task"], rows, contract["validation_rows"])
    prior._validate_parent_prefix(arrays, ancestor)
    if {key: array_summary(value) for key, value in arrays.items()} != record["data_arrays"]:
        raise ValueError("old54 parent arrays differ")
    if json.loads((directory / "whitening.json").read_text()) != ancestor["whitening"].to_dict():
        raise ValueError("old54 parent whitening differs")
    binding = {"directory": str(directory), "run": identity(directory / "run.json"),
        "data": identity(directory / "data.npz"), "protocol": identity(record["protocol_path"]),
        "checkpoint": identity(final / "checkpoint.json"), "train_rows": rows,
        "validation_rows": contract["validation_rows"],
        "continuation_rng_state": deepcopy(record["numpy_generator_states"]["after_append"])}
    return {"task": record["task"], "arrays": arrays, "identity": binding}


def _read_historical(path, old_task):
    path = Path(path).resolve()
    if path.suffix == ".npz":
        path = path.with_suffix(".json")
    record = json.loads(path.read_text())
    payload = path.parent / record["payload"]
    if (record.get("format") != "heston_spread_scale_recovery_inputs_v1" or record["task"] != old_task
            or payload.resolve().parent != path.parent or digest(payload) != record["payload_sha256"]):
        raise ValueError("paired historical512 physical truths differ")
    with np.load(payload, allow_pickle=False) as archive:
        truth, ids = archive["truth"].copy(), archive["row_id"].copy()
    if truth.shape != (512, 6) or ids.shape != (512,) or not np.isfinite(truth).all():
        raise ValueError("512 finite historical physical/c truths required")
    return {"truth": truth, "row_id": ids,
        "identity": {"manifest": identity(path), "payload": identity(payload),
                     "truth": array_summary(truth), "row_id": array_summary(ids)}}


def shared_contracts(old_task, task):
    matches = []
    old, new = old_task["records"], task["records"]
    for i in range(len(old["mid"])):
        for j in range(len(new["mid"])):
            if all(old[key][i] == new[key][j] for key in GEOMETRY_KEYS):
                matches.append({"old_index": i, "new_index": j,
                    "geometry": {key: new[key][j] for key in GEOMETRY_KEYS}})
    if len(matches) != 1:
        raise ValueError("exactly one shared contract with identical geometry/carry/spread is required")
    return matches


class NotebookWhitening:
    """The native notebook float64 ZCA equation, with explicit63-feature state."""
    def __init__(self, observations):
        x = np.asarray(observations)
        if x.ndim != 2 or x.shape[1] != 63 or len(x) < 64 or not np.isfinite(x).all():
            raise ValueError("finite native training matrix with63 features required")
        self.mean, self.covariance = x.mean(axis=0), np.cov(x, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)
        self.floor = self.eigenvalues[-1] * 1e-8
        if self.floor <= 0:
            raise ValueError("positive covariance scale required")
        self.matrix = (self.eigenvectors / np.sqrt(np.maximum(self.eigenvalues, self.floor))) @ self.eigenvectors.T
        self.training_rows = len(x)

    def transform(self, values):
        values = np.atleast_2d(values)
        if np.iscomplexobj(values) or values.ndim != 2 or values.shape[1] != 63 or not np.isfinite(values).all():
            raise ValueError("finite real63-feature rows required")
        return (values - self.mean) @ self.matrix

    def to_dict(self):
        return {"format": "heston_pr87_grid_whitening_v1", "quote_count": 63, "training_rows": self.training_rows,
            "mean": self.mean.tolist(), "covariance": self.covariance.tolist(),
            "eigenvalues": self.eigenvalues.tolist(), "eigenvectors": self.eigenvectors.tolist(),
            "eigenvalue_floor": float(self.floor), "whitening_matrix": self.matrix.tolist(),
            "validation_used_for_fit": False, "relative_eigenvalue_floor": 1e-8}

    @classmethod
    def from_dict(cls, record):
        if (record.get("format") != "heston_pr87_grid_whitening_v1" or record.get("quote_count") != 63
                or record.get("validation_used_for_fit") is not False or record.get("relative_eigenvalue_floor") != 1e-8):
            raise ValueError("invalid revised-grid whitening identity")
        self = cls.__new__(cls)
        self.mean, self.covariance, self.eigenvalues, self.eigenvectors, self.matrix = (
            np.asarray(record[key], dtype=float) for key in
            ("mean", "covariance", "eigenvalues", "eigenvectors", "whitening_matrix"))
        self.floor, self.training_rows = record["eigenvalue_floor"], record["training_rows"]
        if (self.mean.shape != (63,) or self.covariance.shape != (63, 63) or self.eigenvalues.shape != (63,)
                or self.eigenvectors.shape != (63, 63) or self.matrix.shape != (63, 63) or self.training_rows < 64
                or not all(np.isfinite(a).all() for a in (self.mean, self.covariance, self.eigenvalues, self.eigenvectors, self.matrix))
                or self.floor != self.eigenvalues[-1] * 1e-8 or self.floor <= 0
                or not np.allclose(self.eigenvectors @ np.diag(self.eigenvalues) @ self.eigenvectors.T,
                                   self.covariance, rtol=1e-10, atol=1e-7)
                or not np.array_equal(self.matrix, (self.eigenvectors / np.sqrt(np.maximum(self.eigenvalues, self.floor))) @ self.eigenvectors.T)):
            raise ValueError("saved revised-grid whitening state differs")
        return self


def bank_contract(task, parent_training, historical_inputs, *, smoke_test_only=False):
    task = read_task(task)
    parent = _read_parent(parent_training, smoke_test_only=smoke_test_only)
    historical = _read_historical(historical_inputs, parent["task"])
    old_target, target = deepcopy(parent["task"]["target"]), deepcopy(task["target"])
    if target["pricing_controls"].pop("fourier_representation", None) != "black_control_variate_v1":
        raise ValueError("versioned Black-control-variate numerical representation required")
    if {k: v for k, v in target.items() if k not in ("quote_count", "geometry_and_spreads_sha256")} != {
            k: v for k, v in old_target.items() if k not in ("quote_count", "geometry_and_spreads_sha256")}:
        raise ValueError("only quote geometry/spreads and the declared numerical representation may change")
    _, keras = _runtime()
    return {"format": "heston_pr87_grid_bank_contract_v1", "predeclared_before_generation": True,
        "task_fingerprint": fingerprint(task), "source": source_bindings(), "parent": parent["identity"],
        "historical_inputs": historical["identity"], "train_rows": 768 if smoke_test_only else 300000,
        "parent_train_rows": parent["identity"]["train_rows"], "appended_rows": 256 if smoke_test_only else 100000,
        "validation_rows": parent["identity"]["validation_rows"], "historical_rows": 512, "quote_count": 63,
        "whitening_rows": 128 if smoke_test_only else 10000, "shared_contracts": shared_contracts(parent["task"], task),
        "simulation": "restore old200k after_append RNG; native sequential simulator: parent200k prior rows then100k fresh priors, then300k new63-feature noises; fixed native/historical truths then fresh noises",
        "shared_error_pairing": "draw all63 noises, then replace shared-column noise in parent200k and native100 with stored (oldquote-oldclean)/(c*h); always reprice; historical512 has fresh63-feature noises",
        "shared_rounding_bound": "8*float64_eps*(abs(newclean)/(c*h)+abs(oldstandardnormal)+1) in standardized residual units",
        "whitening": "native notebook ZCA fit only to first10000 new-grid noisy train rows, frozen for bothfits; all63 features retained",
        "seed": SEED, "max_wall_seconds": MAX_BANK_SECONDS, "smoke_test_only": smoke_test_only,
        "versions": {n: metadata.version(n) for n in ("bayesflow", "keras", "jax", "jaxlib", "numpy")},
        "python": platform.python_version(), "platform": platform.platform(), "backend": keras.backend.backend(),
        "devices": [str(d) for d in __import__("jax").devices()],
        "fresh_noise_per_pass": False, "reserved_arrays_opened": False, "automatic_retry": False,
        "scientific_acceptance": False}


def protocol_contract(task, parent_training, historical_inputs, *, train_rows=200000, smoke_test_only=False):
    bank = bank_contract(task, parent_training, historical_inputs, smoke_test_only=smoke_test_only)
    supported = (512, 768) if smoke_test_only else (200000, 300000)
    if type(train_rows) is not int or train_rows not in supported:
        raise ValueError("only predeclared200k/300k grid fits (512/768 smoke) supported")
    bf, keras = _runtime()
    keras.utils.set_random_seed(SEED)
    epochs, steps = (2 if smoke_test_only else 300), (train_rows + 255) // 256
    workflow = build_workflow(task)
    optimizer = workflow.build_optimizer(epochs, steps, strategy="offline")
    return {"format": "heston_pr87_grid_training_contract_v1", "predeclared_before_fit": True,
        "task_fingerprint": bank["task_fingerprint"], "source": bank["source"], "bank_contract": bank,
        "train_rows": train_rows, "validation_rows": bank["validation_rows"], "epochs": epochs, "batch_size": 256,
        "batches_per_epoch": steps, "tail_batch_rows": train_rows % 256 or 256,
        "example_presentations": epochs * train_rows, "expected_optimizer_updates": epochs * steps,
        "checkpoint_epochs": [1, 2] if smoke_test_only else [100, 200, 250, 300],
        "primary_checkpoint": "final_epoch_only", "archival_selection": "first strictly smallest finite native val_loss; passive until fit ends",
        "target_transform": "raw physical coordinates; concatenate then rename only; no clipping/rejection/target constraint",
        "standardize": "inference_variables", "seed": SEED,
        "resolved_defaults": {"optimizer": _stable_config(keras.optimizers.serialize(optimizer)),
            "network": _stable_config(keras.saving.serialize_keras_object(workflow.inference_network)),
            "fit_offline_signature": str(inspect.signature(bf.BasicWorkflow.fit_offline))},
        "versions": {n: metadata.version(n) for n in ("bayesflow", "keras", "jax", "jaxlib", "numpy")},
        "python": platform.python_version(), "platform": platform.platform(),
        "backend": keras.backend.backend(), "devices": [str(d) for d in __import__("jax").devices()],
        "max_wall_seconds": MAX_TRAIN_SECONDS, "smoke_test_only": smoke_test_only,
        "automatic_extension": False, "retries": 0, "scientific_acceptance": False}


def _truth(arrays, split, names):
    return np.column_stack([arrays[f"{split}_{name}"].reshape(-1) for name in names])


def simulate_bank(task, parent, historical, contract, *, check_budget=lambda: None, progress=lambda n: None):
    """Native all-priors/all-noises ordering, with one paired stored residual."""
    bf, _ = _runtime()
    model = HestonSpreadPilot(task["records"], infer_c=True, c_bounds=(.5, 2.))
    names, bounds = model.parameter_names, np.asarray(model.bounds)
    spread = np.asarray(model.records["half_spread"])
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(parent["identity"]["continuation_rng_state"])
    old_train = _truth(parent["arrays"], "train", names)
    old_validation = _truth(parent["arrays"], "validation", names)
    shared = contract["shared_contracts"][0]
    old_index, new_index = shared["old_index"], shared["new_index"]
    payload, states, pairing = {}, {}, {}
    state = {"split": None, "prior_index": 0, "observation_index": 0, "priced": 0}
    clean_rows, quote_rows = [], []

    def prior_draw():
        check_budget()
        split, index = state["split"], state["prior_index"]
        state["prior_index"] += 1
        if split == "train":
            values = old_train[index] if index < len(old_train) else rng.uniform(bounds[:, 0], bounds[:, 1])
        else:
            values = (old_validation if split == "validation" else historical["truth"])[index]
        return dict(zip(names, values, strict=True))

    def observation_model(initial_variance, kappa, theta, vol_of_vol, rho, c):
        check_budget()
        parameters = np.array([initial_variance, kappa, theta, vol_of_vol, rho, c])
        clean = model.prices(parameters)
        quotes = clean + c * spread * rng.standard_normal(63)
        split, index = state["split"], state["observation_index"]
        state["observation_index"] += 1
        if split == "validation" or (split == "train" and index < len(old_train)):
            old_clean = parent["arrays"][split + "_clean_prices"][index, old_index]
            old_quote = parent["arrays"][split + "_quotes"][index, old_index]
            scale = c * spread[new_index]
            old_z = (old_quote - old_clean) / scale
            quotes[new_index] = clean[new_index] + scale * old_z
            recovered = (quotes[new_index] - clean[new_index]) / scale
            error = abs(recovered - old_z)
            bound = 8 * np.finfo(np.float64).eps * (abs(clean[new_index]) / scale + abs(old_z) + 1)
            if error > bound:
                raise FloatingPointError("shared stored-normal residual recomposition exceeds float64 bound")
            stats = pairing.setdefault(split, {"rows": 0, "clean_bitwise_equal_rows": 0,
                "max_absolute_clean_shift": 0., "max_old_quote_roundtrip_error": 0.,
                "max_standard_normal_reconstruction_error": 0., "max_standard_normal_rounding_bound": 0.})
            stats["rows"] += 1
            stats["clean_bitwise_equal_rows"] += int(clean[new_index] == old_clean)
            for key, value in (("max_absolute_clean_shift", abs(clean[new_index] - old_clean)),
                    ("max_old_quote_roundtrip_error", abs(old_clean + scale * old_z - old_quote)),
                    ("max_standard_normal_reconstruction_error", error),
                    ("max_standard_normal_rounding_bound", bound)):
                stats[key] = max(stats[key], float(value))
        clean_rows.append(clean.copy()); quote_rows.append(quotes.copy())
        state["priced"] += 1
        total = contract["train_rows"] + contract["validation_rows"] + contract["historical_rows"]
        if state["priced"] % 1000 == 0 or state["priced"] == total:
            progress(state["priced"])
        return {"observations": quotes / spread}

    simulator = bf.make_simulator([prior_draw, observation_model])
    for split, rows in (("train", contract["train_rows"]), ("validation", contract["validation_rows"]),
                        ("historical", contract["historical_rows"])):
        state.update(split=split, prior_index=0, observation_index=0)
        clean_rows.clear(); quote_rows.clear()
        states["before_" + split] = deepcopy(rng.bit_generator.state)
        values = simulator.sample(rows)
        payload.update({f"{split}_{key}": np.asarray(value) for key, value in values.items()})
        payload[split + "_clean_prices"], payload[split + "_quotes"] = np.asarray(clean_rows), np.asarray(quote_rows)
        states["after_" + split] = deepcopy(rng.bit_generator.state)
    payload["validation_row_id"] = np.asarray([f"native-validation:{i:06d}" for i in range(contract["validation_rows"])])
    payload["historical_row_id"] = historical["row_id"].copy()
    _validate_arrays(payload, task, contract)
    _validate_pairing(payload, task, parent, historical, contract)
    return payload, states, pairing


def _validate_arrays(arrays, task, contract):
    expected = {}
    for split, rows in (("train", contract["train_rows"]), ("validation", contract["validation_rows"]),
                        ("historical", contract["historical_rows"])):
        expected.update({f"{split}_{name}": (rows, 1) for name in task["target"]["parameter_names"]})
        expected.update({f"{split}_{name}": (rows, 63) for name in ("observations", "quotes", "clean_prices")})
    numeric = set(expected)
    expected.update(validation_row_id=(contract["validation_rows"],), historical_row_id=(contract["historical_rows"],))
    if set(arrays) != set(expected) or any(arrays[k].shape != shape for k, shape in expected.items()):
        raise ValueError("revised-grid bank array inventory or dimensions differ")
    if any(np.iscomplexobj(arrays[k]) or not np.isfinite(arrays[k]).all() for k in numeric):
        raise ValueError("finite real revised-grid bank arrays required")
    spread = np.asarray(task["records"]["half_spread"])
    bounds = np.asarray(task["target"]["prior_bounds"])
    for split in ("train", "validation", "historical"):
        if not np.array_equal(arrays[split + "_observations"], arrays[split + "_quotes"] / spread):
            raise ValueError("new-grid raw quote/native observation reconstruction differs")
        truth = _truth(arrays, split, task["target"]["parameter_names"])
        if np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1]):
            raise ValueError("generated physical/c truth lies outside its unchanged prior")


def _validate_pairing(arrays, task, parent, historical, contract):
    names = task["target"]["parameter_names"]
    for split, rows in (("train", contract["parent_train_rows"]), ("validation", contract["validation_rows"])):
        if not np.array_equal(_truth(arrays, split, names)[:rows], _truth(parent["arrays"], split, names)):
            raise ValueError("paired old-grid physical/c truths changed")
    if (not np.array_equal(_truth(arrays, "historical", names), historical["truth"])
            or not np.array_equal(arrays["historical_row_id"], historical["row_id"])):
        raise ValueError("historical512 physical/c truths or row identities changed")
    shared = shared_contracts(parent["task"], task)
    if shared != contract["shared_contracts"]:
        raise ValueError("shared contract geometry/carry/spread identity changed")
    old_i, new_i = shared[0]["old_index"], shared[0]["new_index"]
    h = task["records"]["half_spread"][new_i]
    for split, rows in (("train", contract["parent_train_rows"]), ("validation", contract["validation_rows"])):
        c = arrays[split + "_c"][:rows, 0]
        old_z = (parent["arrays"][split + "_quotes"][:, old_i] - parent["arrays"][split + "_clean_prices"][:, old_i]) / (c * h)
        clean = arrays[split + "_clean_prices"][:rows, new_i]
        new_z = (arrays[split + "_quotes"][:rows, new_i] - clean) / (c * h)
        bound = 8 * np.finfo(np.float64).eps * (np.abs(clean) / (c * h) + np.abs(old_z) + 1)
        if np.any(np.abs(new_z - old_z) > bound):
            raise ValueError("shared-contract stored standard-normal errors changed beyond declared rounding")


def _payload_destination(path):
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parents[2]
    if path == repo or (repo in path.parents and path.relative_to(repo).parts[0] != "local-artifacts"):
        raise ValueError("full bank/model payloads require untracked local-artifacts or outside-Git destination")
    return path


def generate_bank(task, parent_training, historical_inputs, protocol, out):
    started = time.monotonic()
    protocol_path = Path(protocol).resolve()
    declaration = json.loads(protocol_path.read_text())
    contract = declaration.get("contract", declaration)
    expected = bank_contract(task, parent_training, historical_inputs, smoke_test_only=contract.get("smoke_test_only"))
    if contract != expected:
        raise ValueError("persist exact revised-grid bank contract before pricing")
    task, out = read_task(task), _payload_destination(out)
    if "task" in declaration and declaration["task"] != task:
        raise ValueError("declared new-grid task differs")
    if "output" in declaration and Path(declaration["output"]).resolve() != out:
        raise ValueError("declared bank output differs")
    out.mkdir(parents=True, exist_ok=False)
    parent = _read_parent(parent_training, smoke_test_only=contract["smoke_test_only"])
    historical = _read_historical(historical_inputs, parent["task"])
    record = {"format": "heston_pr87_grid_bank_v1", "status": "started", "task": task,
        "task_fingerprint": fingerprint(task), "protocol": contract, "protocol_path": str(protocol_path),
        "protocol_sha256": digest(protocol_path), "data_payload": "data.npz", "parent": contract["parent"],
        "historical_inputs": contract["historical_inputs"], "smoke_test_only": contract["smoke_test_only"]}
    write_json(out / "bank.json", record)

    def check_budget():
        if time.monotonic() - started >= contract["max_wall_seconds"]:
            raise TimeoutError("bounded bank generation exhausted; no retry or extension")

    def progress(count):
        total = contract["train_rows"] + contract["validation_rows"] + contract["historical_rows"]
        write_json(out / "generation-progress.json", {"completed_pricing_calls": count,
            "total_pricing_calls": total, "elapsed_seconds": time.monotonic() - started,
            "status": "complete" if count == total else "running"})
        print(f"New-grid checked labels: {count}/{total}", flush=True)
        check_budget()

    try:
        progress(0)
        arrays, states, pairing = simulate_bank(task, parent, historical, contract,
            check_budget=check_budget, progress=progress)
        write_npz(out / "data.npz", arrays)
        whitening = NotebookWhitening(arrays["train_observations"][:contract["whitening_rows"]])
        write_json(out / "whitening.json", whitening.to_dict())
        record.update(status="complete", data_payload_sha256=digest(out / "data.npz"),
            data_arrays={k: array_summary(v) for k, v in arrays.items()}, whitening_sha256=digest(out / "whitening.json"),
            numpy_generator_states=states, shared_error_pairing=pairing,
            pricing_calls=contract["train_rows"] + contract["validation_rows"] + contract["historical_rows"],
            elapsed_seconds=time.monotonic() - started)
        check_budget(); write_json(out / "bank.json", record); write_json(out / "generation.json", record)
        return record
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started)
        write_json(out / "bank.json", record); write_json(out / "generation.json", record)
        raise


def load_bank(directory, expected_contract=None):
    directory = Path(directory).resolve()
    record = json.loads((directory / "bank.json").read_text())
    task, contract = read_task(record["task"]), record["protocol"]
    protocol_path = Path(record["protocol_path"])
    declaration = json.loads(protocol_path.read_text())
    expected = bank_contract(task, contract["parent"]["directory"], contract["historical_inputs"]["manifest"]["path"],
                             smoke_test_only=contract.get("smoke_test_only"))
    if (record.get("format") != "heston_pr87_grid_bank_v1" or record.get("status") != "complete"
            or contract != expected or (expected_contract is not None and contract != expected_contract)
            or declaration.get("contract", declaration) != contract
            or record["task_fingerprint"] != fingerprint(task) or record["parent"] != contract["parent"]
            or digest(protocol_path) != record["protocol_sha256"] or record["data_payload"] != "data.npz"
            or digest(directory / "bank.json") != digest(directory / "generation.json")
            or digest(directory / "data.npz") != record["data_payload_sha256"]
            or digest(directory / "whitening.json") != record["whitening_sha256"]):
        raise ValueError("revised-grid bank source/task/payload/whitening identity differs")
    with np.load(directory / "data.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    _validate_arrays(arrays, task, contract)
    parent = _read_parent(contract["parent"]["directory"], smoke_test_only=contract["smoke_test_only"])
    historical = _read_historical(contract["historical_inputs"]["manifest"]["path"], parent["task"])
    _validate_pairing(arrays, task, parent, historical, contract)
    whitening = NotebookWhitening.from_dict(json.loads((directory / "whitening.json").read_text()))
    if (record["data_arrays"] != {k: array_summary(v) for k, v in arrays.items()}
            or whitening.to_dict() != NotebookWhitening(arrays["train_observations"][:contract["whitening_rows"]]).to_dict()):
        raise ValueError("revised-grid bank arrays or first10k-only whitening changed")
    return arrays, record


def _save_checkpoint(path, workflow, record, whitening_path, epoch, logs):
    saved = prior._save_checkpoint(path, workflow, record, whitening_path, epoch, logs)
    saved["format"] = "heston_pr87_grid_checkpoint_v1"
    write_json(path / "checkpoint.json", saved)
    return saved


def train(bankdir, protocol, out, *, train_rows):
    started = time.monotonic()
    protocol_path, bankdir, out = Path(protocol).resolve(), Path(bankdir).resolve(), _payload_destination(out)
    declaration = json.loads(protocol_path.read_text())
    contract = declaration.get("contract", declaration)
    arrays, bank = load_bank(bankdir, contract["bank_contract"])
    task = bank["task"]
    expected = protocol_contract(task, contract["bank_contract"]["parent"]["directory"],
        contract["bank_contract"]["historical_inputs"]["manifest"]["path"],
        train_rows=train_rows, smoke_test_only=contract.get("smoke_test_only"))
    if contract != expected:
        raise ValueError("persist exact per-fit revised-grid training contract before fitting")
    if "contract" in declaration and (declaration.get("task") != task
            or Path(declaration["output"]).resolve() != out):
        raise ValueError("declared revised-grid training task/output differs")
    if "bank_protocol" in declaration and (declaration["bank_protocol"]["sha256"] != bank["protocol_sha256"]
            or identity(declaration["bank_protocol"]["path"]) != declaration["bank_protocol"]):
        raise ValueError("declared shared bank protocol differs")
    out.mkdir(parents=True, exist_ok=False)
    record = {"format": "heston_pr87_grid_run_v1", "status": "started", "task": task,
        "task_fingerprint": fingerprint(task), "parameter_names": task["target"]["parameter_names"],
        "bounds": task["target"]["prior_bounds"], "protocol": contract,
        "protocol_path": str(protocol_path), "protocol_sha256": digest(protocol_path),
        "bank_path": str(bankdir), "bank_record_sha256": digest(bankdir / "bank.json"),
        "data_payload": str(bankdir / "data.npz"), "data_payload_sha256": bank["data_payload_sha256"],
        "data_arrays": bank["data_arrays"], "whitening_sha256": bank["whitening_sha256"],
        "train_rows": train_rows, "primary_epoch": contract["epochs"],
        "smoke_test_only": contract["smoke_test_only"], "checkpoints": [], "pricing_calls": 0}
    write_json(out / "run.json", record)

    def check_budget():
        if time.monotonic() - started >= contract["max_wall_seconds"]:
            raise TimeoutError("bounded native grid fit exhausted; no retry or extension")

    try:
        _, keras = _runtime(); keras.utils.set_random_seed(SEED)
        whitening = NotebookWhitening.from_dict(json.loads((bankdir / "whitening.json").read_text()))
        shutil.copyfile(bankdir / "whitening.json", out / "whitening.json")
        names = task["target"]["parameter_names"]
        data = {split: {name: arrays[f"{split}_{name}"][:train_rows] if split == "train" else arrays[f"{split}_{name}"]
                       for name in [*names, "observations"]} for split in ("train", "validation")}
        for split in data:
            data[split]["observations"] = whitening.transform(data[split]["observations"])
        workflow = build_workflow(task)
        rows, best = [], {"loss": float("inf"), "state": None, "epoch": None, "logs": None}

        class Archive(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                actual = _stable_config(keras.optimizers.serialize(self.model.optimizer))
                if actual != contract["resolved_defaults"]["optimizer"]:
                    raise ValueError("native revised-grid optimizer differs")
                record["resolved_optimizer"] = actual
                record["resolved_estimator_config"] = _stable_config(self.model.get_config())
                record["resolved_compile_config"] = _stable_config(self.model.get_compile_config())
                write_json(out / "run.json", record); check_budget()

            def on_train_batch_begin(self, batch, logs=None):
                check_budget()

            def on_train_batch_end(self, batch, logs=None):
                check_budget()

            def on_epoch_end(self, epoch, logs=None):
                check_budget()
                logs = {k: float(v) for k, v in (logs or {}).items()}; epoch += 1
                if not all(np.isfinite(v) for v in logs.values()):
                    raise FloatingPointError("nonfinite native grid history")
                rows.append({"epoch": epoch, **logs}); write_json(out / "native-epoch-history.json", rows)
                if "val_loss" in logs and logs["val_loss"] < best["loss"]:
                    best.update(loss=logs["val_loss"], state=_state(self.model), epoch=epoch, logs=logs)
                if epoch in contract["checkpoint_epochs"]:
                    path = out / "checkpoints" / f"epoch-{epoch:06d}"
                    _save_checkpoint(path, workflow, record, out / "whitening.json", epoch, logs)
                    record["checkpoints"].append({"epoch": epoch, "path": str(path),
                        "checkpoint_sha256": digest(path / "checkpoint.json")})
                write_json(out / "run.json", record); check_budget()

        history = workflow.fit_offline(data=data["train"], validation_data=data["validation"],
            epochs=contract["epochs"], batch_size=contract["batch_size"], callbacks=[Archive()])
        check_budget()
        if len(history.epoch) != contract["epochs"] or int(workflow.estimator.optimizer.iterations.numpy()) != contract["expected_optimizer_updates"]:
            raise RuntimeError("incomplete declared native grid epochs/updates")
        write_json(out / "native-history.json", {k: [float(x) for x in v] for k, v in history.history.items()})
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
            native_epoch_history_sha256=digest(out / "native-epoch-history.json"), elapsed_seconds=time.monotonic() - started)
        check_budget(); write_json(out / "run.json", record); write_json(out / "training.json", record)
        return record
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started)
        write_json(out / "run.json", record); write_json(out / "training.json", record)
        raise


class RawQuoteWorkflow:
    def __init__(self, workflow, task, whitening):
        self.workflow, self.task, self.whitening, self.estimator = workflow, task, whitening, workflow.estimator

    def sample(self, raw_quotes, count, seed=0):
        if type(count) is not int or count < 1 or type(seed) is not int or seed < 0 or np.iscomplexobj(raw_quotes):
            raise ValueError("positive count/nonnegative integer seed/real raw quotes required")
        quotes = np.atleast_2d(np.asarray(raw_quotes, dtype=float))
        if quotes.ndim != 2 or quotes.shape[1] != 63 or not len(quotes) or not np.isfinite(quotes).all():
            raise ValueError("complete finite revised63-grid raw quotes required")
        conditions = self.whitening.transform(quotes / np.asarray(self.task["records"]["half_spread"]))
        values = self.workflow.sample(conditions={"observations": conditions}, num_samples=count, sample_shape=(), seed=seed)
        result = np.concatenate([np.asarray(values[n]) for n in self.task["target"]["parameter_names"]], axis=-1)
        if result.shape != (len(quotes), count, 6):
            raise FloatingPointError("invalid raw physical posterior sample shape")
        return result


def sample(workflow, task, raw_quotes, count, seed=0):
    if fingerprint(read_task(task)) != fingerprint(workflow.task):
        raise ValueError("sampling task differs from loaded revised-grid checkpoint")
    return workflow.sample(raw_quotes, count, seed)


def load_checkpoint(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory / "checkpoint.json").read_text())
    if record.get("format") != "heston_pr87_grid_checkpoint_v1":
        raise ValueError("new revised-grid checkpoint identity required")
    task, contract = read_task(record["task"]), record["protocol"]
    protocol_path = Path(record["protocol_path"])
    declaration = json.loads(protocol_path.read_text())
    training = directory.parent.parent
    run = json.loads((training / "run.json").read_text())
    bankdir = Path(record["bank_path"])
    arrays, bank = load_bank(bankdir, contract["bank_contract"])
    expected = protocol_contract(task, contract["bank_contract"]["parent"]["directory"],
        contract["bank_contract"]["historical_inputs"]["manifest"]["path"], train_rows=contract["train_rows"],
        smoke_test_only=contract.get("smoke_test_only"))
    entries = [run["best_validation_archive"]] if directory.name == "best-validation" else run["checkpoints"]
    entries = [q for q in entries if Path(q["path"]).resolve() == directory]
    if (run.get("status") != "complete" or run.get("format") != "heston_pr87_grid_run_v1"
            or contract != expected or declaration.get("contract", declaration) != contract
            or record["task_fingerprint"] != fingerprint(task) or bank["task"] != task
            or record["task"] != run["task"] or record["protocol"] != run["protocol"]
            or digest(protocol_path) != record["protocol_sha256"] or run["protocol_sha256"] != record["protocol_sha256"]
            or digest(training / "run.json") != digest(training / "training.json")
            or record["parameter_names"] != task["target"]["parameter_names"] or record["bounds"] != task["target"]["prior_bounds"]
            or record["train_rows"] != contract["train_rows"] or run["train_rows"] != contract["train_rows"]
            or record["native_epoch"] not in range(1, contract["epochs"] + 1)
            or record["actual_optimizer_updates"] != record["native_epoch"] * contract["batches_per_epoch"]
            or run["completed_epochs"] != contract["epochs"] or run["actual_optimizer_updates"] != contract["expected_optimizer_updates"]
            or record["bank_record_sha256"] != digest(bankdir / "bank.json") or run["bank_record_sha256"] != record["bank_record_sha256"]
            or Path(record["data_payload"]).resolve() != bankdir / "data.npz"
            or record["data_payload_sha256"] != bank["data_payload_sha256"] or run["data_payload_sha256"] != record["data_payload_sha256"]
            or record["data_arrays"] != bank["data_arrays"] or record["whitening_sha256"] != bank["whitening_sha256"]
            or digest(training / "whitening.json") != record["whitening_sha256"]
            or len(entries) != 1 or entries[0]["epoch"] != record["native_epoch"]
            or digest(directory / "checkpoint.json") != entries[0]["checkpoint_sha256"]
            or any(digest(directory / name) != record[key] for name, key in
                (("model.keras", "model_sha256"), ("state.npz", "state_sha256"), ("whitening.json", "whitening_sha256")))):
        raise ValueError("revised-grid checkpoint task/source/bank/whitening/full payload identity differs")
    workflow = build_workflow(task); workflow.load_estimator(directory / "model.keras")
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        keys = {f"{group}_{i:04d}" for group in GROUPS for i in range(len(record["state"][group]))}
        if set(archive.files) != keys:
            raise ValueError("revised-grid full-state inventory differs")
        state = tuple([archive[f"{group}_{i:04d}"].copy() for i in range(len(record["state"][group]))] for group in GROUPS)
    _assign_state(workflow.estimator, state)
    actual = _state_summary(workflow.estimator, _state(workflow.estimator))
    if ({g: [{k: q[k] for k in ("shape", "dtype", "sha256")} for q in v] for g, v in actual.items()}
            != {g: [{k: q[k] for k in ("shape", "dtype", "sha256")} for q in v] for g, v in record["state"].items()}
            or int(workflow.estimator.optimizer.iterations.numpy()) != record["actual_optimizer_updates"]
            or _stable_config(__import__("keras").optimizers.serialize(workflow.estimator.optimizer)) != contract["resolved_defaults"]["optimizer"]):
        raise ValueError("revised-grid native full state differs after reload")
    whitening = NotebookWhitening.from_dict(json.loads((directory / "whitening.json").read_text()))
    return RawQuoteWorkflow(workflow, task, whitening), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    for key in ("task", "parent", "historical-inputs", "protocol", "out"):
        generate.add_argument("--" + key, required=True)
    generate.add_argument("--worker", action="store_true")
    fit = commands.add_parser("train")
    for key in ("bank", "protocol", "out"):
        fit.add_argument("--" + key, required=True)
    fit.add_argument("--train-rows", type=int, required=True); fit.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        generate_bank(args.task, args.parent, args.historical_inputs, args.protocol, args.out)
    else:
        train(args.bank, args.protocol, args.out, train_rows=args.train_rows)


if __name__ == "__main__":
    main()

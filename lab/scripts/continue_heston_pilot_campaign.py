"""Continue the frozen 100k-row Heston trajectory without resetting its optimizer.

Production use extends only its 50,000-update checkpoint to 100,000 updates.
The serialized Adam moments, standardization/seed state and 50k CosineDecay
schedule are retained. Absolute data/noise steps continue at 50,000. No bank
generation or reserved-test reads are supported by this tool.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train_heston_pilot_campaign as producer
from scripts.train_heston_pilot import sample, training_data
from scripts.train_full_surface_budget import BatchPlan


def array_fingerprint(variables):
    """Hash ordered dtype/shape/value arrays; variable names remain descriptive."""
    digest = hashlib.sha256()
    specifications = []
    for variable in variables:
        values = np.asarray(variable.numpy())
        shape = list(values.shape)
        header = {"dtype": str(values.dtype), "shape": shape}
        digest.update(producer.canonical(header).encode())
        digest.update(np.ascontiguousarray(values).tobytes())
        specifications.append({"name": str(getattr(variable, "path", getattr(variable, "name", "unnamed"))), **header})
    return {"sha256": digest.hexdigest(), "variables": specifications}


def state_record(model):
    """Serializable hashes of the learned state and full optimizer slots."""
    if hasattr(model, "jax_state_sync"):
        model.jax_state_sync()
    optimizer = model.optimizer
    result = {"trainable": array_fingerprint(model.trainable_variables),
              "non_trainable": array_fingerprint(model.non_trainable_variables),
              "non_trainable_without_seeds": array_fingerprint(
                  [variable for variable in model.non_trainable_variables if "seed_generator_state" not in variable.path]),
              "standardization": array_fingerprint(model.standardizer.variables),
              "optimizer": array_fingerprint(optimizer.variables),
              "optimizer_config": optimizer.get_config(),
              "optimizer_config_sha256": producer.fingerprint(optimizer.get_config()),
              "iterations": int(optimizer.iterations.numpy()),
              "learning_rate": float(np.asarray(optimizer.learning_rate))}
    generator = getattr(model, "seed_generator", None)
    if generator is not None:
        result["estimator_seed"] = array_fingerprint([generator.state])
    return result


def assert_same_state(before, after):
    for key in ("trainable", "non_trainable", "standardization", "optimizer", "estimator_seed"):
        if key in before and (key not in after or before[key]["sha256"] != after[key]["sha256"]):
            raise ValueError(f"parent {key} state changed before the first continuation update")
    for key in ("iterations", "optimizer_config_sha256", "learning_rate"):
        if before[key] != after[key]:
            raise ValueError(f"parent optimizer {key} changed before the first continuation update")


def absolute_batch(train, plan, step, halfspread, market_vega, seed):
    indices = plan.indices(step)
    quotes, c, parameters = producer.augmented_batch(
        train["clean"][indices], train["parameters"][indices], halfspread, market_vega, step, seed)
    return indices, quotes, c, parameters


def expected_iteration(start, relative_batch, actual):
    expected = start + int(relative_batch) + 1
    if actual != expected:
        raise RuntimeError(f"continuation optimizer iteration {actual} differs from expected {expected}")
    return expected


def seed_variables(model):
    return [(index, variable) for index, variable in enumerate(model.non_trainable_variables)
            if "seed_generator_state" in variable.path]


def recover_original_rng(metadata, parent_run, records, bank_path):
    """Replay initialization and two tiny updates to recover linear RNG counters.

    The frozen .keras archive omits SeedGenerator variables. Dropout is active
    in this spline's default MLP, so these variables cannot be ignored. The
    original initialization uses one fixed Keras seed and the same public
    build-from-data route. Two bounded updates establish each counter's stride.
    The independent resumed-versus-uninterrupted smoke verifies this recovery.
    """
    import keras
    with np.load(Path(bank_path) / "bank.npz", allow_pickle=False) as archive:
        train = {"parameters": archive["train_parameters"][:metadata["train_rows"]],
                 "clean": archive["train_clean"][:metadata["train_rows"]]}
    seed = metadata["seed"]
    template = producer.build_workflow(metadata["names"], np.asarray(metadata["bounds"]), metadata["schedule_updates"])
    plan = BatchPlan(metadata["train_rows"], parent_run["batch_size"], 2, seed)
    h, vega = np.asarray(records["half_spread"]), np.asarray(records["market_vega"])
    mask = np.ones(len(h), dtype=bool)

    class Dataset(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)

        def __len__(self):
            return 2

        def __getitem__(self, step):
            _, quotes, c, theta = absolute_batch(train, plan, int(step), h, vega, seed)
            return template.adapter(training_data(quotes, c, theta, records, mask, metadata["names"]))

    dataset = Dataset()
    mock = keras.tree.map_structure(keras.ops.convert_to_tensor, dataset[0])
    template.estimator.build_from_data(mock)
    initial = [np.asarray(variable.numpy()).copy() for _, variable in seed_variables(template.estimator)]
    states = []

    class Capture(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if hasattr(self.model, "jax_state_sync"):
                self.model.jax_state_sync()
            states.append([np.asarray(variable.numpy()).copy() for _, variable in seed_variables(self.model)])

    template.estimator.fit(dataset=dataset, epochs=1, shuffle=False, verbose=0, callbacks=[Capture()])
    if len(states) != 2 or len(initial) != len(states[0]) or len(initial) != len(states[1]):
        raise ValueError("RNG calibration did not capture exactly two complete training steps")
    recovered, descriptions = [], []
    for index, (before, first, second) in enumerate(zip(initial, states[0], states[1])):
        if (before.shape != (2,) or not np.issubdtype(before.dtype, np.integer)
                or first[0] != before[0] or second[0] != before[0]
                or first[1] < before[1] or second[1] - first[1] != first[1] - before[1]):
            raise ValueError("SeedGenerator is not a fixed seed plus linear per-update counter")
        stride = int(first[1] - before[1])
        desired = before.copy()
        desired[1] = int(before[1]) + metadata["actual_optimizer_updates"] * stride
        recovered.append(desired)
        descriptions.append({"seed_variable_index": index, "initial": before.tolist(),
                             "after_one_update": first.tolist(), "after_two_updates": second.tolist(),
                             "counter_stride": stride, "recovered_parent": desired.tolist()})
    return recovered, {"method": "Replay frozen original initialization; public build_from_data and two original absolute training batches; extrapolate verified constant RNG counter stride",
                       "parent_updates": metadata["actual_optimizer_updates"], "seed_states": descriptions,
                       "gradient_rng_variables": sum(item["counter_stride"] > 0 for item in descriptions),
                       "scope": "Recovery is verified against an independent uninterrupted smoke trajectory before production use."}


def assign_recovered_seeds(model, recovered):
    variables = seed_variables(model)
    if len(variables) != len(recovered):
        raise ValueError("reconstructed architecture has a different number of seed variables")
    for (_, variable), values in zip(variables, recovered):
        if tuple(variable.shape) != values.shape or np.dtype(variable.dtype) != values.dtype:
            raise ValueError("reconstructed seed variable shape or dtype differs")
        variable.assign(values)


def save_state_sidecar(model, directory):
    variables = model.non_trainable_variables
    path = Path(directory) / "continuation-state.npz"
    np.savez_compressed(path, **{f"variable_{index:03d}": np.asarray(variable.numpy())
                               for index, variable in enumerate(variables)})
    return {"file": path.name, "sha256": producer.digest(path),
            "non_trainable_state": array_fingerprint(variables)}


def load_preserved_checkpoint(directory):
    workflow, record = producer.load_checkpoint(directory)
    sidecar = record.get("continuation_state")
    if sidecar is not None:
        path = Path(directory) / sidecar["file"]
        if path.resolve().parent != Path(directory).resolve() or producer.digest(path) != sidecar["sha256"]:
            raise ValueError("continuation state sidecar path or hash differs")
        variables = workflow.estimator.non_trainable_variables
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {f"variable_{i:03d}" for i in range(len(variables))}:
                raise ValueError("continuation sidecar variable count differs")
            for index, variable in enumerate(variables):
                values = archive[f"variable_{index:03d}"]
                if values.shape != tuple(variable.shape) or values.dtype != np.dtype(variable.dtype):
                    raise ValueError("continuation sidecar shape or dtype differs")
                variable.assign(values)
        if array_fingerprint(variables)["sha256"] != sidecar["non_trainable_state"]["sha256"]:
            raise ValueError("continuation non-trainable state did not restore exactly")
    return workflow, record


def load_parent_inputs(parent_path, design_path, bank_path, *, total_updates, smoke_test_only):
    """Bind the immutable parent, frozen task and projected non-test bank roles."""
    parent_path, bank_path = Path(parent_path), Path(bank_path)
    if not (parent_path / "run.json").exists():
        raise ValueError("resume requires a final parent run directory with run.json")
    run = json.loads((parent_path / "run.json").read_text())
    model, metadata = load_preserved_checkpoint(parent_path)
    for key in ("task", "task_fingerprint", "design_sha256", "names", "bounds", "train_rows", "seed", "schedule_updates"):
        if run[key] != metadata[key]:
            raise ValueError(f"parent run/checkpoint {key} differs")
    start = metadata["actual_optimizer_updates"]
    if start != metadata["planned_updates"] or start >= total_updates:
        raise ValueError("parent must be a completed shorter trajectory")
    if not smoke_test_only and (start != 50000 or total_updates != 100000
            or metadata["train_rows"] != 100000 or metadata["schedule_updates"] != 50000
            or run["batch_size"] != 128):
        raise ValueError("production continuation is only the 100k-row, 50k-to-100k extension with the original 50k schedule")
    task, records, pilot = producer.make_task(design_path, kappa_max=metadata["bounds"][1][1], xi_max=metadata["bounds"][3][1])
    if task != metadata["task"] or producer.fingerprint(task) != metadata["task_fingerprint"]:
        raise ValueError("continuation design or task differs from the parent")
    manifest_path = bank_path / "bank.json"
    if (not manifest_path.exists() or not (bank_path / "bank.npz").exists()
            or bank_path.resolve() != Path(metadata["bank_path"]).resolve()
            or producer.digest(manifest_path) != metadata["bank_record_sha256"]):
        raise ValueError("continuation bank path or record differs from the parent; generation is unsupported")
    manifest = json.loads(manifest_path.read_text())
    payload_path = bank_path / manifest["payload"]
    if (payload_path.resolve().parent != bank_path.resolve()
            or producer.digest(payload_path) != metadata["bank_payload_sha256"]
            or manifest["payload_sha256"] != metadata["bank_payload_sha256"]
            or manifest["identity"]["task_fingerprint"] != metadata["task_fingerprint"]
            or manifest["identity"]["task"] != task):
        raise ValueError("continuation bank payload or task binding differs")
    if not smoke_test_only and (manifest["identity"]["train_rows"] != 100000
            or manifest["identity"]["validation_rows"] != 5000 or manifest["identity"]["test_rows"] != 2000):
        raise ValueError("production bank split counts differ from the frozen campaign")
    if producer.digest(Path(producer.__file__)) != metadata["source_sha256"]["campaign_trainer"]:
        raise ValueError("frozen original campaign producer has changed")
    # This existing loader is invoked only after existence/hash checks. It can
    # neither generate a missing bank here nor read reserved test members.
    payload, loaded = producer.load_or_generate_bank(bank_path, manifest["identity"], pilot)
    if loaded != manifest:
        raise ValueError("bank record changed during continuation loading")
    return model, metadata, run, records, pilot, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "design", "bank", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--total-updates", type=int, default=100000)
    parser.add_argument("--checkpoint-updates", type=int, nargs="+")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--diagnostic-draws", type=int, default=512)
    parser.add_argument("--smoke-test-only", action="store_true")
    args = parser.parse_args()
    if min(args.total_updates, args.log_every, args.diagnostic_draws) < 1:
        parser.error("update and diagnostic settings must be positive")
    if args.out.exists() or args.out.resolve().is_relative_to(args.parent.resolve()):
        raise ValueError("continuation requires a fresh output directory outside its immutable parent")
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import keras
    raw_metadata = json.loads((args.parent / "checkpoint.json").read_text())
    parent_declaration = json.loads((args.parent / "run.json").read_text())
    declared_design = json.loads(args.design.read_text())
    if (producer.digest(args.parent / "model.keras") != raw_metadata["model_sha256"]
            or producer.digest(args.design) != raw_metadata["design_sha256"]
            or producer.digest(args.bank / "bank.npz") != raw_metadata["bank_payload_sha256"]):
        raise ValueError("parent, design or bank differs before RNG reconstruction")
    keras.utils.set_random_seed(raw_metadata["seed"])
    recovered_rng, rng_record = recover_original_rng(raw_metadata, parent_declaration, declared_design["records"], args.bank)
    workflow, metadata, parent_run, records, pilot, payload = load_parent_inputs(
        args.parent, args.design, args.bank, total_updates=args.total_updates, smoke_test_only=args.smoke_test_only)
    archived_state = state_record(workflow.estimator)
    assign_recovered_seeds(workflow.estimator, recovered_rng)
    recovered_state = state_record(workflow.estimator)
    for key in ("trainable", "standardization", "optimizer", "non_trainable_without_seeds"):
        if archived_state[key]["sha256"] != recovered_state[key]["sha256"]:
            raise ValueError("RNG recovery modified learned, optimizer or standardization state")
    start, seed, rows = metadata["actual_optimizer_updates"], metadata["seed"], metadata["train_rows"]
    checkpoints = sorted(set(args.checkpoint_updates or [75000, 100000]))
    if (not checkpoints or checkpoints[-1] != args.total_updates
            or any(point <= start or point > args.total_updates for point in checkpoints)
            or (not args.smoke_test_only and checkpoints != [75000, 100000])):
        raise ValueError("checkpoints must be predeclared future endpoints ending at the total update count")
    parent_state = state_record(workflow.estimator)
    if parent_state["iterations"] != start:
        raise ValueError("loaded optimizer does not retain the parent iteration")
    if not args.smoke_test_only and not np.isclose(parent_state["learning_rate"], 5e-5, rtol=1e-6, atol=0.):
        raise ValueError("the unchanged parent schedule is not at its expected learning-rate floor")
    run = copy.deepcopy(parent_run)
    run.update(format="heston_pilot_continuation_run_v1", scientific_acceptance=False,
               planned_updates=args.total_updates, start_updates=start,
               schedule_updates=metadata["schedule_updates"], smoke_test_only=args.smoke_test_only,
               parent={"directory": str(args.parent.resolve()), "checkpoint_sha256": producer.digest(args.parent / "checkpoint.json"),
                       "model_sha256": metadata["model_sha256"], "run_sha256": producer.digest(args.parent / "run.json"),
                       "optimizer_state_sha256": parent_state["optimizer"]["sha256"],
                       "standardization_state_sha256": parent_state["standardization"]["sha256"]},
               checkpoint_updates=checkpoints,
               continuation_rule="Load full serialized optimizer/standardization/seed state; do not compile or restart. Original schedule remains fixed. Data, tau and noise use absolute steps.",
               training_loss_scope="callback loss is the current training batch; fixed probes match the parent endpoint",
               legacy_parent_rng_scope="The original .keras omitted SeedGenerator states. Original initialization and linear training-counter replay recover active dropout RNG; new checkpoints preserve full non-trainable sidecars.")
    run["source_sha256"]["continuation_trainer"] = producer.digest(__file__)
    train = producer.paired_subset(payload, rows)
    plan = BatchPlan(rows, parent_run["batch_size"], args.total_updates, seed)
    mask = np.ones(len(records["mid"]), dtype=bool)
    h, vega = np.asarray(records["half_spread"]), pilot.market_vega

    class Dataset(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)

        def __len__(self):
            return args.total_updates - start

        def __getitem__(self, relative):
            if not 0 <= int(relative) < len(self):
                raise IndexError("continuation batch outside the predeclared interval")
            _, quotes, c, theta = absolute_batch(train, plan, start + int(relative), h, vega, seed)
            return workflow.adapter(training_data(quotes, c, theta, records, mask, pilot.names))

    def fixed_probe(clean, theta, count, namespace):
        quotes, c, parameters = producer.augmented_batch(clean[:count], theta[:count], h, vega, 0, seed, namespace=namespace)
        return workflow.adapter(training_data(quotes, c, parameters, records, mask, pilot.names))

    counts = parent_run["curve_probe"]["rows"]
    probes = {"train": fixed_probe(train["clean"], train["parameters"], counts["train"], 2),
              "validation": fixed_probe(payload["validation_clean"], payload["validation_parameters"], counts["validation"], 1)}

    def probe_losses():
        if hasattr(workflow.estimator, "jax_state_sync"):
            workflow.estimator.jax_state_sync()
        return {key + "_probe_loss": float(keras.ops.convert_to_numpy(
            workflow.estimator.compute_metrics(**value, stage="validation")["loss"])) for key, value in probes.items()}

    initial_probes = probe_losses()
    for key, value in initial_probes.items():
        if not np.isclose(value, metadata["curve_probe_metrics"][key], rtol=1e-6, atol=1e-6):
            raise ValueError("resumed fixed probe differs from its parent endpoint")
    assert_same_state(parent_state, state_record(workflow.estimator))
    first_indices, _, _, _ = absolute_batch(train, plan, start, h, vega, seed)
    args.out.mkdir(parents=True, exist_ok=False)
    producer.write_json(args.out / "run.json", run)
    resume_record = {"format": "heston_pilot_resume_state_v1", "parent_state": parent_state,
        "legacy_archived_state_before_rng_recovery": archived_state, "rng_reconstruction": rng_record,
        "initial_fixed_probes": initial_probes, "first_absolute_data_step": start,
        "first_batch_row_ids": train["row_id"][first_indices].tolist(),
        "first_batch_indices_sha256": hashlib.sha256(np.asarray(first_indices, dtype="<i8").tobytes()).hexdigest(),
        "source_sha256": run["source_sha256"], "exact_parent_state_before_fit": True}
    producer.write_json(args.out / "resume.json", resume_record)
    run["resume_record_sha256"] = producer.digest(args.out / "resume.json")
    producer.write_json(args.out / "run.json", run)
    curve = [{"updates": start, "continuation_updates": 0, "seconds": 0., **initial_probes}]
    producer.write_json(args.out / "curve.json", {"points": curve, "selection": "predeclared absolute update endpoints"})
    started = time.perf_counter()

    class Guard(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.updates = start
            self.visits = np.zeros(rows, dtype=np.int64)
            self.order = hashlib.sha256()

        def on_train_begin(self, logs=None):
            if int(self.model.steps_per_execution) != 1:
                raise ValueError("continuation requires exact one-update callback accounting")
            assert_same_state(parent_state, state_record(self.model))
            resume_record["exact_parent_state_at_train_begin"] = True
            producer.write_json(args.out / "resume.json", resume_record)
            run["resume_record_sha256"] = producer.digest(args.out / "resume.json")
            producer.write_json(args.out / "run.json", run)

        def on_train_batch_end(self, batch, logs=None):
            actual = int(self.model.optimizer.iterations.numpy())
            expected_iteration(start, batch, actual)
            if actual != self.updates + 1:
                raise RuntimeError("continuation callback skipped or repeated an optimizer update")
            indices = plan.indices(self.updates)
            self.visits[indices] += 1
            self.order.update(np.asarray([self.updates], dtype="<i8").tobytes())
            self.order.update(np.asarray(indices, dtype="<i8").tobytes())
            self.updates = actual

        def on_train_end(self, logs=None):
            if self.updates != args.total_updates or int(self.model.optimizer.iterations.numpy()) != args.total_updates:
                raise RuntimeError("continuation ended before its requested absolute iteration")

    guard = Guard()

    class Monitor(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            actual = int(self.model.optimizer.iterations.numpy())
            if actual % args.log_every and actual not in checkpoints:
                return
            if actual != guard.updates:
                raise RuntimeError("continuation monitor and update guard disagree")
            point = {"updates": actual, "continuation_updates": actual - start,
                     "seconds": time.perf_counter() - started,
                     "batch_training_loss": float(logs["loss"]), **probe_losses()}
            if any(not np.isfinite(value) for value in point.values()):
                raise FloatingPointError("nonfinite continuation curve")
            curve.append(point)
            producer.write_json(args.out / "curve.json", {"points": curve, "selection": "predeclared absolute update endpoints"})
            print(json.dumps(point), flush=True)
            if actual in checkpoints:
                destination = args.out if actual == args.total_updates else args.out / "checkpoints" / f"update-{actual:06d}"
                if destination != args.out:
                    destination.mkdir(parents=True, exist_ok=False)
                self.model.save(destination / "model.keras")
                record = producer._checkpoint_record(run, destination, actual, point, time.perf_counter() - started)
                record["continuation_state"] = save_state_sidecar(self.model, destination)
                record.update(continuation_parent=run["parent"], start_optimizer_updates=start,
                              resumed_optimizer_state_sha256=parent_state["optimizer"]["sha256"],
                              resume_record_sha256=producer.digest(args.out / "resume.json"),
                              smoke_test_only=args.smoke_test_only)
                producer.write_json(destination / "checkpoint.json", record)
                restored, _ = load_preserved_checkpoint(destination)
                # Verify weights, standardization and every optimizer slot survived serialization.
                assert_same_state(state_record(self.model), state_record(restored.estimator))
                market = np.asarray(records["mid"])[None, :]
                before = sample(workflow, pilot.names, market, np.ones(1), records, mask, 16, seed + 31)
                after = sample(restored, pilot.names, market, np.ones(1), records, mask, 16, seed + 31)
                if not np.array_equal(before, after):
                    raise RuntimeError("continuation checkpoint changed same-seed posterior draws")
                draws = sample(restored, pilot.names, market, np.ones(1), records, mask, args.diagnostic_draws, seed + 32)[0]
                np.savez_compressed(destination / "condition-diagnostics.npz", market_c1=draws)
                record.update(same_seed_reload_exact=True, optimizer_and_standardization_reload_exact=True,
                    diagnostic_payload_sha256=producer.digest(destination / "condition-diagnostics.npz"),
                    market_c1_parameter_quantiles_05_50_95={name: np.quantile(draws[:, i], [.05, .5, .95]).tolist() for i, name in enumerate(pilot.names)})
                producer.write_json(destination / "checkpoint.json", record)

    history = workflow.estimator.fit(dataset=Dataset(), epochs=1, shuffle=False, verbose=0, callbacks=[guard, Monitor()])
    if (producer.digest(args.parent / "model.keras") != run["parent"]["model_sha256"]
            or producer.digest(args.bank / "bank.npz") != metadata["bank_payload_sha256"]):
        raise ValueError("immutable parent or bank changed during continuation")
    record = json.loads((args.out / "checkpoint.json").read_text())
    record.update(format="heston_pilot_campaign_continuation_training_v1", updates=guard.updates,
                  continuation_updates=guard.updates - start, training_seconds=time.perf_counter() - started,
                  history={key: [float(value) for value in values] for key, values in history.history.items()},
                  continuation_exposure={"row_visits_min": int(guard.visits.min()), "row_visits_max": int(guard.visits.max()),
                                         "absolute_batch_order_sha256": guard.order.hexdigest()},
                  curve_sha256=producer.digest(args.out / "curve.json"))
    producer.write_json(args.out / "training.json", record)
    print(json.dumps({"out": str(args.out), "start_updates": start, "actual_updates": guard.updates,
                      "seconds": record["training_seconds"], "scientific_acceptance": False}), flush=True)


if __name__ == "__main__":
    main()

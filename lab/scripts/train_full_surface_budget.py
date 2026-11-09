"""Fresh-run, exact-update training mode for the bounded Heston study.

The epoch trainer and its recovery format remain separate. This mode uses the
public BayesFlow dataset and estimator.fit interfaces with a finite schedule of
batches. IDs/counts are retained for accounting and never passed to the adapter.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


class BatchPlan:
    """A bounded reproducible row schedule, independent of dataset probes/prefetch."""

    def __init__(self, rows, batch_size, updates, seed):
        self.rows = _positive_integer(rows, "rows")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.updates = _positive_integer(updates, "updates")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(seed)
        self.batches_per_pass = (self.rows + self.batch_size - 1) // self.batch_size

    @lru_cache(maxsize=2)
    def _permutation(self, epoch):
        result = np.random.default_rng(np.random.SeedSequence([0x42554447, self.seed, epoch])).permutation(self.rows)
        result.flags.writeable = False
        return result

    def indices(self, step):
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or not 0 <= step < self.updates:
            raise IndexError("batch step is outside the predeclared update budget")
        epoch, batch = divmod(int(step), self.batches_per_pass)
        start = batch * self.batch_size
        return self._permutation(epoch)[start:min(start + self.batch_size, self.rows)]

    def noise(self, step, shape):
        # Repeated build/probe requests return the same batch. The independent
        # standard-normal observation law is unchanged; no online pricing occurs.
        return np.random.default_rng(np.random.SeedSequence([0x4E4F4953, self.seed, int(step)])).standard_normal(shape)


def make_dataset(data, adapter, plan):
    """Use a public PyDataset; one finite epoch contains exactly plan.updates."""
    import keras

    class BudgetDataset(keras.utils.PyDataset):
        def __init__(self):
            # No producer threads or unbounded queue of full surfaces.
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)

        def __len__(self):
            return plan.updates

        def __getitem__(self, step):
            indices = plan.indices(step)
            batch = {key: np.take(value, indices, axis=0) for key, value in data.items()}
            quotes = np.array(batch["observations"], copy=True)
            quotes += plan.noise(step, quotes.shape)
            batch["observations"] = quotes
            return adapter(batch)

    return BudgetDataset()


def update_guard(plan):
    """Assert actual per-batch optimizer work and account only committed updates."""
    import keras

    class ExactUpdates(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.updates = 0
            self.visits = np.zeros(plan.rows, dtype=np.int64)
            self.order = hashlib.sha256()
            self.batch_sizes = {}

        def on_train_begin(self, logs=None):
            if int(self.model.optimizer.iterations.numpy()) != 0:
                raise ValueError("update-budget mode requires a fresh optimizer; resume is unsupported")
            if int(self.model.steps_per_execution) != 1:
                raise ValueError("exact update accounting requires steps_per_execution=1 before training")

        def on_train_batch_end(self, batch, logs=None):
            actual = int(self.model.optimizer.iterations.numpy())
            if actual != self.updates + 1 or int(batch) != self.updates:
                raise RuntimeError("optimizer updates differ from the finite batch schedule")
            indices = plan.indices(self.updates)
            self.visits[indices] += 1
            self.order.update(np.asarray(indices, dtype="<i8").tobytes())
            size = len(indices)
            self.batch_sizes[str(size)] = self.batch_sizes.get(str(size), 0) + 1
            self.updates = actual
            if actual > plan.updates:
                raise RuntimeError("optimizer exceeded its predeclared update budget")

        def on_train_end(self, logs=None):
            if self.updates != plan.updates or int(self.model.optimizer.iterations.numpy()) != plan.updates:
                raise RuntimeError("training stopped before the exact requested update budget")

    return ExactUpdates()


def schedule_record(optimizer, updates):
    """Record the serialized schedule and actual discrete LR samples before fit."""
    schedule = optimizer.get_config()["learning_rate"]
    # Public serialization supplies the complete rule; representative samples
    # include every step of the very short runs and critical schedule boundaries.
    import keras
    rule = keras.optimizers.schedules.deserialize(schedule)
    steps = (list(range(updates + 1)) if updates <= 100
             else sorted({0, 1, int(.05 * updates), updates // 2, updates - 1, updates}))
    return {"serialized": schedule, "discrete_learning_rates": {str(step): float(rule(step)) for step in steps}}


def run_budget_training(args, trainer):
    """Train one immutable fresh endpoint; never overwrite or resume old output."""
    from sbi4finance.calibration.performance import PerformanceRecorder, synchronize_model, training_callback

    updates = _positive_integer(args.updates, "updates")
    if args.family != "flow_matching" or args.encoder != "vector":
        raise ValueError("the declared update-budget study requires flow_matching with vector conditions")
    if getattr(args, "checkpoint_selection", "validation_loss") != "final":
        raise ValueError("update-budget mode requires --checkpoint-selection final (endpoint only)")
    manifest = json.loads((args.bank / "manifest.json").read_text())
    contract = trainer.run_contract(args, manifest)
    contract["format"] = "full_surface_update_budget_run_v1"
    contract["settings"] = {"updates": updates, "batch_size": args.batch_size, "seed": args.seed,
                            "fresh_quote_noise": True, "checkpoint_selection": "endpoint",
                            "steps_per_execution": 1,
                            "data_order": "seeded within-bank permutations; finite ordered batch schedule",
                            "noise_stream": "standard-normal keyed by seed and batch; probe invariant"}
    contract["source_sha256"]["scripts/train_full_surface_budget.py"] = trainer.sha256(__file__)
    trainer.recovery.check_run_directory(args.out, contract, False)
    recorder = PerformanceRecorder()
    trainer.recovery.reserve_run_directory(args.out, contract, False)

    def journal(status, **extra):
        destination = args.out / "budget-run.json"
        temporary = args.out / ".budget-run.json.tmp"
        temporary.write_text(json.dumps({"format": "full_surface_update_budget_journal_v1", "status": status,
                                          "requested_updates": updates, "automatic_retry": False,
                                          **extra}, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)

    journal("preparing")
    try:
        with recorder.phase("data_loading"):
            loaded_manifest, splits = trainer.bank.load_bank(args.bank, splits=("train", "validation"),
                                                             roles=trainer.bank.TRAINING_ROLES)
            if trainer.sha256(args.bank / "manifest.json") != contract["dataset"]["manifest_sha256"]:
                raise ValueError("bank manifest changed while loading its immutable snapshot")
            if loaded_manifest != manifest:
                raise ValueError("loaded bank manifest differs from the predeclared snapshot")
            task = trainer.full_surface.validate_task(manifest["configuration"]["task"])
            if task["model"] != "hkde":
                raise ValueError("this update-budget study is restricted to the Heston option task")
            config = contract["network"]
            train = trainer.full_surface.training_data(splits["train"], task, "vector", clean=True)
            validation = trainer.full_surface.training_data(splits["validation"], task, "vector")
            ids = np.asarray(splits["train"]["example_id"])
            if len(ids) != len(train["observations"]) or len(np.unique(ids)) != len(ids):
                raise ValueError("unique train IDs are required for exact exposure accounting")
        with recorder.phase("neural_imports"):
            import keras
            import bayesflow  # noqa: F401 - finish import-time seed consumption first
        keras.utils.set_random_seed(args.seed)
        plan = BatchPlan(len(ids), args.batch_size, updates, args.seed)
        with recorder.phase("workflow_construction"):
            optimizer = trainer.inference.offline_optimizer(updates)
            learning_rate = schedule_record(optimizer, updates)
            workflow = trainer.full_surface.build_workflow(task, optimizer=optimizer, **config)
            # Use public estimator.compile/fit so a multi-batch execution block
            # cannot silently pass the endpoint before a callback is invoked.
            workflow.estimator.compile(optimizer=optimizer, steps_per_execution=1)
            dataset = make_dataset(train, workflow.adapter, plan)
            from bayesflow.datasets import OfflineDataset
            validation_dataset = OfflineDataset(validation, args.batch_size, workflow.adapter, shuffle=False)
            guard = update_guard(plan)
            phases = training_callback(recorder)
        journal("training")
        phases.fit_started = recorder.now()
        with recorder.phase("training_fit", synchronize=lambda: synchronize_model(workflow.estimator)):
            fitted = workflow.estimator.fit(dataset=dataset, epochs=1, shuffle=False, verbose=0,
                                             validation_data=validation_dataset,
                                             callbacks=[guard, phases])
        if int(optimizer.iterations.numpy()) != updates:
            raise RuntimeError("actual optimizer iterations differ from the requested budget")
        history = {key: [float(v) for v in values] for key, values in fitted.history.items()}
        if any(not np.isfinite(v) for values in history.values() for v in values):
            raise FloatingPointError("nonfinite loss; preserve failed run")
        with recorder.phase("checkpoint_serialization", synchronize=lambda: synchronize_model(workflow.estimator)):
            workflow.estimator.save(args.out / "model.keras")
        with recorder.phase("checkpoint_reload"):
            restored = trainer.full_surface.build_workflow(task, **config)
            restored.load_estimator(args.out / "model.keras")
        with recorder.phase("posterior_sampling", synchronize=lambda: synchronize_model(restored.estimator)):
            quotes = splits["validation"]["observations"][:2]
            before = trainer.full_surface.sample(workflow, task, quotes, 16, encoder="vector", seed=args.seed + 11)
            after = trainer.full_surface.sample(restored, task, quotes, 16, encoder="vector", seed=args.seed + 11)
        if not np.array_equal(before, after):
            raise RuntimeError("checkpoint reload changed same-device, same-seed draws")
        np.savez_compressed(args.out / "training-exposure.npz", example_id=ids, presentations=guard.visits)
        work = {"optimizer_updates": guard.updates, "example_presentations": int(guard.visits.sum()),
                "unique_rows_seen": int(np.count_nonzero(guard.visits)), "available_training_rows": len(ids),
                "batch_size_counts": guard.batch_sizes, "ordered_row_indices_sha256": guard.order.hexdigest(),
                "exposure_file": "training-exposure.npz", "exposure_sha256": trainer.sha256(args.out / "training-exposure.npz"),
                "architecture": config, "optimizer_schedule": learning_rate,
                "simulation_calls": 0, "reserved_test_evaluated": False,
                "scope": "Actual optimizer-committed batches; dataset probes excluded from presentations."}
        report = {"format": "full_surface_training_v1", "task_fingerprint": manifest["task_fingerprint"],
                  "task_id": task["task_id"], "network": config, "run_contract": contract,
                  "checkpoint_sha256": trainer.sha256(args.out / "model.keras"), "history": history,
                  "selected_epoch": None, "selected_update": guard.updates,
                  "selection": "predeclared exact-update endpoint", "checkpoint_selection": "endpoint",
                  "same_seed_reload_exact": True, "training_surfaces": len(ids),
                  "validation_surfaces": len(validation["observations"]), "reserved_test_evaluated": False,
                  "scientific_validation": False, "performance": recorder.report(work=work),
                  "flow_matching_integrate_kwargs": dict(workflow.inference_network.integrate_kwargs)}
        (args.out / "task.json").write_text(json.dumps(task, indent=2, allow_nan=False) + "\n")
        (args.out / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        journal("complete", optimizer_updates=guard.updates, checkpoint_sha256=report["checkpoint_sha256"])
        return report
    except BaseException as error:
        journal("failed", error_type=type(error).__name__, error=str(error),
                performance=recorder.report(work={"scope": "incomplete attempt; no successful checkpoint claim"}))
        raise

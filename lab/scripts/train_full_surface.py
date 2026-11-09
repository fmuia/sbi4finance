"""Train a complete-surface BF2.1 estimator, preserving interrupted runs.

Use the locked ``core`` / ``core-gpu`` environment. This reads only training
and validation data; known curves are conditions only for the explicit known-curve
rough task. No test split or sparse layout is used.
One run binds its immutable bank snapshot, architecture and full schedule.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

_IMPORT_STARTED = (time.perf_counter(), time.process_time())

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sbi4finance.calibration import bank_collection as bank, full_surface, inference  # noqa: E402
from sbi4finance.runtime_sources import option_inference_runtime_hashes  # noqa: E402
from studies import train_calibration as recovery  # noqa: E402
from sbi4finance.calibration.performance import PerformanceRecorder, synchronize_model, training_callback  # noqa: E402

_IMPORT_ENDED = (time.perf_counter(), time.process_time())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_contract(args, manifest):
    """Reuse protected run storage, with core architecture/sampling bound explicitly."""
    config = full_surface.configuration(family=args.family, encoder=args.encoder,
                                        widths=tuple(int(w) for w in args.widths.split(",")),
                                        summary_dim=args.summary_dim, flow_steps=args.flow_steps,
                                        mean_steps=args.mean_steps)
    if min(args.epochs, args.batch_size, args.chunk_epochs) < 1 or args.seed < 0:
        raise ValueError("positive epochs/batch/chunk sizes and nonnegative seed required")
    selection = getattr(args, "checkpoint_selection", "validation_loss")
    if selection not in ("validation_loss", "final"):
        raise ValueError("checkpoint selection must be validation_loss or final")
    sources = [Path(__file__), Path(recovery.__file__), ROOT / "src/sbi4finance/runtime_sources.py", *sorted((ROOT / "src/sbi4finance/calibration").glob("*.py"))]
    return {"format": "full_surface_training_run_v1", "network": config,
            "dataset": {"manifest_sha256": sha256(args.bank / "manifest.json"),
                        "task_fingerprint": manifest["task_fingerprint"],
                        "collection_id": manifest.get("collection_id")},
            "runtime": recovery.runtime_versions(),
            "environment_sha256": {name: sha256(ROOT / name) for name in ("pyproject.toml", "pixi.lock")},
            "settings": {"epochs": args.epochs, "batch_size": args.batch_size,
                         "chunk_epochs": args.chunk_epochs, "seed": args.seed, "fresh_quote_noise": True,
                         "checkpoint_selection": selection},
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
            "shared_runtime_source_sha256": option_inference_runtime_hashes(manifest["configuration"]["task"])}


def run_training(args):
    if getattr(args, "updates", None) is not None:
        from scripts.train_full_surface_budget import run_budget_training
        return run_budget_training(args, sys.modules[__name__])
    started = time.perf_counter()
    performance = PerformanceRecorder()
    performance.phases["worker_module_imports"] = {
        "wall_seconds": _IMPORT_ENDED[0] - _IMPORT_STARTED[0],
        "process_cpu_seconds": _IMPORT_ENDED[1] - _IMPORT_STARTED[1], "calls": 1}
    manifest = json.loads((args.bank / "manifest.json").read_text())
    contract = run_contract(args, manifest)
    state = recovery.check_run_directory(args.out, contract, True)
    with performance.phase("data_loading"):
        manifest, splits = bank.load_bank(args.bank, splits=("train", "validation"), roles=bank.TRAINING_ROLES)
    if sha256(args.bank / "manifest.json") != contract["dataset"]["manifest_sha256"]:
        raise ValueError("bank manifest changed while loading; use its immutable snapshot")
    task = full_surface.validate_task(manifest["configuration"]["task"])
    config = contract["network"]
    with performance.phase("data_preparation"):
        train = full_surface.training_data(splits["train"], task, config["encoder"], clean=True)
        validation = full_surface.training_data(splits["validation"], task, config["encoder"])
    augment = full_surface.fresh_noise(args.seed, config["encoder"])

    with performance.phase("backend_imports"):
        import keras
        # BayesFlow imports create initializers which consume Python RNG seeds.
        # Finish those imports before resetting the training seed, so a cold CLI
        # and a process that already imported BayesFlow initialize identically.
        import bayesflow  # noqa: F401
    keras.utils.set_random_seed(args.seed)
    batches = int(np.ceil(len(train["observations"]) / args.batch_size))
    with performance.phase("workflow_construction"):
        workflow = full_surface.build_workflow(task, optimizer=inference.offline_optimizer(args.epochs * batches), **config)
    timing_callback = training_callback(performance)
    recovery.reserve_run_directory(args.out, contract, True)

    class BestCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.epoch = np.inf, 0

        def on_epoch_end(self, epoch, logs=None):
            value = float(logs["val_loss"])
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite validation loss; preserve run for inspection")
            if value < self.best:
                self.best, self.epoch = value, epoch + 1
                with performance.phase("validation_best_serialization"):
                    self.model.save(args.out / "model.keras")

    best = BestCheckpoint()
    history, resumed_at, done = {"loss": [], "val_loss": []}, [], 0
    if state is not None:
        with performance.phase("resume_reload"):
            workflow.load_estimator(args.out / state["checkpoint_files"]["last"])
            synchronize_model(workflow.estimator)
        workflow.optimizer = workflow.estimator.optimizer
        workflow._needs_compile = False
        if int(workflow.optimizer.iterations.numpy()) != state["epochs_done"] * batches:
            raise RuntimeError("restored optimizer step count differs from committed training state")
        augment.rng.bit_generator.state = state["noise_rng"]
        history, done = state["history"], state["epochs_done"]
        resumed_at = state["resumed_at_epochs"] + [done]
        best.best, best.epoch = state["best_val_loss"], state["best_epoch"]
        shutil.copyfile(args.out / state["checkpoint_files"]["best"], args.out / "model.keras")

    initial_done = done
    initial_updates = int(workflow.optimizer.iterations.numpy())
    ready = time.perf_counter()
    while done < args.epochs:
        stop = min(done + args.chunk_epochs, args.epochs)
        # Outside fit, Keras has reattached the model/optimizer variables. Do
        # not synchronize on every batch and distort the workload being timed.
        timing_callback.fit_started = performance.now()
        with performance.phase("fit_including_validation_and_callbacks"):
            chunk = workflow.fit_offline(data=train, validation_data=validation, epochs=stop, initial_epoch=done,
                                         batch_size=args.batch_size, augmentations=augment, keep_optimizer=True,
                                         callbacks=[timing_callback, best], verbose=0).history
            synchronize_model(workflow.estimator)
        if (len(chunk["loss"]) != stop - done or len(chunk["val_loss"]) != stop - done
                or int(workflow.optimizer.iterations.numpy()) != stop * batches):
            raise RuntimeError("fit did not complete the declared epochs/optimizer updates; preserve run for inspection")
        for key in history:
            history[key] += [float(value) for value in chunk[key]]
        done = stop
        state = {"epochs_done": done, "best_val_loss": best.best, "best_epoch": best.epoch, "history": history,
                 "resumed_at_epochs": resumed_at, "noise_rng": augment.rng.bit_generator.state,
                 "planned": {"epochs": args.epochs, "batch_size": args.batch_size,
                             "rows": len(train["observations"])}, "run_contract": contract}
        with performance.phase("serialization_and_resume_commit"):
            state = recovery.commit_resume_state(args.out, state, workflow.estimator.save)
        print(f"epoch {done}/{args.epochs}: best validation loss {best.best:.4f}", flush=True)

    actual_updates = int(workflow.optimizer.iterations.numpy())
    policy = contract["settings"]["checkpoint_selection"]
    # Protected best/last files remain unchanged. Publication follows the
    # policy bound before training, including when a completed run is restored.
    if policy == "final":
        shutil.copyfile(args.out / state["checkpoint_files"]["last"], args.out / "model.keras")
    selected_epoch = args.epochs if policy == "final" else best.epoch
    with performance.phase("selected_checkpoint_reload"):
        workflow.load_estimator(args.out / "model.keras")
        synchronize_model(workflow.estimator)
    with performance.phase("comparison_workflow_construction_and_reload"):
        restored = full_surface.build_workflow(task, **config)
        restored.load_estimator(args.out / "model.keras")
        synchronize_model(restored.estimator)
    quotes = splits["validation"]["observations"][:2]
    curves = splits["validation"].get("context_values")
    curves = None if curves is None else curves[:2]
    with performance.phase("posterior_sampling_reload_check"):
        before = full_surface.sample(workflow, task, quotes, 16, encoder=config["encoder"], seed=args.seed + 11,
                                     context_values=curves)
        after = full_surface.sample(restored, task, quotes, 16, encoder=config["encoder"], seed=args.seed + 11,
                                    context_values=curves)
    if not np.array_equal(before, after):
        raise RuntimeError("checkpoint reload changed same-device, same-seed draws")
    # Epoch mode consumes every row exactly once per completed epoch. Count
    # committed optimizer work before reloading a potentially earlier best
    # checkpoint; augmentation probes/prefetch are not training presentations.
    rows = len(train["observations"])
    identifiers = splits["train"].get("example_id")
    identity = (None if identifiers is None else hashlib.sha256(
        json.dumps(np.asarray(identifiers).tolist(), separators=(",", ":")).encode()).hexdigest())
    work = {"scope": "completed optimizer work across this run; invocation counters exclude recovered epochs",
            "optimizer_updates": actual_updates, "optimizer_updates_this_invocation": actual_updates - initial_updates,
            "selected_checkpoint_optimizer_updates": int(workflow.estimator.optimizer.iterations.numpy()),
            "completed_epochs": done, "completed_epochs_this_invocation": done - initial_done,
            "training_presentations": done * rows,
            "training_presentations_this_invocation": (done - initial_done) * rows,
            "unique_training_rows": rows if done else 0,
            "unique_training_rows_this_invocation": rows if done > initial_done else 0,
            "training_example_ids_sha256": identity,
            "row_count_basis": "all rows in each verified complete offline epoch; excludes adapter/build probes",
            "batch_size": args.batch_size, "batches_per_epoch": batches,
            "last_batch_size": rows - (batches - 1) * args.batch_size,
            "full_batches": done * (rows // args.batch_size),
            "partial_batches": done if rows % args.batch_size else 0,
            "architecture": config,
            "optimizer": "AdamW(weight_decay=0.005, clipnorm=1.5)",
            "learning_rate": {"rule": "linear warmup then cosine decay", "peak": 0.0005,
                              "initial": 0.00005, "warmup_updates": int(.05 * args.epochs * batches),
                              "total_updates": args.epochs * batches},
            "online_simulated_rows": 0, "pricing_seconds": 0.0,
            "reload_check": {"observations": len(quotes), "draws_per_observation": 16, "sample_calls": 2}}
    report = {"format": "full_surface_training_v1", "task_fingerprint": manifest["task_fingerprint"],
              "task_id": task["task_id"], "network": config, "run_contract": contract,
              "checkpoint_sha256": sha256(args.out / "model.keras"), "history": history,
              "selected_epoch": selected_epoch,
              "selection": ("predeclared final epoch" if policy == "final"
                            else "minimum validation loss within this family"),
              "checkpoint_selection": policy,
              "validation_best_epoch": best.epoch, "validation_best_loss": best.best,
              "validation_best_checkpoint_sha256": sha256(args.out / state["checkpoint_files"]["best"]),
              "final_checkpoint_sha256": sha256(args.out / state["checkpoint_files"]["last"]),
              "flow_matching_integrate_kwargs": (dict(workflow.inference_network.integrate_kwargs)
                                                  if config["family"] == "flow_matching" else None),
              "resumed_at_epochs": resumed_at, "same_seed_reload_exact": True,
              "training_surfaces": len(train["observations"]), "validation_surfaces": len(validation["observations"]),
              "reserved_test_evaluated": False,
              "conditioning": ("complete quotes/design/noise and supplied known curve levels"
                               if full_surface.uses_known_curve(task)
                               else "complete quotes/design/noise only; no supplied curve"),
              "scientific_validation": False, "timing_seconds": {"setup": ready - started,
                 "training_and_reload": time.perf_counter() - ready}}
    report["performance"] = performance.report(work=work)
    report["performance"]["imports_scope"] = "worker_module_imports occurred at module load, outside invocation total; backend_imports may be warm"
    report["performance"]["pricing_scope"] = "no online simulation/pricing in offline training; fresh quote noise only"
    report["performance"]["cache_state"] = "existing process and compiler caches are not cleared; first-fit phases include any required build/compilation"
    (args.out / "task.json").write_text(json.dumps(task, indent=2, allow_nan=False) + "\n")
    (args.out / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=full_surface.FAMILIES, default="spline")
    parser.add_argument("--encoder", choices=full_surface.ENCODERS, default="vector")
    parser.add_argument("--widths", default="128,128")
    parser.add_argument("--summary-dim", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=None,
                        help="Explicit legacy Euler step count; omit for BayesFlow's native FlowMatching solver.")
    parser.add_argument("--mean-steps", type=int, default=4)
    parser.add_argument("--updates", type=int, default=None,
                        help="Fresh Heston fit with exactly this many optimizer updates; requires --checkpoint-selection final. Epoch/chunk settings are unused.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-selection", choices=("validation_loss", "final"), default="validation_loss",
                        help="Predeclare the published model; validation-best and final recovery files are both retained.")
    args = parser.parse_args()
    with recovery.training_lock(args.out):
        report = run_training(args)
    print(json.dumps({"selected_epoch": report["selected_epoch"], "selected_update": report.get("selected_update"), "reload_exact": True,
                      "scientific_validation": False}))


if __name__ == "__main__":
    main()

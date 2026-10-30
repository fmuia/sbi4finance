"""Train a masked arbitrary-history posterior; never read the reserved test split."""

import argparse
import gc
import hashlib
from importlib import metadata
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import bank, inference  # noqa: E402


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"raw": raw, "raw_unit": "bytes" if sys.platform == "darwin" else "KiB",
            "bytes": int(raw if sys.platform == "darwin" else raw * 1024)}


def memory_projection(task, training_count, validation_count, validation_cases, batch_size):
    steps, targets = task["n_observations"], len(task["parameter_names"])
    row_bytes = steps * (2 * 4 + 1) + (targets + 2) * 4
    training_bytes = training_count * row_bytes
    validation_bytes = min(validation_count, validation_cases) * len(task["prefix_lengths"]) * row_bytes
    raw_bank_bytes = training_count * (steps * 8 + targets * 8)
    # Cover bank concatenation, neural input conversion, multiple host copies,
    # per-batch workspaces and a fixed import/runtime allowance. GPU activation
    # memory is checked separately by production-shape GPU readiness.
    estimate = max(raw_bank_bytes * 2, raw_bank_bytes + training_bytes,
                   4 * (training_bytes + validation_bytes)) + 1024**3
    estimate += batch_size * steps * 2 * 4 * 16
    return {"training_input_bytes": training_bytes, "validation_input_bytes": validation_bytes,
            "raw_training_arrays_bytes": raw_bank_bytes, "estimated_peak_host_bytes": estimate,
            "host_guard_bytes": 12 * 1024**3,
            "projected_100000_training_input_bytes": 100_000 * row_bytes,
            "estimate_note": "conservative host estimate; full-shape GPU smoke checks device allocation"}


def run_training(args):
    started = time.monotonic()
    for name in ("epochs", "batch_size", "validation_cases"):
        inference.integer(getattr(args, name), name)
    inference.integer(args.seed, "seed", minimum=0)
    if args.max_seconds <= 0:
        raise ValueError("max-seconds must be positive")
    manifest = bank.load_manifest(args.bank)
    task = inference.validate_task(manifest["task"])
    memory = memory_projection(task, manifest["split_counts"]["train"],
                               manifest["split_counts"]["validation"],
                               args.validation_cases, args.batch_size)
    if memory["estimated_peak_host_bytes"] > memory["host_guard_bytes"]:
        raise MemoryError("projected path-training host memory exceeds 12 GiB guard; use a streamed bank loader")
    train_split = bank.load_split(args.bank, "train")
    training_count = len(train_split["observations"])
    if training_count == 0:
        raise ValueError("training split must contain trajectories")
    training_identity = {key + "_sha256": hashlib.sha256(train_split[key].tobytes()).hexdigest()
                         for key in ("parameters", "trajectory_id")}
    train = inference.training_data(train_split, task)
    del train_split
    gc.collect()
    validation_split = bank.load_split(args.bank, "validation")
    count = min(args.validation_cases, len(validation_split["observations"]))
    if count == 0:
        raise ValueError("validation split must contain trajectories")
    validation_split = {k: v[:count].copy() if isinstance(v, np.ndarray) else v
                        for k, v in validation_split.items()}
    panels = [inference.training_data(validation_split, task,
                                     lengths=np.full(count, n, dtype=np.int64))
              for n in task["prefix_lengths"]]
    validation = {key: np.concatenate([panel[key] for panel in panels]) for key in panels[0]}
    del panels
    memory["actual_training_input_bytes"] = sum(value.nbytes for value in train.values())
    memory["actual_validation_input_bytes"] = sum(value.nbytes for value in validation.values())
    gc.collect()
    network = dict(inference.DEFAULT_NETWORK)
    for name in ("hidden_dim", "summary_dim", "depth", "bins"):
        value = getattr(args, name, None)
        if value is not None:
            network[name] = inference.integer(value, name)
    if getattr(args, "width", None) is not None:
        network["widths"] = [inference.integer(args.width, "width")] * 2
    import bayesflow  # noqa: F401
    import keras
    keras.utils.set_random_seed(args.seed)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.5)
    workflow = inference.build_workflow(task, network=network, optimizer=optimizer)
    augment = inference.random_prefixes(task, args.seed + 701)
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "task.json", task)
    versions = {}
    for package in ("bayesflow", "keras", "jax", "numpy", "scipy"):
        versions[package] = metadata.version(package)
    report = {"format": inference.FORMAT, "task_sha256": inference.fingerprint(task),
              "bank_manifest_sha256": inference.sha256(args.bank / "manifest.json"),
              "network": network, "runtime": versions,
              "normalization": "fixed_prior_units_no_fitted_standardizer_v1",
              "history_training": "uniform_integer_per_trajectory_1_to_max_with_mask",
              "supported_history_lengths": [1, task["n_observations"]],
              "validation_landmarks": task["prefix_lengths"],
              "validation_cases_per_length": count,
              "checkpoint_selection": "minimum equally weighted landmark validation loss",
              "training_trajectories": training_count, "training_identity": training_identity,
              "host_memory": memory, "peak_rss": peak_rss(),
              "planned_epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
              "max_seconds": args.max_seconds, "epochs_completed": 0,
              "status": "initializing", "reserved_test_evaluated": False,
              "scientific_validation": False, "resume_supported": False,
              "source_sha256": {str(p.relative_to(ROOT)): inference.sha256(p)
                                for p in (Path(__file__), Path(inference.__file__), Path(bank.__file__))},
              "environment_sha256": {name: inference.sha256(ROOT / name)
                                     for name in ("pyproject.toml", "pixi.lock")},
              "history": {"loss": [], "val_loss": []}}
    atomic_json(args.out / "training.json", report)

    class Checkpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = np.inf

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss, val_loss = float(logs["loss"]), float(logs["val_loss"])
            if not np.isfinite([loss, val_loss]).all():
                raise FloatingPointError("nonfinite path posterior loss")
            report["history"]["loss"].append(loss)
            report["history"]["val_loss"].append(val_loss)
            report["epochs_completed"] = epoch + 1
            self.model.save(args.out / "final.keras")
            if val_loss < self.best:
                self.best = val_loss
                self.model.save(args.out / "model.keras")
                report["selected_epoch"] = epoch + 1
                report["selected_validation_loss"] = val_loss
            report["model_sha256"] = inference.sha256(args.out / "model.keras")
            report["final_sha256"] = inference.sha256(args.out / "final.keras")
            report["optimizer_steps"] = int(self.model.optimizer.iterations.numpy())
            report["elapsed_seconds"] = time.monotonic() - started
            report["peak_rss"] = peak_rss()
            report["prefix_generation_histogram"] = augment.exposure.tolist()
            report["prefix_histogram_note"] = "includes dataset build/probe calls; optimizer_steps records actual updates"
            report["status"] = "running"
            if report["elapsed_seconds"] >= args.max_seconds:
                self.model.stop_training = True
                report["status"] = "soft_wall_stop"
            atomic_json(args.out / "training.json", report)
            print(f"epoch {epoch + 1}/{args.epochs}: loss={loss:.5g} val={val_loss:.5g} "
                  f"elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    workflow.fit_offline(data=train, validation_data=validation, epochs=args.epochs,
                         batch_size=args.batch_size, augmentations=augment,
                         keep_optimizer=True, callbacks=[Checkpoint()], verbose=0)
    if report["epochs_completed"] == args.epochs:
        report["status"] = "completed"
    # Reload the selected estimator twice; identical seeded draws are a software
    # portability check, not posterior validation or exact training continuation.
    workflow.load_estimator(args.out / "model.keras")
    restored = inference.build_workflow(task, network=network)
    restored.load_estimator(args.out / "model.keras")
    observed = validation_split["observations"][:2]
    arbitrary = min(17, task["n_observations"])
    a = inference.sample(workflow, task, observed[:, :arbitrary], 8, seed=args.seed + 19)
    b = inference.sample(restored, task, observed[:, :arbitrary], 8, seed=args.seed + 19)
    if not np.array_equal(a, b):
        raise RuntimeError("same-device same-seed checkpoint reload changed posterior draws")
    report["same_seed_reload_exact"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_rss"] = peak_rss()
    atomic_json(args.out / "training.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=2700)
    parser.add_argument("--validation-cases", type=int, default=256)
    for name in ("hidden-dim", "summary-dim", "depth", "bins", "width"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args()
    result = run_training(args)
    print(json.dumps({k: result[k] for k in ("status", "epochs_completed", "selected_epoch", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()

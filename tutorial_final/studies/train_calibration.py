"""Train the amortized calibration posterior on a calibration bank (studies S2 and S3).

    python tutorial_final/studies/train_calibration.py --bank BANK --out NEW_DIR \
        [--epochs 40 --batch-size 256 --augmentations 4 --seed 0] [--fresh-noise]

Training quotes are the clean prices of each training surface plus quote
errors of the declared noise model: --augmentations fixed draws per surface,
or with --fresh-noise a new draw for every batch, so no noisy surface is seen
twice. Validation keeps the bank's stored noise and only selects the epoch. The test split
is not read. The run directory receives model.keras, task.json and
training.json, and the reload is checked with same-seed draws.

With --thin-layout LAYOUT.json (study S2-T, sbi4finance.calibration.thin) the network reads only the layout's
quotes, with the layout's noise times a noise scale drawn for every surface of every batch, and infers the
forward-variance levels with the parameters; validation then uses one fixed draw of the scale and the noise.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sbi4finance.calibration import bank_collection as bank  # noqa: E402
from sbi4finance.calibration import inference, thin  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions():
    """Inspect installed distributions without importing training backends."""
    return {"python": platform.python_version(), "keras_backend_env": os.environ.get("KERAS_BACKEND"),
            **{name: package_metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}}


@contextmanager
def training_lock(path):
    """One POSIX training writer per output directory, including same-run recovery.

    Keep the sibling lock file after release: deleting it could let another
    process lock a replacement inode while a worker still holds the original.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.training.lock"
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another trainer owns --out {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_contract(args, manifest):
    """Identity of one fixed dataset and training schedule; independent of its directory name."""
    widths = tuple(int(value) for value in args.widths.split(","))
    if (args.epochs < 1 or args.batch_size < 1 or args.augmentations < 1 or args.chunk_epochs < 0
            or args.seed < 0 or not widths or any(width < 1 for width in widths)
            or not math.isfinite(args.time_alpha) or args.time_alpha <= -1):
        raise ValueError("positive training sizes/widths, a nonnegative seed/chunk size and time-alpha > -1 required")
    return {
        "format": "calibration_training_run_v1",
        "dataset": {"manifest_sha256": sha256(args.bank / "manifest.json"),
                    "task_fingerprint": manifest["task_fingerprint"],
                    "collection_id": manifest.get("collection_id")},
        "layout_sha256": None if args.thin_layout is None else sha256(args.thin_layout),
        "runtime": runtime_versions(),
        "environment_sha256": {name: sha256(Path(__file__).resolve().parents[2] / name)
                               for name in ("pyproject.toml", "pixi.lock")},
        "settings": {"epochs": args.epochs, "batch_size": args.batch_size, "augmentations": args.augmentations,
                     "fresh_noise": args.fresh_noise, "seed": args.seed, "widths": list(widths),
                     "time_alpha": args.time_alpha, "chunk_epochs": args.chunk_epochs},
        "source_sha256": {"train_calibration.py": sha256(__file__), "inference.py": sha256(inference.__file__),
                          "thin.py": sha256(thin.__file__), "bank_collection.py": sha256(bank.__file__),
                          "gbm_bank.py": sha256(Path(__file__).resolve().parents[2] / "src/sbi4finance/gbm_bank.py"),
                          **{f"calibration/{path.name}": sha256(path) for path in sorted(
                              (Path(__file__).resolve().parents[2] / "src/sbi4finance/calibration").glob("*.py"))}},
    }


def check_run_directory(path, contract, resumable):
    """Read-only preflight. An incomplete first chunk is preserved and requires a new output directory."""
    path = Path(path)
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError("training output is not a directory: use a new --out")
    if not any(path.iterdir()):
        return None
    required = ("run-contract.json", "resume.json")
    if not resumable or any(not (path / name).is_file() for name in required):
        raise ValueError("nonempty training output has no complete resumable state: use a new --out")
    saved_contract = json.loads((path / "run-contract.json").read_text())
    state = json.loads((path / "resume.json").read_text())
    if saved_contract != contract or state.get("run_contract") != contract:
        raise ValueError("dataset, layout, source or training settings changed: use a new --out")
    done = state.get("epochs_done")
    history = state.get("history")
    best_epoch = state.get("best_epoch")
    planned = state.get("planned", {})
    if (type(done) is not int or not 0 < done <= contract["settings"]["epochs"]
            or type(best_epoch) is not int or not 1 <= best_epoch <= done
            or not isinstance(history, dict) or not {"loss", "val_loss"} <= set(history)
            or any(not isinstance(values, list) or len(values) != done for values in history.values())
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for values in history.values() for value in values)
            or not isinstance(state.get("resumed_at_epochs"), list)
            or any(type(epoch) is not int or not 0 < epoch <= done for epoch in state["resumed_at_epochs"])
            or type(state.get("best_val_loss")) not in (int, float)
            or not math.isfinite(state["best_val_loss"])
            or not isinstance(planned, dict)
            or planned.get("epochs") != contract["settings"]["epochs"]
            or planned.get("batch_size") != contract["settings"]["batch_size"]
            or type(planned.get("rows")) is not int or planned["rows"] < 1):
        raise ValueError("invalid training resume metadata: use a new --out")
    minimum = min(history["val_loss"])
    if state["best_val_loss"] != minimum or best_epoch != history["val_loss"].index(minimum) + 1:
        raise ValueError("invalid training resume best epoch/loss: use a new --out")
    files = state.get("checkpoint_files", {})
    checkpoints = state.get("checkpoint_sha256", {})
    if not isinstance(files, dict) or set(files) != {"last", "best"}:
        raise ValueError("missing committed training checkpoints: use a new --out")
    for label, relative in files.items():
        if not isinstance(relative, str):
            raise ValueError("invalid committed checkpoint path: use a new --out")
        parts = Path(relative).parts
        if (len(parts) != 3 or parts[0] != ".resume" or not parts[1].startswith(f"epoch-{done:06d}-")
                or parts[2] != f"{label}.keras" or not (path / relative).resolve().is_relative_to(path.resolve())):
            raise ValueError("invalid committed checkpoint path: use a new --out")
    if (Path(files["last"]).parent != Path(files["best"]).parent
            or not isinstance(checkpoints, dict) or set(checkpoints) != set(files.values())
            or any(not (path / name).is_file() or sha256(path / name) != checkpoints[name] for name in checkpoints)):
        raise ValueError("training checkpoint and resume state differ: use a new --out")
    return state


def reserve_run_directory(path, contract, resumable):
    """Reserve an empty directory without deleting or replacing any pre-existing data."""
    if check_run_directory(path, contract, resumable) is not None:
        return
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "run-contract.json").open("x") as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)
        stream.write("\n")


def commit_resume_state(path, state, save_last):
    """Publish a resume pointer only after both immutable checkpoint copies are complete.

    A failed attempt leaves the previous pointer and checkpoint copies untouched.
    Unique attempt directories also permit retrying an interrupted chunk safely.
    """
    path = Path(path)
    checkpoint_root = path / ".resume"
    checkpoint_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"epoch-{state['epochs_done']:06d}-", dir=checkpoint_root))
    save_last(directory / "last.keras")
    shutil.copyfile(path / "model.keras", directory / "best.keras")
    files = {label: (directory / f"{label}.keras").relative_to(path).as_posix() for label in ("last", "best")}
    committed = {**state, "checkpoint_files": files,
                 "checkpoint_sha256": {name: sha256(path / name) for name in files.values()}}
    temporary = path / "resume.json.tmp"
    temporary.write_text(json.dumps(committed, allow_nan=False) + "\n")
    temporary.replace(path / "resume.json")
    return committed


def train_in_chunks(args, task, train, validation, augment, widths, resume_path, builder, contract):
    """Chunks of --chunk-epochs on one learning-rate schedule; immutable checkpoint copies and resume.json.

    The best epoch by validation loss is saved as model.keras across all chunks.
    On a rerun, the approximator, its optimizer state (step count and moments)
    and the noise generator's state are restored from the last chunk.
    This is same-run recovery; backend shuffle/time randomness need not reproduce
    a bitwise-identical uninterrupted training trajectory.
    """
    import keras

    class BestCheckpoint(keras.callbacks.Callback):
        def __init__(self, path, best, epoch):
            super().__init__()
            self.path, self.best, self.epoch = path, best, epoch

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get("val_loss")
            if value is not None and value < self.best:
                self.best, self.epoch = float(value), epoch + 1
                self.model.save(self.path)

    rows = train["clean_scaled"] if "clean_scaled" in train else train["observations"]  # thin data: clean_scaled
    batches = int(np.ceil(len(rows) / args.batch_size))
    workflow = builder(task, subnet_widths=widths, time_power_law_alpha=args.time_alpha,
                       optimizer=inference.offline_optimizer(args.epochs * batches))
    state = check_run_directory(args.out, contract, True) if resume_path.exists() else None
    history = {"loss": [], "val_loss": []}
    resumed_at, done = [], 0
    best = BestCheckpoint(args.out / "model.keras", np.inf, 0)
    if state:
        planned = {"epochs": args.epochs, "batch_size": args.batch_size, "rows": len(rows)}
        if state.get("run_contract") != contract or state.get("planned") != planned:
            # The learning-rate schedule is fixed by the first run's epochs and batches; resuming with others would
            # continue the old schedule (at a learning rate of 0 past its end). Train into a new --out instead.
            raise ValueError(f"this run was started with {state.get('planned')}, not {planned}: use a new --out")
        workflow.approximator = keras.saving.load_model(args.out / state["checkpoint_files"]["last"])
        workflow.optimizer = workflow.approximator.optimizer
        workflow._needs_compile = False
        if int(workflow.optimizer.iterations.numpy()) != state["epochs_done"] * batches:
            raise RuntimeError("restored optimizer step count does not match the saved state")
        if augment is not None:
            augment.rng.bit_generator.state = state["noise_rng"]
        history, resumed_at, done = state["history"], state["resumed_at_epochs"] + [state["epochs_done"]], state["epochs_done"]
        best.best, best.epoch = state["best_val_loss"], state["best_epoch"]
        # A later interrupted chunk may have changed the working best file.
        shutil.copyfile(args.out / state["checkpoint_files"]["best"], args.out / "model.keras")
    while done < args.epochs:
        stop = min(done + args.chunk_epochs, args.epochs)
        chunk = workflow.fit_offline(data=train, validation_data=validation, epochs=stop, initial_epoch=done,
                                     batch_size=args.batch_size, augmentations=augment, keep_optimizer=True,
                                     callbacks=[best], verbose=0).history
        for key in history:
            history[key] += [float(v) for v in chunk[key]]
        done = stop
        record = {"epochs_done": done, "best_val_loss": best.best, "best_epoch": best.epoch, "history": history,
                  "resumed_at_epochs": resumed_at,
                  "planned": {"epochs": args.epochs, "batch_size": args.batch_size, "rows": len(rows)},
                  "noise_rng": augment.rng.bit_generator.state if augment is not None else None,
                  "run_contract": contract}
        commit_resume_state(args.out, record, workflow.approximator.save)
        print(f"epoch {done}/{args.epochs}: best validation loss {best.best:.4f} at epoch {best.epoch}", flush=True)
    workflow.load_approximator(args.out / "model.keras")
    return workflow, history, resumed_at


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--augmentations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--widths", default="256,256,256,256")
    parser.add_argument("--fresh-noise", action="store_true",
                        help="draw a new quote error for every batch instead of --augmentations fixed copies")
    parser.add_argument("--time-alpha", type=float, default=-0.5,
                        help="flow-matching training times t = u^(1 + alpha). BayesFlow's default, 0.5, favours the noise "
                             "end; -0.5 (density 2t) favours the data end, where a sharp posterior's location is set, "
                             "and took Heston's v0 from 28%% to 82%% agreement with the exact posterior")
    parser.add_argument("--thin-layout", type=Path, default=None,
                        help="train the thin-chain network (S2-T) on this layout instead of the liquid network")
    parser.add_argument("--chunk-epochs", type=int, default=0,
                        help="train in chunks of this many epochs, saving a resumable state after each; rerunning "
                             "the same command after an interruption continues from the last chunk")
    args = parser.parse_args()
    with training_lock(args.out):
        run_training(args)


def run_training(args):
    started = time.perf_counter()
    metadata = json.loads((args.bank / "manifest.json").read_text())
    contract = run_contract(args, metadata)
    resumable = args.chunk_epochs > 0
    check_run_directory(args.out, contract, resumable)
    manifest, splits = bank.load_bank(args.bank, splits=("train", "validation"), roles=bank.TRAINING_ROLES)
    if (sha256(args.bank / "manifest.json") != contract["dataset"]["manifest_sha256"]
            or manifest["task_fingerprint"] != contract["dataset"]["task_fingerprint"]):
        raise ValueError("dataset changed while loading: use its immutable snapshot")
    task = manifest["configuration"]["task"]
    layout = None if args.thin_layout is None else thin.load_layout(args.thin_layout, task)
    if layout is not None:
        if layout["task_fingerprint"] != manifest["task_fingerprint"]:
            raise ValueError("the layout was built for another task")
        train = thin.thin_clean_data(splits["train"], task, layout)
        augment = thin.thin_noise(args.seed)
        validation = thin.thin_validation_data(splits["validation"], task, layout, args.seed)
        builder = thin.build_thin_workflow
    elif args.fresh_noise:
        train = inference.clean_training_data(splits["train"], task)
        augment = inference.fresh_noise(args.seed)
    else:
        train = inference.augmented_training_data(splits["train"], task, augmentations=args.augmentations,
                                                  seed=args.seed)
        augment = None
    if layout is None:
        validation = inference.training_data(splits["validation"], task)
        builder = inference.build_workflow
    resume_path = args.out / "resume.json"
    reserve_run_directory(args.out, contract, resumable)

    import bayesflow as bf
    import jax
    import keras

    keras.utils.set_random_seed(args.seed)
    widths = tuple(int(w) for w in args.widths.split(","))
    ready = time.perf_counter()
    resumed_at = []
    if not resumable:
        workflow = builder(task, args.out, subnet_widths=widths, time_power_law_alpha=args.time_alpha)
        history = workflow.fit_offline(data=train, validation_data=validation, epochs=args.epochs,
                                       batch_size=args.batch_size, augmentations=augment, verbose=0).history
        workflow.load_approximator()
    else:
        workflow, history, resumed_at = train_in_chunks(args, task, train, validation, augment, widths, resume_path,
                                                        builder, contract)
    trained = time.perf_counter()
    restored = builder(task, subnet_widths=widths)
    restored.load_approximator(args.out / "model.keras")
    if layout is None:
        observations = splits["validation"]["observations"][:2]
        context = splits["validation"]["context_values"][:2]
        before = inference.sample(workflow, task, observations, context, 128, seed=args.seed + 11)
        after = inference.sample(restored, task, observations, context, 128, seed=args.seed + 11)
    else:
        idx, _ = thin.layout_arrays(layout)
        quotes = splits["validation"]["reference_prices"][:2][:, idx]
        before = thin.sample_thin(workflow, task, layout, quotes, 128, seed=args.seed + 11)
        after = thin.sample_thin(restored, task, layout, quotes, 128, seed=args.seed + 11)
    if not np.array_equal(before, after):
        raise RuntimeError("checkpoint reload changed same-seed draws")
    losses = [float(v) for v in history["val_loss"]]
    report = {
        "study": "train_calibration", "evidence_scope": "S", "measure": "Q",
        "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
        "bank_manifest_sha256": sha256(args.bank / "manifest.json"),
        "checkpoint_sha256": sha256(args.out / "model.keras"),
        "run_contract": contract,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "quote_noise": ("the layout's noise times a log-uniform scale, fresh per batch" if layout is not None
                                     else "fresh standard-normal error per batch" if args.fresh_noise
                                     else f"{args.augmentations} fixed draws per surface"),
                     "augmentations": None if args.fresh_noise else args.augmentations,
                     "seed": args.seed, "subnet_widths": list(widths), "sampler": inference.SAMPLER_DESCRIPTION,
                     "selected_epoch": int(np.argmin(losses)) + 1, "selection": "minimum validation loss",
                     "optimizer": "AdamW, weight decay 5e-3, clipnorm 1.5; 5% warm-up to 5e-4, cosine decay to 0 "
                                  "over all epochs (BayesFlow's offline default)",
                     "chunk_epochs": args.chunk_epochs or None, "resumed_at_epochs": resumed_at,
                     "time_power_law_alpha": args.time_alpha,
                     "thin_layout": None if layout is None else {"path": str(args.thin_layout),
                                                                 "sha256": sha256(args.thin_layout),
                                                                 "ticker": layout["ticker"],
                                                                 "contracts": len(layout["contract_indices"]),
                                                                 "scale_bounds": list(thin.SCALE_BOUNDS)}},
        "training_surfaces": len(splits["train"]["parameters"]), "training_rows": len(train["clean_scaled" if layout is not None else "observations"]),
        "validation_surfaces": len(validation["observations"]), "reserved_test_evaluated": False,
        "same_seed_reload_exact": True,
        "timing_seconds": {"load_and_setup": ready - started, "training_including_compile": trained - ready,
                           "reload_check": time.perf_counter() - trained},
        "cpu_count": __import__("os").cpu_count(),
        "history": {key: [float(v) for v in values] for key, values in history.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "bayesflow": bf.__version__,
                     "keras": keras.__version__, "jax": jax.__version__},
        "source_sha256": {"train_calibration.py": sha256(__file__), "inference.py": sha256(inference.__file__),
                          "thin.py": sha256(thin.__file__)},
    }
    (args.out / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "task.json").write_text(json.dumps(task, indent=1) + "\n")
    print(json.dumps({"training_minutes": (trained - ready) / 60, "selected_epoch": report["training"]["selected_epoch"],
                      "best_val_loss": min(losses), "reload_exact": True}, indent=1))


if __name__ == "__main__":
    main()

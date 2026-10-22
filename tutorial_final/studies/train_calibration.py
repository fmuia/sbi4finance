"""Train the amortized calibration posterior on a calibration bank (studies S2 and S3).

    python tutorial_final/studies/train_calibration.py --bank BANK --out NEW_DIR \
        [--epochs 40 --batch-size 256 --augmentations 4 --seed 0] [--fresh-noise]

Training quotes are the clean prices of each training surface plus quote
errors of the declared noise model: --augmentations fixed draws per surface,
or with --fresh-noise a new draw for every batch, so no noisy surface is seen
twice. Validation keeps the bank's stored noise and only selects the epoch. The test split
is not read. The run directory receives model.keras, task.json and
training.json, and the reload is checked with same-seed draws.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sbi4finance.calibration import bank, inference  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_in_chunks(args, task, train, validation, augment, widths, resume_path):
    """Chunks of --chunk-epochs on one learning-rate schedule; after each, last.keras and resume.json.

    The best epoch by validation loss is saved as model.keras across all chunks.
    On a rerun, the approximator, its optimizer state (step count and moments)
    and the noise generator's state are restored from the last chunk.
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

    batches = int(np.ceil(len(train["observations"]) / args.batch_size))
    workflow = inference.build_workflow(task, subnet_widths=widths, time_power_law_alpha=args.time_alpha,
                                        optimizer=inference.offline_optimizer(args.epochs * batches))
    state = json.loads(resume_path.read_text()) if resume_path.exists() else None
    history = {"loss": [], "val_loss": []}
    resumed_at, done = [], 0
    best = BestCheckpoint(args.out / "model.keras", np.inf, 0)
    if state:
        workflow.approximator = keras.saving.load_model(args.out / "last.keras")
        workflow.optimizer = workflow.approximator.optimizer
        workflow._needs_compile = False
        if int(workflow.optimizer.iterations.numpy()) != state["epochs_done"] * batches:
            raise RuntimeError("restored optimizer step count does not match the saved state")
        if augment is not None:
            augment.rng.bit_generator.state = state["noise_rng"]
        history, resumed_at, done = state["history"], state["resumed_at_epochs"] + [state["epochs_done"]], state["epochs_done"]
        best.best, best.epoch = state["best_val_loss"], state["best_epoch"]
    while done < args.epochs:
        stop = min(done + args.chunk_epochs, args.epochs)
        chunk = workflow.fit_offline(data=train, validation_data=validation, epochs=stop, initial_epoch=done,
                                     batch_size=args.batch_size, augmentations=augment, keep_optimizer=True,
                                     callbacks=[best], verbose=0).history
        for key in history:
            history[key] += [float(v) for v in chunk[key]]
        done = stop
        workflow.approximator.save(args.out / "last.keras")
        record = {"epochs_done": done, "best_val_loss": best.best, "best_epoch": best.epoch, "history": history,
                  "resumed_at_epochs": resumed_at,
                  "noise_rng": augment.rng.bit_generator.state if augment is not None else None}
        temporary = resume_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record) + "\n")
        temporary.replace(resume_path)
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
                             "and took Heston's v0 from 28% to 82% agreement with the exact posterior")
    parser.add_argument("--chunk-epochs", type=int, default=0,
                        help="train in chunks of this many epochs, saving a resumable state after each; rerunning "
                             "the same command after an interruption continues from the last chunk")
    args = parser.parse_args()
    started = time.perf_counter()
    manifest, splits = bank.load_bank(args.bank)
    task = manifest["configuration"]["task"]
    if args.fresh_noise:
        train = inference.clean_training_data(splits["train"], task)
        augment = inference.fresh_noise(args.seed)
    else:
        train = inference.augmented_training_data(splits["train"], task, augmentations=args.augmentations,
                                                  seed=args.seed)
        augment = None
    validation = inference.training_data(splits["validation"], task)
    resume_path = args.out / "resume.json"
    resumable = args.chunk_epochs > 0
    args.out.mkdir(parents=True, exist_ok=resumable)
    if resumable and not resume_path.exists():
        for stale in args.out.iterdir():  # an interrupted first chunk leaves nothing to resume from
            stale.unlink()

    import bayesflow as bf
    import jax
    import keras

    keras.utils.set_random_seed(args.seed)
    widths = tuple(int(w) for w in args.widths.split(","))
    ready = time.perf_counter()
    resumed_at = []
    if not resumable:
        workflow = inference.build_workflow(task, args.out, subnet_widths=widths, time_power_law_alpha=args.time_alpha)
        history = workflow.fit_offline(data=train, validation_data=validation, epochs=args.epochs,
                                       batch_size=args.batch_size, augmentations=augment, verbose=0).history
        workflow.load_approximator()
    else:
        workflow, history, resumed_at = train_in_chunks(args, task, train, validation, augment, widths, resume_path)
    trained = time.perf_counter()
    observations = splits["validation"]["observations"][:2]
    context = splits["validation"]["context_values"][:2]
    before = inference.sample(workflow, task, observations, context, 128, seed=args.seed + 11)
    restored = inference.build_workflow(task, subnet_widths=widths)
    restored.load_approximator(args.out / "model.keras")
    after = inference.sample(restored, task, observations, context, 128, seed=args.seed + 11)
    if not np.array_equal(before, after):
        raise RuntimeError("checkpoint reload changed same-seed draws")
    losses = [float(v) for v in history["val_loss"]]
    report = {
        "study": "train_calibration", "evidence_scope": "S", "measure": "Q",
        "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
        "bank_manifest_sha256": sha256(args.bank / "manifest.json"),
        "checkpoint_sha256": sha256(args.out / "model.keras"),
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "quote_noise": ("fresh standard-normal error per batch" if args.fresh_noise
                                     else f"{args.augmentations} fixed draws per surface"),
                     "augmentations": None if args.fresh_noise else args.augmentations,
                     "seed": args.seed, "subnet_widths": list(widths), "sampler": inference.SAMPLER_DESCRIPTION,
                     "selected_epoch": int(np.argmin(losses)) + 1, "selection": "minimum validation loss",
                     "optimizer": "AdamW, weight decay 5e-3, clipnorm 1.5; 5% warm-up to 5e-4, cosine decay to 0 "
                                  "over all epochs (BayesFlow's offline default)",
                     "chunk_epochs": args.chunk_epochs or None, "resumed_at_epochs": resumed_at,
                     "time_power_law_alpha": args.time_alpha},
        "training_surfaces": len(splits["train"]["parameters"]), "training_rows": len(train["observations"]),
        "validation_surfaces": len(validation["observations"]), "reserved_test_evaluated": False,
        "same_seed_reload_exact": True,
        "timing_seconds": {"load_and_setup": ready - started, "training_including_compile": trained - ready,
                           "reload_check": time.perf_counter() - trained},
        "cpu_count": __import__("os").cpu_count(),
        "history": {key: [float(v) for v in values] for key, values in history.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "bayesflow": bf.__version__,
                     "keras": keras.__version__, "jax": jax.__version__},
        "source_sha256": {"train_calibration.py": sha256(__file__), "inference.py": sha256(inference.__file__)},
    }
    (args.out / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "task.json").write_text(json.dumps(task, indent=1) + "\n")
    print(json.dumps({"training_minutes": (trained - ready) / 60, "selected_epoch": report["training"]["selected_epoch"],
                      "best_val_loss": min(losses), "reload_exact": True}, indent=1))


if __name__ == "__main__":
    main()

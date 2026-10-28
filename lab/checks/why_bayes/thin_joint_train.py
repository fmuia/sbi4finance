"""Train the six-target thin network (thin_joint.py) on a thin joint bank, in resumable chunks.

FlowMatching, BayesFlow's offline optimizer built once for the whole run (inference.offline_optimizer), a fresh
noise scale and quote error for every batch (thin.thin_noise), and the best epoch by validation loss in
model.keras. After every --chunk-epochs epochs the approximator with its optimizer state goes to last.keras and the
epoch count, history and noise generator's state to resume.json; rerunning the same command continues from there.
This is studies/train_calibration.py's train_in_chunks for the six joint targets. The record holds the settings,
the history, the bank's identity and the package versions.

Needs the joint task of PR #48. Run from lab/:
python checks/why_bayes/thin_joint_train.py --bank BANK --out RUN [--epochs 200 --batch-size 256 --chunk-epochs 10
    --seed 0]
R4's settings for the ten-target thin network: --widths 512,512,512,512 --epochs 300 --time-alpha -0.5
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from sbi4finance.calibration import bank, inference, thin

import thin_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-epochs", type=int, default=10)
    parser.add_argument("--widths", default="256,256,256,256")
    parser.add_argument("--time-alpha", type=float, default=0.5, help="FlowMatching's time_power_law_alpha")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest, splits = bank.load_bank(args.bank, splits=("train", "validation"))
    task = manifest["configuration"]["task"]
    train = thin_joint.clean_data(splits["train"], task)
    validation = thin_joint.validation_data(splits["validation"], task, args.seed)
    resume_path = args.out / "resume.json"
    args.out.mkdir(parents=True, exist_ok=True)

    import bayesflow as bf
    import jax
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

    keras.utils.set_random_seed(args.seed)
    widths = tuple(int(w) for w in args.widths.split(","))
    batches = int(np.ceil(len(train["clean_scaled"]) / args.batch_size))
    planned = {"epochs": args.epochs, "batch_size": args.batch_size, "rows": len(train["clean_scaled"]),
               "widths": list(widths), "time_power_law_alpha": args.time_alpha, "seed": args.seed}
    workflow = thin_joint.build_workflow(task, subnet_widths=widths, time_power_law_alpha=args.time_alpha,
                                         optimizer=inference.offline_optimizer(args.epochs * batches))
    augment = thin.thin_noise(args.seed)
    history, resumed_at, done = {"loss": [], "val_loss": []}, [], 0
    best = BestCheckpoint(args.out / "model.keras", np.inf, 0)
    state = json.loads(resume_path.read_text()) if resume_path.exists() else None
    if state:
        if state["planned"] != planned:
            raise ValueError(f"this run was started with {state['planned']}, not {planned}: use a new --out")
        workflow.approximator = keras.saving.load_model(args.out / "last.keras")
        workflow.optimizer = workflow.approximator.optimizer
        workflow._needs_compile = False
        if int(workflow.optimizer.iterations.numpy()) != state["epochs_done"] * batches:
            raise RuntimeError("restored optimizer step count does not match the saved state")
        augment.rng.bit_generator.state = state["noise_rng"]
        history, done = state["history"], state["epochs_done"]
        resumed_at = state["resumed_at_epochs"] + [done]
        best.best, best.epoch = state["best_val_loss"], state["best_epoch"]
    ready = time.perf_counter()
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
                  "resumed_at_epochs": resumed_at, "planned": planned, "noise_rng": augment.rng.bit_generator.state,
                  "seconds_this_session": time.perf_counter() - ready}
        temporary = resume_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record) + "\n")
        temporary.replace(resume_path)
        print(f"epoch {done}/{args.epochs}: validation loss {history['val_loss'][-1]:.4f}, best {best.best:.4f} at "
              f"epoch {best.epoch}; {time.perf_counter() - ready:.0f} s this session", flush=True)
    trained = time.perf_counter()
    restored = thin_joint.load(args.out, task, subnet_widths=widths)
    quotes = splits["validation"]["reference_prices"][:2]
    draws = thin_joint.sample(restored, task, quotes, 64, seed=args.seed + 11)
    record = {
        "bank": {"path": str(args.bank), "manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
                 "train": len(train["clean_scaled"]), "validation": len(validation["observations"])},
        "task": {"task_id": task["task_id"], "targets": thin_joint.NAMES, "prior_bounds": task["prior_bounds"],
                 "contracts": len(task["context"]["contracts"])},
        "training": {**planned, "chunk_epochs": args.chunk_epochs, "resumed_at_epochs": resumed_at,
                     "noise": f"fresh quote error and log-uniform noise scale on {list(thin.SCALE_BOUNDS)} per batch",
                     "optimizer": "inference.offline_optimizer: AdamW, warm-up then cosine decay over all epochs",
                     "best_epoch": best.epoch, "best_val_loss": best.best, "history": history},
        "reload_check": {"finite_draws": bool(np.isfinite(draws).all()), "shape": list(draws.shape)},
        "seconds_last_session": {"load": ready - started, "train": trained - ready},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "bayesflow": bf.__version__,
                     "keras": keras.__version__, "jax": jax.__version__},
    }
    (args.out / "training.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"{args.out}: best validation loss {best.best:.4f} at epoch {best.epoch} of {args.epochs}")


if __name__ == "__main__":
    main()

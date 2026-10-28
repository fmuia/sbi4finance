"""Train the six-target thin network (thin_joint.py) on a thin joint bank.

FlowMatching with four layers of 256, BayesFlow's offline optimizer, a fresh noise scale and quote error for every
batch (thin.thin_noise), and the best epoch by validation loss. The record holds the settings, the history, the
bank's identity and the package versions.

Needs the joint task of PR #48. Run from lab/:
python checks/why_bayes/thin_joint_train.py --bank BANK --out RUN [--epochs 200 --batch-size 256 --seed 0]
R4's settings for the ten-target thin network: --widths 512,512,512,512 --epochs 300 --time-alpha -0.5
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from sbi4finance.calibration import bank, thin

import thin_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--widths", default="256,256,256,256")
    parser.add_argument("--time-alpha", type=float, default=0.5, help="FlowMatching's time_power_law_alpha")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest, splits = bank.load_bank(args.bank, splits=("train", "validation"))
    task = manifest["configuration"]["task"]
    train = thin_joint.clean_data(splits["train"], task)
    validation = thin_joint.validation_data(splits["validation"], task, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)

    import bayesflow as bf
    import jax
    import keras

    keras.utils.set_random_seed(args.seed)
    widths = tuple(int(w) for w in args.widths.split(","))
    workflow = thin_joint.build_workflow(task, args.out, subnet_widths=widths, time_power_law_alpha=args.time_alpha)
    ready = time.perf_counter()
    history = workflow.fit_offline(data=train, validation_data=validation, epochs=args.epochs,
                                   batch_size=args.batch_size, augmentations=thin.thin_noise(args.seed),
                                   verbose=0).history
    trained = time.perf_counter()
    restored = thin_joint.load(args.out, task, subnet_widths=widths)
    quotes = splits["validation"]["reference_prices"][:2]
    draws = thin_joint.sample(restored, task, quotes, 64, seed=args.seed + 11)
    losses = np.asarray(history["val_loss"], dtype=float)
    record = {
        "bank": {"path": str(args.bank), "manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
                 "train": len(train["clean_scaled"]), "validation": len(validation["observations"])},
        "task": {"task_id": task["task_id"], "targets": thin_joint.NAMES, "prior_bounds": task["prior_bounds"],
                 "contracts": len(task["context"]["contracts"])},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "widths": list(widths), "seed": args.seed,
                     "time_power_law_alpha": args.time_alpha,
                     "noise": f"fresh quote error and log-uniform noise scale on {list(thin.SCALE_BOUNDS)} per batch",
                     "best_epoch": int(np.argmin(losses)) + 1, "best_val_loss": float(losses.min()),
                     "history": {k: [float(v) for v in vals] for k, vals in history.items()}},
        "reload_check": {"finite_draws": bool(np.isfinite(draws).all()), "shape": list(draws.shape)},
        "seconds": {"load": ready - started, "train": trained - ready},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "bayesflow": bf.__version__,
                     "keras": keras.__version__, "jax": jax.__version__},
    }
    (args.out / "training.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"{args.out}: best validation loss {losses.min():.4f} at epoch {np.argmin(losses) + 1} of {args.epochs}; "
          f"{trained - ready:.0f} s training")


if __name__ == "__main__":
    main()

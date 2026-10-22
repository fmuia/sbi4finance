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
    args.out.mkdir(parents=True, exist_ok=False)

    import bayesflow as bf
    import jax
    import keras

    keras.utils.set_random_seed(args.seed)
    widths = tuple(int(w) for w in args.widths.split(","))
    workflow = inference.build_workflow(task, args.out, subnet_widths=widths)
    ready = time.perf_counter()
    history = workflow.fit_offline(data=train, validation_data=validation, epochs=args.epochs,
                                   batch_size=args.batch_size, augmentations=augment, verbose=0)
    trained = time.perf_counter()
    workflow.load_approximator()
    observations = splits["validation"]["observations"][:2]
    context = splits["validation"]["context_values"][:2]
    before = inference.sample(workflow, task, observations, context, 128, seed=args.seed + 11)
    restored = inference.build_workflow(task, subnet_widths=widths)
    restored.load_approximator(args.out / "model.keras")
    after = inference.sample(restored, task, observations, context, 128, seed=args.seed + 11)
    if not np.array_equal(before, after):
        raise RuntimeError("checkpoint reload changed same-seed draws")
    losses = [float(v) for v in history.history["val_loss"]]
    report = {
        "study": "train_calibration", "evidence_scope": "S", "measure": "Q",
        "task_id": task["task_id"], "task_fingerprint": manifest["task_fingerprint"],
        "bank_manifest_sha256": sha256(args.bank / "manifest.json"),
        "checkpoint_sha256": sha256(args.out / "model.keras"),
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "quote_noise": ("fresh standard-normal error per batch" if args.fresh_noise
                                     else f"{args.augmentations} fixed draws per surface"),
                     "augmentations": None if args.fresh_noise else args.augmentations,
                     "seed": args.seed, "subnet_widths": list(widths), "sampler": f"{inference.EULER_STEPS} Euler steps",
                     "selected_epoch": int(np.argmin(losses)) + 1, "selection": "minimum validation loss"},
        "training_surfaces": len(splits["train"]["parameters"]), "training_rows": len(train["observations"]),
        "validation_surfaces": len(validation["observations"]), "reserved_test_evaluated": False,
        "same_seed_reload_exact": True,
        "timing_seconds": {"load_and_setup": ready - started, "training_including_compile": trained - ready,
                           "reload_check": time.perf_counter() - trained},
        "cpu_count": __import__("os").cpu_count(),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
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

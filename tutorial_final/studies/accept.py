"""Copy an accepted study run or checkpoint into tutorial_final, recording what was copied and changed.

    python tutorial_final/studies/accept.py --run RUN_DIR --to results/calibration/heston/s2 [--keep-surfaces 20]
    python tutorial_final/studies/accept.py --checkpoint RUN_DIR --to checkpoints/heston

Results: every .json and .png is copied unchanged. A .npz is copied unchanged
unless --keep-surfaces is given, in which case the arrays "draws", "truth" and
"rows" keep only their first N surfaces (the notebook shows a few; the study
JSON keeps every number). Checkpoints: task.json and training.json are copied
unchanged; model.keras is copied with the optimizer's state removed, which
leaves the network's weights and therefore its draws unchanged (checked by
sampling both with the same seed) and cuts the file to about a third.

Each destination receives ACCEPTED.json: the source directory, the SHA-256 of
every source file, and every change made. The destination must be new.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

FINAL = Path(__file__).resolve().parents[1]
SLICED = ("draws", "truth", "rows")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accept_run(source, destination, keep):
    changes = []
    for path in sorted(source.iterdir()):
        if path.suffix in (".json", ".png"):
            shutil.copy2(path, destination / path.name)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            trimmed = {key for key, value in arrays.items()
                       if keep is not None and key in SLICED and value.ndim >= 1 and len(value) > keep}
            if trimmed:
                arrays = {key: value[:keep] if key in trimmed else value for key, value in arrays.items()}
                np.savez_compressed(destination / path.name, **arrays)
                changes.append(f"{path.name}: kept the first {keep} surfaces of {sorted(trimmed)}")
            else:
                shutil.copy2(path, destination / path.name)
    return changes


def strip_optimizer(source, destination):
    """model.keras without the optimizer's variables; the weights file keeps every other group unchanged."""
    import h5py

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(tmp)
        with h5py.File(tmp / "model.weights.h5", "r") as old, h5py.File(tmp / "stripped.h5", "w") as new:
            for key in old:
                if key == "optimizer":
                    group = new.create_group("optimizer")
                    group.create_group("vars")
                    for name, value in old["optimizer"].attrs.items():
                        group.attrs[name] = value
                else:
                    old.copy(old[key], new, name=key)
            for name, value in old.attrs.items():
                new.attrs[name] = value
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ("metadata.json", "config.json"):
                archive.write(tmp / name, name)
            archive.write(tmp / "stripped.h5", "model.weights.h5")


def same_draws(original, stripped):
    from sbi4finance.calibration import inference, tasks, teaching

    task, workflow = teaching.load_network(original)
    _, copy = teaching.load_network(stripped)
    rng = np.random.default_rng(0)
    contracts = len(task["context"]["contracts"])
    observations = 5.0 * rng.standard_normal((2, contracts)) * tasks.sigma(task) + 0.05
    k = tasks.context_size(task)
    context = np.full((2, k), 0.04) if k else None
    a = inference.sample(workflow, task, observations, context, 64, seed=3)
    b = inference.sample(copy, task, observations, context, 64, seed=3)
    return bool(np.array_equal(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_kind = parser.add_mutually_exclusive_group(required=True)
    source_kind.add_argument("--run", type=Path)
    source_kind.add_argument("--checkpoint", type=Path)
    parser.add_argument("--to", required=True, help="destination relative to tutorial_final/")
    parser.add_argument("--keep-surfaces", type=int)
    args = parser.parse_args()
    source = (args.run or args.checkpoint).resolve()
    destination = FINAL / args.to
    destination.mkdir(parents=True, exist_ok=False)
    record = {"accepted": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": str(source),
              "source_sha256": {p.name: sha256(p) for p in sorted(source.iterdir()) if p.is_file()}}
    if args.run:
        record["changes"] = accept_run(source, destination, args.keep_surfaces)
    else:
        for name in ("task.json", "training.json"):
            shutil.copy2(source / name, destination / name)
        strip_optimizer(source / "model.keras", destination / "model.keras")
        if not same_draws(source, destination):
            shutil.rmtree(destination)
            raise RuntimeError("the stripped checkpoint does not reproduce the original's draws")
        record["changes"] = ["model.keras: optimizer state removed; same-seed draws checked identical to the "
                             "original's, whose SHA-256 the study reports cite as checkpoint_sha256"]
    (destination / "ACCEPTED.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"to": str(destination.relative_to(FINAL)), "files": sorted(p.name for p in destination.iterdir()),
                      "changes": record["changes"]}, indent=1))


if __name__ == "__main__":
    main()

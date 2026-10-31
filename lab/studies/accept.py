"""Publish tutorial assets with an explicit copy receipt and preserved scientific scope.

    python lab/studies/accept.py --run RUN_DIR --to results/calibration/heston/s2
    python lab/studies/accept.py --replay CACHE_DIR --to backbones/EXAMPLE_ID
    python lab/studies/accept.py --checkpoint RUN_DIR --to local-checkpoints/MODEL_ID

Native BF2.1 checkpoint bytes and task/training metadata are preserved, retaining
bound hashes. Historical BF2.0 exports keep their original optimizer-stripping
route and same-seed comparison. Replay exports first validate the full bound
case, copy its bytes unchanged and explicitly retain scientific_acceptance=false.
ACCEPTED.json is a copy receipt, not a new scientific-acceptance decision.

Result JSON/PNG are unchanged; optional --keep-surfaces slices only legacy run
arrays, never bound replay bundles. Every destination must be new and asset-local.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ASSETS = Path(__file__).resolve().parents[2] / "tutorial" / "assets"
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
    record = json.loads((Path(original) / "training.json").read_text())
    if record["training"].get("thin_layout"):
        return same_thin_draws(original, stripped)
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


def same_thin_draws(original, stripped):
    """As same_draws, for a thin-chain checkpoint (study S2-T): the layout's quotes and the noise scale 1."""
    import warnings

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_thin import load_thin_network
    from sbi4finance.calibration import thin

    task, layout, workflow, _ = load_thin_network(original)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        _, _, copy, _ = load_thin_network(stripped)
    _, noise = thin.layout_arrays(layout)
    rng = np.random.default_rng(0)
    quotes = 0.05 + 5.0 * noise * rng.standard_normal((2, len(noise)))
    a = thin.sample_thin(workflow, task, layout, quotes, 64, seed=3)
    b = thin.sample_thin(copy, task, layout, quotes, 64, seed=3)
    return bool(np.array_equal(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_kind = parser.add_mutually_exclusive_group(required=True)
    source_kind.add_argument("--run", type=Path)
    source_kind.add_argument("--checkpoint", type=Path)
    source_kind.add_argument("--replay", type=Path, help="verified development replay; preserves scope and bytes")
    parser.add_argument("--to", required=True, help="destination relative to tutorial/assets/")
    parser.add_argument("--keep-surfaces", type=int)
    args = parser.parse_args()
    source = (args.run or args.checkpoint or args.replay).resolve()
    destination = (ASSETS / args.to).resolve()
    if not destination.is_relative_to(ASSETS.resolve()) or destination == ASSETS.resolve():
        raise ValueError("destination must be inside tutorial/assets/")
    if destination.exists():
        raise FileExistsError(f"{destination} exists: the destination must be new")
    # Everything is built in a staging directory and moved into place only when every check has passed, so a
    # failed acceptance leaves nothing behind and can simply be rerun.
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)  # left by a failed earlier attempt
    staging.mkdir(parents=True)
    record = {"accepted": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": str(source),
              "source_sha256": {p.name: sha256(p) for p in sorted(source.iterdir()) if p.is_file()}}
    if args.replay:
        from sbi4finance.calibration import full_surface_teaching as replay
        bundle = json.loads((source / "bundle.json").read_text())
        if bundle["format"] == "trajectory_teaching_replay_v1":
            from sbi4finance.path_inference.teaching import load_cache
            loader = load_cache
        else:
            loader = replay.load_cache if bundle["format"] == "full_surface_teaching_v1" else replay.load_neural_replay
        loader(source)
        if args.keep_surfaces is not None:
            raise ValueError("bound replay arrays cannot be sliced")
        record["changes"] = accept_run(source, staging, None)
        record["scientific_acceptance"] = False
        record["scope"] = "verified development replay copied byte-for-byte; no scientific promotion"
    elif args.run:
        record["changes"] = accept_run(source, staging, args.keep_surfaces)
    else:
        for name in ("task.json", "training.json"):
            shutil.copy2(source / name, staging / name)
        metadata = json.loads((source / "training.json").read_text())
        native = metadata.get("format") in ("full_surface_training_v1", "path_posterior_v1")
        if native:
            model_key = "model_sha256" if metadata["format"] == "path_posterior_v1" else "checkpoint_sha256"
            if sha256(source / "model.keras") != metadata[model_key]:
                raise ValueError("native checkpoint bytes differ from training record")
            if metadata["format"] == "path_posterior_v1":
                from sbi4finance.path_inference import inference as path_inference
                task = path_inference.validate_task(json.loads((source / "task.json").read_text()))
                if path_inference.fingerprint(task) != metadata["task_sha256"]:
                    raise ValueError("native path task differs from training record")
            shutil.copy2(source / "model.keras", staging / "model.keras")
            record["changes"] = ["native BF2.1 model and metadata copied unchanged; checkpoint hash retained"]
            record["scientific_acceptance"] = False
        else:
            strip_optimizer(source / "model.keras", staging / "model.keras")
        if not native and not same_draws(source, staging):
            shutil.rmtree(staging)
            raise RuntimeError("the stripped checkpoint does not reproduce the original's draws")
        if not native:
            record["changes"] = ["model.keras: optimizer state removed; same-seed draws checked identical to the "
                                 "original's, whose SHA-256 the study reports cite as checkpoint_sha256"]
    (staging / "ACCEPTED.json").write_text(json.dumps(record, indent=2) + "\n")
    staging.rename(destination)
    print(json.dumps({"to": str(destination.relative_to(ASSETS) if destination.is_relative_to(ASSETS) else destination), "files": sorted(p.name for p in destination.iterdir()),
                      "changes": record["changes"]}, indent=1))


if __name__ == "__main__":
    main()

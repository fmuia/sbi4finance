"""Small, hash-bound trajectory replays; no neural backend is needed to read them."""

import json
from pathlib import Path
import shutil

import numpy as np

from . import inference


FORMAT = "trajectory_teaching_replay_v1"
MODELS = ("heston", "rough-known", "rough-joint")


def load_cache(directory):
    directory = Path(directory).resolve()
    bundle = json.loads((directory / "bundle.json").read_text())
    if (bundle.get("format") != FORMAT or bundle.get("scientific_acceptance") is not False
            or bundle.get("source_split") != "validation"
            or bundle.get("reserved_test_evaluated") is not False
            or bundle.get("model") not in MODELS):
        raise ValueError("invalid trajectory development replay scope")
    required = {"task.json", "training.json", "evaluation.json", "posterior.npz"}
    if set(bundle.get("files", {})) != required:
        raise ValueError("trajectory replay must bind every required payload")
    for name, digest in bundle["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or inference.sha256(path) != digest:
            raise ValueError(f"trajectory replay payload hash mismatch: {name}")
    task = inference.validate_task(json.loads((directory / "task.json").read_text()))
    training = json.loads((directory / "training.json").read_text())
    evaluation = json.loads((directory / "evaluation.json").read_text())
    if (inference.fingerprint(task) != bundle["task_sha256"]
            or task["model"] != bundle["model"]
            or training["task_sha256"] != bundle["task_sha256"]
            or training["model_sha256"] != bundle["checkpoint_sha256"]
            or evaluation["task_sha256"] != bundle["task_sha256"]
            or evaluation["checkpoint_sha256"] != bundle["checkpoint_sha256"]
            or evaluation["bank_manifest_sha256"] != training["bank_manifest_sha256"]
            or evaluation["split"] != "validation" or evaluation["offset"] != 0
            or evaluation["reserved_test_evaluated"] is not False):
        raise ValueError("trajectory replay task, model, bank or source cohort differs")
    with np.load(directory / "posterior.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    lengths = bundle["cached_lengths"]
    if (not lengths or lengths != sorted(set(lengths))
            or any(inference.integer(n, "cached length") > task["n_observations"] for n in lengths)
            or set(arrays) != {"observations", "truth", "trajectory_id"} | {f"draws_{n}" for n in lengths}):
        raise ValueError("invalid cached trajectory lengths or arrays")
    count, dimension = len(bundle["case_ids"]), len(task["parameter_names"])
    if (not count or arrays["observations"].shape != (count, task["n_observations"], 1)
            or arrays["truth"].shape != (count, dimension)
            or arrays["trajectory_id"].tolist() != bundle["case_ids"]
            or not np.isfinite(arrays["observations"]).all()):
        raise ValueError("trajectory replay observations, truth or case identities differ")
    bounds = np.asarray(task["prior_bounds"])
    for name in ["truth"] + [f"draws_{n}" for n in lengths]:
        values = arrays[name]
        shape = (count, dimension) if name == "truth" else (count, evaluation["draws"], dimension)
        if (values.shape != shape or not np.isfinite(values).all()
                or np.any(values <= bounds[:, 0]) or np.any(values >= bounds[:, 1])):
            raise ValueError(f"trajectory replay shape/support mismatch: {name}")
    return task, bundle, arrays


def prepare(checkpoint, evaluation_directory, destination, *, cases=4):
    """Curate first declared validation cases, preserving the selected draw rows exactly."""
    checkpoint, evaluation_directory, destination = map(Path, (checkpoint, evaluation_directory, destination))
    inference.integer(cases, "cases")
    if destination.exists():
        raise FileExistsError("trajectory replay destination must be new")
    task = inference.validate_task(json.loads((checkpoint / "task.json").read_text()))
    training = json.loads((checkpoint / "training.json").read_text())
    evaluation = json.loads((evaluation_directory / "evaluation.json").read_text())
    source_arrays = evaluation_directory / "teaching-cases.npz"
    if (inference.sha256(checkpoint / "model.keras") != training["model_sha256"]
            or inference.sha256(source_arrays) != evaluation["teaching_cases_sha256"]):
        raise ValueError("source checkpoint or teaching draws fail their recorded hash")
    with np.load(source_arrays, allow_pickle=False) as data:
        if cases > len(data["truth"]):
            raise ValueError("requested cases exceed saved source panel")
        arrays = {key: data[key][:cases].copy() for key in data.files}
    destination.mkdir(parents=True)
    try:
        for name in ("task.json", "training.json"):
            shutil.copy2(checkpoint / name, destination / name)
        shutil.copy2(evaluation_directory / "evaluation.json", destination / "evaluation.json")
        np.savez_compressed(destination / "posterior.npz", **arrays)
        bundle = {"format": FORMAT, "role": "cached_development_trajectory_replay",
                  "scientific_acceptance": False, "reserved_test_evaluated": False,
                  "source_split": "validation", "source_offset": 0, "model": task["model"],
                  "task_sha256": inference.fingerprint(task),
                  "checkpoint_sha256": training["model_sha256"],
                  "cached_lengths": sorted(int(key.removeprefix("draws_")) for key in arrays if key.startswith("draws_")),
                  "case_ids": arrays["trajectory_id"].tolist(),
                  "source_case_panel_sha256": inference.sha256(source_arrays),
                  "files": {name: inference.sha256(destination / name)
                            for name in ("task.json", "training.json", "evaluation.json", "posterior.npz")},
                  "limitations": "Development illustration from prespecified validation cases. Approximate neural posteriors; marginal diagnostics do not certify conditional or joint accuracy. No real-market validity claim. Intervals can widen as returns accumulate."}
        (destination / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
        load_cache(destination)
    except Exception:
        shutil.rmtree(destination)
        raise
    return bundle

"""New, immutable actual63 compression experiments; historical producers stay frozen.

Run from lab with the locked core environment. Large outputs belong outside Git.
Only complete fixed63 inputs are supported by this S1 trainer. Quote-set, masking
and geometry helpers have separate contracts and cannot reuse these checkpoints.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
sys.path.insert(0, str(LAB))
from scripts import heston_grid_collaborator as delivery
from sbi4f_tutorial.heston_options import experimental_inference as inference

FORMAT = "heston_flexible_experiment_v1"
STUDY_FORMAT = "heston_flexible_study_v1"
NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def update_study(root, directory, run):
    """Serialize concurrent writers; never lose another fit's status update."""
    import fcntl
    root, directory = Path(root).resolve(), Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "study.json"
        study = read(path) if path.exists() else {
            "format": STUDY_FORMAT, "stage": "S1", "scientific_acceptance": False, "runs": []}
        if study.get("format") != STUDY_FORMAT:
            raise ValueError("incompatible study index")
        label = directory.name
        entry = {"label": label, "directory": str(directory), "status": run["status"],
                 "representation": run["configuration"]["representation"],
                 "summary_dim": run["configuration"]["summary_dim"],
                 "completed_epochs": run.get("completed_epochs", 0),
                 "train_rows": run["train_rows"], "seed": run["seed"]}
        study["runs"] = [r for r in study["runs"] if r["directory"] != str(directory)] + [entry]
        study["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write(path, study)


def condition_quotes(quotes, task, whitening):
    """Raw price -> Y/h -> frozen ZCA exactly once; masks/layouts unsupported."""
    if np.ma.isMaskedArray(quotes) or np.iscomplexobj(quotes):
        raise ValueError("complete real raw prices required; masks are unsupported")
    values = np.asarray(quotes, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != 63 or not len(values) or not np.isfinite(values).all():
        raise ValueError("complete finite fixed63 prices required")
    spread = np.asarray(task["records"]["half_spread"], dtype=np.float64)
    return whitening.transform(values / spread)


def _save_model(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".writing.keras")
    model.save(temporary)
    temporary.replace(path)


def train(bundle, out, *, representation="mlp", summary_dim=32, rows=300000,
          epochs=300, batch_size=256, seed=9011, seconds=3600, study_root=None):
    if (type(rows) is not int or not 32 <= rows <= 300000 or type(epochs) is not int or epochs < 1
            or type(batch_size) is not int or batch_size < 1 or type(seed) is not int or seed < 0
            or not 0 < seconds <= 3600):
        raise ValueError("invalid rows/epochs/batch/seed/bounded wall time")
    config = inference.configuration(representation=representation,
                                     summary_dim=summary_dim if representation == "mlp" else None)
    out, bundle = Path(out).resolve(), Path(bundle).resolve()
    if out.exists():
        raise FileExistsError("new experiment output identity required")
    task, arrays, bank = delivery.load_bank(bundle)
    whitening = delivery._whitening(bundle / "bank/whitening.json")
    data = {}
    for split, count in (("train", rows), ("validation", 100)):
        data[split] = {name: arrays[f"{split}_{name}"][:count] for name in NAMES}
        data[split]["observations"] = whitening.transform(arrays[f"{split}_observations"][:count])
    # Do not keep three full raw-price arrays alive throughout training.
    del arrays
    out.mkdir(parents=True)
    shutil.copyfile(bundle / "bank/whitening.json", out / "whitening.json")
    write(out / "task.json", task)
    preprocessing = {"format": "heston_fixed63_preprocessing_v1", "kind": "frozen_zca",
                     "input_units": "raw_price", "network_input": "ZCA(Y/h)",
                     "supported_layout": task["target"]["geometry_and_spreads_sha256"],
                     "mask_support": False, "geometry_variation_support": False,
                     "whitening_sha256": digest(out / "whitening.json")}
    write(out / "preprocessing.json", preprocessing)
    run = {"format": FORMAT, "status": "prepared", "configuration": config,
           "task": task, "seed": seed, "train_rows": rows, "validation_rows": 100,
           "epochs": epochs, "batch_size": batch_size, "max_seconds": seconds,
           "completed_epochs": 0, "smoke_test_only": rows < 200000 or epochs < 300,
           "data_payload_sha256": bank["data_payload_sha256"],
           "bundle": str(bundle), "task_sha256": digest(out / "task.json"),
           "preprocessing_sha256": digest(out / "preprocessing.json"),
           "source": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "runner_sha256": digest(__file__), "inference_sha256": digest(inference.__file__)},
           "versions": {name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "jaxlib", "numpy")},
           "platform": platform.platform(), "started_utc": datetime.now(timezone.utc).isoformat(),
           "checkpoint_policy": "final primary; minimum native val_loss secondary; no diagnostic selection",
           "noise_policy": "immutable stored noisy rows; no fresh-noise augmentation",
           "support_policy": "retain raw unbounded draws; no clipping/rejection/resampling",
           "resume_supported": False, "scientific_acceptance": False,
           "evaluation_scope": "reused development panels; no fresh acceptance claim", "pricing_calls": 0}
    history = {"epoch": [], "loss": [], "val_loss": []}
    write(out / "run.json", run)
    write(out / "history.json", history)
    study_root = Path(study_root).resolve() if study_root else out.parent
    started = time.monotonic()
    workflow = None

    def persist():
        run["elapsed_seconds"] = time.monotonic() - started
        write(out / "run.json", run)
        update_study(study_root, out, run)

    def deadline():
        if time.monotonic() - started > seconds:
            raise TimeoutError("declared fit wall time exhausted")

    try:
        import keras
        import jax
        workflow = inference.build_workflow(task, config, seed=seed)
        run.update(status="training", devices=[str(d) for d in jax.devices()])
        persist()

        class Archive(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                run["resolved_optimizer"] = keras.optimizers.serialize(self.model.optimizer)
                persist()

            def on_train_batch_begin(self, batch, logs=None):
                deadline()

            def on_epoch_end(self, epoch, logs=None):
                logs = {key: float(value) for key, value in (logs or {}).items()}
                if not all(np.isfinite(value) for value in logs.values()):
                    raise FloatingPointError("nonfinite epoch; last valid checkpoint preserved")
                number = epoch + 1
                for key in history:
                    history[key].append(number if key == "epoch" else logs[key])
                run["completed_epochs"] = number
                run["optimizer_updates"] = int(self.model.optimizer.iterations.numpy())
                if logs["val_loss"] < run.get("best_validation_loss", float("inf")):
                    _save_model(self.model, out / "best.keras")
                    run.update(best_epoch=number, best_validation_loss=logs["val_loss"],
                               best_model_sha256=digest(out / "best.keras"))
                if number % 25 == 0 or number == epochs:
                    _save_model(self.model, out / "latest.keras")
                    run["latest_model_sha256"] = digest(out / "latest.keras")
                write(out / "history.json", history)
                persist()
                deadline()

        workflow.fit_offline(data=data["train"], validation_data=data["validation"],
                             epochs=epochs, batch_size=batch_size, callbacks=[Archive()], verbose=0)
        deadline()
        if run["completed_epochs"] != epochs:
            raise RuntimeError("fit did not reach declared endpoint")
        _save_model(workflow.estimator, out / "model.keras")
        run.update(status="complete", model_sha256=digest(out / "model.keras"),
                   presentations=rows * epochs)
        persist()
        return run
    except BaseException as error:
        run.update(status="incomplete" if isinstance(error, TimeoutError) else "failed",
                   error_type=type(error).__name__, error=str(error))
        persist()
        raise


def load_run(directory, *, endpoint="final"):
    directory = Path(directory).resolve()
    run, task, preprocessing = (read(directory / name) for name in
                                ("run.json", "task.json", "preprocessing.json"))
    if (run.get("format") != FORMAT or run.get("status") != "complete"
            or task != run["task"] or digest(directory / "task.json") != run["task_sha256"]
            or digest(directory / "preprocessing.json") != run["preprocessing_sha256"]
            or preprocessing.get("mask_support") is not False
            or preprocessing.get("geometry_variation_support") is not False
            or digest(inference.__file__) != run["source"]["inference_sha256"]):
        raise ValueError("complete compatible experiment/task/preprocessing source required")
    if endpoint not in ("final", "best"):
        raise ValueError("endpoint must be final or best")
    model_name, key = ("model.keras", "model_sha256") if endpoint == "final" else ("best.keras", "best_model_sha256")
    if digest(directory / model_name) != run[key]:
        raise ValueError("checkpoint payload differs")
    expected_versions = {name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "jaxlib", "numpy")}
    if run.get("versions") != expected_versions:
        raise ValueError("checkpoint native environment versions differ")
    delivery._whitening(directory / "whitening.json")
    workflow = inference.build_workflow(task, run["configuration"], seed=run["seed"])
    workflow.load_estimator(directory / model_name)
    return workflow, task, preprocessing, run


def sample_run(directory, quotes, num_samples=512, seed=261112, *, endpoint="final"):
    workflow, task, _, _ = load_run(directory, endpoint=endpoint)
    whitening = delivery._whitening(Path(directory) / "whitening.json")
    conditions = condition_quotes(quotes, task, whitening)
    return inference.sample(workflow, conditions, list(NAMES), num_samples, seed=seed)


def _evaluation_bank(directory, task, preprocessing, run, bundle=None):
    """Relocate evaluation inputs explicitly without changing training provenance."""
    directory = Path(directory)
    bundle_path = Path(run["bundle"] if bundle is None else bundle).resolve()
    bank_task, arrays, bank = delivery.load_bank(bundle_path)
    whitening_sha256 = digest(bundle_path / "bank/whitening.json")
    if (bank_task != task or bank["data_payload_sha256"] != run["data_payload_sha256"]
            or bank["whitening_sha256"] != preprocessing["whitening_sha256"]
            or whitening_sha256 != preprocessing["whitening_sha256"]
            or digest(directory / "whitening.json") != whitening_sha256):
        raise ValueError("evaluation bundle task, bank or whitening differs from the training run")
    provenance = {"bundle_override_used": bundle is not None, "resolved_bundle": str(bundle_path),
                  "bundle_manifest_sha256": digest(bundle_path / "bundle.json"),
                  "data_payload_sha256": bank["data_payload_sha256"],
                  "whitening_sha256": whitening_sha256,
                  "training_run_sha256": digest(directory / "run.json"),
                  "evaluator_sha256": digest(__file__)}
    return arrays, provenance


def evaluate(directory, *, endpoint="final", population_rows=512, draws=512, conditional_draws=4096,
             bundle=None):
    """Reuse raw-draw diagnostics and hash-checked target-matched full MCMC chains."""
    from scripts import evaluate_heston_pr87_grid as diagnostics
    directory = Path(directory).resolve()
    destination = directory / ("evaluation.json" if endpoint == "final" else "best-evaluation.json")
    if destination.exists():
        raise FileExistsError("evaluation identity already exists")
    if (type(population_rows) is not int or not 1 <= population_rows <= 512
            or type(draws) is not int or draws < 32 or type(conditional_draws) is not int or conditional_draws < 32):
        raise ValueError("invalid development evaluation allocation")
    started = time.monotonic()
    workflow, task, preprocessing, run = load_run(directory, endpoint=endpoint)
    arrays, evaluation_source = _evaluation_bank(directory, task, preprocessing, run, bundle)
    quotes = arrays["historical_quotes"][:population_rows]
    truth = np.column_stack([arrays["historical_" + name][:population_rows, 0] for name in NAMES])
    whitening = delivery._whitening(directory / "whitening.json")

    def query(values, count, seed):
        x = condition_quotes(values, task, whitening)
        return inference.sample(workflow, x, list(NAMES), count, seed=seed)

    population = np.concatenate([query(quotes[start:start + 16], draws, 262112 + start)
                                 for start in range(0, len(quotes), 16)])
    summary, quantiles = diagnostics.population_summary(population, truth, quotes, task["target"], 263112)
    medians = np.median(population, axis=1)
    lower, upper = np.quantile(population, [.05, .95], axis=1)
    parameters = {name: {"rmse": float(np.sqrt(np.mean((medians[:, j] - truth[:, j]) ** 2))),
                         "coverage90": float(np.mean((truth[:, j] >= lower[:, j]) & (truth[:, j] <= upper[:, j]))),
                         "mean_width90": float(np.mean(upper[:, j] - lower[:, j])),
                         "median_width90": float(np.median(upper[:, j] - lower[:, j]))}
                  for j, name in enumerate(NAMES)}
    result = {"format": "heston_flexible_evaluation_v1", "status": "started", "endpoint": endpoint,
              "population_rows": population_rows, "draws_per_case": draws, "parameters": parameters,
              "population": summary, "cases": {}, "scientific_acceptance": False,
              "evaluation_source": evaluation_source,
              "scope": "reused historical development cases; raw draws retained"}
    write(destination.with_suffix(".progress.json"), result)
    payload = {"population": population, "truth": truth, "quantiles": quantiles}
    reference_root = LAB / "evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/recovery-bank-independent/references"
    panel_path = LAB / "evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/cv-preparation/conditional-panel.json"
    panel = read(panel_path)
    if panel["task"] != task["target"]:
        raise ValueError("conditional panel target differs")
    for index, case in enumerate(panel["cases"]):
        case_id = case["id"]
        observation = np.asarray(case["observations"], dtype=np.float64)
        record_path = reference_root / case_id / "run.json"
        reference = read(record_path)
        values = query(observation[None], conditional_draws, 261112 + index)[0]
        entry = {"reference_ready": False, "parameters": {},
                 "support": diagnostics.raw_support(values, task["target"]["prior_bounds"], list(NAMES))}
        chain_path = Path(reference.get("chain_payload_path", ""))
        if chain_path.is_file() and digest(chain_path) != reference.get("chain_payload_sha256"):
            raise ValueError("conditional reference chain payload differs")
        if (reference.get("reference_ready") is True and reference["task"] == task["target"]
                and np.array_equal(reference["observations"], observation) and chain_path.is_file()
                and digest(chain_path) == reference["chain_payload_sha256"]):
            with np.load(chain_path, allow_pickle=False) as archive:
                chains = archive["parameters"].copy()
            comparison = diagnostics.descriptive_comparison(values, chains, list(NAMES), panel["products"],
                                                            task["target"]["prior_bounds"])
            flat = chains.reshape(-1, len(NAMES))
            for j, name in enumerate(NAMES):
                q = np.quantile(values[:, j], [.05, .95])
                r = np.quantile(flat[:, j], [.05, .95])
                entry["parameters"][name] = {
                    "mean_error_in_reference_sd": float((np.mean(values[:, j]) - np.mean(flat[:, j])) / np.std(flat[:, j], ddof=1)),
                    "width90_ratio": float((q[1] - q[0]) / (r[1] - r[0]))}
            entry.update(reference_ready=True, comparison=comparison,
                         reference_record_sha256=digest(record_path), reference_chain_sha256=digest(chain_path))
        result["cases"][case_id] = entry
        payload[case_id] = values
        write(destination.with_suffix(".progress.json"), result)
    payload_path = directory / ("draws.npz" if endpoint == "final" else "best-draws.npz")
    np.savez_compressed(payload_path, **payload)
    result.update(status="complete", seconds=time.monotonic() - started,
                  draws_sha256=digest(payload_path))
    write(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="role", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--bundle", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--study-root", type=Path)
    fit.add_argument("--representation", choices=("direct", "mlp"), default="mlp")
    fit.add_argument("--summary-dim", type=int, default=32)
    fit.add_argument("--rows", type=int, default=300000)
    fit.add_argument("--epochs", type=int, default=300)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--seed", type=int, default=9011)
    fit.add_argument("--seconds", type=float, default=3600)
    fit.add_argument("--evaluate", action="store_true")
    check = commands.add_parser("evaluate")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--bundle", type=Path, help="Relocated verified bundle; must match training task, bank and whitening")
    check.add_argument("--endpoint", choices=("final", "best"), default="final")
    check.add_argument("--population-rows", type=int, default=512)
    check.add_argument("--draws", type=int, default=512)
    check.add_argument("--conditional-draws", type=int, default=4096)
    args = vars(parser.parse_args())
    role = args.pop("role")
    if role == "train":
        should_evaluate = args.pop("evaluate")
        result = train(**args)
        if should_evaluate:
            evaluate(args["out"])
            if result["best_epoch"] != result["epochs"]:
                evaluate(args["out"], endpoint="best")
    else:
        result = evaluate(**args)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()

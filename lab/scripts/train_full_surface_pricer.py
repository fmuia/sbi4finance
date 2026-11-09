"""Fit a forward price emulator from existing clean bank labels.

Reads training and validation only. This produces an approximate deterministic
likelihood component, not an independently validated pricing reference.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration import bank_collection, full_surface, pricer_net, tasks  # noqa: E402
from sbi4finance.gbm_bank import _write_json  # noqa: E402
from sbi4finance.runtime_sources import option_pricer_runtime_hashes  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error_summary(errors):
    """Show individual quote errors and their accumulation across the surface."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or not errors.size or not np.isfinite(errors).all():
        raise ValueError("finite nonempty surface-by-contract errors required")
    quantiles = [0.5, 0.9, 0.99, 1.0]
    def summary(values):
        return dict(zip(map(str, quantiles), np.quantile(values, quantiles).tolist()))
    return {"units": "quote-noise standard deviations",
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "absolute_quote_error_quantiles": summary(np.abs(errors)),
            "surface_rms_error_quantiles": summary(np.sqrt(np.mean(errors**2, axis=1))),
            "surface_l2_error_quantiles": summary(np.linalg.norm(errors, axis=1)),
            "signed_mean_error_per_contract": errors.mean(axis=0).tolist(),
            "interpretation": "Discrepancies against numerical labels mix emulator error and label randomness; "
                              "they do not measure absolute price bias or grid error. "
                              "Per-quote accuracy alone does not bound posterior distortion across all contracts."}


def run_training(args):
    if args.out.exists():
        raise FileExistsError(f"use a new pricer output directory: {args.out}")
    widths = tuple(int(w) for w in args.widths.split(","))
    if (not widths or min(widths) < 1 or min(args.epochs, args.batch_size, args.patience) < 1
            or args.seed < 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("positive widths/training settings and nonnegative seed required")
    before = time.perf_counter()
    manifest_hash = digest(args.bank / "manifest.json")
    roles = ("parameters", "reference_prices", "context_values", "example_id", "pricing_standard_errors")
    manifest, splits = bank_collection.load_bank(args.bank, splits=("train", "validation"), roles=roles)
    if digest(args.bank / "manifest.json") != manifest_hash:
        raise ValueError("bank manifest changed while loading")
    task = full_surface.validate_task(manifest["configuration"]["task"])
    if any(not len(splits[name]["parameters"]) for name in ("train", "validation")):
        raise ValueError("nonempty training and validation splits required")
    sources = [Path(__file__), ROOT / "src/sbi4finance/runtime_sources.py", *sorted((ROOT / "src/sbi4finance/calibration").glob("*.py"))]
    record = {"format": "full_surface_pricer_training_v1", "status": "training",
              "task_fingerprint": manifest["task_fingerprint"], "task_id": task["task_id"],
              "bank_manifest_sha256": manifest_hash, "collection_id": manifest.get("collection_id"),
              "generator_source_sha256": manifest.get("source_sha256"),
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "shared_runtime_source_sha256": option_pricer_runtime_hashes(task),
              "environment_sha256": {name: digest(ROOT / name) for name in ("pyproject.toml", "pixi.lock")},
              "versions": {"python": platform.python_version(),
                           **{name: metadata.version(name) for name in ("keras", "jax", "numpy", "scipy")}},
              "settings": {"widths": list(widths), **{name: getattr(args, name) for name in
                           ("epochs", "batch_size", "learning_rate", "patience", "seed")}},
              "training_surfaces": len(splits["train"]["parameters"]),
              "validation_surfaces": len(splits["validation"]["parameters"]),
              "reserved_test_evaluated": False, "scientific_acceptance": False,
              "target": "clean numerical prices divided by fixed quote SD; no observation-noise augmentation",
              "conditioning": ("three model parameters and supplied log curve levels" if full_surface.uses_known_curve(task)
                               else "physical model parameters through logits; joint auxiliary curves excluded"),
              "scope": "Validation selects the epoch and measures interpolation error against bank labels. "
                       "Independent price/refinement and posterior checks remain required."}
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "run.json", record)
    try:
        pricer, fit = pricer_net.train(task, splits["train"], splits["validation"],
            widths=widths, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            patience=args.patience, seed=args.seed)
        checkpoint = args.out / "checkpoint"
        pricer.save(checkpoint)
        _write_json(checkpoint / "task.json", task)
        restored = pricer_net.Pricer.load(checkpoint)
        x = pricer_net.features(task, splits["validation"]["parameters"], splits["validation"].get("context_values"))
        if not np.array_equal(pricer(x), restored(x)):
            raise RuntimeError("saved NumPy pricer changed validation predictions")
        label_se = splits["validation"]["pricing_standard_errors"] / tasks.sigma(task)
        if not np.isfinite(label_se).all() or np.any(label_se < 0):
            raise ValueError("invalid reported label MCSE")
        record.update(fit=fit, pricer_sha256=digest(checkpoint / "pricer.npz"), same_reload_exact=True,
                      validation_errors=error_summary(pricer_net.errors(restored, task, splits["validation"])),
                      validation_label_reported_mcse={"units": "quote-noise standard deviations",
                          "median": float(np.median(label_se)), "p99": float(np.quantile(label_se, .99)),
                          "max": float(label_se.max()),
                          "interpretation": "Reported label MCSE only; not independent price error or a grid-bias bound."},
                      status="completed", seconds=time.perf_counter() - before)
        if digest(args.bank / "manifest.json") != manifest_hash:
            raise ValueError("bank manifest changed during training")
        if (any(digest(ROOT / name) != value for name, value in record["source_sha256"].items())
                or option_pricer_runtime_hashes(task) != record["shared_runtime_source_sha256"]):
            raise ValueError("source changed during training")
        pricer_net.save_record(checkpoint, record)
        _write_json(args.out / "run.json", record)
        return record
    except BaseException as error:
        record.update(status="incomplete", seconds=time.perf_counter() - before,
                      failure={"type": type(error).__name__, "error": str(error)})
        _write_json(args.out / "run.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--widths", default="256,256,256,256")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.bank, args.out = args.bank.resolve(), args.out.resolve()
    record = run_training(args)
    print(json.dumps({"checkpoint": str(args.out / "checkpoint"),
                      "validation_rmse_quote_noise": record["validation_errors"]["rmse"],
                      "scientific_acceptance": False}))


if __name__ == "__main__":
    main()

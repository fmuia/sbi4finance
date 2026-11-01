"""Independent per-model pilot or bounded-memory integrity gate for offline banks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import bank, inference


def write_report(out, name, value):
    temporary = out / (name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / name)


def teaching_contract(task):
    expected = bank.teaching_task(task["model"])
    if bank.task_fingerprint(task) != bank.task_fingerprint(expected):
        raise ValueError("bank differs from the declared harmonized teaching task")
    if (task["dt_years"] != 1 / 252 or task["n_observations"] != 1260
            or "drift" not in task["parameter_names"] or "drift" in task["fixed"]):
        raise ValueError("daily five-year observations and inferred drift required")


def integrity(directory, out):
    started = time.monotonic()
    manifest = bank.load_manifest(directory)
    teaching_contract(manifest["task"])
    if manifest["split_counts"] != {"train": 1_000_000, "validation": 4000, "test": 4000}:
        raise ValueError("incomplete scale-up bank counts")
    counts, byte_count, parts = {}, 0, 0
    for split, records in manifest["splits"].items():
        counts[split] = 0
        for record in records:
            arrays = bank._load_part(directory, manifest, split, record)
            counts[split] += len(arrays["parameters"])
            byte_count += (directory / record["path"]).stat().st_size
            parts += 1
            del arrays
        if counts[split] != manifest["split_counts"][split]:
            raise ValueError("shard counts do not match manifest")
    result = {"passed": True, "scope": "all shard bytes, finite arrays, labels and split-qualified identities; no posterior evaluation",
              "task_sha256": manifest["task_sha256"], "bank_manifest_sha256": inference.sha256(directory / "manifest.json"),
              "counts": counts, "parts": parts, "payload_bytes": byte_count,
              "elapsed_seconds": time.monotonic() - started,
              "reserved_test_evaluated": False, "scientific_acceptance": False}
    write_report(out, "integrity.json", result)
    return result


def pilot(model, out, *, cases=2048, repeats=3):
    started = time.monotonic()
    task = bank.teaching_task(model)
    teaching_contract(task)
    # Compile/warm the numerical kernel before measuring repeated production chunks.
    bank.generate(task, "validation", 0, 8, seed=20261031091)
    measurements = []
    for repeat in range(repeats):
        destination = out / f"benchmark-{repeat}"
        before = time.monotonic()
        bank.write_bank(task, destination, cases, 16, 16, chunk=128, seed=20261031101 + repeat)
        measurements.append(time.monotonic() - before)
    forecast = float(np.median(measurements) * (1_008_000 / (cases + 32)) * 1.25 + 180)
    predicted_bytes = 1_008_000 * (1260 * 8 + len(task["parameter_names"]) * 8 + 256)
    free = shutil.disk_usage(out).free
    sample = bank.generate(task, "validation", 0, 256, seed=20261031201)
    inference.conditioning_data(sample["observations"], task)
    mu = sample["parameters"][:, task["parameter_names"].index("drift")]
    statistics = {}
    for n in (21, 252, 756, 1260):
        x = sample["observations"][:, :n, 0]
        statistics[str(n)] = {"terminal_log_return_quantiles": np.quantile(x.sum(axis=1), [.01, .5, .99]).tolist(),
                             "annualized_realized_variance_quantiles": np.quantile(np.sum(x * x, axis=1) / (n / 252), [.01, .5, .99]).tolist(),
                             "maximum_absolute_daily_return": float(np.max(np.abs(x)))}
    result = {"format": "path_scaleup_pilot_v1", "model": model, "task": task,
              "task_sha256": bank.task_fingerprint(task), "benchmark_rows_per_repeat": cases + 32,
              "generation_write_verify_seconds": measurements,
              "forecast_full_bank_seconds_with_margin": forecast, "bank_wall_limit_seconds": 4500,
              "predicted_bank_bytes": predicted_bytes, "free_filesystem_bytes": free,
              "drift_draw_range": [float(mu.min()), float(mu.max())],
              "prior_predictive": statistics, "elapsed_seconds": time.monotonic() - started,
              "scope": "execution, numerical finiteness, reproducibility contracts and bounded cost; not a continuous-model or posterior certificate",
              "reserved_test_evaluated": False, "scientific_acceptance": False,
              "source_sha256": {str(p.relative_to(ROOT)): inference.sha256(p)
                                  for p in (Path(__file__), Path(bank.__file__))}}
    result["passed"] = bool(forecast <= 4380 and free >= predicted_bytes * 2 + 4 * 1024**3)
    write_report(out, "pilot.json", result)
    if not result["passed"]:
        raise RuntimeError(f"{model} pilot exceeds its time/storage admission margin; other model lanes are independent")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=bank.MODELS)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.model) == bool(args.bank):
        parser.error("choose exactly one of --model or --bank")
    args.out.mkdir(parents=True, exist_ok=False)
    result = integrity(args.bank, args.out) if args.bank else pilot(args.model, args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in ("task", "prior_predictive", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()

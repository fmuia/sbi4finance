"""A PyMC/Fourier Heston reference for one full-surface bank observation.

This is an offline instructor tool, not notebook execution or an accepted
reference collection. Small smoke runs must not be presented as converged MCMC.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import bank, bank_collection, reference  # noqa: E402
from sbi4finance.gbm_bank import _write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    manifest, data = bank_collection.load_bank(args.bank, splits=[args.split],
                                               roles=["observations", "example_id"])
    task = manifest["configuration"]["task"]
    split = data[args.split]
    if not 0 <= args.index < len(split["example_id"]):
        raise ValueError("index must select an existing surface in the chosen split")
    quotes = split["observations"][args.index:args.index + 1]
    log_likelihood = reference.heston_log_likelihood(task, quotes)
    args.out.mkdir(parents=True, exist_ok=False)
    record = {
        "format": "full_surface_pymc_reference_v1", "task": task,
        "target": "independent Gaussian quote noise around deterministic refined Fourier prices",
        "sampler": "PyMC DEMetropolisZ; no gradients; sequential independent chains",
        "bank_manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
        "surface_id": str(split["example_id"][args.index]), "split": args.split,
        "observed_prices": quotes[0, :, 0].tolist(),
        "settings": {key: getattr(args, key) for key in ("draws", "tune", "chains", "seed")},
        "versions": {"python": platform.python_version(),
                     **{key: metadata.version(key) for key in ("pymc", "pytensor", "arviz-stats", "numpy", "scipy")}},
        "source_sha256": {"reference.py": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                          "calibration_reference.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "generator_source_sha256": manifest.get("source_sha256"),
        "reference_pricing_source_sha256": bank.source_hashes(task),
        "environment_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                               for name in ("pyproject.toml", "pixi.lock")},
    }
    _write_json(args.out / "reference.json", record)
    started = time.perf_counter()
    try:
        idata = reference.sample_deterministic_reference(
            log_likelihood, task["prior_bounds"], task["parameter_names"], draws=args.draws,
            tune=args.tune, chains=args.chains, seed=args.seed, progressbar=not args.no_progress)
        arrays = {"parameters": idata["posterior"]["parameters"].transpose("chain", "draw", "parameter").values}
        for name, value in idata["sample_stats"].data_vars.items():
            if set(value.dims) == {"chain", "draw"}:
                arrays[f"sample_stats_{name}"] = value.transpose("chain", "draw").values
        np.savez_compressed(args.out / "chains.npz", **arrays)
        diagnostics = reference.chain_diagnostics(idata)
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        _write_json(args.out / "diagnostics.json", diagnostics)
        print(json.dumps({"out": str(args.out), "shape": list(arrays["parameters"].shape),
                          "mcmc_diagnostics_passed": diagnostics["mcmc_diagnostics_passed"]}))
    except Exception as error:
        _write_json(args.out / "failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()

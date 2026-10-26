"""A PyMC/Fourier Heston reference for one full-surface bank observation.

This is an offline instructor tool, not notebook execution or an accepted
reference collection. Small smoke runs must not be presented as converged MCMC.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi4finance.calibration import bank, bank_collection, reference  # noqa: E402
from sbi4finance.gbm_bank import _write_json  # noqa: E402


class PartialDrawWriter:
    """Native PyMC callback: retain draws atomically, without a resume promise.

    The parent invokes callbacks for both sequential and parallel sampling.
    Physical draws include warmup flags. Files contain no DEMetropolisZ history,
    adaptation or RNG state and cannot resume the Markov chain.
    """

    def __init__(self, path, *, chains, tune, draws, every=100):
        if any(type(n) is not int or n < 1 for n in (chains, tune, draws, every)):
            raise ValueError("positive chains/tune/draws/checkpoint frequency required")
        self.path = Path(path)
        self.path.mkdir(exist_ok=False)
        self.chains, self.tune, self.draws, self.every = chains, tune, draws, every
        self.counts = [0] * chains
        self.started = time.perf_counter()
        self._progress()

    def _progress(self):
        record = {"format": "pymc_partial_draw_progress_v1", "sampler_resumable": False,
                  "scientific_acceptance": False, "planned_chains": self.chains,
                  "planned_tune": self.tune, "planned_draws": self.draws,
                  "counts_including_warmup": self.counts,
                  "posterior_draw_counts": [max(n - self.tune, 0) for n in self.counts],
                  "seconds": time.perf_counter() - self.started,
                  "scope": "partial trace only; tuning flags must be honored; no sampler/adaptation/RNG state"}
        temporary = self.path / "progress.json.tmp"
        _write_json(temporary, record)
        temporary.replace(self.path / "progress.json")

    def __call__(self, *, trace, draw):
        chain, count = int(draw.chain), int(draw.draw_idx) + 1
        if not 0 <= chain < self.chains or count != len(trace):
            raise ValueError("unexpected PyMC callback chain/draw indexing")
        if count != 1 and count % self.every and count != self.tune + self.draws:
            return
        # NDArray traces preallocate: retain only the draws actually recorded.
        values = np.asarray(trace.get_values("parameters"))[:count]
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("invalid physical parameter draws in the partial trace")
        destination = self.path / f"chain-{chain:03d}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, parameters=values, draw_idx=np.arange(count),
                                tuning=np.arange(count) < self.tune, chain=np.asarray(chain))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        self.counts[chain] = count
        self._progress()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1, help="Parallel chains, at most the allocated CPU count.")
    parser.add_argument("--mp-context", choices=("spawn", "forkserver"), default="spawn")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Persist partial per-chain draws at this interval.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    reference.validate_parallelism(args.cores, args.chains, args.mp_context)
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
        "sampler": "PyMC DEMetropolisZ; no gradients; native independent chains with explicit cores/context",
        "bank_manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
        "surface_id": str(split["example_id"][args.index]), "split": args.split,
        "observed_prices": quotes[0, :, 0].tolist(),
        "settings": {key: getattr(args, key) for key in ("draws", "tune", "chains", "cores", "mp_context", "checkpoint_every", "seed")},
        "initialization": reference.reference_initialization(task["prior_bounds"], args.chains, args.seed),
        "partial_trace": {"directory": "partial", "sampler_resumable": False,
                          "scope": "draw persistence only; never pool warmup or substitute for convergence checks"},
        "versions": {"python": platform.python_version(),
                     **{key: metadata.version(key) for key in ("pymc", "pytensor", "arviz-stats", "numpy", "scipy")}},
        "source_sha256": {"reference.py": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                          "pymc_reference_op.py": hashlib.sha256((Path(reference.__file__).parent / "pymc_reference_op.py").read_bytes()).hexdigest(),
                          "calibration_reference.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "generator_source_sha256": manifest.get("source_sha256"),
        "reference_pricing_source_sha256": bank.source_hashes(task),
        "environment_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                               for name in ("pyproject.toml", "pixi.lock")},
    }
    _write_json(args.out / "reference.json", record)
    started = time.perf_counter()
    try:
        partial = PartialDrawWriter(args.out / "partial", chains=args.chains, tune=args.tune,
                                    draws=args.draws, every=args.checkpoint_every)
        idata = reference.sample_deterministic_reference(
            log_likelihood, task["prior_bounds"], task["parameter_names"], draws=args.draws,
            tune=args.tune, chains=args.chains, seed=args.seed, progressbar=not args.no_progress,
            cores=args.cores, mp_context=args.mp_context, callback=partial)
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

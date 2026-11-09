"""PyMC reference for one full-surface observation.

Heston defaults to deterministic Fourier pricing. Optional rough forward-pricer
references target an approximate emulated model, not independent pricing truth.

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
    parser.add_argument("--pricer", type=Path,
                        help="Optional task-bound rough emulator directory: pricer.npz, task.json and pricer.json. Approximate target.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1, help="Parallel chains, at most the allocated CPU count.")
    parser.add_argument("--mp-context", choices=("spawn", "forkserver"), default="spawn")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Persist partial per-chain draws at this interval.")
    parser.add_argument("--initialization", choices=("prior", "data-fit"), default=None,
                        help="Default: prior starts for Fourier; data-only fitted starts for an emulator.")
    parser.add_argument("--fit-starts", type=int, default=None, help="Data-fit multistarts; default min(4, chains).")
    parser.add_argument("--fit-budget", type=int, default=2000, help="Total maximum Powell function evaluations across starts.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    args.initialization = args.initialization or ("data-fit" if args.pricer is not None else "prior")
    reference.validate_parallelism(args.cores, args.chains, args.mp_context)
    declared_task = json.loads((args.bank / "manifest.json").read_text())["configuration"]["task"]
    observed_curve = args.pricer is not None and declared_task["task_id"] == "rough_calibration_v1"
    roles = ["observations", "example_id"] + (["context_values"] if observed_curve else [])
    manifest, data = bank_collection.load_bank(args.bank, splits=[args.split],
                                               roles=roles)
    task = manifest["configuration"]["task"]
    if task != declared_task:
        raise ValueError("bank task changed while selecting the reference inputs")
    split = data[args.split]
    if not 0 <= args.index < len(split["example_id"]):
        raise ValueError("index must select an existing surface in the chosen split")
    quotes = split["observations"][args.index:args.index + 1]
    pricer_identity, known_curve = None, None
    if args.pricer is None:
        log_likelihood = reference.heston_log_likelihood(task, quotes)
    else:
        pricer, pricer_identity = reference.load_forward_pricer(args.pricer, task)
        if task["task_id"] == "rough_calibration_v1":
            known_curve = split["context_values"][args.index:args.index + 1].copy()
        log_likelihood = reference.pricer_log_likelihood(task, quotes, pricer, known_curve)
    args.out.mkdir(parents=True, exist_ok=False)
    record = {
        "format": "full_surface_pymc_reference_v1", "task": task,
        "target": ("independent Gaussian quote noise around deterministic refined Fourier prices"
                   if args.pricer is None else
                   "independent Gaussian quote noise around deterministic forward-pricer emulator; approximate price target"),
        "likelihood_kind": "fourier" if args.pricer is None else "forward-pricer-emulator",
        "scientific_acceptance": False,
        "sampler": "PyMC DEMetropolisZ; no gradients; native independent chains with explicit cores/context",
        "bank_manifest_sha256": hashlib.sha256((args.bank / "manifest.json").read_bytes()).hexdigest(),
        "surface_id": str(split["example_id"][args.index]), "split": args.split,
        "observed_prices": quotes[0, :, 0].tolist(),
        "known_curve_levels": None if known_curve is None else known_curve[0].tolist(),
        "settings": {key: getattr(args, key) for key in ("draws", "tune", "chains", "cores", "mp_context", "checkpoint_every",
                                                        "initialization", "fit_starts", "fit_budget", "seed")},
        "initialization": reference.reference_initialization(task["prior_bounds"], args.chains, args.seed),
        "partial_trace": {"directory": "partial", "sampler_resumable": False,
                          "scope": "draw persistence only; never pool warmup or substitute for convergence checks"},
        "versions": {"python": platform.python_version(),
                     **{key: metadata.version(key) for key in ("pymc", "pytensor", "arviz-stats", "numpy", "scipy")}},
        "source_sha256": {"reference.py": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                          "pymc_reference_op.py": hashlib.sha256((Path(reference.__file__).parent / "pymc_reference_op.py").read_bytes()).hexdigest(),
                          "reference_initialization.py": hashlib.sha256((Path(reference.__file__).parent / "reference_initialization.py").read_bytes()).hexdigest(),
                          "variance_functionals.py": hashlib.sha256((Path(reference.__file__).parent / "variance_functionals.py").read_bytes()).hexdigest(),
                          "calibration_reference.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "generator_source_sha256": manifest.get("source_sha256"),
        "reference_pricing_source_sha256": bank.source_hashes(task),
        "environment_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                               for name in ("pyproject.toml", "pixi.lock")},
    }
    if pricer_identity is not None:
        from sbi4finance.calibration import pricer_net
        from sbi4finance.gbm_bank import _json
        from sbi4finance.runtime_sources import option_pricer_runtime_hashes

        record["emulator"] = pricer_identity
        record["task_fingerprint"] = bank.task_fingerprint(manifest["configuration"])
        record["observation_sha256"] = hashlib.sha256(_json(record["observed_prices"]).encode()).hexdigest()
        record["known_curve_sha256"] = (None if known_curve is None else
            hashlib.sha256(_json(record["known_curve_levels"]).encode()).hexdigest())
        record["source_sha256"]["pricer_net.py"] = hashlib.sha256(Path(pricer_net.__file__).read_bytes()).hexdigest()
        record["shared_runtime_source_sha256"] = option_pricer_runtime_hashes(task)
        record["reference_pricing_source_sha256"] = None
        record["approximation_scope"] = (
            "MCMC targets the fixed fitted emulator with declared Gaussian quote noise. "
            "Emulator bias and training-pricer randomness/covariance are not propagated. "
            "Independent pricing/refinement and posterior/product checks remain required; "
            "convergence alone does not establish an exact-model reference.")
    if args.initialization == "data-fit":
        record["initialization"] = {"kind": "data-fit", "status": "pending",
                                    "fit_starts": args.fit_starts, "optimizer_evaluation_budget": args.fit_budget,
                                    "scope": "transformed-space MAP initialization has not completed"}
    _write_json(args.out / "reference.json", record)
    started = time.perf_counter()
    try:
        initialization = None
        if args.initialization == "data-fit":
            initialization = reference.data_fit_initialization(
                log_likelihood, task["prior_bounds"], args.chains, args.seed,
                fit_starts=args.fit_starts, fit_budget=args.fit_budget)
            record["initialization"] = initialization
            _write_json(args.out / "reference.json", record)
        partial = PartialDrawWriter(args.out / "partial", chains=args.chains, tune=args.tune,
                                    draws=args.draws, every=args.checkpoint_every)
        idata = reference.sample_deterministic_reference(
            log_likelihood, task["prior_bounds"], task["parameter_names"], draws=args.draws,
            tune=args.tune, chains=args.chains, seed=args.seed, progressbar=not args.no_progress,
            cores=args.cores, mp_context=args.mp_context, callback=partial, initialization=initialization)
        arrays = {"parameters": idata["posterior"]["parameters"].transpose("chain", "draw", "parameter").values}
        for name, value in idata["sample_stats"].data_vars.items():
            if set(value.dims) == {"chain", "draw"}:
                arrays[f"sample_stats_{name}"] = value.transpose("chain", "draw").values
        np.savez_compressed(args.out / "chains.npz", **arrays)
        diagnostics = reference.chain_diagnostics(idata)
        if task["model"] == "hkde":
            diagnostics["variance_functionals"] = reference.variance_functional_diagnostics(idata, task)
        else:
            diagnostics["derived_quantities"] = reference.rough_derived_diagnostics(idata, task)
            diagnostics["parameter_mcmc_diagnostics_passed"] = diagnostics["mcmc_diagnostics_passed"]
            diagnostics["mcmc_diagnostics_passed"] = bool(
                diagnostics["parameter_mcmc_diagnostics_passed"]
                and diagnostics["derived_quantities"]["mcmc_diagnostics_passed"])
            diagnostics["target_scope"] = "approximate emulator posterior, not independently validated model prices"
        if pricer_identity is not None:
            for filename, expected in pricer_identity["input_sha256"].items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                    raise ValueError("bound emulator files changed during reference sampling")
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        _write_json(args.out / "diagnostics.json", diagnostics)
        print(json.dumps({"out": str(args.out), "shape": list(arrays["parameters"].shape),
                          "mcmc_diagnostics_passed": diagnostics["mcmc_diagnostics_passed"]}))
    except Exception as error:
        _write_json(args.out / "failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()

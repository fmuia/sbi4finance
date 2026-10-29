"""Read the shipped BayesFlow 2.0.14 Heston estimator on three stored references.

No bank, pricing, fitting, MCMC or notebook execution is performed. The full
serialized adapter/standardizer is restored in its original BayesFlow version.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "JAX_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                         "--xla_cpu_parallel_codegen_split_count=1 intra_op_parallelism_threads=1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from sbi4finance.calibration import inference, validation
from sbi4finance.gbm_bank import _json, _write_json
from compare_full_surface_reference import _reference_issues, _variance_product_comparison

MODEL_SHA = "2d8a8a700f8c8a27c60e777656df14769b5e03e9e47ad717ffb21e1bffb2055e"
TASK_SHA = "5e499526ff8f29a72f49785a1a36006b1d769d26bca8a6f83f7a8c8ac4d431fa"
DRAW_COUNT = 10000
SEED_ROOT = 271103


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_archive(checkpoint):
    """Check shipped task and current stripped artifact against historical provenance."""
    task = json.loads((checkpoint / "task.json").read_text())
    training = json.loads((checkpoint / "training.json").read_text())
    accepted = json.loads((checkpoint / "ACCEPTED.json").read_text())
    if (metadata.version("bayesflow") != "2.0.14" or digest(checkpoint / "model.keras") != MODEL_SHA
            or hashlib.sha256(_json(task).encode()).hexdigest() != TASK_SHA
            or training["task_fingerprint"] != TASK_SHA
            or training["checkpoint_sha256"] != accepted["source_sha256"]["model.keras"]
            or training["training"]["time_power_law_alpha"] != -0.5
            or training["training"]["subnet_widths"] != [256] * 4
            or training["training"]["sampler"] != inference.SAMPLER_DESCRIPTION):
        raise ValueError("unchanged shipped Heston checkpoint and BayesFlow2.0.14 runtime required")
    for filename in ("training.json", "task.json"):
        if digest(checkpoint / filename) != accepted["source_sha256"][filename]:
            raise ValueError("archived metadata no longer matches its accepted source record")
    return task, training, accepted


def check_reference(case, task):
    root = case / "reference"
    reference = json.loads((root / "reference.json").read_text())
    diagnostics = json.loads((root / "diagnostics.json").read_text())
    precision = json.loads((case / "precision/precision.json").read_text())
    hashes = {name: digest(root / name) for name in ("reference.json", "chains.npz", "diagnostics.json")}
    names = task["parameter_names"]
    if (reference["task"] != task or reference["format"] != "full_surface_pymc_reference_v1"
            or reference["target"] != "independent Gaussian quote noise around deterministic refined Fourier prices"
            or _reference_issues(diagnostics, names)
            or (root / "failure.json").exists()
            or precision["input_sha256"] != hashes
            or precision["all_points_priced"] is not True
            or precision["pricing_sources_match_reference"] is not True
            or precision["source_surface_id"] != reference["surface_id"]
            or precision["base_numerics"] != task["numerics"]):
        raise ValueError("exact matching and diagnostically checked Heston reference/precision inputs required")
    quotes = np.asarray(reference["observed_prices"], dtype=float)
    inference.conditioning_data(quotes[None, :], None, task)
    with np.load(root / "chains.npz", allow_pickle=False) as archive:
        chains = archive["parameters"]
    bounds = np.asarray(task["prior_bounds"])
    if (chains.shape != (4, 8000, len(names)) or not np.isfinite(chains).all()
            or np.any(chains < bounds[:, 0]) or np.any(chains > bounds[:, 1])):
        raise ValueError("the completed reviewed four-chain reference is required")
    return reference, diagnostics, precision, chains, hashes


def cpu_seconds():
    used = resource.getrusage(resource.RUSAGE_SELF)
    return used.ru_utime + used.ru_stime


def stop_limit(signum, frame):
    raise TimeoutError(f"bounded archived comparison hit signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "tutorial_final/checkpoints/heston")
    parser.add_argument("--references", type=Path, default=ROOT / "docs/pilot-results/2026-10-27/heston-reviewed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wall_start = time.perf_counter()
    # Limits count process CPU across all threads. Soft CPU limit leaves time
    # to save a failure; the absolute hard process limit is one CPU-hour.
    resource.setrlimit(resource.RLIMIT_CPU, (3590, 3600))
    signal.signal(signal.SIGXCPU, stop_limit)
    signal.signal(signal.SIGPROF, stop_limit)
    signal.signal(signal.SIGALRM, stop_limit)
    signal.setitimer(signal.ITIMER_PROF, max(1., 3590 - cpu_seconds()))
    signal.alarm(3600)
    args.out.mkdir(parents=True, exist_ok=False)
    run = {"format": "archived_heston_three_reference_comparison_v1", "status": "preflight",
           "scientific_acceptance": False, "no_pricing_or_mcmc_or_bank_evaluation": True,
           "draws_per_case": DRAW_COUNT, "seed_root": SEED_ROOT, "cpu_hard_limit_seconds": 3600,
           "wall_limit_seconds": 3600, "cases": []}
    try:
        task, training, accepted = check_archive(args.checkpoint)
        inputs = [check_reference(args.references / f"case-{i}", task) for i in range(3)]
        seeds = [int(np.random.SeedSequence([0x41524348, SEED_ROOT, i]).generate_state(1)[0]) for i in range(3)]
        if len(set(seeds)) != 3:
            raise ValueError("reduced latent seed collision")
        # These are Keras/JAX latent streams, not any pricing/PCG64 stream.
        # Audit all earlier recorded scalar seeds before using this namespace.
        old_seeds = set()
        def collect(value, selected=False):
            if isinstance(value, dict):
                for key, item in value.items():
                    collect(item, selected or "seed" in key.lower())
            elif isinstance(value, list):
                for item in value:
                    collect(item, selected)
            elif selected and type(value) is int:
                old_seeds.add(value)
        for path in (ROOT / "docs/pilot-results").rglob("*.json"):
            collect(json.loads(path.read_text()))
        if any(seed in old_seeds for seed in seeds) or SEED_ROOT in old_seeds:
            raise ValueError("new archived namespace collides with a recorded previous scalar seed")
        run.update(status="declared", task_fingerprint=TASK_SHA, task=task,
                   checkpoint_sha256={name: digest(args.checkpoint / name) for name in
                                      ("model.keras", "task.json", "training.json", "ACCEPTED.json")},
                   original_training_checkpoint_sha256=training["checkpoint_sha256"],
                   archive_changes=accepted["changes"], archived_training_versions=training["versions"],
                   versions={name: metadata.version(name) for name in
                             ("bayesflow", "keras", "jax", "jaxlib", "numpy", "scipy")},
                   python=platform.python_version(), platform=platform.platform(),
                   thread_environment={key: os.environ[key] for key in os.environ
                                       if key.endswith("NUM_THREADS") or key in
                                       ("KERAS_BACKEND", "JAX_PLATFORMS", "XLA_FLAGS")},
                   sampling={"description": inference.SAMPLER_DESCRIPTION,
                             "time_power_law_alpha": training["training"]["time_power_law_alpha"],
                             "seeds": seeds, "seed_sequence_entropy": [[0x41524348, SEED_ROOT, i] for i in range(3)],
                             "audit": "new scalar seeds and root disjoint from retained prior seed values; actual Keras/JAX initial states checked before drawing; no pricing RNG initialized"},
                   reference_input_sha256=[x[-1] for x in inputs],
                   source_sha256={name: digest(path) for name, path in
                                  (("compare_archived_heston.py", __file__),
                                   ("calibration/inference.py", inference.__file__),
                                   ("calibration/validation.py", validation.__file__),
                                   ("compare_full_surface_reference.py", ROOT / "scripts/compare_full_surface_reference.py"))})
        _write_json(args.out / "declaration.json", run)  # Never overwritten; before loading/sampling.
        run["declaration_sha256"] = digest(args.out / "declaration.json")
        _write_json(args.out / "run.json", run)
        import keras
        import jax
        states = [np.asarray(keras.random.SeedGenerator(seed).state).tolist() for seed in seeds]
        if len({tuple(state) for state in states}) != 3 or any(device.platform != "cpu" for device in jax.devices()):
            raise ValueError("distinct actual Keras initial states and CPU-only backend required")
        run["sampling"]["keras_jax_initial_states"] = states
        run["jax_devices"] = [str(device) for device in jax.devices()]
        run["status"] = "loading"
        _write_json(args.out / "run.json", run)
        workflow = inference.build_workflow(task, subnet_widths=tuple(training["training"]["subnet_widths"]),
                                            time_power_law_alpha=-0.5)
        workflow.load_approximator(args.checkpoint / "model.keras")
        if workflow.approximator.inference_network.get_config()["time_power_law_alpha"] != -0.5:
            raise ValueError("serialized archived FlowMatching time law changed")
        for i, (reference, diagnostics, precision, chains, hashes) in enumerate(inputs):
            before = time.perf_counter()
            neural = inference.sample(workflow, task, np.asarray(reference["observed_prices"])[None, :],
                                      None, DRAW_COUNT, seed=seeds[i])[0]
            if (neural.shape != (DRAW_COUNT, len(task["parameter_names"])) or not np.isfinite(neural).all()
                    or np.any(neural < np.asarray(task["prior_bounds"])[:, 0])
                    or np.any(neural > np.asarray(task["prior_bounds"])[:, 1])):
                raise ValueError("invalid inverse-adapted archived posterior draws")
            agreement = validation.agreement(neural, chains.reshape(-1, chains.shape[-1]))
            parameters = {name: {"standardized_mean_error": agreement["standardized_mean_error"][j],
                                 "width_ratio_90": agreement["width_ratio"][j]} for j, name in enumerate(task["parameter_names"])}
            passed = all(row["standardized_mean_error"] <= .3 and abs(row["width_ratio_90"] - 1) <= .25
                         for row in parameters.values())
            row = {"format": "archived_heston_case_comparison_v1", "scientific_acceptance": False,
                   "case_index": i, "surface_id": reference["surface_id"], "sampling_seed": seeds[i],
                   "sampling_seconds": time.perf_counter() - before,
                   "reference_input_sha256": hashes, "observed_prices": reference["observed_prices"],
                   "proposed_parameter_agreement": {"all_checks_passed": passed, "parameters": parameters},
                   "variance_product_comparison": _variance_product_comparison(task, neural, chains, diagnostics),
                   "scope": "descriptive development comparison; historical 50k training differs from new20k fits"}
            dest = args.out / f"case-{i}"
            dest.mkdir()
            np.savez_compressed(dest / "neural_draws.npz", parameters=neural)
            _write_json(dest / "comparison.json", row)
            run["cases"].append(row)
            run.update(status="running", cpu_seconds=cpu_seconds(), wall_seconds=time.perf_counter() - wall_start)
            _write_json(args.out / "run.json", run)
        run["status"] = "completed_development_comparison"
        for name, sha in run["checkpoint_sha256"].items():
            if digest(args.checkpoint / name) != sha:
                raise ValueError("checkpoint changed during evaluation")
        for i in range(3):
            for name, sha in run["reference_input_sha256"][i].items():
                if digest(args.references / f"case-{i}/reference" / name) != sha:
                    raise ValueError("reference changed during evaluation")
    except BaseException as error:
        run.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        signal.setitimer(signal.ITIMER_PROF, 0)
        run.update(cpu_seconds=cpu_seconds(), wall_seconds=time.perf_counter() - wall_start)
        _write_json(args.out / "run.json", run)
        print(json.dumps({k: run[k] for k in ("status", "cpu_seconds", "wall_seconds")}))


if __name__ == "__main__":
    main()

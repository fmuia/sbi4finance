"""One bounded PMMH feasibility chain for a validation-history Heston prefix.

This specialized atom-aware particle sampler is not PyMC or an accepted
reference. Submit separate --chain identities to assess cross-chain mixing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numba
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import bank  # noqa: E402
from sbi4finance.path_inference.pmmh import (  # noqa: E402
    load_initial_checkpoint, particle_log_likelihood, prior_logit_to_physical,
    proposal_cholesky, run_chain, validate_reference_task,
)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_case(path, case, length):
    manifest = bank.load_manifest(path)
    task = manifest["task"]
    validate_reference_task(task)
    if not 0 <= case < manifest["split_counts"]["validation"]:
        raise ValueError("case-index must identify an existing validation trajectory")
    if not task["min_history"] <= length <= task["n_observations"]:
        raise ValueError("length lies outside the declared task history range")
    for record in manifest["splits"]["validation"]:
        if record["start"] <= case < record["start"] + record["count"]:
            arrays = bank._load_part(path, manifest, "validation", record)
            row = case - record["start"]
            return task, arrays["observations"][row, :length, 0].copy(), {
                "split": "validation", "reserved_test_evaluated": False,
                "case_index": case, "trajectory_id": str(arrays["trajectory_id"][row]),
                "source_shard": record["path"], "source_shard_sha256": record["sha256"],
                "task_sha256": manifest["task_sha256"], "length": length,
                "elapsed_years": length * task["dt_years"], "truth_used_for_initialization_or_likelihood": False}
    raise RuntimeError("validation shard not found")


def _save_phase(path, values, task, chain):
    physical = np.asarray([prior_logit_to_physical(z, task) for z in values["z"]])
    if not len(physical):
        physical = np.empty((0, len(task["parameter_names"])))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, parameters=physical[None],
                            **{k: v[None] for k, v in values.items()}, chain_id=np.array([chain]))
    temporary.replace(path)


def _load_proposal(path, task, observation):
    if path is None:
        return np.eye(len(task["parameter_names"])), None
    config = json.loads(Path(path).read_text())
    if (config.get("schema") != "path-pmmh-proposal-covariance-v1"
            or config.get("coordinates") != "prior_cdf_logits"
            or config.get("task_sha256") != bank.task_fingerprint(task)
            or config.get("parameter_names") != task["parameter_names"]
            or config.get("length") != observation["length"]
            or config.get("trajectory_id") != observation["trajectory_id"]
            or config.get("source_shard_sha256") != observation["source_shard_sha256"]):
        raise ValueError("proposal covariance task/coordinates/observation identity differs")
    matrix = np.asarray(config["covariance"], dtype=float)
    proposal_cholesky(matrix, len(task["parameter_names"]))
    return matrix, {"path": str(path), "sha256": _sha256(path), "configuration": config}


def _noise_only(args, task, returns, observation, checkpoint):
    """Independent PF replicates at a saved point, never a changed-count chain."""
    if any(n < 2 for n in args.noise_particle_counts):
        raise ValueError("noise particle counts must be >=2")
    z, accepted, provenance = checkpoint
    physical = prior_logit_to_physical(z, task)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "observations.npz", log_returns=returns,
                        dt_years=np.full(len(returns), task["dt_years"]))
    started = time.perf_counter()
    arrays, runs = {}, []
    for particles in sorted(set(args.noise_particle_counts)):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [args.seed, args.phase_seed, args.chain, args.length, particles, 0x4e4f4953])))
        seeds = rng.integers(0, np.iinfo(np.uint64).max, args.noise_repeats, dtype=np.uint64)
        estimates, seconds = [], []
        for seed in seeds:
            tick = time.perf_counter()
            estimates.append(particle_log_likelihood(returns, task, physical,
                particles=particles, seed=int(seed)))
            seconds.append(time.perf_counter() - tick)
        values = np.asarray([item.log_value for item in estimates])
        finite = values[np.isfinite(values)]
        arrays[f"particles_{particles}_log_likelihood"] = values
        arrays[f"particles_{particles}_seeds"] = seeds
        arrays[f"particles_{particles}_seconds"] = np.asarray(seconds)
        runs.append({"particles": particles, "repeats": args.noise_repeats,
                     "collapse_count": int(np.sum(~np.isfinite(values))),
                     "finite_log_likelihood_mean": float(finite.mean()) if len(finite) else None,
                     "finite_log_likelihood_sd": float(finite.std(ddof=1)) if len(finite) > 1 else None,
                     "seconds_total": float(np.sum(seconds)), "seconds_median": float(np.median(seconds)),
                     "minimum_ancestor_ess": min(item.minimum_ess for item in estimates)})
        print(f"chain {args.chain}: {args.noise_repeats} independent likelihoods at {particles} particles", flush=True)
    np.savez_compressed(args.out / "likelihood-noise.npz", **arrays)
    report = {"schema": "path-pmmh-fixed-point-noise-v1", "status": "completed_noise_diagnostic_only",
              "scientific_acceptance": False, "sampling_performed": False,
              "old_likelihood_reused_for_new_particle_count": False,
              "initial_checkpoint": provenance, "task": task, "observation": observation,
              "chain_id": args.chain, "root_seed": args.seed, "phase_seed": args.phase_seed,
              "point_z": z.tolist(), "point_physical": physical.tolist(),
              "source_accepted_log_likelihood": accepted.log_value,
              "runs": runs, "seconds": time.perf_counter() - started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "numba": numba.__version__},
              "first_evaluation_may_include_numba_compilation": True,
              "interpretation": "Independent finite-repeat likelihood-noise/runtime measurements at the saved point. Finite-log SD excludes collapsed zero estimates, which are counted separately. No new chain or reference-accuracy claim.",
              "source_sha256": {"pmmh.py": _sha256(ROOT / "src/sbi4finance/path_inference/pmmh.py"),
                                "path_pmmh.py": _sha256(Path(__file__))},
              "files": [{"file": name, "sha256": _sha256(args.out / name)} for name in
                        ("observations.npz", "likelihood-noise.npz")]}
    _write_json(args.out / "noise-summary.json", report)
    return report


def run(args):
    for key, minimum in (("case_index", 0), ("chain", 0), ("tune", 0), ("draws", 1),
                         ("particles", 2), ("noise_repeats", 2), ("checkpoint_every", 1),
                         ("initial_attempts", 1), ("phase_seed", 0), ("seed", 0)):
        if getattr(args, key) < minimum:
            raise ValueError(f"{key} must be >= {minimum}")
    task, returns, observation = _load_case(args.bank, args.case_index, args.length)
    checkpoint_state = None
    if args.initial_checkpoint is not None:
        checkpoint_state = load_initial_checkpoint(args.initial_checkpoint, task, returns, observation,
            particles=None if args.noise_only else args.particles)
    if args.noise_only:
        if checkpoint_state is None:
            raise ValueError("--noise-only requires --initial-checkpoint")
        return _noise_only(args, task, returns, observation, checkpoint_state)
    proposal_covariance, proposal_provenance = _load_proposal(args.proposal_covariance, task, observation)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "observations.npz", log_returns=returns,
                        dt_years=np.full(len(returns), task["dt_years"]))
    chain_seed = int(np.random.SeedSequence([args.seed, 0x50484153, args.phase_seed,
                                           args.case_index, args.length, args.chain, 0])
                     .generate_state(1, dtype=np.uint64)[0])
    if checkpoint_state is not None and chain_seed == checkpoint_state[2]["source_chain_seed"]:
        raise ValueError("new phase must use a fresh RNG identity")
    initial_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([args.seed, args.chain, 1])))
    # Prior-based dispersion; no synthetic labels or neural results are used.
    initial_u = np.clip(initial_rng.random(len(task["parameter_names"])),
                        np.finfo(float).eps, 1 - np.finfo(float).eps)
    initial_z = np.log(initial_u) - np.log1p(-initial_u)

    def estimator(z, seed):
        return particle_log_likelihood(returns, task, prior_logit_to_physical(z, task),
                                       particles=args.particles, seed=seed)

    started = time.perf_counter()
    report = {"schema": "heston-path-pmmh-feasibility-v1", "task": task,
              "observation": observation, "chain_id": args.chain, "root_seed": args.seed,
              "chain_seed": chain_seed, "tune": args.tune, "draws": args.draws,
              "phase_seed": args.phase_seed, "exact_resume": False,
              "execution_mode": "new_phase_from_accepted_extended_state" if checkpoint_state is not None else "new_prior_initialized_phase",
              "particles": args.particles, "status": "running_not_accepted_reference",
              "scientific_acceptance": False,
              "method": "particle marginal random-walk Metropolis-Hastings with independent fresh particle likelihood proposals in prior-CDF logits",
              "reason_for_specialized_method": "unbiased atom-aware particle likelihood for signed full-truncation substeps; no differentiable deterministic NUTS target supplied",
              "framework": "custom atom-aware PMMH; not PyMC",
              "target": "same independent priors and declared-substep discrete P Heston law as the bank; latent paths integrated by an unbiased particle likelihood estimate",
              "particle_filter": ("adapted one-step conditional transition" if task["numerics"]["substeps"] == 1 else
                                  "bootstrap variance Brownian substep paths; integrate independent asset Brownian noise out within each observation; systematic terminal signed-state resampling"),
              "particle_rng": ("PCG64 independent resampling and transition streams" if task["numerics"]["substeps"] == 1 else
                               "Numba MT19937 with uint32 seed derived from retained uint64 PF seed through SeedSequence"),
              "transformation": "inverse-prior-CDF(sigmoid(z)); log prior in z = sum(logsigmoid(z)+logsigmoid(-z))",
              "adaptation": "Robbins-Monro scalar scale during discarded tuning only; frozen for retained draws" if not args.no_adapt else "fixed proposal scale throughout",
              "proposal_scale_initial": args.proposal_scale,
              "proposal_covariance": proposal_covariance.tolist(),
              "proposal_covariance_provenance": proposal_provenance,
              "proposal_geometry": "fixed symmetric positive-definite covariance in prior-CDF logits; only scalar scale adapts during discarded tuning",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "numba": numba.__version__}}
    report["source_sha256"] = {str(path.relative_to(ROOT)): _sha256(path) for path in (
        Path(__file__).resolve(), ROOT / "src/sbi4finance/path_inference/pmmh.py",
        ROOT / "src/sbi4finance/path_inference/reference.py", ROOT / "src/sbi4finance/path_inference/bank.py",
        ROOT / "src/sbi4finance/heston.py")}
    _write_json(args.out / "pmmh-summary.json", report)

    def checkpoint(iterations, arrays, state):
        if iterations % args.checkpoint_every and iterations != args.tune + args.draws:
            return
        tune_count = min(iterations, args.tune)
        _save_phase(args.out / "tune.npz", {k: v[:tune_count] for k, v in arrays.items()}, task, args.chain)
        _save_phase(args.out / "draws.npz", {k: v[args.tune:iterations] for k, v in arrays.items()}, task, args.chain)
        report.update(completed_iterations=iterations, seconds=time.perf_counter() - started,
                      sampler_state=state)
        _write_json(args.out / "pmmh-summary.json", report)
        print(f"chain {args.chain}: {iterations}/{args.tune + args.draws} iterations", flush=True)

    try:
        screening = []
        initial_estimate = None
        for attempt in range(0 if checkpoint_state is not None else args.initial_attempts):
            if attempt:
                initial_u = np.clip(initial_rng.random(len(task["parameter_names"])),
                                    np.finfo(float).eps, 1 - np.finfo(float).eps)
                initial_z = np.log(initial_u) - np.log1p(-initial_u)
            initial_seed = int(initial_rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            candidate = estimator(initial_z, initial_seed)
            screening.append({"attempt": attempt, "likelihood_seed": initial_seed,
                              "physical_parameters": prior_logit_to_physical(initial_z, task).tolist(),
                              "log_likelihood_estimate": candidate.log_value if np.isfinite(candidate.log_value) else None,
                              "collapsed": candidate.collapse})
            if (np.isfinite(candidate.log_value)
                    and (initial_estimate is None or candidate.log_value > initial_estimate.log_value)):
                initial_estimate = candidate
                selected_initial_z = initial_z.copy()
        report["initialization_screening"] = screening
        report["initialization_rule"] = "bounded independent uniform prior-CDF positions, highest finite estimated likelihood among screened candidates; machine-epsilon endpoint guard for initialization only; target prior unchanged"
        if checkpoint_state is not None:
            selected_initial_z, initial_estimate, provenance = checkpoint_state
            report["initial_checkpoint"] = provenance
            report["initialization_rule"] = "validated last accepted parameter/log-likelihood/PF-seed state preserved; same task/observations/particle count; fresh explicitly salted phase RNG; no exact resume claim"
        if initial_estimate is None:
            raise RuntimeError("bounded initialization found no positive particle likelihood; increase particles or inspect atom support")
        initial_z = selected_initial_z
        result = run_chain(estimator, initial_z, tune=args.tune, draws=args.draws, seed=chain_seed,
                           proposal_scale=args.proposal_scale, adapt=not args.no_adapt, callback=checkpoint,
                           initial_estimate=initial_estimate, proposal_covariance=proposal_covariance)
        retained = np.asarray([prior_logit_to_physical(z, task) for z in result["draws"]["z"]])
        points = {"initial": prior_logit_to_physical(initial_z, task),
                  "posterior_coordinate_median": np.median(retained, axis=0)}
        noises = []
        noise_arrays = {}
        for index, (name, point) in enumerate(points.items()):
            noise_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([chain_seed, index, 2])))
            seeds = noise_rng.integers(0, np.iinfo(np.uint64).max, args.noise_repeats, dtype=np.uint64)
            estimates = [particle_log_likelihood(returns, task, point, particles=args.particles,
                                                 seed=int(seed)) for seed in seeds]
            values = np.array([x.log_value for x in estimates])
            finite = values[np.isfinite(values)]
            noise_arrays[f"{name}_log_likelihood_estimates"] = values
            noise_arrays[f"{name}_seeds"] = seeds
            noises.append({"point": name, "physical_parameters": point.tolist(),
                           "repeats": args.noise_repeats, "collapse_count": int(np.sum(~np.isfinite(values))),
                           "finite_log_likelihood_mean": float(finite.mean()) if len(finite) else None,
                           "finite_log_likelihood_sd": float(finite.std(ddof=1)) if len(finite) > 1 else None,
                           "interpretation": "pointwise finite-repeat noise diagnostic; does not establish mixing or posterior accuracy"})
        np.savez_compressed(args.out / "likelihood-noise.npz", **noise_arrays)
        report.update(status="completed_feasibility_only_not_accepted_reference", seconds=time.perf_counter() - started,
                      initial=result["initial"], final_rng_state=result["final_rng_state"],
                      proposal_scale_final=result["final_proposal_scale"],
                      acceptance_tune=float(result["tune"]["accepted"].mean()) if args.tune else None,
                      acceptance_draws=float(result["draws"]["accepted"].mean()),
                      collapsed_draw_proposals=int(result["draws"]["proposal_collapsed"].sum()),
                      physical_mean=retained.mean(axis=0).tolist(),
                      physical_sd=retained.std(axis=0).tolist(),
                      physical_q05=np.quantile(retained, 0.05, axis=0).tolist(),
                      physical_q95=np.quantile(retained, 0.95, axis=0).tolist(),
                      likelihood_noise=noises,
                      interpretation="Combine independent chain files and examine rank Rhat, bulk/tail ESS, repeat-particle precision and SBI agreement before acceptance. A short completed chain is feasibility evidence only.")
        report["files"] = [{"file": name, "sha256": _sha256(args.out / name)} for name in
                           ("observations.npz", "tune.npz", "draws.npz", "likelihood-noise.npz")]
        _write_json(args.out / "pmmh-summary.json", report)
    except Exception as error:
        report.update(status="failed_no_reference_accepted", error=repr(error), seconds=time.perf_counter() - started)
        _write_json(args.out / "pmmh-summary.json", report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--length", type=int, default=252)
    parser.add_argument("--chain", type=int, default=0)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261030091)
    parser.add_argument("--proposal-scale", type=float, default=0.35)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--noise-repeats", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--initial-attempts", type=int, default=64)
    parser.add_argument("--proposal-covariance", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--phase-seed", type=int, default=1)
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("--noise-particle-counts", type=int, nargs="+", default=[512, 1024, 2048])
    report = run(parser.parse_args())
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "acceptance_draws": report.get("acceptance_draws")}, indent=2))


if __name__ == "__main__":
    main()

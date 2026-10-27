"""Native adaptive FlowMatching on three frozen Heston observations only.

Run inside one15-minute4-CPU/GPU allocation. Inputs/defaults are frozen before
loading the cached model. The Slurm limit is the hard outer wall cap; partial
case files remain development evidence when a solver call is interrupted.
"""

import argparse
import ast
import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))
from scripts import compare_full_surface_reference as comparison  # noqa: E402
from sbi4finance.calibration import full_surface, validation, variance_functionals  # noqa: E402
from sbi4finance.gbm_bank import _json  # noqa: E402

RECIPE = ROOT/"configs/heston_cached_native_solver.json"
OLD = ROOT/"docs/pilot-results/2026-10-27/followup/heston/flow-matching-500"
REFERENCES = ROOT/"docs/pilot-results/2026-10-27/heston-reviewed"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_settings():
    """Read this pinned library's defaults without importing/initializing a model."""
    if metadata.version("bayesflow") != "2.1.0":
        raise ValueError("the locked BayesFlow2.1 runtime is required")
    package = Path(metadata.distribution("bayesflow").locate_file("bayesflow"))
    defaults, integrator = package/"networks/defaults.py", package/"utils/integrate.py"
    assignment = next(node for node in ast.parse(defaults.read_text()).body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "FLOW_MATCHING_INTEGRATE_DEFAULTS"
                              for target in node.targets))
    settings = ast.literal_eval(assignment.value.args[0])
    if settings != {"method": "tsit5", "steps": "adaptive"}:
        raise ValueError("unexpected installed native FlowMatching policy; do not substitute another solver")
    tree = ast.parse(integrator.read_text())
    integrate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "integrate")
    signature = dict(zip([arg.arg for arg in integrate.args.args][-len(integrate.args.defaults):], integrate.args.defaults))
    settings.update({name: ast.literal_eval(signature[name]) for name in ("min_steps", "max_steps")})
    adaptive = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "integrate_adaptive")
    for node in ast.walk(adaptive):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "kwargs" and len(node.args) == 2):
            key = ast.literal_eval(node.args[0])
            if key == "atol":
                settings[key] = ast.literal_eval(node.args[1])
            elif key == "rtol":
                if not isinstance(node.args[1], ast.IfExp) or ast.unparse(node.args[1].test) != "len(state) == 1":
                    raise ValueError("unrecognized sampling-versus-density tolerance rule")
                settings[key] = ast.literal_eval(node.args[1].body)
    if not {"method", "steps", "min_steps", "max_steps", "atol", "rtol"} <= settings.keys():
        raise ValueError("could not identify all installed native sampling defaults")
    return settings, {str(path): digest(path) for path in (defaults, integrator,
        package/"networks/inference/flow_matching/flow_matching.py", package/"networks/inference/inference_network.py",
        package/"workflows/basic_workflow.py", package/"estimators/basic_estimator.py")}


def checked_inputs(euler_run, references, recipe):
    checkpoint = euler_run/"checkpoint"
    training = json.loads((checkpoint/"training.json").read_text())
    task = json.loads((checkpoint/"task.json").read_text())
    if (digest(checkpoint/"model.keras") != recipe["checkpoint_sha256"]
            or digest(checkpoint/"training.json") != recipe["training_metadata_sha256"]
            or hashlib.sha256(_json(task).encode()).hexdigest() != recipe["task_fingerprint"]
            or training["task_fingerprint"] != recipe["task_fingerprint"]):
        raise ValueError("the exact completed20k/500-epoch FlowMatching checkpoint is required")
    full_surface.validate_task(task)
    environment = {name: digest(ROOT/name) for name in ("pyproject.toml", "pixi.lock")}
    runtime = {"python": platform.python_version(), "keras_backend_env": os.environ.get("KERAS_BACKEND"),
               **{name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}}
    if environment != training["run_contract"]["environment_sha256"] or runtime != training["run_contract"]["runtime"]:
        raise ValueError("the original locked BF2.1 runtime/environment and JAX backend are required")
    hashes = {str(checkpoint/name): digest(checkpoint/name) for name in ("model.keras", "training.json", "task.json")}
    cases = []
    for index in recipe["case_indices"]:
        old, case = euler_run/f"comparisons/case-{index}", references/f"case-{index}"
        reference = json.loads((case/"reference/reference.json").read_text())
        diagnostics = json.loads((case/"reference/diagnostics.json").read_text())
        precision = json.loads((case/"precision/precision.json").read_text())
        previous = json.loads((old/"comparison.json").read_text())
        files = ("reference/reference.json", "reference/diagnostics.json", "reference/chains.npz", "precision/precision.json")
        if (digest(old/"neural_draws.npz") != recipe["euler_draw_sha256"][index]
                or digest(old/"comparison.json") != digest(OLD/f"comparisons/case-{index}/comparison.json")
                or any(digest(case/name) != digest(REFERENCES/f"case-{index}"/name) for name in files)
                or reference["task"] != task or reference["surface_id"] != f"validation:{index}"
                or previous["sampling_seed"] != recipe["seed_base"]+index
                or previous["neural_draws_requested"] != recipe["draws_per_case"]
                or comparison._reference_issues(diagnostics, task["parameter_names"])
                or diagnostics.get("variance_functionals", {}).get("mcmc_diagnostics_passed") is not True
                or precision.get("all_points_priced") is not True or precision.get("pricing_sources_match_reference") is not True):
            raise ValueError(f"case{index} must use the exact reviewed reference and frozen Euler comparison/draws")
        for name in files:
            hashes[str(case/name)] = digest(case/name)
        for name in ("comparison.json", "neural_draws.npz"):
            hashes[str(old/name)] = digest(old/name)
        cases.append({"index": index, "seed": recipe["seed_base"]+index, "surface_id": reference["surface_id"],
                      "observation_bank_manifest_sha256": reference["bank_manifest_sha256"],
                      "observed_prices": reference["observed_prices"], "reference": str(case), "euler": str(old)})
    return {"checkpoint": str(checkpoint), "task": task, "cases": cases, "sha256": hashes,
            "runtime": runtime, "environment_sha256": environment,
            "scope": "exact three original readiness observations; no20k-bank validation/test arrays opened"}


def native_draws(workflow, task, quotes, count, seed):
    stored = copy.deepcopy(workflow.inference_network.integrate_kwargs)
    if stored.get("method") != "euler" or stored.get("steps") != 100:
        raise ValueError("the restored cached checkpoint must retain its stored Euler100 configuration")
    result = workflow.sample(num_samples=count, conditions=full_surface.conditioning_data(quotes[None, :], task),
                             sample_shape=(), seed=seed, method="tsit5", steps="adaptive")
    if workflow.inference_network.integrate_kwargs != stored:
        raise RuntimeError("call-time native overrides mutated the stored integrator configuration")
    draws = np.concatenate([np.asarray(result[name], dtype=np.float64) for name in task["parameter_names"]], axis=-1)
    bounds = np.asarray(task["prior_bounds"])
    if (draws.shape != (1, count, len(bounds)) or not np.isfinite(draws).all()
            or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
        raise ValueError("native sampling returned invalid physical posterior draws")
    return draws[0]


def score(task, draws, euler, chains, diagnostics):
    reference = chains.reshape(-1, chains.shape[-1])
    agreement = validation.agreement(draws, reference)
    names, rows, paired = task["parameter_names"], {}, {}
    for j, name in enumerate(names):
        error, width = agreement["standardized_mean_error"][j], agreement["width_ratio"][j]
        rows[name] = {"standardized_mean_error": error, "width_ratio_90": width,
                      "mean_error_within_0_3_reference_sd": bool(error <= .3),
                      "relative_width_error_within_25_percent": bool(abs(width-1) <= .25),
                      "reference_mean": float(reference[:, j].mean()), "reference_sd": float(reference[:, j].std()),
                      "neural_mean": float(draws[:, j].mean()),
                      "reference_q05_q95": np.quantile(reference[:, j], [.05, .95]).tolist(),
                      "neural_q05_q95": np.quantile(draws[:, j], [.05, .95]).tolist(),
                      "reference_diagnostics": diagnostics.get("parameters", {}).get(f"parameters[{name}]", {})}
        delta, sd = draws[:, j]-euler[:, j], reference[:, j].std()
        paired[name] = {"signed_mean_change_over_reference_sd": float(delta.mean()/sd),
                        "paired_rms_change_over_reference_sd": float(np.sqrt(np.mean(delta**2))/sd),
                        "paired_abs_change_q50_q90_over_reference_sd": (np.quantile(abs(delta), [.5, .9])/sd).tolist(),
                        "signed_q05_q95_change_over_reference_sd": ((np.quantile(draws[:, j], [.05, .95])-
                                                                    np.quantile(euler[:, j], [.05, .95]))/sd).tolist()}
    products = comparison._variance_product_comparison(task, draws, chains, diagnostics)
    native_products = variance_functionals.heston_variance_functionals(task, draws)
    old_products = variance_functionals.heston_variance_functionals(task, euler)
    return {"proposed_parameter_agreement": {"parameters": rows, "all_checks_passed": all(
                row["mean_error_within_0_3_reference_sd"] and row["relative_width_error_within_25_percent"] for row in rows.values())},
            "variance_product_comparison": products, "paired_parameter_changes_from_euler": paired,
            "paired_analytic_product_changes_from_euler": {name: {
                "signed_mean_change": float((value-old_products[name]).mean()),
                "paired_rms_change": float(np.sqrt(np.mean((value-old_products[name])**2))),
                "signed_q10_q90_change": (np.quantile(value, [.1, .9])-np.quantile(old_products[name], [.1, .9])).tolist()}
                for name, value in native_products.items()}}


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 4:
        raise ValueError("one4-CPU/GPU Slurm allocation is required; no login-node inference")
    recipe = json.loads(RECIPE.read_text())
    started = time.perf_counter()
    inputs = checked_inputs(args.euler_run, args.references, recipe)
    settings, library_sources = native_settings()
    sources = [Path(__file__), RECIPE, Path(full_surface.__file__), Path(validation.__file__),
               Path(variance_functionals.__file__), Path(comparison.__file__)]
    args.out.mkdir(parents=True, exist_ok=False)
    state = {"format": "heston_cached_native_solver_run_v1", "recipe": recipe, "status": "inputs_frozen",
             "scientific_acceptance": False, "inputs": inputs, "native_effective_settings": settings,
             "call_time_overrides": {key: settings[key] for key in ("method", "steps")},
             "library_source_sha256": library_sources, "source_sha256": {str(p): digest(p) for p in sources},
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "cases": [{"index": c["index"], "status": "pending"} for c in inputs["cases"]],
             "allocation_wall_cap_seconds": recipe["resources"]["wall_seconds"],
             "scope": "native solver sensitivity only; no new fit/pricing/MCMC,500-validation panel or acceptance claim"}
    write(args.out/"run.json", state)  # precedes environment initialization and model loading/sampling
    def interrupted(signum, frame):
        raise SystemExit(f"received signal{signum}; preserve partial cases, no automatic retry")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        import jax
        state["devices"] = [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()]
        if not any(d["platform"] == "gpu" for d in state["devices"]):
            raise ValueError("an actual JAX GPU is required inside the allocation")
        task, workflow, _ = full_surface.load_checkpoint(inputs["checkpoint"])
        if task != inputs["task"]:
            raise ValueError("loaded checkpoint task changed")
        for case, progress in zip(inputs["cases"], state["cases"]):
            if time.perf_counter()-started >= recipe["internal_wall_seconds"]:
                raise TimeoutError("internal wall allowance exhausted; no further case starts")
            progress["status"] = "running"
            state["status"] = f"case-{case['index']}"
            write(args.out/"run.json", state)
            before = time.perf_counter()
            draws = native_draws(workflow, task, np.asarray(case["observed_prices"]), recipe["draws_per_case"], case["seed"])
            root = Path(case["reference"])
            with np.load(root/"reference/chains.npz", allow_pickle=False) as archive:
                chains = archive["parameters"]
            with np.load(Path(case["euler"])/"neural_draws.npz", allow_pickle=False) as archive:
                euler = archive["parameters"]
            diagnostics = json.loads((root/"reference/diagnostics.json").read_text())
            report = {"format": "heston_native_solver_comparison_v1", "scientific_acceptance": False,
                      "surface_id": case["surface_id"], "sampling_seed": case["seed"],
                      "observation_bank_manifest_sha256": case["observation_bank_manifest_sha256"],
                      "native_effective_settings": settings, "seconds": time.perf_counter()-before,
                      "paired_design": "same fresh integer seed, observation and10k draw shape as saved Euler; deterministic solvers",
                      **score(task, draws, euler, chains, diagnostics)}
            destination = args.out/f"case-{case['index']}"
            destination.mkdir()
            np.savez_compressed(destination/"native_draws.npz", parameters=draws)
            write(destination/"comparison.json", report)
            progress.update(status="completed", seconds=report["seconds"],
                            proposed_parameter_checks_passed=report["proposed_parameter_agreement"]["all_checks_passed"])
            write(args.out/"run.json", state)
        if any(digest(Path(path)) != expected for path, expected in inputs["sha256"].items()):
            raise ValueError("frozen input bytes changed during the check")
        state.update(status="completed_development_comparisons", seconds=time.perf_counter()-started)
        write(args.out/"run.json", state)
        return state
    except BaseException as error:
        state.update(status="incomplete" if isinstance(error, (TimeoutError, SystemExit)) else "failed",
                     seconds=time.perf_counter()-started, failure={"type": type(error).__name__, "error": str(error)})
        write(args.out/"run.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("euler-run", "references", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()

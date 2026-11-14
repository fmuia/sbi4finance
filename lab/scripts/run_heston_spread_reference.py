"""Predeclare observations and run bounded native PyMC Heston spread references.

Run one case per process. Four chains run sequentially; CPU and wall limits
include initialization, both predeclared sampling attempts and diagnostics.
Partial traces are preserved but cannot resume sampler/adaptation/RNG state.
"""
import argparse
from contextlib import AbstractContextManager
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbi4finance.calibration.heston_spread import HestonSpreadPilot
from sbi4finance.calibration import reference
from sbi4f_tutorial.heston_options import model
from scripts.calibration_reference import PartialDrawWriter


CASE_IDS = ("synthetic-broad", "synthetic-market-like", "synthetic-boundary", "market")
BOUNDARY_TRUTH = [.0105, .12, .012, 2.9, -.94]
ATTEMPTS = ((2000, 2000), (6000, 12000))  # tuning, retained draws per chain
MAX_SECONDS = 2700


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def values_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def write_json(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def expected_variance(draws, interval):
    """Analytic average variance on fixed horizons; preserve all leading axes."""
    values = np.asarray(draws, dtype=float)
    start, end = np.asarray(interval, dtype=float)
    if (values.ndim < 1 or values.shape[-1] < 5 or not np.isfinite(values).all()
            or not np.isfinite([start, end]).all() or not 0 <= start < end
            or np.any(values[..., 1] <= 0)):
        raise ValueError("finite physical Heston draws and 0 <= start < end are required")
    v0, kappa, theta = (values[..., i] for i in range(3))
    weight = np.exp(-kappa * start) * (-np.expm1(-kappa * (end-start))) / (kappa * (end-start))
    return theta + (v0-theta) * weight


def product_values(draws, products):
    return np.stack([expected_variance(draws, interval) for interval in products.values()], axis=-1)


class Budget(AbstractContextManager):
    """Enforce one-process CPU plus elapsed limits, including BLAS threads."""

    def __init__(self, seconds=MAX_SECONDS):
        if not np.isfinite(seconds) or not 0 < seconds <= MAX_SECONDS:
            raise ValueError("budget must be positive and no more than 2700 seconds")
        self.seconds = float(seconds)

    def check(self):
        if time.monotonic()-self.wall_start >= self.seconds or time.process_time()-self.cpu_start >= self.seconds:
            raise TimeoutError("reference CPU or wall budget exhausted")

    def usage(self):
        return {"wall_seconds": time.monotonic()-self.wall_start,
                "cpu_seconds": time.process_time()-self.cpu_start, "limit_seconds": self.seconds}

    def __enter__(self):
        from threadpoolctl import threadpool_limits
        self.wall_start, self.cpu_start = time.monotonic(), time.process_time()
        self.old_limit = resource.getrlimit(resource.RLIMIT_CPU)
        self.old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGXCPU, signal.SIGTERM)}
        self.old_timer = signal.getitimer(signal.ITIMER_REAL)

        def exhausted(signum, frame):
            raise TimeoutError(f"reference budget/interruption signal {signum}")

        for sig in self.old_handlers:
            signal.signal(sig, exhausted)
        soft = math.ceil(self.cpu_start+self.seconds)
        if self.old_limit[1] != resource.RLIM_INFINITY:
            soft = min(soft, self.old_limit[1])
        resource.setrlimit(resource.RLIMIT_CPU, (soft, self.old_limit[1]))
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        self.threads = threadpool_limits(limits=1)
        self.threads.__enter__()
        return self

    def __exit__(self, *args):
        signal.setitimer(signal.ITIMER_REAL, 0)
        resource.setrlimit(resource.RLIMIT_CPU, self.old_limit)
        for sig, handler in self.old_handlers.items():
            signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, *self.old_timer)
        self.threads.__exit__(*args)


def price_audit(pilot, parameters, observations=None):
    """Explicit finer controls beyond the accepted rung, not a shifted ladder."""
    base, controls = pilot.prices(parameters, return_diagnostics=True)
    nodes, cutoff = 2*controls["nodes"], 2*controls["cutoff"]
    finer = pilot._rung_prices(parameters, pilot._indices(None), nodes, cutoff)
    shift = float(np.max(abs(finer-base)/pilot.measurement_sigma))
    ll_shift = None
    if observations is not None:
        sigma = pilot.observation_sigma(parameters)
        y = np.asarray(observations)
        ll_shift = float(-.5*np.sum(((y-finer)/sigma)**2-((y-base)/sigma)**2))
    passed = bool(np.isfinite(finer).all() and np.isfinite(shift)
                  and shift <= pilot.numerical_fraction
                  and (ll_shift is None or np.isfinite(ll_shift) and abs(ll_shift) <= .1))
    return finer, {"base_nodes": controls["nodes"], "base_cutoff": controls["cutoff"],
        "finer_nodes": nodes, "finer_cutoff": cutoff,
        "price_shift_over_minimum_measurement_sd": shift, "log_likelihood_shift": ll_shift,
        "pointwise_gates_passed": passed,
        "scope": "pointwise price/likelihood precision, not global posterior stability"}


def source_hashes():
    from sbi4finance.calibration import heston_pilot, heston_spread, reference_initialization, pymc_reference_op
    paths = {"runner": Path(__file__), "spread_likelihood": Path(heston_spread.__file__),
             "checked_pricer": Path(heston_pilot.__file__), "runtime_model": Path(model.__file__),
             "reference": Path(reference.__file__), "initialization": Path(reference_initialization.__file__),
             "likelihood_op": Path(pymc_reference_op.__file__),
             "partial_writer": Path(sys.modules[PartialDrawWriter.__module__].__file__)}
    lab = Path(__file__).resolve().parents[1]
    paths.update({name: lab/name for name in ("pyproject.toml", "pixi.lock")})
    return {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in paths.items()}


def prepare_panel(design_path, probe_cases_path, historical_summary_path, out, *, seed=261111):
    """Freeze three truths and common c=1 observations before new neural fits."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if type(seed) is not int or seed < 0:
        raise ValueError("panel seed must be a nonnegative integer")
    design = json.loads(Path(design_path).read_text())
    fixed, joint = HestonSpreadPilot(design["records"]), HestonSpreadPilot(design["records"], infer_c=True)
    if len(fixed.records["mid"]) != 54:
        raise ValueError("the reference panel requires the exact complete 54-quote design")
    broad = json.loads(Path(probe_cases_path).read_text())["cases"][0]
    if broad["validation_row"] != 0:
        raise ValueError("broad truth must be the predeclared validation row zero")
    historical = json.loads(Path(historical_summary_path).read_text())["cases"]["full-wide-xi-extended"]
    if historical["run"]["design_sha256"] != digest(design_path):
        raise ValueError("historical market reference does not bind the same design")
    quantiles = historical["posterior"]["parameter_quantiles_05_50_95"]
    truths = [broad["physical_parameters"], [quantiles[name][1] for name in fixed.names], BOUNDARY_TRUTH]
    panel = {"format": "heston_spread_reference_panel_v1", "predeclared_before_new_neural_fits": True,
        "design_path": str(Path(design_path).resolve()), "design_sha256": digest(design_path),
        "fixed_task": fixed.task_definition(), "joint_task": joint.task_definition(),
        "products": design["products"], "seed": seed,
        "truth_sources": {"broad_probe_cases": {"path": str(Path(probe_cases_path).resolve()), "sha256": digest(probe_cases_path)},
            "historical_market_summary": {"path": str(Path(historical_summary_path).resolve()), "sha256": digest(historical_summary_path),
                "case": "full-wide-xi-extended"}, "boundary": "manual near-boundary physical parameters; retained if difficult"},
        "source_sha256": source_hashes(), "reserved_test_arrays_opened": False,
        "scope": "Development observations, not acceptance data; market-like truth is synthetic, not a known market truth.", "cases": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    declaration = out.with_suffix(".declaration.json")
    if declaration.exists():
        raise FileExistsError(declaration)
    write_json(declaration, {**panel, "truths": dict(zip(CASE_IDS[:3], truths))})
    panel.update(declaration_path=str(declaration.resolve()), declaration_sha256=digest(declaration))
    try:
        with Budget():
            for index, (case_id, truth) in enumerate(zip(CASE_IDS[:3], truths)):
                truth = np.asarray(truth, dtype=float)
                if truth.shape != (5,) or np.any(truth < fixed.bounds[:, 0]) or np.any(truth > fixed.bounds[:, 1]):
                    raise ValueError("predeclared truth outside fixed physical prior")
                clean, precision = price_audit(fixed, truth)
                if not precision["pointwise_gates_passed"]:
                    raise FloatingPointError(f"predeclared case {case_id} fails price precision: {precision}")
                errors = np.random.default_rng(seed+index).standard_normal(54)
                observations = clean + fixed.records["half_spread"]*errors
                panel["cases"].append({"id": case_id, "kind": "synthetic", "physical_truth": truth.tolist(),
                    "generating_c": 1., "seed": seed+index, "clean_prices": clean.tolist(),
                    "standard_errors": errors.tolist(), "observations": observations.tolist(),
                    "observations_sha256": values_digest(observations), "clean_prices_sha256": values_digest(clean),
                    "pricing_audit": precision})
            observations = fixed.records["mid"]
            panel["cases"].append({"id": "market", "kind": "market", "physical_truth": None,
                "generating_c": None, "observations": observations.tolist(), "observations_sha256": values_digest(observations)})
            write_json(out, panel)
    except BaseException as error:
        write_json(out.with_suffix(".failure.json"), {"error_type": type(error).__name__, "error": str(error),
                   "completed_cases": [case["id"] for case in panel["cases"]], "scientific_acceptance": False})
        raise
    return panel


def load_case(panel_path, case_id, *, infer_c=False, c=1.):
    panel = json.loads(Path(panel_path).read_text())
    if panel["format"] != "heston_spread_reference_panel_v1" or [case["id"] for case in panel["cases"]] != list(CASE_IDS):
        raise ValueError("panel format or predeclared cases differ")
    if digest(panel["declaration_path"]) != panel["declaration_sha256"]:
        raise ValueError("predeclared truth file hash differs")
    declaration = json.loads(Path(panel["declaration_path"]).read_text())
    for key in ("design_sha256", "fixed_task", "joint_task", "products", "seed", "truth_sources"):
        if declaration[key] != panel[key]:
            raise ValueError("panel differs from the predeclared design, target or truth sources")
    design = json.loads(Path(panel["design_path"]).read_text())
    if digest(panel["design_path"]) != panel["design_sha256"] or panel["products"] != design["products"]:
        raise ValueError("design identity or product horizons differ")
    fixed = HestonSpreadPilot(design["records"])
    fixed.validate_task(panel["fixed_task"])
    HestonSpreadPilot(design["records"], infer_c=True).validate_task(panel["joint_task"])
    selected = next((case for case in panel["cases"] if case["id"] == case_id), None)
    if selected is None:
        raise ValueError("case id is not predeclared")
    y = np.asarray(selected["observations"], dtype=float)
    if y.shape != (54,) or not np.isfinite(y).all() or values_digest(y) != selected["observations_sha256"]:
        raise ValueError("case observation dimensions, finiteness or hash differ")
    if selected["kind"] == "synthetic":
        clean, error = np.asarray(selected["clean_prices"]), np.asarray(selected["standard_errors"])
        truth = np.asarray(selected["physical_truth"])
        if (case_id not in CASE_IDS[:3] or clean.shape != y.shape or error.shape != y.shape or truth.shape != (5,)
                or not np.isfinite(clean).all() or not np.isfinite(truth).all()
                or not np.array_equal(truth, declaration["truths"][case_id])
                or np.any(truth < fixed.bounds[:, 0]) or np.any(truth > fixed.bounds[:, 1])
                or selected["generating_c"] != 1. or values_digest(clean) != selected["clean_prices_sha256"]
                or not np.array_equal(error, np.random.default_rng(selected["seed"]).standard_normal(54))
                or not np.array_equal(y, clean+error*fixed.records["half_spread"])):
            raise ValueError("synthetic truth, clean prices or c=1 noise recomposition differs")
    elif (case_id != "market" or selected["kind"] != "market" or selected["physical_truth"] is not None
            or selected["generating_c"] is not None or not np.array_equal(y, fixed.records["mid"])):
        raise ValueError("market observation is not the frozen design midpoint vector")
    return panel, selected, HestonSpreadPilot(design["records"], c=c, infer_c=infer_c), y


def diagnose(draws, names, products):
    import xarray as xr
    def tree(values, labels):
        return xr.DataTree.from_dict({"posterior": xr.Dataset({"parameters": (("chain", "draw", "parameter"), values)},
                                                             coords={"parameter": labels})})
    result = reference.chain_diagnostics(tree(draws, names))
    result["products"] = reference.chain_diagnostics(tree(product_values(draws, products), list(products)))
    result["parameter_mcmc_diagnostics_passed"] = result["mcmc_diagnostics_passed"]
    result["mcmc_diagnostics_passed"] = bool(result["mcmc_diagnostics_passed"] and result["products"]["mcmc_diagnostics_passed"])
    return result


def run_reference(panel_path, case_id, out, payload_out, *, seed=261112, infer_c=False, c=1.,
                  smoke=False, budget_seconds=MAX_SECONDS, fit_budget=2000):
    if type(seed) is not int or seed < 0 or type(fit_budget) is not int or not 80 <= fit_budget <= 2000:
        raise ValueError("nonnegative integer seed and fit_budget between 80 and 2000 required")
    panel, case, pilot, y = load_case(panel_path, case_id, infer_c=infer_c, c=c)
    out, payload_out = Path(out), Path(payload_out)
    out.mkdir(parents=True, exist_ok=False)
    payload_out.mkdir(parents=True, exist_ok=False)
    run = {"format": "heston_spread_reference_run_v1", "status": "started", "case_id": case_id,
        "case_kind": case["kind"], "panel_path": str(Path(panel_path).resolve()), "panel_sha256": digest(panel_path),
        "design_sha256": panel["design_sha256"], "observation_sha256": values_digest(y),
        "observations": y.tolist(), "physical_truth": case["physical_truth"], "products": panel["products"],
        "task": pilot.task_definition(), "names": pilot.names, "bounds": pilot.bounds.tolist(),
        "source_sha256": source_hashes(), "settings": {"seed": seed, "chains": 4, "cores": 1,
            "fit_budget": fit_budget, "attempts": [[20,20]] if smoke else [list(v) for v in ATTEMPTS],
            "cpu_and_wall_seconds": budget_seconds, "blas_threads": 1},
        "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in
            ("pymc", "pytensor", "arviz-stats", "numpy", "scipy", "xarray")}},
        "smoke": smoke, "scientific_acceptance": False, "reserved_test_arrays_opened": False,
        "sampler": "native PyMC DEMetropolisZ; bounded physical Uniform priors; deterministic Fourier likelihood",
        "scope": "Reference convergence and sampled numerical precision only; no neural or market-model acceptance.",
        "attempts": []}
    write_json(out/"run.json", run)
    write_json(payload_out/"run.json", run)
    budget = Budget(budget_seconds)
    try:
        with budget:
            for index, (tune, retained) in enumerate(((20,20),) if smoke else ATTEMPTS):
                directory = payload_out/f"attempt-{index}"
                directory.mkdir()
                calls = 0
                def likelihood(point):
                    nonlocal calls
                    budget.check()
                    calls += 1
                    if calls % 250 == 0:
                        write_json(out/"progress.json", {"attempt": index, "phase": "likelihood", "evaluations": calls, **budget.usage()})
                    return pilot.log_likelihood(point, observations=y)
                initialization = reference.data_fit_initialization(likelihood, pilot.bounds, 4, seed,
                    fit_starts=4, fit_budget=fit_budget)
                write_json(directory/"initialization.json", initialization)
                partial = PartialDrawWriter(directory/"partial", chains=4, tune=tune, draws=retained, every=250)
                def callback(**kwargs):
                    budget.check()
                    partial(**kwargs)
                idata = reference.sample_deterministic_reference(likelihood, pilot.bounds, pilot.names,
                    draws=retained, tune=tune, chains=4, cores=1, seed=seed,
                    progressbar=False, initialization=initialization, callback=callback)
                draws = np.asarray(idata["posterior"]["parameters"].transpose("chain", "draw", "parameter").values)
                if (draws.shape != (4, retained, len(pilot.names)) or not np.isfinite(draws).all()
                        or np.any(draws < pilot.bounds[:,0]) or np.any(draws > pilot.bounds[:,1])):
                    raise ValueError("retained draw shape, finiteness or physical support differs")
                arrays = {"parameters": draws, "products": product_values(draws, panel["products"])}
                for name, value in idata["sample_stats"].data_vars.items():
                    if set(value.dims) == {"chain", "draw"}:
                        arrays["sample_stats_"+name] = value.transpose("chain", "draw").values
                temporary = directory/"chains.npz.tmp"
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush(); os.fsync(stream.fileno())
                temporary.replace(directory/"chains.npz")
                diagnostics = diagnose(draws, pilot.names, panel["products"])
                precision = []
                for chain in range(4):
                    for draw in (0, retained-1):
                        _, check = price_audit(pilot, draws[chain,draw], y)
                        precision.append({"chain": chain, "draw": draw, **check})
                diagnostics["numerical_precision"] = precision
                diagnostics["pointwise_precision_passed"] = all(point["pointwise_gates_passed"] for point in precision)
                diagnostics["chain_payload_path"] = str((directory/"chains.npz").resolve())
                diagnostics["chain_payload_sha256"] = digest(directory/"chains.npz")
                diagnostics["smoke"] = smoke
                diagnostics["scientific_acceptance"] = False
                diagnostics["reference_ready"] = bool(not smoke and diagnostics["mcmc_diagnostics_passed"] and diagnostics["pointwise_precision_passed"])
                write_json(directory/"diagnostics.json", diagnostics)
                attempt = {"attempt": index, "tune": tune, "draws": retained, "likelihood_evaluations": calls,
                    "mcmc_diagnostics_passed": diagnostics["mcmc_diagnostics_passed"],
                    "pointwise_precision_passed": diagnostics["pointwise_precision_passed"],
                    "payload_directory": str(directory.resolve()), "diagnostics_sha256": digest(directory/"diagnostics.json"), **budget.usage()}
                run["attempts"].append(attempt)
                if not diagnostics["pointwise_precision_passed"]:
                    raise FloatingPointError("retained-point pricing/likelihood precision failed; no sampler escalation")
                if diagnostics["mcmc_diagnostics_passed"] or smoke or index == len(ATTEMPTS)-1:
                    run.update(status="completed", reference_ready=diagnostics["reference_ready"],
                        selected_attempt=index, chain_payload_path=diagnostics["chain_payload_path"],
                        chain_payload_sha256=diagnostics["chain_payload_sha256"])
                    write_json(out/"diagnostics.json", diagnostics)
                    write_json(payload_out/"diagnostics.json", diagnostics)
                    flat = draws.reshape(-1,len(pilot.names))
                    posterior = {"parameter_quantiles_05_50_95": {name: np.quantile(flat[:,i],[.05,.5,.95]).tolist() for i,name in enumerate(pilot.names)},
                        "parameter_means": dict(zip(pilot.names, np.mean(flat,axis=0).tolist())),
                        "products": {name: {"interval": interval, "quantiles_05_50_95": np.quantile(expected_variance(flat,interval),[.05,.5,.95]).tolist()}
                            for name,interval in panel["products"].items()}}
                    write_json(out/"posterior.json", posterior)
                    break
                write_json(out/"run.json", run)
            run["usage"] = budget.usage()
        write_json(out/"run.json", run)
        write_json(payload_out/"run.json", run)
        return run
    except BaseException as error:
        run.update(status="failed", reference_ready=False, error_type=type(error).__name__, error=str(error), usage=budget.usage())
        write_json(out/"run.json", run)
        write_json(payload_out/"run.json", run)
        write_json(out/"failure.json", run)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    panel = sub.add_parser("prepare-panel")
    for name in ("design", "probe-cases", "historical-summary", "out"):
        panel.add_argument("--"+name, type=Path, required=True)
    panel.add_argument("--seed", type=int, default=261111)
    run = sub.add_parser("run")
    for name in ("panel", "out", "payload-out"):
        run.add_argument("--"+name, type=Path, required=True)
    run.add_argument("--case", choices=CASE_IDS, required=True)
    run.add_argument("--seed", type=int, default=261112)
    run.add_argument("--infer-c", action="store_true")
    run.add_argument("--c", type=float, default=1.)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--budget-seconds", type=int, default=MAX_SECONDS)
    run.add_argument("--fit-budget", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "prepare-panel":
        prepare_panel(args.design,args.probe_cases,args.historical_summary,args.out,seed=args.seed)
        print(json.dumps({"panel":str(args.out),"sha256":digest(args.out),"cases":list(CASE_IDS)}))
    else:
        result = run_reference(args.panel,args.case,args.out,args.payload_out,seed=args.seed,
            infer_c=args.infer_c,c=args.c,smoke=args.smoke,budget_seconds=args.budget_seconds,fit_budget=args.fit_budget)
        print(json.dumps({"out":str(args.out),"reference_ready":result["reference_ready"],"smoke":args.smoke}))


if __name__ == "__main__":
    main()

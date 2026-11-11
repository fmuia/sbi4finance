"""Bounded native-PyMC references for a new63-quote observation-only panel.

The historical wrappers require54 quotes. This version changes panel and job
binding only; finance, initialization, PyMC sampling, diagnostics, pricing
refinement, partial writes and CPU/wall enforcement use canonical helpers.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROCESS_STARTED = time.monotonic()
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_reference as shared
from scripts.evaluate_heston_spread_baseline import check_reference_binding
from sbi4finance.calibration import heston_grid
from sbi4f_tutorial.heston_options import control_variate

read = lambda path: json.loads(Path(path).read_text())
digest, values_digest, write_json = shared.digest, shared.values_digest, shared.write_json
CASES = ("synthetic-broad", "synthetic-market-like", "market")
SEED, INNER_SECONDS, OUTER_SECONDS = 261112, 900, 950


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def checked(spec):
    if not {"path", "sha256"} <= set(spec) or binding(spec["path"]) != {k: spec[k] for k in ("path", "sha256")}:
        raise ValueError("reference input identity differs")
    return Path(spec["path"])


def source_hashes():
    from scripts import evaluate_heston_spread_baseline as comparison
    return {**shared.source_hashes(), "grid_reference": binding(__file__), "binding_validator": binding(comparison.__file__),
        "grid_numerical_pilot": binding(heston_grid.__file__), "black_control_variate": binding(control_variate.__file__),
        "black_formula": binding(control_variate.gbm.__file__)}


def settings_contract(*, smoke_test_only=False):
    if type(smoke_test_only) is not bool: raise ValueError("explicit smoke mode required")
    return {"chains": 4, "cores": 1, "seed": SEED, "fit_budget": 80 if smoke_test_only else 2000,
        "attempts": [[20, 20]] if smoke_test_only else [list(pair) for pair in shared.ATTEMPTS],
        "inner_cpu_and_wall_seconds": 120 if smoke_test_only else INNER_SECONDS,
        "outer_total_wall_seconds": 150 if smoke_test_only else OUTER_SECONDS,
        "r_hat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400,
        "escalation": "canonical6k/12k second attempt only after first2k/2k diagnostic failure; never after numerical failure",
        "initialization": "canonical four data-only Powell fits and dispersed starts; no truths or neural estimates",
        "likelihood": "canonical normalized Gaussian with sigma_i=c*half_spread_i and -quote_count*log(c)",
        "smoke_test_only": smoke_test_only, "automatic_retry_or_budget_extension": False,
        "truth_available_to_sampler": False, "reserved_arrays_opened": False}


def load_case(panel_path, case_id):
    panel = read(panel_path)
    if (panel.get("format") != "heston_pr87_grid_reference_panel_v1"
            or [case.get("id") for case in panel.get("cases", [])] != list(CASES)
            or not set(panel) <= {"format", "design", "task", "products", "cases", "scope", "source", "preparation_protocol"}):
        raise ValueError("three-case observation-only grid reference panel required")
    if "source" in panel: checked(panel["source"])
    if "preparation_protocol" in panel: checked(panel["preparation_protocol"])
    design_path = checked(panel["design"])
    design = read(design_path)
    pilot = heston_grid.HestonGridPilot(design["records"], infer_c=True, c_bounds=(.5, 2.))
    pilot.validate_task(panel["task"])
    if (len(pilot.records["mid"]) != 63 or len(pilot.names) != 6 or pilot.names[-1] != "c"
            or panel["products"] != design["products"]):
        raise ValueError("canonical unchanged six-target priors and new63 design required")
    selected = None
    for case in panel["cases"]:
        if set(case) != {"id", "observations", "observations_sha256"}:
            raise ValueError("sampler panel must contain observations only, never generating truths")
        y = np.asarray(case["observations"], dtype=float)
        if y.shape != (len(pilot.records["mid"]),) or not np.isfinite(y).all() or values_digest(y) != case["observations_sha256"]:
            raise ValueError("63-quote reference observation dimensions or identity differ")
        if case["id"] == "market" and not np.array_equal(y, pilot.records["mid"]):
            raise ValueError("market reference must use frozen63 design midpoints")
        if case["id"] == case_id: selected = case
    if selected is None: raise ValueError("case is not predeclared")
    return panel, selected, pilot, np.asarray(selected["observations"], dtype=float)


def job_spec(panel_path, case_id):
    panel, case, pilot, y = load_case(panel_path, case_id)
    return {"id": case_id, "case_id": case_id, "mode": "joint", "task": pilot.task_definition(),
        "names": pilot.names, "bounds": pilot.bounds.tolist(), "seed": SEED,
        "observation_sha256": values_digest(y), "truth_available_to_sampler": False}


def prepare_protocol(panel_path, evidence_root, payload_root, out, *, smoke_test_only=False):
    panel_path, evidence_root, payload_root, out = map(lambda p: Path(p).resolve(), (panel_path, evidence_root, payload_root, out))
    if out.exists(): raise FileExistsError("new grid reference protocol identity required")
    panel = read(panel_path); jobs = []
    for case in CASES:
        spec = job_spec(panel_path, case)
        spec.update(evidence_out=str(evidence_root/case), payload_out=str(payload_root/case))
        if Path(spec["evidence_out"]).exists() or Path(spec["payload_out"]).exists():
            raise FileExistsError("reference case outputs must be new identities")
        jobs.append(spec)
    protocol = {"format": "heston_pr87_grid_reference_protocol_v1", "predeclared_before_sampling": True, "protocol_path": str(out),
        "panel_path": str(panel_path), "panel_sha256": digest(panel_path), "design_path": panel["design"]["path"],
        "design_sha256": panel["design"]["sha256"], "products": panel["products"],
        "evidence_root": str(evidence_root), "payload_root": str(payload_root), "jobs": jobs,
        "settings": settings_contract(smoke_test_only=smoke_test_only), "source_sha256": source_hashes(),
        "campaign_limits": {"maximum_concurrent_reference_processes": 2,
            "maximum_sum_inner_cpu_seconds": len(jobs) * (120 if smoke_test_only else INNER_SECONDS),
            "whole_campaign_wall_seconds": len(jobs) * (150 if smoke_test_only else OUTER_SECONDS)},
        "truth_available_to_sampler": False, "reserved_arrays_opened": False, "scientific_acceptance": False}
    out.parent.mkdir(parents=True, exist_ok=True); write_json(out, protocol)
    return protocol


def validate_protocol(protocol, case_id, *, smoke_test_only=False):
    if (protocol.get("format") != "heston_pr87_grid_reference_protocol_v1"
            or protocol.get("predeclared_before_sampling") is not True
            or protocol.get("settings") != settings_contract(smoke_test_only=smoke_test_only)
            or source_hashes() != protocol.get("source_sha256")
            or read(protocol["protocol_path"]) != protocol
            or digest(protocol["panel_path"]) != protocol["panel_sha256"]
            or digest(protocol["design_path"]) != protocol["design_sha256"]
            or [job.get("case_id") for job in protocol.get("jobs", [])] != list(CASES)
            or protocol.get("truth_available_to_sampler") is not False
            or protocol.get("reserved_arrays_opened") is not False
            or protocol.get("scientific_acceptance") is not False
            or protocol.get("campaign_limits") != {"maximum_concurrent_reference_processes": 2,
                "maximum_sum_inner_cpu_seconds": 3*(120 if smoke_test_only else INNER_SECONDS),
                "whole_campaign_wall_seconds": 3*(150 if smoke_test_only else OUTER_SECONDS)}):
        raise ValueError("frozen grid reference protocol, sources or limits differ")
    selected = None
    for spec in protocol["jobs"]:
        expected = job_spec(protocol["panel_path"], spec["case_id"])
        if (any(spec.get(key) != value for key, value in expected.items())
                or Path(spec["evidence_out"]) != Path(protocol["evidence_root"])/spec["id"]
                or Path(spec["payload_out"]) != Path(protocol["payload_root"])/spec["id"]):
            raise ValueError("reference job target, observations, seed or output differs")
        if spec["case_id"] == case_id: selected = spec
    if selected is None: raise ValueError("one predeclared case required")
    return selected


def run_reference(protocol_path, case_id, *, smoke_test_only=False):
    protocol = read(protocol_path)
    spec = validate_protocol(protocol, case_id, smoke_test_only=smoke_test_only)
    panel, case, pilot, y = load_case(protocol["panel_path"], case_id)
    out, payload_out = map(Path, (spec["evidence_out"], spec["payload_out"]))
    if out.exists() or payload_out.exists(): raise FileExistsError("reference outputs are immutable; no retry")
    out.mkdir(parents=True); payload_out.mkdir(parents=True)
    settings = protocol["settings"]
    run = {"format": "heston_pr87_grid_reference_run_v1", "status": "started", "case_id": case_id,
        "panel_path": protocol["panel_path"], "panel_sha256": protocol["panel_sha256"],
        "design_sha256": protocol["design_sha256"], "observation_sha256": values_digest(y), "observations": y.tolist(),
        "products": panel["products"], "task": pilot.task_definition(), "names": pilot.names, "bounds": pilot.bounds.tolist(),
        "protocol": binding(protocol_path), "source_sha256": source_hashes(), "settings": {"seed": settings["seed"],
            "chains": 4, "cores": 1, "fit_budget": settings["fit_budget"], "attempts": settings["attempts"],
            "cpu_and_wall_seconds": settings["inner_cpu_and_wall_seconds"], "blas_threads": 1},
        "versions": {"python": platform.python_version(), **{name: metadata.version(name) for name in
            ("pymc", "pytensor", "arviz-stats", "numpy", "scipy", "xarray")}},
        "smoke": smoke_test_only, "sampler": "canonical native PyMC DEMetropolisZ; bounded physical Uniform priors",
        "truth_available_to_sampler": False, "reserved_test_arrays_opened": False, "scientific_acceptance": False, "attempts": []}
    write_json(out/"run.json", run)
    budget = shared.Budget(settings["inner_cpu_and_wall_seconds"])
    try:
        with budget:
            for index, (tune, retained) in enumerate(settings["attempts"]):
                directory = payload_out/f"attempt-{index}"; directory.mkdir(); calls = 0
                def likelihood(point):
                    nonlocal calls
                    budget.check(); calls += 1
                    if calls % 250 == 0:
                        write_json(out/"progress.json", {"attempt": index, "evaluations": calls, **budget.usage()})
                    return pilot.log_likelihood(point, observations=y)
                initialization = shared.reference.data_fit_initialization(likelihood, pilot.bounds, 4, settings["seed"],
                    fit_starts=4, fit_budget=settings["fit_budget"])
                write_json(directory/"initialization.json", initialization)
                partial = shared.PartialDrawWriter(directory/"partial", chains=4, tune=tune, draws=retained, every=250)
                def callback(**kwargs): budget.check(); partial(**kwargs)
                idata = shared.reference.sample_deterministic_reference(likelihood, pilot.bounds, pilot.names,
                    draws=retained, tune=tune, chains=4, cores=1, seed=settings["seed"], progressbar=False,
                    initialization=initialization, callback=callback)
                draws = np.asarray(idata["posterior"]["parameters"].transpose("chain", "draw", "parameter").values)
                if (draws.shape != (4, retained, 6) or not np.isfinite(draws).all()
                        or np.any(draws < pilot.bounds[:, 0]) or np.any(draws > pilot.bounds[:, 1])):
                    raise ValueError("canonical reference draw shape or support differs")
                arrays = {"parameters": draws, "products": shared.product_values(draws, panel["products"])}
                for name, value in idata["sample_stats"].data_vars.items():
                    if set(value.dims) == {"chain", "draw"}: arrays["sample_stats_"+name] = value.transpose("chain", "draw").values
                with (directory/"chains.npz").open("wb") as stream:
                    np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
                diagnostics = shared.diagnose(draws, pilot.names, panel["products"])
                precision = []
                for chain in range(4):
                    for draw in (0, retained-1):
                        _, check = shared.price_audit(pilot, draws[chain, draw], y)
                        precision.append({"chain": chain, "draw": draw, **check})
                diagnostics.update(numerical_precision=precision,
                    pointwise_precision_passed=all(item["pointwise_gates_passed"] for item in precision),
                    chain_payload_path=str((directory/"chains.npz").resolve()), chain_payload_sha256=digest(directory/"chains.npz"),
                    smoke=smoke_test_only, scientific_acceptance=False)
                diagnostics["reference_ready"] = bool(not smoke_test_only and diagnostics["mcmc_diagnostics_passed"] and diagnostics["pointwise_precision_passed"])
                write_json(directory/"diagnostics.json", diagnostics)
                run["attempts"].append({"attempt": index, "tune": tune, "draws": retained, "likelihood_evaluations": calls,
                    "mcmc_diagnostics_passed": diagnostics["mcmc_diagnostics_passed"],
                    "pointwise_precision_passed": diagnostics["pointwise_precision_passed"],
                    "diagnostics_sha256": digest(directory/"diagnostics.json"), **budget.usage()})
                if not diagnostics["pointwise_precision_passed"]:
                    raise FloatingPointError("sampled canonical pricing refinement failed; no escalation")
                if diagnostics["mcmc_diagnostics_passed"] or smoke_test_only or index == len(settings["attempts"])-1:
                    run.update(status="completed", reference_ready=diagnostics["reference_ready"], selected_attempt=index,
                        chain_payload_path=diagnostics["chain_payload_path"], chain_payload_sha256=diagnostics["chain_payload_sha256"])
                    write_json(out/"diagnostics.json", diagnostics)
                    flat = draws.reshape(-1, 6)
                    write_json(out/"posterior.json", {"parameter_quantiles_05_50_95":
                        {name: np.quantile(flat[:, j], [.05, .5, .95]).tolist() for j, name in enumerate(pilot.names)},
                        "parameter_means": dict(zip(pilot.names, flat.mean(axis=0).tolist()))})
                    break
                write_json(out/"run.json", run)
            run["usage"] = budget.usage(); budget.check()
        validate_protocol(protocol, case_id, smoke_test_only=smoke_test_only)
        if digest(protocol_path) != run["protocol"]["sha256"]: raise ValueError("reference declaration changed")
        write_json(out/"run.json", run); write_json(payload_out/"run.json", run)
        return run
    except BaseException as error:
        run.update(status="failed", reference_ready=False, error_type=type(error).__name__, error=str(error), usage=budget.usage())
        write_json(out/"run.json", run); write_json(out/"failure.json", run); write_json(payload_out/"run.json", run)
        raise


def matched_reference(protocol, case_id, target, observations, protected):
    """Read only ready new-target chains; never substitute historical54 chains."""
    spec = validate_protocol(protocol, case_id)
    if spec["task"] != target or spec["observation_sha256"] != values_digest(observations):
        raise ValueError("grid reference does not target this exact posterior")
    directory = Path(spec["evidence_out"]); run_path = directory/"run.json"; diagnostic_path = directory/"diagnostics.json"
    if not run_path.is_file(): return None, {"reference_ready": False, "status": "inconclusive", "reason": "new63 reference missing"}
    record = read(run_path); protected[str(run_path.resolve())] = digest(run_path)
    if (record.get("format") != "heston_pr87_grid_reference_run_v1"
            or record.get("protocol") != binding(protocol["protocol_path"])
            or record.get("source_sha256") != protocol["source_sha256"] or record.get("smoke") is not False):
        raise ValueError("new63 reference run provenance or production identity differs")
    if record.get("status") != "completed" or not diagnostic_path.is_file():
        return None, {"reference_ready": False, "status": "inconclusive", "reason": "new63 reference failed or capped: "+str(record.get("error_type", record.get("status")))+" "+str(record.get("error", ""))[:300], "reference_run": binding(run_path)}
    diagnostics = read(diagnostic_path); chains_path = Path(record["chain_payload_path"])
    check_reference_binding(record, diagnostics, chains_path, target=target, design_sha256=protocol["design_sha256"],
        case_id=case_id, observations=observations, panel_sha256=protocol["panel_sha256"], products=protocol["products"], diagnostics_path=diagnostic_path)
    result = {"reference_run": binding(run_path), "reference_diagnostics": binding(diagnostic_path),
        "reference_chain_payload": binding(chains_path), "reference_ready": bool(diagnostics.get("reference_ready") is True
            and diagnostics.get("mcmc_diagnostics_passed") is True and diagnostics.get("pointwise_precision_passed") is True)}
    for item in (result["reference_run"], result["reference_diagnostics"], result["reference_chain_payload"]): protected[item["path"]] = item["sha256"]
    if result["reference_ready"] is not True:
        return None, {**result, "status": "inconclusive", "reason": "new63 convergence or sampled precision unresolved"}
    with np.load(chains_path, allow_pickle=False) as archive: chains = archive["parameters"].copy()
    retained = record["attempts"][record["selected_attempt"]]["draws"]
    if chains.shape != (4, retained, 6) or not np.isfinite(chains).all() or np.any(chains < np.asarray(spec["bounds"])[:, 0]) or np.any(chains > np.asarray(spec["bounds"])[:, 1]):
        raise ValueError("new63 diagnosed chain shape or physical prior support differs")
    return chains, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "protocol", "out", "payload-out"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--case-id", choices=CASES, required=True)
    parser.add_argument("--smoke-test-only", action="store_true"); parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(); protocol = read(args.protocol)
    spec = validate_protocol(protocol, args.case_id, smoke_test_only=args.smoke_test_only)
    if (Path(protocol["panel_path"]).resolve() != args.panel.resolve() or Path(spec["evidence_out"]).resolve() != args.out.resolve()
            or Path(spec["payload_out"]).resolve() != args.payload_out.resolve()): parser.error("CLI identities differ from frozen job")
    if args.worker:
        result = run_reference(args.protocol, args.case_id, smoke_test_only=args.smoke_test_only)
        print(json.dumps({"status": result["status"], "case": args.case_id, "reference_ready": result["reference_ready"]}))
    else:
        if args.out.exists() or args.payload_out.exists(): raise FileExistsError("new reference identities required")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"])
        try: code = process.wait(timeout=max(.001, protocol["settings"]["outer_total_wall_seconds"]-(time.monotonic()-PROCESS_STARTED)))
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(); args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out.parent/(args.out.name+".outer-timeout.json"), {"status": "killed_at_declared_outer_wall_limit", "reference_ready": False, "automatic_retry": False})
            raise TimeoutError("grid reference outer wall cap exhausted")
        if code: raise SystemExit(code)

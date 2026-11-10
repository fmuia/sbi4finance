"""Joint six-parameter Heston MCMC on a frozen observation-only recovery panel.

Truth, clean prices and generating noise are intentionally absent from the
sampler panel. Native PyMC sampling, initialization, numerical-price refinement
and diagnostics reuse the unchanged reference helpers. One named job runs per
process; full chains and incomplete attempts retain their identities.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from importlib import metadata

PROCESS_STARTED = time.monotonic()
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from scripts import run_heston_spread_reference as shared

HestonSpreadPilot, reference = shared.HestonSpreadPilot, shared.reference
Budget, PartialDrawWriter = shared.Budget, shared.PartialDrawWriter
digest, values_digest, write_json = shared.digest, shared.values_digest, shared.write_json
price_audit, diagnose, product_values = shared.price_audit, shared.diagnose, shared.product_values
expected_variance = shared.expected_variance
ATTEMPTS = shared.ATTEMPTS
FROZEN_REFERENCE_SHA256 = "69b6267580b91c49b691a867326f8f64ce332dad445204e1fe00166ff6d2f575"
BASE_SEED, INNER_SECONDS, OUTER_SECONDS = 261120, 900, 950


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {**shared.source_hashes(), "recovery_runner": {
        "path": str(Path(__file__).resolve()), "sha256": digest(__file__)}}


def settings_contract(case_count=8, *, smoke_test_only=False):
    if type(case_count) is not int or not 1 <= case_count <= 8 or type(smoke_test_only) is not bool:
        raise ValueError("one to eight predeclared recovery cases required")
    return {"case_count": case_count, "chains": 4, "cores": 1,
        "base_seed": BASE_SEED, "seed_rule": "base_seed + fixed observation-panel case index",
        "fit_budget": 80 if smoke_test_only else 2000,
        "attempts": [[20, 20]] if smoke_test_only else [list(value) for value in ATTEMPTS],
        "inner_cpu_and_wall_seconds": 120 if smoke_test_only else INNER_SECONDS,
        "outer_total_wall_seconds": 150 if smoke_test_only else OUTER_SECONDS,
        "r_hat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400,
        "escalation": "one 6k/12k attempt only if initial 2k/2k MCMC diagnostics fail; never after numerical failure",
        "initialization": "four multistart data-only Powell fits, bounded priors and distinct dispersed starts; no generating truth",
        "likelihood": "normalized independent Gaussian price likelihood with sigma_i=c*half_spread_i, including -54*log(c)",
        "smoke_test_only": smoke_test_only, "automatic_retry_or_budget_extension": False,
        "reserved_arrays_opened": False, "truth_available_to_sampler": False}


def load_case(panel_path, case_id):
    panel = read(panel_path)
    allowed = {"format", "design_path", "design_sha256", "joint_task", "products", "cases", "scope",
        "source", "population_observations_sha256"}
    if (panel.get("format") != "heston_spread_recovery_observation_panel_v1"
            or not set(panel) <= allowed or not 1 <= len(panel.get("cases", [])) <= 8):
        raise ValueError("observation-only recovery panel format or fields differ")
    if "source" in panel:
        source = panel["source"]
        if set(source) != {"path", "sha256"} or digest(source["path"]) != source["sha256"]:
            raise ValueError("observation-panel producer binding differs")
    if ("population_observations_sha256" in panel
            and re.fullmatch(r"[0-9a-f]{64}", panel["population_observations_sha256"]) is None):
        raise ValueError("population observation hash must be a SHA256 identity")
    cases = panel["cases"]
    if (any(set(case) != {"id", "row_id", "observations", "observations_sha256"} for case in cases)
            or any(not isinstance(case["id"], str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", case["id"]) is None
                or not isinstance(case["row_id"], str) or re.fullmatch(r"validation:[0-9]{8}", case["row_id"]) is None
                for case in cases)
            or len({case["id"] for case in cases}) != len(cases)
            or len({case["row_id"] for case in cases}) != len(cases)):
        raise ValueError("case fields/identities differ; truths and clean prices must remain outside the sampler panel")
    if digest(panel["design_path"]) != panel["design_sha256"]:
        raise ValueError("recovery design identity differs")
    design = read(panel["design_path"])
    pilot = HestonSpreadPilot(design["records"], infer_c=True, c_bounds=(.5, 2.))
    pilot.validate_task(panel["joint_task"])
    if (len(pilot.records["mid"]) != 54 or len(pilot.names) != 6 or pilot.names[-1] != "c"
            or panel["products"] != design["products"]):
        raise ValueError("recovery must use the unchanged complete six-target Heston design")
    selected = next((case for case in cases if case["id"] == case_id), None)
    if selected is None:
        raise ValueError("case must be in the predeclared observation panel")
    for case in cases:
        y = np.asarray(case["observations"], dtype=float)
        if y.shape != (54,) or not np.isfinite(y).all() or values_digest(y) != case["observations_sha256"]:
            raise ValueError("recovery observation dimensions, values or hash differ")
    return panel, selected, pilot, np.asarray(selected["observations"], dtype=float)


def job_spec(panel_path, case_id, index):
    panel, case, pilot, y = load_case(panel_path, case_id)
    if type(index) is not int or not 0 <= index < len(panel["cases"]) or panel["cases"][index]["id"] != case_id:
        raise ValueError("fixed case index must match the declared observation order")
    return {"id": case_id, "case_id": case_id, "row_id": case["row_id"], "seed": BASE_SEED+index,
        "task": pilot.task_definition(), "names": pilot.names, "bounds": pilot.bounds.tolist(),
        "observation_sha256": values_digest(y), "truth_available_to_sampler": False}


def prepare_protocol(panel_path, evidence_root, payload_root, out, *, smoke_test_only=False):
    panel_path, evidence_root, payload_root, out = map(Path, (panel_path, evidence_root, payload_root, out))
    if out.exists() or digest(shared.__file__) != FROZEN_REFERENCE_SHA256:
        raise ValueError("protocol identity exists or frozen reference helper differs")
    panel = read(panel_path)
    jobs = []
    for index, case in enumerate(panel["cases"]):
        spec = job_spec(panel_path, case["id"], index)
        spec.update(evidence_out=str((evidence_root/case["id"]).resolve()),
            payload_out=str((payload_root/case["id"]).resolve()))
        if Path(spec["evidence_out"]).exists() or Path(spec["payload_out"]).exists():
            raise FileExistsError("recovery case output identity already exists")
        jobs.append(spec)
    protocol = {"format": "heston_spread_recovery_reference_protocol_v1", "predeclared_before_sampling": True,
        "panel_path": str(panel_path.resolve()), "panel_sha256": digest(panel_path),
        "design_path": panel["design_path"], "design_sha256": panel["design_sha256"],
        "products": panel["products"], "evidence_root": str(evidence_root.resolve()), "payload_root": str(payload_root.resolve()),
        "jobs": jobs, "source_sha256": source_hashes(),
        "settings": settings_contract(len(jobs), smoke_test_only=smoke_test_only),
        "campaign_limits": {"maximum_concurrent_reference_processes": 2,
            "whole_campaign_wall_seconds": 4500, "maximum_sum_inner_cpu_seconds": len(jobs)*INNER_SECONDS,
            "enforcement": "root campaign supervisor; each runner enforces its own inner and outer limits"},
        "truth_available_to_sampler": False, "reserved_arrays_opened": False, "scientific_acceptance": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, protocol)
    return protocol


def validate_protocol(protocol, panel_path, case_id, out, payload_out, *, smoke_test_only=False):
    if (protocol.get("format") != "heston_spread_recovery_reference_protocol_v1"
            or digest(shared.__file__) != FROZEN_REFERENCE_SHA256
            or Path(protocol["panel_path"]).resolve() != Path(panel_path).resolve()
            or digest(panel_path) != protocol["panel_sha256"]
            or digest(protocol["design_path"]) != protocol["design_sha256"]
            or source_hashes() != protocol["source_sha256"]):
        raise ValueError("recovery protocol source, panel or design binding differs")
    panel, case, pilot, y = load_case(panel_path, case_id)
    if (protocol["settings"] != settings_contract(len(panel["cases"]), smoke_test_only=smoke_test_only)
            or protocol["products"] != panel["products"] or protocol["design_sha256"] != panel["design_sha256"]
            or protocol["truth_available_to_sampler"] is not False
            or protocol["reserved_arrays_opened"] is not False
            or [job["case_id"] for job in protocol["jobs"]] != [item["id"] for item in panel["cases"]]):
        raise ValueError("predeclared recovery settings, count, order or target differs")
    selected = None
    for index, spec in enumerate(protocol["jobs"]):
        expected = job_spec(panel_path, spec["case_id"], index)
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("recovery job target, observation, row or seed differs")
        if (Path(spec["evidence_out"]) != Path(protocol["evidence_root"])/spec["id"]
                or Path(spec["payload_out"]) != Path(protocol["payload_root"])/spec["id"]):
            raise ValueError("recovery output differs from declared root")
        if spec["case_id"] == case_id:
            selected = spec
    if (selected is None or Path(selected["evidence_out"]).resolve() != Path(out).resolve()
            or Path(selected["payload_out"]).resolve() != Path(payload_out).resolve()):
        raise ValueError("selected case output differs from predeclared job")
    return panel, case, pilot, y, selected


def run_reference(panel_path, case_id, protocol_path, out, payload_out, *, smoke_test_only=False):
    protocol = read(protocol_path)
    panel, case, pilot, y, spec = validate_protocol(protocol, panel_path, case_id, out, payload_out,
        smoke_test_only=smoke_test_only)
    smoke = smoke_test_only
    seed = spec["seed"]
    budget_seconds = protocol["settings"]["inner_cpu_and_wall_seconds"]
    fit_budget = protocol["settings"]["fit_budget"]
    out, payload_out = Path(out), Path(payload_out)
    out.mkdir(parents=True, exist_ok=False)
    payload_out.mkdir(parents=True, exist_ok=False)
    run = {"format": "heston_spread_recovery_reference_run_v1", "status": "started", "case_id": case_id,
        "case_kind": "synthetic_validation", "row_id": case["row_id"], "panel_path": str(Path(panel_path).resolve()), "panel_sha256": digest(panel_path),
        "design_sha256": panel["design_sha256"], "observation_sha256": values_digest(y),
        "observations": y.tolist(), "truth_available_to_sampler": False, "products": panel["products"],
        "protocol_path": str(Path(protocol_path).resolve()), "protocol_sha256": digest(protocol_path),
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
        if source_hashes() != protocol["source_sha256"] or digest(protocol_path) != run["protocol_sha256"]:
            raise ValueError("reference source or protocol changed during sampling")
        validate_protocol(protocol, panel_path, case_id, out, payload_out, smoke_test_only=smoke_test_only)
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
    for name in ("panel", "protocol", "out", "payload-out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        result = run_reference(args.panel, args.case_id, args.protocol, args.out, args.payload_out,
            smoke_test_only=args.smoke_test_only)
        print(json.dumps({"status": result["status"], "reference_ready": result["reference_ready"],
            "case_id": args.case_id, "smoke": args.smoke_test_only}))
        return
    protocol = read(args.protocol)
    validate_protocol(protocol, args.panel, args.case_id, args.out, args.payload_out,
        smoke_test_only=args.smoke_test_only)
    if args.out.exists() or args.payload_out.exists():
        raise FileExistsError("recovery case identities are immutable; no automatic retry")
    timeout = protocol["settings"]["outer_total_wall_seconds"]
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"])
    try:
        code = process.wait(timeout=max(.001, timeout-(time.monotonic()-PROCESS_STARTED)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out.parent/(args.out.name+".outer-timeout.json"),
            {"status": "killed_at_declared_outer_wall_limit", "case_id": args.case_id,
             "outer_wall_seconds": timeout, "reference_ready": False, "automatic_retry": False})
        raise TimeoutError("recovery reference outer wall limit exhausted")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

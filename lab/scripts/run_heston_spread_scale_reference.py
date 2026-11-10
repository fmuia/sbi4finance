"""Source-bound consumer of the unchanged fixed/joint-c Heston reference runner.

Predeclare a bounded scale study, then execute exactly one named job per process.
Every target conditions on the same frozen observations; no quotes are generated,
no bank arrays are read, and the old c=1 runs remain immutable.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_reference as producer
from scripts.evaluate_heston_spread_baseline import check_reference_binding
from sbi4finance.calibration.heston_spread import conditional_c_log_density, conditional_c_mode


MODES = {"half": {"infer_c": False, "c": .5},
         "double": {"infer_c": False, "c": 2.},
         "joint": {"infer_c": True, "c": 1.}}
CASE_MODES = (("synthetic-broad", "half"), ("synthetic-broad", "double"), ("synthetic-broad", "joint"),
              ("synthetic-market-like", "half"), ("synthetic-market-like", "double"), ("synthetic-market-like", "joint"),
              ("market", "joint"))
INNER_BUDGET_SECONDS = 900
OUTER_WALL_SECONDS = 950
SEED = 261112
FIT_BUDGET = 2000
FROZEN_PRODUCER_SHA256 = "69b6267580b91c49b691a867326f8f64ce332dad445204e1fe00166ff6d2f575"


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    from scripts import evaluate_heston_spread_baseline as evaluator
    return {**producer.source_hashes(),
            "scale_consumer": {"path": str(Path(__file__).resolve()), "sha256": producer.digest(__file__)},
            "reference_binding": {"path": str(Path(evaluator.__file__).resolve()), "sha256": producer.digest(evaluator.__file__)}}


def job_spec(panel_path, case_id, mode):
    if (case_id, mode) not in CASE_MODES:
        raise ValueError("job is not in the bounded scale matrix")
    panel, case, pilot, y = producer.load_case(panel_path, case_id, **MODES[mode])
    if panel["fixed_task"]["c_bounds"] != [.5, 2.] or panel["joint_task"]["prior_bounds"][-1] != [.5, 2.]:
        raise ValueError("the frozen Uniform c support must remain[.5,2]")
    return {"id": case_id+"-c-"+mode, "case_id": case_id, "mode": mode,
        "runner_kwargs": dict(MODES[mode]), "task": pilot.task_definition(),
        "names": pilot.names, "bounds": pilot.bounds.tolist(), "observation_sha256": producer.values_digest(y),
        "generating_c": case["generating_c"], "case_kind": case["kind"],
        "physical_truth": case["physical_truth"], "same_observations_as_fixed_c1": True,
        "role": "synthetic inference study" if case["kind"] == "synthetic" else "separate market application/misfit diagnostic"}


def validate_job(spec, panel_path):
    expected = job_spec(panel_path, spec["case_id"], spec["mode"])
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError("scale job target/observations/prior differs from declaration: "+key)
    return expected


def fixed_phi_oracle(panel_path, case_id):
    """Use saved generating prices only: p(c|phi_truth,y), never p(c|y)."""
    _, case, pilot, y = producer.load_case(panel_path, case_id)
    if case["kind"] != "synthetic":
        raise ValueError("a generating-phi oracle is unavailable for market observations")
    residual = (y-np.asarray(case["clean_prices"]))/pilot.records["half_spread"]
    log_density = lambda c: conditional_c_log_density(c, residual, (.5, 2.))
    density = lambda c: np.exp(log_density(c))
    integral, error = quad(density, .5, 2., epsabs=1e-10, epsrel=1e-10, limit=100)
    if not np.isfinite([integral, error]).all() or abs(integral-1.) > 1e-8:
        raise FloatingPointError("saved-price conditional c oracle failed normalization")
    quantiles = [brentq(lambda c: quad(density, .5, c, epsabs=1e-10, epsrel=1e-10)[0]-probability,
                       .5, 2., xtol=1e-10) for probability in (.05, .5, .95)]
    return {"conditioning": "physical parameters fixed at saved generating truth", "physical_truth": case["physical_truth"],
        "generating_c": 1., "residual_sum_squares_over_half_spread": float(residual@residual),
        "quotes": len(residual), "mode": conditional_c_mode(residual, (.5, 2.)),
        "quantiles_05_50_95": quantiles, "density_integral": integral,
        "observation_sha256": producer.values_digest(y), "clean_prices_sha256": case["clean_prices_sha256"],
        "price_calls": 0, "scope": "One-dimensional conditional oracle at fixed phi. Not the full joint posterior or evidence that phi is known during inference."}


def reused_c1_reference(panel_path, references, case_id):
    panel, _, pilot, y = producer.load_case(panel_path, case_id)
    directory = Path(references)/case_id
    run_path, diagnostic_path = directory/"run.json", directory/"diagnostics.json"
    record, diagnostic = read(run_path), read(diagnostic_path)
    if (record.get("status") != "completed" or diagnostic.get("reference_ready") is not True
            or record["task"]["infer_c"] or record["task"]["fixed_c"] != 1.):
        raise ValueError("only completed ready matched c=1 references may be reused")
    chain_path = Path(record["chain_payload_path"])
    check_reference_binding(record, diagnostic, chain_path, target=pilot.task_definition(),
        design_sha256=panel["design_sha256"], case_id=case_id, observations=y,
        panel_sha256=producer.digest(panel_path), products=panel["products"], diagnostics_path=diagnostic_path)
    return {"case_id": case_id, "run_path": str(run_path.resolve()), "run_sha256": producer.digest(run_path),
        "diagnostics_path": str(diagnostic_path.resolve()), "diagnostics_sha256": producer.digest(diagnostic_path),
        "chain_payload_path": str(chain_path.resolve()), "chain_payload_sha256": producer.digest(chain_path),
        "task": record["task"], "observation_sha256": record["observation_sha256"], "reference_ready": True,
        "action": "reuse immutable existing reference; no c=1 rerun"}


def prepare_protocol(panel_path, references, evidence_root, payload_root, out):
    panel_path, references, evidence_root, payload_root, out = map(Path, (panel_path, references, evidence_root, payload_root, out))
    if out.exists() or evidence_root.exists() or payload_root.exists():
        raise FileExistsError("scale protocol/job output roots must be new identities")
    if producer.digest(producer.__file__) != FROZEN_PRODUCER_SHA256:
        raise ValueError("the original reference producer changed")
    panel = read(panel_path)
    c1 = {case: reused_c1_reference(panel_path, references, case)
          for case in ("synthetic-broad", "synthetic-market-like", "market")}
    jobs = []
    for case_id, mode in CASE_MODES:
        spec = job_spec(panel_path, case_id, mode)
        spec.update(evidence_out=str((evidence_root/spec["id"]).resolve()),
            payload_out=str((payload_root/spec["id"]).resolve()))
        jobs.append(spec)
    protocol = {"format": "heston_spread_scale_reference_protocol_v1", "predeclared_before_scale_runs": True,
        "panel_path": str(panel_path.resolve()), "panel_sha256": producer.digest(panel_path),
        "declaration_path": panel["declaration_path"], "declaration_sha256": panel["declaration_sha256"],
        "design_path": panel["design_path"], "design_sha256": panel["design_sha256"], "products": panel["products"],
        "evidence_root": str(evidence_root.resolve()), "payload_root": str(payload_root.resolve()),
        "fixed_c1_references": c1, "jobs": jobs, "source_sha256": source_hashes(),
        "settings": {"chains": 4, "cores": 1, "seed": SEED, "fit_budget": FIT_BUDGET,
            "attempts": [list(attempt) for attempt in producer.ATTEMPTS],
            "escalation": "one6k/12k attempt only when first2k/2k attempt fails MCMC diagnostics; never after numerical failure",
            "inner_cpu_and_wall_seconds": INNER_BUDGET_SECONDS, "outer_total_wall_seconds": OUTER_WALL_SECONDS,
            "outer_limit_scope": "includes interpreter/import/metadata overhead outside the old900-second Budget; not a900-second total-process claim",
            "blas_threads": 1, "automatic_retry_or_budget_extension": False},
        "campaign_limits": {"maximum_concurrent_reference_processes": 2, "whole_campaign_wall_seconds": 3900,
            "maximum_sum_inner_cpu_seconds": 6300,
            "enforcement": "Root campaign controller; this one-job consumer enforces only its own900-second inner and950-second outer limits."},
        "fixed_phi_conditional_oracles": {case: fixed_phi_oracle(panel_path, case)
            for case in ("synthetic-broad", "synthetic-market-like")},
        "joint_c_summary": {"prior": "Uniform(.5,2) in physical c", "lower_tail_threshold": .55,
            "upper_tail_threshold": 1.95, "also_report": "physical-c Pearson correlations and marginal c quantiles; descriptive, not causality or identifiability certification"},
        "excluded": {"synthetic-boundary": "previous capped reference remains inconclusive; no repeat in this matrix",
                     "market-fixed-c-half-and-double": "deferred; only joint-c market application is in this bounded matrix"},
        "observation_model": "Y=P(phi)+c*half_spread*Z; normalized Gaussian includes -n*log(c)",
        "geometry_physical_priors_and_pricing_gate": "unchanged; gate remains.02*(.5*half_spread), independent of fixed/inferred c",
        "reserved_arrays_opened": False, "synthetic_observations_generated": 0,
        "scientific_acceptance": False,
        "scope": "Fixed-scale posterior sensitivity plus six-dimensional joint references on the same observations. Market boundary pressure is a working-model diagnostic, not permission to enlarge the c prior."}
    out.parent.mkdir(parents=True, exist_ok=True)
    producer.write_json(out, protocol)
    return protocol


def validate_protocol(protocol, job_id):
    if protocol.get("format") != "heston_spread_scale_reference_protocol_v1":
        raise ValueError("unsupported scale-reference protocol")
    if (producer.digest(protocol["panel_path"]) != protocol["panel_sha256"]
            or producer.digest(protocol["declaration_path"]) != protocol["declaration_sha256"]
            or producer.digest(protocol["design_path"]) != protocol["design_sha256"]
            or source_hashes() != protocol["source_sha256"]):
        raise ValueError("scale protocol source/data identity differs")
    settings = protocol["settings"]
    if (settings["inner_cpu_and_wall_seconds"] != INNER_BUDGET_SECONDS
            or settings["outer_total_wall_seconds"] != OUTER_WALL_SECONDS
            or settings["attempts"] != [list(attempt) for attempt in producer.ATTEMPTS]
            or settings["seed"] != SEED or settings["fit_budget"] != FIT_BUDGET
            or settings["chains"] != 4 or settings["cores"] != 1):
        raise ValueError("predeclared reference sampler/budget differs")
    if [(job["case_id"], job["mode"]) for job in protocol["jobs"]] != list(CASE_MODES):
        raise ValueError("predeclared scale matrix differs")
    selected = [job for job in protocol["jobs"] if job["id"] == job_id]
    if len(selected) != 1:
        raise ValueError("one predeclared job id required")
    spec = selected[0]
    validate_job(spec, protocol["panel_path"])
    if (Path(spec["evidence_out"]) != Path(protocol["evidence_root"])/job_id
            or Path(spec["payload_out"]) != Path(protocol["payload_root"])/job_id):
        raise ValueError("job output path differs from declared root")
    for key, expected in protocol["fixed_c1_references"].items():
        if reused_c1_reference(protocol["panel_path"], Path(expected["run_path"]).parent.parent, key) != expected:
            raise ValueError("reused c=1 reference changed")
    return spec


def summarize_run(protocol, spec):
    directory = Path(spec["evidence_out"])
    record = read(directory/"run.json")
    panel = read(protocol["panel_path"])
    if (record["task"] != spec["task"] or record["names"] != spec["names"] or record["bounds"] != spec["bounds"]
            or record["case_id"] != spec["case_id"] or record["panel_sha256"] != protocol["panel_sha256"]
            or record["design_sha256"] != protocol["design_sha256"] or record["products"] != protocol["products"]
            or record["observation_sha256"] != spec["observation_sha256"]
            or record["settings"]["cpu_and_wall_seconds"] != INNER_BUDGET_SECONDS):
        raise ValueError("reference result differs from predeclared target/data/budget")
    _, _, _, y = producer.load_case(protocol["panel_path"], spec["case_id"], **spec["runner_kwargs"])
    if not np.array_equal(record["observations"], y):
        raise ValueError("reference observations differ byte-for-byte")
    result = {"status": record["status"], "reference_ready": False, "job_id": spec["id"],
        "run_path": str((directory/"run.json").resolve()), "run_sha256": producer.digest(directory/"run.json"),
        "usage": record.get("usage"), "task": spec["task"], "observation_sha256": spec["observation_sha256"],
        "same_observations_as_fixed_c1": True, "scientific_acceptance": False}
    if record.get("status") != "completed":
        result.update(status="inconclusive", reason="reference failed or exhausted its declared budget; partial data are not a converged posterior")
        return result
    diagnostic_path = directory/"diagnostics.json"
    diagnostic = read(diagnostic_path)
    chain_path = Path(record["chain_payload_path"])
    check_reference_binding(record, diagnostic, chain_path, target=spec["task"], design_sha256=protocol["design_sha256"],
        case_id=spec["case_id"], observations=y, panel_sha256=protocol["panel_sha256"], products=panel["products"], diagnostics_path=diagnostic_path)
    result.update(reference_ready=bool(diagnostic["reference_ready"]), diagnostics_sha256=producer.digest(diagnostic_path),
        chain_payload_path=str(chain_path), chain_payload_sha256=producer.digest(chain_path),
        parameter_mcmc_diagnostics=diagnostic["parameters"], products_mcmc_diagnostics=diagnostic["products"]["parameters"],
        pointwise_precision_passed=diagnostic["pointwise_precision_passed"])
    if not result["reference_ready"]:
        result.update(status="inconclusive", reason="completed trace fails MCMC or sampled numerical precision gates")
        return result
    with np.load(chain_path, allow_pickle=False) as archive:
        draws = archive["parameters"].copy()
    expected_draws = record["attempts"][record["selected_attempt"]]["draws"]
    bounds = np.asarray(spec["bounds"])
    if (draws.shape != (4, expected_draws, len(spec["names"])) or not np.isfinite(draws).all()
            or np.any(draws < bounds[:, 0]) or np.any(draws > bounds[:, 1])):
        raise ValueError("reference chain shape/support differs")
    flat = draws.reshape(-1, len(spec["names"]))
    result["parameter_quantiles_05_50_95"] = {name: np.quantile(flat[:, index], [.05, .5, .95]).tolist()
        for index, name in enumerate(spec["names"])}
    if spec["mode"] == "joint":
        if spec["names"][-1] != "c" or np.any(flat.std(axis=0) <= 0):
            raise ValueError("finite nondegenerate six-dimensional joint draws required")
        result["joint_c"] = {"mean": float(flat[:, -1].mean()), "sd": float(flat[:, -1].std(ddof=1)),
            "quantiles_05_50_95": np.quantile(flat[:, -1], [.05, .5, .95]).tolist(),
            "probability_c_le_055": float(np.mean(flat[:, -1] <= .55)),
            "probability_c_ge_195": float(np.mean(flat[:, -1] >= 1.95)),
            "physical_parameter_c_correlations": dict(zip(spec["names"][:5], np.corrcoef(flat, rowvar=False)[-1, :5].tolist())),
            "scope": "Full joint marginal summaries after convergence gates. Distinct from the fixed-generating-phi conditional oracle; boundary mass is descriptive and does not justify prior expansion."}
    return result


def run_one(protocol_path, job_id):
    protocol_path = Path(protocol_path)
    protocol = read(protocol_path)
    spec = validate_protocol(protocol, job_id)
    directory = Path(protocol["evidence_root"])
    consumer_path = directory/(job_id+".consumer.json")
    if consumer_path.exists() or Path(spec["evidence_out"]).exists() or Path(spec["payload_out"]).exists():
        raise FileExistsError("scale reference job identities are immutable; no automatic retry")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    consumption = {"format": "heston_spread_scale_reference_consumption_v1", "status": "started",
        "protocol_path": str(protocol_path.resolve()), "protocol_sha256": producer.digest(protocol_path),
        "job": spec, "source_sha256": source_hashes(), "reserved_arrays_opened": False,
        "scientific_acceptance": False, "sampler_attempts_are_not_resumable": True}
    # Exclusive creation prevents two controllers claiming the same immutable
    # job while its old-runner output directory is not yet present.
    with consumer_path.open("x") as stream:
        json.dump(consumption, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        producer.run_reference(protocol["panel_path"], spec["case_id"], spec["evidence_out"], spec["payload_out"],
            seed=SEED, budget_seconds=INNER_BUDGET_SECONDS, fit_budget=FIT_BUDGET, **spec["runner_kwargs"])
        if source_hashes() != protocol["source_sha256"] or producer.digest(protocol_path) != consumption["protocol_sha256"]:
            raise ValueError("source/protocol changed during reference run")
        result = summarize_run(protocol, spec)
    except BaseException as error:
        consumption.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic()-started)
        if (Path(spec["evidence_out"])/"run.json").exists():
            try:
                consumption["result"] = summarize_run(protocol, spec)
            except BaseException as binding_error:
                consumption["result_binding_error"] = {"type": type(binding_error).__name__, "message": str(binding_error)}
        producer.write_json(consumer_path, consumption)
        raise
    consumption.update(status="completed", result=result, seconds=time.monotonic()-started)
    producer.write_json(consumer_path, consumption)
    return consumption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("panel", "references", "evidence-root", "payload-root", "out"):
        prepare.add_argument("--"+name, type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--job", required=True)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "prepare":
        protocol = prepare_protocol(args.panel, args.references, args.evidence_root, args.payload_root, args.out)
        print(json.dumps({"protocol": str(args.out), "sha256": producer.digest(args.out), "jobs": [job["id"] for job in protocol["jobs"]]}))
    elif args.worker:
        result = run_one(args.protocol, args.job)
        print(json.dumps({"job": args.job, "reference_ready": result["result"]["reference_ready"], "seconds": result["seconds"]}))
    else:
        protocol = read(args.protocol)
        spec = validate_protocol(protocol, args.job)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], env=env)
        try:
            code = process.wait(timeout=OUTER_WALL_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            directory = Path(protocol["evidence_root"])
            directory.mkdir(parents=True, exist_ok=True)
            producer.write_json(directory/(args.job+".hard-timeout.json"), {"status": "killed_at_outer_wall_limit",
                "outer_total_wall_seconds": OUTER_WALL_SECONDS, "inner_cpu_and_wall_seconds": INNER_BUDGET_SECONDS,
                "job_id": args.job, "reference_ready": False, "automatic_retry": False,
                "scope": "Metadata/import/cleanup overhead is included by this outer cap; old partial artifacts remain immutable."})
            raise TimeoutError("scale reference outer wall cap exhausted")
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()

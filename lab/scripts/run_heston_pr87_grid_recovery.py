"""Separately declared recovery after an operational reference timeout.

Financial task, bank producer, training and diagnostics are unchanged. Reference
failures stay visible and cannot kill an independent bank or fit. This version
uses fresh immutable outputs and a new bounded reference budget; no original
failed records are resumed, overwritten or relabelled as successful.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_heston_spread_scale_capacity as shared

FORMAT = "heston_pr87_grid_recovery_campaign_protocol_v1"
RUNS = (("200k-300", 200000), ("300k-300", 300000))
LIMITS = {"maximum_bank_workers": 1, "bank_seconds": 3600, "maximum_fits": 2,
    "fit_seconds": 3600, "maximum_concurrent_fits": 1,
    "maximum_concurrent_references": 2, "reference_inner_seconds": 2400, "reference_outer_seconds": 2450,
    "maximum_reference_jobs": 3, "evaluation_seconds": 900, "maximum_endpoints_per_fit": 2,
    "maximum_concurrent_phase_workers": 1, "maximum_concurrent_workers": 3,
    "new_grid_pricing_calls": 300612, "automatic_retry": False, "unplanned_extension": False}
ENDPOINTS = {"primary": "final_epoch300", "secondary": "first strictly smallest native val_loss only when epoch differs from300",
    "posterior_diagnostics_select_checkpoint": False}


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": shared.digest(path)}


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked(spec):
    if identity(spec["path"]) != {key: spec[key] for key in ("path", "sha256")}:
        raise ValueError("frozen source/input changed: " + spec["path"])
    return Path(spec["path"])


def readiness(preparation, reference_smoke):
    preparation, reference_smoke = Path(preparation).resolve(), Path(reference_smoke).resolve()
    pilot_path = preparation / "numerical-pilot" / "run.json"
    pilot, task = shared.read(pilot_path), shared.read(preparation / "task.json")
    if (pilot.get("status") != "complete" or len(pilot.get("points", [])) != 98
            or [row["index"] for row in pilot["points"]] != list(range(98))
            or not all(row.get("pointwise_gates_passed") is True for row in pilot["points"])
            or pilot.get("reserved_arrays_opened") is not False
            or task.get("target", {}).get("quote_count") != 63
            or task["target"]["pricing_controls"].get("fourier_representation") != "black_control_variate_v1"
            or task["target"]["prior_bounds"] != [[.01, .16], [.1, 10.], [.01, .16], [.1, 3.], [-.95, -.1], [.5, 2.]]):
        raise ValueError("complete98-point CV pilot and unchanged broad-prior63 task required")
    preparation_protocol = shared.read(checked(pilot["protocol"]))
    panel = shared.read(checked(pilot["panel"]))
    if (preparation_protocol.get("task") != identity(preparation / "task.json")
            or preparation_protocol.get("design") != identity(preparation / "design.json")
            or panel.get("task") != task["target"]
            or panel.get("design") != preparation_protocol["design"]):
        raise ValueError("successful CV pilot must bind the current task/design and conditional panel")
    for spec in preparation_protocol["sources"].values(): checked(spec)
    smoke = shared.read(reference_smoke)
    if (smoke.get("format") != "heston_pr87_grid_reference_run_v1" or smoke.get("status") != "completed"
            or smoke.get("smoke") is not True or not smoke.get("attempts")
            or smoke.get("task") != task["target"]
            or smoke.get("truth_available_to_sampler") is not False):
        raise ValueError("completed native CV reference software smoke required")
    checked(smoke["protocol"])
    return {"pilot": identity(pilot_path), "preparation_protocol": pilot["protocol"],
        "task": identity(preparation / "task.json"), "design": identity(preparation / "design.json"),
        "conditional_panel": pilot["panel"], "reference_smoke": identity(reference_smoke),
        "scope": "Numerical/software readiness only; no posterior-quality acceptance."}


def preservation(previous_evidence, previous_payload, coverage, failed_preparation):
    previous_evidence, previous_payload, coverage, failed_preparation = map(Path,
        (previous_evidence, previous_payload, coverage, failed_preparation))
    prior_path = previous_evidence / "preserved-artifacts-before.json"
    prior = shared.read(prior_path)
    files = {}
    for row in prior["files"]:
        path = checked(row)
        if path.stat().st_size != row["bytes"]: raise ValueError("preserved file size changed")
        files[str(path.resolve())] = row
    for folder in (previous_evidence, previous_payload, coverage, failed_preparation):
        if not folder.is_dir(): raise ValueError("preservation root missing: " + str(folder))
        for path in sorted(folder.rglob("*")):
            if path.is_file(): files[str(path.resolve())] = {**identity(path), "bytes": path.stat().st_size}
    rows = [files[key] for key in sorted(files)]
    return {"format": "heston_pr87_grid_preserved_before_v1", "prior_manifest": identity(prior_path),
        "scope": "previous565 plus completed300epoch evidence/payloads, grid coverage and original failed preparation; immutable",
        "files": rows, "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows)}


def source_bindings():
    from scripts import reproduce_heston_pr87_grid as producer
    from scripts import evaluate_heston_pr87_grid_recovery as evaluator
    from scripts import render_heston_pr87_grid as renderer
    from scripts import export_heston_pr87_grid_figures as exporter
    from scripts import run_heston_pr87_grid_reference_recovery as references
    from scripts import evaluate_heston_pr87_followup as parent_evaluator
    from scripts import render_heston_pr87_followup as parent_renderer
    from scripts import evaluate_heston_pr87_100k as prior_evaluator
    from scripts import render_heston_pr87_100k as prior_renderer
    from scripts import evaluate_heston_pr87_reproduction as original_evaluator
    from scripts import render_heston_pr87_reproduction as original_renderer
    from scripts import export_heston_pr87_reproduction_figures as plotting
    from sbi4finance.calibration import heston_grid
    from sbi4f_tutorial.heston_options import control_variate
    return {"controller": identity(__file__), "producer": identity(producer.__file__),
        "evaluator": identity(evaluator.__file__), "renderer": identity(renderer.__file__),
        "exporter": identity(exporter.__file__), "grid_reference": identity(references.__file__),
        "parent_evaluator": identity(parent_evaluator.__file__), "parent_renderer": identity(parent_renderer.__file__),
        "parent_100k_evaluator": identity(prior_evaluator.__file__), "parent_100k_renderer": identity(prior_renderer.__file__),
        "parent_original_evaluator": identity(original_evaluator.__file__), "parent_original_renderer": identity(original_renderer.__file__),
        "archived_plotting_helper": identity(plotting.__file__), "grid_pilot": identity(heston_grid.__file__),
        "black_control_variate": identity(control_variate.__file__), "black_formula": identity(control_variate.gbm.__file__)}


def prepare(preparation, parent, historical_inputs, old54_control, reference_smoke, evidence, payload, *, native_smoke):
    from scripts import reproduce_heston_pr87_grid as producer
    from scripts import evaluate_heston_pr87_grid_recovery as evaluator
    from scripts import run_heston_pr87_grid_reference_recovery as references
    preparation, parent, historical_inputs, old54_control, reference_smoke, evidence, payload = map(
        lambda p: Path(p).resolve(), (preparation, parent, historical_inputs, old54_control, reference_smoke, evidence, payload))
    gate = readiness(preparation, reference_smoke)
    task = shared.read(gate["task"]["path"])
    parent_record, control = shared.read(parent / "run.json"), shared.read(old54_control)
    if (parent_record.get("status") != "complete" or parent_record.get("protocol", {}).get("train_rows") != 200000
            or parent_record.get("completed_epochs") != 300
            or control.get("format") != "heston_pr87_followup_evaluation_v1" or control.get("status") != "complete"
            or control.get("endpoint") != "primary_final" or control.get("native_epoch") != 300
            or control.get("training_run") != identity(parent / "run.json")):
        raise ValueError("completed old54/200k300 parent and its primary control required")
    planned = [evidence / "campaign-protocol.json", evidence / "bank-protocol.json", evidence / "reference-protocol.json",
        evidence / "preserved-artifacts-before.json", payload / "bank", payload / "bank.log", evidence / "references", payload / "references"]
    planned += [folder / name for name, _ in RUNS for folder in (evidence, payload)]
    if any(path.exists() for path in planned): raise FileExistsError("new immutable grid campaign identities required")
    old_evidence = Path(parent_record["protocol_path"]).parent.parent
    old_payload = parent.parent.parent
    coverage = preparation.parent.parent / "grid-coverage"
    inventory = preservation(old_evidence, old_payload, coverage, preparation.parent / "preparation")
    # Bind every first-attempt record and payload before creating recovery outputs.
    aborted_path = preparation.parent / "campaign.json"
    aborted = shared.read(aborted_path)
    if aborted.get("status") != "failed" or aborted.get("error") != "grid worker process/output failed: reference_synthetic-market-like":
        raise ValueError("the preserved reference-timeout campaign is required")
    first_payload = Path(shared.read(preparation.parent / "bank-protocol.json")["output"]).parent
    files = {row["path"]: row for row in inventory["files"]}
    for folder in (preparation.parent, first_payload):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                files[str(path.resolve())] = {**identity(path), "bytes": path.stat().st_size}
    inventory["files"] = [files[key] for key in sorted(files)]
    inventory.update(file_count=len(files), total_bytes=sum(row["bytes"] for row in files.values()),
        scope="Prior749 plus all original failed grid campaign evidence/payloads; immutable before this separately declared recovery")
    sources = source_bindings()
    native = shared.read(native_smoke)
    if (native.get("format") != "heston_pr87_grid_real_native_smoke_v1" or native.get("status") != "complete"
            or native.get("producer") != sources["producer"] or native.get("real_checked_labels") != 1312
            or [fit.get("rows") for fit in native.get("fits", [])] != [512, 768]
            or not all(fit.get("seeded_reload_exact") is True and fit.get("draw_shape") == [2, 3, 6]
                       for fit in native["fits"])):
        raise ValueError("completed real-price bank and both native-fit/reload software smokes required")
    checked(native["bank"])
    for fit in native["fits"]: checked(fit["run"])
    bank_contract = producer.bank_contract(task, parent, historical_inputs)
    contracts = {name: producer.protocol_contract(task, parent, historical_inputs, train_rows=rows) for name, rows in RUNS}
    if any(contract["bank_contract"] != bank_contract for contract in contracts.values()):
        raise ValueError("both fits must use the exact same frozen bank")
    evidence.mkdir(parents=True, exist_ok=True); payload.mkdir(parents=True, exist_ok=True)
    preserved_path = evidence / "preserved-artifacts-before.json"
    write_new(preserved_path, inventory)
    reference_path = evidence / "reference-protocol.json"
    references.prepare_protocol(gate["conditional_panel"]["path"], evidence / "references", payload / "references", reference_path)
    declared = datetime.now(timezone.utc).isoformat()
    bank_path = evidence / "bank-protocol.json"
    bank = {"format": "heston_pr87_grid_bank_study_v1", "predeclared_before_generation": True,
        "declared_utc": declared, "contract": bank_contract, "task": task, "task_file": gate["task"],
        "output": str(payload / "bank"), "sources": sources, "readiness": gate,
        "preserved_artifacts": identity(preserved_path), "scientific_acceptance": False}
    write_new(bank_path, bank)
    campaign = {"format": FORMAT, "declared_utc": declared, "predeclared_before_bank": True, "limits": LIMITS,
        "sources": sources, "readiness": gate, "preserved_artifacts": identity(preserved_path),
        "parent_training": str(parent), "historical_inputs": identity(historical_inputs),
        "old54_control": identity(old54_control), "bank_protocol": identity(bank_path),
        "bank_log": str(payload / "bank.log"), "reference_protocol": identity(reference_path),
        "reference_logs": {case: str(payload / ("reference-" + case + ".log")) for case in references.CASES},
        "reference_cache_root": str(payload / "pytensor-cache"), "runs": {},
        "native_smoke": identity(native_smoke),
        "rough_initiation": "No rough jobs in this controller. A separate bounded rough numerical initiation needs satisfactory predeclared Heston diagnostics; development success grants no official acceptance.",
        "reserved_arrays_opened": False, "scientific_acceptance": False}
    for name, rows in RUNS:
        run_evidence, run_payload = evidence / name, payload / name
        run_evidence.mkdir(); run_payload.mkdir()
        study = {"format": "heston_pr87_grid_study_v1", "declared_utc": declared,
            "predeclared_before_bank": True, "predeclared_before_fit": True, "contract": contracts[name],
            "task": task, "task_file": gate["task"], "bank_protocol": identity(bank_path),
            "bank_path": str(payload / "bank"), "old54_control": identity(old54_control),
            "reference_protocol": identity(reference_path), "conditional_panel": gate["conditional_panel"],
            "sources": sources, "campaign_limits": LIMITS, "endpoint_policy": ENDPOINTS,
            "output": str(run_payload / "training"), "training_log": str(run_payload / "training.log"),
            "evaluation": evaluator.evaluation_contract(task["target"]), "evaluations": {
                "primary": {"checkpoint": "epoch-000300", "output": str(run_evidence / "evaluation"),
                    "payload_output": str(run_payload / "evaluation"), "log": str(run_payload / "evaluation.log")},
                "secondary": {"checkpoint": "best-validation", "output": str(run_evidence / "evaluation-best-validation"),
                    "payload_output": str(run_payload / "evaluation-best-validation"), "log": str(run_payload / "evaluation-best-validation.log")}},
            "scientific_acceptance": False, "reserved_arrays_opened": False,
            "comparison": "New63 task and observation noise profile; unchanged broad physical/c prior. Both fits reprice one common300k bank, keep old200k truths paired, freeze first10k new63 ZCA and native100 validation.200k/300k complete300passes:60m/90m presentations, not equal compute."}
        path = run_evidence / "protocol.json"
        write_new(path, study)
        campaign["runs"][name] = {"train_rows": rows, "protocol": identity(path)}
    path = evidence / "campaign-protocol.json"
    campaign["recovery_of"] = identity(aborted_path)
    campaign["operational_recovery"] = {
        "original_failed_campaign": identity(aborted_path),
        "reason": "A900s reference cap killed its independent bank at290k labels before final payload save. No numerical failure occurred.",
        "scientific_changes": "None: exact bank contract, all prior/noise/architecture/optimizer/epochs/endpoint and numerical gates retained.",
        "new_reference_budget": "Fresh3-case protocol,2400s inner/2450s outer each; same seeds/initialization/attempt schedule. Ready broad case repeated only for one coherent source-bound protocol, never selective omission.",
        "failure_isolation": "A reference timeout/failure is retained as inconclusive and cannot stop bank/fits; missing references never become acceptance evidence.",
        "resume_or_silent_extension": False}
    write_new(path, campaign)
    return path


def output_paths(campaign):
    bank = shared.read(campaign["bank_protocol"]["path"])
    paths = [bank["output"], campaign["bank_log"]]
    reference = shared.read(campaign["reference_protocol"]["path"])
    paths += [value for job in reference["jobs"] for value in (job["evidence_out"], job["payload_out"], campaign["reference_logs"][job["id"]])]
    for spec in campaign["runs"].values():
        study = shared.read(spec["protocol"]["path"])
        paths += [study["output"], study["training_log"]]
        paths += [entry[key] for entry in study["evaluations"].values() for key in ("output", "payload_output", "log")]
    return paths


def validate(campaign):
    if (campaign.get("format") != FORMAT or campaign.get("limits") != LIMITS
            or campaign.get("predeclared_before_bank") is not True or campaign.get("scientific_acceptance") is not False
            or list(campaign.get("runs", {})) != [name for name, _ in RUNS]):
        raise ValueError("predeclared bounded grid campaign required")
    for spec in [*campaign["sources"].values(), *campaign["readiness"].values(), campaign["historical_inputs"],
                 campaign["old54_control"], campaign["bank_protocol"], campaign["reference_protocol"], campaign["preserved_artifacts"]]:
        if isinstance(spec, dict) and {"path", "sha256"} <= set(spec): checked(spec)
    if campaign["sources"]["controller"] != identity(__file__): raise ValueError("controller source changed")
    if campaign.get("native_smoke"): checked(campaign["native_smoke"])
    bank = shared.read(campaign["bank_protocol"]["path"])
    contract = bank["contract"]
    if (contract.get("format") != "heston_pr87_grid_bank_contract_v1" or contract.get("train_rows") != 300000
            or contract.get("validation_rows") != 100 or contract.get("historical_rows") != 512
            or contract.get("quote_count") != 63 or contract.get("whitening_rows") != 10000):
        raise ValueError("common300k/63-feature bank contract required")
    for spec in contract["source"].values(): checked(spec)
    reference = shared.read(campaign["reference_protocol"]["path"])
    if (reference.get("format") != "heston_pr87_grid_reference_protocol_v1"
            or reference["settings"]["smoke_test_only"] is not False
            or reference["settings"]["inner_cpu_and_wall_seconds"] != 2400
            or reference["settings"]["outer_total_wall_seconds"] != 2450
            or reference["campaign_limits"]["maximum_concurrent_reference_processes"] != 2
            or [job["id"] for job in reference["jobs"]] != ["synthetic-broad", "synthetic-market-like", "market"]):
        raise ValueError("three production references with declared two-worker bounds required")
    for name, rows in RUNS:
        spec = campaign["runs"][name]
        study = shared.read(checked(spec["protocol"]))
        fit = study["contract"]
        if (spec["train_rows"] != rows or fit.get("format") != "heston_pr87_grid_training_contract_v1"
                or fit.get("train_rows") != rows or fit.get("epochs") != 300 or fit.get("batch_size") != 256
                or fit.get("bank_contract") != contract or study["task"] != bank["task"]
                or study["bank_protocol"] != campaign["bank_protocol"] or study["reference_protocol"] != campaign["reference_protocol"]
                or study["sources"] != campaign["sources"] or study["endpoint_policy"] != ENDPOINTS
                or study.get("predeclared_before_bank") is not True):
            raise ValueError("both immutable native300-epoch fit protocols must share the declared bank/reference target")
    for row in shared.read(campaign["preserved_artifacts"]["path"])["files"]:
        checked(row)
        if Path(row["path"]).stat().st_size != row["bytes"]: raise ValueError("preserved size changed")
    paths = [str(Path(path).resolve()) for path in output_paths(campaign)]
    if len(set(paths)) != len(paths): raise ValueError("all worker output identities must be distinct")
    return paths


def reference_environment(env, cache):
    result = dict(env)
    flags = [item for item in result.get("PYTENSOR_FLAGS", "").split(",") if item
             and item.split("=", 1)[0].strip() not in ("base_compiledir", "compiledir")]
    result["PYTENSOR_FLAGS"] = ",".join([*flags, "base_compiledir=" + str(cache)])
    return result


def run(protocol_path, out):
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    campaign = shared.read(protocol_path)
    paths = validate(campaign)
    if out.exists() or out.with_suffix(".claim.json").exists() or any(Path(path).exists() for path in paths):
        raise FileExistsError("new grid outputs required; no retries")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_new(out.with_suffix(".claim.json"), identity(protocol_path))
    result = {"format": "heston_pr87_grid_campaign_v1", "status": "running", "protocol": identity(protocol_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "jobs": [], "runs": {}, "references": {}, "automatic_retry": False, "scientific_acceptance": False,
        "recovery_of": campaign["recovery_of"], "operational_recovery": campaign["operational_recovery"]}
    env = dict(os.environ)
    env.update(KERAS_BACKEND="jax", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR="/private/tmp/sbi4f-matplotlib")
    active = {}
    ref = shared.read(campaign["reference_protocol"]["path"])
    pending_refs = deque(ref["jobs"])
    started = time.monotonic()
    shared.write(out, result)

    def spawn(name, command, log_path, cap, expected, kind, child_env=None):
        validate(campaign)
        entry = {"id": name, "command": command, "log": str(log_path), "wall_seconds_limit": cap,
            "kind": kind, "status": "running"}
        stream = Path(log_path).open("x")
        try: process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=child_env or env)
        except BaseException: stream.close(); raise
        entry["pid"] = process.pid
        result["jobs"].append(entry)
        active[name] = {"process": process, "stream": stream, "started": time.monotonic(),
            "entry": entry, "expected": Path(expected), "kind": kind}
        shared.write(out, result)
        return name

    def fill_references():
        while pending_refs and sum(job["kind"] == "reference" for job in active.values()) < 2:
            spec = pending_refs.popleft()
            command = [sys.executable, campaign["sources"]["grid_reference"]["path"], "--panel", ref["panel_path"],
                "--protocol", campaign["reference_protocol"]["path"], "--case-id", spec["id"],
                "--out", spec["evidence_out"], "--payload-out", spec["payload_out"], "--worker"]
            child_env = reference_environment(env, Path(campaign["reference_cache_root"]) / spec["id"])
            spawn("reference_" + spec["id"], command, campaign["reference_logs"][spec["id"]], 2450,
                Path(spec["evidence_out"]) / "run.json", "reference", child_env)

    def failed_reference(name, job, reason, elapsed, code=None):
        job["stream"].close()
        entry = job["entry"]
        entry.update(status="failed_inconclusive", seconds=elapsed, returncode=code, error=reason)
        case_id = name.removeprefix("reference_")
        record = identity(job["expected"]) if job["expected"].is_file() else None
        if record is not None: entry["record"] = record
        result["references"][case_id] = {"run": record, "reference_ready": False,
            "scope": "inconclusive; independent worker failed/capped, no case substitution", "error": reason}
        del active[name]
        shared.write(out, result)

    def pump():
        for name, job in list(active.items()):
            entry, process = job["entry"], job["process"]
            code = process.poll()
            elapsed = time.monotonic() - job["started"]
            if code is None:
                if elapsed >= entry["wall_seconds_limit"]:
                    if job["kind"] == "reference":
                        shared.stop(process)
                        failed_reference(name, job, "declared outer wall cap exhausted", elapsed, process.poll())
                        continue
                    entry.update(status="wall_cap", seconds=elapsed)
                    raise TimeoutError("grid worker exceeded its wall cap: " + name)
                continue
            job["stream"].close()
            entry.update(returncode=code, seconds=elapsed)
            allowed = "completed" if job["kind"] == "reference" else "complete"
            if code or not job["expected"].is_file() or shared.read(job["expected"]).get("status") != allowed:
                if job["kind"] == "reference":
                    failed_reference(name, job, "reference process/output failed", elapsed, code)
                    continue
                entry["status"] = "failed"
                raise RuntimeError("grid worker process/output failed: " + name)
            record = shared.read(job["expected"])
            entry.update(status="complete", record=identity(job["expected"]))
            if job["kind"] == "reference":
                result["references"][record["case_id"]] = {"run": entry["record"],
                    "reference_ready": record.get("reference_ready") is True,
                    "scope": "ready" if record.get("reference_ready") is True else "inconclusive; no case substitution"}
            del active[name]
            shared.write(out, result)
        fill_references()

    def wait_for(name=None):
        while ((name in active) if name is not None else (pending_refs or any(job["kind"] == "reference" for job in active.values()))):
            pump()
            time.sleep(.1)

    try:
        bank = shared.read(campaign["bank_protocol"]["path"])
        command = [sys.executable, campaign["sources"]["producer"]["path"], "generate", "--task", campaign["readiness"]["task"]["path"],
            "--parent", campaign["parent_training"], "--historical-inputs", campaign["historical_inputs"]["path"],
            "--protocol", campaign["bank_protocol"]["path"], "--out", bank["output"], "--worker"]
        bank_id = spawn("bank", command, campaign["bank_log"], 3600, Path(bank["output"]) / "bank.json", "bank")
        fill_references()
        wait_for(bank_id)
        for name, rows in RUNS:
            study_path = campaign["runs"][name]["protocol"]["path"]
            study = shared.read(study_path)
            command = [sys.executable, study["sources"]["producer"]["path"], "train", "--bank", bank["output"],
                "--protocol", study_path, "--out", study["output"], "--train-rows", str(rows), "--worker"]
            job_id = spawn(name + "_training", command, study["training_log"], 3600,
                Path(study["output"]) / "training.json", "fit")
            wait_for(job_id)
            training = shared.read(Path(study["output"]) / "training.json")
            best = int(training["best_validation_archive"]["epoch"])
            result["runs"][name] = {"training": identity(Path(study["output"]) / "training.json"), "best_validation_epoch": best}
            wait_for()  # Every reference process must finish before any posterior evaluation.
            for role, spec in study["evaluations"].items():
                if role == "secondary" and best == 300:
                    result["runs"][name]["secondary_skipped"] = "same epoch300 primary"
                    continue
                checkpoint = Path(study["output"]) / "checkpoints" / spec["checkpoint"]
                command = [sys.executable, study["sources"]["evaluator"]["path"], "--checkpoint", str(checkpoint),
                    "--protocol", study_path, "--out", spec["output"], "--payload-out", spec["payload_output"], "--worker"]
                job_id = spawn(name + "_" + role + "_evaluation", command, spec["log"], 900,
                    Path(spec["output"]) / "evaluation.json", "evaluation")
                wait_for(job_id)
        wait_for()
        validate(campaign)
        if identity(protocol_path) != result["protocol"]: raise ValueError("campaign protocol changed")
        result.update(status="complete", seconds=time.monotonic() - started,
            all_references_ready=all(row.get("reference_ready") is True for row in result["references"].values()) and len(result["references"]) == 3)
        shared.write(out, result)
        return result
    except BaseException as error:
        for job in active.values():
            shared.stop(job["process"])
            job["stream"].close()
            if job["entry"]["status"] == "running": job["entry"]["status"] = "stopped_after_peer_failure"
        result.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic() - started)
        shared.write(out, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for key in ("preparation", "parent", "historical-inputs", "old54-control", "reference-smoke", "evidence", "payload"):
        prep.add_argument("--" + key, required=True, type=Path)
    prep.add_argument("--native-smoke", required=True, type=Path)
    launch = commands.add_parser("run")
    launch.add_argument("--protocol", required=True, type=Path)
    launch.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.preparation, args.parent, args.historical_inputs, args.old54_control,
            args.reference_smoke, args.evidence, args.payload, native_smoke=args.native_smoke))
    else:
        print(json.dumps(run(args.protocol, args.out)), flush=True)

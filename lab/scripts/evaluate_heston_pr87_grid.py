"""Raw six-target diagnostics for the revised63-quote native FlowMatching fits.

Truth rows remain paired to the immutable old54 control; observations and all
conditional references belong to the new financial target. No old54 chain is
substituted for a new63 reference. Every raw neural draw is retained.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_pr87_followup as parent_evaluation
from scripts import run_heston_pr87_grid_reference as reference

fixed, baseline = parent_evaluation.fixed, parent_evaluation.baseline
read, binding, checked_binding = parent_evaluation.read, parent_evaluation.binding, parent_evaluation.checked_binding
raw_support, population_summary = parent_evaluation.raw_support, parent_evaluation.population_summary
descriptive_comparison = parent_evaluation.descriptive_comparison
SEED, ROWS, DRAWS, CONDITIONAL_DRAWS = 261112, 512, 512, 4096
NATIVE_ROWS, NATIVE_DRAWS, NATIVE_SEED, WALL_SECONDS = 100, 300, 709012, 900


def evaluation_contract(target):
    if (target.get("infer_c") is not True or target.get("quote_count") != 63
            or len(target.get("parameter_names", [])) != 6 or target["parameter_names"][-1] != "c"
            or target.get("prior_bounds", [])[-1:] != [[.5, 2.]]
            or target.get("pricing_controls", {}).get("fourier_representation") != "black_control_variate_v1"):
        raise ValueError("unchanged six-target posterior on the revised63-quote grid required")
    return {"native_validation_rows": NATIVE_ROWS, "native_validation_draws": NATIVE_DRAWS,
        "native_sampling_seed": NATIVE_SEED, "historical_population_rows": ROWS,
        "historical_draws_per_case": DRAWS, "historical_seed": SEED,
        "historical_population_batch_rows": 16,
        "historical_population_sampling_seed": "261112 + 1000 + zero_based_batch_start",
        "historical_population_sbc_seed": SEED + 2000, "conditional_cases": list(reference.CASES),
        "conditional_draws": CONDITIONAL_DRAWS, "conditional_sampling_seed": "261112 + zero_based_case_index",
        "order": ["native_validation", "historical_population", "new63_conditionals"],
        "raw_draw_policy": "retain every raw unbounded draw; no clipping, filtering, rejection or resampling",
        "calibration": "all raw finite draws; any nonfinite draw makes the whole panel's calibration undefined",
        "support": "per-sample and per-dimension outside-prior, nonphysical and nonfinite counts",
        "checkpoint_selection": "final epoch300 primary; one minimum-validation secondary only when its epoch differs",
        "supported_train_rows": [200000, 300000], "epochs": 300,
        "presentations": "60m for200k300,90m for300k300; old54 control200k300 has60m",
        "equal_compute_data_size_comparison": False,
        "truth_pairing": "same native100 and historical512 six-target truths as old54 control; new63 observations",
        "conditional_reference": "new63 target-matched nativePyMC references only; missing or unready stays inconclusive",
        "sbc_scope": "six-marginal DKW screen within each panel; reused development data, no simultaneous acceptance family",
        "max_posterior_evaluation_seconds_per_endpoint": WALL_SECONDS,
        "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False}


def training_accounting(run):
    contract = run.get("protocol", {}); rows = contract.get("train_rows")
    if rows not in (200000, 300000): raise ValueError("revised-grid training rows must be200k or300k")
    steps = (rows + 255) // 256
    if (run.get("format") != "heston_pr87_grid_run_v1" or run.get("status") != "complete"
            or run.get("completed_epochs") != 300 or run.get("actual_optimizer_updates") != 300 * steps
            or contract.get("epochs") != 300 or contract.get("batch_size") != 256
            or contract.get("validation_rows") != 100 or contract.get("batches_per_epoch") != steps
            or contract.get("expected_optimizer_updates") != 300 * steps):
        raise ValueError("completed revised-grid300-epoch native accounting required")
    return {"train_rows": rows, "steps_per_epoch": steps, "epochs": 300,
        "primary_optimizer_updates": 300 * steps, "primary_presentations": rows * 300,
        "tail_batch_rows": rows % 256 or 256}


def checkpoint_endpoint(checkpoint, metadata, run):
    accounting = training_accounting(run); epoch = metadata.get("native_epoch"); checkpoint = Path(checkpoint).resolve()
    if (type(epoch) is not int or not 1 <= epoch <= 300
            or metadata.get("actual_optimizer_updates") != epoch * accounting["steps_per_epoch"]):
        raise ValueError("revised-grid checkpoint epoch or optimizer accounting differs")
    if checkpoint == Path(run["primary_checkpoint"]).resolve() and epoch == 300: return "primary_final"
    best = run.get("best_validation_archive", {})
    if (epoch != 300 and best.get("epoch") == epoch and best.get("path")
            and checkpoint == Path(best["path"]).resolve()): return "secondary_best_validation"
    raise ValueError("only final300 and distinct predeclared minimum-validation endpoints are permitted")


def paired_bank_inputs(run, task, control, protected):
    bank_path = Path(run["bank_path"]).resolve(); record_path = bank_path/"bank.json"
    if fixed.digest(record_path) != run["bank_record_sha256"]: raise ValueError("grid bank record identity differs")
    bank = read(record_path); payload = Path(run["data_payload"]).resolve()
    if (bank.get("format") != "heston_pr87_grid_bank_v1" or bank.get("status") != "complete"
            or bank.get("task") != task or bank.get("protocol") != run["protocol"]["bank_contract"]
            or payload != bank_path/bank["data_payload"] or fixed.digest(payload) != run["data_payload_sha256"]
            or bank.get("whitening_sha256") != run["whitening_sha256"]):
        raise ValueError("complete common revised-grid bank binding differs")
    protected[str(record_path)] = run["bank_record_sha256"]; protected[str(payload)] = run["data_payload_sha256"]
    names = task["target"]["parameter_names"]; quote_count = task["target"]["quote_count"]
    inputs = {}
    with np.load(payload, allow_pickle=False) as archive:
        for label, prefix, rows, old_panel in (("native_validation", "validation", 100, "native_validation"), ("population", "historical", 512, "population")):
            quotes, observations = archive[prefix+"_quotes"].copy(), archive[prefix+"_observations"].copy()
            truth = np.column_stack([archive[prefix+"_"+name].reshape(-1) for name in names])
            if (quotes.shape != (rows, quote_count) or truth.shape != (rows, 6) or not np.isfinite(quotes).all()
                    or not np.array_equal(observations, quotes/np.asarray(task["records"]["half_spread"]))
                    or baseline.observations_digest(truth) != control[old_panel]["truth_sha256"]):
                raise ValueError("new63 panel truth pairing or exact condition reconstruction differs")
            ids = archive[prefix+"_row_id"].copy()
            if ids.shape != (rows,) or len(set(ids.tolist())) != rows:
                raise ValueError("paired panel row identifiers differ")
            if prefix == "historical" and not np.array_equal(ids, np.array([f"validation:{i:08d}" for i in range(rows)])):
                raise ValueError("historical512 paired row identifiers differ")
            inputs[label] = (quotes, truth, ids)
    return inputs, bank


def evaluate(checkpoint, protocol_path, out, payload_out, *, max_seconds=WALL_SECONDS):
    started = time.monotonic(); checkpoint, protocol_path, out, payload_out = map(Path, (checkpoint, protocol_path, out, payload_out))
    if out.exists() or payload_out.exists(): raise FileExistsError("revised-grid evaluation identities must be new")
    if not 0 < max_seconds <= WALL_SECONDS: raise ValueError("evaluation budget must be in(0,900]")
    protocol = read(protocol_path); protected = {}
    from scripts import reproduce_heston_pr87_grid as trainer
    workflow, metadata = trainer.load_checkpoint(checkpoint)
    task, target = metadata["task"], metadata["task"]["target"]
    if (metadata.get("format") != "heston_pr87_grid_checkpoint_v1" or protocol.get("task") != task
            or protocol.get("evaluation") != evaluation_contract(target)
            or protocol.get("sources", {}).get("evaluator") != binding(__file__)
            or protocol["sources"].get("parent_evaluator") != binding(parent_evaluation.__file__)
            or protocol["sources"].get("grid_reference") != binding(reference.__file__)
            or Path(metadata["protocol_path"]).resolve() != protocol_path.resolve()
            or metadata["protocol_sha256"] != fixed.digest(protocol_path)):
        raise ValueError("source-frozen revised-grid checkpoint/evaluation protocol required")
    for spec in protocol["sources"].values(): checked_binding(spec, protected)
    checked_binding(binding(protocol_path), protected)
    run_path = checkpoint.parent.parent/"run.json"; run = read(run_path)
    endpoint = checkpoint_endpoint(checkpoint, metadata, run); accounting = training_accounting(run)
    protected[str(run_path.resolve())] = fixed.digest(run_path)
    for name in ("checkpoint.json", "model.keras", "state.npz", "whitening.json"):
        path = checkpoint/name
        if path.is_file(): protected[str(path.resolve())] = fixed.digest(path)
    control = read(checked_binding(protocol["old54_control"], protected))
    if (control.get("format") != "heston_pr87_followup_evaluation_v1"
            or control.get("status") != "complete" or control.get("native_epoch") != 300 or control.get("training_rows") != 200000
            or control.get("endpoint") != "primary_final" or control["task"]["quote_count"] != 54
            or control["task"]["parameter_names"] != target["parameter_names"] or control["task"]["prior_bounds"] != target["prior_bounds"]):
        raise ValueError("preserved old54/200k300 control with unchanged target priors required")
    inputs, bank = paired_bank_inputs(run, task, control, protected)
    ref_protocol_path = checked_binding(protocol["reference_protocol"], protected); ref_protocol = read(ref_protocol_path)
    panel_path = checked_binding(protocol["conditional_panel"], protected)
    if Path(ref_protocol["panel_path"]).resolve() != panel_path.resolve(): raise ValueError("conditional panel/reference protocol differs")
    reference.validate_protocol(ref_protocol, reference.CASES[0])
    panel, _, _, _ = reference.load_case(panel_path, reference.CASES[0])
    if panel["task"] != target: raise ValueError("new63 conditional target differs")
    out.mkdir(parents=True); payload_out.mkdir(parents=True)
    record = {"format": "heston_pr87_grid_evaluation_v1", "status": "started", "task": target,
        "protocol": binding(protocol_path), "checkpoint": binding(checkpoint/"checkpoint.json"),
        "training_run": binding(run_path), "native_epoch": metadata["native_epoch"], "endpoint": endpoint,
        "actual_optimizer_updates": metadata["actual_optimizer_updates"], "training_rows": accounting["train_rows"],
        "training_presentations": metadata["native_epoch"]*accounting["train_rows"], "training_accounting": accounting,
        "old54_control": protocol["old54_control"], "reference_protocol": protocol["reference_protocol"],
        "conditional_panel": protocol["conditional_panel"], "common_bank": binding(Path(run["bank_path"])/"bank.json"),
        "seed": SEED, "native_seed": NATIVE_SEED, "cases": {}, "max_seconds": max_seconds,
        "reserved_arrays_opened": False, "new_reference_sampling": False, "scientific_acceptance": False,
        "truth_pairing": "native100 and historical512 truth identities equal old54 control; new63 quotes target revised likelihood",
        "loss_interpretation": "native FlowMatching objective, not density NLL",
        "source_sha256": {"evaluator": binding(__file__), "trainer": binding(trainer.__file__), "reference_consumer": binding(reference.__file__)}}
    fixed.write_json(out/"run.json", record)
    def deadline():
        if time.monotonic()-started >= max_seconds: raise TimeoutError("revised-grid posterior evaluation cap exhausted")
    def query(quotes, count, seed):
        deadline(); values = np.asarray(trainer.sample(workflow, task, quotes, count, seed=seed)); deadline()
        if values.shape != (len(quotes), count, 6): raise ValueError("raw revised-grid posterior shape differs")
        return values
    try:
        for label, count, sampling_seed, sbc_seed, batch in (("native_validation", 300, NATIVE_SEED, NATIVE_SEED+2000, 100), ("population", 512, SEED+1000, SEED+2000, 16)):
            quotes, truth, ids = inputs[label]
            draws = np.concatenate([query(quotes[start:start+batch], count, sampling_seed+start) for start in range(0, len(quotes), batch)])
            raw_path = payload_out/(label+"-draws.npz"); fixed.write_npz(raw_path, {"draws": draws})
            summary, quantiles = population_summary(draws, truth, quotes, target, sbc_seed)
            summary["raw_draw_payload"] = binding(raw_path)
            if quantiles is not None:
                path = out/("native-recovery.npz" if label == "native_validation" else "recovery.npz")
                fixed.write_npz(path, {"truth": truth, "quantiles": quantiles.astype(np.float32), "row_id": ids}); summary["recovery_payload"] = binding(path)
            record[label] = summary; fixed.write_json(out/"run.json", record)
        conditional = {}
        for index, case_id in enumerate(reference.CASES):
            _, _, _, observation = reference.load_case(panel_path, case_id)
            draws = query(observation[None], CONDITIONAL_DRAWS, SEED+index)[0]; conditional[case_id] = draws
            chains, ref_binding = reference.matched_reference(ref_protocol, case_id, target, observation, protected)
            entry = {"observation_sha256": baseline.observations_digest(observation), "support": raw_support(draws, target["prior_bounds"], target["parameter_names"]), **ref_binding}
            if chains is not None: entry["comparison"] = descriptive_comparison(draws, chains, target["parameter_names"], panel["products"], target["prior_bounds"])
            record["cases"][case_id] = entry; fixed.write_json(out/"run.json", record)
        path = payload_out/"conditional-draws.npz"; fixed.write_npz(path, conditional)
        record["conditional_draws"] = {**binding(path), "draws_per_case": CONDITIONAL_DRAWS}
        if any(fixed.digest(path) != sha for path, sha in protected.items()): raise ValueError("protected revised-grid input changed")
        deadline(); record.update(status="complete", seconds=time.monotonic()-started, protected_inputs_unchanged=True)
        fixed.write_json(out/"evaluation.json", record); fixed.write_json(out/"run.json", record)
        return record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.monotonic()-started)
        fixed.write_json(out/"failure.json", record); fixed.write_json(out/"run.json", record); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "protocol", "out", "payload-out"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=WALL_SECONDS); parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= WALL_SECONDS: parser.error("wall-seconds must be in(0,900]")
    if not args.worker:
        try: raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], timeout=args.wall_seconds).returncode)
        except subprocess.TimeoutExpired: raise SystemExit(124)
    result = evaluate(args.checkpoint,args.protocol,args.out,args.payload_out,max_seconds=args.wall_seconds)
    print(json.dumps({"status": result["status"], "epoch": result["native_epoch"], "seconds": result["seconds"]}))

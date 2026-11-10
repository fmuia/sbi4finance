"""One predeclared 50k->60k learning-rate comparison for the fixed-c Heston flow.

The saved Adam moments, iteration, standardizers and auxiliary RNG arrays are
restored exactly. Only its learning-rate schedule changes. No automatic retry
or extension, pricing, reserved-pool access or scientific acceptance is provided.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_heston_spread_baseline as trainer
from scripts import continue_heston_spread_baseline as continuation


START, ENDPOINTS, TOTAL = 50000, (55000, 60000), 60000
BOUNDARY = 54999  # PiecewiseConstantDecay includes the boundary in the first interval.
RATES = (5e-6, 5e-7)
MAX_WALL_SECONDS = 300
FROZEN_CONTINUATION_SHA256 = "3c14cb7461d4c0ab741f0a9d279ff99fad75fa20f58f0f1185f265e72d84fc9b"


def source_paths():
    from scripts import evaluate_heston_spread_baseline as evaluator
    from scripts import run_heston_spread_reference as reference
    from sbi4f_tutorial.heston_options import model
    spread = Path(sys.modules[trainer.HestonSpreadPilot.__module__].__file__)
    return {"annealing": Path(__file__), "trainer": Path(trainer.__file__),
        "continuation": Path(continuation.__file__), "evaluator": Path(evaluator.__file__),
        "reference": Path(reference.__file__), "batch_plan": Path(sys.modules[trainer.BatchPlan.__module__].__file__),
        "spread_pilot": spread, "heston_pilot": spread.with_name("heston_pilot.py"), "heston_model": Path(model.__file__)}


def validate_parent(parent):
    trainer.validate_task(parent["task"])
    if (parent.get("format") != "heston_spread_baseline_checkpoint_v1"
            or parent.get("actual_optimizer_updates") != START or parent.get("planned_updates") != START
            or parent.get("dropout_zero_control") is not True or parent["train_rows"] != 20000
            or parent["batch_size"] != 128 or parent["seed"] != 9011 or parent["schedule_updates"] != 10000
            or parent["architecture"] != {"family": "spline", "widths": [128, 128], "subnet_dropout": 0.}
            or parent["probe_rows"] != {"train": 512, "validation": 512}
            or parent["task_fingerprint"] != trainer.fingerprint(parent["task"])
            or parent["names"] != parent["task"]["target"]["parameter_names"]
            or parent["bounds"] != parent["task"]["target"]["prior_bounds"]
            or parent["design_sha256"] != parent["task"]["design_sha256"]
            or parent["source_sha256"]["trainer"] != continuation.FROZEN_TRAINER_SHA256
            or trainer.digest(trainer.__file__) != continuation.FROZEN_TRAINER_SHA256
            or parent["source_sha256"].get("continuation") != FROZEN_CONTINUATION_SHA256
            or trainer.digest(continuation.__file__) != FROZEN_CONTINUATION_SHA256
            or not parent.get("paired_default_reference", {}).get("initial_deterministic_probes_exact")):
        raise ValueError("annealing requires the frozen 50k fixed-c zero-dropout parent")
    schedule = parent["optimizer_schedule"]["serialized"]
    if (schedule["class_name"] != "CosineDecay" or schedule["config"]["decay_steps"] != 10000
            or schedule["config"]["initial_learning_rate"] != 5e-4 or schedule["config"]["alpha"] != .1):
        raise ValueError("parent original optimizer schedule differs")
    for name, path in source_paths().items():
        if name in ("batch_plan", "spread_pilot", "heston_pilot", "heston_model"):
            if trainer.digest(path) != parent["source_sha256"][name]:
                raise ValueError("parent mathematical/data-order source differs")


def make_schedule():
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import keras
    return keras.optimizers.schedules.PiecewiseConstantDecay([BOUNDARY], list(RATES), name="heston_spread_two_stage_annealing")


def hashes(model):
    return {group: [row["sha256"] for row in rows]
            for group, rows in trainer.state_summary(model, trainer.current_state(model)).items()}


def install_schedule(workflow, parent):
    """Use the public setter, and prove that no numerical state was reset."""
    model = workflow.estimator
    old_config, before = model.optimizer.get_config(), hashes(model)
    if (old_config["learning_rate"] != parent["optimizer_schedule"]["serialized"]
            or int(model.optimizer.iterations.numpy()) != START):
        raise ValueError("restored optimizer schedule/iteration differs before assignment")
    model.optimizer.learning_rate = make_schedule()
    new_config = model.optimizer.get_config()
    if ({key: value for key, value in old_config.items() if key != "learning_rate"}
            != {key: value for key, value in new_config.items() if key != "learning_rate"}
            or before != hashes(model)):
        raise RuntimeError("learning-rate assignment changed Adam configuration or model arrays")
    schedule = make_schedule()
    sample_steps = (50000, 54999, 55000, 59999, 60000)
    values = [float(schedule(step)) for step in sample_steps]
    if not np.allclose(values, (RATES[0], RATES[0], RATES[1], RATES[1], RATES[1]), rtol=1e-7, atol=0):
        raise RuntimeError("annealing schedule boundary values differ")
    return {"serialized": new_config["learning_rate"], "absolute_step_rates": dict(zip(map(str, sample_steps), values)),
            "all_saved_array_hashes_unchanged_at_assignment": True, "other_adam_configuration_unchanged": True,
            "old_serialized": old_config["learning_rate"]}


def prepare(checkpoint, out, panel_path, references, protocol_path):
    checkpoint, out, panel_path, references, protocol_path = map(Path, (checkpoint, out, panel_path, references, protocol_path))
    if out.exists() or protocol_path.exists():
        raise FileExistsError("annealing output/protocol identities are immutable")
    parent = json.loads((checkpoint/"checkpoint.json").read_text())
    validate_parent(parent)
    if (trainer.digest(checkpoint/"model.keras") != parent["model_sha256"]
            or trainer.digest(checkpoint/"state.npz") != parent["state_sha256"]):
        raise ValueError("parent payload hashes differ")
    from scripts.run_heston_spread_reference import load_case
    from scripts.evaluate_heston_spread_baseline import check_reference_binding
    panel = json.loads(panel_path.read_text())
    if panel["fixed_task"] != parent["task"]["target"] or panel["design_sha256"] != parent["design_sha256"]:
        raise ValueError("annealing panel differs from parent task")
    reference_bindings = {}
    protected = [checkpoint/name for name in ("checkpoint.json", "model.keras", "state.npz")]
    protected += [panel_path, Path(panel["declaration_path"]), Path(parent["design_path"]), Path(parent["bank_path"])/"bank.json"]
    bank_record = json.loads((Path(parent["bank_path"])/"bank.json").read_text())
    continuation.validate_bank_binding(parent, {"bank_record_sha256": trainer.digest(Path(parent["bank_path"])/"bank.json"),
        "bank_payload_sha256": bank_record["payload_sha256"], "bank_identity": bank_record["identity"]})
    protected += [Path(parent["bank_path"])/bank_record["payload"]]
    for selected in panel["cases"]:
        case_id = selected["id"]
        _, _, _, y = load_case(panel_path, case_id)
        run_path, diag_path = references/case_id/"run.json", references/case_id/"diagnostics.json"
        run_record = json.loads(run_path.read_text())
        diagnostic = json.loads(diag_path.read_text()) if diag_path.exists() else None
        ready = run_record.get("status") == "completed" and diagnostic is not None and diagnostic.get("reference_ready") is True
        if ready:
            chain_path = Path(run_record["chain_payload_path"])
            check_reference_binding(run_record, diagnostic, chain_path, target=parent["task"]["target"],
                design_sha256=parent["design_sha256"], case_id=case_id, observations=y,
                panel_sha256=trainer.digest(panel_path), products=panel["products"], diagnostics_path=diag_path)
            protected += [chain_path]
        if ready != (case_id in ("synthetic-broad", "synthetic-market-like", "market")):
            raise ValueError("the declared three ready references plus inconclusive boundary differ")
        reference_bindings[case_id] = {"ready": ready, "run_path": str(run_path.resolve()),
            "run_sha256": trainer.digest(run_path)}
        protected += [run_path]
        if ready:
            reference_bindings[case_id]["diagnostics_sha256"] = trainer.digest(diag_path)
            protected += [diag_path]
        else:
            # A capped failed reference has no final diagnostics file. Bind its
            # actual failure and first-attempt evidence without inventing one.
            failure_path = references/case_id/"failure.json"
            first_path = references/case_id/"first-attempt-diagnostics.json"
            if (case_id != "synthetic-boundary" or run_record.get("reference_ready") is not False
                    or not failure_path.exists() or not first_path.exists()):
                raise ValueError("boundary reference failure evidence differs")
            reference_bindings[case_id].update(status="inconclusive", failure_sha256=trainer.digest(failure_path),
                first_attempt_diagnostics_sha256=trainer.digest(first_path))
            protected += [failure_path, first_path]
    protected += list(source_paths().values())
    protocol = {"format": "heston_spread_annealing_protocol_v1", "predeclared_before_fit": True,
        "parent_checkpoint": str(checkpoint.resolve()), "parent_checkpoint_sha256": trainer.digest(checkpoint/"checkpoint.json"),
        "parent_model_sha256": parent["model_sha256"], "parent_state_sha256": parent["state_sha256"],
        "out": str(out.resolve()), "panel": str(panel_path.resolve()), "panel_sha256": trainer.digest(panel_path),
        "references": str(references.resolve()), "reference_bindings": reference_bindings,
        "task": parent["task"], "task_fingerprint": parent["task_fingerprint"], "start_updates": START,
        "target_updates": TOTAL, "checkpoint_updates": list(ENDPOINTS), "max_wall_seconds": MAX_WALL_SECONDS,
        "stages": [{"absolute_steps_inclusive": [50000, 54999], "learning_rate": RATES[0]},
                   {"absolute_steps_inclusive": [55000, 59999], "learning_rate": RATES[1]}],
        "seed": 9011, "train_rows": 20000, "batch_size": 128, "fixed_c": 1., "dropout": 0.,
        "retained": "all Adam moments/iteration, model/standardizer/metrics/RNG arrays; original absolute row/noise schedule",
        "only_changed": "optimizer learning-rate schedule", "fit_attempts_max": 1, "automatic_extension": False,
        "evaluation": {"seed": 261112, "conditional_draws": 4096, "validation_rows": 512, "validation_draws": 512,
                       "replicates": 400, "boundary": "inconclusive; reference exhausted its declared budget", "market": "separate application diagnostic"},
        "source_sha256": {name: {"path": str(path.resolve()), "sha256": trainer.digest(path)} for name, path in source_paths().items()},
        "protected_input_sha256": {str(path.resolve()): trainer.digest(path) for path in protected},
        "reserved_arrays_opened": False, "price_calls": 0, "scientific_acceptance": False,
        "scope": "One bounded optimization comparison prompted by local update-scale evidence. No data/capacity/noise change or causal/stationary-jitter claim."}
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_json(protocol_path, protocol)
    return protocol


def verify_protocol(protocol, checkpoint, out):
    if (protocol["format"] != "heston_spread_annealing_protocol_v1"
            or Path(checkpoint).resolve() != Path(protocol["parent_checkpoint"])
            or Path(out).resolve() != Path(protocol["out"])
            or protocol["checkpoint_updates"] != list(ENDPOINTS)
            or protocol["stages"] != [{"absolute_steps_inclusive": [50000, 54999], "learning_rate": RATES[0]},
                                      {"absolute_steps_inclusive": [55000, 59999], "learning_rate": RATES[1]}]):
        raise ValueError("annealing protocol target/schedule/path differs")
    if any(trainer.digest(path) != value for path, value in protocol["protected_input_sha256"].items()):
        raise ValueError("a predeclared input/source hash differs")


def run(checkpoint, out, protocol_path):
    started = time.monotonic()
    checkpoint, out, protocol_path = map(Path, (checkpoint, out, protocol_path))
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, checkpoint, out)
    if out.exists():
        raise FileExistsError(out)
    parent = json.loads((checkpoint/"checkpoint.json").read_text())
    validate_parent(parent)
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import bayesflow
    import jax
    import keras
    keras.utils.set_random_seed(9011)
    workflow, loaded = trainer.load_checkpoint(checkpoint)
    if loaded != parent:
        raise ValueError("parent record changed during loading")
    model = workflow.estimator
    if (bool(jax.config.x64_enabled) != parent["numeric_configuration"]["jax_enable_x64"]
            or keras.backend.floatx() != parent["numeric_configuration"]["keras_floatx"]
            or any(metadata.version(name) != version for name, version in parent["versions"].items() if name != "python")
            or any(trainer.digest(Path(__file__).resolve().parents[1]/name) != value for name, value in parent["environment_sha256"].items())):
        raise ValueError("parent numerical environment differs")
    state, restoration = continuation.restore_full_state(workflow, parent, checkpoint/"state.npz")
    schedule_record = install_schedule(workflow, parent)
    data, bank = trainer.load_clean_bank(parent["bank_path"], parent["task"], parent["train_rows"])
    continuation.validate_bank_binding(parent, bank)
    task = parent["task"]
    probes = {name: trainer.fixed_probe(workflow, task, data[name], 512, 9011, namespace)
              for name, namespace in (("train", 1), ("validation", 2))}
    def probe_values():
        trainer.assign_state(model, state)
        return {name+"_probe_loss": float(np.asarray(model.compute_metrics(**batch, stage="validation")["loss"]))
                for name, (_, batch) in probes.items()}
    initial = probe_values()
    if any(initial[key] != parent["probe_metrics"][key] for key in initial):
        raise ValueError("annealing initial fixed probes differ from parent")
    if hashes(model) != {group: [row["sha256"] for row in rows] for group, rows in parent["state"].items()}:
        raise RuntimeError("initial probe/schedule assignment changed saved arrays")
    out.mkdir(parents=True)
    trainer.write_json(out/"initial-probes.json", initial)
    resume = {"protocol_path": str(protocol_path.resolve()), "protocol_sha256": trainer.digest(protocol_path),
        "restoration": restoration, "learning_rate_assignment": schedule_record, "absolute_data_and_noise_steps": True}
    trainer.write_json(out/"resume-protocol.json", resume)
    run_record = {**parent, "format": "heston_spread_baseline_run_v1", "planned_updates": TOTAL,
        "checkpoint_updates": list(ENDPOINTS), "annealing": resume, "optimizer_schedule": schedule_record,
        "parent_actual_optimizer_updates": START, "scientific_acceptance": False, "resume_supported": False}
    run_record["source_sha256"] = {**parent["source_sha256"], "annealing": trainer.digest(__file__)}
    trainer.write_json(out/"run.json", run_record)
    plan = trainer.BatchPlan(20000, 128, TOTAL, 9011)
    visits = np.zeros(20000, dtype=np.int64)
    step_fn = trainer.compiled_step(model)
    curve, last_checkpoint, proofs = [], None, {}
    step, phase, fields = START, "before_update", []
    candidate, gradients, metrics, indices, quotes, raw, adapted = None, None, {}, [], [], {}, {}
    def deadline():
        if time.monotonic()-started >= MAX_WALL_SECONDS:
            raise TimeoutError("annealing total wall budget exhausted")
    try:
        for step in range(START, TOTAL):
            deadline()
            phase, fields = "before_update", []
            candidate, gradients, metrics, indices, quotes, raw, adapted = None, None, {}, [], [], {}, {}
            indices, quotes, raw, adapted = trainer.adapted_batch(workflow, task, data["train"], plan, step, 9011)
            fields = ["adapted_input:"+key for key, value in adapted.items() if not np.isfinite(value).all()]
            if fields:
                raise FloatingPointError("nonfinite annealing adapted input")
            if step in (START, 55000):
                trainer.write_json(out/f"batch-{step:06d}.json", {"absolute_step": step,
                    "indices_sha256": trainer.array_summary(indices)["sha256"], "quotes_sha256": trainer.array_summary(quotes)["sha256"]})
            phase = "optimizer_update"
            metrics, candidate, gradients, flags = step_fn(state, adapted)
            fields = [key for key, value in jax.device_get(flags).items() if not bool(value)]
            if fields:
                raise FloatingPointError("nonfinite annealing update: "+", ".join(fields))
            if int(np.asarray(candidate[2][0])) != step+1:
                raise RuntimeError("annealing Adam iteration differs from absolute step")
            state = candidate
            visits[indices] += 1
            if (step+1) % 1000 == 0:
                phase = "fixed_probe"
                point = {"updates": step+1, "seconds": time.monotonic()-started,
                    "batch_training_loss": float(np.asarray(metrics["loss"])), **probe_values()}
                if not np.isfinite(list(point.values())).all():
                    metrics, candidate, gradients = point, None, None
                    raise FloatingPointError("nonfinite annealing fixed probe")
                curve.append(point)
                trainer.write_json(out/"learning-curve.json", curve)
                print(json.dumps(point), flush=True)
            if step+1 in ENDPOINTS:
                phase = "checkpoint_and_reload"
                destination = out/"checkpoints"/f"update-{step+1:06d}"
                trainer.checkpoint(destination, workflow, state, run_record, step+1, curve[-1])
                before = trainer.sample(workflow, task, probes["validation"][0][:2], 16, seed=9022)
                restored, saved = trainer.load_checkpoint(destination)
                _, proof = continuation.restore_full_state(restored, saved, destination/"state.npz")
                after = trainer.sample(restored, task, probes["validation"][0][:2], 16, seed=9022)
                if (not np.array_equal(before, after) or not proof["all_array_hashes_exact"]
                        or restored.estimator.optimizer.get_config()["learning_rate"] != schedule_record["serialized"]):
                    raise RuntimeError("annealed endpoint state/schedule/seeded inference differs on reload")
                proofs[str(step+1)] = {**proof, "seeded_draws_exact": True, "schedule_exact": True}
                last_checkpoint = str(destination.resolve())
            trainer.write_json(out/"progress.json", {"status": "training", "completed_updates": step+1,
                "seconds": time.monotonic()-started, "last_checkpoint": last_checkpoint})
        deadline()
        trainer.write_npz(out/"exposure.npz", {"row_id": data["train"]["row_id"], "annealing_presentations": visits,
            "presentations_before_annealing": continuation.visit_counts(plan, 0, START)})
        phase = "evaluation"
        from scripts.evaluate_heston_spread_baseline import evaluate
        evaluation_paths = {}
        for endpoint in ENDPOINTS:
            deadline()
            result = evaluate(out/"checkpoints"/f"update-{endpoint:06d}", protocol["panel"], protocol["references"],
                parent["bank_path"], **{key: value for key, value in protocol["evaluation"].items()
                                      if key in ("seed", "conditional_draws", "validation_rows", "validation_draws", "replicates")})
            target = out/f"evaluation-{endpoint:06d}.json"
            trainer.write_json(target, result)
            evaluation_paths[str(endpoint)] = {"path": str(target.resolve()), "sha256": trainer.digest(target)}
        deadline()
        verify_protocol(protocol, checkpoint, out)
        result = {"status": "complete", "completed_updates": TOTAL, "continued_updates": TOTAL-START,
            "seconds": time.monotonic()-started, "checkpoint_reload_proofs": proofs, "evaluations": evaluation_paths,
            "annealing_row_presentations": int(visits.sum()), "all_parent_array_hashes_restored": True,
            "protected_inputs_unchanged": True, "reserved_arrays_opened": False, "fit_attempts": 1,
            "automatic_extension": False, "scientific_acceptance": False}
        trainer.write_json(out/"training.json", result)
        trainer.write_json(out/"progress.json", result)
        evidence = protocol_path.parent/"annealing"
        evidence.mkdir(exist_ok=False)
        for name in ("training.json", "learning-curve.json", "initial-probes.json", "resume-protocol.json", "run.json", "evaluation-055000.json", "evaluation-060000.json"):
            (evidence/name).write_bytes((out/name).read_bytes())
        return result
    except BaseException as error:
        if phase != "optimizer_update":
            candidate, gradients = None, None
        if phase == "fixed_probe":
            adapted = {**adapted, **{f"probe_{name}_{key}": value for name, (_, batch) in probes.items() for key, value in batch.items()}}
        failure = trainer.capture_failure(out, model, state, candidate, gradients, step=step, indices=indices, quotes=quotes,
            raw=raw, adapted=adapted, fields=fields, metrics=metrics, error=error, last_checkpoint=last_checkpoint, phase=phase)
        trainer.write_json(out/"progress.json", {"status": "failed", "completed_updates": failure["completed_finite_updates"],
            "failure": "failure/failure.json", "seconds": time.monotonic()-started, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_only:
        if not args.panel or not args.references:
            parser.error("preparation requires --panel and --references")
        prepare(args.checkpoint, args.out, args.panel, args.references, args.protocol)
        print(json.dumps({"status": "predeclared", "protocol": str(args.protocol), "sha256": trainer.digest(args.protocol)}))
    elif args.worker:
        print(json.dumps(run(args.checkpoint, args.out, args.protocol)), flush=True)
    else:
        if args.out.exists():
            raise FileExistsError(args.out)
        env = dict(os.environ, KERAS_BACKEND="jax", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], env=env)
        try:
            code = process.wait(timeout=MAX_WALL_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            args.out.mkdir(parents=True, exist_ok=True)
            trainer.write_json(args.out/"hard-timeout.json", {"status": "killed_at_wall_limit", "wall_seconds": MAX_WALL_SECONDS,
                "scientific_acceptance": False, "automatic_retry": False})
            raise TimeoutError("annealing supervisor wall limit exhausted")
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()

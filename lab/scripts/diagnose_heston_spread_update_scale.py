"""Eight independent, discarded Adam updates from the saved zero-dropout 50k state.

This is a local sensitivity diagnostic, not fitting, a continuation or a test of
stationary optimizer variability. No checkpoint or bank member is written.
"""
import argparse
import inspect
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
from scripts.evaluate_heston_spread_baseline import check_reference_binding
from scripts.run_heston_spread_reference import load_case


STEPS = tuple(range(50000, 50008))
CASES = ("synthetic-broad", "synthetic-market-like")
DRAWS = 4096
LATENT_SEED = 261120
DELTA_SCALE = .1
MAX_WALL_SECONDS = 300


def counterfactual_states(before, candidate, scale=DELTA_SCALE):
    """Keep the native candidate's other state; change only learned weights."""
    if len(before) != 4 or len(candidate) != 4 or len(before[0]) != len(candidate[0]):
        raise ValueError("four matching native model-state groups required")
    if not 0 < scale < 1:
        raise ValueError("counterfactual scale must be strictly between zero and one")
    scaled = [old + scale * (new-old) for old, new in zip(before[0], candidate[0], strict=True)]
    return {"native": candidate, "scaled_weight_delta": (scaled, *candidate[1:]),
            "candidate_other_state_only": (before[0], *candidate[1:])}


def shift_summary(shifts):
    values = np.asarray(shifts, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite batch-by-parameter standardized shifts required")
    return {"rms": np.sqrt(np.mean(values**2, axis=0)).tolist(),
            "max_abs": np.max(np.abs(values), axis=0).tolist(),
            "signed_mean": values.mean(axis=0).tolist()}


def state_hashes(model, state):
    return {group: [row["sha256"] for row in rows]
            for group, rows in trainer.state_summary(model, state).items()}


def optimizer_check(model, record):
    config = model.optimizer.get_config()
    if (type(model.optimizer).__name__ != "Adam" or config.get("weight_decay") is not None
            or config.get("use_ema") is not False or config.get("gradient_accumulation_steps") is not None
            or config["learning_rate"] != record["optimizer_schedule"]["serialized"]
            or any(getattr(variable, "constraint", None) is not None for variable in model.trainable_variables)):
        raise ValueError("one-step LR interpretation requires the unchanged unconstrained Adam recipe")
    rate = float(model.optimizer.learning_rate)
    if not np.isclose(rate, 5e-5, rtol=1e-7, atol=0):
        raise ValueError("saved optimizer is not at its original 5e-5 floor")
    return {"config": config, "learning_rate": rate, "scaled_delta_equivalent_rate": rate*DELTA_SCALE,
            "interpretation": "One-step weight-delta scaling at identical prestate/gradient/Adam slots; candidate optimizer and other arrays are retained. Float32 interpolation need not be bit-identical to rerunning Adam at the lower LR. No future trajectory equivalence is claimed."}


def run(checkpoint, panel_path, references, out):
    started = time.monotonic()
    checkpoint, panel_path, references, out = map(Path, (checkpoint, panel_path, references, out))
    if out.exists() or out.with_suffix(".protocol.json").exists():
        raise FileExistsError("diagnostic outputs are immutable")
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import bayesflow
    import jax
    import keras
    keras.utils.set_random_seed(9011)
    workflow, record = trainer.load_checkpoint(checkpoint)
    if (record["actual_optimizer_updates"] != 50000 or not record.get("dropout_zero_control")
            or record["schedule_updates"] != 10000 or record["seed"] != 9011
            or record["architecture"]["subnet_dropout"] != 0.
            or record["source_sha256"]["trainer"] != continuation.FROZEN_TRAINER_SHA256
            or trainer.digest(trainer.__file__) != continuation.FROZEN_TRAINER_SHA256):
        raise ValueError("diagnostic requires the frozen 50k zero-dropout checkpoint")
    state, restoration = continuation.restore_full_state(workflow, record, checkpoint/"state.npz")
    model, task = workflow.estimator, record["task"]
    base_hashes = state_hashes(model, state)
    if int(model.optimizer.iterations.numpy()) != 50000:
        raise ValueError("saved optimizer iteration differs")
    optimizer = optimizer_check(model, record)
    if any(float(layer.rate) != 0. for layer in trainer.dropout_layers(model)):
        raise ValueError("diagnostic does not support active dropout")
    if (any(trainer.digest(Path(__file__).resolve().parents[1]/name) != value
            for name, value in record["environment_sha256"].items())
            or bool(jax.config.x64_enabled) != record["numeric_configuration"]["jax_enable_x64"]
            or keras.backend.floatx() != record["numeric_configuration"]["keras_floatx"]):
        raise ValueError("checkpoint numerical environment differs")
    data, bank = trainer.load_clean_bank(record["bank_path"], task, record["train_rows"])
    continuation.validate_bank_binding(record, bank)
    names, observations, reference_info = record["names"], [], {}
    protected = [checkpoint/name for name in ("checkpoint.json", "model.keras", "state.npz")]
    protected += [panel_path, Path(record["design_path"]), Path(record["bank_path"])/"bank.json"]
    bank_record = json.loads((Path(record["bank_path"])/"bank.json").read_text())
    protected += [Path(record["bank_path"])/bank_record["payload"]]
    for case_id in CASES:
        panel, selected, _, y = load_case(panel_path, case_id)
        if panel["fixed_task"] != task["target"] or panel["design_sha256"] != task["design_sha256"]:
            raise ValueError("panel target differs from checkpoint")
        directory = references/case_id
        run_path, diagnostic_path = directory/"run.json", directory/"diagnostics.json"
        run_record, diagnostic = json.loads(run_path.read_text()), json.loads(diagnostic_path.read_text())
        if run_record.get("status") != "completed" or diagnostic.get("reference_ready") is not True:
            raise ValueError("a ready matched reference is required")
        chain_path = Path(run_record["chain_payload_path"])
        check_reference_binding(run_record, diagnostic, chain_path, target=task["target"],
            design_sha256=task["design_sha256"], case_id=case_id, observations=y,
            panel_sha256=trainer.digest(panel_path), products=panel["products"], diagnostics_path=diagnostic_path)
        sd = np.array([diagnostic["parameters"][f"parameters[{name}]"]["sd"] for name in names])
        if not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError("finite positive reference SD required")
        reference_info[case_id] = {"sd": sd.tolist(), "mean": [diagnostic["parameters"][f"parameters[{name}]"]["mean"] for name in names],
            "observation_sha256": selected["observations_sha256"], "physical_truth": selected["physical_truth"],
            "reference_run_path": str(run_path.resolve()), "reference_diagnostics_path": str(diagnostic_path.resolve()),
            "chain_payload_read": "hash only; no array decompression"}
        observations.append(y)
        protected += [run_path, diagnostic_path, chain_path]
    protected += [Path(panel["declaration_path"])]
    sources = {"diagnostic": Path(__file__), "trainer": Path(trainer.__file__),
        "continuation": Path(continuation.__file__), "case_helper": Path(inspect.getfile(load_case)),
        "reference_binding": Path(inspect.getfile(check_reference_binding)),
        "native_step": Path(inspect.getfile(type(model).stateless_train_step)),
        "keras_adam": Path(inspect.getfile(type(model.optimizer)))}
    protected += list(sources.values())
    protected_hashes = {str(path.resolve()): trainer.digest(path) for path in protected}
    plan = trainer.BatchPlan(record["train_rows"], record["batch_size"], STEPS[-1]+1, record["seed"])
    batches = []
    for step in STEPS:
        indices = plan.indices(step)
        quotes = trainer.noisy_quotes(data["train"]["clean"][indices], task["records"]["half_spread"], step, record["seed"])
        batches.append({"absolute_step": step, "rows": len(indices), "indices_sha256": trainer.array_summary(indices)["sha256"],
                        "quotes_sha256": trainer.array_summary(quotes)["sha256"]})
    protocol = {"format": "heston_spread_update_scale_protocol_v1", "task": task, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_updates": 50000, "steps": list(STEPS), "case_ids": list(CASES), "posterior_draws": DRAWS,
        "common_latent_seed": LATENT_SEED, "weight_delta_scale": DELTA_SCALE, "optimizer": optimizer,
        "independent_from_same_saved_prestate": True, "batches": batches, "references": reference_info,
        "source_sha256": {key: {"path": str(path.resolve()), "sha256": trainer.digest(path)} for key, path in sources.items()},
        "protected_input_sha256": protected_hashes, "restoration": restoration,
        "reserved_arrays_opened": False, "price_calls": 0, "fits": 0, "committed_updates": 0,
        "scope": "Eight local, discarded next-step perturbations; no stationary jitter, causal, data-sufficiency or acceptance claim."}
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_json(out.with_suffix(".protocol.json"), protocol)
    quotes = np.array(observations)

    def posterior_means(trial_state):
        trainer.assign_state(model, trial_state)
        before = state_hashes(model, trainer.current_state(model))
        draws = trainer.sample(workflow, task, quotes, DRAWS, seed=LATENT_SEED)
        if before != state_hashes(model, trainer.current_state(model)):
            raise RuntimeError("explicit-seed inference mutated model state")
        return draws.astype(np.float64).mean(axis=1)

    base_means = posterior_means(state)
    sd = np.array([reference_info[case_id]["sd"] for case_id in CASES])
    step_fn, trials = trainer.compiled_step(model), []
    try:
        for declaration, step in zip(batches, STEPS, strict=True):
            trainer.assign_state(model, state)
            indices, batch_quotes, raw, adapted = trainer.adapted_batch(workflow, task, data["train"], plan, step, record["seed"])
            if (trainer.array_summary(indices)["sha256"] != declaration["indices_sha256"]
                    or trainer.array_summary(batch_quotes)["sha256"] != declaration["quotes_sha256"]):
                raise ValueError("batch differs from declaration")
            metrics, candidate, gradients, flags = step_fn(state, adapted)
            if not all(bool(value) for value in jax.device_get(flags).values()):
                raise FloatingPointError("nonfinite discarded native update")
            if int(np.asarray(candidate[2][0])) != 50001:
                raise ValueError("a trial did not start from the saved Adam iteration")
            candidates = counterfactual_states(state, candidate)
            means = {variant: posterior_means(value) for variant, value in candidates.items()}
            trial = {**declaration, "native_candidate_optimizer_iteration": 50001,
                "native_batch_training_loss": float(np.asarray(metrics["loss"])),
                "mean_shifts_in_reference_sd": {variant: ((value-base_means)/sd).tolist() for variant, value in means.items()},
                "posterior_means": {variant: value.tolist() for variant, value in means.items()}}
            trials.append(trial)
            trainer.assign_state(model, state)
            if state_hashes(model, trainer.current_state(model)) != base_hashes:
                raise RuntimeError("saved prestate not exactly restored after discarded trial")
        summaries = {}
        for index, case_id in enumerate(CASES):
            by_variant = {variant: np.array([trial["mean_shifts_in_reference_sd"][variant][index] for trial in trials])
                          for variant in candidates}
            summary = {variant: shift_summary(values) for variant, values in by_variant.items()}
            native_rms = np.array(summary["native"]["rms"])
            for variant in ("scaled_weight_delta", "candidate_other_state_only"):
                summary[variant]["rms_ratio_to_native"] = np.divide(summary[variant]["rms"], native_rms,
                    out=np.zeros_like(native_rms), where=native_rms != 0).tolist()
            # The difference isolates the learned-weight contribution at updated normalization.
            summary["native_minus_other_state_only"] = shift_summary(by_variant["native"]-by_variant["candidate_other_state_only"])
            summaries[case_id] = {"parameter_names": names, "base_posterior_mean": base_means[index].tolist(), **summary}
        if any(trainer.digest(path) != value for path, value in protected_hashes.items()):
            raise RuntimeError("a protected input/source file changed during diagnostic")
        result = {**protocol, "format": "heston_spread_update_scale_diagnostic_v1", "protocol_sha256": trainer.digest(out.with_suffix(".protocol.json")),
            "status": "completed", "trials": trials, "summary": summaries, "seconds": time.monotonic()-started,
            "all_saved_arrays_restored": state_hashes(model, trainer.current_state(model)) == base_hashes,
            "protected_files_unchanged": True, "sampling_state_unchanged": True,
            "candidate_other_state_scope": "Native nontrainable, optimizer and metrics state with old trainables. Inference does not use optimizer/metrics; dropout is zero, so this measures normalization effects and any remaining inference-relevant nontrainable change."}
        trainer.write_json(out, result)
        print(json.dumps({"status": "completed", "out": str(out), "summary": summaries, "seconds": result["seconds"]}), flush=True)
        return result
    finally:
        trainer.assign_state(model, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        env = dict(os.environ, KERAS_BACKEND="jax", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        try:
            subprocess.run(command, env=env, check=True, timeout=MAX_WALL_SECONDS)
        except subprocess.TimeoutExpired:
            raise SystemExit("discarded-update diagnostic exceeded its 300-second wall limit")
        return
    run(args.checkpoint, args.panel, args.references, args.out)


if __name__ == "__main__":
    main()

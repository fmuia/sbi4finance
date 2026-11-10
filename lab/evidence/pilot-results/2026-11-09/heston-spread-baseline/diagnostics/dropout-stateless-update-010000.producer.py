"""Read-only 2x2 stateless normalization/dropout probe of one frozen checkpoint."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.lab), str(args.lab / "src"), str(args.lab.parent / "tutorial/src")]
    from scripts import train_heston_spread_baseline as trainer
    import jax
    import keras
    import bayesflow as bf
    started = time.monotonic()
    if args.out.exists():
        raise FileExistsError(args.out)
    workflow, record = trainer.load_checkpoint(args.checkpoint)
    task, model = record["task"], workflow.estimator
    if record["actual_optimizer_updates"] != 10000 or task["target"]["infer_c"] or task["target"]["fixed_c"] != 1.:
        raise ValueError("diagnostic is predeclared for the fixed-c1 10k checkpoint")
    source = Path(trainer.__file__)
    hashes = {"diagnostic_source": trainer.digest(__file__), "trainer_source": trainer.digest(source),
              "checkpoint_record": trainer.digest(args.checkpoint / "checkpoint.json"),
              "checkpoint_model": trainer.digest(args.checkpoint / "model.keras"),
              "checkpoint_state": trainer.digest(args.checkpoint / "state.npz"),
              "provisional_evaluation": trainer.digest(args.evaluation)}
    if hashes["trainer_source"] != record["source_sha256"]["trainer"]:
        raise ValueError("frozen producer source differs")
    bank = Path(record["bank_path"])
    bank_record = json.loads((bank / "bank.json").read_text())
    if (trainer.digest(bank / "bank.json") != record["bank_record_sha256"]
            or trainer.digest(bank / bank_record["payload"]) != record["bank_payload_sha256"]):
        raise ValueError("historical bank identity changed")
    with np.load(bank / bank_record["payload"], allow_pickle=False) as archive:
        clean = archive["validation_clean"][:512].copy()
        physical = archive["validation_parameters"][:512].copy()
    quotes = trainer.noisy_quotes(clean, task["records"]["half_spread"], 0, record["seed"], namespace=2)
    data = workflow.adapter(trainer.training_data(task, quotes, physical))
    base = trainer.current_state(model)
    before = trainer.state_summary(model, base)
    dropout = []
    variables = model.non_trainable_variables
    for layer in model._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, keras.layers.Dropout) and layer.rate > 0:
            index = next(i for i, variable in enumerate(variables) if variable is layer.seed_generator.state)
            dropout.append({"index": index, "rate": float(layer.rate), "layer": layer.name})
    if len({row["index"] for row in dropout}) != 24 or any(row["rate"] != .05 for row in dropout):
        raise ValueError("expected the frozen architecture's 24 active dropout layers at rate .05")
    normalizer = model.standardizer.standardize_layers["inference_conditions"]
    stored_std = np.asarray(normalizer.moving_std(0))
    stored_mean = np.asarray(normalizer.moving_mean[0])
    stored_count = float(np.asarray(normalizer.count[0]))
    noise_sd = 1. / stored_std

    def rekey(rep):
        state = tuple(list(group) for group in base)
        for item in dropout:
            index = item["index"]
            value = np.array(state[1][index], copy=True)
            value[0] = np.random.SeedSequence([0x44525044, 261113, rep, index]).generate_state(1)[0]
            state[1][index] = jax.numpy.asarray(value)
        return state

    conditions, stats = {}, {}
    for norm_stage, dropout_stage, label in (
        ("validation", "validation", "saved_norm_dropout_off"),
        ("training", "training", "updated_norm_dropout_on"),
        ("validation", "training", "saved_norm_dropout_on"),
        ("training", "validation", "updated_norm_dropout_off"),
    ):
        original = model.standardizer.maybe_standardize
        def normalize(*values, **kwargs):
            kwargs["stage"] = norm_stage
            return original(*values, **kwargs)
        with patch.object(model.standardizer, "maybe_standardize", new=normalize):
            function = jax.jit(lambda state: model.stateless_compute_metrics(
                state[0], state[1], state[3], data=data, stage=dropout_stage))
            first = function(rekey(0))
            jax.block_until_ready(first)
            losses = [float(np.asarray(first[0]))]
            for rep in range(1, 32):
                losses.append(float(np.asarray(function(rekey(rep))[0])))
        if not np.isfinite(losses).all():
            raise FloatingPointError("nonfinite diagnostic loss")
        conditions[label] = {"losses": losses, "mean": float(np.mean(losses)),
            "sd_across_dropout_streams": float(np.std(losses, ddof=1)),
            "mcse_of_stream_average": float(np.std(losses, ddof=1)/np.sqrt(len(losses))),
            "minimum": min(losses), "maximum": max(losses)}
        with keras.StatelessScope(list(zip(model.non_trainable_variables, first[1][1], strict=True))):
            changed_std = np.asarray(normalizer.moving_std(0))
            changed_mean = np.asarray(normalizer.moving_mean[0])
            changed_count = float(np.asarray(normalizer.count[0]))
        stats[label] = {"returned_count": changed_count,
            "max_relative_input_std_change": float(np.max(np.abs(changed_std/stored_std-1))),
            "max_input_mean_change_in_saved_sd": float(np.max(np.abs((changed_mean-stored_mean)/stored_std)))}
    # Verify that the override reproduces both unmodified native stage calls.
    native = {}
    for stage, label in (("validation", "saved_norm_dropout_off"), ("training", "updated_norm_dropout_on")):
        value = model.stateless_compute_metrics(base[0], rekey(0)[1], base[3], data=data, stage=stage)[0]
        native[stage] = float(np.asarray(value))
        if abs(native[stage]-conditions[label]["losses"][0]) > 2e-4:
            raise ValueError("factorization does not reproduce native stage semantics")
    deterministic = conditions["saved_norm_dropout_off"]["mean"]
    expected = record["probe_metrics"]["validation_probe_loss"]
    if abs(deterministic-expected) > 2e-4:
        raise ValueError("validation data do not reproduce the saved fixed probe")
    differences = {}
    for label, left, right in (
        ("dropout_effect_saved_norm", "saved_norm_dropout_on", "saved_norm_dropout_off"),
        ("dropout_effect_updated_norm", "updated_norm_dropout_on", "updated_norm_dropout_off"),
        ("normalization_effect_dropout_off", "updated_norm_dropout_off", "saved_norm_dropout_off"),
        ("normalization_effect_dropout_on", "updated_norm_dropout_on", "saved_norm_dropout_on"),
    ):
        values = np.asarray(conditions[left]["losses"])-conditions[right]["losses"]
        differences[label] = {"mean_nats_per_case": float(values.mean()),
            "sd_across_paired_streams": float(values.std(ddof=1)),
            "mcse": float(values.std(ddof=1)/np.sqrt(len(values)))}
    after = trainer.state_summary(model, trainer.current_state(model))
    if before != after:
        raise ValueError("diagnostic changed live model/optimizer/standardizer/RNG state")
    if hashes["trainer_source"] != trainer.digest(source) or hashes["checkpoint_record"] != trainer.digest(args.checkpoint / "checkpoint.json"):
        raise ValueError("frozen inputs changed during diagnostic")
    evaluation = json.loads(args.evaluation.read_text())
    result = {"format": "heston_spread_stateless_dropout_diagnostic_v1", "scientific_acceptance": False,
        "scope": "read-only pointwise objective diagnostic, no training, gradients, posterior correction or reserved test reads",
        "hashes": hashes, "checkpoint": str(args.checkpoint.resolve()), "task_fingerprint": record["task_fingerprint"],
        "actual_optimizer_updates": 10000, "validation_rows": len(clean),
        "probe_protocol": "exact producer validation probe: first512 validation rows; noise namespace2 step0 seed9011",
        "diagnostic_replicates": 32, "dropout_rekey_protocol": "PCG64-independent SeedSequence([0x44525044,261113,rep,variable_index]); replace active dropout seed only in stateless argument copies",
        "dropout_layers": len(dropout), "dropout_rate": .05,
        "dropout_multiplier_sd_relative_to_unmasked_activation": float(np.sqrt(.05/.95)),
        "conditions": conditions, "paired_differences": differences, "normalization_returned_state": stats,
        "native_stage_checks": native, "saved_validation_probe_loss": expected,
        "saved_input_normalization": {"count": stored_count,
            "across_prior_input_std_quantiles_0_25_50_75_100": np.quantile(stored_std, [0,.25,.5,.75,1]).tolist(),
            "measurement_noise_sd_in_standardized_input_quantiles": np.quantile(noise_sd, [0,.25,.5,.75,1]).tolist(),
            "per_quote_measurement_noise_sd_in_standardized_input": noise_sd.tolist(),
            "units": "input is Y/h; measurement noise SD1 before global standardization; standardized SD1/input_std"},
        "model_optimizer_standardizer_rng_unchanged": True,
        "state_integrity": {group: {"variable_count": len(rows),
            "before_sha256": trainer.fingerprint(rows), "after_sha256": trainer.fingerprint(after[group]),
            "all_arrays_finite": all(row["all_finite"] for row in rows), "unchanged": rows == after[group]}
            for group, rows in before.items()},
        "reserved_arrays_opened": False, "simulation_calls": 0,
        "limitations": ["Dropout probes use the existing learned weights; they cannot establish which training intervention improves posterior accuracy.",
            "Training stage updates cumulative standardization and activates hidden dropout; factorial calls separate these effects without committing returned state.",
            "The .229 hidden dropout multiplier SD is in hidden-activation units, not quote-price units, and cannot be treated as added market measurement noise.",
            "The native .keras archive does not preserve every SeedGenerator; explicit independent diagnostic streams are supplied, with no exact training-resume claim.",
            "Losses are BF transformed-and-standardized-target NLLs; they are not physical-coordinate NLLs or model-fit residuals.",
            "Broad synthetic conditional-reference failure already exists; actual-market failure alone cannot identify an algorithmic cause."],
        "conditional_context": {name: case["status"] for name, case in evaluation["cases"].items()},
        "seconds": time.monotonic()-started,
        "versions": {name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_json(args.out, result)
    print(json.dumps({"out": str(args.out), "conditions": {name: {key:value for key,value in row.items() if key != "losses"} for name,row in conditions.items()},
                      "paired_differences": differences, "normalization": stats, "input_noise_sd_quantiles": result["saved_input_normalization"]["measurement_noise_sd_in_standardized_input_quantiles"],
                      "unchanged": True, "seconds": result["seconds"]}, indent=2))


if __name__ == "__main__":
    main()

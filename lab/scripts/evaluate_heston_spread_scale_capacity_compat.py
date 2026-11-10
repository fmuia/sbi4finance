"""Restore CLI-authored schedule metadata before the frozen read-only evaluator.

Keras serializes a schedule class created in __main__ with module=null. The
training worker reloads successfully in that same context. Importing the exact
class under scripts.* changes only this serialization field. Restore its
original module attribution, then retain every original loader/evaluator guard.
No trainer, checkpoint, model, schedule formula or evaluator source is edited.
"""
import argparse
import hashlib
import json
import marshal
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import continue_heston_spread_scale as continuation
from scripts import evaluate_heston_spread_scale_continuation as evaluator

fixed = continuation.fixed


def read(path):
    return json.loads(Path(path).read_text())


def code_hashes(cls):
    return {name: hashlib.sha256(marshal.dumps(getattr(cls, name).__code__)).hexdigest()
        for name in ("__init__", "__call__", "get_config")}


def archive_schedule(path):
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    found = []
    def visit(value):
        if isinstance(value, dict):
            if value.get("class_name") == "SpreadExposureCosine":
                found.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(config)
    if len(found) != 1:
        raise ValueError("native archive must contain exactly one exposure schedule")
    return found[0]


def restore_cli_module(cls, expected):
    """Only the serialization module may differ; executable methods stay exact."""
    import keras
    original_module = cls.__module__
    instance = cls(**expected["config"])
    before = keras.optimizers.schedules.serialize(instance)
    if (expected.get("module", "missing") is not None
            or expected.get("registered_name") != "sbi4finance>SpreadExposureCosine"
            or expected.get("class_name") != "SpreadExposureCosine"
            or {k: v for k, v in before.items() if k != "module"}
                != {k: v for k, v in expected.items() if k != "module"}
            or original_module != "scripts.continue_heston_spread_scale"):
        raise ValueError("compatibility is restricted to the original CLI module attribution")
    executable_before = code_hashes(cls)
    config = expected["config"]
    batches = (config["train_rows"]+config["batch_size"]-1)//config["batch_size"]
    start = config["start_updates"]
    steps = sorted(set([0, start, start+batches-1, start+batches,
        *[start+passes*batches for passes in (25, 100, 250, 399, 400, 401, 500)]]))
    rates_before = np.array([float(instance(step)) for step in steps])
    cls.__module__ = "__main__"
    after = keras.optimizers.schedules.serialize(instance)
    rates_after = np.array([float(instance(step)) for step in steps])
    if (after != expected or code_hashes(cls) != executable_before
            or not np.array_equal(rates_before, rates_after)
            or keras.saving.get_registered_object("sbi4finance>SpreadExposureCosine") is not cls):
        raise RuntimeError("module attribution changed executable code, rates or registration")
    expected_rates = []
    for step in steps:
        relative = max(step-start, 0)
        examples = (relative//batches)*config["train_rows"]+min((relative%batches)*config["batch_size"], config["train_rows"])
        fraction = min(examples/(config["decay_passes"]*config["train_rows"]), 1.)
        expected_rates.append(config["minimum"]+(config["maximum"]-config["minimum"])*.5*(1+np.cos(np.pi*fraction)))
    if not np.allclose(rates_after, expected_rates, rtol=2e-7, atol=1e-11):
        raise ValueError("restored schedule rates disagree with the declared exposure cosine")
    return {"original_module": original_module, "restored_module": cls.__module__,
        "serialization_before": before, "serialization_after": after,
        "executable_method_sha256": executable_before, "same_registered_class_object": True,
        "numeric_rates_bitwise_unchanged": True,
        "rates": [{"absolute_update": step, "actual": float(rate), "independent_formula": float(expected_rate)}
            for step, rate, expected_rate in zip(steps, rates_after, expected_rates, strict=True)]}


def validate_addendum(addendum, study_path):
    if (addendum.get("format") != "heston_capacity_evaluation_cli_compatibility_v1"
            or addendum.get("original_study") != {"path": str(Path(study_path).resolve()), "sha256": fixed.digest(study_path)}
            or addendum.get("bootstrap_source") != {"path": str(Path(__file__).resolve()), "sha256": fixed.digest(__file__)}
            or addendum.get("scope") != "startup-only read compatibility; unchanged target, training, samples, seeds and frozen evaluator"
            or addendum.get("automatic_retry") is not False):
        raise ValueError("CLI compatibility addendum differs")
    expected = {"trainer": continuation.__file__, "evaluator": evaluator.__file__}
    for key, path in expected.items():
        if addendum["frozen_sources"].get(key) != {"path": str(Path(path).resolve()), "sha256": fixed.digest(path)}:
            raise ValueError("frozen compatibility source differs")
    evaluator.saved.verify_artifacts(addendum, {})


def evaluate(checkpoint, study, out, payload_out, compatibility):
    started = time.monotonic()
    checkpoint, study, out, payload_out, compatibility = map(Path,
        (checkpoint, study, out, payload_out, compatibility))
    addendum = read(compatibility)
    validate_addendum(addendum, study)
    proof_path = out.parent/(out.name+".compatibility.json")
    if proof_path.exists() or out.exists() or payload_out.exists():
        raise FileExistsError("compatible evaluation and proof identities must be new")
    record = read(checkpoint/"checkpoint.json")
    expected = record["optimizer_schedule"]
    if archive_schedule(checkpoint/"model.keras") != expected:
        raise ValueError("native archive and checkpoint exposure schedule differ")
    protected = {str(path.resolve()): fixed.digest(path) for path in
        (compatibility, study, Path(__file__), Path(continuation.__file__), Path(evaluator.__file__),
         checkpoint/"checkpoint.json", checkpoint/"model.keras", checkpoint/"state.npz")}
    serialization = restore_cli_module(continuation.schedule_class(), expected)
    workflow, restored = continuation.load_checkpoint(checkpoint)
    if restored != record:
        raise ValueError("checkpoint metadata changed during exact frozen loading")
    actual = fixed.state_summary(workflow.estimator, fixed.current_state(workflow.estimator))
    groups = {group: [item["sha256"] for item in actual[group]]
        == [item["sha256"] for item in record["state"][group]] for group in fixed.GROUPS}
    if (not all(groups.values()) or workflow.estimator.optimizer.get_config()["learning_rate"] != expected
            or int(workflow.estimator.optimizer.iterations.numpy()) != record["actual_optimizer_updates"]):
        raise ValueError("compatible frozen loading changed checkpoint arrays, iteration or schedule")
    proof = {"format": "heston_capacity_evaluation_cli_compatibility_proof_v1",
        "compatibility_addendum": {"path": str(compatibility.resolve()), "sha256": fixed.digest(compatibility)},
        "checkpoint_record": {"path": str((checkpoint/"checkpoint.json").resolve()), "sha256": fixed.digest(checkpoint/"checkpoint.json")},
        "bootstrap_source": addendum["bootstrap_source"], "frozen_sources": addendum["frozen_sources"],
        "serialization": serialization, "all_saved_array_groups_exact": groups,
        "array_counts": {group: len(actual[group]) for group in fixed.GROUPS},
        "optimizer_iteration_exact": True, "frozen_loader_all_guards_passed": True,
        "new_optimizer_updates": 0, "posterior_sampling_during_proof": False,
        "checkpoint_payloads_unchanged": True, "seconds": time.monotonic()-started}
    if any(fixed.digest(path) != digest for path, digest in protected.items()):
        raise ValueError("protected compatibility input changed during proof")
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.write_json(proof_path, proof)
    del workflow
    # The original consumer still performs all identity, source, target, state,
    # support, reference and population checks. Its output stays verbatim.
    result = evaluator.evaluate(checkpoint, study, out, payload_out)
    if any(fixed.digest(path) != digest for path, digest in protected.items()):
        raise ValueError("protected compatibility input changed during evaluation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "study", "out", "payload-out", "compatibility"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.study, args.out, args.payload_out, args.compatibility)
    print(result["synthetic_conditional_status"])


if __name__ == "__main__":
    main()

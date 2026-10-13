"""Train the frozen rough-Bergomi pilot after its independent reference passes."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from sbi4finance import option_bank, rough_inference
from sbi4finance.gbm_bank import _sha256


def accepted_inputs(bank_path, reference_path, protocol_path):
    """Validate recorded identities before creating a checkpoint or importing ML."""
    protocol = json.loads(protocol_path.read_text())
    config = json.loads(Path(protocol["bank_config"]).read_text())
    reference_config = json.loads(Path(protocol["reference_config"]).read_text())
    reference = json.loads((reference_path / "reference.json").read_text())
    if (not reference["passed"]
            or reference["protocol_sha256"] != _sha256(protocol_path)
            or reference["task"] != config["task"]
            or reference["reference_bank_configuration"] != reference_config
            or reference_config["task"] != config["task"]
            or reference["bank_manifest_sha256"] != _sha256(bank_path / "manifest.json")
            or reference["reference_bank_manifest_sha256"]
               != _sha256(reference_path / "reference-bank-manifest.json")
            or reference["artifact_sha256"] != {"reference.npz": _sha256(reference_path / "reference.npz")}):
        raise ValueError("matching accepted independent numerical reference required before training")
    reference_manifest = json.loads((reference_path / "reference-bank-manifest.json").read_text())
    if reference_manifest["configuration"] != reference_config:
        raise ValueError("reference bank configuration differs from the frozen experiment")
    expected_sources = {"check_rough_reference.py", "rough_reference.py", *option_bank.source_files("rough")}
    if set(reference["source_sha256"]) != expected_sources:
        raise ValueError("accepted reference has an incomplete or unexpected source set")
    for name, digest in reference["source_sha256"].items():
        source = (Path(__file__).with_name(name) if name == "check_rough_reference.py"
                  else Path(option_bank.__file__).with_name(name))
        if digest != _sha256(source):
            raise ValueError(f"accepted reference source changed: {name}")
    manifest, splits = option_bank.load_bank(bank_path)
    if manifest["configuration"] != config:
        raise ValueError("bank does not match the frozen experiment")
    for label, saved in (("bank", manifest), ("reference bank", reference_manifest)):
        if set(saved["source_sha256"]) != set(option_bank.source_files("rough")):
            raise ValueError(f"{label} has an unexpected generator source set")
        for name, digest in saved["source_sha256"].items():
            if digest != _sha256(Path(option_bank.__file__).with_name(name)):
                raise ValueError(f"{label} generator source changed: {name}")
    spec = protocol["training"]
    if (spec["density"], spec["depth"], spec["widths"], spec["bins"]) != ("coupling_spline", 4, [64, 64], 8):
        raise ValueError("protocol architecture differs from the implemented baseline")
    lo, hi = protocol["selection_validation_ids"]
    audit_lo, audit_hi = protocol["audit_validation_ids"]
    if not (0 <= lo < hi <= audit_lo < audit_hi <= len(splits["validation"]["parameters"])):
        raise ValueError("selection and audit ranges must be ordered and disjoint")
    return protocol, config, reference, manifest, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/rough_inference_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol, config, reference, manifest, splits = accepted_inputs(args.bank, args.reference, args.protocol)
    spec = protocol["training"]
    lo, hi = protocol["selection_validation_ids"]
    selection = {key: values[lo:hi] for key, values in splits["validation"].items()}
    train = rough_inference.augmented_training_data(
        splits["train"], config["task"], augmentations=spec["quote_augmentations_per_base"],
        seed=spec["augmentation_seed"])
    validation = rough_inference.training_data(selection, config["task"])
    args.out.mkdir(parents=True, exist_ok=False)

    import bayesflow as bf
    import jax
    import keras

    keras.utils.set_random_seed(spec["seed"])
    workflow = rough_inference.build_workflow(config["task"], args.out)
    ready = time.perf_counter()
    history = workflow.fit_offline(data=train, validation_data=validation,
                                   epochs=spec["epochs"], batch_size=spec["batch_size"], verbose=0)
    trained = time.perf_counter()
    workflow.load_approximator()
    conditions = rough_inference.conditioning_data(selection["observations"][:2], config["task"])
    before = workflow.sample(num_samples=128, conditions=conditions, seed=spec["seed"] + 11)
    restored = rough_inference.build_workflow(config["task"])
    restored.load_approximator(args.out / "model.keras")
    after = restored.sample(num_samples=128, conditions=conditions, seed=spec["seed"] + 11)
    if not all(np.array_equal(before[name], after[name]) for name in rough_inference.PARAMETER_NAMES):
        raise RuntimeError("checkpoint reload changed same-seed joint draws")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q",
        "observation_law": protocol["observation_law"],
        "protocol_sha256": _sha256(args.protocol),
        "reference_report_sha256": _sha256(args.reference / "reference.json"),
        "reference_bank_manifest_sha256": reference["reference_bank_manifest_sha256"],
        "bank_manifest_sha256": _sha256(args.bank / "manifest.json"), "bank_configuration": config,
        "checkpoint_sha256": _sha256(args.out / "model.keras"), "adapter_id": rough_inference.ADAPTER_ID,
        "training": spec, "selection_validation_ids": [lo, hi],
        "training_base_examples": len(splits["train"]["parameters"]),
        "training_examples": len(train["observations"]), "selection_examples": len(selection["parameters"]),
        "augmentation_scope": "Fresh quote errors per clean training base; shared theta and pricing randomness within each group. Selection and audit are unaugmented.",
        "augmentation_rng": "PCG64; SeedSequence(rough augmentation tag, seed, SHA256 task words, SHA256 split-qualified base ID words); base-major rows",
        "same_seed_reload_exact": True, "reserved_test_evaluated": False,
        "timing_seconds": {"load_and_setup": ready - started, "training_including_compile": trained - ready,
                           "reload_and_draw_check": time.perf_counter() - trained},
        "peak_rss_bytes": int(peak if platform.system() == "Darwin" else peak * 1024),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "bayesflow": bf.__version__,
                     "keras": keras.__version__, "jax": jax.__version__, "devices": [str(d) for d in jax.devices()]},
        "source_sha256": {"train_rough.py": _sha256(__file__), "rough_inference.py": _sha256(rough_inference.__file__)},
        "acceptance": "checkpoint/reload only; conditional, joint, SBC and target-mean evaluation required",
    }
    for name, content in (("training.json", report), ("task.json", config["task"]),
                          ("protocol.json", protocol), ("bank-manifest.json", manifest)):
        (args.out / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"training_seconds": trained - ready, "reload_exact": True,
                      "training_base_examples": report["training_base_examples"],
                      "training_examples": report["training_examples"],
                      "checkpoint_bytes": (args.out / "model.keras").stat().st_size}, indent=2))


if __name__ == "__main__":
    main()

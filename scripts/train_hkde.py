"""Train one frozen restricted HKDE baseline; leave scientific evaluation separate."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import bayesflow as bf
import jax
import keras
import numpy as np

from sbi4finance import hkde_inference, option_bank, training_workflow
from check_hkde_reference import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/hkde_inference_protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(Path(protocol["bank_config"]).read_text())
    reference = json.loads((args.reference / "reference.json").read_text())
    if (not reference["passed"] or reference["protocol_sha256"] != sha(args.protocol)
            or reference["task"] != config["task"]
            or reference["reference_grid_sha256"] != sha(args.reference / "reference-grid.npz")):
        raise ValueError("matching accepted numerical reference required before training")
    expected_sources = {"check_hkde_reference.py", "hkde_reference.py", *option_bank.source_files("hkde")}
    if set(reference["source_sha256"]) != expected_sources:
        raise ValueError("accepted reference has an incomplete or unexpected source set")
    for name, digest in reference["source_sha256"].items():
        source = (Path(__file__).with_name(name) if name == "check_hkde_reference.py"
                  else Path(option_bank.__file__).with_name(name))
        if digest != sha(source):
            raise ValueError(f"accepted reference source changed: {name}")
    manifest, splits = option_bank.load_bank(args.bank)
    if manifest["configuration"] != config:
        raise ValueError("bank does not match the frozen experiment")
    for name, digest in manifest["source_sha256"].items():
        if digest != sha(Path(option_bank.__file__).with_name(name)):
            raise ValueError(f"bank generator source changed: {name}")
    spec = protocol["training"]
    if (spec["density"], spec["depth"], spec["widths"], spec["bins"]) != ("coupling_spline", 4, [64, 64], 8):
        raise ValueError("protocol architecture differs from the implemented baseline")
    lo, hi = protocol["selection_validation_ids"]
    selection = {key: values[lo:hi] for key, values in splits["validation"].items()}
    train = hkde_inference.training_data(splits["train"], config["task"])
    validation = hkde_inference.training_data(selection, config["task"])
    training_workflow.reserve_output_dir(args.out)
    keras.utils.set_random_seed(spec["seed"])
    workflow = hkde_inference.build_workflow(config["task"], args.out)
    ready = time.perf_counter()
    history, lifecycle = training_workflow.fit_and_select(
        workflow, training_data=train, validation_data=validation,
        epochs=spec["epochs"], batch_size=spec["batch_size"], seed=spec["seed"], verbose=0)
    trained = time.perf_counter()
    small = hkde_inference.conditioning_data(selection["observations"][:2], config["task"])
    def restore():
        restored = hkde_inference.build_workflow(config["task"])
        restored.load_approximator(args.out / "model.keras")
        return restored
    lifecycle["same_seed_reload_exact"] = training_workflow.verify_reload(
        workflow, restore=restore,
        sample=lambda candidate: candidate.sample(num_samples=128, conditions=small, seed=spec["seed"] + 11))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "experiment_id": protocol["experiment_id"], "evidence_scope": "S", "measure": "Q",
        "protocol_sha256": sha(args.protocol), "reference_report_sha256": sha(args.reference / "reference.json"),
        "bank_manifest_sha256": sha(args.bank / "manifest.json"), "bank_configuration": config,
        "checkpoint_sha256": sha(args.out / "model.keras"), "adapter_id": hkde_inference.ADAPTER_ID,
        "training": spec, "selection_validation_ids": [lo, hi],
        "lifecycle": lifecycle,
        "training_examples": len(splits["train"]["parameters"]), "selection_examples": len(selection["parameters"]),
        "same_seed_reload_exact": True, "reserved_test_evaluated": False,
        "timing_seconds": {"load_and_setup": ready - started, "training_including_compile": trained - ready,
                           "reload_and_draw_check": time.perf_counter() - trained},
        "peak_rss_bytes": int(peak if platform.system() == "Darwin" else peak * 1024),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "bayesflow": bf.__version__,
                     "keras": keras.__version__, "jax": jax.__version__, "devices": [str(d) for d in jax.devices()]},
        "source_sha256": {"train_hkde.py": sha(__file__), "hkde_inference.py": sha(hkde_inference.__file__),
                          "training_workflow.py": sha(training_workflow.__file__)},
        "acceptance": "checkpoint/reload only; conditional, joint, SBC and target-price evaluation required",
    }
    for name, content in (("training.json", report), ("task.json", config["task"]),
                          ("protocol.json", protocol), ("bank-manifest.json", manifest)):
        (args.out / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"training_seconds": trained - ready, "reload_exact": True,
                      "checkpoint_bytes": (args.out / "model.keras").stat().st_size}, indent=2))


if __name__ == "__main__":
    main()

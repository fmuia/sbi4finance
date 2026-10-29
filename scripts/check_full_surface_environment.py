"""Check the isolated BayesFlow 2.1 runtime without running a tutorial or training.

    pixi run --locked -e core core-check
    pixi run --locked -e core-gpu core-check --require-gpu
"""

import argparse
from importlib import metadata
import json
import os
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="Fail if JAX cannot see a GPU.")
    args = parser.parse_args()
    if os.environ.get("KERAS_BACKEND") != "jax":
        raise RuntimeError("Activate the core environment: KERAS_BACKEND must be jax")
    if metadata.version("bayesflow") != "2.1.0":
        raise RuntimeError("New full-surface candidates require BayesFlow 2.1.0; use -e core")

    import bayesflow as bf
    import jax
    import keras

    if keras.backend.backend() != "jax":
        raise RuntimeError("Keras did not activate its JAX backend")
    # Check the actual public constructors required by the planned family screen.
    for constructor in (bf.networks.CouplingFlow, bf.networks.FlowMatching, bf.experimental.MeanFlow):
        if not callable(constructor):
            raise RuntimeError("Missing a required BayesFlow 2.1 network constructor")
    devices = [{"platform": device.platform, "kind": device.device_kind} for device in jax.devices()]
    if args.require_gpu and not any(device["platform"] == "gpu" for device in devices):
        raise RuntimeError("No JAX GPU device: check the core-gpu environment and cluster driver")
    versions = {name: metadata.version(name) for name in ("bayesflow", "keras", "jax", "numpy", "scipy")}
    try:
        versions["pymc"] = metadata.version("pymc")
    except metadata.PackageNotFoundError:
        pass
    print(json.dumps({"python": platform.python_version(), "versions": versions,
                      "keras_backend": keras.backend.backend(), "devices": devices,
                      "scope": "runtime/API availability only; no numerical or posterior validation"}, indent=2))


if __name__ == "__main__":
    main()

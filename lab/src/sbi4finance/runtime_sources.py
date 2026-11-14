"""Source identities for numerical implementations shared with the tutorial.

Keys are stable repository paths even when the runtime is installed elsewhere.
Callers select the modules on their producer's numerical path; presentation,
notebooks and asset caches do not enter a bank's producer identity.
"""

import hashlib
from importlib.util import find_spec
from pathlib import Path


def runtime_hashes(module_names):
    result = {}
    for name in sorted(set(module_names)):
        if not name.startswith("sbi4f_tutorial."):
            raise ValueError("a tutorial runtime module is required")
        spec = find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"shared runtime source is unavailable: {name}")
        path = Path(spec.origin)
        if path.suffix != ".py" or not path.is_file():
            raise RuntimeError(f"shared runtime source must be a readable Python file: {name}")
        key = "tutorial/src/" + name.replace(".", "/") + ".py"
        result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def option_runtime_hashes(task):
    common = ("sbi4f_tutorial.shared.bank_helpers", "sbi4f_tutorial.shared.gbm",
              "sbi4f_tutorial.shared.surface", "sbi4f_tutorial.shared.pricing",
              "sbi4f_tutorial.shared.tasks")
    model = {"hkde": ("sbi4f_tutorial.heston_options.model", "sbi4f_tutorial.heston_options.simulator"),
             "rough": ("sbi4f_tutorial.rough_options.model", "sbi4f_tutorial.rough_options.simulator")}[task["model"]]
    return runtime_hashes((*common, *model))


def trajectory_runtime_hashes(task):
    modules = ["sbi4f_tutorial.trajectories.simulation", "sbi4f_tutorial.shared.gbm"]
    if task["model"] == "heston":
        modules.append("sbi4f_tutorial.heston_options.simulator")
    else:
        modules.extend(("sbi4f_tutorial.rough_options.model", "sbi4f_tutorial.rough_options.simulator"))
    return runtime_hashes(modules)


def option_inference_runtime_hashes(task):
    """Current option adapter, validation, architecture and sampler sources."""
    modules = ["sbi4f_tutorial.shared.bank_helpers", "sbi4f_tutorial.shared.full_surface",
               "sbi4f_tutorial.shared.tasks", "sbi4f_tutorial.shared.surface",
               "sbi4f_tutorial.shared.pricing", "sbi4f_tutorial.shared.gbm"]
    # Task support is checked by the model validators. Pricing imports the rough
    # kernels, which are recorded even for a Heston task rather than hidden.
    modules.extend(("sbi4f_tutorial.heston_options.model", "sbi4f_tutorial.rough_options.model",
                    "sbi4f_tutorial.rough_options.simulator"))
    return runtime_hashes(modules)


def path_inference_runtime_hashes():
    """Current physical-parameter transforms, masks, GRU adapter and sampler."""
    return runtime_hashes(("sbi4f_tutorial.trajectories.inference",))


def option_pricer_runtime_hashes(task):
    """Forward-emulator features, training, export and numerical dependencies."""
    return {**option_inference_runtime_hashes(task),
            **runtime_hashes(("sbi4f_tutorial.rough_options.surrogate",
                              "sbi4f_tutorial.shared.parameter_transforms"))}

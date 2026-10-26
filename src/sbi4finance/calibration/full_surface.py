"""BayesFlow 2.1 inference on the complete 63-contract tutorial design.

This is deliberately separate from the archived supplied-curve/2.0 workflow.
The joint rough model's curve values are auxiliary labels, never conditions.
Set encoders here represent complete surfaces, not missing-quote support.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import surface, tasks
from ..gbm_bank import _integer, _json

FAMILIES = ("affine", "spline", "flow_matching", "mean_flow")
ENCODERS = ("vector", "deepset", "set_transformer")


def validate_task(task):
    """Accept only the declared complete-design Heston or joint rough task."""
    tasks.validate_task(task)
    contracts, canonical = task["context"]["contracts"], surface.design_grid()
    same_design = (len(contracts) == len(canonical)
                   and all(c["option_type"] == d["option_type"]
                           and np.allclose([c["strike"], c["maturity"]], [d["strike"], d["maturity"]],
                                           rtol=1e-12, atol=0.0)
                           for c, d in zip(contracts, canonical)))
    if not same_design:
        raise ValueError("the core tutorial requires the complete ordered 63-contract design")
    joint = getattr(tasks, "is_joint_curve_task", lambda _: False)(task)
    heston = (task["model"] == "hkde"
              and task["parameter_names"] == ["initial_variance", "theta", "vol_of_vol", "rho"])
    if not (heston or joint):
        raise ValueError("core inference supports four-target Heston or joint-curve rough, not supplied curves")
    return task


def observations(values, task):
    """Validate [B,63] or [B,63,1]; never discard additional channels."""
    validate_task(task)
    if (np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values))) or np.iscomplexobj(values):
        raise ValueError("missing/masked or complex prices are not supported by the core input contract")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != 63 or not len(values) or not np.isfinite(values).all():
        raise ValueError("complete finite prices [batch,63] or [batch,63,1] required; missing quotes unsupported")
    return values


def conditioning_data(values, task, encoder="vector"):
    """Prices/noise and declared contract coordinates only; no generating curve."""
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {ENCODERS}")
    prices = observations(values, task)
    noise = tasks.sigma(task)
    scaled = prices / noise
    if encoder == "vector":
        return {"observations": scaled}
    contracts = task["context"]["contracts"]
    coordinates = np.asarray([[np.log(c["strike"]), c["maturity"], float(c["option_type"] == "call"), sd]
                              for c, sd in zip(contracts, noise, strict=True)])
    tokens = np.concatenate([scaled[..., None], np.broadcast_to(coordinates, (*prices.shape, 4))], axis=-1)
    return {"observations": tokens}


def training_data(split, task, encoder="vector", *, clean=False):
    """Conditions plus ordered targets; stored context/curve arrays are ignored."""
    data = conditioning_data(split["reference_prices" if clean else "observations"], task, encoder)
    if ((np.ma.isMaskedArray(split["parameters"]) and np.any(np.ma.getmaskarray(split["parameters"])))
            or np.iscomplexobj(split["parameters"])):
        raise ValueError("masked or complex parameter targets are unsupported")
    parameters = np.asarray(split["parameters"], dtype=np.float64)
    bounds = np.asarray(task["prior_bounds"], dtype=np.float64)
    if (parameters.shape != (len(data["observations"]), len(bounds)) or not np.isfinite(parameters).all()
            or np.any(parameters <= bounds[:, 0]) or np.any(parameters >= bounds[:, 1])):
        raise ValueError("finite target rows strictly inside the declared prior box required")
    return {**data, **{name: parameters[:, i:i + 1] for i, name in enumerate(task["parameter_names"])}}


def fresh_noise(seed, encoder="vector"):
    """Fresh quote errors on prices alone; coordinate and noise channels are fixed."""
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {ENCODERS}")
    rng = np.random.default_rng(np.random.SeedSequence([0x46554C4C, _integer(seed, "noise seed")]))

    def augment(batch):
        result = dict(batch)
        values = np.array(batch["observations"], copy=True)
        if encoder == "vector":
            values += rng.standard_normal(values.shape)
        else:
            values[..., 0] += rng.standard_normal(values.shape[:-1])
        result["observations"] = values
        return result

    augment.rng = rng
    return augment


def configuration(*, family="spline", encoder="vector", widths=(128, 128), summary_dim=32,
                  flow_steps=100, mean_steps=4):
    """Serializable architecture and sampling choices, bound into run identity."""
    if family not in FAMILIES or encoder not in ENCODERS:
        raise ValueError(f"family/encoder must be from {FAMILIES}/{ENCODERS}")
    widths = [_integer(w, "width", minimum=1) for w in widths]
    if len(widths) < 2:
        raise ValueError("at least two subnet blocks required, including MeanFlow's velocity head")
    return {"family": family, "encoder": encoder, "widths": widths,
            "summary_dim": _integer(summary_dim, "summary_dim", minimum=1),
            "flow_steps": _integer(flow_steps, "flow_steps", minimum=1),
            "mean_steps": _integer(mean_steps, "mean_steps", minimum=1)}


def build_workflow(task, checkpoint_dir=None, *, optimizer=None, **settings):
    """Small BF2.1 family/encoder factory; each family retains its time convention."""
    validate_task(task)
    config = configuration(**settings)
    import bayesflow as bf
    if bf.__version__ != "2.1.0":
        raise RuntimeError("use the locked core environment (BayesFlow 2.1.0); archived environments remain separate")

    names = task["parameter_names"]
    adapter = bf.Adapter()
    for name, (lower, upper) in zip(names, task["prior_bounds"], strict=True):
        adapter = adapter.constrain(name, lower=float(lower), upper=float(upper), inclusive="none")
    destination = "inference_conditions" if config["encoder"] == "vector" else "summary_variables"
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(list(names), into="inference_variables")
               .concatenate(["observations"], into=destination))
    family = config["family"]
    subnet = {"widths": config["widths"]}
    if family in ("affine", "spline"):
        network = bf.networks.CouplingFlow(transform=family, subnet_kwargs=subnet)
    elif family == "flow_matching":
        # BF2.1 integrates noise at t=1 back to data at t=0 itself. No archived reversed grid.
        network = bf.networks.FlowMatching(subnet_kwargs=subnet,
                                          integrate_kwargs={"method": "euler", "steps": config["flow_steps"]})
    else:
        network = bf.experimental.MeanFlow(subnet_kwargs=subnet, steps=config["mean_steps"], density_head=False)
    summary = None
    if config["encoder"] == "deepset":
        summary = bf.networks.DeepSet(summary_dim=config["summary_dim"], dropout=0.0)
    elif config["encoder"] == "set_transformer":
        summary = bf.networks.SetTransformer(summary_dim=config["summary_dim"], dropout=0.0)
    return bf.BasicWorkflow(adapter=adapter, inference_network=network, summary_network=summary,
                            standardize=["inference_variables", destination], optimizer=optimizer,
                            checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
                            save_weights_only=False, save_best_only=True)


def sample(workflow, task, values, num_samples, *, encoder="vector", seed=0, mean_steps=None):
    """Joint draws [batch,draws,targets]; no contract axis becomes a sample axis."""
    count = _integer(num_samples, "num_samples", minimum=1)
    kwargs = {}
    if mean_steps is not None:
        if type(workflow.inference_network).__name__ != "MeanFlow":
            raise ValueError("mean_steps is only applicable to MeanFlow")
        kwargs["steps"] = _integer(mean_steps, "mean_steps", minimum=1)
    conditions = conditioning_data(values, task, encoder)
    draws = workflow.sample(num_samples=count, conditions=conditions, sample_shape=(), seed=_integer(seed, "seed"),
                            **kwargs)
    result = np.concatenate([np.asarray(draws[name], dtype=np.float64) for name in task["parameter_names"]], axis=-1)
    expected = (len(conditions["observations"]), count, len(task["parameter_names"]))
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"invalid posterior output: expected {expected}, received {result.shape}")
    return result


def load_checkpoint(directory):
    """Restore a core estimator, validating the task and recorded checkpoint bytes."""
    directory = Path(directory)
    task = validate_task(json.loads((directory / "task.json").read_text()))
    record = json.loads((directory / "training.json").read_text())
    try:
        settings = configuration(**record["network"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid core architecture metadata") from error
    if (record.get("format") != "full_surface_training_v1"
            or settings != record["network"]
            or record["network"] != record.get("run_contract", {}).get("network")
            or record.get("task_fingerprint") != hashlib.sha256(_json(task).encode()).hexdigest()
            or hashlib.sha256((directory / "model.keras").read_bytes()).hexdigest() != record["checkpoint_sha256"]):
        raise ValueError("incompatible core checkpoint or changed task/checkpoint/architecture metadata")
    workflow = build_workflow(task, **settings)
    workflow.load_estimator(directory / "model.keras")
    family_class = {"affine": "CouplingFlow", "spline": "CouplingFlow",
                    "flow_matching": "FlowMatching", "mean_flow": "MeanFlow"}[settings["family"]]
    summary_class = {"vector": None, "deepset": "DeepSet", "set_transformer": "SetTransformer"}[settings["encoder"]]
    summary = workflow.summary_network
    if (type(workflow.inference_network).__name__ != family_class
            or (None if summary is None else type(summary).__name__) != summary_class
            or (settings["family"] in ("affine", "spline")
                and workflow.inference_network.transform != settings["family"])):
        raise ValueError("saved estimator architecture differs from its recorded family/encoder")
    return task, workflow, record

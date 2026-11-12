"""Fixed-grid Heston compression experiments with the native six-target head.

Inputs are complete, already-whitened 63-vectors. This module neither fits nor
applies whitening and does not support missing quotes or moving contracts.
Importing it initializes no neural backend. Build a workflow (or explicitly
call ``register_serializables``) before loading a saved custom-summary model.
"""

from collections.abc import Mapping
from importlib import import_module

import numpy as np


FORMAT = "heston_fixed_grid_experimental_inference_v1"
PARAMETER_NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c")
INPUT_FEATURES = 63
_SUMMARY_CLASS = None


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _widths(values, name):
    try:
        result = [_integer(value, name) for value in values]
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of positive integers") from error
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def configuration(representation="direct", summary_dim=None, *, summary_widths=(128, 128),
                  posterior_widths=(128, 128, 128), posterior_dropout=.1, summary_activation="silu"):
    """Return a JSON-safe architecture record; direct retains the baseline head.

    MLP outputs are unconstrained learned summaries. Physical targets retain
    their raw coordinates and BayesFlow's native inference-variable scaling;
    no support transform, clipping, rejection or extra summary loss is added.
    """
    if representation not in ("direct", "mlp"):
        raise ValueError("representation must be direct or mlp")
    if representation == "direct":
        if summary_dim is not None:
            raise ValueError("direct representation has no summary dimension")
    else:
        summary_dim = _integer(summary_dim, "summary_dim")
    if (isinstance(posterior_dropout, (bool, np.bool_)) or not np.isscalar(posterior_dropout)
            or not np.isrealobj(posterior_dropout) or not np.isfinite(posterior_dropout)
            or not 0 <= posterior_dropout < 1):
        raise ValueError("posterior_dropout must lie in [0,1)")
    if not isinstance(summary_activation, str) or not summary_activation:
        raise ValueError("summary_activation must name a Keras activation")
    return {"format": FORMAT, "representation": representation, "summary_dim": summary_dim,
            "summary_widths": _widths(summary_widths, "summary_widths"),
            "posterior_widths": _widths(posterior_widths, "posterior_widths"),
            "posterior_dropout": float(posterior_dropout), "summary_activation": summary_activation,
            "input_features": INPUT_FEATURES,
            "target_transform": "raw_coordinates_native_inference_variable_standardization"}


def _configuration(value):
    if not isinstance(value, Mapping):
        raise ValueError("an experimental inference configuration is required")
    arguments = ("representation", "summary_dim", "summary_widths", "posterior_widths",
                 "posterior_dropout", "summary_activation")
    try:
        expected = configuration(**{name: value[name] for name in arguments})
    except KeyError as error:
        raise ValueError("incomplete experimental inference configuration") from error
    if dict(value) != expected:
        raise ValueError("unsupported experimental inference configuration")
    return expected


def register_serializables():
    """Lazily register the MLP summary for safe native Keras save/reload.

    This is also the entry point when loading an estimator directly with Keras
    in a fresh process. ``build_workflow`` performs registration automatically.
    """
    global _SUMMARY_CLASS
    if _SUMMARY_CLASS is not None:
        return _SUMMARY_CLASS
    import bayesflow as bf
    keras = import_module("keras")

    @keras.saving.register_keras_serializable(package="sbi4f_tutorial.heston_options")
    class FixedGridMLPSummary(bf.networks.SummaryNetwork):
        def __init__(self, summary_dim, widths=(128, 128), activation="silu", **kwargs):
            super().__init__(**kwargs)
            self.summary_dim = _integer(summary_dim, "summary_dim")
            self.widths = _widths(widths, "summary widths")
            self.activation = activation
            self.input_spec = keras.layers.InputSpec(ndim=2, axes={-1: INPUT_FEATURES})
            self.hidden = [keras.layers.Dense(width, activation=activation, name=f"hidden_{index}")
                           for index, width in enumerate(self.widths)]
            self.output_layer = keras.layers.Dense(self.summary_dim, name="summary")

        def call(self, inputs, training=None):
            result = inputs
            for layer in self.hidden:
                result = layer(result)
            return self.output_layer(result)

        def get_config(self):
            return {**super().get_config(), "summary_dim": self.summary_dim,
                    "widths": self.widths, "activation": self.activation}

    _SUMMARY_CLASS = FixedGridMLPSummary
    return _SUMMARY_CLASS


def build_workflow(task, configuration, seed=9011):
    """Build a BayesFlow 2.1 workflow without modifying the historical producer."""
    config = _configuration(configuration)
    target = task.get("target", task)
    if tuple(target.get("parameter_names", ())) != PARAMETER_NAMES or target.get("quote_count") != INPUT_FEATURES:
        raise ValueError("the ordered actual63 six-target Heston task is required")
    seed = _integer(seed, "seed", minimum=0) % (2 ** 32)
    import bayesflow as bf
    keras = import_module("keras")
    if bf.__version__ != "2.1.0":
        raise RuntimeError("experimental Heston inference requires locked BayesFlow 2.1.0")
    keras.utils.set_random_seed(seed)
    summary_type = register_serializables()
    key = "summary_variables" if config["representation"] == "mlp" else "inference_conditions"
    adapter = (bf.Adapter().keep([*PARAMETER_NAMES, "observations"])
               .concatenate(PARAMETER_NAMES, into="inference_variables").rename("observations", key))
    summary = None
    if config["representation"] == "mlp":
        summary = summary_type(config["summary_dim"], widths=config["summary_widths"],
                               activation=config["summary_activation"])
    return bf.BasicWorkflow(adapter=adapter, summary_network=summary,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={
            "widths": tuple(config["posterior_widths"]), "dropout": config["posterior_dropout"]}),
        standardize="inference_variables")


def sample(workflow, conditions, names, num_samples, seed=0):
    """Sample raw physical targets as [batch,draws,6] from whitened conditions.

    The condition dictionary deliberately accepts only ``observations``. This
    prevents truth, clean prices or noise realizations entering the inference
    path accidentally. Sampling retains out-of-prior draws for diagnostics.
    """
    if tuple(names) != PARAMETER_NAMES:
        raise ValueError("the canonical ordered six target names are required")
    count = _integer(num_samples, "num_samples")
    seed = _integer(seed, "seed", minimum=0) % (2 ** 32)
    if isinstance(conditions, Mapping):
        if set(conditions) != {"observations"}:
            raise ValueError("only observations may be supplied as conditions")
        conditions = conditions["observations"]
    if np.ma.isMaskedArray(conditions) or np.iscomplexobj(conditions):
        raise ValueError("complete real unmasked whitened observations are required")
    observations = np.atleast_2d(np.asarray(conditions))
    if (observations.ndim != 2 or observations.shape[1] != INPUT_FEATURES
            or not len(observations) or not np.isfinite(observations).all()):
        raise ValueError("observations must be finite complete [batch,63] whitened vectors")
    values = workflow.sample(conditions={"observations": observations}, num_samples=count,
                             sample_shape=(), seed=seed)
    result = np.concatenate([np.asarray(values[name]) for name in names], axis=-1)
    if result.shape != (len(observations), count, len(PARAMETER_NAMES)) or not np.isfinite(result).all():
        raise FloatingPointError("invalid six-target posterior sample shape or values")
    return result

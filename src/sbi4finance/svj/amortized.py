"""SVJ BayesFlow networks, online/saved-bank training and a compiled sampler.

A summary network compresses the 1,000 x 2 input of one window into 32 learned statistics, and a
flow-matching network carries standard normal noise to the joint posterior of the nine targets
(eight parameter coordinates and log spot volatility at the last day). Online training uses fresh
simulations; saved-bank training reuses finite training examples, with validation selecting the epoch.
A new window costs one summary pass and 50 Euler steps of the velocity network, compiled once.
"""

import json
import os
import platform
import time
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")  # must precede the bayesflow import

import bayesflow as bf
import jax
import jax.numpy as jnp
import keras
import numpy as np

from .. import training_workflow
from . import adapters, artifacts, bank as banks, model

SUMMARY_DIM = 32
EULER_STEPS = 50
CHECKPOINT_NAME = "model"

# Budgets from the design document: the reference run (2.56 million series) and the laptop variant.
PRESETS = {
    "reference": {"epochs": 50, "batches_per_epoch": 400, "batch_size": 128},
    "laptop": {"epochs": 10, "batches_per_epoch": 200, "batch_size": 128},
    "smoke": {"epochs": 2, "batches_per_epoch": 10, "batch_size": 64},
}


def build_adapter():
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .rename("targets", "inference_variables")
        .rename("features", "summary_variables")
    )


def build_summary_network():
    """Convolutions (8, 5, 5 days wide; strides 4, 2, 2) read by forward and backward GRUs, plus a skip branch."""
    return bf.networks.TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM,
        filters=(32, 64, 64),
        kernel_sizes=(8, 5, 5),
        strides=(4, 2, 2),
        recurrent_dim=64,
        bidirectional=True,
        dropout=0.05,
        skip_steps=4,
    )


def build_inference_network():
    """BayesFlow's default flow matching: TimeMLP with five FiLM residual blocks, tau = u^1.5 in training."""
    return bf.networks.FlowMatching(integrate_kwargs={"method": "euler", "steps": EULER_STEPS})


def build_workflow(seed=None, checkpoint_dir=None):
    return bf.BasicWorkflow(
        simulator=model.SVJSimulator(seed),
        adapter=build_adapter(),
        summary_network=build_summary_network(),
        inference_network=build_inference_network(),
        standardize="inference_variables",  # the returns are not standardized: their scale identifies volatility
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        checkpoint_name=CHECKPOINT_NAME,
        save_weights_only=False,
        save_best_only=True,
    )


def train(out_dir, epochs, batches_per_epoch, batch_size, seed=2026, n_validation=512, *, bank=None):
    """Train online or from a finite bank in a NEW run directory.

    Both modes select the minimum-validation-loss epoch and verify a saved
    reload. A bank's test split is read for storage validation only; its arrays
    never reach fitting, normalization, epoch selection or smoke sampling.
    """
    if Path(out_dir).exists():
        raise FileExistsError(f"training output already exists: {out_dir}")
    if type(n_validation) is not int or n_validation <= 0:
        raise ValueError("n_validation must be a positive integer")
    task = banks.default_task()
    manifest = None
    if bank is None:
        training = None
        validation = model.training_batch(n_validation, seed=seed + 1)
    else:
        manifest, splits = banks.load_bank(bank)
        training_workflow.validate_training_splits(splits)
        task = manifest["configuration"]["task"]
        if task != banks.default_task():
            raise ValueError("the current SVJ inference/audit workflow requires the complete default prior")
        training = adapters.bank_batch(splits["train"], task)
        validation = adapters.bank_batch(splits["validation"], task)
        n_validation = len(validation["targets"])
    out_dir = training_workflow.reserve_output_dir(out_dir)
    keras.utils.set_random_seed(seed)
    workflow = build_workflow(seed=seed, checkpoint_dir=out_dir)
    start = time.time()
    history, lifecycle = training_workflow.fit_and_select(
        workflow, training_data=training, validation_data=validation, epochs=epochs,
        batch_size=batch_size, seed=seed,
        num_batches_per_epoch=batches_per_epoch if bank is None else None, verbose=2,
    )
    def restored_workflow():
        restored = build_workflow()
        restored.load_approximator(out_dir / f"{CHECKPOINT_NAME}.keras")
        return restored

    lifecycle["same_seed_reload_exact"] = training_workflow.verify_reload(
        workflow, restore=restored_workflow,
        sample=lambda candidate: AmortizedPosterior(candidate.approximator).sample(
            validation["features"][:1], num_samples=8, seed=seed + 11),
        save=lambda candidate: save_approximator(candidate.approximator, out_dir / f"{CHECKPOINT_NAME}.keras"),
    )
    elapsed = time.time() - start
    record = {
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch if bank is None else int(np.ceil(len(training["targets"]) / batch_size)),
        "batch_size": batch_size,
        "simulated_series": lifecycle["training_exposures"] if bank is None else 0,
        "seed": seed,
        "validation_seed": seed + 1 if bank is None else None,
        "n_validation": n_validation,
        "seconds": round(elapsed, 1),
        "cpu_count": os.cpu_count(),
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "window": model.WINDOW,
        "targets": list(model.TARGET_NAMES),
        "versions": {"bayesflow": bf.__version__, "keras": keras.__version__, "jax": jax.__version__},
        "lifecycle": lifecycle,
        "bank_manifest_sha256": artifacts.sha256(Path(bank) / "manifest.json") if bank is not None else None,
        "bank_configuration": manifest["configuration"] if manifest is not None else None,
        "source_sha256": {"svj/amortized.py": artifacts.sha256(__file__),
                          "svj/adapters.py": artifacts.sha256(adapters.__file__),
                          "svj/artifacts.py": artifacts.sha256(artifacts.__file__),
                          "svj/model.py": artifacts.sha256(model.__file__),
                          "svj/bank.py": artifacts.sha256(banks.__file__),
                          "training_workflow.py": artifacts.sha256(training_workflow.__file__)},
        "runtime_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "platform": platform.platform(), "devices": [str(d) for d in jax.devices()]},
        "environment_lock_sha256": artifacts.sha256(Path(__file__).resolve().parents[3] / "pixi.lock")
        if (Path(__file__).resolve().parents[3] / "pixi.lock").is_file() else None,
        "scientific_acceptance": False,
    }
    (out_dir / "training.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    artifacts.write_checkpoint_sidecar(out_dir, task=task)
    return workflow.approximator, record


def save_approximator(approximator, path):
    """Save without optimizer state or compile config, so the checkpoint holds only what inference needs."""
    optimizer, compiled = getattr(approximator, "optimizer", None), approximator.compiled
    try:
        approximator.optimizer = None
        approximator.compiled = False
        approximator.save(path)
    finally:
        approximator.optimizer = optimizer
        approximator.compiled = compiled


def load_approximator(path, *, steps=EULER_STEPS):
    artifacts.validate_checkpoint(path, steps=steps, expected_task=banks.default_task())
    path = Path(path)
    if path.is_dir():
        path = path / f"{CHECKPOINT_NAME}.keras"
    with warnings.catch_warnings():
        # the checkpoint carries no optimizer state on purpose; Keras warns when it builds a fresh optimizer
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        return keras.saving.load_model(path)


class AmortizedPosterior:
    """Compiled sampler: summary network, then a fixed-step Euler solve of the learned flow.

    The same solver is used by the calibration checks, so they validate the sampler that runs in production.
    """

    def __init__(self, approximator, steps=EULER_STEPS):
        if type(steps) is not int or steps != EULER_STEPS:
            raise ValueError("the supported SVJ sampler uses exactly 50 Euler steps")
        self.approximator = approximator
        self.steps = steps
        summary_network = approximator.summary_network
        inference_network = approximator.inference_network
        standardizer = approximator.standardizer

        def summarize(features):
            return summary_network(features, training=False)

        def solve(features, noise):
            summaries = summary_network(features, training=False)
            conditions = jnp.broadcast_to(
                summaries[:, None, :], noise.shape[:2] + (summaries.shape[-1],)
            )

            def euler_step(k, x):
                time = k.astype(x.dtype) / steps
                return x + inference_network.velocity(x, time=time, conditions=conditions, training=False) / steps

            x = jax.lax.fori_loop(0, steps, euler_step, noise)
            return standardizer.maybe_standardize(x, key="inference_variables", stage="inference", forward=False)

        self._summarize = jax.jit(summarize)
        self._solve = jax.jit(solve)

    @classmethod
    def load(cls, path, steps=EULER_STEPS):
        posterior = cls(load_approximator(path, steps=steps), steps)
        posterior.task = banks.default_task()
        posterior.artifact = artifacts.validate_checkpoint(path, steps=steps, expected_task=posterior.task)
        return posterior

    @staticmethod
    def _features(data):
        return adapters.native_features(data)

    def summaries(self, data, chunk=256):
        """Learned summary statistics, shape (n, 32). `data` is returns (n, T) or features (n, T, 2)."""
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("chunk must be a positive integer")
        features = self._features(data)
        out = []
        for start in range(0, len(features), chunk):
            block = features[start : start + chunk]
            pad = chunk - len(block)
            if pad:
                block = np.concatenate([block, np.repeat(block[:1], pad, 0)])
            out.append(np.asarray(self._summarize(block))[: chunk - pad])
        return np.concatenate(out)

    def sample(self, data, num_samples=1_000, seed=0, max_points=25_000):
        """Posterior draws of the nine targets, shape (n, num_samples, 9), in target coordinates."""
        if any(type(value) is not int or value <= 0 for value in (num_samples, max_points)):
            raise ValueError("num_samples and max_points must be positive integers")
        features = self._features(data)
        rng = np.random.default_rng(seed)
        per_call = max(1, max_points // num_samples)
        out = []
        for start in range(0, len(features), per_call):
            block = features[start : start + per_call]
            pad = per_call - len(block) if len(features) > per_call else 0
            if pad:
                block = np.concatenate([block, np.repeat(block[:1], pad, 0)])
            noise = rng.standard_normal((len(block), num_samples, model.N_TARGETS)).astype("float32")
            out.append(np.asarray(self._solve(block, noise))[: len(block) - pad])
        return np.concatenate(out).astype(float)

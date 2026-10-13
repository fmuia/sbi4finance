"""Amortized posterior for the SVJ model: BayesFlow networks, online training and a compiled sampler.

A summary network compresses the 1,000 x 2 input of one window into 32 learned statistics, and a
flow-matching network carries standard normal noise to the joint posterior of the nine targets
(eight parameter coordinates and log spot volatility at the last day). Training is online: every
batch is freshly simulated, so the network never sees a series twice. After training, a new window
costs one summary pass and 50 Euler steps of the velocity network, compiled once.
"""

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")  # must precede the bayesflow import

import bayesflow as bf
import jax
import jax.numpy as jnp
import keras
import numpy as np

from . import model

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
    )


def train(out_dir, epochs, batches_per_epoch, batch_size, seed=2026, n_validation=512):
    """Train online from the prior and save the approximator and a training record under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(seed)
    workflow = build_workflow(seed=seed, checkpoint_dir=out_dir)
    validation = model.training_batch(n_validation, seed=seed + 1)
    start = time.time()
    history = workflow.fit_online(
        epochs=epochs,
        num_batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        validation_data=validation,
        verbose=2,
    )
    elapsed = time.time() - start
    save_approximator(workflow.approximator, out_dir / f"{CHECKPOINT_NAME}.keras")
    record = {
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch_size": batch_size,
        "simulated_series": epochs * batches_per_epoch * batch_size,
        "seed": seed,
        "n_validation": n_validation,
        "seconds": round(elapsed, 1),
        "cpu_count": os.cpu_count(),
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "window": model.WINDOW,
        "targets": list(model.TARGET_NAMES),
        "versions": {"bayesflow": bf.__version__, "keras": keras.__version__, "jax": jax.__version__},
    }
    (out_dir / "training.json").write_text(json.dumps(record, indent=1))
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


def load_approximator(path):
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
        return cls(load_approximator(path), steps)

    @staticmethod
    def _features(data):
        data = np.asarray(data)
        if data.ndim < 3:  # returns (T,) or (n, T); features are (n, T, 2)
            data = model.network_input(np.atleast_2d(data))
        return data.astype("float32")

    def summaries(self, data, chunk=256):
        """Learned summary statistics, shape (n, 32). `data` is returns (n, T) or features (n, T, 2)."""
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

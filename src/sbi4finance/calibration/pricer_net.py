"""The neural pricer: deep calibration's forward map, and MCMC on it (studies S4 and S6).

A multilayer perceptron maps the parameters, in the logit coordinates of the
prior box, and the log forward-variance levels to the 63 clean prices divided
by their quote-noise standard deviations. Its error is therefore measured in
the units of the likelihood: an error of 0.1 shifts each quote's log-likelihood
by at most about 0.1 times the residual. It is trained on the bank's clean
reference prices, which cost nothing beyond the bank itself.

The trained weights are mirrored in a NumPy forward pass, so the Metropolis
chains of the neural-likelihood route need no framework call per step.
"""

import json
from pathlib import Path

import numpy as np

from . import correction, tasks


def features(task, parameters, context_values=None):
    z = correction.to_unbounded(np.atleast_2d(parameters), correction.bounds_of(task))
    if tasks.context_size(task):
        return np.hstack([z, np.log(np.atleast_2d(context_values))])
    return z


def targets(task, reference_prices):
    return np.atleast_2d(reference_prices) / tasks.sigma(task)


class Pricer:
    """Standardized inputs -> MLP (gelu) -> standardized outputs, evaluated in NumPy."""

    def __init__(self, weights, x_mean, x_sd, y_mean, y_sd):
        self.weights = [(np.asarray(w), np.asarray(b)) for w, b in weights]
        self.x_mean, self.x_sd = np.asarray(x_mean), np.asarray(x_sd)
        self.y_mean, self.y_sd = np.asarray(y_mean), np.asarray(y_sd)

    def __call__(self, x):
        h = (np.atleast_2d(x) - self.x_mean) / self.x_sd
        for w, b in self.weights[:-1]:
            a = h @ w + b
            h = 0.5 * a * (1.0 + np.tanh(0.7978845608028654 * (a + 0.044715 * a**3)))  # gelu, tanh form
        w, b = self.weights[-1]
        return (h @ w + b) * self.y_sd + self.y_mean

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        arrays = {"x_mean": self.x_mean, "x_sd": self.x_sd, "y_mean": self.y_mean, "y_sd": self.y_sd}
        for i, (w, b) in enumerate(self.weights):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        np.savez(path / "pricer.npz", **arrays)

    @classmethod
    def load(cls, path):
        with np.load(Path(path) / "pricer.npz") as saved:
            layers = len([k for k in saved.files if k.startswith("w")])
            return cls([(saved[f"w{i}"], saved[f"b{i}"]) for i in range(layers)],
                       saved["x_mean"], saved["x_sd"], saved["y_mean"], saved["y_sd"])


def train(task, train_split, validation_split, *, widths=(256, 256, 256, 256), epochs=300, batch_size=512,
          learning_rate=1e-3, patience=30, seed=0):
    """Fit the MLP by mean squared error in noise units; keep the epoch with the best validation loss."""
    import keras

    keras.utils.set_random_seed(seed)
    x_train = features(task, train_split["parameters"], train_split.get("context_values"))
    y_train = targets(task, train_split["reference_prices"])
    x_val = features(task, validation_split["parameters"], validation_split.get("context_values"))
    y_val = targets(task, validation_split["reference_prices"])
    x_mean, x_sd = x_train.mean(0), x_train.std(0) + 1e-12
    y_mean, y_sd = y_train.mean(0), y_train.std(0) + 1e-12
    layers = [keras.Input((x_train.shape[1],))]
    model = keras.Sequential(layers + [keras.layers.Dense(w, activation="gelu") for w in widths]
                             + [keras.layers.Dense(y_train.shape[1])])
    schedule = keras.optimizers.schedules.CosineDecay(learning_rate, epochs * int(np.ceil(len(x_train) / batch_size)))
    model.compile(optimizer=keras.optimizers.Adam(schedule), loss="mse")
    stop = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    history = model.fit((x_train - x_mean) / x_sd, (y_train - y_mean) / y_sd, epochs=epochs, batch_size=batch_size,
                        validation_data=((x_val - x_mean) / x_sd, (y_val - y_mean) / y_sd), callbacks=[stop], verbose=0)
    weights = [(layer.get_weights()[0], layer.get_weights()[1]) for layer in model.layers]
    pricer = Pricer(weights, x_mean, x_sd, y_mean, y_sd)
    return pricer, {"epochs_run": len(history.history["loss"]),
                    "best_val_mse_standardized": float(min(history.history["val_loss"]))}


def errors(pricer, task, split):
    """Prediction error in quote-noise units, per surface and contract."""
    predicted = pricer(features(task, split["parameters"], split.get("context_values")))
    return predicted - targets(task, split["reference_prices"])


def log_posterior(pricer, task, observation, context_values=None):
    """log p(y | theta) p(theta) |d theta / d z| as a function of logit coordinates z [N, P]."""
    bounds = correction.bounds_of(task)
    y = np.asarray(observation, dtype=float).reshape(-1) / tasks.sigma(task)
    context = None if context_values is None else np.log(np.atleast_2d(context_values))

    def target(z):
        z = np.atleast_2d(z)
        x = z if context is None else np.hstack([z, np.repeat(context, len(z), axis=0)])
        residual = pricer(x) - y
        return -0.5 * np.sum(residual**2, axis=1) + correction.log_jacobian(z, bounds)

    return target


def metropolis(log_target, starts, scale, num_steps, seed):
    """Vectorized random-walk Metropolis: one chain per row of starts [C, P]; returns [C, steps, P]."""
    rng = np.random.default_rng(seed)
    z = np.array(starts, dtype=float)
    current = log_target(z)
    chol = np.linalg.cholesky(np.atleast_2d(scale))
    chains = np.empty((len(z), num_steps, z.shape[1]))
    accepted = np.zeros(len(z))
    for step in range(num_steps):
        proposal = z + rng.standard_normal(z.shape) @ chol.T
        value = log_target(proposal)
        accept = np.log(rng.random(len(z))) < value - current
        z[accept], current[accept] = proposal[accept], value[accept]
        accepted += accept
        chains[:, step] = z
    return chains, accepted / num_steps


def save_record(path, record):
    (Path(path) / "pricer.json").write_text(json.dumps(record, indent=2) + "\n")

"""The neural pricer: deep calibration's forward map, and MCMC on it (studies S4 and S6).

A multilayer perceptron maps the parameters, in the logit coordinates of the
prior box to the 63 clean prices divided by their quote-noise standard
deviations. The known-curve task also supplies log forward-variance levels;
the joint task uses its six parameters only, excluding auxiliary curve labels.
For a scaled quote residual r and emulator error d, the absolute log-likelihood
change is bounded by abs(r*d) + d*d/2; these contributions accumulate across
contracts. It is trained on the bank's clean
reference prices, which cost nothing beyond the bank itself.

The trained weights are mirrored in a NumPy forward pass, so the Metropolis
chains of the neural-likelihood route need no framework call per step.
"""

import json
from pathlib import Path

import numpy as np

from . import correction, tasks
from sbi4f_tutorial.rough_options.surrogate import (
    features, _log_known_curve, quote_noise_mse, targets, Pricer, train, errors,
)


def log_posterior(pricer, task, observation, context_values=None):
    """log p(y | theta) p(theta) |d theta / d z| as a function of logit coordinates z [N, P]."""
    bounds = correction.bounds_of(task)
    y = np.asarray(observation, dtype=float).reshape(-1) / tasks.sigma(task)
    context = (_log_known_curve(task, context_values, 1)
               if tasks.context_size(task) and not tasks.is_joint_curve_task(task) else None)

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

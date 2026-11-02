"""Participant-only views of verified cached examples; neural imports are explicit."""

import json
from pathlib import Path

import numpy as np

from . import full_surface_teaching as replay
from .paths import ASSETS

CATALOGUE = ASSETS / "backbones" / "artifacts.json"

PARAMETER_LABELS = {
    "initial_variance": ("Initial variance v₀", "v₀"),
    "theta": ("Long-run variance θ", "θ"),
    "vol_of_vol": ("Volatility of variance σᵥ", "σᵥ"),
    "rho": ("Price–variance correlation ρ", "ρ"),
    "hurst": ("Roughness H", "H"),
    "eta": ("Variance amplitude η", "η"),
    "short_variance": ("Initial curve level S", "S"),
    "long_variance": ("Long-run curve level L", "L"),
    "decay": ("Curve decay k", "k"),
}


def examples(prefix=None):
    names = list(json.loads(CATALOGUE.read_text())["artifacts"])
    return names if prefix is None else [name for name in names if name.startswith(prefix)]


def load_example(name):
    entry = json.loads(CATALOGUE.read_text())["artifacts"][name]
    if not (CATALOGUE.parent / entry["cache"]).resolve().is_relative_to(ASSETS.resolve()):
        raise ValueError("cached examples must remain under tutorial/assets/")
    return replay.load_tutorial_artifact(CATALOGUE, name)


def load_live(name, directory):
    """Load only a matching BF2.1 native bundle in the explicit downloaded-asset cache."""
    path = Path(directory).expanduser().resolve()
    if not path.is_relative_to(ASSETS.resolve()) or not path.is_relative_to((ASSETS / "local-checkpoints").resolve()):
        raise ValueError("download matching models under tutorial/assets/local-checkpoints/ first")
    load_example(name)
    return replay.load_tutorial_checkpoint(CATALOGUE, name, path)


def parameter_rows(task, arrays):
    draws = arrays["neural_draws"]
    bounds = np.asarray(task["prior_bounds"])
    q = np.quantile(draws, [.05, .5, .95], axis=0)
    return [{"parameter": PARAMETER_LABELS.get(name, (name, name))[0], "prior": f"{bounds[i, 0]:.4g} to {bounds[i, 1]:.4g}",
             "median": float(q[1, i]), "central 90%": f"{q[0, i]:.4g} to {q[2, i]:.4g}"}
            for i, name in enumerate(task["parameter_names"])]


def corner(task, arrays):
    """All target marginals and pairs, preserving the jointly sampled rows."""
    import matplotlib.pyplot as plt

    sets = {"neural": arrays["neural_draws"]}
    names = task["parameter_names"]
    n = len(names)
    if "reference_chains" in arrays:
        sets["PyMC reference"] = arrays["reference_chains"].reshape(-1, n)
    fig, axes = plt.subplots(n, n, figsize=(max(7, 1.8 * n), max(7, 1.8 * n)), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            for label, draws in sets.items():
                if i == j:
                    ax.hist(draws[:, i], bins=35, density=True, histtype="step", label=label)
                else:
                    ids = np.linspace(0, len(draws) - 1, min(1000, len(draws)), dtype=int)
                    ax.scatter(draws[ids, j], draws[ids, i], s=2, alpha=.18, rasterized=True)
            if "truth" in arrays:
                if i == j:
                    ax.axvline(arrays["truth"][i], color="black", linestyle=":", linewidth=1)
                else:
                    ax.plot(arrays["truth"][j], arrays["truth"][i], "k+", markersize=6)
            if i == n - 1:
                ax.set_xlabel(PARAMETER_LABELS.get(names[j], (names[j], names[j]))[1], fontsize=9)
            if j == 0:
                ax.set_ylabel(PARAMETER_LABELS.get(names[i], (names[i], names[i]))[1], fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig

"""Shared helpers for the course figures: colours, paths, data loaders, Black-Scholes in forward units."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "src"))
from sbi4finance.calibration import surface as sf  # noqa: E402,F401

OUT = ROOT / "course" / "images"
COLORS = ["#087F8C", "#D97732", "#738095", "#B33E43", "#5B8C3A", "#7A4E9C", "#2F2F2F"]
GREY = "0.55"
DAYS = np.array(sf.DESIGN_DAYS)
DELTAS = np.array(sf.DESIGN_DELTAS)
SPY_FILE = ROOT / "tutorial" / "assets" / "results" / "calibration" / "spy" / "surface.json"
VOO_FILE = ROOT / "course" / "scripts" / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv"


def tidy(*axes):
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def load_spy():
    return json.loads(SPY_FILE.read_text())


def load_voo():
    """Daily simple returns of VOO, as stored in the file, and the derived log returns."""
    frame = pd.read_csv(VOO_FILE, index_col=0, parse_dates=True)
    simple = frame["VOO"].dropna()
    return simple, np.log1p(simple)


def d12(k, w):
    s = np.sqrt(w)
    d1 = (-k + 0.5 * w) / s
    return d1, d1 - s


def call(k, w):
    """Black-Scholes call in forward units, as a function of log-moneyness k and total variance w."""
    d1, d2 = d12(k, w)
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def put(k, w):
    return call(k, w) - 1.0 + np.exp(k)


def otm(k, w):
    return np.where(np.asarray(k) <= 0, put(k, w), call(k, w))

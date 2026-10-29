"""Draw joint_is_A.png from joint_is_A.npz and joint_is_A-products.npz.

Left: the posterior of (long variance, decay) against the Gauss-Newton 90% region and the prior box. Centre and
right: the 1-to-2-year forward variance and the 2-year 90% put under the posterior, under the Gauss-Newton normal
(delta method), and with the curve held flat beyond one year and only (H, eta, rho) uncertain.

Run from lab/: python checks/why_bayes/joint_is_figure.py
"""

from pathlib import Path

import numpy as np
from scipy import stats

from sbi4finance.calibration.plots import BASELINE, INK, INK_2, SERIES, SURFACE, plt, style

HERE = Path("checks/why_bayes")
LONG_BOX, DECAY_BOX = (0.01, 0.16), (0.25, 8.0)
POSTERIOR, GAUSS_NEWTON, FLAT = SERIES


def forward_variance(theta, start=1.0, end=2.0):
    """The forward variance from start to end years under the joint task's curve."""
    s, l, k = theta[3:]
    w = l * np.array([start, end]) + (s - l) * -np.expm1(-k * np.array([start, end])) / k
    return (w[1] - w[0]) / (end - start)


def delta_sd(f, theta, cov, h=1e-6):
    g = np.array([(f(theta + e) - f(theta - e)) / (2 * h) for e in np.eye(len(theta)) * h])
    return float(np.sqrt(g @ cov @ g))


def densities(ax, values, weights, normals, truth, scale, xlabel, title):
    """The posterior as a weighted histogram, each classical answer as its normal density, the truth as a line."""
    centres = [c for _, c, _, _ in normals]
    sds = [s for _, _, s, _ in normals]
    lo = min(np.quantile(values, 0.001), min(c - 3.5 * s for c, s in zip(centres, sds)))
    hi = max(np.quantile(values, 0.999), max(c + 3.5 * s for c, s in zip(centres, sds)))
    grid = np.linspace(scale * lo, scale * hi, 400)
    ax.hist(scale * values, bins=np.linspace(grid[0], grid[-1], 35), weights=weights, density=True,
            color=POSTERIOR, alpha=0.35, linewidth=0, label="posterior")
    for label, centre, sd, color in normals:
        ax.plot(grid, stats.norm.pdf(grid, scale * centre, scale * sd), color=color, linewidth=2, label=label)
    ax.axvline(scale * truth, color=INK, linewidth=1.2, linestyle="--", label="truth")
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7, loc="upper right")


style()
d = np.load(HERE / "joint_is_A.npz")
p = np.load(HERE / "joint_is_A-products.npz")
ok, w, draws, truth, cov = d["valid"], d["weights"], d["draws"], d["truth"], d["gn_cov"]
shown = np.random.default_rng(0).choice(np.flatnonzero(ok), size=600, p=w[ok] / w[ok].sum())  # display only

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)

ax = axes[0]
for span in ((-0.2, LONG_BOX[0]), (LONG_BOX[1], 0.3)):
    ax.axvspan(*span, color=BASELINE, alpha=0.25, linewidth=0)
ax.axhspan(-3, DECAY_BOX[0], color=BASELINE, alpha=0.25, linewidth=0)
vals, vecs = np.linalg.eigh(cov[np.ix_([4, 5], [4, 5])])
angle = np.linspace(0, 2 * np.pi, 400)
ring = vecs @ (np.sqrt(vals * stats.chi2.ppf(0.9, 2))[:, None] * np.vstack([np.cos(angle), np.sin(angle)]))
ax.plot(truth[4] + ring[0], truth[5] + ring[1], color=GAUSS_NEWTON, linewidth=2, label="Gauss–Newton 90% region")
ax.scatter(draws[shown, 4], draws[shown, 5], s=9, color=POSTERIOR, alpha=0.35, linewidth=0, label="posterior draws")
ax.scatter([truth[4]], [truth[5]], marker="*", s=90, color=INK, edgecolor=SURFACE, linewidth=0.6, label="truth", zorder=5)
ax.text(-0.07, -2.0, "outside the prior", ha="center", fontsize=8, color=INK_2)
ax.set_xlim(-0.13, 0.25)
ax.set_ylim(-2.5, 4.3)
ax.set_xlabel("long variance")
ax.set_ylabel("decay (per year)")
ax.set_title("The curve's long end: posterior against Gauss–Newton", loc="left")
ax.legend(fontsize=7, loc="upper right")

# With the curve held flat beyond one year the forward variance is the 6-to-12-month level: a point, not a range.
flat_level = forward_variance(truth, 0.5, 1.0)
densities(axes[1], d["curve_products"][ok, 1], w[ok],
          [("Gauss–Newton, delta method", forward_variance(truth), delta_sd(forward_variance, truth, cov), GAUSS_NEWTON)],
          forward_variance(truth), 100, "forward variance, one to two years (×100)",
          "Beyond the last quote: the 1–2 year forward variance")
axes[1].axvline(100 * flat_level, color=FLAT, linewidth=2, label="curve flat beyond one year")
axes[1].legend(fontsize=7, loc="upper right")

densities(axes[2], p["prices"][:, 1], p["weights"],
          [("Gauss–Newton, delta method", p["mark"][1], p["delta_sd"][1], GAUSS_NEWTON),
           ("curve flat beyond one year", p["flat"][1], p["flat_sd"][1], FLAT)],
          p["mark"][1], 100, "2-year 90% put (% of the forward)", "The 2-year 90% put")

fig.savefig(HERE / "joint_is_A.png", dpi=160)

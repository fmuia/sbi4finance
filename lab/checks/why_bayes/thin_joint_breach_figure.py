"""Draw DIR/breach.png from thin_joint_breach.py's DIR/summary.json: each method's breach rate per product and side,
for the methods the summary holds (laplace_delta stays in the table only).

Run from lab/: python checks/why_bayes/thin_joint_breach_figure.py DIR
"""

import json
import sys
from pathlib import Path

import numpy as np

from sbi4finance.calibration.plots import BASELINE, INK, INK_2, SERIES, plt, style

METHODS = {"posterior": ("posterior (network)", SERIES[0], "o"),
           "gauss_newton": ("Gauss–Newton, delta method", SERIES[1], "s"),
           "hurst_fixed": ("Gauss–Newton, H fixed at 0.10", SERIES[2], "^"),
           "flat_curve": ("Gauss–Newton, curve flat beyond 6 months", "#eda100", "D"),
           "laplace": ("Laplace at the posterior mode", "#e87ba4", "v")}
LABELS = {"variance_swap_6m": "variance swap\n6m", "volatility_swap_6m": "volatility\nswap 6m",
          "forward_start_3m_6m": "forward start\n3m→6m", "put_6m_80": "put 6m\n80%", "cliquet_6m": "cliquet\n6m",
          "forward_start_6m_1y": "forward start\n6m→1y"}

out = Path(sys.argv[1])
report = json.loads((out / "summary.json").read_text())
n = report["chains"]
shown = {m: spec for m, spec in METHODS.items() if m in report["methods"]}
style()
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True, constrained_layout=True)
x = np.arange(len(report["products"]))
for ax, side, title in ((axes[0], "long", "Long side: true price below the lower bound"),
                        (axes[1], "short", "Short side: true price above the upper bound")):
    ax.axhline(0.1, color=INK, linewidth=1, linestyle="--")
    for k, (method, (label, colour, marker)) in enumerate(shown.items()):
        r = report["methods"][method]
        rate = np.array(r[f"{side}_breaches"]) / n
        ci = np.array(r[f"{side}_wilson"])
        offset = (k - (len(shown) - 1) / 2) * 0.15
        ax.errorbar(x + offset, rate, yerr=[rate - ci[:, 0], ci[:, 1] - rate], fmt=marker, color=colour,
                    markersize=6, capsize=0, elinewidth=1.2, label=label)
    ax.set_xticks(x, [LABELS.get(p, p) for p in report["products"]], fontsize=8)
    ax.axvspan(len(x) - 1.5, len(x) - 0.5, color=BASELINE, alpha=0.18, linewidth=0)
    ax.set_title(title, loc="left")
axes[0].set_ylabel(f"share of {n} chains breaching the 90% bound")
axes[1].text(len(x) - 1, axes[1].get_ylim()[1] * 0.97, "beyond the\nlast quote", ha="center", va="top", fontsize=8,
             color=INK_2)
axes[0].text(-0.4, 0.105, "nominal 10%", fontsize=8, color=INK, va="bottom")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=min(len(shown), 3), fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.savefig(out / "breach.png", dpi=160, bbox_inches="tight")

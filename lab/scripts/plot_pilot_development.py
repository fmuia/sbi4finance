"""Reproduce two development figures from the recorded Oscar JSON artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def curve_shapes(root):
    a = read(root/"curve-followup/A/report.json")
    b = read(root/"curve-followup/B/report.json")
    if a["case"] != b["case"] or a["observations"] != b["observations"]:
        raise ValueError("curve comparisons must share the same truth and observed quotes")
    time = np.linspace(0, 1, 401)
    b0, b1, b2, tau1, tau2 = a["case"]["curve_coefficients"]
    truth = b0+b1*np.exp(-time/tau1)+b2*(time/tau2)*np.exp(-time/tau2)
    curves = [("Constructed hump truth", truth, "#183b56", "-")]
    for label, report, color, style in (("Exponential fit A: decay cap 8", a, "#c76c25", "--"),
                                        ("Exponential fit B: decay cap 32", b, "#20877c", "-.")):
        short, long, decay = report["fit"]["parameters"][3:]
        curves.append((label, long+(short-long)*np.exp(-decay*time), color, style))
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for label, values, color, style in curves:
        ax.plot(12*time, values, label=label, color=color, linestyle=style, linewidth=2.2)
    ax.set(xlabel="Time (months)", ylabel="Annualized forward variance", xlim=(0, 12), ylim=(0.01, 0.085))
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Initial-curve shape: a controlled hump challenge", fontsize=13, y=0.98)
    ax.set_title("Point fits only; no posterior uncertainty bands", fontsize=10, pad=12)
    fig.text(0.5, 0.015, "Oscar development screen, 26 Oct 2026. Not a market-frequency or acceptance result.",
             ha="center", fontsize=8, color="#505050")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(root/"curve-shape-comparison.png", dpi=155)
    plt.close(fig)


def heston_coverage(root):
    reports = [(60, read(root/"heston-development/evaluation-60epochs.json"), "#5277b3", "--", "o"),
               (300, read(root/"heston-development/evaluation-300epochs.json"), "#20877c", "-", "s")]
    first = reports[0][1]
    if len(first["example_ids"]) != 500 or any(r["example_ids"] != first["example_ids"] for _, r, *_ in reports):
        raise ValueError("expected the same 500 recorded validation cases")
    names = ["Initial variance", "Long-run variance", "Volatility of variance", "Correlation"]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.3), sharex=True, sharey=True)
    for j, ax in enumerate(axes.flat):
        ax.plot([0, 1], [0, 1], color="#7c7c7c", linewidth=1.2, linestyle=":", label="Nominal diagonal")
        for epochs, report, color, style, marker in reports:
            levels = sorted(report["coverage"], key=float)
            nominal = np.array([float(k) for k in levels])
            values = np.array([report["coverage"][k]["coverage"][j] for k in levels])
            interval = np.array([report["coverage"][k]["wilson"][j] for k in levels])
            ax.errorbar(nominal, values, yerr=np.array([values-interval[:, 0], interval[:, 1]-values]),
                        color=color, linestyle=style, marker=marker, markersize=4, linewidth=1.5,
                        elinewidth=1, capsize=2, label=f"{epochs} epochs")
        ax.set(title=names[j], xlim=(0.45, 0.99), ylim=(0.43, 1.015))
        ax.set_xticks([0.5, 0.8, 0.9, 0.95], ["50%", "80%", "90%", "95%"], rotation=30, ha="right", fontsize=9)
        ax.set_yticks([0.5, 0.7, 0.9, 1], ["50%", "70%", "90%", "100%"])
        ax.grid(alpha=0.18)
        if j >= 2:
            ax.set_xlabel("Nominal posterior interval coverage")
        if j % 2 == 0:
            ax.set_ylabel("Observed coverage")
    fig.suptitle("Heston coverage during baseline development", fontsize=13, y=0.99)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=9)
    fig.text(0.5, 0.015, "Same 500 validation cases; pointwise 95% Wilson intervals. Development evidence, not acceptance.",
             ha="center", fontsize=8, color="#505050")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(root/"heston-coverage.png", dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"evidence/pilot-results/2026-10-26")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    curve_shapes(args.results)
    heston_coverage(args.results)


if __name__ == "__main__":
    main()

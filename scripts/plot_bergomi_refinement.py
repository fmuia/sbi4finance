"""Standalone review figures from the frozen coupled-grid report, without simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/results/bergomi-coupled-refinement.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/bergomi-refinement-figures"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for row, model in enumerate(report["models"]):
        budget = model["absolute_budget_currency"]
        for col, comparison_index in enumerate((3, 2)):
            ax = axes[row, col]
            for case in model["cases"]:
                comparison = case["comparisons"][comparison_index]
                n = len(comparison["difference"])
                x = case["case_index"] * (n + 2) + np.arange(n)
                ax.errorbar(x, comparison["difference"],
                            yerr=4 * np.asarray(comparison["paired_standard_errors"]),
                            fmt=".", ms=4, capsize=1.5, color=colors[case["case_index"]])
            ax.axhspan(-budget, budget, color="#298c55", alpha=.07)
            ax.axhline(0, color=".6", lw=.7)
            for bound in (-budget, budget):
                ax.axhline(bound, color="#b94a48", ls="--", lw=1)
            ax.set_xticks(np.arange(5) * (n + 2) + (n - 1) / 2,
                          ["corner 0", "corner 1", "corner 2", "corner 3", "center"])
            ax.set_title(f"{model['model'].replace('_', ' ')}: "
                         f"{comparison['coarse_steps']} → {comparison['fine_steps']} steps")
            ax.set_ylabel("Fine minus coarse price (currency)")
            ax.set_xlabel("Each group shows nine contracts in saved panel order")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Bergomi refinement: paired differences ± four estimated SEs\n"
                 "Dashed limits: ±0.025 currency; finite-grid evidence, not a continuum-bias bound")
    outputs = []
    for suffix in ("png", "svg"):
        path = args.out / f"coupled-grid-differences.{suffix}"
        fig.savefig(path, dpi=170)
        outputs.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    markers = ("o", "s", "^", "D")
    for ax, model in zip(axes, report["models"]):
        largest = 0
        for index, pair in enumerate(report["protocol"]["comparisons"]):
            independent = np.concatenate([c["comparisons"][index]["independent_standard_errors"] for c in model["cases"]])
            paired = np.concatenate([c["comparisons"][index]["paired_standard_errors"] for c in model["cases"]])
            largest = max(largest, independent.max(), paired.max())
            ax.scatter(independent, paired, s=15, alpha=.7, marker=markers[index], label=f"{pair[0]}→{pair[1]}")
        ax.plot([0, largest * 1.05], [0, largest * 1.05], color=".5", ls="--", label="equal SE")
        ax.set_xlim(0, largest * 1.05)
        ax.set_ylim(bottom=0)
        ax.set_title(model["model"].replace("_", " "))
        ax.set_xlabel("Counterfactual independent-run SE (currency)")
        ax.set_ylabel("Paired difference SE (currency)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Pairing reduces uncertainty in the grid comparison\n"
                 "Independent-run SE uses the same samples' marginal variances; no second run")
    for suffix in ("png", "svg"):
        path = args.out / f"pairing-precision.{suffix}"
        fig.savefig(path, dpi=170)
        outputs.append(path)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"report": str(args.report), "report_sha256": sha(args.report),
                "script_sha256": sha(Path(__file__)),
                "outputs": {p.name: sha(p) for p in outputs}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

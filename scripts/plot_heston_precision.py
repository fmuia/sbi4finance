"""Plot committed Heston precision evidence without simulation or inference.

Uses the first prespecified neural repeat and second refined reference. The
NPZ hash and plotted predictive summaries are verified against their report.
Source hashes and the signed-tail note are embedded in the PNG metadata.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/results/heston-precision"
NEURAL_COLOR = "#c06b48"
REFERENCE_COLOR = "#333333"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=EVIDENCE / "report.json")
    parser.add_argument("--out", type=Path, default=EVIDENCE, help="directory for a new case-0-comparison.png")
    args = parser.parse_args()
    output = args.out / "case-0-comparison.png"
    if output.exists():
        parser.error("case-0-comparison.png exists in --out; choose another directory")
    report = json.loads(args.report.read_text())
    artifact = report["plot_arrays"]
    if (report["split"] != "validation" or report["reserved_test_accessed"] or report["is_sbc"]
            or report["original_gate_passed"] or artifact["file"] != "case-0-draws.npz"
            or artifact["neural_seed"] != 11101 or artifact["reference_filter_seed"] != 27102):
        raise ValueError("expected the frozen post-hoc validation:0 precision evidence")
    array_path = args.report.parent / artifact["file"]
    if file_hash(array_path) != artifact["sha256"]:
        raise ValueError("draw artifact does not match its reported hash")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = dict(archive)
    count = report["draws_per_case_per_repeat"]
    for name in ("neural_joint", "reference_joint"):
        if arrays[name].shape != (count, 2) or not np.isfinite(arrays[name]).all():
            raise ValueError("expected finite matched joint draw pairs")
    for name in ("neural_future_loss", "reference_future_loss"):
        if arrays[name].shape != (count,) or not np.isfinite(arrays[name]).all():
            raise ValueError("expected finite one-dimensional future losses")
    fine = report["case0_fine_references"][1]
    selected = fine["neural_comparisons"][0]
    if (fine["seed"] != artifact["reference_filter_seed"] or selected["neural_seed"] != artifact["neural_seed"]
            or selected["reference_draw_seed"] != artifact["reference_draw_seed"]
            or selected["shared_future_seed"] != artifact["shared_future_seed"]):
        raise ValueError("plot selection does not match the prespecified repeat/reference")
    quantiles = {}
    for name in ("neural", "reference"):
        quantiles[name] = 100 * np.quantile(arrays[f"{name}_future_loss"], report["quantile_order"])
        np.testing.assert_allclose(quantiles[name] / 100, selected[f"{name}_loss_q05_median_q95"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(arrays["neural_joint"].mean(axis=0), selected["neural"]["mean"], rtol=0, atol=1e-12)

    neural = arrays["neural_joint"][:, 1]
    reference = arrays["reference_joint"][:, 1]
    tail_note = (
        f"Signed-state samples: neural {int(np.sum(neural < 0))}/{count} below zero, "
        f"range [{neural.min():.8g}, {neural.max():.8g}]; "
        f"reference {int(np.sum(reference < 0))}/{count} below zero, "
        f"range [{reference.min():.8g}, {reference.max():.8g}]. "
        "Negative signed auxiliary states are allowed by this discrete model; these are sample ranges, not support bounds."
    )
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.7), gridspec_kw={"width_ratios": [1.05, 1]})
    bins = np.linspace(min(neural.min(), reference.min()), max(neural.max(), reference.max()), 190)
    for values, label, color, mean in (
        (reference, "Refined reference", REFERENCE_COLOR, selected["reference"]["mean"][1]),
        (neural, "Frozen neural", NEURAL_COLOR, selected["neural"]["mean"][1]),
    ):
        density, edges = np.histogram(values, bins=bins, density=True)
        left.stairs(density, edges, color=color, linewidth=1.6, label=label)
        left.axvline(mean, color=color, linestyle="--", linewidth=1.1)
    left.axvline(0, color="#aaaaaa", linewidth=0.7)
    left.set(title="Signed terminal state: central density", xlabel="Auxiliary variance (annualized variance units)",
             ylabel="Probability density", xlim=(-0.001, 0.027))
    left.legend(loc="upper right", bbox_to_anchor=(0.99, 0.54), fontsize=9)
    left.text(0.98, 0.38, "Dashed lines: means\n"
              f"State-mean shift: +{selected['mean_error_in_reference_sd'][1]:.3f} reference SD\n"
              f"Neural state SD: +{100 * selected['relative_std_error'][1]:.1f}%",
              transform=left.transAxes, ha="right", va="top", fontsize=9)
    # The main density is zoomed; this inset preserves the entire sampled range.
    inset = left.inset_axes([0.60, 0.63, 0.37, 0.23])
    for values, y, color in ((neural, 1, NEURAL_COLOR), (reference, 0, REFERENCE_COLOR)):
        inset.plot([values.min(), values.max()], [y, y], color=color, linewidth=1.5, marker="|")
    inset.axvline(0, color="#aaaaaa", linewidth=0.7)
    inset.set(title="Full sample ranges", yticks=[0, 1], yticklabels=["Ref.", "Neural"],
              xticks=[-0.03, 0, 0.03], ylim=(-0.6, 1.6), xlim=(-0.031, 0.038))
    inset.tick_params(labelsize=7)
    inset.title.set_fontsize(8)

    for name, y, label, color in (("neural", 1, "Frozen neural", NEURAL_COLOR),
                                 ("reference", 0, "Refined reference", REFERENCE_COLOR)):
        q05, median, q95 = quantiles[name]
        right.plot([q05, q95], [y, y], color=color, linewidth=3, marker="|")
        right.plot(median, y, marker="o", color=color, markersize=7)
        for q, qlabel in ((q05, "5%"), (median, "50%"), (q95, "95%")):
            right.text(q, y + 0.17, f"{qlabel}: {q:.2f}%", color=color, ha="center", fontsize=9)
    right.axvline(0, color="#bbbbbb", linestyle=":", linewidth=1)
    right.set(title="Ten-day future loss: central 90% interval", xlabel="Relative loss (% of initial wealth)",
              yticks=[0, 1], yticklabels=["Refined reference", "Frozen neural"],
              xlim=(-3.7, 3.7), ylim=(-0.55, 1.70))
    right.text(0.5, 0.04, "Line: 5%–95% predictive interval; dot: median\n"
               f"Neural loss SD: +{100 * selected['predictive_std_relative_error']:.1f}%\n"
               "Negative loss denotes a gain.", transform=right.transAxes, ha="center", fontsize=9)
    fig.suptitle("State inference and future losses: the remaining discrepancy", fontsize=14, y=0.965)
    fig.text(0.5, 0.885, "validation:0 • Frozen bounded-time spline • 16,384 draws per method • Refined reference: 96 grid × 16,384 particles",
             ha="center", fontsize=9)
    fig.text(0.5, 0.063, "DEVELOPMENT DIAGNOSTIC — NOT SBC OR A NEW HOLDOUT. Original acceptance remains failed.\n"
             f"State density is zoomed; inset shows all sample extrema. Signed negatives: neural {np.sum(neural < 0)}/{count:,}, reference {np.sum(reference < 0)}/{count:,}.\n"
             "Displayed loss intervals describe future outcomes under the fitted model, not Monte Carlo confidence intervals.",
             ha="center", va="center", fontsize=8.5)
    fig.subplots_adjust(top=0.79, bottom=0.24, left=0.075, right=0.985, wspace=0.30)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"ReportSHA256": file_hash(args.report), "ArraySHA256": file_hash(array_path),
                "PlotScriptSHA256": file_hash(Path(__file__)), "SignedTailNote": tail_note,
                "Selection": json.dumps(artifact), "Scope": "Post-hoc validation diagnostic; no simulation or inference rerun"}
    fig.savefig(output, dpi=170, metadata=metadata)
    plt.close(fig)
    print(json.dumps({"out": str(output), "sha256": file_hash(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()

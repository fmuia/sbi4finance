"""Render new display-only recovery PNGs; preserve the original study byte for byte.

Run from lab/ with --archive pointing to the extracted private
heston-summary-dimensions-20261114 archive. Only compact figures/provenance are
written. Raw truths/draws stay private; no model loading or inference is needed.
"""

import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np


LAB = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def display_labels(notebook):
    for node in ast.walk(ast.parse(Path(notebook).read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MATH_LABELS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("the notebook's MATH_LABELS are required")


def export(archive, source_root, output, notebook):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    archive, source_root, output = Path(archive), Path(source_root), Path(output)
    record = json.loads((source_root / "results.json").read_text())
    source_manifest = archive / "evidence/recovery/manifest.json"
    if digest(source_manifest) != record["producer"]["recovery_manifest_sha256"]:
        raise ValueError("the original recovery manifest differs")
    exporter = archive / "source-snapshot/lab/scripts/export_heston_summary_recovery.py"
    if digest(exporter) != json.loads(source_manifest.read_text())["exporter_sha256"]:
        raise ValueError("the preserved recovery exporter differs")
    spec = importlib.util.spec_from_file_location("_preserved_summary_exporter", exporter)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    baseline = archive / "baseline/direct-final"
    results = original.load_inputs(archive / "evidence", archive / "runs", baseline / "evaluation.json",
                                   baseline / "population-draws.npz", baseline / "recovery.npz")
    labels = display_labels(notebook)
    fits = {fit["dimension"]: fit for fit in record["fits"]}
    for item in results:
        fit = fits[item["dimension"]]
        if (item["source_draws_sha256"] != fit["recovery_figure"]["source_draws_sha256"]
                or item["truth_sha256"] != record["identities"]["truth_sha256"]
                or item["observations_sha256"] != record["identities"]["observations_sha256"]
                or any(item["metrics"][key] != fit["endpoints"]["primary_final"]["population"][key]
                       for key in ("rmse", "coverage90", "median_width90"))):
            raise ValueError("raw sources or reproduced metrics differ from the lesson")
    lower = np.minimum(original.BOUNDS[:, 0], np.min([item["quantiles"][0].min(axis=0) for item in results], axis=0))
    upper = np.maximum(original.BOUNDS[:, 1], np.max([item["quantiles"][2].max(axis=0) for item in results], axis=0))
    margin = .04 * (upper - lower)
    limits = np.column_stack((lower - margin, upper + margin))
    if not np.array_equal(limits, json.loads(source_manifest.read_text())["shared_axis_limits"]):
        raise ValueError("reproduced shared axes differ from the original figures")
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    for item in results:
        dimension = item["dimension"]
        title = "Direct 63-input model" if dimension is None else f"MLP summary: d = {dimension}"
        color = original.COLORS[dimension]
        figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.3), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            truth = item["truth"][:, j]
            lower, median, upper = item["quantiles"][:, :, j]
            ax.vlines(truth, lower, upper, color=color, alpha=.14, linewidth=.7, zorder=1)
            ax.scatter(truth, median, color=color, alpha=.6, s=9, edgecolors="none", zorder=3)
            ax.plot(limits[j], limits[j], linestyle="--", color="#374151", linewidth=1, zorder=2)
            for bound in original.BOUNDS[j]:
                ax.axhline(bound, color="#acb5bd", linestyle=":", linewidth=.7)
                ax.axvline(bound, color="#acb5bd", linestyle=":", linewidth=.7)
            ax.set(xlim=limits[j], ylim=limits[j], xlabel="Simulated truth",
                   ylabel="Posterior median and 90% interval", title=labels[j])
            ax.set_aspect("equal", adjustable="box")
            ax.text(.035, .96, f"RMSE {item['metrics']['rmse'][j]:.3g}\n90% coverage {item['metrics']['coverage90'][j]:.1%}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 3})
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
        figure.suptitle(f"{title} — parameter recovery at final epoch 300", fontsize=14)
        figure.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=color, label="Posterior median"),
                               Line2D([], [], color=color, alpha=.45, label="Central 90% posterior interval"),
                               Line2D([], [], color="#374151", linestyle="--", label="Perfect point recovery"),
                               Line2D([], [], color="#acb5bd", linestyle=":", label="Prior bounds")],
                      loc="outside lower center", ncol=4, frameon=False, fontsize=9)
        filename = "direct-final.png" if dimension is None else f"mlp{dimension}-final.png"
        path = output / filename
        figure.savefig(path, dpi=150, metadata={"Software": "Heston summary notation exporter"})
        plt.close(figure)
        figures.append({"dimension": dimension, "file": filename, "sha256": digest(path),
                        "source_archived_image_sha256": fits[dimension]["recovery_figure"]["sha256"],
                        **{key: item[key] for key in ("source_draws_sha256", "truth_sha256", "observations_sha256", "metrics")}})
    manifest = {"format": "heston_summary_notation_figures_v1", "display_only": True,
                "labels": labels, "parameter_names": record["protocol"]["parameter_names"],
                "rows": 512, "draws_per_case": 512, "shared_axis_limits": limits.tolist(),
                "source_results_sha256": digest(source_root / "results.json"),
                "source_recovery_manifest_sha256": digest(source_manifest), "source_exporter_sha256": digest(exporter),
                "exporter_sha256": digest(__file__), "figures": figures,
                "scope": "Only display labels changed. All 512 truths, raw-draw quantiles, metrics and common axes are preserved. Original images and study identities remain unchanged."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=LAB / "notebooks/data/heston_summary_dimensions")
    parser.add_argument("--output", type=Path, default=LAB / "notebooks/data/heston_summary_aligned_notation")
    parser.add_argument("--notebook", type=Path, default=LAB / "notebooks/heston_spx_calibration.py")
    args = parser.parse_args()
    result = export(args.archive, args.source_root, args.output, args.notebook)
    print(json.dumps({"figures": len(result["figures"]), "display_only": True}))

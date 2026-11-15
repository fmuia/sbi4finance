"""Update only the panel titles of the notation-aligned summary recovery PNGs.

``export_heston_summary_notation.py`` renders these figures from the private
experiment archive. When only the notebook's display labels change, this tool
updates the committed figures without that archive. It rebuilds the exporter's
figure layout without any data, at matplotlib's stock defaults, once with the
manifest's labels and once with the notebook's ``MATH_LABELS``. The two layouts
must place every axis identically, and the old-label rebuild must reproduce the
committed pixels exactly wherever the two rebuilds differ. Only those pixels are
replaced; all other pixels, including every plotted truth, median and interval,
are copied unchanged. The manifest records the previous labels and images.

Run from lab/ after changing ``MATH_LABELS``:

    python scripts/relabel_heston_summary_notation.py
"""

import argparse
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from scripts.export_heston_summary_notation import display_labels  # noqa: E402


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _loader():
    spec = importlib.util.spec_from_file_location("_heston_summary_notation_for_relabel",
                                                  LAB / "notebooks/_heston_summary_notation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(plt, labels, title, metrics, limits, bounds):
    """The exporter's figure without its data: the same axes, limits, text and legend."""
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.3), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(limits[j], limits[j], linestyle="--", color="#374151", linewidth=1, zorder=2)
        for bound in bounds[j]:
            ax.axhline(bound, color="#acb5bd", linestyle=":", linewidth=.7)
            ax.axvline(bound, color="#acb5bd", linestyle=":", linewidth=.7)
        ax.set(xlim=limits[j], ylim=limits[j], xlabel="Simulated truth",
               ylabel="Posterior median and 90% interval", title=labels[j])
        ax.set_aspect("equal", adjustable="box")
        ax.text(.035, .96, f"RMSE {metrics['rmse'][j]:.3g}\n90% coverage {metrics['coverage90'][j]:.1%}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 3})
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    figure.suptitle(f"{title} — parameter recovery at final epoch 300", fontsize=14)
    figure.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="#000000", label="Posterior median"),
                           Line2D([], [], color="#000000", alpha=.45, label="Central 90% posterior interval"),
                           Line2D([], [], color="#374151", linestyle="--", label="Perfect point recovery"),
                           Line2D([], [], color="#acb5bd", linestyle=":", label="Prior bounds")],
                  loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    buffer = io.BytesIO()
    figure.savefig(buffer, dpi=150, format="png")
    positions = [ax.get_position().bounds for ax in axes.flat]
    plt.close(figure)
    return buffer.getvalue(), positions


def relabel(data, source_root, notebook):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()  # the stock style the exporter ran with, whatever the local matplotlibrc says
    from matplotlib import ft2font
    import matplotlib.pyplot as plt
    from PIL import Image, PngImagePlugin

    data, source_root = Path(data), Path(source_root)
    record = json.loads((source_root / "results.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    old, new = manifest["labels"], display_labels(notebook)
    if new == old:
        return manifest
    _loader().load_figures(data, source_root, record, old)  # the committed figures bind to the frozen study
    limits = np.asarray(manifest["shared_axis_limits"], dtype=float)
    bounds = np.asarray(record["protocol"]["prior_bounds"], dtype=float)
    previous, changed, updates = {}, {}, []
    for item in manifest["figures"]:  # verify every figure before any file is written
        dimension, path = item["dimension"], data / item["file"]
        title = "Direct 63-input model" if dimension is None else f"MLP summary: d = {dimension}"
        rendered = {}
        for name, labels in (("old", old), ("new", new)):
            png, positions = _layout(plt, labels, title, item["metrics"], limits, bounds)
            rendered[name] = (np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")), positions)
        (before, before_axes), (after, after_axes) = rendered["old"], rendered["new"]
        if not np.allclose(before_axes, after_axes, rtol=0, atol=1e-12):
            raise ValueError(f"{item['file']}: the new labels move the axes; rerun the full exporter")
        image = np.asarray(Image.open(path).convert("RGBA"))
        if image.shape != before.shape:
            raise ValueError(f"{item['file']}: the committed image and the rebuilt layout differ in size")
        mask = np.any(before != after, axis=-1)
        if not mask.any():
            raise ValueError(f"{item['file']}: the label change leaves no visible difference")
        rows, columns = np.nonzero(mask)
        box = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))
        if not np.array_equal(image[box], before[box]):
            raise ValueError(f"{item['file']}: the committed labels are not reproduced exactly; "
                             "rerun the full exporter")
        updated = image.copy()
        updated[mask] = after[mask]
        updates.append((item, path, updated))
        changed[item["file"]] = {"rows": [int(rows.min()), int(rows.max())],
                                 "columns": [int(columns.min()), int(columns.max())],
                                 "pixels": int(mask.sum())}
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "Heston summary notation relabeller")
    for item, path, updated in updates:
        previous[item["file"]] = item["sha256"]
        Image.fromarray(updated, "RGBA").save(path, dpi=(150, 150), pnginfo=info)
        item["sha256"] = digest(path)
    manifest["labels"] = new
    manifest.setdefault("relabelled", []).append({
        "tool_sha256": digest(__file__), "previous_labels": old, "previous_figure_sha256": previous,
        "changed_pixels": changed, "matplotlib": matplotlib.__version__,
        "freetype": ft2font.__freetype_version__,
        "scope": "Only panel-title pixels changed. A data-free rebuild of the exporter layout with the "
                 "previous labels reproduced each replaced region exactly, and both label sets give "
                 "identical axes. Every plotted value, axis and metric is unchanged."})
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _loader().load_figures(data, source_root, record, new)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=LAB / "notebooks/data/heston_summary_aligned_notation")
    parser.add_argument("--source-root", type=Path, default=LAB / "notebooks/data/heston_summary_dimensions")
    parser.add_argument("--notebook", type=Path, default=LAB / "notebooks/heston_spx_calibration.py")
    args = parser.parse_args()
    result = relabel(args.data, args.source_root, args.notebook)
    print(json.dumps({"labels": result["labels"], "figures": len(result["figures"])}, ensure_ascii=False))

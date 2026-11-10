"""Export scientific Matplotlib figures from saved PR87 diagnostics only.

This post-run presentation consumer reads hash-bound recovery quantiles, SBC
ranks and the native epoch history. It never loads a trained model, prices an
option, queries a posterior or opens native MCMC chains. The frozen compact
HTML renderer remains unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_heston_pr87_reproduction as saved

binding, read = saved.binding, saved.read
BLUE, RED, GRAY = saved.BLUE, saved.RED, saved.GRAY
LABELS = (r"Initial variance $v_0$", r"Mean reversion $\kappa$", r"Long-run variance $\theta$",
    r"Volatility $\sigma_v$", r"Correlation $\rho$", r"Noise scale $c$")


def figure_files(fig, out, name, title, description):
    files = {}
    for extension in ("png", "svg"):
        path = out / (name + "." + extension)
        fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
        files[extension] = {**binding(path), "bytes": path.stat().st_size}
    plt.close(fig)
    return {"title": title, "description": description, "files": files}


def recovery_figure(recovery, target, *, title, interval_rows):
    """Show every saved median; include every displayed interval in axis limits."""
    truth, q = recovery
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.1), layout="constrained")
    chosen = np.asarray(interval_rows, dtype=int)
    extents = []
    for j, axis in enumerate(axes.flat):
        lo, hi = target["prior_bounds"][j]
        lower = min(lo, float(q[1, :, j].min()), float(q[0, chosen, j].min()))
        upper = max(hi, float(q[1, :, j].max()), float(q[2, chosen, j].max()))
        margin = max(.025 * (upper - lower), np.finfo(float).eps)
        axis.plot([lo, hi], [lo, hi], "--", color=GRAY, lw=1, label="Identity")
        axis.scatter(truth[:, j], q[1, :, j], s=10, color=BLUE, alpha=.45, edgecolors="none", label="Every posterior median")
        axis.errorbar(truth[chosen, j], q[1, chosen, j],
            yerr=np.stack([q[1, chosen, j] - q[0, chosen, j], q[2, chosen, j] - q[1, chosen, j]]),
            fmt="none", color=BLUE, capsize=2, elinewidth=.7, label="Selected central 90% intervals")
        axis.set(title=LABELS[j], xlabel="Generating value", ylabel="Posterior median and central 90% interval",
            xlim=(lo - .025 * (hi - lo), hi + .025 * (hi - lo)), ylim=(lower - margin, upper + margin))
        axis.grid(alpha=.15)
        extents.append({"target": target["parameter_names"][j], "plotted_y_limits": list(axis.get_ylim()),
            "minimum_displayed_interval": float(q[0, chosen, j].min()),
            "maximum_displayed_interval": float(q[2, chosen, j].max()),
            "minimum_all_medians": float(q[1, :, j].min()), "maximum_all_medians": float(q[1, :, j].max())})
    fig.suptitle(title, fontsize=14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    return fig, {"all_rows_plotted": len(truth), "interval_zero_based_rows": chosen.tolist(),
        "quantiles": [.05, .5, .95], "raw_draws_clipped_or_filtered": False, "axis_extents": extents}


def sbc_figure(summary):
    ranks = np.asarray(summary["fractional_ranks"], dtype=float)
    if ranks.shape != (summary["rows"], 6) or not np.isfinite(ranks).all():
        raise ValueError("saved finite six-target SBC ranks required")
    grid = np.linspace(0, 1, 201)
    difference = np.stack([np.searchsorted(np.sort(ranks[:, j]), grid, side="right") / len(ranks) - grid for j in range(6)], axis=1)
    bound = summary["rank_dkw"]["dkw_bound"]
    extent = max(.12, bound * 1.2, float(np.abs(difference).max()) * 1.08)
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.5), layout="constrained")
    for j, axis in enumerate(axes.flat):
        axis.axhspan(-bound, bound, color="#e1eee7", label="Six-marginal DKW band")
        axis.axhline(0, color=GRAY, lw=.8)
        axis.plot(grid, difference[:, j], color=BLUE, lw=1.5, label="Raw posterior rank ECDF − uniform CDF")
        axis.set(title=LABELS[j], xlabel="Fractional posterior rank", ylabel="ECDF − uniform CDF", xlim=(0, 1), ylim=(-extent, extent))
        axis.grid(alpha=.15)
    fig.suptitle("Historical 512-case full-prior panel: all six marginal SBC screens", fontsize=14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    return fig


def learning_figure(curve):
    if len(curve) != 200 or [point["epoch"] for point in curve] != list(range(1, 201)):
        raise ValueError("complete 200-epoch native history required")
    fig, axis = plt.subplots(figsize=(10.7, 4), layout="constrained")
    for key, color, label in (("loss", BLUE, "Training"), ("val_loss", RED, "Validation")):
        values = np.asarray([point[key] for point in curve], dtype=float)
        if not np.isfinite(values).all(): raise ValueError("finite native objective history required")
        axis.plot(range(1, 201), values, color=color, label=label, lw=1.4)
    axis.set(xlabel="Native fit_offline epoch over the same fresh 10,000 noisy datasets",
        ylabel="FlowMatching objective", title="Native training trajectory: final epoch 200 is the primary endpoint")
    axis.grid(alpha=.2); axis.legend(frameon=False)
    return fig


def build(protocol_path, evaluation_path, out, *, native_recovery=True):
    protocol_path, evaluation_path, out = map(lambda value: Path(value).resolve(), (protocol_path, evaluation_path, out))
    if out.exists(): raise FileExistsError("scientific figure export identity must be new")
    protected = {}
    protocol = saved.checked(binding(protocol_path), protected)
    record = saved.checked(binding(evaluation_path), protected)
    if (record.get("format") != "heston_pr87_reproduction_evaluation_v1" or record.get("status") != "complete"
            or record.get("native_epoch") != 200 or record.get("protocol") != binding(protocol_path)
            or record.get("task") != protocol["task"]["target"] or record.get("protected_inputs_unchanged") is not True
            or record.get("new_reference_sampling") is not False or record.get("reserved_arrays_opened") is not False):
        raise ValueError("complete immutable final-epoch PR87 diagnostics required")
    historical = saved.load_recovery(record["population"], protected)
    native = saved.load_recovery(record["native_validation"], protected) if native_recovery else None
    if historical is None or (native_recovery and native is None):
        raise ValueError("finite unfiltered saved raw-draw recovery quantiles required")
    training = Path(record["checkpoint"]["path"]).parent.parent.parent
    training_record = saved.checked(binding(training / "run.json"), protected)
    curve_path = training / "native-epoch-history.json"
    curve_binding = binding(curve_path)
    if training_record.get("native_epoch_history_sha256") != curve_binding["sha256"]:
        raise ValueError("native epoch history differs from the completed training record")
    curve = saved.checked(curve_binding, protected)
    out.mkdir(parents=True)
    exports = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
            "axes.spines.right": False, "svg.fonttype": "none", "savefig.transparent": False}):
        exports["native-learning-curves"] = figure_files(learning_figure(curve), out, "native-learning-curves",
            "Native training and validation objective",
            "FlowMatching objective, not density NLL; absolute magnitudes cannot be compared with historical spline losses.")
        fig, details = recovery_figure(historical, record["task"],
            title="Historical 512-case panel: all posterior medians; 32 fixed central 90% intervals",
            interval_rows=np.arange(0, 512, 16))
        exports["historical-recovery"] = {**figure_files(fig, out, "historical-recovery",
            "Historical full-prior six-target recovery", "All 512 medians and fixed-index 32 intervals; displayed raw out-of-prior results expand the y axis."), **details}
        exports["historical-sbc"] = figure_files(sbc_figure(record["population"]), out, "historical-sbc",
            "Historical full-prior marginal SBC", "Simultaneous DKW band over six marginal targets within this panel; descriptive reused development data, not independent acceptance.")
        if native_recovery:
            fig, details = recovery_figure(native, record["task"], title="Native notebook validation: 100 datasets × 300 draws; all medians",
                interval_rows=np.arange(0, 100, 3))
            exports["native-recovery"] = {**figure_files(fig, out, "native-recovery",
                "Native 100-case notebook recovery", "All 100 saved medians; 34 fixed-index central 90% intervals. The fitted validation split is development evidence."), **details}
    if any(saved.evaluation.fixed.digest(path) != sha for path, sha in protected.items()):
        raise ValueError("protected saved diagnostic input changed during figure export")
    manifest = {"format": "heston_pr87_reproduction_scientific_figures_v1", "protocol": binding(protocol_path),
        "evaluation": binding(evaluation_path), "native_history": curve_binding,
        "producer": binding(__file__), "presentation_helper": binding(saved.__file__),
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__}, "exports": exports,
        "protected_inputs": [{"path": path, "sha256": sha} for path, sha in sorted(protected.items())],
        "protected_inputs_unchanged": True, "model_loaded": False, "new_neural_sampling": False,
        "new_pricing": False, "new_reference_sampling": False, "scientific_acceptance": False,
        "scope": "Post-run scientific plotting from saved raw-draw summaries; independent of the pre-fit frozen compact HTML presentation."}
    saved.evaluation.fixed.write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "evaluation", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--skip-native-recovery", action="store_true")
    args = parser.parse_args()
    result = build(args.protocol, args.evaluation, args.out, native_recovery=not args.skip_native_recovery)
    print(json.dumps({"exports": len(result["exports"]), "manifest": str(args.out / "manifest.json")}))

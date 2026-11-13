"""Render hash-bound final-checkpoint recovery plots without loading a model.

Only PNG figures and portable provenance enter the evidence directory. Raw neural
samples and simulated truths stay in their original private payloads. All sources
are verified against the already-exported compact study before any plot is saved.
Run from lab/ with --evidence, --run-root and --baseline-evaluation; optional
--baseline-draws/--baseline-recovery overrides support relocated private payloads.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c")
LABELS = ("Initial variance $v_0$", "Mean reversion $\\kappa$", "Long-run variance $\\theta$",
          "Volatility of variance $\\eta$", "Correlation $\\rho$", "Quote-noise multiplier $c$")
BOUNDS = np.array(((.01, .16), (.1, 10.), (.01, .16), (.1, 3.), (-.95, -.1), (.5, 2.)))
COLORS = {None: "#4c5964", 64: "#84613f", 32: "#8660b2", 16: "#38946b",
          8: "#2976ac", 4: "#d17b28", 2: "#c54f52"}


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_file(path, expected):
    path = Path(path)
    if digest(path) != expected:
        raise ValueError("source checksum differs: " + path.name)
    return path


def summarize_draws(draws, truth, compact):
    """Reproduce plotted quantiles and every displayed recovery metric from raw draws."""
    population = compact["population"]
    if (draws.shape != (512, 512, 6) or truth.shape != (512, 6)
            or not np.isfinite(draws).all() or not np.isfinite(truth).all()
            or tuple(population["parameter_names"]) != NAMES
            or population["rows"] != 512 or population["draws_per_case"] != 512
            or population["raw_draws_used"] != 262144
            or sha256(np.asarray(truth, dtype="<f8").tobytes()).hexdigest() != population["truth_sha256"]):
        raise ValueError("complete matched raw population and truth identity required")
    support = population["support"]
    if (any(support.get(key) != 0 for key in ("draws_discarded", "draws_clipped", "draws_resampled"))
            or support.get("outside_prior_draws_retained_for_calibration") is not True):
        raise ValueError("all raw draws must be retained without support filtering")
    outside = ((draws < BOUNDS[:, 0]) | (draws > BOUNDS[:, 1])).sum(axis=(0, 1))
    if not np.array_equal(outside, support["outside_prior"]["per_dimension_count"]):
        raise ValueError("raw support counts differ")
    quantiles = np.quantile(draws, [.05, .5, .95], axis=1)
    lower, median, upper = quantiles
    metrics = {"rmse": np.sqrt(np.mean((median - truth) ** 2, axis=0)),
               "coverage90": np.mean((truth >= lower) & (truth <= upper), axis=0),
               "median_width90": np.median(upper - lower, axis=0)}
    for key, actual in metrics.items():
        expected = [compact["parameters"][name][key] for name in NAMES]
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-14):
            raise ValueError("saved recovery metric differs: " + key)
    return quantiles, {key: value.tolist() for key, value in metrics.items()}


def load_inputs(evidence, run_root, baseline_evaluation, baseline_draws=None, baseline_recovery=None):
    evidence, run_root = Path(evidence), Path(run_root)
    study = read(evidence / "study.json")
    entries = [entry for entry in study["runs"] if entry["endpoint"] == "primary_final"]
    if (study.get("format") != "heston_flexible_study_v1"
            or len(entries) != 7 or {entry["summary_dim"] for entry in entries} != set(COLORS)):
        raise ValueError("all seven final endpoints are required")
    results = []
    for entry in sorted(entries, key=lambda e: (e["summary_dim"] is not None, -(e["summary_dim"] or 0))):
        directory = (evidence / entry["directory"]).resolve()
        if evidence.resolve() not in directory.parents:
            raise ValueError("study directory escapes evidence")
        compact = read(directory / "evaluation.json")
        run = read(directory / "run.json")
        if (compact["endpoint"] != "primary_final" or compact["epoch"] != 300
                or compact["status"] != "complete" or run["status"] != "complete"
                or run["train_rows"] != 300000 or run["completed_epochs"] != 300):
            raise ValueError("matched completed final-300 runs are required")
        dimension = entry["summary_dim"]
        if dimension is None:
            source_path = checked_file(baseline_evaluation, compact["source_evaluation_sha256"])
            source = read(source_path)
            if source.get("endpoint") != "primary_final" or source.get("native_epoch") != 300:
                raise ValueError("historical baseline final endpoint differs")
            binding = source["population"]["raw_draw_payload"]
            draws_path = checked_file(baseline_draws or binding["path"], binding["sha256"])
            binding = source["population"]["recovery_payload"]
            truth_path = checked_file(baseline_recovery or binding["path"], binding["sha256"])
            with np.load(draws_path, allow_pickle=False) as archive:
                draws = archive["draws"].copy()
            with np.load(truth_path, allow_pickle=False) as archive:
                truth = archive["truth"].copy()
            extra = {"source_recovery_sha256": digest(truth_path)}
        else:
            source_directory = run_root / directory.parent.name
            source_path = checked_file(source_directory / "evaluation.json", compact["source_evaluation_sha256"])
            source = read(source_path)
            if source.get("endpoint") != "final" or source.get("status") != "complete":
                raise ValueError("MLP final evaluation differs")
            draws_path = checked_file(source_directory / "draws.npz", source["draws_sha256"])
            if digest(draws_path) != run["source_hashes"]["draws_sha256"]:
                raise ValueError("MLP compact raw-draw binding differs")
            with np.load(draws_path, allow_pickle=False) as archive:
                draws, truth = archive["population"].copy(), archive["truth"].copy()
            extra = {}
        quantiles, metrics = summarize_draws(draws, truth, compact)
        if results and (not np.array_equal(truth, results[0]["truth"])
                        or compact["population"]["observations_sha256"] != results[0]["observations_sha256"]
                        or run["data_payload_sha256"] != results[0]["data_payload_sha256"]):
            raise ValueError("all models must share truths, observations and the training bank")
        results.append({"dimension": dimension, "endpoint": "primary_final", "epoch": 300,
                        "truth": truth, "quantiles": quantiles, "metrics": metrics,
                        "source_evaluation_sha256": digest(source_path), "source_draws_sha256": digest(draws_path),
                        "compact_evaluation_sha256": digest(directory / "evaluation.json"),
                        "truth_sha256": compact["population"]["truth_sha256"],
                        "observations_sha256": compact["population"]["observations_sha256"],
                        "data_payload_sha256": run["data_payload_sha256"], **extra})
    return results


def export(evidence, run_root, baseline_evaluation, *, baseline_draws=None, baseline_recovery=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results = load_inputs(evidence, run_root, baseline_evaluation, baseline_draws, baseline_recovery)
    lower = np.minimum(BOUNDS[:, 0], np.min([item["quantiles"][0].min(axis=0) for item in results], axis=0))
    upper = np.maximum(BOUNDS[:, 1], np.max([item["quantiles"][2].max(axis=0) for item in results], axis=0))
    margin = .04 * (upper - lower)
    limits = np.column_stack((lower - margin, upper + margin))
    out = Path(evidence) / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for item in results:
        dimension = item["dimension"]
        label = "Direct 63-input model" if dimension is None else f"MLP summary: d = {dimension}"
        color = COLORS[dimension]
        figure, axes = plt.subplots(2, 3, figsize=(12.5, 8.3), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            truth = item["truth"][:, j]
            lower, median, upper = item["quantiles"][:, :, j]
            ax.vlines(truth, lower, upper, color=color, alpha=.14, linewidth=.7, zorder=1)
            ax.scatter(truth, median, color=color, alpha=.6, s=9, edgecolors="none", zorder=3)
            ax.plot(limits[j], limits[j], linestyle="--", color="#374151", linewidth=1, zorder=2)
            for bound in BOUNDS[j]:
                ax.axhline(bound, color="#acb5bd", linestyle=":", linewidth=.7)
                ax.axvline(bound, color="#acb5bd", linestyle=":", linewidth=.7)
            ax.set(xlim=limits[j], ylim=limits[j], xlabel="Simulated truth", ylabel="Posterior median and 90% interval",
                   title=LABELS[j])
            ax.set_aspect("equal", adjustable="box")
            ax.text(.035, .96, f"RMSE {item['metrics']['rmse'][j]:.3g}\n90% coverage {item['metrics']['coverage90'][j]:.1%}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 3})
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
        figure.suptitle(f"{label} — parameter recovery at final epoch 300", fontsize=14)
        figure.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=color, label="Posterior median"),
                               Line2D([], [], color=color, alpha=.45, label="Central 90% posterior interval"),
                               Line2D([], [], color="#374151", linestyle="--", label="Perfect point recovery"),
                               Line2D([], [], color="#acb5bd", linestyle=":", label="Prior bounds")],
                      loc="outside lower center", ncol=4, frameon=False, fontsize=9)
        filename = "direct-final.png" if dimension is None else f"mlp{dimension}-final.png"
        path = out / filename
        figure.savefig(path, dpi=150, metadata={"Software": "Heston summary recovery exporter"})
        plt.close(figure)
        portable = {key: value for key, value in item.items() if key not in ("truth", "quantiles")}
        portable.update(figure=str(path.relative_to(evidence)), figure_sha256=digest(path), metrics_reproduced=True,
                        rows=512, draws_per_case=512, intervals_plotted=512)
        records.append(portable)
    manifest = {"format": "heston_summary_recovery_figures_v1", "parameter_names": list(NAMES),
                "endpoint": "primary_final", "epoch": 300, "rows": 512, "draws_per_case": 512,
                "source_study_sha256": digest(Path(evidence) / "study.json"),
                "exporter_sha256": digest(__file__), "shared_axis_limits": limits.tolist(),
                "prior_bounds": BOUNDS.tolist(), "figures": records, "scientific_acceptance": False,
                "plot_policy": "All 512 truths, posterior medians and central-90% intervals; common axes per parameter across seven models; raw draws retained without clipping or rejection.",
                "scope": "Reused development cases; figures reveal recovery and information retention, not a new acceptance decision."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--baseline-draws", type=Path)
    parser.add_argument("--baseline-recovery", type=Path)
    args = parser.parse_args()
    manifest = export(args.evidence, args.run_root, args.baseline_evaluation,
                      baseline_draws=args.baseline_draws, baseline_recovery=args.baseline_recovery)
    print(json.dumps({"figures": len(manifest["figures"]), "rows_per_figure": manifest["rows"], "all_metrics_reproduced": True}))


if __name__ == "__main__":
    main()

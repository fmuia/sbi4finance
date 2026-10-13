"""Plot the saved fixed-contract HKDE surrogate experiment; no pricing or fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


COLORS = {"surrogate": "#087F8C", "direct": "#D97732", "gate": "#B33E43", "gray": "#738095"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cache(directory):
    """Read available cache; verify associated arrays before plotting results."""
    directory = Path(directory)
    report_path = directory / "report.json"
    if not report_path.exists():
        return None, {}, {}
    report = json.loads(report_path.read_text())
    paths = {"report.json": report_path}
    for name in ("checks.npz", "model.npz"):
        path = directory / name
        if path.exists():
            if report["artifact_sha256"].get(name) != sha256(path):
                raise ValueError(f"surrogate report/{name} hash mismatch")
            paths[name] = path
    checks = {}
    if "checks.npz" in paths:
        with np.load(paths["checks.npz"], allow_pickle=False) as saved:
            checks = dict(saved)
    return report, checks, paths


def _finish(figure, title, note):
    figure.suptitle(title, fontsize=13)
    figure.text(.5, .02, note, ha="center", va="bottom", fontsize=8.5)
    figure.tight_layout(rect=(0, .12, 1, .92))
    return figure


def parameter_errors(report, checks):
    shape = tuple(int(value) for value in checks["grid_shape"])
    parameters = np.asarray(checks["grid_parameters"])
    errors = np.asarray(checks["grid_error"])
    if parameters.shape != (int(np.prod(shape)), 2) or errors.shape != parameters.shape:
        raise ValueError("inconsistent parameter/error grid shapes")
    axes = [np.unique(parameters[:, dimension]) for dimension in range(2)]
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    if shape != tuple(map(len, axes)) or not np.array_equal(parameters, expected):
        raise ValueError("parameter grid must retain declared C-order")
    maximum_error = float(np.max(np.abs(errors)))
    limit = max(maximum_error, np.finfo(float).eps)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    figure, panels = plt.subplots(1, 2, figsize=(11.4, 5.1), layout="constrained")
    reshaped = errors.reshape(*shape, 2)
    for index, axis in enumerate(panels):
        image = axis.pcolormesh(axes[0], axes[1], reshaped[:, :, index].T,
                               shading="nearest", cmap="coolwarm", norm=norm, rasterized=True)
        contract = report["targets"][index]
        axis.set(title=f"Put K={contract['strike']:g}, T={contract['maturity']:g} years",
                 xlabel=r"Long-run variance $\theta$", ylabel=r"Jump intensity $\lambda$ (per year)")
    figure.colorbar(image, ax=panels, label="Surrogate − Fourier price (currency)", shrink=.75, format="%.1e")
    figure.suptitle(f"Independent {shape[0]}×{shape[1]} stress-grid pricing errors", fontsize=13)
    # Constrained layout handles the shared colorbar; reserve a separate bottom
    # annotation through the layout engine rather than calling tight_layout.
    figure.get_layout_engine().set(rect=(0, .12, 1, .88))
    figure.text(.5, .025,
                f"Largest observed |error|: {maximum_error:.3g} currency; frozen maximum-price gate: "
                f"{report['thresholds']['max_abs_price_error_currency']:g}.\n"
                "The endpoint-inclusive grid was not used for fitting. These finite checks are not a uniform continuum error bound.",
                ha="center", va="bottom", fontsize=8.5)
    return figure


def _case_axis(axis, ids):
    labels = [name.replace("validation:", "").replace("stress:", "S") for name in ids]
    axis.set_xticks(np.arange(len(ids)), labels, fontsize=8)
    axis.set_xlim(-.6, len(ids) - .4)
    stress = [i for i, name in enumerate(ids) if name.startswith("stress:")]
    if stress:
        axis.axvspan(stress[0] - .5, stress[-1] + .5, color="#EDF0F4", zorder=-5)
    axis.set_xlabel("Existing validation panel ID; shaded S0–S3 are fixed stress panels")


def paired_errors(report):
    propagation = report["propagation"]
    ids = propagation["case_ids"]
    x = np.arange(len(ids))
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 7.6), sharex=True)
    floor = 1e-16
    any_zero = False
    for key, label, color, marker in (
        ("neural_paired_quantile_shift", "Same joint neural draws", COLORS["surrogate"], "o"),
        ("grid_paired_quantile_shift", "Same weighted reference grid", COLORS["direct"], "s"),
    ):
        error = np.max(np.abs(np.asarray(propagation[key])), axis=(1, 2))
        any_zero |= bool(np.any(error == 0))
        axes[0].plot(x, np.maximum(error, floor), marker=marker, ms=4, lw=1, color=color, label=label)
    substitution_gate = report["thresholds"]["max_abs_paired_quantile_error_currency"]
    axes[0].axhline(substitution_gate, color=COLORS["gate"], ls="--", lw=1,
                    label=f"Surrogate substitution gate: {substitution_gate:g}")
    axes[0].set(yscale="log", ylabel="Maximum paired quantile shift\n(currency, log scale)")
    axes[0].set_title("Forward substitution: change only the pricer", loc="left", fontsize=11)
    axes[0].legend(fontsize=8, loc="best")
    for key, label, color, marker, style in (
        ("neural_direct_minus_grid_quantiles", "Neural + Fourier versus reference", COLORS["direct"], "o", "-"),
        ("neural_surrogate_minus_grid_quantiles", "Neural + surrogate versus reference", COLORS["surrogate"], "x", "--"),
    ):
        error = np.max(np.abs(np.asarray(propagation[key])), axis=(1, 2))
        axes[1].plot(x, error, marker=marker, ms=5, lw=1.2, ls=style, color=color, label=label)
    inference_gate = report["baseline_inference"]["gate_currency"]
    axes[1].axhline(inference_gate, color=COLORS["gate"], ls="--", lw=1,
                    label=f"Existing inference-price gate: {inference_gate:g}")
    axes[1].set_ylabel("Maximum error versus reference\n(currency)")
    axes[1].set_title("Full price distribution: the old inference discrepancy remains visible", loc="left", fontsize=11)
    axes[1].legend(fontsize=8, loc="best")
    for axis in axes:
        _case_axis(axis, ids)
    axes[0].set_xlabel("")
    note = ("Each point is the maximum absolute difference over two contracts and 5%/50%/95% quantiles; no new posterior was trained.\n"
            "Top and bottom use different scales. Surrogate acceptance cannot erase the original calibration failure.")
    if any_zero:
        note += " Exact zeros are displayed at 1e−16 on the log axis."
    return _finish(figure, "Separate forward-price approximation from posterior-inference error", note)


def timing_figure(report):
    rows = [(name, report["benchmark"][name]) for name in ("single", "batch")]
    figure, axis = plt.subplots(figsize=(10.8, 4.8))
    positions = np.arange(len(rows))
    all_times = []
    for offset, route, label, color in ((-.17, "direct", "Direct Fourier", COLORS["direct"]),
                                       (.17, "surrogate", "Polynomial surrogate", COLORS["surrogate"])):
        values = np.array([result[route + "_median_seconds"] for _, result in rows]) * 1000
        all_times.extend(values)
        axis.barh(positions + offset, values, height=.28, label=label, color=color, alpha=.85)
        for position, value, (_, result) in zip(positions + offset, values, rows):
            repeats = np.asarray(result[route + "_seconds"]) * 1000
            axis.scatter(repeats, np.full(len(repeats), position), color="#253648", marker="|", s=70, zorder=4)
            axis.text(max(value, float(repeats.max())) * 1.14, position, f"{value:.3g} ms", va="center", fontsize=9)
    axis.set_yticks(positions, [f"{result['row_count']:,} parameter pair(s)\n{result['speedup']:.1f}× median ratio"
                               for _, result in rows])
    axis.set(xscale="log", xlabel="Wall time per pricing call (milliseconds, log scale)",
             xlim=(min(all_times) / 2, max(all_times) * 4))
    axis.invert_yaxis()
    axis.legend(loc="upper right", fontsize=9)
    return _finish(figure, "Measured query cost: recompute the same two prices on the same inputs",
                   "Bars show median warm in-process CPU time; black ticks are individual timed repeats. Input validation is included.\n"
                   "Direct pricing uses the current looped Fourier implementation. No cached-label lookup or neural inference is timed here.")


def figures(report, checks):
    if report is None:
        return {}
    result = {}
    if all(key in checks for key in ("grid_shape", "grid_parameters", "grid_error")):
        result["parameter-errors"] = parameter_errors(report, checks)
    if "propagation" in report:
        result["paired-errors"] = paired_errors(report)
    if "benchmark" in report:
        result["timing"] = timing_figure(report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/results/hkde-surrogate"))
    parser.add_argument("--out", type=Path, help="new output directory; default RESULTS/plots")
    args = parser.parse_args()
    report, checks, inputs = load_cache(args.results)
    rendered = figures(report, checks)
    if not rendered:
        parser.error("no saved surrogate results found; no price fitting or inference was run")
    out = args.out or args.results / "plots"
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams["svg.hashsalt"] = "hkde-price-surrogate-v1"
    artifacts = {}
    for name, figure in rendered.items():
        for extension in ("png", "svg"):
            path = out / f"{name}.{extension}"
            figure.savefig(path, dpi=170, metadata={"Date": None} if extension == "svg" else None)
            artifacts[path.name] = sha256(path)
        plt.close(figure)
    manifest = {"scope": "saved synthetic Q supervised price-surrogate diagnostics; no new inference or test evaluation",
                "input_sha256": {name: sha256(path) for name, path in inputs.items()},
                "plot_script_sha256": sha256(__file__), "artifact_sha256": artifacts,
                "figures": list(rendered)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

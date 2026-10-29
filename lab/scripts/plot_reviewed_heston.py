"""Review saved Heston draws without sampling, repricing or an acceptance verdict."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import arviz_stats as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration.variance_functionals import heston_variance_functionals  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values, neural):
    """Preserve chain axes for descriptive time blocks and ArviZ quantile MCSE."""
    mean, sd = float(values.mean()), float(values.std())
    halves, quarters = np.array_split(values, 2, axis=1), np.array_split(values, 4, axis=1)
    half_means = np.stack([part.mean(axis=1) for part in halves], axis=1)
    quarter_means = np.stack([part.mean(axis=1) for part in quarters], axis=1)
    quantiles = np.quantile(values, [.05, .10, .90, .95])
    neural_quantiles = np.quantile(neural, [.05, .10, .90, .95])
    tail_fractions = np.stack([(values <= quantiles[0]).mean(axis=1),
                               (values >= quantiles[3]).mean(axis=1)], axis=1)
    def longest_run(chain):
        boundaries = np.r_[0, np.flatnonzero(np.diff(chain) != 0) + 1, len(chain)]
        return int(np.diff(boundaries).max())
    mcse = {f"q{int(100*p):02d}": float(az.mcse(values, method="quantile", prob=p)) for p in (.1, .9)}
    width = float(quantiles[3]-quantiles[0])
    return {"mean": mean, "sd": sd, "chain_means": values.mean(axis=1).tolist(),
            "chain_mean_offset_over_pooled_sd": ((values.mean(axis=1)-mean)/sd).tolist(),
            "chain_half_means": half_means.tolist(), "chain_quarter_means": quarter_means.tolist(),
            "chain_half2_minus_half1_over_pooled_sd": ((half_means[:, 1]-half_means[:, 0])/sd).tolist(),
            "pooled_half2_minus_half1_over_pooled_sd": float((halves[1].mean()-halves[0].mean())/sd),
            "chain_quarter_mean_offset_over_pooled_sd": ((quarter_means-mean)/sd).tolist(),
            "chain_pooled_lower5_upper5_tail_fractions": tail_fractions.tolist(),
            "longest_unchanged_run_per_chain": [longest_run(chain) for chain in values],
            "q05_q10_q90_q95": quantiles.tolist(), "reference_90_percent_width": width,
            "quantile_mcse": mcse, "quantile_mcse_over_90_percent_width": {k: v/width for k, v in mcse.items()},
            "neural_mean": float(neural.mean()), "neural_q05_q10_q90_q95": neural_quantiles.tolist(),
            "signed_neural_mean_error_over_reference_sd": float((neural.mean()-mean)/sd),
            "neural_to_reference_90_percent_width_ratio": float((neural_quantiles[3]-neural_quantiles[0])/width),
            "absolute_neural_q10_q90_errors": np.abs(neural_quantiles[1:3]-quantiles[1:3]).tolist()}


def review(case, out):
    reference_path = case/"reference/reference.json"
    reference = json.loads(reference_path.read_text())
    comparison = json.loads((case/"comparison/comparison.json").read_text())
    with np.load(case/"reference/chains.npz", allow_pickle=False) as archive:
        chains = archive["parameters"]
        acceptance = archive.get("sample_stats_accepted")
    with np.load(case/"comparison/neural_draws.npz", allow_pickle=False) as archive:
        neural = archive["parameters"]
    names = reference["task"]["parameter_names"]
    expected = (reference["settings"]["chains"], reference["settings"]["draws"], len(names))
    if (chains.shape != expected or neural.ndim != 2 or neural.shape[1] != len(names)
            or not np.isfinite(chains).all() or not np.isfinite(neural).all()
            or digest(reference_path) != comparison["reference_input_sha256"]["reference.json"]
            or digest(case/"reference/chains.npz") != comparison["reference_input_sha256"]["chains.npz"]):
        raise ValueError("finite chain/draw axes and exact saved comparison inputs are required")
    values = {name: chains[:, :, j] for j, name in enumerate(names)}
    estimated = {name: neural[:, j] for j, name in enumerate(names)}
    values.update(heston_variance_functionals(reference["task"], chains))
    estimated.update(heston_variance_functionals(reference["task"], neural))
    statistics = {name: describe(value, estimated[name]) for name, value in values.items()}
    inputs = ["reference/reference.json", "reference/chains.npz", "reference/diagnostics.json",
              "comparison/comparison.json", "comparison/neural_draws.npz", "precision/precision.json"]
    report = {"format": "saved_heston_trace_review_v1", "scientific_acceptance": False,
              "scope": "descriptive saved-draw review; not a stationarity test, mode-coverage proof or neural acceptance",
              "surface_id": reference["surface_id"], "reference_chain_shape": list(chains.shape),
              "warmup_excluded": True, "draws_trimmed_or_thinned_for_statistics": False,
              "quantile_mcse_method": "arviz_stats.mcse(method='quantile'), preserving chain/draw axes",
              "statistics": statistics, "input_sha256": {name: digest(case/name) for name in inputs},
              "source_sha256": {"scripts/plot_reviewed_heston.py": digest(__file__),
                  "calibration/variance_functionals.py": digest(ROOT/"src/sbi4finance/calibration/variance_functionals.py")},
              "versions": {name: metadata.version(name) for name in ("arviz-stats", "matplotlib", "numpy")},
              "precision": comparison["precision"], "manual_review": "required; figures and block summaries do not issue a pass"}
    if acceptance is not None:
        report["native_mean_accepted_per_chain"] = acceptance.mean(axis=1).tolist()
    out.mkdir(parents=True, exist_ok=False)
    (out/"review.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    colors = plt.get_cmap("tab10").colors[:len(chains)]
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    for axis, (name, value) in zip(axes.flat, values.items()):
        for chain, color in zip(value, colors):
            axis.plot(np.arange(len(chain))+1, chain, color=color, lw=.35, alpha=.25, rasterized=True)
            blocks = np.array_split(np.arange(len(chain)), 32)
            axis.plot([block.mean()+1 for block in blocks], [chain[block].mean() for block in blocks],
                      color=color, lw=1.25)
        axis.axvline(value.shape[1]/2, color=".35", ls=":", lw=.8)
        axis.axhline(value.mean(), color="black", lw=.7, ls="--")
        axis.set(title=name.replace("_", " "), xlabel="Retained draw", ylabel="Physical units")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(alpha=.15)
    for index, color in enumerate(colors):
        axes[0, 0].plot([], [], color=color, label=f"Chain {index+1}")
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper right")
    figure.suptitle(f"Heston {reference['surface_id']}: retained traces\n"
                    "Faint: every retained draw; solid: 250-draw block means; warmup excluded", fontsize=12)
    figure.savefig(out/"traces.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    for axis, (name, value) in zip(axes.flat, values.items()):
        pooled = value.reshape(-1)
        bins = np.linspace(min(pooled.min(), estimated[name].min()), max(pooled.max(), estimated[name].max()), 81)
        axis.hist(pooled, bins=bins, density=True, histtype="step", color="black", lw=1.5, label="PyMC reference")
        axis.hist(estimated[name], bins=bins, density=True, color="#e68632", alpha=.4, label="Saved neural posterior")
        row = statistics[name]
        axis.set(title=name.replace("_", " "), xlabel="Physical units", ylabel="Density")
        axis.text(.03, .97, f"Mean error: {row['signed_neural_mean_error_over_reference_sd']:+.2f} reference SD\n"
                  f"90% width: {row['neural_to_reference_90_percent_width_ratio']:.2f} × reference",
                  transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.grid(alpha=.15)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    figure.suptitle(f"Heston {reference['surface_id']}: saved marginal comparisons\n"
                    "32,000 correlated reference draws; 10,000 neural draws; descriptive development evidence", fontsize=12)
    figure.savefig(out/"marginals.png", dpi=170)
    plt.close(figure)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = review(args.case, args.out or args.case/"review")
    print(json.dumps({"surface_id": report["surface_id"], "scientific_acceptance": False,
                      "out": str(args.out or args.case/"review")}))


if __name__ == "__main__":
    main()

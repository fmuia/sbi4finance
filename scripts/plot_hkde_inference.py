"""Render the saved restricted HKDE audit; no simulation, training or test access.

PYTHONPATH=src python scripts/plot_hkde_inference.py \
    --reference /path/to/reference --evaluation /path/to/validation --out /new/path

The manifest records exact input/artifact hashes. These checks establish which
saved results the figures depict, not the correctness of the entire pipeline.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"neural": "#087F8C", "reference": "#D97732", "prior": "#738095",
          "gate": "#B33E43", "text": "#243347"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_axis(axis, case_ids):
    positions = np.arange(len(case_ids))
    labels = [name.split(":")[1] if name.startswith("validation:") else "S"+name.split(":")[1]
              for name in case_ids]
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_xlim(-.6, len(case_ids)-.4)
    stress = [i for i, name in enumerate(case_ids) if name.startswith("stress:")]
    if stress:
        axis.axvspan(stress[0]-.5, stress[-1]+.5, color="#E9EDF2", zorder=-5)
    axis.set_xlabel("Validation panel ID; shaded S0–S3 are fixed stress panels")


def footer(figure, text):
    figure.text(.5, .025, text, ha="center", va="bottom", fontsize=8.5, color=COLORS["text"])


def updating(evaluation, checks):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)
    names = [r"Long-run variance $\theta$", r"Jump intensity $\lambda$"]
    errors = np.array([case["mean_error_in_ref_sd"] for case in evaluation["cases"]])
    prior = np.asarray(evaluation["prior_baseline"]["mean_error_in_ref_sd"])
    x = np.arange(len(evaluation["cases"]))
    prior_std = np.asarray(evaluation["prior_baseline"]["std"])
    reference_std = checks["reference_std"] / prior_std
    neural_std = checks["conditional_draws"].std(axis=1, ddof=1) / prior_std
    gate = evaluation["thresholds"]["max_abs_mean_error_in_ref_sd"]
    for dimension in range(2):
        axis = axes[0, dimension]
        axis.axhspan(-gate, gate, color=COLORS["neural"], alpha=.1, label=f"Mean-error gate ±{gate:g}")
        axis.axhline(0, color="#CBD3DB", lw=.8)
        axis.scatter(x, prior[:, dimension], color=COLORS["prior"], marker="x", label="Copying the prior", s=32)
        axis.scatter(x, errors[:, dimension], color=COLORS["neural"], label="Neural posterior", s=25)
        axis.set_title(names[dimension])
        axis.set_ylabel("Mean error / reference posterior SD")
        case_axis(axis, evaluation["case_ids"])
        axis.set_xlabel("")
        axis = axes[1, dimension]
        axis.axhline(1, color=COLORS["prior"], linestyle="--", label="Prior SD")
        axis.plot(x, reference_std[:, dimension], "o-", color=COLORS["reference"], label="Reference posterior", ms=4, lw=1)
        axis.plot(x, neural_std[:, dimension], "s--", color=COLORS["neural"], label="Neural posterior", ms=3.5, lw=1)
        axis.set_ylabel("Posterior SD / prior SD")
        axis.set_ylim(.12, 1.08)
        case_axis(axis, evaluation["case_ids"])
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[1, 0].legend(loc="lower left", fontsize=8)
    figure.suptitle("The network learns from the panel rather than reproducing the prior", fontsize=14)
    footer(figure, f"Mean-error RMSE: prior {evaluation['prior_baseline']['mean_rmse_in_ref_sd']:.3f}; "
           f"neural {evaluation['metrics']['mean_rmse_in_ref_sd']:.3f} reference SD. "
           "Smaller posterior SD shows information gain, not proof of accuracy.")
    figure.tight_layout(rect=(0, .07, 1, .94))
    return figure


def sbc(evaluation, checks):
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ranks = checks["randomized_ranks"]
    report = evaluation["sbc"]
    bound = report["rank_dkw_bound"]
    grid = np.linspace(0, 1, 2049)
    for dimension, axis in enumerate(axes):
        empirical = np.searchsorted(np.sort(ranks[:, dimension]), grid, side="right") / len(ranks)
        axis.axhspan(-bound, bound, color=COLORS["neural"], alpha=.1)
        axis.axhline(bound, color=COLORS["gate"], linestyle="--", lw=1)
        axis.axhline(-bound, color=COLORS["gate"], linestyle="--", lw=1)
        axis.axhline(0, color="#8793A0", lw=.8)
        axis.plot(grid, empirical-grid, color=COLORS["neural"], lw=1.3)
        axis.set(xlim=(0, 1), ylim=(-1.35*bound, 1.35*bound), xlabel="Randomized rank u = (r + U) / (L + 1)",
                 ylabel="Empirical CDF minus uniform CDF")
        axis.set_title([r"Long-run variance $\theta$", r"Jump intensity $\lambda$"][dimension])
        lower, upper = report["coverage_90_wilson_interval"][dimension]
        axis.text(.03, .96, f"KS {report['rank_ks'][dimension]:.4f} < {bound:.4f}\n"
                  f"90% interval coverage: {100*report['coverage_90'][dimension]:.1f}%\n"
                  f"Individual 95% Wilson interval: {100*lower:.1f}–{100*upper:.1f}%",
                  transform=axis.transAxes, va="top", fontsize=8.5,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 3})
    figure.suptitle("Marginal SBC passes; conditional price accuracy remains a separate check", fontsize=13)
    footer(figure, f"{report['cases']} independent synthetic panels; {report['draws_per_case']} draws each. "
           "Dashed DKW limits control both rank checks at family α = 0.05.\n"
           "The shaded band concerns rank uniformity, not a joint-posterior or real-market guarantee.")
    figure.tight_layout(rect=(0, .12, 1, .92))
    return figure


def price_errors(evaluation, reference, checks):
    figure, axes = plt.subplots(2, 1, figsize=(12.4, 7.5), sharex=True, sharey=True)
    cases = evaluation["cases"]
    errors = np.array([case["target_quantile_error_currency"] for case in cases])
    uncertainty = np.array([case["target_quantile_mc95_interval"] for case in cases])
    uncertainty -= checks["reference_target_quantiles"][..., None]
    x = np.arange(len(cases))
    gate = evaluation["thresholds"]["max_abs_target_quantile_error_currency"]
    shades = ["#3564A4", "#087F8C", "#B46934"]
    for target, axis in enumerate(axes):
        axis.axhspan(-gate, gate, color="#EDF4F4", zorder=-4)
        axis.axhline(0, color="#8793A0", lw=.8)
        for value in (-gate, gate):
            axis.axhline(value, color=COLORS["gate"], ls="--", lw=1)
        for quantile, (offset, label, color) in enumerate(zip([-.2, 0, .2], ["5th percentile", "Median", "95th percentile"], shades)):
            center = errors[:, target, quantile]
            spread = np.stack([center-uncertainty[:, target, quantile, 0],
                               uncertainty[:, target, quantile, 1]-center])
            axis.errorbar(x+offset, center, yerr=spread, fmt="o", color=color, ms=3.5,
                          elinewidth=.7, capsize=1.8, label=label)
        contract = reference["target_contracts"][target]
        axis.set_title(f"Excluded target: European put, K = {contract['strike']:g}, T = {contract['maturity']:g} years", loc="left")
        axis.set_ylabel("Neural − reference quantile\n(currency per option)")
        case_axis(axis, evaluation["case_ids"])
    axes[0].set_xlabel("")
    axes[0].legend(loc="lower right", fontsize=8.5, ncol=3)
    worst = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
    case, target, quantile = worst
    axes[target].annotate(f"{evaluation['case_ids'][case]}\n{errors[worst]:+.4f}",
                          xy=(case+[-.2, 0, .2][quantile], errors[worst]), xytext=(case+1.4, errors[worst]-.007),
                          fontsize=8, color=COLORS["gate"], arrowprops={"arrowstyle": "->", "color": COLORS["gate"], "lw": .8})
    axes[0].set_ylim(min(-gate*1.5, float(uncertainty.min())-.012), max(gate*1.2, float(uncertainty.max())+.006))
    figure.suptitle(f"The frozen checkpoint misses the ±{gate:.2f} downstream-price gate", fontsize=14)
    footer(figure, "Bars: individual 95% order-statistic intervals from 4,096 neural draws. "
           "They quantify sampling error, not neural bias or grid error.\n"
           "All fixed audit cases are shown; validation failures remain recorded and the reserved final test is untouched.")
    figure.tight_layout(rect=(0, .09, 1, .94))
    return figure


def joint(evaluation, reference, grid, checks):
    errors = np.array([case["target_quantile_error_currency"] for case in evaluation["cases"]])
    case = int(np.unravel_index(np.argmax(np.abs(errors)), errors.shape)[0])
    axis0, axis1 = grid["axis0"], grid["axis1"]
    weights = grid["weights"][case]
    density = weights / grid["quadrature_weights"]
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(weights[order])
    levels = [float(density[order[np.searchsorted(cumulative, probability)]]) for probability in (.9, .5)]
    figure, axis = plt.subplots(figsize=(7.9, 6.6))
    selected = checks["conditional_draws"][case, ::4]  # Fixed stride, no favorable draw selection.
    axis.scatter(selected[:, 0], selected[:, 1], s=6, alpha=.18, color=COLORS["neural"],
                 edgecolors="none", label="1,024 of 4,096 neural draws")
    contours = axis.contour(axis0, axis1, density.reshape(len(axis0), len(axis1)).T,
                            levels=levels, colors=[COLORS["reference"]]*2, linewidths=[1.7, 2.2])
    axis.clabel(contours, fmt={level: text for level, text in zip(levels, ["90% reference", "50% reference"])}, fontsize=9)
    for values, marker, color, label in ((reference["case_summaries"][case]["mean"], "X", COLORS["reference"], "Reference mean"),
                                          (evaluation["cases"][case]["neural_mean"], "D", COLORS["neural"], "Neural mean"),
                                          (checks["truth"][case], "*", "#273444", "Synthetic generating parameter")):
        axis.scatter(*values, s=100 if marker != "*" else 145, marker=marker, color=color,
                     edgecolors="white", linewidths=.6, label=label, zorder=5)
    axis.set(xlim=(axis0[0], axis0[-1]), ylim=(axis1[0], axis1[-1]),
             xlabel=r"Long-run variance $\theta$ (annualized variance)",
             ylabel=r"Jump intensity $\lambda$ (per year)")
    axis.legend(loc="upper right", fontsize=8)
    reference_corr = reference["case_summaries"][case]["correlation"]
    neural_corr = evaluation["cases"][case]["neural_correlation"]
    axis.text(.025, .035, f"Correlation: reference {reference_corr:.3f}; neural {neural_corr:.3f}\n"
              "The full rectangle is the independent uniform prior support.", transform=axis.transAxes, fontsize=9)
    figure.suptitle(f"{evaluation['case_ids'][case]}: strong dependence, different posterior centers", fontsize=13)
    footer(figure, "Contours contain 50% / 90% of the numerical reference mass. "
           "They are not confidence regions for the generating truth.\n"
           "This case has the largest saved target-price quantile error; the checkpoint remains provisional.")
    figure.tight_layout(rect=(0, .105, 1, .94))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    paths = {"reference.json": args.reference/"reference.json", "reference-grid.npz": args.reference/"reference-grid.npz",
             "neural-validation.json": args.evaluation/"neural-validation.json", "validation-checks.npz": args.evaluation/"validation-checks.npz"}
    reference = json.loads(paths["reference.json"].read_text())
    evaluation = json.loads(paths["neural-validation.json"].read_text())
    for path, expected in ((paths["reference.json"], evaluation["reference_report_sha256"]),
                           (paths["reference-grid.npz"], reference["reference_grid_sha256"]),
                           (paths["validation-checks.npz"], evaluation["checks_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"input hash mismatch: {path.name}")
    with np.load(paths["reference-grid.npz"], allow_pickle=False) as archive:
        grid = dict(archive)
    with np.load(paths["validation-checks.npz"], allow_pickle=False) as archive:
        checks = dict(archive)
    if reference["case_ids"] != evaluation["case_ids"] or not all(np.array_equal(grid[key], checks[key]) for key in ("case_ids", "observations", "truth")):
        raise ValueError("reference and neural comparison cases differ")
    if evaluation["reserved_test_evaluated"]:
        raise ValueError("this figure set is specifically the frozen validation-only audit")
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": COLORS["text"],
                         "text.color": COLORS["text"], "axes.titleweight": "normal",
                         "savefig.facecolor": "white", "svg.hashsalt": "hkde-inference-audit-v1"})
    worst_case = int(np.unravel_index(np.argmax(np.abs([case["target_quantile_error_currency"] for case in evaluation["cases"]])),
                                     (len(evaluation["cases"]), 2, 3))[0])
    worst_id = evaluation["case_ids"][worst_case].split(":")[-1]
    plots = [("prior-to-posterior", updating(evaluation, checks), "Prior comparison and posterior contraction across all audit cases."),
             ("sbc-ranks", sbc(evaluation, checks), "Randomized-rank ECDF deviations and individual coverage intervals."),
             ("target-price-errors", price_errors(evaluation, reference, checks), "Every saved downstream quantile error with sampling uncertainty and the fixed price gate."),
             (f"joint-posterior-case-{worst_id}", joint(evaluation, reference, grid, checks), "Joint numerical reference and neural draws for the largest downstream price error.")]
    repository = Path(__file__).resolve().parents[1]
    input_paths = {name: str(path.resolve().relative_to(repository)) if path.resolve().is_relative_to(repository) else path.name
                   for name, path in paths.items()}
    manifest = {"experiment_id": evaluation["experiment_id"], "scope": "S synthetic validation only; frozen checkpoint remains provisional",
                "input_sha256": {input_paths[name]: sha256(path) for name, path in paths.items()},
                "plot_script": str(Path(__file__).resolve().relative_to(repository)),
                "plot_script_sha256": sha256(__file__), "checkpoint_sha256": evaluation["checkpoint_sha256"],
                "reserved_test_evaluated": False, "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "figures": []}
    for name, figure, caption in plots:
        outputs = {}
        for extension in ("png", "svg"):
            destination = args.out/f"{name}.{extension}"
            figure.savefig(destination, dpi=160, metadata={"Date": None} if extension == "svg" else None)
            outputs[destination.name] = sha256(destination)
        plt.close(figure)
        manifest["figures"].append({"name": name, "caption": caption, "output_sha256": outputs})
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(args.out), "figures": len(plots), "inputs_verified": True}))


if __name__ == "__main__":
    main()

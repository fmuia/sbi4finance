"""Plot sealed aggregate trajectory evaluations without reopening any cohort.

Receipt content hashes and the explicitly supplied local evaluation JSONs are
checked before creating the destination. Oscar paths in receipts are provenance
strings only: this tool never follows their canonical state directories.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np


MODELS = ("heston", "rough-known", "rough-joint")
LENGTHS = (1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756, 1260)
NAMES = {"heston": "Heston (prototype)", "rough-known": "Rough Bergomi · known curve",
         "rough-joint": "Rough Bergomi · inferred curve"}
COLORS = {"heston": "#2563a7", "rough-known": "#d16b26", "rough-joint": "#87549d"}
PARAMETERS = {"kappa": "Mean reversion κ", "long_variance": "Long variance L",
              "vol_of_vol": "Vol. of variance σ", "rho": "Leverage ρ", "drift": "Drift μ",
              "hurst": "Roughness H", "eta": "Volatility scale η",
              "short_variance": "Short variance S", "decay": "Curve decay k"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sealed(path):
    """Verify local receipt content only; never resolve its recorded paths."""
    value = json.loads(Path(path).read_text())
    declared = value.get("receipt_sha256")
    content = {k: v for k, v in value.items() if k != "receipt_sha256"}
    if declared != fingerprint(content):
        raise ValueError(f"sealed receipt content hash differs: {path}")
    return value


def _vector(value, dimension, name):
    values = np.asarray(value, dtype=float)
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise ValueError(f"invalid finite parameter vector: {name}")
    return values


def validate_inputs(selection_path, assessment_path, evaluations):
    """Validate all identities/design/aggregate cells before any output write."""
    selection, assessment = read_sealed(selection_path), read_sealed(assessment_path)
    phase = {"path_workshop_confirmation_v1": "confirmation",
             "path_workshop_final_characterization_v1": "final"}.get(assessment.get("format"))
    if (selection.get("format") != "path_workshop_selection_v1" or phase is None
            or assessment.get("selection_sha256") != selection["receipt_sha256"]
            or set(evaluations) != set(MODELS) or set(selection["models"]) != set(MODELS)
            or set(assessment["evaluations"]) != set(MODELS)
            or set(assessment["models"]) != set(MODELS)
            or selection.get("scientific_acceptance") is not False
            or assessment.get("scientific_acceptance") is not False
            or assessment.get("execution_pass") is not True):
        raise ValueError("selection/assessment identity, model set, phase or execution differs")
    design = selection["design"]
    cohort = design[phase]
    final = phase == "final"
    if (tuple(design["lengths"]) != LENGTHS or design["draws"] != 512
            or assessment.get("reserved_test_evaluated") is not final):
        raise ValueError("frozen12-length/512-draw design or assessment cohort differs")
    if cohort != ({"split": "test", "offset": 0, "cases": 4000} if final else
                  {"split": "validation", "offset": 1024, "cases": 2976}):
        raise ValueError("unexpected frozen independent-case cohort")
    reports, sources = {}, {}
    selection_file_sha = sha256(selection_path)
    for name in MODELS:
        directory = Path(evaluations[name]).resolve()
        path = directory / "evaluation.json"
        digest = sha256(path)
        if digest != assessment["evaluations"][name]["sha256"]:
            raise ValueError(f"{name}: local evaluation hash differs from sealed assessment")
        report = json.loads(path.read_text())
        expected = selection["models"][name]
        task = expected["task"]
        dimension = len(task["parameter_names"])
        if (fingerprint(task) != expected["task_sha256"] or task["model"] != name
                or task["dt_years"] != 1 / 252 or task["n_observations"] != 1260
                or "drift" not in task["parameter_names"]
                or report.get("format") != "path_development_evaluation_v1"
                or report.get("split") != cohort["split"] or report.get("offset") != cohort["offset"]
                or report.get("cases") != cohort["cases"] or report.get("draws") != design["draws"]
                or report.get("seed") != design["seed"]
                or report.get("phase") != ("final-test" if final else "confirmation")
                or report.get("selection_sha256") != selection_file_sha
                or report.get("scientific_acceptance") is not False
                or report.get("reserved_test_evaluated") is not final
                or sorted(map(int, report["lengths"])) != list(LENGTHS)
                or report.get("targets") != task["parameter_names"]
                or any(report.get(field) != expected[key] for field, key in
                       (("checkpoint_sha256", "model_sha256"), ("task_sha256", "task_sha256"),
                        ("bank_manifest_sha256", "bank_manifest_sha256")))
                or assessment["models"][name].get("execution_pass") is not True):
            raise ValueError(f"{name}: task/model/bank/cohort/seed/draw/prefix identity differs")
        if final:
            opening = report.get("test_opening_receipt", {})
            content = {k: v for k, v in opening.items() if k != "receipt_sha256"}
            if (opening.get("receipt_sha256") != fingerprint(content)
                    or report.get("test_opening_receipt_sha256") != opening["receipt_sha256"]
                    or opening.get("selection_sha256") != selection["receipt_sha256"]
                    or opening.get("model_sha256") != expected["model_sha256"]):
                raise ValueError(f"{name}: embedded once-only test receipt differs")
        for length in LENGTHS:
            metric = report["lengths"][str(length)]
            assessed = assessment["models"][name]["lengths"][str(length)]
            if (metric.get("cases") != cohort["cases"] or metric.get("draws_per_case") != design["draws"]
                    or metric.get("nonfinite_count") != 0
                    or not np.isclose(metric.get("elapsed_years"), length / 252, rtol=0, atol=1e-14)):
                raise ValueError(f"{name}:{length}: aggregate cases/draws/time/finite execution differs")
            coverage = _vector(metric["coverage"]["0.9"]["coverage"], dimension, "coverage")
            mse = _vector(metric["mean_squared_error"], dimension, "MSE")
            baseline = _vector(metric["prior_mean_squared_error"], dimension, "baseline MSE")
            ratio = _vector(metric["mse_to_prior_mean_ratio"], dimension, "risk ratio")
            support = _vector(metric["out_of_prior_fraction"], dimension, "support violations")
            if (np.any(coverage < 0) or np.any(coverage > 1) or np.any(mse < 0)
                    or np.any(baseline <= 0) or np.any(support != 0)
                    or not np.allclose(ratio, mse / baseline, rtol=1e-10, atol=1e-12)
                    or not np.allclose(coverage * cohort["cases"], np.round(coverage * cohort["cases"]), rtol=0, atol=1e-7)
                    or not np.allclose(coverage, assessed["coverage90"], rtol=0, atol=1e-12)
                    or len(assessed["nominal_inside_interval"]) != dimension
                    or not all(isinstance(x, bool) for x in assessed["nominal_inside_interval"])):
                raise ValueError(f"{name}:{length}: invalid or inconsistent coverage/risk aggregates")
            expected_criteria = ["one_sided_minimum_nominal_coverage_for_atomic_target" if name == "heston" and target == "drift"
                                 else "two_sided_nominal_compatibility" for target in task["parameter_names"]]
            if assessed.get("coverage_criterion") != expected_criteria:
                raise ValueError(f"{name}:{length}: atomic-target coverage criterion differs")
        reports[name] = report
        sources[name] = {"local_evaluation_json": str(path), "evaluation_sha256": digest,
                         "task_sha256": report["task_sha256"], "model_sha256": report["checkpoint_sha256"],
                         "bank_manifest_sha256": report["bank_manifest_sha256"],
                         "case_identities_sha256": report["case_identities_sha256"],
                         "targets": report["targets"]}
    return selection, assessment, reports, sources, phase, cohort


def _save(figure, destination, name):
    figure.savefig(destination / f"{name}.png", dpi=180, facecolor="white")
    figure.savefig(destination / f"{name}.pdf", facecolor="white",
                   metadata={"Creator": "sbi4finance plot_path_workshop_results.py", "CreationDate": None, "ModDate": None})


def render(selection_path, assessment_path, evaluations, out):
    destination = Path(out).resolve()
    if destination.exists():
        raise FileExistsError("figure destination must be new")
    selection, assessment, reports, sources, phase, cohort = validate_inputs(selection_path, assessment_path, evaluations)
    # No output, cache or plotting-library initialization precedes validation.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sbi4finance-path-plot-mpl"))
    if "XDG_CACHE_HOME" not in os.environ:
        font_cache = Path(tempfile.gettempdir()) / "sbi4finance-path-font-cache"
        font_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(font_cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    caption_phase = "Untouched validation confirmation" if phase == "confirmation" else "Reserved-test characterization"
    cohort_caption = (f"{caption_phase} · {cohort['cases']:,} independent histories per model · "
                      f"{selection['design']['draws']} posterior draws per history")
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
             "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
             "legend.fontsize": 9, "figure.dpi": 110, "savefig.bbox": "tight"}
    destination.mkdir(parents=True, exist_ok=False)
    outlined = []
    try:
        with plt.rc_context(style):
            figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
            figure.suptitle("Learning physical drift from daily returns", fontsize=16, y=.98)
            figure.text(.5, .90, cohort_caption, ha="center", fontsize=10, color="#404852")
            years = np.asarray(LENGTHS) / 252
            for name in MODELS:
                drift = reports[name]["targets"].index("drift")
                ratios = [reports[name]["lengths"][str(n)]["mse_to_prior_mean_ratio"][drift] for n in LENGTHS]
                coverage = [100 * reports[name]["lengths"][str(n)]["coverage"]["0.9"]["coverage"][drift] for n in LENGTHS]
                options = {"color": COLORS[name], "label": NAMES[name], "linewidth": 2,
                           "linestyle": "--" if name == "heston" else "-", "marker": "o", "markersize": 3.5}
                axes[0].plot(years, ratios, **options)
                axes[1].plot(years, coverage, **options)
            axes[0].axhline(1, color="#57606a", linestyle=":", linewidth=1)
            axes[1].axhline(90, color="#57606a", linestyle=":", linewidth=1)
            axes[0].set(title="Posterior-mean estimation error", ylabel="MSE / own prior-mean baseline MSE")
            axes[1].set(title="Empirical coverage of drift intervals", ylabel="Coverage of nominal 90% intervals (%)")
            axes[0].set_ylim(bottom=0)
            axes[1].set_ylim(min(80, axes[1].get_ylim()[0]), 100.5)
            for axis in axes:
                axis.set_xscale("log")
                axis.set_xticks(np.array([1, 21, 252, 504, 1260]) / 252, ["1d", "21d", "1y", "2y", "5y"])
                axis.set_xlabel("Observed history · 252 daily returns per year")
                axis.grid(axis="y", alpha=.2)
                axis.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
            axes[0].legend(loc="lower left", frameon=False)
            figure.subplots_adjust(left=.085, right=.985, bottom=.25, top=.79, wspace=.30)
            figure.text(.085, .105, "Each model uses its own prior-mean risk baseline. Prefixes are nested; points along a curve are dependent.", fontsize=9, color="#404852")
            figure.text(.085, .055, "Heston is a prototype with drift atoms; central intervals may overcover. Descriptive marginals are not a posterior certificate.", fontsize=9, color="#404852")
            _save(figure, destination, "drift-learning")
            plt.close(figure)

            dimensions = [len(reports[name]["targets"]) for name in MODELS]
            figure, axes = plt.subplots(3, 1, figsize=(12.4, 10.6), gridspec_kw={"height_ratios": dimensions})
            figure.suptitle("Marginal coverage across the common daily history grid", fontsize=16, y=.985)
            figure.text(.5, .952, cohort_caption, ha="center", fontsize=10, color="#404852")
            minimum = min(min(reports[name]["lengths"][str(n)]["coverage"]["0.9"]["coverage"]) for name in MODELS for n in LENGTHS)
            normalization = TwoSlopeNorm(vmin=min(80, 5 * np.floor(100 * minimum / 5)), vcenter=90, vmax=100)
            ticks = [f"{n}d" if n < 252 else f"{n // 252}y" for n in LENGTHS]
            image = None
            for axis, name in zip(axes, MODELS, strict=True):
                coverage = 100 * np.asarray([reports[name]["lengths"][str(n)]["coverage"]["0.9"]["coverage"] for n in LENGTHS]).T
                image = axis.imshow(coverage, aspect="auto", cmap="RdBu", norm=normalization, interpolation="nearest")
                axis.set_title(NAMES[name], loc="left", pad=7, color=COLORS[name], fontweight="bold")
                axis.set_yticks(np.arange(len(reports[name]["targets"])), [PARAMETERS[x] for x in reports[name]["targets"]])
                axis.set_xticks(np.arange(len(LENGTHS)), ticks)
                axis.tick_params(length=0, pad=6)
                for col, length in enumerate(LENGTHS):
                    flags = assessment["models"][name]["lengths"][str(length)]["nominal_inside_interval"]
                    for row, target in enumerate(reports[name]["targets"]):
                        value = coverage[row, col]
                        axis.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=9,
                                  color="white" if value < normalization.vmin + 3 or value > 97 else "#19232d")
                        if not flags[row]:
                            axis.add_patch(Rectangle((col - .48, row - .46), .96, .92, fill=False, edgecolor="#151515", linewidth=1.2))
                            outlined.append({"model": name, "parameter": target, "length": length, "coverage90": float(value / 100)})
                axis.set_xticks(np.arange(-.5, len(LENGTHS), 1), minor=True)
                axis.set_yticks(np.arange(-.5, len(reports[name]["targets"]), 1), minor=True)
                axis.grid(which="minor", color="white", linewidth=1.2)
                axis.tick_params(which="minor", bottom=False, left=False)
                for spine in axis.spines.values():
                    spine.set_visible(False)
            figure.subplots_adjust(left=.17, right=.89, bottom=.14, top=.90, hspace=.45)
            color_axis = figure.add_axes([.915, .26, .015, .50])
            bar = figure.colorbar(image, cax=color_axis)
            bar.set_label("Empirical coverage (%) · nominal 90%")
            axes[-1].set_xlabel("Observed history · d = daily returns, y = 252 daily returns")
            figure.text(.17, .077, "Black outline: failed sealed simultaneous marginal coverage screen; no uncertainty bands are plotted.", fontsize=9, color="#404852")
            figure.text(.17, .047, "Heston drift uses an undercoverage screen only: central intervals for atomic posteriors can legitimately overcover.", fontsize=9, color="#404852")
            figure.text(.17, .017, "Independent unit: a whole simulated history. Nested prefixes are dependent. Marginal coverage does not certify conditional or joint accuracy.", fontsize=9, color="#404852")
            _save(figure, destination, "marginal-coverage")
            plt.close(figure)

        metadata = {"format": "path_workshop_aggregate_figures_v1", "phase": phase,
                    "cohort": cohort, "draws_per_history": selection["design"]["draws"], "lengths": list(LENGTHS),
                    "source_code_sha256": sha256(__file__), "selection_file_sha256": sha256(selection_path),
                    "selection_receipt_sha256": selection["receipt_sha256"],
                    "assessment_file_sha256": sha256(assessment_path), "assessment_receipt_sha256": assessment["receipt_sha256"],
                    "assessment_status": assessment["status"], "assessment_passed": assessment["passed"],
                    "evaluations": sources, "outlined_failed_coverage_cells": outlined,
                    "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(destination.iterdir())},
                    "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                    "risk_ratio": "Reported MSE divided by same-model/same-cohort prior-mean baseline MSE. Evaluator prior mean comes from its seeded100k prior draws; this is distinct from the assessment's analytic-prior paired-risk screen. No new simulations or risk estimates were generated.",
                    "coverage": "Reported marginal central90% interval truth coverage over independent histories; equal to sealed assessment coverage. Heatmap outlines use its frozen simultaneous-screen decisions, not posthoc pointwise tests.",
                    "uncertainty": "No confidence intervals or bands fabricated or added. Prefixes share histories and are dependent; posterior draws are not independent evaluation cases.",
                    "scope": "Common daily1260-return design with twelve frozen prefix evaluations. Own-model baselines; curves are not a ranking of stochastic models. Heston remains a prototype with drift atoms.",
                    "scientific_acceptance": False, "asset_publication": False,
                    "reserved_test_evaluated": phase == "final",
                    "receipt_verification": "Local content seals verified; supplied local evaluation JSON hashes verified against assessment. No canonical Oscar path was opened."}
        (destination / "figure-metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    except Exception:
        plt.close("all")
        shutil.rmtree(destination)
        raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--assessment", required=True)
    parser.add_argument("--evaluation", action="append", required=True, metavar="MODEL=LOCALDIR")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluations = {}
    for value in args.evaluation:
        name, separator, directory = value.partition("=")
        if not separator or name not in MODELS or not directory or name in evaluations:
            parser.error("provide each model exactly once as MODEL=LOCALDIR")
        evaluations[name] = directory
    report = render(args.selection, args.assessment, evaluations, args.out)
    print(json.dumps({"status": "descriptive_figures_complete", "phase": report["phase"],
                      "independent_histories_per_model": report["cohort"]["cases"], "out": str(Path(args.out).resolve())}))


if __name__ == "__main__":
    main()

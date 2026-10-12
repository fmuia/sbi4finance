"""Render cached restricted rough-Bergomi diagnostics without inference or pricing.

Reports remain the evidence; titles do not assume that any acceptance gate
passed. Missing reports are supported by the notebook, while this export CLI
requires at least one drawable result and writes a new output directory.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"neural": "#087F8C", "reference": "#D97732", "prior": "#738095", "gate": "#B33E43"}
PARAMETERS = (r"Roughness $H$", r"Amplitude $\eta$")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reports(directory):
    """Load available cache only; absent artifacts never trigger computation."""
    directory = Path(directory)
    heldout = (directory / "neural-test.json").exists()
    if heldout and (directory / "neural-validation.json").exists():
        raise ValueError("keep held-out and development evaluations in separate result directories")
    evaluation_filename = "neural-test.json" if heldout else "neural-validation.json"
    reports, paths = {}, {}
    for name, filename in (("reference", "reference.json"), ("evaluation", evaluation_filename),
                           ("training", "training.json"), ("cluster", "cluster.json")):
        path = directory / filename
        if path.exists():
            reports[name] = json.loads(path.read_text())
            paths[filename] = path
    checks = {}
    checks_path = directory / ("test-checks.npz" if heldout else "validation-checks.npz")
    if checks_path.exists():
        with np.load(checks_path, allow_pickle=False) as saved:
            # Graphs need ranks only; never load a neural framework/checkpoint.
            checks = {key: saved[key] for key in ("randomized_ranks", "prior_randomized_ranks") if key in saved}
        paths[checks_path.name] = checks_path
    evaluation = reports.get("evaluation", {})
    if "reference" in reports and "reference_report_sha256" in evaluation:
        if sha256(paths["reference.json"]) != evaluation["reference_report_sha256"]:
            raise ValueError("evaluation/reference report hash mismatch")
    expected = evaluation.get("checks_sha256")
    if expected is not None and checks_path.exists() and sha256(checks_path) != expected:
        raise ValueError("evaluation/checks hash mismatch")
    reference = reports.get("reference", {})
    reference_checks = directory / "reference-checks.npz"
    expected_reference = reference.get("artifact_sha256", {}).get(reference_checks.name)
    if expected_reference is not None and reference_checks.exists():
        # Bind cached reference evidence without loading any observation/bank arrays.
        if sha256(reference_checks) != expected_reference:
            raise ValueError("reference/checks hash mismatch")
        paths[reference_checks.name] = reference_checks
    return reports, checks, paths


def _case_axis(axis, ids):
    labels = [name.replace("validation:", "").replace("test:", "").replace("stress:", "S") for name in ids]
    axis.set_xticks(np.arange(len(ids)), labels, fontsize=8)
    axis.set_xlim(-0.7, len(ids) - 0.3)
    stress = [i for i, name in enumerate(ids) if name.startswith("stress:")]
    if stress:
        axis.axvspan(stress[0] - .5, stress[-1] + .5, color="#EDF0F4", zorder=-5)
    axis.set_xlabel("Held-out test panel ID (fixed first 16; no stress replacements)"
                    if ids and all(name.startswith("test:") for name in ids) else
                    "Validation panel ID; shaded S0–S3 are fixed stress panels")


def _finish(figure, title, note):
    figure.suptitle(title, fontsize=13)
    figure.text(.5, .018, note, ha="center", va="bottom", fontsize=8.5)
    figure.tight_layout(rect=(0, .09, 1, .93))
    return figure


def _reference_summary(case, reference, index):
    if "reference_summary" in case:
        return case["reference_summary"]
    return reference["cases"][index]["reference"]["summary"]


def sbc_figure(evaluation, checks):
    report = evaluation["sbc"]
    ranks = np.asarray(checks.get("randomized_ranks", report.get("randomized_ranks", [])))
    if ranks.ndim != 2 or ranks.shape[1] != 2:
        return None
    prior_report = evaluation.get("prior_baseline", {}).get("sbc", {})
    prior_ranks = np.asarray(checks.get("prior_randomized_ranks", prior_report.get("randomized_ranks", [])))
    bound = report["rank_dkw_bound"]
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    grid = np.linspace(0, 1, 2049)
    limit = 1.35 * bound
    for dimension, axis in enumerate(axes):
        axis.axhspan(-bound, bound, color=COLORS["neural"], alpha=.08)
        for value in (-bound, bound):
            axis.axhline(value, color=COLORS["gate"], ls="--", lw=1)
        axis.axhline(0, color="#A3AAB4", lw=.7)
        for samples, label, color in ((ranks, "Neural", COLORS["neural"]),
                                      (prior_ranks, "Prior control", COLORS["prior"])):
            if samples.ndim == 2 and samples.shape[1] == 2:
                ecdf = np.searchsorted(np.sort(samples[:, dimension]), grid, side="right") / len(samples)
                axis.plot(grid, ecdf - grid, label=label, color=color, lw=1.3)
                limit = max(limit, 1.12 * float(np.max(np.abs(ecdf - grid))))
        axis.set(title=PARAMETERS[dimension], xlabel="Randomized rank u", ylabel="Empirical CDF − uniform CDF", xlim=(0, 1))
        axis.text(.03, .95, f"Neural KS: {report['rank_ks'][dimension]:.4f}\n"
                  f"DKW limit: {bound:.4f}\n90% coverage: {report['coverage_90'][dimension]:.3f}",
                  transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    for axis in axes:
        axis.set_ylim(-limit, limit)
    axes[1].legend(loc="lower right", fontsize=9)
    return _finish(figure, "Held-out marginal SBC and the prior control" if evaluation.get("split") == "test" else
                   "Marginal SBC and the prior control",
                   f"{report['cases']} independent audit panels; {report['draws_per_case']} draws each. "
                   f"DKW limits use family α={report.get('sbc_family_alpha', .05):g} across H and η.\n"
                   "An algorithm that ignores observations can pass this marginal check; conditional comparisons remain necessary.")


def intervals_figure(evaluation, reference):
    cases = evaluation["cases"]
    ids = evaluation.get("case_ids", [case["case_id"] for case in cases])
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 7.8), sharex=True)
    prior_cases = evaluation.get("prior_baseline", {}).get("cases", [])
    x = np.arange(len(cases))
    for dimension, axis in enumerate(axes):
        for label, color, offset in (("Prior", COLORS["prior"], -.23),
                                     ("Reference", COLORS["reference"], 0),
                                     ("Neural", COLORS["neural"], .23)):
            if label == "Prior" and len(prior_cases) != len(cases):
                continue
            summaries = (prior_cases if label == "Prior" else cases if label == "Neural" else
                         [_reference_summary(case, reference, i) for i, case in enumerate(cases)])
            qkey, mkey = (("quantiles", "mean") if label == "Reference" else ("parameter_quantiles", "neural_mean"))
            low = np.array([summary[qkey][dimension][0] for summary in summaries])
            high = np.array([summary[qkey][dimension][2] for summary in summaries])
            mean = np.array([summary[mkey][dimension] for summary in summaries])
            # Mean need not lie inside an equal-tail interval; draw segments and
            # centers independently instead of forcing nonnegative error bars.
            axis.vlines(x + offset, low, high, color=color, lw=1.5, alpha=.9)
            axis.scatter(x + offset, mean, color=color, s=18, label=label, zorder=4)
        axis.set_ylabel(PARAMETERS[dimension])
        _case_axis(axis, ids)
    axes[0].set_xlabel("")
    axes[0].legend(ncol=3, loc="upper right", fontsize=9)
    return _finish(figure, "Posterior updating: mean and 90% marginal interval",
                   "Intervals describe parameter uncertainty; they are not Monte Carlo error bars. "
                   "Broad H can be compatible with weak identification.\n"
                   "Compare neural and reference values before interpreting contraction as learning.")


def selected_case_figure(evaluation, reference, index):
    case = evaluation["cases"][index]
    summary = _reference_summary(case, reference, index)
    prior_cases = evaluation.get("prior_baseline", {}).get("cases", [])
    entries = [("Reference", summary["mean"], summary["quantiles"], COLORS["reference"]),
               ("Neural", case["neural_mean"], case["parameter_quantiles"], COLORS["neural"])]
    if len(prior_cases) > index:
        prior = prior_cases[index]
        entries.insert(0, ("Prior", prior["neural_mean"], prior["parameter_quantiles"], COLORS["prior"]))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for dimension, axis in enumerate(axes):
        for position, (label, means, quantiles, color) in enumerate(entries):
            axis.hlines(position, quantiles[dimension][0], quantiles[dimension][2], color=color, lw=3)
            axis.scatter(means[dimension], position, color=color, s=40, zorder=3)
        axis.set_yticks(np.arange(len(entries)), [item[0] for item in entries])
        axis.set_xlabel(PARAMETERS[dimension])
        axis.set_ylim(-.5, len(entries) - .5)
    figure.tight_layout()
    return figure


def conditional_figure(evaluation):
    cases = evaluation["cases"]
    ids = evaluation.get("case_ids", [case["case_id"] for case in cases])
    thresholds = evaluation["thresholds"]
    figure, axes = plt.subplots(3, 1, figsize=(12.6, 8.2), sharex=True)
    x = np.arange(len(cases))
    for axis, field, threshold, label in (
        (axes[0], "mean_error_in_ref_sd", "max_abs_mean_error_in_ref_sd", "Mean error /\nreference SD"),
        (axes[1], "relative_std_error", "max_abs_relative_std_error", "Neural SD /\nreference SD − 1"),
        (axes[2], "correlation_error", "max_abs_correlation_error", "Neural − reference\ncorrelation"),
    ):
        gate = thresholds[threshold]
        axis.axhspan(-gate, gate, color=COLORS["neural"], alpha=.07)
        axis.axhline(0, color="#A3AAB4", lw=.7)
        for value in (-gate, gate):
            axis.axhline(value, color=COLORS["gate"], ls="--", lw=1)
        values = np.array([case[field] for case in cases])
        if values.ndim == 2:
            for dimension, name in enumerate(PARAMETERS):
                axis.plot(x, values[:, dimension], "o-", ms=4, lw=.8,
                          color=[COLORS["neural"], COLORS["reference"]][dimension], label=name)
        else:
            axis.plot(x, values, "o-", ms=4, lw=.8, color=COLORS["neural"])
        axis.set_ylabel(label)
        _case_axis(axis, ids)
    axes[0].legend(ncol=2, loc="best", fontsize=8)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    return _finish(figure, "Conditional parameter and dependence diagnostics",
                   "Dashed lines are the frozen per-case gates. The separate pooled mean-RMSE and interval-mass gates also apply.\n"
                   "Reference adequacy is checked independently; these differences do not establish continuum or market accuracy.")


def target_figure(evaluation, reference):
    cases = evaluation["cases"]
    ids = evaluation.get("case_ids", [case["case_id"] for case in cases])
    errors = np.asarray([case["target_mean_error_currency"] for case in cases])
    envelopes = np.asarray([case["target_mean_error_plus_2mcse_currency"] for case in cases])
    two_se = np.maximum(envelopes - np.abs(errors), 0)
    gate = evaluation["thresholds"]["max_target_mean_error_plus_2mcse_currency"]
    contracts = reference.get("target_contracts", [])
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 7), sharex=True)
    x = np.arange(len(cases))
    for target, axis in enumerate(axes):
        for value in (-gate, gate):
            axis.axhline(value, color=COLORS["gate"], ls="--", lw=1)
        axis.axhspan(-gate, gate, color=COLORS["neural"], alpha=.07)
        axis.axhline(0, color="#A3AAB4", lw=.7)
        axis.errorbar(x, errors[:, target], yerr=two_se[:, target], fmt="o", ms=4,
                      capsize=2, color=COLORS["neural"], elinewidth=1)
        if len(contracts) > target:
            contract = contracts[target]
            axis.set_title(f"Excluded put: K={contract['strike']:g}, T={contract['maturity']:g} years", loc="left")
        else:
            axis.set_title(f"Excluded target {target + 1} (stored contract order)", loc="left")
        axis.set_ylabel("Neural − reference mean price\n(currency per option)")
        _case_axis(axis, ids)
    axes[0].set_xlabel("")
    return _finish(figure, "Downstream posterior mean prices with sampling uncertainty",
                   "Bars are ±2 combined Monte Carlo SE; acceptance requires |error| + 2 SE ≤ the dashed gate.\n"
                   "Fresh target paths are independent of conditioning paths. This is a mean-price check, not a price-quantile or bias bound.")


def reference_figure(reference):
    cases = reference["cases"]
    figure, axes = plt.subplots(3, 1, figsize=(12.2, 8.6), sharex=True)
    ids = [case["case_id"] for case in cases]
    x = np.arange(len(cases))
    thresholds = reference.get("thresholds", {})
    for axis, key, gate_name, ylabel, minimum in (
        (axes[0], "ess", "min_ess", "Importance effective\nsample size", True),
        (axes[1], "max_weight", "max_weight", "Largest normalized\nimportance weight", False),
    ):
        values = np.asarray([case["reference"][key] for case in cases])
        axis.plot(x, values, "o-", color=COLORS["reference"], ms=4)
        axis.set_ylabel(ylabel)
        if gate_name in thresholds:
            gate = thresholds[gate_name]
            axis.axhline(gate, color=COLORS["gate"], ls="--", lw=1,
                         label=f"Frozen {'minimum' if minimum else 'maximum'}: {gate:g}")
            failed = np.flatnonzero(values < gate if minimum else values > gate)
            axis.scatter(x[failed], values[failed], marker="x", s=65, linewidths=2,
                         color=COLORS["gate"], zorder=5)
            for index in failed:
                axis.annotate(f"{ids[index]}: {values[index]:.4g}", (x[index], values[index]),
                              xytext=(8, -16), textcoords="offset points", fontsize=8,
                              color=COLORS["gate"])
            axis.legend(loc="upper right", fontsize=8)
    for dimension, label in enumerate(PARAMETERS):
        axes[2].plot(x, [case["reference"]["mean_mcse_in_ref_sd"][dimension] for case in cases],
                     "o-", ms=4, label=label, color=[COLORS["neural"], COLORS["reference"]][dimension])
    axes[2].set_ylabel("Mean MCSE /\nreference SD")
    if "max_mean_mcse_in_ref_sd" in thresholds:
        gate = thresholds["max_mean_mcse_in_ref_sd"]
        axes[2].axhline(gate, color=COLORS["gate"], ls="--", lw=1, label=f"Frozen maximum: {gate:g}")
    for axis in axes:
        _case_axis(axis, ids)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].legend(fontsize=8, loc="upper right", ncol=3)
    return _finish(figure, "Held-out importance-reference precision" if reference.get("split") == "test" else
                   "Independent importance-reference precision",
                   "Red crosses identify failed ESS/weight gates; passing mean MCSE does not override them.\n"
                   "Independent-block and target-mean precision have additional gates. These checks do not prove complete posterior support.")


def figures(reports, checks):
    result = {}
    reference = reports.get("reference", {})
    evaluation = reports.get("evaluation", {})
    if reference.get("cases"):
        result["reference-precision"] = reference_figure(reference)
    if evaluation.get("cases"):
        result["posterior-intervals"] = intervals_figure(evaluation, reference)
        result["conditional-errors"] = conditional_figure(evaluation)
        if "target_mean_error_currency" in evaluation["cases"][0]:
            result["target-mean-errors"] = target_figure(evaluation, reference)
        if "sbc" in evaluation:
            sbc = sbc_figure(evaluation, checks)
            if sbc is not None:
                result["marginal-sbc"] = sbc
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/results/rough-inference"))
    parser.add_argument("--out", type=Path, help="new directory; default RESULTS/plots")
    args = parser.parse_args()
    reports, checks, paths = load_reports(args.results)
    rendered = figures(reports, checks)
    if not rendered:
        parser.error("no drawable saved results found; no inference was run")
    destination = args.out or args.results / "plots"
    destination.mkdir(parents=True, exist_ok=False)
    plt.rcParams["svg.hashsalt"] = "rough-inference-pilot-v1"
    artifacts = {}
    for name, figure in rendered.items():
        for suffix in ("png", "svg"):
            path = destination / f"{name}.{suffix}"
            figure.savefig(path, dpi=170, metadata={"Date": None} if suffix == "svg" else None)
            artifacts[path.name] = sha256(path)
        plt.close(figure)
    heldout = reports.get("evaluation", reports.get("reference", {})).get("split") == "test"
    scope = ("cached held-out synthetic Q diagnostics; no bank access or new inference" if heldout else
             "cached synthetic Q finite-numerical-model diagnostics; no final-test access")
    manifest = {"scope": scope,
                "input_sha256": {name: sha256(path) for name, path in paths.items()},
                "plot_script_sha256": sha256(__file__), "artifact_sha256": artifacts,
                "figures": list(rendered)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

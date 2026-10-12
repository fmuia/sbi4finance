"""Review cached HKDE importance-correction evidence without running inference."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"corrected": "#087F8C", "raw": "#D97732", "reference": "#596B84", "gate": "#B33E43"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cache(directory):
    directory = Path(directory)
    reports, checks, paths = {}, {}, {}
    for split in ("validation", "test"):
        report_path = directory / f"{split}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["split"] != split:
                raise ValueError("correction report split mismatch")
            reports[split] = report
            paths[report_path.name] = report_path
            check_path = directory / f"{split}-checks.npz"
            if check_path.exists():
                if report["checks_sha256"] != sha256(check_path):
                    raise ValueError(f"{split}: correction checks checksum mismatch")
                with np.load(check_path, allow_pickle=False) as saved:
                    checks[split] = dict(saved)
                paths[check_path.name] = check_path
    if "test" in reports and "validation" in reports:
        if reports["test"]["validation_report_sha256"] != sha256(paths["validation.json"]):
            raise ValueError("new test is bound to another validation report")
    diagnostic_path = directory / "test-reference-pit.npz"
    audit_path = directory / "test-independent-audit.json"
    if diagnostic_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["reference_pit_checks_sha256"] != sha256(diagnostic_path):
            raise ValueError("reference-PIT diagnostic checksum mismatch")
        with np.load(diagnostic_path, allow_pickle=False) as saved:
            checks["reference_pit"] = dict(saved)
        paths.update(reference_pit=diagnostic_path, independent_test_audit=audit_path)
    return reports, checks, paths


def _finish(figure, title, note):
    figure.suptitle(title, fontsize=13)
    figure.text(.5, .02, note, ha="center", va="bottom", fontsize=8.5)
    figure.tight_layout(rect=(0, .10, 1, .93))
    return figure


def _case_axis(axis, ids):
    labels = [name.replace("validation:", "").replace("test:", "T").replace("stress:", "S") for name in ids]
    axis.set_xticks(np.arange(len(ids)), labels, fontsize=8)
    axis.set_xlim(-.6, len(ids) - .4)
    stress = [index for index, name in enumerate(ids) if name.startswith("stress:")]
    if stress:
        axis.axvspan(stress[0]-.5, stress[-1]+.5, color="#EDF0F4", zorder=-5)


def price_error_figure(reports, baseline):
    available = [(split, reports[split]) for split in ("validation", "test")
                 if reports.get(split, {}).get("cases")]
    figure, axes = plt.subplots(len(available), 1, figsize=(12.2, 3.5*len(available)+1), squeeze=False)
    for axis, (split, report) in zip(axes[:, 0], available):
        cases = report["cases"]
        ids = [case["case_id"] for case in cases]
        x = np.arange(len(cases))
        errors = np.array([np.max(np.abs(case["target_quantile_error_currency"])) for case in cases])
        axis.plot(x, errors, "o-", color=COLORS["corrected"], ms=4,
                  label=f"Corrected participant ({report['participant_samples']:,} proposals)")
        precision = report.get("precision", {}).get("cases", [])
        if len(precision) == len(cases):
            more_errors = [np.max(np.abs(case["target_quantile_error_currency"])) for case in precision]
            axis.plot(x, more_errors, "s--", color=COLORS["reference"], ms=3,
                      label=f"Corrected precision ({report['precision_samples']:,} proposals)")
        if split == "validation":
            old = {case["case_id"]: case for case in baseline["cases"]}
            if all(case_id in old for case_id in ids):
                axis.plot(x, [np.max(np.abs(old[case_id]["target_quantile_error_currency"])) for case_id in ids],
                          "x-", color=COLORS["raw"], ms=5, label="Preserved raw-neural baseline")
        gate = report["thresholds"]["max_abs_target_quantile_error_currency"]
        axis.axhline(gate, color=COLORS["gate"], ls="--", lw=1, label=f"Unchanged gate: {gate:g}")
        status = "all frozen gates pass" if report["passed"] else "overall acceptance failed; see saved gates"
        axis.set(title=f"{split.capitalize()}: conditional price distribution check ({status})",
                 ylabel="Maximum absolute quantile error\n(currency per option)")
        _case_axis(axis, ids)
        axis.set_xlabel("Conditional panel ID; S denotes fixed stress cases")
        axis.legend(fontsize=8, ncol=2, loc="best")
    return _finish(figure, "Raw versus corrected pricing uncertainty: separate estimator, unchanged tolerance",
                   "Each point maximizes error over two target contracts and 5%/50%/95% weighted price quantiles.\n"
                   "Weighted finite-pool estimates retain Monte Carlo error; the old raw checkpoint and old posterior test are unchanged.")


def calibration_figure(reports, checks):
    available = [(split, reports[split], checks[split]) for split in ("validation", "test")
                 if "calibration_randomized_ranks" in checks.get(split, {})]
    if not available:
        return None
    figure, axes = plt.subplots(len(available), 2, figsize=(11.5, 3.8*len(available)+1), squeeze=False)
    grid = np.linspace(0, 1, 2049)
    for row, (split, report, arrays) in enumerate(available):
        calibration = report["calibration"]
        ranks = np.asarray(arrays["calibration_randomized_ranks"])
        bound = calibration["rank_dkw_bound"]
        for dimension, axis in enumerate(axes[row]):
            empirical = np.searchsorted(np.sort(ranks[:, dimension]), grid, side="right") / len(ranks)
            axis.plot(grid, empirical-grid, color=COLORS["corrected"], lw=1.3)
            axis.axhspan(-bound, bound, color=COLORS["corrected"], alpha=.08)
            for value in (-bound, bound):
                axis.axhline(value, color=COLORS["gate"], ls="--", lw=1)
            axis.axhline(0, color="#8793A0", lw=.7)
            limit = max(1.3*bound, 1.1*float(np.max(np.abs(empirical-grid))))
            axis.set(xlim=(0, 1), ylim=(-limit, limit), xlabel="Randomized SIR rank",
                     ylabel="Empirical CDF − uniform CDF",
                     title=f"{split.capitalize()}: " + [r"long-run variance $\theta$", r"jump intensity $\lambda$"][dimension])
            axis.text(.03, .96, f"Cases: {len(ranks)}; KS: {calibration['rank_ks'][dimension]:.4f}\n"
                      f"DKW limit: {bound:.4f}; 90% coverage: {calibration['coverage_90'][dimension]:.3f}",
                      transform=axis.transAxes, va="top", fontsize=8.5,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    return _finish(figure, "Calibration of the finite-pool sampling-importance-resampling algorithm",
                   "Each case uses 2,048 proposals and 512 joint resamples; resamples share one finite approximation.\n"
                   "DKW limits control two marginals within each split at family α=0.05. Conditional and joint checks remain separate.")


def efficiency_figure(reports, checks, protocol):
    available = [(split, reports[split]) for split in ("validation", "test")
                 if reports.get(split, {}).get("cases")]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
    for split, report in available:
        color = COLORS["corrected"] if split == "validation" else COLORS["reference"]
        cases = report["cases"]
        for column, field in enumerate(("ess_fraction", "max_weight")):
            values = np.sort([case[field] for case in cases])
            axes[0, column].plot(values, np.arange(1, len(values)+1)/len(values), "o-", ms=3,
                                 label=f"{split}: {report['participant_samples']:,} proposals", color=color)
            precision_cases = report.get("precision", {}).get("cases", [])
            if precision_cases:
                precision_values = np.sort([case[field] for case in precision_cases])
                axes[0, column].plot(precision_values, np.arange(1, len(precision_values)+1)/len(precision_values),
                                     "s--", ms=3, label=f"{split}: {report['precision_samples']:,} proposals", color=color, alpha=.7)
        arrays = checks.get(split, {})
        for column, key in enumerate(("calibration_ess_fraction", "calibration_max_weight")):
            if key in arrays:
                values = np.sort(arrays[key])
                axes[1, column].plot(values, np.arange(1, len(values)+1)/len(values),
                                     label=f"{split}: {len(values)} panels", color=color)
    for row, settings in enumerate((protocol["conditional"], protocol["calibration"])):
        axes[row, 0].axvline(settings["min_ess_fraction"], color=COLORS["gate"], ls="--", lw=1)
        axes[row, 1].axvline(settings["max_weight"], color=COLORS["gate"], ls="--", lw=1)
        for column, axis in enumerate(axes[row]):
            axis.set(ylabel="Fraction of panels at or below x", ylim=(0, 1.05),
                     xlabel=["Importance ESS / proposal count", "Largest normalized weight"][column],
                     title=["Conditional participant and precision pools", "Calibration pools"][row])
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8, loc="best")
            else:
                message = ("Calibration not run: conditional acceptance failed"
                           if any(report.get("calibration", {}).get("skipped") for _, report in available)
                           else "Calibration arrays are not available; see report for run status")
                axis.text(.5, .5, message,
                          ha="center", va="center", transform=axis.transAxes, fontsize=9,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95})
    return _finish(figure, "Proposal efficiency for the participant and calibration budgets",
                   "ESS fractions must exceed their dashed minimum; maximum weights must fall below their dashed limit.\n"
                   "High observed efficiency cannot rule out a missed mode; independent reference comparisons still determine accuracy.")


def live_joint_figure(live):
    grid = live["grid"]
    mass = live["reference_weights"]
    density = mass / grid["quadrature_weights"]
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(mass[order])
    levels = [float(density[order[np.searchsorted(cumulative, probability)]]) for probability in (.9, .5)]
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    weights = live["weights"]
    sizes = 3 + 35*weights/weights.max()
    axis.scatter(live["parameters"][:, 0], live["parameters"][:, 1], s=sizes,
                 color=COLORS["corrected"], alpha=.2, edgecolors="none", label="Live proposals; size follows weight")
    contours = axis.contour(*grid["axes"], density.reshape(grid["shape"]).T,
                            levels=levels, colors=[COLORS["reference"]]*2, linewidths=[1.4, 2])
    axis.clabel(contours, fmt={level: label for level, label in zip(levels, ("90% reference", "50% reference"))}, fontsize=8)
    for mean, marker, color, label in ((live["summary"]["mean"], "D", COLORS["corrected"], "Corrected weighted mean"),
                                      (live["reference_summary"]["mean"], "X", COLORS["reference"], "Reference mean")):
        axis.scatter(*mean, marker=marker, s=85, color=color, edgecolors="white", linewidths=.6, label=label, zorder=5)
    axis.set(xlabel=r"Long-run variance $\theta$", ylabel=r"Jump intensity $\lambda$ (per year)",
             xlim=(grid["axes"][0][0], grid["axes"][0][-1]), ylim=(grid["axes"][1][0], grid["axes"][1][-1]))
    axis.legend(fontsize=8, loc="best")
    return _finish(figure, f"{live['case_id']}: live joint correction against the independent grid",
                   "Contours contain 50%/90% of numerical reference mass. Point sizes are illustrative weights, not density estimates.\n"
                   "The full rectangle is the prior support; θ and λ remain paired throughout target pricing.")


def live_target_figure(live):
    figure, axes = plt.subplots(1, 2, figsize=(11.3, 4.5))
    for target, axis in enumerate(axes):
        prices = live["target_prices"][:, target]
        reference_prices = live["grid"]["target_prices"][:, target]
        bins = np.linspace(min(prices.min(), reference_prices.min()), max(prices.max(), reference_prices.max()), 45)
        axis.hist(reference_prices, bins=bins, weights=live["reference_weights"], density=True,
                  histtype="step", color=COLORS["reference"], lw=1.7, label="Independent reference")
        axis.hist(prices, bins=bins, weights=live["weights"], density=True, color=COLORS["corrected"],
                  alpha=.35, label="Live correction + surrogate")
        contract = live["targets"][target]
        axis.set(title=f"Put K={contract['strike']:g}, T={contract['maturity']:g} years",
                 xlabel="Expected option price (currency)", ylabel="Weighted posterior density")
    axes[0].legend(fontsize=8)
    return _finish(figure, "Parameter uncertainty propagated into two fixed target prices",
                   "Each weighted row is a numerical expectation price at one joint parameter pair, not one realized payoff.\n"
                   "Histograms visualize parameter-induced price uncertainty; finite-pool error and numerical price error remain separate.")


def reference_pit_figure(arrays, bound):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
    grid = np.linspace(0, 1, 2049)
    for dimension, axis in enumerate(axes):
        for key, label, color, style in (
            ("reference_pit_129", "Numerical reference PIT (129 grid)", COLORS["reference"], "-"),
            ("sir_randomized_ranks", "Finite-pool SIR ranks", COLORS["corrected"], "--"),
        ):
            values = np.sort(arrays[key][:, dimension])
            cdf = np.searchsorted(values, grid, side="right") / len(values)
            axis.plot(grid, cdf-grid, color=color, ls=style, lw=1.4, label=label)
        for value in (-bound, bound):
            axis.axhline(value, color=COLORS["gate"], ls=":", lw=1)
        axis.axhline(0, color="#8793A0", lw=.7)
        axis.set(xlim=(0, 1), ylim=(-.12, .12), xlabel="Posterior CDF at truth / randomized rank",
                 ylabel="Empirical CDF − uniform CDF",
                 title=[r"Long-run variance $\theta$", r"Jump intensity $\lambda$"][dimension])
        axis.legend(fontsize=8, loc="upper left")
    return _finish(figure, "The same held-out cohort shows the rank shift under the numerical reference",
                   "Post-hoc explanation using saved panels and numerical CDF interpolation; no new test or proposal draws.\n"
                   "The reference shares the θ failure. This supports investigating cohort variability; the frozen test is still failed.")


def figures(reports, checks, baseline, protocol):
    result = {}
    if any(report.get("cases") for report in reports.values()):
        result["price-errors"] = price_error_figure(reports, baseline)
        result["proposal-efficiency"] = efficiency_figure(reports, checks, protocol)
    calibration = calibration_figure(reports, checks)
    if calibration is not None:
        result["calibration"] = calibration
    if "reference_pit" in checks and "test" in reports:
        result["reference-pit"] = reference_pit_figure(checks["reference_pit"], reports["test"]["calibration"]["rank_dkw_bound"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/results/hkde-correction"))
    parser.add_argument("--out", type=Path, help="new directory; default RESULTS/plots")
    parser.add_argument("--protocol", type=Path, default=Path("configs/hkde_correction_boundary_protocol.json"))
    args = parser.parse_args()
    root = args.protocol.resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    baseline_path = root / protocol["baseline_results"] / "neural-validation.json"
    baseline = json.loads(baseline_path.read_text())
    reports, checks, paths = load_cache(args.results)
    if any(report["experiment_id"] != protocol["experiment_id"] for report in reports.values()):
        parser.error("saved reports belong to a different protocol; select its --protocol explicitly")
    rendered = figures(reports, checks, baseline, protocol)
    if not rendered:
        parser.error("no saved corrected results found; no inference was run")
    destination = args.out or args.results / "plots"
    destination.mkdir(parents=True, exist_ok=False)
    plt.rcParams["svg.hashsalt"] = "hkde-defensive-correction-v1"
    artifacts = {}
    for name, figure in rendered.items():
        for extension in ("png", "svg"):
            path = destination / f"{name}.{extension}"
            figure.savefig(path, dpi=170, metadata={"Date": None} if extension == "svg" else None)
            artifacts[path.name] = sha256(path)
        plt.close(figure)
    paths.update(protocol=args.protocol, baseline=baseline_path)
    manifest = {"scope": "cached synthetic Q correction diagnostics; no new inference/test access",
                "input_sha256": {name: sha256(path) for name, path in paths.items()},
                "plot_script_sha256": sha256(__file__), "artifact_sha256": artifacts, "figures": list(rendered)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

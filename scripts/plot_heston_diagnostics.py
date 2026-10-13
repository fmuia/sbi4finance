"""Plot saved Heston development diagnostics without simulation or inference.

By default reads only committed validation reports, rank arrays and a prior
summary. Optional --prior-bank reproduces that summary from manifest.json and
train.npz only. The final test split is never accessed. Outputs are four PNGs
and a small source/artifact manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/results/heston-time-context"
DEFAULT_OUT = ROOT / "docs/results/heston-diagnostic-review"
ARTIFACTS = ("sbc-rank-ecdf.png", "conditional-reference-errors.png", "gru-gate-saturation.png",
             "prior-posterior-comparison.png")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_prior(bank, report, report_path):
    """Read only the exact training file's terminal states; no return conditioning."""
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (file_hash(manifest_path) != report["bank_manifest_sha256"]
            or manifest["configuration"]["task"] != report["task"]):
        raise ValueError("prior bank must match the recorded neural experiment")
    train = manifest["splits"]["train"]
    train_path = bank / "train.npz"
    if train["file"] != "train.npz" or file_hash(train_path) != train["sha256"]:
        raise ValueError("training artifact differs from its manifest")
    with np.load(train_path, allow_pickle=False) as saved:
        state = saved["terminal_state"]
        ids = saved["example_id"]
    count = train["examples"]
    if (state.shape != (count, 1) or not np.isfinite(state).all()
            or ids.tolist() != [f"train:{i}" for i in range(count)]):
        raise ValueError("training terminal-state shape, values or IDs differ")
    task = report["task"]
    if task["prior"] != "log_uniform":
        raise ValueError("this prior summary requires log-uniform theta")
    low, high = task["theta_bounds"]
    theta_mean = (high - low) / np.log(high / low)
    theta_std = np.sqrt((high**2 - low**2) / (2 * np.log(high / low)) - theta_mean**2)
    means = np.array([theta_mean, state.mean()])
    stds = np.array([theta_std, state.std(ddof=1)])
    cases = report["evaluation"]["reference_cases"]
    errors = np.array([(means - case["reference"]["mean"]) / case["reference"]["std"] for case in cases])
    return {
        "scope": "Unconditional joint-model prior versus conditional posteriors; descriptive evidence of data updating, not an acceptance test",
        "target_order": report["evaluation"]["target_order"], "prior_mean": means.tolist(), "prior_std": stds.tolist(),
        "theta_prior": "Analytic moments of log_uniform(theta_bounds)",
        "state_prior": f"Empirical marginal terminal signed state from all {count} training-bank prior/simulator pairs; fixed initial variance and one year dynamics, no conditioning on observed returns",
        "prior_sample_count": count, "state_prior_mean_mcse": float(stds[1] / np.sqrt(count)),
        "bank_manifest_sha256": file_hash(manifest_path),
        "prior_source_file": "train.npz only; test split not opened", "prior_source_sha256": file_hash(train_path),
        "neural_report_sha256": file_hash(report_path),
        "prior_copy_reference_mean_rmse_in_sd": float(np.sqrt(np.mean(errors**2))),
        "neural_reference_mean_rmse_in_sd": report["evaluation"]["metrics"]["mean_rmse_in_reference_sd"],
        "cases": [{"example_id": case["example_id"], "prior_copy_mean_error_in_reference_sd": error.tolist(),
                   "neural_std_divided_by_prior_std": (np.array(case["neural_std"]) / stds).tolist()}
                  for case, error in zip(cases, errors)],
    }


def load_evidence(prior_path):
    paths = {
        "validation": EVIDENCE / "neural-validation.json",
        "baseline": ROOT / "docs/results/heston-inference-followup/neural-validation.json",
        "checks": EVIDENCE / "validation-checks.npz",
        "legacy": EVIDENCE / "summary-legacy.json",
        "bounded": EVIDENCE / "summary-explicit-time.json",
        "prior": prior_path,
    }
    reports = {key: json.loads(path.read_text()) for key, path in paths.items() if key != "checks"}
    current, baseline = reports["validation"], reports["baseline"]
    for key in ("bank_manifest_sha256", "reference_report_sha256", "task", "split"):
        if current[key] != baseline[key]:
            raise ValueError(f"comparison mismatch: {key}")
    if current["split"] != "validation" or current["evaluation"]["thresholds"] != baseline["evaluation"]["thresholds"]:
        raise ValueError("expected the unchanged validation design and thresholds")
    expected = [f"validation:{i}" for i in range(16)]
    for key, neural in (("legacy", baseline), ("bounded", current)):
        diagnostic = reports[key]
        if (diagnostic["split"] != "validation" or diagnostic["reserved_test_accessed"]
                or diagnostic["example_ids"] != expected or diagnostic["task"] != current["task"]
                or diagnostic["source_sha256"]["checkpoint/model.keras"] != neural["checkpoint_sha256"]):
            raise ValueError(f"diagnostic identity mismatch: {key}")
    if (reports["legacy"]["root_seed"] != reports["bounded"]["root_seed"]
            or reports["legacy"]["source_sha256"]["config.json"]
            != reports["bounded"]["source_sha256"]["config.json"]):
        raise ValueError("gate diagnostics must use the same configured examples")

    evaluation = current["evaluation"]
    count, draws = evaluation["cases"], evaluation["draws_per_case"]
    if (count, draws) != (256, 2048):
        raise ValueError("this bounded review expects 256 histories and 2,048 draws each")
    with np.load(paths["checks"], allow_pickle=False) as saved:
        # Deliberately do not load the saved future outcomes or any bank/test data.
        ranks = saved["ranks"]
        ids = saved["example_id"]
        means, stds = saved["posterior_mean"], saved["posterior_std"]
    if (ranks.shape != (count, 2) or not np.issubdtype(ranks.dtype, np.integer)
            or np.any(ranks < 0) or np.any(ranks > draws)
            or ids.tolist() != [f"validation:{i}" for i in range(count)]):
        raise ValueError("saved rank arrays do not match the reported design")
    cases = evaluation["reference_cases"]
    if [case["example_id"] for case in cases] != expected:
        raise ValueError("conditional checks must retain all 16 preselected histories")
    np.testing.assert_allclose(means[:16], [case["neural_mean"] for case in cases], rtol=0, atol=1e-14)
    np.testing.assert_allclose(stds[:16], [case["neural_std"] for case in cases], rtol=0, atol=1e-14)
    prior = reports["prior"]
    if (prior["bank_manifest_sha256"] != current["bank_manifest_sha256"]
            or prior["neural_report_sha256"] != file_hash(paths["validation"])
            or prior["target_order"] != evaluation["target_order"]
            or [case["example_id"] for case in prior["cases"]] != expected):
        raise ValueError("prior summary is not bound to this bank and neural report")
    if (np.shape(prior["prior_mean"]) != (2,) or np.shape(prior["prior_std"]) != (2,)
            or not np.isfinite([prior["prior_mean"], prior["prior_std"]]).all()
            or np.any(np.array(prior["prior_std"]) <= 0)):
        raise ValueError("prior summary needs two finite means and positive SDs")

    uniform_ranks = (ranks + np.random.default_rng(evaluation["seed"] + 1).random(ranks.shape)) / (draws + 1)
    ordered = np.sort(uniform_ranks, axis=0)
    deviations = np.maximum(
        np.max(np.arange(1, count + 1)[:, None] / count - ordered, axis=0),
        np.max(ordered - np.arange(count)[:, None] / count, axis=0),
    )
    np.testing.assert_allclose(deviations, evaluation["rank_uniformity_max_ecdf_error"], rtol=0, atol=1e-14)
    return reports, paths, ordered, deviations


def save_rank_plot(ordered, evaluation, out):
    count = len(ordered)
    bound = evaluation["rank_dkw_bound_two_targets"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    for column, (ax, label) in enumerate(zip(axes, ("Long-run variance θ", "Signed terminal state aₜ"))):
        values = ordered[:, column]
        # Both sides of each ECDF jump: exact piecewise-linear F_n(u) - u.
        x = np.r_[0.0, np.repeat(values, 2), 1.0]
        y = np.r_[0.0, np.column_stack((np.arange(count) / count - values,
                                       np.arange(1, count + 1) / count - values)).ravel(), 0.0]
        ax.axhspan(-bound, bound, color="#dce5ee", alpha=0.75, label="Recorded DKW diagnostic band")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.plot(x, y, color="#35688d", linewidth=1.4, label="Marginal rank ECDF − uniform")
        ax.set(title=label, xlabel="Randomized normalized rank u", xlim=(0, 1), ylim=(-0.12, 0.12))
        ax.text(0.03, 0.04, f"Max deviation = {evaluation['rank_uniformity_max_ecdf_error'][column]:.4f}\nBand = ±{bound:.4f}",
                transform=ax.transAxes, fontsize=10)
    axes[0].set_ylabel("Fₙ(u) − u")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Marginal SBC ranks: 256 joint prior/simulator histories × 2,048 neural draws", fontsize=13)
    fig.text(0.5, 0.025, "DEVELOPMENT VALIDATION reused for model selection — not an independent calibration certificate or a joint/conditional guarantee.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.83, bottom=0.20, wspace=0.12)
    fig.savefig(out / ARTIFACTS[0], dpi=160)
    plt.close(fig)


def save_conditional_plot(evaluation, out):
    records = evaluation["reference_cases"]
    means = np.array([case["mean_error_in_reference_sd"] for case in records])
    endpoints = np.array([case["predictive_interval_error_in_sd"] for case in records])
    np.testing.assert_allclose(np.max(np.abs(means)), evaluation["metrics"]["maximum_abs_mean_error_in_reference_sd"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.max(np.abs(endpoints)), evaluation["metrics"]["maximum_predictive_interval_error_in_sd"], rtol=0, atol=1e-14)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    designs = [
        (means, ("θ mean", "Signed-state mean"), "maximum_abs_mean_error_in_reference_sd",
         "Mean error / corresponding\nreference posterior SD", "Parameter/state conditional means"),
        (endpoints, ("5% loss quantile", "95% loss quantile"), "maximum_predictive_interval_error_in_sd",
         "Endpoint error / reference\nfuture-loss SD", "Ten-day predictive interval endpoints"),
    ]
    x = np.arange(len(records))
    for ax, (values, labels, gate, ylabel, title) in zip(axes, designs):
        limit = evaluation["thresholds"][gate]
        ax.axhspan(-limit, limit, color="#eeeeee", zorder=0)
        ax.axhline(limit, color="#666666", linestyle="--", linewidth=1, label=f"Declared pilot tolerance ±{limit:.2f}")
        ax.axhline(-limit, color="#666666", linestyle="--", linewidth=1)
        ax.axhline(0, color="#333333", linewidth=0.8)
        for i, color in enumerate(("#4d749b", "#c06b48")):
            ax.bar(x + (i - 0.5) * 0.35, values[:, i], width=0.35, color=color, label=labels[i])
        ax.set(ylabel=ylabel, title=title, xlim=(-0.7, len(records) - 0.3))
        ax.legend(loc="upper right", fontsize=8, ncols=3)
    axes[1].set(xticks=x, xticklabels=[str(i) for i in x], xlabel="Preselected validation history ID (validation:0 through validation:15)")
    fig.suptitle("Conditional reference comparisons — NOT SBC", fontsize=14)
    fig.text(0.5, 0.025, "DEVELOPMENT VALIDATION • Errors are neural minus finite reference; the two panels use different SD denominators.\nThe 0.40 / 0.15 limits are declared pilot engineering tolerances, not universal statistical tests.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.17, left=0.13, hspace=0.30)
    fig.savefig(out / ARTIFACTS[1], dpi=160)
    plt.close(fig)


def save_gate_plot(reports, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    for ax, gate in zip(axes, ("update", "reset")):
        for key, label, color in (("legacy", "Raw-index spline", "#4d749b"),
                                  ("bounded", "Bounded-time spline", "#c06b48")):
            records = reports[key]["gate_diagnostics"]
            ax.plot([r["time_index"] + 1 for r in records],
                    [100 * r[gate]["fraction_abs_preactivation_above_5"] for r in records],
                    marker="o", color=color, label=label)
        ax.set(title=f"GRU {gate} gate", xlabel="Observed day", xticks=[1, 32, 128, 252], ylim=(-2, 102))
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Preactivations with |z| > 5 (%)")
    fig.suptitle("Engineering diagnosis: bounded time removes the measured gate saturation", fontsize=13)
    fig.text(0.5, 0.025, "Same 16 validation histories; four recorded positions; separately trained weights.\nThis is NOT SBC and is not an inference acceptance gate. The bounded-time posterior still fails validation.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.83, bottom=0.21, wspace=0.12)
    fig.savefig(out / ARTIFACTS[2], dpi=160)
    plt.close(fig)


def save_prior_plot(evaluation, prior, out):
    cases = evaluation["reference_cases"]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    for column, (ax, label) in enumerate(zip(axes, ("Long-run variance θ", "Signed terminal variance state aₜ"))):
        mean, std = prior["prior_mean"][column], prior["prior_std"][column]
        ax.axhspan(mean - std, mean + std, color="#dddddd", alpha=0.75, label="Unconditional prior mean ± SD")
        ax.axhline(mean, color="#777777", linestyle="--", linewidth=1.2)
        ax.errorbar(x - 0.13, [c["reference"]["mean"][column] for c in cases],
                    yerr=[c["reference"]["std"][column] for c in cases],
                    fmt="o", capsize=3, markersize=4, color="#333333", label="Reference posterior mean ± SD")
        ax.errorbar(x + 0.13, [c["neural_mean"][column] for c in cases],
                    yerr=[c["neural_std"][column] for c in cases],
                    fmt="o", capsize=3, markersize=4, color="#c06b48", label="Neural posterior mean ± SD")
        ax.set(ylabel="Variance units (annualized)", title=label, xlim=(-0.7, len(cases) - 0.3))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncols=3, fontsize=9)
    axes[1].set(xticks=x, xticklabels=[str(i) for i in x], xlabel="Preselected validation history ID (validation:0 through validation:15)")
    fig.suptitle("The posterior changes with the observed history; it is not a copy of the prior", fontsize=13)
    fig.text(0.5, 0.03, f"DEVELOPMENT VALIDATION • Means ± one standard deviation, NOT credible intervals, support bounds or synthetic truths.\nTheta prior: analytic log-uniform. Terminal-state prior: {prior['prior_sample_count']:,} unconditional training simulations.\nSome theta posteriors remain broad. Visible data updating does not establish full joint or conditional accuracy.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.11, hspace=0.34)
    fig.savefig(out / ARTIFACTS[3], dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prior-bank", type=Path,
                        help="optionally reproduce prior-comparison.json from matching manifest.json and train.npz only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / name).exists() for name in (*ARTIFACTS, "figures-manifest.json")):
        raise FileExistsError("choose a directory without existing diagnostic figure artifacts")
    prior_path = DEFAULT_OUT / "prior-comparison.json"
    if args.prior_bank is not None:
        report_path = EVIDENCE / "neural-validation.json"
        prior_path = args.out / "prior-comparison.json"
        prior = summarize_prior(args.prior_bank, json.loads(report_path.read_text()), report_path)
        prior_path.write_text(json.dumps(prior, indent=2) + "\n")
    reports, sources, ordered, deviations = load_evidence(prior_path)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    evaluation = reports["validation"]["evaluation"]
    save_rank_plot(ordered, evaluation, args.out)
    save_conditional_plot(evaluation, args.out)
    save_gate_plot(reports, args.out)
    save_prior_plot(evaluation, reports["prior"], args.out)
    manifest = {
        "purpose": "saved bounded-time Heston development diagnostics; no inference, simulation or test evaluation",
        "split": "validation", "reserved_test_accessed": False,
        "bank_manifest_sha256": reports["validation"]["bank_manifest_sha256"],
        "reference_report_sha256": reports["validation"]["reference_report_sha256"],
        "checkpoint_sha256": reports["validation"]["checkpoint_sha256"],
        "source_sha256": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): file_hash(path)
                          for path in sources.values()},
        "plot_script_sha256": file_hash(Path(__file__)),
        "artifacts_sha256": {name: file_hash(args.out / name) for name in ARTIFACTS},
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "rank_reconstruction": {"jitter_seed": evaluation["seed"] + 1,
                                "draws": evaluation["draws_per_case"], "cases": evaluation["cases"],
                                "max_ecdf_errors": deviations.tolist(),
                                "dkw_bound": evaluation["rank_dkw_bound_two_targets"]},
        "limits": "Validation was reused for selection; marginal SBC diagnostics are not an independent certificate. "
                  "Conditional errors use a finite reference; gate saturation is an engineering diagnostic.",
    }
    (args.out / "figures-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "artifacts": list(ARTIFACTS)}, indent=2))


if __name__ == "__main__":
    main()

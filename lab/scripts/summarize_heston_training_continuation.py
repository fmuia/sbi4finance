"""Source-bound summary of the longer Heston trajectory and its failed endpoint.

Read only saved development evaluations, density scores, curves and execution
records. Original results and model payloads are never modified. Missing 100k
results are represented as unavailable, not as failed posterior comparisons.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "src"))

import numpy as np

from scripts.summarize_heston_training_study import compact_comparison


PHYSICAL = ["initial_variance", "kappa", "theta", "vol_of_vol", "rho"]
DISPLAY = [r"$v_0$", r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def read_cell(root, updates):
    directory = root / "evaluations" / f"n100000-u{updates:06d}"
    path = directory / "evaluation.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    checkpoint = raw["checkpoint"]
    if checkpoint["train_rows"] != 100000 or checkpoint["actual_optimizer_updates"] != updates:
        raise ValueError("evaluation is labelled with the wrong training cell")
    payload = directory / "draws.npz"
    if digest(payload) != raw["draws_sha256"]:
        raise ValueError("evaluation draw payload changed after analysis")
    with np.load(payload, allow_pickle=False) as archive:
        truth = np.asarray(archive["validation_truth"], dtype="<f8")
        observed = np.asarray(archive["validation_observations"], dtype="<f8")
        market = np.asarray(archive["market_draws"])
    count = raw["settings"]["cases"]
    if truth.shape != (count, 6) or observed.shape != (count, 54) or market.shape[1:] != (6,):
        raise ValueError("development truth, observation or market dimensions differ")
    if not np.isfinite(truth).all() or not np.isfinite(observed).all() or not np.isfinite(market).all():
        raise ValueError("finite saved development arrays are required")
    tau = raw["market"]["parameters"]["tau"]
    if (not np.isclose(market[:, 5].astype(float).mean(), tau["neural_mean"], rtol=0., atol=1e-12)
            or not np.allclose(np.quantile(market[:, 5], [.05, .95]), tau["neural_q05_q95"], rtol=0., atol=1e-12)):
        raise ValueError("saved market discrepancy summaries differ from the evaluated draws")
    return {"raw": raw, "updates": updates, "evaluation_sha256": digest(path),
            "development_truth_sha256": value_digest(truth),
            "development_observations_sha256": value_digest(observed),
            "draws_sha256": raw["draws_sha256"], "market_reference_identity": raw["market"]["identity"]}


def build_report(args):
    root, out = args.root.resolve(), args.out.resolve()
    continuation = root / "100k-continuation"
    density_path = out / "continuation-density-score.json"
    protocol_path = out / "continuation-protocol.json"
    density = json.loads(density_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    panel_path = out / "reference-panel.json"
    panel = json.loads(panel_path.read_text())
    bank_path = root / "bank" / "bank.json"
    bank = json.loads(bank_path.read_text())
    cells = {updates: read_cell(root, updates) for updates in (50000, 75000, 100000)}
    if cells[50000] is None or cells[75000] is None:
        raise ValueError("the baseline and valid 75k evaluation must be complete")
    if cells[100000] is not None or protocol["status"] != "training_failed":
        raise ValueError("this report is bound to the preserved failed continuation, with no 100k checkpoint")
    present = {step: cell for step, cell in cells.items() if cell is not None}
    identities = {(cell["raw"]["checkpoint"]["task_fingerprint"], cell["raw"]["cohort_sha256"],
                   cell["development_truth_sha256"], cell["development_observations_sha256"],
                   cell["raw"]["settings"]["seed"], cell["raw"]["script_sha256"])
                  for cell in present.values()}
    if len(identities) != 1:
        raise ValueError("continuation cells differ in target, actual development data or evaluation protocol")
    baseline_checkpoint = present[50000]["raw"]["checkpoint"]
    run_path = continuation / "run.json"
    run = json.loads(run_path.read_text())
    if (run["parent"]["model_sha256"] != baseline_checkpoint["model_sha256"]
            or run["task_fingerprint"] != baseline_checkpoint["task_fingerprint"]
            or run["schedule_updates"] != 50000 or run["start_updates"] != 50000
            or run["planned_updates"] != 100000 or protocol["parent_model_sha256"] != baseline_checkpoint["model_sha256"]):
        raise ValueError("continuation parent or unchanged optimizer schedule differs")
    for step, cell in present.items():
        raw = cell["raw"]
        cp = raw["checkpoint"]
        if (cp["task_fingerprint"] != bank["identity"]["task_fingerprint"]
                or cp["bank_payload_sha256"] != bank["payload_sha256"]
                or cp["bank_record_sha256"] != digest(bank_path)
                or raw["reference_panel_sha256"] != digest(panel_path)
                or cell["market_reference_identity"] != present[50000]["market_reference_identity"]):
            raise ValueError("bank, conditional panel or market reference identity differs")
        if step > 50000 and cp["continuation_parent"]["model_sha256"] != baseline_checkpoint["model_sha256"]:
            raise ValueError("continued checkpoint has a different parent")
    if density["baseline_cell"]["model_sha256"] != baseline_checkpoint["model_sha256"]:
        raise ValueError("density-score baseline differs from the evaluated baseline model")
    logp_path = Path(density["logp_payload_path"])
    if digest(logp_path) != density["logp_payload_sha256"]:
        raise ValueError("physical density payload changed after scoring")
    with np.load(logp_path, allow_pickle=False) as archive:
        logp = {int(key[1:]): np.asarray(archive[key], dtype=float) for key in archive.files if key.startswith("u")}
    for step, cell in present.items():
        record = density["baseline_cell"] if step == 50000 else density["cells"][str(step)]
        if (record["model_sha256"] != cell["raw"]["checkpoint"]["model_sha256"]
                or record["evaluation_record_sha256"] != cell["evaluation_sha256"]
                or record["evaluation_payload_sha256"] != cell["draws_sha256"]
                or record["truth_sha256"] != cell["development_truth_sha256"]
                or record["observations_sha256"] != cell["development_observations_sha256"]
                or step not in logp or not np.isfinite(logp[step]).all()
                or not np.isclose(-logp[step].mean(), record["mean_physical_nll"], rtol=0., atol=1e-12)):
            raise ValueError("physical density score and checkpoint/development identity differ")
        if step > 50000:
            delta = logp[50000] - logp[step]
            pair = density["paired_later_minus_50k"][str(step)]
            if (not np.isclose(delta.mean(), pair["mean_nll_difference"], rtol=0., atol=1e-12)
                    or not np.isclose(delta.std(ddof=1)/np.sqrt(len(delta)), pair["paired_observation_standard_error"], rtol=0., atol=1e-12)):
                raise ValueError("paired physical density-score extraction differs from its saved observations")
    parent_curve_path = root / "train-n100000" / "curve.json"
    curve_path = continuation / "curve.json"
    parent_points = json.loads(parent_curve_path.read_text())["points"]
    later_points = json.loads(curve_path.read_text())["points"]
    if parent_points[-1]["updates"] != 50000 or later_points[0]["updates"] != 50000:
        raise ValueError("learning curves do not join at the parent endpoint")
    for key in ("train_probe_loss", "validation_probe_loss"):
        if not np.isclose(parent_points[-1][key], later_points[0][key], rtol=1e-6, atol=1e-6):
            raise ValueError("fixed continuation probes differ from the parent probes")
    points = parent_points + later_points[1:]
    if (any(b["updates"] <= a["updates"] for a, b in zip(points, points[1:]))
            or any(not np.isfinite(p[key]) for p in points for key in ("train_probe_loss", "validation_probe_loss"))):
        raise ValueError("finite strictly ordered logged learning curves are required")
    unavailable = {}
    if cells[100000] is None:
        failure = density.get("unavailable_endpoints", {}).get("100000")
        log_path = root / "continuation.log"
        if (protocol["status"] != "training_failed" or failure is None
                or failure["status"] != "training_failed_no_checkpoint"
                or failure["continuation_log_sha256"] != digest(log_path)
                or failure["curve_sha256"] != digest(curve_path)
                or failure["continuation_run_sha256"] != digest(run_path)
                or "FloatingPointError: nonfinite continuation curve" not in log_path.read_text()
                or (continuation / "checkpoint.json").exists()):
            raise ValueError("the missing 100k endpoint is not bound to the recorded stopped run")
        if failure["last_recorded_finite_curve_point"] != points[-1]:
            raise ValueError("failure record and final finite learning-curve point differ")
        unavailable["100000"] = {**failure, "endpoint_evaluated": False,
            "first_nonfinite_optimizer_update": None,
            "diagnostic_limit": "The exact first nonfinite update or curve component is not established by the monitor exception."}
    compact = {}
    for step, cell in present.items():
        raw = cell["raw"]
        tau = raw["market"]["parameters"]["tau"]
        compact[str(step)] = {"updates": step, "model_sha256": raw["checkpoint"]["model_sha256"],
            "task_fingerprint": raw["checkpoint"]["task_fingerprint"],
            "evaluation_sha256": cell["evaluation_sha256"], "draws_sha256": cell["draws_sha256"],
            "development_truth_sha256": cell["development_truth_sha256"],
            "development_observations_sha256": cell["development_observations_sha256"],
            "market": compact_comparison(raw["market"]),
            "market_tau": {key: tau[key] for key in ("neural_mean", "neural_q05_q95", "reference_mean", "reference_q05_q95")},
            "conditional_references": {case["id"]: compact_comparison(raw["conditional_references"][case["id"]]) for case in panel["cases"]},
            "parameter_calibration": raw["parameter_calibration"], "product_calibration": raw["product_calibration"],
            "neural_support": raw["neural_support"],
            "numeric_support_boundary_draw_count": raw["numeric_support_boundary_draw_count"]}
    return {"format": "heston_training_continuation_results_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_acceptance": False, "reserved_final_2000_opened": False,
            "scope": "Same 100k-row bank, one training trajectory/seed, unchanged 50k learning-rate horizon; development evidence only. The unreached 100k endpoint has no model or evaluation.",
            "source_sha256": {"summary": digest(__file__), "compact_comparison": digest(LAB/"scripts/summarize_heston_training_study.py")},
            "input_sha256": {"bank_record": digest(bank_path), "panel": digest(panel_path),
                "parent_curve": digest(parent_curve_path), "continuation_curve": digest(curve_path),
                "continuation_run": digest(run_path), "continuation_protocol": digest(protocol_path), "density_score": digest(density_path)},
            "parent_model_sha256": baseline_checkpoint["model_sha256"], "cells": compact,
            "unavailable_endpoints": unavailable, "merged_fixed_probe_curve": points,
            "physical_density": density, "case_categories": {case["id"]: case["category"] for case in panel["cases"]},
            "interpretation": ["Continued optimization improved average development physical NLL at the valid 75k checkpoint.",
                "Average density improvement does not establish conditional posterior agreement; coordinates and products are assessed separately.",
                "The trajectory ended on a nonfinite curve check after the final finite 87k logged probe; the valid 75k model remains available.",
                "No capacity limit or optimizer failure mechanism is inferred from these results."],
            "limits": ["Case-level agreement combines six coordinates and two products; a collective failure does not mean every coordinate fails.",
                "Six of twelve declared synthetic cases were run; the other six remain not run.",
                "Physical-NLL intervals are paired development-observation uncertainty; parameter comparison bars are approximate simultaneous Monte Carlo intervals; tau bars are posterior central90% intervals.",
                "Pointwise numerical precision, MCMC convergence and neural approximation quality are separate checks."]}


def render(report, path):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"sbi4f-continuation-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.7))
    steps = sorted(int(step) for step in report["cells"])
    colors = {50000: "#225f9f", 75000: "#ca6519", 100000: "#24816a"}
    curve = report["merged_fixed_probe_curve"]
    ax = axes[0, 0]
    ax.plot([p["updates"]/1000 for p in curve], [p["validation_probe_loss"] for p in curve], color="#225f9f", label="Validation probe")
    ax.plot([p["updates"]/1000 for p in curve], [p["train_probe_loss"] for p in curve], color="#777777", linestyle="--", label="Training probe")
    ax.axvline(50, color="#888888", linestyle=":", linewidth=1)
    last = curve[-1]["updates"]/1000
    if report["unavailable_endpoints"]:
        ax.axvspan(last, 100, color="#b92525", alpha=.07)
        ax.annotate("Run stopped; last\nfinite probe at 87k", xy=(last, curve[-1]["validation_probe_loss"]),
                    xytext=(48, -2.5), arrowprops={"arrowstyle": "->", "color": "#8c2929"}, color="#8c2929")
    ax.set(xlim=(0, 100), xlabel="Total optimizer updates (thousands)", ylabel="Standardized-target NLL",
           title="A  Fixed probes continued to improve")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.2)

    ax = axes[0, 1]
    ax.axhline(0, color="#777777", linewidth=1)
    for step in steps:
        if step == 50000:
            mean, low, high = 0., 0., 0.
        else:
            pair = report["physical_density"]["paired_later_minus_50k"][str(step)]
            mean = pair["mean_nll_difference"]
            low, high = pair["normal_approximation_ci95"]
        ax.errorbar(step/1000, mean, yerr=[[mean-low], [high-mean]], fmt="o", color=colors[step], capsize=5)
    if report["unavailable_endpoints"]:
        ax.text(100, -.01, "No 100k\ncheckpoint", ha="center", va="top", color="#777777", fontsize=9)
    ax.set(xticks=[50, 75, 100], xlim=(43, 108), xlabel="Total optimizer updates (thousands)",
           ylabel="Physical NLL change vs 50k (nats/observation)", title="B  Average density score improved at 75k")
    ax.grid(alpha=.2)

    for row, metric, interval, band, title, ylabel in (
            (1, "signed_standardized_mean_difference", "signed_standardized_mean_interval", (-.3, .3),
             "C  Market: physical parameter mean errors", "Mean error / reference posterior SD"),
            (2, "width_ratio_90", "width_ratio_interval", (.75, 1.25),
             "E  Market: physical parameter interval widths", "Neural / reference 90% width")):
        ax = axes[row, 0]
        ax.axhspan(*band, color="#3a8d54", alpha=.12, label="Declared agreement range")
        for j, step in enumerate(steps):
            parameters = report["cells"][str(step)]["market"]["parameters"]
            values = np.array([parameters[name][metric] for name in PHYSICAL])
            bounds = np.array([parameters[name][interval] for name in PHYSICAL]).T
            x = np.arange(5) + (j-(len(steps)-1)/2)*.16
            ax.errorbar(x, values, yerr=np.maximum(0, [values-bounds[0], bounds[1]-values]), fmt="o",
                        color=colors[step], label=f"{step//1000}k updates", capsize=3)
        ax.set(xticks=np.arange(5), xticklabels=DISPLAY, title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
        if row == 1:
            ax.axhline(0, color="#777777", linewidth=.8)
            ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    reference = report["cells"]["50000"]["market_tau"]
    lower, upper = np.asarray(reference["reference_q05_q95"])*100
    ax.axhspan(lower, upper, color="#555555", alpha=.12, label="Reference central 90%")
    ax.axhline(reference["reference_mean"]*100, color="#333333", linestyle="--", label="Reference mean")
    for step in steps:
        values = report["cells"][str(step)]["market_tau"]
        mean = values["neural_mean"]*100
        low, high = np.asarray(values["neural_q05_q95"])*100
        ax.errorbar(step/1000, mean, yerr=[[mean-low], [high-mean]], fmt="o", color=colors[step], capsize=5)
    ax.set(xticks=[50, 75, 100], xlim=(43, 108), xlabel="Total optimizer updates (thousands)",
           ylabel=r"$\tau$ (IV percentage points)", title=r"D  Market: $\tau$ remains far from the reference")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=.2)

    ax = axes[2, 1]
    cases = ["market", *report["case_categories"]]
    status_values = {"pass": 0, "inconclusive": 1, "fail": 2, "not_run": 3, "unavailable": 4}
    color_values = ["#4c9c65", "#e6b85b", "#bd4e48", "#ececec", "#999999"]
    matrix = []
    for case in cases:
        row = []
        for step in (50000, 75000, 100000):
            if str(step) not in report["cells"]:
                status = "unavailable"
            elif case == "market":
                status = report["cells"][str(step)]["market"]["status"]
            else:
                status = report["cells"][str(step)]["conditional_references"][case]["status"]
            row.append(status_values[status])
        matrix.append(row)
    ax.imshow(matrix, cmap=ListedColormap(color_values), vmin=-.5, vmax=4.5, aspect="auto")
    labels = ["Market", *[f"Case {int(case.split(':')[-1]):02d}" for case in cases[1:]]]
    ax.set(yticks=np.arange(len(cases)), yticklabels=labels, xticks=[0, 1, 2], xticklabels=["50k", "75k", "100k"],
           xlabel="Total optimizer updates", title="F  Collective reference agreement")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(handles=[Patch(color=color, label=label) for color, label in zip(color_values,
              ["Pass", "Inconclusive", "Fail", "Not run", "No checkpoint"])],
              loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Longer Heston training: better average density, conditional errors remain\n"
                 "100k clean surfaces · same model and seed · 75k checkpoint valid, 100k endpoint not reached", fontsize=15, y=.985)
    fig.text(.5, .017, "B: paired 95% development-observation interval. C/E: approximate simultaneous Monte Carlo intervals.\n"
             "D: posterior mean and central 90% intervals. F combines six parameters plus two products; individual coordinates may pass.\n"
             "Development evidence only; reserved acceptance cases remain unopened. The failure mechanism is not established.",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.91, bottom=.14, hspace=.43, wspace=.33)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args)
    args.out.mkdir(parents=True, exist_ok=True)
    figure = args.out / "optimization-continuation.png"
    render(report, figure)
    report["figure_sha256"] = digest(figure)
    path = args.out / "continuation-results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"available_endpoints": list(report["cells"]), "unavailable_endpoints": list(report["unavailable_endpoints"]),
                      "summary": str(path), "figure": str(figure)}))


if __name__ == "__main__":
    main()

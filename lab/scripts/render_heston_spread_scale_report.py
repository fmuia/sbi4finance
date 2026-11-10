"""Presentation-only HTML from a frozen noise-scale summary and compact artifacts.

CLI: --summary summary.json --out report.html. Artifact entries contain path
and sha256; relative paths resolve beside the summary. Preferred summary keys:
artifacts.{study,reference_protocol,panel,training_curve,training_record,
training_protocol,trainer_source,fixed_helper_source,recovery?,evaluations}, evaluations keyed by update count,
and references (normalized compact records with source artifacts). No model,
bank or posterior-chain payload is read. recovery is an optional environment
addendum JSON; population quantiles use hash-bound sibling recovery.npz files.
"""

import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.heston_spread_report_figures import _svg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c")
LABELS = {"initial_variance": "Initial variance v₀", "kappa": "Mean reversion κ",
    "theta": "Long-run variance θ", "vol_of_vol": "Volatility of volatility ξ",
    "rho": "Correlation ρ", "c": "Spread multiplier c",
    "control": "Short-window expected variance", "affected": "Later-window expected variance"}
CASES = ("synthetic-broad", "synthetic-market-like", "market")
CASE_LABELS = {"synthetic-broad": "Broad synthetic", "synthetic-market-like": "Market-like synthetic", "market": "Actual market"}
ENDPOINTS = (10000, 50000, 60000)
MODES = ("half", "one", "double", "joint")
MODE_LABELS = {"half": "Fixed c=.5", "one": "Fixed c=1", "double": "Fixed c=2", "joint": "Joint unknown c"}
COLORS = {10000: "#487f9f", 50000: "#287b6a", 60000: "#654a97"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value):
    return f"{float(value):.5g}"


def badge(value):
    if value not in ("pass", "fail", "inconclusive"):
        return f'<span class="muted">{esc(value)}</span>'
    return f'<span class="badge {value}">{esc(value)}</span>'


def section(title, body, ident):
    return f'<section id="{esc(ident)}"><h2>{esc(title)}</h2>{body}</section>'


def table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def chart(svg, caption):
    return '<div class="chart">'+svg+'</div><p class="caption">'+esc(caption)+'</p>'


def source_function(path, name):
    source = Path(path).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise ValueError(f"source function {name} missing from {path}")


def zero_dropout_call(path):
    source = Path(path).read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run_training")
    toggle = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                  and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                  and node.value.func.attr == "switch_dropout_zero")
    updates = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == "step_fn"]
    if not updates or min(updates) <= toggle.lineno:
        raise ValueError("dropout switch does not precede the first native update")
    return ast.get_source_segment(source, toggle)


def stylesheet(path):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CSS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("existing report CSS not found")


class Artifacts:
    """Hash-check allowed presentation inputs without following model/bank paths."""
    def __init__(self, summary_path):
        self.base = Path(summary_path).resolve().parent
        self.sources = []

    def path(self, spec, label, *, recovery=False):
        path = Path(spec["path"])
        path = path if path.is_absolute() else self.base / path
        path = path.resolve()
        allowed = (".npz",) if recovery else (".json", ".py", ".md")
        if path.suffix not in allowed or path.name in ("bank.json", "bank.npz", "chains.npz", "model.keras", "state.npz"):
            raise ValueError(f"not a permitted compact presentation input: {label}")
        actual = digest(path)
        if actual != spec["sha256"]:
            raise ValueError(f"presentation input hash differs: {label}")
        self.sources.append({"label": label, "path": str(path), "sha256": actual})
        return path

    def json(self, spec, label):
        return json.loads(self.path(spec, label).read_text())


def validate_reference_recovery(study, artifacts, inputs):
    original = study["reference_protocol"]
    current = artifacts["reference_protocol"]
    if artifacts.get("recovery") is None:
        if original["sha256"] != current["sha256"]:
            raise ValueError("changed reference protocol requires an explicit recovery addendum")
        return None
    recovery = inputs.json(artifacts["recovery"], "Reference environment recovery addendum")
    if recovery.get("format") != "heston_spread_scale_reference_recovery_v1":
        raise ValueError("unsupported reference-recovery addendum")
    for key, expected in (("original_study", artifacts["study"]),
                          ("original_reference_protocol", original),
                          ("recovery_reference_protocol", current)):
        actual = recovery[key]
        bound_path = inputs.path(actual, "Recovery binding: "+key)
        expected_path = Path(expected["path"])
        expected_path = expected_path if expected_path.is_absolute() else inputs.base / expected_path
        if actual["sha256"] != expected["sha256"] or bound_path != expected_path.resolve():
            raise ValueError("reference recovery path/hash binding differs: "+key)
    return recovery


def quantiles(record, name, *, product=False):
    container = record.get("products", {}) if product else record.get("parameter_quantiles_05_50_95", {})
    value = container.get(name)
    if isinstance(value, dict):
        value = value.get("quantiles_05_50_95")
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.any(np.diff(value) < 0):
        raise ValueError(f"invalid compact quantiles: {name}")
    return value


def reference_records(summary, inputs):
    result = {}
    for ref in summary.get("references", []):
        key = ref["case_id"], ref["mode"]
        if key in result or key[0] not in CASES or key[1] not in MODES:
            raise ValueError("duplicate or unsupported reference summary identity")
        raw = []
        for i, source in enumerate(ref.get("sources", [])):
            value = inputs.json(source, f"{key[0]} / {key[1]} reference source {i+1}")
            raw.extend([value, value.get("result", {})])
        if ref.get("reference_ready") is True:
            expected = ref["parameter_quantiles_05_50_95"]
            if not any(value.get("parameter_quantiles_05_50_95") == expected for value in raw):
                raise ValueError("ready reference quantiles lack a matching compact source")
            for name in (*NAMES[:5], *(("c",) if key[1] == "joint" else ())):
                if quantiles(ref, name) is None:
                    raise ValueError("ready reference omits a declared parameter")
            for name in ref.get("products", {}):
                q = quantiles(ref, name, product=True)
                if not any((quantiles(value, name, product=True) is not None
                            and np.array_equal(quantiles(value, name, product=True), q)) for value in raw):
                    raise ValueError("reference product quantiles lack a matching compact source")
            if ref.get("joint_c") is not None:
                expected_c = {k: v for k, v in ref["joint_c"].items() if k != "scope"}
                if not any(isinstance(value.get("joint_c"), dict)
                           and all(value["joint_c"].get(k) == v for k, v in expected_c.items()) for value in raw):
                    raise ValueError("joint scale/correlation summary lacks a matching compact source")
        result[key] = ref
    return result


def sensitivity(references, products):
    body = '<p>The observed prices are identical within each case. These are different assumed posterior targets: the quantiles and width ratios below are descriptive sensitivity, never pass/fail accuracy tests.</p>'
    for case in CASES[:2]:
        rows = []
        for name in (*NAMES[:5], *products):
            product = name in products
            base = references.get((case, "one"), {})
            base_q = quantiles(base, name, product=product) if base.get("reference_ready") else None
            cells = [esc(LABELS.get(name, name))]
            for mode in MODES:
                ref = references.get((case, mode), {})
                q = quantiles(ref, name, product=product) if ref.get("reference_ready") else None
                if q is None:
                    cells.append('<span class="muted">Reference unavailable / inconclusive</span>')
                    continue
                ratio = "" if base_q is None or base_q[2] == base_q[0] else f'<small>Width / fixed-c1 width: {(q[2]-q[0])/(base_q[2]-base_q[0]):.3f}×</small>'
                cells.append(f'[{fmt(q[0])}, <strong>{fmt(q[1])}</strong>, {fmt(q[2])}]'+ratio)
            rows.append(cells)
        body += '<h3>'+esc(CASE_LABELS[case])+'</h3>'+table(["Quantity · q05, median, q95", *[MODE_LABELS[m] for m in MODES]], rows)
    return body+'<p>Unknown-c physical marginals integrate over the data-dependent posterior p(c|y). They are not an equal mixture of the three fixed-scale curves. The old c=1 neural estimator is not used at c=.5 or 2.</p>'


def joint_scale(references, reference_protocol):
    oracles = reference_protocol.get("fixed_phi_conditional_oracles", {})
    rows, ready = [], []
    for case in CASES:
        ref = references.get((case, "joint"), {})
        q = quantiles(ref, "c") if ref.get("reference_ready") else None
        oracle = oracles.get(case)
        rows.append([esc(CASE_LABELS[case]), "Unavailable / inconclusive" if q is None else esc(f"[{fmt(q[0])}, {fmt(q[1])}, {fmt(q[2])}]"),
            "No known market truth" if oracle is None else esc(str([round(x, 5) for x in oracle["quantiles_05_50_95"]]))])
        if q is not None:
            ready.append((case, ref))
    body = '<p>The joint reference infers physical parameters and c together. The fixed-physical oracle instead conditions on the saved generating parameters. Its interval is not a replacement reference for the joint marginal.</p>'
    body += table(["Case", "Joint marginal c · q05/median/q95", "Fixed-generating-physical oracle · q05/median/q95"], rows)
    if ready:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        for index, (case, ref) in enumerate(ready):
            q = quantiles(ref, "c")
            axes[0].errorbar(q[1], index+.1, xerr=[[q[1]-q[0]], [q[2]-q[1]]], fmt="o", color="#654a97", capsize=3)
            if case in oracles:
                o = np.asarray(oracles[case]["quantiles_05_50_95"])
                axes[0].errorbar(o[1], index-.1, xerr=[[o[1]-o[0]], [o[2]-o[1]]], fmt="D", color="#487f9f", capsize=3)
            corr = ref.get("joint_c", {}).get("physical_parameter_c_correlations", {})
            if corr:
                vals = np.asarray([np.nan if corr.get(name) is None else corr[name] for name in NAMES[:5]], dtype=float)
                present = np.asarray([corr.get(name) is not None for name in NAMES[:5]])
                if not np.isfinite(vals[present]).all() or np.any(abs(vals[present]) > 1+1e-12):
                    raise ValueError("invalid compact posterior correlations")
                axes[1].plot(np.arange(5), vals, "o-", label=CASE_LABELS[case])
        axes[0].set(yticks=np.arange(len(ready)), yticklabels=[CASE_LABELS[c] for c, _ in ready],
                    xlim=(.5, 2.), xlabel="c · central 90% posterior interval", title="Joint marginal: circles\nFixed-φ oracle: diamonds")
        axes[1].axhline(0, color="#84929a", lw=.8)
        axes[1].set(xticks=np.arange(5), xticklabels=["v₀", "κ", "θ", "ξ", "ρ"], ylim=(-1, 1),
                    ylabel="Pearson correlation with c", title="Joint reference dependence · descriptive")
        if axes[1].lines[1:]: axes[1].legend(fontsize=8)
        body += chart(_svg(fig, "Joint scale marginals and physical-scale dependence", "Ready references only. Oracle intervals condition on known generating physical parameters and are not joint inference checks."),
            "Prior support is [.5,2]. Undefined correlations are omitted, not replaced by zero. Small Pearson correlations do not establish independence: scale–width dependence can remain without linear correlation.")
    body += '<p>Posterior pressure near c=2 can reflect insufficient residual-scale support, model discrepancy or data/prior conditions. It does not identify pure measurement noise or authorize retrospective prior expansion. Market scale has no generating truth.</p>'
    return body


def conditional_tables(evaluations, products):
    body = '''<p>Reference readiness requires four-chain rank-normalized R-hat ≤1.01 and bulk/tail ESS ≥400 for the parameters and two variance products, plus sampled numerical price refinement within .02×(.5h)=.01h, independently of inferred c. These sampled checks are not a prior-wide pricing certificate. A ready reference can still reveal a neural approximation error.</p>
      <p>Same-target neural mean error must be at most .3 reference SD and its central-90% width ratio must lie in [.75,1.25], with Monte Carlo uncertainty and the predeclared 144-check family accounted for. Each endpoint retains both synthetic cases and the separate market application. Within each case, 16 individual checks cover mean and width for six parameters plus two analytic variance quantities. Their Monte Carlo intervals are not posterior credible intervals.</p>
      <p>The predeclared terminal endpoint, 60k, appears first; it was not selected as the best result. The earlier 10k and 50k tables remain below in expandable sections, and all tables are included when printing.</p>'''
    for step in (60000, 10000, 50000):
        evaluation = evaluations.get(step, {})
        history = step != 60000
        body += (f'<details class="conditional-history"><summary>{step:,} updates · earlier predeclared endpoint</summary>'
                 if history else f'<h3>{step:,} updates · predeclared terminal endpoint</h3>')
        if not evaluation:
            body += '<p class="muted">Evaluation unavailable.</p>'
            if history:
                body += '</details>'
            continue
        for case in CASES:
            entry = evaluation.get("cases", {}).get(case, {})
            body += '<h4>'+esc(CASE_LABELS[case])+' · '+badge(entry.get("status", "inconclusive"))+'</h4>'
            if "c_mean" in entry and "c_sd" in entry:
                body += (f'<p>Neural c mean {entry["c_mean"]:.5f}, SD {entry["c_sd"]:.5f}. '
                         'Uniform-prior mean 1.25, SD 0.43301. These are descriptive information checks, '
                         'not an identifiability verdict or a replacement for the matched reference.</p>')
            marginal = entry.get("reference_c_marginal")
            if marginal is not None:
                body += f'<p>Matched reference c mean {marginal["mean"]:.5f}, SD {marginal["sd"]:.5f}. Pearson dependence summaries, when supplied, are descriptive rather than copula acceptance.</p>'
            comparison = entry.get("comparison")
            if comparison is None:
                body += '<p class="muted">'+esc(entry.get("reason", "Matched ready reference unavailable."))+'</p>'; continue
            rows = []
            for name in (*NAMES, *products):
                p = comparison["parameters"].get(name)
                if p is None:
                    reason = esc(comparison.get("reason", "Individual metric unavailable."))
                    rows.append([esc(LABELS.get(name, name)), badge("inconclusive")+"<small>"+reason+"</small>", badge("inconclusive")])
                    continue
                mi, wi = p["signed_standardized_mean_interval"], p["width_ratio_interval"]
                rows.append([esc(LABELS.get(name, name)),
                    f'{p["absolute_standardized_mean_difference"]:.3f} SD {badge(p["mean_status"])}<small>Signed MC interval [{mi[0]:+.3f}, {mi[1]:+.3f}]</small>',
                    f'{p["width_ratio_90"]:.3f}× {badge(p["width_status"])}<small>MC interval [{wi[0]:.3f}, {wi[1]:.3f}]</small>'])
            body += table(["Quantity", "Absolute mean error · limit .3 reference SD", "90% width ratio · limits [.75,1.25]"], rows)
        if history:
            body += '</details>'
    return body+'<p>The predeclared 144-check family spans three endpoints, three cases, eight quantities and two metrics. All endpoints remain visible; this report does not select a best checkpoint. Product checks do not constitute a complete test of joint shape or tails.</p>'


def population_records(evaluations):
    records = {}
    for step, evaluation in evaluations.items():
        p = evaluation.get("population")
        if p is None: continue
        ranks = np.asarray(p["fractional_ranks"], dtype=float)
        if (p["parameter_names"] != list(NAMES) or ranks.shape != (512, 6)
                or not np.isfinite(ranks).all() or np.any(ranks < 0) or np.any(ranks >= 1)
                or len(p["rank_dkw"]["passed"]) != 6):
            raise ValueError("population figure requires six declared marginal rank series")
        records[step] = p, ranks
    return records


def population_figures(records):
    if not records:
        return '<p class="muted">No completed population diagnostics are available.</p>'
    rows = [[f"{step:,}", str(sum(p["rank_dkw"]["passed"]))+"/6", fmt(p["rank_dkw"]["dkw_bound"]),
             esc(", ".join(f'{name}: {p["coverage"]["0.9"]["coverage"][i]:.3f}' for i, name in enumerate(NAMES)))]
            for step, (p, _) in sorted(records.items())]
    body = table(["Updates", "Marginal DKW screens", "DKW bound", "90% development coverage"], rows)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True, layout="constrained")
    grid = np.linspace(0, 1, 201)
    for i, axis in enumerate(axes.flat):
        for step, (p, ranks) in sorted(records.items()):
            difference = np.searchsorted(np.sort(ranks[:, i]), grid, side="right") / len(ranks) - grid
            axis.plot(grid, difference, color=COLORS[step], label=f"{step//1000}k")
        bounds = [float(p["rank_dkw"]["dkw_bound"]) for p, _ in records.values()]
        if not np.allclose(bounds, bounds[0], rtol=0, atol=1e-14):
            raise ValueError("different DKW bands need explicit separate display")
        axis.axhspan(-bounds[0], bounds[0], color="#dbece1", zorder=0)
        axis.axhline(0, color="#6d7d87", lw=.7)
        axis.set(title=LABELS[NAMES[i]], xlabel="Fractional rank", ylabel="ECDF − uniform CDF", xlim=(0, 1))
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    body += chart(_svg(fig, "Six-parameter development SBC at all three endpoints", "Randomized finite-posterior ranks. The DKW adjustment is across six marginals within each checkpoint, not across all three endpoints."),
        "The same 512 development physical rows are reused, with generating c independently Uniform(.5,2). Green reference bands adjust across six parameters per checkpoint; they do not supply simultaneous control across all three endpoints or conditional-on-truth calibration.")
    if 60000 in records:
        ranks = records[60000][1]
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.7), sharex=True, sharey=True, layout="constrained")
        for i, axis in enumerate(axes.flat):
            axis.hist(ranks[:, i], bins=np.linspace(0, 1, 11), color=COLORS[60000], edgecolor="white")
            axis.axhline(51.2, color="#436753", ls="--", lw=1)
            axis.set(title=LABELS[NAMES[i]], xlabel="Fractional rank", ylabel="Datasets per bin", xlim=(0, 1))
        body += chart(_svg(fig, "60k six-parameter SBC rank histograms", "Ten equal bins for each of six parameters. The dashed 51.2 count is an expectation, not a confidence interval."),
            "Ten bins, 512 datasets. Approximate flatness is a marginal diagnostic, not proof of accurate per-observation or joint inference.")
    return body


def recovery_data(spec, inputs, evaluation, step=60000):
    path = inputs.path(spec, f"{step:,}-update saved recovery quantiles", recovery=True)
    population = evaluation["population"]
    if population["recovery_sha256"] != digest(path):
        raise ValueError("recovery artifact does not bind the final evaluation")
    with np.load(path, allow_pickle=False) as archive:
        truth, q, ids = archive["truth"].copy(), archive["quantiles"].astype(float), archive["row_id"].copy()
    if (truth.shape != (512, 6) or q.shape != (3, 512, 6) or not np.isfinite(truth).all()
            or not np.isfinite(q).all() or np.any(np.diff(q, axis=0) < 0)
            or value_digest(truth) != population["truth_sha256"]
            or not np.array_equal(ids, np.array([f"validation:{i:08d}" for i in range(512)]))):
        raise ValueError("recovery dimensions, truth identity, ordering or row IDs differ")
    return truth, q


def scale_information(recoveries, bounds):
    """Descriptive recovery/contraction; no posterior identifiability verdict."""
    lo, hi = bounds
    prior_width, prior_median = .9*(hi-lo), .5*(lo+hi)
    rows, metrics = [], {}
    for step in ENDPOINTS:
        if step not in recoveries:
            rows.append([f"{step:,}", "Unavailable", "Unavailable", "Unavailable", "Unavailable"])
            continue
        truth, q = recoveries[step]
        actual, prediction = truth[:, -1], q[1, :, -1]
        rmse = float(np.sqrt(np.mean((prediction-actual)**2)))
        prior_rmse = float(np.sqrt(np.mean((prior_median-actual)**2)))
        correlation = (None if np.ptp(prediction) == 0 else float(np.corrcoef(actual, prediction)[0, 1]))
        width = float(np.median(q[2, :, -1]-q[0, :, -1]))
        metrics[str(step)] = {"c_posterior_median_rmse": rmse,
            "c_prior_median_predictor_rmse_same_truths": prior_rmse,
            "c_truth_median_pearson_correlation": correlation,
            "median_c_posterior90_width": width, "uniform_c_prior90_width": prior_width,
            "median_posterior90_width_over_prior90_width": width/prior_width,
            "scope": "Descriptive saved-quantile recovery/contraction; not a likelihood-identifiability diagnosis."}
        rows.append([f"{step:,}", fmt(rmse), fmt(prior_rmse),
            "Undefined: constant medians" if correlation is None else f"{correlation:+.4f}",
            f"{width:.4f} / {prior_width:.4f} = {width/prior_width:.3f}×"])
    body = '''<div class="callout warn"><strong>SBC calibration is different from learning information.</strong>
      <p>A predictor that ignores the quotes and returns the prior can pass marginal prior-predictive SBC.
      Near-prior c intervals therefore need to be read with recovery and matched conditional references.
      Neural ignorance does not establish that c is unidentified by the model.</p></div>'''
    body += table(["Updates", "c median RMSE", "Constant prior-median RMSE", "Truth–median correlation", "Median posterior 90% width / prior 90% width"], rows)
    body += '<p>The constant baseline always predicts the Uniform-prior median 1.25 on these same truths. A width ratio near 1 indicates little interval contraction in the neural output; it is not a pass/fail criterion. Correct posterior medians need not be Uniform, and perfect recovery is not expected from noisy observations. Ready joint MCMC establishes the conditional target; this table alone cannot diagnose model identifiability.</p>'
    return body, metrics


def conditional_scale_information(evaluations):
    rows = []
    for case in CASES[:2]:
        entry = evaluations.get(60000, {}).get("cases", {}).get(case, {})
        marginal = entry.get("reference_c_marginal")
        comparison = (entry.get("comparison") or {}).get("parameters", {}).get("c")
        if marginal is None or comparison is None:
            continue
        values = [entry["c_mean"], entry["c_sd"], marginal["mean"], marginal["sd"]]
        if not np.isfinite(values).all():
            raise ValueError("nonfinite ready conditional c information summary")
        rows.append([esc(CASE_LABELS[case]), f'{entry["c_mean"]:.4f} / {entry["c_sd"]:.4f}',
                     f'{marginal["mean"]:.4f} / {marginal["sd"]:.4f}', badge(comparison["status"])])
    if not rows:
        return '<p class="muted">No ready synthetic joint-reference scale comparison is available yet. No model-identifiability conclusion is inferred from the neural output.</p>'
    return '<h3>Ready synthetic conditional evidence at the final endpoint</h3>'+table(
        ["Case", "Neural c mean / SD", "Joint-reference c mean / SD", "Declared c mean/width check"], rows)+\
        '<p>A narrow joint-reference c posterior alongside a near-prior neural posterior is a case-specific approximation failure: the specified data/model contain information the estimator misses. It is not global non-identifiability. These comparisons concern the full joint marginal, unlike the known-physical oracle.</p>'


def recovery_figures(truth, q, bounds):
    chosen = np.arange(0, 512, 16)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), layout="constrained")
    for i, axis in enumerate(axes.flat):
        lo, hi = bounds[i]
        axis.scatter(truth[:, i], q[1, :, i], s=10, color="#617784", alpha=.4, edgecolors="none")
        axis.errorbar(truth[chosen, i], q[1, chosen, i],
            yerr=np.stack([q[1, chosen, i]-q[0, chosen, i], q[2, chosen, i]-q[1, chosen, i]]),
            fmt="o", ms=2.5, color=COLORS[60000], capsize=2, elinewidth=.8)
        axis.plot([lo, hi], [lo, hi], "--", color="#334e5c", lw=1)
        margin = .025*(hi-lo)
        axis.set(title=LABELS[NAMES[i]], xlabel="Generating value", ylabel="Posterior median",
                 xlim=(lo-margin, hi+margin), ylim=(lo-margin, hi+margin), aspect="equal")
    body = chart(_svg(fig, "60k recovery for all six parameters", "All 512 posterior medians are shown. Marginal 90-percent intervals use the same 32 index-selected datasets, indices 0,16,...,496, in every panel."),
        "All 512 medians; 32 marginal 90% intervals selected by fixed row index, not outcome. Bars are posterior intervals, not Monte Carlo error bars. Full-prior plots can look excellent despite material local conditional errors.")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6), layout="constrained")
    for i, axis in enumerate(axes.flat):
        axis.errorbar(truth[chosen, i], q[1, chosen, i]-truth[chosen, i],
            yerr=np.stack([q[1, chosen, i]-q[0, chosen, i], q[2, chosen, i]-q[1, chosen, i]]),
            fmt="o", ms=3, color=COLORS[60000], capsize=2, elinewidth=.8)
        axis.axhline(0, color="#334e5c", ls="--", lw=1)
        axis.set(title=LABELS[NAMES[i]], xlabel="Generating value", ylabel="Median − truth")
    body += chart(_svg(fig, "Same 32 cases: six-parameter recovery residuals", "Median-minus-truth and shifted marginal 90-percent posterior intervals. These are the same fixed index-selected datasets, not additional inference."),
        "Detail view of exactly the same 32 cases. Vertical scales vary by parameter and use physical units.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    grid = np.linspace(.5, 2., 201)
    axes[0].plot(grid, (grid-.5)/1.5, "--", color="#778791", label="Uniform prior CDF")
    for value, label, color in ((truth[:, -1], "Generating c", "#287b6a"), (q[1, :, -1], "Posterior median c", COLORS[60000])):
        ordered = np.sort(value)
        axes[0].step(ordered, np.arange(1, len(ordered)+1)/len(ordered), where="post", label=label, color=color)
    axes[0].set(xlabel="c", ylabel="ECDF", xlim=(.5, 2.), ylim=(0, 1), title="Generating c and median-c distributions")
    axes[0].legend(fontsize=8)
    axes[1].scatter(truth[:, -1], q[1, :, -1], s=13, alpha=.45, color=COLORS[60000])
    axes[1].plot([.5, 2.], [.5, 2.], "--", color="#334e5c")
    axes[1].set(xlabel="Generating c", ylabel="Posterior median c", xlim=(.5, 2.), ylim=(.5, 2.), aspect="equal",
                title="c recovery across the full declared prior")
    body += chart(_svg(fig, "Generating scale ECDF and posterior-median recovery", "Posterior medians need not follow the prior distribution even under correct inference. The median ECDF is descriptive, not an SBC uniformity test."),
        "Posterior-median ECDF is not expected to be Uniform under exact inference; this is descriptive recovery. SBC uses posterior ranks, not the distribution of medians.")
    return body


def build(summary_path, out):
    summary_path, out = Path(summary_path).resolve(), Path(out)
    summary = json.loads(summary_path.read_text())
    inputs = Artifacts(summary_path)
    inputs.sources.append({"label": "Frozen report summary", "path": str(summary_path), "sha256": digest(summary_path)})
    artifacts = summary.get("artifacts", summary)
    study = inputs.json(artifacts["study"], "Scientific study protocol")
    reference_protocol = inputs.json(artifacts["reference_protocol"], "Reference protocol")
    environment_recovery = validate_reference_recovery(study, artifacts, inputs)
    panel = inputs.json(artifacts["panel"], "Fixed observation panel")
    training = inputs.json(artifacts["training_record"], "Completed training record")
    training_protocol = inputs.json(artifacts["training_protocol"], "Training protocol")
    curve = inputs.json(artifacts["training_curve"], "Fixed training/validation probes")
    trainer_source = inputs.path(artifacts["trainer_source"], "Executed six-target trainer source")
    fixed_helper_source = inputs.path(artifacts["fixed_helper_source"], "Frozen fixed-task helper source")
    if (study["joint_task"]["parameter_names"] != list(NAMES) or panel["joint_task"] != study["joint_task"]
            or training["task"]["target"] != study["joint_task"]
            or training["source_sha256"]["trainer"] != digest(trainer_source)
            or training["source_sha256"]["fixed_helper"] != digest(fixed_helper_source)
            or training_protocol["trainer_sha256"] != digest(trainer_source)
            or study["panel"]["sha256"] != artifacts["panel"]["sha256"]
            or training["protocol_sha256"] != artifacts["training_protocol"]["sha256"]
            or training["task_fingerprint"] != training_protocol["task_fingerprint"]):
        raise ValueError("report sources disagree on joint target or executed trainer")
    toggle = training["initialization_control"]
    if (len(toggle["rates_after"]) != 24 or any(rate != 0. for rate in toggle["rates_after"])
            or not toggle["array_state_unchanged"]):
        raise ValueError("report cannot claim 24 state-preserving zero-dropout layers")
    setting_rows = [["Targets", "Six: five physical Heston parameters plus c; six-dimensional weights differ from the old five-target model."],
        ["Flow", esc(str(training["architecture"]))],
        ["Physical training rows / batch", f'{training["train_rows"]:,} / {training["batch_size"]}'],
        ["Update budget / endpoints", f'{training["planned_updates"]:,} / '+esc(str(study["joint_neural_fit"]["endpoints"]))],
        ["Optimizer", "Adam; moments and iteration retained across schedule changes."],
        ["Effective dropout", "24 existing Dropout rates switched from .05 to 0 before the first update; all arrays unchanged."],
        ["Seed", esc(training["seed"])]]
    setting_rows += [["Version: "+name, esc(value)] for name, value in training["versions"].items()]
    settings_html = table(["Setting", "Executed recipe"], setting_rows)
    rate_rows = [[esc(str(stage["t_inclusive"])), esc(stage.get("rule", fmt(stage["rate"]) if "rate" in stage else ""))]
                 for stage in study["joint_neural_fit"]["schedule"]]
    settings_html += table(["Absolute update steps, inclusive", "Learning rate / schedule"], rate_rows)
    evaluations, evaluation_paths = {}, {}
    for key, spec in artifacts.get("evaluations", summary.get("evaluations", {})).items():
        step = int(key)
        if step not in ENDPOINTS or step in evaluations:
            raise ValueError("undeclared report endpoint")
        record = inputs.json(spec, f"{step:,}-update evaluation")
        if (record["actual_optimizer_updates"] != step or record["task"] != study["joint_task"]
                or record["study_protocol_sha256"] != artifacts["study"]["sha256"]
                or record["reference_protocol_sha256"] != artifacts["reference_protocol"]["sha256"]
                or record["panel_sha256"] != artifacts["panel"]["sha256"]):
            raise ValueError("evaluation endpoint/target differs")
        evaluations[step] = record
        value_path = Path(spec["path"])
        evaluation_paths[step] = value_path if value_path.is_absolute() else inputs.base / value_path
    references = reference_records(summary, inputs)
    products = panel["products"]
    recoveries = {}
    for step, evaluation in evaluations.items():
        if "population" not in evaluation:
            continue
        sibling = evaluation_paths[step].with_name("recovery.npz")
        if sibling.is_file():
            spec = {"path": str(sibling.resolve()), "sha256": evaluation["population"]["recovery_sha256"]}
            recoveries[step] = recovery_data(spec, inputs, evaluation, step)
    information_body, information_metrics = scale_information(recoveries, study["joint_task"]["prior_bounds"][-1])
    information_body += conditional_scale_information(evaluations)
    css_path = Path(__file__).with_name("render_heston_spread_report.py")
    svg_path = Path(__file__).with_name("heston_spread_report_figures.py")
    for label, path in (("Presentation producer", Path(__file__)), ("Shared stylesheet source", css_path), ("Shared SVG helper", svg_path)):
        inputs.sources.append({"label": label, "path": str(path.resolve()), "sha256": digest(path)})
    for label, spec in summary.get("sources", {}).items():
        inputs.path(spec, "Summary producer source: "+label)
    snapshot = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pieces = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Heston: one Gaussian noise scale</title><style>'+stylesheet(css_path)+'</style></head><body><main>',
        '<header><p class="eyebrow">Fixed 54-contract Heston · noise-scale development</p><h1>One Gaussian scale: assumptions, joint inference and measured checks</h1>',
        '<p class="lead">Fixed-scale sensitivity and an inferred-scale posterior answer different questions. All predeclared checkpoints and unresolved references remain visible.</p>',
        '<div class="actions"><button onclick="window.print()">Print report</button><button id="download">Download source inventory</button></div></header>',
        '<nav><a href="#targets">Targets</a><a href="#information">Calibration vs information</a><a href="#sensitivity">Fixed sensitivity</a><a href="#scale">Joint scale</a><a href="#training">Training</a><a href="#conditional">Conditional checks</a><a href="#population">Population</a><a href="#recovery">Recovery</a><a href="#code">Source code</a><a href="#sources">Sources</a></nav>']
    pieces.append(section("Same observations, two inference questions", '''<div class="equation">Yᵢ = Pᵢ(φ) + c hᵢ Zᵢ, &nbsp; Zᵢ ∼ N(0,1)<small>Known frozen half-spreads h; one shared scale c. Conditional quote errors are independent.</small></div>
      <p>Fixed c=.5,1,2 changes the assumed likelihood at identical observed prices. Joint inference instead learns six targets: five physical Heston parameters and c∼Uniform(.5,2). The generating c is never an input; the estimator receives only 54 ordered Y/h values.</p>
      <p>The known-scale physical MAP is unchanged by global c rescaling under the fixed Uniform physical prior, but posterior means and widths can change. Joint physical marginals use p(c|y), not an unweighted average of fixed-scale curves. Half-spread-as-SD and the c prior remain conventions.</p>
      <p>The old 63 grid was symmetric at 20%; the current 54 is already put-heavy and remains frozen here. The audited E63 proposal adds short-expiry/wing coverage but lacks a full-year PM expiry and its original final +2 call. No grid repricing, spread-profile change, tau component or Bergomi work enters this study.</p>''', "targets"))
    if environment_recovery is not None:
        pieces.append('<div class="callout"><strong>Explicit reference environment recovery</strong><p>The report follows the separately hash-bound recovery addendum and recovered reference protocol. The original study and original reference protocol remain bound and preserved; this is not a silent change of observations, targets or priors.</p></div>')
    pieces.append(section("Calibration does not establish information learning", information_body, "information"))
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "heston-spread-scale-report-v1", "font.family": "DejaVu Sans", "font.size": 9}):
        pieces.append(section("Fixed-scale posterior sensitivity", sensitivity(references, products), "sensitivity"))
        pieces.append(section("Joint scale versus a known-physical oracle", joint_scale(references, reference_protocol), "scale"))
        fig, axis = plt.subplots(figsize=(11, 4), layout="constrained")
        for key, label, color in (("train_probe_loss", "Fixed train probe", "#487f9f"), ("validation_probe_loss", "Fixed validation probe", "#654a97")):
            points = [(p["updates"], p[key]) for p in curve if key in p]
            if points:
                x, y = np.asarray(points).T
                if not np.isfinite(y).all(): raise ValueError("nonfinite compact loss curve")
                axis.plot(x/1000, y, label=label, color=color)
        for endpoint in ENDPOINTS: axis.axvline(endpoint/1000, color="#c4cdd2", ls=":", lw=.8)
        axis.set(xlabel="Absolute optimizer updates (thousands)", ylabel="Fixed-probe loss · learned target coordinates", title="One predeclared six-target trajectory; all endpoints retained")
        axis.legend(fontsize=9)
        pieces.append(section("Training recipe and objective", chart(_svg(fig, "Six-target fixed training and validation probes", "Probe losses from the compact training record only. No fitting or model loading."),
            "The 10k/50k/60k checkpoints are predeclared. Probe loss uses the trainer's learned target coordinates; no physical-density ranking is inferred. Five-target and six-target NLLs have different dimensions and are not an accuracy ranking.")+'''<p>The spline uses six coupling blocks and 128×128 conditioners. Construction uses dropout .05, then 24 existing layers are switched to effective dropout 0 before the first update without changing state arrays. Adam moments are retained at rate changes. The source below shows the executing code.</p>'''+settings_html, "training"))
        pieces.append(section("Neural versus matched joint references", conditional_tables(evaluations, products), "conditional"))
        pieces.append(section("Six-parameter population calibration", population_figures(population_records(evaluations)), "population"))
        if 60000 in recoveries:
            truth, q = recoveries[60000]
            recovery_body = recovery_figures(truth, q, np.asarray(study["joint_task"]["prior_bounds"]))
        else:
            recovery_body = '<p class="muted">A hash-bound final recovery artifact was not supplied; no values are reconstructed or sampled.</p>'
        pieces.append(section("Final 60k recovery, including c", recovery_body, "recovery"))
    code = '<p>Exact AST-extracted functions from the hash-bound executed trainer; the module is not imported or run. The construction rate .05 is followed by the separately verified 24-layer zero-dropout switch.</p>'
    for name in ("build_workflow", "augmented_batch", "draw_c", "install_annealing"):
        code += '<details><summary>'+esc(name)+'</summary><pre><code>'+esc(source_function(trainer_source, name))+'</code></pre></details>'
    code += '<h3>The effective zero-dropout action</h3><p>This exact call occurs in run_training before the first native optimizer update:</p><pre><code>'+esc(zero_dropout_call(trainer_source))+'</code></pre>'
    code += '<details><summary>Frozen helper: switch_dropout_zero</summary><pre><code>'+esc(source_function(fixed_helper_source, "switch_dropout_zero"))+'</code></pre></details>'
    pieces.append(section("Executed network, augmentation and learning-rate code", code, "code"))
    source_rows = []
    for source in inputs.sources:
        path = Path(source["path"])
        try:
            relative = path.relative_to(ROOT)
            blob = subprocess.run(["git", "show", f"{snapshot}:{relative.as_posix()}"], cwd=ROOT, capture_output=True)
            committed = blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == source["sha256"]
            display = '<a href="https://github.com/fmuia/sbi4finance/blob/'+snapshot+'/'+relative.as_posix()+'">'+esc(relative)+'</a>' if committed else esc(relative)+'<small>Local/ignored or not present at this commit; SHA-bound.</small>'
        except ValueError:
            display = esc(path)+'<small>External local artifact; SHA-bound.</small>'
        source_rows.append([esc(source["label"]), display, '<span class="hash">'+source["sha256"]+'</span>'])
    pieces.append(section("Sources and scope", '<p>Presentation only: compact JSON records, exact source text and the named recovery quantiles. No bank, model or posterior-chain payload was read; no pricing, training or posterior sampling occurred. Missing results remain unavailable. The report does not confer scientific or release acceptance.</p><p>Git snapshot: <code>'+esc(snapshot)+'</code>. GitHub links are emitted only when the bytes exist at that commit.</p>'+table(["Source", "Path", "SHA-256"], source_rows), "sources"))
    payload = json.dumps({"format": "heston_spread_scale_report_sources_v1", "git_revision": snapshot,
        "summary_sha256": digest(summary_path), "sources": inputs.sources,
        "derived_c_information": information_metrics,
        "scientific_acceptance": False}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    pieces.append('<footer>Development report · all predeclared endpoints · no automatic release or Bergomi authorization.</footer><script id="report-data" type="application/json">'+payload+'</script>'+'''<script>
      document.getElementById('download').addEventListener('click',function(){const text=JSON.stringify(JSON.parse(document.getElementById('report-data').textContent),null,2);const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='heston-scale-report-sources.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
      let conditionalPrintState=null;
      window.addEventListener('beforeprint',function(){if(conditionalPrintState===null){conditionalPrintState=Array.from(document.querySelectorAll('details.conditional-history'),function(element){const prior=element.open;element.open=true;return[element,prior]})}});
      window.addEventListener('afterprint',function(){if(conditionalPrintState!==null){conditionalPrintState.forEach(function(item){item[0].open=item[1]});conditionalPrintState=null}});
      </script></main></body></html>''')
    # Refuse a presentation assembled while any declared input was changing.
    if any(digest(source["path"]) != source["sha256"] for source in inputs.sources):
        raise ValueError("presentation input changed during rendering")
    out.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(line.rstrip() for line in "".join(pieces).splitlines())+"\n"
    out.write_text(rendered, encoding="utf-8")
    return {"out": str(out), "sha256": digest(out), "bytes": out.stat().st_size,
            "sources": len(inputs.sources), "scientific_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.summary, args.out)))


if __name__ == "__main__":
    main()

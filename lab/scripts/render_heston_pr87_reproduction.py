"""Compact offline HTML for the single PR87 FlowMatching reproduction.

Read saved summaries, recovery quantiles and existing compact MCMC records.
No model loading, posterior queries, pricing or reference sampling occurs.
Inline, hand-built SVG keeps the primary Paseo preview below one MiB.
"""
import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_heston_pr87_reproduction as evaluation

read, binding = evaluation.read, evaluation.binding
LABELS = ("Initial variance v₀", "Mean reversion κ", "Long-run variance θ", "Volatility σᵥ", "Correlation ρ", "Noise scale c")
BLUE, RED, GRAY, GREEN = "#236b9a", "#b84935", "#596a77", "#2e8067"
LIMIT_BYTES = 1024 * 1024


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=3):
    if value is None: return "—"
    try:
        number = float(value)
        return f"{number:.{digits}g}" if np.isfinite(number) else str(number)
    except (ValueError, TypeError): return esc(value)


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(v)}</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def checked(spec, protected):
    path = evaluation.checked_binding(spec, protected)
    return read(path)


def load_recovery(summary, protected):
    if summary["calibration_status"] != "defined": return None
    spec = summary["recovery_payload"]
    path = evaluation.checked_binding(spec, protected)
    with np.load(path, allow_pickle=False) as archive:
        truth, quantiles = archive["truth"].copy(), archive["quantiles"].astype(float)
    if (truth.shape != (summary["rows"], 6) or quantiles.shape != (3, summary["rows"], 6)
            or evaluation.baseline.observations_digest(truth) != summary["truth_sha256"]
            or not np.isfinite(quantiles).all() or np.any(np.diff(quantiles, axis=0) < 0)):
        raise ValueError("saved raw-draw recovery payload differs")
    return truth, quantiles


def selected_references(spec, population, target, protected):
    """Read existing selected-case compact intervals; never open native chains."""
    if population is None: return None
    manifest = checked(spec, protected)
    selection = checked(manifest["truth_selection"], protected)
    panel = checked(manifest["panel"], protected)
    truth, neural_q = population
    indices = selection["selection"]["zero_based_population_indices"]
    if (len(indices) != 8 or panel["joint_task"] != target
            or selection["population_truth_sha256"] != evaluation.baseline.observations_digest(truth)
            or not np.array_equal(np.argsort(truth[:, -1], kind="stable")[list(range(32, 512, 64))], indices)):
        raise ValueError("existing selected intervals use different target or population rows")
    selected_truth = np.asarray([case["truth"] for case in selection["cases"]])
    if not np.array_equal(truth[indices], selected_truth):
        raise ValueError("selected truth rows differ")
    q = np.full((3, 8, 6), np.nan)
    reasons = []
    for i, (case, observed) in enumerate(zip(selection["cases"], panel["cases"], strict=True)):
        if (case["population_index"] != indices[i] or case["id"] != observed["id"]
                or case["observations_sha256"] != observed["observations_sha256"]):
            raise ValueError("selected case identities differ")
        ref = next(item for item in manifest["references"] if item["case_id"] == case["id"])
        run = checked(ref["run"], protected) if ref.get("run") else {}
        diagnostics = checked(ref["diagnostics"], protected) if ref.get("diagnostics") else {}
        posterior = checked(ref["posterior"], protected) if ref.get("posterior") else None
        if run and (run["task"] != target or run["observation_sha256"] != case["observations_sha256"]):
            raise ValueError("selected reference target or quotes differ")
        ready = (run.get("status") in ("complete", "completed") and run.get("reference_ready") is True
            and diagnostics.get("reference_ready") is True and posterior is not None)
        reasons.append("ready" if ready else "unready: retained without a reference interval")
        if ready:
            q[:, i, :] = np.asarray([posterior["parameter_quantiles_05_50_95"][name] for name in target["parameter_names"]]).T
    return {"indices": indices, "truth": selected_truth, "neural_quantiles": neural_q[:, indices, :],
        "reference_quantiles": q, "reference_ready": np.isfinite(q).all(axis=(0, 2)), "reasons": reasons}


def text(x, y, value, *, size=11, color=GRAY, anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{esc(value)}</text>'


def line(x1, y1, x2, y2, color=GRAY, width=1, dash=None):
    extra = '' if dash is None else f' stroke-dasharray="{dash}"'
    return f'<path d="M{x1:.1f},{y1:.1f}L{x2:.1f},{y2:.1f}" stroke="{color}" stroke-width="{width}" fill="none"{extra}/>'


def path(points, color=BLUE, width=1.5):
    return '<path d="' + ' '.join(('M' if i == 0 else 'L') + f'{x:.1f},{y:.1f}' for i, (x, y) in enumerate(points)) + f'" stroke="{color}" stroke-width="{width}" fill="none"/>'


def circle(x, y, color=BLUE, radius=2, opacity=1):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="{opacity}"/>'


def svg_panel(title, panels, description, width=1020, height=550):
    content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title><rect width="100%" height="100%" fill="white"/>'
    content += ''.join(panels) + '</svg>'
    return f'<figure>{content}<figcaption><strong>{esc(title)}.</strong> {esc(description)}</figcaption></figure>'


def axes(j, limits_x, limits_y, xlabel, ylabel):
    col, row = j % 3, j // 3
    x, y, w, h = 58 + col * 338, 28 + row * 266, 258, 192
    xlo, xhi = limits_x; ylo, yhi = limits_y
    project_x = lambda value: x + (value - xlo) / (xhi - xlo) * w
    project_y = lambda value: y + h - (value - ylo) / (yhi - ylo) * h
    parts = [text(x + w / 2, y - 9, LABELS[j], size=12, anchor="middle"),
        line(x, y + h, x + w, y + h, "#9da9b2"), line(x, y, x, y + h, "#9da9b2")]
    for value in np.linspace(xlo, xhi, 3):
        px = project_x(value)
        parts += [line(px, y + h, px, y + h + 4, "#9da9b2"), text(px, y + h + 17, fmt(value), size=9, anchor="middle")]
    for value in np.linspace(ylo, yhi, 3):
        py = project_y(value)
        parts += [line(x - 4, py, x, py, "#9da9b2"), text(x - 7, py + 3, fmt(value), size=9, anchor="end")]
    parts += [text(x + w / 2, y + h + 35, xlabel, size=10, anchor="middle"),
        f'<text transform="translate({x - 43},{y + h / 2}) rotate(-90)" font-size="10" fill="{GRAY}" text-anchor="middle">{esc(ylabel)}</text>']
    return parts, project_x, project_y


def recovery_plot(recovery, target, label, rows=100):
    truth, q = recovery
    chosen = np.linspace(0, len(truth) - 1, min(rows, len(truth)), dtype=int)
    bars = chosen[::max(1, len(chosen) // 16)]
    panels = []
    for j, (lo, hi) in enumerate(target["prior_bounds"]):
        low = min(lo, float(q[0, bars, j].min()), float(q[1, chosen, j].min()))
        high = max(hi, float(q[2, bars, j].max()), float(q[1, chosen, j].max()))
        margin = .025 * (high - low)
        parts, px, py = axes(j, (lo, hi), (low - margin, high + margin), "Generating value", "Posterior median / 90% interval")
        parts.append(line(px(lo), py(lo), px(hi), py(hi), "#a0acb4", dash="3 3"))
        for i in bars: parts.append(line(px(truth[i, j]), py(q[0, i, j]), px(truth[i, j]), py(q[2, i, j]), BLUE, .7))
        for i in chosen: parts.append(circle(px(truth[i, j]), py(q[1, i, j]), radius=1.7, opacity=.55))
        panels.extend(parts)
    return svg_panel(label + ": all six target recoveries", panels,
        f"{len(chosen)} evenly spaced rows shown; 90% intervals for {len(bars)} fixed rows. Metrics below use all {len(truth)} rows and every raw draw. Axis extent includes displayed out-of-prior intervals.")


def sbc_plot(summary):
    ranks = np.asarray(summary["fractional_ranks"])
    band = summary["rank_dkw"]["dkw_bound"]
    grid = np.linspace(0, 1, 101)
    departures = np.stack([np.searchsorted(np.sort(ranks[:, j]), grid, side="right") / len(ranks) - grid for j in range(6)], axis=1)
    extent = max(.12, band * 1.2, float(np.abs(departures).max()) * 1.08)
    panels = []
    for j in range(6):
        parts, px, py = axes(j, (0, 1), (-extent, extent), "Fractional posterior rank", "ECDF − uniform CDF")
        parts += [f'<rect x="{px(0):.1f}" y="{py(band):.1f}" width="{px(1)-px(0):.1f}" height="{py(-band)-py(band):.1f}" fill="#e1eee7"/>',
            line(px(0), py(0), px(1), py(0), "#9da9b2", .7), path([(px(x), py(y)) for x, y in zip(grid, departures[:, j])])]
        panels.extend(parts)
    return svg_panel("Marginal SBC on the historical 512-case panel", panels,
        "Shading is the six-parameter union-adjusted DKW band within this panel. Reused development data and a single run do not provide independent estimator acceptance.")


def selected_plot(selected):
    q, reference, truth = selected["neural_quantiles"], selected["reference_quantiles"], selected["truth"]
    panels = []
    for j in range(6):
        all_values = np.r_[q[:, :, j].ravel(), reference[:, :, j].ravel(), truth[:, j]]
        all_values = all_values[np.isfinite(all_values)]
        lo, hi = float(all_values.min()), float(all_values.max()); margin = max((hi - lo) * .06, .001)
        parts, px, py = axes(j, (0, 9), (lo - margin, hi + margin), "Selected case, ascending generating c", "Parameter value and central90% interval")
        for i in range(8):
            x = i + 1
            parts += [line(px(x - .08), py(q[0, i, j]), px(x - .08), py(q[2, i, j]), BLUE, 2), circle(px(x - .08), py(q[1, i, j]), BLUE, 2.5), circle(px(x), py(truth[i, j]), "#202e39", 2)]
            if selected["reference_ready"][i]:
                parts += [line(px(x + .08), py(reference[0, i, j]), px(x + .08), py(reference[2, i, j]), RED, 2), circle(px(x + .08), py(reference[1, i, j]), RED, 2.5)]
        panels.extend(parts)
    return svg_panel("Frozen eight-case interval comparison", panels,
        "Blue: raw FlowMatching posterior. Red: existing ready joint MCMC. Black: generating truth. All eight cases remain; an unready reference has no red interval. These cases were originally selected at eight fixed generating-c ranks.")


def learning_plot(curve):
    if not curve: return ''
    series = []
    for key, color in (("loss", BLUE), ("val_loss", RED)):
        pairs = [(int(p.get("epoch", i + 1)), p.get(key, p.get("logs", {}).get(key))) for i, p in enumerate(curve)]
        pairs = [(x, float(y)) for x, y in pairs if y is not None and np.isfinite(float(y))]
        if pairs: series.append((key, color, pairs))
    if not series: return ''
    xmin, xmax = 1, max(x for _, _, pairs in series for x, _ in pairs)
    ys = [y for _, _, pairs in series for _, y in pairs]; ymin, ymax = min(ys), max(ys)
    if ymax == ymin: ymax += 1
    px = lambda x: 65 + (x - xmin) / max(xmax - xmin, 1) * 885
    py = lambda y: 215 - (y - ymin) / (ymax - ymin) * 175
    pieces = [line(65, 215, 950, 215, "#9da9b2"), line(65, 40, 65, 215, "#9da9b2"), text(500, 254, "Native fit_offline epoch over the same fresh 10,000 noisy datasets", anchor="middle")]
    for key, color, pairs in series:
        pieces += [path([(px(x), py(y)) for x, y in pairs], color), text(700, 20 if key == "loss" else 36, "Training" if key == "loss" else "Validation", color=color)]
    for x in (1, 50, 100, 150, 200): pieces.append(text(px(x), 233, str(x), anchor="middle"))
    for y in np.linspace(ymin, ymax, 4): pieces.append(text(58, py(y) + 3, fmt(y), anchor="end"))
    return svg_panel("Native training and validation objective", pieces,
        "This is the FlowMatching objective, not a density negative log likelihood. Its absolute magnitude cannot be compared with the earlier spline loss.", height=275)


def metrics_table(summary):
    if summary["calibration_status"] != "defined": return '<p class="notice">Calibration is undefined because at least one raw draw is nonfinite. All raw draws remain in the saved payload.</p>'
    recovery, support = summary["recovery"], summary["support"]
    rows = []
    for j, label in enumerate(LABELS):
        ci = summary["coverage"]["0.9"]["wilson"][j]
        rows.append([label, fmt(recovery["median_rmse"][j], 4), fmt(recovery["prior_midpoint_rmse"][j], 4),
            percent(summary["coverage"]["0.9"]["coverage"][j]), f'{percent(ci[0])}–{percent(ci[1])}',
            fmt(recovery["median_width90_over_prior"][j]), fmt(summary["rank_dkw"]["rank_ks"][j]),
            "inside band" if summary["rank_dkw"]["passed"][j] else "outside band",
            percent(support["outside_prior"]["per_dimension_fraction"][j])])
    return table(["Target", "Median RMSE", "Prior-midpoint RMSE", "90% coverage", "95% Wilson interval", "90% width / prior", "SBC KS", "SBC screen", "Raw draws outside prior"], rows)


def support_table(summary):
    support = summary["support"]
    rows = [[label, str(support["outside_prior"]["per_dimension_count"][j]), percent(support["outside_prior"]["per_dimension_fraction"][j]),
        str(support["nonphysical"]["per_dimension_count"][j]), str(support["nonfinite"]["per_dimension_count"][j])] for j, label in enumerate(LABELS)]
    rows.append(["Any target in a raw six-vector", str(support["outside_prior"]["sample_count"]), percent(support["outside_prior"]["sample_fraction"]),
        str(support["nonphysical"]["sample_count"]), str(support["nonfinite"]["sample_count"])])
    return table(["Raw target", "Outside-prior count", "Outside-prior fraction", "Nonphysical count", "Nonfinite count"], rows)


def conditional_tables(record):
    result = []
    for case, entry in record["cases"].items():
        comparison = entry.get("comparison", {})
        result.append('<h3>' + esc(case) + '</h3>')
        if comparison.get("status") != "defined":
            result.append('<p>Comparison unavailable: ' + esc(comparison.get("status", entry.get("reason", "existing reference unavailable"))) + '</p>')
            continue
        rows = [[label, fmt(comparison["neural_mean"][j], 4), fmt(comparison["reference_mean"][j], 4),
            fmt(comparison["standardized_mean_error"][j]), fmt(comparison["width_ratio"][j]),
            percent(entry["support"]["outside_prior"]["per_dimension_fraction"][j])] for j, label in enumerate(LABELS)]
        result.append(table(["Target", "Flow mean", "Joint-MCMC mean", "|Mean difference| / ref SD", "90% width ratio", "Raw outside prior"], rows))
        result.append(f'<p>Any-target outside prior: {percent(entry["support"]["outside_prior"]["sample_fraction"])}; nonphysical samples: {entry["support"]["nonphysical"]["sample_count"]} / {entry["support"]["samples"]}.</p>')
        dep = comparison.get("dependence", {})
        if dep.get("status") == "defined":
            result.append(f'<p>Across the 15 unique physical-coordinate Pearson pairs: maximum |correlation difference| {fmt(dep["max_absolute_correlation_difference"])}; median {fmt(dep["median_absolute_correlation_difference"])}. This describes dependence without testing full-joint equality.</p>')
        product = comparison.get("derived_products", {})
        result.append('<p>Derived products: ' + esc(product.get("status", "unavailable")) + ('. ' + esc(product["reason"]) if product.get("reason") else '.') + '</p>')
    return ''.join(result)


def build(protocol_path, evaluation_path, out, *, full_out=None):
    protocol_path, evaluation_path, out = map(Path, (protocol_path, evaluation_path, out))
    if out.exists() or (full_out and Path(full_out).exists()):
        raise FileExistsError("PR87 report output identity must be new")
    protected = {}
    protocol = checked(binding(protocol_path), protected)
    record = checked(binding(evaluation_path), protected)
    if (record.get("format") != "heston_pr87_reproduction_evaluation_v1" or record.get("status") != "complete"
            or record["protocol"] != binding(protocol_path) or record["task"] != protocol["task"]["target"]
            or record["native_epoch"] != 200 or record["new_reference_sampling"] is not False
            or record["reserved_arrays_opened"] is not False or record.get("protected_inputs_unchanged") is not True
            or protocol.get("evaluation") != evaluation.evaluation_contract(record["task"])):
        raise ValueError("completed frozen final-epoch PR87 evaluation required")
    if protocol.get("sources", {}).get("renderer") != binding(__file__):
        raise ValueError("PR87 renderer differs from source frozen before fit")
    native, population = record["native_validation"], record["population"]
    if (native["rows"] != 100 or native["draws_per_case"] != 300 or population["rows"] != 512 or population["draws_per_case"] != 512):
        raise ValueError("native or historical recovery panel differs")
    native_recovery, recovery = load_recovery(native, protected), load_recovery(population, protected)
    selected = selected_references(record["matched_mcmc_report"], recovery, record["task"], protected)
    training_dir = Path(record["checkpoint"]["path"]).parent.parent.parent
    curve_path = training_dir / "native-epoch-history.json"
    curve = checked(binding(curve_path), protected) if curve_path.is_file() else []
    if isinstance(curve, dict): curve = curve.get("history", curve.get("epochs", []))
    chunks = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PR87 native FlowMatching reproduction</title>',
        '<style>body{font:15px/1.55 system-ui,sans-serif;color:#20313e;background:#f4f7f8;margin:0}main{max-width:1190px;margin:auto;padding:30px 22px 65px}h1{font-size:30px;line-height:1.2;margin:10px 0}h2{font-size:23px;margin-top:38px}h3{margin-top:25px}p{max-width:1000px}.kicker{color:#236b9a;letter-spacing:.08em;font-size:12px;font-weight:700}.notice{background:#fff6df;border-left:4px solid #bf8731;padding:14px 18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:12px;margin:14px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #dfe5e9;white-space:nowrap}th{background:#e9eff3}figure{margin:20px 0;padding:12px;background:white;border:1px solid #dfe5e9;border-radius:8px}svg{display:block;width:100%;height:auto}figcaption{font-size:12px;color:#536672;padding:7px}details{margin:18px 0}summary{cursor:pointer;font-weight:650}code{font-size:12px;overflow-wrap:anywhere}footer{margin-top:35px;border-top:1px solid #ccd7df;padding-top:18px;font-size:12px;color:#596a77}</style><main>',
        '<div class="kicker">ONE RUN · FINAL EPOCH 200 · RAW SIX-TARGET POSTERIOR</div><h1>PR87 native FlowMatching reproduction</h1>',
        '<p>Fresh 10,000 training datasets and 100 validation datasets; training-only full-covariance ZCA; native 128×128×128 FlowMatching with dropout 0.1; raw, unbounded six-target adapter; native target standardization, default optimizer, 200 epochs and batch 256. Seed 9011 is the deliberate reproducibility addition to an originally unseeded notebook.</p>',
        '<p class="notice">This reproduces the whole notebook recipe. Fresh simulation draws, network family, target support and optimizer differ from the earlier spline study, so the comparison cannot attribute changes to architecture alone. Every raw draw is retained; no clipping, prior rejection or posterior filtering is applied.</p>']
    if population["calibration_status"] == "defined":
        pc = population["coverage"]["0.9"]["coverage"]
        nc = native["coverage"]["0.9"]["coverage"] if native["calibration_status"] == "defined" else None
        chunks.append(f'<p><strong>On the unchanged historical 512-case panel</strong>, central90% intervals cover {percent(pc[-1])} of c truths and {percent(min(pc[:5]))}–{percent(max(pc[:5]))} of physical-parameter truths. The c posterior-median RMSE is {fmt(population["recovery"]["median_rmse"][-1],4)}; {sum(population["rank_dkw"]["passed"])}/6 marginal SBC screens fall inside their DKW band.</p>')
        if nc: chunks.append(f'<p>The notebook-sized100-case check gives c coverage {percent(nc[-1])}; its 95% Wilson interval is {percent(native["coverage"]["0.9"]["wilson"][-1][0])}–{percent(native["coverage"]["0.9"]["wilson"][-1][1])}. The 100-case and 512-case panels use different observations, and the smaller panel has greater sampling uncertainty.</p>')
    chunks += [learning_plot(curve), '<h2>Notebook-sized native recovery:100×300</h2>',
        '<p>These are the fresh held-out validation datasets from the same run. The first posterior evaluation is the native single 100-row call with 300 draws per case. It is also the validation split used during fitting, so this is development evidence.</p>', metrics_table(native)]
    if native_recovery is not None: chunks.append(recovery_plot(native_recovery, record["task"], "Native100-case validation"))
    chunks += ['<h2>Historical full-prior panel:512×512</h2>',
        '<p>The exact saved 512 raw quote vectors and six generating targets match the earlier reports by both file and array fingerprints. Posterior seeds and three conditional reference identities remain unchanged.</p>', metrics_table(population)]
    if recovery is not None:
        chunks += [recovery_plot(recovery, record["task"], "Historical512-case panel", rows=128), sbc_plot(population)]
    if selected is not None:
        chunks += ['<h2>Selected posterior intervals against existing joint MCMC</h2>', selected_plot(selected),
            table(["Selected case", "Population row", "Reference"], [[str(i + 1), str(index), reason] for i, (index, reason) in enumerate(zip(selected["indices"], selected["reasons"]))])]
    chunks += ['<h2>Raw posterior support</h2>',
        '<p>Counts below include all raw samples, including values outside the declared prior. Nonphysical means a nonpositive variance, reversion, volatility or noise scale; correlation outside[−1,1]; or a nonfinite value. Prior support is narrower than physical validity. Zero draws are discarded, clipped or resampled.</p>',
        '<h3>Native100×300</h3>', support_table(native), '<h3>Historical512×512</h3>', support_table(population),
        '<h2>Three unchanged conditional cases:4096 draws each</h2>',
        '<p>Means, central90% widths and dependence use all six raw target coordinates and the existing ready joint-MCMC chains. The market comparison tests inference under the assumed likelihood; market-model adequacy requires predictive or residual checks.</p>', conditional_tables(record),
        '<details><summary>All coverage levels and diagnostic scope</summary>']
    for label, summary in (("Native100", native), ("Historical512", population)):
        if summary["calibration_status"] == "defined":
            chunks.append('<h3>' + label + '</h3>' + table(["Target", "50%", "80%", "90%", "95%"], [[name] + [percent(summary["coverage"][level]["coverage"][j]) for level in ("0.5", "0.8", "0.9", "0.95")] for j, name in enumerate(LABELS)]))
    chunks += ['<p>Coverage Wilson intervals describe finite population uncertainty; marginal SBC bands adjust over six parameters within each panel. Neither establishes full-joint posterior equality, scientific acceptance or independent performance after model selection. Earlier checkpoints and the minimum-validation checkpoint are archived without further posterior queries.</p></details>',
        '<footer>Offline report from saved summaries and intervals. No neural inference, new pricing or MCMC occurs during rendering.<br>Notebook commit: <code>ddf3619faf48b8f4b0ee56ee841ef24124a508d1</code><br>Protocol SHA256: <code>' + binding(protocol_path)["sha256"] + '</code><br>Evaluation SHA256: <code>' + binding(evaluation_path)["sha256"] + '</code></footer></main></html>']
    document = ''.join(chunks)
    if len(document.encode()) >= LIMIT_BYTES:
        raise ValueError("compact Paseo report must remain below one MiB")
    if any(evaluation.fixed.digest(path) != sha for path, sha in protected.items()):
        raise ValueError("a protected report input changed during rendering")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document)
    summary = {"format": "heston_pr87_reproduction_report_v1", "protocol": binding(protocol_path), "evaluation": binding(evaluation_path),
        "report": binding(out), "report_bytes": len(document.encode()), "inline_svg_count": document.count('<svg'),
        "limit_bytes": LIMIT_BYTES, "protected_inputs_unchanged": True, "producer": binding(__file__), "scientific_acceptance": False}
    if full_out:
        # The compact report already includes all six targets, both panels and
        # every conditional table. An optional full identity may preserve it.
        Path(full_out).write_text(document)
        summary["full_report"] = binding(full_out)
    evaluation.fixed.write_json(out.with_name("report-summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "evaluation", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--full-out", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.protocol, args.evaluation, args.out, full_out=args.full_out)))


if __name__ == "__main__":
    main()

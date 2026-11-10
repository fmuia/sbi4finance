"""Read-only inline SVG figures for the standalone Heston development report.

All inputs are small evidence JSON files or supplied counterfactual noise ratios.
No model payload, bank member, pricing routine or reserved data is accessed.
"""
from html import escape
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


_NAMES = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho")
_SYMBOLS = (r"$v_0$", r"$\kappa$", r"$\theta$", r"$\xi$", r"$\rho$")
_STATUS = {"pass": ("#28744b", "o"), "fail": ("#b33f35", "s"),
           "inconclusive": ("#a66c10", "D")}


def _read(path: Path):
    return json.loads(path.read_text())


def _svg(fig, title: str, description: str) -> str:
    output = StringIO()
    fig.savefig(output, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    text = output.getvalue()
    text = text[text.index("<svg"):]
    prefix = "heston-"+hashlib.sha256(title.encode()).hexdigest()[:10]+"-"
    for identifier in re.findall(r'\bid="([^"]+)"', text):
        text = text.replace('id="'+identifier+'"', 'id="'+prefix+identifier+'"')
        text = text.replace('href="#'+identifier+'"', 'href="#'+prefix+identifier+'"')
        text = text.replace('url(#'+identifier+')', 'url(#'+prefix+identifier+')')
    text = text.replace("<svg ", '<svg role="img" aria-label="'+escape(title, quote=True)+'" ', 1)
    end = text.index(">")+1
    return text[:end]+"\n<title>"+escape(title)+"</title><desc>"+escape(description)+"</desc>"+text[end:]


def _conditional(evidence: Path) -> str:
    evaluation = _read(evidence/"annealing"/"evaluation-060000.json")
    rows = []
    for case, label in (("synthetic-broad", "Broad"), ("synthetic-market-like", "Market-like")):
        values = evaluation["cases"][case]["comparison"]["parameters"]
        rows.extend((label+"  "+symbol, values[name]) for name, symbol in zip(_NAMES, _SYMBOLS, strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.5), sharey=True, layout="constrained")
    left, right = axes
    left.axvspan(-.3, .3, color="#dbece1", zorder=0)
    left.axvline(0, color="#42594a", lw=.7)
    right.axvspan(.75, 1.25, color="#dbece1", zorder=0)
    right.axvline(1., color="#42594a", lw=.7)
    mean_limits, width_limits = [.3], [.75, 1.25]
    for y, (_, record) in enumerate(rows):
        for axis, point_key, interval_key, status_key, collected in (
            (left, "signed_standardized_mean_difference", "signed_standardized_mean_interval", "mean_status", mean_limits),
            (right, "width_ratio_90", "width_ratio_interval", "width_status", width_limits),
        ):
            value = float(record[point_key])
            lo, hi = map(float, record[interval_key])
            if not np.isfinite([value, lo, hi]).all() or not lo <= value <= hi:
                raise ValueError("conditional point/MC interval is nonfinite or inconsistent")
            status = record[status_key]
            color, marker = _STATUS[status]
            axis.errorbar(value, y, xerr=np.array([[value-lo], [hi-value]]), color=color,
                          fmt=marker, ms=6, capsize=3, elinewidth=1.25, zorder=3)
            collected.extend([lo, hi])
    for axis in axes:
        axis.axhline(4.5, color="#9da4aa", lw=.8, ls=":")
        axis.set_ylim(9.65, -.65)
        axis.grid(axis="x", color="#e3e6e9", lw=.7)
        axis.tick_params(axis="y", length=0, pad=8)
        axis.spines[["top", "right"]].set_visible(False)
    left.set_yticks(np.arange(10), [label for label, _ in rows])
    extent = max(abs(float(v)) for v in mean_limits)*1.12
    left.set_xlim(-extent, extent)
    right.set_xlim(min(.65, min(width_limits)*.95), max(width_limits)*1.06)
    left.set(title="Signed mean error", xlabel="Neural − reference mean / reference SD")
    right.set(title="Central-90% interval width", xlabel="Neural width / reference width")
    handles = [Line2D([], [], marker=marker, color=color, ls="", label=status.title(), markersize=6)
               for status, (color, marker) in _STATUS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               title="Individual MC-aware check; error bars show Monte Carlo uncertainty")
    fig.suptitle("60k endpoint: narrower posteriors, remaining conditional failures", fontsize=13)
    return _svg(fig, "60k Heston conditional posterior checks",
        "Ten rows show five physical parameters for broad and market-like synthetic cases. "
        "Left: signed mean errors with the declared minus 0.3 to plus 0.3 reference-SD band. "
        "Right: 90-percent width ratios with the declared 0.75 to 1.25 band. "
        "Squares are fail, diamonds inconclusive and circles pass according to source MC-aware checks. "
        "The full 60k endpoint remains unaccepted.")


def _training(evidence: Path) -> str:
    baseline = _read(evidence/"training-fixed-c1"/"learning-curve.json")
    zero = (_read(evidence/"training-fixed-c1-dropout-zero"/"learning-curve.json")
            +_read(evidence/"training-fixed-c1-dropout-zero-continuation"/"learning-curve.json"))
    annealed = _read(evidence/"annealing"/"learning-curve.json")
    fig, axis = plt.subplots(figsize=(10.8, 4.5), layout="constrained")
    for points, label, color, endpoints in (
        (baseline, "Five-target pilot · dropout .05", "#a65331", (2000, 10000)),
        (zero, "Dropout 0 · fit and continuation to 50k", "#487f9f", (2000, 10000, 25000, 50000)),
        ([zero[-1]]+annealed, "Same 50k state · lower learning rates", "#654a97", (55000, 60000)),
    ):
        x = np.array([p["updates"] for p in points], dtype=float)/1000
        y = np.array([p["validation_probe_loss"] for p in points], dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite training probe record")
        axis.plot(x, y, color=color, label=label, lw=1.7)
        chosen = np.isin(x*1000, endpoints)
        axis.scatter(x[chosen], y[chosen], color=color, s=26, zorder=3)
    axis.axvspan(50, 55, color="#ede8f4", zorder=0)
    axis.axvspan(55, 60, color="#e5dced", zorder=0)
    axis.axvline(50, color="#8a7b99", ls=":", lw=.8)
    axis.axvline(55, color="#8a7b99", ls=":", lw=.8)
    axis.text(52.5, .98, "LR 5e−6", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    axis.text(57.5, .91, "LR 5e−7", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    axis.set(xlabel="Absolute optimizer updates (thousands)",
             ylabel="Fixed validation NLL (transformed target)",
             title="Predeclared runs and endpoints; lower loss does not select an accepted model")
    axis.grid(alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right", bbox_to_anchor=(.84, .98), fontsize=9)
    return _svg(fig, "Heston development training progress",
        "Fixed validation negative log likelihood is shown for this five-target pilot's dropout 0.05 fit to 10k, "
        "the paired zero-dropout fit and continuation to 50k, and a separate learning-rate comparison "
        "from that exact 50k state to 60k. Learning rates are 5e-6 for 50k to 55k and 5e-7 for 55k to 60k. "
        "Dots mark declared diagnostic endpoints; none is accepted or selected as best.")


def _noise(ratios: np.ndarray) -> str:
    values = np.asarray(ratios, dtype=float)
    if values.shape != (54,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("counterfactual noise figure needs 54 finite positive ordered-contract ratios")
    fig, axis = plt.subplots(figsize=(10.8, 3.25), layout="constrained")
    axis.plot(np.arange(1, 55), values, color="#487f9f", marker="o", ms=3, lw=1.1)
    axis.axhline(1., color="#4f6459", ls="--", lw=1, label="Same scale")
    axis.set(yscale="log", xlabel="Contract index in the frozen 54-quote order",
             ylabel="Old-rule SD / observed half-spread",
             title="Counterfactual old noise rule on the same 54 contracts")
    axis.grid(axis="y", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    return _svg(fig, "Counterfactual observation noise comparison on the same 54 contracts",
        "Ordered positive ratios compare the old synthetic noise rule with observed half-spreads "
        "on the same frozen 54 contracts. The logarithmic axis and equality line show scale changes. "
        "This is a counterfactual noise calculation, not an old-versus-new market calibration study.")


def _sbc_records(evidence):
    stages = (("default-010000", "Dropout .05 / 10k", "#a65331"),
              ("zero-010000", "Dropout 0 / 10k", "#85a9bd"),
              ("zero-050000", "Dropout 0 / 50k", "#286784"),
              ("annealed-060000", "Dropout 0 / 60k + lower LR", "#654a97"))
    result = []
    for ident, label, color in stages:
        record = _read(evidence/"sbc-development"/(ident+".json"))
        ranks = np.asarray(record["fractional_ranks"], dtype=float)
        if (ranks.shape != (512,5) or not np.isfinite(ranks).all()
                or np.any(ranks < 0) or np.any(ranks >= 1)
                or record["parameter_names"] != list(_NAMES)
                or record["metrics_exact_match"] is not True):
            raise ValueError("SBC figure requires exactly reproduced development ranks")
        result.append((record, ranks, label, color))
    return result


def _sbc_histogram(evidence):
    record, ranks, _, color = _sbc_records(evidence)[-1]
    fig, axes = plt.subplots(1,5,figsize=(12.0,3.0),sharex=True,sharey=True,layout="constrained")
    for index, axis in enumerate(axes):
        axis.hist(ranks[:,index], bins=np.linspace(0,1,11), color=color, edgecolor="white", linewidth=.7)
        axis.axhline(512/10,color="#2d6147",ls="--",lw=1)
        axis.set(title=_SYMBOLS[index],xlabel="Fractional rank",xlim=(0,1),ylim=(0,None))
        axis.set_xticks((0,.5,1))
        axis.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Datasets per bin")
    fig.suptitle("60k checkpoint: SBC rank histograms across 512 simulated datasets",fontsize=13)
    fig.supxlabel("10 equal bins; dashed line = uniform expected count 51.2. Random fluctuation is expected.",fontsize=9)
    return _svg(fig,"Latest Heston SBC rank histograms",
        "Five histograms show randomized fractional ranks of each generating parameter among 512 posterior draws, "
        "on the same 512 held-out development datasets. Uniform ranks should be approximately flat. "
        "The dashed expected count is not a confidence band. All five marginal DKW screens pass at this checkpoint.")


def _sbc_ecdf(evidence):
    from sbi4finance.calibration.validation import ecdf_difference
    records = _sbc_records(evidence)
    fig, axes = plt.subplots(1,5,figsize=(12.0,3.5),sharex=True,sharey=True,layout="constrained")
    band = float(records[-1][0]["rank_dkw"]["dkw_bound"])
    for axis in axes:
        axis.axhspan(-band,band,color="#dbece1",zorder=0)
        axis.axhline(0,color="#446152",lw=.7)
        axis.set(xlim=(0,1),xlabel="Fractional rank")
        axis.set_xticks((0,.5,1))
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(alpha=.15)
    for record, ranks, label, color in records:
        grid, difference = ecdf_difference(ranks)
        for index, axis in enumerate(axes):
            axis.plot(grid,difference[:,index],color=color,lw=1.3 if record is records[-1][0] else 1.,
                      label=label,alpha=1. if record is records[-1][0] else .85)
    for index, axis in enumerate(axes):
        axis.set_title(_SYMBOLS[index])
    axes[0].set_ylabel("Rank ECDF − uniform CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=9,
               title=f"Green band: ±{band:.4f}, simultaneous marginal DKW reference (reused development pool)")
    fig.suptitle("SBC development history: zero is uniform; large departures flag rank distortion",fontsize=13)
    return _svg(fig,"Heston SBC rank ECDF comparison",
        "Five panels compare rank ECDF minus the uniform CDF for the original 10k dropout fit, "
        "zero-dropout 10k and 50k checkpoints and the lower-rate 60k checkpoint. "
        "The green simultaneous marginal DKW reference band is diagnostic on this repeatedly reused development pool, "
        "not fresh acceptance evidence. Passing these screens does not establish per-dataset or joint posterior accuracy.")


def make_figures(evidence: Path, noise_ratios: np.ndarray) -> dict[str, str]:
    """Return self-contained SVG strings; write no output files."""
    evidence = Path(evidence)
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "heston-spread-report-v1",
                         "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10}):
        return {"conditional": _conditional(evidence), "training": _training(evidence),
                "noise": _noise(noise_ratios), "sbc_histogram": _sbc_histogram(evidence),
                "sbc_ecdf": _sbc_ecdf(evidence)}

"""Plot retained joint100k posterior audits; no model, simulation or inference.

All inputs are compact committed audit summaries and saved case draws. The
curve prior's pointwise quantiles use deterministic one-dimensional quadrature.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/pilot-results/2026-10-29"
OUT = EVIDENCE / "joint100k-sanity-figures"
LABELS = ["H", "eta", "rho", "Short variance", "Long variance", "Decay kappa"]
SHORT = ["H", "eta", "rho", "Short var.", "Long var.", "Decay"]
BLUE, ORANGE, RED, GREY = "#087e8b", "#d58a21", "#a23649", "#9ba6ad"
INPUTS, OUTPUTS = [], []


def read(path):
    INPUTS.append(path)
    return json.loads(path.read_text())


def load(path):
    INPUTS.append(path)
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def save(fig, name):
    for suffix in ("png", "svg"):
        path = OUT / (name + "." + suffix)
        metadata = {"Date": None} if suffix == "svg" else None
        fig.savefig(path, dpi=170, facecolor="white", metadata=metadata)
        OUTPUTS.append(path)
    plt.close(fig)


def contraction(audit):
    rows = audit["parameters"]
    pos = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 5.2), sharey=True)
    for ax in axes:
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-.6, 5.65)
    for y, row in zip(pos, rows):
        q = row["width90_ratio"]
        axes[0].plot([q["q10"], q["q90"]], [y, y], color=BLUE, lw=3, alpha=.65)
        axes[0].plot(q["median"], y, "o", color=BLUE, ms=6)
        axes[1].plot(row["mean_recovery"]["matched_rmse_ratio"], y, "o", color=BLUE, ms=6)
        axes[1].plot(row["mean_recovery"]["offdiagonal_rmse_ratio"], y, "x", color=ORANGE, ms=7, mew=1.7)
        axes[2].plot(row["coverage90"]["matched"], y, "o", color=BLUE, ms=6)
        axes[2].plot(row["coverage90"]["offdiagonal"], y, "x", color=ORANGE, ms=7, mew=1.7)
    axes[0].set_yticks(pos, LABELS)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    axes[0].axvline(1, color=GREY, ls="--", lw=1.2)
    axes[1].axvline(1, color=GREY, ls="--", lw=1.2)
    axes[2].axvline(.9, color=GREY, ls="--", lw=1.2)
    axes[0].set_xlim(0, 1.13)
    axes[1].set_xlim(0, 1.55)
    axes[2].set_xlim(0, 1)
    axes[0].set_title("90% interval width / prior interval width", fontsize=10)
    axes[1].set_title("Mean-estimate RMSE / prior-midpoint RMSE", fontsize=10)
    axes[2].set_title("Coverage of nominal 90% intervals", fontsize=10)
    axes[0].set_xlabel("1 = same width as the uniform prior")
    axes[1].set_xlabel("1 = predicting the prior midpoint")
    axes[2].set_xlabel("Matched cases versus wrong-case truths")
    fig.suptitle("Joint rough Bergomi: informative contraction and data dependence", fontsize=15, y=.98)
    fig.text(.5, .91, "2,000 validation cases x 1,000 retained draws; physical parameter coordinates", ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="Correct case pairing"),
                        Line2D([], [], marker="x", color=ORANGE, ls="", label="Every off-diagonal case pairing")],
               loc="lower center", bbox_to_anchor=(.55, .067), ncol=2, frameon=False)
    fig.text(.5, .025, "Left bars: 10th-90th percentiles across cases, with median dots. Wrong-case controls reuse the same draws and widths.",
             ha="center", fontsize=9, color="#47535b")
    fig.subplots_adjust(left=.115, right=.985, bottom=.24, top=.82, wspace=.15)
    save(fig, "contraction-and-case-pairing")


def corner(case, meta, interior_count):
    values, truth, bounds = case["parameters"], case["truth"], case["prior_bounds"]
    dims = values.shape[1]
    limits = []
    for j in range(dims):
        lo, hi = np.quantile(values[:, j], [.001, .999])
        lo, hi = min(lo, truth[j]), max(hi, truth[j])
        limits.append((lo - .08 * (hi - lo), hi + .08 * (hi - lo)))
    fig, axes = plt.subplots(dims, dims, figsize=(11.8, 11.8))
    for i in range(dims):
        for j in range(dims):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if i == j:
                x = np.linspace(*limits[j], 250)
                mean, sd = values[:, j].mean(), values[:, j].std(ddof=1)
                density = np.exp(-.5 * ((x - mean) / sd)**2) / (sd * np.sqrt(2 * np.pi))
                ax.hist(values[:, j], bins=32, density=True, color=BLUE, alpha=.45, edgecolor="white", linewidth=.3)
                ax.plot(x, density, color=ORANGE, ls="--", lw=1.1)
                ax.axhline(1 / np.diff(bounds[j])[0], color=GREY, ls=":", lw=1)
                ax.axvline(truth[j], color=RED, lw=1.1)
                ax.set_yticks([])
            else:
                ax.scatter(values[:, j], values[:, i], s=3.5, color=BLUE, alpha=.23, edgecolors="none", rasterized=True)
                cov = np.cov(values[:, [j, i]], rowvar=False)
                eigenvalues, vectors = np.linalg.eigh(cov)
                order = np.argsort(eigenvalues)[::-1]
                eigenvalues, vectors = eigenvalues[order], vectors[:, order]
                radii = np.sqrt(eigenvalues * (-2 * np.log(.1)))
                angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
                ax.add_patch(Ellipse(values[:, [j, i]].mean(axis=0), 2*radii[0], 2*radii[1], angle=angle,
                                     fill=False, color=ORANGE, ls="--", lw=.8, alpha=.85))
                ax.plot(truth[j], truth[i], "+", color=RED, ms=6, mew=1.1)
                ax.set_ylim(limits[i])
                ax.text(.04, .94, "r = %+.2f" % meta["pearson"][i][j], transform=ax.transAxes,
                        va="top", fontsize=7, bbox={"facecolor":"white", "alpha":.7, "edgecolor":"none", "pad":1})
            ax.set_xlim(limits[j])
            if i == dims-1:
                ax.set_xlabel(SHORT[j], fontsize=10)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(SHORT[i], fontsize=10)
            elif j > 0:
                ax.tick_params(labelleft=False)
    fig.suptitle("Joint rough Bergomi: complete six-parameter corner | " + meta["case_id"], fontsize=15, y=.985)
    fig.text(.55, .942, "Saved amortized draws; conditional PyMC comparison pending", ha="center", fontsize=11)
    text = ("Exploratory geometry example\nSelected from %d cases with all six\ntruths and bulk posteriors inside priors.\n\nBlue: retained posterior draws\nRed: generating parameters\nOrange: moment-matched Gaussian\n(dashed 90%% ellipses; not posterior contours)\nGray dotted: physical uniform-prior density\n\nAxes zoom to the saved posterior.\nThis plot does not establish conditional accuracy." % interior_count)
    fig.text(.64, .83, text, va="top", fontsize=10, linespacing=1.5, color="#3a4b55")
    fig.subplots_adjust(left=.075, right=.985, bottom=.075, top=.91, wspace=.30, hspace=.15)
    save(fig, "complete-corner-" + meta["case_id"].replace(":", "-"))


def prior_curve_quantiles(times, bounds, nodes=256):
    """Pointwise 5/95% quantiles for independent uniform S,L,k, without MC."""
    z, weights = np.polynomial.legendre.leggauss(nodes)
    k = (bounds[5, 0] + bounds[5, 1])/2 + z*np.diff(bounds[5])[0]/2
    result = np.empty((2, len(times)))
    for j, t in enumerate(times):
        if t == 0:
            result[:, j] = bounds[3, 0] + np.array([.05, .95])*np.diff(bounds[3])[0]
            continue
        a = np.exp(-k*t); b = 1-a
        origin = a*bounds[3, 0] + b*bounds[4, 0]
        A, B = a*np.diff(bounds[3])[0], b*np.diff(bounds[4])[0]
        def cdf(x):
            y = x-origin
            v = (np.maximum(y, 0)**2-np.maximum(y-A, 0)**2-np.maximum(y-B, 0)**2
                 +np.maximum(y-A-B, 0)**2)/(2*A*B)
            return np.dot(weights, np.clip(v, 0, 1))/2
        for qi, q in enumerate((.05, .95)):
            low, high = bounds[3:5, 0].min(), bounds[3:5, 1].max()
            for _ in range(45):
                mid = (low+high)/2
                if cdf(mid) < q: low = mid
                else: high = mid
            result[qi, j] = (low+high)/2
    return result


def curves(cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(12, 4.9), squeeze=False)
    quadrature_error = 0.0
    reference = cases[0][0]
    for ax, (case, role) in zip(axes[0], cases):
        t = case.get("curve_times", reference["curve_times"])
        values, truth = case["parameters"], case["truth"]
        curve = values[:, 4, None]+(values[:, 3]-values[:, 4])[:, None]*np.exp(-values[:, 5, None]*t)
        true = truth[4]+(truth[3]-truth[4])*np.exp(-truth[5]*t)
        if "forward_variance" in case:
            if not np.allclose(curve, case["forward_variance"], rtol=1e-12, atol=1e-14):
                raise ValueError("Curve formula differs from the saved candidate")
        prior = prior_curve_quantiles(t, case["prior_bounds"])
        check = prior_curve_quantiles(t, case["prior_bounds"], nodes=128)
        quadrature_error = max(quadrature_error, float(np.max(np.abs(prior-check))))
        q = np.quantile(curve, [.05, .5, .95], axis=0)
        ax.fill_between(t, *prior, color=GREY, alpha=.27, label="Prior pointwise 90% interval")
        ax.fill_between(t, q[0], q[2], color=BLUE, alpha=.3, label="Posterior pointwise 90% interval")
        ax.plot(t, q[1], color=BLUE, lw=1.8, label="Posterior median")
        ax.plot(t, true, color=RED, ls="--", lw=1.3, label="Generating curve")
        for maturity in case.get("observed_maturities", reference["observed_maturities"]):
            ax.axvline(maturity, color="#b7c5cb", lw=.6, alpha=.55, zorder=0)
        ax.set(title=str(case["case_id"])+": "+role, xlabel="Maturity (years)", ylabel="Annualized forward variance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, .17)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Learning the variance curve despite uncertainty in decay", fontsize=14, y=.98)
    fig.text(.5, .025, "Intervals concern the initial forward-variance curve, not future variance paths. Prior quantiles use deterministic quadrature.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, top=.83, bottom=.28, wspace=.23)
    save(fig, "curve-function-and-weak-decay")
    return quadrature_error


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "svg.hashsalt":"sbi-joint100k-sanity"})
    audit = read(EVIDENCE / "joint100k-contraction-audit/audit.json")
    geometry = read(EVIDENCE / "joint100k-tradeoffs/tradeoffs.json")
    contraction(audit)
    cases = {}
    for meta in geometry["candidates"]:
        case = load(EVIDENCE / "joint100k-tradeoffs" / meta["saved_npz"])
        corner(case, meta, geometry["strict_interior_cases"])
        cases[meta["row"]] = case
    weak = geometry["separate_weak_decay_candidate"]
    weak_path = EVIDENCE / "joint100k-tradeoffs" / weak["saved_npz"]
    if not weak_path.exists(): raise FileNotFoundError("The retained weak-decay case is required")
    weak_case = load(weak_path)
    error = curves([(cases[1983], "interior curve trade-off"), (weak_case, "separate weak-decay case")])
    manifest = {"scope":"Plots of retained learned posteriors and audit summaries; not conditional reference validation",
                "new_inference_or_simulation":False, "quadrature_128_vs_256_max_abs_difference":error,
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS},
                "outputs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUTS},
                "numpy":np.__version__, "matplotlib":matplotlib.__version__}
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"figures":len(OUTPUTS), "quadrature_max_abs_difference":error, "out":str(OUT)}))


if __name__ == "__main__": main()

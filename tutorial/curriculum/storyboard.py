"""Figures of the tutorial curriculum (tutorial/curriculum/).

    python tutorial/curriculum/storyboard.py

Writes to tutorial/curriculum/images/:
  storyboard.png        the tutorial and the minute map of every slot;
  slot-N.png            the minute map of slot N alone, for that slot's file;
  ladder.png            the three models of the tutorial and what one posterior costs under each.
The segments and minutes are the plans of the slot files; change both together.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

TUTORIAL_DIR = Path(__file__).resolve().parents[1]
OUT = TUTORIAL_DIR / "curriculum" / "images"
COSTS = TUTORIAL_DIR / "assets" / "results" / "calibration" / "costs" / "costs.json"

PRESENTER = {"F": "#087F8C", "A": "#D97732", "S": "#738095", "FA": "#7A4E9C"}
NAMES = {"F": "Francesco", "A": "Alex", "S": "Stefan and Jerry", "FA": "Francesco and Alex"}
ALPHA = {"live": 1.0, "cached": 0.45, "slide": 1.0}
TEXT = {"live": "white", "cached": "#1E1E1E", "slide": "#1E1E1E"}

# (label, minutes, presenter, mode); mode is "live" (computed on stage), "cached" (loaded from
# results/) or "slide" (static text or code on screen).
TUTORIAL = ("The tutorial: one smile, three models, 115 minutes", [
    ("1  The problem:\nBayes for\nvolatility models", 15, "F", "slide"),
    ("2  What is\namortized Bayes?", 15, "A", "live"),
    ("3  Network\narchitectures", 10, "S", "cached"),
    ("4  Heston on the\nSPY smile, in\nBayesFlow", 20, "S", "live"),
    ("5  Rough Bergomi on the SPY smile:\nfrom the posterior to a reserve", 40, "FA", "live"),
    ("6  Priors and\nmisspecification", 15, "S", "live"),
])

SLOTS = {
    1: ("Slot 1, 15 minutes, Beamer slides", [
        ("The\nproblem", 1.5, "F", "slide"),
        ("The ingredients on Black–Scholes: the data,\nthe likelihood, prior and posterior,\na price's range, the predictive check", 5, "F", "slide"),
        ("What the quotes\nleave open", 2, "F", "slide"),
        ("What exact posteriors\ncost; amortized\ninference", 3, "F", "slide"),
        ("The afternoon,\ntakeaways,\npresenters", 3, "F", "slide"),
    ]),
    2: ("Slot 2, 15 minutes", [
        ("Two ways\nto pay", 2, "A", "slide"),
        ("A network learns\nslot 1's posterior", 3, "A", "live"),
        ("Why the minimizer\nis the posterior", 3, "A", "slide"),
        ("When it pays:\nbreak-even", 2.5, "A", "cached"),
        ("The workflow\nand SBC", 3, "A", "live"),
        ("Roots and\nhandoff", 1.5, "A", "slide"),
    ]),
    3: ("Slot 3, 10 minutes", [
        ("Two networks,\ntwo jobs", 1, "S", "slide"),
        ("Flows and flow matching:\nnoise carried to a posterior", 4, "S", "cached"),
        ("Reading the data:\nsummary networks", 3, "S", "slide"),
        ("Details that\nmatter", 2, "S", "slide"),
    ]),
    4: ("Slot 4, 20 minutes", [
        ("Heston and\nthe question", 2, "S", "slide"),
        ("Simulator\nand prior", 3, "S", "live"),
        ("Bank, adapter,\nnetwork", 2, "S", "slide"),
        ("Train", 2, "S", "live"),
        ("Check: SBC and the\nexact posterior", 4, "S", "cached"),
        ("Use: SPY's\nposterior", 4, "S", "live"),
        ("What it says\nabout Heston", 2, "S", "live"),
        ("To\nslot 5", 1, "S", "slide"),
    ]),
    5: ("Slot 5, 40 minutes", [
        ("The\nquestion", 3, "F", "cached"),
        ("Why rough\nBergomi", 4, "F", "live"),
        ("What the exact\nposterior costs", 3, "F", "cached"),
        ("The amortized\nposterior for SPY", 6, "A", "live"),
        ("Is it right without\nan exact answer?", 7, "A", "cached"),
        ("The classical\nroute", 4, "F", "cached"),
        ("Prices, range\nand reserve", 5, "F", "live"),
        ("Two\nmodels", 1, "F", "cached"),
        ("Where the quotes stop:\nEWG, a thin chain", 5, "F", "cached"),
        ("What it\nassumes", 2, "F", "slide"),
    ]),
    6: ("Slot 6, 15 minutes", [
        ("What the answer\ndepends on", 1, "S", "slide"),
        ("Changing the prior\nwithout retraining", 5, "S", "live"),
        ("The noise\nmodel", 2, "S", "live"),
        ("When the model is wrong:\nfour detectors", 5, "S", "cached"),
        ("Across\nmodels", 1, "S", "cached"),
        ("Close", 1, "S", "slide"),
    ]),
}


def draw_row(ax, y, segments, height=0.62):
    total = sum(minutes for _, minutes, _, _ in segments)
    start = 0.0
    for label, minutes, who, mode in segments:
        width = minutes / total
        colour = PRESENTER[who]
        gap = 0.0025  # a white gap keeps neighbouring segments of one colour distinct
        box = (start + gap, y), width - 2 * gap, height
        face = "white" if mode == "slide" else colour
        ax.add_patch(Rectangle(*box, facecolor=face, alpha=ALPHA[mode], edgecolor="none"))
        ax.add_patch(Rectangle(*box, facecolor="none", edgecolor=colour, linewidth=1.4))
        ax.text(start + width / 2, y + height / 2, label, ha="center", va="center",
                fontsize=7.0 if width >= 0.035 else 5.6, color=TEXT[mode], linespacing=1.15)
        shown = f"{minutes:g} min"
        ax.text(start + width / 2, y - 0.07, shown, ha="center", va="top", fontsize=6.3, color="0.35")
        start += width


def legend(ax, y):
    handles = [Patch(facecolor=PRESENTER[k], edgecolor=PRESENTER[k], label=NAMES[k]) for k in PRESENTER]
    handles += [Patch(facecolor="0.3", edgecolor="0.3", label="live: computed on stage"),
                Patch(facecolor="0.3", alpha=0.45, edgecolor="0.3", label="cached: loaded from results/"),
                Patch(facecolor="white", edgecolor="0.3", label="slide or code on screen")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=7, fontsize=7.4,
              frameon=False, handlelength=1.6, columnspacing=1.3)


def rows_figure(rows, path, row_height=1.12):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13.5, 0.55 + 1.0 * n))
    ys = [row_height * (n - 1 - i) for i in range(n)]
    for (title, segments), y in zip(rows, ys):
        draw_row(ax, y, segments)
        ax.text(0.0, y + 0.70, title, ha="left", va="bottom", fontsize=9.3, fontweight="bold", color="#1E1E1E")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.62, row_height * (n - 1) + 1.0)
    ax.axis("off")
    legend(ax, 0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ladder(path):
    """What one likelihood evaluation and one exact posterior cost under each model, on one 4-core machine."""
    costs = json.loads(COSTS.read_text())
    heston_eval = costs["heston_likelihood_evaluation_cpu_s"]["median"]
    heston_exact = costs["exact_heston_posterior"]["cpu_seconds"]
    rough_eval = costs["rough_likelihood_evaluation_cpu_s"]["median"]
    rough_exact = costs["exact_rough_posterior"]["cpu_hours"] * 3600
    network = costs["rough_amortized_posterior_1000_draws_cpu_s"]["median"]
    audit = 512 * 1.25 * rough_eval  # the full audit: 512 proposals, each after a pilot at a quarter of the paths
    importance = 1024 * rough_eval  # importance sampling from a Gauss-Newton proposal on SPY (checks/rough_spy_is.py)
    # Black-Scholes: one quote's price takes about 10 microseconds and its grid posterior about
    # 0.2 milliseconds (measured with sbi4finance.calibration.surface, 23 Oct 2026).
    models = ["Black–Scholes\n1 parameter, closed form", "Heston\n4 parameters, Fourier integral",
              "Rough Bergomi\n3 parameters and a curve, Monte Carlo"]
    evals = [1.0e-5, heston_eval, rough_eval]
    exact = [2.0e-4, heston_exact, rough_exact]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    x = range(len(models))
    w = 0.36
    ax.bar([i - w / 2 for i in x], evals, w, color="#738095", label="one likelihood evaluation")
    bars = ax.bar([i + w / 2 for i in x], exact, w, color="#087F8C", label="one exact posterior")
    bars[2].set_hatch("//")
    bars[2].set_edgecolor("white")
    ax.axhline(network, color="#D97732", linewidth=1.6, linestyle="--")
    ax.text(-0.45, network * 1.35, f"amortized network: {network:.1f} CPU-s for 1,000 draws (1.4 s on the clock)",
            fontsize=8, color="#D97732")
    ax.axhline(audit, color="#B33E43", linewidth=1.2, linestyle=":")
    ax.text(-0.45, audit * 1.35, f"audit of one rough-Bergomi smile: about {audit / 60:.0f} CPU-min (512 proposals)",
            fontsize=8, color="#B33E43")
    ax.plot([2 + w / 2], [importance], marker="D", color="#7A4E9C", markersize=6, zorder=5)
    ax.annotate(f"importance sampling,\nGauss–Newton proposal:\nabout {importance / 60:.0f} min (one SPY smile)",
                xy=(2 + w / 2, importance), xytext=(1.42, 20), fontsize=7.4, color="#7A4E9C",
                arrowprops={"arrowstyle": "->", "color": "#7A4E9C", "lw": 0.9})
    labels_eval = ["10 µs (one quote)", f"{1000 * heston_eval:.0f} ms (one rung)", f"{rough_eval:.1f} s"]
    labels_exact = ["0.2 ms (vectorized grid)", f"{heston_exact / 60:.0f} min (one rung)",
                    f"chain: at least {rough_exact / 3600:.0f} h"]
    for i in x:
        ax.text(i - w / 2, evals[i] * 1.5, labels_eval[i], ha="center", fontsize=7.6)
        inside = i == 1  # the Heston label would sit on the audit line; write it inside the bar
        ax.text(i + w / 2, exact[i] * (0.25 if inside else 1.5), labels_exact[i], ha="center", fontsize=7.6,
                color="white" if inside else "black")
    ax.set_yscale("log")
    ax.set_ylim(3e-6, 1e6)
    ax.set_xticks(list(x), models, fontsize=8.5)
    ax.set_ylabel("CPU-seconds, log scale")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def heston_variance_swaps(path):
    """The exact Heston posterior's variance-swap curve on SPY against SPY's replicated curve (slot 4's ending)."""
    run = json.loads((TUTORIAL_DIR / "assets" / "data" / "heston-spy-exact.json").read_text())
    vs = run["variance_swap_vol_pct"]
    days = vs["days"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.fill_between(days, vs["q05"], vs["q95"], color="#738095", alpha=0.35, linewidth=0,
                    label="Heston posterior, 90% interval")
    ax.plot(days, vs["q50"], color="#738095", linewidth=1.8, label="Heston posterior, median")
    ax.plot(days, vs["spy_replicated"], "o", color="#B33E43", markersize=5,
            label="SPY, replicated from the whole chain")
    for d, model, market in zip(days, vs["q50"], vs["spy_replicated"]):
        ax.annotate(f"{model - market:+.1f}", (d, market), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color="#B33E43")
    ax.set_xscale("log")
    ax.set_xticks(days, ["1w", "2w", "1m", "2m", "3m", "6m", "1y"])
    ax.minorticks_off()
    ax.set_xlabel("maturity")
    ax.set_ylabel("variance-swap volatility (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Heston on the SPY smile of 21 Oct 2026: precise, and below the market (labels: model minus market)",
                 fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_figure([TUTORIAL] + [SLOTS[k] for k in sorted(SLOTS)], OUT / "storyboard.png")
    for k, row in SLOTS.items():
        rows_figure([row], OUT / f"slot-{k}.png")
    ladder(OUT / "ladder.png")
    heston_variance_swaps(OUT / "heston-spy-variance-swaps.png")
    print(OUT)


if __name__ == "__main__":
    main()

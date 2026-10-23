"""Storyboard of the ICAIF '26 tutorial: the six slots, and minute maps of slots 1, 4 and 5.

Writes tutorial_final/images/curriculum-storyboard.png. The segments and their minutes are the
plans of tutorial_final/CURRICULUM.md, sections 2 to 5; change both together.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tutorial_final" / "images" / "curriculum-storyboard.png"

PRESENTER = {"F": "#087F8C", "A": "#D97732", "S": "#738095", "FA": "#7A4E9C"}
NAMES = {"F": "Francesco", "A": "Alex", "S": "Stefan", "FA": "Francesco and Alex"}

# (label, minutes, presenter, mode); mode is "live" (computed on stage), "cached" (loaded from
# results/), or "slide" (static text).
ROWS = [
    ("The tutorial, 115 minutes", [
        ("1  Problem:\nBayes for\nvolatility models", 15, "F", "slide"),
        ("2  What is\namortized Bayes?", 15, "A", "slide"),
        ("3  Network\narchitectures", 10, "S", "slide"),
        ("4  BayesFlow on\nVOO returns (GBM)", 20, "S", "live"),
        ("5  SPY smile to a reserve:\nrough Bergomi, audited", 40, "FA", "live"),
        ("6  Priors and\nmisspecification", 15, "S", "slide"),
    ]),
    ("Slot 1, 15 minutes", [
        ("Two fits,\ntwo prices", 2, "F", "cached"),
        ("One quote gives a posterior;\ntwo quotes reject the model", 5, "F", "live"),
        ("Why desks don't:\nthe 2x2 map and its costs", 4, "F", "cached"),
        ("Train once,\ncheck always", 2, "F", "slide"),
        ("The\nafternoon", 2, "F", "slide"),
    ]),
    ("Slot 4, 20 minutes", [
        ("VOO's\nquestion", 2, "S", "live"),
        ("Simulator\nand prior", 3, "S", "live"),
        ("Adapter and\nnetworks", 3, "S", "slide"),
        ("Train", 2, "S", "live"),
        ("Check: diagnostics\nand the exact posterior", 4, "S", "live"),
        ("Use: posterior,\nVaR, ES", 3, "S", "live"),
        ("VOO rejects\nthe model", 2, "S", "live"),
        ("To\nslot 5", 1, "S", "slide"),
    ]),
    ("Slot 5, 40 minutes", [
        ("The desk's\nquestion", 3, "F", "cached"),
        ("Why rough\nBergomi", 5, "F", "live"),
        ("Why no exact\nposterior", 3, "F", "cached"),
        ("The amortized\nposterior for SPY", 6, "A", "live"),
        ("Is it right? SBC, the exact\nHeston check, the audit", 8, "A", "cached"),
        ("The classical\nroute", 4, "F", "cached"),
        ("Prices, range\nand reserve", 6, "F", "live"),
        ("Across\nmodels", 3, "F", "cached"),
        ("What it\nassumes", 2, "F", "slide"),
    ]),
]

STYLE = {"live": dict(alpha=1.0, hatch=None, text="white"),
         "cached": dict(alpha=0.45, hatch=None, text="#1E1E1E"),
         "slide": dict(alpha=0.0, hatch=None, text="#1E1E1E")}


def draw_row(ax, y, segments, height=0.62):
    total = sum(minutes for _, minutes, _, _ in segments)
    start = 0.0
    for label, minutes, who, mode in segments:
        width = minutes / total
        colour = PRESENTER[who]
        face = colour if mode != "slide" else "white"
        gap = 0.0025  # white gap so that neighbouring segments of one colour stay distinct
        box = (start + gap, y), width - 2 * gap, height
        ax.add_patch(Rectangle(*box, facecolor=face, alpha=STYLE[mode]["alpha"] or 1.0, edgecolor="none"))
        ax.add_patch(Rectangle(*box, facecolor="none", edgecolor=colour, linewidth=1.4))
        ax.text(start + width / 2, y + height / 2, label, ha="center", va="center", fontsize=7.2,
                color=STYLE[mode]["text"], linespacing=1.15)
        ax.text(start + width / 2, y - 0.07, f"{minutes} min", ha="center", va="top", fontsize=6.4, color="0.35")
        start += width


def main():
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ys = [3.15, 2.1, 1.05, 0.0]
    for (title, segments), y in zip(ROWS, ys):
        draw_row(ax, y, segments)
        ax.text(0.0, y + 0.70, title, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#1E1E1E")
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.62, 4.05)
    ax.axis("off")
    handles = [Patch(facecolor=PRESENTER[k], edgecolor=PRESENTER[k], label=NAMES[k]) for k in PRESENTER]
    handles += [Patch(facecolor="0.3", edgecolor="0.3", label="live: computed on stage"),
                Patch(facecolor="0.3", alpha=0.45, edgecolor="0.3", label="cached: loaded from results/"),
                Patch(facecolor="white", edgecolor="0.3", label="slide or code on screen")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=7.6,
              frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()

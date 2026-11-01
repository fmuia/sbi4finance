"""Export slot 1's figures for the slides: run the notebook with the deck's palette and save each figure as a PDF.

Run from tutorial/:

    python slides/slot1/export_figures.py [--rubik PATH/TO/Rubik-Variable.ttf]

The notebook is not modified. For this run only, the colours of `sbi4f_tutorial.plots` become the BayesOps theme's
(navy, periwinkle, magenta, teal on white) and every figure is drawn at 72% of its notebook size, so its text stays
legible at the width of a slide. With --rubik (the variable font in the Overleaf project's fonts/ folder), the figures
use Rubik like the slides; without it, matplotlib's default sans-serif. The PDFs land in slides/slot1/figures/slot1/.
"""

import argparse
import importlib.util
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from sbi4f_tutorial import plots  # noqa: E402

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parents[1] / "notebooks" / "slot1_intro.py"
OUT = HERE / "figures" / "slot1"
SCALE = 0.72  # the notebook draws figures 9.6 in wide; the slides show them 5 to 6 in wide
NAMES = {  # the first panel's title -> the file slot1.tex includes
    "SPY's option chain": "chain-reach", "SPY's smile at seven": "smiles", "one smile: level": "smile-shape",
    "the 63 contracts": "grid-contracts", "step 1: the error": "noise-rule", "SPY at 1 month": "quote-information",
    "the model maps": "one-price", "prior 20%": "prior-sweep", "posterior of σ from simulated": "more-prices",
    "the one-month put": "put-range", "the put at δ": "predictive-check", "the assumed noise": "noise-frames",
    "SPY's smile, Black": "heston-smile", "how far a product": "products", "Heston on SPY, 63": "tails",
    "Heston, 15 SPY": "curved-posterior", "forward-start call": "reserves", "the cost of": "valuation-run",
    "SPY's prices in noise": "noise-units", "SPY's forward-variance": "variance-curve",
}


def add_rubik(variable_font, folder):
    """Static Regular and SemiBold instances of Rubik's variable font, registered with matplotlib."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    for weight, name in ((400, "Rubik-Regular.ttf"), (600, "Rubik-SemiBold.ttf")):
        font = instancer.instantiateVariableFont(TTFont(variable_font), {"wght": weight}, updateFontNames=True)
        font.save(folder / name)
        fm.fontManager.addfont(str(folder / name))
    return ["Rubik", "DejaVu Sans"]  # DejaVu supplies the Greek letters Rubik lacks


def deck_palette(families):
    """The theme's colours (aviation #15153B, neonlake #6969FF) on the notebook's figure style."""
    plots.SURFACE, plots.INK, plots.INK_2, plots.MUTED = "#ffffff", "#15153b", "#3f3f63", "#7c7c99"
    plots.GRID, plots.BASELINE = "#e7e7f1", "#c4c4d6"
    plots.SERIES = ("#6969ff", "#c2187a", "#1a9c86")  # periwinkle, magenta, teal: they differ in lightness too
    notebook_style = plots.style

    def style():
        notebook_style()
        plt.rcParams.update({
            "font.family": families, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
            "figure.facecolor": plots.SURFACE, "axes.facecolor": plots.SURFACE, "savefig.facecolor": plots.SURFACE,
            "axes.edgecolor": plots.BASELINE, "axes.labelcolor": plots.INK_2, "axes.titlecolor": plots.INK,
            "grid.color": plots.GRID, "xtick.color": plots.MUTED, "ytick.color": plots.MUTED,
            "xtick.labelcolor": plots.INK_2, "ytick.labelcolor": plots.INK_2, "legend.labelcolor": plots.INK_2,
        })

    plots.style = style


def run_notebook():
    """Run the notebook, keeping every figure it creates, drawn at SCALE of its size."""
    figures, make_figure = [], plt.figure

    def figure(*args, **kwargs):
        if kwargs.get("figsize") is not None:
            kwargs["figsize"] = tuple(SCALE * x for x in kwargs["figsize"])
        figures.append(make_figure(*args, **kwargs))
        return figures[-1]

    plt.figure = figure
    spec = importlib.util.spec_from_file_location("slot1_intro", NOTEBOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.run()
    plt.figure = make_figure
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rubik", type=Path, help="Rubik's variable font, e.g. the Overleaf project's fonts/Rubik-Variable.ttf")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        families = add_rubik(args.rubik, Path(folder)) if args.rubik else ["DejaVu Sans"]
        deck_palette(families)
        figures = run_notebook()
        saved = []
        for fig in figures:
            first = next((t for t in (ax.get_title(loc="left") or ax.get_title() for ax in fig.axes) if t), "")
            name = next((v for k, v in NAMES.items() if first.startswith(k)), None)
            if name is None:
                raise SystemExit(f"a figure titled {first!r} has no file name in NAMES")
            fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
            saved.append(name)
    missing = sorted(set(NAMES.values()) - set(saved))
    if missing:
        raise SystemExit(f"no figure for {missing}: a figure's title changed in the notebook")
    print(f"{len(saved)} figures in {OUT}")


if __name__ == "__main__":
    main()

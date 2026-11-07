"""Export slot 1's figures for the slides: run the notebook with the deck's palette and save each figure as a PDF.

Run from tutorial/:

    python slides/slot1/export_figures.py [--rubik PATH/TO/Rubik-Variable.ttf]

The notebook is not modified. For this run only, the colours of `sbi4f_tutorial.plots` become the BayesOps theme's
(navy, periwinkle, magenta, teal on white) and every figure is drawn at 72% of its notebook size, so its text stays
legible at the width of a slide. With --rubik (the variable font in the Overleaf project's fonts/ folder, or Google
Fonts' Rubik[wght].ttf), the figures use Rubik like the slides; without it, matplotlib's default sans-serif.

Only the figures the deck includes are saved (the values of NAMES that are not None); the others are recognized by
their title and skipped. One figure does not come from the notebook: `same-short-end`, drawn here from Heston's
variance-swap formula alone. The PDFs land in slides/slot1/figures/slot1/.
"""

import argparse
import importlib.util
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sbi4f_tutorial import plots  # noqa: E402
from sbi4f_tutorial.shared import plots as shared_plots  # noqa: E402

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parents[1] / "notebooks" / "slot1_intro.py"
OUT = HERE / "figures" / "slot1"
SCALE = 0.72  # the notebook draws figures 9.6 in wide; the slides show them 5 to 6 in wide
NAMES = {  # the first panel's title -> the file slot1.tex includes, or None for a figure the deck does not use
    "SPY's option chain": "chain-reach", "SPY's smile at seven": "smiles", "one smile: level": "smile-shape",
    "the 63 contracts": "grid-contracts", "step 1: the error": "noise-rule", "SPY at 1 month": "quote-information",
    "the model maps": "one-price", "prior 20%": "prior-sweep", "posterior of σ from simulated": None,
    "the one-month put": "put-range", "the put at δ": "predictive-check", "the assumed noise": None,
    "SPY's smile, Black": None, "how far a product": None, "Heston on SPY, 63": None,
    "Heston, 15 SPY": None, "forward-start call": None, "the cost of": None,
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
    """The theme's colours (aviation #15153B, neonlake #6969FF) on the notebook's figure style.

    `sbi4f_tutorial.plots` re-exports `sbi4f_tutorial.shared.plots`; the notebook reads the first and the shared
    teaching helpers the second, so both are patched.
    """
    colours = {
        "SURFACE": "#ffffff", "INK": "#15153b", "INK_2": "#3f3f63", "MUTED": "#7c7c99",
        "GRID": "#e7e7f1", "BASELINE": "#c4c4d6",
        "SERIES": ("#6969ff", "#c2187a", "#1a9c86"),  # periwinkle, magenta, teal: they differ in lightness too
    }
    notebook_style = shared_plots.style

    def style():
        notebook_style()
        plt.rcParams.update({
            "font.family": families, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
            "figure.facecolor": colours["SURFACE"], "axes.facecolor": colours["SURFACE"],
            "savefig.facecolor": colours["SURFACE"], "axes.edgecolor": colours["BASELINE"],
            "axes.labelcolor": colours["INK_2"], "axes.titlecolor": colours["INK"], "grid.color": colours["GRID"],
            "xtick.color": colours["MUTED"], "ytick.color": colours["MUTED"], "xtick.labelcolor": colours["INK_2"],
            "ytick.labelcolor": colours["INK_2"], "legend.labelcolor": colours["INK_2"],
        })

    for module in (plots, shared_plots):
        for name, value in colours.items():
            setattr(module, name, value)
        module.style = style


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


def heston_variance_swap(v0, theta, kappa, maturity):
    """Heston's fair variance-swap level: the average expected variance, theta + (v0 - theta)(1 - e^{-kT}) / (kT)."""
    weight = (1 - np.exp(-kappa * maturity)) / (kappa * maturity)
    return theta + (v0 - theta) * weight


def same_short_end():
    """Three Heston calibrations that agree on the quoted short end and part beyond it, from the formula alone.

    The quoted levels are illustrative: the variance-swap levels at 1, 3 and 6 months of v0 = 0.018, theta = 0.066,
    kappa = 2. For kappa = 1 and 4, (v0, theta) are refitted by least squares to those three levels; the variance
    swap is linear in (v0, theta), so the fit is exact linear algebra. Returns the figure and the numbers it shows.
    """
    quoted = np.array([1 / 12, 0.25, 0.5])
    target = heston_variance_swap(0.018, 0.066, 2.0, quoted)
    fits = {}
    for kappa in (1.0, 2.0, 4.0):
        weight = (1 - np.exp(-kappa * quoted)) / (kappa * quoted)
        design = np.column_stack([weight, 1 - weight])  # columns: v0, theta
        (v0, theta), *_ = np.linalg.lstsq(design, target, rcond=None)
        miss = np.max(np.abs(np.sqrt(heston_variance_swap(v0, theta, kappa, quoted)) - np.sqrt(target)))
        fits[kappa] = {"v0": v0, "theta": theta, "max_miss_points": 100 * miss,
                       "two_year": 100 * np.sqrt(heston_variance_swap(v0, theta, kappa, 2.0))}

    maturity = np.linspace(1 / 52, 3, 400)
    plots.style()
    fig, ax = plt.subplots(figsize=(SCALE * 9.6, SCALE * 3.6), constrained_layout=True)
    ax.axvspan(0, 0.5, color=plots.SERIES[0], alpha=0.07, lw=0)
    ax.text(0.03, 27.4, "maturities the quotes cover", fontsize=8, color=plots.INK_2)
    for colour, (kappa, fit) in zip((plots.SERIES[1], plots.INK, plots.SERIES[2]), fits.items()):
        level = 100 * np.sqrt(heston_variance_swap(fit["v0"], fit["theta"], kappa, maturity))
        ax.plot(maturity, level, color=colour, lw=1.8,
                label=f"κ = {kappa:g}, long-run volatility {100 * np.sqrt(fit['theta']):.0f}%")
        ax.plot([2.0], [fit["two_year"]], "o", color=colour, ms=4)
        ax.text(2.08, fit["two_year"] + 0.15, f"{fit['two_year']:.1f}%", va="bottom", fontsize=8, color=colour)
    ax.errorbar(quoted, 100 * np.sqrt(target), yerr=0.3, fmt="o", color=plots.INK, ms=3.5, capsize=2,
                label="the levels the quotes fix, ± 0.3 points")
    ax.axvline(2.0, color=plots.BASELINE, lw=0.8, ls=":")
    ax.text(1.97, 27.4, "2-year variance swap", fontsize=8, color=plots.INK_2, ha="right")
    ax.set_xlim(0, 3)
    ax.set_ylim(13, 28.5)
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("variance-swap volatility (%)")
    ax.set_title("same short end, different long end: three Heston calibrations", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    return fig, fits


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
            key = next((k for k in NAMES if first.startswith(k)), None)
            if key is None:
                raise SystemExit(f"a figure titled {first!r} has no entry in NAMES")
            if NAMES[key] is not None:
                fig.savefig(OUT / f"{NAMES[key]}.pdf", bbox_inches="tight", pad_inches=0.02)
                saved.append(NAMES[key])
        fig, fits = same_short_end()
        fig.savefig(OUT / "same-short-end.pdf", bbox_inches="tight", pad_inches=0.02)
        saved.append("same-short-end")
    missing = sorted({name for name in NAMES.values() if name} - set(saved))
    if missing:
        raise SystemExit(f"no figure for {missing}: a figure's title changed in the notebook")
    print(f"{len(saved)} figures in {OUT}")
    for kappa, fit in fits.items():
        print(f"same-short-end, kappa {kappa:g}: long-run volatility {100 * np.sqrt(fit['theta']):.1f}%, "
              f"largest miss on the quoted levels {fit['max_miss_points']:.2f} points, 2-year {fit['two_year']:.1f}%")


if __name__ == "__main__":
    main()

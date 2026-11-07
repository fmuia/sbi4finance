# Slot 1 slides

The 15-minute talk of slot 1, *The problem: Bayes for volatility models*, as Beamer source in the tutorial's BayesOps
theme. The workshop opens with Beamer talks (introduction, amortized Bayes, architectures) and then works through the
three application notebooks; this deck is the first talk. It has 12 content frames after the title page, and 12 backup
frames after `\appendix`. The plan, the minute map, and the source and status of every number are in
[`curriculum/01-slot1-the-problem.md`](../../curriculum/01-slot1-the-problem.md).

| File | What |
|---|---|
| `slot1.tex` | The deck: a main document of its own, using `\usepackage[cover]{bayesops}` like the project's `main.tex` |
| `figures/slot1/*.pdf` | The 13 figures the deck includes, in the theme's colours, as vector PDFs |
| `export_figures.py` | Regenerates the figures: twelve from the notebook, one from Heston's variance-swap formula |

## In the Overleaf project

The theme (`bayesops.sty`), its fonts (`fonts/`) and its logos (`assets/`) live in the tutorial's Overleaf project and
are not copied here. Upload `slot1.tex` and the folder `figures/slot1/` to that project, keeping the paths, then choose
Menu › Main document › `slot1.tex` and the LuaLaTeX compiler, which the theme needs. The project's `main.tex` is left
as it is.

The deck adds a few helpers to the theme, at the top of `slot1.tex`: `\stat` for a headline number, three block
colours (definition, reassuring result, warning), `\highlight`, `\source` for a source line, the ragged table column
`L`, and `\todo` for a placeholder. It loads `unicode-math` with Latin Modern Math and the `array` package. Dashes are
typed as Unicode (–), because the theme's title fonts do not turn `--` into a dash.

Before the talk, fill the `\todo` placeholders on the last frame (the presenters' affiliations and who presents the
return histories) and confirm the assignments, which follow the curriculum of 24 October. Speaker notes are in
`\note{}`; Beamer prints them only with `\setbeameroption{show notes on second screen}`.

## Regenerating the figures

From `tutorial/`, after a change to the notebook:

```bash
pixi run --locked python slides/slot1/export_figures.py --rubik PATH/TO/fonts/Rubik-Variable.ttf
```

`--rubik` takes the variable font from the Overleaf project's `fonts/` folder, or Google Fonts' `Rubik[wght].ttf`, so
that the figures use the slides' typeface; without it they use matplotlib's default sans-serif. The script runs the
notebook once with the theme's colours, saves the figures the deck includes under the names `slot1.tex` uses, skips the
notebook's other figures, and draws `same-short-end.pdf` from the formula. The notebook itself is not changed.

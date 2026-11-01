# Slot 1 slides

The 15-minute talk of slot 1, *The problem: Bayes for volatility models*, as Beamer source in the tutorial's BayesOps
theme. It follows [`notebooks/slot1_intro.py`](../../notebooks/slot1_intro.py) section by section: 21 content frames,
the theme's section dividers, takeaways and five backup frames. The plan, the time budget and the source and status of
every number are in [`curriculum/01-slot1-the-problem.md`](../../curriculum/01-slot1-the-problem.md).

| File | What |
|---|---|
| `slot1.tex` | The deck: a main document of its own, using `\usepackage[cover]{bayesops}` like the project's `main.tex` |
| `figures/slot1/*.pdf` | The notebook's figures, in the theme's colours, as vector PDFs |
| `export_figures.py` | Regenerates the figures from the notebook |

## In the Overleaf project

The theme (`bayesops.sty`), its fonts (`fonts/`) and its logos (`assets/`) live in the tutorial's Overleaf project and
are not copied here. Upload `slot1.tex` and the folder `figures/slot1/` to that project, keeping the paths, then choose
Menu › Main document › `slot1.tex` and the LuaLaTeX compiler, which the theme needs. The project's `main.tex` is left
as it is.

The deck adds only a few helpers to the theme, at the top of `slot1.tex`: `\stat` for a headline number, three block
colours (definition, reassuring result, warning) and `\highlight`. It loads `unicode-math` with Latin Modern Math.
Dashes are typed as Unicode (–), because the theme's title fonts do not turn `--` into a dash.

## Regenerating the figures

From `tutorial/`, after a change to the notebook:

```bash
pixi run --locked python slides/slot1/export_figures.py --rubik PATH/TO/fonts/Rubik-Variable.ttf
```

`--rubik` takes the variable font from the Overleaf project's `fonts/` folder so that the figures use the slides'
typeface; without it they use matplotlib's default sans-serif. The script runs the notebook once with the theme's
colours and saves each figure under the name `slot1.tex` includes. The notebook itself is not changed.

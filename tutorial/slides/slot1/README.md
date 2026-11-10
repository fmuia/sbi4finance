# Slot 1 slides

The 15-minute opening talk, *Introduction: Bayes for volatility models*, as Beamer source in the tutorial's BayesOps
theme. The workshop opens with Beamer talks (introduction, amortized Bayes, architectures) and then works through the
application notebooks; this deck is the first talk. It has 15 content frames after the title page, one of them in two
steps, and 5 backup frames after `\appendix`. The plan, the minute map, and the source and status of every number are
in [`curriculum/01-slot1-the-problem.md`](../../curriculum/01-slot1-the-problem.md).

| File | What |
|---|---|
| `slot1.tex` | The deck: a main document of its own, using `\usepackage[cover]{bayesops}` like the project's `main.tex` |
| `figures/slot1/*.pdf` | The 8 figures the deck includes, in the theme's colours, as vector PDFs drawn at the size the slides show them |
| `figures/slot1/*.jpg` | The four presenters' photos, cropped square as on the tutorial's webpage |
| `export_figures.py` | Regenerates the figures and prints every number the slides quote |
| `illustration.py` | The running example behind seven of the figures: Heston on simulated quotes |

## The argument

The talk has five parts, and each key line follows from the previous one.

1. **The problem** (slides 1–3). A model calibrated to quoted options prices what the market does not quote. The
   calibration is one fit among many that are consistent with the quotes, so a price computed from it has a range.
2. **The exercise** (slides 4–5). Volatility models are the example. The five steps (model, simulate, train, infer,
   check) and the questions they answer are general; the methods are the message, more than the examples.
3. **Why Bayes** (slides 6–9). The posterior keeps every fit, weighted. Where the quotes determine the price, the
   classical range and the posterior agree, and the deck says so. What the quotes leave open is set by an assumption:
   fixing a parameter is a prior of zero width. One set of draws then prices every product, and a book.
4. **Why simulation-based inference** (slides 10–12). A posterior costs thousands of model evaluations. A network
   trained once on simulations returns it in seconds, which also makes its coverage testable.
5. **The close** (slides 13–15). Three takeaways, the team, and the rest of the tutorial.

## How a slide is written

Every main slide has one reading order, so that the audience knows what to read first:

1. **The title and the key line** (`\key`) are what everyone reads. The key line is one plain sentence, or two short
   ones, with one idea; `\highlight` marks its few essential words. Most key lines fit on one line.
2. **The figure** is the evidence. Most slides have exactly one.
3. **Everything else is optional**: a strip under the figure of at most three label–value pairs (`\pairs`), each
   value a few words, and a source line (`\source`). Numbers and definitions go there unless they are the point.

Write plain statements, not slogans, and keep the logical order. Explanations go in the speaker notes (`\note{}`),
never in sentences on the slide. No paragraphs, cards or blocks on the main slides; a table only where the table is
the content, as a list of one-line rows (the six questions) or of one row per part (the programme). The backup frames
after `\appendix` hold the rest. One slide, *One Fit Among Many*, has two steps: the quotes with the best fit, then the
other fits and the maturity nobody quotes.

## In the Overleaf project

The theme (`bayesops.sty`), its fonts (`fonts/`) and its logos (`assets/`) live in the tutorial's Overleaf project and
are not copied here. Upload `slot1.tex` and the folder `figures/slot1/` to that project, keeping the paths, then choose
Menu › Main document › `slot1.tex` and the LuaLaTeX compiler, which the theme needs. The project's `main.tex` is left
as it is.

The deck adds a few helpers to the theme, at the top of `slot1.tex`: `\key`, `\pairs` and `\bignum` for the reading
order above; `\detail` for the backup frames; `\highlight`, `\source` for a source line, `\fig` for a figure,
`\person` for a presenter, `\rowrule` for the light rule between the rows of the two lists (slides 5 and 15), the
ragged table column `L`, and the diagram styles, which apply only inside a picture opened with `[deck]`. If the theme
already defines one of these names, LaTeX stops at that line: rename the helper. The deck also uses the theme's
`\TitleFont` for the key lines and for the lists.
The deck loads `unicode-math` with Latin Modern Math and the `array`, `booktabs` and `tikz` packages. Dashes are typed
as Unicode (–), because the theme's title fonts do not turn `--` into a dash.

**The deck has not been compiled with the real theme.** It was compiled with LuaLaTeX (TeX Live 2023) against a
stand-in that has the theme's two colours, Rubik titles and a frame body 14 mm shorter than that of Beamer's default
layout, once with Inter and once with Rubik as the text font; no frame overflows in either. After the upload, check
that no frame overflows in the real theme. If one does, lower that figure's height in `export_figures.py` (the second
argument of `canvas`) and export again.

Speaker notes are in `\note{}`; Beamer prints them only with `\setbeameroption{show notes on second screen}`.

## Regenerating the figures

From `tutorial/`:

```bash
pixi run --locked python slides/slot1/export_figures.py --rubik PATH/TO/fonts/Rubik-Variable.ttf
```

`--rubik` takes the variable font from the Overleaf project's `fonts/` folder, or Google Fonts' `Rubik[wght].ttf`, so
that the figures use the slides' typeface; without it they use matplotlib's default sans-serif. The script needs only
the tutorial's own dependencies, and `fontTools` for `--rubik`.

The running example takes about four minutes to compute on four cores, mostly five posteriors by importance
sampling. Its numbers are cached under `tutorial/.cache/slides-slot1/`, which Git ignores, and reused until
`illustration.py` or the settings at the top of `export_figures.py` change; `--recompute` ignores the cache. The
script ends by printing every number the slides quote, and lists those that `slot1.tex` does not contain. After a
change to the example, update the slides it names (2, 3, 7, 8, 9 and the first backup frame) and the curriculum page.

## The running example

Seven figures come from one example, so that the argument stays on one case: Heston with known parameters generates
35 option prices (7 strikes at 1, 3 and 6 months, 1 and 2 years), each observed once with the tutorial's error band of
0.3 volatility points at the money and 1 point at |δ| = 2. The same surface is then read with quotes that reach
2 years or stop at 1 month. For each reading the example computes the least-squares fit with its Gauss–Newton
covariance and the delta-method range of a product, and the posterior under a uniform prior on the application's
parameter box. The products are a 2-year put struck at 90% of the forward and a call that starts at the money in
1 year and pays at 2 years.

- **It is an illustration, not a result of the tutorial.** The model is right by construction, the quote errors are
  independent, and one simulated surface is shown. The seed is the date of the tutorial, fixed before any figure was
  drawn. On this surface the 6-month delta-method range ends 0.01 below the price under the generating parameters,
  and the range with κ fixed at 1 misses it; both are shown as they came out.
- **Prices** are the tutorial runtime's Fourier prices, at the quadrature rung the Heston lesson uses. The
  forward-start call is one more Fourier integral (Lucic 2003; Kruse and Nögel 2005), checked against the vanilla call
  as the start date goes to zero, against Black–Scholes as the volatility of volatility goes to zero, and against a
  Monte Carlo simulation.
- **The posterior is computed by importance sampling**, not by Markov chain Monte Carlo, so that the script needs only
  the tutorial's dependencies. The proposal mixes the prior with four multivariate t components in logit coordinates,
  placed along the posterior's longest direction and fitted over three rounds to a flattened posterior (the likelihood
  to the power one half), so that a curved valley and its tails are covered. It is then sampled until 6,000 effective
  draws or 120,000 proposals. The script reports the effective sample size (3,200 to 6,900) and the share of failed
  pricings (0.01%, counted as zero likelihood).
- **Each of the five posteriors was compared once with four long PyMC DEMetropolisZ chains** (R-hat at most 1.004, at
  least 1,800 effective draws). Posterior means agree within 0.05 posterior standard deviations, standard deviations
  within 17%, and the 5%, 50% and 95% points of the 2-year put within 4% of the width of its 90% range. A first
  version of the sampler, with one t component fitted to the posterior itself, had put the put's 95% point on the
  3-month reading at 6.65 where the chains give 6.81; the comparison caught it. The chains are not part of this
  folder, because PyMC is not a dependency of the tutorial.

The eighth figure, `coverage.pdf`, is not from the example: it plots the stored validation of the Heston network of
study S2 (`assets/results/calibration/heston/s2/validation.json`, claim C7).

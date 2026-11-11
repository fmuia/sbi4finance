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
| `illustration.py` | The running example behind seven of the figures: Heston on the Heston notebook's simulated quotes |
| `spx-design.json` | The tutorial's 63-quote SPX design of 9 November 2026 (strikes, maturities, forwards, discounts, half-spreads), with its source |
| `market-like-case.json` | The Heston notebook's simulated case "market-like parameters": its 63 observed prices, generating parameters and noise draws, with their source |

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
   trained once on simulations returns it in under a second, which also makes its coverage testable; it is valid for
   data like its training data, which the SPX quotes are not for Heston.
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

**The deck has not been compiled with the real theme.** After the update of 11 November it was compiled with
LuaLaTeX (TeX Live 2023) against a stand-in that has the theme's colours, Rubik titles on Inter text, a frame title
with an accent rule and a footline; no frame overflows. After the upload, check that no frame overflows in the real
theme. If one does, lower that figure's height in `export_figures.py` (the second
argument of `canvas`) and export again.

Speaker notes are in `\note{}`; Beamer prints them only with `\setbeameroption{show notes on second screen}`.

## Regenerating the figures

From `tutorial/`:

```bash
pixi run --locked python slides/slot1/export_figures.py --rubik PATH/TO/fonts/Rubik-Variable.ttf
```

`--rubik` takes the variable font from the Overleaf project's `fonts/` folder, or Google Fonts' `Rubik[wght].ttf`, so
that the figures use the slides' typeface; without it they use matplotlib's default sans-serif. The committed figures
were drawn with `--rubik`. The script needs only
the tutorial's own dependencies, and `fontTools` for `--rubik`.

The running example takes about half an hour to compute on four cores, mostly five posteriors by importance
sampling. Its numbers are cached under `tutorial/.cache/slides-slot1/`, which Git ignores, and reused until
`illustration.py`, `spx-design.json`, `market-like-case.json` or the settings at the top of `export_figures.py`
change; `--recompute` ignores the cache. The script ends by printing every number the slides quote, and lists those that `slot1.tex` does not contain. After a
change to the example, update the slides it names (2, 3, 6, 7, 8, 9 and the first backup frame) and the curriculum
page.

## The running example

Seven figures come from one example, so that the argument stays on one case. It is the simulated case
"market-like parameters" of the tutorial's Heston notebook (`lab/notebooks/heston_spx_calibration.py`, on branch
`claude/trusting-dijkstra-hmgvcm` as of 11 November 2026), in the setting of the option application
(`lab/plans/option-application/`):

- **Design.** The strikes, maturities, forwards, discounts and half-spreads of the 63 actual SPX quotes of
  9 November 2026: 9 strikes at standardized moneyness z = ln(K/F)/(0.16√T) from −5 to +2 on each of 7 expiries,
  from 7 to 325 days ([`spx-design.json`](spx-design.json), copied with its source from the application's record).
  The figures show them, as the notebook does, at δ = ln(K/F)/(0.20√T), from about −4 to 1.6. The market's mids are
  not used.
- **Quotes.** The notebook's 63 observed prices of the case ([`market-like-case.json`](market-like-case.json), copied
  with their source and SHA-256 from its conditional panel): the slides and the notebook read the same numbers. The
  slides' pricer reproduces the case's clean prices to within 10⁻⁵ of a half-spread, which `illustration.py` checks
  on every load.
- **Model and truth.** Heston with the application's parameters (v₀, κ, v∞, σ_v, ρ; the lab documents write v̄ and ξ) and their uniform prior box, at the
  notebook's case "market-like parameters", the application's reference case "market-like, simulated" (v₀ 0.0271,
  κ 4.34, v∞ 0.0565, σ_v 1.80, ρ −0.667).
- **Error law.** The application's: each quote is the model price plus c × its half-spread × a standard normal, with
  c = 1 in the simulation. In the posterior c is inferred under its prior U(0.5, 2), with the likelihood's −n ln c
  term; the classical route estimates it by maximum likelihood, √(R/n), R the sum of squared misses in half-spreads.
- **Readings.** The same surface is read with quotes up to 325, 81, 30, 14 and 7 days (63, 45, 27, 18 and 9 quotes).
  Slides 2, 3, 6, 8 and 9 use the quotes up to 14 days; slide 7 uses all five readings.
- **Products.** The application's forward-start call, struck at the money in 6 months and paid at 1 year; the
  forward-starting variance over its later window, 81 to 325 days; and the listed 325-day at-the-money call, which a
  desk could hold against the forward start (slide 9).

For each reading the example computes the least-squares fit with its Gauss–Newton covariance and the delta-method
range of a product, and the posterior.

- **It is an illustration, not a result of the tutorial.** The model is right by construction, the quote errors are
  independent, and one simulated surface is shown, the notebook's (its noise draws are a little small: 0.93 half-spreads
  root mean square over the 63 quotes, against 1 expected). On the real SPX quotes Heston misses by about 13 half-spreads and c goes to its upper bound of 2 (the
  application's Section 6.5); the example, generated with c = 1, does not show that misfit.
- **Prices** are Lewis Fourier integrals with the analytic Black control variate of the application (its Section 4.1),
  on the tutorial runtime's Heston characteristic function: the 7-day put at z = −5 is worth about 10⁻⁴ of the
  forward and a half-spread about 10⁻⁵, which the plain formula cannot resolve. The quotes are priced with 2,048
  Gauss–Legendre nodes up to 1,200; the next rung (4,096 nodes, 2,400) moves no price by more than
  6 × 10⁻⁷ of its half-spread at 200 posterior draws of every reading, against the application's gate of 0.01. The
  forward-start call is one more Fourier integral (Lucic 2003; Kruse and Nögel 2005), checked against the vanilla call
  as the start date goes to zero and against Black–Scholes as the volatility of volatility goes to zero (and, at the
  money, against a Monte Carlo simulation when the deck was first built); the forward-starting variance is the closed
  form of the application's Section 4.1.
- **The error scale c is integrated out in closed form.** With the uniform prior on [0.5, 2], the likelihood's
  integral over c is an incomplete gamma function of R, so the sampler works on the five Heston parameters, and each
  draw then receives a c from its exact conditional (a truncated gamma variable in R/(2c²)).
- **The posterior is computed by importance sampling**, not by Markov chain Monte Carlo, so that the script needs only
  the tutorial's dependencies. The proposal mixes the prior with multivariate t components in logit coordinates,
  placed along the posterior's longest direction and fitted over several rounds to a flattened posterior, so that a
  curved valley and its tails are covered; it is then sampled until 5,000 effective draws or 72,000 proposals. On the
  7-day reading the posterior is a thin curved ridge whose ends are set by the box: there the proposal is adapted over
  six rounds to the likelihood to the power 0.3, with eight slices and 30% of the proposals from the prior, up to
  150,000 proposals, and the 4,000 resampled draws are then moved by 100 sweeps of differential-evolution Metropolis
  at the posterior. The script reports the effective sample size (1,259 to 5,442 on four readings; 452 on the
  7-day reading, before its moves) and the share of failed pricings (0.02% to 0.04%, counted as zero likelihood).
- **The posteriors were checked against Markov chain Monte Carlo** on the same target (the five Heston parameters and
  c, uniform priors, the same likelihood):
  - *325 days*, all 63 quotes: against the Heston notebook's own PyMC reference for this case (four DEMetropolisZ
    chains of 12,000 draws, `lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/`). Every 5%, 50% and 95%
    point of the six parameters and of the later-window variance agrees within 0.02 of the reference's 90% width;
    `export_figures.py` prints the comparison.
  - *14 days*, the opening example, against four DEMetropolisZ chains of 40,000 draws after 20,000 tuning steps,
    started from dispersed least-squares fits (R-hat at most 1.008, at least 487 effective draws): means within 0.06
    standard deviations, standard deviations within 10% (the sampler's tails of κ and the long-run variance are a
    little longer), the forward start's 5%, 50% and 95% points within 2% of the width of its 90% range.
  - *7 days*: the same check, with 80,000 draws after 40,000, is being rerun on this surface.

  The chains are not part of this folder, because PyMC is not a dependency of the tutorial.

The eighth figure, `coverage.pdf`, is not from the example: it plots the checks of the option application's Heston
network on 512 surfaces simulated from the prior, six targets (its Section 8.2: 300,000 training rows, flow matching,
whitened inputs). Its numbers are copied into `export_figures.py` with their source, because the slides' code reads
nothing outside `tutorial/`.

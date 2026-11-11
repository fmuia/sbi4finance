# Slot 1 slides

The 15-minute opening talk, *Introduction: Bayes for volatility models*, as Beamer source in the tutorial's BayesOps
theme. The workshop opens with Beamer talks (introduction, amortized Bayes, architectures) and then works through the
application notebooks; this deck is the first talk. It has 16 content frames after the title page, one of them in two
steps, and 7 backup frames after `\appendix`. The plan, the minute map, and the source and status of every number are
in [`curriculum/01-slot1-the-problem.md`](../../curriculum/01-slot1-the-problem.md).

| File | What |
|---|---|
| `slot1.tex` | The deck: a main document of its own, using `\usepackage[cover]{bayesops}` like the project's `main.tex` |
| `figures/slot1/*.pdf` | The 9 figures the deck includes, in the theme's colours, as vector PDFs drawn at the size the slides show them |
| `figures/slot1/*.jpg` | The four presenters' photos, cropped square as on the tutorial's webpage |
| `export_figures.py` | Regenerates the figures and prints every number the slides quote |
| `illustration.py` | The running example behind seven of the figures: a thin option chain simulated from Heston |
| `check_posterior.py` | An independent check of the example's posteriors by Metropolis chains; nothing on the slides depends on it |
| `recorded-results.json` | The recorded results of the Heston SPX study that slides 10 to 13 quote, each with its source file, commit and SHA-256 |
| `spx-design.json` | The tutorial's 63-quote SPX design of 9 November 2026 (strikes, maturities, forwards, discounts, half-spreads), with its source |
| `market-like-case.json` | The Heston notebook's simulated case "market-like parameters": its 63 observed prices, generating parameters and noise draws, with their source |

## The argument

The talk has five parts, and each key line follows from the previous one.

1. **The problem** (slides 1–3). A model calibrated to quoted options prices what the market does not quote. The
   calibration is one fit among many that are consistent with the quotes, so a price computed from it has a range.
2. **The exercise** (slides 4–5). Volatility models are the example. The five steps (model, simulate, train, infer,
   check) and the questions they answer are general; the methods are the message, more than the examples.
3. **Why Bayes** (slides 6–9). The posterior keeps every fit, weighted. Where the quotes determine the price, the
   classical range and the posterior agree, and the deck says so, also for the Heston notebook's own 63 quotes. What
   the quotes leave open is set by an assumption: a fixed parameter is an assumption of zero width, a prior states
   its width. One set of draws then prices every product, and a book.
4. **Why simulation-based inference** (slides 10–13). An exact posterior costs minutes for Heston and months for
   rough Bergomi. A network trained once on simulations returns it in 0.3 seconds, which makes its coverage testable.
   The tests say where the network holds and where it does not.
5. **The close** (slides 14–16). Three takeaways, the team, and the rest of the tutorial.

## Two kinds of evidence

The deck shows two kinds of evidence, never on the same slide. The rule above the strip of facts carries a tag that
says which one a slide shows.

| Tag | Slides | What it is | Where it comes from |
|---|---|---|---|
| ILLUSTRATION · SIMULATED QUOTES | 2, 3, 6, 7, 8, 9 | One running example, a thin option chain simulated from Heston, the model right by construction. It shows a mechanism. | `illustration.py` |
| RESULT · HESTON ON SPX · AS OF 11 NOV | 10, 11, 12, 13 | What the tutorial's Heston study on 63 SPX quotes had recorded on 11 November 2026 | `recorded-results.json` |

Slide 13 states the status of every result with a mark and a legend: a check that holds, one that holds with a
caveat, one that raised a flag, and one still to come. Slide 10 separates measured costs from estimated ones by
colour and by label. On 11 November the status was:

| Check | Status | Recorded |
|---|---|---|
| Speed | holds | 0.27 s per posterior (615 posteriors in 164 s), against 14.0 to 26.6 CPU-minutes for an exact posterior |
| Coverage on 512 simulated surfaces | holds for 5 of 6 parameters | 90% intervals contain the truth 89.3 to 92.4% of the time for κ, v∞, σ_v, ρ and c; 97.5% for v₀, whose intervals are too wide |
| Against exact posteriors on two simulated surfaces | caveat | means within 0.96 exact standard deviations; 90% intervals 1.05 to 2.9 times as wide as the exact ones |
| On the SPX quotes of 9 November | flag | Heston misses the quotes by 13.4 half-spreads (root mean square), so they are unlike the network's training data; the market results use the exact posterior |
| Rough Bergomi | pending | the simulation bank was being generated; no result is shown |

These are development results (`scientific_acceptance` is false in their run records), not studies of the claims
ledger, and the deck cites no claim of the ledger. Update slide 13 and `recorded-results.json` together if the
record changes before the talk.

## How a slide is written

Every main slide has one reading order, so that the audience knows what to read first:

1. **The title and the key line** (`\key`) are what everyone reads. The key line is one plain sentence, or two short
   ones, with one idea; `\highlight` marks its few essential words. Every key line fits on one line.
2. **The figure** is the evidence. Most slides have exactly one.
3. **Everything else is optional**: a strip under the figure of at most three label–value pairs (`\pairs`), each
   value a few words, and a source line (`\source`). Numbers and definitions go there unless they are the point.

Write plain statements, not slogans, and keep the logical order. Explanations go in the speaker notes (`\note{}`),
never in sentences on the slide. No paragraphs, cards or blocks on the main slides; a table only where the table is
the content, as a list of one-line rows (the six questions, the five checks) or of one row per part (the programme).
The backup frames after `\appendix` hold the rest. One slide, *One Fit Among Many*, has two steps: the quotes with
the best fit, then the other fits and the maturity nobody quotes.

The speaker notes hold about 1,700 words for 14 minutes 20 seconds, about 120 words a minute.

## In the Overleaf project

The theme (`bayesops.sty`), its fonts (`fonts/`) and its logos (`assets/`) live in the tutorial's Overleaf project and
are not copied here. Upload `slot1.tex` and the folder `figures/slot1/` to that project, keeping the paths, then choose
Menu › Main document › `slot1.tex` and the LuaLaTeX compiler, which the theme needs. The project's `main.tex` is left
as it is.

The deck adds a few helpers to the theme, at the top of `slot1.tex`: `\key` and `\pairs` for the reading order above,
with `\pairs[illustration]` and `\pairs[result]` for the evidence tag; `\gap`, `\footgap`, `\fig` and `\figmax` for
the vertical layout; `\holds`, `\caveat`, `\flagged` and `\pending` for the status marks of slide 13; `\detail` for
the backup frames; `\highlight`, `\source` for a source line, `\person` for a presenter, `\rowrule` for the light rule
between the rows of a list, the ragged table column `L`, and the diagram styles, which apply only inside a picture
opened with `[deck]`. If the theme already defines one of these names, LaTeX stops at that line: rename the helper.
The deck also uses the theme's `\TitleFont` for the key lines and for the lists. It loads `unicode-math` with Latin
Modern Math and the `array`, `booktabs` and `tikz` packages. Dashes are typed as Unicode (–), because the theme's
title fonts do not turn `--` into a dash.

Every slide is set from the top: the title, then the key line. The strip and the source line sit at the bottom, and
the figure or diagram is centred in between (`\gap` is stretchable space). Every figure is 1.60 inches tall and is
shown unscaled.

**The deck has not been compiled with the real theme.** On 11 November it was compiled with LuaLaTeX (TeX Live 2023)
against a stand-in that has the theme's colours, Rubik titles on Inter text, a frame title with an accent rule and a
footline; no frame overflows there. After the upload, check that no frame overflows in the real theme. If a figure
slide does, lower `\figmax` at the top of `slot1.tex`: every figure shrinks to it.

Speaker notes are in `\note{}`; Beamer prints them only with `\setbeameroption{show notes on second screen}`.

## Regenerating the figures

From `tutorial/`:

```bash
pixi run --locked python slides/slot1/export_figures.py --rubik PATH/TO/fonts/Rubik-Variable.ttf
```

`--rubik` takes the variable font from the Overleaf project's `fonts/` folder, or Google Fonts' `Rubik[wght].ttf`, so
that the figures use the slides' typeface; without it they use matplotlib's default sans-serif. The committed figures
were drawn with `--rubik`. The script needs only the tutorial's own dependencies, and `fontTools` for `--rubik`.

The running example takes about ten minutes to compute on four cores, mostly five posteriors. Its numbers are cached
under `tutorial/.cache/slides-slot1/`, which Git ignores, and reused until `illustration.py`, `spx-design.json`,
`market-like-case.json` or the computation in `export_figures.py` changes; `--recompute` ignores the cache. The
script ends by printing every number the slides quote, and lists those that `slot1.tex` does not contain. After a
change to the example, update the slides it names (2, 3, 6, 7, 8, 9 and the backup frames) and the curriculum page.

## The running example

Seven figures come from one example, so that the argument stays on one case. The example keeps the model, the prior,
the error law and the "market-like" parameters of the tutorial's Heston notebook
(`lab/notebooks/heston_spx_calibration.py`, on branch `claude/trusting-dijkstra-hmgvcm` as of 11 November 2026) and
changes one thing, the chain. The notebook's 63 SPX quotes are so many and so tight that they pin every Heston
parameter: its 90% intervals are at most about 1% of the prior's range, and a figure of them shows nothing. The example's
chain is thinner and quoted less tightly, as for a single stock, so that what the quotes leave open can be seen.

- **Model and truth.** Heston with the notebook's parameters (v₀, κ, v∞, σ_v, ρ) at its case "market-like
  parameters": v₀ 0.0271, κ 4.34, v∞ 0.0565, σ_v 1.80, ρ −0.667.
- **Chain.** 35 out-of-the-money options: 7 strikes at standardized moneyness δ = ln(K/F)/(0.16√T) of −3.5, −2.25,
  −1.25, −0.5, 0, 0.5 and 1, on each of 1, 3 and 6 months and 1 and 2 years.
- **Half-spreads.** 0.3 volatility points at the money, rising by 0.35 points per unit of |δ| to 1.5 points at
  δ = −3.5, converted to prices with each quote's own vega. That is about ten times the half-spreads of the SPX
  quotes (0.02 to 0.5 points, median 0.06).
- **Error law.** The notebook's: each quote is the model price plus c × its half-spread × a standard normal, with
  c = 1 in the simulation and each quote observed once (seed 20261114).
- **Prior.** The notebook's uniform box (v₀ and v∞ from 0.01 to 0.16, κ from 0.1 to 10, σ_v from 0.1 to 3, ρ from
  −0.95 to −0.10), with c inferred under U(0.5, 2).
- **Readings.** The same surface is read with quotes up to 2 years, 1 year, 6 months, 3 months and 1 month (35, 28,
  21, 14 and 7 quotes). Slides 2, 3, 6, 8 and 9 use the quotes up to 3 months; slide 7 uses all five readings.
- **Products.** A 2-year put struck at 90% of the forward, and a forward-start call struck at the money in 1 year and
  paid at 2 years (slide 9 holds the call against the put).

For each reading the example computes two things.

- **The classical route.** Least squares in half-spreads from several starts; the error scale by its unbiased
  estimate, ĉ² = R/(n − p), with R the sum of squared misses in half-spreads, n quotes and p fitted parameters; the
  Gauss–Newton covariance; Student-t intervals, for a parameter or, by the delta method, for a product. Where the fit
  sits on a bound of the box, no interval is reported (the 1-month reading: κ and v∞ are on their bounds).
- **The posterior.** The error scale c is integrated out in closed form: with its uniform prior the likelihood's
  integral over c is an incomplete gamma function of R. The five Heston parameters are then sampled by importance
  sampling in logit coordinates, from a mixture of the prior with multivariate t components that is adapted over
  four rounds to a flattened posterior, up to 6,000 effective draws or 200,000 proposals; 4,000 draws are resampled
  and moved by 50 sweeps of differential-evolution Metropolis at the posterior; each draw then receives a c from its
  exact conditional. The script reports the effective sample size before the moves (2,041 to 7,152) and the share
  of failed pricings (none).

What was checked:

- **Prices** are Lewis Fourier integrals with an analytic Black control variate, on 1,024 Gauss–Legendre nodes up to
  600, for many parameter sets at once. They match the tutorial runtime's prices (the tests compare the two), and the
  next rung (2,048 nodes, 1,200) moves no price by more than 3 × 10⁻⁵ of its half-spread at 200 posterior draws of
  every reading, against the notebook's gate of 0.01. The forward-start call is one more Fourier integral (Kruse and
  Nögel 2005); the tests check it against the vanilla call as the start date goes to zero and against Black–Scholes
  as the volatility of volatility goes to zero.
- **The posteriors** were checked against an independent sampler on the same target, eight adaptive Metropolis
  chains of 30,000 draws after 30,000 (`check_posterior.py`), on the two readings where the posterior is furthest
  from a normal. Quotes up to 3 months: the 5%, 50% and 95% points of the 2-year put agree within 1.3% of the width
  of its 90% range, and those of every parameter within 2.4%; split R-hat at most 1.04. Quotes up to 1 month: within
  3.2% for the put and 3.5% for the parameters; split R-hat at most 1.01.
- **The classical route** was checked on the notebook's own data. On its 63 simulated quotes
  (`market-like-case.json` on the design of `spx-design.json`), the least-squares 90% intervals of this code match
  the notebook's exact posterior (four MCMC chains of 12,000 draws) for all five Heston parameters: widths within 2%
  and centres within 0.03 posterior standard deviations. Slide 7 quotes this, a backup frame shows the table, and a
  test repeats it.

**Why these samplers, and their limits.** The tutorial's environment does not include PyMC, and the slides' code may
use only the tutorial's dependencies, so the example's posterior and its check are computed by this folder's own
samplers: importance sampling with Metropolis moves in `illustration.py`, and adaptive Metropolis chains in
`check_posterior.py`. They serve a teaching illustration and are not certified reference posteriors: the diagnostics
are a split R-hat and a batch-means effective sample size, the chains of the check accept 4 to 9% of their proposals,
and the agreement between the two samplers is stated to a few percent of a range's width. A test compares the
sampler with the delta-method range on the reading where every quote is read. The exact posteriors that the deck
quotes as results are the Heston notebook's PyMC references.

**It is an illustration, not a result of the tutorial.** The model is right by construction, the quote errors are
independent, and one simulated surface is shown. It supports statements about a mechanism, not about a market. On
the real SPX quotes the open question is the model, not the parameters: Heston misses the quotes by 13 half-spreads.
The backup frame *The Example and the Notebook* sets the two side by side.

## The recorded results

Slides 10 to 13 and two backup frames quote the Heston SPX study as recorded on 11 November 2026. The numbers are
copied into [`recorded-results.json`](recorded-results.json) with the branch, path, commit and SHA-256 of each source
file, because the slides' code reads nothing outside `tutorial/`. The costs of rough Bergomi on slide 10 are not
measured: they are the Heston run's 146,000 pricings (or 1,000 pricings for importance sampling) at the 60 to 105
CPU-seconds that one rough-Bergomi pricing of the quotes takes, and the slide labels them "estimated".

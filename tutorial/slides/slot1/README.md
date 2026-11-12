# Slot 1 slides

The 15-minute opening talk, *Introduction: Bayes for volatility models*, as Beamer source in the tutorial's BayesOps
theme. The workshop opens with Beamer talks (introduction, amortized Bayes, architectures) and then works through the
application notebooks; this deck is the first talk. It has 19 content frames after the title page, one of them in
three steps, and 13 backup frames after `\appendix`. The plan, the minute map, and the source and status of every number are
in [`curriculum/01-slot1-the-problem.md`](../../curriculum/01-slot1-the-problem.md).

| File | What |
|---|---|
| `slot1.tex` | The deck: a main document of its own, using `\usepackage[cover]{bayesops}` like the project's `main.tex` |
| `figures/slot1/*.pdf` | The 15 figures the deck includes, in the theme's colours, as vector PDFs drawn at the size the slides show them |
| `figures/slot1/*.jpg` | The four presenters' photos, cropped square as on the tutorial's webpage |
| `export_figures.py` | Regenerates the figures of the option example and of the recorded results, and prints every number their slides quote |
| `intro_figures.py` | Draws the four figures of the opening argument (slides 1 to 4) and prints every number those slides quote |
| `intro_example.py` | The opening example: a portfolio of ten US industries and a share in stocks, computed from estimated inputs, with and without a prior |
| `intro-inputs.json` | The inputs of the opening example: averages and covariances of monthly US excess returns, October 2006 to September 2026, computed from Kenneth French's data library, with the SHA-256 of its two files |
| `illustration.py` | The running example behind most of the figures: a thin option chain simulated from Heston, and a forward-start call |
| `check_posterior.py` | An independent check of the example's posteriors by Metropolis chains; nothing on the slides depends on it |
| `recorded-results.json` | The recorded results of the Heston SPX study that slide 15 and two backup frames quote, each with its source file, commit and SHA-256 |
| `spx-design.json` | The tutorial's 63-quote SPX design of 9 November 2026 (strikes, maturities, forwards, discounts, half-spreads), with its source |
| `market-like-case.json` | The Heston notebook's simulated case "market-like parameters": its 63 observed prices, generating parameters and noise draws, with their source |

## The argument

The talk introduces Bayes for volatility calibration and sets up the rest of the tutorial. It has five parts.

1. **Why uncertainty, and what Bayes adds** (slides 1–5), on one example from 20 years of US returns. An
   optimizer amplifies the errors of its inputs: raising one average return by a third of its standard error moves a
   portfolio weight by 21 points. The uncertainty of an estimate changes a decision: with it, a buy-and-hold
   investor holds fewer stocks the longer the horizon. Bayes gives the parameters a probability distribution, from a
   prior and the data. The prior states a belief and how strongly it is held, and the decision follows it. Bayes
   handles bounds, several fits, hidden variables and non-normal returns, at the cost of a prior, a full model and
   computation; with a lot of data it agrees with classical methods. Slide 5 ends on why the tutorial uses option
   calibration: every building block can be shown and checked on it.
2. **The example and the exercise** (slides 6–8). A model calibrated to quoted options prices what the market does
   not quote. Volatility models are the example; the five steps (model, simulate, train, infer, check) and the
   questions they answer are general.
3. **Bayes for calibration** (slides 9–14). Bayes' rule; what a prior and a likelihood are; the posterior, computed
   as draws by a sampler, and what it shows; from the posterior to the price of an exotic; why Bayes; and what
   happens when the quotes run thin. Least squares with a quadratic approximation also gives ranges, and the deck
   says so: it is the limit of Bayes when the quotes are informative.
4. **The cost, and what comes next** (slides 15–16). An exact posterior costs minutes to months. Amortized inference
   pays that cost once: one handover slide, to the next talk.
5. **The close** (slides 17–19). A summary in three pictures, the team, and the rest of the tutorial.

The validation of the network (the coverage test and the checks) is not part of this talk: it belongs to the worked
example. Those slides are in the backup.

## Three kinds of evidence

The rule above a slide's strip of facts carries a tag that says which kind of evidence the slide shows.

| Tag | Slides | What it is | Where it comes from |
|---|---|---|---|
| ILLUSTRATION · US RETURNS 2006–2026 | 1, 2, 3, 4 | The opening example: two decisions computed from 20 years of monthly US returns, a portfolio of ten industries and a share in stocks. It shows why estimates need their uncertainty, and what a prior does. | `intro_example.py`, from `intro-inputs.json` |
| ILLUSTRATION · SIMULATED QUOTES | 10, 11, 12, 14, and two backup frames | One running example: a thin option chain simulated from Heston, the model right by construction, and one exotic, a forward-start call. It shows a mechanism. | `illustration.py` |
| RESULT · HESTON ON SPX · AS OF 11 NOV | 15, and two backup frames | What the tutorial's Heston study on 63 SPX quotes had recorded on 11 November 2026 | `recorded-results.json` |

Slide 15 gives costs as orders of magnitude from the tutorial's own runs, and marks the rough Bergomi costs as
estimated. The backup frames *The Coverage Test* and *What the Checks Find* give the network's status on
11 November:

| Check | Status | Recorded |
|---|---|---|
| Speed | holds | 0.27 s per posterior (615 posteriors in 164 s), against 14.0 to 26.6 CPU-minutes for an exact posterior |
| Coverage on 512 simulated surfaces | holds for 5 of 6 parameters | 90% intervals contain the truth 89.3 to 92.4% of the time for κ, v∞, σ_v, ρ and c; 97.5% for v₀, whose intervals are too wide |
| Against exact posteriors on two simulated surfaces | caveat | means within 0.96 exact standard deviations; 90% intervals 1.05 to 2.9 times as wide as the exact ones |
| Rough Bergomi | pending | the simulation bank was being generated; no result is shown |

These are development results (`scientific_acceptance` is false in their run records), not studies of the claims
ledger, and the deck cites no claim of the ledger. The deck does not quote Heston's misfit on the SPX quotes.

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
The backup frames after `\appendix` hold the rest. One slide, *What a Posterior Looks Like*, has three steps: a
sampler's first steps, the posterior draws, then least squares with its quadratic region.

The speaker notes hold about 2,120 words, about 18 minutes at 115 words a minute: the opening of 12 November
(slides 1 to 5, about 670 words) is longer than the two slides it replaced, and the rest of the deck has not yet
been trimmed to bring the talk back to 15 minutes.

## In the Overleaf project

The theme (`bayesops.sty`), its fonts (`fonts/`) and its logos (`assets/`) live in the tutorial's Overleaf project and
are not copied here. Upload `slot1.tex` and the folder `figures/slot1/` to that project, keeping the paths, then choose
Menu › Main document › `slot1.tex` and the LuaLaTeX compiler, which the theme needs. The project's `main.tex` is left
as it is.

The deck adds a few helpers to the theme, at the top of `slot1.tex`: `\key` and `\pairs` for the reading order above,
with `\pairs[returns]`, `\pairs[illustration]` and `\pairs[result]` for the evidence tag; `\gap`, `\footgap`, `\fig` and `\figmax` for
the vertical layout; `\holds`, `\caveat`, `\flagged` and `\pending` for the status marks of the checks; `\detail` for
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
footline; no frame overflows there. On 12 November the deck with its new opening was compiled against a plainer
stand-in (the theme's colours, Rubik titles on Inter text, a plain frame title and a footline); no frame overflows
there either. After the upload, check that no frame overflows in the real theme. If a figure
slide does, lower `\figmax` at the top of `slot1.tex`: every figure shrinks to it.

Speaker notes are in `\note{}`; Beamer prints them only with `\setbeameroption{show notes on second screen}`.

## Regenerating the figures

From `tutorial/`:

```bash
pixi run --locked python slides/slot1/export_figures.py --rubik PATH/TO/fonts/Rubik-Variable.ttf
```

The opening argument's four figures have their own script, which runs in a few seconds and reads only
`intro-inputs.json`:

```bash
pixi run --locked python slides/slot1/intro_figures.py --rubik PATH/TO/fonts/Rubik-Variable.ttf
```

`--rubik` takes the variable font from the Overleaf project's `fonts/` folder, or Google Fonts' `Rubik[wght].ttf`, so
that the figures use the slides' typeface; without it they use matplotlib's default sans-serif. The committed figures
were drawn with `--rubik`. The script needs only the tutorial's own dependencies, and `fontTools` for `--rubik`.

The running example takes about ten minutes to compute on four cores, mostly five posteriors. Its numbers are cached
under `tutorial/.cache/slides-slot1/`, which Git ignores, and reused until `illustration.py`, `spx-design.json`,
`market-like-case.json` or the computation in `export_figures.py` changes; `--recompute` ignores the cache. The
script ends by printing every number the slides quote, and lists those that `slot1.tex` does not contain. After a
change to the example, update the slides it names (10, 11, 12, 14 and the backup frames) and the curriculum page.

## The opening example

Slides 1 to 4 use one small example on real returns, so that the case for uncertainty and for Bayes is made on
numbers and not on a list. It is computed by [`intro_example.py`](intro_example.py) and drawn by
[`intro_figures.py`](intro_figures.py), which prints every number below.

- **Data.** Monthly returns in excess of the one-month bill rate, October 2006 to September 2026 (240 months), from
  Kenneth French's data library (CRSP): the ten value-weighted industry portfolios, and the market. The script reads
  their averages and covariances from [`intro-inputs.json`](intro-inputs.json), which records the library's two
  files with their SHA-256; `intro_example.build_inputs` recomputes it from those files.
- **The portfolio** (slide 1). The fully invested portfolio of the ten industries with the highest Sharpe ratio,
  short positions allowed, from the sample averages and covariances taken as exact: the plug-in route. Technology
  has the highest average, 15.7% a year with a standard error of 4.3. Raising it by 10%, to 17.3%, is a change of
  0.37 standard errors. Technology's weight goes from 74% to 95% and retail's from 31% to 18%. Raising each industry
  in turn moves one weight by 3 to 26 points.
- **The horizon** (slide 2). The share in stocks of a buy-and-hold investor with power utility and risk aversion 5,
  as in Barberis (2000), Section II: excess log returns are i.i.d. normal, and the share is chosen once, between 0
  and 99%. The market's mean is 9.1% a year with a standard error of 3.5, and its volatility 15.9%. With these taken
  as exact the share is 83 to 84% at every horizon from 1 to 20 years. With a flat prior on the mean and the log
  variance, the cumulative return is Student-t with a variance that grows faster than the horizon, and the share
  falls from 79% at 1 year to 58% at 10 and 45% at 20.
- **The prior** (slides 3 and 4). The belief that the ten industries have the same expected return: mu ~ N(eta 1,
  Sigma / tau) with eta diffuse (Jorion 1986). It is worth tau months of data, so its weight against 240 months of
  data is tau / (240 + tau). Weight 0 is the plug-in route; weight 1 takes the belief as certain and gives the
  minimum-variance portfolio, in which technology has 12% whatever the averages. The spread of the ten averages
  suggests a weight of 56% (Jorion's estimate), a prior worth 26 years of data: technology then has 39%, and 48%
  after the change. Slide 3 shows that prior for one parameter, technology's expected return, with the common value
  taken as given: prior 8.9% ± 3.8 a year, data 15.7% ± 4.3, posterior 11.9% ± 2.8.

**What the example supports, and what it does not.**

- It is an illustration on one window of real returns, not a result of the tutorial, and it makes no claim about
  which portfolio performs better out of sample.
- The smaller move under the prior comes from the prior. With a flat prior the Bayesian weights are the plug-in
  weights, because the predictive covariance is proportional to the sample covariance.
- The slides keep the prior's weight fixed when the average is raised, as a belief stated before the data. If the
  weight is estimated again from the changed averages it falls to 49%, and technology moves by 15 points, not 9.
- With no short positions the plug-in portfolio holds three industries and the change moves technology from 54% to
  66%; under the prior it holds five, and the largest move is 7 points.
- The horizon effect depends on the length of the sample: Barberis reports about 10 points at a 10-year horizon
  with 43 years of data and 35 points with 10 years of data. Here, with 20 years, it is 25 points.

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
- **Product.** An exotic, the option application's forward-start call: its strike is set at the money in 6 months
  and it pays at 1 year. Slide 12 adds a second forward start, struck in 9 months.
- **Readings.** The same surface is read with fewer and fewer quoted expiries, always including 1 year, when the
  product pays: all five expiries (35 quotes), 1 month to 1 year (28), 3 months to 1 year (21), 6 months and 1 year
  (14), and 1 year only (7). Slides 10, 11 and 12 use the two expiries that are the product's own dates; slide 14
  uses all five readings; the backup frame *What an Assumption Does* uses the one expiry.

**Why the product's dates are quoted.** The talk is about exotics: products whose price the quotes do not fix even
where the quotes are. With quotes at 6 months and at 1 year, the options fix the distribution at each date, and the
forward start depends on how the two dates are linked, which the quotes pin only through the model's parameters. A
product beyond the last quoted expiry would show a larger range, but that is extrapolating the vanilla surface in
maturity, which is not what an exotic is and not what the talk argues. The honest size of the effect follows from
this choice: with the two dates quoted the forward start's 90% range is 4.39 to 5.00, 13% of its price, against 5%
for the quoted 1-year at-the-money call under the same draws; with four expiries it is 4.56 to 4.81.

For each reading the example computes two things.

- **The classical route.** Least squares in half-spreads from several starts; the error scale by its unbiased
  estimate, ĉ² = R/(n − p), with R the sum of squared misses in half-spreads, n quotes and p fitted parameters; the
  Gauss–Newton covariance; Student-t intervals, for a parameter or, by the delta method, for a product. Where the fit
  sits on a bound of the box, no interval is reported (the reading with one expiry: today's variance is on its
  lower bound).
- **The posterior.** The error scale c is integrated out in closed form: with its uniform prior the likelihood's
  integral over c is an incomplete gamma function of R. The five Heston parameters are then sampled by importance
  sampling in logit coordinates, from a mixture of the prior with multivariate t components that is adapted over
  four rounds to a flattened posterior, up to 6,000 effective draws or 200,000 proposals; 4,000 draws are resampled
  and moved by 50 sweeps of differential-evolution Metropolis at the posterior; each draw then receives a c from its
  exact conditional. The script reports the effective sample size before the moves (2,903 to 7,152) and the share
  of failed pricings (none).

What was checked:

- **Prices** are Lewis Fourier integrals with an analytic Black control variate, on 1,024 Gauss–Legendre nodes up to
  600, for many parameter sets at once. They match the tutorial runtime's prices (the tests compare the two), and the
  next rung (2,048 nodes, 1,200) moves no price by more than 5 × 10⁻⁶ of its half-spread at 200 posterior draws of
  every reading, against the notebook's gate of 0.01. The forward-start call is one more Fourier integral (Kruse and
  Nögel 2005); the tests check it against the vanilla call as the start date goes to zero and against Black–Scholes
  as the volatility of volatility goes to zero.
- **The posteriors** were checked against an independent sampler on the same target, eight adaptive Metropolis
  chains of 30,000 draws after 30,000 (`check_posterior.py`), on the two readings where the posterior is furthest
  from a normal. Quotes at 6 months and 1 year: the 5%, 50% and 95% points of the forward start agree within 2.2% of
  the width of its 90% range, and those of four parameters within 3.7%; the 95% point of the long-run variance
  differs by 20% of its range (0.0655 by the chains, 0.0698 here: the sampler's upper tail, at slow mean reversion,
  is a little longer), which the forward start does not feel; split R-hat at most 1.01. Quotes at 1 year only:
  within 7.3% for the forward start (3.02 to 4.84 by the chains, 2.89 to 4.90 here) and 8.3% for the parameters;
  split R-hat at most 1.02. This is the least precise reading. The slides use it only to show a wide range set by
  the prior, and quote that range to one decimal.
- **The classical route** was checked on the notebook's own data. On its 63 simulated quotes
  (`market-like-case.json` on the design of `spx-design.json`), the least-squares 90% intervals of this code match
  the notebook's exact posterior (four MCMC chains of 12,000 draws) for all five Heston parameters: widths within 2%
  and centres within 0.03 posterior standard deviations. Slides 11 and 14 quote this, a backup frame shows the table, and a
  test repeats it.

**Why these samplers, and their limits.** The tutorial's environment does not include PyMC, and the slides' code may
use only the tutorial's dependencies, so the example's posterior and its check are computed by this folder's own
samplers: importance sampling with Metropolis moves in `illustration.py`, and adaptive Metropolis chains in
`check_posterior.py`. They serve a teaching illustration and are not certified reference posteriors: the diagnostics
are a split R-hat and a batch-means effective sample size, the chains of the check accept 5 to 8% of their proposals,
and the agreement between the two samplers is stated to a few percent of a range's width. A test compares the
sampler with the delta-method range on the reading where every quote is read. The exact posteriors that the deck
quotes as results are the Heston notebook's PyMC references.

**It is an illustration, not a result of the tutorial.** The model is right by construction, the quote errors are
independent, and one simulated surface is shown. It supports statements about a mechanism, not about a market. On
the tutorial's 63 SPX quotes the same calculation gives ranges too narrow to see, which is why the example thins the
chain. The backup frame *The Example and the Notebook* sets the two side by side.

The sampler's path on the first step of slide 11 is a short random-walk Metropolis chain on the same posterior, run
by `export_figures.py` from a fixed starting guess; it illustrates how draws are obtained and feeds no number.

## The recorded results

Slide 15 and two backup frames quote the Heston SPX study as recorded on 11 November 2026. The numbers are
copied into [`recorded-results.json`](recorded-results.json) with the branch, path, commit and SHA-256 of each source
file, because the slides' code reads nothing outside `tutorial/`. The costs of rough Bergomi on slide 15 are not
measured: they are the Heston run's 146,000 pricings (or 1,000 pricings for importance sampling) at the 60 to 105
CPU-seconds that one rough-Bergomi pricing of the quotes takes, and the slide labels them "estimated".

# Slot 1. The problem: Bayes for volatility models

Francesco · 15 minutes · first slot of the afternoon. Part of the [curriculum](00-overview.md).

![Minute map of slot 1](images/slot-1.png)

Rebuilt on 30 October 2026, after the organizers' call of 29 October. Slot 1 introduces the objects the
afternoon uses, with a short window on why they matter. It follows the arc of a standard introduction to Bayes,
translated to option quotes, and leaves hierarchies out. It opens on a diagram of a desk's daily task and the
question it raises, then presents the data: the grid read from SPY's quotes and the noise of each price. It
builds Bayesian inference step by step on Black–Scholes: the model as a map from $\sigma$ to a price, least
squares and the likelihood, prior and posterior, more quotes, the range of a price, and a check of the model.
Heston repeats the steps with four parameters and shows when the posterior differs from least squares, and the
cost of exact posteriors leads to amortization. After the takeaways, as reference, come the details of the
data that the later slots use, the formulas and the study's assumptions. The notebook is
[`slot1_intro.py`](../notebooks/slot1_intro.py); its structure, style and formatting follow the slot 5 SVJ
notebook of #38 (branch `feat/tutorial-slot5-svj-advanced`, commit `d00ccb1`).

## The slot in three sentences

**The question.** What does a calibration know, and how sure is it?

**The answer.**

1. A calibration is an estimate from noisy quotes; its posterior becomes a range for any price computed from it.
2. The posterior is often the classical answer with error bars, but not always: where the quotes say nothing, the prior decides, so it must be stated.
3. Exact posteriors get expensive as models get realistic, and many are needed: every name in a valuation run, every smile of a validation. Amortization, from slot 2 on, makes them affordable.

## What the audience leaves with

- The objects of the afternoon: the quotes $y$ (63 prices, each with its noise), the model, the parameters $\theta$, the prior, the likelihood, the posterior, a product's range, the checks, and the network $q(\theta \mid y)$.
- Least squares keeps the peak of the likelihood; the posterior keeps the whole curve. More quotes narrow it, around the truth when the model can produce the data.
- The data as every later slot uses it: 63 prices in units of the forward, on 7 maturities × 9 standardized moneyness points, each with a noise fixed by rule; networks read the prices in noise units; SPY's seven forward-variance levels; simulated smiles drawn from one task file per model.
- What the study assumes: one date and one underlying, a grid read from cleaned quotes, European pricing of American options, independent Gaussian quote noise, three models with constant parameters, uniform priors.
- The posterior is conditional on its model. A model that cannot produce the data gives a precise and wrong range.
- On a liquid smile with a well-identified model, the classical interval is the posterior's Gaussian limit. Where the quotes leave a direction open, the posterior is not an ellipse.
- The three models form a ladder: Black–Scholes (closed form), Heston (exact likelihood, slot 4) and rough Bergomi (Monte Carlo only, slot 5). Each step fits the smile better and makes the exact posterior more expensive.

## The arc: a standard introduction to Bayes, on option quotes

| A generic introduction (*Statistical Rethinking*, ch. 2–3; *Bayesian Data Analysis*, ch. 1) | Slot 1 |
| --- | --- |
| The question, and the data story: how unknowns produce data | Sections 1 and 2: the desk's task and its question; the quotes, the grid and the noise of each price |
| Prior × likelihood → posterior, on a grid | Section 3: the Black–Scholes $\sigma$ from one quote, least squares as the peak of the likelihood, and more quotes narrowing the posterior |
| Posterior samples give derived quantities | Section 4: another option priced under every draw |
| A posterior predictive check | Section 5: the predicted quote against the market's, and the flat prediction against the smile |
| The quadratic approximation, and when it fails | Section 6: Heston's posterior is an ellipse; rough Bergomi's need not be |
| Computing the posterior: grid, quadratic, MCMC | Section 7: and a fourth rung, amortized inference |

Each section answers one question and hands the next one over.

## Section by section

| Min | Section | The question | On screen | Mode |
| --- | --- | --- | --- | --- |
| 0–1 | 1 · The problem | What does a desk do with option quotes, and what does one calibration not tell it? | A diagram: quotes, calibration, the calibrated model, prices of products that are not quoted, the book; the question: how sure are the parameters, and the prices computed from them? | slide |
| 1–2.5 | 2 · The data | What is quoted, and how precisely? | The 63 contracts as strikes against maturity, with SPY's forward, and the smile at those points; each quote ± its noise at one month and one year: tight at and below the money, open above it | cached |
| 2.5–5.5 | 3 · Bayes with one parameter | What does one quote say about $\sigma$? | The model as a map from $\sigma$ to a price, and least squares' $\chi^2$ (12.84%, with $\chi^2 \le 1$ from 12.55% to 13.13%); prior × likelihood → posterior on a grid of 4,001 volatilities, with a choice of priors; seven simulated quotes narrowing the posterior like $1/\sqrt n$ | live |
| 5.5–6.5 | 4 · Using the posterior | How sure is a price the model computes? | The one-month put at $\delta = -2$ priced under 4,000 draws: a 90% range of 0.05 to 0.14 bp; the rules that ask for ranges | live |
| 6.5–8.5 | 5 · Checking the model | Does the model reproduce data it has not seen? | The market's 14.1 bp against the predicted quotes, 9.5 standard deviations away, and the flat prediction against the smile; with both quotes, a noise slider (about 4.8 times the noise reconciles them) | live |
| 8.5–11.5 | 6 · A model with a smile: Heston | What changes when the model can produce the smile? | Heston fitted to SPY's smile and the effect of the correlation (live Fourier pricing); its exact posterior on SPY and a 1-year variance swap priced under it; the least-squares/Bayes table; Heston's posterior is an ellipse (C11), rough Bergomi's for an almost flat curve is not | cached |
| 11.5–14 | 7 · What it costs, and amortized inference | Why not a posterior for every smile? | The methods' ladder (grid, Gaussian, MCMC, amortized) and the measured costs (C8); the cost of one valuation run by route, with a slider; the amortized pipeline | cached |
| 14–15 | Close | What does the afternoon build on? | The objects of the afternoon and four takeaways; next, slot 2 | slide |

The notebook reads every number live; the values above are today's.

After the takeaways the notebook holds reference material that is not presented: the data in detail (how the
grid is read from the chain, what a network reads, the variance curve and the skew, simulated smiles and the
task files), the formulas, and the study's assumptions. The presenter names it in the close; the later slots
rely on it.

## The live notebook

| Cells | Compute | Budget |
| --- | --- | --- |
| Section 2 | SPY prices, noise and the cleaned chain's expirations from `spy/surface.json`; implied volatilities, and those of each price ± its noise; the identity of the grid and noise with `backbones/heston-final500/task.json` | < 1 s |
| Section 3 | $\chi^2$ and the grid posterior of $\sigma$ from the at-the-money one-month quote, flat prior on 5% to 60%; three normal priors centred on 20% for comparison; seven at-the-money quotes simulated at 13% (fixed seed) and their posteriors | < 0.1 s per choice |
| Section 4 | 4,000 draws of $\sigma$, the put under each, its quantiles | < 0.1 s |
| Section 5 | The posterior predictive of the put's quote; both likelihoods, the joint posterior, standardized misses; a noise multiple from 1 to 10 | < 0.1 s per move |
| Section 6 | Three Heston smiles by Fourier pricing; the SPY Heston posterior draws of `data/heston-spy-exact.npy`; `heston-final500` (PyMC reference draws) and `rough-joint100k-validation1962` (network draws), loaded offline with NumPy; Gaussian ellipses; C11 from `heston/s6/classical.json` | about 3 s |
| Section 7 | C8 from `costs/costs.json`; the cost of one valuation run by route | < 0.1 s per move |
| The data in detail | The rates and grid readings; prices in noise units; the forward-variance levels, variance-swap volatilities, at-the-money skew and its power law | < 0.1 s |

No network is loaded. The notebook runs headless in about 9 seconds.

## Numbers on screen, and their status

| Number | Source | Status |
| --- | --- | --- |
| 63 prices on a grid of 7 maturities × 9 moneyness points, read from SPY's quotes of 21 Oct 2026 | `spy/surface.json` (S8) | verified |
| The grid and noise are those of every backbone task | computed live | verified by construction |
| 32 expirations, 5,358 cleaned quotes; every grid point inside the quoted strikes | `spy/surface.json` diagnostics (S8) | verified |
| Each quote's noise in implied volatility at one month: about ±0.3 points at the money and ±0.5 at $\delta = -2$; at $\delta$ = 1.5 and 2 the lower end reaches zero | computed live from `spy/surface.json` | verified by construction (live) |
| Least squares on the one-month at-the-money quote: $\sigma$ = 12.84%, $\chi^2 \le 1$ from 12.55% to 13.13% | computed live | verified by construction (live) |
| Seven at-the-money quotes simulated at $\sigma$ = 13%: posterior standard deviation 0.30 points from one quote, 0.11 from seven | computed live, fixed seed | verified by construction (live) |
| Treasury yields 4.26% (3 months) and 4.79% (2 years); 35 grid points read from two expirations, 28 from one, none extrapolated | `spy/surface.json` (S8) | verified |
| Prices in noise units from 0.2 (1 week, $\delta = 2$) to 56 (1 year, at the money); the calls at $\delta$ = 1.5 and 2 below one noise unit at every maturity, where SPY's implied volatility is 12% to 15% | computed live from `spy/surface.json` | verified by construction (live) |
| Variance swaps replicated on 17 of 32 expirations; seven forward-variance levels; variance-swap volatilities from 13.6 (1 week) to 21.8 points (1 year) | `spy/surface.json` (S8) | verified |
| The at-the-money skew falls like $T^{-0.32}$ from one week to one year | C1, `spy/surface.json` (S8) | verified |
| Banks of 100,000 training and 5,000 validation smiles per model, with 5,000 (Heston) or 7,000 (rough Bergomi) held back; slot 4's SPY network from an earlier Heston bank of 50,000. Heston prices refined to 0.1 noise sd; rough Bergomi's Monte Carlo error below 0.25 noise sd (0.5 at the last rung), with 32,768 to 2,097,152 paths | `backbones/*/training.json` and `task.json`, `checkpoints/heston/training.json`, course [status](../../course/status.md) | verified |
| Heston on SPY: posterior mean $v_0$ = 0.018, $\theta$ = 0.066, $\sigma_v$ = 1.14, $\rho$ = −0.63 ($\kappa$ = 2); 1-year variance swap 20.92 to 21.55 volatility points (90%) | `data/heston-spy-exact.json` and `.npy`, the feasibility run of 23 Oct that slot 4 uses | not an accepted study; S8-H replaces it |
| $\sigma$ = 12.84% ± 0.30; put range 0.05–0.14 bp; quote 14.1 bp; miss 9.5 sd; about 4.8 times the noise | computed live from `spy/surface.json` | verified by construction (live) |
| Heston: about 90% of the exact draws inside the Gaussian's 90% ellipse | `backbones/heston-final500`, PyMC DEMetropolisZ reference | development example, not an accepted study |
| Gauss–Newton 90% coverage 87–92% against the posterior's 87–90%, on 200 Heston smiles; no Gauss–Newton interval on 11 | C11, `heston/s6/classical.json` | verified |
| Rough Bergomi with an almost flat curve: the decay's 90% interval spans most of its prior; the matching Gaussian puts mass outside the prior box | `backbones/rough-joint100k-validation1962`, network draws | development replay; no conditional reference yet (J2) |
| Heston 20 ms and about 4 CPU-min, both at one quadrature rung; rough Bergomi 5.0 CPU-s and at least 17 CPU-h by the chain; network 1.4 s (3.6 CPU-s) | `costs/costs.json` (C8) | verified |
| Rough Bergomi by importance sampling from a Gauss–Newton proposal on SPY's 63 quotes: 1,024 evaluations at 5.0 CPU-s, about 85 CPU-min | `lab/checks/rough_spy_is.py`; [v2-plan](../../course/v2-plan.md), Section 2.4 | verified |

## What may be said, and what may not

| Say | Do not say | Why |
| --- | --- | --- |
| "The at-the-money quote is known to 0.3 volatility points, and the quotes at $\delta = \pm 2$ to 1 point at a 20% volatility: the noise the tutorial assumes." | "The quote's bid–ask is 0.3 points." | The noise is the design's floor, wider than SPY's spreads on that day (study S8; course, Lecture 5, Section 5.4). |
| "The 63 prices are read from SPY's quotes by smoothing and interpolation." | "The 63 prices are quoted." | Each is read from a spline through an expiration's implied volatilities; 35 of them interpolate between two expirations (study S8). |
| "SPY's skew falls roughly like $T^{-0.32}$." | "So $H$ is about 0.18." | Under rough Bergomi with a large $\eta$ the fitted exponent is steeper than $H - \tfrac12$ (slot 5). |
| "On an almost flat curve, the network's posterior is not an ellipse." | "Rough Bergomi's posteriors are banana-shaped." | One development case, network draws, no exact reference yet. |
| "Where the quotes say nothing, the stated prior decides." | "The prior is arbitrary." | The prior is stated and tested; slot 6 varies it. |
| "At least 17 CPU-hours by a random-walk chain, or about 85 CPU-minutes by importance sampling." | "17 CPU-hours", or the chain's cost alone. | The chain's cost is a lower bound (study S9); importance sampling targets the same exact posterior ([v2-plan](../../course/v2-plan.md), Section 2.4). |
| "1.4 seconds for 1,000 draws." | "Milliseconds." | Not what was measured. |
| "Art. 9 for the calibrated parameters, Art. 11 for the choice of model." | "Art. 11 asks for the range of calibrations." | Art. 11(1) excludes calibrations from market-derived parameters, which Art. 9 covers (Lecture 6, Section 6.6). |
| "A posterior range is an input to a valuation adjustment." | That a posterior quantile is a complete prudent-valuation procedure. | It is a within-model range ([tutorial review](TUTORIAL-REVIEW.md)). |

## Hard questions

1. **"With a flat prior, one quote is just least squares."** Yes. Section 3's prior choice shows when the prior moves the answer; section 6 shows where it decides it. Priors are slot 6's subject.
2. **"Black–Scholes is a straw man."** It is the one-parameter case where prior, likelihood and posterior can be drawn on one axis. Section 5 shows it failing on SPY, and section 6 moves to a model that can produce a smile.
3. **"The right panel shows network draws. How do you know the shape is not the network's?"** Not yet from an exact reference: that is pending (J2), and the slide says so. The geometry explains it: when the short and long levels of the curve coincide, its decay barely changes a price.
4. **"Why a fixed noise rule and not each day's bid–ask?"** Real and simulated smiles must carry the same noise, or a network trained on simulations would read real quotes as a different kind of data. On 21 October the rule is wider than SPY's half-spread at every grid point, and slot 6 varies its level.
5. **"Why a grid, and not the chain's own quotes?"** The grid gives real and simulated smiles the same 63 inputs, so one network serves both. Set networks, discussed in slot 3, would read a chain directly and remove the grid.
6. **"Heston calibrates in a second; why would I need a network?"** For one smile, the exact Heston posterior is affordable, and slot 4 computes it. A validation needs thousands of posteriors, and a valuation across a book needs one for every name (slot 2). Rough Bergomi has no fast pricer at all (slot 5).

## What is left out, and why

- **Hierarchies across names** (the sector of names): agreed on 29 October to keep them out of slot 1.
- **The classical-versus-Bayes toy of Lecture 9.7**: replaced by a case from the afternoon's own models.
- **The long-end products table**: pricing products is slot 5's; further exotics are an extension.
- **The 15-quote Heston panel**: sparse quote chains are deferred.
- **The HKDE two-optima slide**: Heston with jumps is outside the core models, and the study is not registered as a claim.
- **Option D, betas across stocks** (branch `feat/tutorial-slot1-intro`), **the proposal's Figure 1**, and **returns and value-at-risk** (overview, §6).

## To build, by the freeze on Wed 4 Nov

1. Alex's review, as agreed on 29 October; a semi-final version by Sunday 1 November.
2. Decide whether section 6's right panel waits for a conditional reference (J2).
3. Remove the assets and modules only the old slot 1 used (`why_bayes.py`, `bs_hierarchy.py`, `quick-checks/bs_hierarchy.json`, `quick-checks/long_end/`, `images/l10-credible-vs-confidence.png`) through the lab's tools, with the repository cleanup.
4. Rehearse with a timer, twice, before the run-through of Mon 9 Nov.

## Course lectures behind slot 1

Lectures 1 and 6 (the products, the rules), 4 and 5 (Black–Scholes, implied volatility), 7 and 8 (calibration as estimation, the posterior and propagation), 14 (cost).

# Slot 1. Introduction: Bayes for volatility models

Francesco · 15 minutes · the first of the opening Beamer talks. Part of the [curriculum](00-overview.md).

![Minute map of slot 1](images/slot-1.png)

Rebuilt on 10 November 2026 for the two-hour workshop of 14 November: Beamer talks for the introduction, amortized
Bayes and architectures, then the application notebooks, each run as model → simulate → train → infer → check. The talk
is [`slides/slot1/slot1.tex`](../slides/slot1/slot1.tex), compiled in the tutorial's Overleaf project
([`slides/slot1/README.md`](../slides/slot1/README.md)). The notebook [`slot1_intro.py`](../notebooks/slot1_intro.py)
remains a supporting demo; the deck's figures do not come from it.

The talk has one aim: that the audience sees why the methods of the tutorial may matter for their own models. It
states the problem on volatility models, says that these are an exercise and that the steps and the questions are
general, argues for Bayes while showing where the classical route gives the same answer, argues for simulation-based
inference in three slides, and ends on the team and the rest of the tutorial. Seven of its eight figures come from
one running example, so the argument stays on one case.

## The slot in three sentences

**The question.** How sure is the price that a calibrated model gives to what the market does not quote, and how is
that uncertainty computed and checked?

**The answer.**

1. A calibration is one fit among many that are consistent with the quotes, so a price computed from it has a range.
2. The posterior is that set of fits, weighted. Where the quotes determine the price, it returns the classical range.
   Where they do not, it states the assumption that sets the range, and one set of draws prices every product.
3. A posterior costs thousands of model evaluations. A network trained once on simulations returns it in under a
   second, which also makes its coverage testable; it is valid for data like its training data.

## What the audience leaves with

- The five steps every notebook follows (model, simulate, train, infer, check), and that only the first is specific to
  volatility models.
- Six questions the methods answer for any model that can be simulated, and the form each takes in the tutorial.
- What a posterior is: every parameter set, weighted by its fit and its prior; least squares is its peak under a
  flat prior.
- Where Bayes changes the answer (what the quotes leave open) and where it does not (quotes that determine the
  price, as the Bernstein–von Mises theorem predicts).
- The habits of calibration read as assumptions: a fixed parameter is a prior of zero width, a free fit has no prior.
- What exact posteriors cost, the idea of amortized inference, and the tests that make a posterior trustworthy:
  coverage on simulated data, and agreement with exact posteriors; and that a network is valid only for data like its
  training data.
- Who presents, and what each later part adds.

## Slide by slide

| Min | Slide | Key line (read by everyone); figure; optional facts | Source and status |
| --- | --- | --- | --- |
| 0:00–0:50 | The Problem | "A calibrated model prices what the market does not quote"; the desk diagram, from quotes to the book; facts: quoted, not quoted, the question | De Spiegeleer et al. 2018; Schoutens et al. 2004 |
| 0:50–2:10 | One Fit Among Many (two steps) | "Calibration returns the best fit, and a price for anything else", then "Many fits are consistent with the quotes. They disagree elsewhere"; quotes at 7 and 14 days with the fit, then 120 posterior draws and the forward smile in 6 months, which no option quotes; facts: the example, the misfit, the forward at-the-money volatility | The running example; illustration |
| 2:10–3:10 | From a Price to a Range | "Each parameter set gives a price. Together they form a range"; the forward-start call under every draw, with the least-squares price, the mark, the 10% point and the reserve; facts: the product, mark and 10% point, the prudent value of a long position | The running example; Reg. (EU) 2016/101 |
| 3:10–4:10 | An Exercise in Inference | "Volatility models are the exercise. The five steps are general"; the five steps and the simulators, with "your simulator"; facts: in finance, in science, what is needed | Cranmer et al. 2020 |
| 4:10–5:10 | Questions These Methods Answer | "The methods answer questions about uncertainty"; six questions, one per line, each with its form in today's examples | The notebooks |
| 5:10–6:10 | From a Fit to a Posterior | "Least squares keeps one fit. The posterior keeps them all, weighted"; the posterior in two planes, with the least-squares point and its ellipse; facts: likelihood, prior, posterior | The running example |
| 6:10–7:20 | Where the Methods Agree | "Where the quotes determine the price, the two ranges agree"; the forward start's classical and posterior ranges as the quotes stop at 325, 81, 30, 14 and 7 days; facts: the classical range, where the quotes decide, quotes at 7 days only | The running example; the Bernstein–von Mises theorem |
| 7:20–8:30 | What the Quotes Leave Open | "What the quotes leave open is set by an assumption"; the forward start's range with κ fixed at 2, 4 and 8, fitted, and under the posterior; facts: fix it, fit it freely, state a prior | The running example; Bergomi 2003; Janek et al. 2011 |
| 8:30–9:20 | One Set of Draws | "One set of posterior draws prices every product, consistently"; two products under the same draws, and a book; facts: a draw, a product, a book | The running example |
| 9:20–10:10 | The Cost of a Posterior | "A posterior takes thousands of model evaluations. A book or a validation takes hundreds of posteriors"; minutes, a day, no formula | The Heston notebook's PyMC references; option application, Section 7.1 |
| 10:10–11:10 | Amortized Inference | "A network trained once on simulations returns the posterior for new data in under a second"; the pipeline; facts: what it needs, new data, where it is valid | Cranmer et al. 2020; Radev et al. 2022; the Heston notebook; option application, Section 7.5 |
| 11:10–12:10 | A Posterior That Can Be Tested | "A 90% range must contain the truth 90% of the time"; stated against measured coverage of the Heston network on 512 simulated surfaces, six targets; facts: the test, the 90% intervals, the cost of 300 exact rough-Bergomi posteriors | Option application, Sections 7.1 and 8.2 |
| 12:10–13:00 | Summary | Three statements, and three questions for the audience's own model | — |
| 13:00–13:40 | The Team | The four presenters: photo, affiliation, background, parts | The tutorial's webpage |
| 13:40–14:30 | The Rest of the Tutorial | "Each part adds one building block"; the timeline, and one row per part: the time, the part over what it adds, the presenters | The tutorial's webpage |

Each main slide follows one reading order: the title and the key line are read by everyone, the figure is the
evidence, and a strip of at most three short label–value facts is optional; explanations go in the speaker notes
([`slides/slot1/README.md`](../slides/slot1/README.md)).

Five backup frames follow: the running example with its numbers, what the quotes leave open and what practice does
(the quick checks behind slide 8), the rules and what a posterior adds to them, the methods that compute a posterior,
and the references.

## The running example

The example is the simulated case "market-like parameters" of the tutorial's Heston notebook
(`lab/notebooks/heston_spx_calibration.py`), the same 63 numbers: Heston with known parameters, close to those that
fit the market, prices the 63 SPX quotes of 9 November 2026 (9 strikes at standardized moneyness δ from about −4 to
1.6 on 7 expiries from 7 to 325 days), and each price is observed once with the error law of the notebook and the
option application, c × its own half-spread × a standard normal, with c = 1. The prior is the notebook's uniform box,
with c inferred under U(0.5, 2). The same
surface is read with quotes up to 325, 81, 30, 14 and 7 days. Slides 2, 3, 6, 8 and 9 use the quotes up to 14 days;
slide 7 uses all five readings. The product is the application's forward-start call, struck at the money in
6 months and paid at 1 year; slide 9 holds it against the listed 325-day at-the-money call. The classical route is
least squares with the delta method, with c by maximum likelihood; the posterior is computed by importance sampling.
[`slides/slot1/README.md`](../slides/slot1/README.md) gives the method, its checks and its limits.

The example is an illustration, labelled so on the slides ("simulated quotes") and in the backup. The model is right
by construction, the quote errors are independent, and one surface is shown, the notebook's. It supports statements about a mechanism, not about a market: on the real quotes Heston misses by about
13 half-spreads and c goes to its bound (option application, Section 6.5).

## Why Bayes, and what the slot does not claim

**The classical route is given its due.** Slide 7 shows least squares with the delta method giving the same range as
the posterior while the quotes reach 30 days or more, which is what the Bernstein–von Mises theorem predicts when the
quotes determine the parameters. The slide says that Bayes is not needed for the range there. The tutorial has not
yet measured the coverage of the two routes on its current SPX design: the delta-method and expiry-drop baselines are
remaining work of the option application (its Section 10.3), and the deck does not cite the earlier study on the
retired κ-fixed task.

**The argument is the assumption, not the shape.** Where the course measured it, on simulated thin chains, a
Gaussian at the mode with the same prior gave the same product ranges as the posterior (Lecture 16, Section 16.4).
The deck therefore argues from what the quotes leave open: fixing a parameter is a prior of zero width, a free fit
has no prior, and a prior states a width that can be discussed and tested (slide 8 and its backup). Slide 6 shows a
curved posterior beside the least-squares ellipse as a picture of what the quotes leave open, not as a claim that
shape drives prices.

**One set of draws.** Slide 9 adds what a set of draws gives beyond a range for one product: every product priced
under the same parameter sets, and a book's range that accounts for how its products move together.

**The rules.** Prudent valuation asks for the point of a range of plausible values at which the bank is 90% confident
(Regulation (EU) 2016/101, Articles 9 and 11): the 10% point for a position it owns, the 90% point for one it owes. Banks meet this today with consensus data and expert ranges
([practice sources](../../course/practice-sources.md)), so the rules are not by themselves a reason for Bayes. The
notes of slide 3 and a backup frame say what a posterior adds: a range derived from the data, with its prior stated,
that can be tested by coverage. That a posterior is an accepted basis for a prudent value is a proposal.

**What a passed test says.** Slide 12 tests the network against the posterior of the stated model, on average over the
prior; a network that returned the prior would also pass, so the application compares it with exact posteriors too.
A passed test covers data like the training data: on the SPX quotes of 9 November, Heston misses by far more than any
error the training allowed, the market lies outside what the network saw, and the application's market results use
the exact posterior. Whether the model describes the market is a separate check, which the later parts take up; the
notes say so.

**Downstream uses.** The talk names marking and reserving products that are not quoted. It makes no claim about
hedging under parameter uncertainty or about value-at-risk.

## Numbers on screen, and their status

`export_figures.py` prints every number of the running example; compare its report with this table after any change.
The application's numbers are as of 11 November 2026 (`lab/plans/option-application/`, branch
`claude/tender-bardeen-l6tivw`).

| Number | Source | Status |
| --- | --- | --- |
| 18 quotes at 7 and 14 days; root-mean-square misfit 1.01 half-spreads for the best fit, 1.05 to 1.48 for the posterior draws (5% to 95%) | the running example | illustration |
| At-the-money forward volatility, 6 months to 1 year: 14.7% to 18.1% across 120 posterior draws (fit 16.1%, generating parameters 16.5%) | the running example | illustration |
| The forward-start call with quotes up to 14 days: least squares 4.55, mark 4.58, 10% point 4.37, reserve 0.21 (5% of the mark), per 100 of forward; 4.66 under the generating parameters | the running example | illustration |
| The forward start's 90% range, delta method against posterior: 4.65–4.67 and 4.65–4.67 (quotes up to 325 days), 4.66–4.70 and 4.66–4.70 (81 days), 4.59–4.74 and 4.58–4.77 (30 days), 4.35–4.75 and 4.31–4.89 (14 days); at 7 days the fit sits on the upper bound of κ and has no range, the posterior gives 2.40–8.56 | the running example | illustration |
| κ fixed at 2, 4 and 8 on the 14-day quotes: 4.41–5.10, 4.37–4.84 and 4.35–4.67; error scale by maximum likelihood 1.17, 1.04 and 1.08, against 1.01 with κ fitted | the running example | illustration |
| 90% widths: the forward start 0.57, the 325-day call 0.64, the forward start minus the call 0.08; correlation across draws 0.996 | the running example | illustration |
| One exact posterior on the 63 SPX quotes: Heston 14–27 CPU-min by MCMC (about 180,000 pricings of the quotes, one core) | the Heston notebook's PyMC references (`lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/`) | recorded |
| Rough Bergomi about 17–29 CPU-h per exact posterior by importance sampling, at 1–2 CPU-min per pricing; 300 exact rough-Bergomi posteriors 5,000–9,000 CPU-h | option application, Section 7.1 | estimated from its pricing cost |
| The Heston network: 300,000 simulated surfaces, under an hour of training on a laptop CPU; about 0.3 s per posterior | the Heston notebook; option application, Section 7.5 | recorded |
| Against the exact posterior on the notebook's simulated quotes, the network's means lie within one reference standard deviation and its intervals are 1.05 to 2.9 times as wide; on SPX, Heston misses by 13 half-spreads and c runs to its bound of 2 | option application, Section 8.3; the notebook's run record | recorded |
| The rough-Bergomi bank: 47–59 hours on 192 CPUs | option application, Section 7.3 | estimate; being generated |
| Coverage of the Heston network on 512 simulated surfaces, six targets: 50.0–67.6%, 78.9–92.6%, 89.3–97.5% and 93.2–99.2% at 50, 80, 90 and 95%; at 90%, 89–92% for five targets and 97.5% for v₀ | option application, Section 8.2 | recorded |
| On the SPX quotes the network is outside its training range; market results use the exact posterior | option application, Section 8.3 | recorded |
| The backup's quick checks: 23% of thin chains; 0.30–0.32 of the posterior's range; 59.5% against 88.0%; 76 of 100 fits on a bound, ranges 1.5–3.3 times the posterior's | the course, Lectures 8.10, 15.6 and 16.4 | quick checks, not results of the option application |

The deck cites no claim of the ledger: C7, C8 and C11 were measured on the earlier Heston task with κ fixed at 2,
which the option application has replaced. The application's results enter `tutorial/assets/` and the ledger only
through the lab's tools; until then the slides name the application's sections.

## What may be said, and what may not

| Say | Do not say | Why |
| --- | --- | --- |
| "Many parameter sets are consistent with the quotes." | "Calibration is broken", or "least squares is wrong." | The best fit is a valid estimate; the point is that the quotes cannot tell it from its neighbours. |
| "This is an illustration on simulated quotes; the model is right by construction." | The example's numbers as results of the tutorial. | One simulated surface, independent errors. |
| "Where the quotes determine the price, the classical range and the posterior agree." | "Classical methods fail", or "Bayes gives better intervals." | Slide 7's upper rows; the Bernstein–von Mises theorem. |
| "Where we measured, the prior made the difference, not the posterior's shape." | "Classical methods miss the posterior's shape." | Lecture 16, Section 16.4: a Gaussian with the same prior matches the posterior on thin chains. |
| "Fixing a parameter is a prior of zero width." | "Desks fix κ at 2." | Fixing κ is documented (Bergomi 2003; Janek et al. 2011); the value differs by desk. |
| "The rules are met today, with consensus data and expert ranges; a posterior adds a range derived from the data, with its assumptions stated, that can be tested." | "Banks cannot meet the rules without Bayes." | Consensus services publish 10th and 90th percentiles; the regulation allows proxies and expert-based ranges. |
| "The reserve is the mark minus the 10% point, for a position the bank owns; the 90% point minus the mark for one it owes." | "The reserve is a VaR." | VaR is a quantile of P&L over a horizon. |
| "A posterior takes thousands of model evaluations: minutes for Heston, about a day for rough Bergomi." | A precise rough-Bergomi cost as measured. | 17–29 CPU-h is estimated from 1,000 pricings at 1–2 CPU-minutes each (application, Section 7.1). |
| "A posterior in under a second, after training." | "Milliseconds", or "instant on the market". | About 0.3 s measured; on the SPX quotes the network was outside its training range. |
| "The network is accurate only for data like its training data, and it is tested before it is used." | "The network gives the posterior." | It is an approximation: on simulated surfaces its intervals are 1.05 to 2.9 times the exact ones, those for v₀ too wide; on SPX it is far off (application, Sections 8.2 and 8.3). |
| "A passed coverage test says the network returns the posterior of the stated model." | "The range is right for the market." | The range is within one model, under a stated error model. |
| "The same steps apply to any model that can be simulated." | That the tutorial shows results outside volatility models. | The examples are Heston, rough Bergomi and return histories. |

## Hard questions

1. **"With a flat prior this is just least squares."** At its peak, yes, and slide 6 says so. The posterior keeps the
   other fits too; they matter where the quotes leave something open (slides 7 and 8).
2. **"A careful classical analysis gives the same range."** Where the quotes determine the price it does, and slide 7
   shows it. The delta method stops when the fit runs to a bound, and a fixed parameter then replaces it with false
   certainty. A profile-likelihood range would still exist there, and it too would be set by the box.
3. **"So the prior is doing the work."** Where the quotes are silent, yes, and it must: the classical habits are
   priors of zero width, or none. Bayes states the width, and the width can be tested.
4. **"Your example is simulated, and the model is true."** Yes, and it is labelled so. It is the Heston notebook's
   own simulated case: the strikes, expiries and half-spreads of the 63 SPX quotes of 9 November, generated with
   c = 1, with an error scale c that is inferred. It shows the mechanism. On the real quotes Heston misses by about 13 half-spreads and c goes to its
   bound of 2; the notebooks show that misfit and what it means for the ranges.
5. **"Banks already meet the rules."** They do. A posterior adds a range derived from the data, with a stated prior,
   that can be tested by coverage. That it is an accepted basis for a prudent value is a proposal.
6. **"Heston calibrates in a second; why a network?"** One exact Heston posterior is affordable (14 to 27 CPU-minutes),
   and the Heston part computes it as the reference. A validation needs hundreds of posteriors, a book one per name
   per day, and rough Bergomi has no fast pricer: about a day per exact posterior.
7. **"Why trust a network?"** Only as far as it passes its checks: coverage on simulated data, agreement with an exact
   posterior where one exists, and data like its training data. The Heston network passes the first two on simulated
   surfaces (with intervals for v₀ too wide) and fails the third on the SPX quotes, so the market results use the
   exact posterior.
8. **"Does the range cover model risk?"** No. It is one model's parameter uncertainty under a stated error model. Two
   models on the same data show part of the rest.
9. **"Why not my model?"** The method does not depend on the model: a prior, a simulator and an error model are the
   first step, and the other four are the same.
10. **"Do you retrain every day?"** In the tutorial one network serves one snapshot's design (strikes, expiries,
   half-spreads); a new design needs its own simulations and network (option application, Section 5.1). A network that
   takes the design as an input is the way to daily use, and the tutorial does not show it.

## What is left out, and why

- **The ingredients on Black–Scholes and one SPY close** (likelihood, prior sweep, predictive check). They stay in
  the notebook `slot1_intro.py`; the talk spends its time on the argument.
- **The defence of the error law, and model error.** The application notebooks own the error model: the half-spread
  as the scale, c inferred, and the smooth model-error extension.
- **Hierarchies across names.** One row of the backup.
- **The posterior's shape as an argument.** See "Why Bayes" above.
- **Hedging and value-at-risk.** Nothing in the repository supports a claim.

## Before the run-through

1. Compile in the Overleaf project with the real theme and check that no frame overflows; the deck has been compiled
   here only against a stand-in for the theme.
2. Confirm the presenters' parts on slides 14 and 15 against the final programme.
3. If the option application's results change before 14 November (rough Bergomi's bank and network, the products
   and the delta-method baselines), update slides 10 to 12 and this page.
4. Rehearse with a timer, twice.

## Sources

De Spiegeleer, Madan, Reyners and Schoutens (2018), SSRN 3191050; Schoutens, Simons and Tistaert (2004); Cranmer,
Brehmer and Louppe (2020); Radev et al. (2022); the course, Lectures 6, 8, 14 and 16.4, and its
[practice sources](../../course/practice-sources.md); the option application, `lab/plans/option-application/`
(results as of 11 November 2026).

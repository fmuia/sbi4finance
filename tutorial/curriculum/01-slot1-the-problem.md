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
3. A posterior costs thousands of model evaluations. A network trained once on simulations returns it in seconds,
   which also makes its coverage testable.

## What the audience leaves with

- The five steps every notebook follows (model, simulate, train, infer, check), and that only the first is specific to
  volatility models.
- Six questions the methods answer for any model that can be simulated, and the form each takes in the tutorial.
- What a posterior is: every parameter set, weighted by its fit and its prior; least squares is its peak under a
  flat prior.
- Where Bayes changes the answer (what the quotes leave open) and where it does not (quotes that determine the
  price; C11).
- The habits of calibration read as assumptions: a fixed parameter is a prior of zero width, a free fit has no prior.
- What exact posteriors cost, the idea of amortized inference, and the test that makes a posterior trustworthy:
  coverage on simulated data.
- Who presents, and what each later part adds.

## Slide by slide

| Min | Slide | Key line (read by everyone); figure; optional facts | Source and status |
| --- | --- | --- | --- |
| 0:00–0:50 | The Problem | "A calibrated model prices what the market does not quote"; the desk diagram, from quotes to the book; facts: quoted, not quoted, the question | De Spiegeleer et al. 2018; Schoutens et al. 2004 |
| 0:50–2:10 | One Fit Among Many (two steps) | "Calibration returns the best fit, and a price for anything else", then "Many fits are consistent with the quotes. They disagree elsewhere"; quotes at 1 and 3 months with the fit, then 120 posterior draws and the 2-year smile nobody quotes; facts: the example, the misfit, the at-the-money volatility at 2 years | The running example; illustration |
| 2:10–3:10 | From a Price to a Range | "Each parameter set gives a price. Together they form a range"; the 2-year put under every draw, with the least-squares price, the mark, the 90% point and the reserve; facts: the product, mark and 90% point, prudent valuation | The running example; Reg. (EU) 2016/101 |
| 3:10–4:10 | An Exercise in Inference | "Volatility models are the exercise. The five steps are general"; the five steps and the simulators, with "your simulator"; facts: in finance, in science, what is needed | Cranmer et al. 2020 |
| 4:10–5:10 | Questions These Methods Answer | "The methods answer questions about uncertainty"; six questions, one per line, each with its form in today's examples | The notebooks |
| 5:10–6:10 | From a Fit to a Posterior | "Least squares keeps one fit. The posterior keeps them all, weighted"; the posterior in two planes, with the least-squares point and its ellipse; facts: likelihood, prior, posterior | The running example |
| 6:10–7:20 | Where the Methods Agree | "Where the quotes determine the price, the two ranges agree"; the 2-year put's classical and posterior ranges as the quotes stop at 2 years, 1 year, 6, 3 and 1 months; facts: the classical range, the measured coverage, quotes at 1 month only | The running example; C11, verified |
| 7:20–8:30 | What the Quotes Leave Open | "What the quotes leave open is set by an assumption"; the put's range with κ fixed at 1, 2 and 4, fitted, and under the posterior; facts: fix it, fit it freely, state a prior | The running example; Bergomi 2003; Janek et al. 2011 |
| 8:30–9:20 | One Set of Draws | "One set of posterior draws prices every product, consistently"; two products under the same draws, and a book; facts: a draw, a product, a book | The running example |
| 9:20–10:10 | The Cost of a Posterior | "A posterior takes thousands of model evaluations. A book or a validation takes thousands of posteriors"; three orders of magnitude | C8, verified |
| 10:10–11:10 | Amortized Inference | "A network trained once on simulations returns the posterior for new data in seconds"; the pipeline; facts: what it needs, new data, where it is valid | Cranmer et al. 2020; Radev et al. 2022; laptop benchmark |
| 11:10–12:10 | A Posterior That Can Be Tested | "A 90% range must contain the truth 90% of the time"; stated against measured coverage of a Heston network on 2,000 simulated smiles; facts: the test, the 90% intervals, the cost for rough Bergomi | S2 (C7, verified); Lecture 14.4 |
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

Heston with known parameters generates 35 option prices: 7 strikes at 1, 3 and 6 months, 1 and 2 years, each observed
once with the tutorial's error band. The same surface is read with quotes that reach 2 years or stop earlier. Slides
2, 3, 6, 8 and 9 use the quotes up to 3 months; slide 7 uses all five readings. The products are a 2-year put struck
at 90% of the forward, and a call that starts at the money in 1 year and pays at 2. The classical route is least
squares with the delta method; the posterior uses a uniform prior on the application's parameter box and is computed
by importance sampling. [`slides/slot1/README.md`](../slides/slot1/README.md) gives the method, its checks and its
limits.

The example is an illustration, labelled so on the slides ("simulated quotes") and in the backup. The model is right
by construction, the quote errors are independent, and one surface is shown, with a seed fixed before any figure was
drawn. It supports statements about a mechanism, not about a market.

## Why Bayes, and what the slot does not claim

**The classical route is given its due.** Slide 7 shows least squares with the delta method giving the same range as
the posterior while the quotes reach one or two years, and cites the verified study in which both kinds of interval
cover at about the stated rate on 200 simulated Heston smiles (C11). The slide says that Bayes is not needed for the
range there.

**The argument is the assumption, not the shape.** Where the course measured it, on simulated thin chains, a
Gaussian at the mode with the same prior gave the same product ranges as the posterior (Lecture 16, Section 16.4).
The deck therefore argues from what the quotes leave open: fixing a parameter is a prior of zero width, a free fit
has no prior, and a prior states a width that can be discussed and tested (slide 8 and its backup). Slide 6 shows a
curved posterior beside the least-squares ellipse as a picture of what the quotes leave open, not as a claim that
shape drives prices.

**One set of draws.** Slide 9 adds what a set of draws gives beyond a range for one product: every product priced
under the same parameter sets, and a book's range that accounts for how its products move together.

**The rules.** Prudent valuation asks for the 90% point of a range of plausible values (Regulation (EU) 2016/101,
Articles 9 and 11). Banks meet this today with consensus data and expert ranges
([practice sources](../../course/practice-sources.md)), so the rules are not by themselves a reason for Bayes. The
notes of slide 3 and a backup frame say what a posterior adds: a range derived from the data, with its prior stated,
that can be tested by coverage. That a posterior is an accepted basis for a prudent value is a proposal.

**What a passed test says.** Slide 12 tests the network against the posterior of the stated model. Whether the model
describes the market is a separate check, which the later parts take up; the notes say so.

**Downstream uses.** The talk names marking and reserving products that are not quoted. It makes no claim about
hedging under parameter uncertainty or about value-at-risk.

## Numbers on screen, and their status

`export_figures.py` prints every number of the running example; compare its report with this table after any change.

| Number | Source | Status |
| --- | --- | --- |
| 14 quotes at 1 and 3 months; root-mean-square misfit 0.7 error bands for the best fit, 0.8 to 1.1 for the posterior draws (5% to 95%) | the running example | illustration |
| At-the-money volatility at 2 years: 15.1% to 20.1% across 120 posterior draws (fit 16.5%, generating parameters 17.0%) | the running example | illustration |
| The 2-year put with quotes up to 3 months: least squares 5.44, mark 5.60, 90% point 6.49, reserve 0.89 (16% of the mark), per 100 of forward; 5.99 under the generating parameters | the running example | illustration |
| The put's 90% range, delta method against posterior: 5.87–6.15 and 5.86–6.15 (quotes up to 2 years), 5.58–6.10 and 5.57–6.10 (1 year), 5.08–5.98 and 5.11–6.05 (6 months), 4.50–6.37 and 4.84–6.89 (3 months); at 1 month the fit sits on two bounds and has no range, the posterior gives 2.66–10.32 | the running example | illustration |
| κ fixed at 1, 2 and 4 on the 3-month quotes: 6.23–7.93, 5.73–7.08 and 5.19–6.21; χ² of 10.8, 9.3 and 7.8 on 14 quotes, 7.6 with κ fitted | the running example | illustration |
| 90% widths: the put 2.05, the forward-start call 1.50, the call minus the put 0.68 | the running example | illustration |
| Heston, 200 simulated smiles: Gauss–Newton 90% intervals cover 87–92%, the posterior's 87–90% | C11, `heston/s6/classical.json` | verified |
| Heston about 4 CPU-min per posterior; rough Bergomi at least 17 CPU-h by a random-walk chain | `costs/costs.json` (C8) | verified |
| Rough Bergomi by importance sampling from a Gauss–Newton proposal on SPY's 63 quotes: about 85 CPU-min | `lab/checks/rough_spy_is.py` | verified |
| 1,000 posterior draws in 1.4 to 3.6 s | [laptop benchmark](../../lab/evidence/pilot-results/2026-10-29/laptop-inference-benchmark/README.md) | measured; one case per model on one machine |
| Coverage of a Heston network on 2,000 simulated smiles: 48.1–50.1%, 78.3–79.6%, 88.7–90.2% and 93.5–95.2% at 50, 80, 90 and 95% | `heston/s2/validation.json` (S2; C7) | verified; the network of the first study, four parameters, the 63-point grid |
| A coverage test on 2,000 rough-Bergomi smiles: about 2 CPU-h with the network after a bank of about 370 CPU-h, about 2,800 by importance sampling | Lecture 14.4, from C8 and S1 | as C8; the bank at the pilot's rate |
| The backup's quick checks: 23% of thin chains; 0.30–0.32 of the posterior's range; 59.5% against 88.0%; 76 of 100 fits on a bound, ranges 1.5–3.3 times the posterior's; no interval on 11 of 200 | Lectures 8.10, 15.6 and 16.4; C11 for the last | quick checks; the last verified |

The costs on slides 10 and 12 are those of the first studies, on the 63-point grid. If the application of 14 November
measures new ones, the notebooks state them; the slides give orders of magnitude and keep the verified figures in the
source lines.

## What may be said, and what may not

| Say | Do not say | Why |
| --- | --- | --- |
| "Many parameter sets are consistent with the quotes." | "Calibration is broken", or "least squares is wrong." | The best fit is a valid estimate; the point is that the quotes cannot tell it from its neighbours. |
| "This is an illustration on simulated quotes; the model is right by construction." | The example's numbers as results of the tutorial. | One simulated surface, independent errors. |
| "Where the quotes determine the price, the classical range and the posterior agree." | "Classical methods fail", or "Bayes gives better intervals." | Slide 7's upper rows; C11. |
| "Where we measured, the prior made the difference, not the posterior's shape." | "Classical methods miss the posterior's shape." | Lecture 16, Section 16.4: a Gaussian with the same prior matches the posterior on thin chains. |
| "Fixing a parameter is a prior of zero width." | "Desks fix κ at 2." | Fixing κ is documented (Bergomi 2003; Janek et al. 2011); the value differs by desk. |
| "The rules are met today, with consensus data and expert ranges; a posterior adds a range derived from the data, with its assumptions stated, that can be tested." | "Banks cannot meet the rules without Bayes." | Consensus services publish 10th and 90th percentiles; the regulation allows proxies and expert-based ranges. |
| "The reserve is the 90% point minus the mark, for a sold position." | "The reserve is a VaR." | VaR is a quantile of P&L over a horizon. |
| "A posterior takes thousands of model evaluations: minutes for Heston, hours for rough Bergomi." | "17 CPU-hours", or the chain's cost alone. | The chain's cost is a lower bound; importance sampling targets the same posterior in about 85 CPU-minutes. |
| "1,000 draws in a few seconds." | "Milliseconds." | 1.4 to 3.6 s measured. |
| "The network is accurate only for data the prior can produce, and it is tested before it is used." | "The network gives the posterior." | It is an approximation; C5 records where the Heston network alone missed its criterion. |
| "A passed coverage test says the network returns the posterior of the stated model." | "The range is right for the market." | The range is within one model, under a stated error model. |
| "The same steps apply to any model that can be simulated." | That the tutorial shows results outside volatility models. | The examples are Heston, rough Bergomi and return histories. |

## Hard questions

1. **"With a flat prior this is just least squares."** At its peak, yes, and slide 6 says so. The posterior keeps the
   other fits too; they matter where the quotes leave something open (slides 7 and 8).
2. **"A careful classical analysis gives the same range."** Where the quotes determine the price it does, and slide 7
   shows it. It stops when the fit runs to a bound, and a fixed parameter then replaces it with false certainty.
3. **"So the prior is doing the work."** Where the quotes are silent, yes, and it must: the classical habits are
   priors of zero width, or none. Bayes states the width, and the width can be tested.
4. **"Your example is simulated, and the model is true."** Yes, and it is labelled so. It shows the mechanism. The
   notebooks take real quotes, a model error and the checks.
5. **"Banks already meet the rules."** They do. A posterior adds a range derived from the data, with a stated prior,
   that can be tested by coverage. That it is an accepted basis for a prudent value is a proposal.
6. **"Heston calibrates in a second; why a network?"** One exact Heston posterior is affordable, and the Heston part
   computes it as the reference. A validation needs thousands of posteriors, a book one per name per day, and rough
   Bergomi has no fast pricer.
7. **"Why trust a network?"** Only as far as it passes its checks: coverage on simulated data, agreement with an exact
   posterior where one exists, and predictive checks on the real data.
8. **"Does the range cover model risk?"** No. It is one model's parameter uncertainty under a stated error model. Two
   models on the same data show part of the rest.
9. **"Why not my model?"** The method does not depend on the model: a prior, a simulator and an error model are the
   first step, and the other four are the same.

## What is left out, and why

- **The ingredients on Black–Scholes and one SPY close** (likelihood, prior sweep, predictive check). They stay in
  the notebook `slot1_intro.py`; the talk spends its time on the argument.
- **The defence of the error band, and model error.** The application notebooks own the error model.
- **Hierarchies across names.** One row of the backup.
- **The posterior's shape as an argument.** See "Why Bayes" above.
- **Hedging and value-at-risk.** Nothing in the repository supports a claim.

## Before the run-through

1. Compile in the Overleaf project with the real theme and check that no frame overflows; the deck has been compiled
   here only against a stand-in for the theme.
2. Confirm the presenters' parts on slides 14 and 15 against the final programme.
3. If the application of 14 November has measured costs or coverage by then, decide whether slides 10 and 12 keep the
   first studies' figures.
4. Rehearse with a timer, twice.

## Sources

De Spiegeleer, Madan, Reyners and Schoutens (2018), SSRN 3191050; Schoutens, Simons and Tistaert (2004); Cranmer,
Brehmer and Louppe (2020); Radev et al. (2022); the course, Lectures 6, 8, 14 and 16.4, and its
[practice sources](../../course/practice-sources.md); [CLAIMS.md](../CLAIMS.md) for C5, C7, C8 and C11.

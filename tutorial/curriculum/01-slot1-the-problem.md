# Slot 1. Introduction: Bayes for volatility models

Francesco · 15 minutes · the first of the opening Beamer talks. Part of the [curriculum](00-overview.md).

![Minute map of slot 1](images/slot-1.png)

Rebuilt on 11 November 2026 for the two-hour workshop of 14 November: Beamer talks for the introduction, amortized
Bayes and architectures, then the application notebooks, each run as model → simulate → train → infer → check. The talk
is [`slides/slot1/slot1.tex`](../slides/slot1/slot1.tex), compiled in the tutorial's Overleaf project
([`slides/slot1/README.md`](../slides/slot1/README.md)). The notebook [`slot1_intro.py`](../notebooks/slot1_intro.py)
remains a supporting demo; the deck's figures do not come from it.

The talk has two jobs: to introduce Bayesian inference for the calibration of volatility models, and to set up the
rest of the tutorial. It opens on why uncertainty matters in finance and what Bayes offers, takes the calibration of
volatility models as its example, says that the steps and the questions are general, shows Bayes on that example
(prior and likelihood, the posterior as draws, from the posterior to the price of an exotic), says where least
squares gives the same answer, names the cost, and hands over to amortized inference.

The validation of the network (coverage, comparison with exact posteriors) is not in this talk: it belongs to the
worked example at 11:40. Those slides are in the backup.

## The slot in three sentences

1. A calibration is an estimate, and decisions need its uncertainty: Bayes returns a distribution over parameters,
   with the assumptions stated.
2. Each posterior draw prices every product, so each price gets a range; with informative quotes least squares with
   a quadratic approximation gives the same ranges, and Bayes is the general case.
3. An exact posterior costs minutes to months; amortized inference pays that cost once, which is the subject of the
   rest of the tutorial.

## What the audience leaves with

- Why a single number is not enough in valuation, regulation, risk and trading.
- What Bayes is, generically: unknowns as distributions, a stated prior, uncertainty that propagates, a result that
  can be checked.
- The five steps every later part follows (model, simulate, train, infer, check), and six questions the methods
  answer for any model that can be simulated.
- What a prior and a likelihood are, on option quotes; that least squares maximizes that same likelihood.
- What a posterior is and how it is computed: draws from a sampler; which parameters the quotes pin and which trade
  off.
- How draws turn into the price range of an exotic, a mark and a reserve.
- That least squares has assumptions too, also gives ranges, and agrees with Bayes when quotes are informative.
- What an exact posterior costs, and that amortized inference is the way around it.
- Who presents, and what each later part adds.

## Slide by slide

| Min | Slide | Key line (read by everyone); what is shown | Evidence | Source |
| --- | --- | --- | --- | --- |
| 0:00–0:50 | Uncertainty in Finance | "A number without a range is not enough to decide"; four rows: valuation, regulation, risk, trading | — | Reg. (EU) 2016/101, Art. 9; IFRS 13, 93(h) |
| 0:50–1:35 | What Bayes Offers | "Bayes treats what is unknown as a distribution"; four rows: one language, prior knowledge, propagation, checks | — | — |
| 1:35–2:25 | The Example: Volatility Models | "A calibrated model prices what the market does not quote"; the desk diagram, from quotes to the book | — | De Spiegeleer et al. 2018; Schoutens et al. 2004 |
| 2:25–3:20 | An Exercise in Inference | "Volatility models are the exercise. The five steps are general"; the five steps and the simulators | — | Cranmer et al. 2020 |
| 3:20–4:05 | Questions These Methods Answer | "The methods answer questions about uncertainty"; six questions, each with its form in today's examples | — | The programme |
| 4:05–4:50 | Calibration: The Bayesian Way | "Bayes' rule turns the quotes into a distribution over parameters"; prior × likelihood ∝ posterior, with one example line each | — | — |
| 4:50–5:50 | Prior and Likelihood | "The prior says what is plausible. The likelihood scores it"; smiles drawn from the prior, and two smiles scored against the quotes | illustration | The running example |
| 5:50–7:15 | What a Posterior Looks Like (three steps) | "A posterior is computed as draws, by a sampler"; "The draws show what the quotes pin, and what they leave open"; "Least squares keeps the peak, and a quadratic region around it"; a sampler's path, the draws in two planes, then least squares | illustration | The running example; the Heston notebook's exact posterior |
| 7:15–8:15 | From Posterior to Prices | "Each draw prices the exotic. Together the prices form a range"; the forward-start call under every draw, with mark, 90% point and reserve | illustration | The running example; Reg. (EU) 2016/101 |
| 8:15–9:25 | Why Bayes | "Bayes makes the calibration's assumptions and uncertainty explicit"; six rows, among them that least squares has assumptions too and is the limit when quotes are rich | — | — |
| 9:25–10:35 | When the Quotes Run Thin | "Rich quotes: the two ranges agree. Thin quotes: Bayes still answers"; the forward start's classical and posterior ranges as five, four, three, two and one expiries are quoted | illustration | The running example; the Heston notebook's exact posterior; the Bernstein–von Mises theorem |
| 10:35–11:35 | The Cost of a Posterior | "One exact posterior costs minutes to months of computing"; CPU time on a logarithmic scale, as orders of magnitude from the tutorial's own runs | result | The Heston notebook's run records; rough Bergomi estimated from its pricing time |
| 11:35–12:15 | Next: Amortized Inference | "Train a network once, on simulations. Then every posterior is fast"; the pipeline, as a handover to the next talk | — | Cranmer et al. 2020; Radev et al. 2022 |
| 12:15–13:05 | Summary | "Bayes returns a distribution. Amortization makes it fast"; three pictures: a posterior, a price range, the cost | — | — |
| 13:05–13:40 | The Team | The four presenters: photo, affiliation, background | — | The tutorial's webpage |
| 13:40–14:20 | The Rest of the Tutorial | "Each part adds one building block"; the timeline, and one row per part | — | The presenters |

Each main slide follows one reading order: the title and the key line are read by everyone, the figure is the
evidence, and a strip of at most three short label–value facts is optional; explanations go in the speaker notes
([`slides/slot1/README.md`](../slides/slot1/README.md)). The notes hold about 1,630 words, about 115 a minute.

Eleven backup frames follow: the coverage test; what the checks find; what an assumption does when one expiry is
quoted; two products and a book; the running example with its numbers; the example beside the notebook; least
squares against the exact posterior on the notebook's 63 quotes; the checks, parameter by parameter; the
prudent-valuation rules; the methods that compute a posterior; and the references.

## The running example

The example keeps the model, the prior, the error law and the "market-like" parameters of the tutorial's Heston
notebook (`lab/notebooks/heston_spx_calibration.py`) and changes one thing, the chain. On the notebook's 63 SPX
quotes Heston's parameters are pinned to about 1% of the prior's range, and any price computed inside Heston has a
range too narrow to draw. The example's chain is thinner and quoted less tightly, as for a single stock: 7 strikes
on each of 1, 3 and 6 months and 1 and 2 years, with half-spreads about ten times those of the SPX quotes. It is
there to show what a posterior is and what it gives, not to claim an effect on SPX.

The product is an exotic, the option application's forward-start call: its strike is set at the money in 6 months
and it pays at 1 year. Slides 7, 8 and 9 use the 14 quotes at 6 months and 1 year, the product's own two dates;
slide 11 reads the same surface with five, four, three, two and one quoted expiries, always including 1 year. The
classical route is least squares with the delta method; the posterior is computed by importance sampling followed by
Metropolis moves, and was checked against independent chains.
[`slides/slot1/README.md`](../slides/slot1/README.md) gives the method, its checks and its limits.

## What the talk claims about Bayes, and what it does not

- **It argues from what the method gives**, not from a contest with least squares: stated assumptions, a
  distribution for any price, a picture of what the quotes determine, a check of the model.
- **Least squares is treated fairly.** It maximizes the same likelihood, it has assumptions too (normal errors,
  fixed weights, a flat prior), and with a quadratic approximation it also gives ranges. On the notebook's 63
  simulated quotes its 90% intervals match the exact posterior within 2% (a backup frame shows the table, and a test
  repeats it). Slide 11 shows where the two part: with two quoted expiries the quadratic region runs past the edge
  of the prior's box, and with one the fit sits on a bound and has no range.
- **The rules.** Prudent valuation asks for a value at 90% confidence (Regulation (EU) 2016/101, Article 9), and
  IFRS 13 for the sensitivity of Level 3 fair values. Banks meet this today with consensus data and expert ranges
  ([practice sources](../../course/practice-sources.md)); a posterior as the basis of a prudent value is a proposal.
- **No claim about SPX.** The example is simulated, and the deck does not quote Heston's misfit on the SPX quotes.
- **No claim about hedging or value-at-risk.**

## Numbers on screen, and their status

`export_figures.py` prints every number of the running example and of the recorded results. The recorded results
are in [`slides/slot1/recorded-results.json`](../slides/slot1/recorded-results.json), each with its source file,
commit and SHA-256; they are development results as recorded on 11 November 2026.

| Number | Where | Source | Status |
| --- | --- | --- | --- |
| The best fit misses the 6-month and 1-year quotes by 1.1 half-spreads; the dashed smile by 10 | slide 7 | the running example | illustration |
| On the notebook's 63 quotes, least squares and the exact posterior agree within 2% | slides 8 and 11 | this deck's least squares; the notebook's PyMC reference | a check, repeated by a test |
| The forward-start call: least squares 4.77, mark 4.70, 90% point 4.92, reserve 0.22 (5% of the mark); 4.66 under the generating parameters | slide 9 | the running example | illustration |
| The forward start's 90% range, delta method against posterior: 4.61–4.81 and 4.61–4.80 (five expiries), 4.56–4.82 and 4.56–4.81 (four), 4.51–4.94 and 4.51–4.90 (three), 4.31–5.23 and 4.39–5.00 (two); with one expiry no classical range, the posterior about 2.9–4.9 | slide 11 | the running example | illustration |
| One exact Heston posterior: 14–27 CPU-minutes, about 146,000 pricings by Fourier integral | slide 12 | the Heston notebook's PyMC references | recorded, measured; shown as orders of magnitude |
| Rough Bergomi: 17–29 CPU-hours by importance sampling, 100–180 CPU-days by MCMC, at 1–2 CPU-minutes per pricing | slide 12 | option application, Section 7.1, and the Heston run's pricing count | estimated, not run |
| The same Heston posterior from a trained network: 0.3 s | slide 12 | the run record | recorded |
| Coverage on 512 simulated surfaces; widths against exact posteriors | backup | the run records | recorded; see the README |

## Hard questions

1. **"With a flat prior this is just least squares."** At its peak, yes: least squares maximizes the same
   likelihood. The posterior keeps the other parameter sets too, and they matter where the quotes leave something
   open (slides 8 and 11).
2. **"A careful classical analysis gives the same range."** With informative quotes it does, and the deck says so.
   The quadratic approximation fails along a curved valley and stops when the fit runs to a bound.
3. **"So the prior is doing the work."** Where the quotes are silent, yes, and something must: bounds, fixed
   parameters and starting points are priors too. Bayes states them.
4. **"Your example is simulated."** Yes, and every slide that uses it says so. On the tutorial's SPX quotes the
   parameters are pinned and the same pictures would show a dot; the example thins the chain to show the mechanism.
5. **"Both dates of the forward start are quoted. Why is it uncertain?"** The options fix the distribution at each
   date; the forward start depends on how the two dates are linked, which the quotes pin only through the model's
   parameters.
6. **"Where do the costs come from?"** From the tutorial's own runs on 63 SPX quotes: about 146,000 pricings per
   exact Heston posterior, milliseconds each by Fourier integral. Rough Bergomi is the same count at one to two
   CPU-minutes per Monte Carlo pricing: an estimate. They are orders of magnitude.
7. **"Why not train a network to price, and run a chain on it?"** That works too. It amortizes the pricer; each
   posterior still costs a chain. Amortizing the posterior makes many posteriors cheap.
8. **"Why trust a network?"** Only as far as it passes its checks; the worked example at 11:40 shows them, and the
   backup has the status on 11 November.
9. **"Does the range cover model risk?"** No. It is one model's parameter uncertainty under a stated error model.

## The supporting notebook

`slot1_intro.py` is an introduction to Bayes on Black–Scholes and one SPY close: the data, Bayes with one parameter,
the range of a price, checking the model, Heston on SPY with quick checks, and the costs. It follows an earlier
version of this talk. The deck does not use it: none of the deck's figures comes from the notebook.

## Before the run-through

1. Compile in the Overleaf project with the real theme and check that no frame overflows; the deck has been compiled
   here only against a stand-in for the theme. If a figure slide overflows, lower `\figmax` at the top of `slot1.tex`.
2. Confirm the presenters on the last slide against the final programme. On 11 November the deck had Stefan Radev
   for amortized inference at 11:15 and Alexander Fengler for the Heston example at 11:40; the public webpage still
   had the earlier order.
3. Rehearse with a timer, twice.

## Sources

De Spiegeleer, Madan, Reyners and Schoutens (2018), SSRN 3191050; Schoutens, Simons and Tistaert (2004); Cranmer,
Brehmer and Louppe (2020); Radev et al. (2022); Talts et al. (2018); van der Vaart (1998), chapter 10; Kruse and
Nögel (2005) for the forward start; the course,
Lectures 6, 8, 14 and 16.4, and its [practice sources](../../course/practice-sources.md); the Heston SPX notebook,
`lab/notebooks/heston_spx_calibration.py`, with its run records, and the option application,
`lab/plans/option-application/` (results as of 11 November 2026).

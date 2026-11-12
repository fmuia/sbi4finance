# Slot 1. Introduction: Bayes for volatility models

Francesco · 15 minutes · the first of the opening Beamer talks. Part of the [curriculum](00-overview.md).

![Minute map of slot 1](images/slot-1.png)

Rebuilt on 11 November 2026 for the two-hour workshop of 14 November: Beamer talks for the introduction, amortized
Bayes and architectures, then the application notebooks, each run as model → simulate → train → infer → check. The talk
is [`slides/slot1/slot1.tex`](../slides/slot1/slot1.tex), compiled in the tutorial's Overleaf project
([`slides/slot1/README.md`](../slides/slot1/README.md)). The notebook [`slot1_intro.py`](../notebooks/slot1_intro.py)
remains a supporting demo; the deck's figures do not come from it.

The opening was rebuilt on 12 November: five slides on why finance needs the uncertainty of its estimates and what
Bayes adds, on one example from 20 years of US returns, replace the two slides that listed where uncertainty matters
and what Bayes offers (both are now in the backup). The rest of the deck is as it was on 11 November and has not yet
been trimmed: the notes now run to about 18 minutes.

The talk has two jobs: to introduce Bayesian inference for the calibration of volatility models, and to set up the
rest of the tutorial. It opens on why finance needs the uncertainty of its estimates and what Bayes adds, on a
portfolio and a share in stocks computed from US returns, takes the calibration of
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

- Why an estimate needs its uncertainty: an optimizer amplifies the errors of its inputs, and the uncertainty of an
  estimate changes a decision.
- What Bayes is: a prior and the data give a distribution for the parameters; the prior states a belief and how
  strongly it is held; what Bayes handles, what it costs, and that with a lot of data it agrees with classical
  methods.
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
| 0:00–1:03 | Uncertainty in Finance: Portfolio Weights | "Inputs are estimates. An optimizer amplifies their errors"; the weights of ten industries before and after one average return is raised by 10%, a third of its standard error | illustration, US returns | The opening example; K. French's data library |
| 1:03–2:11 | Uncertainty in Finance: The Horizon | "Uncertain estimates lower the share in stocks at long horizons"; the share in stocks against the horizon, with the estimates taken as exact and with their uncertainty | illustration, US returns | The opening example; Barberis 2000 |
| 2:11–3:19 | Bayesian Inference | "Bayes gives the parameters a probability distribution"; the prior, the data and the posterior of one expected return, and three reasons finance uses Bayes | illustration, US returns | The opening example; Rachev et al. 2008 |
| 3:19–4:32 | The Prior | "The prior states what you believe, and how strongly"; one portfolio weight as the prior goes from no weight to certainty | illustration, US returns | The opening example; Jorion 1986 |
| 4:32–5:49 | What Bayes Handles, and What It Costs | "Bayes matters most when data are few or an input is left open"; two lists, and why the tutorial uses option calibration | — | Rachev et al. 2008 |
| 5:49–6:39 | The Example: Volatility Models | "A calibrated model prices what the market does not quote"; the desk diagram, from quotes to the book | — | De Spiegeleer et al. 2018; Schoutens et al. 2004 |
| 6:39–7:34 | An Exercise in Inference | "Volatility models are the exercise. The five steps are general"; the five steps and the simulators | — | Cranmer et al. 2020 |
| 7:34–8:19 | Questions These Methods Answer | "The methods answer questions about uncertainty"; six questions, each with its form in today's examples | — | The programme |
| 8:19–9:04 | Calibration: The Bayesian Way | "Bayes' rule turns the quotes into a distribution over parameters"; prior × likelihood ∝ posterior, with one example line each | — | — |
| 9:04–10:04 | Prior and Likelihood | "The prior says what is plausible. The likelihood scores it"; smiles drawn from the prior, and two smiles scored against the quotes | illustration | The running example |
| 10:04–11:29 | What a Posterior Looks Like (three steps) | "A posterior is computed as draws, by a sampler"; "The draws show what the quotes pin, and what they leave open"; "Least squares keeps the peak, and a quadratic region around it"; a sampler's path, the draws in two planes, then least squares | illustration | The running example; the Heston notebook's exact posterior |
| 11:29–12:29 | From Posterior to Prices | "Each draw prices the exotic. Together the prices form a range"; the forward-start call under every draw, with mark, 90% point and reserve | illustration | The running example; Reg. (EU) 2016/101 |
| 12:29–13:39 | Why Bayes | "Bayes makes the calibration's assumptions and uncertainty explicit"; six rows, among them that least squares has assumptions too and is the limit when quotes are rich | — | — |
| 13:39–14:49 | When the Quotes Run Thin | "Rich quotes: the two ranges agree. Thin quotes: Bayes still answers"; the forward start's classical and posterior ranges as five, four, three, two and one expiries are quoted | illustration | The running example; the Heston notebook's exact posterior; the Bernstein–von Mises theorem |
| 14:49–15:49 | The Cost of a Posterior | "One exact posterior costs minutes to months of computing"; CPU time on a logarithmic scale, as orders of magnitude from the tutorial's own runs | result | The Heston notebook's run records; rough Bergomi estimated from its pricing time |
| 15:49–16:29 | Next: Amortized Inference | "Train a network once, on simulations. Then every posterior is fast"; the pipeline, as a handover to the next talk | — | Cranmer et al. 2020; Radev et al. 2022 |
| 16:29–17:19 | Summary | "Bayes returns a distribution. Amortization makes it fast"; three pictures: a posterior, a price range, the cost | — | — |
| 17:19–17:54 | The Team | The four presenters: photo, affiliation, background | — | The tutorial's webpage |
| 17:54–18:34 | The Rest of the Tutorial | "Each part adds one building block"; the timeline, and one row per part | — | The presenters |

Each main slide follows one reading order: the title and the key line are read by everyone, the figure is the
evidence, and a strip of at most three short label–value facts is optional; explanations go in the speaker notes
([`slides/slot1/README.md`](../slides/slot1/README.md)). The notes hold about 2,120 words: about 18 and a half
minutes at 115 a minute, the pace of the table above. The talk has 15: the slides after the opening are still to be
trimmed. The minute map pictured at the top of this page predates the new opening.

Thirteen backup frames follow: the coverage test; what the checks find; what an assumption does when one expiry is
quoted; two products and a book; the running example with its numbers; the example beside the notebook; least
squares against the exact posterior on the notebook's 63 quotes; the checks, parameter by parameter; the two slides
that opened the deck until 12 November (where uncertainty matters, and what Bayes offers); the prudent-valuation
rules; the methods that compute a posterior; and the references.

## The running example

The example keeps the model, the prior, the error law and the "market-like" parameters of the tutorial's Heston
notebook (`lab/notebooks/heston_spx_calibration.py`) and changes one thing, the chain. On the notebook's 63 SPX
quotes Heston's parameters are pinned to about 1% of the prior's range, and any price computed inside Heston has a
range too narrow to draw. The example's chain is thinner and quoted less tightly, as for a single stock: 7 strikes
on each of 1, 3 and 6 months and 1 and 2 years, with half-spreads about ten times those of the SPX quotes. It is
there to show what a posterior is and what it gives, not to claim an effect on SPX.

The product is an exotic, the option application's forward-start call: its strike is set at the money in 6 months
and it pays at 1 year. Slides 10, 11 and 12 use the 14 quotes at 6 months and 1 year, the product's own two dates;
slide 14 reads the same surface with five, four, three, two and one quoted expiries, always including 1 year. The
classical route is least squares with the delta method; the posterior is computed by importance sampling followed by
Metropolis moves, and was checked against independent chains.
[`slides/slot1/README.md`](../slides/slot1/README.md) gives the method, its checks and its limits.

## What the talk claims about Bayes, and what it does not

- **It argues from what the method gives**, not from a contest with least squares: stated assumptions, a
  distribution for any price, a picture of what the quotes determine, a check of the model.
- **Least squares is treated fairly.** It maximizes the same likelihood, it has assumptions too (normal errors,
  fixed weights, a flat prior), and with a quadratic approximation it also gives ranges. On the notebook's 63
  simulated quotes its 90% intervals match the exact posterior within 2% (a backup frame shows the table, and a test
  repeats it). Slide 14 shows where the two part: with two quoted expiries the quadratic region runs past the edge
  of the prior's box, and with one the fit sits on a bound and has no range.
- **The rules.** Prudent valuation asks for a value at 90% confidence (Regulation (EU) 2016/101, Article 9), and
  IFRS 13 for the sensitivity of Level 3 fair values. Banks meet this today with consensus data and expert ranges
  ([practice sources](../../course/practice-sources.md)); a posterior as the basis of a prudent value is a proposal.
- **The opening example is an illustration** on one window of US returns. The smaller move of the portfolio under
  the prior comes from the prior: with a flat prior the Bayesian weights are the plug-in weights. The deck makes no
  claim about which portfolio performs better out of sample
  ([`slides/slot1/README.md`](../slides/slot1/README.md), *The opening example*).
- **No claim about SPX.** The example is simulated, and the deck does not quote Heston's misfit on the SPX quotes.
- **No claim about hedging or value-at-risk.**

## Numbers on screen, and their status

`intro_figures.py` prints every number of the opening example, and `export_figures.py` every number of the running
example and of the recorded results. The recorded results
are in [`slides/slot1/recorded-results.json`](../slides/slot1/recorded-results.json), each with its source file,
commit and SHA-256; they are development results as recorded on 11 November 2026.

| Number | Where | Source | Status |
| --- | --- | --- | --- |
| Technology's average return raised by 10%, from 15.7% to 17.3% a year (0.37 of its standard error, 4.3): its weight goes from 74% to 95%, retail's from 31% to 18% | slide 1 | the opening example | illustration |
| The market's mean return 9.1% a year ± 3.5, volatility 15.9%: 83–84% in stocks at every horizon with the estimates taken as exact; 45% at 20 years with their uncertainty, and 58% at 10 years in the notes | slide 2 | the opening example, after Barberis (2000), Figure 1 | illustration |
| Technology's expected return: prior 8.9% ± 3.8, data 15.7% ± 4.3, posterior 11.9% ± 2.8 a year; with 90% probability between 7.2% and 16.5% | slide 3 | the opening example | illustration |
| Technology's weight: 74% to 95% with no prior; 39% to 48% under a prior worth 26 years of data (a weight of 56%); 12% with the prior taken as certain | slide 4 | the opening example; Jorion (1986) | illustration |
| The best fit misses the 6-month and 1-year quotes by 1.1 half-spreads; the dashed smile by 10 | slide 10 | the running example | illustration |
| On the notebook's 63 quotes, least squares and the exact posterior agree within 2% | slides 11 and 14 | this deck's least squares; the notebook's PyMC reference | a check, repeated by a test |
| The forward-start call: least squares 4.77, mark 4.70, 90% point 4.92, reserve 0.22 (5% of the mark); 4.66 under the generating parameters | slide 12 | the running example | illustration |
| The forward start's 90% range, delta method against posterior: 4.61–4.81 and 4.61–4.80 (five expiries), 4.56–4.82 and 4.56–4.81 (four), 4.51–4.94 and 4.51–4.90 (three), 4.31–5.23 and 4.39–5.00 (two); with one expiry no classical range, the posterior about 2.9–4.9 | slide 14 | the running example | illustration |
| One exact Heston posterior: 14–27 CPU-minutes, about 146,000 pricings by Fourier integral | slide 15 | the Heston notebook's PyMC references | recorded, measured; shown as orders of magnitude |
| Rough Bergomi: 17–29 CPU-hours by importance sampling, 100–180 CPU-days by MCMC, at 1–2 CPU-minutes per pricing | slide 15 | option application, Section 7.1, and the Heston run's pricing count | estimated, not run |
| The same Heston posterior from a trained network: 0.3 s | slide 15 | the run record | recorded |
| Coverage on 512 simulated surfaces; widths against exact posteriors | backup | the run records | recorded; see the README |

## Hard questions

1. **"With a flat prior this is just least squares."** At its peak, yes: least squares maximizes the same
   likelihood. The posterior keeps the other parameter sets too, and they matter where the quotes leave something
   open (slides 11 and 14).
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

Rachev, Hsu, Bagasheva and Fabozzi (2008), *Bayesian Methods in Finance*; Barberis (2000), *Journal of Finance*
55(1); Jorion (1986), *JFQA* 21(3); Black and Litterman (1992); Kenneth French's data library, for the returns of the
opening example; De Spiegeleer, Madan, Reyners and Schoutens (2018), SSRN 3191050; Schoutens, Simons and Tistaert (2004); Cranmer,
Brehmer and Louppe (2020); Radev et al. (2022); Talts et al. (2018); van der Vaart (1998), chapter 10; Kruse and
Nögel (2005) for the forward start; the course,
Lectures 6, 8, 14 and 16.4, and its [practice sources](../../course/practice-sources.md); the Heston SPX notebook,
`lab/notebooks/heston_spx_calibration.py`, with its run records, and the option application,
`lab/plans/option-application/` (results as of 11 November 2026).

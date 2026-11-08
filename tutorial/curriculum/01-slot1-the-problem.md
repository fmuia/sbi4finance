# Slot 1. The problem: Bayes for volatility models

Francesco · 15 minutes · the first of the opening Beamer talks. Part of the [curriculum](00-overview.md).

![Minute map of slot 1](images/slot-1.png)

Rebuilt on 7 November 2026 for the workshop's current format: Beamer talks for the introduction, amortized Bayes
and architectures, then the three application notebooks (Heston options, rough Bergomi options, return histories),
each run as model → simulate → train → infer → check. The talk is [`slides/slot1/slot1.tex`](../slides/slot1/slot1.tex),
compiled in the tutorial's Overleaf project ([`slides/slot1/README.md`](../slides/slot1/README.md)). The notebook
[`slot1_intro.py`](../notebooks/slot1_intro.py) remains as a supporting demo and as the source of the deck's figures;
it holds more than the talk shows.

The narrative follows the introduction of De Spiegeleer, Madan, Reyners and Schoutens (2018): desks calibrate,
price, hedge and measure risk every day, on thousands of options and underliers, with models priced by Fourier
integrals or Monte Carlo. To that problem the slot adds one fact the literature has measured: a calibration is one of
many that fit (Schoutens, Simons and Tistaert 2004). It then introduces the ingredients of the afternoon on
Black–Scholes and one close of SPY, argues for Bayes through what the quotes leave open, and ends on what exact
posteriors cost, amortized inference, the afternoon's map, the takeaways and the presenters.

## The slot in three sentences

**The question.** Which price do we mark for what the market does not quote, with what range, fast enough for every
name, every day?

**The answer.**

1. A calibration is an estimate: what the quotes leave open is decided by a prior, stated or not.
2. Bayes states that prior, carries it into every price, and checks the model against the data; where the quotes determine everything, it returns the classical answer.
3. Exact posteriors do not scale to a book or a validation. Amortized inference learns them once from simulations, and is checked like any learned surrogate.

## What the audience leaves with

- The ingredients every later talk and notebook uses: SPY's chain, the 63 prices of the fixed grid, the tolerance of each price, the model as a map from parameters to prices, the likelihood, the prior, the posterior, a product's range, and the predictive check.
- The habits of calibration read as priors: least squares (flat), a penalty toward yesterday's value (normal), a fixed parameter (zero width), a flat curve beyond the last quote and a proxy for a thin name (zero width).
- Where Bayes changes the answer (inputs the quotes leave open) and where it does not (a liquid, well-identified smile, C11).
- That the rules asking for a range are met today, with consensus marks and expert ranges, and what Bayes adds to them: a range derived from the data, with its assumptions stated, that can be tested.
- What exact posteriors cost, why a book and a validation need many of them, and the idea of amortized inference.
- The afternoon: two more talks and three notebooks, with their data, models and checks.

## Slide by slide

| Min | Slide | Key line (read by everyone); figure; optional facts | Source and status |
| --- | --- | --- | --- |
| 0:00–1:30 | The Problem | "The calibrated parameters are uncertain, and so are the prices computed from them"; the desk diagram; facts: daily, for many underlyings; Fourier or Monte Carlo pricing; seven models with similar fits and different exotic prices | De Spiegeleer et al. 2018; Schoutens et al. 2004 |
| 1:30–2:30 | The Data | "The data are 63 SPY option prices, each with an error band"; SPY's smile; facts: date, grid, error band | S8, verified |
| 2:30–3:30 | The Likelihood | "The likelihood measures how well each value of σ reproduces the observed price"; one-price figure; facts: Black–Scholes and one price, 12.84% ± 0.30, the formula | By construction |
| 3:30–4:30 | Prior and Posterior | "The posterior combines the prior with the likelihood. The prior matters only where the data are weak"; the prior sweep; facts: least squares, penalties and fixed parameters as priors | By construction |
| 4:30–5:30 | Pricing Under the Posterior | "Pricing a product under each posterior draw gives the distribution of its price"; the put's distribution; facts: the put, its 90% point (43% against 37% linearized), prudent valuation | By construction; Reg. (EU) 2016/101 |
| 5:30–6:30 | Checking the Model | "A posterior predictive check compares simulated quotes with the observed ones. Black–Scholes fails it on SPY"; the predictive check; facts: 9.5 sd, the smile, the models used next | By construction |
| 6:30–8:30 | Beyond the Quotes | "Some quantities are not determined by the quotes. Their uncertainty is set by the prior"; three Heston fits with the same short end; facts: the example, 21.3% to 25.3% at two years, fixing gives too narrow a range | Formula; quick checks in the backup |
| 8:30–9:45 | The Cost of Exact Posteriors | "An exact posterior requires about 10,000 model evaluations. A book or a validation requires thousands of posteriors"; three numbers | C8, verified |
| 9:45–11:30 | Amortized Inference | "A network is trained once on simulated data. It then returns posteriors for new data in seconds"; the pipeline; facts: 1.4–3.6 s, the coverage test, valid within the prior | C8; laptop benchmark; Lecture 14.4 |
| 11:30–12:45 | This Afternoon | "The three notebooks follow the same five steps"; the steps and the three notebooks; facts: the next two talks, synthetic worked cases | Main README and notebooks |
| 12:45–13:45 | Summary | Three statements; next, Amortized Bayes | — |
| 13:45–14:30 | Presenters | Names, affiliations (to fill), who presents what | To confirm |

Each main slide follows one reading order: the title and the key line are read by everyone, the figure is the
evidence, and a strip of at most three short label–value facts is optional; explanations go in the speaker notes ([`slides/slot1/README.md`](../slides/slot1/README.md)).

Twelve backup frames follow: SPY's chain and its cleaning, how a grid price is read, the tolerance of a price, what a
price says about volatility, level, skew and curvature, what a network reads, the variance curve, what the quotes leave open and
what practice does (the table behind slide 7), the evidence in detail, where the posterior's
shape matters, the methods that compute a posterior, and the references.

## Why Bayes, and what the slot does not claim

**The rules.** Prudent valuation asks for the 90% point of a range of plausible values (Regulation (EU) 2016/101,
Articles 9 and 11); IFRS 13 §93(h) and model-risk guidance ask for sensitivities and an understanding of parameter
uncertainty. Banks meet these today: consensus services publish 10th and 90th percentile marks, including for
autocallables and cliquets (Totem), and where no consensus exists the regulation allows expert-based ranges, proxies
and alternative calibrations ([practice sources](../../course/practice-sources.md)). The rules are therefore not by
themselves a reason for Bayes. What Bayes adds is quality where consensus does not reach: a range derived from the
data, with its prior stated, that can be tested by coverage. The slot says this on the range slide and in the notes.

**The prior, not the shape.** Where the course measured it, on simulated thin chains, a Gaussian at the mode with the
same prior gave the same product ranges as the posterior (Lecture 16, Section 16.4). The argument is therefore the
prior: each classical habit fixes what the quotes leave open with a prior of zero width, or leaves it with none, and
both fail in measurable ways (slide 7 and its backup). Cases where the shape itself matters exist and are kept in the backup: a price
convex in a parameter, a valley cut by bounds (Heston below a vol of vol of 0.3), a bent ridge (the rough-Bergomi long
end with the curve inferred), and two optima (Heston with jumps).

**Downstream uses.** The talk names marking and reserving products that are not quoted, with published evidence
(Schoutens, Simons and Tistaert 2004) and the course's quick checks. The repository has no result on hedging under
parameter uncertainty or on value-at-risk from return histories; the talk does not claim one. The trajectory
notebook's "risk" screens are statistical estimation-risk checks, not financial risk.

## Numbers on screen, and their status

| Number | Source | Status |
| --- | --- | --- |
| 13,114 options on 35 expirations; 5,358 quotes kept on 32 | S8, as quoted in the course, Lectures 1.6 and 15.3 | verified |
| 63 prices on 7 maturities × 9 moneyness points; 35 read from two expirations, 28 from one, none extrapolated | `spy/surface.json` (S8) | verified |
| Tolerance $s(\delta) = 0.3 + 0.35\lvert\delta\rvert$ points: ±0.3 at the money, 1 point at $\delta = \pm 2$; one volatility point is 11.5 bp of the forward (about \$0.89) at one month | computed live from `spy/surface.json` | verified by construction |
| $\sigma$ = 12.84% ± 0.30 points from the 1-month at-the-money price | computed live | verified by construction |
| Normal priors centred on 20%: ±2 points give 12.99%, ±0.3 give 16.4%, ±0.1 give 19.3% | computed live | verified by construction |
| The put at $\delta = -2$: a sold position's reserve 43% of the mark under the posterior, 37% by the delta method | computed live | verified by construction |
| The market's put quote 9.5 sd from the posterior predictive | computed live | verified by construction |
| $H$ fixed at 0.10: the truth above the 90% bound on 23% of 100 thin chains (variance swap and put); without a prior, 76 of 100 fits on a bound and bands 1.5–3.3 times the posterior's; a Laplace approximation with the prior within 5% of the posterior's widths | `lab/checks/why_bayes/` (course, Lecture 16.4) | quick check |
| SPY's long end: flat extrapolation gives 0.32, 0.30 and 0 of the posterior's 90% range (2-to-3-year forward start, 5-year put, 3-year variance swap); the forward start's range 19.4% to 36.8% of its price as the prior's sd goes from 0.1 to 0.3 | `lab/checks/long_end/` (Lecture 15.6) | quick check |
| A sector of 2 liquid and 6 illiquid names: 90% intervals cover 59.5% by proxy, 88.0% under hierarchical Bayes | `lab/checks/bs_hierarchy.py` (Lecture 8.10) | quick check; Black–Scholes, model right by construction |
| Heston, 200 smiles: no Gauss–Newton interval on 11; Gauss–Newton coverage 87–92% against the posterior's 87–90% | C11, `heston/s6/classical.json` | verified |
| Heston 20 ms per evaluation and about 4 CPU-min per posterior; rough Bergomi 5.0 CPU-s and at least 17 CPU-h by the chain; network 3.6 CPU-s for 1,000 draws | `costs/costs.json` (C8) | verified |
| Rough Bergomi by importance sampling from a Gauss–Newton proposal on SPY: 1,024 evaluations, about 85 CPU-min | `lab/checks/rough_spy_is.py` | verified |
| About $10^4$ evaluations per exact posterior | C8: 251 CPU-s at 20 ms; 17 CPU-h at 5.0 CPU-s | as C8 |
| A coverage test on 2,000 smiles: about 2 CPU-h with the network after a bank of about 370 CPU-h, about 2,800 by importance sampling | Lecture 14.4, from C8 and S1 | as C8; the bank at the pilot's rate |
| 1,000 posterior draws in 1.4–3.6 s | [laptop benchmark](../../lab/evidence/pilot-results/2026-10-29/laptop-inference-benchmark/README.md) | measured; one-case latency on one machine |
| Same short end: $\kappa$ = 1, 2 and 4 match the 1-, 3- and 6-month levels within 0.16 points; at two years 25.3%, 23.3% and 21.3% | `slides/slot1/export_figures.py`, the Heston formula | by construction, illustrative levels |
| Shape cases (backup): $\sigma_v$ interval 1.48 times the posterior's on 28 smiles, 6 with two optima; a range 0.37 times Gauss–Newton's, the flat curve 21% low and the 2-year put 7.4% low; a variance swap from −18% to +67% | Lectures 11.6, 15.6 and 6.4 | quick checks; the last a study of 9 October, not a registered claim |

## What may be said, and what may not

| Say | Do not say | Why |
| --- | --- | --- |
| "Each price is known only to within an error band: 0.3 volatility points at the money, the tutorial's assumption." | "The quote's bid–ask is 0.3 points." | The rule is the design's floor, wider than SPY's half-spreads at every grid point (S8). |
| "The 63 prices are read from SPY's quotes by smoothing and interpolation." | "The 63 prices are quoted." | Each is read from a spline through an expiration's implied volatilities (S8). |
| "The rules are met today, with consensus marks and expert ranges; Bayes gives a range derived from the data, with its assumptions stated, that can be tested." | "Banks cannot meet the rules without Bayes." | Totem publishes 10th and 90th percentiles; Art. 3(3)(a) and 11(4) allow proxies and expert-based ranges. |
| "Where we measured, the prior made the difference, not the posterior's shape." | "Classical methods miss the posterior's shape." | Lecture 16, Section 16.4: a Laplace approximation with the same prior matches the posterior on thin chains. |
| "Fixing $H$ at 0.10 is our analogy to fixing $\kappa$ or $\beta$." | "Desks fix $H$ at 0.10." | Fixing $\kappa$ and $\beta$ is documented; fixing $H$ is not ([practice sources](../../course/practice-sources.md)). |
| "On a liquid, well-identified smile the classical route covers at its nominal rate." | "Gauss–Newton is wrong." | C11: 87–92% on 200 Heston smiles. |
| "These are quick checks on simulated data." | Slide 7's numbers as accepted results. | S10, S11 and J3 are pending. |
| "SPY's skew falls roughly like $T^{-0.32}$." | "So $H$ is about 0.18." | Under rough Bergomi with a large $\eta$ the fitted exponent is steeper than $H - \tfrac12$. |
| "At least 17 CPU-hours by a random-walk chain, or about 85 CPU-minutes by importance sampling." | "17 CPU-hours", or the chain's cost alone. | The chain's cost is a lower bound; importance sampling targets the same posterior. |
| "1,000 draws in a second or a few." | "Milliseconds." | 1.4–3.6 s measured. |
| "The reserve is a 90% quantile, like a VaR, of today's valuation." | "The reserve is a VaR." | VaR is a quantile of P&L over a horizon. |
| "Art. 9 for the calibrated parameters, Art. 11 for the choice of model." | "Art. 11 asks for the range of calibrations." | Art. 11(1) excludes calibrations from market-derived parameters, which Art. 9 covers. |

## Hard questions

1. **"With a flat prior, one quote is just least squares."** Yes, and the slide says so: least squares is the posterior's peak under a flat prior. The posterior differs where the quotes leave something open (slide 7).
2. **"Black–Scholes is a straw man."** It is the vehicle for the ingredients, with one parameter on one axis, and the predictive check rejects it on stage; the afternoon's models produce a smile.
3. **"Banks already meet the rules."** They do, with consensus marks, expert ranges, proxies and alternative calibrations. Bayes adds a range that comes from the data with a stated prior, and that can be tested by coverage; on thin names and unidentified inputs the classical habits are biased (a fixed input) or inflated (no prior).
4. **"So the prior is doing the work."** Where the quotes are silent, yes, and it must: the classical habits are priors of zero width. Bayes states the width and tests it; on SPY's long end the forward start's range moves from 19.4% to 36.8% of its price as the prior's width triples.
5. **"Does the posterior's shape matter?"** Rarely, in what the course measured. The backup lists the cases where it does.
6. **"Your evidence is quick checks."** Yes, and labelled so; the full studies (S10, S11, J3) are pending. The liquid case, where the classical route works, is a verified study (C11).
7. **"Why a fixed tolerance rule and not each day's bid–ask?"** Real and simulated smiles must carry the same noise, or a network trained on simulations would read real quotes as a different kind of data. The rule is wider than SPY's half-spread at every grid point.
8. **"Why a grid, and not the chain's own quotes?"** The grid gives real and simulated smiles the same 63 inputs, so one network serves both. Set networks (Architectures) would read a chain directly.
9. **"Heston calibrates in a second; why a network?"** One exact Heston posterior is affordable, and the Heston notebook computes it as the reference. A validation needs thousands of posteriors, and a book one per name; rough Bergomi has no fast pricer.
10. **"Why not our own model?"** The method does not depend on the model: draw parameters from the posterior, price the product under each draw with the bank's own pricer, and read the 90% point.

## The supporting notebook

`slot1_intro.py` runs headless under the current runtime (76 cells, no errors, about 8 seconds on 7 November 2026)
and loads no network. Its sections follow the earlier version of this talk: the data, Bayes with one parameter, the
range of a price, checking the model, Heston on SPY with the quick checks of `lab/checks/heston_checks.py`, and the
costs. The deck shows only its first four sections; the Heston quick checks stay in the notebook, with their status
labels, and are not on any slide. `slides/slot1/export_figures.py` runs it to produce the deck's figures.

## What is left out, and why

- **Heston's product ranges, reserves, tails and curved posteriors on SPY, the second-model gap and the money cards.** Quick checks and pilot results; they belong to the application notebooks, with their own status.
- **The frequentist reading of the error bar, more quotes narrowing the posterior, and the noise multiplied by 4.8.** Not needed for the argument in 15 minutes; in the notebook.
- **The defence of the noise rule.** In the backup only; on the main slides the tolerance is stated once.
- **Hierarchies across names as a section** (agreed on 29 October): one row of slide 7 and the backup.
- **Hedging and value-at-risk.** Nothing in the repository supports a claim.

## To build before the run-through

1. Fill the presenters' affiliations, confirm the assignments of 24 October, and name who presents the return histories.
2. Compile in the Overleaf project with the real theme and check that no frame overflows; the deck has been compiled here only against a stand-in for the theme.
3. If S10, S11 or J3 land before the tutorial, replace slide 7's quick checks with their results.
4. Rehearse with a timer, twice.

## Sources

De Spiegeleer, Madan, Reyners and Schoutens (2018), SSRN 3191050; Schoutens, Simons and Tistaert (2004); the
course, Lectures 1, 4 to 8, 11.6, 14, 15.6 and 16.4, and its [practice sources](../../course/practice-sources.md);
[CLAIMS.md](../CLAIMS.md) for C1, C8 and C11.

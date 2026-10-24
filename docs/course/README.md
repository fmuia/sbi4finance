# Amortized Bayesian calibration of volatility models: a lecture series

Version 2, 24 October 2026 · AI & Analytics Solutions SAS di Muia Francesco

This folder is a course in sixteen lectures. It builds, from the basics, everything needed to understand and teach
the advanced example of the ICAIF '26 tutorial: the calibration of a rough volatility model to option quotes, with
a posterior, and the propagation of that posterior into the prices of products the quotes do not determine.

The course is a superset of the tutorial and of the [calibration design document](../design/calibration-design-document.md).
[The version-2 plan](v2-plan.md) says what the course argues and why, and which options would strengthen it; [the outline](v2-outline.md) says where each
section of version 1 went; [practice-sources.md](practice-sources.md) lists the practices the course may describe,
with their sources; [status.md](status.md) gives the state of every study. Version 1 (fourteen lectures) is at
commit 19a6ea4.

## The argument of the course

The course makes one argument in four claims. [Lecture 6, Section 6.8](06-marking-a-product-the-smile-does-not-quote.md#68-the-argument-of-the-course)
states it, each lecture's opening says which part it serves, and
[Lecture 16, Section 16.8](16-a-thin-chain-ewg.md#168-the-argument-claim-by-claim) gathers the evidence and says
what the pending runs decide.

1. **Where the quotes determine what a product depends on, the classical answer is right**, and the posterior
   returns it (Lectures 7, 8, 11 and 15).
2. **Where they do not, the two answers can part.** The classical route fixes the open input and gives it no range,
   or fits it along a weakly informed direction where its linear range can be wrong. The posterior gives the input a
   stated prior and carries it into the range. Three cases: beyond the last quote (Lecture 15), a thin chain
   (Lecture 16), and a name that quotes almost nothing, in a sector of names that quote more (Lecture 8).
3. **Simulation-based inference makes the posterior checkable** for a model priced only by simulation. One
   posterior is affordable without it when a classical fit gives a good proposal. A check of coverage on thousands
   of simulated smiles is not: for 2,000 smiles, about 2,800 CPU-hours by importance sampling, against about 2 with a
   network after a training bank of about 370 (Lectures 9 and 14).
4. **The posterior is exact for its model, its prior and its noise model, and for nothing else**, so each is stated
   and tested (Lectures 8, 15 and 16).

Today quick checks show claim 1 on SPY and claim 2 beyond SPY's last quote, the latter with a Laplace
approximation and a size set by the prior. A simulated sector shows claim 2 for pooling, and pilot smiles show part of
claim 4. The thin chain, the network's own checks, and the network as a proposal where a classical fit fails, wait
for the runs of [`tutorial_final/CLUSTER-RUNS.md`](../../tutorial_final/CLUSTER-RUNS.md).

## Who the course is for

The reader knows probability (random variables, densities, conditional expectations, the normal distribution),
calculus and some Python. The reader is not assumed to know finance, option pricing, stochastic calculus, rough
volatility or neural networks. Each of these is introduced from zero and developed to the level a quantitative
practitioner works at.

## How the course is organized

Each part needs the one before it. Black–Scholes carries the simple concepts, Heston the BayesFlow workflow, and
rough Bergomi the full examples.

| Lecture | Title | What it establishes |
| --- | --- | --- |
| **Part I** | **Markets and the first model** | |
| 1 | [Options, quotes and volatility](01-options-quotes-and-volatility.md) | What an option is, what a market quotes on a liquid and a thin chain, and what volatility means in data |
| 2 | [No-arbitrage without a model](02-no-arbitrage-without-a-model.md) | Discounting, the forward, put–call parity, forward units, the bounds on option prices, American exercise and dividends |
| 3 | [Brownian motion, SDEs and Itô's lemma](03-brownian-motion-and-ito.md) | The mathematics of continuous-time randomness; why volatility is measurable and the drift is not |
| 4 | [The Black–Scholes model](04-black-scholes.md) | Hedging, the risk-neutral measure, the pricing formula, the sensitivities, a term structure and the forward start |
| 5 | [Implied volatility and the smile](05-implied-volatility-and-the-smile.md) | How quotes are read, the smile and what it says, the design grid, and the variance swap |
| **Part II** | **Calibration and Bayes on Black–Scholes** | |
| 6 | [Marking a product the smile does not quote](06-marking-a-product-the-smile-does-not-quote.md) | The marking procedure, the products, one smile with several calibrations, what the rules ask for, the baselines that fix what the quotes do not, and the argument of the course |
| 7 | [Calibration as estimation](07-calibration-as-estimation.md) | The observation model, the likelihood, least squares, the Gauss–Newton interval and where it fails, multi-start and regularization |
| 8 | [Bayesian calibration](08-bayesian-calibration.md) | The prior and the posterior, identification, credible against confidence intervals, propagation to products, Metropolis–Hastings, importance sampling, pooling across names, where the posterior and the classical answer part |
| 9 | [Amortized inference](09-amortized-inference.md) | Why simulated pairs carry the posterior, neural posterior estimation, flow matching, simulation-based calibration and contraction |
| **Part III** | **Heston and the BayesFlow workflow** | |
| 10 | [The Heston model](10-heston.md) | A random variance correlated with the price, Fourier pricing, how each parameter shapes the smile, Heston's products |
| 11 | [Calibrating Heston](11-calibrating-heston.md) | Least squares on SPY, the profile, the exact posterior, and why on SPY it agrees with the classical interval |
| 12 | [The amortized workflow on Heston](12-the-amortized-workflow-on-heston.md) | The BayesFlow code, the bank, every check against an exact posterior, the audit, the predictive check, changing the prior |
| **Part IV** | **Rough Bergomi: the full examples** | |
| 13 | [The rough Bergomi model](13-rough-bergomi.md) | Roughness, the Volterra driver, the forward-variance curve, the hybrid scheme, the power-law skew, what the smile identifies |
| 14 | [Why the exact posterior is expensive](14-why-the-exact-posterior-is-expensive.md) | The Monte Carlo likelihood, the cost of an exact chain and of importance sampling, and who needs many posteriors |
| 15 | [A liquid smile: SPY](15-a-liquid-smile-spy.md) | The rough network, its checks, SPY's posterior, where it agrees with Gauss–Newton, products, the long end |
| 16 | [A thin chain: EWG](16-a-thin-chain-ewg.md) | Why a thin chain and which one, the procedure on EWG, coverage on simulated chains, EWG's posterior, model error, and the argument claim by claim |

## Conventions

- **Units.** Time is in years. Real maturities are trading days over 252. SPY's surface of 21 October 2026 counts
  weekdays (exchange holidays not removed), as version 1 did; the thin chains count NYSE trading days, which makes
  a calendar year about 252 days (Lecture 16). Volatility is annualized and written as a fraction (0.20) or a
  percentage (20%). "One volatility point" is 0.01.
- **Forward units.** From Lecture 2 on, option prices are divided by the discount factor and by the forward, so
  that the forward is 1 and a price is a fraction of it.
- **Notation.** The symbols follow the design document. Each lecture defines a symbol when it first appears, and
  the [design document's symbols table](../design/calibration-design-document.md#symbols) collects them.
- **Data.** The daily returns of VOO from 3 January 2020 to 4 September 2026
  (`data/SPMO_VOO_2020-01-01_2026-09-04.csv`); SPY's surface of 21 October 2026 on the design grid
  (`tutorial_final/results/calibration/spy/surface.json`); four thin chains of 22 and 23 October 2026, derived data
  only (`tutorial_final/checks/thin_chains/`). Raw quote files are not committed; `data/SOURCES.md` gives their
  URLs.
- **Numbers that need further runs.** A number written as ⟨Sn⟩ is left blank on purpose. Study Sn produces it;
  [`tutorial_final/CLUSTER-RUNS.md`](../../tutorial_final/CLUSTER-RUNS.md) says how.
- **Figures.** Every figure is drawn by a script in `scripts/course/` (named by version 1's lecture numbers) or by
  a check in `tutorial_final/checks/`, from closed-form formulas, simulations with fixed seeds, or the data above.
  Each figure has a paragraph that says how to read it.

## Where each section of the design document is covered

| Design-document section | Lecture |
| --- | --- |
| Executive summary, steps 1–7 | 1, 6 (step 1); 4, 5, 10, 13 (step 2); 8 (step 3); 14 (step 4); 7 (step 5); 9, 12, 15 (step 6); 12, 15 (step 7) |
| The two claims; Summary; Three routes from one smile to a product price | 14, 15 |
| Why Heston is the check and not the example | 10, 11, 14 |
| The practitioner problem | 1 (exotics and marking), 6 |
| Executive summary of the model (inputs and outputs, data, what it captures and misses, the output for SPY) | 6, 15; the data in 1, 5, 7 and 12 |
| Notation and key terms: four volatility objects | 1 (realized), 5 (implied, forward variance), 13 (instantaneous) |
| The model: Black–Scholes and the smile | 4, 5 |
| The model: Heston | 10 |
| The model: rough Bergomi; the prior; the prior predictive check; the 63 model prices; the skew's term structure | 13 (model, pricing, skew), 8 (prior), 12 and 15 (prior predictive check) |
| The model: from parameters to a product price | 6, 8 |
| The observation model (quotes, design grid, noise model, identification) | 5 (design grid), 7 (quote and noise), 8 and 13 (identification) |
| Why the likelihood is expensive | 14 |
| Classical calibration and where it fails | 7, 11 |
| Amortized design | 9, 12, 15 |
| Validation plan | 9, 12, 15, 16 |
| From posterior to prices | 8 (method), 15 and 16 (results) |
| Planned feasibility study (S1–S9) | [status.md](status.md), with each study cited where its subject is taught |
| Tutorial narrative; risks and open decisions | 16, `tutorial_final/curriculum/` |
| Appendix A. Calibration practice and the rules | 6 |
| Appendix B. Black–Scholes, implied volatility, variance swap | 2, 4, 5 |
| Appendix C. Heston and the Fourier pricer | 10 |
| Appendix D. Rough Bergomi | 13 |
| Appendix E. Calibration as estimation | 7 |
| Appendix F. Amortized inference | 9, 12 |
| Appendix G. Validation | 9, 12 |
| Appendix H. Products under the posterior | 1 (payoffs), 6 (definitions), 8 (method), 15 (results) |
| Appendix I. The real surface | 2 (forward and discount), 5 (design grid, forward-variance curve), 15 |
| Appendix J. Fourier pricing, Heston against rough Bergomi | 10, 13, 15 |
| References | Each lecture lists the references it uses |

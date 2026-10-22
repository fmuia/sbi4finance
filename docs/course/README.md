# Amortized Bayesian calibration of volatility models: a lecture series

Oct 22, 2026 · AI & Analytics Solutions SAS di Muia Francesco

This folder is a course in fourteen lectures. It builds, from the basics, everything needed to understand and teach the advanced example of the ICAIF '26 tutorial: the calibration of a rough volatility model to an option smile, with a posterior, and the propagation of that posterior into the prices of products the smile does not pin down.

The course is a superset of the [calibration design document](../design/calibration-design-document.md). The design document states the method and the planned evidence for the organizers. The course explains every concept the design document uses, from what an option is to how an amortized posterior is validated, and adds the derivations, figures, limits and interpretations that a first reading needs. The design document is unchanged; the [map at the end of this page](#where-each-section-of-the-design-document-is-covered) says where each of its sections is covered here.

## Who the course is for

The reader knows probability (random variables, densities, conditional expectations, the normal distribution), calculus and some Python. The reader is not assumed to know finance, option pricing, stochastic calculus, rough volatility or neural networks. Each of these is introduced from zero and developed to the level a quantitative practitioner works at.

## How the course is organized

The lectures follow one line of argument in four parts. Each part needs the one before it.

1. **Part I builds the models.** It explains what is traded and quoted, what can be said without a model, the mathematics of continuous-time randomness, and then three models of increasing realism: Black–Scholes, Heston and rough Bergomi.
2. **Part II explains why the problem matters.** It describes what a derivatives desk does every morning with these models, where the calibration is uncertain, and what the regulations ask for.
3. **Part III explains what Bayesian inference adds.** It treats calibration as statistical inference, first by least squares and then by the posterior, and shows on simple models what the posterior reveals that a point estimate hides.
4. **Part IV explains how to make the posterior affordable.** It shows why the exact posterior of rough Bergomi costs hours per surface, how simulation-based amortized inference reduces that to seconds, and how the result is validated.

| Lecture | Title | What it establishes | Status |
| --- | --- | --- | --- |
| **Part I** | **The models** | | |
| 1 | [Options, quotes and volatility](01-options-quotes-and-volatility.md) | What an option is, what a market quotes, what a desk must price, and what volatility means in data | Written |
| 2 | [No-arbitrage without a model](02-no-arbitrage-without-a-model.md) | Discounting, the forward, put–call parity, forward units, the bounds and shape constraints on option prices, and the link between prices and the distribution of the future price | Written |
| 3 | [Brownian motion, stochastic differential equations and Itô's lemma](03-brownian-motion-and-ito.md) | The mathematics of continuous-time randomness, why volatility is measurable and the drift is not | Written |
| 4 | [The Black–Scholes model](04-black-scholes.md) | Hedging, the disappearance of the drift, the risk-neutral measure, the pricing formula, the sensitivities, and the model's limits | Written |
| 5 | [Implied volatility and the smile](05-implied-volatility-and-the-smile.md) | How quotes are read, what the market's smile looks like, what it says about the distribution of returns, and the variance swap | Written |
| 6 | [Stochastic volatility: the Heston model](06-heston.md) | A random variance correlated with the price, the characteristic function, Fourier pricing, how each parameter shapes the smile, a fit to SPY, and where Heston's smile has the wrong shape | Written |
| 7 | [Rough volatility: the rough Bergomi model](07-rough-bergomi.md) | Evidence of roughness, the Volterra driver, the forward-variance curve as an input, the hybrid scheme and the Monte Carlo pricer, the power-law skew, and what the smile identifies | Written |
| **Part II** | **Why the problem matters** | | |
| 8 | [Calibration on a derivatives desk](08-calibration-on-a-desk.md) | Daily calibration, the products the smile does not pin down (variance swap, volatility swap, forward start, cliquet), one model with several calibrations, the literature on calibration risk, and what the rules ask for | Written |
| **Part III** | **What Bayesian inference adds** | | |
| 9 | [Calibration as estimation](09-calibration-as-estimation.md) | The observation model of a quote, the likelihood, least squares as maximum likelihood, the Gauss–Newton interval and where it fails, multi-start, regularization and deep calibration | Written |
| 10 | [Bayesian calibration](10-bayesian-calibration.md) | The prior and the posterior, Black–Scholes on SPY as a misspecified model, identification, credible against confidence intervals, Metropolis–Hastings, a Heston posterior, and propagation to product prices and reserves | Written |
| **Part IV** | **Making the posterior affordable** | | |
| 11 | [Why the exact posterior is expensive](11-why-the-exact-posterior-is-expensive.md) | The rough-Bergomi likelihood as a Monte Carlo run, the effect of Monte Carlo error on it, the cost of an exact chain, and the alternatives | Written |
| 12 | [Amortized inference](12-amortized-inference.md) | Why simulated pairs carry the posterior, neural posterior estimation, flow matching and its sampler, the tutorial's network and training bank, and the neural-pricer alternative | Written |
| 13 | [Validating an amortized posterior](13-validating-an-amortized-posterior.md) | The prior predictive check, simulation-based calibration, contraction and coverage, agreement with exact posteriors, the importance-sampling audit, and the posterior predictive check | Written |
| 14 | [The study and the tutorial](14-the-study-and-the-tutorial.md) | The two claims, the nine feasibility studies and where they stand, the costs, the real SPY smile, products and reserves, and the tutorial's narrative and risks | Written |

## Conventions

- **Units.** Time is in years. Real maturities are counted in business days (weekdays; exchange holidays are not removed) and divided by 252, which is the repository's convention: the "one-year" design maturity of 252 business days from 21 October 2026 ends on 8 October 2027, 352 calendar days later. Lecture 2, Section 2.3 shows the one place where this choice visibly changes a number. Volatility is annualized and written as a fraction (0.20) or a percentage (20%). "One volatility point" is 0.01.
- **Forward units.** From Lecture 2 on, option prices are divided by the discount factor and by the forward, so that the forward is 1 and a price is a fraction of it. Lecture 2 explains why this removes interest rates and dividends from every formula.
- **Notation.** The symbols follow the design document. Each lecture defines a symbol when it first appears, and the [design document's symbols table](../design/calibration-design-document.md#symbols) collects them.
- **Data.** Real data come from two files in the repository: the daily returns of VOO, an S&P 500 index fund, from 3 January 2020 to 4 September 2026 (`data/SPMO_VOO_2020-01-01_2026-09-04.csv`), and the SPY option surface of 21 October 2026 on the tutorial's design grid (`tutorial_final/results/calibration/spy/surface.json`, study S8 of the design document).
- **Numbers that need further runs.** A number written as ⟨Sn⟩ is left blank on purpose. It is produced by study Sn of the design document's planned feasibility study, and it is filled in when that study has run.
- **Figures.** Every figure is drawn by a script in `scripts/course/`, one per lecture, from closed-form formulas, simulations with fixed seeds, or the data files above. Each figure has a paragraph that says how to read it.

## Where each section of the design document is covered

The table lists every main section of the design document and the lecture that covers it.

| Design-document section | Lecture |
| --- | --- |
| Executive summary, steps 1–7 | 1, 8 (step 1); 4–7 (step 2); 10 (step 3); 11 (step 4); 9 (step 5); 12 (step 6); 13 (step 7) |
| The two claims; Summary; Three routes from one smile to a product price | 14 |
| Why Heston is the check and not the example | 6, 7, 11 |
| The practitioner problem | 1 (exotics and marking), 8 |
| Executive summary of the model (inputs and outputs, data, what it captures and misses, the daily run, the output for SPY) | 8, 14; the data subsection in 1, 5, 9 and 12 |
| Notation and key terms: four volatility objects | 1 (realized), 5 (implied, forward variance), 7 (instantaneous) |
| Notation and key terms: symbols; statistical vocabulary | Defined where first used across 1–14 |
| The model: Black–Scholes and the smile | 4, 5 |
| The model: Heston | 6 |
| The model: rough Bergomi; the prior; prior predictive check; computing the 63 model prices; the example surface; the skew's term structure | 7 (model, pricing, skew), 10 (prior), 13 (prior predictive check) |
| The model: from parameters to a product price | 8 |
| The observation model (quotes, design grid, noise model, identification) | 5 (design grid), 9 (quote and noise), 7 and 10 (identification) |
| Why the likelihood is expensive | 11 |
| Classical calibration and where it fails | 9 |
| Amortized design | 12 |
| Validation plan | 13 |
| From posterior to prices | 10 (method), 14 (results) |
| Planned feasibility study (S1–S9) | 14, with each study also cited where its subject is taught |
| Tutorial narrative; risks and open decisions | 14 |
| Appendix A. Calibration practice and the rules | 8 |
| Appendix B. Black–Scholes, implied volatility, variance swap | 2 (forward, discount, parity), 4 (formulas in forward units), 5 (implied volatility, log contract, variance swap) |
| Appendix C. Heston and the Fourier pricer | 6 |
| Appendix D. Rough Bergomi | 7 |
| Appendix E. Calibration as estimation | 9 |
| Appendix F. Amortized inference | 12 |
| Appendix G. Validation | 13 |
| Appendix H. Products under the posterior | 1 (payoffs), 8 (prices), 10 (reserve) |
| Appendix I. The real surface | 2 (forward and discount), 5 (design grid, forward-variance curve), 14 (use in the tutorial) |
| Appendix J. Fourier pricing, Heston against rough Bergomi | 6, 7 |
| References | Each lecture lists the references it uses |

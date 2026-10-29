# Amortized Bayesian calibration of volatility models: a lecture series

Version 2, 24 October 2026 · AI & Analytics Solutions SAS di Muia Francesco

This folder is a course in sixteen lectures. It builds, from the basics, everything needed to understand and teach
the advanced example of the ICAIF '26 tutorial: the calibration of a rough volatility model to option quotes, with
a posterior, and the propagation of that posterior into the prices of products the quotes do not determine.

The course is a superset of the tutorial.
[The plan](v2-plan.md) says what the course argues and why, and which options would strengthen it;
[practice-sources.md](practice-sources.md) lists the practices the course may describe, with their sources;
[status.md](status.md) gives the state of every study.

The proposed [numerical experiment plan](../lab/plans/NUMERICAL-EXPERIMENT-PLAN.md) covers the next data, training and reference campaign. The [tutorial review](../tutorial/curriculum/TUTORIAL-REVIEW.md) assesses how that evidence supports the live learning path.

## The argument of the course

**The message.** To price a product the quotes do not determine, a calibration must assume something about what they
leave open: a parameter the smile does not identify, the forward-variance curve beyond the last quote, the curve
itself on a thin chain. The classical route makes that assumption silently. It fixes the open input, which is a prior
of zero width, or fits it with no prior at all, which the quotes then cannot pin down. Bayesian calibration states the
assumption as a prior with a width, carries it into every product's range by one rule, and lets it be tested; where
the quotes determine everything, it returns the classical answer. Rough Bergomi, the course's model for the smile's
term structure, is priced only by simulation, and its exact posterior costs hours per chain. Simulation-based
inference makes that posterior affordable on every chain and checkable on thousands of simulated ones.

**What the course sets out to show.** Two things. Where, why and by how much the posterior's range differs from the
classical one, on the models and the quotes a bank uses. And that for a simulation-priced model the posterior can be
computed and checked at the scale a bank needs. The quick checks so far sharpen the first: where the two differ on
products, the stated prior makes the difference more than the posterior's shape (Lecture 16, Section 16.4).

| Part | What it adds to the argument |
| --- | --- |
| I. Markets and the first model (1–5) | What the quotes determine and what they leave open: the forward smile, the curve beyond the last quote, a thin chain's wide spreads |
| II. Calibration and Bayes on Black–Scholes (6–9) | The problem and the argument (6); the classical answer, and the prior hidden in its baselines and penalties (7); the posterior, which states that prior and carries it into every range (8); amortized inference and the checks it makes affordable (9) |
| III. Heston and the BayesFlow workflow (10–12) | The first claim on real quotes, against an exact posterior (11); the amortized workflow checked against exact posteriors (12) |
| IV. Rough Bergomi: the full examples (13–16) | A model priced only by simulation, and what its exact posterior costs (13, 14); agreement on a liquid smile and the long end beyond its last quote (15); a thin chain, and the argument claim by claim (16) |

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

Today quick checks show claim 1 on SPY, and claim 2 beyond the last quote: on SPY with a Laplace approximation and a
size set by the prior, and with the curve inferred on one simulated smile, by importance sampling. On 100 simulated
thin chains with the curve inferred, they show claim 2 there too: least squares without a prior over-reserves, a
parameter fixed by hand is biased, and a Laplace approximation with the posterior's prior matches the posterior. A
simulated sector shows claim 2 for pooling, and pilot smiles show part of claim 4. The joint runs on full surfaces
(⟨J1⟩ to ⟨J3⟩, on PR #48), the thin-chain studies on the full bank, the network's own checks, and the network as a
proposal where a classical fit fails, wait for the runs of [`lab/plans/CLUSTER-RUNS.md`](../lab/plans/CLUSTER-RUNS.md).

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

For a worked explanation of the neural-pricer alternative, see [forward pricing emulation and fair inference comparisons](forward-pricing-emulation.md), including training data, MCMC, missing quotes and the current implementation's limits.

## Conventions

- **Units.** Time is in years. Real maturities are trading days over 252. SPY's surface of 21 October 2026 counts
  weekdays (exchange holidays not removed); the thin chains count NYSE trading days, which makes
  a calendar year about 252 days (Lecture 16). Volatility is annualized and written as a fraction (0.20) or a
  percentage (20%). "One volatility point" is 0.01.
- **Forward units.** From Lecture 2 on, option prices are divided by the discount factor and by the forward, so
  that the forward is 1 and a price is a fraction of it.
- **Notation.** Each lecture defines a symbol when it first appears; [the table below](#symbols) collects them.
- **Data.** The daily returns of VOO from 3 January 2020 to 4 September 2026
  (`course/scripts/data/SPMO_VOO_2020-01-01_2026-09-04.csv`); SPY's surface of 21 October 2026 on the design grid
  (`tutorial/assets/results/calibration/spy/surface.json`); four thin chains of 22 and 23 October 2026, derived data
  only (`lab/checks/thin_chains/`). Raw quote files are not committed; `lab/data/SOURCES.md` gives their
  URLs.
- **Numbers that need further runs.** A number written as ⟨Sn⟩ is left blank on purpose. Study Sn produces it;
  [`lab/plans/CLUSTER-RUNS.md`](../lab/plans/CLUSTER-RUNS.md) says how. A number written as ⟨Jn⟩ comes from the
  six-target joint runs of PR #48, which infer the forward-variance curve; [status.md](status.md#the-joint-runs)
  lists them.
- **Figures.** Every figure is drawn by a script in `course/scripts/` (named by an earlier numbering of the lectures) or by
  a check in `lab/checks/`, from closed-form formulas, simulations with fixed seeds, or the data above.
  Four figures are copied from study results instead (`images/study-*.png`: Figures 6.2, 15.4, 15.5 and 15.8),
  and their captions say from which. Each figure has a paragraph that says how to read it.

### Symbols

The symbols of the advanced example, with their prior medians or design values. The row of the training curves gives the pilot's curve prior; Lecture 15, Section 15.1 gives the full bank's.

| Symbol | Meaning | Units | Prior median or design value |
| --- | --- | --- | --- |
| S\_t, F\_T, D\_T | Price, forward for maturity T, discount factor | currency, currency, – | – |
| k | Log-moneyness, ln(K / F\_T) | – | – |
| δ | Standardized moneyness, k / (σ\_ref √T) | – | −2 to 2 in steps of 0.5 |
| σ\_ref | Reference volatility of the design grid | fraction per year | 0.20 |
| T | Maturity, in trading days over 252 | years | 5, 10, 21, 42, 63, 126, 252 days |
| C | Number of quotes in the panel | – | 63 |
| y\_c, P\_c(θ; ξ\_0), σ\_c | Observed quote, model price, quote-noise standard deviation of contract c | forward units | – |
| c(k, T), p(k, T) | Black–Scholes call and put in forward units | – | – |
| H | Hurst exponent of the variance driver; roughness | – | 0.185, uniform on (0.02, 0.35) |
| η | Volatility of variance | per year^H | 2.0, uniform on (0.5, 3.5) |
| ρ | Correlation between price and variance shocks | – | −0.595, uniform on (−0.99, −0.2) |
| θ | The three parameters (H, η, ρ), in the unbounded coordinates of the adapter | – | – |
| ξ\_0(t) | Forward-variance curve, constant on each of the 7 intervals [T\_{j−1}, T\_j) between design maturities, with T\_0 = 0 | variance per year | level σ\_0 = 25% a year, slope s = 0.15 |
| σ\_0, s, u\_j | Level, slope and log perturbations of the training curves: the level on interval j is σ\_0² (1 + s t\_j) e^(u\_j), with t\_j the interval's midpoint | fraction per year, per year, – | σ\_0 ~ U(10%, 40%), s ~ U(−0.3, 0.6), u\_j ~ U(−0.2, 0.2) independently |
| ξ\_short, ξ\_long, λ | The six-target task's curve, inferred with (H, η, ρ): ξ\_0(t) = ξ\_long + (ξ\_short − ξ\_long) e^(−λt), with short variance, long variance and decay (Lecture 15, Section 15.2) | variance per year, variance per year, per year | uniform on (0.01, 0.16), (0.01, 0.16) and (0.25, 8) |
| W\_t, W^⊥\_t | Brownian motion driving the variance and the correlated price shock; independent Brownian motion for the rest of the price shock | – | – |
| Y\_t | The Volterra driver, √(2H) ∫₀ᵗ (t − s)^(H−½) dW\_s, with Var(Y\_t) = t^(2H) | – | – |
| V\_t, Q\_T | Instantaneous and realized variance | variance per year | – |
| v\_0, θ\_H, κ, σ\_v, ρ\_H | Heston's initial variance, long-run variance, mean reversion, vol-of-vol and correlation | variance, variance, per year, –, – | κ fixed at 2 |
| φ(u) | Characteristic function of ln(S\_T / F\_T) | – | – |
| N, n | Monte Carlo paths and time steps per maturity block | – | 2^15 paths, doubled up to 2^21 until the error budget holds; 80 steps over 10 days, 504 over 3 months, 504 over a year |
| M, L | Audit proposals; posterior draws per surface | – | M = 256 at 2^13 paths per block for rough Bergomi (the pilot's setting; the full study sets M from the pilot's efficiencies), 2,048 with the Fourier likelihood for Heston; L = 1,000 |
| Π, p\_Π | A product's payoff and price | forward units | – |
| T\_1, T\_2 | Forward-start dates: strike set at T\_1, expiry T\_2 | years | 63 and 126 days |

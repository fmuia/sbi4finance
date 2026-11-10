# Option application on SPX · Layer 3: full detail

*Layer 3 of 3 · reference · [Index](README.md) · [Layer 1](1-overview.md) ·
[Layer 2](2-walkthrough.md) · [Glossary](glossary.md)*

*State on the evening of 10 November 2026, for the tutorial of 14 November. This layer
replaces the plan of 9 November ([PR #81](https://github.com/fmuia/sbi4finance/pull/81)):
it keeps every item of that plan, adds the derivations, and integrates the results pushed
on 10 November by Alex ([#82](https://github.com/fmuia/sbi4finance/pull/82) to
[#85](https://github.com/fmuia/sbi4finance/pull/85),
[#87](https://github.com/fmuia/sbi4finance/pull/87)) and Stefan
([#86](https://github.com/fmuia/sbi4finance/pull/86),
[#87](https://github.com/fmuia/sbi4finance/pull/87)). Where those results change the
plan, the step says so; [Where we are, and what remains](#where-we-are-and-what-remains)
lists what is settled, what is running and what is still to be decided.*

**Sources and their labels.**
- "First checks": quick scratch runs of 9 November, committed with their scripts in
  [`design-and-error/`](../../evidence/pilot-results/2026-11-09/design-and-error/README.md).
- "Noise session": exact fits and test networks of 9 November from a separate session;
  not committed, quoted where used, tentative.
- "Alex's study": the spread-noise study and its whitening and checkpoint-replay
  diagnostics, in draft PR #85 and its handover PR #87, with the work orders in #82, #83
  and #84.
- "The pilot": Alex's European-quote pilot of 9 November, now on GitHub as branch
  `study/heston-european-observations-20261109` (evidence through `78cf9de`).

**How to read this layer.** Each step gives the definitions, the derivations, the
numbers behind each choice, and the implementation (code paths, settings, costs).
Textbook background is in the [course](../../../course/README.md); the table in
[Audience](#audience) says which lecture supports which step.

## Goals

**1. A value and a range for products that no market prices.** For each of the four SPX
closes $`d \in`$ {9, 10, 11, 12 November 2026}, each model $`m \in`$ {Heston, rough
Bergomi} and each product $`j`$ (a 6-month-to-1-year forward-start call, 3-month and
1-year volatility swaps, a 1-year down-and-in put), report the posterior 5%, 50% and 95%
quantiles, the mark, the prudent value at 90% and the reserve; and the same for an
example book. Report two checks that the market prices independently of the fit: the
model-free variance swap at 1, 3 and 12 months, and the listed 2-year 90% put.

**2. SBI makes this fast and testable.** The amortized posteriors must agree with
reference posteriors on real and simulated days (V4), be calibrated on their own
simulator (V3), and give product ranges whose coverage is measured on hundreds to
thousands of simulated days (D3), a study that is unaffordable with exact posteriors
for rough Bergomi (step 7).

**3. The limits.** Each output is labelled with what it covers: parameter uncertainty
within one model, under a stated error model. Model risk beyond the parameters, hedging
and regulatory acceptance are outside.

**Success criteria.** Numerical: Monte Carlo errors at most 2% of each 90% range
(step 9); pricing refinement within 0.01 of the half-spread (step 4). Inference: V3 and
V4 (step 8); if the network fails them, real-day results come from the references and
the failure is shown. Model: V5 and V6 are reported, not tuned (step 8).

**Not goals.** Hedge ratios under the posterior; a complete model-risk reserve; a claim
that regulators accept a posterior-based AVA.

The lesson that runs through the goals: a range is only as honest as its error model.
Step 6 shows how a wrong one makes ranges "precise but wrong", and step 7 shows that a
network can also fail to read the error scale at all unless its inputs are well
conditioned.

## Audience

The ICAIF '26 tutorial (two hours, 14 November) speaks to AI and statistics researchers
who know posteriors, MCMC and neural networks but not options, and to finance
practitioners who know options and calibration but not SBI. The material becomes:
- **Slot 4, Heston on SPX:** steps 1 to 9 with an exact posterior to check against;
- **Slot 5, rough Bergomi on SPX:** the same steps for a model priced only by simulation,
  then the two models side by side;
- **Slot 6, priors and misspecification:** the error-model comparison D4.

The current state of each notebook is in [Where we are](#where-we-are-and-what-remains).

This layer is for us. Background for each step, from the course:

| Step | Background in the course |
| --- | --- |
| 1 | [Lecture 1](../../../course/01-options-quotes-and-volatility.md) (options, quotes), [2](../../../course/02-no-arbitrage-without-a-model.md) (forwards, parity, forward units), [5](../../../course/05-implied-volatility-and-the-smile.md) (implied volatility, the variance swap) |
| 2 | [Lecture 7](../../../course/07-calibration-as-estimation.md) (least squares, Gauss–Newton interval) |
| 3 | [Lecture 6](../../../course/06-marking-a-product-the-smile-does-not-quote.md) (marking, the rules), [8](../../../course/08-bayesian-calibration.md) (posterior, propagation) |
| 4 | [Lecture 3](../../../course/03-brownian-motion-and-ito.md) (Itô), [4](../../../course/04-black-scholes.md) (Black–Scholes), [10](../../../course/10-heston.md) (Heston), [13](../../../course/13-rough-bergomi.md) (rough Bergomi) |
| 5, 6 | [Lecture 5](../../../course/05-implied-volatility-and-the-smile.md) (reading a chain), [7](../../../course/07-calibration-as-estimation.md) (observation models) |
| 7 | [Lecture 9](../../../course/09-amortized-inference.md) (amortized inference) |
| 8 | [Lectures 12](../../../course/12-the-amortized-workflow-on-heston.md) and [14](../../../course/14-why-the-exact-posterior-is-expensive.md) (checks, references, costs) |
| 9 | [Lectures 6](../../../course/06-marking-a-product-the-smile-does-not-quote.md), [11](../../../course/11-calibrating-heston.md) and [15](../../../course/15-a-liquid-smile-spy.md) (products under the posterior) |

The course predates this plan in three ways that matter here: it uses SPY, a symmetric
63-point grid and the old fixed noise rule. Where they differ, this layer governs.

## The practitioner's point of view

**The daily cycle.** At the close, the desk marks every position; the marks give the
day's P&L and the balance sheet. Product control, independent of the traders, verifies
the marks against market data and computes valuation adjustments; the bank then reports
fair values in its accounts and prudent values in its capital.

**Fair value (IFRS 13).** The price that would be received to sell an asset, or paid to
transfer a liability, in an orderly transaction between market participants. Its inputs
are ranked: quoted prices for the identical instrument in an active market (Level 1),
other observable inputs (Level 2), and unobservable inputs (Level 3). A listed SPX option
is Level 1. A forward start, a volatility swap or a barrier put whose value depends
significantly on inputs nobody quotes is Level 3, and the bank discloses how its value
would change under reasonably possible alternative inputs (paragraph 93(h)).

**Prudent value (EU).** Commission Delegated Regulation (EU) 2016/101, the technical
standard under CRR Article 105(14), asks for each fair-valued position a prudent value: a
point where the bank is "90 % confident" it could exit the valuation exposure "at that
price or better" (Article 9(5)(a) for market price uncertainty, Article 11(3) for model
risk). The difference between fair value and prudent value is an additional valuation
adjustment. CRR Article 34 then requires institutions to "deduct from Common Equity Tier
1 capital the amount of any additional value adjustments necessary". Three categories
matter here:
- **Market price uncertainty (Article 9)**, for the uncertainty of valuation inputs.
  Article 11(1) sends here the "model risk which arises due to calibrations from market
  derived parameters": the spread of a model calibrated to quotes is an Article 9 input.
- **Close-out costs (Article 10)**, for the cost of exiting at bid or ask rather than mid.
  The half-spreads of step 1 measure them for listed options; they are not computed here.
- **Model risk (Article 11)**, "due to the potential existence of a range of different
  models or model calibrations, which are used by market participants", to be measured
  where possible as "a range of plausible valuations produced from alternative
  appropriate modelling and calibration approaches" (Article 11(3)). Heston against rough
  Bergomi belongs here.

Where data cannot give a range, the Regulation allows an expert-based approach
(Articles 9(5)(b) and 11(4)).

**Aggregation.** The Annex combines the individual AVAs of these three categories over
valuation exposures. With FV the fair value, PV the prudent value and EV an expected value
from the range of possible values:

```math
\text{Method 1:}\ \ \mathrm{APVA} = (1-\alpha)\,(\mathrm{FV}-\mathrm{PV}),\qquad
\text{Method 2:}\ \ \mathrm{APVA} = \max\{0,\ (\mathrm{FV}-\mathrm{PV}) - \alpha\,(\mathrm{EV}-\mathrm{PV})\},\qquad
\mathrm{AVA} = \sum \mathrm{APVA},
```

with $`\alpha = 50\%`$ since 1 January 2021 (66% before). The factor is a fixed allowance
for the fact that many exposures do not reach their 90% points at once. When the fair
value is the posterior mean and EV is also the posterior mean, the two methods coincide:
each exposure counts for half of its standalone adjustment. Step 9 measures the actual
offset from the joint posterior draws, for comparison only; the regulatory factor stays
fixed.

**Current practice.** One point calibration per day; ranges from alternative calibrations
and models, from consensus services that pool banks' prices (some publish 10th and 90th
percentiles for prudent valuation), or from expert judgment. The hard inputs are those
nobody quotes: the forward smile, the deep wings, the vol of vol. The course's
[practice sources](../../../course/practice-sources.md) list what is documented and what
is not.

**What we propose.** The spread of a posterior over the parameters of a model calibrated
to quotes, carried through every product, is an Article 9 input; the comparison of two
models on the same information is an Article 11 question. That a posterior is an
acceptable basis for an AVA is our proposal, not established practice.

So the practitioner needs, for every product, a mark and a defensible range; for the
book, a range that accounts for offsets between products; and a statement of what the
range does not cover.

## Notation and conventions

- **Time.** Valuation at a close, time 0. The plan counts maturities in business days
  from the valuation session (NYSE holidays, `chain.trading_days`), $`T = \text{days}/252`$.
  Alex's frozen design records exact settlement times under ACT/365F (actual seconds over
  365 days). The two conventions must be reconciled before rough Bergomi's grids are
  built (step 4).
- **Spot, forward, discount.** $`S_t`$ is the index, $`F(t)`$ the forward for maturity
  $`t`$, $`D(t)`$ the discount factor. With deterministic rates and dividend yield,

  ```math
  S_t = F(t)\,X_t,\qquad X_0 = 1,\qquad \frac{dX_t}{X_t} = \sqrt{V_t}\,dB_t,\qquad B = \rho\,W + \sqrt{1-\rho^2}\,W^{\perp},
  ```

  where $`V`$ is the instantaneous variance, $`W`$ drives $`V`$, and $`W^{\perp}`$ is
  independent of $`W`$.
- **Forward units.** A put with strike $`K`$ and maturity $`T`$ is worth
  $`D(T)F(T)\,\mathbb{E}[(\tilde K - X_T)^+]`$ with $`\tilde K = K/F(T)`$; its
  log-moneyness is $`k = \ln \tilde K`$, and its standardized moneyness
  $`\delta = k/(0.16\sqrt T)`$. Model prices are computed in these units.
- **Implied volatility and vega.** $`\sigma_{\text{imp}}(k,T)`$ reproduces a forward-unit
  price through Black–Scholes; the forward-unit vega is
  $`\nu = \partial P/\partial\sigma = \sqrt T\,\varphi(d_1)`$.
- **Half-spreads.** $`h^P_i`$ is quote $`i`$'s half-spread in forward-unit price (the
  code's `half_spread`); $`h_i = h^P_i/\nu_i`$ is the same in vol points, at the quote's
  own implied volatility.
- **Integrated quantities along a path.**

  ```math
  I_t = \int_0^t V_s\,ds,\qquad J_t = \int_0^t \sqrt{V_s}\,dW_s,\qquad J^{\perp}_t = \int_0^t \sqrt{V_s}\,dW^{\perp}_s ,
  ```

  so that, by Itô's formula for $`\ln X`$,

  ```math
  \ln X_t = \rho\,J_t + \sqrt{1-\rho^2}\,J^{\perp}_t - \tfrac12\,I_t,\qquad J^{\perp}_t \mid W \sim \mathcal N(0, I_t).
  ```

  The conditional law holds because, given the whole path of $`W`$, $`J^{\perp}_t`$ is an
  integral of a known function against an independent Brownian motion. This structure is
  exact for both models and underlies every Monte Carlo estimator of steps 4 and 9
  (`lab/src/sbi4finance/calibration/products.py`).
- **Units.** One vol point ("vp") is 0.01 in decimal volatility.
- **Parameters and their code names.**

  | Symbol | Meaning | Code name |
  | --- | --- | --- |
  | $`v_0`$ | Heston initial variance | `initial_variance` |
  | $`\bar{v}`$ | Heston long-run variance | `theta` |
  | $`\kappa`$ | Heston mean-reversion speed | `kappa` |
  | $`\xi`$ | Heston vol of vol | `vol_of_vol` |
  | $`\rho`$ | correlation of index and variance shocks (both models) | `rho` |
  | $`H`$, $`\eta`$ | rough Bergomi Hurst exponent and vol of vol | `hurst`, `eta` |
  | $`\xi_s`$, $`\xi_l`$, $`\lambda`$ | rough Bergomi curve: short level, long level, decay | `short_variance`, `long_variance`, `decay` |
  | $`c`$ | multiplier of the half-spread | `c` |
  | $`\tau`$ | model-error size, decimal volatility | `tau` |

  $`\theta`$ denotes either model's full parameter vector; Alex's code writes `phi` for
  Heston's.

## 1. What has a price, and what does not

### 1.1 Chains and expiries

- **Source.** Cboe delayed quotes `_SPX.json`, fetched after each close on 9, 10, 11 and
  12 November (Cboe serves only the current snapshot). The 9 November close: last trade
  16:14:59 ET, SHA-256 `209765af…a8a1`. Raw files stay outside Git; each hash is recorded
  in `lab/data/SOURCES.md`.
- **Expiries.** PM-settled SPXW wherever listed; AM-settled SPX (root `SPX`) only on dates
  with no SPXW, which supplies the expiries beyond 30 September 2027. SPXO is a different
  product and is excluded. Alex's study keeps to SPXW only and does not substitute AM
  contracts silently; the one-year expiry is therefore an open point (step 5).
- **Valuation date.** The session of the last trade (`parse_cboe_json`).

### 1.2 Discount and forward

- **Discount.** $`D(T) = e^{-r(T)T}`$, with $`r(T)`$ linear in $`T`$ between that day's
  3-month and 2-year Treasury constant-maturity yields, converted from bond-equivalent
  yield $`y`$ to continuous compounding, $`r = 2\ln(1 + y/2)`$ (`chain.rate_curve`).
- **Forward, from put–call parity.** A call minus a put with the same strike and maturity
  pays $`(S_T-K)^+ - (K-S_T)^+ = S_T - K`$ at $`T`$, whatever happens, so its price is that
  of a forward contract:

  ```math
  C(K,T) - P(K,T) = D(T)\,\big(F(T) - K\big)\quad\Longrightarrow\quad F(T) = K + \frac{C(K,T) - P(K,T)}{D(T)} .
  ```

  We take the median of this over the six strike pairs nearest the money
  (`chain.forward_given_discount`). SPX is European, so parity holds with no
  early-exercise term; for SPY's American puts it does not, which biased both the forward
  and the smile by up to about 1.1 vp at the 1-year at-the-money put (noise session,
  binomial estimate).
- **Between expiries** (used by the barrier product), $`\ln F(t)`$ is interpolated
  linearly in $`t`$.
- Alex's frozen design stores each expiry's parity regression: forward, discount, implied
  rate, their conditional standard errors and the anchor contracts. Every third paired
  near-spot strike is reserved for this "carry" and never used as an observation.

### 1.3 Cleaning (`chain.clean_expiration`)

Keep only out-of-the-money quotes, puts with $`k \le 0`$ and calls with $`k \gt 0`$.
Drop a quote if its bid is zero, if its ask is at or below its bid, if its relative
spread (ask − bid)/mid exceeds 50%, if its mid lies outside the no-arbitrage bounds (no
implied volatility exists), or if its implied volatility is more than 25% away from the
5-point moving average of its neighbours. Alex's design applies a stricter spread filter
(half-spread over mid at most 25%) and keeps only distinct actual quotes.

### 1.4 Implied volatility, vega and half-spreads in vol points

In forward units the Black–Scholes put is

```math
p(\tilde K, T; \sigma) = \tilde K\,\Phi(-d_2) - \Phi(-d_1),\qquad d_{1} = \frac{-k + \sigma^2 T/2}{\sigma\sqrt T},\quad d_2 = d_1 - \sigma\sqrt T ,
```

and its vega is $`\nu = \sqrt T\,\varphi(d_1)`$. For each kept quote: the mid in forward
units $`p = \text{mid}/(DF)`$, its implied volatility by bisection
(`surface.implied_vol`), and the half-spread in vol points

```math
h = \frac{(\text{ask}-\text{bid})/2\,/\,(DF)}{\sqrt T\,\varphi\big(d_1(\sigma_{\text{imp}})\big)} ,
```

using the vega at the quote's own implied volatility. This removes the conversion at a
fixed 20% volatility criticized in step 6.

On SPX on 9 November (first checks), half-spreads were 0.02–0.05 vp at the money,
0.05–0.17 down to −6 at-the-money standard deviations, 0.11–0.25 at −6 to −8, and
0.10–0.37 for calls at +2 to +3. The 54 quotes of Alex's design have half-spreads of
0.02–0.10 vp (step 5).

### 1.5 The products and the checks

| Product | Payoff | Role |
| --- | --- | --- |
| Forward-start call, $`T_1`$ = 6 months, $`T_2`$ = 1 year | $`(S_{T_2} - m\,S_{T_1})^+`$ at $`T_2`$, with $`m = F(T_2)/F(T_1)`$ | Marked |
| Volatility swap, $`T`$ = 3 months and 1 year | $`\sqrt{\mathrm{RV}_T} - K_{\text{vol}}`$ per unit notional, with $`\mathrm{RV}_T`$ the realized variance | Marked |
| Down-and-in put, $`T`$ = 1 year, strike $`K = S_0`$, barrier $`B = 0.7\,S_0`$, daily closes | $`(K - S_T)^+\,\mathbf{1}\{\min_i S_{t_i} \le B\}`$ | Marked |
| Variance swap, 1, 3 and 12 months | $`\mathrm{RV}_T - K_{\text{var}}`$ | Check (model-free) |
| Listed 2-year 90% put | Listed option, December 2028 expiry, strike nearest $`0.9\,F(T)`$ | Check (extrapolation) |

Realized variance over $`N`$ daily log returns $`r_i`$ is
$`\mathrm{RV} = (252/N)\sum_i r_i^2`$; for a continuous path it tends to
$`I_T/T`$. The fair strikes are $`K_{\text{var}}(T) = \mathbb{E}[I_T]/T`$ and
$`K_{\text{vol}}(T) = \mathbb{E}[\sqrt{I_T/T}]`$ under the pricing probabilities. The
down-and-in put is the risk inside autocallable notes: their buyers in effect sell it to
the issuing bank, which is therefore long it.

**The smile region each product needs** (from the discussion of 9 November):
- *Forward start:* the near-the-money level and skew at 6 to 12 months, and above all the
  forward smile at $`T_1`$, which no listed option fixes. It comes from the model's
  dynamics, inferred from the skew's term structure.
- *Volatility swaps:* the whole strip of strikes, which sets the variance level and is
  weighted towards puts, plus the vol of vol, from the wings and short maturities, since
  $`K_{\text{vol}} \approx \sqrt{K_{\text{var}}}`$ minus a convexity term (step 9).
- *Down-and-in put:* puts at and below 70% out to 1 year, and the spot–volatility
  dynamics ($`\rho`$ and the vol of vol). At 1 year the barrier sits near
  $`\delta \approx -2.3`$: $`\ln 0.7/0.16 \approx -2.2`$, a little lower once carry lifts
  the forward above spot.

### 1.6 What listed options determine: two derivations

**Marginals, not the joint law (Breeden–Litzenberger).** Write the call price as
$`C(K) = D\int (s-K)^+ p_T(s)\,ds`$. Differentiating twice in $`K`$,

```math
\frac{\partial C}{\partial K} = -D\int_K^\infty p_T(s)\,ds,\qquad \frac{\partial^2 C}{\partial K^2} = D\,p_T(K).
```

So the calls at one maturity, across all strikes, fix the distribution of $`S_T`$ at that
maturity and the price of every payoff $`g(S_T)`$. They say nothing about how the laws at
two dates are joined, which a forward start, a volatility swap or a barrier needs.

**The variance swap is model-free.** For a continuous path, Itô's formula gives
$`d\ln X_t = dX_t/X_t - \tfrac12 V_t\,dt`$, so

```math
I_T = \int_0^T V_t\,dt = 2\int_0^T \frac{dX_t}{X_t} - 2\ln X_T .
```

The first term is a martingale under the pricing probabilities, so
$`\mathbb{E}[I_T] = -2\,\mathbb{E}[\ln X_T]`$. For any $`x \gt 0`$,

```math
-\ln x = (1 - x) + \int_0^1 \frac{(\tilde k - x)^+}{\tilde k^2}\,d\tilde k + \int_1^\infty \frac{(x - \tilde k)^+}{\tilde k^2}\,d\tilde k ,
```

as can be checked by differentiating both sides twice. Taking expectations, the first term
vanishes because $`\mathbb{E}[X_T] = 1`$, and the integrals become forward-unit put and
call prices. With $`\tilde k = e^k`$,

```math
K_{\text{var}}(T)\,T = \mathbb{E}[I_T] = 2\int \frac{\mathrm{OTM}(\tilde k)}{\tilde k^2}\,d\tilde k = 2\int \mathrm{OTM}(k)\,e^{-k}\,dk =: w(T),
```

where $`\mathrm{OTM}`$ is the out-of-the-money forward-unit price (puts for $`k \lt 0`$,
calls for $`k \gt 0`$). This is the VIX construction. It is exact for a continuous
path; its errors in practice are strike discretization, truncation of the wings and
jumps, reported as a tolerance (step 5).

So the three marked products need a model, and the variance swap is a check on any model
fitted to part of the strip.

## 2. Calibration gives one value and hides the rest

### 2.1 Pricing under the model

Under the pricing probabilities $`X`$ is a martingale, and a payoff $`\Pi`$ paid at
$`T`$ is worth

```math
P_\Pi(\theta) = D(T)\,\mathbb{E}_\theta[\Pi] .
```

For a vanilla this is a function of the law of $`X_T`$ alone; for the three products it
depends on the law of the path.

### 2.2 The classical answer

- **Least squares in implied volatility,** as practised:
  $`\hat\theta = \arg\min_\theta \sum_i \big(\sigma^{\text{market}}_i - \sigma_i(\theta)\big)^2`$,
  from 8 starting points spread over the prior box.
- **Maximum likelihood,** with the likelihood of step 6 and its error size estimated with
  $`\theta`$.
- The mark is $`P_\Pi(\hat\theta)`$.

### 2.3 What it hides: weak directions

Near $`\hat\theta`$, write the design's prices as $`P(\theta) \approx P(\hat\theta) +
\mathcal{J}\,(\theta - \hat\theta)`$, with $`\mathcal{J} = \partial P/\partial\theta`$ the
$`n \times p`$ Jacobian. Under Gaussian errors with covariance $`\Sigma`$, the
log-likelihood is locally quadratic with curvature $`\mathcal{J}^\top\Sigma^{-1}\mathcal{J}`$,
the Fisher information. Its eigenvectors with small eigenvalues are directions in which
the parameters can move far while the quotes barely change. A product $`P_j`$ that loads
on such a direction, $`\nabla P_j^\top v \ne 0`$ for a weak eigenvector $`v`$, inherits a
large uncertainty. First checks quantify how strongly this depends on which quotes exist:
removing the maturities beyond two months raises the posterior standard deviation of
$`\bar{v}`$ by 303% and of $`\kappa`$ by 232% (step 5).

### 2.4 Two classical ranges (baselines of step 9)

**The delta method.** Linearizing as above, the generalized-least-squares estimator has

```math
\widehat{\mathrm{Cov}}(\hat\theta) = \big(\mathcal{J}^\top \hat\Sigma^{-1} \mathcal{J}\big)^{-1},\qquad
\mathrm{sd}_j = \sqrt{\nabla P_j^\top\,\widehat{\mathrm{Cov}}(\hat\theta)\,\nabla P_j},\qquad
\text{range: } P_j(\hat\theta) \pm 1.645\,\mathrm{sd}_j ,
```

with $`\hat\Sigma`$ the error covariance at the fit (step 6), Jacobians by central
differences, and common random numbers for rough Bergomi. It is exact when $`P`$ and
$`P_j`$ are linear in $`\theta`$ and the errors Gaussian; it misleads where the
likelihood is curved, bounded by the prior box, or multimodal.

**Expiry-drop refits.** Refit with each of the design's expiries removed in turn, and
report the minimum to maximum of the product values: the "recalibrate on other instrument
sets" practice.

## 3. The posterior turns one value into a distribution

### 3.1 Posterior and push-forward

```math
p(\theta \mid y) \;\propto\; p(y \mid \theta)\,p(\theta).
```

A product's value is a function of the parameters, so the posterior over $`\theta`$
induces a distribution over the value, its push-forward. With posterior draws
$`\theta^{(1)},\dots,\theta^{(S)}`$, $`S = 1{,}000`$,

```math
\hat P_j^{(s)} = \hat P_j\big(\theta^{(s)};\ Z\big),\qquad s = 1,\dots,S,
```

where $`Z`$ is one fixed set of random normals shared by all draws (common random
numbers, step 9). The spread of $`\hat P_j^{(s)}`$ across $`s`$ is then due to the
parameters, not to Monte Carlo noise.

The draws are the network's (1,000 per day), or, from the references, the pooled Heston
MCMC draws thinned to 1,000, or 1,000 rough Bergomi draws resampled by their importance
weights.

### 3.2 Summaries

- **Mark:** the posterior mean $`\bar P_j = \frac1S\sum_s \hat P_j^{(s)}`$, which minimizes
  the posterior expected squared error. The posterior median and the point-calibration
  price are reported beside it.
- **Prudent value at 90%,** on the side that hurts the position:
  $`\mathrm{PV}_j = q_{0.10}`$ for a long position and $`q_{0.90}`$ for a short one, where
  $`q_\alpha`$ is the $`\alpha`$-quantile of $`\hat P_j^{(s)}`$.
- **Reserve:** $`R_j = |\bar P_j - \mathrm{PV}_j|`$ (`products.reserve`, with point
  $`= \bar P_j`$).

### 3.3 The book

With positions $`n_j`$ (positive long, negative short), each draw prices the whole book:

```math
B^{(s)} = \sum_j n_j\,\hat P_j^{(s)},\qquad \mathrm{PV}_{\text{book}} = q_{0.10}\big(B^{(s)}\big),\qquad R_{\text{book}} = \bar B - \mathrm{PV}_{\text{book}} .
```

The **example book** is long one down-and-in put (as an autocallable issuer is), long one
forward start and short one 1-year volatility swap. Report:
- the **diversification ratio** $`R_{\text{book}} / \sum_j R_j`$, which the joint draws
  measure and the Regulation sets at $`1-\alpha = 0.5`$;
- the **correlation matrix** of $`\hat P_j^{(s)}`$ across draws, which says why the
  ratio is what it is.

### 3.4 The simulator the next steps build

The posterior needs a prior (step 4), the quotes (step 5) and a likelihood (step 6).
Together they define the generative model

```math
\theta \sim p(\theta),\qquad y = P(\theta) + \varepsilon,\qquad \varepsilon \mid \theta \sim p(\varepsilon \mid \theta),
```

with $`P(\theta)`$ the clean prices at the $`n`$ design contracts. Step 7 learns to
invert it; step 8 checks the inversion.

## 4. Two models

### 4.1 Heston

```math
\frac{dX_t}{X_t} = \sqrt{V_t}\,dB_t,\qquad dV_t = \kappa\,(\bar{v} - V_t)\,dt + \xi\sqrt{V_t}\,dW_t,\qquad dB_t\,dW_t = \rho\,dt,\qquad V_0 = v_0 .
```

**Priors** (the pilot's, independent and uniform):

| Parameter | Prior | Note |
| --- | --- | --- |
| $`v_0`$, $`\bar{v}`$ | 0.01 to 0.16 | volatilities of 10% to 40% |
| $`\kappa`$ | 0.1 to 10 | free; the tutorial's lesson fixed it at 2 |
| $`\xi`$ | 0.1 to 3 | the tutorial's lesson capped it at 1.5 |
| $`\rho`$ | −0.95 to −0.1 | |

The least-squares fit to SPX on 9 November (first checks, intraday chain) is
$`\kappa = 7.26`$, $`v_0 = 0.0329`$, $`\bar{v} = 0.0506`$, $`\xi = 2.53`$,
$`\rho = -0.717`$: inside this box, outside the lesson's. If a day's posterior piles up
against $`\xi = 3`$ or $`\kappa = 10`$, widen the box and retrain.

**Feller condition.** $`2\kappa\bar{v} \ge \xi^2`$ keeps the variance strictly positive.
The SPX fit gives $`2\kappa\bar{v} = 0.735`$ against $`\xi^2 = 6.40`$. It is not imposed:
when it fails, zero is reachable but the process is still well defined, and Fourier
prices remain exact.

**Characteristic function** of $`\ln X_T`$ (`heston_options.model.characteristic_function`),
in the numerically stable form, with $`b = \kappa - \rho\xi iu`$,
$`d = \sqrt{b^2 + \xi^2(u^2 + iu)}`$ and $`g = (b-d)/(b+d)`$:

```math
\ln\varphi_T(u) = \frac{\kappa\bar{v}}{\xi^2}\Big[(b-d)\,T - 2\ln\frac{1-g\,e^{-dT}}{1-g}\Big] + v_0\,\frac{b-d}{\xi^2}\,\frac{1-e^{-dT}}{1-g\,e^{-dT}} .
```

**Call price in forward units** (Lewis's formula, contour shifted to
$`\mathrm{Im}\,u = -1/2`$; `european_price`):

```math
C(\tilde K, T) = 1 - \frac{\sqrt{\tilde K}}{\pi}\int_0^\infty \frac{\mathrm{Re}\big[e^{iu\ln(1/\tilde K)}\,\varphi_T(u - \tfrac i2)\big]}{u^2 + \tfrac14}\,du,\qquad P = C - 1 + \tilde K .
```

The derivation (Fourier transform of the payoff, then a shift of the integration contour)
is in [Lecture 10](../../../course/10-heston.md); the put follows from parity in forward
units.

**Quadrature and its accuracy gate.** Gauss–Legendre on $`[0, U]`$ with $`N`$ nodes, on
the ladder $`(N, U) \in`$ {(512, 400), (1024, 600), (2048, 1500), (4096, 3000),
(8192, 6000), (16384, 9600)}; Alex's audit added the last rung. A rung is accepted when
it and the next price inside the no-arbitrage bounds and agree within
$`0.02 \times 0.5\,h^P = 0.01\,h^P`$, a gate set by the half-spread alone, whatever the
inferred error scale. Alex's study also verifies the likelihood's sensitivity at
posterior points and an alternative integrator. The checked speed-ups of PR #77 apply.
Pricing the design costs 20–50 ms.

**Expected variance and the variance swap.** Taking expectations of the variance equation,
$`d\,\mathbb{E}[V_t] = \kappa(\bar{v} - \mathbb{E}[V_t])\,dt`$, so

```math
\mathbb{E}[V_t] = \bar{v} + (v_0 - \bar{v})\,e^{-\kappa t},\qquad
K_{\text{var}}(T) = \frac1T\int_0^T \mathbb{E}[V_t]\,dt = \bar{v} + (v_0 - \bar{v})\,\frac{1 - e^{-\kappa T}}{\kappa T} .
```

**Short maturities.** Heston's at-the-money skew tends to a constant as $`T \to 0`$; the
market's keeps steepening. This is the motivation for rough Bergomi.

### 4.2 Rough Bergomi, with the curve inferred jointly

As implemented in `rough_options/simulator.py`:

```math
V_t = \xi_0(t)\,\exp\!\Big(\eta\,Y_t - \tfrac12\,\eta^2\,\mathrm{Var}(Y_t)\Big),\qquad Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW_s,\qquad \frac{dX_t}{X_t} = \sqrt{V_t}\,dB_t .
```

- **Variance of the driver.** $`\mathrm{Var}(Y_t) = 2H\int_0^t (t-s)^{2H-1}\,ds = t^{2H}`$.
- **Mean.** $`Y_t`$ is Gaussian, so $`\mathbb{E}[e^{\eta Y_t - \frac12\eta^2\mathrm{Var}(Y_t)}] = 1`$
  and $`\mathbb{E}[V_t] = \xi_0(t)`$: the curve is the expected future variance.
- **Roughness.** For $`H \lt 1/2`$ the kernel $`(t-s)^{H-1/2}`$ diverges as $`s \to t`$:
  the latest shocks dominate, and paths are rougher than Brownian motion. The
  at-the-money skew then grows like $`T^{H-1/2}`$ as $`T \to 0`$. Whether rough models fit
  SPX better than Markovian ones is disputed (Abi Jaber and Li, 2024); D7 compares the two
  on our data.
- **The curve,** Heston-shaped (`pricing.heston_variance_values`):

  ```math
  \xi_0(t) = \xi_s\,e^{-\lambda t} + \xi_l\,\big(1 - e^{-\lambda t}\big),\qquad K_{\text{var}}(T) = \xi_l + (\xi_s - \xi_l)\,\frac{1 - e^{-\lambda T}}{\lambda T} .
  ```

  It is the same family as Heston's $`\mathbb{E}[V_t]`$, with $`\xi_s`$, $`\xi_l`$ and
  $`\lambda`$ in the roles of $`v_0`$, $`\bar{v}`$ and $`\kappa`$. Both models therefore
  learn the variance level from the same quotes in the same way, and the variance-swap
  check (V6) treats them equally. In the current tutorial, rough Bergomi was handed a curve
  replicated from strikes Heston never saw.
- **Priors,** uniform (the current joint task's bounds): $`H`$ 0.02 to 0.35, $`\eta`$ 0.5
  to 3.5, $`\rho`$ −0.99 to −0.2, $`\xi_s`$ and $`\xi_l`$ 0.01 to 0.16, $`\lambda`$ 0.25 to
  8 per year. Alex's refresh plan treats these as starting controls to review, and keeps
  difficult low-$`H`$, high-$`\eta`$ cases rather than narrowing the prior for numerical
  convenience.

**Simulation.** The hybrid scheme of Bennedsen, Lunde and Pakkanen: the singular last cell
of the kernel is simulated exactly, older cells with averaged kernels, the convolution by
FFT. The normalizer is the discrete $`\mathrm{Var}(Y_i)`$, so
$`\mathbb{E}[V_i] = \xi_0(t_i)`$ holds exactly on the grid. The task's blocks: 10 days on
1,280 steps (maturities up to 10 business days), 3 months on 2,016 steps (up to 63) and
1 year on 2,016 steps (up to 252); one business day is 128, 32 and 8 steps. Maturities
must fall on these grids: the design's exact ACT/365F times (step 5) need an explicit
mapping, which Alex's refresh plan lists among the "exact-time gaps" to inventory.

**The vanilla estimator** (`pricing.rough_block`). From the decomposition of $`\ln X_T`$
in the notation section, split $`-\tfrac12 I_T = -\tfrac12\rho^2 I_T - \tfrac12(1-\rho^2)I_T`$:

```math
X_T = \underbrace{e^{\rho J_T - \frac12\rho^2 I_T}}_{F'} \times e^{\sqrt{1-\rho^2}\,J^{\perp}_T - \frac12(1-\rho^2) I_T} .
```

Given $`W`$, the second factor is lognormal with mean 1 and log-variance
$`w = (1-\rho^2)I_T`$. So, conditionally, $`X_T`$ is a Black–Scholes underlying with
forward $`F'`$ and total variance $`w`$, and

```math
P(\tilde K, T) = \mathbb{E}\Big[\mathrm{BS}_{\text{put}}\big(F',\ \tilde K,\ w\big)\Big],
```

an average of closed-form prices over variance paths (conditional Monte Carlo).
Variance reduction: antithetic pairs, and two zero-mean control variates per maturity,
$`e^{\rho J_T - \frac12\rho^2 I_T} - 1`$ (a martingale started at 1) and
$`I_T - \sum_i \xi_0(t_i)\,\Delta t`$, with regression coefficients; the standard error
comes from the controlled residuals.

**Accuracy target.** Per contract, in forward-unit price,

```math
\mathrm{SE}_i \le 0.25\,\sqrt{(c_{\min}\,h^P_i)^2 + (\tau_{\min}\,\nu_i)^2},\qquad c_{\min} = 0.5,\quad \tau_{\min} = 0.2\ \text{vp},
```

a quarter of the smallest total error the prior allows, about 0.05 vp at the money. The
tutorial's current target, a quarter of its fixed noise, is about 0.075 vp at the money
but very loose in the call wing of low-volatility draws; the new one is tighter there. The
path ladder runs from 32,768 to 2,097,152 and stops per contract at the target. The
remaining Monte Carlo covariance $`\Sigma_{\text{MC}}`$ enters training as part of the
simulated observation and the reference likelihood as an added covariance (step 6).

*Consequence of the error-model decision (step 6).* The floor $`\tau_{\min}`$ is what
keeps this target affordable. Under the spread-only law with $`c = 1`$, the target becomes
$`0.25\,h^P_i`$, about 0.005–0.012 vp at the money: 4 to 10 times smaller, so 16 to 100
times more paths per contract at the same variance reduction, since the standard error
falls as one over the square root of the paths. That is an estimate from the targets
alone, to be measured on the 200-surface pilot before any bank is built. Alex's refresh
plan sets the same tighter scale as the gate for Monte Carlo, grid and curve errors.

### 4.3 The same information for both

Both models see the same quotes, the same error model, the same curve family, and infer
the same error-scale parameters. Neither sees information the other does not.

## 5. Which quotes

### 5.1 Coverage criteria

The principle: cover what the goal depends on, where the name quotes reliably, within the
model's scope. First checks measured each region's role (Heston on SPX, smooth model
error; rise in posterior standard deviation when the region is removed):

| Region removed | $`\kappa`$ | $`v_0`$ | $`\bar{v}`$ | $`\xi`$ | $`\rho`$ | 1-year variance swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deep puts, below $`\delta = -3`$ | +59% | +9% | +33% | +51% | +3% | +27% |
| Near puts, $`-3`$ to $`-0.5`$ | +3% | +4% | +5% | +2% | +4% | +4% |
| At the money, within 0.5 | +17% | +21% | +9% | +25% | +15% | +11% |
| Calls above $`+0.5`$ | +11% | +9% | +6% | +17% | +72% | +7% |
| Maturities up to 1 month | +63% | +79% | +3% | +108% | +15% | +1% |
| Maturities from 2 months | +232% | +24% | +303% | +15% | +37% | +287% |

A design must therefore contain, in standard deviations of a 16% volatility (SPX's
at-the-money level):
1. puts below $`\delta = -3`$ at short and at long maturities;
2. a point at the money, and calls at about $`+0.5`$, $`+1`$ and $`+2`$; without the
   near-the-money calls the uncertainty of $`\rho`$ grows 2.4 to 4.1 times;
3. maturities from 1 week to 1 year, with at least two up to 1 month and two from
   2 months.

Coverage matters more than density: under smooth error, doubling the points within the
current range gains 2–17%, while widening the range gains 23–49% on $`\kappa`$,
$`\bar{v}`$, $`\xi`$ and the variance swap.

### 5.2 The pilot's 54 quotes, checked

The pilot's design (`designs/put_heavy.json` on the pilot branch) is now on GitHub and
was checked for this layer. It holds 54 actual SPXW quotes from an intraday snapshot of
the 9 November chain (15:13 New York time, index 7,571.54), frozen with strike, expiry, forward, discount,
half-spread, implied volatility and vega:

| Expiry | Calendar days | Quotes | $`\delta`$ from | to | Half-spreads (vp) |
| --- | ---: | ---: | ---: | ---: | --- |
| 30 Nov 2026 | 21 | 9 | −3.69 | +0.93 | 0.04–0.09 |
| 31 Dec 2026 | 52 | 9 | −3.83 | +0.96 | 0.03–0.10 |
| 29 Jan 2027 | 81 | 9 | −3.87 | +0.98 | 0.03–0.09 |
| 31 Mar 2027 | 142 | 9 | −4.00 | +0.99 | 0.03–0.06 |
| 30 Jun 2027 | 233 | 9 | −4.08 | +1.04 | 0.03–0.08 |
| 30 Sep 2027 | 325 | 9 | −4.29 | +1.07 | 0.02–0.07 |

Each expiry carries targets near $`\delta`$ = −4, −3, −2, −1.5, −1, −0.5, 0, +0.5, +1.
Against the criteria:
- deep puts below −3 at short and long maturities: **met** (−3.7 to −4.3);
- an at-the-money point and calls at +0.5 and +1: **met**; calls at +2: **missed**
  (the furthest is +1.07);
- maturities: **missed** at the short end. The shortest is 21 days, so there is no
  1-week point and only one maturity up to a month; four maturities lie beyond 2 months,
  the longest at 325 days.

It misses several criteria, so by the plan's rule the design becomes layout E
(section 5.3). Alex's work order reaches the same conclusion independently and starts from
layout E. Note also that the design's half-spreads, 0.02–0.10 vp, are 10 to 50 times
smaller than Heston's measured misfit on SPX (step 6).

### 5.3 Layout E, and its audit against the actual chain

- **Targets:** 7 maturities (5, 10, 21, 42, 63, 126 and 252 business days) times
  $`\delta \in`$ {−5, −3.5, −2.25, −1.25, −0.5, 0, +0.5, +1, +2}, with
  $`k = 0.16\,\delta\sqrt T`$ and puts at $`\delta \le 0`$.
- **Posterior standard deviation relative to the tutorial's current grid** (Heston on
  SPX, first checks):

  | Error model | $`\kappa`$ | $`v_0`$ | $`\bar{v}`$ | $`\xi`$ | $`\rho`$ | 1-year variance swap |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Smooth, $`\tau`$ = 1 vp, $`\ell`$ = (0.75, 1.15) | 0.65 | 0.92 | 0.72 | 0.68 | 1.09 | 0.75 |
  | Smooth, $`\tau`$ = 2.5 vp, $`\ell`$ = (1.6, 2.0) | 0.78 | 0.81 | 0.77 | 0.83 | 1.31 | 0.78 |

  E narrows everything except $`\rho`$, which it widens by 9–31%.
- **Audit against the actual chain (Alex's study):** 62 of the 63 literal targets have a
  supported actual SPXW quote; there is no 1-year PM-settled expiry; an explicit revision
  of the last call to +1.5 supports every selected location but still misses the +2 and
  1-year goals.
- **Rules agreed in Alex's work order:** select distinct actual European PM-settled
  quotes after the quality filters; record realized distances, spreads, unsupported
  targets and provenance; revise unsupported targets explicitly; never interpolate an
  artificial observation or substitute an AM-settled contract silently; freeze strikes,
  exact settlement times and quote types; keep carry anchors disjoint from fitted quotes.
- **Caveats** (Alex's review of this plan's evidence): the layout comparison used per-expiry
  at-the-money scales while E is written at a fixed 16%; the Fisher comparisons assume
  smooth error; under the spread-only law E's optimality is not established. The literal
  actual-quote candidate is to be evaluated under the intended likelihood on a small
  representative panel before production labels.
- **Reading:** on the design's own day, `chain.build_surface(rows, rate=…, deltas=LAYOUT_E,
  sigma_ref=0.16)` reads a grid; the actual-quote selection replaces it in Alex's track.

### 5.4 The other days: an open decision

The plan reads days 2 to 4 at the design's fixed $`(k_i, T_i)`$ so that one network serves
all four days:
1. a cubic smoothing spline of implied volatility against $`\delta = k/(0.16\sqrt{T_e})`$
   per expiry, weighted by $`1/h`$, with smoothing factor equal to the number of quotes
   (`chain.smile_spline`);
2. for each target, the nearest expiry on each side of $`T_i`$ whose cleaned strikes reach
   $`k_i`$; an expiry that stops short is skipped, never extrapolated;
3. linear interpolation of the total implied variance $`\sigma^2 T`$ between the two
   expiries (one expiry: hold its volatility); half-spreads interpolated the same way;
4. a record of readings from two expiries, from one, and outside quoted strikes (must
   be zero);
5. prices, half-spreads and vegas in that day's forward units; vegas kept frozen (they
   change by a few percent a day); that day's $`c`$ set to the median ratio of its
   half-spreads to the frozen ones.

The spline error is about 0.01 vp; interpolating between expiries adds up to 0.05–0.25 vp
(noise session; an upper bound from leaving one expiry out). It is smooth across strikes,
so a smooth model-error term absorbs it, and $`\tau`$ on read days is reported separately.
A new `chain.read_points(rows, points, rate)` is needed.

Alex's work orders take the opposite position for the baseline: actual quotes only, no
interpolation, and "a different snapshot is not silently supported by this fixed-design
model"; a changed spread vector is not to be replaced by a median ratio without a
verified proportionality assumption. The options for D6 are:
- (a) the reader above, which is coherent only with a model-error term that absorbs
  reading error;
- (b) select actual quotes nearest the same targets each day, which changes the design
  and needs either one network per day or a network that conditions on the design (the
  flexible-input track of #83);
- (c) on days 2 to 4, use reference posteriors only (Heston MCMC at 15–40 CPU-minutes a
  day is affordable), with the network shown on day 1.

Option (c) is the cheapest for the tutorial; (a) is the plan's; (b) is the long-term
route.

Whatever the option, every design point must lie inside the quoted strikes on every day
used, and this is checked. On 9 November SPX puts were quoted down to −12 to −16
at-the-money standard deviations up to 2 months and about −8 at 6 to 12 months, and calls
to +2.2 to +3.1.

### 5.5 Data held out of every fit

- **Model-free variance swap.** For each expiry whose cleaned strip reaches $`\delta \le -8`$
  and $`\delta \ge 1.8`$ in 20%-volatility units (the existing replication criterion;
  with `sigma_ref=0.16`, pass `replication_deltas=(-10, 2.25)` to keep the same reach),
  compute $`w(T_e)`$ by the trapezoid rule over the strip (step 1.6), interpolate $`w`$
  linearly in $`T`$ to 21, 63 and 252 business days, and report $`\sqrt{w(T)/T}`$.
  Discretization, truncation and jumps are reported as a tolerance, not tuned away.
- **The listed 2-year 90% put:** the listed expiry nearest 2 years (December 2028), the
  strike nearest $`0.9\,F(T)`$; record its mid, half-spread in vol points and implied
  volatility.
- **Held-out quotes:** cleaned quotes not in the design (other strikes and expiries);
  carry anchors are excluded from both.

### 5.6 Evidence behind the design choices

**The reviewer's grid issue, reassessed.**

| Claim | Verdict |
| --- | --- |
| Symmetric ±2 at a fixed 20% volatility | Right; the noise rule is tied to 20% too |
| Put quotes reach $`\delta`$ = −8 to −17 | Right in substance: −8.2 to −14.5 in 20%-volatility units on SPY, 21 October (−13 to −21 in at-the-money units) |
| A third of the points are far calls with almost no information | 21 of 63 at $`\delta \ge +1`$. Their low weight comes from the 20% vega conversion: 5–7 vp of effective noise against half-spreads of 0.1–0.4 vp |
| Some points have no quotes | Wrong: 0 of 63 lie outside quoted strikes (SPY 21 October and 6 November, SPX 9 November) |
| Repricing costs 1.3 CPU-h per 100k Heston surfaces and 330–370 CPU-h for rough Bergomi | Right at today's accuracy targets |
| Fix: scale by at-the-money volatility and shift towards puts | The shift is right, but only with a vol-point error model; scaling alone is neutral on one date |

**The reviewer's fix under three error models.** Grids: G1, −4 to +1 in at-the-money
units (the proposal); G2, the same deltas at 20%; G3, ±2 in at-the-money units. Posterior
standard deviation relative to the current grid:
- *Under the tutorial's rule:* G1 and G2 look 2–12 times tighter for Heston and 5–25 times
  for rough Bergomi, because deep puts get only 0.003–0.03 vp of noise: false precision.
  Rough Bergomi's Monte Carlo error at $`\delta = -4`$ would be 40–65 times that noise.
- *Under a consistent, independent vol-point error:* Heston (150 prior draws), G1: $`v_0`$
  1.07, $`\bar{v}`$ 0.89, $`\xi`$ 0.83, $`\rho`$ 1.48, variance swap 0.83. Rough Bergomi
  (8 points), G1: $`H`$ 0.58, $`\eta`$ 0.77, $`\rho`$ 1.41, variance swap 0.68. G3 stays
  near 1 (0.95–1.0).
- *Under smooth model error (Heston on SPX):* widening to −6…+2 gains 23–49% on
  $`\kappa`$, $`\bar{v}`$, $`\xi`$ and the variance swap; doubling the density gains 2–17%;
  $`\rho`$ needs the near-the-money calls.

**Coverage by goal and by name.**

| Goal | What to cover |
| --- | --- |
| Reprice the hedges | every liquid quote, weighted by the error model |
| Variance or volatility swap | the whole strip into the deep puts |
| Forward start, cliquet | at-the-money level and skew across the forward dates; short maturities for the vol of vol |
| Downside barrier, autocallable | long-dated puts down to the barrier |
| Model comparison | where the models differ, with the same data for every model |

| Name | Design | Dominant error |
| --- | --- | --- |
| SPX | wide and put-heavy, 1 week to 1 year | model error; European, so clean forwards |
| SPY | as SPX | plus early exercise on puts, so prefer SPX |
| Thin ETF (EWG: 4 expiries to 6 months; half-spreads 1.2–2 vp) | what the chain has, narrow | quote error; the prior beyond the quotes |
| Single stock | skip earnings expiries | early exercise, dividends, events |

**Coordinates.** The plan uses a fixed 16% scale, not each surface's at-the-money
volatility. On one date, scaling by at-the-money volatility adds no information (G3); it
makes strike locations depend on the data, so the simulator would place them from each
simulated surface and the network would have to condition on them; it pays off only
across names and volatility regimes.

**Not adopted.** A set encoder over all raw quotes (it needs variable quote layouts in
simulation: the deferred sparse-chain work, now sequenced by #83); a denser symmetric
grid (under smooth error, density adds little).

## 6. How quotes differ from the model

### 6.1 Two candidate laws

Two observation laws are now on the table, and the choice between them is the main open
decision of this application.

- **Spread only** (Alex's adopted baseline, PRs #82 and #85):

  ```math
  y_i = P_i(\theta) + c\,h^P_i\,Z_i,\qquad Z_i \sim \mathcal N(0,1)\ \text{independent}.
  ```

  The primary teaching target fixes $`c = 1`$ and infers the five Heston parameters.
  Fixed-scale sensitivity applies $`c`$ = 0.5, 1 and 2 to the same observed prices
  (Alex's references give physical widths about half and twice those at $`c = 1`$). A
  separate extension infers $`c \sim U(0.5, 2)`$ jointly. Alex's stated rule: Heston can
  fail to fit a real chain even when inference is correct; report that rather than
  inflate a noise scale until the market looks well explained.
- **Spread plus smooth model error** (this plan):

  ```math
  y = P(\theta) + c\,h^P \odot Z + \tau\,\nu \odot (L\,U),\qquad Z,\,U \sim \mathcal N(0, I_n)\ \text{independent},\qquad L L^\top = K_\ell ,
  ```

  with $`c`$ known (1 on market data, drawn in training), $`\tau`$ the model-error size in
  decimal volatility, inferred, and $`K_\ell`$ the squared-exponential correlation over
  $`(\delta_i, \ln T_i)`$:

  ```math
  (K_\ell)_{ij} = \exp\!\Big(-\frac{(\delta_i-\delta_j)^2}{2\ell_\delta^2} - \frac{(\ln T_i-\ln T_j)^2}{2\ell_T^2}\Big).
  ```

  The spread-only law is the special case $`\tau = 0`$.
- **Earlier laws, for reference:** the original lesson used a fixed implied-volatility
  profile converted with Black–Scholes vega at 20%; the completed $`\tau`$ pilot used
  independent errors with standard deviation $`\sqrt{(c\,h^P_i)^2 + (\tau\,\nu_i)^2}`$ and
  $`\tau \sim U(0, 2\ \text{vp})`$.

Half-spread as one Gaussian standard deviation is a convention in every version.

### 6.2 Likelihoods

**Spread only.** With $`R(\theta) = \sum_i \big((y_i - P_i(\theta))/h^P_i\big)^2`$,

```math
\log p(y \mid \theta, c) = -\frac{R(\theta)}{2c^2} - n\ln c - \sum_i \ln h^P_i - \frac n2 \ln 2\pi .
```

The $`-n\ln c`$ term must be kept: without it inference about $`c`$ changes. At fixed
$`\theta`$ the conditional mode is $`\hat c = \sqrt{R(\theta)/n}`$, clipped to the prior
interval (Alex's known-$`\theta`$ oracle, `conditional_c_mode`); it is an oracle, not the
joint posterior.

**Smooth model error.**

```math
\Sigma(\tau, c) = c^2\,\mathrm{diag}\big((h^P)^2\big) + \tau^2\,\mathrm{diag}(\nu)\,K_\ell\,\mathrm{diag}(\nu)\ \big(+\ \Sigma_{\text{MC}}(\theta)\big),
```

```math
\log p(y \mid \theta, \tau) = -\tfrac12\,r^\top\Sigma^{-1}r - \tfrac12\log\det\Sigma - \tfrac n2\log 2\pi,\qquad r = y - P(\theta).
```

Computed with a Cholesky factor of $`\Sigma`$ and $`10^{-10}`$ jitter on the diagonal of
$`K_\ell`$. The log-determinant is what stops $`\tau`$ from growing without bound.
$`\Sigma_{\text{MC}}`$ is rough Bergomi's Monte Carlo price covariance at $`\theta`$, zero
for Heston.

**Vol-unit form.** For each quote,
$`y_i - P_i(\theta) = \mathrm{BS}(\sigma^{\text{market}}_i) - \mathrm{BS}(\sigma^{\text{model}}_i) \approx \nu_i\,(\sigma^{\text{market}}_i - \sigma^{\text{model}}_i)`$
to first order. Dividing the residuals by $`\nu`$,

```math
\frac{r}{\nu} \sim \mathcal N\big(0,\ c^2\,\mathrm{diag}(h^2) + \tau^2 K_\ell\big),
```

with $`h = h^P/\nu`$ in volatility units: the implied-volatility formulation, to first
order. Working in prices keeps negative simulated prices usable without inversion.

### 6.3 What the data say: the error budget

| Source | Size (vp) | From |
| --- | --- | --- |
| Bid–ask, SPX 9 November | 0.02–0.05 at the money; 0.05–0.17 down to −6; 0.11–0.25 at −6 to −8; 0.10–0.37 for calls at +2 to +3 | first checks |
| Bid–ask, the pilot's 54 quotes | 0.02–0.10, median 0.04 | this layer, from the design file |
| Bid–ask, SPY 6 November | 0.04–0.13; 0.26–0.37 at $`\delta = +2`$ | noise session |
| Spline reading | about 0.01; neighbours correlate +0.38 | noise session |
| Interpolation between expiries | 0.05–0.25 (upper bound) | noise session |
| Early exercise (SPY only) | up to about 1.1 at the 1-year at-the-money put | noise session |
| Heston model error | SPX: 1.0 RMS, 17 times the median half-spread; residuals correlate 0.80 between strikes 0.5 apart and 0.85 between adjacent maturities. SPY: 0.8–1.2 RMS, about 99% of the residual variance; correlation 0.56–0.78 | first checks; noise session |
| Heston on the market chain under spread only | joint MCMC puts $`c`$ at its upper bound 2: the misfit exceeds twice the half-spread | Alex's study |

The last two rows say the same thing from two directions: on SPX the misfit is about an
order of magnitude larger than the quotes' own uncertainty, and it is smooth.

### 6.4 Information about the error scale: three derivations

**How precisely the multiplier can be learned (spread only).** At the true $`\theta`$,
$`R/c^2 \sim \chi^2_n`$; with $`p`$ parameters fitted, about $`\chi^2_{n-p}`$. The
estimate $`\hat c = c\,\sqrt{\chi^2_m/m}`$, $`m = n - p`$, has relative standard
deviation about $`1/\sqrt{2m}`$ (delta method on $`\mathrm{Var}(\chi^2_m/m) = 2/m`$). With
$`n = 54`$, $`p = 5`$: 0.10, so a 90% interval about $`0.33\,c`$ wide. Alex's matched MCMC
gives $`c`$ RMSE 0.100 and widths around 0.36; the whitened network 0.365 (step 7). The
information is there.

**How precisely the model-error size can be learned (smooth error).** In volatility units with
$`\Sigma = c^2 D + \tau^2 K`$, $`D = \mathrm{diag}(h^2)`$, the Fisher information is
$`I(\tau^2) = \tfrac12\,\mathrm{tr}\big(\Sigma^{-1}K\,\Sigma^{-1}K\big)`$. If $`D \approx
d\,I`$ and $`K = \sum_k \lambda_k e_k e_k^\top`$,

```math
I(\tau^2) = \frac12\sum_k \frac{\lambda_k^2}{(c^2 d + \tau^2\lambda_k)^2} \approx \frac{m_{\text{eff}}}{2\tau^4},\qquad m_{\text{eff}} = \text{number of } k \text{ with } \tau^2\lambda_k \gg c^2 d ,
```

so $`\mathrm{sd}(\tau)/\tau \approx 1/\sqrt{2\,m_{\text{eff}}}`$: only directions where
model error dominates quote noise inform $`\tau`$, and the parameters can absorb up to
$`p`$ of them. Computed on the pilot's 54 quotes with their own half-spreads (this
layer's calculation, $`\theta`$ held fixed): $`\mathrm{sd}(\tau)/\tau`$ is 0.11–0.12 with
$`\ell = (0.75, 1.15)`$ ($`m_{\text{eff}} \approx`$ 34–40) and 0.15–0.17 with
$`\ell = (1.6, 2.0)`$ ($`m_{\text{eff}} \approx`$ 17–21), for $`\tau`$ from 1 to 2.5 vp.
$`\tau`$ is identifiable, somewhat less sharply than $`c`$, and less so the smoother the
error.

**Why independent errors make posteriors too narrow.** Fit a parameter along a direction
$`a`$ by least squares while the true residual covariance is $`\tau^2 K`$. The true
variance of the estimate is $`\tau^2\,a^\top K a/(a^\top a)^2`$; the independent model
claims $`\tau^2/(a^\top a)`$. Their ratio is $`a^\top K a/(a^\top a)`$, which is large when
$`a`$ is smooth across quotes. For a level shift, $`a = \mathbf{1}`$, the ratio is
$`\mathbf{1}^\top K\mathbf{1}/n`$, the average number of strongly correlated neighbours.
On the pilot's design it is 11 with $`\ell = (0.75, 1.15)`$ and 26 with
$`\ell = (1.6, 2.0)`$: the independent model understates that standard deviation by a
factor of 3.4 to 5.1. Variance levels, and hence variance and volatility swaps, load on
such smooth directions.

### 6.5 Settings of the smooth-error law

- **Prior of the model-error size** $`\tau`$: log-uniform on $`[0.2, 5]`$ vp, for both models. The smooth amplitude
  was estimated at 2.1–2.6 vp (noise session, tentative), above the pilot's cap of 2.
  Below 0.2 vp, model error is comparable to processing error, which this design does not
  resolve, and the floor bounds rough Bergomi's Monte Carlo cost (step 4). The pilot's
  independent version is kept for D4.
- **Correlation lengths,** fixed per model before training by profile likelihood on the
  first two days:

  ```math
  \hat\ell = \arg\max_\ell \sum_{t \in \{1, 2\}} \max_{\theta, \tau} \log p(y_t \mid \theta, \tau, \ell),\qquad \ell_\delta \in \{0.75, 1, 1.5, 2\},\quad \ell_T \in \{1, 1.5, 2\}.
  ```

  Heston uses the Fourier likelihood; rough Bergomi the fixed-seed Monte Carlo pricer with
  Gauss–Newton steps (the classical route of study S6); if rough Bergomi's profile is flat,
  it uses Heston's $`\ell`$.
- **An independent model-error part** $`\tau_n\,\nu \odot U'`$ is added only if V2 needs it.
- **The multiplier** $`c`$: 1 on the design's own day; on other days as decided in step 5.4;
  $`U(0.5, 2)`$ in training, as a known input.

### 6.6 What it replaces

The tutorial's current rule adds independent price noise equal to the Black–Scholes vega
at 20% volatility times 0.3 vp at the money, rising to 1.0 at $`|\delta| \ge 2`$ (in 20%
units). It is about 8 times SPY's bid–ask and 6–15 times SPX's at the money. In vol
points it equals the stated values only where a quote's own volatility is 20%: on SPY,
21 October, the median effective noise is 0.44 vp at $`\delta = -2`$ and 5.4 vp
(maximum 7.0) at $`\delta = +2`$; at the prior's centre, 0.28–0.55. Its fit: $`\chi^2`$ =
66.5 on 63 quotes (21 October), 508 (6 November). It is neither quote noise nor a stated
model error, but an unestimated, independent allowance that alone sets every posterior
width. It is replaced for the new tasks only; existing task identities and assets are
unchanged. The new tasks are `heston_spx_v1`, extending the pilot's, and
`rough_joint_spx_v1`.

### 6.7 Consequences already measured

- **Coverage** (noise session): networks trained on the current rule, or on independent
  errors, covered the truth 22–72% and 18–58% of the time at a nominal 90%, when the real
  misfit was smooth and of the measured size.
- **The variance-swap gap** (the current tutorial's case against Heston). On SPY,
  21 October, the exact Heston posterior under the rule puts variance swaps 0.6–1.7 vp
  below the replicated values, with bands 0.3–0.6 vp wide. On SPY, 6 November, the
  1-week shortfall stays on both the current and a put-heavy grid (−3.9 and −3.2 vp),
  and the 1-week deep put misses by 11.5 vp: the short-end shortfall is real misfit.
  Under independent $`\tau`$ the bands (0.5–1.1 vp) exclude the gap (−0.5 to −1.6 vp);
  under smooth $`\tau`$ the bands (1.8–3.0 vp) include it from 1 month to 1 year (noise
  session, tentative). The comparison was also unequal (step 4.2).

### 6.8 The decision, and what Alex's results add

- **Agreed:** price-space Gaussian errors; half-spreads in the scale; actual quotes;
  negative simulated prices kept; a known $`c`$ as the primary baseline.
- **Disputed:** whether the market application carries a model-error term. Alex's
  baseline omits it and reports the misfit; this plan includes it because a prudent
  range for a product must account for the model's measured misfit, which is about ten
  times the quotes' uncertainty and smooth. Alex's market result, $`c`$ at its bound
  under spread only, is the misfit showing through.
- **A reconciliation that both tracks support:** stage 1, spread only with $`c = 1`$,
  validates inference and teaches what the quotes identify if the model were right; stage
  2 adds the smooth model error for the product ranges on market data, and reports
  $`\tau`$ and the residual map (D5). Alex's caution applies to stage 2 too: $`\tau`$
  cannot tell structural model error from data or convention problems, so it is reported,
  not used to make the fit look good.
- Stage 2 inherits every network requirement found in stage 1, above all input
  conditioning (step 7), because inferring $`\tau`$ is the same kind of problem as
  inferring $`c`$: reading the size of a residual.

## 7. Computing the posterior: simulation-based inference

### 7.1 Why not compute every posterior exactly

| | Heston | Rough Bergomi |
| --- | --- | --- |
| One likelihood evaluation | 20–50 ms (Fourier) | 25–50 CPU-s (Monte Carlo, at the step 4 target) |
| One reference posterior | 15–40 CPU-min by MCMC (Alex's cap: 45 per case) | 7–14 CPU-h by importance sampling |
| Coverage study D3 | 1,300 cases: about 325–870 CPU-h exact | 300 cases: about 2,100–4,200 CPU-h exact |
| D3 with the network | about 11 CPU-h, mostly product pricing | about 12 CPU-h |

V3 needs 2,000 more posteriors per model. One posterior is affordable without SBI when a
good proposal exists; thousands are not, and validation, daily use and many names all
need thousands.

### 7.2 The amortization identity and the training objective

Draw $`\theta \sim p(\theta)`$ and $`y \sim p(y \mid \theta)`$. The pair is a draw from
$`p(\theta, y) = p(\theta)\,p(y \mid \theta) = p(y)\,p(\theta \mid y)`$: read from the
right, $`\theta`$ is a posterior draw for its own $`y`$. For a conditional density
$`q(\theta \mid y)`$,

```math
\mathbb{E}_{p(\theta, y)}\big[-\log q(\theta \mid y)\big] = \mathbb{E}_{p(y)}\Big[\mathrm{KL}\big(p(\cdot \mid y)\,\big\|\,q(\cdot \mid y)\big)\Big] + \mathbb{E}_{p(y)}\Big[\mathcal{H}\big(p(\cdot \mid y)\big)\Big],
```

where the second term does not depend on $`q`$. Minimizing the simulated negative
log-density therefore drives $`q`$ to the posterior for $`p(y)`$-almost every $`y`$:
neural posterior estimation, used by the spline coupling flows of Alex's study. BayesFlow's
flow matching, used in the tutorial and in Stefan's notebook, trains a velocity field by
regression instead; its optimum is the same conditional distribution. In both cases the
approximation error can differ from one $`y`$ to another, which is why the checks of
step 8 test single observations as well as averages.

### 7.3 Simulation banks

| | Heston (`heston_spx_v1`) | Rough Bergomi (`rough_joint_spx_v1`) |
| --- | --- | --- |
| Draws | The pilot's bank: 100,000 surfaces at its 54 contracts. A new design (layout E) needs repricing: about 1.3 CPU-h per 100,000 | 100,000: 90,000 train, 5,000 validation, 5,000 test |
| Pricing | Fourier, rungs within $`0.01\,h^P`$ | Hybrid-scheme Monte Carlo to the step 4 target |
| Stored per draw | $`\theta`$; clean prices; rung; refinement difference | $`\theta`$; prices; Monte Carlo covariance; paths per block |
| Failures | Recorded and rejected, never silently redrawn | Same |
| Seeds | As in the pilot | `SeedSequence([task fingerprint words, example id, purpose])`, independent of batch order |
| Cost | Exists; cheap to reprice | 3.3 CPU-h per 1,000 surfaces at today's target; 660–1,300 CPU-h at the step 4 target; more under the spread-only law (step 4); measure on a 200-surface pilot first |

The clean banks do not depend on $`\tau`$, $`c`$, $`h`$ or $`\ell`$; errors are added at
training time. New code: `lab/studies/spx_rough/bank.py`, reusing the tutorial runtime's
rough pricer and writing a manifest with SHA-256.

How densely the bank covers the prior matters: on 512 development surfaces, the median
distance to the nearest training surface, in half-spread units, is 94.8 with 20,000 rows
and 66.6 with 100,000, against a noise norm of about 7.35 (Alex's bank-resolution audit).
The network must interpolate between surfaces that are far apart relative to the noise.

### 7.4 Input conditioning: why the network ignored the error scale, and the fix

This is the main new result of 10 November. It concerns any network that must infer an
error scale, $`c`$ or $`\tau`$, and so applies to this plan's networks in full.

**The finding.** Alex's joint network for the five Heston parameters and $`c`$ recovered
the physical parameters almost perfectly, but returned $`c`$'s prior: median 90% width
1.336 against the prior's 1.35, RMSE 0.445 against 0.437 for always answering the prior
median, truth–estimate correlation about zero. A 100,000-surface bank, 50 million extra
training examples and batches of 128, 512, 1,024 and 4,098 changed nothing. Marginal SBC
still passed, because returning the prior is calibrated on average. Joint MCMC on eight
observations stratified by $`c`$ converged on seven and recovered $`c`$ (RMSE 0.100,
correlation 0.977, 90% widths around 0.36): the information was there and the network
missed it.

**The geometry.** The network's input is $`x = y/h^P`$ (elementwise). Under the
spread-only law, $`x = s(\theta) + c\,z`$, with $`s = P/h^P`$ and $`z`$ standard normal,
independent of $`\theta`$ and $`c`$. Across the prior,

```math
\mathrm{Cov}(x) = \mathrm{Cov}_\theta\big(s(\theta)\big) + \mathbb{E}[c^2]\,I,\qquad \mathbb{E}[c^2] = 1.75\ \text{for}\ c \sim U(0.5, 2).
```

One parameter vector moves all 54 prices together, so the first term is huge and of low
effective rank: three directions hold 98.2% of the clean-price variance, while the noise
is about 1.3 half-spreads in every direction (Alex's conditioning diagnostic). The
information about $`c`$ lives in the residual,
$`\hat c^2 \approx \sum_i (x_i - s_i(\theta))^2/n`$, that is, in the directions the
parameters cannot explain: $`n - p = 49`$ for a flat price surface; because the surface
curves with the parameters, noise dominates 33 of the 54 in the diagnostic (step 6.4).

**Why ordinary standardization fails.** Per-feature scaling divides each quote by its own
standard deviation. The noise directions are combinations of all 54 quotes, so they stay
buried: the condition number of the scaled input covariance is 2.2 million (1.09 million
before scaling). An illustrative two-quote example: $`x_1 = s + c z_1`$,
$`x_2 = s + c z_2`$, with a common move $`s`$ of standard deviation, say, 150
half-spreads. The only information about $`c`$ is in
$`x_1 - x_2 = c\,(z_1 - z_2)`$. After standardization this difference has standard
deviation about 0.012 while each input has 1; to read $`c`$, the network must learn the
weights (1, −1) to about one part in a thousand and then square the result, and with 54
quotes it needs about 49 such cancellations.

**Why more data or steps did not help.** For gradient descent on a linear first layer, the
curvature of the loss in the weights is the input covariance. The step size is capped by
the largest eigenvalue, so progress along the smallest is about $`1/\kappa`$ per step,
with $`\kappa`$ the condition number. The batch-128 continuation made 391,000 updates,
about 0.2 times $`\kappa`$: far too few. More data and longer training add steps but do
not change $`\kappa`$; bigger batches reduce gradient noise, not curvature; per-feature
scaling and Adam are both diagonal, while the bad directions are rotations. (The argument
is exact only for plain gradient descent on a linear layer; it is a guide, not a proof.)
It also explains why the five physical parameters were learned well: they live in the
large directions.

**The fix: full (ZCA) whitening.** With $`\mu`$ and $`C`$ the mean and covariance of the
noisy training inputs, $`C = Q\,\mathrm{diag}(\lambda)\,Q^\top`$,

```math
W = Q\,\mathrm{diag}\Big(1/\sqrt{\max(\lambda_k,\ 10^{-8}\lambda_{\max})}\Big)\,Q^\top,\qquad u = W\,(x - \mu),\qquad \mathrm{Cov}(u) \approx I .
```

The map is invertible and keeps all 54 coordinates, so it changes neither the information
nor the exact posterior, only the difficulty of the network's task. The eigenvalue floor
limits amplification; it was inactive in the pilot. It rescales the noise-dominated
directions to unit variance and in effect hands the network the residual.

**Rules for using it** (Alex's specification, PR #85):
- fit $`\mu`$ and $`W`$ on noisy training inputs only, in float64 before the final float32
  cast; never refit on validation, reserved or market observations;
- persist and hash the mean, matrix, eigenvalues and floor with the training, source,
  design and spread-profile bindings, as part of the model artifact; apply the transform
  exactly once at training, validation and prediction, including market calls;
- transform the whole noisy vector. In whitened coordinates the noise covariance is
  $`c^2\,W W^\top`$; generating `whiten(clean) + c Z` would change the model;
- a dense whitener belongs to one ordered layout: a new design (layout E), a different
  model (rough Bergomi) or variable locations and masks (#83) each need their own.

**For the smooth-error law.** The input noise covariance becomes
$`c^2 I + \tau^2\,\mathrm{diag}(1/h)\,K_\ell\,\mathrm{diag}(1/h)`$, with $`h`$ in vol
points; at $`\tau = 1`$ vp and $`h \approx 0.04`$ vp the model-error variance is hundreds
of times the quote noise in its leading directions, and those directions are smooth, like
the parameter-driven ones. Inferring $`\tau`$ is therefore at least as badly conditioned
as inferring $`c`$, and whitening is a requirement, not an option, for this plan's
networks.

**The controlled evidence** (Alex's paired pilot; identical initial states, the same 10,000
noisy training rows repeated for 200 passes, batch 256, 8,000 updates, Adam with cosine
decay from $`5\times10^{-4}`$ to $`5\times10^{-5}`$, six-block spline coupling flows with
[128, 128] conditioners and zero dropout; the arms differ only in the input scaling):

| Same 512 validation observations | Per-feature scaling | Full ZCA whitening |
| --- | ---: | ---: |
| Input condition number | 2.2 million | about 1 |
| $`c`$ posterior-median RMSE | 0.442 | 0.170 |
| $`c`$ truth–median correlation | −0.001 | 0.923 |
| Median 90% width of $`c`$ | 1.338 | 0.365 |
| Nominal-90% coverage of $`c`$ | 87.7% | 74.0% |
| Marginal SBC screens passed | 3 of 6 | 0 of 6 |

Stefan's notebook (PR #87, `lab/notebooks/heston_6_parameter_bayesflow.ipynb`) found the
fix: ZCA on $`y/h^P`$, flow matching (three 128-unit layers, dropout 0.1), 10,000
simulations, 200 epochs at batch 256, trained in 16 seconds, $`c`$ correlation 0.887. It
changed several things at once and has no calibration check; the paired pilot attributes
the gain to the transform.

### 7.5 Generalization: the whitened network overfits its fixed noise

The whitened network's 90% intervals are too narrow: final coverage 55–61% for the
physical parameters and 74% for $`c`$, with all six SBC screens failing. Its validation
loss is lowest at epoch 83 (−6.38) and rises to −1.41 by epoch 200 while the training loss
keeps falling. An exact replay that kept earlier checkpoints:

| Epoch | $`c`$ RMSE | $`c`$ correlation | Median 90% width of $`c`$ | $`c`$ coverage | Physical coverage | SBC screens within band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.168 | 0.924 | 0.478 | 80.9% | 79.7–84.6% | 2 of 6 |
| 75 | 0.163 | 0.928 | 0.444 | 79.7% | 76.8–82.2% | 0 of 6 |
| 83 | 0.168 | 0.924 | 0.428 | 78.9% | 74.2–78.7% | 0 of 6 |
| 100 | 0.170 | 0.922 | 0.418 | 78.7% | 66.8–73.8% | 0 of 6 |
| 200 | 0.170 | 0.923 | 0.365 | 74.0% | 54.9–60.9% | 0 of 6 |

Earlier stopping helps coverage, but no checkpoint is calibrated, and the errors are not
uniform: on the broad synthetic case, epoch-50 physical intervals are 3.7–9.6 times the
reference widths (mean errors up to 16.7 reference standard deviations), epoch 83's are
2.9–6.9 times, while population coverage is too low. No single interval multiplier
repairs this.

**The likely cause.** Each of the 10,000 training rows kept one noise draw for all 200
passes. Once whitening exposes the noise directions, the network can memorize those
draws; without whitening it could not see them, which is why only the whitened network
overfits. **This plan's training draws fresh errors at every exposure**, which removes the
mechanism by construction. Alex's next isolated experiment tests exactly this: the same
10,000 clean surfaces, frozen whitener, initialization, architecture, batch and 8,000-step
schedule, with $`c`$ and the 54 errors refreshed each pass, compared at matched epochs.
If needed, the following single changes are more distinct surfaces from the 100,000 bank,
then one regularization or flow-matching comparison. The fixed-$`c`$ teaching network is
then fitted with the settled recipe, and the recipe is frozen before a fresh confirmation
on the reserved 2,000-case pool, which stays unopened during development.

### 7.6 The networks of this plan

- **Conditions.** The quotes divided by their half-spreads, $`y_i/h^P_i`$, whitened as in
  7.4; and, because $`c`$ is known in this plan, the log measurement scale
  $`\log(c\,h^P_i/(D_iF_i))`$, standardized. When $`c`$ is inferred instead, nothing that
  contains it may be an input (Alex's rule); when it is known and varies, the network
  must be trained over its declared range with $`c`$ as an explicit condition.
- **Training augmentation, fresh at every exposure:**
  1. draw $`\ln\tau \sim U[\ln 0.002, \ln 0.05]`$ and $`c \sim U(0.5, 2)`$;
  2. draw $`Z, U \sim \mathcal N(0, I_n)`$ independently;
  3. form $`y = P_{\text{bank}} + c\,h^P \odot Z + \tau\,\nu \odot (L U)`$, with $`L`$ the
     Cholesky factor of $`K_\ell`$, computed once; for rough Bergomi $`P_{\text{bank}}`$
     already carries its Monte Carlo error;
  4. whiten with the frozen transform.
- **Targets,** through the precision-safe bounded (logit-box) transform computed in
  float64 (Alex's study found and repaired a float32 rounding failure near the bounds):
  Heston $`(v_0, \kappa, \bar{v}, \xi, \rho, \ln\tau)`$; rough Bergomi
  $`(H, \eta, \rho, \xi_s, \xi_l, \lambda, \ln\tau)`$.
- **Architecture and training.** Start from the recipe Alex's study settles: a bounded
  six-block spline coupling flow with [128, 128] conditioners and zero dropout (a dropout
  of 0.05 cost 1.9 nats of objective; removing it narrowed over-wide intervals from
  4.8–10 to 1.6–1.8 times the reference), Adam with cosine decay, and a low final learning
  rate (annealing to $`5\times10^{-7}`$ narrowed the fixed-$`c`$ widths to 1.13–1.41 times
  the reference). Rough Bergomi uses the same settings.
- **Selection** of configuration, seed and checkpoint on simulated data only, never on
  the real days, with the rule frozen before confirmation.
- **Code.** $`\tau`$ and $`c`$ are kept apart from the pricing parameters, because
  `tasks.model_parameters` passes every inferred name to the pricer.
- **Cost.** A Heston fit of 30,000 updates takes about 100 s on one L40S GPU; the
  checkpoint replay ran 4,000 updates in 68 worker-seconds. Plan 10–20 fits per model.
- **Earlier evidence:** the 8 November Heston training matrix on the tutorial's old task
  needed at least 30,000 updates to pass rank and coverage checks and still missed
  conditional checks (`NUMERICAL-EXPERIMENT-PLAN.md`, P4).

## 8. Checking it

### 8.1 Reference posteriors

**Heston, exact.**
- Likelihood: step 6, with Fourier prices; the pilot's normalized likelihood
  (`lab/src/sbi4finance/calibration/heston_pilot.py`), or the spread-only one
  (`heston_spread.py`), with the smooth covariance where it applies.
- Sampler: PyMC through `sample_deterministic_reference`, uniform priors, the
  log-likelihood as a `Potential`, `DEMetropolisZ` (gradient-free; NUTS would need
  validated gradients of the Fourier likelihood, per the MCMC policy). Initialization from
  the data and prior only (`reference.data_fit_initialization`), never from the truth or
  the network.
- Budget (Alex's current specification): 4 chains, 2,000 tuning steps and 2,000 retained
  draws, with one declared escalation to 6,000 and 12,000; at most 45 CPU-minutes per case.
  Hitting the cap records an incomplete reference, not convergence.
- Acceptance: rank-normalized $`\hat R \le 1.01`$, and bulk and tail effective sample
  sizes of at least 400, for parameters and reported products. Record seeds, chain and draw
  structure, sampler, initialization and exact versions.
- Cases: each SPX day, and 8 simulated cases spread over the prior and near SPX.
- Comparison point: under the pilot's independent-error model, the market case gave
  $`\tau`$ mean 0.61 vp, 90% interval 0.52–0.72 vp. The smooth model's $`\tau`$ is the
  amplitude of a correlated field and is expected to be larger.
- Status: seven of eight $`c`$-stratified joint references are ready (row 487
  unresolved); the fixed-$`c`$ broad, market-like and actual-market references pass; the
  boundary reference exhausted its 2,700-second limit and is inconclusive.

**Rough Bergomi, by importance sampling with the network as proposal.** Draw
$`\theta_s \sim q(\theta \mid y)`$, $`s = 1, \dots, 1{,}000`$, and weight

```math
\tilde w_s = \frac{p(\theta_s)\,\hat L(y \mid \theta_s)}{q(\theta_s \mid y)},
```

with $`\hat L`$ the Gaussian likelihood of step 6, Monte Carlo prices from a fixed-seed
pricer and $`\Sigma_{\text{MC}}`$. The fixed seed makes $`\hat L`$ a deterministic function
of $`\theta`$, so the reference is the exact posterior of that numerical model, which
differs negligibly from the continuous one when the Monte Carlo error is much smaller than
the total error. Diagnostics: effective sample size $`(\sum_s \tilde w_s)^2/\sum_s \tilde
w_s^2 \ge 200`$ and Pareto-smoothed $`\hat k \lt 0.7`$; otherwise the network is reported
as not validated for that case, after one retry with the network widened by a factor 1.5.
Cost 7–14 CPU-h per case at the step 4 target.

### 8.2 The checks, with their criteria

**V1 · Prior predictive.** Draw 10,000 surfaces from the prior with errors. Every SPX
day's quotes must lie inside the 1–99% band at every design point.

**V2 · Does the error model describe the residuals?** At the fitted $`(\theta, \tau, \ell)`$
with $`\Sigma = LL^\top`$, the whitened residuals $`u = L^{-1}r`$ are independent standard
normals if the error model is right. So $`u^\top u \sim \chi^2_n`$, with mean $`n`$ and
variance $`2n`$, and:
- $`u^\top u/n`$ must lie within $`1 \pm 2\sqrt{2/n}`$;
- the lag-1 autocorrelation of $`u`$ along strikes, within each expiry, must be below 0.3.

Apply the same test to the pilot's independent-error reference on its market case. Under
the spread-only law the analogue is $`R(\hat\theta)/n \approx c^2`$; Alex's market result,
$`c`$ pressed against 2, is this check failing.

**V3 · Calibration on the network's own simulator.** On 2,000 test simulations, for each
target, the rank of the truth among 127 posterior draws must be uniform on 0…127
($`\chi^2`$ test on 16 bins at the 1% level, Bonferroni across targets), and central 90%
intervals must cover 87–93%. Why ranks are uniform: if $`q`$ is the posterior, the truth
and the 127 draws are 128 exchangeable draws from the same distribution given $`y`$.
Necessary, not sufficient: a network that returns the prior passes, and Alex's fixed-$`c`$
network passed 5 of 5 rank screens with coverage 0.916–0.934 while failing the conditional
checks.

**V4 · Conditional accuracy against the references.** For each reference case:
- (network mean − reference mean) / reference standard deviation within ±0.3 for every
  parameter;
- ratio of 90% widths within [0.75, 1.25] (the repository's current margin; the
  9 November plan wrote [0.8, 1.25]);
- the error scale's median ($`\tau`$ or $`c`$) within ±20%;
- each product's 5%, 50% and 95% quantiles within ±10% of the reference 90% range;
- parameter dependence and analytic variance-product checks (Alex's additions).

Status: the fixed-$`c = 1`$ Heston network at 60,000 updates has broad-case physical widths
1.13–1.41 times the reference, mean errors of 0.78 reference standard deviations for
$`v_0`$ and −0.41 for $`\bar{v}`$; on the market-like case $`\kappa`$ and $`\bar{v}`$ pass
while $`\xi`$ and $`\rho`$ fail. It is not accepted.

**V4b · Diagnosing a network failure.** The pilot's $`\tau`$ network fails V4 on the market
day ($`\tau`$ mean 0.14 vp against the reference's 0.61); its continuation to 88,000 updates
hit a non-finite loss, cause unresolved. Two candidate causes, with different predictions:
- *input conditioning* (7.4): an unwhitened network cannot read the error scale. This is
  now demonstrated for $`c`$, and Alex notes that the $`\tau`$ checkpoints also failed
  their own simulator's conditional checks, where the error law is right by construction.
  Prediction: failure even on simulated data with independent errors, cured by
  whitening;
- *the error law*: trained on independent errors, the network meets smooth market
  residuals. Prediction: agreement on independent-error simulations, failure on
  smooth-error simulations and on the market.

The test compares network and reference, with and without whitening, on (1) simulated
cases with independent errors, (2) simulated cases with smooth errors of the same size,
and (3) the market day, and measures the correlation of neighbouring whitened residuals on
the market day. Repeat on the smooth-error network.

**V5 · Held-out quotes.** For a held-out quote with model price $`P_*(\theta)`$, vega
$`\nu_*`$ and half-spread $`h^P_*`$, the model error there and the residuals $`r`$ at the
design are jointly Gaussian. With $`k_* = \tau^2\,\nu_*\,\mathrm{diag}(\nu)\,K_\ell(\text{design}, x_*)`$
their covariance, conditioning gives the predictive draw for each posterior draw
$`(\theta, \tau)`$:

```math
y_* \sim \mathcal N\Big(P_*(\theta) + k_*^\top\Sigma^{-1}r,\ \ \tau^2\nu_*^2 - k_*^\top\Sigma^{-1}k_* + (c\,h^P_*)^2\Big),\qquad r = y - P(\theta).
```

The model error at the unused quote is predicted from the design's residuals (kriging),
and the quote's own half-spread is added. Report the coverage of 90% intervals by region
(beyond $`\delta = -5`$, beyond $`+2`$, unused expiries); expect 80–97%; not tuned. Under
the spread-only law the predictive is $`P_*(\theta) + c\,h^P_* Z_*`$, with no kriging term,
and it will undercover wherever the misfit is real.

**V6 · Market checks.** Reported per day and maturity, inside or outside the 90% band, not
tuned:
- the model-free variance-swap volatility against the posterior of $`\sqrt{K_{\text{var}}(T)}`$
  at 1, 3 and 12 months;
- the listed 2-year 90% put's implied volatility against its predictive band from V5. At
  $`\ln T = \ln 2`$, far from the design, the kriged model error reverts towards zero with
  variance up to $`\tau^2\nu_*^2`$.

**If a check fails.** V3 or V4: the real-day results come from the references, and the
network is shown on simulated data with its scores. V5 or V6: a finding about the model,
reported as such.

### 8.3 Population checks and conditional checks answer different questions

Alex's evaluations use the same 512 prior-generated observations for population checks
(recovery, coverage, SBC) and a few fixed observations with references for conditional
checks. The two disagree in instructive ways: pooled calibration can pass while single
posteriors are wrong, and some conditional widths move towards the reference with more
training while population coverage worsens. Both are required, and neither replaces the
other. Network–reference agreement on the market day tests the inference under the
assumed model; whether the model fits the market is a separate question for V2, V5 and V6.

### 8.4 Status of the networks

| Network | Data | Status |
| --- | --- | --- |
| Heston, five targets, $`c = 1`$ (the primary teaching target) | The pilot's 54 quotes | Not accepted: conditional widths 1.13–1.41 times the reference, mean errors up to 0.78 SD |
| Heston, six targets with $`c`$, unwhitened | Same | Ignores $`c`$; not accepted |
| Heston, six targets with $`c`$, whitened | Same | Learns $`c`$; undercovers; fresh-noise test next |
| Heston with independent $`\tau`$ (the pilot) | Same | Fails conditional checks, including on its own simulator |
| Heston and rough Bergomi with smooth $`\tau`$ (this plan) | Layout E | Not started; needs the settled recipe and the whitener |
| Rough Bergomi, any | Any new design | Gated on Heston acceptance (#84); read-only readiness only |

## 9. From the posterior to the products

### 9.1 Principle

A product's value is a functional of the model's law, $`P_j(\theta) = \mathbb{E}_\theta[\text{discounted payoff}]`$,
and the posterior induces its distribution (step 3). For each draw, simulate paths and
average; all draws share one set of normals (common random numbers), so Monte Carlo noise
shifts all draws together and is measured separately (9.9). $`\tau`$ does not enter: it
describes how listed options are observed, not the dynamics. The error model acts on the
products only through the posterior of $`\theta`$.

### 9.2 The same numerical model for fit and products

- **Heston:** the fit uses Fourier prices, the products Monte Carlo. Check on each day
  that Monte Carlo reprices the design's options within 0.05 vp of Fourier at the
  posterior median; if not, halve the time step.
- **Rough Bergomi:** the products use the same hybrid scheme and 2,016 steps per year as
  the bank.

### 9.3 Path simulation

- **Heston:** full-truncation Euler for $`V`$, with $`V^+ = \max(V, 0)`$ in the drift and
  the volatility,
  $`V_{t+\Delta} = V_t + \kappa(\bar{v} - V_t^+)\Delta + \xi\sqrt{V_t^+}\,\Delta W`$,
  with 4 sub-steps per business day ($`\Delta = 1/1008`$), as in `products.heston_paths`
  with finer steps; $`I`$ and $`J`$ stored at every business day.
- **Rough Bergomi:** `products.rough_paths` on the hybrid scheme, 2,016 steps per year,
  $`I`$ and $`J`$ stored at every business day (every 8 steps).
- **Normals:** antithetic pairs; the independent normals for $`J^{\perp}`$ (one per business
  day) are drawn once and shared by all draws.
- **Horizons:** 1 year for the swaps, the forward start and the barrier; 2 years for the
  2-year put check, with the curve family extended beyond 1 year (stated).

### 9.4 Variance swap (check)

$`K_{\text{var}}(T) = \mathbb{E}[I_T]/T`$ in closed form for both models (step 4),
reported as $`\sqrt{K_{\text{var}}}`$ in vol points at 1, 3 and 12 months and compared with
the model-free $`\sqrt{w(T)/T}`$ (V6, D4). A traded variance swap uses daily log returns
with a fixed annualization; the difference from the continuous strike is small at these
maturities and is stated.

### 9.5 Volatility swap

$`K_{\text{vol}}(T) = \mathbb{E}\big[\sqrt{I_T/T}\big]`$, estimated by the path mean of
$`\sqrt{I_T/T}`$ with antithetic pairs and a control variate (`products.volatility_swap`):
$`I_T/T - \mathbb{E}[I_T]/T`$ for rough Bergomi, whose discrete mean is exact; $`J_T`$ for
Heston, whose Euler scheme changes the discrete mean of $`I`$ but leaves $`J`$ with mean
zero.

**Convexity check.** Expanding $`\sqrt{x}`$ to second order around
$`m = K_{\text{var}} = \mathbb{E}[x]`$, with $`x = I_T/T`$,

```math
\sqrt{x} \approx \sqrt{m} + \frac{x - m}{2\sqrt m} - \frac{(x-m)^2}{8\,m^{3/2}}\quad\Longrightarrow\quad K_{\text{vol}} \approx \sqrt{K_{\text{var}}} - \frac{\mathrm{Var}(I_T/T)}{8\,K_{\text{var}}^{3/2}} .
```

The correction is set by the variance of realized variance, that is by the vol of vol,
which is why this product depends on the wings and the short maturities. Maturities 63
and 252 business days; prices in vol points.

### 9.6 Forward-start call, 6 months to 1 year

**Reduction.** With $`S_t = F(t)X_t`$ and $`m = F(T_2)/F(T_1)`$,
$`S_{T_2} - m\,S_{T_1} = F(T_2)\,(X_{T_2} - X_{T_1})`$, so

```math
\text{Value} = D(T_2)\,F(T_2)\ \mathbb{E}\big[X_{T_1}\,(X_{T_2}/X_{T_1} - 1)^+\big].
```

**Conditional formula.** Given the path of $`W`$, $`X_{T_1}`$ depends on $`W^{\perp}`$ over
$`[0, T_1]`$ and $`X_{T_2}/X_{T_1}`$ on $`W^{\perp}`$ over $`[T_1, T_2]`$: they are
conditionally independent. Each is conditionally lognormal (notation section), and
$`\mathbb{E}[X_{T_1} \mid W] = e^{\rho J_{T_1} - \frac12\rho^2 I_{T_1}}`$. Hence
(`products.forward_start_call_spot`, moneyness 1)

```math
\mathbb{E}\big[\,\cdot \mid W\big] = e^{\rho J_{T_1} - \frac12\rho^2 I_{T_1}}\ \mathrm{BS}_{\text{call}}\Big(F' = e^{\rho(J_{T_2} - J_{T_1}) - \frac12\rho^2(I_{T_2} - I_{T_1})},\ 1,\ w = (1-\rho^2)(I_{T_2} - I_{T_1})\Big).
```

Controls: the forward difference
$`e^{\rho J_{T_2} - \frac12\rho^2 I_{T_2}} - e^{\rho J_{T_1} - \frac12\rho^2 I_{T_1}}`$
(mean zero) and the integrated-variance control of 9.5.

**Why it is the key exotic.** Its value depends on the forward smile at $`T_1`$, which
today's smile does not determine. Heston's forward skew flattens through mean reversion;
rough Bergomi's persists. This is where D7 expects the models to disagree. Report in
forward units at $`T_2`$ and in index points ($`\times\,D(T_2)F(T_2)`$).

### 9.7 Down-and-in put, 1 year, strike 100%, barrier 70%, daily closes

**Payoff** at $`T`$ = 1 year: $`(K - S_T)^+\,\mathbf{1}\{\min_i S_{t_i} \le B\}`$, with
$`K = S_0`$, $`B = 0.7\,S_0`$ and $`t_i`$ the 252 business-day closes.

**The spot path from the stored quantities.** Given $`W`$, $`J^{\perp}`$ is a Gaussian
process with independent increments of variance $`\Delta I_i = I_{t_i} - I_{t_{i-1}}`$,
so, exactly given the variance path,

```math
\ln X_{t_i} = \rho\,J_{t_i} + \sqrt{1-\rho^2}\,J^{\perp}_{t_i} - \tfrac12\,I_{t_i},\qquad J^{\perp}_{t_i} = J^{\perp}_{t_{i-1}} + \sqrt{\Delta I_i}\ \zeta_i,\qquad \zeta_i \sim \mathcal N(0,1)\ \text{iid}.
```

Then $`S_{t_i} = F(t_i)X_{t_i}`$, with carry from the parity forwards. The barrier test is
$`X_{t_i} \le B/F(t_i)`$: in forward units the barrier falls as the forward rises.

**Estimator.** The mean of $`D(T)(K - S_T)^+\mathbf{1}\{\text{hit}\}`$, with antithetic pairs
over the variance normals and over $`\zeta`$, and the vanilla put $`(K - S_T)^+`$ as a
control variate with its exact value (Fourier for Heston; the conditional estimator of
step 4 on the same paths for rough Bergomi) and a regression coefficient. The per-path
identity down-and-in + down-and-out = vanilla is checked.

**New code** in `lab/src/sbi4finance/calibration/products.py`, with unit tests:
1. *Black–Scholes limit.* With $`\xi \to 0`$ and $`v_0 = \bar{v} = \sigma^2`$, compare
   with the closed-form continuously monitored down-and-in put at the shifted barrier
   $`B\,e^{-0.5826\,\sigma\sqrt{\Delta t}}`$ (the Broadie–Glasserman–Kou correction for
   discrete monitoring; $`0.5826 = -\zeta(1/2)/\sqrt{2\pi}`$). Pass within 3 standard
   errors plus 0.5% of the price.
2. *Monitoring frequency.* Halving it must move the price in the direction and by about
   the size the correction predicts.
3. *Determinism.* The same seed gives the same price; the antithetic estimator is unbiased
   against a long reference run.

**Why it matters.** It is the downside barrier inside autocallables, among the most common
structured products, and the issuing bank is long it. It depends on the deep put wing and
on the spot–volatility dynamics ($`\rho`$, $`\xi`$): the put-heavy design is for it.

### 9.8 The listed 2-year 90% put (check)

Model price at the listed expiry ($`T \approx`$ 2.1 years) and strike: Fourier for Heston;
for rough Bergomi the conditional estimator on 2-year paths, with the curve family
extended beyond 1 year. Convert to the model's implied volatility at
$`(\ln(K/F(T)), T)`$ and compare with the market's, ± its half-spread, against the
predictive band of V5 and V6: extrapolation in maturity, checked against a listed price.

### 9.9 Monte Carlo accuracy

- **Paths:** choose $`N`$ per product and model so that the standard error at the
  posterior median is at most 2% of that product's posterior 90% range; expect 20,000 to
  100,000 antithetic paths for Heston and 10,000 to 50,000 for rough Bergomi.
- **Draw-sampling error of quantiles:** bootstrap over the $`S`$ draws. For
  $`S = 1{,}000`$, the 90% quantile's error in probability level is about
  $`\sqrt{0.9 \times 0.1/1000} \approx 0.01`$.
- **Path error of quantiles:** rerun all draws with an independent seed and report the
  differences.
- **Requirement:** both errors at most 2% of the 90% range; otherwise raise $`N`$ or $`S`$.

### 9.10 Summaries, prudent value and the book

Mark, prudent value and reserve as in step 3. The example book:

| Position | Size | Why |
| --- | --- | --- |
| Down-and-in put, 1 year | long 1 | the issuer's side of autocallables |
| Forward-start call, 6 months to 1 year | long 1 | as in the plan |
| Volatility swap, 1 year | short 1 | as in the plan |

The products are in different units: the put and the forward start in index points per
unit of index, the volatility swap in vol points per unit of vega notional. The book needs
one currency unit, so the volatility swap's vega notional must be fixed when the book is
built; the plan does not yet specify it. Report the book's prudent value
$`q_{0.10}(B^{(s)})`$, its reserve, the diversification ratio against the sum of the
standalone reserves, compared with the Regulation's fixed 0.5, and the correlation matrix
of the product values across draws.

### 9.11 Baselines

- **Point calibration** (step 2.2): least squares in implied volatility from 8 starts,
  and the maximum-likelihood version with the error scale estimated. A value with no
  range.
- **Delta method** (step 2.4), with product gradients by central differences under common
  random numbers.
- **Expiry-drop refits** (step 2.4).

## 10. What the results will show, and what they will not

### 10.1 The demonstrations

| | Computation | Output |
| --- | --- | --- |
| D1 | Per day, model and product: posterior 5/50/95%, mark, point price, delta-method range, expiry-drop range, reserve | One table per model; a figure of ranges by product. Listed options reprice tightly, the forward start is wide, the 2-year put is set against its listed price |
| D2 | Book reserve against the sum of standalone reserves; correlations of products across draws | Table: book against sum, diversification ratio, against the Regulation's 0.5 |
| D3 | Simulation study. Heston: 1,000 prior draws plus 300 near SPX, 200 posterior draws each. Rough Bergomi: 300 cases, 100 draws each. Per case: truth, observation with errors, network posterior, product distribution, true product price under an independent seed at 4 times the paths | Coverage of the 90% product ranges with binomial ±2 standard errors (±1.9% on 1,000 cases, ±3.5% on 300), against the delta method, the refits (300 cases) and the point calibration (0%) |
| D4 | One day, Heston reference, under: half-spreads only ($`c = 1`$); plus independent $`\tau`$ (the pilot's); plus smooth $`\tau`$ (adopted); the adopted model at $`c`$ = 0.5 and 2 | Variance-swap bands against the model-free value; product ranges as the error model grows. Alex's same-observation $`c`$ = 0.5/1/2 references are the first row's sensitivity |
| D5 | Posterior of $`\tau`$ per day and model; map of the posterior-mean residual $`\bar r = y - \mathbb{E}[P(\theta) \mid y]`$ on the design, in vol points ($`\bar r/\nu`$) | Where each model misfits |
| D6 | 9–12 November: marks and ranges per product; daily point calibrations | Day-to-day changes of the marks against the widths of the ranges; four days, illustrative; depends on the decision of step 5.4 |
| D7 | Heston against rough Bergomi on the same $`y`$, $`h`$, $`\ell`$ and design: $`\tau`$, V5 coverage and held-out log predictive density, V6, product ranges side by side | Model comparison; stacking weights on held-out quotes and the weighted mixture of product draws, stated as a combination, not a reserve |

The stacking weights maximize the held-out log predictive density of the mixture over the
simplex $`\Delta`$:

```math
\hat w = \arg\max_{w \in \Delta}\ \sum_i \log \sum_m w_m\,p_m(y^*_i \mid y).
```

### 10.2 What the ranges are not

- They are parameter uncertainty within each model, under the stated error model. The
  model-error term covers misfit on quoted points, not dynamics that the quotes cannot
  reveal, such as the forward smile. D7 shows how far the two models disagree; neither
  range is a complete model-risk reserve.
- On real data, the true value of a forward start or a barrier is never observed; the
  real-data checks are V5 and V6.
- Hedge ratios under the posterior are not shown.
- A posterior-based AVA is a proposal, not established practice.

### 10.3 Limitations and known risks

1. **The smooth error can absorb parameter signal.** A flexible model error and the
   parameters can trade off. The correlation lengths are fixed from the first two days'
   fits; wider posteriors are reported as the honest result; V2 and V4 detect problems.
2. **The error law is assumed:** Gaussian, with a squared-exponential correlation.
   $`\tau`$ cannot separate structural model error from data or convention problems.
3. **The networks may miss the references,** as every Heston network so far has; the
   real-data results then come from the references.
4. **Input conditioning.** A network that must read an error scale needs whitened inputs;
   the whitener is tied to one ordered design and must be refitted for each new design
   or model.
5. **Fixed-noise overfitting.** Training rows that keep one noise draw let a whitened
   network memorize it; this plan's fresh draws avoid it, and Alex's next experiment tests
   it.
6. **Pooled checks pass while single posteriors are wrong;** V4 is binding.
7. **Rough Bergomi's bank carries Monte Carlo error** inside its simulated observations;
   at the step 4 target this makes posteriors slightly conservative, not biased. Under the
   spread-only law the target tightens and the cost rises steeply (step 4).
8. **Fourier accuracy at extreme parameters,** large $`\kappa`$ and $`\xi`$, for short-dated deep puts: the
   rung checks apply, and rejected draws are recorded.
9. **Event risk inside the shortest maturity** (a US inflation release that week): the
   smooth term absorbs it, and the text says so.
10. **Four days** illustrate day-to-day behaviour; they are not a history.
11. **The runtime's 63-contract assumptions** (`shared/full_surface.py`,
    `full_surface_teaching.py` and the notebooks' 7 × 9 reshapes) must accept a declared
    design of any size.
12. **Raw data:** each close must be fetched that evening; files stay outside Git,
    recorded by SHA-256.
13. **The public site** carries no unreleased results (AGENTS.md).

## Where we are, and what remains

### Settled across the tracks

- SPX: European PM-settled SPXW quotes, actual contracts, no interpolation on the
  design's own day, carry anchors disjoint from fitted quotes.
- Price-space Gaussian errors with the half-spread as the scale (a convention), a known
  $`c`$ as the primary baseline, negative simulated prices kept.
- Heston priors with $`\kappa`$ free and $`\xi`$ up to 3.
- The coverage principle; the pilot's 54 quotes give way to an actual-quote layout built
  from the layout E targets (this layer's check and Alex's audit agree).
- Whitened network inputs, fitted on training data only, persisted with the model and
  applied once.
- References with PyMC under the MCMC policy; population and conditional checks both
  binding.

### Running: Alex's Heston work order (PRs #82 and #85)

| Unit | Work | Status, 10 November evening |
| --- | --- | --- |
| H1 | Spread-only likelihood, task identities, known-$`\theta`$ oracle for $`c`$ | Done |
| H2 | Reuse the 54 clean prices; references | Done: seven of eight joint references ready; fixed-$`c`$ boundary reference inconclusive |
| H3 | Checkpoint audit; fresh-noise test; fixed $`c = 1`$ teaching fit | Audit done (no checkpoint calibrated); fresh-noise test next; teaching fit not accepted |
| H4 | Revised actual quotes; price only changed contracts | Audit done (62 of 63 targets; no 1-year PM expiry; last call at +1.5); pricing gated |
| H5 | Repeat the accepted recipe on the revised layout | Not started |
| H6 | Freeze the recipe; fresh confirmation; promotion | Not started |

The rough Bergomi refresh (#84) is limited to read-only readiness until Heston is
accepted; the flexible-input track (#83) is a specification.

### Open decisions

1. **The error model for the market application** (step 6.8): spread only, or spread plus
   smooth model error. Evidence: $`c`$ at its bound on the market; misfit about 17 times
   the half-spread and smooth; $`\tau`$ identifiable to 11–17%; a spread-only rough
   Bergomi needs 16 to 100 times more Monte Carlo paths. Proposal: stage 1 spread only,
   stage 2 with model error for the product ranges.
2. **Days 2 to 4** (step 5.4): the reader, a design per day, or references only. For the
   tutorial, references only on days 2 to 4 is the cheapest.
3. **The long end and the far call:** an AM-settled 1-year SPX contract or the 11-month PM
   expiry; the +2 call or the +1.5 revision.
4. **Time convention:** business days over 252 (the plan, rough grids) or ACT/365F (the
   frozen design).
5. **What the tutorial shows on 14 November:** which of D1 to D7 fit the slots. Alex's
   work orders list the products and the prudent-value discussion as separate decisions.
6. **Slot 4:** Stefan's PR #86 makes the current notebook a synthetic, 20-minute workflow
   on the old 63-point grid and old noise rule, without the SPY sections; the SPX version
   of 13.1 depends on an accepted network (H3) or on showing references on the real day.
7. **The book's volatility-swap notional** (9.10).

### If the network is not accepted by 14 November

The plan's failure policy applies: real-day results come from the reference posteriors
(Heston MCMC is available for every day at 15–40 CPU-minutes), and the network is shown on
simulated data with its V3 and V4 scores. The whitening result is itself teachable: a
network can miss information that is plainly in the data, pooled SBC will not notice, and
a change of input coordinates fixes it.

### Corrections to the current text

Needed whatever else changes; verify line numbers when editing.

| Where | Now | Replace with |
| --- | --- | --- |
| `tutorial/notebooks/slot4_heston_bayesflow.py:155-158` | noise "specified in implied volatility, 0.3 / 1.0 points" | the error model of step 6; the old rule described as an assumed allowance, about 8 times the bid–ask, whose vol-point meaning varies |
| `slot4_heston_bayesflow.py:943-975` | the wing caveat in a cell stopped by `mo.stop` | superseded by steps 1–6 in the rebuilt notebook |
| `slot4_heston_bayesflow.py:1028-1032` | "variance swaps sit below the market … precise, and wrong" | D4, D5 and D7 |
| `slot4` §8 (from line 536), `slot5_calibration.py` §7 (from line 563) | historical SPY workflows | the SPX application, for both models; #86 already removes slot 4's SPY sections |
| `slot5_calibration.py:155-163` | the same noise text | as for slot 4 |
| `slot5_calibration.py:650-656` | "variance swaps match SPY's by construction" as the reason to switch models | the curve is inferred from the same quotes for both models; the reasons to compare are the skew's term structure and D7 |
| `tutorial/curriculum/00-overview.md:48`; `04-slot4…md:26, 46, 63, 68`; `05-slot5…md:29, 49, 115, 131`; `06-slot6…md:28, 61` | the variance-swap gap as the case against Heston; "the design's noise" | aligned with the above |
| `tutorial/CLAIMS.md` | claims tied to the old rule or the gap | regenerated through `pixi run claims` |
| `lab/src/sbi4finance/calibration/chain.py:147-149, 264-265` | half-spread converted with the quote's own vega, then back to price at 20% vega | not used by the new tasks (step 1.4 keeps vol units) |
| `tutorial/src/sbi4f_tutorial/shared/{full_surface,full_surface_teaching}.py`; the notebooks' 7 × 9 reshapes | 63-contract checks and plots | accept a declared design of any size |
| `tutorial/src/sbi4f_tutorial/shared/inference.py:36-40` | `use_sampler` reportedly reverses its time grid under BayesFlow 2.1 (noise session) | verify; fix if any live path calls it (`full_surface.sample` does not) |

Assets enter `tutorial/assets/` only through the lab tools (`lab/studies/accept.py`, with a
bundle format for each new task, then `pixi run manifest` and `pixi run claims`). Before
release: tutorial `pixi run test`, `test-notebooks` and `check`; lab `pixi run test` and
`release-check`.

### Tutorial integration

**Slot 4, Heston on SPX.** Rewritten in place; `pixi run heston-options` unchanged.
1. The job: what a desk calibrates each day, and what Bayes and SBI add.
2. SPX data and the design: the smile, where the quotes sit, which region pins which
   parameter (step 5).
3. The error budget: bid–ask against model error; the error model (step 6).
4. The model: Heston, its priors and simulator, with the code visible, and one training
   pair.
5. The amortized posterior on a synthetic case: SBC and coverage (V3); the network against
   the exact reference (V4); whitening, and why it is needed.
6. SPX, four days: posterior, $`\tau`$, the residual map (D5), the held-out and market
   checks (V5, V6).
7. Products: D1–D4 and D6.
8. Scope: what the ranges cover and what they do not (step 10).

Retire the historical SPY section from the live path; old assets stay until a
dependency-checked retirement batch. Current state: Stefan's #86 (synthetic workflow,
about 20 minutes) awaits review.

**Slot 5, rough Bergomi on SPX.** Rewritten in place; `pixi run rough-options` unchanged.
1. Why another model: SPX's at-the-money skew term structure (`smile_statistics`,
   `skew_power_law`) and Heston's residual map from slot 4.
2. The model: its curve family, Monte Carlo pricing, the accuracy target.
3. Inference: the same design and error model, the network, the importance-sampling
   reference.
4. SPX: posterior, $`\tau`$, residual map, held-out and market checks.
5. Products under rough Bergomi, then D7: the two models side by side and their
   combination.
6. Scope.

Retire the historical SPY studies from the live path. Current state: Alex's branch
`lesson/rough-options-workflow-20261107` (no PR yet) makes the rough simulator and both
neural workflows explicit and propagates calibration uncertainty; production on a new
design is gated (#84).

**Slot 6, priors and misspecification.** D4 is the material for the noise part: half-spreads
only, independent $`\tau`$, smooth $`\tau`$, and $`c`$ from 0.5 to 2; a slider maps to $`c`$
and to the size of $`\tau`$.

### Compute

| Item | Estimate |
| --- | --- |
| Heston bank | exists for the 54 quotes; about 1.3 CPU-h per 100,000 for layout E |
| Rough bank, 100,000 | 660–1,300 CPU-h at the step 4 target (pilot of 200 surfaces first); far more under the spread-only law |
| Networks | Heston about 100 s per 30,000-update fit on one L40S; rough Bergomi of the same order; 10–20 fits per model |
| Heston references, 12 posteriors | 3–8 CPU-h (at most 45 CPU-min each) |
| Rough references, 12 cases | 85–170 CPU-h |
| Products per day, both models | under 2 CPU-h |
| D3 | Heston about 11 CPU-h; rough Bergomi about 12 CPU-h |
| Data | four SPX closes, about 14 MB each, outside Git |

### What stays right in the current implementation

A fixed observation design shared by simulator, network and reference;
out-of-the-money quotes in forward units, $`\sqrt T`$ scaling, and maturities from 1 week
to 1 year; spline reading without extrapolation (0 points outside quoted strikes);
numerical accuracy tied to the error scale; the recovery, SBC and reference checks. The
synthetic main path is internally consistent; what is wrong there is the stated meaning of
the noise.

### Evidence index

- [`design-and-error/summary.json`](../../evidence/pilot-results/2026-11-09/design-and-error/summary.json)
  and its scripts: the numbers behind steps 1, 5 and 6 (first checks).
- The noise session, 9 November: SPY error budget, exact fits under three error models,
  test-network coverage, variance-swap bands under smooth error. Not committed.
- Alex's study, PR #85 (draft), on branch `feat/heston-spread-baseline-20261109`, under
  `lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/`: the spread-only
  likelihood and fixed-$`c`$ development (`README.md`), the noise-scale study
  (`noise-scale/`), the larger-bank comparison and conditioning diagnostic
  (`noise-scale/capacity-study/`), the paired whitening pilot
  (`noise-scale/capacity-study/whitening-pilot/`) and the checkpoint replay
  (`…/whitening-pilot/checkpoint-replay/`), the grid audit (`grid-coverage/`).
- The handover, PR #87: the investigation brief and portable recovery inputs
  (`noise-scale/README.md`, `recovery-inputs.{json,npz}`), and Stefan's notebook
  `lab/notebooks/heston_6_parameter_bayesflow.ipynb`.
- Work orders: `lab/plans/HESTON-NOISE-GRID-PLAN.md` (#82),
  `HESTON-FLEXIBLE-INPUTS-PLAN.md` (#83), `BERGOMI-OBSERVATION-REFRESH-PLAN.md` (#84).
- The pilot, branch `study/heston-european-observations-20261109`, evidence through
  `78cf9de`: the design `designs/put_heavy.json` (checked in step 5.2), the $`\tau`$
  networks and their failure records under `heston-observations/training-study/`.
- Tutorial assets: `results/calibration/spy/surface.json` (S8);
  `data/heston-spy-exact.json`; `results/calibration/{heston,rough-pilot}/s1/bank-quality.json`
  (costs).
- `lab/plans/NUMERICAL-EXPERIMENT-PLAN.md`, P4: the 8 November Heston training matrix.
- Regulation: Commission Delegated Regulation (EU) 2016/101, Articles 9, 10 and 11 and
  the Annex; CRR (Regulation (EU) No 575/2013) Article 34; IFRS 13. The course's
  [practice sources](../../../course/practice-sources.md) record what practice is
  documented.

## Back to the goals, the audience and the practitioner

**Goals.**
1. *Marks and ranges.* Steps 3 and 9 define them, D1 and D2 report them, and step 9's
   numerical requirements make them reproducible. On 10 November the product code (the
   barrier estimator and its tests) and the rough bank are not yet built, and no network
   is accepted; the references can carry the real-day results.
2. *Fast and testable.* Steps 7 and 8. The amortization identity makes it possible; the
   whitening result shows what it takes: a network can ignore an identifiable error scale
   until its inputs are well conditioned, and then overfit fixed noise unless the noise is
   redrawn. V3, V4 and D3 test the result; none of the Heston networks passes V4 yet.
3. *The limits.* Step 10 states them, and step 6 states the choice that most affects them:
   whether the market ranges include the model's measured misfit.

**Audience.** Researchers in AI and statistics see a real inverse problem in which two
things, not the network's size, decide whether the posterior is right: the error model
(step 6) and the conditioning of the network's inputs (step 7), together with a chain of
checks that separates approximation error from model error (step 8). Practitioners see
calibration ranges, book aggregation and a model comparison from one consistent
calculation, mapped onto the prudent-valuation rules, with a plain statement of what the
ranges leave out. Both should leave able to say, for their own model, whether they need
this, what it costs, and how to check it.

**Practitioner.** A desk and its product control would get, every day: a mark and a range
for every product and for the book, with tested coverage; a measure of each model's misfit
($`\tau`$ and the residual map) that flags when a model stops describing the market; and a
comparison of two models on the same information. In the terms of the rules, the
posterior's range is an Article 9 input, the comparison of models belongs to Article 11,
and the joint draws measure the offset between products that the Annex fixes at 50%. They
would still have to choose the models, handle the model risk beyond one model's parameters
and the other AVA categories, decide how to hedge, take the method through model
validation, and have their regulator accept a posterior as the basis of a prudent value.

# Option application on SPX · Layer 3: full detail

*Layer 3 of 3 · reference · [Index](README.md) · [Layer 1](1-overview.md) ·
[Layer 2](2-walkthrough.md) · [Glossary](glossary.md) · Results as of 13 November 2026*

Each step gives the definitions, the derivations, the numbers behind each choice and the
implementation: code paths, settings and costs. Every quantity is defined by a formula
where it first appears, and the symbols are those of the tutorial's Heston notebook,
`lab/notebooks/heston_spx_calibration.py`
([PR #93](https://github.com/fmuia/sbi4finance/pull/93)). The Heston results are those the
notebook shows: they come from its run record and from the study in
[PR #85](https://github.com/fmuia/sbi4finance/pull/85). The rough Bergomi data generation
is in [PR #84](https://github.com/fmuia/sbi4finance/pull/84) and
[PR #88](https://github.com/fmuia/sbi4finance/pull/88). Every figure is drawn by
[`figure-sources/make_figures.py`](figure-sources/make_figures.py) from
[`figure-sources/data.json`](figure-sources/data.json) and
[`figure-sources/notebook.json`](figure-sources/notebook.json), whose sources and their
SHA-256 hashes are listed in the files; [Evidence and figures](#evidence-and-figures) says
which file each number comes from.

## Goals

**1. A value and a range for products that no market prices.** Under Heston, fitted to
63 SPX quotes with an error model scaled by their half-spreads, report for each product
the posterior 5%, 50% and 95% quantiles, the mark, the prudent value at 90% and the
reserve. The products are a forward-starting variance swap on two windows, a forward-start
call from 3 to 6 months and a 3-month volatility swap. Rough Bergomi, fitted to the same
quotes with the same error model, follows; a 1-year down-and-in put and an example book
are next (10.3).

**2. SBI makes this fast and testable.** The amortized posterior must be calibrated on
simulated surfaces, agree with reference posteriors where those can be computed, and
return a posterior in under a second, so that ranges can be produced every day and their
coverage tested on hundreds of simulated surfaces.

**3. The model and the limits.** Check the model on the market quotes, measure how much
its misfit moves the products, and label each output with what it covers: parameter
uncertainty within one model, under a stated error model.

**Success criteria.**
- *Numerical:* option prices refined until two grids agree within 0.01 of the quote's
  half-spread (step 4); product prices simulated with the same random numbers for every
  posterior draw, so that their spread is the parameters' (step 9).
- *Inference:* coverage and rank tests on simulated surfaces; agreement with the reference
  posteriors (step 8). Where the network does not agree, results come from the reference.
- *Model:* the fit to the market is measured and reported, and the products are priced
  under refits to parts of the quotes as well as under the posterior (steps 6, 8 and 9).

**Not goals.** Hedge ratios under the posterior; a complete model-risk reserve; a claim
that regulators accept a posterior-based AVA.

## Audience

The ICAIF '26 tutorial (two hours, 14 November) speaks to AI and statistics researchers
who know posteriors, MCMC and neural networks but not options, and to finance
practitioners who know options and calibration but not SBI. For Heston on SPX the material
is a notebook, `lab/notebooks/heston_spx_calibration.py`, which runs steps 4 to 9 with an
exact posterior to check against and closes with the model's misfit (steps 6 and 10);
10.4 maps its sections onto the steps. Rough Bergomi on SPX follows the same steps once
its simulations are complete.

Background for each step, from the [course](../../../course/README.md):

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

The course's worked examples use SPY, a symmetric grid of 63 points and a fixed noise
level; the definitions below are the ones used for SPX. Two symbols differ from the
course: there $`\theta`$ is Heston's long-run variance and $`\sigma_{\text{ref}} = 0.20`$;
here, as in the tutorial's notebook, $`\theta`$ is the vector of unknowns, the long-run
variance is $`v_\infty`$, and $`\sigma_{\text{ref}} = 0.16`$.

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
  appropriate modelling and calibration approaches" (Article 11(3)). A second model, or
  calibrations of one model to different quotes (9.6), belong here.

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

with $`\alpha = 50\%`$. The factor is a fixed allowance for the fact that many exposures
do not reach their 90% points at once. When the fair value and EV are both the posterior
mean, the two methods coincide: each exposure counts for half of its standalone
adjustment. The joint posterior draws of a book would measure the actual offset, for
comparison (3.3); the regulatory factor stays fixed.

**Current practice.** One point calibration per day; ranges from alternative calibrations
and models, from consensus services that pool banks' prices (some publish 10th and 90th
percentiles for prudent valuation), or from expert judgment. The hard inputs are those
nobody quotes: the forward smile, the deep wings, the vol of vol. The course's
[practice sources](../../../course/practice-sources.md) list what is documented.

**What we propose.** The spread of a posterior over the parameters of a model calibrated
to quotes, carried through every product, is an Article 9 input; the spread across models,
or across calibrations of one model to different quotes, is an Article 11 question. That a posterior is an
acceptable basis for an AVA is our proposal, not established practice.

So the practitioner needs, for every product, a mark and a defensible range; for the
book, a range that accounts for offsets between products; and a statement of what the
range does not cover.

## Notation and conventions

**Counts and indices.**

| Symbol | Value | Meaning |
| --- | --- | --- |
| $`n`$ | 63 | quotes, indexed by $`i`$ |
| $`d`$ | 6 for Heston, 7 for rough Bergomi | unknowns: the model's parameters and the multiplier $`c`$ |
| $`N`$ | 300,000 | simulated training rows, indexed by the superscript $`(n)`$ |
| $`N_w`$ | 10,000 | training rows used to fit the whitening |
| $`M`$, $`L`$ | 512, 512 | simulated test surfaces (index $`m`$) and network draws on each (index $`l`$) |
| $`S`$ | 1,000; 4,096 in 8.3; 40 in 9.6 | draws used from one posterior, indexed by the superscript $`(s)`$ |

A **surface** is one set of 63 prices at the design's contracts, simulated or observed.

**Time.** Valuation on 9 November 2026 at 20:13:37 UTC (15:13 New York). Each expiry
settles at 16:00 New York; $`T`$ is the time to it in years of 365 days (ACT/365F, actual
seconds over $`365 \times 86{,}400`$).

**Market data.** Quote $`i`$ is a put or a call with strike $`K_i`$ and expiry $`T_i`$.
From its bid and ask,

```math
y_i = \frac{\text{ask}_i + \text{bid}_i}{2}\ \ \text{(mid)},\qquad h_i = \frac{\text{ask}_i - \text{bid}_i}{2}\ \ \text{(half-spread)},
```

both in index points; $`y = (y_1, \dots, y_n)`$ and $`h = (h_1, \dots, h_n)`$.

**Spot, forward, discount.** $`S_t`$ is the index, $`F(T)`$ the forward for expiry $`T`$,
$`D(T)`$ the discount factor. The index in forward units is $`X_t = S_t/F(t)`$:

```math
\frac{dX_t}{X_t} = \sqrt{V_t}\,dW^F_t,\qquad X_0 = 1,\qquad W^F = \rho\,W^V + \sqrt{1-\rho^2}\,W^{\perp},
```

where $`V_t`$ is the instantaneous variance, the Brownian motion $`W^V`$ drives $`V`$,
and $`W^{\perp}`$ is a Brownian motion independent of $`W^V`$. For one expiry, the
notebook's forward price is $`F_t = F_0\,X_t`$.

**Moneyness.** For strike $`K`$ and expiry $`T`$,

```math
k = \ln\frac{K}{F(T)}\ \ \text{(log-moneyness)},\qquad \delta = \frac{k}{\sigma_{\text{ref}}\sqrt{T}}\ \ \text{(standardized moneyness)},\qquad \sigma_{\text{ref}} = 0.16 .
```

$`\delta`$ measures the distance of the strike from the forward in units of
$`\sigma_{\text{ref}}\sqrt{T}`$, the standard deviation the log index would have at expiry
under a 16% volatility. It only places and labels strikes: prices use each contract's
actual strike, forward, discount factor and expiry. The design's files call it `z`.

**Implied volatility and vega.** $`\sigma^{\text{imp}}_i`$ is the volatility at which the
Black–Scholes formula gives the price of quote $`i`$; its vega is
$`\nu_i = \partial(\text{price})/\partial\sigma`$ at that volatility. One vol point ("vp")
is 0.01 of volatility. A half-spread in vol points is $`h_i/\nu_i`$.

**Unknowns and model prices.**

| Model | $`\theta`$ |
| --- | --- |
| Heston | $`(v_0, \kappa, v_\infty, \sigma_v, \rho, c)`$ |
| Rough Bergomi | $`(H, \eta, \rho, \xi_s, \xi_l, \lambda, c)`$ |

$`\theta \in \mathbb{R}^d`$ collects everything inferred: the model's parameters and, as
its last component, the multiplier $`c`$ of the error model. $`P_i(\theta)`$ is the
model's price of quote $`i`$, in the units of $`y_i`$; it does not depend on $`c`$.
$`P(\theta) = (P_1(\theta), \dots, P_n(\theta))`$.

**Errors.** The error model of step 6 is

```math
y_i = P_i(\theta) + c\,h_i\,\epsilon_i,\qquad \epsilon_i \sim \mathcal N(0, 1)\ \text{independent},\qquad r_i(\theta) = \frac{y_i - P_i(\theta)}{h_i} ,
```

with $`r_i(\theta)`$ the **scaled residual**: the miss of the model on quote $`i`$, in
half-spreads.

**Products.** Product $`j`$ has payoff $`\Pi_j`$ and value
$`G_j(\theta) = D\,\mathbb{E}_\theta[\Pi_j]`$, a function of the parameters. Under draw
$`s`$ of the posterior its value is $`G_j^{(s)} = G_j(\theta^{(s)})`$; a book holds
$`\omega_j`$ units of product $`j`$ (negative when short).

**The network** (step 7): scaled quotes $`x = y/h`$, whitened quotes
$`\tilde x = W(x - \mu)`$, standardized unknowns $`\tilde\theta`$, flow state $`z_t`$,
velocity network $`v_\phi`$ with weights $`\phi`$, and its posterior
$`q_\phi(\theta \mid y)`$. They are defined in 7.4 and 7.6.

**Integrated quantities along a path.**

```math
I_t = \int_0^t V_s\,ds,\qquad J_t = \int_0^t \sqrt{V_s}\,dW^V_s,\qquad J^{\perp}_t = \int_0^t \sqrt{V_s}\,dW^{\perp}_s ,
```

so that, by Itô's formula for $`\ln X`$,

```math
\ln X_t = \rho\,J_t + \sqrt{1-\rho^2}\,J^{\perp}_t - \tfrac12\,I_t,\qquad J^{\perp}_t \mid W^V \sim \mathcal N(0, I_t).
```

The conditional law holds because, given the whole path of $`W^V`$, $`J^{\perp}_t`$ is an
integral of a known function against an independent Brownian motion. This structure is
exact for both models and underlies every Monte Carlo estimator of steps 4 and 9.

**Parameters and their code names.**

| Symbol | Meaning | Code name |
| --- | --- | --- |
| $`v_0`$ | Heston initial variance $`V_0`$ | `initial_variance` |
| $`\kappa`$ | Heston speed of mean reversion, per year | `kappa` |
| $`v_\infty`$ | Heston long-run variance | `theta` |
| $`\sigma_v`$ | Heston volatility of variance ("vol of vol") | `vol_of_vol` |
| $`\rho`$ | correlation of the shocks to index and variance (both models) | `rho` |
| $`H`$, $`\eta`$ | rough Bergomi Hurst exponent and vol of vol | `hurst`, `eta` |
| $`\xi_s`$, $`\xi_l`$, $`\lambda`$ | rough Bergomi variance curve: short level, long level, decay | `short_variance`, `long_variance`, `decay` |
| $`c`$ | multiplier of the half-spreads in the error model | `c` |

## 1. What has a price, and what does not

### 1.1 The data

- **Source.** Cboe delayed quotes for SPX options (`_SPX.json`), snapshot of 9 November
  2026; raw file 14.1 MB, SHA-256 `8c730e11…9458`, kept outside Git. The index stood at
  7,571.54.
- **Series.** SPXW, European and cash-settled on the closing value at 16:00 New York on
  the expiry, which list expiries on almost every trading day. Early-close, same-day and
  non-SPXW markets are excluded.
- **Quality.** A quote is usable if its bid is positive, its ask is above its bid and its
  half-spread is at most 25% of its mid.

### 1.2 Discount and forward

**Put–call parity.** A call minus a put with the same strike and maturity pays
$`(S_T-K)^+ - (K-S_T)^+ = S_T - K`$ at $`T`$, whatever happens, so its price is that of a
forward contract:

```math
C(K,T) - \mathrm{Put}(K,T) = D(T)\,\big(F(T) - K\big) = D(T)F(T) - D(T)\,K .
```

So the call price minus the put price is a straight line in $`K`$, with slope $`-D(T)`$
and intercept $`D(T)F(T)`$.

**The fit.** For one expiry, the **carry anchors** are every third strike within 5% of
spot that has both a call and a put, at least 15 of them, indexed by $`a`$. With
$`q_a`$ the call mid minus the put mid at strike $`K_a`$, and
$`s_a = \sqrt{h_{\text{call},a}^2 + h_{\text{put},a}^2}`$ the combined half-spread,
weighted least squares gives the discount factor and the forward:

```math
\big(\hat D, \hat b\big) = \arg\min_{D,\,b}\ \sum_a \Big(\frac{q_a - b + D\,K_a}{s_a}\Big)^2,\qquad \hat F = \hat b/\hat D .
```

The covariance of $`(\hat b, \hat D)`$ from the regression gives the uncertainty of
$`\hat D`$ and $`\hat F`$. Eleven expiries have such a fit: the seven of the design (step
5) and four others, among them the 30 November expiry that closes the short window of
step 9. The anchors are never used as observations. SPX options are European, so parity
holds exactly, with no early-exercise term.

### 1.3 Implied volatility, vega and half-spreads in vol points

In forward units the Black–Scholes put is

```math
p(\tilde K, T; \sigma) = \tilde K\,\Phi(-d_2) - \Phi(-d_1),\qquad d_{1} = \frac{-k + \sigma^2 T/2}{\sigma\sqrt T},\quad d_2 = d_1 - \sigma\sqrt T ,
```

and its vega is $`\nu = \sqrt T\,\varphi(d_1)`$. For each quote, the mid in forward units
gives the implied volatility by root-finding, and the half-spread in vol points is

```math
\frac{h}{\nu} = \frac{(\text{ask}-\text{bid})/2\,/\,(DF)}{\sqrt T\,\varphi\big(d_1(\sigma_{\text{imp}})\big)} ,
```

at the quote's own implied volatility. On the 63 quotes it ranges from 0.025 to 0.28 vp,
with a median of 0.064 (step 5).

### 1.4 The products and the checks

| Product | Payoff | Role |
| --- | --- | --- |
| Forward-starting variance swap, windows $`[0, 21\ \text{days}]`$ and $`[81, 325\ \text{days}]`$ | Realized variance over the window minus a strike agreed today | Priced; closed form for each draw (9.2) |
| Forward-start call, $`T_1`$ = 3 months, $`T_2`$ = 6 months | $`(S_{T_2}/S_{T_1} - 1)^+`$ per unit of notional, at $`T_2`$ | Priced; by simulation (9.5) |
| Volatility swap, 3 months | $`\sqrt{\mathrm{RV}_T} - K_{\text{vol}}`$ per unit notional | Priced; by simulation (9.4) |
| Down-and-in put, 1 year, strike $`K = S_0`$, barrier $`B = 0.7\,S_0`$, daily closes | $`(K - S_T)^+\,\mathbf{1}\{\min_i S_{t_i} \le B\}`$ | Next; estimator specified (9.7) |
| Variance swap at a listed expiry | $`\mathrm{RV}_T - K_{\text{var}}`$ | Check against the strip of listed options (next) |
| Listed 2-year 90% put | A listed option beyond the last fitted expiry | Check of extrapolation in maturity (next) |

Realized variance over $`n_d`$ trading days with closes $`S_{t_0}, \dots, S_{t_{n_d}}`$ is
the annualized mean of the squared daily log returns,
$`\mathrm{RV} = \frac{252}{n_d}\sum_{k=1}^{n_d} \big(\ln(S_{t_k}/S_{t_{k-1}})\big)^2`$;
for a continuous path it tends to $`I_T/T`$. The fair strikes are
$`K_{\text{var}}(t_1,t_2) = \mathbb{E}[I_{t_2} - I_{t_1}]/(t_2-t_1)`$ and
$`K_{\text{vol}}(T) = \mathbb{E}[\sqrt{I_T/T}]`$ under the pricing probabilities. The
down-and-in put is the risk inside autocallable notes: their buyers in effect sell it to
the issuing bank, which is therefore long it.

**The region of the smile each product needs.**
- *Forward-starting variance:* the term structure of the variance level, so the
  expiries around the window ends; the strips of listed options at those expiries fix it
  for a continuous path (1.5).
- *Forward start:* the near-the-money level and skew at 3 to 6 months, and above all the
  forward smile, the smile that will prevail at $`T_1`$, which no listed option fixes. It
  comes from the model's dynamics, inferred from the skew's term structure.
- *Volatility swap:* the variance level, plus the vol of vol, from the wings and the short
  maturities, since $`K_{\text{vol}} \approx \sqrt{K_{\text{var}}}`$ minus a convexity term
  (9.4).
- *Down-and-in put:* puts at and below 70% out to 1 year, and the spot–volatility
  dynamics ($`\rho`$ and the vol of vol). At 1 year the barrier sits near $`\delta = -2.5`$.

![Simulated index paths showing the payoffs of the three products](images/products.png)

**Figure 1.** Simulated paths of the index under Heston at its SPX posterior mean (step 4),
in forward units. Left: each forward-start call fixes its strike at the dot, 3 months in,
and pays the rise above it at 6 months. Centre: running realized volatility over 3 months,
path by path, against one fixed strike. At these parameters the variance stays low most of
the time and spikes now and then (the Feller condition fails by a factor of six, 4.1), so
most paths realize less than the strike and a few much more. Right: the expected variance
over time and its average over the two windows, the strikes of the two forward-starting
variance swaps.

### 1.5 What listed options determine: two derivations

**Marginals, not the joint law (Breeden–Litzenberger).** Write the call price as
$`C(K) = D\int (s-K)^+ p_T(s)\,ds`$. Differentiating twice in $`K`$,

```math
\frac{\partial C}{\partial K} = -D\int_K^\infty p_T(s)\,ds,\qquad \frac{\partial^2 C}{\partial K^2} = D\,p_T(K).
```

So the calls at one maturity, across all strikes, fix the distribution of $`S_T`$ at that
maturity and the price of every payoff $`g(S_T)`$. They say nothing about how the laws at
two dates are joined, which a forward start, a volatility swap or a barrier needs.

**The variance swap from the strip.** For a continuous path, Itô's formula gives
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
calls for $`k \gt 0`$). This is the VIX construction; the forward-starting strike is
$`(w(t_2) - w(t_1))/(t_2 - t_1)`$.

![The variance swap integrand for the 325-day expiry](images/variance-swap-strip.png)

**Figure 2.** The integrand $`2\,\mathrm{OTM}(k)\,e^{-k}`$ for the 325-day expiry, from Heston
prices at the SPX posterior mean; the area is $`w(T)`$. Integrated over the strip it gives
a variance-swap volatility of 21.81%, against 21.89% from the closed form of step 4: the
difference is the truncated tails. Most of the area comes from puts, and the design's nine
quotes (dots) sample it.

In practice the errors of the strip construction are strike discretization, truncation of
the wings and jumps, reported as a tolerance.

## 2. Calibration gives one value and hides the rest

### 2.1 Pricing under the model

Under the pricing probabilities $`X`$ is a martingale, and a product with payoff $`\Pi`$
paid at $`T`$ is worth

```math
G(\theta) = D(T)\,\mathbb{E}_\theta[\Pi] .
```

For a listed option this depends on the law of $`X_T`$ alone; for the products of 1.4 it
depends on the law of the path.

### 2.2 The classical answer

Calibrate by least squares and mark the product at the fitted parameters:

```math
\hat\theta = \arg\min_\theta \sum_{i=1}^{n} \Big(\sigma^{\text{imp}}_i - \sigma^{\text{model}}_i(\theta)\Big)^2
\qquad\text{or}\qquad
\hat\theta = \arg\min_\theta \sum_{i=1}^{n} r_i(\theta)^2,\qquad \text{mark} = G(\hat\theta) ,
```

in implied volatility or in price weighted by the half-spreads, from several starting
points spread over the prior box. The maximum-likelihood version uses the likelihood of
step 6 and estimates $`c`$ with the model's parameters.

### 2.3 What it hides: weak directions

Near $`\hat\theta`$, write the 63 prices as
$`P(\theta) \approx P(\hat\theta) + \mathcal{J}\,(\theta - \hat\theta)`$, with
$`\mathcal{J} = \partial P/\partial\theta`$ the Jacobian, an $`n \times p`$ matrix for
$`p`$ model parameters. Under Gaussian errors with covariance $`\Gamma`$ the
log-likelihood is locally quadratic, with curvature
$`\mathcal{J}^\top\Gamma^{-1}\mathcal{J}`$, the Fisher information. Its eigenvectors with
small eigenvalues are directions in which the parameters can move far while the quotes
barely change. A product whose gradient $`\nabla G_j`$ has a component along such a
direction inherits a large uncertainty. Which directions are weak depends on which quotes
exist (5.2).

### 2.4 Two classical ranges

**The delta method.** Linearizing as above, the generalized least-squares estimator has

```math
\widehat{\mathrm{Cov}}(\hat\theta) = \big(\mathcal{J}^\top \hat\Gamma^{-1} \mathcal{J}\big)^{-1},\qquad
\mathrm{sd}_j = \sqrt{\nabla G_j^\top\,\widehat{\mathrm{Cov}}(\hat\theta)\,\nabla G_j},\qquad
\text{range: } G_j(\hat\theta) \pm 1.645\,\mathrm{sd}_j ,
```

with $`\hat\Gamma`$ the error covariance at the fit, Jacobians by central differences, and
common random numbers for rough Bergomi. It is exact when $`P`$ and $`G_j`$ are linear in
$`\theta`$ and the errors Gaussian; it misleads where the likelihood is curved, bounded by
the prior box, or has several modes.

**Refits to parts of the quotes.** Refit by least squares to subsets of the quotes and
report the minimum to maximum of the product values. 9.6 uses four subsets of the SPX
quotes: all 63, the three shortest expiries, the four longest, and the quotes near the
money. Refitting with each of the seven expiries removed in turn is a finer version.

## 3. The posterior turns one value into a distribution

### 3.1 Posterior and push-forward

```math
p(\theta \mid y) \;\propto\; p(y \mid \theta)\,p(\theta).
```

A product's value $`G_j(\theta)`$ is a function of the parameters, so the posterior over
$`\theta`$ induces a distribution over the value, its push-forward. With posterior draws
$`\theta^{(1)},\dots,\theta^{(S)}`$,

```math
G_j^{(s)} = G_j\big(\theta^{(s)}\big),\qquad s = 1,\dots,S .
```

When $`G_j`$ is computed by simulation, every draw uses the same set of random normals
(common random numbers, step 9), so that the spread of $`G_j^{(s)}`$ across $`s`$ is due
to the parameters and not to Monte Carlo noise. The draws $`\theta^{(s)}`$ are the
network's, or the reference's: thinned for MCMC, resampled by their weights for
importance sampling.

![Exact SPX posterior draws of Heston and the variance strike they imply](images/posterior-to-product.png)

**Figure 3.** The exact SPX posterior of Heston, 1,000 draws, and what it implies for one
product. Left: the draws in the plane of $`\kappa`$ and $`\sqrt{v_\infty}`$ (their
correlation is −0.87), coloured by the product's value under each draw. Right: the
81-to-325-day variance strike (9.2) under every draw, as a volatility: mark 22.61%, 90%
range 22.53% to 22.70%, prudent value of a long position 22.55%.

### 3.2 Summaries

For product $`j`$, from the values $`G_j^{(1)}, \dots, G_j^{(S)}`$, with $`q_\alpha`$ their
$`\alpha`$-quantile:

```math
\underbrace{\bar G_j = \frac1S\sum_{s=1}^{S} G_j^{(s)}}_{\text{mark}},\qquad
\underbrace{\mathrm{PV}_j = q_{0.10}\ \ \text{if long},\quad q_{0.90}\ \ \text{if short}}_{\text{prudent value}},\qquad
\underbrace{R_j = \big|\bar G_j - \mathrm{PV}_j\big|}_{\text{reserve}} .
```

The mark is the posterior mean, which minimizes the posterior expected squared error; the
median and the point-calibration value $`G_j(\hat\theta)`$ are reported beside it. The
prudent value is the 90% point on the side that hurts the position.

### 3.3 The book

With positions $`\omega_j`$ (positive long, negative short), each draw prices the whole
book:

```math
B^{(s)} = \sum_j \omega_j\,G_j^{(s)},\qquad \bar B = \frac1S\sum_{s=1}^{S} B^{(s)},\qquad \mathrm{PV}_{\text{book}} = q_{0.10}\big(B^{(1)}, \dots, B^{(S)}\big),\qquad R_{\text{book}} = \bar B - \mathrm{PV}_{\text{book}} .
```

The example book, which is among the next steps (10.3), is long one down-and-in put (as
an autocallable issuer is), long one forward start and short one volatility swap. Two more
numbers explain its reserve:
the **diversification ratio** $`R_{\text{book}} / \sum_j |\omega_j|\,R_j`$, which the joint
draws measure and the Regulation sets at $`1-\alpha = 0.5`$, and the **correlation
matrix** of the $`G_j^{(s)}`$ across draws. The products are in different units (index
points per unit of index for the put and the forward start, vol points per unit of vega
notional for the volatility swap), so the book fixes the swap's vega notional in currency.

### 3.4 The simulator the next steps build

The posterior needs a prior $`p(\theta)`$ (step 4), the quotes with their half-spreads
(step 5) and a likelihood $`p(y \mid \theta)`$ (step 6). Together they define a generative
model, a recipe that produces a surface from nothing:

```math
\theta \sim p(\theta),\qquad y_i = P_i(\theta) + c\,h_i\,\epsilon_i,\quad \epsilon_i \sim \mathcal N(0, 1),\quad i = 1, \dots, n .
```

Step 7 learns to invert it, from $`y`$ back to $`\theta`$; step 8 checks the inversion.

## 4. The model

The model is Heston (4.1). Rough Bergomi (4.2) is the next model, for what Heston misses
at short expiries (6.5). Both move the index in forward units as
$`dX_t/X_t = \sqrt{V_t}\,dW^F_t`$ and differ in how the variance $`V`$ behaves.

### 4.1 Heston

```math
\frac{dX_t}{X_t} = \sqrt{V_t}\,dW^F_t,\qquad dV_t = \kappa\,(v_\infty - V_t)\,dt + \sigma_v\sqrt{V_t}\,dW^V_t,\qquad dW^F_t\,dW^V_t = \rho\,dt,\qquad V_0 = v_0 .
```

| Parameter | Meaning | Prior (uniform) | SPX, exact posterior mean ± sd |
| --- | --- | --- | --- |
| $`v_0`$ | today's variance | 0.01 to 0.16 | 0.02807 ± 0.00010 (volatility 16.75%) |
| $`\kappa`$ | speed of mean reversion, per year | 0.1 to 10 | 5.07 ± 0.10 |
| $`v_\infty`$ | long-run variance | 0.01 to 0.16 | 0.05349 ± 0.00033 (volatility 23.13%) |
| $`\sigma_v`$ | vol of vol | 0.1 to 3 | 1.804 ± 0.019 |
| $`\rho`$ | correlation of index and variance shocks | −0.95 to −0.1 | −0.6967 ± 0.0019 |
| $`c`$ | error multiplier (step 6) | 0.5 to 2 | 1.9993 ± 0.0007 |

Given the model and the error law, the quotes pin every parameter to a small fraction of
its prior range; step 6 explains why $`c`$ sits at its upper bound.

**Feller condition.** $`2\kappa v_\infty \ge \sigma_v^2`$ keeps the variance strictly positive.
At the SPX posterior mean $`2\kappa v_\infty = 0.54`$ and $`\sigma_v^2 = 3.26`$. It is not
imposed: zero is then reachable, but the process is well defined and Fourier prices stay
exact.

**Expected variance.** Taking expectations of the variance equation removes the $`dW`$
term, which has mean zero, so $`d\,\mathbb{E}[V_t] = \kappa\,(v_\infty - \mathbb{E}[V_t])\,dt`$
and

```math
\mathbb{E}[V_t] = v_\infty + (v_0 - v_\infty)\,e^{-\kappa t} .
```

Averaging over a window gives the strike of a forward-starting variance swap (1.4):

```math
K_{\text{var}}(t_1, t_2) = \frac{1}{t_2 - t_1}\int_{t_1}^{t_2}\mathbb{E}[V_t]\,dt = v_\infty + (v_0 - v_\infty)\,e^{-\kappa t_1}\,\frac{1 - e^{-\kappa (t_2 - t_1)}}{\kappa\,(t_2 - t_1)} ,
```

and $`t_1 = 0`$ gives the ordinary variance-swap strike $`K_{\text{var}}(T)`$.

**Characteristic function** of $`\ln X_T`$ (`heston_options.model.characteristic_function`),
in the form that avoids branch-cut jumps, with $`b = \kappa - \rho\sigma_v iu`$,
$`d = \sqrt{b^2 + \sigma_v^2(u^2 + iu)}`$ and $`g = (b-d)/(b+d)`$:

```math
\ln\varphi_T(u) = \frac{\kappa v_\infty}{\sigma_v^2}\Big[(b-d)\,T - 2\ln\frac{1-g\,e^{-dT}}{1-g}\Big] + v_0\,\frac{b-d}{\sigma_v^2}\,\frac{1-e^{-dT}}{1-g\,e^{-dT}} .
```

**Call price in forward units** (Lewis's formula: the payoff's Fourier transform, with the
integration contour moved to $`\mathrm{Im}\,u = -1/2`$; derivation in
[Lecture 10](../../../course/10-heston.md)), with $`k = \ln\tilde K`$:

```math
\frac{C}{DF} = 1 - \frac{\sqrt{\tilde K}}{\pi}\int_0^\infty \frac{\mathrm{Re}\big[e^{-iuk}\,\varphi_T(u - \tfrac i2)\big]}{u^2 + \tfrac14}\,du .
```

**The Black control variate** (`heston_options.control_variate`). A lognormal $`X_T`$ with
mean 1 and total variance $`w`$ has $`\varphi^{B}_T(u - \tfrac i2) = e^{-\frac12 w\,(u^2 + \frac14)}`$,
and the same formula prices it exactly. Subtracting the two formulas,

```math
\frac{P}{DF} = p_{\text{Black}}(\tilde K, w) + \frac{\sqrt{\tilde K}}{\pi}\int_0^\infty \frac{\mathrm{Re}\big[e^{-iuk}\,\big(\varphi^{B}_T(u - \tfrac i2) - \varphi_T(u - \tfrac i2)\big)\big]}{u^2 + \tfrac14}\,du ,
```

for calls and puts alike, since the parity terms cancel between the two models. Here
$`p_{\text{Black}}(\tilde K, w)`$ is the forward-unit Black–Scholes price of the same
option (1.3) with total variance $`\sigma^2 T = w`$, and we take
$`w = \mathbb{E}[I_T] = T\,K_{\text{var}}(T)`$. The integral is the same as before, so the
price is unchanged; what changes is the arithmetic. A deep put is worth very little (the
7-day put at $`\delta = -5`$ is worth $`2\times10^{-4}`$ in forward units), and the plain formula
obtains it as a difference of numbers close to 1, losing digits. Here the Black price
carries the value with full relative precision and the integral is a small correction.

**Quadrature and its accuracy gate.** Gauss–Legendre on $`[0, U]`$ with $`N`$ nodes, on the
ladder $`(N, U)`$ = (256, 150), (512, 300), (1,024, 600), (2,048, 1,200), (4,096, 2,400),
(8,192, 4,800), (16,384, 9,600). A rung is accepted when its prices and the next rung's
lie within the no-arbitrage bounds and agree within $`0.01\,h_i`$ for every quote, that is
$`0.02\,c_{\min}h_i`$ with $`c_{\min} = 0.5`$: refinement moves no price by more than 2% of
the smallest error the prior allows. A frozen replay over 98 parameter points (64 drawn
from the prior, the 32 corners of the prior box, and two simulated cases) passes at every
point: the largest refinement difference is $`6.5\times10^{-5}`$ of the smallest error
standard deviation, and the largest change in log-likelihood $`4.7\times10^{-6}`$. The same
check repeated at reference-posterior draws (step 8) gives differences below
$`10^{-6}`$ of that standard deviation. One pricing of the 63 quotes costs 3 to 11 ms.

**Skew and curvature.** With $`\rho \lt 0`$, volatility rises when the index falls, so
puts cost more: the skew. A larger $`\sigma_v`$ makes the variance more random and fattens
both tails: the curvature. It also steepens the skew and breaks the Feller condition.

**A ceiling on the short-expiry skew.** Write $`\psi(T)`$ for the at-the-money skew, the
slope of implied volatility in log-moneyness $`k`$ at the money:

```math
\psi(T) = \frac{\partial\,\sigma^{\text{imp}}(k, T)}{\partial k}\Big|_{k=0} .
```

Over a short time the variance barely moves from $`v_0`$, and the expansion of Heston's
smile in small $`T`$ ([Lecture 10](../../../course/10-heston.md), Section 10.8) gives

```math
\lim_{T \to 0} |\psi(T)| = \frac{|\rho|\,\sigma_v}{4\sqrt{v_0}} .
```

For expiries well below the reversion time $`1/\kappa`$ the skew sits at this ceiling and
grows no further, while index skews keep steepening as the expiry shrinks. At the
market-like parameters of step 8 the skew is 0.27 at one year, 0.86 at 30 days and 1.51
at 7 days, and stays within 10% of its ceiling, 1.82, below about 5 days. 6.5 shows the
consequence on SPX.

### 4.2 Rough Bergomi, the next model

```math
V_t = \xi_0(t)\,\exp\!\Big(\eta\,Y_t - \tfrac12\,\eta^2\,t^{2H}\Big),\qquad Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW^V_s,\qquad \frac{dX_t}{X_t} = \sqrt{V_t}\,dW^F_t,\qquad dW^F_t\,dW^V_t = \rho\,dt .
```

**Variance of the driver.** $`\mathrm{Var}(Y_t) = 2H\int_0^t (t-s)^{2H-1}\,ds = t^{2H}`$.

**Mean.** $`Y_t`$ is Gaussian, so $`\mathbb{E}\big[e^{\eta Y_t - \frac12\eta^2 t^{2H}}\big] = 1`$
and $`\mathbb{E}[V_t] = \xi_0(t)`$: the curve is the expected future variance.

**Roughness.** For $`H \lt 1/2`$ the kernel $`(t-s)^{H-1/2}`$ diverges as $`s \to t`$, so
the latest shocks weigh most and paths are rougher than Brownian motion. The at-the-money
skew then grows like $`T^{H-1/2}`$ as $`T \to 0`$, the shape of index skews at short
maturities. Whether rough models describe SPX better than Markovian ones is debated (Abi
Jaber and Li, 2024); the two models here are compared on the same information.

**The curve** (`heston_exponential_v1`), inferred jointly with the other parameters:

```math
\xi_0(t) = \xi_l + (\xi_s - \xi_l)\,e^{-\lambda t},\qquad K_{\text{var}}(t_1, t_2) = \xi_l + (\xi_s - \xi_l)\,e^{-\lambda t_1}\,\frac{1 - e^{-\lambda (t_2 - t_1)}}{\lambda\,(t_2 - t_1)} .
```

It is the same family as Heston's $`\mathbb{E}[V_t]`$, with $`\xi_s`$, $`\xi_l`$ and
$`\lambda`$ in the roles of $`v_0`$, $`v_\infty`$ and $`\kappa`$, so both models learn the
variance level from the quotes in the same way.

**Priors,** uniform and independent: $`H`$ from 0.02 to 0.35, $`\eta`$ from 0.5 to 3.5,
$`\rho`$ from −0.99 to −0.2, $`\xi_s`$ and $`\xi_l`$ from 0.01 to 0.16, $`\lambda`$ from
0.25 to 8 per year, and $`c`$ from 0.5 to 2 as for Heston.

**Simulation.** The hybrid scheme of Bennedsen, Lunde and Pakkanen: the singular cell of
the kernel next to $`t`$ is simulated exactly, older cells with averaged kernel values, and
the convolution by FFT. The normalizer is the discrete $`\mathrm{Var}(Y_i)`$, so
$`\mathbb{E}[V_i] = \xi_0(t_i)`$ holds exactly on the grid. Each of the seven expiries has
its own grid, ending exactly at its ACT/365F maturity: 622, 1,241, 664, 1,327, 1,791, 950
and 1,796 steps, which is about 128 steps per business day up to two weeks, 32 up to three
months and 8 beyond.

**The vanilla estimator** (`shared.pricing.rough_block`). Split
$`-\tfrac12 I_T = -\tfrac12\rho^2 I_T - \tfrac12(1-\rho^2)I_T`$ in the decomposition of
$`\ln X_T`$ (notation section):

```math
X_T = \underbrace{e^{\rho J_T - \frac12\rho^2 I_T}}_{F'} \times e^{\sqrt{1-\rho^2}\,J^{\perp}_T - \frac12(1-\rho^2) I_T} .
```

Given $`W^V`$, the second factor is lognormal with mean 1 and log-variance
$`w = (1-\rho^2)I_T`$. So, conditionally, $`X_T`$ is a Black–Scholes underlying with
forward $`F'`$ and total variance $`w`$, and

```math
\frac{P(\tilde K, T)}{DF} = \mathbb{E}\Big[\mathrm{BS}_{\text{put}}\big(F',\ \tilde K,\ w\big)\Big],
```

an average of closed-form prices over variance paths. Two further reductions: antithetic
pairs, and two control variates per maturity with known mean zero,
$`e^{\rho J_T - \frac12\rho^2 I_T} - 1`$ and $`I_T - \sum_i \xi_0(t_i)\,\Delta t`$, with
regression coefficients. The standard error comes from the controlled residuals.

**Accuracy.** Paths double from 32,768 to 2,097,152 until the standard error of every
price is at most $`0.25\,\sigma_{\text{ref}}`$, with $`\sigma_{\text{ref}} = 4.41\times10^{-4}`$
in forward-unit price, a fixed numerical scale; the last rung accepts up to
$`0.5\,\sigma_{\text{ref}}`$. The target, $`1.1\times10^{-4}`$, is about 3.3 times the median
half-spread of the 63 quotes in the same units ($`3.3\times10^{-5}`$). The bank records
each price's standard error. Where it is not small compared with $`c\,h_i`$, the Monte
Carlo noise is part of the simulated observations, and the network learns a
correspondingly wider posterior. The price is that of the model on its time grid; the
effect of a finer grid is recorded on a panel, not imposed.

**Cost.** Pricing the 63 quotes for one parameter set takes a median of about 1 CPU-minute
and a mean of about 1.7 (90% of sets under 4 minutes). A set that exceeds 860 seconds is
recorded as a failure and not redrawn: 7 of the first 2,764. The bank of 300,000 sets has
been running on Brown's Oscar cluster since 11 November, on up to 192 CPUs: 47 hours if
every CPU stays busy, about 59 hours with the batch structure, from the measured times.
Rows that ran out of time are recomputed separately with their original parameters,
errors and random streams (271 of them by 12 November), and merged by row, so that no row
is drawn twice. Sets for validation (5,000) and testing (7,000) follow. Generating the
bank does not by itself show that a network trained on it is calibrated; that is the work
of steps 7 and 8.

### 4.3 The same information for both

Both models see the same 63 quotes, the same error law with $`c`$ inferred under the same
prior, and the same family for the variance level. Neither sees information the other
does not, so differences between their product ranges come from their dynamics.

## 5. Which quotes

### 5.1 The selection rule

**Expiries.** For each of 5, 10, 21, 42, 63, 126 and 252 business days (counted as
days/252 years), the listed SPXW expiry whose exact maturity is nearest.

**Places along each expiry.** Nine targets in standardized moneyness,
$`\delta \in`$ {−5, −3.5, −2.25, −1.25, −0.5, 0, +0.5, +1, +2}, with a put at $`\delta \le 0`$ and a
call at $`\delta \gt 0`$, so that every quote is out of the money (puts below the forward,
calls above).

**Eligible quotes.** Positive bid, ask above bid, half-spread at most 25% of the mid, and a
mid strictly inside the European no-arbitrage bounds ($`0 \lt P \lt DK`$ for a put,
$`0 \lt C \lt DF`$ for a call). Carry anchors and a separate set of held-out quotes are
excluded.

**Selection.** For each target $`\delta^\ast`$ of an expiry, the eligible quote of the
right type, not already used, that is nearest to it: $`i^\ast = \arg\min_i |\delta_i - \delta^\ast|`$.
A target counts as supported when $`|\delta_{i^\ast} - \delta^\ast| \le 0.35`$. At the 325-day expiry no call
supports $`\delta = +2`$, so its last target is +1.5.

| Expiry | Days | $`T`$ (years) | Forward | $`\delta`$ from | to | Half-spreads (vp) | At-the-money vol |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 16 Nov 2026 | 7 | 0.0193 | 7,573.5 | −5.02 | +1.96 | 0.028–0.280 | 15.2% |
| 23 Nov 2026 | 14 | 0.0384 | 7,577.3 | −5.01 | +2.03 | 0.047–0.251 | 15.4% |
| 9 Dec 2026 | 30 | 0.0823 | 7,586.9 | −5.11 | +1.96 | 0.039–0.190 | 15.3% |
| 8 Jan 2027 | 60 | 0.1645 | 7,615.0 | −4.74 | +1.87 | 0.041–0.145 | 15.4% |
| 29 Jan 2027 | 81 | 0.2220 | 7,632.7 | −5.09 | +1.99 | 0.025–0.099 | 15.6% |
| 30 Apr 2027 | 172 | 0.4712 | 7,705.4 | −5.10 | +2.00 | 0.025–0.107 | 16.4% |
| 30 Sep 2027 | 325 | 0.8904 | 7,830.9 | −5.15 | +1.49 | 0.026–0.070 | 17.1% |

Every selected quote lies within 0.26 of its target (within 0.15 except at 60 days, where
the deepest put is at −4.74). The design is frozen with each quote's strike, exact
maturity, forward, discount, mid and half-spread
([`design.json`](https://github.com/fmuia/sbi4finance/blob/1c276bb72c1a8d43d1055327934f359708ba1e3f/lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/cv-preparation/design.json)).

![The 63 quotes: implied volatility by expiry, and their places in moneyness and maturity](images/smile-and-design.png)

**Figure 4.** Left: the smile at the 63 quotes, one line per expiry. Right: the design in
$`(\delta, T)`$, coloured by each quote's half-spread in vol points (median 0.064, from 0.025
to 0.28). The widest are the calls at $`\delta \approx 2`$ in the first two weeks.

On the scale of $`\delta`$ the seven smiles are nearly parallel. From the money to the
deepest put, implied volatility rises by 20.5, 22.3, 24.5, 23.7, 25.8, 25.9 and 25.3 vol
points at the seven expiries, from 7 to 325 days: the skew per unit of $`\delta`$ is about
the same at one week as at eleven months. That is what the model must reproduce; in
log-moneyness $`k = \sigma_{\text{ref}}\sqrt{T}\,\delta`$ it means a skew that keeps growing
as the expiry shrinks (6.5). The half-spread is 0.18% of the mid at
best, near the money, and 20% at worst, for the cheapest contract, a 7-day call worth
0.25 index points.

Using actual quotes adds no interpolation error. Because the design is fixed, one trained
network serves this snapshot; a new design needs its own simulations and network.

### 5.2 What each region pins

A local calculation shows how much each region of quotes contributes. At the SPX
posterior mean $`\hat\theta`$, with the Jacobian $`\mathcal{J} = \partial P/\partial\theta`$
by central differences and independent errors of one half-spread, the Fisher information
and the local posterior covariance are

```math
\mathcal{I} = \mathcal{J}^\top\,\mathrm{diag}(1/h^2)\,\mathcal{J},\qquad \mathrm{Cov}(\theta \mid y) \approx \mathcal{I}^{-1},\qquad \mathrm{Var}\big(G(\theta) \mid y\big) \approx \nabla G^\top\,\mathcal{I}^{-1}\,\nabla G ,
```

for any quantity $`G(\theta)`$ derived from the parameters, such as a variance strike. Removing the quotes of
a region $`R`$ gives $`\mathcal{I}_{-R}`$; the rise in posterior standard deviation is the
ratio of the square roots of the diagonals of $`\mathcal{I}_{-R}^{-1}`$ and
$`\mathcal{I}^{-1}`$, minus one. It does not depend on $`c`$, which scales every variance
alike.

| Region removed | $`v_0`$ | $`\kappa`$ | $`v_\infty`$ | $`\sigma_v`$ | $`\rho`$ | Short-window variance | Later-window variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deep puts, $`\delta \le -2.25`$ (21 quotes) | +56% | +76% | +75% | +87% | +3% | +79% | +71% |
| Near puts, $`\delta`$ = −1.25 and −0.5 (14) | +21% | +9% | +20% | +8% | +17% | +21% | +25% |
| At the money, $`\delta = 0`$ (7) | +4% | +3% | +1% | +2% | +4% | +3% | +0% |
| Calls, $`\delta \gt 0`$ (21) | +124% | +30% | +53% | +77% | +791% | +122% | +76% |
| Expiries up to 1 month (27) | +144% | +70% | +23% | +92% | +10% | +106% | +11% |
| Expiries from 2 months (36) | +101% | +846% | +1502% | +107% | +157% | +39% | +1684% |

![Rise in posterior standard deviation when one region of quotes is removed](images/information-by-region.png)

**Figure 5.** The table as a chart. Bars beyond the scale are labelled with the factor by
which the standard deviation grows.

- The calls pin $`\rho`$: without them its uncertainty grows ninefold.
- The expiries from 2 months pin the long-run level: without them $`v_\infty`$ and the
  later-window variance become 16 and 18 times less certain, and $`\kappa`$ 9.5 times.
- The expiries up to 1 month pin today's variance and the short-window variance.
- The deep puts matter for every parameter except $`\rho`$, by 56% to 87%.
- The at-the-money point adds little once its neighbours are present.

So a design must reach deep puts and far calls, at short and at long maturities.

### 5.3 Coverage over density

Exact posteriors confirm this on simulated surfaces. A narrower design of 54 quotes, with
six expiries from 21 to 325 days (none shorter than three weeks), puts to about
$`\delta = -4`$ and calls to about $`\delta = +1`$, was compared with the 63-quote design on two simulated surfaces with
the same parameters. Under the 63 quotes the exact posterior's 90% intervals for the five
Heston parameters are 41% to 87% as wide, and $`\rho`$'s narrows the most on one of the two
(to 41%). The quotes, their half-spreads and the error draws differ between the two
designs, so this comparison describes two cases rather than proving a general ranking.

### 5.4 Data kept out of every fit

- **Carry anchors:** the parity pairs of 1.2, used only for forwards and discounts.
- **Held-out quotes:** a separate set of quotes, excluded from the selection, for scoring
  predictions out of sample.
- **Variance-swap strips:** at each expiry, all usable out-of-the-money quotes, for the
  check of 1.5.
- **The listed 2-year 90% put:** the listed expiry nearest 2 years and the strike nearest
  $`0.9\,F(T)`$, for a check of extrapolation in maturity.

## 6. How quotes differ from the model

### 6.1 The observation law

For the $`n = 63`$ quotes,

```math
y_i = P_i(\theta) + c\,h_i\,\epsilon_i,\qquad \epsilon_i \sim \mathcal N(0, 1)\ \text{independent},\qquad c \sim U(0.5,\ 2) .
```

The three factors of the error term play different roles:
- $`h_i`$ is **data**: the half-spread of quote $`i`$, read from the snapshot. Like the
  strike and the expiry it is known and fixed, the same in every simulation. The value at
  which a trade would happen lies between bid and ask, so the half-spread is the natural
  scale of a quote's uncertainty; reading it as one standard deviation is a convention.
- $`\epsilon_i`$ is the **random error**: a fresh standard normal draw for every quote
  and every simulated surface.
- $`c`$ is an **unknown**: one multiplier for all 63 quotes, inferred with the model's
  parameters and never given to the network as an input. It lets the data say how large
  the errors are, relative to the half-spreads.

The errors are added to prices, not to implied volatilities, so a simulated price that
falls below zero stays usable. The law has a single source of error, quote noise: it
assumes that the model, at the right parameters, prices every contract exactly. If the
model itself is off, the only way this law can absorb it is a larger $`c`$ (6.5).

### 6.2 Likelihood

Each $`y_i`$ is normal with mean $`P_i(\theta)`$ and standard deviation $`c\,h_i`$. With
the scaled residuals $`r_i(\theta) = (y_i - P_i(\theta))/h_i`$,

```math
\log p(y \mid \theta) = -\frac{1}{2c^2}\sum_{i=1}^{n} r_i(\theta)^2 - n\ln c - \sum_{i=1}^{n} \ln h_i - \frac n2 \ln 2\pi .
```

The term $`-n\ln c`$ comes from the normalizing constant of the Gaussian. It penalizes a
large $`c`$; without it the posterior would raise $`c`$ to explain any miss. Setting the
derivative in $`c`$ to zero at fixed model parameters gives the conditional mode

```math
\hat c = \sqrt{\frac1n \sum_{i=1}^{n} r_i(\theta)^2} ,
```

the root-mean-square scaled residual, clipped to the prior's range.

### 6.3 How precisely the error scale can be learned

At the true parameters $`r_i = c\,\epsilon_i`$, so $`\sum_i r_i^2/c^2`$ is a sum of
$`n`$ squared standard normals, a $`\chi^2_n`$ variable. When the $`p = 5`$ Heston
parameters are fitted, about $`p`$ of the $`n`$ directions in which the quotes can move
are absorbed by the fit, and $`\sum_i r_i^2/c^2 \approx \chi^2_{n-p}`$ with
$`n - p = 58`$. A $`\chi^2_{58}`$ divided by 58 has variance $`2/58`$, so by the delta
method

```math
\frac{\mathrm{sd}(\hat c)}{c} \approx \frac{1}{\sqrt{2\,(n - p)}} = \frac{1}{\sqrt{2 \times 58}} \approx 0.093 ,
```

and a 90% interval for $`c`$ is about $`2 \times 1.645 \times 0.093 \approx 0.31`$ times
$`c`$ wide. The exact posteriors of the two simulated cases of step 8, both generated with
$`c = 1`$, have 90% intervals for $`c`$ of width 0.28 and 0.30. The quotes determine
$`c`$ well; step 7 shows what a network needs in order to read it.

### 6.4 The fixed-scale case

With $`c`$ fixed, the likelihood is a least-squares criterion with weights
$`1/(c\,h_i)^2`$. Near the mode the posterior covariance is $`c^2\,(\mathcal{J}^\top\mathrm{diag}(1/h^2)\mathcal{J})^{-1}`$,
so the posterior widths of the physical parameters scale with $`c`$: fixing $`c`$ at 0.5
or 2 halves or doubles them, as long as the prior box does not bind. The error model sets
the width of every range that follows.

### 6.5 What the SPX quotes say

The exact posterior on the market puts $`c`$ at 1.9993 ± 0.0007, against an upper bound of
2: the quotes ask for errors larger than the prior allows. To see why, take the mean
$`\bar\theta`$ of the exact posterior and measure the misses of the model in two units: in
vol points, $`e_i = \sigma^{\text{imp}}_i - \sigma^{\text{model}}_i(\bar\theta)`$, and in
half-spreads, the scaled residuals $`r_i(\bar\theta)`$.

![Market quotes against Heston's best fit, and the misses by expiry](images/heston-misfit.png)

**Figure 6.** Left: market quotes (dots) and Heston at its exact posterior mean (lines), for
three expiries. At 7 days the model cannot produce the steep skew of the deep puts. Right:
market minus model in vol points, for all seven expiries; the red band, ± one median
half-spread, is barely visible.

| Measure | Value |
| --- | --- |
| Root mean square of $`e_i`$, $`\sqrt{\frac1n\sum_i e_i^2}`$ | 1.38 vp |
| Largest $`e_i`$ in absolute value (7-day put at $`\delta = -5`$) | 6.0 vp |
| Root mean square of $`e_i`$ by expiry, 7 / 14 / 30 days | 2.2 / 1.8 / 1.4 vp |
| Root mean square of $`e_i`$ by expiry, 60 to 325 days | 0.84–0.93 vp |
| Median half-spread, $`h_i/\nu_i`$ | 0.064 vp |
| Root mean square of $`r_i`$, $`\sqrt{\frac1n\sum_i r_i^2}`$ | 13.4 half-spreads (the law allows about $`c \le 2`$) |
| Correlation of $`e_i`$ between neighbouring strikes of one expiry | 0.66 |
| Correlation of $`e_i`$ between adjacent expiries, same target $`\delta`$ | 0.94 |

**Four signs that the model is wrong.** All four come from the exact posterior; the
network plays no part in them.
1. $`c`$ sits at its bound: median 1.9995 against an upper limit of 2.
2. The fit misses at every expiry. Per expiry, the root mean square of $`r_i`$ is 10.7 to
   17.0 half-spreads, against a largest allowed error of 2 (table below).
3. The misses are systematic. The deepest put is priced too low at every expiry, by 1.9 to
   6.0 vol points, and the 14 puts with $`\delta \lt -3`$ give 42% of
   $`\sum_i r_i^2`$. Elsewhere the fit is a compromise that misses by up to 1.2 vol points
   in both directions.
4. The fit needs extreme parameters: $`\sigma_v = 1.80`$, so $`\sigma_v^2 = 3.26`$ against
   $`2\kappa v_\infty = 0.54`$; the Feller condition fails by a factor of six.

| Expiry (days) | 7 | 14 | 30 | 60 | 81 | 172 | 325 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Root mean square of $`r_i`$ (half-spreads) | 13.8 | 10.7 | 14.7 | 11.0 | 17.0 | 10.7 | 14.3 |
| Largest $`r_i`$ | 23.9 | 21.0 | 28.5 | 18.5 | 35.4 | 23.3 | 31.5 |
| $`\delta`$ of the largest | −5.02 | −5.01 | −5.11 | −4.74 | +0.50 | −5.10 | −5.15 |

The largest miss is at the deepest put at six of the seven expiries.

**The likelihood of the error scale.** At fixed model parameters the likelihood of 6.2 is,
as a function of $`c`$, proportional to $`c^{-n}\exp\big(-\sum_i r_i^2/(2c^2)\big)`$, which
peaks at $`\hat c = \sqrt{\frac1n\sum_i r_i^2}`$. At the posterior mean that is 13.4. The
multiplier scales every error by the same factor, so it cannot improve the fit, only
measure it; the prior's bound stops it at 2.

**Which feature is missing.** Measure each smile's skew $`\psi(T)`$ and curvature at the
money by fitting a quadratic in $`k`$ through the five quotes nearest the money, as in
[Lecture 5](../../../course/05-implied-volatility-and-the-smile.md): the skew is the
linear coefficient and the curvature twice the quadratic one.

![What Heston misses at short expiries: the skew ceiling, the curvature and the 7-day smile](images/heston-skew-ceiling.png)

**Figure 7.** Heston's best fit (the posterior mean) against the SPX quotes. Left: the
at-the-money skew $`|\psi(T)|`$ of each smile, Heston's skew at any expiry from one day to
one year, its ceiling $`|\rho|\,\sigma_v/(4\sqrt{v_0}) = 1.88`$ (4.1), and a power law
fitted to the SPX skews. Centre: the at-the-money curvature. Right: the 7-day smile.

| Expiry (days) | 7 | 14 | 30 | 60 | 81 | 172 | 325 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPX skew $`\psi(T)`$ | −1.36 | −1.11 | −0.85 | −0.63 | −0.55 | −0.40 | −0.30 |
| Best fit's skew | −1.59 | −1.27 | −0.95 | −0.67 | −0.59 | −0.42 | −0.30 |
| SPX curvature | 13.1 | 9.8 | 6.3 | 4.0 | 3.0 | 1.3 | 0.53 |
| Best fit's curvature | 11.0 | 17.9 | 12.4 | 7.1 | 5.1 | 1.8 | 0.58 |
| Rise from the money to the deepest put, SPX (vp) | 20.5 | 22.3 | 24.5 | 23.7 | 25.8 | 25.9 | 25.3 |
| The same, best fit (vp) | 13.7 | 17.5 | 21.4 | 22.4 | 24.1 | 23.9 | 22.5 |

- *Skew.* The SPX skew follows a power law, $`|\psi(T)| \propto T^{-0.40}`$, and keeps
  growing as the expiry shrinks. Over the quoted expiries the fit follows it within 16%,
  with a similar exponent, −0.44. But its 7-day skew is already 85% of the ceiling, so
  shorter expiries could add at most 15%.
- *Curvature.* The SPX smile gets more curved as the expiry shrinks. The fit's curvature
  peaks at 14 days and falls below it; under about 3 days it turns negative.
- *The deep put wing.* From the money to the deepest put, SPX rises 20.5 vol points at
  7 days and the fit only 13.7; from 60 days on the shortfall is 1.2 to 2.8 points.
- *The 7-day smile.* Near the money ($`|\delta| \lt 1.3`$) the fit is within 1.2 vol
  points; at the deepest put it is 6.0 too low. With one diffusive variance factor, Heston
  cannot put that much crash risk into one week without breaking the rest of the smile.

What Heston lacks is a short-expiry smile that keeps steepening. Rough Bergomi's skew grows
like $`T^{H-1/2}`$ with no ceiling (4.2).

**What follows.** On SPX the model, not the quotes, is the main source of error, and the
misses are smooth across strikes and maturities. The posterior is then the uncertainty of
Heston's parameters given that Heston is the model, with errors capped at twice the
half-spread: it leaves out the model's own error. Widening the prior on $`c`$ would not
repair this. Independent errors would treat the 63 smooth misses as 63 independent pieces
of evidence, when they form a few smooth bumps, and the posterior would remain too narrow.
A smooth model-error term (6.6) lets the posterior admit the misfit instead of pushing
$`c`$ against its bound; the misfit itself stays, and removing it needs a different
model.

### 6.6 The smooth model-error extension

The natural extension adds a smooth error in implied volatility:

```math
y = P(\theta) + c\,h \odot \epsilon + \tau\,\nu \odot U,\qquad \epsilon \sim \mathcal N(0, I_n),\qquad U \sim \mathcal N(0, K_\ell)\ \text{independent},
```

with $`\nu`$ the quotes' vegas, $`\tau`$ the model error's size in volatility (inferred),
$`\odot`$ the elementwise product, and $`K_\ell`$ a squared-exponential correlation over
$`(\delta_i, \ln T_i)`$:

```math
(K_\ell)_{ij} = \exp\!\Big(-\frac{(\delta_i-\delta_j)^2}{2\ell_\delta^2} - \frac{(\ln T_i-\ln T_j)^2}{2\ell_T^2}\Big).
```

The law of 6.1 is the special case $`\tau = 0`$.

**Likelihood.** The errors $`y - P(\theta)`$ are jointly normal with mean zero and
covariance $`\Gamma`$:

```math
\Gamma(\tau, c) = c^2\,\mathrm{diag}(h^2) + \tau^2\,\mathrm{diag}(\nu)\,K_\ell\,\mathrm{diag}(\nu),\qquad
\log p(y \mid \theta, \tau) = -\tfrac12\,\big(y - P(\theta)\big)^\top\Gamma^{-1}\big(y - P(\theta)\big) - \tfrac12\log\det\Gamma - \tfrac n2\log 2\pi ,
```

computed with a Cholesky factor of $`\Gamma`$. The log-determinant plays the role of
$`-n\ln c`$: it stops $`\tau`$ from growing without bound. For rough Bergomi the Monte
Carlo price covariance at $`\theta`$ is added to $`\Gamma`$.

**Volatility units.** To first order,
$`y_i - P_i(\theta) \approx \nu_i\,(\sigma^{\text{market}}_i - \sigma^{\text{model}}_i)`$,
so dividing the residuals by $`\nu`$,

```math
\frac{y - P(\theta)}{\nu} \sim \mathcal N\big(0,\ c^2\,\mathrm{diag}(h_{\text{vol}}^2) + \tau^2 K_\ell\big),\qquad h_{\text{vol}} = h/\nu :
```

independent quote noise in vol points plus a smooth curve of model error.

**How precisely the model-error size can be learned.** In volatility units, with
$`\Gamma = c^2 D + \tau^2 K`$, $`D = \mathrm{diag}(h_{\text{vol}}^2)`$, the Fisher information
for $`\tau^2`$ is $`\tfrac12\,\mathrm{tr}\big(\Gamma^{-1}K\,\Gamma^{-1}K\big)`$. If
$`D \approx d\,I`$ and $`K = \sum_k \lambda_k e_k e_k^\top`$,

```math
\mathcal{I}(\tau^2) = \frac12\sum_k \frac{\lambda_k^2}{(c^2 d + \tau^2\lambda_k)^2} \approx \frac{m_{\text{eff}}}{2\tau^4},\qquad m_{\text{eff}} = \text{number of } k \text{ with } \tau^2\lambda_k \gg c^2 d ,
```

so $`\mathrm{sd}(\tau)/\tau \approx 1/\sqrt{2\,m_{\text{eff}}}`$: only the directions in which
model error dominates quote noise inform $`\tau`$. On the 63 quotes, with their own
half-spreads, $`c = 1`$ and $`\theta`$ held fixed:

| Correlation lengths $`(\ell_\delta, \ell_T)`$ | $`\tau`$ | $`\mathrm{sd}(\tau)/\tau`$ | $`m_{\text{eff}}`$ |
| --- | ---: | ---: | ---: |
| (0.75, 1.15) | 1 vp | 0.108 | 43 |
| (0.75, 1.15) | 2.5 vp | 0.099 | 51 |
| (1.6, 2.0) | 1 vp | 0.151 | 22 |
| (1.6, 2.0) | 2.5 vp | 0.134 | 28 |

So $`\tau`$ is determined to about 10–15%, less sharply the smoother the error. Fitting
$`\theta`$ as well absorbs up to $`p`$ of these directions.

**Why independent errors make posteriors too narrow.** Estimate a quantity along a
direction $`a`$ of the quotes by least squares while the true residual covariance is
$`\tau^2 K`$. The true variance of the estimate is $`\tau^2\,a^\top K a/(a^\top a)^2`$; an
independent-error model claims $`\tau^2/(a^\top a)`$. Their ratio,

```math
\frac{a^\top K a}{a^\top a} ,
```

is large when $`a`$ is smooth across quotes. For a level shift, $`a = \mathbf{1}`$, it is
$`\mathbf{1}^\top K\mathbf{1}/n`$, the average number of strongly correlated neighbours. On
the 63 quotes the independent model understates that standard deviation by a factor of
2.9 with $`\ell = (0.75, 1.15)`$ and 4.7 with $`\ell = (1.6, 2.0)`$. Variance levels, and
with them variance and volatility swaps, load on such smooth directions.

**Settings.** $`\tau`$ log-uniform on 0.2 to 5 vp for both models; the correlation lengths
fixed per model before training by profile likelihood on the quotes, over
$`\ell_\delta \in`$ {0.75, 1, 1.5, 2} and $`\ell_T \in`$ {1, 1.5, 2}; $`c`$ as in 6.1.

**What it means for the network.** In the network's coordinates $`y/h`$ the noise
covariance becomes $`c^2 I + \tau^2\,\mathrm{diag}(\nu/h)\,K_\ell\,\mathrm{diag}(\nu/h)`$. At
$`\tau = 1`$ vp and a median half-spread of 0.064 vp, the model-error variance is hundreds
of times the quote noise in its leading directions, and those directions are smooth, like
the parameters' own. Reading $`\tau`$ is therefore the same kind of problem as reading
$`c`$, and needs the same preparation of the input (7.6).

The extension is specified and its identifiability computed; posteriors and networks that
include it are among the next steps (10.3).

## 7. Computing the posterior: simulation-based inference

Steps 4 to 6 define the posterior of the unknowns $`\theta`$ given the quotes $`y`$:

```math
p(\theta \mid y) = \frac{p(\theta)\,p(y \mid \theta)}{p(y)},\qquad p(y) = \int p(\theta)\,p(y \mid \theta)\,d\theta .
```

This step computes it. The argument has five links:
1. evaluating the right-hand side at one $`\theta`$ costs one pricing of the 63 quotes (7.1);
2. one posterior needs about 180,000 such evaluations, and the work needs hundreds of
   posteriors (7.1, 7.2);
3. the pricings can instead be done once, on simulated parameters, because simulated
   pairs $`(\theta, y)`$ already contain the posterior of every $`y`$ (7.3);
4. a network learns the posterior from those pairs; 7.4 says what goes into it and what
   comes out, 7.5 how it is trained;
5. the network can read the error scale $`c`$ only if its input is whitened (7.6).

### 7.1 What one posterior costs when computed directly

**One likelihood evaluation is one pricing.** For one candidate $`\theta`$, the likelihood
of 6.2 is

```math
p(y \mid \theta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\,c\,h_i}\,\exp\!\Big(-\frac{\big(y_i - P_i(\theta)\big)^2}{2\,c^2 h_i^2}\Big).
```

The quotes $`y_i`$ and half-spreads $`h_i`$ are data, and $`c`$ is a component of
$`\theta`$. The only thing to compute is the vector of $`n = 63`$ model prices
$`P(\theta) = (P_1(\theta), \dots, P_n(\theta))`$. We call this computation one
**pricing**; a likelihood evaluation costs one pricing.

| | Heston | Rough Bergomi |
| --- | --- | --- |
| How one pricing is computed | 63 Fourier integrals of a closed-form characteristic function (4.1) | Monte Carlo: 32,768 to 2,097,152 simulated variance paths on seven time grids (4.2) |
| Cost of one pricing | 3 to 11 ms | 1 to 2 CPU-minutes |

**One posterior needs many evaluations.** There is no formula for $`p(\theta \mid y)`$,
because the integral $`p(y)`$ has none. Markov chain Monte Carlo (MCMC) draws from the
posterior without it: a chain proposes a new $`\theta`$, evaluates
$`p(\theta)\,p(y \mid \theta)`$ there, accepts or rejects the proposal, and repeats. The
cost of a posterior is therefore

```math
\text{cost of one posterior} = (\text{number of likelihood evaluations}) \times (\text{cost of one pricing}) .
```

| | Heston | Rough Bergomi |
| --- | --- | --- |
| Likelihood evaluations for one posterior by MCMC | about 180,000: 34,000 in a short first run, 146,000 in the run that passed (measured, 8.1) | of the same order |
| Cost of one posterior | 17 to 32 minutes on one core (measured) | 180,000 × 1 to 1.75 CPU-minutes = 125 to 220 CPU-days (not run) |

![What each step costs, and the total time for many sets of quotes](images/posterior-costs.png)

**Figure 8.** Left: the cost of each step, on a logarithmic scale; hatched bars are paid
once, and the lighter part of a bar is the upper end of its range. Heston's times are
measured; the rough Bergomi time follows from the number of evaluations and the measured
cost of one pricing. Right: the total time for many sets of quotes. Simulation and
training are paid once, 46 minutes in all; MCMC is paid again for every set, and the 512
sets of the calibration check (8.2) would take about 230 hours.

### 7.2 How many posteriors the work needs

| Purpose | Posteriors needed |
| --- | --- |
| The result: each model's posterior on the market quotes (step 9) | 1 per model and day |
| Calibration check (8.2): a 90% interval is checked by counting how often it contains a known truth, so it needs simulated surfaces, each with its own posterior. With $`M`$ surfaces the count has standard error $`\sqrt{0.9 \times 0.1/M}`$: 1.3% for $`M = 512`$ | 512 per model |
| Comparison with exact posteriors (8.3) | 3 per model |
| Daily use | 1 per model, day and underlying |

By MCMC, the 512 posteriors of the calibration check would cost about 230 hours for
Heston, at the 27 minutes of an average run, and 175 to 310 CPU-years for rough
Bergomi. The posterior on the market is cheap;
showing that its ranges can be trusted is what costs.

### 7.3 The idea: do the pricings once, on simulated parameters

**Simulated pairs.** Run the generative model of 3.4 $`N`$ times, with $`n`$ = 1, …,
$`N`$:

```math
\theta^{(n)} \sim p(\theta),\qquad y^{(n)} = P\big(\theta^{(n)}\big) + c^{(n)}\,h \odot \epsilon^{(n)},\qquad \epsilon^{(n)} \sim \mathcal N(0, I_{63}) ,
```

where $`\odot`$ is the elementwise product and $`c^{(n)}`$ is the last component of
$`\theta^{(n)}`$. Each pair costs one pricing. The table of $`N`$ = 300,000 pairs, the
**bank**, costs 300,000 pricings: fewer than two posteriors by MCMC.

**Why the pairs contain every posterior.** A pair made this way is a draw from the joint
density $`p(\theta)\,p(y \mid \theta)`$. By the definition of conditional probability the
same density factors the other way:

```math
p(\theta)\,p(y \mid \theta) = p(y)\,p(\theta \mid y) .
```

Read from the right, drawing a pair is the same as drawing quotes $`y`$ from $`p(y)`$ and
then parameters from the posterior of those quotes. So, among all simulated pairs whose
quotes equal a given $`y`$, the parameters are distributed as $`p(\theta \mid y)`$. The
bank is a table of examples of the map from quotes to a draw from their posterior
([Lecture 9](../../../course/09-amortized-inference.md), Figure 8.1, shows this on a toy
model).

**A network learns the map.** With 63 real-valued quotes, no simulated $`y^{(n)}`$ ever
equals the observed one, so the posterior cannot be looked up in the table. A neural
network, with weights $`\phi`$, learns from all pairs at once how the distribution of
$`\theta`$ changes with $`y`$. Its output for quotes $`y`$ is a distribution
$`q_\phi(\theta \mid y)`$, meant to equal $`p(\theta \mid y)`$ for every $`y`$ the
generative model can produce.

**What this buys.**

| | Pricings | Paid |
| --- | ---: | --- |
| The bank | 300,000 | once per model and design |
| Each posterior from the trained network | 0 | under 0.3 s each |
| Each posterior by MCMC | about 180,000 | every time |

The cost of inference is paid once and spread over every later posterior: it is
**amortized**. For Heston the bank took 16 minutes and the training 30 minutes on the
processor of a laptop; for rough Bergomi the bank takes 47 to 59 hours on 192 CPUs (4.2),
fewer pricings than two posteriors by MCMC would need.

**Two ways to use the simulations.** Simulated pairs can train either of two networks.
- A **neural pricer** learns the clean prices of the five pricing parameters
  $`\vartheta = (v_0, \kappa, v_\infty, \sigma_v, \rho)`$:
  $`\widehat P_\psi(\vartheta) \approx P(\vartheta)`$, with weights $`\psi`$. It replaces
  the pricer in the likelihood of 6.2, and MCMC samples the result with the same
  six-parameter prior, once for every set of quotes.
- An **amortized estimator**, the route taken here, learns the posterior
  $`q_\phi(\theta \mid y)`$ itself from noisy simulated quotes, so a new set of quotes needs
  no MCMC.

With the neural pricer and the error model of 6.1, the likelihood is

```math
\log\widehat p(y \mid \vartheta, c) = -\frac12\sum_{i=1}^{63}\Big[\Big(\frac{y_i - \widehat P_{\psi,i}(\vartheta)}{c\,h_i}\Big)^2 + 2\ln(c\,h_i)\Big] + \text{constant} ,
```

where the $`\ln(c\,h_i)`$ terms matter because $`c`$ is inferred. The checks differ: a
neural pricer's errors must be small against the quote noise, and each MCMC run must
converge; an amortized estimator must be calibrated across simulated sets and agree with
exact posteriors (step 8). A fair comparison of the two keeps the contracts, the spreads
and the prior the same. Heston's Fourier pricing is already fast, so its exact reference
uses the pricer directly; a neural pricer pays off where pricing is expensive, as for
rough Bergomi.

### 7.4 What the network takes in and what it returns

![What the estimator takes in and returns, at training and at use](images/network-inputs-outputs.png)

**Figure 9.** Top: training, done once. Bottom: use. The three quantities fitted in
training (orange) are the whitening transform $`(\mu, W)`$, the weights $`\phi`$ of the
velocity network, and the means and standard deviations $`(m_\theta, s_\theta)`$ of the
unknowns.

| Object | Symbol | Size | What it is |
| --- | --- | ---: | --- |
| Quotes | $`y`$ | 63 | mid prices of the 63 contracts, in index points |
| Scaled quotes | $`x = y/h`$, that is $`x_i = y_i/h_i`$ | 63 | each quote in units of its own half-spread |
| Whitened quotes | $`\tilde x = W\,(x - \mu)`$ | 63 | the network's input; $`\mu`$ and $`W`$ are defined in 7.6 |
| Unknowns | $`\theta = (v_0, \kappa, v_\infty, \sigma_v, \rho, c)`$ | 6 | Heston's parameters and the multiplier |
| Standardized unknowns | $`\tilde\theta = (\theta - m_\theta)/s_\theta`$, elementwise | 6 | $`m_\theta`$, $`s_\theta`$: mean and standard deviation of each unknown over the bank |
| Noise | $`z_1 \sim \mathcal N(0, I_6)`$ | 6 | a fresh random vector for every posterior draw |
| Flow state | $`z_t`$, $`t \in [0, 1]`$ | 6 | a point that moves from the noise $`z_1`$ to a draw $`z_0`$ |
| Velocity network | $`v_\phi(z_t, t, \tilde x)`$ | 6 + 1 + 63 in, 6 out | three hidden layers of 128 units; the only trained function |
| Posterior draws | $`\theta^{(1)}, \dots, \theta^{(S)}`$ | $`S`$ × 6 | the output |

**At use,** for quotes $`y`$:
1. compute $`x = y/h`$ and $`\tilde x = W(x - \mu)`$;
2. draw $`z_1 \sim \mathcal N(0, I_6)`$;
3. solve the ordinary differential equation $`dz_t/dt = v_\phi(z_t, t, \tilde x)`$ from
   $`t = 1`$ down to $`t = 0`$, starting from $`z_1`$ (7.5);
4. return $`\theta = m_\theta + s_\theta \odot z_0`$;
5. repeat steps 2 to 4 with new noise for each of the $`S`$ draws.

$`q_\phi(\theta \mid y)`$ is, by definition, the distribution of the vector returned in
step 4. The network itself is deterministic; the spread of the draws comes from the noise
$`z_1`$.

Three things do not enter. The strikes, expiries and half-spreads are the same for every
input, since the design is fixed. $`c`$ is an output, never an input. And there is no
summary network: the 63 numbers enter directly.

**In BayesFlow.** The notebook builds the estimator in three parts: an adapter that names
the six parameters as the inference variables and the whitened quotes as the condition, a
`FlowMatching` inference network, and a `BasicWorkflow` that standardizes the parameters
and trains:

```python
adapter = (bf.Adapter()
           .concatenate(NAMES, into="inference_variables")   # theta: the six parameters
           .rename("observations", "inference_conditions"))   # x-tilde: the whitened quotes
network = bf.networks.FlowMatching(subnet_kwargs={"widths": (128,) * 3, "dropout": 0.1})
workflow = bf.BasicWorkflow(adapter=adapter, inference_network=network, standardize="inference_variables")
history = workflow.fit_offline(data=train, validation_data=validation, epochs=300, batch_size=256)
draws = workflow.sample(conditions={"observations": x_tilde}, num_samples=4096)
```

Each design choice has alternatives in BayesFlow:

| Choice | Here | Alternatives |
| --- | --- | --- |
| What is learned | the posterior $`q_\phi(\theta \mid y)`$ | a neural pricer, a likelihood or a likelihood ratio, each used inside a new MCMC run per set of quotes (7.3) |
| Summary network | none: a fixed vector of 63 whitened quotes | `DeepSet` or `SetTransformer` for quote sets of varying size and layout |
| Inference network | `FlowMatching` | `CouplingFlow` (exact density, one-pass sampling), `DiffusionModel`, `ConsistencyModel` (few sampling steps) |
| Parameter space | standardized and unbounded, so a few draws leave the prior | bounded transforms in the adapter, so that every draw respects the prior |
| Training data | a fixed set of simulations (`fit_offline`), one error draw per pair | fresh simulations in every batch (`fit_online`) |
| Amortized over | quotes of these 63 contracts with these spreads | contract terms and spreads as extra inputs, to reuse one network across days |

### 7.5 How the network is trained: flow matching

**The task.** For each $`\tilde x`$, the network must turn the noise distribution
$`\mathcal N(0, I_6)`$ into the posterior of $`\tilde\theta`$. It does so through a
velocity field: a rule that says, for a point at position $`z`$ at time $`t`$, in which
direction and how fast it moves. Training must find the field under which noise at
$`t = 1`$ arrives at $`t = 0`$ distributed as the posterior.

**From a bank row to a straight line.** Take a row $`(\tilde\theta^{(n)}, \tilde x^{(n)})`$,
an independent noise vector $`z_1 \sim \mathcal N(0, I_6)`$ and a time $`t`$, and place a
point on the straight line between the row's parameters and the noise:

```math
z_t = (1 - t)\,\tilde\theta^{(n)} + t\,z_1,\qquad \frac{dz_t}{dt} = z_1 - \tilde\theta^{(n)} .
```

The point is at the parameters when $`t = 0`$ and at the noise when $`t = 1`$, and it
moves with the constant velocity $`z_1 - \tilde\theta^{(n)}`$.

**The loss.** The network is asked to predict that velocity from the point, the time and
the row's quotes:

```math
\mathcal{L}(\phi) = \mathbb{E}\Big[\big\| v_\phi\big(z_t,\ t,\ \tilde x^{(n)}\big) - \big(z_1 - \tilde\theta^{(n)}\big) \big\|^2\Big] ,
```

with the expectation over the rows $`n`$, the noise $`z_1`$ and the time $`t`$, drawn
between 0 and 1 and more often near 0. Nothing else is needed: no likelihood, no density,
no pricing.

**Why minimizing it gives the posterior.** Many lines pass through the same point $`z`$
at the same time $`t`$, coming from different parameters and noise vectors (Figure 10,
left). A function of $`(z, t, \tilde x)`$ cannot follow each of them. The function that
minimizes a squared error is the conditional mean, so the best velocity is the average
over the lines through that point, among the rows whose quotes are $`\tilde x`$:

```math
v^\ast(z, t, \tilde x) = \mathbb{E}\big[\,z_1 - \tilde\theta \ \big|\ z_t = z,\ \tilde x\,\big] .
```

By 7.3, the parameters of the rows with quotes $`\tilde x`$ are draws from the posterior.
So, for a fixed $`\tilde x`$, the points $`z_t`$ are distributed as the noise at $`t = 1`$
and as the posterior of $`\tilde\theta`$ at $`t = 0`$. The averaged field moves a cloud of
points so that, at every time, the cloud has the distribution of $`z_t`$. To see it, let
$`p_t`$ be the density of $`z_t`$ and $`f`$ any smooth function:

```math
\frac{d}{dt}\,\mathbb{E}\big[f(z_t)\big] = \mathbb{E}\big[\nabla f(z_t) \cdot (z_1 - \tilde\theta)\big] = \mathbb{E}\big[\nabla f(z_t) \cdot v^\ast(z_t, t, \tilde x)\big] = -\int f(z)\ \nabla \cdot \big(p_t\,v^\ast\big)(z)\,dz .
```

The second equality replaces the velocity by its average given $`z_t`$; the third
integrates by parts. Since $`f`$ is arbitrary,
$`\partial_t p_t + \nabla \cdot (p_t\,v^\ast) = 0`$. That is the equation obeyed by the
density of points that move with velocity $`v^\ast`$. Started from the same density at
$`t = 1`$, the two agree at every $`t`$, and in particular at $`t = 0`$ (Lipman et al.
2023; Liu, Gong and Liu 2023; for posteriors, Wildberger et al. 2023).
[Lecture 9](../../../course/09-amortized-inference.md), Section 9.4, gives the same
argument with time running the other way, from noise at 0 to the target at 1.

![Flow matching on a one-parameter toy model](images/flow-matching.png)

**Figure 10.** A toy model with one parameter, $`\theta \sim U(-2, 2)`$ and
$`y = \theta^3/4 + 0.3\,\epsilon`$, for which the averaged velocity can be computed
exactly, without a network.
- *Left:* training pairs. Each line joins a noise draw (at $`t = 1`$) to a simulated
  parameter (at $`t = 0`$). Lines cross. The lines whose simulated $`y`$ is near 1 (red)
  end where $`\theta`$ explains $`y = 1`$; those with $`y`$ near −0.2 (blue) end elsewhere.
- *Centre:* sampling. Points started from the same noise follow the averaged velocity
  for $`y = 1`$ (red) or $`y = -0.2`$ (blue). Paths for one $`y`$ do not cross.
- *Right:* their end points against the exact posteriors: mean 1.51 and standard
  deviation 0.25 against 1.51 and 0.26 for $`y = 1`$; −0.22 and 0.69 against −0.24 and
  0.68 for $`y = -0.2`$.

The observation enters only as an input of the velocity, and that is enough to send the
same noise to two different posteriors. The Heston network does the same with six
parameters and 63 inputs.

**Sampling.** The equation $`dz_t/dt = v_\phi(z_t, t, \tilde x)`$ is solved from $`t = 1`$
to $`t = 0`$ with an adaptive Runge–Kutta solver (Tsit5).

**Another way to fit a posterior network.** A network can instead output a density
$`q_\phi(\theta \mid y)`$ and be trained by minimizing
$`-\frac1N\sum_n \log q_\phi(\theta^{(n)} \mid y^{(n)})`$ (neural posterior estimation).
In expectation this loss equals the average over $`y`$ of the Kullback–Leibler divergence
from the posterior to $`q_\phi`$, plus a constant, so it is also minimized by the
posterior (Lecture 9, Section 9.2). The comparison of 7.6 used such a network; the Heston
network here uses flow matching.

### 7.6 Preparing the input: whitening

**What the scaled quotes look like.** Write $`s(\theta) = P(\theta)/h`$ for the model
prices in half-spreads. By 6.1,

```math
x = s(\theta) + c\,\epsilon,\qquad \mathrm{Cov}(x) = \mathrm{Cov}\big(s(\theta)\big) + \mathbb{E}[c^2]\,I_{63},\qquad \mathbb{E}[c^2] = \frac{2^3 - 0.5^3}{3 \times 1.5} = 1.75 ,
```

with the covariances taken over the prior. The two terms have very different sizes:
- *the parameters* move all 63 quotes together, by hundreds to thousands of half-spreads,
  along a few directions (level, slope, curvature, term structure);
- *the errors* move each quote independently, by $`c`$ half-spreads.

**Where the information about the error scale is.** Let $`\lambda_1 \ge \dots \ge \lambda_{63}`$
be the eigenvalues of $`\mathrm{Cov}(x)`$ and $`u_k`$ its eigenvectors. In a direction
$`u_k`$ that the parameters do not move, $`\lambda_k \approx 1.75`$ and the coordinate of
an input is pure error:

```math
u_k^\top (x - \mu) \approx c\ u_k^\top \epsilon \sim \mathcal N(0, c^2) .
```

So the size of $`x`$ along these directions measures $`c`$. With $`K`$ of them,

```math
\hat c = \sqrt{\frac1K \sum_{k} \big(u_k^\top (x - \mu)\big)^2}
```

estimates $`c`$, with relative error about $`1/\sqrt{2K}`$ when those directions are pure
error (the precision argument of 6.3).

![Variance of the inputs by direction, correlations before and after whitening, and the error scale read from the small directions](images/whitening-spectrum.png)

**Figure 11.** From the notebook's 1,000 surfaces simulated from the prior at the 63
quotes (its Section 2.2); $`\mu`$ and the directions are fitted on 700 of them and applied
to the other 300, as in the notebook's Section 2.3.
- *First panel:* the variances $`\lambda_k`$. The largest is $`1.9 \times 10^6`$ and the
  smallest 1.0: a ratio (condition number) of 1.8 million. The first three directions
  hold 97% of the variance. 33 directions (orange) lie below 3.5, twice the error
  variance: there the inputs carry quote errors and little else.
- *Second and third panels:* correlations between the 63 inputs, before and after
  whitening. Before, almost every pair is strongly correlated (median absolute
  correlation 0.70); after, the largest is 0.26.
- *Last panel:* $`\hat c`$ computed from the 33 orange directions against the true $`c`$,
  on the 300 held-out surfaces: correlation 0.84, relative error 27%. The error scale is in
  the data, in the smallest directions. The notebook shows the same thing another way:
  the correlation of $`|u_k^\top(x - \mu)|/\sqrt{\lambda_k}`$ with $`c`$ averages 0.34 over
  the 30 directions of lowest variance and 0.06 over the 10 of highest.

**Why a network does not find it unaided.** A network fed $`x`$, or each $`x_i`$ rescaled
on its own, sees inputs whose informative directions for $`c`$ are a million times
smaller in variance than the dominant ones; rescaling each input separately leaves the
condition number at 2.1 million, because the small directions are combinations of all 63
quotes. For gradient descent on a linear first layer, the curvature of the loss in the
weights is the input covariance: the step size is limited by the largest eigenvalue, and
progress along the smallest is about one part in the condition number per step. More data
or more steps do not change the condition number. (The argument is exact only for a linear
layer; it is a guide.) The same reasoning explains why the Heston parameters are learned
easily: they live in the large directions.

**The transform.** From the first $`N_w`$ = 10,000 rows of the bank, compute the mean and
covariance of the scaled quotes and the eigendecomposition of the covariance:

```math
\mu = \frac{1}{N_w}\sum_{n=1}^{N_w} x^{(n)},\qquad \Sigma = \frac{1}{N_w - 1}\sum_{n=1}^{N_w} \big(x^{(n)} - \mu\big)\big(x^{(n)} - \mu\big)^\top = U\,\mathrm{diag}(\lambda_1, \dots, \lambda_{63})\,U^\top ,
```

with $`U`$ the orthogonal matrix whose columns are the $`u_k`$. The whitening matrix and
the whitened input are

```math
W = U\,\mathrm{diag}\big(\lambda_1^{-1/2}, \dots, \lambda_{63}^{-1/2}\big)\,U^\top = \Sigma^{-1/2},\qquad \tilde x = W\,(x - \mu),\qquad \mathrm{Cov}(\tilde x) = W\,\Sigma\,W^\top = I_{63} .
```

In words: rotate onto the directions $`u_k`$, divide the coordinate along each by its
standard deviation $`\sqrt{\lambda_k}`$, rotate back. Eigenvalues below
$`10^{-8}\lambda_1`$ would be raised to that floor; none is.

**What it changes, and what it does not.**
- $`W`$ is invertible, so $`\tilde x`$ contains exactly the information of $`x`$, and the
  exact posterior given $`\tilde x`$ is the posterior given $`y`$.
- Every direction now has variance 1, so the 33 error directions weigh as much in the
  input as the leading ones. In whitened coordinates the errors have covariance
  $`c^2\,W W^\top`$.
- This choice of $`W`$ (ZCA whitening) is symmetric. Among the matrices that give identity
  covariance, it keeps $`\tilde x`$ closest to $`x`$, so each whitened input still
  corresponds mostly to one quote.

**A controlled comparison.** Same simulated data (a 54-quote design, 10,000 training
rows), same network (a spline coupling flow trained by neural posterior estimation), same
initial weights; only the input scaling differs.

![The inferred multiplier with and without whitening, in a controlled comparison](images/whitening-comparison.png)

**Figure 12.** Posterior median of $`c`$, with 90% intervals, against the true $`c`$ on
512 simulated surfaces. Left: each input scaled on its own; the network returns the prior
of $`c`$ whatever the data. Right: whitened inputs.

| 512 simulated surfaces | Each input scaled on its own | Whitened |
| --- | ---: | ---: |
| Condition number of the input covariance | 2.2 million | 1 |
| Correlation of the true $`c`$ with its posterior median | 0.00 | 0.92 |
| Root-mean-square error of the posterior median of $`c`$ | 0.44 | 0.17 |
| Median width of the 90% interval for $`c`$ | 1.34 (the prior's is 1.35) | 0.37 |

**Rules.**
- Fit $`\mu`$ and $`W`$ on training rows only, with their errors included, in float64;
  never on validation, test or market quotes.
- Store them with the network, and apply the same transform at training, validation and
  use, the market included.
- Whiten the whole noisy vector $`x`$. Adding errors after whitening clean prices would
  change the error model.
- One transform per model and design: a new set of quotes, or rough Bergomi, gets its own.

### 7.7 The training run

| Item | Value |
| --- | --- |
| Rows | $`N`$ = 300,000 pairs $`(\theta^{(n)}, y^{(n)})`$; each keeps one error draw $`\epsilon^{(n)}`$ for the whole run; 100 more rows for validation |
| Input | $`\tilde x^{(n)} = W(x^{(n)} - \mu)`$ |
| Target | $`\tilde\theta^{(n)}`$; no transform keeps draws inside the prior box, so draws that fall outside it are kept and counted (8.2) |
| Network | $`v_\phi`$: three hidden layers of 128 units, a 32-dimensional embedding of $`t`$, dropout 0.1 |
| Passes | 300 passes over the rows (90 million presentations), in batches of 256: 1,172 updates per pass, 351,600 in all |
| Optimizer | AdamW, weight decay 0.005, gradient-norm clipping at 1.5 |
| Learning rate | from $`5 \times 10^{-5}`$ to $`5 \times 10^{-4}`$ over the first 17,580 updates (5%), then to zero along a cosine |
| Seed | 9011 |
| Network used | the one after the last pass; the pass with the lowest validation loss (259) was also evaluated and is no better |
| Software | BayesFlow's `FlowMatching` network in a `BasicWorkflow`, trained with `fit_offline` (7.4) |
| Cost | 30 minutes of training on the processor of an Apple-silicon laptop; 615 posteriors drawn for step 8 in 164 s, under 0.3 s each |

### 7.8 Rough Bergomi

The same construction, with $`\theta = (H, \eta, \rho, \xi_s, \xi_l, \lambda, c)`$, seven
numbers, its own bank (4.2) and its own $`(\mu, W)`$. The differences are in cost: a
pricing takes minutes, so the bank is the expensive part (7.3), and the exact reference of
step 8 uses importance sampling, which needs about 1,000 pricings per posterior where MCMC
needs about 180,000. Its bank is being generated (4.2); the network and its checks
follow.

## 8. Checking it

The network's $`q_\phi(\theta \mid y)`$ is an approximation, and the model behind it is an
assumption. The notebook runs five checks:

| Check | Question | What can be wrong | What the check needs |
| --- | --- | --- | --- |
| Posterior predictive (8.4) | Can the inferred parameters regenerate the quotes? | the model, or the posterior | posterior draws and the pricer |
| Calibration (8.2) | Does a 90% interval contain the truth 90% of the time? | on average over the surfaces the model can produce, $`q_\phi`$ is not the posterior | many simulated surfaces with known parameters |
| Recovery (8.2) | Does the posterior median find the true parameters? | $`q_\phi`$ is uninformative or biased | the same surfaces |
| Comparison with the exact posterior (8.3) | Does the network reproduce the posterior of one $`y`$? | for one given $`y`$, $`q_\phi(\theta \mid y)`$ is not the posterior | the exact posterior of that $`y`$ (8.1) |
| The model on the market (8.4) | Can Heston describe the market at all? | the model, so that even the exact posterior answers the wrong question | the market quotes |

### 8.1 Reference posteriors

**Heston, exact, by MCMC.**
- *Target:* $`p(\theta)\,p(y \mid \theta)`$, with the uniform prior of 4.1 and the
  likelihood of 6.2 computed with Fourier prices, written in PyMC.
- *Sampler:* DEMetropolisZ, which needs no gradients. Gradient-based samplers would need
  validated gradients of the Fourier likelihood ([MCMC policy](../mcmc-policy.md)).
- *Schedule:* 4 chains. A first run of 2,000 tuning steps and 2,000 draws per chain; if
  its diagnostics fail, one declared extension to 6,000 and 12,000. All three cases
  needed the extension.
- *Starts:* maximum-likelihood fits to the data alone (Powell's method) and dispersed
  points; never the truth or the network.
- *Acceptance:* rank-normalized $`\hat R \le 1.01`$ and bulk and tail effective sample
  sizes of at least 400, for every component of $`\theta`$ and for the two variance
  strikes of 9.2; and the pricing check of 4.1 at sampled draws.

| Case | Parameters that generated it | Likelihood evaluations, both runs (accepted run) | Minutes on one core, both runs (accepted run) | Max $`\hat R`$ | Min bulk / tail ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| Simulated, market-like | $`v_0`$ 0.027, $`\kappa`$ 4.34, $`v_\infty`$ 0.056, $`\sigma_v`$ 1.80, $`\rho`$ −0.67, $`c`$ 1 | 180,174 (146,087) | 31.6 (26.3) | 1.003 | 1,958 / 3,120 |
| Simulated, stressed market | $`v_0`$ 0.136, $`\kappa`$ 3.08, $`v_\infty`$ 0.083, $`\sigma_v`$ 1.72, $`\rho`$ −0.78, $`c`$ 1 | 180,188 (146,094) | 16.8 (14.1) | 1.002 | 2,183 / 2,863 |
| SPX quotes, 9 November 2026 | unknown | 180,226 (146,113) | 32.1 (26.8) | 1.002 | 2,078 / 2,301 |

The market-like case uses parameters within 15% of the market posterior. The stressed
case is a high-volatility draw from the prior: volatility starts at 37% and reverts
towards 29%. The times are for one CPU core of a cloud machine; the same runs took 7 to
15 minutes on the workstation that produced the study. The references were computed
twice, on the two machines: their means agree within 0.07 posterior standard deviations
and their 90% widths within 2.3%.

**Rough Bergomi, by importance sampling with the network as proposal.** Draw
$`\theta^{(s)} \sim q_\phi(\theta \mid y)`$, $`s = 1, \dots, 1{,}000`$, and weight each
draw by the ratio of the exact posterior density, up to a constant, to the network's
(planned, once the rough Bergomi network exists):

```math
\tilde w_s = \frac{p\big(\theta^{(s)}\big)\,\hat p\big(y \mid \theta^{(s)}\big)}{q_\phi\big(\theta^{(s)} \mid y\big)} ,
```

where $`\hat p(y \mid \theta)`$ is the likelihood of 6.2 computed with Monte Carlo prices
from a fixed seed, with their Monte Carlo covariance added to the error covariance. The
fixed seed makes $`\hat p`$ a deterministic function of $`\theta`$, so the weighted draws
target the exact posterior of that numerical model. This costs 1,000 pricings, not
180,000, because the network has already put the draws where the posterior is. It is
accepted when the effective sample size
$`\big(\sum_s \tilde w_s\big)^2 / \sum_s \tilde w_s^2`$ is at least 200 and the
Pareto-smoothed tail index $`\hat k`$ is below 0.7; otherwise the network is reported as
not checked on that case.

### 8.2 Population checks on simulated surfaces

**The test set.** $`M = 512`$ surfaces simulated as in 7.3, each with its known
parameters $`\theta^{\ast}_m`$ and quotes $`y_m`$, and for each of them $`L = 512`$ draws
$`\theta^{(m,l)} \sim q_\phi(\theta \mid y_m)`$. Below, $`j`$ indexes the six components
of $`\theta`$.

**Recovery.** How close the posterior median of component $`j`$ is to the truth, across
the surfaces: their correlation over $`m`$, and the average width of the 90% intervals as
a share of the prior's range.

![Recovery: true parameter against posterior median on 512 simulated surfaces](images/heston-recovery.png)

**Figure 13.** True value $`\theta^{\ast}_{m,j}`$ against the posterior median, with 90%
intervals on 60 of the surfaces. The five Heston parameters are recovered almost exactly;
$`c`$, which is read from the errors alone, has wider intervals.

**Coverage.** Let $`I^{\alpha}_{m,j}`$ be the central interval of level $`\alpha`$ of the
draws $`\theta^{(m,l)}_j`$, from their $`(1-\alpha)/2`$ to their $`(1+\alpha)/2`$
quantile. The coverage is the share of surfaces whose truth lies inside:

```math
\widehat{\mathrm{cov}}_j(\alpha) = \frac1M \sum_{m=1}^{M} \mathbf{1}\big\{\theta^{\ast}_{m,j} \in I^{\alpha}_{m,j}\big\} ,
```

which should equal $`\alpha`$ up to a binomial standard error
$`\sqrt{\alpha(1-\alpha)/M}`$, 1.3% at $`\alpha = 0.9`$.

**Rank test (simulation-based calibration).** The fractional rank of the truth among the
draws is

```math
u_{m,j} = \frac1L \sum_{l=1}^{L} \mathbf{1}\big\{\theta^{(m,l)}_j \lt \theta^{\ast}_{m,j}\big\} .
```

If $`q_\phi(\cdot \mid y_m)`$ is the posterior, the truth and the $`L`$ draws are $`L+1`$
exchangeable draws from the same distribution given $`y_m`$, so $`u_{m,j}`$ is uniform on
$`[0, 1]`$. The rank plot compares the empirical distribution of the 512 ranks with the
uniform, through

```math
\Delta_j(z) = \frac1M \sum_{m=1}^{M} \mathbf{1}\{u_{m,j} \le z\} - z,\qquad z \in [0, 1] .
```

For uniform ranks, $`M\,(\Delta_j(z) + z)`$ is binomial with $`M`$ trials and probability
$`z`$. The notebook's band (Säilynoja, Bürkner and Vehtari, 2022) bounds $`\Delta_j`$ at
every $`z`$ at once, with level $`1 - 0.05/6`$ for each component, so that a calibrated
network keeps all six curves inside with probability 95%; it is narrow at 0 and 1, where a
correct curve varies least. The study's record also gives the largest distance
$`D_j = \sup_z |\Delta_j(z)|`$, against the Dvoretzky–Kiefer–Wolfowitz bound at 5% shared
over the six components,

```math
D_j \ \le\ \sqrt{\frac{\ln(2 \times 6 / 0.05)}{2M}} = 0.073 .
```

Both criteria single out the same component. A network that returned the prior would also
pass (the left panel of Figure 12 is such a case), so the check is necessary, not
sufficient; recovery and the comparison of 8.3 test sharpness.

![Coverage and simulation-based calibration of the Heston network](images/heston-calibration.png)

**Figure 14.** Left: $`\widehat{\mathrm{cov}}_j(\alpha)`$ against $`\alpha`$; a calibrated
posterior lies on the diagonal, within the grey band. Right: $`\Delta_j(z)`$ for the six
components; a calibrated network keeps all six curves inside the grey band with
probability 95%.

| | $`v_0`$ | $`\kappa`$ | $`v_\infty`$ | $`\sigma_v`$ | $`\rho`$ | $`c`$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| $`\widehat{\mathrm{cov}}_j(0.5)`$ | 67.6% | 53.3% | 55.7% | 51.6% | 53.5% | 50.0% |
| $`\widehat{\mathrm{cov}}_j(0.8)`$ | 92.6% | 81.6% | 84.4% | 83.0% | 84.8% | 78.9% |
| $`\widehat{\mathrm{cov}}_j(0.9)`$ | 97.5% | 90.0% | 92.4% | 89.3% | 92.0% | 90.2% |
| $`\widehat{\mathrm{cov}}_j(0.95)`$ | 99.2% | 94.1% | 95.9% | 93.2% | 96.7% | 94.9% |
| Rank plot | leaves the band | inside | inside | inside | inside | inside |
| $`D_j`$ (bound 0.073) | 0.135 | 0.028 | 0.048 | 0.051 | 0.048 | 0.034 |
| Correlation of truth and posterior median | 0.99999 | 0.9997 | 0.9987 | 0.9999 | 0.996 | 0.95 |
| Average width of the 90% interval, share of the prior range | 0.35% | 2.0% | 1.3% | 1.2% | 3.4% | 28% |
| 95% confidence interval of $`\widehat{\mathrm{cov}}_j(0.9)`$ | 95.7–98.5% | 87.1–92.3% | 89.8–94.4% | 86.3–91.7% | 89.3–94.0% | 87.4–92.5% |
| Draws outside the prior range | 0.12% | 0.31% | 0.12% | 0.46% | 0.98% | 1.13% |

- **Five of the six components are calibrated** by two criteria: the 95% confidence
  interval of their 90% coverage contains 90%, and their rank curve stays inside the band
  (their $`D_j`$ is also below the bound). At
  the 50% and 80% levels $`v_\infty`$ and $`\rho`$ cover a little more than nominal
  (55.7% and 84.8%).
- **One component is not:** the intervals for $`v_0`$ are too wide. The truth falls
  near their centre too often. If a normal posterior is widened by a factor $`k`$, its nominal 90% interval
  covers $`\Pr(|N| \le 1.645\,k)`$ for a standard normal $`N`$; 97.5% gives
  $`k \approx 1.36`$, and the 50% and 80% levels give 1.46 and 1.39. So these intervals
  are about 1.4 times as wide as they should be: the network is conservative about
  $`v_0`$, not overconfident. The shortest quotes pin $`v_0`$ to a posterior standard
  deviation of 0.00016 on average, on a prior range of 0.15; a resolution finer than the
  network reaches is a possible cause, not a tested one.
- **The multiplier is recovered:** the root-mean-square error of the posterior median of
  $`c`$ is 0.135, and its 90% intervals cover 90.2%.
- **3.0% of all draws have a component slightly outside the prior range,** most often
  $`c`$ or $`\rho`$. They are kept in every check, not clipped.

### 8.3 Conditional checks against the references

For each of the three cases, draw $`S = 4{,}096`$ times from the network, with the seed of
its recorded evaluation, and compare component by component with the reference posterior
$`p`$ of 8.1. Two numbers per
component: the difference of the means in reference standard deviations, and the ratio of
the widths of the 90% intervals,

```math
\mathrm{ME}_j = \frac{\big|\,\mathbb{E}_{q_\phi}[\theta_j \mid y] - \mathbb{E}_{p}[\theta_j \mid y]\,\big|}{\mathrm{sd}_{p}(\theta_j \mid y)},\qquad
\mathrm{WR}_j = \frac{q^{0.95}_{\phi, j} - q^{0.05}_{\phi, j}}{q^{0.95}_{p, j} - q^{0.05}_{p, j}} ,
```

with $`q^{\alpha}_{\cdot, j}`$ the $`\alpha`$-quantile of $`\theta_j`$ under the network or
the reference. Exact agreement is $`\mathrm{ME}_j = 0`$ and $`\mathrm{WR}_j = 1`$; we read
$`\mathrm{ME}_j \lt 0.3`$ together with $`0.75 \le \mathrm{WR}_j \le 1.25`$ as agreement.

![Network against reference posterior on two simulated surfaces and the SPX quotes](images/heston-vs-reference.png)

**Figure 15.** For each case and component, the 90% interval of the reference (grey) and
of the network (colour), centred on the reference median and measured in reference
widths. On the two simulated surfaces the network brackets the reference with wider
intervals. On the SPX quotes it is far away (note the wider scale).

| Case | | $`v_0`$ | $`\kappa`$ | $`v_\infty`$ | $`\sigma_v`$ | $`\rho`$ | $`c`$ | Variance strike, 0 to 21 days | Variance strike, 81 to 325 days |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Market-like | $`\mathrm{ME}`$ | 0.97 | 0.11 | 0.16 | 0.32 | 0.06 | 0.33 | 0.87 | 0.14 |
| Market-like | $`\mathrm{WR}`$ | 2.91 | 1.07 | 1.43 | 1.13 | 2.31 | 1.27 | 2.74 | 1.89 |
| Stressed | $`\mathrm{ME}`$ | 0.73 | 0.29 | 0.14 | 0.02 | 0.62 | 0.40 | 0.78 | 0.11 |
| Stressed | $`\mathrm{WR}`$ | 1.88 | 1.55 | 2.05 | 1.44 | 2.72 | 1.37 | 2.09 | 1.83 |
| SPX | $`\mathrm{ME}`$ | 11.1 | 5.8 | 11.9 | 1.8 | 26.7 | 37.3 | 9.2 | 12.4 |
| SPX | $`\mathrm{WR}`$ | 5.7 | 1.7 | 3.4 | 2.0 | 4.8 | 59 | 5.3 | 4.2 |

The parameter columns compare with the references of 8.1, as the notebook does. The two
variance-strike columns come from the recorded evaluation, which compared with the first
computation of the references; the two computations agree within 0.07 standard deviations
in their means and 2.3% in their widths (8.1).

- **Simulated surfaces.** The centres are close: the median $`\mathrm{ME}_j`$ over the
  12 parameter posteriors is 0.31 and the largest 0.97. The intervals are wider: 10 of
  the 12 are more than 25% wider than the exact ones, up to 2.9 times, and none is
  narrower. One of the 12 ($`\kappa`$, market-like) meets both criteria. The network does
  not claim more precision than the quotes support, but it does not extract all the
  information in them either. This is compatible with 8.2: calibration on average does
  not imply sharp posteriors on each surface.
- **SPX quotes.** The network is far from the reference: 21% of its draws have
  $`c \gt 2`$, outside the prior range, and its means for the Heston parameters lie 1.8 to
  27 reference standard deviations away (37 for $`c`$). 8.4 explains why. Market results
  therefore use the reference posterior.

A calibrated posterior can still be too wide: a network that returned the prior would be
perfectly calibrated and useless. Only an exact reference shows how much sharpness is
lost.

### 8.4 Checks of the model: replicated quotes and the market

Everything in 8.2 and 8.3 uses surfaces simulated from the model, for which the model is
right by construction. On market quotes the model itself is in question.

**Before training: could the generative model have produced the market?** Simulate
surfaces from the prior and compare them with the market quotes (a prior predictive
check).

![Smiles simulated from the prior against the market smile at three expiries](images/prior-predictive.png)

**Figure 16.** Implied volatilities of 150 of the notebook's 1,000 surfaces drawn from the
prior (blue) and of the market (orange), at three expiries. The prior produces calm and
stressed markets, flat and steep skews, and the market lies inside this range, on its
low-volatility side at 7 days. Among the 1,000 simulated surfaces, each market quote lies
between the 6th and the 56th percentile of its simulated values; the lowest is the 7-day
call at $`\delta = 2.0`$. Of the simulated quotes, 0.4% have no implied volatility,
because the error pushed a very cheap option below its intrinsic value; they remain valid
data, since errors are added to prices.

This shows that the prior does not exclude the data, quote by quote. It does not show
that some $`\theta`$ fits all 63 quotes at once.

**After inference: does the fitted model reproduce the quotes within the assumed
errors?** Under the error model, the scaled residuals
$`r_i(\theta) = (y_i - P_i(\theta))/h_i`$ at the true $`\theta`$ are independent
$`\mathcal N(0, c^2)`$, so their root mean square should be close to $`c`$, at most 2. On
the SPX quotes, at the mean of the exact posterior, it is 13.4, and the residuals follow
a smooth pattern across strikes (6.5). No Heston parameters reproduce these quotes within
the assumed errors.

**Posterior predictive check.** Draw $`\theta^{(m)}`$ from the posterior, price the 63
contracts and add fresh errors with each draw's own multiplier:

```math
\tilde y^{(m)}_i = P_i\big(\theta^{(m)}\big) + c^{(m)}\,h_i\,\epsilon^{(m)}_i,\qquad \epsilon^{(m)}_i \sim \mathcal N(0, 1) .
```

The prices $`P_i(\theta^{(m)})`$ carry the parameter uncertainty; the replicated quotes
$`\tilde y^{(m)}_i`$ add independent quote errors. If the model and the posterior are
right, the observed quotes look like one more replicated set: the band of
$`\tilde y^{(m)}_i - y_i`$ covers zero at most quotes. The notebook prices 40 draws of
each posterior, keeping each price paired with its own $`c`$; network draws outside the
prior are left out, because the pricer's accuracy is checked on the prior's range.

![The posterior predictive check on simulated quotes and on the SPX quotes](images/posterior-predictive.png)

**Figure 17.** The check at the 81-day expiry, from 40 draws of the exact posterior, in
half-spreads: model prices minus the observed quotes (purple) and replicated quotes minus
the observed quotes (green), pointwise 90% bands. Left: simulated market-like quotes; the
replicated band covers zero, and the model prices follow the true prices (dashed) more
closely than any single quote measures them. Right: the SPX quotes.

| Data set | RMS of the posterior-mean prices minus the quotes, 81 days (half-spreads) | Median $`c`$ of the draws |
| --- | ---: | ---: |
| Simulated, market-like | 0.69 | 0.92 |
| Simulated, stressed market | 1.13 | 0.97 |
| SPX quotes | 17.0 | 2.00 |

On the simulated sets the misses are of the size of $`c`$, as they should be. On the SPX
quotes they are eight times the largest $`c`$ the prior allows, with the systematic pattern
of 6.5.

**Why the network and the reference disagree on the market.** The network was trained on
surfaces that lie within $`c \le 2`$ half-spreads of a Heston surface. The market quotes
lie 13.4 half-spreads, root mean square, from the best-fitting one: an input unlike any
in the bank. Like any
learned function, the network is reliable only on inputs that resemble its training
data. The disagreement of 8.3 is therefore a symptom of the model's misfit, not a failure
of training, and comparing with an exact reference is what reveals it.

**Checks still to compute:**
- prediction of the held-out quotes (5.4);
- the model's variance-swap strikes against those of the listed strips (1.5);
- the listed 2-year 90% put against the model's prediction (9.8).

Under the smooth model-error law of 6.6, the prediction for a held-out quote with model
price $`P_\star(\theta)`$, vega $`\nu_\star`$ and half-spread $`h_\star`$ includes the
model error carried over from the design. Let
$`k_\star = \tau^2\,\nu_\star\,\mathrm{diag}(\nu)\,K_\ell(\text{design}, \star)`$ be the
covariance between the model error at the held-out point and at the 63 quotes. Gaussian
conditioning gives, for each posterior draw,

```math
y_\star \sim \mathcal N\Big(P_\star(\theta) + k_\star^\top\,\Gamma^{-1}\big(y - P(\theta)\big),\ \ \tau^2\nu_\star^2 - k_\star^\top\,\Gamma^{-1} k_\star + (c\,h_\star)^2\Big) .
```

Under the law of 6.1 the prediction is $`P_\star(\theta) + c\,h_\star\,\epsilon_\star`$,
which will cover too rarely wherever the misfit is real.

## 9. From the posterior to the products

### 9.1 Principle

A product's value is a function of the parameters,
$`G_j(\theta) = \mathbb{E}_\theta[\text{discounted payoff}]`$, and the posterior induces its
distribution (step 3). For each posterior draw, price each product. Path-dependent products
need simulation: every draw uses the same normals (common random numbers), so Monte Carlo
noise moves all draws together and the spread across draws is the parameters' (9.9). The
error scales $`c`$ and $`\tau`$ do not enter: they describe how listed options are
observed, not how the index moves.

### 9.2 Forward-starting variance: closed form

For each draw the strike is the closed form of 4.1 (Heston) or 4.2 (rough Bergomi). Two
windows are marked: a short one, from today to the 30 November 2026 expiry (21 days), and
a later one, from the 29 January to the 30 September 2027 expiries (81 to 325 days). Under
the exact SPX posterior of Heston:

| Window $`[t_1, t_2]`$ | Mark $`\bar G`$ | Posterior sd | 90% range, $`q_{0.05}`$ to $`q_{0.95}`$ | Prudent value, long, $`q_{0.10}`$ | Reserve $`\bar G - q_{0.10}`$ |
| --- | ---: | ---: | --- | ---: | ---: |
| 0 to 21 days | 17.73% | 0.029 vp | 17.69% to 17.78% | 17.70% | 0.036 vp |
| 81 to 325 days | 22.61% | 0.050 vp | 22.53% to 22.70% | 22.55% | 0.065 vp |

Here $`G = \sqrt{K_{\text{var}}(t_1, t_2)}`$, the strike as a volatility, and the numbers
come from 1,000 draws of the exact posterior (Figure 3 shows the second row). A variance
swap is usually sized by its vega notional, the amount paid per vol point of realized
volatility above the strike; per 100,000 of vega notional, these reserves are about 3,600
and 6,500 before discounting.

The ranges are a few hundredths of a vol point wide, while at these maturities Heston
misses the quotes by 0.8 to 2.2 vol points (6.5). They are the uncertainty of Heston's
parameters given that Heston is right, with errors capped at twice the half-spread. Under
the smooth model-error law of 6.6 they would widen: the variance level loads on exactly
the smooth directions that law adds.

On the two simulated surfaces, the network's values for both windows lie within 0.9
reference standard deviations of the reference's, with 90% ranges 1.8 to 2.7 times as wide
(8.3).

### 9.3 Path simulation

- **Heston, as the notebook prices it:** full-truncation Euler for the variance, with
  $`V^+ = \max(V, 0)`$ in the drift and the volatility,
  $`V_{t+\Delta} = V_t + \kappa(v_\infty - V_t^+)\,\Delta + \sigma_v\sqrt{V_t^+\,\Delta}\,Z`$,
  over 6 months in 252 steps ($`\Delta = 1/504`$ year), for 4,096 paths
  (`calibration.products.heston_paths`). Each path stores $`I_t`$ and $`J_t`$, the
  integrals of the notation section, as sums over the steps.
- **No index path is drawn.** Given the variance path, the log index between two dates is
  Gaussian (notation section), so each product is a Black–Scholes formula per variance
  path, averaged over the paths.
- **Common random numbers:** the same normals (seed 20261115) for every posterior draw and
  every refit, so that the spread of prices across draws is the parameters' and not
  Monte Carlo noise.
- **Units:** forward units, with zero carry and no discounting; the forward-start call as
  a percentage of its notional, the volatility swap as an annualized volatility in percent.
- **Rough Bergomi, next:** the hybrid scheme of 4.2, with $`I`$ and $`J`$ stored in the
  same layout (`calibration.products.rough_paths`).
- **Fit and products on one model:** Heston's fit uses Fourier prices and its products use
  Euler paths. A check that the paths reprice the design's options within 0.05 vp of
  Fourier at the posterior median, with the time step halved otherwise, is among the next
  steps (10.3).

### 9.4 Volatility swap, 3 months

$`K_{\text{vol}}(T) = \mathbb{E}\big[\sqrt{I_T/T}\big]`$ with $`T`$ = 3 months, estimated by
the path mean of $`\sqrt{I_T/T}`$ with one control variate
(`calibration.products.volatility_swap`): $`J_T`$ for Heston, whose Euler scheme shifts the
mean of $`I`$ but leaves $`J`$ with mean zero; $`I_T/T`$ minus its exact mean for rough
Bergomi, whose discrete mean is exact.

**Convexity.** Expanding $`\sqrt{x}`$ to second order around
$`m = K_{\text{var}} = \mathbb{E}[x]`$, with $`x = I_T/T`$:

```math
\sqrt{x} \approx \sqrt{m} + \frac{x - m}{2\sqrt m} - \frac{(x-m)^2}{8\,m^{3/2}}\qquad\Longrightarrow\qquad K_{\text{vol}} \approx \sqrt{K_{\text{var}}} - \frac{\mathrm{Var}(I_T/T)}{8\,K_{\text{var}}^{3/2}} .
```

The gap between the volatility and the variance strike is set by the variance of realized
variance, that is by the vol of vol. This is why the product depends on the wings and the
short maturities, which pin the vol of vol (5.2).

### 9.5 Forward-start call, 3 to 6 months

**Payoff.** At $`T_2`$ = 6 months, per unit of notional, $`(S_{T_2}/S_{T_1} - 1)^+`$ with
$`T_1`$ = 3 months: the percentage rise of the index between the two dates, if positive.
In forward units with zero carry, $`S_{T_2}/S_{T_1} = X_{T_2}/X_{T_1}`$.

**Conditional formula.** Write $`\Delta I = I_{T_2} - I_{T_1}`$ and
$`\Delta J = J_{T_2} - J_{T_1}`$. Given the path of $`W^V`$, the notation section gives

```math
\ln\frac{X_{T_2}}{X_{T_1}} \;\Big|\; W^V \;\sim\; \mathcal N\Big(\rho\,\Delta J - \tfrac12\,\Delta I,\ \ (1-\rho^2)\,\Delta I\Big),
```

so, with $`F' = e^{\rho\,\Delta J - \frac12\rho^2\,\Delta I}`$ and $`w = (1-\rho^2)\,\Delta I`$
(`calibration.products.forward_start_call`),

```math
\mathbb{E}\Big[\Big(\frac{X_{T_2}}{X_{T_1}} - 1\Big)^+ \;\Big|\; W^V\Big] = \mathrm{BS}_{\text{call}}\big(F',\ 1,\ w\big),
```

the undiscounted Black–Scholes call with forward $`F'`$, strike 1 and total variance $`w`$.
Two controls with mean zero reduce the noise: $`F' - 1`$ and $`\Delta J`$.

**Why it matters.** Its value depends on the smile that will prevail at $`T_1`$, which no
listed option shows. Heston's future skew flattens through mean reversion, rough
Bergomi's persists: this is where the two models are expected to differ most.

### 9.6 Results: posterior ranges and refits

**Posterior ranges.** For each data set, 40 posterior draws inside the prior are priced
with the common random numbers of 9.3: draws of the exact posterior, and of the network
when it is loaded. On the market-like simulation the true parameters are known, so the
true prices are too.

**Refits.** Heston is refitted by least squares in half-spreads, minimizing
$`\sum_{i \in A} \big((y_i - P_i(\vartheta))/h_i\big)^2`$ over the five pricing
parameters $`\vartheta`$ within the prior box, from the posterior mean, for four subsets
$`A`$ of the SPX quotes: all 63; the three shortest expiries (7 to 30 days); the four
longest (60 to 325 days); and the quotes near the money ($`|\delta| \le 1.3`$). Each refit
prices the same products.

| | Forward-start call (% of notional) | Volatility swap (%) | $`\kappa`$ | $`\sigma_v`$ | RMS miss on its quotes (half-spreads) |
| --- | --- | --- | ---: | ---: | ---: |
| Market-like simulation: true value | 3.127 | 15.79 | 4.34 | 1.80 | |
| Market-like simulation: exact posterior, 90% range | 3.125 to 3.139 | 15.77 to 15.80 | | | |
| SPX: exact posterior, 90% range (medians of $`\kappa`$, $`\sigma_v`$) | 3.164 to 3.188 | 16.14 to 16.21 | 5.07 | 1.80 | 13.4 |
| SPX: refit to all 63 quotes | 3.170 | 16.16 | 5.07 | 1.80 | 13.4 |
| SPX: refit to the 3 shortest expiries | 3.579 | 17.63 | 10.00 | 2.10 | 10.8 |
| SPX: refit to the 4 longest expiries | 3.130 | 16.35 | 4.08 | 1.61 | 12.6 |
| SPX: refit to the quotes near the money | 3.282 | 16.21 | 4.54 | 1.30 | 8.4 |

![The forward-start call under posterior draws and under refits](images/refits.png)

**Figure 18.** The forward-start call under 40 draws of the exact posterior (dots; the bar
is the 90% range) and under the four refits (diamonds). Top: the market-like simulation,
where Heston is right. Bottom: the SPX quotes.

- **Where the model is right,** both ranges contain the true value; the forward-start
  call's range is ±0.22% of its price.
- **On SPX,** the refits spread 18 times wider than the posterior range for the
  forward-start call and 21 times for the volatility swap. The refit to the shortest
  expiries puts $`\kappa`$ at its upper bound of 10 and $`\sigma_v`$ at 2.1, to reach their
  skew; the refit near the money needs only $`\sigma_v = 1.30`$, because it no longer
  has to reach the deep puts. Even on its own quotes each refit misses by 8 to 13
  half-spreads.
- **The network on SPX** gives ranges that differ from the exact posterior's, because the
  SPX quotes lie outside its training data (8.4).

A posterior measures uncertainty within the model. On SPX the model is wrong, so its
parameters depend on which quotes it is fitted to, and the posterior cannot see that. A
desk would hold a reserve sized to the refit spread, add a model-error term (6.6), or use
a better model. The posterior range matters most where the quotes leave the price open:
with few or wide quotes, or with a model flexible enough that the quotes do not pin what
the product depends on.

### 9.7 Down-and-in put, 1 year, strike 100%, barrier 70%, daily closes

Specified, not yet priced (10.3).

**Payoff** at $`T`$ = 1 year: $`(K - S_T)^+\,\mathbf{1}\{\min_i S_{t_i} \le B\}`$, with
$`K = S_0`$, $`B = 0.7\,S_0`$ and $`t_i`$ the 252 closes.

**Where it sits in the design.** At 1 year the forward is about 3.9% above spot, so the
barrier is near $`\delta = -2.5`$ and the strike near $`\delta = -0.24`$. At 325 days the design's
puts have strikes at 48%, 61% and 74% of spot, so the deep puts bracket the barrier
(Figure 4).

**The index path from the stored quantities.** Given $`W^V`$, $`J^{\perp}`$ has independent
Gaussian increments with variances $`\Delta I_i = I_{t_i} - I_{t_{i-1}}`$, so, exactly given
the variance path,

```math
\ln X_{t_i} = \rho\,J_{t_i} + \sqrt{1-\rho^2}\,J^{\perp}_{t_i} - \tfrac12\,I_{t_i},\qquad J^{\perp}_{t_i} = J^{\perp}_{t_{i-1}} + \sqrt{\Delta I_i}\ \zeta_i,\qquad \zeta_i \sim \mathcal N(0,1)\ \text{independent}.
```

Then $`S_{t_i} = F(t_i)\,X_{t_i}`$, with the forwards of 1.2. The barrier test is
$`X_{t_i} \le B/F(t_i)`$: in forward units the barrier falls as the forward rises.

**Estimator.** The path mean of $`D(T)\,(K - S_T)^+\,\mathbf{1}\{\text{hit}\}`$, with
antithetic pairs and the vanilla put $`(K - S_T)^+`$ as a control variate with its exact
value (Fourier for Heston; the conditional estimator of 4.2 on the same paths for rough
Bergomi). The identity down-and-in + down-and-out = vanilla is checked path by path.

**Tests of the new estimator** (to be added to `calibration.products`):
1. *Black–Scholes limit.* With $`\sigma_v \to 0`$ and $`v_0 = v_\infty = \sigma^2`$, compare with
   the closed-form continuously monitored down-and-in put at the shifted barrier
   $`B\,e^{-0.5826\,\sigma\sqrt{\Delta t}}`$, the Broadie–Glasserman–Kou correction for
   daily monitoring ($`0.5826 = -\zeta(1/2)/\sqrt{2\pi}`$, with $`\zeta`$ the Riemann zeta
   function). Pass within 3 standard errors plus 0.5% of the price.
2. *Monitoring frequency.* Halving it must move the price in the direction, and by about
   the size, the correction predicts.
3. *Determinism.* The same seed gives the same price.

**Why it matters.** It is the risk inside autocallable notes, and the issuing bank is long
it. It depends on the deep put wing out to 1 year and on how volatility rises when the
index falls ($`\rho`$ and the vol of vol). At Heston's SPX posterior mean, about 13% of
simulated 1-year paths touch the barrier.

### 9.8 The listed 2-year 90% put (check)

Specified, not yet computed (10.3). The model price at the listed expiry nearest 2 years and the strike nearest
$`0.9\,F(T)`$: Fourier for Heston, the conditional estimator on 2-year paths for rough
Bergomi. Converted to implied volatility and compared with the market's, ± its
half-spread: extrapolation in maturity, checked against a listed price.

### 9.9 Monte Carlo accuracy

- **In 9.6:** 4,096 paths and 40 draws per range, with common random numbers, so the
  simulation error of a price is shared by all draws and does not widen a range. With 40
  draws each end of a 90% range rests on about two draws, so the ranges are rough; the
  factor of about 20 between the refits and the posterior ranges does not hinge on that
  precision.
- **For the full set of products and the book (10.3):**
  - *paths:* per product and model, enough that the standard error at the posterior
    median is at most 2% of the product's posterior 90% range; 20,000 to 100,000
    antithetic paths are expected for Heston and 10,000 to 50,000 for rough Bergomi;
  - *draws:* with $`S = 1{,}000`$ draws, the 90% quantile's error in probability is about
    $`\sqrt{0.9 \times 0.1/1000} \approx 0.01`$; reported by bootstrap over the draws;
  - *paths, for quantiles:* all draws rerun with an independent seed, and the differences
    reported;
  - *requirement:* with $`\mathrm{SE}_j`$ either of these errors for product $`j`$,
    $`\mathrm{SE}_j \le 0.02\,(q_{0.95} - q_{0.05})`$; otherwise more paths or draws.

### 9.10 Summaries, prudent value and the book

Mark, prudent value and reserve as in 3.2. The example book, among the next steps (10.3):

| Position | Size | Why |
| --- | --- | --- |
| Down-and-in put, 1 year | long 1 | the issuer's side of autocallables |
| Forward-start call | long 1 | exposure to the future smile |
| Volatility swap | short 1 | exposure to the vol of vol |

The put is in index points per unit of index, the forward start a percentage of its
notional, the volatility swap in vol points per unit of vega notional; the book fixes each
notional in currency. Reported: the book's prudent value $`q_{0.10}(B^{(s)})`$, its
reserve, the diversification ratio $`R_{\text{book}}/\sum_j |\omega_j|\,R_j`$ against the
Regulation's fixed 0.5, and the correlation matrix of the product values across draws.

### 9.11 Baselines

- **Point calibration** (2.2): least squares in implied volatility from 8 starts, and the
  maximum-likelihood version with $`c`$ estimated. A value with no range.
- **Refits to parts of the quotes** (2.4): computed in 9.6, on four subsets; refits that
  leave out one expiry at a time are a finer version.
- **Delta method** (2.4), with product gradients by central differences under common
  random numbers: next.

## 10. What the results show, and what they do not

### 10.1 Results for Heston on SPX, 9 November 2026

- **Cost.** One bank of $`N`$ = 300,000 simulated surfaces (16 minutes) and one training
  run (30 minutes) give a network that returns a posterior in under 0.3 s, against 17 to
  32 minutes by MCMC (7.1 to 7.3).
- **Calibration on average.** On $`M`$ = 512 simulated surfaces the network is calibrated
  for five of the six components of $`\theta`$; its intervals for $`v_0`$ are about 1.4
  times too wide (8.2).
- **Recovery.** The posterior medians of the five Heston parameters correlate with the
  truth at 0.996 or more, and that of $`c`$ at 0.95 (8.2).
- **Single surfaces.** On two simulated surfaces its means lie within one reference
  standard deviation of the exact posterior's (median 0.31), and its 90% intervals are
  1.07 to 2.9 times as wide; one of 12 parameter posteriors meets both agreement
  criteria (8.3).
- **The error scale.** The information about $`c`$ sits in the 33 low-variance directions
  of the inputs. Whitening is what lets the network read it; it then recovers $`c`$ with
  a root-mean-square error of 0.135 (6.3, 7.6, 8.2).
- **The design.** Calls pin $`\rho`$, expiries beyond 2 months pin the long-run level,
  and the deep puts matter for everything else (5.2).
- **The market.** Heston misses the SPX quotes by 1.38 vp root mean square, 13.4
  half-spreads, at every expiry, most at short maturities and in the deep put wing, and
  $`c`$ goes to its bound. Its skew stops at a ceiling of 1.88 as the expiry shrinks,
  while the SPX skew follows $`T^{-0.40}`$ (6.5). The market quotes are unlike any surface
  in the bank, so the network and the exact posterior disagree there, and market results
  use the exact posterior (8.3, 8.4).
- **Products.** Variance strikes of 17.73% (0 to 21 days) and 22.61% (81 to 325 days),
  with 90% ranges less than 0.2 vol points wide (9.2). For the forward-start call and the
  volatility swap, the posterior ranges contain the true prices on the market-like
  simulation; on SPX, refits to parts of the quotes spread 18 and 21 times wider than the
  posterior ranges (9.6).

### 10.2 What the ranges are not

- They are parameter uncertainty within one model, under a stated error model. Where the
  model misfits, as Heston does on SPX, they are too narrow: refits of the same model to
  different quotes move prices about 20 times more (9.6). The smooth model-error term of
  6.6 lets the posterior admit the misfit; it covers misfit on quoted points, not
  dynamics the quotes cannot reveal, such as the future smile, and only a different model
  removes the misfit. Refits and a second model show how far the model's own error moves
  a price; neither is a complete model-risk reserve.
- On real data the true value of a forward start or a barrier is never observed; the
  real-data checks are those of 8.4.
- Hedge ratios under the posterior are not shown.
- A posterior-based AVA is a proposal, not established practice.

**Known risks.**
1. *The smooth error can absorb parameter signal:* a flexible error and the parameters can
   trade off. The correlation lengths are fixed before training; wider posteriors are
   reported as they are.
2. *The error law is assumed:* Gaussian, with a squared-exponential correlation. $`\tau`$
   cannot tell structural model error from data problems.
3. *Rough Bergomi's simulated prices carry Monte Carlo error,* recorded per quote (4.2).
4. *Fourier accuracy at extreme parameters* (large $`\kappa`$ and $`\sigma_v`$, short deep
   puts): the gate of 4.1 applies everywhere and rejected parameter sets are recorded.
5. *Event risk inside the shortest expiry:* a scheduled announcement before the 7-day
   expiry, such as an inflation release, adds a jump that neither model contains; the
   error terms absorb it.
6. *Raw data:* each close is fetched that evening and kept outside Git, recorded by
   SHA-256.

### 10.3 What remains

1. **Rough Bergomi.** Its bank of 300,000 surfaces is being generated on the Oscar cluster
   (4.2). Then its validation and test surfaces, its network with the recipe of 7.5 and
   its own whitening, its checks (8.1 to 8.3, with importance sampling as the reference),
   and the two models side by side on the same quotes. It addresses what 6.5 finds
   missing: a short-expiry skew without a ceiling.
2. **The smooth model-error term** (6.6): exact posteriors first, then networks trained
   with $`\tau`$.
3. **More products and the book:** the down-and-in put with its estimator and tests (9.7),
   the book (9.10), the delta method (9.11), and the products under more draws and paths
   (9.9).
4. **The intervals for today's variance**, $`v_0`$, which are too wide (8.2).
5. **More checks of the model on the market:** held-out quotes, variance swaps from the
   full strips, the listed 2-year put (8.4, 9.8), and the repricing check of 9.3.
6. **A recorded render of the notebook with the pretrained estimator.** Its run record
   lists the estimator's cells as run so far only with stand-ins. The network numbers
   here are those of the recorded evaluation, drawn with the seed and number of draws the
   notebook uses.

### 10.4 Tutorial integration

The Heston notebook, `lab/notebooks/heston_spx_calibration.py`, follows the steps:

| Notebook section | Steps | What it shows |
| --- | --- | --- |
| The data | 1, 5 | the 63 SPXW quotes of 20:13 UTC on 9 November 2026, standardized moneyness, the selection rules |
| 1. Model | 4, 6 | Heston's dynamics and four features (4.1), the observation model with $`c`$ (6.1) |
| 2. Simulate | 3, 7 | the simulator, illustrative paths, the prior predictive check, whitening (7.6) |
| 3. Train | 7 | the two ways to use simulations (7.3), the network (7.4), its training record (7.7), and a small network trained live |
| 4. Infer | 7, 8 | the three data sets, the PyMC reference (8.1), the posteriors, the costs |
| 5. Check | 8 | the posterior predictive check, calibration, recovery, the comparison with PyMC, Heston against SPX |
| 6. Use | 9 | the forward-start call and the volatility swap under the posterior and under refits (9.6) |
| What Heston misses | 6, 10 | the four signs, where the fit misses, the missing feature, what can be done within Heston (6.5) |
| Takeaways | 10 | |

**Running it.** From `lab/`, with `HESTON_BUNDLE` set to the folder of the pretrained
estimator (a separate 465 MB archive, described in `lab/plans/training-data-guide.md`):

```bash
HESTON_BUNDLE=/path/to/bundle pixi run --locked -e core-reference marimo edit notebooks/heston_spx_calibration.py
```

Without the bundle, the notebook shows the PyMC reference alone, and the cells that need
the network stay off. `marimo export html` with the same setting renders a static copy with
every output. Two buttons run optional work: training a small network on 3,000 simulated
surfaces for 40 passes, in about a minute on a CPU, and resampling the 512 surfaces of
8.2 with fresh draws.

Rough Bergomi on SPX gets the same treatment once its bank and network exist. The
tutorial's own Heston notebook, `tutorial/notebooks/slot4_heston_bayesflow.py`, keeps its
four-parameter configuration on SPY. Results of the SPX notebook are not yet promoted to
`tutorial/assets/`, and enter there only through the lab's tools
(`lab/studies/accept.py`, then `pixi run manifest` and `pixi run claims`); before release,
the tutorial's `pixi run test`, `test-notebooks` and `check`, and the lab's `pixi run test`
and `release-check`.

## Back to the goals, the audience and the practitioner

**Goals.**
1. *Marks and ranges* come from steps 3 and 9: one posterior, every product priced under
   every draw. For Heston on SPX the forward-starting variance strikes, a forward-start
   call and a volatility swap are priced; the refits show that Heston's misfit, not the
   uncertainty of its parameters, sets how far a price can move. The down-and-in put and
   the book follow the methods of 9.7 and 9.10.
2. *Fast and testable:* steps 7 and 8. Pricing once, on simulated pairs that already
   contain every posterior, makes it possible; whitening the inputs makes the error scale
   readable; the checks show five of six components calibrated and close agreement with
   exact posteriors on simulated surfaces, while the market results use the exact
   posterior.
3. *The model and the limits:* steps 6, 8 and 10. The error model of step 6 is the choice
   that most affects the ranges. On SPX, Heston's misfit is about twenty times the quotes'
   half-spreads in vol points, its skew stops at a ceiling, and the ranges do not include
   the misfit until the model-error term does.

**Audience.** Researchers in AI and statistics see a real inverse problem in which three
things, not the network's size, decide whether the posterior can be used: the error model
(step 6), the conditioning of the network's inputs (step 7), and whether the model fits the
data (steps 6 and 8), with a chain of checks that separates approximation error from model
error (step 8). Practitioners see calibration ranges and refits from one consistent
calculation, mapped onto the prudent-valuation rules, with a plain statement of what the
ranges leave out. Both should leave able to say, for their own model, whether they need
this, what it costs, and how to check it.

**Practitioner.** A desk and its product control would get, every day:
- a mark and a range for every product, from one consistent calculation, with tested
  coverage;
- a measure of how well the model fits that day's quotes ($`c`$, $`\tau`$ and the map of
  misses), which flags the days when a model stops describing the market;
- the spread of prices across refits to different quotes and, with a second model,
  across models.

In the terms of the rules, the posterior's range is an Article 9 input, the spread across
refits and models belongs to Article 11, and the joint draws of a book would measure the
offset between products that the Annex fixes at 50%. The desk would still have to choose
the models, handle the model risk beyond one model's parameters and the other AVA
categories, such as close-out costs, decide how to hedge, take the method through model
validation, and have its regulator accept a posterior as the basis of a prudent value.

## Evidence and figures

**Heston study:** [PR #85](https://github.com/fmuia/sbi4finance/pull/85), commit
[`1c276bb`](https://github.com/fmuia/sbi4finance/tree/1c276bb72c1a8d43d1055327934f359708ba1e3f/lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid),
under `lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/`:

| What | File |
| --- | --- |
| The 63 quotes, forwards and discounts (steps 1, 5) | `cv-preparation/design.json` |
| The three data sets of 8.1 and their true parameters | `cv-preparation/conditional-panel.json`, `conditional-truth-audit.json` |
| Pricing replay over 98 points (4.1) | `cv-preparation/numerical-pilot/run.json` |
| Bank and training settings (7.3, 7.7) | `recovery-bank-independent/300k-300/protocol.json`, `campaign.json` |
| Population and conditional checks (8.2, 8.3) | `recovery-bank-independent/300k-300/evaluation/evaluation.json`, `recovery.npz` |
| Reference posteriors, first computation (8.1) | `recovery-bank-independent/references/{synthetic-market-like,synthetic-broad,market}/` |
| 63 against 54 quotes (5.3) | `recovery-bank-independent/reference-width-comparison.json` |
| Controlled whitening comparison (7.6) | `../noise-scale/capacity-study/whitening-pilot/` |

**Heston notebook and its run record:** branch `claude/awesome-fermat-544u9v`, commit
`cf34bfc` ([PR #93](https://github.com/fmuia/sbi4finance/pull/93), which builds on
[PR #89](https://github.com/fmuia/sbi4finance/pull/89) and PR #85). The notebook
`lab/notebooks/heston_spx_calibration.py` presents the configuration of PR #85 in six
stages (model, simulate, train, infer, check, use), between a description of the data and
an analysis of what Heston misses. Its run record,
`lab/evidence/pilot-results/2026-11-11/heston-actual63-notebook/`, holds the reference
posteriors recomputed with the same sampler and seed (`reference-comparison.json`,
`reference-display-draws.npz`, `references/<case>/`); the parameter table of 4.1, the
costs of 7.1 and 8.1, the comparisons of 8.3, the misfit of 6.5 and the product numbers of
step 9 use them. The trained network and its bank are distributed outside Git
(`lab/plans/training-data-guide.md` on that branch).

**The network recipe** (whitened $`y/h`$; flow matching with three 128-unit layers and
dropout 0.1) is that of Stefan's notebook,
`lab/notebooks/heston_6_parameter_bayesflow.ipynb` in
[PR #87](https://github.com/fmuia/sbi4finance/pull/87); the study scales it to 300,000 rows
and 300 passes.

**Rough Bergomi bank:** the plan and its status in
`lab/plans/BERGOMI-OBSERVATION-REFRESH-PLAN.md`
([PR #84](https://github.com/fmuia/sbi4finance/pull/84)); the production code, evidence
and repair packaging on branch `feat/bergomi-actual63-production-20261111`
([PR #88](https://github.com/fmuia/sbi4finance/pull/88)), under
`lab/evidence/pilot-results/2026-11-11/bergomi-observation-refresh/`
(`campaign-summary.json`, and the timing audit in `joint-legacy-budget-300k/`).

**Code.** In the tutorial runtime: the Heston characteristic function and pricer
(`tutorial/src/sbi4f_tutorial/heston_options/model.py`), the rough simulator
(`rough_options/simulator.py`) and its vanilla estimator (`shared/pricing.py`). On the
branch of PR #85: the Black control variate (`heston_options/control_variate.py`) and the
whitening transform (`shared/whitening.py`). In the lab: product pricing
(`lab/src/sbi4finance/calibration/products.py`).

**Figures and numbers.**
- [`figure-sources/extract_data.py`](figure-sources/extract_data.py) copies the numbers
  the figures need from the study's files into
  [`figure-sources/data.json`](figure-sources/data.json), with each source's path,
  commit and SHA-256.
- [`figure-sources/notebook_numbers.py`](figure-sources/notebook_numbers.py) repeats the
  notebook's computations from a checkout of its branch, without marimo or BayesFlow, and
  writes [`figure-sources/notebook.json`](figure-sources/notebook.json): every number the
  notebook prints that does not need the trained network, recomputed with its seeds and
  code, and the network's recorded results on the three data sets. The notebook's numbers
  quoted in the layers come from this file, with the commit and the SHA-256 of each source.
- [`figure-sources/make_figures.py`](figure-sources/make_figures.py) draws every figure in
  `images/` from these two files and the tutorial's Heston pricer. Three figures come from
  computations made in the script itself, each stated in its docstring: Figures 11 and 16
  from the notebook's 1,000 surfaces simulated from the prior, recomputed with the same
  seed and in the same order, and Figure 10 from a one-parameter toy model. The numbers
  quoted from these computations (the misfit, the information by region, the strip check,
  the precision of $`\tau`$, the spectrum of the inputs, the product summaries, the costs)
  are written to [`figure-sources/numbers.json`](figure-sources/numbers.json).

To recompute the notebook's numbers, run the first command from the `lab/` folder of a
checkout of the notebook's branch, with `DOCS` the path of this folder; then redraw from
the repository root:

```bash
PYTHONPATH=.:src:../tutorial/src python "$DOCS/figure-sources/notebook_numbers.py" --out "$DOCS/figure-sources/notebook.json" --commit "$(git rev-parse HEAD)"
```

```bash
PYTHONPATH=tutorial/src python lab/plans/option-application/figure-sources/make_figures.py
```

**Regulation and accounting:** Commission Delegated Regulation (EU) 2016/101, Articles 9,
10 and 11 and the Annex; CRR (Regulation (EU) No 575/2013), Article 34; IFRS 13. The
course's [practice sources](../../../course/practice-sources.md) record what practice is
documented.

# Option application on SPX · Layer 2: walkthrough

*Layer 2 of 3 · about 30 minutes · [Index](README.md) · [Layer 1](1-overview.md) ·
[Layer 3](3-full-detail.md) · [Glossary](glossary.md)*

*10 November 2026: the plan for the tutorial of 14 November
([PR #81](https://github.com/fmuia/sbi4finance/pull/81)). There are no results yet.
Numbers marked "first checks" come from quick runs of 9 November
([evidence](../../evidence/pilot-results/2026-11-09/design-and-error/README.md)); numbers
marked "noise session" come from exact fits and test networks of the same day, which are
not committed. [Layer 3](3-full-detail.md) gives every derivation and integrates the
results pushed on 10 November by Alex and Stefan; the main ones are summarized here.*

This layer takes each step of [Layer 1](1-overview.md) one level deeper: what we do, why,
the formulas that carry it, and the numbers behind each choice. The steps have the same
numbers in every layer.

## Goals

The three goals of Layer 1, made precise:

1. **A value and a range for products that no market prices.** For a forward-start call,
   two volatility swaps and a down-and-in put on the S&P 500, give on each of four days
   (9 to 12 November 2026) a mark, a 90% range, a prudent value and the range of a small
   book. All of it comes from one posterior per day and per model, under two models
   (Heston and rough Bergomi) that see the same data, the same quotes and the same error
   model.
2. **SBI makes this fast and testable.** Show that a network trained once returns each
   day's posterior in seconds; that it agrees with an exact posterior wherever one can be
   computed; and that its 90% ranges contain the truth about 90% of the time on hundreds
   to thousands of simulated days, a test that would cost thousands of CPU-hours with
   exact posteriors.
3. **The limits.** State what the range covers (one model's parameter uncertainty, under
   a stated error model) and what it does not (model risk beyond the parameters, hedging,
   regulatory acceptance).

A lesson runs through all three: a range is only as honest as the error model behind it.
With the wrong error model the ranges come out "precise but wrong" (step 6).

## Audience

The ICAIF '26 tutorial lasts two hours and speaks to two groups. AI and statistics
researchers know posteriors, MCMC and neural networks but not options; finance
practitioners know options and calibration but not SBI. Every step below is written so
that each group can follow the other's part: finance terms and statistical checks are
both explained where they first appear.

In the tutorial this material becomes two notebooks and part of a third:
- **Slot 4, Heston on SPX:** steps 1 to 9, with an exact posterior to check every claim
  against.
- **Slot 5, rough Bergomi on SPX:** the same steps for a model priced only by
  simulation, then the two models side by side.
- **Slot 6, priors and misspecification:** the comparison of error models (D4 in
  step 10).

This document is for us, and assumes no finance background.

## The practitioner's point of view

**The daily cycle.** At each close, a bank's trading desk values every position it
holds. This is **marking**, and the marks give the day's profit and loss (**P&L**) and the
balance sheet. A separate function, **product control**, independent of the traders,
checks the marks against market data and computes valuation adjustments. There are two
cases:
- **Mark to market.** A listed option has a quote, and its mark is the market price,
  usually the midpoint of bid and ask.
- **Mark to model.** A product agreed privately, **over the counter**, has no quote. Its
  mark is computed with a model **calibrated** to listed prices: the model's parameters
  are set so that it reproduces them.

**What the rules ask for.**
- *Accounting.* Positions are carried at **fair value**, the price an orderly sale would
  fetch (IFRS 13). A fair value that depends significantly on inputs nobody quotes is
  "Level 3", and the bank must disclose how much it would change under reasonably
  possible alternative inputs (IFRS 13, paragraph 93(h)).
- *Prudent valuation (EU).* Regulation 2016/101 asks for a **prudent value**: a point
  where the bank is "90 % confident" it could **exit** the position at that price or
  better (Article 9(5)(a), Article 11(3)). The shortfall of the prudent value from the
  fair value is an **additional valuation adjustment (AVA)**, deducted from Common Equity
  Tier 1, the bank's core capital (CRR, Article 34).
- Two of the AVA categories concern calibration. **Market price uncertainty**
  (Article 9) covers the uncertainty of valuation inputs; Article 11(1) sends there the
  "model risk which arises due to calibrations from market derived parameters". **Model
  risk** (Article 11) covers "a range of different models or model calibrations, which
  are used by market participants", to be measured where possible by "a range of
  plausible valuations produced from alternative appropriate modelling and calibration
  approaches".
- The individual AVAs are then aggregated with a factor $`\alpha = 50\%`$ (the
  Regulation's Annex). Under its Method 1, each exposure counts for
  $`(1-\alpha)(\mathrm{FV} - \mathrm{PV})`$, half its standalone size: a fixed allowance
  for the fact that many positions do not all reach their 90% points at once.

**What practitioners do today.** They run one point calibration per day. They build the
range from alternative calibrations and models, from consensus services that pool banks'
prices (some publish 10th and 90th percentiles for prudent valuation), or from expert
judgment, which the Regulation allows where data cannot give a range (Articles 9(5)(b)
and 11(4)). The hard cases are the inputs nobody quotes: the future shape of the smile,
the deep wings, the vol of vol. Those are exactly what the products below depend on.

**What we propose.** Make the range part of the calibration: a posterior over the
parameters, carried through every product. The spread that comes from calibrating to
quotes is an Article 9 input; the comparison of two models is an Article 11 question.
That a posterior is an acceptable basis for an AVA is our proposal, not established
practice, and the tutorial says so.

So, for every product, the practitioner needs a mark and a range that can be defended,
and for the book a range that accounts for how the products offset each other.

## 1. What has a price, and what does not

The problem starts with what the market prices directly.

**Options.** A **call** with strike $`K`$ and maturity $`T`$ pays $`(S_T-K)^+`$ at
$`T`$, and a **put** pays $`(K-S_T)^+`$, where $`x^+ = \max(x, 0)`$ and $`S_T`$ is the
index level at $`T`$. SPX options, on the S&P 500 index itself, are **European** (they
can be exercised only at maturity) and **cash-settled**. We use the **SPXW** series,
settled on the closing level, which expire on almost every trading day; the monthly SPX
series, settled on opening prices, supply the expiries beyond September 2027. Earlier
versions of the tutorial used SPY, a fund that tracks the index, whose options are
**American**: early exercise adds up to about 1.1 vol points to a 1-year at-the-money put
(noise session), which biases both the forward and the smile. SPX has no such term.

**Discount, forward and forward units.** A payment at $`T`$ is worth $`D(T)`$ today, the
**discount factor**, here taken from US Treasury yields between 3 months and 2 years. The
**forward** $`F(T)`$ is the price agreed today for the index delivered at $`T`$; with
deterministic interest rate $`r`$ and dividend yield $`q`$, $`F(T) = S_0\,e^{(r-q)T}`$. We
read it from the quotes by **put–call parity**, which holds exactly for European options:

```math
C(K,T) - P(K,T) = D(T)\,\big(F(T) - K\big),
```

taking the median over the six strike pairs nearest the money. Prices are then expressed
in **forward units** (divided by $`D(T)F(T)`$), and strikes by their **log-moneyness**
$`k = \ln(K/F(T))`$.

**Implied volatility and the smile.** The Black–Scholes formula prices an option as if
the index had a constant volatility $`\sigma`$. The $`\sigma`$ that reproduces a quote is
its **implied volatility**. Across strikes and maturities it forms the **smile**; for SPX,
mostly a downward **skew**. On 9 November, SPX's at-the-money implied volatility was
15.6–17.1% across maturities, and deep puts were far higher. We measure strike distances
in standard deviations at a fixed 16% volatility,

```math
\delta = \frac{k}{0.16\,\sqrt T},
```

so $`\delta = -3`$ is a strike three standard deviations below the forward. A quote's
**vega** $`\nu`$, the sensitivity of its price to its implied volatility (in forward
units $`\nu = \sqrt T\,\varphi(d_1)`$, with $`\varphi`$ the standard normal density),
converts between price and vol points.

**Bid, ask and half-spread.** We keep only **out-of-the-money** quotes, the liquid ones:
puts with $`k \le 0`$ and calls with $`k \gt 0`$. A quote is dropped if it has no bid, if
its ask is not above its bid, if its spread exceeds 50% of its mid, if its mid violates
the no-arbitrage bounds, or if its implied volatility jumps away from its neighbours'.
The **half-spread**, converted to vol points at the quote's own implied volatility, says
how precisely the market prices it. On SPX, 9 November: 0.02–0.05 vol points at the
money, 0.05–0.17 down to $`\delta \approx -6`$, 0.11–0.25 between $`-6`$ and $`-8`$, and
0.10–0.37 for calls at $`+2`$ to $`+3`$ (first checks).

**The products.** Banks sell over-the-counter products on the index, often inside
**structured notes** sold to investors. Three are marked; two more quantities serve as
checks.

| Product | Payoff | What it depends on |
| --- | --- | --- |
| Forward-start call, 6 months to 1 year | $`(S_{T_2} - m\,S_{T_1})^+`$ at $`T_2 = 1`$ year, with $`T_1 = 6`$ months and $`m = F(T_2)/F(T_1)`$ | The at-the-money level and skew between 6 months and 1 year, and above all the **forward smile**: the smile that will prevail in 6 months, which no listed option shows |
| Volatility swaps, 3 months and 1 year | Realized volatility over the period minus a strike agreed today | The variance level across the whole strip of strikes (weighted towards puts), and the vol of vol, from the wings and the short maturities |
| Down-and-in put, 1 year, strike 100%, barrier 70% | $`(K-S_T)^+`$ if some daily close is at or below $`0.7\,S_0`$, else zero | Deep puts down to 70% out to 1 year, and how volatility rises when the index falls |
| Variance swap (check) | Realized variance minus a strike agreed today | The whole strip: its strike can be computed from listed options without a model |
| Listed 2-year 90% put (check) | A listed put beyond the design's last maturity | Extrapolation in maturity |

**Realized variance** over a period is the annualized sum of squared daily log returns,
and **realized volatility** is its square root. The down-and-in put is the risk inside
**autocallable** notes, among the most common structured products: their buyers in effect
sell the bank this put.

**Why the three need a model.** Listed options at one maturity fix the distribution of
$`S_T`$ at that maturity: the second derivative of the call price in the strike is the
discounted density of $`S_T`$ (Breeden and Litzenberger),

```math
\frac{\partial^2 C}{\partial K^2}(K,T) = D(T)\,p_T(K).
```

They say nothing about how the distributions at different dates are joined. A forward
start depends on the joint law of $`S_{T_1}`$ and $`S_{T_2}`$, a volatility swap on the
whole path of the variance, a barrier on the path's minimum. Two models that agree on
every listed price can disagree on all three. The variance swap is the exception that
proves the rule: for a continuous path, realized variance can be replicated by a fixed
position in options plus daily trading in the index, so its strike follows from listed
prices (the construction behind the VIX index). That makes it a check: a model fitted to
some quotes must reproduce it.

So the three products are marked to model, and the next question is how the model is
set.

## 2. Calibration gives one value and hides the rest

A product without a price needs a model, and a model needs its parameters $`\theta`$. The
model values a payoff $`\Pi`$ as its expected discounted payoff,

```math
P_\Pi(\theta) = D(T)\,\mathbb{E}_\theta[\Pi],
```

where the expectation is under the model's **pricing probabilities** (also called the
risk-neutral measure). Under them the index in forward units, $`X_t = S_t/F(t)`$, has no
drift, so that no trading strategy earns a riskless profit.

**The classical answer.** Calibrate by least squares on implied volatilities,

```math
\hat\theta = \arg\min_\theta \sum_{i=1}^{n} \big(\sigma_i^{\text{market}} - \sigma_i^{\text{model}}(\theta)\big)^2 ,
```

from several starting points (eight in the plan), and mark the product at
$`P_\Pi(\hat\theta)`$. A version with a likelihood (step 6) also estimates the size of the
model error, by maximum likelihood.

**What it hides.** The quotes pin some directions in parameter space tightly and leave
others loose, and which ones depends on which quotes there are. In Heston on SPX,
removing the maturities beyond two months raises the posterior standard deviation of the
long-run variance by 303% and that of the mean-reversion speed by 232% (first checks).
Some loose directions barely move the listed prices but move a product a lot. The forward
start is the clearest case: listed prices constrain the forward smile only through the
model's dynamics. A single best fit reports none of this.

**Two classical ways to put a range on it,** both used as baselines in step 9:
- the **delta method**: linearize the prices around $`\hat\theta`$, take the covariance
  of $`\hat\theta`$ from the curvature of the fit, and carry it to the product through
  the product's gradient. It is exact for linear problems and can mislead where the fit
  is curved or runs into the edge of the parameter range;
- **refits on subsets**: recalibrate with one maturity left out at a time, and take the
  spread of the resulting product values.

The posterior of step 3 needs no linearization, and it states its assumptions.

## 3. The posterior turns one value into a distribution

To see how far the value can move, we replace the single best fit by a distribution.
Bayes' rule gives the **posterior**, the distribution of the parameters given the quotes
$`y`$:

```math
p(\theta \mid y) \;\propto\; p(y \mid \theta)\; p(\theta).
```

The **prior** $`p(\theta)`$ says which parameters are plausible before seeing data (step
4). The **likelihood** $`p(y \mid \theta)`$ says how probable the quotes are under
$`\theta`$; it comes from the error model (step 6).

**From parameters to products.** Each product's value is a function of the parameters,
so the posterior over $`\theta`$ induces a distribution over the value. With draws
$`\theta^{(1)}, \dots, \theta^{(S)}`$ from the posterior ($`S = 1{,}000`$):

```math
P_j^{(s)} = P_j\big(\theta^{(s)}\big),\qquad s = 1, \dots, S .
```

From these values:
- the **mark** is the mean, $`\bar P_j = \frac1S \sum_s P_j^{(s)}`$; the median and the
  point-calibration price are reported beside it;
- the **prudent value** is on the side that hurts the position: the 10% quantile
  $`q_{0.10}`$ for a product the bank owns, the 90% quantile $`q_{0.90}`$ for a product it
  owes;
- the **reserve** is the gap between the two, $`|\bar P_j - \mathrm{PV}_j|`$.

**The book.** A desk holds positions $`n_j`$ in several products. Each posterior draw is
one complete, consistent set of parameters, so it prices every product at once, and the
book's value per draw is

```math
B^{(s)} = \sum_j n_j\,P_j^{(s)} .
```

The book's prudent value is the 10% quantile of $`B^{(s)}`$. Comparing the book's reserve
with the sum of the products' standalone reserves measures how much the products offset
each other under the posterior. The Regulation replaces this offset with the fixed factor
$`\alpha = 50\%`$; the joint draws measure it. The example book is long one down-and-in
put, as a bank that issues autocallables is (the notes' buyers in effect sell it the
put), long one forward start and short one 1-year volatility swap.

**The three ingredients.** The posterior needs a model with its prior (step 4), the
quotes (step 5), and an error model that turns the two into a likelihood (step 6).
Together they define how quotes are generated:

```math
\theta \sim p(\theta),\qquad y = P(\theta) + \text{error},
```

with $`P(\theta)`$ the model's prices at the chosen quotes. This is a simulator, and
step 7 learns to invert it.

## 4. Two models

The first ingredient is the model.

**Heston.** In forward units, the index and its variance follow

```math
\frac{dX_t}{X_t} = \sqrt{V_t}\,dB_t,\qquad dV_t = \kappa\,(\bar{v} - V_t)\,dt + \xi\sqrt{V_t}\,dW_t,\qquad dB_t\,dW_t = \rho\,dt ,
```

where $`B`$ and $`W`$ are Brownian motions with correlation $`\rho`$. The parameters:

| Parameter | Meaning | Prior (uniform) |
| --- | --- | --- |
| $`v_0`$ | today's variance | 0.01 to 0.16 (volatility 10% to 40%) |
| $`\bar{v}`$ | long-run variance | 0.01 to 0.16 |
| $`\kappa`$ | speed of **mean reversion** towards $`\bar{v}`$ | 0.1 to 10 |
| $`\xi`$ | **vol of vol**: how much the variance fluctuates | 0.1 to 3 |
| $`\rho`$ | correlation of index and variance shocks (negative: volatility rises when the index falls) | −0.95 to −0.1 |

These are the priors of Alex's pilot. They contain the 9 November least-squares fit to
SPX ($`\kappa = 7.3`$, $`\xi = 2.5`$, $`\rho = -0.72`$), which the tutorial's earlier box
($`\kappa`$ fixed at 2, $`\xi \le 1.5`$) excluded. The **Feller condition**
$`2\kappa\bar{v} \ge \xi^2`$, which keeps the variance away from zero, is not imposed. The
SPX fit violates it nine times over ($`2\kappa\bar{v} = 0.73`$ against
$`\xi^2 = 6.4`$), and that is allowed: the model is still well defined, and its prices are
still exact.

*Pricing.* The characteristic function of $`\ln X_T`$ is known in closed form, and one
Fourier integral turns it into an option price. The integral is computed on a ladder of
finer and finer grids until two consecutive grids agree within 0.01 of the quote's
half-spread. Pricing all the quotes takes 20–50 milliseconds.

*Expected variance.* The variance reverts on average to $`\bar{v}`$,

```math
\mathbb{E}[V_t] = \bar{v} + (v_0 - \bar{v})\,e^{-\kappa t},\qquad
K_{\text{var}}(T) = \frac1T\int_0^T \mathbb{E}[V_t]\,dt = \bar{v} + (v_0 - \bar{v})\,\frac{1 - e^{-\kappa T}}{\kappa T},
```

where $`K_{\text{var}}(T)`$ is the fair strike of a variance swap. At short maturities
Heston's at-the-money skew levels off to a constant, while the market's keeps steepening.

**Rough Bergomi.** The variance is the exponential of a rough Gaussian process:

```math
V_t = \xi_0(t)\,\exp\!\Big(\eta\,Y_t - \tfrac12\,\eta^2\,t^{2H}\Big),\qquad Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW_s ,
```

and the index follows $`dX_t/X_t = \sqrt{V_t}\,dB_t`$ with $`dB_t\,dW_t = \rho\,dt`$.
Since $`\mathrm{Var}(Y_t) = t^{2H}`$, the correction term makes $`\mathbb{E}[V_t] = \xi_0(t)`$:
the curve $`\xi_0`$ is the expected future variance.
- The **Hurst exponent** $`H`$ sets the roughness. The kernel $`(t-s)^{H-1/2}`$ blows up
  as $`s \to t`$ when $`H \lt 1/2`$, so the latest shocks weigh most and paths are much
  rougher than Brownian motion; estimates from realized volatility put $`H`$ near 0.1. As
  a result the at-the-money skew grows like a power of maturity, $`T^{H-1/2}`$, as maturity
  shrinks: the shape index smiles show. Whether this makes rough Bergomi the better
  model for SPX is debated (Abi Jaber and Li, 2024, argue it fails on the smile's global
  shape); step 10 (D7) compares the two models on our data.
- $`\eta`$ is the vol of vol, and $`\rho`$ the correlation.
- The curve has the same shape as Heston's expected variance:

  ```math
  \xi_0(t) = \xi_s\,e^{-\lambda t} + \xi_l\,\big(1 - e^{-\lambda t}\big),
  ```

  with a short-term level $`\xi_s`$, a long-term level $`\xi_l`$ and a speed
  $`\lambda`$. So both models learn the variance level from the same quotes in the same
  way, and the variance-swap check (step 8) treats them equally. In the earlier tutorial
  rough Bergomi was handed a curve replicated from strikes Heston never saw, which made
  the comparison unequal.
- Priors, uniform: $`H`$ from 0.02 to 0.35, $`\eta`$ from 0.5 to 3.5, $`\rho`$ from −0.99
  to −0.2, $`\xi_s`$ and $`\xi_l`$ from 0.01 to 0.16, $`\lambda`$ from 0.25 to 8.

*Pricing.* There is no formula, only Monte Carlo simulation. The hybrid scheme of
Bennedsen, Lunde and Pakkanen simulates $`Y`$ exactly near the singular point of the
kernel and by a fast convolution elsewhere. Given a path of $`W`$, $`\ln X_T`$ is Gaussian,
so each path contributes a Black–Scholes price (conditional Monte Carlo), and paired
("antithetic") paths and control variates reduce the noise further. Paths are added, from
32,768 up to about 2 million, until each price's standard error is at most a quarter of
the smallest error the prior allows (about 0.05 vol points at the money). This costs
25–50 CPU-seconds per set of quotes, and the remaining Monte Carlo covariance enters the
likelihood.

**Same information for both.** Both models see the same quotes, the same error model and
the same curve family for the expected variance, and both infer the model-error size
$`\tau`$ (step 6).

## 5. Which quotes

The second ingredient is the data the posterior conditions on.

**Data.** SPX closing quotes of 9, 10, 11 and 12 November, from Cboe's delayed-quote
feed, fetched each evening because Cboe serves only the current snapshot. The raw files
stay outside Git, identified by their SHA-256 hashes.

**The design.** A fixed set of $`n`$ actual SPXW quotes, each frozen with its strike,
expiry, forward, discount factor, half-spread and vega. It is fixed because one network can serve all days only if every
observation has the same shape. Using actual quotes, rather than points interpolated on a
grid, avoids reading errors on the design's own day.

**Why heavy in puts: what each region pins.** First checks measured, for Heston on SPX
with smooth model error, how much the posterior standard deviation of each quantity rises
when one region of quotes is removed:

| Region removed | Rise in posterior standard deviation |
| --- | --- |
| Deep puts, below $`\delta = -3`$ | $`\kappa`$ +59%, $`\xi`$ +51%, $`\bar{v}`$ +33%, 1-year variance swap +27% |
| At the money, within 0.5 of it | $`v_0`$ +21%, $`\xi`$ +25% |
| Calls, above $`\delta = +0.5`$ | $`\rho`$ +72% |
| Maturities up to 1 month | $`\xi`$ +108%, $`v_0`$ +79%, $`\kappa`$ +63% |
| Maturities from 2 months | $`\bar{v}`$ +303%, variance swap +287%, $`\kappa`$ +232% |

**Coverage criteria.** The design must contain, with $`\delta`$ in units of the 16%
volatility:
1. puts below $`\delta = -3`$, at short and at long maturities;
2. a point at the money, and calls near $`+0.5`$, $`+1`$ and $`+2`$ (without the
   near-the-money calls the uncertainty of $`\rho`$ grows 2.4 to 4.1 times);
3. maturities from 1 week to 1 year, with at least two up to 1 month and two from
   2 months.

Coverage matters more than density: under smooth model error, doubling the number of
points within the current range narrows the posteriors by 2–17%, while widening the
range narrows $`\kappa`$, $`\bar{v}`$, $`\xi`$ and the variance swap by 23–49% (first
checks).

**The pilot's 54 quotes, checked.** Alex's pilot used 54 quotes on six expiries from 21
to 325 days, nine per expiry from $`\delta \approx -4`$ to $`+1`$, with half-spreads of
0.02–0.10 vol points. They meet the put criterion but miss the 1-week maturity, the second
maturity under a month, and the $`+2`$ calls. The design is therefore rebuilt as actual
quotes at the targets of "layout E" (7 maturities from 1 week to 1 year, times 9 values of
$`\delta`$ from −5 to +2): 62 of the 63 targets are quoted, there is no 1-year expiry
settled on the close, and the last call is revised to $`+1.5`$ (Alex's audit). The final
contract list is not frozen yet.

**The other days.** The plan reads 10 to 12 November at the design's places in
$`(k, T)`$: a smoothing spline of implied volatility against $`\delta`$ for each expiry,
linear interpolation of the total implied variance $`\sigma^2 T`$ between the two
expiries around $`T`$, and no extrapolation beyond the quoted strikes. Reading adds about
0.01 vol points (the spline) to 0.05–0.25 (interpolation between expiries); the error is
smooth across strikes, so the model-error term absorbs it. The vegas stay frozen (they
change by a few percent a day), and the day's half-spreads enter through the multiplier
$`c`$ of step 6. Alex's baseline instead uses actual quotes only and treats another day
as another design; the choice is open (Layer 3, step 5.4).

**Data kept out of the fit,** for the checks of step 8:
- the **model-free variance swap**: for each expiry whose strikes reach far enough into
  both wings, the total variance

  ```math
  w(T) = 2\int \mathrm{OTM}(k)\,e^{-k}\,dk ,
  ```

  over the out-of-the-money prices in forward units, read at 1, 3 and 12 months as a
  volatility $`\sqrt{w(T)/T}`$ (the VIX construction);
- the **listed 2-year 90% put**: the December 2028 expiry, at the listed strike nearest
  $`0.9\,F(T)`$;
- **held-out quotes**: every cleaned quote that is not in the design.

## 6. How quotes differ from the model

The third ingredient links the first two: it is the likelihood.

**Observation equation.** For the $`n`$ design quotes, in forward-unit prices,

```math
y_i = P_i(\theta) + \nu_i\,\big(c\,h_i\,Z_i + \tau\,U_i\big),\qquad Z \sim \mathcal N(0, I),\quad U \sim \mathcal N(0, K_\ell),\quad Z,\ U\ \text{independent}.
```

- $`P_i(\theta)`$ is the model's price, and $`\nu_i`$ the quote's market vega, frozen.
- $`h_i`$ is the quote's half-spread in vol points, frozen with the design, and $`c`$ a
  known multiplier: 1 on the design's own day; on the other days, the median ratio of
  that day's half-spreads to the frozen ones (step 5); drawn between 0.5 and 2 in
  training, so that the network handles tighter and wider quotes. Reading a half-spread
  as one standard deviation is a convention.
- $`\tau`$ is the size of the model error, in vol points, inferred with $`\theta`$.
- $`K_\ell`$ is the correlation between quotes, squared-exponential in their coordinates:

  ```math
  \big(K_\ell\big)_{ij} = \exp\!\Big(-\frac{(\delta_i-\delta_j)^2}{2\ell_\delta^2} - \frac{(\ln T_i-\ln T_j)^2}{2\ell_T^2}\Big).
  ```

  Nearby quotes have correlated model errors; the lengths $`\ell_\delta`$ and $`\ell_T`$
  say how near is near.

Divided by $`\nu_i`$, the equation says that in vol points, to first order,
$`\sigma_i^{\text{market}} - \sigma_i^{\text{model}}(\theta) \approx c\,h_i Z_i + \tau\,U_i`$:
Layer 1's equation.

**Likelihood.** The errors are Gaussian with covariance

```math
\Sigma(\tau, c) = \mathrm{diag}(\nu)\,\big(c^2\,\mathrm{diag}(h^2) + \tau^2 K_\ell\big)\,\mathrm{diag}(\nu)\ \ \big(+\ \Sigma_{\text{MC}}\ \text{for rough Bergomi}\big),
```

so, with residuals $`r = y - P(\theta)`$,

```math
\log p(y \mid \theta, \tau) = -\tfrac12\, r^\top \Sigma^{-1} r - \tfrac12 \log\det\Sigma - \tfrac n2 \log 2\pi .
```

The log-determinant term penalizes a large $`\tau`$; without it, $`\tau`$ would grow until
it explained everything.

**Why this form: the error budget.**

| Source | Size, in vol points |
| --- | --- |
| Bid–ask on SPX | 0.02–0.37, depending on the region (step 1) |
| Reading the chain on the other days | 0.01–0.25 |
| Heston's model error on SPX | 1.0 root mean square, 17 times the median half-spread; correlation 0.80 between strikes half a standard deviation apart, 0.85 between adjacent maturities (first checks) |
| Heston's model error on SPY | 0.8–1.2 root mean square, about 99% of the residual variance (noise session) |

The model error dominates, and it is smooth.

**What it replaces.** The tutorial's current rule adds independent noise in price equal to
the Black–Scholes vega at a fixed 20% volatility times 0.3 vol points at the money,
rising to 1.0 in the wings. That is about 8 times SPY's bid–ask. In vol points it equals
the stated values only where a quote's own volatility is 20%: SPY's far calls get 5–7 vol
points, its puts at $`\delta = -2`$ get 0.4–0.9. It is neither quote noise nor a stated
model error, but an unestimated, independent allowance that alone sets every posterior
width.

**Why smooth matters.** Treated as independent, $`n`$ correlated residuals count as
$`n`$ independent pieces of evidence, and posteriors come out too narrow. In the noise
session, networks trained on the current rule or on independent errors covered the truth
22–72% and 18–58% of the time, at a nominal 90%, when the real misfit was smooth and of
the measured size. It also changes a headline of the current tutorial: Heston's variance
swaps sit 0.6–1.7 vol points below SPY's replicated values, a gap that looks significant
only under independent errors. Under smooth model error the bands widen to 1.8–3.0 vol
points and contain the market from 1 month to 1 year (noise session, tentative).

**Settings.**
- $`\tau`$: log-uniform from 0.2 to 5 vol points, for both models. The noise session
  estimated the smooth amplitude at 2.1–2.6 vol points, above the pilot's cap of 2; below
  0.2, model error is comparable to processing error, and the floor also bounds rough
  Bergomi's Monte Carlo cost.
- $`\ell_\delta`$ and $`\ell_T`$: fixed per model before training, at the values that
  maximize the likelihood of the first two days (maximized over $`\theta`$ and $`\tau`$),
  on the grid $`\ell_\delta \in \{0.75, 1, 1.5, 2\}`$, $`\ell_T \in \{1, 1.5, 2\}`$.
- An extra independent model-error term is added only if the checks of step 8 (V2) ask
  for it.

**Where the implementation stands.** Alex's pilot used the same price form with an
independent model error ($`K_\ell = I`$) and $`\tau`$ uniform from 0 to 2 vol points; its
version is kept for comparison (D4). His current baseline (PRs #82 and #85) keeps only the
quote term ($`\tau = 0`$), with $`c`$ fixed at 1 as the primary teaching target and $`c`$
inferred as an extension. On the market chain its exact posterior puts $`c`$ at its upper
bound of 2: the misfit shows through. The two tracks support a two-stage reading: the
quote term alone to validate inference and teach what the quotes identify if the model
were right, then the smooth model error for the product ranges on market data. Which one
the market application uses is the main open decision (Layer 3, step 6.8).

## 7. Computing the posterior: simulation-based inference

With the three ingredients the posterior is defined; computing it is the next problem.

**Why not compute every posterior exactly.**

| | Heston | Rough Bergomi |
| --- | --- | --- |
| One likelihood evaluation | 20–50 ms (Fourier) | 25–50 CPU-s (Monte Carlo) |
| One exact posterior | 15–40 CPU-min (MCMC) | 7–14 CPU-h (importance sampling) |
| Simulated cases in the coverage study (D3) | 1,300 | 300 |

For rough Bergomi the coverage study alone would take 2,100–4,200 CPU-hours with exact
posteriors, against about 12 with the network. One posterior is affordable without SBI;
many posteriors are not, and validation, daily use and many names all need many.

**The amortization identity.** Draw $`\theta`$ from the prior and $`y`$ from the
simulator. The pair is a draw from the joint distribution, which factors both ways:

```math
p(\theta, y) = p(\theta)\,p(y \mid \theta) = p(y)\,p(\theta \mid y).
```

Read from the right, $`\theta`$ is a draw from the posterior given its own $`y`$. A
network $`q(\theta \mid y)`$ trained on many such pairs can therefore learn the posterior
for every $`y`$ at once. In its simplest form it maximizes the average of
$`\log q(\theta \mid y)`$ over the simulated pairs, whose optimum is the posterior itself;
the tutorial's BayesFlow configuration uses flow matching, a different loss with the same
target.

**Simulation banks.** The expensive part, the model prices at the design's quotes, does not
depend on the errors. It is computed once and stored as a **bank**: 100,000 draws from
the prior with their clean prices.
- Heston's bank exists for the pilot's 54 quotes; the rebuilt design needs repricing,
  about 1.3 CPU-hours.
- Rough Bergomi's is new: 90,000 draws for training, 5,000 for validation and 5,000 for
  testing, an estimated 660–1,300 CPU-hours at the accuracy target of step 4, measured
  first on a pilot of 200.
- Failed draws are recorded and rejected, never silently redrawn.

**Training.** The errors are drawn fresh each time a bank entry is used: $`\ln\tau`$
uniform, $`c`$ uniform on 0.5 to 2, $`Z`$ and $`U`$ normal, and
$`y = P_{\text{bank}} + \nu\,(c\,h\,Z + \tau\,U)`$. So the network sees each set of
prices under many error draws, and the error model can change without a new bank. The
network reads the quotes divided by their half-spreads, together with the log of the
measurement scale, and returns the joint posterior of the model parameters and
$`\ln\tau`$, through a transform that maps each prior range onto the real line. The
configuration and the seed are chosen on simulated data only, never on the real days.

**Whitened inputs: the finding of 10 November.** A network given these inputs, merely
standardized one by one, ignored the error scale entirely: inferring $`c`$, Alex's
network returned its prior (truth–estimate correlation about 0), although MCMC on the same
data recovers $`c`$ well. The reason is geometric. All quotes move together with the
parameters, by far more than the noise, so the residual that carries the error scale is
a tiny combination of nearly identical inputs (input condition number about 2 million),
which training cannot resolve. Stefan's notebook showed the fix and Alex's controlled
pilot isolated it: a fixed full **whitening** of the inputs (a rotation and rescaling,
fitted on training simulations, that makes them uncorrelated with unit variance) raised
the correlation to 0.92. The whitened network then overfits when each training row keeps
one noise draw, giving intervals that are too narrow; drawing fresh errors at every pass,
as this plan does, is the remedy being tested. Inferring $`\tau`$ is the same kind of
problem, so this plan's networks whiten their inputs too.

**Cost after training.** A Heston training run takes about 100 seconds per 30,000
updates on one GPU, and 10–20 runs per model are planned. After that, a day's posterior
takes seconds.

## 8. Checking it

A network's posterior is an approximation. It is used only as far as it passes checks,
and each check answers one question.

**Reference posteriors,** computed directly, to check the network against:
- *Heston, exact:* MCMC on the Fourier likelihood, with PyMC's DEMetropolisZ sampler (it
  needs no gradients), 4 chains of 2,000 tuning steps and 2,000 retained draws, at most
  45 CPU-minutes per case. Accepted when the chains agree ($`\hat R \le 1.01`$) and give
  at least 400 effective draws for every parameter and reported product.
- *Rough Bergomi, by importance sampling:* draw 1,000 parameter sets from the network,
  weight each by prior times likelihood over the network's density, and use a fixed-seed
  Monte Carlo likelihood, so that the weights are a deterministic function of
  $`\theta`$. Accepted when the effective sample size is at least 200 and the Pareto
  diagnostic $`\hat k`$ is below 0.7; otherwise the network is reported as not validated
  for that case.
- *Cases:* every real day, plus 8 simulated cases spread over the prior and near SPX.

**The checks.**

| Check | Question | Passes when |
| --- | --- | --- |
| V1, prior predictive | Can the model, its prior and its error model produce each real day? | Every quote lies inside the 1–99% band of 10,000 simulations from the prior |
| V2, error-model fit | Do exact fits leave residuals like those the error model describes? | Whitened residuals have lag-1 correlation below 0.3 along strikes, and mean square within $`1 \pm 2\sqrt{2/n}`$ |
| V3, calibration | On 2,000 simulated cases, are the network's posteriors calibrated? | The rank of the truth among posterior draws is uniform (SBC), and 90% intervals cover 87–93% |
| V4, accuracy | Does the network match the reference on each case? | Means within 0.3 reference standard deviations; 90% widths within a factor 0.75–1.25; $`\tau`$'s median within 20%; product quantiles within 10% of the reference range |
| V5, held-out quotes | Do the fits predict quotes they did not use? | Coverage of 90% predictive intervals between 80% and 97% by region; reported, not tuned |
| V6, market | Do the fits match two market prices they never used? | Reported per day: the model-free variance swap at 1, 3 and 12 months against the band of $`\sqrt{K_{\text{var}}}`$, and the listed 2-year put against its predictive band; not tuned |

Three terms in the table:
- **Whitened residuals** are the residuals transformed by the inverse of the Cholesky
  factor of $`\Sigma`$. If the error model is right, they are independent standard
  normals, so their correlation and mean square test it.
- **SBC** (simulation-based calibration): over simulated cases, the rank of the true
  value among the posterior draws is uniform if and only if the posteriors are calibrated
  on average.
- In V5 the model error at an unused quote is predicted from the design's residuals
  through the correlation $`K_\ell`$ (kriging), and the quote's own half-spread is added.

**V4b: why did the pilot's network fail?** Alex's 75,000-update network fails V4 on the
market day: its $`\tau`$ has posterior mean 0.14 vol points, against the exact
posterior's 0.61 (its continuation to 88,000 updates hit a non-finite loss, cause
unresolved). Two causes are possible. The leading one is now input conditioning (step 7):
its inputs were not whitened, and the $`\tau`$ networks failed even on their own
simulator, where the error law is right by construction. The other is the error law:
trained on independent errors, the network meets smooth market residuals. The test
compares network and reference, with and without whitening, on (1) simulated cases with
independent errors, (2) simulated cases with smooth errors of the same size, and (3) the
market day, and measures the correlation of neighbouring whitened residuals on the market
day. Failure already on (1) without whitening points to conditioning; agreement on (1)
with failure on (2) and (3) points to the error law.

**Status.** No Heston network passes V4 yet. The fixed-$`c`$ teaching network has
intervals 1.13–1.41 times the reference widths on a broad simulated case; the whitened
network that infers $`c`$ learns it but undercovers (55–61% at a nominal 90% for the
physical parameters). Pooled checks can pass while single posteriors are wrong, which is
why V4 is binding.

**If a check fails.** If V3 or V4 fails, the real-day results come from the reference
posteriors, and the network is shown on simulated data with its scores. If V5 or V6
fails, that is a finding about the model, reported as such.

## 9. From the posterior to the products

With a checked posterior, the products can be priced.

**Principle.** For each of the $`S = 1{,}000`$ posterior draws (the network's draws, or the
reference's draws thinned or resampled), simulate index paths under that draw's
parameters and average the discounted payoff. $`\tau`$ does not enter: it describes how
listed options are observed, not how the index moves. The error model acts on the
products only through the posterior of $`\theta`$.

**Common random numbers.** All draws share one set of random normals, so the estimated
values differ across draws only through $`\theta`$. Monte Carlo noise shifts all draws
together and is measured separately. Each product's standard error must be at most 2% of
its 90% range, and so must the errors of its quantiles (from a bootstrap over draws and a
rerun with a new seed).

**One idea for all products.** Given the path of the Brownian motion $`W`$ that drives the
variance, the log-index is Gaussian:

```math
\ln X_t = \rho\,J_t + \sqrt{1-\rho^2}\,J^{\perp}_t - \tfrac12\,I_t,\qquad I_t = \int_0^t V_s\,ds,\quad J_t = \int_0^t \sqrt{V_s}\,dW_s,\quad J^{\perp}_t \mid W \sim \mathcal N(0, I_t).
```

So a simulated variance path gives the integrated variance $`I`$ and the integral $`J`$,
and the rest of the randomness can be integrated in closed form or sampled exactly. This
holds for both models.

**Each product.**
- *Variance swap:* $`K_{\text{var}}(T) = \mathbb{E}[I_T]/T`$, in closed form for both
  models (step 4), reported as a volatility $`\sqrt{K_{\text{var}}}`$.
- *Volatility swap:* $`K_{\text{vol}}(T) = \mathbb{E}\big[\sqrt{I_T/T}\big]`$, the path
  average with a control variate. As a sanity check,
  $`K_{\text{vol}} \approx \sqrt{K_{\text{var}}} - \mathrm{Var}(I_T/T)/(8K_{\text{var}}^{3/2})`$:
  the gap between the two swaps grows with the vol of vol.
- *Forward start:* given $`W`$, $`X_{T_1}`$ and $`X_{T_2}/X_{T_1}`$ are independent, so the
  value is an average, over variance paths, of a forward factor times a Black–Scholes
  call on the variance between $`T_1`$ and $`T_2`$ (formula in Layer 3). Heston's forward
  skew flattens through mean reversion, while rough Bergomi's persists: this is where the
  two models are expected to disagree most.
- *Down-and-in put:* the index at the 252 daily closes is built exactly from the stored
  increments of $`I`$ and $`J`$ plus fresh normals, and the barrier is checked on the
  closes. The vanilla put serves as a control variate, since a down-and-in put plus the
  matching down-and-out put equals the vanilla on every path. Unit tests cover the
  Black–Scholes limit (with the Broadie–Glasserman–Kou correction for daily monitoring),
  the response to the monitoring frequency, and determinism.
- *2-year put:* Heston by Fourier; rough Bergomi by conditional Monte Carlo on 2-year
  paths, with the curve extended beyond 1 year (stated as an assumption).

**Consistency between fit and products.** The products use path simulation: for Heston a
full-truncation Euler scheme with 4 steps per business day, for rough Bergomi the same
hybrid scheme as the bank. The check: the simulation reprices the design's options within
0.05 vol points of the Fourier prices at the posterior median, on each day.

**Baselines.**
- *Point calibration:* least squares in implied volatility (8 starts) and its
  maximum-likelihood version. A value with no range.
- *Delta method* at $`\hat\theta`$, with $`J_P = \partial P/\partial\theta`$ the Jacobian
  of the design's prices and $`\hat\Sigma`$ the error covariance at the fit:

  ```math
  \widehat{\mathrm{Cov}}(\hat\theta) = \big(J_P^\top \hat\Sigma^{-1} J_P\big)^{-1},\qquad \mathrm{sd}_j = \sqrt{\nabla P_j^\top\,\widehat{\mathrm{Cov}}(\hat\theta)\,\nabla P_j},\qquad \text{range } P_j(\hat\theta) \pm 1.645\,\mathrm{sd}_j .
  ```

- *Expiry-drop refits:* refit with each of the design's expiries removed in turn, and
  take the minimum to maximum of the product values.

## 10. What the results will show, and what they will not

Finally, what the outputs will establish, and where they stop.

| | Computation | What it shows |
| --- | --- | --- |
| D1 | Per day, model and product: posterior 5%, 50% and 95%, mark, point price, delta-method and refit ranges, reserve | One posterior prices every product; ranges differ by how well the quotes pin each product: listed options reprice tightly, the forward start is wide, the 2-year put is set against its listed price |
| D2 | Book reserve against the sum of standalone reserves; correlations of product values across draws | How ranges aggregate across a book |
| D3 | Simulation study: 1,300 Heston cases, 300 rough Bergomi cases, each with its true product prices | Coverage of the 90% product ranges (binomial ±2 standard errors: ±1.9% on 1,000 cases, ±3.5% on 300), against the delta method, the refits and the point calibration, which never covers |
| D4 | One day, Heston reference, under four error models: half-spreads only; plus independent $`\tau`$ (the pilot's); plus smooth $`\tau`$ (adopted); smooth with $`c`$ = 0.5 and 2 | That "precise but wrong" comes from the error model; variance-swap bands against the model-free value |
| D5 | Posterior of $`\tau`$ per day and model; map of the mean residual, in vol points | How wrong each model is, and where |
| D6 | 9 to 12 November: marks, ranges, daily point calibrations | How marks move from day to day against the width of their ranges (four days: illustrative) |
| D7 | Heston against rough Bergomi on the same data: $`\tau`$, held-out fit, market checks, product ranges, and a weighted combination (stacking weights fitted on held-out quotes) | A model comparison, stated as a combination, not a reserve |

**What the ranges are not.**
- They are uncertainty about one model's parameters, under the stated error model. The
  model-error term covers misfit on the quoted points, not dynamics the quotes cannot
  reveal, such as the forward smile. D7 shows how far the two models disagree; neither
  range is a complete model-risk reserve.
- On real data the true value of a forward start or a barrier put is never observed. The
  real-data checks are V5 and V6.
- Hedge ratios under the posterior are not shown.
- A posterior-based AVA is a proposal.

**Known risks.**
1. A flexible model error can absorb part of the parameter signal. The correlation
   lengths are fixed from the first two days, wider posteriors are reported as the honest
   result, and V2 and V4 detect problems.
2. The error law is assumed: Gaussian, with a squared-exponential correlation. $`\tau`$
   cannot tell structural model error from data or convention problems.
3. The networks may still miss the references, as the pilot's did, and as the 8 November
   Heston training matrix did. The real-data results then come from the references.
4. Rough Bergomi's bank carries Monte Carlo error inside its simulated quotes; at the
   accuracy target this makes posteriors slightly conservative, not biased.
5. Fourier pricing is hardest at extreme $`\kappa`$ and $`\xi`$ for short-dated deep
   puts; the grid checks apply, and rejected draws are recorded.
6. A US inflation release falls inside the shortest maturity; the smooth term absorbs
   it, and the text says so.
7. Four days illustrate day-to-day behaviour; they are not a history.
8. The raw quotes must be fetched each evening.
9. A network must read error scales from whitened inputs. The whitener belongs to one
   ordered design and is refitted for every new design or model, and a whitened network
   overfits fixed training noise unless the noise is redrawn.

## Back to the goals, the audience and the practitioner

**Goals.**
1. *Marks and ranges* come from steps 3 and 9 (D1, D2): one posterior per day and model,
   every product priced under every draw, and a range for the book.
2. *Fast and testable:* steps 7 and 8 (V3, V4, D3). Posteriors take seconds, are checked
   against exact references, and their coverage is tested on 1,300 and 300 simulated
   cases. If the network fails, the references carry the real-data results.
3. *Limits:* step 10, with D4 and D7. The ranges are within-model and depend on the
   error model; they are not a reserve for model risk.

The lesson that the error model sets the width is shown directly by D4.

**Audience.** Researchers in AI and statistics see a real inverse problem in which the
error model, not the network, decides whether the posterior is right, and a chain of
checks (V1 to V6) that separates network error from model error. Practitioners see
calibration ranges and book aggregation from one consistent calculation, with tested
coverage and a plain statement of what it leaves out. Both should leave able to say, for
their own model, whether they need this (do their products depend on what the quotes
leave open?), what it costs (a bank and a training once, then seconds a day, with exact
references where affordable), and how to check it.

**Practitioner.** A desk and its product control would get, every day:
- a mark and a range for every product and for the book, from one consistent
  calculation, with tested coverage;
- a measure of how badly each model misfits ($`\tau`$ and the residual map), which flags
  the days when a model stops describing the market;
- a comparison of two models on the same information.

They would still have to choose the models, handle model risk beyond one model's
parameters (Article 11) and the other AVA categories, such as close-out costs, decide how
to hedge, put the method through model validation, and have their regulator accept it.
In the terms of the rules, the posterior's range is an input to Article 9, the two-model
comparison belongs to Article 11, and the joint draws measure the offset between products
that the Regulation's aggregation sets at a fixed 50%.

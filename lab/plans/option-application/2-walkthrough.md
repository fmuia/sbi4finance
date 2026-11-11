# Option application on SPX · Layer 2: walkthrough

*Layer 2 of 3 · about 30 minutes · [Index](README.md) · [Layer 1](1-overview.md) ·
[Layer 3](3-full-detail.md) · [Glossary](glossary.md) · Results as of 11 November 2026*

Each step of [Layer 1](1-overview.md), one level deeper: what we do, why, the formulas
that carry it, and the numbers behind each choice. The steps have the same numbers in
every layer.

## Goals

1. **A value and a range for products that no market prices.** For products on the
   S&P 500 that are not listed (a forward-starting variance swap, a forward-start call, a
   volatility swap and a down-and-in put), give a mark, a 90% range and a prudent value,
   and the range of a small book. All of it comes from one posterior per model, under two
   models (Heston and rough Bergomi) that see the same 63 quotes and the same error model.
2. **SBI makes this fast and testable.** A network trained once returns a posterior in
   under a second. It is tested on hundreds of simulated surfaces with known parameters,
   and against exact posteriors wherever they can be computed.
3. **The limits.** State what the range covers (one model's parameter uncertainty, under
   a stated error model) and what it does not (the model's own error, hedging, regulatory
   acceptance).

Two things decide whether such a posterior is right, and both run through the steps
below: the error model, which says how far quotes may sit from the model (step 6), and
the way the network reads its inputs (step 7).

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
- **Slot 6, priors and misspecification:** how the error scale and the model's misfit
  change the posterior (steps 6 and 10).

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
the deep wings, the vol of vol. Those are what the products below depend on.

**What we propose.** Make the range part of the calibration: a posterior over the
parameters, carried through every product. The spread that comes from calibrating to
quotes is an Article 9 input; the comparison of two models is an Article 11 question.
That a posterior is an acceptable basis for an AVA is our proposal, not established
practice.

So, for every product, the practitioner needs a mark and a range that can be defended,
and for the book a range that accounts for how the products offset each other.

## 1. What has a price, and what does not

The problem starts with what the market prices directly.

**Options.** A **call** with strike $`K`$ and maturity $`T`$ pays $`(S_T-K)^+`$ at
$`T`$, and a **put** pays $`(K-S_T)^+`$, where $`x^+ = \max(x, 0)`$ and $`S_T`$ is the
index level at $`T`$. SPX options, on the S&P 500 index itself, are **European** (they
can be exercised only at maturity) and **cash-settled**. We use the **SPXW** series,
settled on the index's closing level, which expire on almost every trading day.

**Discount, forward and forward units.** A payment at $`T`$ is worth $`D(T)`$ today, the
**discount factor**. The **forward** $`F(T)`$ is the price agreed today for the index
delivered at $`T`$; with interest rate $`r`$ and dividend yield $`q`$,
$`F(T) = S_0\,e^{(r-q)T}`$. Both are read from the quotes by **put–call parity**, which
holds exactly for European options:

```math
C(K,T) - P(K,T) = D(T)\,\big(F(T) - K\big).
```

For each expiry, a straight-line fit of $`C - P`$ against $`K`$, over pairs of
near-the-money calls and puts kept aside for this purpose, gives $`D(T)`$ from its slope
and $`F(T)`$ from its intercept. Prices are then expressed in **forward units** (divided by $`D(T)F(T)`$).

**Implied volatility and the smile.** The Black–Scholes formula prices an option as if
the index had a constant volatility $`\sigma`$. The $`\sigma`$ that reproduces a quote is
its **implied volatility**. Across strikes and maturities it forms the **smile**; for SPX,
mostly a downward **skew** (Figure 3, step 5). A quote's **vega** $`\nu`$, the sensitivity
of its price to its implied volatility, converts between price and **vol points** (0.01
of volatility): a half-spread $`h`$ in price is $`h/\nu`$ in volatility.

**The products.** Banks sell over-the-counter products on the index, often inside
**structured notes** sold to investors.

| Product | Payoff | What it depends on |
| --- | --- | --- |
| Forward-starting variance swap, window $`[t_1, t_2]`$ | Realized variance over the window minus a strike agreed today | The expected variance over the window: the term structure of the smile |
| Forward-start call, 6 months to 1 year | $`(S_{T_2} - m\,S_{T_1})^+`$ at $`T_2 = 1`$ year, with $`T_1 = 6`$ months and $`m = F(T_2)/F(T_1)`$ | The **forward smile**: the smile that will prevail in 6 months, which no listed option shows |
| Volatility swap, 1 year | Realized volatility minus a strike agreed today | The variance level, and how much it fluctuates (the vol of vol) |
| Down-and-in put, 1 year, strike 100%, barrier 70% | $`(K-S_T)^+`$ if some daily close is at or below $`0.7\,S_0`$, else zero | Deep puts out to 1 year, and how volatility rises when the index falls |

**Realized variance** over a period is the annualized sum of squared daily log returns,
and **realized volatility** is its square root. The down-and-in put is the risk inside
**autocallable** notes, among the most common structured products: their buyers in effect
sell the bank this put.

![Simulated index paths showing the payoffs of the four products](images/products.png)

**Figure 1.** Simulated paths of the index under Heston, at its parameters fitted to SPX
(step 6). Top left: each forward-start call fixes its strike at the dot, 6 months in, and
pays at 1 year. Top right: running realized volatility, path by path, against one fixed
strike. Bottom left: 12% of paths touch the 70% barrier; on those, the put exists. Bottom
right: the expected variance over time and its average over two windows, the strikes of
two forward-starting variance swaps.

**Why they need a model.** Listed options at one maturity fix the distribution of
$`S_T`$ at that maturity: the second derivative of the call price in the strike is the
discounted density of $`S_T`$ (Breeden and Litzenberger),

```math
\frac{\partial^2 C}{\partial K^2}(K,T) = D(T)\,p_T(K).
```

They say nothing about how the distributions at different dates are joined. A forward
start depends on the joint law of $`S_{T_1}`$ and $`S_{T_2}`$, a volatility swap on the
whole path of the variance, a barrier on the path's minimum. Two models that agree on
every listed price can disagree on all of them. The forward-starting variance swap is a
partial exception: for a continuous path, its strike follows from the strips of listed
options at the two window ends (the construction behind the VIX index), so a model fitted
to some quotes must reproduce it there.

## 2. Calibration gives one value and hides the rest

A product without a price needs a model, and a model needs its parameters $`\theta`$. The
model values a payoff $`\Pi`$ as its expected discounted payoff,

```math
P_\Pi(\theta) = D(T)\,\mathbb{E}_\theta[\Pi],
```

where the expectation is under the model's **pricing probabilities** (also called the
risk-neutral measure). Under them the index in forward units, $`X_t = S_t/F(t)`$, has no
drift, so that no trading strategy earns a riskless profit.

**The classical answer.** Calibrate by least squares, for example on implied
volatilities,

```math
\hat\theta = \arg\min_\theta \sum_{i=1}^{n} \big(\sigma_i^{\text{market}} - \sigma_i^{\text{model}}(\theta)\big)^2 ,
```

from several starting points, and mark the product at $`P_\Pi(\hat\theta)`$.

**What it hides.** The quotes pin some directions in parameter space tightly and leave
others loose, and which ones depends on which quotes there are (step 5). Some loose
directions barely move the listed prices but move a product a lot. The forward start is
the clearest case: listed prices constrain the forward smile only through the model's
dynamics. A single best fit reports none of this.

**Two classical ways to put a range on it:**
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
- the **mark** is the mean, $`\bar P_j = \frac1S \sum_s P_j^{(s)}`$;
- the **prudent value** is on the side that hurts the position: the 10% quantile
  $`q_{0.10}`$ for a product the bank owns, the 90% quantile $`q_{0.90}`$ for a product it
  owes;
- the **reserve** is the gap between the two, $`|\bar P_j - \mathrm{PV}_j|`$.

![Schematic: posterior draws pushed through a pricer give a distribution of values](images/posterior-to-product.png)

**Figure 2.** Schematic of the push-forward. Left: posterior draws, each one complete set of
parameters; the parameters are correlated. Right: the product value under every draw.
The mean is the mark, the 10% quantile the prudent value of a long position.

**The book.** A desk holds positions $`n_j`$ in several products. Each posterior draw is
one complete, consistent set of parameters, so it prices every product at once, and the
book's value per draw is

```math
B^{(s)} = \sum_j n_j\,P_j^{(s)} .
```

The book's prudent value is the 10% quantile of $`B^{(s)}`$. Comparing the book's reserve
with the sum of the products' standalone reserves measures how much the products offset
each other under the posterior; the Regulation replaces this offset with the fixed
factor $`\alpha = 50\%`$. The example book is long one down-and-in put, as a bank that
issues autocallables is, long one forward start and short one 1-year volatility swap.

**The three ingredients.** The posterior needs a model with its prior (step 4), the
quotes (step 5), and an error model that turns the two into a likelihood (step 6).
Together they define how quotes are generated:

```math
\theta \sim p(\theta),\qquad y = P(\theta) + \text{error},
```

with $`P(\theta)`$ the model's prices at the 63 quotes. This is a simulator, and step 7
learns to invert it.

## 4. Two models

The first ingredient is the model.

**Heston.** In forward units, the index and its variance follow

```math
\frac{dX_t}{X_t} = \sqrt{V_t}\,dB_t,\qquad dV_t = \kappa\,(\bar{v} - V_t)\,dt + \xi\sqrt{V_t}\,dW_t,\qquad dB_t\,dW_t = \rho\,dt ,
```

where $`B`$ and $`W`$ are Brownian motions with correlation $`\rho`$:

| Parameter | Meaning | Prior (uniform) | SPX, 9 November (exact posterior mean) |
| --- | --- | --- | --- |
| $`v_0`$ | today's variance | 0.01 to 0.16 (volatility 10% to 40%) | 0.0281 (volatility 16.8%) |
| $`\bar{v}`$ | long-run variance | 0.01 to 0.16 | 0.0535 (volatility 23.1%) |
| $`\kappa`$ | speed of **mean reversion** towards $`\bar{v}`$ | 0.1 to 10 | 5.07 |
| $`\xi`$ | **vol of vol**: how much the variance fluctuates | 0.1 to 3 | 1.80 |
| $`\rho`$ | correlation of index and variance shocks | −0.95 to −0.1 | −0.70 |

The SPX values violate the **Feller condition** $`2\kappa\bar{v} \ge \xi^2`$, which keeps
the variance away from zero ($`2\kappa\bar{v} = 0.54`$ against $`\xi^2 = 3.26`$). That is
allowed: the model is still well defined, and its prices are still exact.

*Pricing.* The characteristic function of $`\ln X_T`$ is known in closed form, and one
Fourier integral turns it into an option price. The integral is written relative to a
Black–Scholes price with the same expected variance, which keeps it accurate for deep,
nearly worthless options. It is computed on a ladder of finer grids until two
consecutive grids agree within 0.01 of the quote's half-spread.

*Expected variance.* The variance reverts on average to $`\bar{v}`$:

```math
\mathbb{E}[V_t] = \bar{v} + (v_0 - \bar{v})\,e^{-\kappa t}.
```

Averaged over a window, this is the strike of a forward-starting variance swap (step 9).

**Rough Bergomi.** The variance is the exponential of a rough Gaussian process:

```math
V_t = \xi_0(t)\,\exp\!\Big(\eta\,Y_t - \tfrac12\,\eta^2\,t^{2H}\Big),\qquad Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW_s ,
```

and the index follows $`dX_t/X_t = \sqrt{V_t}\,dB_t`$ with $`dB_t\,dW_t = \rho\,dt`$.
Since $`\mathrm{Var}(Y_t) = t^{2H}`$, the correction term makes $`\mathbb{E}[V_t] = \xi_0(t)`$:
the curve $`\xi_0`$ is the expected future variance.

The **Hurst exponent** $`H`$ sets the roughness. The kernel $`(t-s)^{H-1/2}`$ blows up as
$`s \to t`$ when $`H \lt 1/2`$, so the latest shocks weigh most and paths are much
rougher than Brownian motion; estimates from realized volatility put $`H`$ near 0.1. The
at-the-money skew then grows like $`T^{H-1/2}`$ as maturity shrinks, which is the shape
of index smiles at short maturities, where Heston falls short (Figure 5, step 6).

The curve has the same shape as Heston's expected variance:

```math
\xi_0(t) = \xi_l + (\xi_s - \xi_l)\,e^{-\lambda t},
```

with a short-term level $`\xi_s`$, a long-term level $`\xi_l`$ and a speed $`\lambda`$.
So both models describe the variance level in the same way.

**Priors,** uniform: $`H`$ from 0.02 to 0.35, $`\eta`$ from 0.5 to 3.5, $`\rho`$ from −0.99
to −0.2, $`\xi_s`$ and $`\xi_l`$ from 0.01 to 0.16, $`\lambda`$ from 0.25 to 8 per year.

*Pricing.* There is no formula, only Monte Carlo simulation, on a time grid built for
each expiry. Given a path of $`W`$, $`\ln X_T`$ is Gaussian, so each path contributes a
Black–Scholes price; paired ("antithetic") paths and control variates reduce the noise
further. Paths are added, up to about 2 million, until the Monte Carlo standard error
falls below a fixed target (about 0.0001 in forward-unit price). Pricing the 63 quotes
once costs 1–2 CPU-minutes.

## 5. Which quotes

The second ingredient is the data the posterior conditions on.

**Data.** The Cboe chain of SPX options on 9 November 2026, read at 15:13 New York time.
The raw file stays outside Git, identified by its SHA-256 hash.

**Coordinates.** Strikes are placed by **standardized moneyness**,

```math
z = \frac{\ln(K/F)}{0.16\,\sqrt{T}} ,
```

the distance of the strike from the forward in standard deviations of the index at a 16%
volatility, close to SPX's at-the-money level (15–17% on that day). It is a coordinate
for placing strikes, not the option's delta. Time $`T`$ is exact calendar time to the
expiry's 16:00 close, in years of 365 days.

**The design.** For each of seven expiries chosen near 1 and 2 weeks and 1, 2, 3, 6 and
12 months, take the actual quote nearest to each of nine targets: $`z`$ = −5, −3.5,
−2.25, −1.25, −0.5 and 0 (puts), +0.5, +1 and +2 (calls). Quotes must be distinct, have
a positive bid below the ask and a half-spread at most 25% of the mid, and stay apart from
the parity pairs used for the forwards.

| Expiry | Days | Quotes | Note |
| --- | ---: | ---: | --- |
| 16 Nov 2026 | 7 | 9 | |
| 23 Nov 2026 | 14 | 9 | |
| 9 Dec 2026 | 30 | 9 | |
| 8 Jan 2027 | 60 | 9 | |
| 29 Jan 2027 | 81 | 9 | |
| 30 Apr 2027 | 172 | 9 | |
| 30 Sep 2027 | 325 | 9 | latest expiry settled on the close; last call at $`z = +1.5`$, the furthest available |

![The 63 quotes: implied volatility by expiry, and their places in moneyness and maturity](images/smile-and-design.png)

**Figure 3.** Left: the smile at the 63 quotes, one line per expiry. Deep puts trade at up to
42% volatility, calls near 13%; short expiries are steeper. Right: the design in
$`(z, T)`$, with each quote's half-spread in vol points as colour. Half-spreads have a
median of 0.064 vol points and range from 0.025 to 0.28; the widest are far calls at
short maturities.

Using actual quotes, rather than points interpolated on a grid, adds no reading error.
Because the design is fixed, one trained network serves this snapshot. Every new design
needs its own simulations and its own network.

**What each region pins.** A local calculation shows how much each region of quotes
contributes. At Heston's SPX posterior, with errors of one half-spread, remove one region
and see how much the posterior standard deviation of each quantity rises:

![Rise in posterior standard deviation when one region of quotes is removed](images/information-by-region.png)

**Figure 4.** Rise in the posterior standard deviation of each parameter, and of two
forward-starting variances (step 9), when one region of the 63 quotes is removed. Local
calculation from the curvature of the likelihood at Heston's SPX posterior mean.

- The calls pin $`\rho`$: without them its uncertainty grows 8.9 times.
- The expiries beyond 2 months pin the long-run level: without them $`\bar{v}`$ and the
  later-window variance become 16 and 18 times less certain, and $`\kappa`$ 9.5 times.
- The expiries up to a month pin today's variance and the short-window variance.
- The deep puts matter for every parameter except $`\rho`$, by 56% to 87%.
- The at-the-money point adds little once its neighbours are there.

So a design must reach deep puts and far calls, at short and long maturities. Covering
these regions matters more than adding points inside them.

## 6. How quotes differ from the model

The third ingredient links the first two: it is the likelihood.

**Observation equation.** For the $`n = 63`$ quotes, in forward-unit prices,

```math
y_i = P_i(\theta) + c\,h_i\,Z_i,\qquad Z_i \sim \mathcal N(0, 1)\ \text{independent},
```

where $`P_i(\theta)`$ is the model's price, $`h_i`$ the quote's half-spread and $`c`$ a
multiplier common to all quotes. The true price lies somewhere between bid and ask, so
the half-spread is the natural scale of a quote's uncertainty; reading it as one standard
deviation is a convention. The multiplier is inferred with the model's parameters, with a
uniform prior from 0.5 to 2: the data say how large the errors are.

**Likelihood.** With $`R(\theta) = \sum_i \big((y_i - P_i(\theta))/h_i\big)^2`$, the sum of
squared misses in half-spreads,

```math
\log p(y \mid \theta, c) = -\frac{R(\theta)}{2c^2} - n\ln c - \sum_i \ln h_i - \frac n2 \ln 2\pi .
```

The term $`-n\ln c`$ matters: it stops $`c`$ from growing to explain everything. At fixed
$`\theta`$ the best $`c`$ is $`\sqrt{R(\theta)/n}`$.

**How precisely the multiplier can be known.** The parameters explain about 5 of the 63 directions
in which the quotes can move; the other 58 carry only error. So $`c`$ is estimated from
about 58 independent pieces of evidence, to a relative precision of about
$`1/\sqrt{2 \times 58} \approx 9\%`$: a 90% interval about 0.3 times $`c`$ wide. The data
determine $`c`$ well, provided the estimator can read them (step 7).

**The fixed-scale case.** With $`c`$ known and equal to 1, the posterior is simply
"Heston with quote errors of one half-spread". Applying $`c`$ = 0.5 or 2 to the same
quotes halves or doubles the widths of the physical parameters' posteriors. This makes the
role of the error model visible in one line.

**What the SPX quotes say.** On the market, the exact posterior puts $`c`$ at its upper
bound: 1.999 out of a maximum of 2. Figure 5 shows why.

![Market quotes against Heston's best fit, and the misses by expiry](images/heston-misfit.png)

**Figure 5.** Left: market quotes (dots) and Heston at its exact posterior mean (lines), for
three expiries. At 7 days the model cannot produce the steep skew of deep puts. Right:
market minus model in vol points, for all seven expiries. The red band, ± one median
half-spread, is barely visible.

- Heston misses the 63 quotes by 1.38 vol points root mean square, and by up to 6.0 at
  the deepest 7-day put. The median half-spread is 0.064 vol points.
- Measured in each quote's own half-spread, the misses are 13 half-spreads root mean
  square; the prior allows at most 2.
- The misses are smooth: they follow the same pattern across neighbouring strikes
  (correlation 0.66 between neighbours) and similar patterns across expiries.

**What it means.** On SPX the model, not the quotes, is the main source of error. The
posterior is then the uncertainty of Heston's parameters given that Heston is the model,
with an error scale capped at twice the half-spread: it does not include the model's own
error, and its ranges are narrower than the misfit warrants. Widening the prior on $`c`$
would not fix this, because independent errors would count the 63 smooth misses as 63
independent pieces of evidence, when they are a few smooth bumps. The natural extension is
a smooth **model-error** term,

```math
y_i = P_i(\theta) + c\,h_i\,Z_i + \tau\,\nu_i\,U_i ,
```

with $`U`$ a smooth random curve across strikes and maturities, $`\nu_i`$ the quote's vega
and $`\tau`$ the model error's size in vol points, inferred with the rest.
[Layer 3](3-full-detail.md) gives its form and shows that the 63 quotes determine
$`\tau`$ to about 10–15%.

## 7. Computing the posterior: simulation-based inference

With the three ingredients the posterior is defined; computing it is the next problem.

![CPU time for one posterior by each method](images/posterior-costs.png)

**Figure 6.** CPU time for one posterior. Heston's exact posterior by MCMC took 7 to 15
minutes on each of three cases. For rough Bergomi, importance sampling needs about 1,000
pricings of the 63 quotes at 1–2 minutes each. The trained network returns a posterior in
under a second.

Testing that ranges are right needs posteriors on hundreds of simulated surfaces with
known parameters. For rough Bergomi, 300 of them would take 5,000 to 9,000 CPU-hours by
importance sampling. One posterior is affordable without SBI; many are not, and
validation, daily use and many underlyings all need many.

**The amortization identity.** Draw $`\theta`$ from the prior and $`y`$ from the
simulator. The pair is a draw from the joint distribution, which factors both ways:

```math
p(\theta, y) = p(\theta)\,p(y \mid \theta) = p(y)\,p(\theta \mid y).
```

Read from the right, $`\theta`$ is a draw from the posterior given its own $`y`$. A
network $`q(\theta \mid y)`$ trained on many such pairs can therefore learn the posterior
for every $`y`$ at once. We use **flow matching** (BayesFlow 2.1): the network learns to
transport simple noise into posterior draws, and its training target is the posterior
itself.

**Simulation banks.** The expensive part, the model prices at the 63 quotes, is computed
once and stored as a **bank** of parameter sets drawn from the prior, with their clean
prices. Each bank row gets one draw of $`c`$ and of the errors $`Z`$.
- Heston: 300,000 rows, priced with the checked Fourier integral.
- Rough Bergomi: 300,000 rows, being generated on the cluster (47 to 59 hours on 192
  CPUs), then 5,000 for validation and 7,000 for testing.

**Training.** The network reads the quotes divided by their half-spreads, $`y_i/h_i`$,
whitened (below), and returns the joint posterior of the model parameters and $`c`$;
$`c`$ is never an input. Heston's network has three hidden layers of 128 units, dropout
0.1 and standardized targets, and is trained with AdamW at batch size 256 for 300 passes
over the bank (351,600 updates): the learning rate warms up over the first 5% of updates,
then decays to zero. Rough Bergomi uses the same recipe.

**Whitened inputs.** All 63 quotes move together with the parameters, by far more than
the errors, and the information about $`c`$ sits in the small differences between them.
Scaling each input on its own leaves those differences buried: in the controlled
comparison below, the scaled inputs' covariance had a condition number of about
2 million, and training could not resolve its smallest directions. **Whitening** fixes this: a fixed rotation and rescaling, fitted on
training simulations, that makes the inputs uncorrelated with unit variance. It loses no
information and does not change the posterior, only how easily the network reads it.

![Whitening: two quotes before and after, and the inferred c with and without it](images/whitening.png)

**Figure 7.** First two panels: two quotes across the prior lie almost on a line, because
both follow the parameters; whitening turns them into a round cloud in which the error
directions are as visible as the rest. Last two panels: a controlled comparison, same
data and same network, differing only in the input scaling. Scaled one input at a time,
the network returns the prior for $`c`$ (correlation with the truth 0.00); whitened, it
recovers $`c`$ (0.92).

Rules: fit the whitening on noisy training simulations only (the first 10,000 rows);
keep all 63 coordinates; apply exactly the same transform at training and at use; fit a
new one for every new design or model.

## 8. Checking it

A network's posterior is an approximation. It is used only as far as it passes checks,
and each check answers one question.

**Reference posteriors,** computed directly, to check the network against.
- *Heston, exact:* MCMC on the Fourier likelihood with PyMC's DEMetropolisZ sampler (it
  needs no gradients): 4 chains of 6,000 tuning steps and 12,000 draws, started from fits
  to the data alone. Accepted when the chains agree ($`\hat R \le 1.01`$) and give at least 400
  effective draws for every parameter. Three cases: a simulated surface from a broad
  region of the prior, a simulated surface near SPX, and the SPX quotes. All three pass,
  in 7 to 15 CPU-minutes each.
- *Rough Bergomi, by importance sampling:* draw 1,000 parameter sets from the network,
  weight each by prior times likelihood over the network's density, and accept when the
  effective sample size is at least 200 and the Pareto diagnostic $`\hat k`$ is below 0.7.

**Checks on simulated surfaces.** 512 surfaces simulated from the prior, each with its
known parameters and 512 posterior draws from the network.

![Recovery: true parameter against posterior median on 512 simulated surfaces](images/heston-recovery.png)

**Figure 8.** True value against posterior median for each parameter, with 90% intervals on
60 surfaces. The five Heston parameters are recovered almost exactly; $`c`$, which is read
from the errors alone, has wider intervals.

![Coverage and simulation-based calibration of the Heston network](images/heston-calibration.png)

**Figure 9.** Left: how often central intervals contain the truth, against their nominal
level; a calibrated posterior lies on the diagonal, within the grey band. Right: the rank
of the truth among posterior draws (simulation-based calibration); a calibrated posterior
stays inside the grey band.

| | $`v_0`$ | $`\kappa`$ | $`\bar{v}`$ | $`\xi`$ | $`\rho`$ | $`c`$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 90% coverage | 97.5% | 90.0% | 92.4% | 89.3% | 92.0% | 90.2% |
| Rank test | outside the band | passes | passes | passes | passes | passes |
| Median 90% width, as a share of the prior's | 0.4% | 1.7% | 0.7% | 1.1% | 1.0% | 32% |

Five of six parameters are calibrated. Today's variance $`v_0`$ has intervals about 1.4
times as wide as they should be: the truth falls too often near the centre. About 3% of draws
fall slightly outside the prior's box; they are kept, not clipped.

**Checks against the references.**

![Network against reference posterior on two simulated surfaces and the SPX quotes](images/heston-vs-reference.png)

**Figure 10.** For each case and parameter, the 90% interval of the reference (grey) and of
the network (colour), centred on the reference median and measured in reference widths.
On the two simulated surfaces the network brackets the reference with wider intervals. On
the SPX quotes it is far away (note the wider scale).

- On the two simulated surfaces, the network's means lie within one reference standard
  deviation of the reference's (within 0.6 for most parameters), and its intervals are 1.05
  to 2.9 times as wide, widest for $`v_0`$ and $`\rho`$.
- On the SPX quotes the network is far from the reference: 21% of its draws put $`c`$
  above 2, and its physical means sit 2 to 27 reference standard deviations away. The
  market lies outside what the network saw in training, since its misfit is larger than
  any error the prior allows (step 6). Market results therefore use the reference
  posterior.

**Population and conditional checks answer different questions.** Coverage and ranks
average over many simulated surfaces; the references check single surfaces. Both are
needed: a network can be calibrated on average and still be wrong on a given day.

**Checks of the model on the market.** Figure 5 is the first: the misses against the
half-spreads. Two more follow: predicting quotes left out of the fit, and comparing the
model's variance-swap strikes with those computed from the full strip of listed options.

## 9. From the posterior to the products

With a checked posterior, the products can be priced.

**Principle.** For each posterior draw, price each product. All draws share the same
random numbers (**common random numbers**), so the spread across draws reflects the
parameters, not simulation noise. $`c`$ does not enter: it describes how quotes are
observed, not how the index moves.

**Forward-starting variance.** The strike of a variance swap on the window
$`[t_1, t_2]`$ is the expected variance averaged over the window. Under Heston,

```math
K_{\text{var}}(t_1, t_2) = \bar{v} + (v_0 - \bar{v})\,e^{-\kappa t_1}\,\frac{1 - e^{-\kappa (t_2 - t_1)}}{\kappa\,(t_2 - t_1)} ,
```

and under rough Bergomi it is the average of $`\xi_0`$ over the window. Two windows are
reported: a short one, from today to 21 days, and a later one, from 81 to 325 days (two of
the design's expiries). At Heston's SPX posterior mean they correspond to volatilities of
17.7% and 22.6% (Figure 1, bottom right). On the simulated surfaces the network's values
for both lie within 0.9 reference standard deviations of the reference's, with ranges 1.8
to 2.7 times as wide.

**Path-dependent products.** For each draw, simulate index paths and average the
discounted payoff:
- the *forward-start call*, from a conditional Black–Scholes formula on each variance
  path;
- the *volatility swap*, from the average of $`\sqrt{I_T/T}`$, with $`I_T`$ the integrated
  variance;
- the *down-and-in put*, from the index at the 252 daily closes, with the vanilla put as
  a control variate.

**Summaries and the book.** Each product gets a mark, a 5–95% range, a prudent value and a
reserve, and the example book gets its own range and reserve, set against the sum of its
products' reserves and against the Regulation's fixed 50%.

**Baselines,** for comparison: the best fit alone, which has no range; a linearized range
around it (the delta method); and refits that leave out one expiry at a time.

## 10. What the results show, and what they do not

For Heston on SPX, the results show:
- a network, trained once, that returns calibrated posteriors in under a second for five
  of six parameters, with intervals for $`v_0`$ about 1.4 times too wide;
- agreement with exact posteriors on simulated surfaces, within about one reference
  standard deviation and with intervals up to about 3 times wider;
- what each region of quotes pins (Figure 4), and so which products the quotes leave
  open;
- that Heston misses the SPX smile by about 20 times the median half-spread, mostly at
  short maturities and deep strikes, which is where rough Bergomi is built to do better.

They do not:
- cover the model's own error. The ranges are uncertainty about one model's parameters.
  Where the model misfits, as Heston does on SPX, they are too narrow, and the smooth
  model-error term of step 6 is the extension that addresses it;
- check the products against a true value on real data, which is never observed;
- show hedging under parameter uncertainty;
- establish that a posterior is an accepted basis for a prudent value. That is our
  proposal, not current practice.

**What remains.** Rough Bergomi's network and checks, once its simulations are
complete; the path-dependent products and the book under the exact SPX posterior; the
checks on left-out quotes and on variance swaps from the full strip; calibration of the
intervals for $`v_0`$; and the model-error term.

## Back to the goals, the audience and the practitioner

**Goals.**
1. *Marks and ranges* come from steps 3 and 9: one posterior per model, every product
   priced under every draw, and a range for the book.
2. *Fast and testable:* steps 7 and 8. Under a second per posterior; calibrated on 512
   simulated surfaces for five of six parameters; close to exact posteriors on simulated
   surfaces; exact posteriors carry the market results.
3. *Limits:* step 10, with the error model of step 6 as the choice that most affects
   them.

**Audience.** Researchers in AI and statistics see a real inverse problem in which two
things, not the network's size, decide whether the posterior is right: the error model and
the conditioning of the network's inputs. Practitioners see calibration ranges, a book
range and a model comparison from one consistent calculation, mapped onto the
prudent-valuation rules, with a plain statement of what they leave out. Both should leave
able to say, for their own model, whether they need this, what it costs, and how to check
it.

**Practitioner.** A desk and its product control would get, every day:
- a mark and a range for every product and for the book, from one consistent
  calculation, with tested coverage;
- a measure of how well the model fits that day's quotes ($`c`$ and the map of misses),
  which flags the days when a model stops describing the market;
- a comparison of two models on the same information.

They would still have to choose the models, handle the model risk beyond one model's
parameters (Article 11) and the other AVA categories, such as close-out costs, decide how
to hedge, take the method through model validation, and have their regulator accept it.
In the terms of the rules, the posterior's range is an input to Article 9, the comparison
of models belongs to Article 11, and the joint draws measure the offset between products
that the Regulation's aggregation sets at a fixed 50%.

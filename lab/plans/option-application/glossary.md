# Glossary

*Shared by the three layers · [Index](README.md)*

Plain-language definitions of the finance and statistics terms the layers use, in
alphabetical order. Each layer also explains a term where it first appears.

**Additional valuation adjustment (AVA).** The amount by which a position's fair value
exceeds its prudent value. EU banks compute AVAs for several sources of valuation
uncertainty, among them market price uncertainty (Article 9 of Regulation 2016/101),
close-out costs (Article 10) and model risk (Article 11). The Regulation's Annex
aggregates these three with a factor $`\alpha = 50\%`$, and the total is deducted from
Common Equity Tier 1 capital (CRR, Article 34).

**Amortized inference.** Paying the cost of inference once, by training a network on
simulated data, so that each later posterior costs seconds. The training cost is spread
("amortized") over all later uses.

**Ask.** The lowest price at which someone currently offers to sell. See *bid*.

**At the money.** An option whose strike is at, or near, the forward level of the index.
"At-the-money volatility" is the implied volatility there.

**Autocallable.** A structured note that pays coupons and is redeemed early ("called
automatically") if the index is above a set level on an observation date. If the index
has fallen below a barrier, here 70% of its starting level, the investor bears the loss
at maturity. In effect the investor has sold the bank a down-and-in put.

**Barrier.** An index level whose crossing switches an option on ("knock-in") or off
("knock-out").

**Bid.** The highest price at which someone currently offers to buy. The **bid–ask
spread** is the ask minus the bid; the *half-spread* is half of it.

**Black–Scholes formula.** The price of a call or put if the index had a constant
volatility $`\sigma`$. For a put with strike $`K`$, maturity $`T`$, forward $`F`$ and
discount factor $`D`$:
$`D\,[K\,\Phi(-d_2) - F\,\Phi(-d_1)]`$, with
$`d_{1,2} = [\ln(F/K) \pm \sigma^2 T/2]/(\sigma\sqrt T)`$ and $`\Phi`$ the standard
normal distribution function. Here it serves to translate prices into implied
volatilities.

**Book.** The set of positions a desk holds. Its value is the sum of its positions'
values.

**Calibration.** Choosing a model's parameters so that its prices reproduce market
prices.

**Call option.** Pays $`\max(S_T - K,\,0)`$ at maturity $`T`$: the right to buy the index
at the strike $`K`$.

**Cash-settled.** Settled by paying the payoff in cash, not by delivering the underlying.
SPX options are cash-settled.

**Cboe.** The exchange (Cboe Global Markets) on which SPX options are listed.

**CET1 (Common Equity Tier 1) capital.** A bank's highest-quality capital, mainly
shareholders' equity and retained earnings. Regulators set minimum ratios of CET1 to
risk-weighted assets; deductions such as AVAs reduce it.

**Common random numbers.** Using the same random numbers in the simulations for every
parameter draw, so that differences between draws come from the parameters and not from
simulation noise.

**Coverage.** How often an interval contains the true value. A well-calibrated 90%
interval has 90% coverage.

**Delta method.** A linear approximation that turns the covariance of fitted parameters
into the variance of a function of them, here a product's price:
$`\mathrm{Var}\,P \approx \nabla P^\top \mathrm{Cov}(\hat\theta)\,\nabla P`$.

**Desk.** A team of traders managing a book of positions.

**Discount factor** ($`D(T)`$). Today's value of one unit of currency paid at time $`T`$:
$`D(T) = e^{-r(T)\,T}`$, with $`r(T)`$ the interest rate for that maturity.

**Down-and-in put.** A put that comes into existence only if the index reaches a barrier
$`B`$ below its starting level during the put's life. Here the barrier is checked on daily
closes: the payoff is $`(K - S_T)^+`$ if some close was at or below $`B`$, and zero
otherwise.

**Error model.** The statement of how observed quotes differ from model prices. Here: a
quote uncertainty the size of the half-spread, plus a smooth model error of size
$`\tau`$. It defines the likelihood.

**European option.** An option that can be exercised only at maturity. An **American**
option can be exercised at any time before. SPX options are European; SPY options are
American.

**Exit.** Ending a position: selling what one owns, or buying back what one owes. Also
called closing out.

**Exotic.** Any derivative other than a plain call or put, usually traded over the
counter, often with a payoff that depends on the index's path.

**Fair value.** The price that would be received to sell an asset, or paid to transfer a
liability, in an orderly transaction between market participants (the accounting
standard IFRS 13). The mark is the bank's estimate of it.

**Forward** ($`F(T)`$). The price agreed today for buying the index at $`T`$, paid at
$`T`$. It equals today's level grown at the interest rate less the dividend yield,
$`F(T) = S_0\,e^{(r-q)T}`$. In this work it is read from the quotes through put–call
parity.

**Forward smile.** The smile that options starting at a future date will show. No listed
option reveals it today, and models disagree about it.

**Forward variance curve** ($`\xi_0(t)`$). The expected future variance at each date
$`t`$ under the pricing probabilities, $`\xi_0(t) = \mathbb{E}[V_t]`$. Rough Bergomi takes
it as a curve, here described by three numbers: a short-term level, a long-term level and
the speed between them. Not to be confused with Heston's vol of vol $`\xi`$.

**Forward-start call.** A call whose strike is fixed at a future date $`T_1`$ in
proportion to the index level then, and which pays at a later date $`T_2`$. Here
$`T_1`$ is 6 months and $`T_2`$ is 1 year. Its value depends on the forward smile.

**Half-spread** ($`h`$). Half the bid–ask spread, $`(\text{ask} - \text{bid})/2`$: the
uncertainty of a quote's price. Divided by vega, it is expressed in vol points.

**Heston model.** A stochastic volatility model in which the variance reverts to a
long-run level and fluctuates in proportion to its square root (Heston, 1993). Its
parameters are $`v_0`$, $`\bar{v}`$, $`\kappa`$, $`\xi`$ and $`\rho`$, and its option
prices come from a Fourier integral.

**Hurst exponent** ($`H`$). In rough Bergomi, the parameter that sets how rough volatility
paths are: $`H = 1/2`$ is as rough as Brownian motion, and $`H \lt 1/2`$ is rougher.
Estimates from data are near 0.1.

**Implied volatility.** The volatility $`\sigma`$ that makes the Black–Scholes formula
reproduce an option's market price. It quotes prices in a common unit across strikes and
maturities.

**Likelihood.** The probability (density) of the observed data given the parameters,
$`p(y \mid \theta)`$, seen as a function of $`\theta`$.

**Listed.** Traded on an exchange, with standard terms and public quotes. The opposite of
over the counter.

**Mark; mark to market; mark to model.** The mark is the value a bank assigns to a
position at the end of a day, for its P&L and balance sheet. Marking to market values a
position at prices readily available from independent sources, such as an exchange.
Marking to model computes the value from market inputs, with a model, when no such price
exists. EU banking rules (the CRR) define both and require marking to market whenever
possible.

**Maturity, expiry.** The date on which an option pays. $`T`$ denotes the time to it, in
years.

**MCMC (Markov chain Monte Carlo).** A family of algorithms that draw from a posterior
using only its density up to a constant. Here it gives Heston's exact posterior.

**Mean reversion** ($`\kappa`$). In Heston, the speed at which the variance is pulled back
to its long-run level $`\bar{v}`$: a gap halves in about $`\ln 2/\kappa`$ years.

**Model error.** The part of the gap between quotes and model prices that comes from the
model being wrong rather than from the quotes. Here it is smooth across strikes and
maturities, with size $`\tau`$.

**Model risk.** The risk that a value is wrong because the model is: other reasonable
models or calibrations would give other values. In EU prudent valuation it is the subject
of Article 11 of Regulation 2016/101.

**Out of the money.** A put whose strike is below the forward, or a call whose strike is
above it: it would pay nothing if the index ended at the forward. These are the liquid
quotes the design uses.

**Over the counter (OTC).** Traded privately between two parties, on terms set by
contract, rather than on an exchange. There is no public price.

**P&L.** Profit and loss: the change in the value of a book, usually per day.

**Posterior.** The distribution of the parameters given the data:
$`p(\theta \mid y) \propto p(y \mid \theta)\,p(\theta)`$.

**Pricing probabilities.** The probabilities under which a price is the expected
discounted payoff; also called the risk-neutral measure. Under them the index grows on
average at the interest rate less its dividend yield. Their existence is equivalent to
the absence of riskless profits.

**Prior.** The distribution of the parameters before seeing the data: which values are
considered plausible.

**Product control.** The bank function, independent of the traders, that verifies the
marks against market data and computes valuation adjustments such as AVAs.

**Prudent value.** Under EU rules, a value at which the bank is 90% confident it could
exit the position at that price or better (Regulation 2016/101, Articles 9 and 11). From
a posterior: the 10% quantile of the value for a position the bank owns, the 90% quantile
for one it owes.

**Put option.** Pays $`\max(K - S_T,\,0)`$ at maturity: the right to sell the index at
the strike $`K`$. It protects against a fall.

**Put–call parity.** For European options with the same strike and maturity,
$`C - P = D(T)\,(F(T) - K)`$: a call minus a put is a forward contract. It is used to read
the forward from the quotes.

**Realized volatility and variance.** Computed from the index's daily log returns $`r_i`$
over a period of $`N`$ trading days: realized variance $`= (252/N)\sum_i r_i^2`$, and
realized volatility is its square root.

**Reserve.** Here, the gap between a product's (or a book's) mark and its prudent value.

**Rough Bergomi.** A stochastic volatility model in which the logarithm of the variance
is driven by a rough process with Hurst exponent $`H \lt 1/2`$ (Bayer, Friz and
Gatheral, 2016). It is priced only by Monte Carlo simulation.

**Rough volatility.** Volatility whose paths are rougher than Brownian motion's. It makes
the skew steepen at short maturities, as index smiles do.

**SBC (simulation-based calibration).** A check of a posterior method: draw parameters
from the prior, simulate data, compute the posterior; the rank of the true parameter
among the posterior draws must then be uniformly distributed.

**SBI (simulation-based inference).** Inference that only needs to simulate data from
the model, not to evaluate its likelihood. Here, a neural network trained on simulated
pairs of parameters and quotes.

**Skew.** The slope of implied volatility against strike. For an equity index it is
negative: low strikes have higher implied volatilities.

**Smile.** Implied volatility as a function of strike and maturity. The name comes from
the U shape it takes in some markets; for an index it is mostly a skew.

**Spot** ($`S_t`$). The index level at time $`t`$.

**SPX, SPXW.** Options on the S&P 500 index, listed on Cboe: European and cash-settled.
SPXW are the series settled on the closing value, with expiries on almost every trading
day; the standard monthly SPX options are settled on the opening prices of their expiry
day.

**SPY.** A fund traded on an exchange that tracks the S&P 500. Its options are American,
so their prices include an early-exercise value; this application uses SPX instead.

**Stochastic volatility model.** A model in which the index's volatility is itself
random, such as Heston and rough Bergomi.

**Strike** ($`K`$). The fixed level in an option's payoff.

**Structured note.** A bond-like product sold to investors whose payoff depends on an
index through options embedded in it, such as an autocallable.

**Variance swap.** Pays the realized variance over a period minus a strike agreed today.
Its fair strike can be computed without a model, from the prices of out-of-the-money
options at all strikes of that maturity. The VIX index is built the same way.

**Vega** ($`\nu`$). The sensitivity of an option's price to its implied volatility,
$`\nu = \partial P/\partial\sigma`$. It converts an error in vol points into an error in
price.

**VIX.** Cboe's volatility index: approximately the 30-day variance-swap rate on the
S&P 500, computed from SPX option prices and quoted as a volatility.

**Vol of vol.** How much volatility itself fluctuates: $`\xi`$ in Heston, $`\eta`$ in
rough Bergomi.

**Vol point.** 0.01 of implied volatility: from 20% to 21% is one vol point.

**Volatility.** The annualized standard deviation of the index's log returns.

**Volatility swap.** Pays the realized volatility over a period minus a strike agreed
today. Unlike the variance swap, it cannot be replicated with listed options alone; its
value also depends on the vol of vol.

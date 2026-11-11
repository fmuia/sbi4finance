# Glossary

*Shared by the three layers · [Index](README.md)*

Plain-language definitions of the finance and statistics terms the layers use, in
alphabetical order. Each layer also explains a term where it first appears.

**ACT/365F.** A time convention: actual elapsed time divided by 365 days. Every maturity
here is measured this way, from the valuation time to the expiry's 16:00 New York close.

**Additional valuation adjustment (AVA).** The amount by which a position's fair value
exceeds its prudent value. EU banks compute AVAs for several sources of valuation
uncertainty, among them market price uncertainty (Article 9 of Regulation 2016/101),
close-out costs (Article 10) and model risk (Article 11). The Regulation's Annex
aggregates these three with a factor $`\alpha = 50\%`$, and the total is deducted from
Common Equity Tier 1 capital (CRR, Article 34).

**Aggregation factor.** In the Annex of Regulation 2016/101, the factor $`\alpha = 50\%`$
by which individual AVAs are reduced when they are added up, a fixed allowance for
diversification between valuation exposures.

**Amortized inference.** Paying the cost of inference once, by training a network on
simulated data, so that each later posterior costs well under a second. The training cost
is spread ("amortized") over all later uses.

**Antithetic paths.** Simulating each random path together with its mirror image (all
normals with flipped sign), which cancels part of the Monte Carlo noise.

**Ask.** The lowest price at which someone currently offers to sell. See *bid*.

**At the money.** An option whose strike is at, or near, the forward level of the index.
At-the-money volatility is the implied volatility there: on 9 November 2026, from 15.2%
at 7 days to 17.1% at 325 days.

**Autocallable.** A structured note that pays coupons and is redeemed early ("called
automatically") if the index is above a set level on an observation date. If the index
has fallen below a barrier, here 70% of its starting level, the investor bears the loss
at maturity. In effect the investor has sold the bank a down-and-in put.

**Bank (simulation bank).** A stored table of parameter sets drawn from the prior, each
with the model's clean prices at the design's quotes. Errors are added to the clean prices
to form the training observations.

**Barrier.** An index level whose crossing switches an option on ("knock-in") or off
("knock-out").

**Bid.** The highest price at which someone currently offers to buy. The **bid–ask
spread** is the ask minus the bid; the *half-spread* is half of it.

**Black–Scholes formula.** The price of a call or put if the index had a constant
volatility $`\sigma`$. For a put with strike $`K`$, maturity $`T`$, forward $`F`$ and
discount factor $`D`$:
$`D\,[K\,\Phi(-d_2) - F\,\Phi(-d_1)]`$, with
$`d_{1,2} = [\ln(F/K) \pm \sigma^2 T/2]/(\sigma\sqrt T)`$ and $`\Phi`$ the standard
normal distribution function. Here it translates prices into implied volatilities, and
serves as a control variate for Heston's Fourier integral.

**Book.** The set of positions a desk holds. Its value is the sum of its positions'
values.

**Breeden–Litzenberger formula.** The second derivative of the call price in the strike
is the discounted density of the index at maturity:
$`\partial^2 C/\partial K^2 = D(T)\,p_T(K)`$. So listed options at one maturity fix the
distribution of the index at that maturity, and nothing about how dates are linked.

**Broadie–Glasserman–Kou correction.** An adjustment that turns a barrier price for
continuous monitoring into one for monitoring at discrete dates, by shifting the barrier
by a factor $`e^{\pm 0.5826\,\sigma\sqrt{\Delta t}}`$.

**Calibrated posterior.** A posterior whose intervals contain the truth as often as their
level says: 90% intervals contain it 90% of the time. Not to be confused with calibrating
a model, which means fitting its parameters.

**Calibration.** Choosing a model's parameters so that its prices reproduce market
prices.

**Call option.** Pays $`\max(S_T - K,\,0)`$ at maturity $`T`$: the right to buy the index
at the strike $`K`$.

**Carry anchors.** Pairs of a call and a put with the same strike near spot, used only to
estimate each expiry's forward and discount factor by put–call parity; they are never
fitted as observations.

**Cash-settled.** Settled by paying the payoff in cash, not by delivering the underlying.
SPX options are cash-settled.

**Cboe.** The exchange (Cboe Global Markets) on which SPX options are listed.

**CET1 (Common Equity Tier 1) capital.** A bank's highest-quality capital, mainly
shareholders' equity and retained earnings. Regulators set minimum ratios of CET1 to
risk-weighted assets; deductions such as AVAs reduce it.

**Characteristic function.** The Fourier transform of a distribution,
$`\varphi(u) = \mathbb{E}[e^{iu\ln X_T}]`$. Heston's is known in closed form, and one
Fourier integral turns it into an option price.

**Cholesky factor.** For a covariance matrix $`\Gamma`$, the lower-triangular $`L`$ with
$`LL^\top = \Gamma`$. Multiplying independent normals by $`L`$ gives correlated ones.

**Close-out costs.** The cost of exiting a position at the bid or ask rather than the mid;
an AVA category of its own (Article 10 of Regulation 2016/101).

**Common random numbers.** Using the same random numbers in the simulations for every
parameter draw, so that differences between draws come from the parameters and not from
simulation noise.

**Condition number.** The ratio of the largest to the smallest eigenvalue of a covariance
(or curvature) matrix. A large one means some directions are tiny next to others, which
gradient-based training resolves very slowly.

**Consensus service.** A service that collects banks' valuations of the same products and
returns the consensus, and sometimes its spread (some publish 10th and 90th percentiles
for prudent valuation).

**Control variate.** A quantity computed on the same paths, or in the same integral, whose
exact value is known; subtracting its error, scaled by a regression coefficient, reduces
the noise of the estimate.

**Coverage.** How often an interval contains the true value. A calibrated 90% interval has
90% coverage.

**CRR.** The EU Capital Requirements Regulation, Regulation (EU) No 575/2013, which sets
banks' capital rules. Its Article 105 asks for prudent valuation, and Article 34 deducts
AVAs from CET1 capital.

**Delta method.** A linear approximation that turns the covariance of fitted parameters
into the variance of a function of them, here a product's price:
$`\mathrm{Var}\,P \approx \nabla P^\top \mathrm{Cov}(\hat\theta)\,\nabla P`$.

**Desk.** A team of traders managing a book of positions.

**Discount factor** ($`D(T)`$). Today's value of one unit of currency paid at time $`T`$:
$`D(T) = e^{-r(T)\,T}`$, with $`r(T)`$ the interest rate for that maturity.

**Diversification ratio.** A book's reserve divided by the sum of its products' standalone
reserves. Below 1 when the products' values offset each other across posterior draws; the
Regulation's aggregation sets it at a fixed 0.5.

**Down-and-in put.** A put that comes into existence only if the index reaches a barrier
$`B`$ below its starting level during the put's life. Here the barrier is checked on daily
closes: the payoff is $`(K - S_T)^+`$ if some close was at or below $`B`$, and zero
otherwise.

**Dropout.** A training device that switches off a random share of a network's units at
each update (10% in the Heston network), so that the network does not depend on any single
one.

**Effective sample size (ESS).** The number of independent draws that a set of correlated
(MCMC) or weighted (importance-sampling) draws is worth. For importance weights $`w_s`$,
$`(\sum_s w_s)^2/\sum_s w_s^2`$.

**Epoch.** One pass of training over the whole training set. With 300,000 rows in batches
of 256, one epoch is 1,172 updates; the Heston network trains for 300.

**Error model.** The statement of how observed quotes differ from model prices; it defines
the likelihood. Here each quote's error is Gaussian with standard deviation $`c`$ times its
half-spread, with the multiplier $`c`$ inferred. A smooth model-error term of size
$`\tau`$ is the extension.

**European option.** An option that can be exercised only at maturity. An **American**
option can be exercised at any time before. SPX options are European; SPY options are
American.

**Exit.** Ending a position: selling what one owns, or buying back what one owes. Also
called closing out.

**Fair value.** The price that would be received to sell an asset, or paid to transfer a
liability, in an orderly transaction between market participants (the accounting
standard IFRS 13). The mark is the bank's estimate of it.

**Fair-value levels (IFRS 13).** Level 1: quoted prices for the identical instrument in an
active market. Level 2: other observable inputs. Level 3: significant inputs nobody
quotes; banks disclose how sensitive such values are to reasonably possible alternative
inputs.

**Feller condition.** In Heston, $`2\kappa v_\infty \ge \sigma_v^2`$: it keeps the variance
away from zero. Violating it is allowed; the model and its Fourier prices remain valid.

**Fisher information.** The expected curvature of the log-likelihood in the parameters. Its
inverse approximates the parameters' covariance; small eigenvalues mark directions the
data barely constrain.

**Flow matching.** A way to train a network that turns random noise into draws from a
target distribution. Each training example joins a simulated parameter vector to a noise
vector by a straight line; the network learns, by least squares, the average velocity of
these lines at every point and time. Drawing from the posterior then means starting from
noise and following that velocity, which is solving an ordinary differential equation.
Used here through BayesFlow 2.1.

**Forward** ($`F(T)`$). The price agreed today for buying the index at $`T`$, paid at
$`T`$. It equals today's level grown at the interest rate less the dividend yield,
$`F(T) = S_0\,e^{(r-q)T}`$. Here it is read from the quotes through put–call parity.

**Forward smile.** The smile that options starting at a future date will show. No listed
option reveals it today, and models disagree about it.

**Forward-start call.** A call whose strike is fixed at a future date $`T_1`$ in
proportion to the index level then, and which pays at a later date $`T_2`$. Here
$`T_1`$ is 6 months and $`T_2`$ is 1 year. Its value depends on the forward smile.

**Forward-starting variance swap.** A variance swap whose period starts at a future date:
it pays the realized variance between $`t_1`$ and $`t_2`$ minus a strike agreed today. Its
fair strike is the expected variance averaged over the window.

**Forward units.** Prices divided by $`D(T)\,F(T)`$, and the index divided by its
forward, $`X_t = S_t/F(t)`$. In these units the index has no drift under the pricing
probabilities.

**Forward variance curve** ($`\xi_0(t)`$). The expected future variance at each date
$`t`$ under the pricing probabilities, $`\xi_0(t) = \mathbb{E}[V_t]`$. Rough Bergomi takes
it as a curve, here described by three numbers: a short-term level $`\xi_s`$, a long-term
level $`\xi_l`$ and the speed $`\lambda`$ between them.

**Full-truncation Euler scheme.** A time-stepping scheme for Heston's variance that
replaces negative values by zero in the drift and the volatility, so the simulation stays
well defined.

**Half-spread** ($`h`$). Half the bid–ask spread, $`(\text{ask} - \text{bid})/2`$: the
uncertainty of a quote's price. Divided by vega, it is expressed in vol points.

**Held-out quotes.** Quotes kept out of every fit, to test the model's predictions out of
sample.

**Heston model.** A stochastic volatility model in which the variance reverts to a
long-run level and fluctuates in proportion to its square root (Heston, 1993). Its
parameters are $`v_0`$, $`v_\infty`$, $`\kappa`$, $`\sigma_v`$ and $`\rho`$, and its option
prices come from a Fourier integral.

**Hurst exponent** ($`H`$). In rough Bergomi, the parameter that sets how rough volatility
paths are: $`H = 1/2`$ is as rough as Brownian motion, and $`H \lt 1/2`$ is rougher.
Estimates from data are near 0.1.

**Hybrid scheme.** The standard way to simulate rough Bergomi (Bennedsen, Lunde and
Pakkanen): the singular part of the kernel near the present is simulated exactly, the
rest by a convolution.

**Implied volatility.** The volatility $`\sigma`$ that makes the Black–Scholes formula
reproduce an option's market price. It quotes prices in a common unit across strikes and
maturities.

**Importance sampling.** Drawing from a convenient distribution $`q`$ and weighting each
draw by $`p/q`$ to obtain expectations under the target $`p`$. Here $`q`$ is the network's
posterior and $`p`$ the exact one.

**Likelihood.** The probability (density) of the observed data given the parameters,
$`p(y \mid \theta)`$, seen as a function of $`\theta`$.

**Listed.** Traded on an exchange, with standard terms and public quotes. The opposite of
over the counter.

**Log-moneyness** ($`k`$). $`k = \ln(K/F(T))`$: how far a strike is from the forward, on a
log scale. See *standardized moneyness*.

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
to its long-run level $`v_\infty`$: a gap halves in about $`\ln 2/\kappa`$ years.

**Model error.** The part of the gap between quotes and model prices that comes from the
model being wrong rather than from the quotes. On SPX it is smooth across strikes and
maturities; the extension of step 6 represents it as a smooth random curve of size
$`\tau`$ in implied volatility.

**Model risk.** The risk that a value is wrong because the model is: other reasonable
models or calibrations would give other values. In EU prudent valuation it is the subject
of Article 11 of Regulation 2016/101.

**Neural posterior estimation (NPE).** Training a conditional density network
$`q(\theta \mid y)`$ on simulated pairs by maximizing $`\log q(\theta \mid y)`$; the
optimum is the posterior.

**Out of the money.** A put whose strike is below the forward, or a call whose strike is
above it: it would pay nothing if the index ended at the forward. These are the liquid
quotes the design uses.

**Over the counter (OTC).** Traded privately between two parties, on terms set by
contract, rather than on an exchange. There is no public price.

**Pareto k-hat** ($`\hat k`$). A diagnostic of importance sampling: it estimates how heavy the
tail of the weights is. Below 0.7 the estimate is reliable.

**P&L.** Profit and loss: the change in the value of a book, usually per day.

**Population and conditional checks.** Population checks (coverage, rank tests) average
over many simulated observations; conditional checks compare single posteriors with
reference posteriors. Population checks can pass while single posteriors are wrong.

**Posterior draw.** One parameter vector $`\theta^{(s)}`$ sampled from the posterior. A
set of draws represents the posterior: any summary (mean, quantile, the value of a
product) is computed from them.

**Posterior.** The distribution of the parameters given the data:
$`p(\theta \mid y) \propto p(y \mid \theta)\,p(\theta)`$.

**Pricing probabilities.** The probabilities under which a price is the expected
discounted payoff; also called the risk-neutral measure. Under them the index grows on
average at the interest rate less its dividend yield. Their existence is equivalent to
the absence of riskless profits.

**Pricing.** Here, one computation of the model's 63 prices $`P(\theta)`$ at one
parameter vector. It is the unit of cost: one likelihood evaluation needs one pricing, a
posterior by MCMC about 146,000, the bank 300,000, a posterior from the trained network
none.

**Prior.** The distribution of the parameters before seeing the data: which values are
considered plausible.

**Prior predictive check.** Simulating surfaces from the prior and the error model, and
checking that the real quotes look like a possible draw.

**Product control.** The bank function, independent of the traders, that verifies the
marks against market data and computes valuation adjustments such as AVAs.

**Prudent value.** Under EU rules, a value at which the bank is 90% confident it could
exit the position at that price or better (Regulation 2016/101, Articles 9 and 11). From
a posterior: the 10% quantile of the value for a position the bank owns, the 90% quantile
for one it owes.

**Push-forward.** The distribution of a function of a random quantity: here, of a
product's value when the parameters follow the posterior.

**Put option.** Pays $`\max(K - S_T,\,0)`$ at maturity: the right to sell the index at
the strike $`K`$. It protects against a fall.

**Put–call parity.** For European options with the same strike and maturity,
$`C - P = D(T)\,(F(T) - K)`$: a call minus a put is a forward contract. It is used to read
the forward and the discount factor from the quotes.

**Quadrature.** Numerical integration as a weighted sum of the integrand at chosen nodes.
Heston prices use Gauss–Legendre quadrature, refined until two successive node sets agree.

**R-hat** ($`\hat R`$). An MCMC diagnostic comparing the spread within and between
chains. Close to 1 (here at most 1.01) means the chains agree.

**Rank test.** See *SBC*.

**Realized volatility and variance.** Computed from the index's daily log returns $`r_i`$
over a period of $`N`$ trading days: realized variance $`= (252/N)\sum_i r_i^2`$, and
realized volatility is its square root.

**Reference posterior.** A posterior computed directly from the likelihood (by MCMC or
importance sampling), used to check the network's.

**Reserve.** Here, the gap between a product's (or a book's) mark and its prudent value.

**Rough Bergomi.** A stochastic volatility model in which the logarithm of the variance
is driven by a rough process with Hurst exponent $`H \lt 1/2`$ (Bayer, Friz and
Gatheral, 2016). It is priced only by Monte Carlo simulation.

**Rough volatility.** Volatility whose paths are rougher than Brownian motion's. It makes
the skew steepen at short maturities, as index smiles do.

**Scaled residual** ($`r_i`$). $`r_i(\theta) = (y_i - P_i(\theta))/h_i`$: the model's miss
on quote $`i`$, in half-spreads. Under the error model its typical size is $`c`$.

**SBC (simulation-based calibration).** A check of a posterior method: draw parameters
from the prior, simulate data, compute the posterior; if the posterior is right, the rank
of the true parameter among the posterior draws is uniformly distributed. Also called the
rank test.

**SBI (simulation-based inference).** Inference that only needs to simulate data from
the model, not to evaluate its likelihood. Here, a neural network trained on simulated
pairs of parameters and quotes.

**Simulated surface.** One set of 63 prices at the design's contracts, produced by the
generative model: parameters drawn from the prior, priced by the model, with errors
added.

**Skew.** The slope of implied volatility against strike. For an equity index it is
negative: low strikes have higher implied volatilities.

**Smile.** Implied volatility as a function of strike and maturity. The name comes from
the U shape it takes in some markets; for an index it is mostly a skew.

**Spline coupling flow.** A normalizing flow: a chain of invertible maps built from
monotone splines that turns a standard normal into an approximate posterior and can
evaluate that approximation's density. Trained by neural posterior estimation.

**Spot** ($`S_t`$). The index level at time $`t`$.

**SPX, SPXW.** Options on the S&P 500 index, listed on Cboe: European and cash-settled.
SPXW are the series settled on the closing value, with expiries on almost every trading
day; the standard monthly SPX options are settled on the opening prices of their expiry
day.

**SPY.** A fund traded on an exchange that tracks the S&P 500. Its options are American,
so their prices include an early-exercise value; this application uses SPX instead.

**Squared-exponential correlation.** $`\exp(-d^2/2\ell^2)`$ between two points at
distance $`d`$: near 1 for nearby points, near 0 beyond a few lengths $`\ell`$. It makes
the model error a smooth random curve.

**Standardized moneyness** ($`\delta`$).
$`\delta = \ln(K/F(T))/(\sigma_{\text{ref}}\sqrt{T})`$ with $`\sigma_{\text{ref}} = 0.16`$:
the distance of a strike from the forward in units of the standard deviation the log
index would have at expiry under a 16% volatility. It only places and labels strikes;
prices use the actual strikes. The design's files call it `z`.

**Standardized unknowns** ($`\tilde\theta`$). Each unknown minus its mean over the bank,
divided by its standard deviation over the bank, so that all six have comparable sizes for
the network. The network's draws are converted back to the original units.

**Stochastic volatility model.** A model in which the index's volatility is itself
random, such as Heston and rough Bergomi.

**Strike** ($`K`$). The fixed level in an option's payoff.

**Structured note.** A bond-like product sold to investors whose payoff depends on an
index through options embedded in it, such as an autocallable.

**Variance swap.** Pays the realized variance over a period minus a strike agreed today.
Its fair strike can be computed without a model, from the prices of out-of-the-money
options at all strikes of that maturity. The VIX index is built the same way.

**Velocity field.** A rule that gives, for a point at a position and a time, the direction
and speed of its motion. In flow matching the network is the velocity field
$`v_\phi(z_t, t, \tilde x)`$: it takes the current point, the time and the whitened quotes,
and returns six numbers.

**Vega** ($`\nu`$). The sensitivity of an option's price to its implied volatility,
$`\nu = \partial P/\partial\sigma`$. It converts an error in vol points into an error in
price.

**Vega notional.** The usual way to size a variance or volatility swap: the amount paid
per vol point by which realized volatility exceeds the strike.

**VIX.** Cboe's volatility index: approximately the 30-day variance-swap rate on the
S&P 500, computed from SPX option prices and quoted as a volatility.

**Vol of vol.** How much volatility itself fluctuates: $`\sigma_v`$ in Heston (the
volatility of the variance), $`\eta`$ in rough Bergomi.

**Vol point.** 0.01 of implied volatility: from 20% to 21% is one vol point.

**Volatility.** The annualized standard deviation of the index's log returns.

**Volatility swap.** Pays the realized volatility over a period minus a strike agreed
today. Unlike the variance swap, it cannot be replicated with listed options alone; its
value also depends on the vol of vol.

**Whitening (ZCA).** A fixed linear change of input coordinates, fitted on training
simulations, that makes the inputs uncorrelated with unit variance. With $`\mu`$ the mean
of the scaled quotes $`x = y/h`$ and $`\Sigma = U\,\mathrm{diag}(\lambda_1, \dots, \lambda_{63})\,U^\top`$
their covariance: $`\tilde x = W(x - \mu)`$,
$`W = U\,\mathrm{diag}(\lambda_1^{-1/2}, \dots, \lambda_{63}^{-1/2})\,U^\top`$. It loses no
information and lets a network read small errors hidden under large common moves.

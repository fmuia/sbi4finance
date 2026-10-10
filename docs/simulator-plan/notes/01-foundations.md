# Financial and statistical foundations

This guide explains the two questions the simulator library serves: what historical returns reveal about an asset's dynamics, and what option quotes reveal about a model used to value other contracts. Both use simulation and Bayesian inference. They use different observations and different probability laws.

Read this before [models and numerics](02-models-and-numerics.md), then continue to [inference and pricing](03-inference-and-pricing.md). The [implementation plan](../README.md) explains when each concept becomes software.

## Prices and returns

Let $S_t>0$ be an asset price at time $t$. A simple return over an interval is

$$
R_i=\frac{S_{t_i}}{S_{t_{i-1}}}-1.
$$

A log return is

$$
r_i=\log S_{t_i}-\log S_{t_{i-1}}=\log(1+R_i).
$$

Simple returns compound by multiplication: $1+R_{1:n}=\prod_i(1+R_i)$. Log returns add: $r_{1:n}=\sum_i r_i$, and the corresponding simple return is $\exp(r_{1:n})-1$. The letter $r_i$ here means a return; an unsubscripted $r$ below denotes an interest rate. Context and explicit field names in code will prevent confusion.

A price moving from 100 to 102 has a simple return of 0.02, or 2%, and a log return of about 0.019803. Decimal values and percentage displays must not be interchanged in a training bank. Log returns also require a positive underlying series; a commodity series that can reach zero or become negative needs a different representation or a different modeled instrument.

An observed history is the complete ordered sequence $(r_1,\ldots,r_n)$ and its observation times. The order carries information about volatility clustering and state evolution. Shuffling days usually changes the inference problem.

## Volatility and hidden state

Volatility describes the scale of random price movements. In a diffusion model, an annualized volatility $\sigma=0.20$ corresponds to instantaneous variance $v=\sigma^2=0.04$ per year. Under constant volatility and a trading-time convention with 252 equal intervals per year, the approximate daily return standard deviation is

$$
\sigma\sqrt{1/252}=0.01260,
$$

or 1.26%. Variance scales with elapsed time; standard deviation scales with its square root in this constant-volatility example. Stochastic variance and jumps make longer-horizon distributions more complicated.

A model parameter describes a persistent feature such as average variance or jump frequency. A state describes the system now, such as today's variance. In stochastic-volatility models, variance changes through time and is not directly observed from one daily return. A large move may reflect high variance, a jump, or an unusually large diffusion shock. Inference must account for this ambiguity.

“Filtering” means estimating the current hidden state using observations available up to now. “Smoothing” means estimating a past hidden state using later observations as well. Forecasting from a freshly reset state answers a different question from forecasting conditional on the observed history.

## Random processes and simulation

A Brownian motion $W_t$ is a continuous random driver whose increments over disjoint time intervals are independent. An increment over length $\Delta t$ is normally distributed with mean zero and variance $\Delta t$:

$$
\Delta W=\sqrt{\Delta t}\,Z,\qquad Z\sim\mathcal N(0,1).
$$

Geometric Brownian motion, or GBM, describes a simple positive-price process:

$$
\frac{dS_t}{S_t}=\mu\,dt+\sigma\,dW_t.
$$

The notation says that a small proportional price change combines systematic growth and a random shock. Here $\mu$ is the expected proportional price growth rate, not the expected log-return rate. The exact update at discrete observation times is

$$
S_{t+\Delta t}=S_t\exp\left[(\mu-\tfrac12\sigma^2)\Delta t
+\sigma\sqrt{\Delta t}Z\right].
$$

The subtraction of $\sigma^2/2$ is the stochastic-calculus correction that makes the expected price growth match $\mu$. This exact formula gives us a valuable first simulator: its error does not depend on using a fine internal time step.

More complicated models introduce a random variance process, jumps, or memory. Often their continuous-time equations cannot be simulated exactly with a simple update. A numerical scheme approximates them at internal time steps. One daily observation may therefore aggregate many internal simulation steps. Learning an approximate simulator perfectly still leaves its numerical approximation error.

## Two probability measures and one Bayesian posterior

The physical measure $\mathbb P$ describes the model's real-world probabilities. It is the natural setting for questions such as “How large might next week's loss be?” Historical returns inform parameters under this measure.

The risk-neutral measure $\mathbb Q$ describes probabilities used for valuation under a specified no-arbitrage pricing model. For a stock with continuous dividend yield $q$, its proportional drift under Q is $r-q$ in the simple diffusion setting, with $r$ the risk-free rate. This is a pricing construction; it does not say investors actually expect stocks to earn that rate.

With stochastic volatility and jumps, absence of arbitrage alone need not select a unique Q measure. We specify a pricing model and estimate its Q parameters from quotes. A P parameter estimate is not automatically a Q estimate. Connecting them requires assumptions about compensation for risks such as volatility and jumps.

A Bayesian probability density $p(\vartheta)$ over parameters is a third object: it represents uncertainty about which parameter values to use. We can do Bayesian inference for P dynamics or Q dynamics. “Bayesian versus frequentist” and “physical versus risk-neutral” describe different distinctions.

| Question | Evidence | Probability law | Typical unknowns |
|---|---|---|---|
| What losses might occur next? | Ordered historical returns | P | Dynamic parameters and current state |
| What does the pricing model imply for another option? | Current option quotes and market inputs | Q | Pricing parameters and unknown variance/curve inputs |

## Options and model prices

A European call gives its holder a payoff $\max(S_T-K,0)$ at maturity $T$, where $K$ is its strike. A European put pays $\max(K-S_T,0)$. “European” refers to exercise at maturity, not the geographic location of trading. A contract specification includes the payoff, maturity and any other terms needed to define its cash flows.

With deterministic discounting, a model price is

$$
V(\vartheta,c)=D(0,T)\,\mathbb E^{\mathbb Q_\vartheta}
[\Phi_c(S_{[0,T]})],\qquad
D(0,T)=\exp\left(-\int_0^T r(u)\,du\right).
$$

Here $c$ identifies a contract, $\Phi_c$ is its payoff, and $\vartheta$ collects the model's unknown parameters or initialization inputs. A path-dependent payoff may depend on the whole trajectory $S_{[0,T]}$, not just its endpoint. The expectation averages over model futures at fixed parameters. This pricing principle and the simplest closed-form option formula originate in the classical option-pricing framework. [Black and Scholes](https://doi.org/10.1086/260062)

For constant $r,q,\sigma$ and remaining maturity $T>0$, the European call reference is

$$
C=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2),\quad
d_1=\frac{\log(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},
\quad d_2=d_1-\sigma\sqrt T,
$$

where $\Phi$ is the standard normal cumulative distribution function. It gives an independent target for the GBM pricing tests. Under these assumptions, put-call parity is $C-P=S_0e^{-qT}-Ke^{-rT}$.

An implied volatility is the volatility value that makes this benchmark formula match a quoted price. It is a way to express a quote, not a direct measurement of the hidden instantaneous variance in a more complex model. A surface contains quotes across strikes and maturities; these jointly constrain a model but may leave individual parameters weakly identified.

An American option permits earlier exercise. Its pricing problem includes an optimal stopping decision. It requires a suitable exercise algorithm and its own validation, which is why American pricing is a separate extension rather than an automatic consequence of adding a Monte Carlo path generator.

## Bayes and identifiability

Let $y$ denote the observed data and $\vartheta$ the unknown parameter vector. A prior $p(\vartheta)$ describes uncertainty before conditioning on these data. A likelihood $p(y\mid\vartheta)$ describes how plausible the observation is under a parameter value. Bayes' rule gives

$$
p(\vartheta\mid y)=
\frac{p(y\mid\vartheta)p(\vartheta)}{p(y)},\qquad
p(y)=\int p(y\mid\vartheta)p(\vartheta)\,d\vartheta.
$$

The posterior represents parameter uncertainty after conditioning. The denominator ensures the density integrates to one; it does not depend on which parameter value is being evaluated. When hidden states are also unknown, we extend the target to their joint posterior with parameters.

Weak identifiability means different parameter combinations produce similar observations. A good point fit can therefore coexist with a broad posterior. More complex models do not necessarily make more parameters learnable. Priors can stabilize inference, but posterior conclusions along weakly informed directions can depend strongly on those priors.

Simulation-based inference, or SBI, uses synthetic parameter–observation pairs to learn an inference rule. In the proposed neural posterior estimation workflow, the network approximates the posterior distribution conditional on an observation. It is trained once over a declared range of parameters and contexts, then reused for compatible new observations. This reuse is called amortization. The mathematical training objective is developed in [inference and pricing](03-inference-and-pricing.md).

## An analytic first inference problem

An intentionally simple task gives an independent correctness check before we use latent volatility. Suppose GBM volatility $\sigma$ is known and the drift $\mu$ has a Gaussian prior:

$$
\mu\sim\mathcal N(m_0,s_0^2),\qquad
r_i\mid\mu\sim\mathcal N\left((\mu-\tfrac12\sigma^2)\Delta t_i,
\sigma^2\Delta t_i\right).
$$

Define $a_i=r_i+\sigma^2\Delta t_i/2$. Completing the square in the prior times likelihood gives an exact Gaussian posterior:

$$
s_n^2=\left(s_0^{-2}+\frac{\sum_i\Delta t_i}{\sigma^2}\right)^{-1},
\qquad
m_n=s_n^2\left(\frac{m_0}{s_0^2}+\frac{\sum_i a_i}{\sigma^2}\right).
$$

Then $\mu\mid r_{1:n}\sim\mathcal N(m_n,s_n^2)$. These equations are obtained directly from the specified Gaussian model and are useful for testing simulator output, dataset labels and a learned posterior separately.

For example, take annual volatility 20%, a prior $\mu\sim\mathcal N(0,0.1^2)$, one year of observations, and cumulative log return 0.03. Then $\sum_i a_i=0.05$, posterior variance is $1/(100+25)=0.008$, posterior mean is 0.01 and posterior standard deviation is about 0.0894. One year's returns provide limited information about drift. The example is illustrative, not a proposed prior for the final finance application.

Notice that information about drift depends on total elapsed time in this model. Increasing sampling frequency over the same year does not automatically identify drift more precisely when volatility is known. This is one reason a simulator ladder should assess identifiability rather than count parameters alone.

There is also an exact posterior predictive reference. For a future horizon $h$ with independent new process shocks, integrating the Gaussian drift posterior gives

$$
r_{\mathrm{future}}\mid r_{1:n}
\sim\mathcal N\left((m_n-\tfrac12\sigma^2)h,
\sigma^2h+s_n^2h^2\right).
$$

The first variance term comes from future price shocks; the second comes from uncertainty about drift. This provides a direct check of the simulation-based prediction step. It relies on known constant volatility and the stated Gaussian prior, rather than being a general formula for stochastic-volatility models.

## Four uncertainties to keep separate

| Uncertainty | Example | How we examine it |
|---|---|---|
| Random future outcomes | Next week's return varies even at known parameters | Simulate future paths |
| Parameter and state uncertainty | Current variance or jump intensity is unknown | Infer the joint posterior |
| Numerical error | A coarse time step biases option values | Refine the numerical method and compare references |
| Model inadequacy | Actual markets differ from the assumed dynamics | Predictive checks, sensitivity analysis and empirical evaluation |

Neural approximation error adds another layer: a trained network may fail to reproduce the target posterior or price map. A narrow neural interval is persuasive only after checking these layers; a simulator-generated dataset does not make the model or the neural approximation automatically correct.

## Check your understanding

1. If annual volatility doubles, what happens to one-day standard deviation in GBM? It doubles. Instantaneous variance increases by a factor of four.
2. Do 50,000 paths used to price one option panel make 50,000 calibration examples? No. They estimate one panel at one model setting; observation examples are created at the experiment level.
3. Does a posterior over Q parameters describe actual future-return probabilities? It describes uncertainty in the pricing model. Real-world forecasts require a specified P model or an explicit connection between measures.
4. Does a numerical price belong to the same distribution as one realized payoff? No. The price is an expectation under the model and discounting convention. The payoff is one random outcome.
5. Why keep GBM if the final model is more complex? Its exact distributions and prices test shared infrastructure against independent answers.

Next: [Models and numerical simulation](02-models-and-numerics.md).

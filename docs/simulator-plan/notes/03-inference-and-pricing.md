# Inference pricing and risk

Status: educational design notes for review; no simulator, network, or dataset described here is implemented by this document. Read the [plan](../README.md), [foundations](01-foundations.md), and [models and numerical methods](02-models-and-numerics.md) first. The [software and data notes](04-software-and-data.md) translate these distinctions into interfaces.

## 1 The object of inference

A simulator draws possible observations from specified assumptions. Bayesian inference asks which unknown quantities remain plausible after observing actual data. Let:

- $\vartheta$ denote the full vector of static model parameters;
- $z_{0:T}$ denote the latent, or unobserved, state history;
- $y$ denote the observed return history or option panel;
- $c$ denote known context: observation times, contracts, rates, masks, and other declared inputs.

For a returns experiment, the generative distribution is

$$
p(\vartheta,z_{0:T},y\mid c)
=p(\vartheta\mid c)\,p(z_{0:T},y\mid\vartheta,c).
$$

The second factor includes initialization, process dynamics, and the observation mechanism. Writing it jointly avoids accidentally assuming that return and variance innovations are independent: leverage explicitly couples them. The posterior is

$$
p(\vartheta,z_T\mid y,c)
\propto p(\vartheta\mid c)
\int p(z_{0:T},y\mid\vartheta,c)\,dz_{0:T-1}.
$$

Integration means accounting for every compatible hidden history, weighted by its probability. We need not reconstruct every past variance value if our goal only requires parameters and a sufficient current state.

For continuous-time Heston with independent price jumps, current variance $v_T$ supplies the stochastic state needed for future relative returns. A numerical scheme may retain additional state: full-truncation Euler's auxiliary variance can be negative and must be preserved for exact continuation of that discrete scheme. Finite-factor Bergomi needs all its factors. Rough Bergomi generally needs a conditional memory representation, not just $v_T$. Calling every target “parameters plus current volatility” would hide this difference.

An initialization law is substantive. Fixing $v_0$, drawing it from a stationary distribution, and inferring it as another parameter define different experiments. Training and reference inference must use the same choice. Likewise, context is not automatically known: an uncertain rate or forward-variance curve requires an observation model or an explicit conditional approximation.

## 2 Two observation problems and two probability measures

One returns training example is a complete ordered history, such as 252 daily log returns, together with its times and mask. One option-calibration example is a complete surface or quote panel with contract coordinates. Individual days and strikes are elements of those examples. They do not become independent inference examples simply because they occupy separate array rows.

Historical returns describe evolution under the **physical measure**, $\mathbb P$. Derivative valuation generally uses a specified **risk-neutral measure**, $\mathbb Q$, under which discounted tradable gains satisfy the appropriate martingale condition. These measures can assign different probabilities to adverse outcomes. A return-based posterior over $\vartheta^{\mathbb P}$ is therefore not automatically a posterior over pricing parameters $\vartheta^{\mathbb Q}$.

Shared simulation machinery does not establish the relationship. Joint inference would need an explicit model of risk premia and a mapping between the measures. Changing only the stock drift is insufficient in general: volatility and jump dynamics can also change. The HKDE paper's option calibration belongs to the risk-neutral workflow. [Agazzotti et al., HKDE calibration and pricing](https://arxiv.org/html/2502.13824v1)

For a quote panel, a simple illustrative observation model is

$$
y_j=V_j(\vartheta^{\mathbb Q};c)+\epsilon_j,
\qquad \epsilon\sim\mathcal N(0,\Sigma),
$$

where $V_j$ is a model price and $\Sigma$ describes quote error or discrepancy. Independent errors correspond to diagonal $\Sigma$, an assumption to justify rather than a universal market fact. Noise in implied volatility is a different likelihood from noise in price because conversion is nonlinear.

Three uncertainties must remain separate:

| Quantity | Meaning | What reduces it? |
|---|---|---|
| Prior/posterior uncertainty | Uncertainty about parameters or hidden state | Informative observations, conditional on the model |
| Observation error/discrepancy | Differences between model observables and recorded data | Better measurements or a better observation model |
| Numerical error | Approximation introduced by integration, pricing, or Monte Carlo | More accurate computation |

The prior specifies plausible unknown quantities before the current data. It is not an extra perturbation to add to each quote. Monte Carlo noise is also not a substitute for market noise. If numerical prices fluctuate materially relative to the declared quote error, the implemented training distribution differs from the intended one.

## 3 What neural posterior estimation learns

Choose a target vector $u$, for example $u=(\vartheta,v_T)$ for a Heston returns task. Generate independent training examples from the declared joint distribution and fit a conditional density $q_\phi(u\mid y,c)$, where $\phi$ denotes network weights.

A likelihood-trained density model minimizes conditional cross-entropy:

$$
\mathcal L(\phi)
=\mathbb E_{p(u,y,c)}[-\log q_\phi(u\mid y,c)]
\approx-\frac1N\sum_{i=1}^N\log q_\phi(u_i\mid y_i,c_i).
$$

Conditioning on each $(y,c)$, this objective equals the posterior entropy plus
$\operatorname{KL}[p(u\mid y,c)\Vert q_\phi(u\mid y,c)]$.
The entropy does not depend on the network. Thus an expressive model, sufficient training, and correct simulation can recover the target posterior in the ideal limit. Finite data and optimization do not guarantee that limit. Conditional-density learning is a foundational route to neural posterior estimation. [Papamakarios and Murray, 2016](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html)

Amortization means paying for simulation and training once, then reusing the network for many compatible datasets. Compatibility includes the prior, observation conventions, supported context, and model. A network trained on fixed 252-day windows does not automatically handle arbitrary histories. [BayesFlow workflow paper](https://arxiv.org/abs/2306.16015)

Flow matching uses a different training objective; the expression above specifically describes conditional density training by log likelihood. Either approach still needs the same target/observation contract and scientific validation.

Joint targets matter. Drawing $\vartheta$ from one marginal posterior and $v_T$ independently from another generally destroys posterior dependence. Keep matched joint draws for forecasting. A network trained only on parameters cannot supply a current-state posterior without an additional conditional filtering procedure.

Transforms help enforce support. For strictly positive variance use $a=\log v_T$; for $\rho\in(-1,1)$, a transformed coordinate such as $\operatorname{atanh}\rho$ is possible. A scheme with an atom at zero or a negative auxiliary state needs a compatible target representation; log transformation cannot handle those values. Do not silently clip them to a small positive number. If $a=h(u)$ is invertible and the network models $a$, then

$$
q_u(u\mid y,c)=q_a(h(u)\mid y,c)
\left|\det\frac{\partial h}{\partial u}\right|.
$$

Sampling requires applying the inverse transform. Evaluating densities in original coordinates requires its Jacobian. Transforms and training-only normalization belong to the checkpoint contract. Deterministically appending redundant targets, such as both variance and its square root, can create a singular target distribution; compute derived quantities after sampling instead.

## 4 A price is an expectation, not one simulated payoff

For a European or other fixed-payoff contract $c_*$, write its time-$t$ model price as

$$
V(\vartheta^{\mathbb Q},z_t;c_*)
=\mathbb E^{\mathbb Q}_{\vartheta^{\mathbb Q},z_t}
[D_{t,\tau}H(S_{t:\tau};c_*)].
$$

Here $H$ is the payoff, $D$ discounts it, and $\tau$ is maturity. A call has $H=(S_\tau-K)^+$. A barrier payoff depends on the path. An American price additionally optimizes over admissible exercise times, so its reference solver and error checks need an optimal-stopping calculation.

Drawing a future path produces one discounted payoff. Pricing averages over future randomness. Consequently, a posterior distribution of model prices differs from a predictive distribution of discounted payoffs. If $W$ is the uncertain parameter/state vector and $G$ the discounted payoff, then

$$
\operatorname{Var}(G\mid y)
=\mathbb E[\operatorname{Var}(G\mid W)\mid y]
+\operatorname{Var}(V(W)\mid y).
$$

The first term is future payoff variability; the second is uncertainty about its model expectation. This identity assumes finite variances. Plotting the first distribution and calling its width “pricing uncertainty” would mix the two questions.

## 5 Three connected pricing workflows

These workflows share model definitions, contracts, and reference pricing cases. The primary proposed development path is posterior inference plus a supervised pricing surrogate. Direct price-posterior inference is an optional later study.

### A. Posterior draws followed by reference pricing

Infer $W^{(b)}=(\vartheta^{\mathbb Q,(b)},z_t^{(b)})$, then calculate
$V^{(b)}=V_{\rm ref}(W^{(b)};c_*)$.
The collection approximates the posterior distribution induced by the pricing map. This is the most transparent baseline: inspect parameters, preserve dependence, and independently check pricing accuracy. Its cost is a reference-pricer call per draw and contract.

The pricer may be Fourier/PROJ, PDE-based, or Monte Carlo, depending on the model and payoff. There is no reason to replace a reliable fast analytic or Fourier calculation merely to make all models use path simulation.

### B. Posterior draws followed by a supervised pricing surrogate

Train a regression model

$$
\widehat V_\psi(W,c_*)\approx V_{\rm ref}(W;c_*),
\qquad
\min_\psi\sum_i w_i
\big[\widehat V_\psi(W_i,c_i)-V_{{\rm ref},i}\big]^2.
$$

Each label should approximate the conditional pricing expectation with controlled error. A single payoff is not a reference price. Although squared-error regression on unbiased single-payoff labels can theoretically learn their conditional mean, it creates a much noisier learning problem and requires its own validation; that is not the proposed initial bank.

At inference time, reuse each joint posterior draw across contracts and evaluate the surrogate. This keeps the interpretation of workflow A while accelerating valuation. It introduces a separate approximation: test price and, if needed, Greek errors throughout relevant posterior regions. Posterior interval width does not automatically account for surrogate error. Ensure small-price contracts receive sensible absolute/relative tolerances rather than unstable percentage errors near zero.

### C. Learn the induced price posterior directly

Generate $W$ and an observed panel $y$, calculate reference values for held-out target contracts, and train

$$
q_\eta(\mathbf V\mid y,c,\mathcal C_*),
\qquad
\mathbf V=(V(W;c_1),\ldots,V(W;c_m)).
$$

This approximates the posterior distribution obtained by pushing uncertainty in $W$ through the pricing map. No solved parameter posterior is needed to make training labels: joint generative samples supply them. A regression predicting only the posterior mean would be a different objective.

The direct route can focus capacity on prices that matter, but hides parameter interpretation and is tied to its target-contract design. Separate scalar models give marginal distributions, not their dependence. Independently sampling those marginals can yield incoherent surfaces or incorrect portfolio uncertainty. Joint targets, shared latent draws, or a coherent parameter-based construction are needed. A large vector of exact prices may lie on a lower-dimensional manifold, another reason not to assume an unrestricted full-dimensional density is always appropriate.

For all three workflows, distinguish a posterior summary from a tradable quote. With common context and common weights, averaging exact arbitrage-free pricing functions preserves their linear static-arbitrage inequalities. However, pointwise credible bounds or independently sampled price marginals need not form an arbitrage-free surface; approximate networks can also violate restrictions. Posterior uncertainty alone supplies neither execution prices nor a hedging guarantee.

## 6 Forecasting losses, VaR, and expected shortfall

Risk forecasting returns to $\mathbb P$. For a long position of current value $S_T$, define future cumulative log return

$$
R_h=\log(S_{T+h}/S_T),\qquad
L_h=S_T-S_{T+h}=S_T(1-e^{R_h}).
$$

Positive $L_h$ is a loss. The corresponding simple return is $e^{R_h}-1$, so negative log return is not exactly percentage loss. Specify whether results are dollars, fractions of current value, or log losses. “Ten days” must also identify its calendar and elapsed-time convention. More complex portfolios need their own valuation and cash-flow definitions.

This expression is price-only loss for an unfinanced position; it excludes dividends, financing and transaction costs. A total-return or portfolio loss definition must incorporate those cash flows consistently with the observed series and simulation model.

For matched posterior parameter/state draws, the predictive loss law is

$$
p(L_h\mid y,c)=\int p(L_h\mid\vartheta,z_T,c)
p(\vartheta,z_T\mid y,c)\,d\vartheta\,dz_T.
$$

Sample a joint draw, continue from its state, and simulate a future loss. Repeat. Starting every forecast from stationary variance would answer a different question from forecasting after the observed history. Rough models must also retain or reconstruct the relevant memory.

At confidence level $\alpha$, define

$$
\operatorname{VaR}_\alpha(L)=\inf\{\ell:F_L(\ell)\ge\alpha\},
\qquad
\operatorname{ES}_\alpha(L)=\frac1{1-\alpha}
\int_\alpha^1\operatorname{VaR}_u(L)\,du.
$$

For continuous distributions, ES is the mean loss in the upper tail beyond VaR. The quantile-integral definition handles probability mass at the threshold correctly. [Rockafellar and Uryasev, general loss distributions](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)

There are two distinct Bayesian outputs:

1. **Predictive risk:** calculate VaR/ES from the pooled predictive loss law.
2. **Uncertain conditional risk:** for each posterior draw, calculate that draw's conditional VaR/ES, then summarize their posterior distribution.

These operations cannot be interchanged. An exact toy example shows why: two parameter settings have equal posterior probability. Under the first, loss is always 0; under the second, always 100. Their conditional 95% VaRs are 0 and 100, averaging 50. The predictive CDF has $F(0)=0.5$ and $F(100)=1$, so its 95% VaR is **100**, not 50. Its ES is also 100. This deliberately simple example proves the distinction without numerical approximation.

Separating state uncertainty from parameter uncertainty requires declared conditional comparisons. Fixing one parameter vector while retaining the original unconditional state draws generally does not preserve $p(z_T\mid\vartheta,y)$. There is no unique additive decomposition of VaR analogous to the variance identity above, and Bayesian integration does not universally increase every risk measure.

## 7 Numerical error and inference validation

For independent discounted payoff draws $G_1,\ldots,G_M$ with finite second moment, an ordinary Monte Carlo price and its conventional estimated standard error are

$$
\widehat V_M=M^{-1}\sum_m G_m,
\qquad \widehat{\mathrm{SE}}_{\rm MC}=s_G/\sqrt M.
$$

Antithetic pairs, randomized quasi-Monte Carlo, or dependent paths require an error estimator respecting their grouping. Shared paths also correlate errors across contracts.

A finite price does not guarantee finite payoff variance. Rough Bergomi can have infinite second stock-price moments in a true-martingale domain, so naive call-payoff Monte Carlo may not support conventional standard errors. [Gassiat's moment analysis](https://arxiv.org/abs/1811.10935) For the initial European panel, one possible solution is to estimate bounded put payoffs and recover calls through put-call parity, using verified martingale conditions and deterministic discounting. The bounded payoff gives finite sampling variance; numerical discretization error still needs separate assessment. The estimator and supported domain must be reviewed before interpreting error bars.

If a price estimate is conditionally unbiased for a discretized price $V_\delta(W)$, then, across posterior draws and independent inner simulations,

$$
\operatorname{Var}(\widehat V\mid y)
=\operatorname{Var}(V_\delta(W)\mid y)
+\mathbb E[\operatorname{Var}(\widehat V\mid W)\mid y].
$$

Thus noisy inner prices can inflate apparent posterior price uncertainty. Report posterior spread and pricing error separately. Repeat selected prices with larger path counts and finer time grids. More paths reduce Monte Carlo error; they do not remove discretization bias. Posterior sampling error and surrogate error are additional components. Tail-risk estimates need enough effective tail samples and repeated or appropriately resampled calculations, not a price-mean standard-error formula.

Reference inference can use analytic posteriors for simple limits, ordinary MCMC/SMC for evaluable quote likelihoods, and particle MCMC for suitable latent-state return models. Proper PMCMC can target the specified discretized-model posterior despite using finitely many particles, under its algorithmic conditions; finite chain length and poor mixing still matter. The absence of a closed-form likelihood does not make neural inference mathematically necessary. [Andrieu, Doucet, and Holenstein](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)

Validation should combine held-out conditional reference cases, posterior dependence checks, and simulation-based calibration (SBC). SBC repeatedly simulates a truth and observations, infers a posterior, and checks ranks of the generating truth among posterior draws. Marginal checks can miss important failures; data-dependent and joint test quantities improve sensitivity. Passing ranks is not proof of accurate inference on every individual dataset. [Modrák et al., SBC test quantities](https://arxiv.org/abs/2211.02383)

Finally, distinguish two checks: agreement with a reference using the **same** discretization tests inference; comparison against higher-accuracy simulation tests numerical bias. Both methods can agree perfectly on the same inadequate simulator. Real-data adequacy is a further question, assessed through relevant predictive diagnostics and sensitivity to prior and discrepancy assumptions.

## 8 Short checks with worked answers

**The simulator returns true terminal variance. Can the encoder receive it?** Only if the experiment declares that variance directly observed. For a returns-only task it is a training target, excluded from conditioning. A noisy volatility proxy can be an observation, but its generation and measurement error must be specified separately.

**A payoff standard deviation is 20 currency units. How accurate is a price based on 10,000 independent paths?** Its estimated Monte Carlo standard error is approximately $20/\sqrt{10{,}000}=0.20$. Reaching 0.10 at unchanged variance takes 40,000 paths. This calculation describes uncertainty in the computed mean, not a posterior interval for the model price. Time-step bias is unaffected.

**A learned price posterior matches every contract's marginal interval. Is portfolio uncertainty validated?** No. If two contracts have uncertain prices $V_1,V_2$, then $\operatorname{Var}(V_1+V_2)=\operatorname{Var}(V_1)+\operatorname{Var}(V_2)+2\operatorname{Cov}(V_1,V_2)$. Matching marginal intervals leaves their covariance unconstrained. Matched parameter draws in workflows A and B preserve the model-induced connection; workflow C needs a corresponding joint construction and validation.

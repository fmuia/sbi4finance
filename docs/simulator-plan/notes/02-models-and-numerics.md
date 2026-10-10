# Models and numerical simulation

This note specifies the mathematical building blocks proposed for `sbi4finance`. It is a review document, not a claim that the simulators exist. Read [Foundations](01-foundations.md) first for the statistical vocabulary; continue with [Inference and pricing](03-inference-and-pricing.md) for observation models, posteriors and financial outputs.

## 1 Notation units and the meaning of a model

Let $S_t>0$ be an asset price and $X_t=\log S_t$, using a fixed currency unit. An observed log return is $r_i=X_{t_i}-X_{t_{i-1}}$; its corresponding simple return is $e^{r_i}-1$. These are different quantities even though they are close for small moves. Use $\vartheta$ for the complete parameter vector.

Time is expressed in years. The task must define how timestamps become year fractions: a trading-day convention such as $1/252$ and an actual-calendar convention are different choices. Let $v_t$ denote instantaneous variance, so $\sqrt{v_t}$ is volatility. Brownian motion $W_t$ has independent Gaussian increments with variance equal to elapsed time:

$$
W_{t+h}-W_t=\sqrt h\,Z,\qquad Z\sim\mathcal N(0,1).
$$

Thus a volatility contribution $\sqrt{v_t}\,\sqrt h\,Z$ is a dimensionless return. Annualized 20% volatility means $\sqrt v=0.20$, hence $v=0.04$, not $v=0.20$.

| Symbol | Meaning | Time dimension |
|---|---|---|
| $\mu,r,q$ | Expected proportional price-growth rate, interest rate, dividend yield | Year$^{-1}$ |
| $v,\bar v$ | Instantaneous and long-run variance | Year$^{-1}$ |
| $\kappa,\lambda$ | Mean-reversion speed and jump intensity | Year$^{-1}$ |
| $\sigma_v$ | Heston variance diffusion coefficient | Year$^{-1}$ |
| $J,\rho$ | Log-jump size and correlation | Dimensionless |

The probability measure is part of the specification. Under the physical measure $P$, $\mu$ describes expected price growth. Under a pricing measure $Q$, the corresponding rate is $r-q$. This does not imply all other parameters are equal under $P$ and $Q$. A model definition, an initialization law and a numerical scheme together determine the distribution actually used to generate a bank.

## 2 GBM an exact first building block

Geometric Brownian motion (GBM) assumes constant variance:

$$
\frac{dS_t}{S_t}=\mu\,dt+\sqrt v\,dW_t,
\qquad
dX_t=(\mu-\tfrac12v)dt+\sqrt v\,dW_t.
$$

The subtraction of half the variance is the Itô correction. It ensures the exponential of the log-price process has the intended proportional drift. On any observation interval $h$, simulate exactly:

$$
r_i=(\mu-\tfrac12v)h+\sqrt{vh}\,Z_i.
$$

There is no time-discretization error in these increments. For a deterministic variance curve $v(t)$, replace $vh$ by $A=\int_t^{t+h}v(u)du$, and replace $\mu h$ by the integrated drift. Numerical quadrature of $A$, if required, is then a separate approximation.

**Worked example.** With $\mu=0.08$, $v=0.04$, and $h=1/252$, the daily log-return mean is $0.06/252\approx0.000238$, or 0.0238%; its standard deviation is $0.20/\sqrt{252}\approx1.26\%$. The expected simple return is $e^{0.08/252}-1\approx0.0318\%$. A daily observation need not be a daily numerical step in more complicated models.

GBM supplies exact return laws and analytic European-option references. It is therefore useful for checking units, priors, observation transformations and the complete bank-to-training pipeline before hidden states are introduced.

## 3 Heston hidden mean-reverting variance

The Heston backbone replaces constant variance by a square-root diffusion:

$$
\begin{aligned}
dX_t&=(\mu-\tfrac12v_t)dt+\sqrt{v_t}\,dW^S_t,\\
dv_t&=\kappa(\bar v-v_t)dt+\sigma_v\sqrt{v_t}\,dW^v_t,\\
d\langle W^S,W^v\rangle_t&=\rho\,dt.
\end{aligned}
$$

Here $\kappa>0$ controls the speed of return toward $\bar v>0$; $\sigma_v>0$ controls random variance fluctuations. Negative $\rho$ couples negative price shocks with positive variance shocks. The original model provides a characteristic-function route to European pricing. [Heston (1993)](https://doi.org/10.1093/rfs/6.2.327)

Conditional on $v_t$, the expected future variance is

$$
E[v_{t+h}\mid v_t]=\bar v+(v_t-\bar v)e^{-\kappa h}.
$$

This follows by taking expectations in the variance equation and solving the resulting linear ordinary differential equation. Its half-life is $\log(2)/\kappa$. Returns reveal variance indirectly through their changing dispersion; observing a return is not observing $v_t$.

The task must declare $v_0$: fixed context, an inferred unknown, a random draw, or the result of burn-in. One possible random initialization is the continuous-time stationary law

$$
v_0\sim\operatorname{Gamma}
\left(\text{shape}=\frac{2\kappa\bar v}{\sigma_v^2},
\ \text{scale}=\frac{\sigma_v^2}{2\kappa}\right).
$$

This is conditional on the sampled parameters. It does not imply the observed market is stationary, nor that a discretized transition has exactly this stationary distribution.

For positive parameters and positive initial variance, the Feller condition $2\kappa\bar v\geq\sigma_v^2$ prevents the continuous-time variance from reaching zero. Its failure does not make the model invalid: variance remains nonnegative but can reach the boundary. Restricting every prior to the Feller region would therefore be a scientific restriction, not a universal validity requirement.

## 4 A discrete scheme is an explicit approximation

Naive Euler simulation can produce negative variance. A transparent baseline is full truncation. Write $a^+=\max(a,0)$, retain an auxiliary value $\widetilde v_n$, and use

$$
\begin{aligned}
\widetilde v_{n+1}
&=\widetilde v_n+\kappa(\bar v-\widetilde v_n^+)h
 +\sigma_v\sqrt{\widetilde v_n^+h}\,Z^v_n,\\
X_{n+1}
&=X_n+(\mu-\tfrac12\widetilde v_n^+)h
 +\sqrt{\widetilde v_n^+h}\,Z^S_n,
\end{aligned}
$$

where the standard normals have correlation $\rho$. Effective variance is $\widetilde v_n^+$; the auxiliary value can be negative. Clipping and overwriting the auxiliary value after every update defines a different scheme. [Full-truncation study](https://repub.eur.nl/pub/18571)

Four internal steps per day produce one observed daily return by summing four log increments. Increasing those steps changes numerical fidelity, not the sampling frequency. The bank must record the scheme, step grid, effective-state convention and precision. If a forecast continues this particular discrete scheme, it must also retain or infer its auxiliary state; replacing a negative auxiliary value by zero changes its next-step law.

Andersen's quadratic-exponential methods are candidates for a more efficient accuracy tradeoff, to be benchmarked against the baseline. Exact sampling of the variance endpoint alone does not give an exact joint price/variance transition: integrated variance and price/variance dependence also matter. [Andersen's simulation methods](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405)

A checkpoint trained on a particular daily discrete model approximates that model's posterior. Replacing its generator by a finer continuous-time approximation requires validation and potentially retraining, even when the replacement is scientifically preferable.

Two kinds of convergence answer different questions. *Strong* convergence asks whether coarse and fine paths stay close when driven by appropriately coupled randomness. *Weak* convergence asks whether expectations of specified functions, such as an option payoff, agree. Matching a European call price does not establish correct short-interval return distributions or terminal-state dependence. Conversely, one visually similar path is not evidence that tail probabilities are accurate. Reference tests should therefore include the actual observations, latent targets and downstream quantities the task will use. Numerical validation belongs to the declared parameter domain: a scheme that works near one fitted parameter vector may fail in a difficult region of the training prior.

## 5 Jumps and drift compensation

Let $N$ be a homogeneous Poisson process of intensity $\lambda$, so its independent increments over intervals of length $h$ have distribution $\operatorname{Poisson}(\lambda h)$. Let $J_1,J_2,\ldots$ be independent, identically distributed log-jump sizes, independent of the Brownian drivers and of the arrival process. A jump multiplies the price by $e^J$; its proportional change is $e^J-1$. Define

$$
c_J=E[e^J-1].
$$

With $\mu$ defined as the total instantaneous expected proportional growth rate, the compensated model is

$$
\frac{dS_t}{S_{t-}}
=(\mu-\lambda c_J)dt+\sqrt{v_t}\,dW^S_t
 +d\!\left(\sum_{k=1}^{N_t}(e^{J_k}-1)\right),
$$

or, in log prices,

$$
dX_t=(\mu-\lambda c_J-\tfrac12v_t)dt
 +\sqrt{v_t}\,dW^S_t+d\!\left(\sum_{k=1}^{N_t}J_k\right).
$$

The jump contribution has expected proportional rate $\lambda c_J$, which cancels the compensator in the drift. Using $E[J]$ there would be incorrect. Some publications instead call the *uncompensated continuous drift* $\mu$; parameter import must resolve that convention explicitly.

Under $Q$, substitute $\mu=r-q$, using the $Q$-jump intensity and jump-size law. Under $P$, estimate or fix $\mu$ according to the returns task. Matching drift formulas alone does not identify the transformation between these parameter sets.

For an interval $h$, draw $n\sim\operatorname{Poisson}(\lambda h)$ and sum $n$ jumps. A Bernoulli draw that permits at most one jump is an approximation. With two jumps per year and $h=1/252$, the expected count is $2/252$, while the probability of at least one is $1-e^{-2/252}$; they are close, not identical. Event timing also matters for path-dependent payoffs.

## 6 Gaussian jumps and HKDE are sibling models

For Gaussian log jumps,

$$
J\sim\mathcal N(m_J,s_J^2),\qquad
c_J=\exp(m_J+\tfrac12s_J^2)-1.
$$

The second expression follows from the normal moment-generating function. Conditional on $n$ arrivals, the aggregate log jump is normal with mean $nm_J$ and variance $ns_J^2$, including a deterministic zero when $n=0$. This gives a convenient exact aggregate-jump sampler. Gaussian *log* jumps imply lognormal price multipliers, not Gaussian proportional price jumps.

For HKDE, use the Heston variance process and the asymmetric double-exponential density

$$
f_J(j)=p_+a_+e^{-a_+j}\mathbf1_{j\geq0}
 +(1-p_+)a_-e^{a_-j}\mathbf1_{j<0},
$$

where $0\leq p_+\leq1$ and $a_+,a_->0$. Select the upward branch with probability $p_+$, then draw an exponential magnitude; otherwise negate an exponential draw with rate $a_-$. These are **rates**, so mean magnitudes are their reciprocals. [Kou and Wang, model definition](https://www.columbia.edu/~sk75/KouWangMS.pdf)

Direct integration gives

$$
E[e^{uJ}]=\frac{p_+a_+}{a_+-u}
 +\frac{(1-p_+)a_-}{a_-+u},
\qquad -a_-<u<a_+,
$$

when both mixture components are present. Therefore

$$
c_J=\frac{p_+a_+}{a_+-1}
 +\frac{(1-p_+)a_-}{a_-+1}-1.
$$

If $p_+>0$, the compensator requires $a_+>1$; boundary components that have zero probability are omitted. Higher price moments require stronger jump-moment conditions and may also be limited by stochastic-volatility dynamics. A finite log-jump variance is not enough to guarantee finite price moments.

The HKDE study couples this jump law to Heston and uses Fourier/PROJ pricing for calibration. That makes HKDE a useful alternative jump specification, not a mandatory stage beyond Gaussian-jump SVJ. [HKDE paper](https://arxiv.org/html/2502.13824v1)

SVCJ adds a specified joint mechanism for price and variance jumps. It is deferred: synchronization, size dependence and variance positivity must be defined before implementation. “SVJ” alone does not settle those choices. [Eraker, Johannes and Polson](https://onlinelibrary.wiley.com/doi/10.1111/1540-6261.00566)

## 7 Bergomi constructing positive variance from Gaussian drivers

Instead of a square-root diffusion, construct variance from a centered Gaussian process $G_t$:

$$
v_t=\xi_0(t)\exp\!\left(\eta G_t
 -\tfrac12\eta^2 C(t,t)\right),
\qquad C(t,u)=\operatorname{Cov}(G_t,G_u).
$$

The positive function $\xi_0(t)$ specifies initial mean variance. Because $E[e^{aG}]=e^{a^2\operatorname{Var}(G)/2}$ for centered Gaussian $G$, the correction ensures $E[v_t]=\xi_0(t)$. This normalization must use the covariance of the driver actually simulated.

A useful representation is

$$
G_t=\int_0^t K(t-s)\,dW^v_s,
\qquad
C(t,u)=\int_0^{\min(t,u)}K(t-s)K(u-s)ds.
$$

The kernel $K$ specifies how past shocks influence present variance. The price obeys $dX_t=(r-q-\tfrac12v_t)dt+\sqrt{v_t}\,dW^S_t$ under the standard $Q$ formulation, with declared correlation between the price and variance Brownian drivers. Bergomi models describe a forward-variance curve, rather than only today's variance. [Bergomi's original model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302)

For maturity $u\geq t$, conditioning this construction gives

$$
\xi_t(u)=E[v_u\mid\mathcal F_t]
=\xi_0(u)\exp\!\left(
\eta\int_0^tK(u-s)dW^v_s
-\tfrac12\eta^2\int_0^tK(u-s)^2ds\right).
$$

This follows by splitting the Gaussian integral into known past and independent future contributions. The future exponential has conditional mean one. The resulting curve will explain the continuation-state requirement below.

## 8 One and two exponential factors

For $K(t-s)=e^{-k(t-s)}$ with $k\geq0$, define the Ornstein–Uhlenbeck factor starting at $Z_0=0$:

$$
Z_t=\int_0^t e^{-k(t-s)}dW^v_s,
\qquad dZ_t=-kZ_tdt+dW^v_t.
$$

Its variance is $A_k(t)=(1-e^{-2kt})/(2k)$, interpreted as $t$ when $k=0$. Hence $G_t=Z_t$ gives a one-factor Bergomi model. Its OU transition is exactly Gaussian, although the coupled price evolution still needs integration.

One explicit two-factor convention is

$$
G_t=c[(1-w)Z^1_t+wZ^2_t],\quad 0\leq w\leq1,
\quad c^{-2}=(1-w)^2+w^2+2\rho_{12}w(1-w),
$$

with $dZ^i_t=-k_iZ^i_tdt+dW^i_t$, $k_i\geq0$, $Z^i_0=0$, and Brownian correlation $\rho_{12}$. Require the denominator to be positive. Its covariance ingredients are

$$
C_{ij}(t)=\rho_{ij}
\frac{1-e^{-(k_i+k_j)t}}{k_i+k_j},\qquad\rho_{ii}=1,
$$

using the continuous limit when the denominator is zero. Therefore

$$
\operatorname{Var}(G_t)=c^2[(1-w)^2C_{11}(t)
+w^2C_{22}(t)+2w(1-w)C_{12}(t)].
$$

The constant $c$ normalizes the instantaneous diffusion variance of the weighted factor, not its variance at every time. Other conventions are possible; importing parameters requires matching them. These formulas follow by integrating the OU kernels and give a concrete convention for review, not a mandated final parameterization.

All Brownian correlations must jointly form a positive-semidefinite matrix:

$$
\begin{pmatrix}
1&\rho_{S1}&\rho_{S2}\\
\rho_{S1}&1&\rho_{12}\\
\rho_{S2}&\rho_{12}&1
\end{pmatrix}\succeq0.
$$

Individually valid numbers in $[-1,1]$ do not guarantee this. Exact OU innovations must also have the covariance obtained from their integrated kernels; simply imposing the instantaneous Brownian correlations on differently damped innovations is generally wrong.

## 9 Rough Bergomi and its normalization

Choose the singular power kernel

$$
K_H(t-s)=\sqrt{2H}(t-s)^{H-1/2},\qquad0<H<1/2.
$$

Then direct integration gives $\operatorname{Var}(G_t)=t^{2H}$, and

$$
v_t=\xi_0(t)\exp\!\left(\eta G_t-\tfrac12\eta^2t^{2H}\right).
$$

Small $H$ makes variance paths locally less smooth. This driver is a Gaussian Volterra process starting at zero, often called Riemann–Liouville fractional Brownian motion; it does not have all the stationary-increment properties of standard fractional Brownian motion. Authors sometimes omit $\sqrt{2H}$ and absorb it into $\eta$, so numerical parameter values are not portable without checking normalization. [Original rough-volatility pricing paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754)

With physical time dimensions retained, $G_t$ has units year$^{H}$, so $\eta$ has units year$^{-H}$. In the OU convention, $\eta$ instead has units year$^{-1/2}$. Both are conventionally reported as numbers with time measured in years. Changing time units without adjusting coefficients changes the model.

At $H=1/2$, the kernel becomes constant. This is the zero-mean-reversion one-factor limit, not arbitrary mean-reverting Bergomi. Rough Heston is another model family and is outside this initial ladder.

## 10 Memory initialization and continuation

For the exponential kernel, the past contribution at a future maturity is $e^{-k(u-t)}Z_t$. Thus one OU factor, the curve specification, parameters and time determine future variance dynamics. Two-factor Bergomi needs both factors.

For the rough kernel, the corresponding past contribution is the function

$$
u\longmapsto\int_0^t K_H(u-s)dW^v_s,\qquad u\geq t.
$$

Its value at $u=t$ cannot recover all its future values. Consequently, knowing $v_t$ is generally insufficient for conditional forecasting. The complete conditional forward-variance curve, or an adequate memory representation for the numerical scheme, is needed. [Forward-variance representations](https://arxiv.org/html/2202.10413v2)

Starting the Gaussian driver at zero is an initialization assumption. Restarting it at zero whenever a historical window begins discards pre-window memory. A returns task must specify prehistory, a distribution over factor/memory states, or a conditional reconstruction procedure. True simulated memory stored as a training label does not make it observable at inference time.

A separately defined $P$-analogue can use these kernels to describe returns, but its $\xi_0$ is then a physical expected-variance profile. An observed risk-neutral variance curve cannot simply be substituted for it. Defining two analogous equations is not a proof of an equivalent measure change.

## 11 Structural complexity versus numerical fidelity

A sum of exponentials,

$$
K_H(s)\approx K^{(m)}(s)=\sum_{j=1}^m a_j e^{-k_js},
$$

turns the rough driver into a finite collection of OU factors driven by the same Brownian motion. Their innovations are correlated, not independent. If weights and rates approximate a chosen rough kernel over a declared horizon, $m$ and the approximation tolerance are numerical settings. If factors are independently parameterized and estimated as the scientific model, they are structural choices. [Markovian approximation study](https://arxiv.org/abs/2007.02113)

For a direct rough implementation, a small-grid Gaussian covariance sampler is a useful reference. Sampling that Gaussian vector exactly does not make the discretized price integral exact. A hybrid scheme handles the kernel singularity near zero specially and approximates the remaining convolution; convolution strategy and grid refinement must be recorded. [Hybrid simulation method](https://arxiv.org/abs/1507.03004)

The exponential normalization must match the chosen approximate driver covariance. Using the exact $t^{2H}$ correction with a mismatched approximate variance need not preserve $E[v_t]=\xi_0(t)$.

## 12 Reference pricing with characteristic functions

A characteristic function encodes a probability distribution through complex-valued expectations. For the cumulative log return under $Q$, define

$$
\varphi_R(u;T)=E^Q\!\left[e^{iu\log(S_T/S_0)}\right],
\qquad u\in\mathbb R,\quad i^2=-1.
$$

For Heston with constant coefficients and initial variance $v_0$, it has the affine form

$$
\varphi_R(u;T)=\exp\!\left[C(T,u)+D(T,u)v_0\right].
$$

Substituting an exponential function of log price and variance into the model's backward generator, and matching the constant and variance coefficients, gives

$$
\begin{aligned}
\partial_T D&=\tfrac12\sigma_v^2D^2
 +(iu\rho\sigma_v-\kappa)D-\tfrac12(u^2+iu),\\
\partial_T C&=iu(r-q)+\kappa\bar vD,
\qquad C(0,u)=D(0,u)=0.
\end{aligned}
$$

These Riccati equations specify the characteristic function without committing to a particular complex-square-root implementation. All coefficients here are pricing-measure parameters. [Heston's characteristic-function construction](https://wwwf.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf)

Independent compensated price jumps multiply the diffusion characteristic function by

$$
\exp\!\left\{\lambda T[\varphi_J(u)-1-iu c_J]\right\},
\qquad
\varphi_J(u)=
\begin{cases}
\exp(iu m_J-\tfrac12s_J^2u^2),&\text{Gaussian log jumps},\\
\dfrac{p_+a_+}{a_+-iu}+
\dfrac{(1-p_+)a_-}{a_-+iu},&\text{double exponential}.
\end{cases}
$$

This follows by conditioning on the Poisson count; the final term accounts for the drift compensation derived above.

If $f_T(x)$ is the log-return density and $g$ a European payoff, its price is

$$
V=e^{-rT}\int_{\mathbb R}g(S_0e^x)f_T(x)\,dx.
$$

Fourier inversion relates $f_T$ to $\varphi_R$. Fourier pricing evaluates this expectation using the characteristic function; PROJ approximates the density in a basis and integrates that approximation against the payoff. It is a pricing algorithm, not a different financial model. [PROJ in the HKDE study](https://arxiv.org/html/2502.13824v1#S3.SS1)

Basic checks are $\varphi_R(0;T)=1$ and $\varphi_R(-u;T)=\overline{\varphi_R(u;T)}$ for real $u$. Where the first exponential moment exists and the discounted total-return process is a true martingale, $\varphi_R(-i;T)=e^{(r-q)T}$. This last expression uses a complex argument and tests compensation, not just normalization. Domain truncation, grid resolution and integration tolerances still create numerical error. Refining those settings should converge toward the same model price; agreement with an independently refined path pricer provides another check.

## 13 Connecting these models to inference and validation

| Building block | Initial task | Required state or reference |
|---|---|---|
| GBM | Parameters from return windows | Exact return distribution; analytic European prices |
| Heston and price-jump variants | Parameters and hidden variance from returns | Joint parameter/state posterior; declared discrete continuation state |
| HKDE under $Q$ | Parameters from option panels | Validated characteristic-function/PROJ prices |
| Finite-factor Bergomi | Selected parameters from returns or quotes | Full factor state for continuation; independently checked prices |
| Rough Bergomi | Parameter or curve calibration; later conditional forecasting | Memory treatment, Gaussian references and numerical refinement |

A reference pricer tests the map from parameters to prices. It is not automatically a likelihood for latent-volatility return histories. Conversely, simulating paths does not by itself define observed quote errors. The next note makes those observation contracts explicit.

Validation should check Gaussian covariances, variance means, jump counts, compensation, analytic limits and refinement across the prior. Martingale tests need mathematically appropriate domains: in standard rough Bergomi, the true-martingale property depends on price/volatility correlation. Merely writing $r-q$ in the drift does not settle it, and high price moments may be infinite. [Gassiat's martingale analysis](https://arxiv.org/abs/1811.10935)

Compiler acceleration does not resolve these mathematical questions. First identify the law, state and acceptable numerical error; then benchmark kernels and storage against those requirements.

# HKDE: double-exponential jumps and European reference prices

This milestone supplies a Heston–Kou simulator and a continuous-model European reference pricer. It creates numerical building blocks for synthetic option observations, not a neural posterior or a market calibration. Its model identifier is `heston_kou_double_exponential_v1`. The current Heston posterior's unfinished accuracy work does not enter these kernels or their numerical checks.

## What model is being simulated?

Write variance as \(v_t\), so its square root is the price diffusion coefficient. Under a caller-declared measure, the dynamics are

\[
\frac{dS_t}{S_{t-}}=(b-\lambda\zeta)dt+\sqrt{v_t}\,dW_t^S+(e^J-1)dN_t,
\qquad dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW_t^v,
\]

with \(d\langle W^S,W^v\rangle_t=\rho\,dt\). The independent compound Poisson jumps have intensity \(\lambda\) per year and log-jump density

\[
f_J(j)=p\eta_+e^{-\eta_+j}\mathbf1_{j\ge0}
 +(1-p)\eta_-e^{\eta_-j}\mathbf1_{j<0}.
\]

Thus an upward jump has mean log size \(1/\eta_+\), and a downward jump has mean log size \(-1/\eta_-\). Rates concern decimal log sizes, not percentage price changes. The compensator is

\[
\zeta=E[e^J]-1=\frac{p}{\eta_+-1}-\frac{1-p}{\eta_-+1}.
\]

An active upward tail requires \(\eta_+>1\) for finite expected price. A zero intensity or zero upward probability removes that restriction on the unused tail; its rate must still be positive. `jump_compensation` returns **\(\lambda\zeta\)**. Supplying `drift=b` means total expected proportional price growth, matching the existing Heston API. Under Q, set \(b=r-q\). Setting `drift=0` and initial price one gives a forward-normalized path, suitable for comparing the terminal multiplier against the Fourier model. Parameters under P and Q are separate; no change of measure is inferred.

This is the Heston plus Kou model family considered by [Agazzotti et al.](https://arxiv.org/html/2502.13824v1). The explicit compensated jump convention follows [Kou and Wang, equations 1–3](https://www.columbia.edu/~sk75/KouWangMS.pdf). We use consistent variance notation above rather than relying on the paper's occasionally ambiguous variance/volatility symbols.

## Simulation and continuation

The Heston diffusion reuses the existing NumPy/Numba full-truncation implementation. Its stored variance state is the **signed auxiliary state**, not a clipped variance. It must be preserved when restarting a path. Effective variance in drift and diffusion coefficients is its positive part.

Poisson thinning gives independent counts over each observation interval:

\[
N_+\sim\operatorname{Poisson}(\lambda p\Delta t),\quad
N_-\sim\operatorname{Poisson}(\lambda(1-p)\Delta t).
\]

The corresponding jump magnitude sums are independent gamma variables with shapes \(N_+,N_-\) and scales \(1/\eta_+,1/\eta_-\), with zero sum for a zero count. This is exact for interval jump totals, including multiple jumps. The diffusion still has time-discretization error. Individual jump times, intrainterval extrema, barrier crossings and American exercise are not represented by this aggregation.

```python
from sbi4finance import hkde

parameters = dict(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.6,
                  jump_intensity=0.8, jump_up_probability=0.35,
                  jump_up_rate=12.0, jump_down_rate=8.0, drift=0.0)
noise = hkde.draw_innovations([0.01] * 50, 4, seed=71,
                             example_ids=[0, 1], parameters=parameters)
paths = hkde.simulate(parameters, [0.01] * 50, 0.04, noise)
```

`log_returns` has shape `[batch,time,1]`; `variance_state` has shape `[batch,time+1,1]`, including the starting state. Brownian inputs are `[batch,time,substeps,2]`, and up/down counts and positive gamma sums are `[batch,time]`. Per-example seeded streams preserve results when reordering or partitioning a batch. The complete variance trajectory, counts and jump totals are simulator truth/diagnostics; they must not enter observed conditioning by accident. Slicing prescribed innovations permits a reproducible segmented-continuation check. A fresh future simulation should use a distinct stream identifier.

## Why a separate reference pricer?

An option price is a **discounted expectation over many possible futures**, not one simulated payoff. To train on synthetic option panels, we first draw Q parameters and context, calculate accurate model expectations, and then add the task's intentional quote noise. The inference network can subsequently learn noisy panel → parameter posterior. Posterior draws can be propagated through the same pricer to obtain price uncertainty for another contract. A pricing surrogate instead learns parameters and contracts → expectation labels. These are distinct tasks.

For deterministic financing, let \(F\) be the forward, \(D\) the discount factor, \(K\) the strike and \(T\) maturity. The characteristic function concerns \(X_T=\log(S_T/F)\):

\[
\phi(u)=\phi_{\rm Heston}(u)\exp\{T[\lambda(\chi_J(u)-1)-iu\lambda\zeta]\},
\qquad \chi_J(u)=\frac{p\eta_+}{\eta_+-iu}+\frac{(1-p)\eta_-}{\eta_-+iu}.
\]

It has \(\phi(0)=\phi(-i)=1\). The Heston factor uses the decaying-exponential Riccati solution, with the deterministic-variance limit handled explicitly when \(\xi=0\). The formula and branch convention are documented in [Forde and Jacquier, equation 5](https://martinforde.github.io/HestonSmallTime.pdf); our validation also solves the underlying Riccati differential equations independently.

The implemented half-shift inversion is

\[
C=D\left[F-\frac{\sqrt{FK}}{\pi}
\int_0^\infty\frac{\Re[e^{iu\log(F/K)}\phi(u-i/2)]}{u^2+1/4}\,du\right].
\]

The put follows from \(C-P=D(F-K)\). This integral follows from Fourier inversion of the terminal payoff; it only requires the half moment already available under the finite first-moment restriction. The current implementation is a direct Fourier quadrature, **not PROJ**, and does not reproduce the paper's barrier pricer. Its validation includes a separate probability-based payoff inversion, analytic limits and path Monte Carlo.

```python
clean_prices = hkde.european_price(
    forward=100.0, strike=[80.0, 100.0, 120.0], discount=0.98,
    maturity=0.5, parameters=parameters, initial_variance=0.04,
    nodes=512, upper_limit=200.0,
)
```

Contracts broadcast; one scalar model parameter set describes a panel. The `drift` field is ignored by the forward-normalized pricer, since carry is already supplied through the forward. `kind="put"` prices puts. `nodes` and `upper_limit` are quadrature controls, not certified error estimates. Material no-arbitrage bound violations raise an error rather than being clipped or silently redrawn.

## Bounded numerical evidence

Run the independent report with:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_hkde.py -v
PYTHONPATH=src python scripts/validate_hkde.py --out /tmp/hkde-validation-new.json
```

The [committed report](results/hkde-validation.json) records its source hashes, seed, controls, budgets, timings and per-contract results. Its scope is eight specified parameter fixtures, three maturities (0.125, 0.5 and 1 year) and three strike/forward ratios (0.8, 1 and 1.2). Fixtures include low variance with large vol-of-vol, one-sided jumps, zero jumps, deterministic variance and the Black–Scholes limit. Monte Carlo adds two complete model cases, with three correlated payoffs per simulated path.

The recorded run passed all eight fixtures and both Monte Carlo cases in 4.34 seconds. Its largest cutoff/node discrepancy was **5.18e-8 of DF**; the alternative payoff inversion differed by at most **8.95e-13**, and the independent Riccati calculation by **4.22e-15**. The six kernel/pricer unit tests also passed. These are the measured bounds on the declared comparisons, not a uniform bound on model bias.

The proposed synthetic pilot box uses \(\kappa\in[1,3]\), \(\theta,v_0\in[0.025,0.065]\), \(\xi\in[0.15,0.45]\), \(\rho\in[-0.8,-0.3]\), \(\lambda\in[0.1,1.2]\), \(p\in[0.2,0.6]\), \(\eta_+\in[8,20]\), \(\eta_-\in[6,15]\). These are deliberately modest tutorial bounds, not fitted market ranges. Fixture coverage is not a guarantee everywhere inside this box. The panel generator must also retain a refinement discrepancy for **each** generated parameter draw and reject/fail unsupported numerical cases without resampling them into a biased prior.

Predeclared comparisons use the following separate budgets:

- Fourier cutoff/node refinement and alternative inversion: absolute error \(2\times10^{-6}\) in units of \(DF\). Controls are 512/200, 1024/400 and 2048/800 (nodes/cutoff).
- Independent characteristic-function Riccati check: \(10^{-8}\) absolute, with 2048/4096 ODE steps compared.
- Path Monte Carlo versus Fourier and paired 128/512-step refinement: \(3\times10^{-4}\) of \(DF\), plus five reported Monte Carlo standard errors. This is a bounded diagnostic allowance, not a confidence bound on the unknown diffusion bias.

Payoffs at different strikes share paths. Their mean covariance is retained; treating these price errors as independent quote noise would be incorrect. The martingale mean and its standard error are recorded separately. Intentional quote noise belongs to the observation task, not these numerical checks.

The mathematical validator admits more parameters than the pilot numerically investigates. Very short maturities, extreme moneyness, near-divergent upward tails, positive correlation, very small mean reversion and much larger vol-of-vol require new refinement/reference checks. The implementation provides no American/barrier accuracy, no neural calibration guarantee, no empirical model adequacy claim, and no inferred physical risk probabilities from Q prices.

# One-factor Bergomi: the forward-variance building block

This implements M6's required one-factor kernel and a reusable European option
panel estimator. It is a **synthetic Q simulation and numerical-pricing core**.
The bank records approximate price labels and numerical uncertainty; these are
not observations of tomorrow's market prices or an accepted neural posterior.
Two-factor Bergomi is optional and is not implemented.

## What is being modeled?

An asset price is the traded quantity. Its instantaneous variance is hidden and
changes over time. The **initial forward-variance curve** \(\xi_0(t)\) describes
the expected future instantaneous variance at each future time, under the
risk-neutral pricing measure Q. It is a known input here, not inferred from a
real market feed. For example, the flat curve 0.04 corresponds to 20% annualized
volatility, since volatility is the square root of variance.

We specify one Gaussian factor, initialized at zero at the original origin:

\[
dX_t=-\kappa X_t\,dt+dW_t,\qquad
q(t)=\operatorname{Var}(X_t)=\frac{1-e^{-2\kappa t}}{2\kappa},
\]
\[
v_t=\xi_0(t)\exp\{\eta X_t-\tfrac12\eta^2q(t)\},\qquad
d\log S_t=(r-q_d-\tfrac12v_t)dt+
\sqrt{v_t}\{\rho\,dW_t+\sqrt{1-\rho^2}\,dW_t^\perp\}.
\]

Here \(q_d\) is the dividend yield, distinct from the factor variance \(q(t)\).
The mean reversion \(\kappa>0\) controls how quickly a shock fades; \(\eta\ge0\)
controls variance fluctuations; \(\rho\) couples asset and volatility shocks.
The subtraction in the exponential ensures \(E_Q[v_t]=\xi_0(t)\), by the Gaussian
exponential-moment formula. The curve is positive, piecewise linear in time,
starts at time zero, and must cover the entire requested horizon. No
extrapolation is silently performed. Rates and dividend yields are constant in
this bounded implementation; all times are in years.

## Exact factors, approximate asset paths

Over one interval of length \(h\),

\[
X_{t+h}=e^{-\kappa h}X_t+I_h,\quad
\operatorname{Var}(I_h)=\frac{1-e^{-2\kappa h}}{2\kappa},\quad
\operatorname{Cov}(I_h,\Delta W)=\frac{1-e^{-\kappa h}}{\kappa}.
\]

Using the same standard normal for the entire factor increment and asset
increment would make their correlation one; that is incorrect for finite
intervals. We construct the exact joint Gaussian pair from an interval Brownian
normal and an independent bridge normal. A third independent normal drives
\(W^\perp\). The asset uses variance at the **start** of each interval. That
adapted log-Euler step preserves the specified discrete conditional mean, but
introduces time-integration error. Exact Gaussian stepping does not remove that
error from option prices.

`simulate_one_factor(parameters, curve, dt_years, innovations, ...)` accepts
prescribed independent normals `[paths, steps, 3]` so numerical comparisons are
reproducible. It returns decimal `log_returns [paths,steps,1]`, `factor_state
[paths,steps+1,1]`, variance, log spot, integrated variance and the correlated
Brownian integral. All state paths are **latent simulator truth**, not observed
inference inputs.

A restart requires the factor state, original curve, elapsed time and current
asset price. Keeping the original normalization \(q(t)\) and evaluating the
original curve at the correct future time is essential. Tests split a path into
two pieces and reproduce uninterrupted returns under the same prescribed
innovations. A common scalar spot is accepted; batched continuation can use
spot one and add each path's stored log spot afterwards. This is a Q continuation
test, not an inference procedure reconstructing state from real returns.

## Why the price estimator averages conditional puts

A price is a discounted expected payoff, not one simulated payoff. For each
volatility-driver path we compute

\[
A_T=\sum_i v_i\Delta t_i,\qquad
J_T=\sum_i\sqrt{v_i}\Delta W_i.
\]

Conditionally on that path, the independent asset Brownian motion can be
integrated out. The terminal log asset price is Gaussian with mean
\(\log S_0+(r-q_d)T-\tfrac12A_T+\rho J_T\) and variance
\((1-\rho^2)A_T\). Its conditional put expectation follows the usual lognormal
option formula. Averaging these conditional expectations is a
Rao–Blackwell variance reduction: it removes independent asset noise without
changing the discrete-model price.

Every discounted conditional put is between zero and \(Ke^{-rT}\). It therefore
has a finite variance, so iid Monte Carlo standard errors are justified.
Contracts share simulated paths; `estimator_covariance` is the covariance of the
**panel's estimated means**, and its off-diagonal entries must be retained.
Calls use \(C=P+S_0e^{-q_dT}-Ke^{-rT}\). The public API restricts leverage to
\(-1\le\rho\le0\), a conservative common martingale-domain convention for this
and the rough branch. It never relies on a sample second moment of raw call
payoffs as evidence of finite population variance. That concern is especially
concrete in rough Bergomi; see [the rough-model plan](simulator-plan/milestones/07-rough-bergomi.md).

`option_panel("one_factor", parameters, curve, contracts, ...)` uses a uniform
asset grid ending at the largest maturity. Every contract maturity must land
on a grid node. Contract order is preserved. Seeds and configurations determine
paths independently of the memory chunk size. This core is NumPy/vectorized;
the measured bounded workload did not require a new Numba kernel.

Three uncertainties remain distinct:

1. Monte Carlo sampling error, reported by standard errors and panel covariance.
2. Factor/asset discretization error, examined by changing the grid; it is **not**
   included in the Monte Carlo error bars.
3. Observation or quote error, supplied separately when constructing a noisy
   simulated option observation; it is **not** added by the pricer.

## Measured bounded checks

Run from the repository root:

```sh
PYTHONPATH=src pixi run python -m unittest discover -s tests -p test_bergomi.py -v
PYTHONPATH=src pixi run python scripts/validate_bergomi.py
```

The [saved report](results/bergomi-validation.json) records source hashes, seeds,
full panel covariance, curve and contracts. Three synthetic scenarios cover
\((\kappa,\eta,\rho)=(2,1,-0.6),(0.5,0.5,-0.2),(4,1.5,-0.9)\). Each uses 32,768
paths and grids of 32, 64 and 128 steps through six months. There are six puts:
strikes 0.9, 1.0 and 1.1 at three and six months, with initial spot one.

The frozen acceptance screen is a refinement difference smaller than **0.002
spot units plus four combined Monte Carlo SEs** per contract; Gaussian and
variance-mean checks use a five-SE screen. All tested fixtures pass. The largest
64-to-128-step price difference is **0.000472 spot units**. The independent
100,000-path factor variance and driver cross-covariance checks are within
0.77 SE of their analytical values; the variance mean is within 1.71 SE. These
seeded screens are numerical development evidence, not simultaneous statistical
confidence claims or guaranteed bounds on remaining continuum bias.

The unit tests additionally verify exact zero-eta GBM/Black–Scholes limits,
matched continuation, shared-contract covariance, chunk invariance, estimator
bounds, and rejection of unsupported inputs. The report's elapsed time is a
local measurement, not a cluster scaling promise.

The separate [bank-specific screen](results/bergomi-bank-numerics.json) tests all
four corners and the center of the actual two-target bank prior: fixed
\(\kappa=2\), \(\eta\in[0.3,1]\), \(\rho\in[-0.8,-0.2]\), flat forward variance
0.04, spot 100, and contracts through one year. Each point compares 262,144-path
independent estimates at 128 and 512 steps. The declared screen is an absolute
difference below **0.025 currency units plus four combined MC SEs**, where 0.025
is 10% of the bank's quote-noise SD. All five screens pass. The largest observed
difference is **0.0309 currency units**, with combined SE **0.0190**. The largest
128-step price-estimator SE across these fixtures is **0.0219**, below the bank's
0.025 sampling-error limit.

This observed refinement difference is larger than 0.025 before allowing for
sampling uncertainty: **passing the screen does not establish bias below
0.025**. The 512-step estimates are also approximate. That limitation is retained
in the bank/report rather than hidden in its sampling error. Reproduce with:

```sh
PYTHONPATH=src pixi run python scripts/check_bergomi_bank_numerics.py --config configs/tasks/bergomi_option_bank.json --output docs/results/bergomi-bank-numerics.json
```

## What this unlocks and what remains

The same observation-bank infrastructure can now generate parameter/curve to
European-panel examples, retain clean approximate price labels, and add explicit
quote noise. Rough Bergomi can reuse the curve, contract and pricing boundary.
No real market data is necessary for that synthetic workflow. Applying a future
trained network to a real panel would require compatible context but would not
create parameter truth labels. Later held-out quote evaluation is a separate
empirical study.

Before a large training bank, test the actual selected parameter support and
price-error budget; these three fixtures do not certify every prior draw. Before
claiming an accepted posterior, train and validate that posterior separately.
Two factors, P forecasting and P/Q linkage remain optional future work.

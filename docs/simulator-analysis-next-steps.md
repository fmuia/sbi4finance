# Analysis to resume after simulator and bank coverage

The next work builds reusable HKDE, one-factor Bergomi and rough-Bergomi
simulation/pricing examples and a shared option-panel bank. It does not reopen
Heston network tuning or select the live tutorial's returns-versus-options
curriculum. A bank that can be generated and loaded is infrastructure; it is
not an accepted posterior or a certificate of numerical accuracy throughout a
prior. The milestone plans remain the acceptance specification.

## 1. Preserve the Heston finding and its boundary

The restricted Heston experiment infers long-run variance and the terminal
signed full-truncation auxiliary state jointly from 252 daily returns. All
other parameters and initialization are fixed. This is a declared discrete
physical-measure task, not full-SVJ inference or inference for a continuously
observed diffusion.

The network learns substantially from observations: the unconditional-prior
baseline's reference-normalized mean RMSE is 3.265, compared with 0.173 for the
bounded-time v2 candidate. Some parameter directions remain broad. Marginal
simulation-based calibration does not establish conditional or joint accuracy.
The remaining low-state discrepancy persists under more precise reference and
neural sampling. The one signed-asinh experiment did not fix it and stopped
under its declared rule. The better v2 candidate remains provisional.

| Current evidence | Consequence |
|---|---|
| v2 maximum conditional mean error 0.4554 SD exceeds 0.40; predictive endpoint error 0.2099 SD exceeds 0.15 | M3.3 posterior acceptance is pending. |
| v3 asinh maximum mean error 0.5268, relative SD error 0.3212 and predictive endpoint error 0.2403 exceed unchanged limits | No further scale/seed/architecture sweep in that increment. |
| Marginal rank checks pass for both candidates | Retain the marginal result, without overriding the conditional failures. |
| Root-43022 final test remains unevaluated | Preserve it; inspected development cases are not a fresh final test. |

Sources: [diagnostics](heston-diagnostics.md), [bounded-time experiment](heston-time-context.md),
[asinh protocol and outcome](heston-state-transform.md), and
[interactive review](../notebooks/05_heston_joint_inference.py).

If we resume inference, use a separately predeclared bounded experiment. First
decide whether the remaining error matters for the intended teaching quantity,
using the existing reference uncertainty and original-unit loss effects.
Then identify one testable mechanism, such as insufficient observation
summaries or conditional-density representation. Do not relax the old gates
after seeing a result, erase failed cases, feed latent variance into the
encoder, or use the reserved test to choose a candidate. Marginal SBC,
conditional references, joint dependence and future-loss checks remain distinct
pieces of evidence. Any newly declared tolerance must be motivated for a new
task and reported alongside, rather than in place of, the old result.

## 2. Finish numerical evidence before scaling option banks

The new kernels and small banks give concrete experiments for the following
checks. Their reports must state the tested parameter cases, grids, path counts
and error budgets; a successful ordinary case does not cover every proposed
prior draw.

| Branch | Next numerical analysis | What remains outside that evidence |
|---|---|---|
| HKDE | Check double-exponential support/compensation, analytic limits, Fourier integration refinement and an independently refined path-price comparison; relate residual errors to quote noise. | Posterior calibration, broad-prior accuracy, American/exotic products and P risk forecasts. |
| One-factor Bergomi | Check OU/asset cross-covariances, variance normalization, matched continuation with elapsed time, and price-grid refinement. | A two-factor scientific model, inference from historical returns and an accepted option posterior. |
| Rough Bergomi | Check the named Volterra driver's covariance and leverage dependence, compare reference and approximate grids, and measure price-refinement errors in difficult roughness/leverage regimes. | Production-scale convolution performance, inferred historical memory and an accepted option posterior. |

For Monte Carlo European pricing, bounded put payoffs have finite sampling
variance. Calls can be recovered through put–call parity within the supported
martingale domain. This avoids treating finite expected call prices as proof
of finite call-payoff variance. Sampling standard errors still omit time-grid
and kernel bias. Shared simulated paths produce correlated errors across
contracts: preserve that dependence when judging a complete surface. A scalar
per-contract error bar is not a simultaneous confidence band.

Bank records should separate clean numerical expectation estimates,
intentionally noisy observed quotes, latent parameter labels, and numerical
diagnostics. Diagnostics computed using true parameters are not observed
conditioning. Failed numerical draws retain their original identities and
reasons; replacing them silently changes the prior. Freeze independent base
surfaces before quote-noise augmentation and keep all derivatives of one
surface in its split.

## 3. Pick one restricted inference task only after its price labels are ready

**First Q experiment now implemented: the existing two-parameter HKDE panel.**
The [measured baseline](hkde-joint-inference.md) now includes the refined grid,
training/reload, conditional/joint checks, SBC and target-price propagation.
Its reference passes, and parameter/SBC gates pass; maximum target-price
quantile error 0.0439 exceeds the frozen 0.03 currency gate. The one-run stop
rule was applied, so this checkpoint remains provisional and its 256 final-test
panels remain unevaluated. Mean RMSE is 0.123 reference SD versus 2.09 for the
prior baseline: this is substantial updating, not a prior copy. The graph-linked
review preserves the failure. Any later bounded follow-up must retain this
record; do not quietly relax the output tolerance or start a broad tuning sweep.

The original choice was a bounded implementation experiment, not selection
of the final curriculum. Its
deterministic Fourier labels and Gaussian quote law permit a direct likelihood
and a two-dimensional numerical posterior reference over `theta` and
`jump_intensity`. The saved bank's largest Fourier refinement difference is
about 1.12e-9 currency, against quote sigma 0.25. The refined posterior
grid, fixed cases and first baseline now provide these comparisons. Keep the
final-test cases separate from further development. A surrogate is a later,
independently accepted step.

There is no need to expand every Bergomi bank before that integration. Their
largest saved MCSE/quote-sigma ratios are 0.0814 (one factor) and 0.0741 (rough).
However, the current `abs(grid difference) <= 0.025 + 4 SE` rule is a discrepancy
screen, not proof that bias is below 0.025. None of the five complete corner/
center panels passes the stronger resolution check
`abs(grid difference) + 4 SE <= 0.025`; the independent runs cannot resolve that
budget. Even a precisely measured two-grid difference is not a continuum bound.

A coherent Bergomi inference pilot can instead declare the implemented law
explicitly:

\[
y=\widehat P_{128,N}(\theta;U)+0.25\epsilon,
\qquad N=262144,\quad \epsilon\sim N(0,I).
\]

Here `U` is the independent Monte Carlo randomness used to estimate a full
panel. Its errors are correlated across contracts. This numerical experiment
already has a valid simulator; it is different from a Gaussian likelihood
centered at an exact continuum price. Independent base draws/splits and full
validation remain necessary before accepting its posterior.

If tighter Bergomi price fidelity becomes necessary, use one bounded **coupled
grid comparison** before increasing every bank's path count. For one factor,
subsample the fine exact OU factors and sum fine Brownian increments into
coarse increments. For the rough hybrid, aggregate independent normal inputs
to coarse standard normals, then rebuild each grid with its own hybrid weights
and normalization; subsampling a fine hybrid field does not give the coarse
model. Compare paired conditional-put expectations and estimate the SE of their
difference. This reduces sampling noise while preserving both discrete
marginals; it measures a grid difference, not uniform continuum accuracy.

For any chosen model, declare a small parameter prior, known context, contract
panel and observation law. Decide explicitly whether the initial
forward-variance curve is known context or an inferred object. Price-label
error must be measured against the intended quote-noise scale before generating
a large training bank. Numerical fidelity is metadata unless a separate
fidelity-conditioned task is deliberately specified.

Three later outputs have different training pairs:

1. **Parameter posterior:** noisy panel and known context → joint Q parameter
   draws; propagate each draw through a validated target pricer.
2. **Pricing surrogate:** parameters and contract/context → accurate reference
   expectation labels. A single payoff is not a price label.
3. **Direct price posterior:** noisy panel/context → target reference prices
   generated from the same parameter draw. Preserve their dependence with the
   panel and the grouped data split.

Use independent synthetic recovery cases, marginal SBC, joint and conditional
reference checks, and target-price-sensitive quantities. Include informative
and weakly identified cases. A posterior that reproduces observations can
still get extrapolated or path-dependent prices wrong. No bank or notebook in
this increment substitutes for that inference validation.

## 4. Scale measured work and preserve the data boundary

Use the same frozen task and stable example IDs locally and on the cluster.
The [option-workload smoke](option-cluster-smoke.md) now measures three actual
24-panel jobs, exact Linux regeneration, complete transfer hashes and selected
Mac numerical replay. Generation/write took 2.45, 111.95 and 116.59 seconds
for HKDE, one-factor and rough Bergomi; job elapsed times also include
verification. Use these bounded measurements before choosing a first modest
training volume, without extrapolating them into a large-scale benchmark.
The initial HKDE portability failure and currency-only tolerance adjustment
remain recorded separately from numerical accuracy. Do not turn a notebook
opening into a generation or training job.

All required **S** work can be synthetic and offline: generated curves and
contracts, clean prices, noisy panels, training, validation and downstream
pricing. Optional **M** conditions a trained network on a compatible real
panel; it does not provide true parameters and does not require target quotes
just to produce a model-price distribution. Optional **E** uses contemporaneous
excluded quotes, or later outcomes for a separately specified P forecast,
without preprocessing leakage. A single real panel is a case study, not a
general backtest. Synthetic checks do not certify market adequacy.

These option tasks are under **Q**. A calibrated Q parameter vector, one-factor
state or rough memory is not a P forecast of tomorrow's returns or market
option chain. Rough returns forecasting would additionally require P dynamics,
initialization and inference of the complete conditional memory. Joint P/Q
modeling remains deferred.

Continue with the separate subplans for [M4](simulator-plan/milestones/04-hkde-and-options.md),
[M5](simulator-plan/milestones/05-amortized-pricing.md),
[M6](simulator-plan/milestones/06-bergomi-factors.md) and
[M7](simulator-plan/milestones/07-rough-bergomi.md). Their remaining numerical
and inference gates stay explicit while simulator and bank coverage expands.

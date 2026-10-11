# Restricted rough-Bergomi posterior pilot

M7.3b follows the [paired numerical study](bergomi-coupled-refinement.md).
The frozen machine-readable contract is
[`rough_inference_protocol.json`](../configs/rough_inference_protocol.json).
This is synthetic Q evidence, with no market acquisition dependency. It tests
one small inference experiment; it does not reopen the curriculum choice.

## What is observed and inferred?

The unknown pair is theta = (H, eta), independently uniform on
[0.1, 0.3] × [0.4, 1.2]. Leverage −0.6, the flat forward-variance curve 0.04,
spot 100, rate 0.02, zero dividend and nine ordered European puts are known.
Each observation is nine noisy currency prices. Neither parameter truth nor
Monte Carlo diagnostics are observed inputs. The network returns paired H/eta
draws; there is no inferred current state or P forecasting task here.

Let U denote all random numbers in a 262,144-path, 512-step hybrid pricing run.
The explicit observation model is

\[
\theta\sim\pi,\quad U\sim p_U,\qquad
Y=\widehat C_{512,262144}(\theta,U)+0.25\epsilon,
\quad \epsilon\sim N(0,I_9).
\]

Thus Monte Carlo noise is part of the declared numerical experiment. It is
small relative to quote noise but is not silently treated as zero. The prior
paired study supports 512 steps for this pilot; it does not prove continuum
accuracy. Existing 128-step banks retain their identities.

## Ordered subplan

1. **Freeze and generate.** Record the protocol before outcomes. Generate 1,024
   training bases, 256 selection cases, 512 audit cases and 256 reserved test
   cases. Fixed-ID cluster chunks must assemble to the existing bank schema,
   independent of scheduling. Retain the 0.025 marginal MCSE screen; preserve
   failures instead of redrawing. Use eight independent quote augmentations
   per training base (8,192 observations, only 1,024 independent bases). All
   augmentations remain within their base split.
2. **Build an independent reference.** Generate 4,096 independent prior rows
   with a separate root seed. The reference bank has two additional, unused
   compatibility rows to retain the common three-split format. Weight clean
   numerical panels, never their already-noisy observations. Check effective
   sample size, largest weight, self-normalized importance MCSE and agreement
   between two independent blocks. These checks are finite-sample evidence,
   not a guarantee that every important region was sampled.
3. **Train once.** A joint four-layer spline coupling flow uses the same small
   architecture as the earlier restricted exercise. Train for at most 80 epochs
   and select on the separate 256-case selection set. Verify same-seed checkpoint
   reload. Freeze a single candidate; no architecture or seed sweep.
4. **Audit inference and price sensitivity.** Compare 16 preselected audit
   panels and four stress panels against the reference. Examine joint moments,
   90% interval mass, and posterior updating relative to the prior. Run marginal
   SBC on all 512 independent audit cases. Compare posterior **mean** prices of
   two excluded puts using fresh target-pricing randomness. Preserve failed
   gates and keep the final test reserved unless every validation gate passes.
5. **Teach the result.** Add a marimo walkthrough of the numerical observation
   law, independent reference, prior control, SBC versus conditional checks,
   measured costs and actual outcomes. Expensive generation and training stay
   opt-in; cached review must run offline with no market fixtures.

## Why the reference has this form

For independent reference rows (theta_i, U_i),

\[
w_i(y)\ \propto\
\exp\left[-\frac{\|y-\widehat C(\theta_i,U_i)\|^2}{2(0.25)^2}\right].
\]

These are prior-importance weights for the joint numerical posterior of
(theta, U). Marginalizing U gives the desired parameter posterior. Adding the
saved pricing covariance to this likelihood would count simulation noise a
second time. The reference is a Monte Carlo approximation, not an exact grid
posterior or a price surrogate. Its error must be reported alongside neural
differences. A weakly identified H can retain a broad prior-like marginal while
eta or their dependence still updates; the prior control tests that distinction.

For an excluded contract, the downstream estimand is

\[
E_{\theta\mid y}[P^{512}_{\rm target}(\theta)].
\]

Estimate each target with independent randomness V, separate from U and quote
noise. Shared conditioning paths would select pricing fluctuations through the
weights and change the estimand. Reference rows use 4,096 target paths; neural
draws use 128, with outer-and-inner sampling error reported for the pooled mean.
No individual noisy payoff average is presented as an exact price, and no
quantiles of these noisy averages are presented as model-price uncertainty.
The two new contracts have no separate continuum-refinement certificate.

## Acceptance scope

All thresholds, IDs, budgets and seeds are in the protocol. Numerical simulation
delivery, reference adequacy, neural acceptance and tutorial execution have
separate statuses. SBC alone cannot certify conditional accuracy or dependence;
passing synthetic tests cannot establish market adequacy. A failed gate is a
reported outcome of this bounded pilot, not a reason to tune until it passes.
Real return windows or option panels remain optional applications and would
require matching context. This Q posterior does not forecast tomorrow's chain
or furnish physical-measure risk probabilities.

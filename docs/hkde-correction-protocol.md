# HKDE: one bounded likelihood-correction follow-up

This follow-up to M5.1 freezes an inference **pipeline**, using the existing
checkpoint without changing its weights. Its purpose is to test whether a
small correction can resolve the downstream price discrepancy and support an
offline M8 options candidate. It does not select the final curriculum, repair
the raw network retrospectively, or change any original acceptance tolerance.
The [settings](../configs/hkde_correction_protocol.json) are committed before
the new scientific run. The earlier failed report and reserved test stay intact.

## Why this is a different estimator

For the same two-parameter HKDE Q task, let `q(a|y)` be the frozen neural
density in **raw parameter units**, `p(a)` the declared uniform prior, and
`L(y|a)` the original nine-quote Gaussian likelihood using the refined Fourier
map. Draw each proposal independently from

\[
g(a\mid y)=0.9q(a\mid y)+0.1p(a),\qquad
w_i\propto L(y\mid a_i)p(a_i)/g(a_i\mid y).
\]

The prior component gives positive proposal density throughout the prior box.
It does not guarantee that a finite sample visits every important region.
Every row uses the **full mixture denominator**, including rows drawn from the
prior. Log-sum-exp stabilizes normalization; weights are never clipped.

BayesFlow's density call already includes logit-adapter and learned
standardization Jacobians. Evaluate one parameter pair per batch row, repeating
that row's observed panel. Adding another manual Jacobian, or silently summing
Jacobians across the sample dimension, would change the target distribution.
Check the density convention independently before evaluating scientific gates.

Self-normalized importance estimates have finite-sample bias and Monte Carlo
error. They are consistent under the usual support/integrability conditions;
they are **not exact finite-sample posteriors**. Report effective sample size,
largest normalized weight, weighted joint moments and financial quantities,
with eight independent proposal-block estimates for uncertainty. High ESS
alone is not evidence that no posterior mode was missed. The independent
quadrature reference supplies the conditional accuracy check here.

The accepted polynomial surrogate evaluates the **two downstream put prices**
on matched joint parameter rows. It does not replace the nine prices required
by the likelihood. Measure sampling, density evaluation, numerical likelihood,
target pricing and loading separately; make no speed advantage assumption.

## Frozen checks and data roles

1. Reuse the original 16 conditional audit panels and four stress panels. Draw
   8,192 IID mixture proposals per panel, seeded independently by case. The
   participant algorithm uses the first 2,048 proposals; the full 8,192 are a
   precision check. **Both budgets must pass** all conditional gates; no
   post-result choice between budgets is permitted. Use
   weighted moments and weighted empirical quantiles, rather than adding
   resampling noise to conditional comparisons. Apply the original six
   parameter/joint/price gates unchanged, plus ESS/N at least 0.5 and largest
   weight at most 0.01. All rows of both target prices retain joint pairing.
2. Audit all 512 original validation panels 256–767. Each gets 2,048 proposals
   and 512 joint multinomial resamples. Run randomized marginal SBC ranks and
   coverage on this **complete sampling-importance-resampling algorithm**,
   with the original two-target 5% DKW criterion. Resamples from one finite
   cloud are dependent after integrating over that cloud; the exact-posterior
   rank-uniformity result is a benchmark, not a theorem for this approximate
   algorithm. Require every audit panel's ESS/N at least 0.1 and maximum
   weight at most 0.05. Truth is used only in evaluation.
3. Only after every validation gate passes, generate a **new** 256-panel test
   at root 95026, using the same observation law and independent named streams.
   Audit its first 16 cases conditionally and all 256 with the same finite-pool
   calibration procedure. Check 65-to-129 reference refinement on those 16
   cases using the original tolerances. No refinement or tuning is triggered
   by test outcomes. The original root-62026 test remains unevaluated.
4. Save exact protocols, source/artifact hashes, weighted proposal clouds for
   conditional review, calibration ranks/efficiencies and measured times.
   Any gate failure stops release without an automatic retry or new seed.

The known validation panels motivated this follow-up, so their improvement
alone is development evidence. The new independent test evaluates the fixed
pipeline. It is not a claim of real-market adequacy, a uniform prior-domain
guarantee, or proof that all latent HKDE parameters can be recovered.

## Teaching handoff

If the complete new experiment passes, an M8 options candidate can load the
saved checkpoint, condition on a labelled synthetic panel, perform the bounded
2,048-proposal correction live, and price its weighted joint draws with the saved surrogate.
Explain why raw neural SBC passed while a conditional price check failed, what
the correction costs, and why likelihood-based correction is available for this
example. A rough/likelihood-intractable task needs a different reference route.

The notebook must run offline without market fixtures or automatic training,
include exercises and exact evidence links, and show the original failure
alongside the corrected result. Optional M/E data are neither necessary nor
silently substituted. Heston's and rough Bergomi's unresolved analyses remain
separate; completing this candidate does not certify those branches.

Before scientific execution, the protocol was clarified to require the same
2,048-proposal budget for participant inference, conditional acceptance and
SIR calibration. The full 8,192 run is an additional precision check, not an
automatic fallback. This avoids transferring calibration claims across budgets.

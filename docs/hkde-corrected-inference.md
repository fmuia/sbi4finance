# HKDE likelihood correction and participant handoff

This is a separately specified follow-up to the [provisional neural baseline](hkde-joint-inference.md).
The saved flow supplies proposals; a numerical-likelihood importance correction
supplies the weighted posterior approximation. The raw checkpoint is unchanged.
See the [boundary-proposal protocol](hkde-correction-boundary-protocol.md) for the estimand, budgets,
accuracy gates and independent-test rule. **Validation passes; the new test
fails its long-run-variance SBC gate.** The corrected pipeline and M8 participant
acceptance remain pending. The frozen stop rule was applied without another
seed, proposal, threshold change or training run.

## What is being completed

The simulator/bank ladder already supplies GBM, Heston, Gaussian price jumps,
HKDE, one-factor Bergomi and rough Bergomi. This increment connects existing
HKDE inference and learned pricing into a bounded options teaching candidate.
It adds no model family, training campaign, provider dependency or framework.

The declared observation is a vector of nine synthetic noisy put prices.
The unknowns are long-run variance and jump intensity; other Q model parameters,
initial variance, contracts, rates and spot are known context. The forward
surrogate returns the model prices of two excluded puts. Neither true synthetic
parameters nor target prices enter the network's observation inputs.

The completed pipeline is intended to be:

1. Load the existing trained flow, task and accepted two-contract surrogate.
2. Draw 2,048 independent proposals from 80% flow / 10% uniform prior /
   10% product-arcsine boundary distribution. This changes the proposal only;
   the posterior prior remains uniform.
3. Evaluate all nine numerical expectation prices per parameter row and the
   declared Gaussian quote likelihood. Divide likelihood times prior density
   by the **full** mixture proposal density; normalize without clipping.
4. Retain matched parameter pairs and weights. Evaluate both target prices
   with the accepted surrogate and summarize their weighted distributions.
5. Compare with the independently refined numerical posterior and inspect
   accuracy, importance efficiency, finite-pool uncertainty and elapsed cost.

The eight-block uncertainty summary is descriptive Monte Carlo precision.
Self-normalized estimates have finite-sample bias. Positive support and high
observed ESS do not certify exploration of every mode. The density implementation
is tested against independently assembled logit/standardizer/network Jacobians,
a normalized-density quadrature, and an analytic nonuniform-proposal experiment.

## The retained first correction

The [first correction](results/hkde-correction-v1/validation.json) passed every
accuracy check, reducing maximum price-quantile error to 0.01261 at the
participant budget and 0.005773 in its precision run. It stopped on precision
weight concentration in `stress:2`; no SBC or new test was run. The
[boundary diagnosis](hkde-correction-boundary-review.md) describes the retained
high-weight row and explains the fixed additional proposal component. No threshold,
prior, likelihood, checkpoint, budget or seed was changed.

## Scientific and teaching evidence

The two proposal budgets (2,048 participant; 8,192 precision check) must both
pass the original six inference gates and additional weight-efficiency limits.
The 512-case prior-predictive calibration study resamples 512 joint draws from
2,048 weighted proposals per case. Its SBC audits this finite-pool algorithm;
shared-cloud resamples are not exact independent posterior draws.

A new 256-panel test at root 95026 is generated only after every validation
check passes. Its first 16 cases get refined conditional references; all 256
get the same calibration audit. It cannot select another checkpoint, proposal,
seed or budget. The original baseline's root-62026 test remains untouched.

The M8 options candidate is [notebook10](https://github.com/fmuia/sbi4finance/blob/feat/hkde-offline-tutorial/notebooks/10_hkde_corrected_workflow.py)
in the following participant-delivery PR.
Its 30-minute lesson explains the task, weight formula, joint dependence,
price uncertainty and diagnostics, then asks short interpretation exercises.
It uses a saved synthetic validation panel and computes one correction live.
Training, base-bank generation and reference-grid pricing are cached and are
never started by opening the notebook. A final curriculum choice remains open.

### Measured validation

The boundary proposal was frozen before this run. Both conditional budgets
passed on all 20 declared cases, and the 512-panel calibration audit passed.

| Validation quantity | 2,048 proposals | 8,192 proposals | Frozen limit |
|---|---:|---:|---:|
| Mean RMSE / reference SD | 0.02928 | 0.01138 | 0.15 |
| Largest absolute mean error / reference SD | 0.06570 | 0.02526 | 0.40 |
| Largest relative SD error | 0.02519 | 0.01559 | 0.25 |
| Largest absolute correlation error | 0.01935 | 0.00712 | 0.10 |
| Largest target-price quantile error (currency) | 0.01110 | 0.00482 | 0.03 |
| Absolute error in average reference mass of 90% intervals | 0.00151 | 0.00042 | 0.03 |
| Smallest importance ESS / proposal count | 0.68777 | 0.68281 | at least 0.50 |
| Largest normalized weight | 0.00256 | 0.00095 | at most 0.01 |

Finite-pool SIR rank KS statistics are 0.03149 and 0.03514, below the two-target
DKW limit 0.06542. Marginal 90% interval coverage is 92.19% and 91.60%; the
saved report also gives individual binomial uncertainty intervals. The smallest
calibration ESS fraction is 0.51058 and largest weight 0.00789, within their
separately declared 0.10/0.05 limits. Marginal SBC alone would not validate
conditional accuracy or dependence; those are checked in the table above.

An [independent saved-artifact audit](results/hkde-correction/validation-independent-audit.json)
reconstructs the weights exactly, conditional diagnostics to within 5e-15, and
rank statistics exactly. It checks source/input hashes and independently
replays parameter seeds. Calibration clouds and individual SIR draws were not
saved, so the audit reconstructs the KS calculation from saved ranks and checks
coverage-count arithmetic; it does not independently reconstruct ranks or
coverage events from those missing individual draws.

The full validation took 1,246.52 seconds (20.78 minutes) locally. Of that,
the 20 full conditional proposal pools required 164.50 seconds of numerical
likelihood work, whereas their two-target surrogate/check step took 0.146
seconds. The 512 calibration panels, including proposal, likelihood and
resampling, took 1,074.35 seconds. These are measured audit costs on this
machine, not a participant latency or a speed comparison with another sampler.

### Where to inspect the diagnostics

- [Price-quantile errors](results/hkde-correction/plots/price-errors.png): compare
  the preserved raw flow with the corrected estimator against the same
  independent reference and 0.03 currency gate. The separate test row assesses
  the fixed correction on new panels; it does not reopen the old flow's test.
- [Proposal efficiency](results/hkde-correction/plots/proposal-efficiency.png):
  inspect both ESS fractions and largest weights for the conditional and
  calibration pools. These diagnose observed weight concentration, not missing
  modes. The [retained v1 plot](results/hkde-correction-v1/plots/proposal-efficiency.png)
  shows why the first correction stopped despite accurate price estimates.
- [Finite-pool SBC](results/hkde-correction/plots/calibration.png): compare
  randomized rank CDFs with the declared two-marginal DKW bounds. This is
  prior-predictive calibration of the complete proposal/weight/resampling
  algorithm; the conditional references supply different evidence.

The raw flow already passed marginal SBC while missing a conditional price
gate. That observed example explains why neither calibration nor ESS can
replace downstream, case-specific reference comparisons.

### New independent test: recorded stop

The [256-panel test report](results/hkde-correction/test.json) uses the separately
reserved root 95026 and is bound to the exact passing validation report. Its
refined numerical reference and all conditional/weight gates pass on the first
16 cases at both budgets. Maximum target-price quantile error is **0.01125**
currency at 2,048 proposals and **0.00638** at 8,192, below 0.03. Minimum
conditional ESS fractions are 0.73186/0.74044, and largest weights are
0.00292/0.00077. The saved report retains all six accuracy metrics.

However, finite-pool SIR calibration on all 256 cases fails for long-run
variance: **KS 0.10325 exceeds the frozen DKW bound 0.09251**. Jump intensity's
KS is 0.07203. The respective 90% coverages are 90.63% and 91.02%; plausible
coverage alone does not override a rank-distribution failure. Calibration
importance efficiency passes (minimum ESS/N 0.53493; maximum weight 0.00666).
This test took 618.53 seconds, including 492.77 seconds for calibration.

The test is now **used and closed** for this experiment. It cannot be reused
as a fresh selection holdout. The original raw-neural test remains untouched.
The initial development review displayed this stop and skipped live execution.
After the [saved-cohort gate audit](hkde-calibration-gate-audit.md), the
[M8 lesson](m8-options-candidate.md) now permits a provisional live calculation
on fixed development case `validation:257`. Its successful conditional check
is separate from this unchanged historical failure and does not establish
population calibration or full participant acceptance. Any new scientific
acceptance experiment must distinguish cohort variability, reference/numerical
effects and finite-pool inference error. No automatic training or seed/budget
sweep follows from this result.

### Explaining the rank shift with the same numerical reference

The [independent test audit](results/hkde-correction/test-independent-audit.json)
reproduces the failed gate, all weight calculations and the new panel/quote-noise
seed identities. It also evaluates the posterior marginal CDF at each of the
256 saved parameter truths using the existing reference grid. This probability
integral transform (PIT) needs no neural proposal or new simulated observation.

| Same held-out cohort | Long-run variance KS | Jump-intensity KS |
|---|---:|---:|
| Finite-pool randomized SIR ranks | 0.10325 | 0.07203 |
| Numerical-reference PIT, 129 nodes per axis | 0.10349 | 0.08212 |
| Numerical-reference PIT, 65 nodes per axis | 0.10340 | 0.08213 |

The [reference-PIT comparison](results/hkde-correction/plots/reference-pit.png)
shows essentially the same long-run-variance shift. Mean SIR minus reference
PIT is 0.00055; their paired correlation is 0.99690. This suggests variability
shared by this finite cohort and supplies no evidence that another proposal
change would fix the failed rank statistic. It does **not** prove absence of
all inference or model errors, and this post-hoc check does not override the
frozen failure.

For reproducibility, the calculation starts from each normalized two-dimensional
reference mass array. Sum over the other coordinate and divide by that axis's
trapezoidal quadrature weights to recover the marginal density. Cumulatively
integrate the density by the trapezoidal rule, normalize its endpoint to one,
and linearly interpolate the CDF at the generating truth. Repeat on every
second grid node for the 65-node comparison. The [saved arrays](results/hkde-correction/test-reference-pit.npz)
retain both PITs, matched SIR ranks, truths and IDs; the audit records their
checksum. This is a numerical quadrature/interpolation reference, not an
analytic posterior oracle.

## Reproduction

Use the pinned Pixi environment from the repository root. The commands create
new output directories and must be run sequentially. A failed stage writes its
report and exits nonzero; do not use it to unlock the next stage.

```bash
PYTHONPATH=src pixi run python scripts/evaluate_hkde_correction.py \
  --protocol configs/hkde_correction_boundary_protocol.json \
  --out /tmp/hkde-correction-validation-new
PYTHONPATH=src pixi run python scripts/evaluate_hkde_correction.py \
  --protocol configs/hkde_correction_boundary_protocol.json \
  --split test --validation-report /tmp/hkde-correction-validation-new/validation.json \
  --out /tmp/hkde-correction-test-new
```

The original bank identity and configuration are checked; its training arrays
need not be present or regenerated. These commands regenerate only the
validation panels and, after acceptance, the new test panels.
The checkpoint, input task, source modules, accepted numerical reference,
surrogate and exact protocols are bound by hashes in the new reports. The
conditional artifacts retain parameter proposals, component identities, log
proposal densities, log likelihoods, normalized weights and matched target
prices. Independent verification can reconstruct the weighting without a
neural training run.

## Boundaries that remain

This small two-parameter Q example has an evaluable likelihood and a reusable
2D reference grid. It is useful for teaching neural approximation and its
correction, not evidence that neural inference is universally faster. Numerical
likelihood calls remain in the corrected route; the two-output
surrogate does not replace them. Any elapsed-time comparison must include this
cost, loading and compilation.

A successful corrected workflow would accept this **pipeline** on its tested
synthetic task, not the raw flow or every HKDE parameter. Heston's recorded
conditional failures, rough Bergomi's failed reference-efficiency case, the
returns-versus-options curriculum decision and optional model extensions remain
separate. Real-market application is optional M; empirical evaluation is E and
needs genuinely held-out outcomes or target quotes. Neither is required for
this S notebook or implied by its synthetic checks.

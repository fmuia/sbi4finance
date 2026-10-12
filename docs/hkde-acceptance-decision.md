# HKDE acceptance decision: retain the provisional teaching scope

**12 October 2026 — issue #24 decision, for review.** Keep the frozen corrected
HKDE workflow **provisional at 2,048 proposals**. Continue the selected-case
teaching demonstration with its existing checks and visible limitations.
Formal scientific participant acceptance remains **false** and data integration
**I4 stays deferred**; I4a remains a separate provisional demonstration.
No new experiment is needed to make this decision.
Published for review in [PR #29](https://github.com/fmuia/sbi4finance/pull/29),
based on PR #19; issue #24 is synchronized and remains open for review.

This completes the decision requested in [issue #24](https://github.com/fmuia/sbi4finance/issues/24)
without claiming that the shared calibration anomaly has been resolved.
The [machine-readable decision](results/hkde-acceptance/decision.json),
[artifact manifest](results/hkde-acceptance/artifact-manifest.json) and updated
[notebook11](../notebooks/11_hkde_calibration_audit.py) accompany this note.
The historical correction reports and acceptance flags remain unchanged.

## Claim and exact scope

For the frozen two-parameter synthetic Q task, the 2,048-proposal correction
passed the specified parameter-moment, correlation, marginal-interval and
excluded-option price-quantile checks on the recorded conditional cases.
Its historical prior-predictive calibration gate failed, with a similar alarm
in the numerical reference. This supports a provisional teaching example and
the reported case-specific comparisons. It does not establish accuracy over
the entire observation domain, general calibration, or a full joint-distribution
error bound.

The estimand is the posterior of `(theta, jump_intensity)` under independent
uniform priors `[0.025,0.065] × [0.25,1]`, conditional on nine European put
quotes with IID Gaussian noise of SD 0.25 currency. Contracts are strikes
90/100/110 within maturities 0.25/0.5/1 year; spot 100, rate 0.02, dividend 0,
initial variance 0.04 and the other HKDE parameters are fixed by the task JSON.
The declared likelihood uses the frozen Fourier price map (1,024 nodes,
upper integration limit 400), rather than a continuum-pricing oracle.

The estimator is `hkde_boundary_defensive_importance_v2`: 2,048 proposals from
80% frozen flow, 10% uniform prior and 10% product-arcsine boundary density.
Every importance denominator uses the full mixture; the numerator retains the
uniform prior and all nine numerical likelihood prices. Keep paired parameter
rows and normalized weights intact. The separately validated surrogate maps
each pair to the two excluded puts `(K=95,T=.75)` and `(K=80,T=1)`.
Their weighted distributions describe uncertainty in Q model prices, not
realized payoffs, future market quotes or physical-measure losses.

The manifest binds the task, checkpoint, likelihood and correction sources,
surrogate, original protocols, saved evidence and environment lock to SHA256
identities at PR #19 head `b588df33d8fa1456c05bae67438d130d5b00ee1a`.
Its 8,192-proposal results are separately labelled precision evidence.
Historical calibration uses 2,048 proposals and 512 joint SIR resamples per
panel. The fixed live example is `validation:257`, a selected development case.
The manifest is an identity check, not proof that the evidence is sufficient.

## Evidence used in this decision

| Question | Evidence | What it establishes |
|---|---|---|
| Numerical reference/model | Refined posterior grids, price refinement and saved-input replay pass their checks. | Numerical stability for the declared observation model. Generator and reference share a Fourier engine; no independent continuum or market certificate. |
| Conditional approximation | The 20-case development panel and 16-case independent test panel pass the six aggregated conditional checks at both budgets, with every case meeting both weight limits. At 2,048, maximum test target-price quantile error is 0.01125 currency, below 0.03. | Recorded finite-case accuracy. Correlation assesses one feature of dependence; it does not validate the entire joint posterior. |
| Prior-predictive calibration | Development SBC passes on 512 panels. On 256 independent panels, corrected theta KS is 0.10325 > 0.09251; numerical-reference theta PIT KS is 0.10349. | The historical test failed. Similar KS distances do not measure conditional posterior distance or identify the cause of the shared shift. |
| Finite-pool uncertainty | Direct weighted-CDF RMS errors on 16 cases are 0.00884/0.00940 at 2,048 proposals; estimated RMS SEs are 0.00976/0.00980. | Measurable approximation error of a scale consistent with SNIS uncertainty. Each CDF is evaluated at one truth threshold; this is not a marginal supremum or joint-CDF bound. |
| Participant delivery | The fixed development example passes its live checks and exports offline. | Execution and selected-case accuracy only. I4a does not complete I4. |

The [gate audit](hkde-calibration-gate-audit.md) shows that 98.85% of ideal-reference
theta rank controls trigger the alarm **conditional on this already-inspected
cohort**. That is not a population false-alarm rate or grounds for clearing the
gate. The original all-cohort proposal pools and individual SIR draws were not
saved, so existing ranks cannot fully separate pool error from resampling.
The two proposal budgets use nested pools, not independent replications.

The raw flow remains provisional after its conditional price failure. The v1
correction's weight-concentration stop also stands. Root **95026 is used and
closed**; root **62026's raw-neural test remains untouched**. Development cases,
post-hoc controls and the selected live example do not become new holdouts.

## Policy for continued provisional use

This policy applies prospectively to how the frozen example is presented and
checked. It retains the original numerical tolerances because the task and
intended outputs are unchanged. It is **not a new experiment protocol** and
does not relabel any inspected data as independent acceptance evidence.

1. Verify the task and artifact identities before interpreting a result.
   A changed task, checkpoint, source, proposal, budget or surrogate is outside
   this decision. Missing or mismatched evidence blocks the comparison.
2. Require the existing numerical-reference prerequisites. The 65/129-node
   posterior-grid comparisons must stay within 0.01 for mean differences in
   fine-reference SD units, relative SD differences and absolute correlation
   differences; 0.0025 currency for target quantiles; and 0.001 for log evidence.
   The price-refinement discrepancy divided by quote SD must be at most 0.001
   (1,024/400 versus 2,048/800 price controls on the declared check points).
   These are stability diagnostics, not certified worst-case error bounds.
3. Evaluate the unchanged conditional and reliability criteria below at the
   actual 2,048 budget. The separately recorded 8,192 precision comparison
   must satisfy its original criteria too; its success cannot substitute for
   the participant result. For the fixed live case use its existing reference
   and saved precision evidence, without starting a new 8,192 run on opening.
4. Report MC uncertainty and every failed check. Passing the selected-case
   checks permits only the label **provisional demonstration; case checks
   passed**. A failure remains visible and does not trigger a seed retry,
   larger pool, threshold adjustment or acceptance override.

| Conditional diagnostic | Retained limit and aggregation |
|---|---|
| Mean error / reference SD | RMS across cases and both parameters ≤0.15; largest absolute error ≤0.40. |
| Relative SD error | Largest absolute error across cases and parameters ≤0.25. |
| Correlation error | Largest absolute error across cases ≤0.10. |
| Target-price quantile error | Largest absolute error across cases, two targets and quantiles 0.05/0.50/0.95 ≤0.03 currency. |
| Reference mass in marginal 90% intervals | For each parameter average the mass across cases; the largest absolute departure of those averages from 0.90 must be ≤0.03. This is not a worst-case interval or cohort coverage guarantee. |
| Proposal weights | Every conditional case ESS/N ≥0.50 and largest normalized weight ≤0.01. |

For one live case the aggregates reduce to that case. Keep the six accuracy
checks distinct from weight diagnostics: ESS alone cannot certify mode coverage
or price accuracy. The 0.03 price limit is 12% of the declared 0.25 quote-noise
SD, an existing tutorial tolerance, not a market-wide economic standard.
Reference target-quantile tolerance 0.0025 is twelve times smaller than 0.03;
neither number is a simultaneous confidence bound. The surrogate retains its
own existing 0.003-currency acceptance gate within its exact two-contract scope.

**Monte Carlo treatment:** show plug-in SNIS mean/CDF SEs and the existing
eight-block mean/quantile precision summaries as approximate diagnostics.
Compare observed discrepancies directly with the fixed limits; do not subtract
two SEs to turn a failure into a pass. These SEs omit finite-sample and numerical
bias and do not certify simultaneous tolerance satisfaction. Increasing SIR
resamples cannot remove shared pool error. No new CDF cutoff is selected around
the observed 0.009 RMS errors.

## Calibration and the shared-alarm rule

Keep the historical two-coordinate DKW rule at family alpha 0.05:
`sqrt(log(2*2/0.05)/(2*N))`, giving 0.092513 at N=256. Its alpha concerns those
two marginal uniformity diagnostics, not every gate or every experiment.
The historical calibration weight limits remain ESS/N ≥0.10 and largest
weight ≤0.05 for every panel. Passing marginal SBC cannot replace conditional
or dependence checks; finite-pool SIR is not an exact IID posterior sampler.

| Evidence state | Disposition |
|---|---|
| Numerical/reference precision or identity prerequisites fail | Reference/comparison blocked; no reliable approximation verdict from that comparison. Investigate separately before further interpretation. |
| Reference prerequisites pass; conditional or weight checks fail | Provisional, with the specific approximation/reliability failure retained. |
| Conditional checks pass; corrected SIR and reference PIT share an alarm | Provisional; calibration unresolved. Do not blame only the correction or waive its failure. **This is the current state.** |
| Reference PIT passes; corrected SIR alarms | Provisional; investigate finite-pool approximation and resampling without assuming a cause. |
| Corrected SIR passes; reference PIT alarms | Provisional; investigate the discrepancy. A more uniform approximation is not evidence of a better posterior. |
| Both calibration diagnostics and conditional checks pass | Report the scoped checks passed. This alone does not certify general calibration, full joint accuracy or formal I4 acceptance. |

A shared calibration alarm alone is not a numerical-reference precision
failure: **reference-blocked** and **calibration unresolved** are different
outcomes. The table is an interpretation policy, not a second retrospective
statistical test or an executable new acceptance gate.

## Why no replication now; when to reopen

The immediate output is an honest selected-case lesson, which already runs
with measured conditional checks. Another cohort is not needed to retain that
scope. The audit identifies no correction-specific defect requiring a proposal
change, and a fresh pass would not erase the failed cohort. The final curriculum
and need for a stronger claim remain open in #26. Spending another experiment
before choosing that claim would supply data without a settled decision target.

Broader acceptance therefore stays deferred. Reopen only for a concrete claim
beyond this lesson, such as conditional reliability over a named prior-predictive
population or formally accepted participant inference. Before fresh outcomes,
specify the population, acceptable failure frequency, relevant joint-distribution
metric if claimed, conditional/price tolerances, numerical-reference precision,
MC uncertainty rule and calibration multiplicity. Freeze sample size justified
by the desired precision, independent seeds, case selection, 2,048 proposals,
512 SIR draws if that output is assessed, metrics, aggregation and one stopping
rule in a new protocol. This decision intentionally supplies no seeds or
executable replication configuration; it cannot unlock a new acceptance run.

If such a study is later justified, retain every outcome in one new directory
and save proposal pairs, densities, weights, reference values, SIR indices,
truths and rank jitter sufficient for independent arithmetic reconstruction.
No automatic cohort expansion, seed search, retraining or repeat-until-pass
loop follows. Measure its actual cost; existing export timings are not a
new inference speed benchmark.

## Delivery and verification

[Notebook11](../notebooks/11_hkde_calibration_audit.py) now teaches the decision,
retained criteria and shared-alarm policy alongside the existing audit and
the original reference-PIT and target-price error graphs. It remains cached
review with a small analytic calculation, and starts no inference experiment.

Run the thin review check using the pinned environment:

```sh
PYTHONPATH=src pixi run python scripts/check_hkde_acceptance_review.py --out /tmp/hkde-acceptance-review
```

Choose a new output directory. The check verifies the full artifact manifest,
runs strict marimo checking and exports an isolated offline copy in a fresh
process. The [verification record](results/hkde-acceptance/verification.json)
records actual results, source identities and visual inspection. This reuses
an existing locked environment; it does not certify a clean installation.

Measured verification: **128 artifact hashes matched; 57 HKDE tests passed;
15 output cells, zero errors and five bound figures exported in 0.883 s**.
All five figure files were visually inspected, and the export contains those
same PNG bytes. Its visible provisional/budget/I4 text was checked in the
rendered session output; a full browser layout inspection was not performed.
The export guard blocked external Python socket/DNS access and scientific
imports while permitting local kernel IPC. This timing measures cached review,
not inference or a comparative speed benchmark.

No positive scientific-acceptance handoff is issued to I4. The data workstream
retains ownership of its local-input verification, and PR #22's notebook is
untouched. Review/merge integration remains #27; nothing is merged by this
decision. A completed #24 decision with provisional status is a valid result.

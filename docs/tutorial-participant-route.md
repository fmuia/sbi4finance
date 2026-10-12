# Selected participant route: synthetic HKDE options and diagnostic reasoning

**Decision recorded 12 October 2026, issue #26.** The user selected the
**Q/options route with a provisional HKDE diagnostic lesson**, following an
exact GBM foundation. The financial question is: *which parameter pairs explain
these nine option premiums, and what model prices do those pairs imply for two
excluded European puts?* The [#27 combined review](stack-integration-review.md) now includes the
selected source and presentation dependencies. Final participant delivery
verification is recorded against its pinned revision.

The scientific decision in [PR #29](https://github.com/fmuia/sbi4finance/pull/29),
at [45bb08d](https://github.com/fmuia/sbi4finance/blob/45bb08d2b091df7bae49fa250424e122f19e3eca/docs/hkde-acceptance-decision.md),
retains the frozen **2,048-proposal selected-case demonstration as provisional**.
It grants no formal scientific participant acceptance; **I4 remains deferred**.
This route uses that scope, not a new acceptance claim. PR #29's decision and
notebook changes are included in this combined checkout.

## Why this question, and what remains optional

| | P / Heston alternative | Selected Q / HKDE route |
|---|---|---|
| Financial question | What does a return history imply about parameters, today's hidden state and conditional future losses? | What do noisy option premiums imply about pricing parameters and excluded contracts' model prices? |
| Observation and context | Ordered decimal log returns `[batch,time,1]` and positive year-fraction increments | Nine currency premiums `[batch,9,1]`; fixed contracts, spot, carry, noise law and model context |
| Inferred quantities | Restricted pilot: long-run variance and signed terminal continuation state, kept paired | Long-run variance and annual jump intensity, kept paired with posterior weights |
| Participant output | Joint parameter/state plot and a ten-day model-based loss distribution | Joint weighted parameter plot, two model-price distributions and weighted 5/50/95% price quantiles |
| Additional concepts | Hidden-state filtering, full-truncation continuation and predictive mixtures | Risk-neutral expectation pricing, quote likelihood, importance correction and price-map error |
| Current limitation | Conditional/state-sensitive failures remain; the restricted pilot is not a full SVJ/SVCJ reproduction | Historical theta SBC gate failed; selected-case agreement and an accepted target surrogate do not accept the posterior |

The graph comparison was part of the review. Heston's
[prior/posterior graph](results/heston-diagnostic-review/prior-posterior-comparison.png)
shows substantial updating, especially for current state. Its
[conditional errors](results/heston-diagnostic-review/conditional-reference-errors.png)
still expose the calm-history state and upper future-loss-quantile failures;
the plotted bars use different reference-SD denominators in the two panels.
HKDE's [joint graph](results/hkde-live-demo/live-joint.png) shows the strong
variance/jump-intensity trade-off; its
[two target-price plots](results/hkde-live-demo/live-target-prices.png) show close
selected-case agreement. Its [reference/PIT graph](results/hkde-correction/plots/reference-pit.png)
shows a shared theta rank alarm, not a resolved population-calibration result.

Heston notebook05, the numerical model reviews06–08, and original MMAR notebook01
remain optional material. The rough held-out
[conditional-failure](https://github.com/fmuia/sbi4finance/blob/f1bfa72/docs/results/rough-heldout-test/plots/conditional-failure.png)
and [SBC](https://github.com/fmuia/sbi4finance/blob/f1bfa72/docs/results/rough-heldout-test/plots/marginal-sbc.png)
graphs are an optional diagnostic companion: marginal SBC passes while the
selected eta posterior is too narrow. Rough remains provisional after one of
seven neural gates failed; its 256-case test is used and closed. These optional
reviews are outside the two-hour route and its required runtime measurement.

## Learning objectives and 120-minute schedule

Participants should be able to identify observation/context/truth roles,
explain what a conditional density network learns, retain joint draws when
propagating uncertainty, distinguish numerical pricing from posterior inference,
and interpret conditional checks separately from calibration. No objective
requires demonstrating frequentist failure, superior tail-risk forecasts,
necessity of SBI, or empirical market adequacy.

The schedule selects sections from existing notebooks; it does not ask
participants to read every development experiment. Exercise time is included.
Setup is completed before the session. The named leads retain the original
schedule's subject allocation and can be reassigned by the team.

Here **GBM01** means [01_gbm_foundations.py](../notebooks/01_gbm_foundations.py),
**HKDE10** means [10_hkde_corrected_workflow.py](../notebooks/10_hkde_corrected_workflow.py),
and **HKDE11** means [11_hkde_calibration_audit.py](../notebooks/11_hkde_calibration_audit.py).
The data walkthrough and specified #22/#29 additions are included here.

| Elapsed | Topic and exact source sections | Activity / output | Lead |
|---|---|---|---|
| 0–10 min | Financial question and P/Q; GBM01 **M0 role records: two different inference questions** and HKDE10 **Observation, known context and unknown targets** | State the selected options question and label observation, context and output | Francesco |
| 10–30 min | GBM01 **An exact posterior to check later approximations**, **From a posterior to future outcomes**, **Separate Q reference: a payoff is not a price** | Change the history-length slider once; compare posterior uncertainty; E1 (4 min) | Stefan |
| 30–45 min | GBM01 **Training, saving and reloading are explicit steps** and **Historical result for increment 2**; HKDE's existing [workflow](../src/sbi4finance/hkde_inference.py) and [training configuration](../configs/hkde_inference_protocol.json) | Trace simulation → training pairs → adapter → conditional density → fixed checkpoint; distinguish amortized training from this lesson's likelihood correction | Alex / Stefan |
| 45–60 min | [Data walkthrough](https://github.com/fmuia/sbi4finance/blob/663ac4edeaf5935b59da5eb3d6767e13b0806037/notebooks/hkde_data_walkthrough.py), **1. What was observed**, **2. The inference boundary**, **3. Keep parameter draws paired**, **4. Price a contract outside the conditioning panel** | Inspect the local synthetic panel and preserved raw result; E2 (3 min) | Stefan |
| 60–75 min | HKDE10 **Why correct a neural posterior?**, **Finite particles remain an approximation** | Read the full-mixture weighting formula and distinguish the surrogate from the likelihood pricer; E3 (4 min) | Alex |
| 75–95 min | HKDE10 **Same observation, same target, different estimators** from #22; **One live synthetic panel**, **Keep the joint posterior when propagating prices**, **What was cached, and what did this machine compute?** | Compare the cached result; explicitly run the fixed 2,048-proposal demonstration or label the saved replay; read both target distributions; E4 (5 min) | Francesco / Alex |
| 95–113 min | HKDE10 **Conditional validation and the new independent test**, **SBC audits the actual finite-pool resampling algorithm**; HKDE11 **Two questions, two comparisons**, the shared-reference graph and #29 decision | Identify what passed and failed, and why the reference alarm does not settle the cause; E5 (6 min) | Alex |
| 113–120 min | HKDE10 **Review trail and exercises**; data walkthrough **What would real data add?** | Discuss a changed prior or panel and the S/M/E boundary; E6 (3 min) and participant exit statement | All |

The architecture segment has a concrete existing example: nine ordered prices
condition a two-dimensional spline coupling flow **directly, with no learned
summary encoder**. The frozen network has four coupling layers, dense subnets
of widths `[64,64]` and eight spline bins. The adapter applies bounded parameter
transforms and training-only global normalization; it returns paired raw-unit
parameters. The recorded fit used 80 epochs and batches of 128. Explain these
choices from the source/configuration; no training runs in the session. A
recurrent summary network is relevant to an ordered returns task, but is not
silently added to this fixed option panel.

## Bounded exercises and expected answers

| Exercise | Prompt, using the existing section/exercise | Expected answer |
|---|---|---|
| E1, 4 min | GBM01 exercises 1, 2 and 5: fix the seed, change 252 to 504 days, then reason about finer sampling within one year. Which quantity labels a pricing surrogate? | More elapsed time reduces drift variance; the realized mean need not move toward truth. Finer sampling at fixed elapsed time does not reduce this known-volatility drift variance. The pricing label is a sufficiently accurate expectation, not one realized payoff. |
| E2, 3 min | HKDE10 exercise 1/data walkthrough section2: mark the nine premiums, contract order, synthetic truth and two target prices as conditioning/context/evaluation. | Only the nine premiums enter the neural input; the complete fixed task supplies context. Truth, auxiliary timestamps and target prices stay outside conditioning. Changing a premium invalidates the old cached posterior. |
| E3, 4 min | HKDE10 exercises 2 and 3: a draw came from the 10% boundary component. Which density appears below its importance weight, and does high ESS prove accuracy? | The denominator is the full 80/10/10 proposal mixture; the numerator retains the uniform inference prior and numerical likelihood. ESS diagnoses weight concentration, not missed modes or posterior accuracy. Compare against the independent posterior grid. |
| E4, 5 min | HKDE10 exercises 4 and 5: use the joint graph to predict what shuffling only jump intensity would change. Explain a 90% target-price interval. | Shuffling destroys the variance/jump trade-off despite preserving marginals. Parameters, normalized weights and target-price rows must stay aligned. The interval describes posterior uncertainty in a Q expectation, not individual payoffs, future quotes or P loss risk. Fourier, surrogate and finite-pool errors remain separate. |
| E5, 6 min | HKDE10 exercise 7/HKDE11 exercise1: compare theta's SIR curve with the reference PIT curve, then contrast a conditional check with SBC. | Both show the historical alarm, which does not isolate correction-specific bias or erase the failed gate. Conditional accuracy concerns a specified panel; prior-predictive SBC aggregates across generated panels. Even a prior-only algorithm can pass a marginal rank check. The scope remains provisional and I4 deferred. |
| E6, 3 min | HKDE10 exercise6: change the prior, request an American option, or ask for tomorrow's losses. What must change? | A changed prior needs a compatible inference/weighting and support assessment; frozen results cannot simply be relabelled. American exercise needs a different supported pricing/task design. Physical loss forecasts need a P model and appropriate state inference; Q calibration supplies no P/Q link. No implementation or new experiment is assigned. |

The participant exit statement is one sentence naming the input, joint output,
downstream expectation, observed diagnostic failure and provisional scope.
Neither rerunning a test cohort nor perturbing the fixed panel is an exercise.

## Route card: task, assets and execution modes

| Item | Bound choice |
|---|---|
| Required scope | **S**: synthetic and offline after installation; no real observations or future market outcomes required |
| Financial task | `hkde_option_panel_v1`, Q, [task JSON](../configs/tasks/hkde_inference_bank.json); independent uniform theta `[0.025,0.065]` and annual jump intensity `[0.25,1]` |
| Input | `validation:257`, nine ordered European put premiums: strikes 90/100/110 within maturities .25/.5/1 year; spot 100, rate .02, dividend 0, quote-noise SD .25 currency; all remaining parameters and initial variance fixed by task |
| Local input dependency | #22's `data/synthetic/hkde-walkthrough/{bundle.json,task.json,posterior.npz,panel/observations.csv,panel/manifest.json}` and `src/sbi4finance/datasets/hkde_teaching.py`; the loader checks full task, order, roles and hashes |
| Checkpoint | [hkde-theta-intensity-provisional](../checkpoints/hkde-theta-intensity-provisional/README.md): `model.keras`, `task.json`, `training.json`; unchanged raw-flow failure remains visible |
| Correction | [Frozen protocol](../configs/hkde_correction_boundary_protocol.json), experiment `hkde_boundary_defensive_importance_v2`; 2,048 proposals; fixed selected-case seed `3667540760`; 80% frozen flow / 10% prior / 10% product-arcsine boundary proposal |
| Likelihood and reference | Nine Fourier prices per row at 1,024 nodes / upper limit 400; `docs/results/hkde-inference/{reference.json,reference-grid.npz,neural-validation.json,validation-checks.npz}`; numerical posterior grid has 129 points per parameter |
| Forward outputs | `docs/results/hkde-surrogate/{model.npz,report.json}` prices only the supported European puts `(K=95,T=.75)` and `(K=80,T=1)`; it does not replace the nine-contract likelihood calculation |
| Preserved evidence | `docs/results/hkde-correction/{validation.json,test.json,validation-checks.npz,test-checks.npz}` and `hkde-correction-v1/validation.json`; gate-audit report/checks/figures, linked live figures and source/artifact bindings remain available |
| Scientific decision dependency | #29 at 45bb08d: `docs/hkde-acceptance-decision.md`, `docs/results/hkde-acceptance/{decision.json,artifact-manifest.json}` and extended notebook11; included in the #27 combined tree |
| Runtime | Locked Pixi CPU environment: Python 3.12, BayesFlow 2.0.14, Keras 3.15.1, JAX 0.11.1; `KERAS_BACKEND=jax`, `PYTHONPATH=src` tied to the actual checkout. No GPU or cluster access needed. Platform and actual memory/runtime must be recorded at final verification. |

The compact table identifies teaching assets, not a replacement integrity
manifest. Existing loaders verify their complete source/artifact bindings;
do not prune dependencies solely because a filename is absent from this card.
Case checks retain the six frozen conditional tolerances in the correction
protocol, including maximum target-quantile error 0.03 currency, ESS fraction
at least 0.5 and maximum normalized weight at most 0.01. A selected-case pass
does not override the historical theta calibration failure.

| Calculation | Intended participant behavior on the #27 combined tree |
|---|---|
| GBM01 illustrations | Live bounded simulation, exact inference, future sampling and European expectation checks. The notebook also creates, loads and deletes a tiny 64/16/8 bank in a temporary directory; no pre-existing raw training bank is required. |
| GBM training / cluster sections | Cached evidence and displayed code only; do not execute the reproduction commands during the route. |
| Data walkthrough | Cached raw 4,096 paired neural draws and their numerical target prices, loaded from the local synthetic bundle; no checkpoint inference. |
| HKDE10 default | Preserve #22's **cached review** behavior, local input validation and raw-versus-corrected comparison; the corrected cache has **8,192 precision proposals**. Do not label it a fresh 2,048-proposal run. |
| HKDE10 opt-in | `demo-hkde-corrected` explicitly enables the provisional 2,048-proposal calculation, including load, proposal sampling/density, nine numerical likelihood prices, weights, two surrogate outputs and comparison with the saved grid. No retries or changed seed/budget. |
| Diagnostic lesson | Cached historical cohort/reference evidence; notebook11 also runs its small analytic rank-variance illustration. No new panels, training or acceptance cohorts. |
| HTML export | Fresh notebook execution and rendering, measured separately from live calculation. Exported HTML is a review artifact; Python-backed controls use marimo. |

If live execution is omitted during a session, use the saved
[joint](results/hkde-live-demo/live-joint.png) and
[price](results/hkde-live-demo/live-target-prices.png) figures and identify their
provenance explicitly. A saved replay remains useful, but is not a successful
execution on the participant's machine. The final delivery check must exercise
both intended cached and opt-in modes.

## Cost evidence and delivery acceptance still required

Current PR #19 notebook10 is **live by default**. Its current-tree preflight is
useful but does not verify the proposed #22 cached-default/local-input behavior.
The target is #27's explicit composition of these changes and #29's scientific
decision; no combination of separate-checkout successes certifies that tree.

Historical PR #19 [execution](results/hkde-live-demo/verification.json) took
13.50 seconds for a fresh-process export and 5.37 seconds within the live cell.
That cell separately recorded load/import 2.56 s, proposal sample .884 s,
density .155 s, total proposal 1.039 s, numerical likelihood 1.767 s,
two-target surrogate .00062 s, and comparison/summary .00436 s.
These figures describe one recorded machine/run. Loading includes framework
initialization; proposal total overlaps sampling/density. They must not be
summed with the export or with timings from other checkouts. #22 separately
records [5.06 seconds live / 14.60 seconds for its isolated two-notebook check](https://github.com/fmuia/sbi4finance/blob/663ac4edeaf5935b59da5eb3d6767e13b0806037/docs/hkde-provisional-demo.md).
Neither is a promised participant time or an amortization speedup benchmark.

The final check must pin the combined commit, install its locked environment
fresh (package networking permitted for installation), then execute all four
selected notebooks in fresh processes with external networking blocked while
permitting local marimo kernel communication. Raw training banks and every
real-market fixture must be absent; required synthetic assets/checkpoints must
remain. Record cached and opt-in behavior, stage and whole-route export costs,
cell errors, exact versions/hashes, every rendered graph and exercise review,
and any platform limitation. Exercise review checks E1's slider change and the
answer interpretations; it does not run new scientific experiments. Perform
this after all executable or figure-affecting lesson edits.

The delivery record must distinguish **setup success**, **offline execution**,
**selected-case numerical checks** and **scientific status**. A completed
provisional tutorial can retain `scientific_acceptance=false`; it cannot
silently mark formal I4 complete. Optional real conditioning (**M**) and held-out
market assessment (**E**) require compatible separate data/designs and must
skip visibly when absent. No provider acquisition, new cluster campaign,
retraining, reserved-test use or P/Q linkage is part of this route.

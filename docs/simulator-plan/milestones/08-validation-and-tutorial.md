# M8 Accumulated validation and a complete teaching path

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** M0, M1, accepted local [M2.1–M2.3](02-banks-and-cluster.md#ordered-implementation-subplan), and the selected scientific/inference substeps. Start the evidence sheet alongside that branch; final delivery requires its checks to pass. Cluster execution is accepted separately at M2.4. Completion of every model branch is not required. Read [inference and pricing](../notes/03-inference-and-pricing.md), [software and data](../notes/04-software-and-data.md), and the [roadmap](../README.md).

## Purpose

**Current boundary:** one bounded HKDE options candidate is being assembled
as **[draft PR #19, 10/10](https://github.com/fmuia/sbi4finance/pull/19)** on `feat/hkde-offline-tutorial`, stacked on
[PR #15](https://github.com/fmuia/sbi4finance/pull/15). Its
[candidate specification](https://github.com/fmuia/sbi4finance/blob/feat/hkde-offline-tutorial/docs/m8-options-candidate.md)
and [notebook10](https://github.com/fmuia/sbi4finance/blob/feat/hkde-offline-tutorial/notebooks/10_hkde_corrected_workflow.py)
target a complete 30-minute S lesson. Corrected development-validation passes,
but the new held-out test fails theta SBC. The draft reviews this retained
stop; its live candidate path remains disabled and M8 scientific acceptance
remains pending. Preparing this candidate does not select the final
returns-versus-options curriculum, replace MMAR, or complete unresolved branches.

Actual option-bank cluster execution is now measured for all three 24-panel
tasks; same-Linux selected-ID reproduction passed. The
[cluster report](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/option-cluster-smoke.md)
also records matched hashes for all 15 transferred files and passed portable
regeneration, including the initial HKDE comparison failure and the documented
tolerance adjustment. These storage/portability checks remain separate from
scientific accuracy.
The first [restricted HKDE experiment](https://github.com/fmuia/sbi4finance/blob/feat/hkde-inference/docs/hkde-joint-inference.md) now
implements that reference/inference path and its review. Parameter and marginal
SBC checks pass, but the target-price quantile gate fails (0.0439 versus 0.03
currency). Its checkpoint remains provisional and final test unused. A verified
review of this outcome does not accept a participant inference workflow;
M5.1's raw posterior remains unaccepted. A separately evaluated corrected
pipeline can qualify on its own declared evidence; it cannot rewrite that
result. Bergomi's remaining label-bias work does not block the
HKDE experiment. No participant acceptance follows from the
cluster runs or provisional checkpoint, and the recorded Heston failures remain unresolved.
Market acquisition cannot block the required offline S delivery.

The [rough H/eta pilot](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/rough-joint-inference.md)
adds saved synthetic banks and a reference audit: 19/20 panels pass, but one
misses the frozen ESS and maximum-weight gates. Neural training and SBC were
not run; the final test stays reserved. Notebook09 is a didactic review of this
reference limitation, not an accepted participant checkpoint. Its next
reference experiment remains separate from M8 selection.

The [M5.2 surrogate](https://github.com/fmuia/sbi4finance/blob/feat/hkde-inference/docs/hkde-price-surrogate.md)
already passes its independent fixed-contract pricing checks. The new
[M5.1f correction subplan](05-amortized-pricing.md#m51f-bounded-likelihood-correction-follow-up)
addresses inference with the frozen checkpoint as a proposal. Its first
90% flow / 10% uniform mixture passed all conditional accuracy gates but
stopped on one boundary-region importance-weight reliability check; no SBC or
new test was run. One frozen 80% flow / 10% uniform / 10% product-arcsine
follow-up now passes every validation gate at both conditional budgets and
the 512-case finite-pool SIR audit. The target prior stays uniform and all gates
stay fixed. Its independent 256-panel test passes all 16 conditional cases at
both budgets and every reliability check, but theta's SIR rank KS 0.103253
exceeds the frozen 0.092513 DKW limit. Lambda's 0.072029 is below it. The test
is not accepted, and no retry or tuning follows. A validation pass or an
informative review notebook cannot qualify the participant artifact.

An [independent reference-PIT diagnostic](https://github.com/fmuia/sbi4finance/blob/feat/hkde-inference/docs/results/hkde-correction/test-independent-audit.json)
finds the same finite-cohort theta shift in the refined numerical posterior
(KS 0.103485; 65-point check 0.103404) as in SIR (0.103253). This is useful
material for the lesson: a failed rank gate does not by itself isolate neural
or proposal bias, and this check does not indicate further proposal tuning.
The numerical reference is not an analytic oracle. The post-hoc diagnosis
changes neither the frozen failure nor the closed test or M8 acceptance status.

Assemble evidence already collected at each milestone into a complete S participant workflow. Validation and didactic review begin with M0 contracts and M1 analytic checks; this milestone closes end-to-end gaps rather than postponing either. The earlier review notebooks explain each development increment, including failed experiments; M8 selects and paces a scientifically accepted application for participants. M is an optional market illustration and E is a separate empirical extension; neither is required for S acceptance.

## Tasks

1. Maintain an evidence matrix separating simulator/numerical correctness, dataset integrity, synthetic inference/prediction accuracy, optional M case studies, and E empirical findings. Reference accepted fixtures and reports; record data origin, available truth, evidence scope, unresolved failures, and unsupported parameter regions.
2. Select one teaching task based on scientific relevance, conditional validation, and measured participant cost. Choose either returns → parameter/state posterior → conditional losses, or option panel → Q posterior → downstream price uncertainty. Reuse accepted explanations and demonstrations from the milestone review notebooks; their existence does not select the curriculum. Keep alternative branches as optional material.
3. Reproduce the complete path from immutable configuration and bank manifest to training, checkpoint reload, adapter application, posterior draws, and the selected financial output. Cache expensive reference calculations with provenance.
4. Combine prior-predictive checks, SBC, parameter recovery, conditional posterior comparisons, joint/state-sensitive functionals, and predictive checks. Where reference MCMC is used, report its convergence and sampling uncertainty. Marginal SBC success cannot overrule an unresolved conditional discrepancy.
5. Run the entire selected S notebook offline on the intended CPU environment with all real-market fixtures absent and provider access disabled. Include generated context, inference, and downstream uncertainty. Measure data loading, checkpoint inference, and forward calculation together; distinguish precomputed synthetic references/checkpoints from participant calculations.
6. Put M/E cells behind explicit optional data-availability gates. M normally applies frozen weights without fabricated truth labels. E evaluates P forecasts against separately held-out later outcomes, or Q prices against same-time excluded target quotes, also excluded from smoothing/input construction. A market fit alone is not E. Record chronological dependence and the tested dates/contracts.

## Ordered implementation subplan

Define M8.1 before the selected branch's final training/evaluation; complete M8.2–M8.4 in order once the relevant model and inference outputs are available. Acceptance is evidenced in the implementation PRs. A returns release uses M3.3. An HKDE options release requires an accepted M5.1 estimator, including M5.1f's explicit likelihood-correction pipeline if used, additionally M5.2 for a price surrogate and M5.3 for direct price inference when those outputs are included. Acceptance of a corrected estimator does not require relabelling its raw proposal checkpoint as an accepted posterior. A rough-Bergomi inference release uses M7.3. Optional extensions need their own accepted results only when included in the released workflow.

| Step | Concrete output | Verification/handoff |
|---|---|---|
| M8.1 — Task and evidence sheet | A named teaching configuration, required outputs, supported domain, independent case bank and predeclared metrics/tolerances, with S/M/E scope and a reference strategy. | Trace every promised output to its model/inference acceptance gate. Specify joint/state-sensitive and difficult-region cases before final evaluation; keep unavailable M/E evidence optional. |
| M8.2 — Reproducible inference and output | A versioned configuration → bank → training → checkpoint reload → posterior → forecast/price run, with reference results, numerical error and sampling uncertainty. | Reproduce the selected task's conditional/joint checks and financial outputs, not only training loss or marginal SBC. Resolve material discrepancies or explicitly narrow the supported scope before teaching it. |
| M8.3 — Offline participant notebook | One compact marimo notebook in `notebooks/`, reusing selected milestone explanations, synthetic fixtures, compatible checkpoint, exercises and a measured CPU run. Record which expensive calculations are cached and how they were produced. | Execute headlessly and export to HTML in a clean supported installation with all market fixtures absent and provider/network access disabled. Inspect the complete chosen S workflow and measure loading, inference and downstream calculation together. No expensive job starts automatically; optional M/E cells skip visibly when unavailable. |
| M8.4 — Acceptance report and handover | A checkpoint card, links to exact artifacts and checks, reproducible run instructions, limitations and remaining work. Keep reports with the implementation documentation; large banks stay in their declared artifact storage. | Mark only verified substeps accepted. Record M2.4 separately if cluster execution is pending, preserve the open status of unused branches, and ensure every teaching claim stays within its numerical, synthetic or optional empirical evidence. |

The mathematical explanations come from the existing notes and milestone review tutorials; this milestone turns one accepted branch into a paced participant lesson rather than requiring all models in one notebook.

## Bounded options candidate: concrete M8 subplans

Use the [frozen boundary-correction task](https://github.com/fmuia/sbi4finance/blob/feat/hkde-inference/configs/hkde_correction_boundary_protocol.json),
without adding a model, training campaign, market provider or new framework.
These are implementations of M8.1–M8.4, not additional PRs or an automatic
curriculum decision.

| Step | Candidate-specific output | Completion evidence still required |
|---|---|---|
| M8.1 — Freeze task and roles | Nine noisy synthetic European-put prices under Q; joint theta/jump-intensity targets; known remaining parameters/context; two excluded put-price outputs; uniform prior and a separately declared three-component proposal | Preserve the original raw-neural failure and v1 reliability stop. Bind the new estimator to protocol, checkpoint, likelihood, surrogate and source hashes. Only S is promised; truths and clean prices never enter observed conditioning. |
| M8.2 — Qualify the corrected pipeline | M5.1f weighted joint inference followed by the accepted M5.2 two-price surrogate; both 2,048 and 8,192 conditional checks, finite-pool SIR calibration, and the new root-95026 test | Every frozen validation gate passes before generating the new test; the test passes without selecting another candidate. The original root-62026 test stays untouched. No acceptance transfers to the raw flow, Heston or rough Bergomi. |
| M8.3 — Execute a 30-minute lesson | Notebook10: task/data roles (5 minutes), importance correction (8), one live posterior/price calculation (7), diagnostic interpretation (7), exercises (3) | Fresh-process headless HTML export with external sockets blocked, market fixtures and raw training banks absent; inspect the rendered graphs and tables. Actually execute the 2,048-proposal calculation on the named saved validation panel. A notebook that skips live work because evidence is missing does not pass. |
| M8.4 — Record and hand over | Exact science and notebook reports, review commands, artifact links, a pipeline/checkpoint card, supported-domain limits and remaining work | Report loading, proposal sampling/density, nine-price numerical likelihood, two-price surrogate and reference comparison, with live versus cached work explicit. Keep self-normalized/SIR approximation, prior-versus-proposal distinction, unchanged failed predecessors and final curriculum choice visible. |

Training, base-bank generation, accepted grid pricing and the scientific
audits remain cached. Opening the notebook starts no such jobs or provider
access. The live calculation retains matched parameter pairs and weights;
its output is parameter-induced uncertainty in Q expectation prices, not
realized payoff risk or a prediction of tomorrow's option chain. A live
finite-pool miss is displayed without a new seed or automatic retry. Scientific
and didactic outcomes must both be recorded before calling this bounded
candidate accepted. **Development-validation has passed; the independent test
failed theta SBC, so scientific participant acceptance remains pending.** The
10/10 draft now passes its
[source-bound review-only offline export](https://github.com/fmuia/sbi4finance/blob/feat/hkde-offline-tutorial/docs/results/hkde-correction/notebook-verification.json):
17 cells, zero errors, 8.2935 seconds, external sockets blocked, raw training
banks and market fixtures absent. It shows the recorded failure and correctly
skips live candidate execution. This verifies the review lesson and its gate
handling; it does not satisfy the complete live M8 workflow.

The [combined-stack verification](https://github.com/fmuia/sbi4finance/blob/feat/hkde-offline-tutorial/docs/results/hkde-correction/combined-stack-verification.json)
uses a clean nonmutating merge-tree of this M8/PR #15 stack and PR #14, passing
171 software tests in 25.263 seconds. PR #15 alone passes 133 tests in
21.604 seconds. Neither software result changes the scientific stop, the
closed test or final curriculum choice.

## Didactic review deliverable

Planned with M8: deliver the selected participant marimo notebook and a short release-review section or companion report tracing its claims to the prior milestone artifacts. Follow the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone), including motivation/math, explicit observation/context/target units, a bounded runnable calculation, actual results, provenance and short exercises. Keep the returns-versus-options choice open until selection. Review notebooks containing provisional checkpoints are useful development material but do not qualify those checkpoints for the accepted participant inference path.

Accept the participant artifact after a fresh-process offline headless run and inspected HTML export, with every real-market fixture absent, no automatic expensive jobs and measured end-to-end CPU cost. Check that its exercises, caveats and displayed pass/fail evidence agree with the selected scientific gates. This does not retroactively complete missing review sections for unused model branches.

## Deliverables

An S acceptance report tied to exact versions, a verified compact participant notebook and release-review material, reproducible synthetic reference assets, a checkpoint card, and separately labelled optional M/E reports and deferred extensions. Preserve the existing MMAR example unless an explicit curriculum decision replaces it.

## Acceptance and tests

Apply predeclared tolerances, including difficult-region and higher-accuracy checks. Confirm independent base splits, no target/future-outcome leakage, valid posterior dependence, correct predictive-mixture versus conditional-risk calculations, and checkpoint reloads. For options, propagate pricing errors; for returns, validate inferred-state continuation. Require the full S workflow to pass without providers or cached market files. Missing M/E assets must skip visibly, never produce synthetic results labelled real. Synthetic forecasts and pricing checks remain model-conditional evidence, not empirical backtests.

## Decision gate and deferred scope

The [M5.2 fixed-contract surrogate](https://github.com/fmuia/sbi4finance/blob/feat/hkde-inference/docs/hkde-price-surrogate.md)
passes its independent pricing checks and has a didactic review. The original
HKDE neural posterior still fails its price-quantile gate, so a faster price map
does not accept the combined M8 workflow. Its new regression test is independent
of the reserved posterior test. Curriculum selection remains open.

Release the selected S track when its material failures are resolved or its supported scope is explicit. Market acquisition/permissions never block that release, and passing S does not imply M/E success. Avoid predetermined claims that frequentist methods fail, Bayes always raises risk estimates, or SBI is necessary. Additional models, scaling, and curriculum replacement remain separate decisions; no unmeasured runtime or speedup becomes a tutorial promise.

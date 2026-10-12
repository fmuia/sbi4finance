# M8 Accumulated validation and a complete teaching path

**Status:** the [HKDE options candidate](../../m8-options-candidate.md) runs an explicitly provisional live demonstration on fixed synthetic `validation:257` with 2,048 proposals and unchanged seed/tolerances. Its used, closed test failed theta SBC; M8 scientific acceptance remains pending. The frozen study stopped without tuning or a rerun. The final curriculum choice is open. Track implementation evidence and didactic completion separately. **Depends on:** M0, M1, accepted local [M2.1–M2.3](02-banks-and-cluster.md#ordered-implementation-subplan), and the selected scientific/inference substeps. Start the evidence sheet alongside that branch; final delivery requires its checks to pass. Cluster execution is accepted separately at M2.4. Completion of every model branch is not required. Read [inference and pricing](../notes/03-inference-and-pricing.md), [software and data](../notes/04-software-and-data.md), and the [roadmap](../README.md).

## Purpose

**Current boundary:** the [actual three-model cluster smoke](../../option-cluster-smoke.md)
and offline marimo reviews establish the tested generation/retrieval route.
They do not complete M8 or accept a new posterior. The current candidate follows
the [restricted HKDE subplan](05-amortized-pricing.md#next-increment-restricted-hkde-baseline)
and its measured follow-ups below. Keep Heston's failures, rough Bergomi's
unresolved reference gate and numerical fidelity limits visible. The final
returns-versus-options curriculum remains open; real-market acquisition cannot
block S delivery.

The first [HKDE baseline](../../hkde-joint-inference.md) is now available for
review. Its numerical reference, parameter checks and marginal SBC pass,
but the downstream price-quantile gate fails. Its verified review notebook
is development teaching material; it does not make the provisional checkpoint
an accepted participant inference path. Final-test evaluation remains reserved.

The [M5.2 fixed-contract surrogate](../../hkde-price-surrogate.md) passes its
independent pricing checks and has a didactic review. The original HKDE neural
posterior still fails its price-quantile gate, so a faster price map does not
accept the combined M8 workflow. Its new regression test is independent of the
reserved posterior test. Curriculum selection remains open.

The [bounded likelihood-correction follow-up](../../hkde-corrected-inference.md)
uses the unchanged network as a proposal, not an accepted standalone posterior.
Its first run passed conditional accuracy but stopped on proposal reliability;
SBC and the new test were not run. The [frozen boundary proposal](../../hkde-correction-boundary-protocol.md)
adds symmetric boundary coverage without changing the prior, likelihood,
checkpoint, seeds, budgets or thresholds. Its 512-case validation passed, then
the new 256-case test passed conditional and reliability checks but failed theta
SIR/SBC (KS 0.103253 > 0.092513). [Notebook 10](../../../notebooks/10_hkde_corrected_workflow.py)
preserves that frozen evidence while running one provisional live development
example. No tuning or test rerun followed. This pending path neither selects the
curriculum nor accepts the Heston or rough-Bergomi inference branches.
The independent-grid PIT also fails theta on the same panels (0.103485), so
the calibration diagnostic does not isolate bias from the correction method.
The frozen acceptance decision remains failed. The [saved-cohort audit](../../hkde-calibration-gate-audit.md)
supports continuing the bounded demonstration without attributing the shared
rank alarm to the correction alone; it does not establish population calibration.


Assemble evidence already collected at each milestone into a complete S participant workflow. Validation and didactic review begin with M0 contracts and M1 analytic checks; this milestone closes end-to-end gaps rather than postponing either. The earlier review notebooks explain each development increment, including failed experiments; M8 selects and paces a scientifically accepted application for participants. M is an optional market illustration and E is a separate empirical extension; neither is required for S acceptance.

## Tasks

1. Maintain an evidence matrix separating simulator/numerical correctness, dataset integrity, synthetic inference/prediction accuracy, optional M case studies, and E empirical findings. Reference accepted fixtures and reports; record data origin, available truth, evidence scope, unresolved failures, and unsupported parameter regions.
2. Select one teaching task based on scientific relevance, conditional validation, and measured participant cost. Choose either returns → parameter/state posterior → conditional losses, or option panel → Q posterior → downstream price uncertainty. Reuse accepted explanations and demonstrations from the milestone review notebooks; their existence does not select the curriculum. Keep alternative branches as optional material.
3. Reproduce the complete path from immutable configuration and bank manifest to training, checkpoint reload, adapter application, posterior draws, and the selected financial output. Cache expensive reference calculations with provenance.
4. Combine prior-predictive checks, SBC, parameter recovery, conditional posterior comparisons, joint/state-sensitive functionals, and predictive checks. Where reference MCMC is used, report its convergence and sampling uncertainty. Marginal SBC success cannot overrule an unresolved conditional discrepancy.
5. Run the entire selected S notebook offline on the intended CPU environment with all real-market fixtures absent and provider access disabled. Include generated context, inference, and downstream uncertainty. Measure data loading, checkpoint inference, and forward calculation together; distinguish precomputed synthetic references/checkpoints from participant calculations.
6. Put M/E cells behind explicit optional data-availability gates. M normally applies frozen weights without fabricated truth labels. E evaluates P forecasts against separately held-out later outcomes, or Q prices against same-time excluded target quotes, also excluded from smoothing/input construction. A market fit alone is not E. Record chronological dependence and the tested dates/contracts.

## Ordered implementation subplan

Define M8.1 before the selected branch's final training/evaluation; complete M8.2–M8.4 in order once the relevant model and inference outputs are available. Acceptance is evidenced in the implementation PRs. A returns release uses M3.3. An HKDE options release requires M5.1, additionally M5.2 for a price surrogate and M5.3 for direct price inference when those outputs are included. A rough-Bergomi inference release uses M7.3. Optional extensions need their own accepted results only when included in the released workflow.

| Step | Concrete output | Verification/handoff |
|---|---|---|
| M8.1 — Task and evidence sheet | **Specified:** [named HKDE candidate](../../m8-options-candidate.md#m81--named-task-and-evidence-boundary), nine Q quotes, two joint parameters and two excluded put-price distributions. [Frozen settings](../../../configs/hkde_correction_boundary_protocol.json) bind the prior, numerical law, references, case splits and unchanged thresholds. | Trace each output to its numerical, conditional/joint or price gate. Keep synthetic truth out of conditioning. S is required; M/E are optional and supply no acceptance shortcut. |
| M8.2 — Reproducible inference and output | **Implemented; frozen test failed theta SBC:** existing bank/training/checkpoint → full-mixture proposals → numerical-likelihood weights → accepted target-price surrogate. [Correction evidence](../../hkde-corrected-inference.md) retains passed validation, the new-test failure, source/artifact bindings and failed predecessors. | Conditional checks at both budgets and reliability passed; they do not override the failed rank gate. The separate 256-panel test is evaluated and frozen. The original neural final test remains unused. No tuning or rerun follows this outcome. Scientific acceptance remains pending. |
| M8.3 — Offline participant notebook | **Provisional live demonstration implemented:** [notebook 10](../../../notebooks/10_hkde_corrected_workflow.py), a paced candidate with equations, input/target roles, joint correction, target-price distributions, cached diagnostics and exercises. Default execution uses fixed `validation:257`, 2,048 proposals and unchanged seed/tolerances. | Run `PYTHONPATH=src pixi run review-hkde-corrected`. The [isolated live export](../../results/hkde-live-demo/verification.json) passes: 17 cells, zero errors, 13.50 seconds overall, including 5.37 seconds live. Market fixtures/raw banks are absent and external networking is blocked. The single-case numerical checks pass (price-quantile error 0.00530 < 0.03, ESS fraction 0.688, maximum weight 0.00208); scientific acceptance remains false. No training or cluster job starts. |
| M8.4 — Acceptance report and handover | **Runnable draft handover; scientific acceptance pending:** [candidate handover](../../m8-options-candidate.md#m84--handover-and-open-work), links to immutable evidence/checkpoint metadata, separate live verification, commands, supported domain and remaining work. | Technical execution and single-case checks are separate from M8.2 scientific acceptance. A provisional live demonstration does not override the used test or establish population calibration. Preserve raw-flow/v1/test failures, Heston/rough-Bergomi pending work and the open curriculum choice. Keep MMAR; empirical claims and broader products need separate evidence. |

The mathematical explanations come from the existing notes and milestone review tutorials; this milestone turns one accepted branch into a paced participant lesson rather than requiring all models in one notebook.

## Didactic review deliverable

The [implemented candidate notebook](../../../notebooks/10_hkde_corrected_workflow.py) and [handover subplan](../../m8-options-candidate.md) trace claims to the prior milestone artifacts. They follow the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone), including motivation/math, explicit observation/context/target units, a bounded runnable calculation, actual results, provenance and short exercises. Its review explains both accepted numerical components and failed inference experiments; successful offline execution does not clear pending scientific acceptance. Keep the returns-versus-options choice open until selection. A corrected pipeline may use the provisional flow as a proposal without accepting it as a standalone posterior.

Accept the participant artifact after a fresh-process offline headless run and inspected HTML export, with every real-market fixture absent, no automatic expensive jobs and measured end-to-end CPU cost. Check that its exercises, caveats and displayed pass/fail evidence agree with the selected scientific gates. This does not retroactively complete missing review sections for unused model branches.

## Deliverables

An S acceptance report tied to exact versions, a verified compact participant notebook and release-review material, reproducible synthetic reference assets, a checkpoint card, and separately labelled optional M/E reports and deferred extensions. Preserve the existing MMAR example unless an explicit curriculum decision replaces it.

## Acceptance and tests

Apply predeclared tolerances, including difficult-region and higher-accuracy checks. Confirm independent base splits, no target/future-outcome leakage, valid posterior dependence, correct predictive-mixture versus conditional-risk calculations, and checkpoint reloads. For options, propagate pricing errors; for returns, validate inferred-state continuation. Require the full S workflow to pass without providers or cached market files. Missing M/E assets must skip visibly, never produce synthetic results labelled real. Synthetic forecasts and pricing checks remain model-conditional evidence, not empirical backtests.

## Decision gate and deferred scope

Release the selected S track when its material failures are resolved or its supported scope is explicit. Market acquisition/permissions never block that release, and passing S does not imply M/E success. Avoid predetermined claims that frequentist methods fail, Bayes always raises risk estimates, or SBI is necessary. Additional models, scaling, and curriculum replacement remain separate decisions; no unmeasured runtime or speedup becomes a tutorial promise.

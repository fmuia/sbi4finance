# M8 Accumulated validation and a complete teaching path

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** M0, M1, accepted local [M2.1–M2.3](02-banks-and-cluster.md#ordered-implementation-subplan), and the selected scientific/inference substeps. Start the evidence sheet alongside that branch; final delivery requires its checks to pass. Cluster execution is accepted separately at M2.4. Completion of every model branch is not required. Read [inference and pricing](../notes/03-inference-and-pricing.md), [software and data](../notes/04-software-and-data.md), and the [roadmap](../README.md).

## Purpose

**Current boundary:** the [actual three-model cluster smoke](../../option-cluster-smoke.md)
and offline marimo reviews establish the tested generation/retrieval route.
They do not complete M8 or accept a new posterior. Next follow the
[restricted HKDE subplan](05-amortized-pricing.md#next-increment-restricted-hkde-baseline):
reference refinement, one modest baseline, conditional/joint and price checks,
and its offline review. Keep Heston's failures and Bergomi's fidelity limits
visible. The final returns-versus-options curriculum remains open; real-market
acquisition cannot block S delivery.

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
| M8.1 — Task and evidence sheet | A named teaching configuration, required outputs, supported domain, independent case bank and predeclared metrics/tolerances, with S/M/E scope and a reference strategy. | Trace every promised output to its model/inference acceptance gate. Specify joint/state-sensitive and difficult-region cases before final evaluation; keep unavailable M/E evidence optional. |
| M8.2 — Reproducible inference and output | A versioned configuration → bank → training → checkpoint reload → posterior → forecast/price run, with reference results, numerical error and sampling uncertainty. | Reproduce the selected task's conditional/joint checks and financial outputs, not only training loss or marginal SBC. Resolve material discrepancies or explicitly narrow the supported scope before teaching it. |
| M8.3 — Offline participant notebook | One compact marimo notebook in `notebooks/`, reusing selected milestone explanations, synthetic fixtures, compatible checkpoint, exercises and a measured CPU run. Record which expensive calculations are cached and how they were produced. | Execute headlessly and export to HTML in a clean supported installation with all market fixtures absent and provider/network access disabled. Inspect the complete chosen S workflow and measure loading, inference and downstream calculation together. No expensive job starts automatically; optional M/E cells skip visibly when unavailable. |
| M8.4 — Acceptance report and handover | A checkpoint card, links to exact artifacts and checks, reproducible run instructions, limitations and remaining work. Keep reports with the implementation documentation; large banks stay in their declared artifact storage. | Mark only verified substeps accepted. Record M2.4 separately if cluster execution is pending, preserve the open status of unused branches, and ensure every teaching claim stays within its numerical, synthetic or optional empirical evidence. |

The mathematical explanations come from the existing notes and milestone review tutorials; this milestone turns one accepted branch into a paced participant lesson rather than requiring all models in one notebook.

## Didactic review deliverable

Planned with M8: deliver the selected participant marimo notebook and a short release-review section or companion report tracing its claims to the prior milestone artifacts. Follow the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone), including motivation/math, explicit observation/context/target units, a bounded runnable calculation, actual results, provenance and short exercises. Keep the returns-versus-options choice open until selection. Review notebooks containing provisional checkpoints are useful development material but do not qualify those checkpoints for the accepted participant inference path.

Accept the participant artifact after a fresh-process offline headless run and inspected HTML export, with every real-market fixture absent, no automatic expensive jobs and measured end-to-end CPU cost. Check that its exercises, caveats and displayed pass/fail evidence agree with the selected scientific gates. This does not retroactively complete missing review sections for unused model branches.

## Deliverables

An S acceptance report tied to exact versions, a verified compact participant notebook and release-review material, reproducible synthetic reference assets, a checkpoint card, and separately labelled optional M/E reports and deferred extensions. Preserve the existing MMAR example unless an explicit curriculum decision replaces it.

## Acceptance and tests

Apply predeclared tolerances, including difficult-region and higher-accuracy checks. Confirm independent base splits, no target/future-outcome leakage, valid posterior dependence, correct predictive-mixture versus conditional-risk calculations, and checkpoint reloads. For options, propagate pricing errors; for returns, validate inferred-state continuation. Require the full S workflow to pass without providers or cached market files. Missing M/E assets must skip visibly, never produce synthetic results labelled real. Synthetic forecasts and pricing checks remain model-conditional evidence, not empirical backtests.

## Decision gate and deferred scope

Release the selected S track when its material failures are resolved or its supported scope is explicit. Market acquisition/permissions never block that release, and passing S does not imply M/E success. Avoid predetermined claims that frequentist methods fail, Bayes always raises risk estimates, or SBI is necessary. Additional models, scaling, and curriculum replacement remain separate decisions; no unmeasured runtime or speedup becomes a tutorial promise.

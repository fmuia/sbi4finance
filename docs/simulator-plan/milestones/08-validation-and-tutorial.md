# M8 Accumulated validation and a complete teaching path

**Status:** proposed; implementation starts after review. **Depends on:** M0–M2 and at least one accepted scientific/inference track selected for delivery. Completion of every model branch is not required. Read [inference and pricing](../notes/03-inference-and-pricing.md), [software and data](../notes/04-software-and-data.md), and the [roadmap](../README.md).

## Purpose

Assemble evidence already collected at each milestone into a complete S participant workflow. Validation begins with M0 contracts and M1 analytic checks; this milestone closes end-to-end gaps rather than postponing testing. M is an optional market illustration and E is a separate empirical extension; neither is required for S acceptance.

## Tasks

1. Maintain an evidence matrix separating simulator/numerical correctness, dataset integrity, synthetic inference/prediction accuracy, optional M case studies, and E empirical findings. Reference accepted fixtures and reports; record data origin, available truth, evidence scope, unresolved failures, and unsupported parameter regions.
2. Select one teaching task based on scientific relevance, conditional validation, and measured participant cost. Choose either returns → parameter/state posterior → conditional losses, or option panel → Q posterior → downstream price uncertainty. Keep alternative branches as optional material.
3. Reproduce the complete path from immutable configuration and bank manifest to training, checkpoint reload, adapter application, posterior draws, and the selected financial output. Cache expensive reference calculations with provenance.
4. Combine prior-predictive checks, SBC, parameter recovery, conditional posterior comparisons, joint/state-sensitive functionals, and predictive checks. Where reference MCMC is used, report its convergence and sampling uncertainty. Marginal SBC success cannot overrule an unresolved conditional discrepancy.
5. Run the entire selected S notebook offline on the intended CPU environment with all real-market fixtures absent and provider access disabled. Include generated context, inference, and downstream uncertainty. Measure data loading, checkpoint inference, and forward calculation together; distinguish precomputed synthetic references/checkpoints from participant calculations.
6. Put M/E cells behind explicit optional data-availability gates. M normally applies frozen weights without fabricated truth labels. E evaluates P forecasts against separately held-out later outcomes, or Q prices against same-time excluded target quotes, also excluded from smoothing/input construction. A market fit alone is not E. Record chronological dependence and the tested dates/contracts.

## Deliverables

An S acceptance report tied to exact versions, a compact participant notebook, reproducible synthetic reference assets, a checkpoint card, and separately labelled optional M/E reports and deferred extensions. Preserve the existing MMAR example unless an explicit curriculum decision replaces it.

## Acceptance and tests

Apply predeclared tolerances, including difficult-region and higher-accuracy checks. Confirm independent base splits, no target/future-outcome leakage, valid posterior dependence, correct predictive-mixture versus conditional-risk calculations, and checkpoint reloads. For options, propagate pricing errors; for returns, validate inferred-state continuation. Require the full S workflow to pass without providers or cached market files. Missing M/E assets must skip visibly, never produce synthetic results labelled real. Synthetic forecasts and pricing checks remain model-conditional evidence, not empirical backtests.

## Decision gate and deferred scope

Release the selected S track when its material failures are resolved or its supported scope is explicit. Market acquisition/permissions never block that release, and passing S does not imply M/E success. Avoid predetermined claims that frequentist methods fail, Bayes always raises risk estimates, or SBI is necessary. Additional models, scaling, and curriculum replacement remain separate decisions; no unmeasured runtime or speedup becomes a tutorial promise.

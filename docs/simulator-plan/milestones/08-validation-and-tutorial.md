# M8 Accumulated validation and a complete teaching path

**Status:** proposed; implementation starts after review. **Depends on:** M0–M2 and at least one accepted scientific/inference track selected for delivery. Completion of every model branch is not required. Read [inference and pricing](../notes/03-inference-and-pricing.md), [software and data](../notes/04-software-and-data.md), and the [roadmap](../README.md).

## Purpose

Assemble evidence already collected at each milestone into a reproducible participant workflow. Validation begins with M0 contracts and M1 analytic checks; this milestone closes end-to-end gaps rather than postponing numerical or inferential testing until the end.

## Tasks

1. Maintain an evidence matrix separating simulator correctness, numerical accuracy, dataset integrity, posterior approximation, and financial-output accuracy. Reference accepted fixtures and reports; record unresolved failures and unsupported parameter regions.
2. Select one teaching task based on scientific relevance, conditional validation, and measured participant cost. Choose either returns → parameter/state posterior → conditional losses, or option panel → Q posterior → downstream price uncertainty. Keep alternative branches as optional material.
3. Reproduce the complete path from immutable configuration and bank manifest to training, checkpoint reload, adapter application, posterior draws, and the selected financial output. Cache expensive reference calculations with provenance.
4. Combine prior-predictive checks, SBC, parameter recovery, conditional posterior comparisons, joint/state-sensitive functionals, and predictive checks. Where reference MCMC is used, report its convergence and sampling uncertainty. Marginal SBC success cannot overrule an unresolved conditional discrepancy.
5. Run the participant notebook on the intended CPU environment and a clean supported installation. Measure data loading, checkpoint inference, and forward calculation together. Document which experiments are precomputed and which participants execute.

## Deliverables

An acceptance report tied to exact versions, a compact participant notebook, reproducible reference assets, a checkpoint card, and a list of deferred extensions. Preserve the existing MMAR example unless an explicit curriculum decision replaces it.

## Acceptance and tests

Apply tolerances set for each task before its final evaluation, including difficult-region and higher-accuracy checks. Confirm split independence, no latent-label leakage, valid posterior dependence, correct predictive-mixture versus conditional-risk calculations, and robust checkpoint reloads. For options, propagate pricing approximation errors; for returns, validate continuation from the inferred state. Present synthetic evidence separately from empirical performance and model-misspecification limitations.

## Decision gate and deferred scope

Release only the selected track whose material failures are resolved or whose narrower supported scope is explicit. Avoid predetermined claims that frequentist methods fail, Bayes always raises risk estimates, or SBI is necessary. Additional models, training-scale increases, and curriculum replacement remain separate review decisions; no unmeasured runtime or speedup becomes a tutorial promise.

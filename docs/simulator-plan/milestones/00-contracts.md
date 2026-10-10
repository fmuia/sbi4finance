# M0 Scientific and software contracts

**Status:** proposed; implementation starts after review. **Depends on:** no implementation milestone. Read [foundations](../notes/01-foundations.md), [software and data](../notes/04-software-and-data.md), and the [roadmap](../README.md).

## Purpose

Agree what a model, observation, inference target, and generated example mean before optimizing kernels. This establishes common building blocks without committing the tutorial to returns or options, and without making the implementation depend on Francesco's pending bundle.

## Tasks

1. Define small immutable specifications for model, observation, inference task, numerical settings, and execution. Keep parameter values and priors outside numerical kernels. Distinguish scientific parameters from discretization, factor-approximation, and execution settings.
2. Review parameter names, units, ordering, supports, drift/jump compensation, probability measure, and initialization conventions. Record whether initial variance or a forward-variance curve is known context, an inferred quantity, or a draw from a declared law. Identify any deliberate incompatibility with a collaborator's parameterization.
3. Define capabilities for path simulation, pricing, and conditional continuation. Continuation state is model-specific: variance for Heston, factors for finite-factor Bergomi, and an explicit memory representation for rough models. Unsupported combinations must fail clearly.
4. Freeze the first task schemas: a complete return window and a complete option panel. Separate observations/context from target truth and numerical diagnostics. Preserve batch, time/contract, and channel axes; distinguish missing observations from zero returns.
5. Specify canonical task identity, stable example IDs, split groups, purpose-separated randomness, failure recording, and checkpoint provenance. Document which settings change the statistical experiment and which merely change execution.

## Deliverables

A reviewed convention table, two worked task specifications, illustrative batch shapes, a minimal capability table, and a short decision record for unresolved scientific choices. The initial package remains inside `sbi4finance`; these documents define the boundary for later code, not a requirement to create every suggested module immediately.

## Acceptance and tests

Walk one returns example and one option example from parameter draw to inference target. Confirm that no target truth reaches network inputs, and that changing shard size cannot change an example's scientific identity. Review P/Q distinctions and model-specific continuation with the mathematical notes. Mark every unresolved convention explicitly; later milestones may implement only the reviewed subset.

## Decision gate and deferred scope

Validate the agreed contracts before M1 implementation. Francesco-specific mappings can remain pending while common contracts proceed. Defer arbitrary component composition, a plugin framework, variable-length generality, joint P/Q risk-premium estimation, and a separate published simulator package.

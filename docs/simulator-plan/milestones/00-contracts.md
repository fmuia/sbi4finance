# M0 Scientific and software contracts

**Status:** M0.1–M0.3 implemented for the first GBM task; later-model details remain deferred. See the [conventions, worked records and limits](../../gbm-reference.md). **Depends on:** no implementation milestone. Read [foundations](../notes/01-foundations.md), [software and data](../notes/04-software-and-data.md), and the [roadmap](../README.md).

## Purpose

Agree what a model, observation, inference target, and generated example mean before optimizing kernels. This establishes common building blocks without committing the tutorial to returns or options, and without making the implementation depend on Francesco's pending bundle.

Keep this a short setup step for the first executable example. A convention table, concrete configuration and two small worked records suffice. Settle GBM's needed choices now; sketch other models' boundaries and resolve their details when their milestone is active. No universal schema library, inheritance hierarchy or exhaustive catalog is required.

Adopt the [S/M/E evidence boundary](https://github.com/fmuia/sbi4finance/blob/1e45918214c57b20120feb1c802cb3220cefcc41/docs/inference-scenarios-and-data.md#three-delivery-and-evidence-levels): S is the required complete synthetic workflow; M is optional market conditioning; E is a separate empirical evaluation. This is a shared contract, not a dependency on merging the data-plan PR or acquiring market data.

## Tasks

1. Define explicit settings for model, observation, inference task, numerics and execution, initially as plain configurations or small dataclasses. Save the configuration used for each run rather than building a specification framework. Keep parameter values and priors outside numerical kernels. Distinguish scientific parameters from discretization, factor-approximation, and execution settings.
2. Review parameter names, units, ordering, supports, drift/jump compensation, probability measure, and initialization conventions. Record whether initial variance or a forward-variance curve is known context, an inferred quantity, or a draw from a declared law. Identify any deliberate incompatibility with a collaborator's parameterization.
3. Define capabilities for path simulation, pricing, and conditional continuation. Continuation state is model-specific: variance for Heston, factors for finite-factor Bergomi, and an explicit memory representation for rough models. Unsupported combinations must fail clearly.
4. Document the first task layouts: a complete return window and a complete option panel. Add `data_origin`, purpose, measure/task, conditioning cutoff or contract selection, available truth labels, and S/M/E evidence scope as simple metadata. Separate observations/context from target truth and numerical diagnostics. Preserve batch, time/contract, and channel axes; distinguish missing observations from zero returns. Market records have no fabricated parameter/state labels. Layouts can change during iteration; record the version and regenerate pilot artifacts rather than requiring migrations.
5. Specify canonical task identity, stable example IDs, split groups, purpose-separated randomness, failure recording, and checkpoint provenance. Document which settings change the statistical experiment and which merely change execution.

## Ordered implementation subplan

The first-task outputs below are recorded in the linked implementation note; broader model contracts are sketched, not frozen. Suggested destinations follow the [module responsibilities](../notes/04-software-and-data.md#responsibilities-and-dependencies), without fixing final filenames or requiring empty modules.

| Step | Concrete output | Verification/handoff |
|---|---|---|
| M0.1 — Conventions and capabilities | A convention table for parameters, units, P/Q measure, initialization, numerical settings and model-specific continuation; a supported-capability table. Keep these contracts with the plan and use them to guide `simulators/`, `observations/` and `pricing/`. | Trace the proposed model ladder through the capability table. State unsupported operations explicitly, including rough-state continuation that has no accepted representation. |
| M0.2 — Two worked task schemas | A synthetic returns specification and option-panel specification, illustrated with paper/mock records for batch shapes, priors/targets, hypothetical context, example/split identity, and S/M/E provenance and truth-label metadata. These guide `tasks/`, `configs/tasks/`, `datasets/` and `adapters/`. | Trace the declared roles of parameter draws, observations and targets without requiring executable simulators. Show that no latent truth or future evaluation outcome enters conditioning, and that neither task needs market fixtures. |
| M0.3 — Accepted contracts and decision record | Consolidated contracts plus a short record of resolved choices and explicitly deferred alternatives. Identify the subset needed for GBM and bank generation. | Hand the agreed subset to M1 and M2. Keep the returns-versus-options teaching choice open; pending collaborator mappings or optional M/E acquisition do not block the common contracts. |

## Deliverables

A reviewed convention table, two worked task specifications, illustrative batch shapes, a minimal capability table, and a short decision record for unresolved scientific choices. The initial package remains inside `sbi4finance`; these documents define the boundary for later code, not a requirement to create every suggested module immediately.

## Acceptance and tests

Walk one worked returns record and one worked option-panel record, including hypothetical contexts, from declared parameter draw to observation and inference target. Use paper/mock values and schema checks; executable generation is accepted in M1 and the relevant later model/pricing milestone, including M4 for HKDE panels. Confirm from the contracts that no target truth reaches network inputs, that no market fixture is required, and that changing shard size cannot change an example's scientific identity. Review P/Q distinctions and model-specific continuation with the mathematical notes. Missing market access or permission affects only M/E; a synthetic substitute must never be labelled real. Mark every unresolved convention explicitly; later milestones may implement only the reviewed subset.

## Decision gate and deferred scope

Validate the GBM-relevant conventions before M1 implementation; do not wait for detailed decisions about every later model. Francesco-specific mappings can remain pending while common contracts proceed. Defer arbitrary component composition, a plugin framework, variable-length generality, joint P/Q risk-premium estimation, and a separate published simulator package.

# D0 Task and local artifact contracts

**Status:** proposed. **Scope:** S foundation, reusable by M/E. **Depends on:** the [scenario specification](../../inference-scenarios-and-data.md); align with PR #3 M0. Return to the [roadmap](../README.md).

## Purpose

Make the statistical objects explicit before implementing file readers. A return history, an option panel, a target contract and a labelled synthetic bank have different roles. Define local observation artifacts that can share PR #3's task meaning without duplicating its model registry or bank schema.

## Dependencies and inputs

Use the agreed R/O/P/D task definitions and the existing repo's environment/layout. Model-specific parameters, initialization and continuation state come from the selected scientific task. Until that contract lands, record the proposed interface and unresolved fields; do not invent an SVJ parameterization to match a filename.

## Implementation sequence

1. Define a small versioned manifest contract for artifact ID/type, schema/task identity, representation, units, source provenance, data origin, purpose, content hashes, conditioning cutoff/contract selection and truth-label availability. Keep evidence scope on reports or linked experiment metadata; market provenance alone does not imply E evidence.
2. Define return-window and option-panel records using named columns and explicit context. Preserve optional unknown metadata as absent/null, distinct from an invalid required field. Specify which records support inspection, normalization, model conditioning or evaluation; a processed-IV file may support fewer capabilities than raw quotes.
3. Separate observations/context/masks, synthetic targets, numerical diagnostics and evaluation outcomes. Real conditioning artifacts contain no manufactured target vectors. Target-contract specifications identify the requested payoff and pricing context; they do not imply that a target market quote exists.
4. Define task compatibility: P/Q measure, return convention or price/IV representation, supported times/contracts, units, fixed versus variable context, masks, target/state ordering and adapter version. Source IDs and paths are provenance, not encoder features. Content integrity and statistical compatibility are separate checks.
5. Freeze two worked schema examples, one R and one O, plus an unlabelled variant and invalid examples. Label hand-constructed values `parser_test` or equivalent; only validated generation supplies scientific training truth. Keep precise priors/noise/grid choices in the owning task configuration.

## Deliverables

A short schema/convention specification, representative manifests and table headers, capability/error categories, and a mapping to PR #3's shared task/manifest fields. Suggested code homes are small helpers under `src/sbi4finance/datasets/` and definitions reused from `tasks/` where those already exist; settle filenames during implementation rather than creating a competing registry. Small fixtures belong in a clearly named test-fixture directory, separate from scientific demonstration data.

## Acceptance and verification

Walk one generated-style return record and one option record through every field's role. Show that unlabelled inference is representable without zero-filled parameter/state targets. Demonstrate how an unknown quote time differs from an invalid observed value. Identify every unresolved scientific convention explicitly and prevent affected model conditioning until it is resolved. Neither a real dataset nor a provider account is needed for acceptance.

## Exit and deferred scope

D1 can implement this local contract once required IO fields are settled. Full model/bank/checkpoint compatibility remains D2. Defer universal schema frameworks, provider adapters, new inference targets and exact reproduction of Francesco's study until its scientific bundle is available.

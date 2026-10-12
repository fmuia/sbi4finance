# D1 Local fixture loading and integrity

**Status:** proposed. **Scope:** S foundation. **Depends on:** [D0](00-contracts.md). Supplies [D1-R](01a-return-windows.md) and [D1-O](01b-option-panels.md). Return to the [roadmap](../README.md).

## Purpose

Provide a small local-file interface used by notebooks and preparation scripts. It reads transparent observations and metadata without installing provider SDKs or invoking a model. Domain-specific return and option normalization lives in the linked subplans.

## Dependencies and inputs

Use D0's versioned manifests, generated parser fixtures and existing project environment. A fixture can validate parsing before a scientific simulator exists; its manifest must not claim numerical or posterior validation. D1 does not produce a synthetic training bank.

## Implementation sequence

1. Implement local table/manifest loading with explicit artifact paths, supported schema versions, named columns and dtypes. Missing files and unsupported representations produce actionable errors; there is no network fallback.
2. Verify required companion files, content hashes, unique row identifiers and declared dimensions. Preserve deterministic row order or apply the documented domain ordering. Record normalization version and source hashes when writing a prepared artifact.
3. Return observations and context separately from optional truth, diagnostics and evaluation data. Select fields explicitly; never expose every numeric column as conditioning. Do not flatten a time series or panel into independent training examples.
4. Add deterministic writing for small prepared fixtures and a compact validation report. Incomplete output must not be accepted as a complete artifact. Keep float64 scientific table values; neural conversion belongs to the shared adapter.
5. Establish generated positive/negative fixtures for both representations, then let each observation branch extend the validators. Keep any market cache outside required tests. Add ignore rules before optional private files enter the workflow.

## Deliverables

A local artifact reader/validator and small-fixture writer, generated parser fixtures, format documentation and a preparation-report structure. Suggested functions are `load_return_window(...)` and `load_option_panel(...)` backed by shared manifest helpers; these are proposed entry points, not APIs that already exist. Reuse any PR #3 metadata/integrity helper that has landed instead of copying it.

## Acceptance and verification

Read/write/read preserves values, declared order, units and unknown fields. Test a missing companion file, altered content/hash, duplicate identity, unsupported schema and malformed numeric field. The import path succeeds with provider packages absent and network access disabled. An unlabelled record remains unlabelled; adding a diagnostics column cannot make it a conditioning channel. Error fixtures exercise these meaningful failure boundaries, rather than just mirroring class construction.

## Exit and deferred scope

Shared IO acceptance allows either observation subplan to proceed. It does not certify return conventions, pricing accuracy, real source freshness or checkpoint compatibility. Defer Parquet, remote stores and general plugin/provider abstractions unless measured requirements justify them. PR #3 M2 continues to own large shards, IDs/RNG, grouped splits and the training batch reader.

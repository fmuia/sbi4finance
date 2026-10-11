# D2 Freeze designs and connect shared scientific contracts

**Status:** proposed. **Scope:** selected S branch, reusable by M/E. **Depends on:** [D1](01-local-fixtures.md) and [D1-R](01a-return-windows.md) or [D1-O](01b-option-panels.md). Return to the [roadmap](../README.md).

## Purpose

Make it impossible to treat a merely well-formed file as compatible with any model/checkpoint. Freeze the observation design and connect local artifacts to PR #3's task, bank and adapter contracts. Keep local validation separate from integration that requires actual scientific assets.

## Dependencies and inputs

Local checks use D0 contracts and prepared fixtures. Actual integration needs PR #3's M0 task schema, M2 verified bank/batch interface, and an accepted model-specific adapter/checkpoint. Record missing dependencies as pending integration, not a reason to duplicate those components here. The options or returns branch alone can support the selected release.

## Implementation sequence

1. Freeze the supported observation design and its compatibility identity: measure/task, representation, units, time/contract ordering, supported context/masks, adapter version and required target/state schema. Distinguish a scientific fingerprint from file hashes and source paths.
2. Validate loaded context against checkpoint support, including fixed curves/grids or explicitly variable inputs. A same-sized input with different maturities, time steps or return convention is not automatically compatible. Do not silently interpolate, pad or rescale into apparent compatibility.
3. Construct observed conditioning through PR #3's shared adapter using an explicit field allowlist. Preserve batch/channel axes. Keep true parameters, state, generator diagnostics, acquisition metadata and future/held-out outcomes outside conditioning unless the task explicitly declares an observable field.
4. Consume a small verified bank from the shared reader and compare its observation/context semantics with the local-fixture route. Use existing IDs, grouped split records and normalization statistics. Checkpoint provenance must identify the scientific configuration, bank and adapter; weights alone are insufficient.
5. Record a compatibility report with passed/rejected capabilities, exact versions and pending dependencies. Add an inference-only record with no targets, demonstrating the same observation contract used at synthetic training and later market application.

## Deliverables

Compatibility checks at the loader-to-adapter boundary, a frozen local design manifest, integration fixtures and a short report. Reuse PR #3's reader/verifier/adapter rather than adding a second bank writer, training loader or transform implementation. Add shared contract functionality in its owning module once if it is missing.

## Acceptance and verification

First accept local contract checks independently. Reject wrong measure/task, units, representation, contract/window design, unsupported missingness and context. Perturb truth/diagnostics/evaluation values while keeping observations fixed: conditioning must remain identical. Then, when actual artifacts exist, test bank-versus-local adapter equivalence and an inference-only call through a compatible reloaded checkpoint. Do not count placeholder weights or parser fixtures as that integration test.

## Exit and deferred scope

D2 is fully accepted only when the selected branch's real integration passes; its local checks may be tracked as a completed substage earlier. D4 also needs accepted posterior and downstream numerical validation from PR #3. Data integrity alone proves neither inference accuracy nor financial usefulness. Defer automatic schema migrations, arbitrary checkpoint adaptation and support for untrained grids or extra observation channels.

# D1-R Chronological return preparation

**Status:** proposed. **Scope:** returns branch, S first. **Depends on:** [D1](01-local-fixtures.md) and the selected return task's conventions. Return to the [roadmap](../README.md).

## Purpose

Turn local prices or explicitly defined returns into one ordered conditioning history with a precise information cutoff. The same observation structure must support generated histories and later unlabelled market histories. This module prepares observations; it neither simulates SVJ paths nor estimates hidden volatility.

## Dependencies and inputs

Start with deterministic generated parser examples. Add genuine generated histories from PR #3 M1/M3 when available. The existing MMAR cache and checkpoint retain their own definitions and are not automatically valid SVJ inputs. Model-specific window length, time convention, return kind and state target are task decisions.

## Implementation sequence

1. Preserve original timestamped levels/returns, units, currency, adjustment/cash-flow policy, source and calendar. Provide configuration for log versus simple returns and decimal versus percent units; do not infer these from magnitude alone.
2. For levels, compute returns using the declared definition. Require positive levels for log returns and handle unsupported cases explicitly. Retain interval start/end and the task's year fraction. For supplied returns, validate their definition without recomputing from an unrelated close column.
3. Resolve sorting, duplicate timestamps, calendar gaps and missingness before constructing windows. Do not zero-fill missing days. The first regular-window task may reject unsupported gaps; masking or irregular-time support requires a matching trained task.
4. Construct fixed-length windows ending at explicit forecast origins. Preserve return scale; avoid per-window standardization that discards volatility information. Link any future evaluation block separately, outside the observations returned to inference.
5. Produce a report of conversions, dropped/rejected rows, original boundaries and resulting window shape. Preserve group/origin identifiers for related windows; PR #3's split policy controls their use in banks. Never claim independent samples simply because windows have different IDs.

## Deliverables

Return preparation helpers, a local `prepare_return_window` script or equivalent function, configuration and generated fixtures. Output columns include interval start/end, `dt_years`, return value and validity, with manifest-defined conventions. Scientific truth is a separate optional target record supplied by the generator. The small local result can later be consumed by the shared adapter; no network download or posterior code belongs here.

## Acceptance and verification

Verify known log/simple conversions and decimal/percent scaling. Test duplicate/out-of-order dates, a missing observation, a calendar gap, nonpositive log-return levels, supplied-return metadata conflicting with the task and a window extending beyond its cutoff. A zero return must remain distinguishable from missing data. Modifying future evaluation values must not change prepared conditioning. Repeated preparation from identical inputs/configuration gives identical observations and lineage.

## Exit and deferred scope

D2 can check the resulting record against the selected task/checkpoint. Exact SVJ reproduction waits for the actual model contract; IO need not wait. Defer automatic corporate-action inference, futures stitching, multivariate portfolios, high-frequency realized-volatility estimators and a real rolling backtest. Their observed channels and financial definitions require separate specification.

# D1-O Option-panel preparation

**Status:** proposed. **Scope:** options branch, S first. **Depends on:** [D1](01-local-fixtures.md) and the selected options task's conventions. Return to the [roadmap](../README.md).

## Purpose

Normalize a local cross-section into one conditioning panel without confusing premiums, implied volatilities, target prices or numerical diagnostics. Use the [option format reference](../../data-loading-plan.md#shared-panel-schema); provider-specific acquisition is optional D3 work.

## Dependencies and inputs

Use generated parser fixtures covering `raw_quote` premium semantics and `processed_iv` semantics, explicitly marked synthetic. Genuine reference-price fixtures come from accepted PR #3 pricing work; hand-written example numbers are not pricing truth. Require the task's contract design, price/IV observation law and fixed/inferred market inputs before conditioning a model.

## Implementation sequence

1. Map named source columns with explicit units. Preserve premium bid/ask versus decimal IV bid/ask, original symbols/root, option type, currency, multiplier, exercise/settlement conventions and time semantics. Unknown raw fields in processed imports stay unknown.
2. Validate row identity, ordered finite quotes where applicable, positive strike/maturity, supported contract conventions and required context. Keep zero bids or unusual values with explicit selection reasons; do not silently repair a surface. Produce capability-specific validation: inspection may succeed where model conditioning lacks necessary inputs.
3. Attach supplied forwards, discount factors and their provenance consistently by expiry. Keep asset forwards separate from forward variance. Invoke an accepted pricing/IV utility only when needed and available; do not implement a second pricer or guess missing curves. Synthetic context is allowed and labelled.
4. Preserve or compute expiry year fractions using the declared conventions. Select and order the fixed contract design reproducibly. Define held-out contracts before any conditioning-dependent smoothing, curve estimation or noise fitting; keep their quote values outside prepared inputs.
5. Write the prepared panel plus exclusions/quality report and context hashes. Where implemented numerical utilities exist, add static pricing-bound/parity checks respecting carry and settlement. Treat those as diagnostics, not proofs of market or model validity.

## Deliverables

Panel normalization helpers, a local preparation entry point/configuration, generated fixtures and a report. Store contract/payoff context separately from optional observed target quotes. An American/barrier contract specification may be representable before its pricing capability is accepted; unsupported model-pricer combinations must still fail clearly.

## Acceptance and verification

Test a premium file versus an IV file, percent versus decimal IV, identical strikes across distinct roots/expiries, missing quote timestamps, mismatched context and invalid quote ordering. Verify deterministic selection and ordering. Missing premium/settlement fields in processed-IV input must not be invented. Changing a held-out quote cannot change conditioning. At D2, the same number of contracts on different coordinates must not pass a fixed-grid checkpoint check.

## Exit and deferred scope

The panel is ready for D2 after its supported capabilities and unresolved fields are explicit. A parsed file is not a calibrated model. Defer automatic smoothing, quote-noise estimation, de-Americanization, variable-grid neural support and exhaustive no-arbitrage repair. Those change the experiment or require separate numerical work; none is needed to validate the first synthetic panel loader.

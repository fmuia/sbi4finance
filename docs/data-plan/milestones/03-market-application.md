# D3 — Optional real-market application

**Status:** proposed. **Scope:** optional M; never a dependency of S. Return to the [roadmap](../README.md).

## Purpose

Apply the selected workflow to one actual market case (M), preserving the distinction between observation and synthetic truth in the [scenario specification](../../inference-scenarios-and-data.md). D3 is optional: unavailable market observations cannot block the complete synthetic teaching core (S). A successful import alone is not a successful market application or evidence of predictive performance.

## Dependencies and inputs

- Complete D2's local contract/validation checks for the selected returns (R) or options (O) branch. The other branch and full checkpoint integration are not required just to prepare an import.
- Local ingestion can then proceed independently. Running inference additionally requires D2's actual scientific integration and a validated, compatible model/checkpoint and conditioning pipeline from the selected simulator track, normally integrated in D4.
- Supply one archived return window or one option panel, its source conventions, provenance and necessary context. Real records have no known generating parameters, jump labels or latent variance labels.

Provider preferences in the [landscape review](../../data-source-landscape.md) remain provisional pending a representative sample, actual cost and permitted use. A usable local export is sufficient. This milestone does not authorize account creation, purchases or provider contact; no such action is required to start local ingestion.

## Implementation sequence

1. **D3.1 Select the case and input representation.** Specify the asset/date, selected task, conditioning boundary and intended claim. For R, fix return convention, adjustment policy, window and calendar. For O, choose processed IV or raw premiums and preserve strikes, expiries, exercise/settlement, valuation time, forwards and discount inputs. Do not require both cases.
2. **D3.2 Inspect and archive one source.** Retain original bytes and hashes in the permitted storage location. Record missing fields honestly. For an options acquisition, resolve historical as-of versus current capture and actual quote-feed/timestamp semantics before building an adapter. Implement at most one provider adapter if local files cannot meet the selected need; participant notebooks never retrieve data.
3. **D3.3 Normalize through the D1 interface.** Reuse D2 validation and produce a deterministic exclusion report. Unknown quote age, settlement or raw premiums remain unknown; they must not become invented fields. A processed-IV import does not acquire raw-quote evidence through normalization. Reject unsupported checkpoint coordinates or channels rather than silently interpolating or replacing the input.
4. **D3.4 Apply the frozen workflow.** Load market conditioning only. For R, infer physical-measure parameters and continuation state, then generate conditional forecasts. For O, infer risk-neutral parameters and propagate uncertainty into declared model prices. Neither operation needs subsequent realized returns or target market quotes. Do not transfer a Q posterior into P forecasting without an additional specified model.
5. **D3.5 Record evidence limits.** Show in-sample fit/predictive checks as such. Record metadata limitations and observed distribution shifts; good fit does not establish true parameters, latent volatility, model correctness or future performance. Missing market files produce an explicit optional-section skip, never synthetic output labelled as market.

## Deliverables

Proposed paths, reusing D1/D2 conventions:

- `configs/data/market/<case_id>.json`: source mapping and selection policy.
- `data/market/<case_id>/`: normalized conditioning plus manifest, only when distribution is permitted; otherwise a documented local location.
- `reports/data/market/<case_id>/`: provenance, exclusions, compatibility and M result summary, respecting source permissions.
- One optional market section in the selected teaching notebook. No second notebook architecture or general provider framework is needed.

## Acceptance and verification

Verify deterministic import, integrity hashes, absence of fabricated truth labels, appropriate P/Q context and D2 compatibility. Exercise the actual case against the validated checkpoint. Remove market files and confirm the S workflow still completes. Ingestion may be recorded as complete while application remains pending checkpoint availability; do not mark all of D3 complete then.

## Exit and deferred scope

Exit with one reproducible, qualified M application. Multiple providers, long historical coverage, simultaneous R/O demonstrations and empirical performance claims are deferred. D5 supplies the separate evaluation design and held-out evidence; market quotes themselves are not known fair-price truth.

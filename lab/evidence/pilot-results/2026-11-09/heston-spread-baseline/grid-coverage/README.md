# Frozen SPXW grid-coverage audit

**9 November 2026; geometry/data evidence only.** The original E targets produce 63 distinct actual quotes, of which 62 meet the declared distance rule. The snapshot cannot supply the original full-year endpoint or the final +2 call. No option-model prices, bank arrays, training or posterior results were read or computed. This is not an accepted replacement grid.

## Reproduction and source

Run from `lab/`, using the existing core-reference environment and a matching complete checkout:

```sh
python scripts/audit_heston_grid_coverage.py \
  --historical-root /path/to/sbi4finance-heston-observations-20261109 \
  --last-call-target 1.5 \
  --out /path/to/new-empty-audit-directory
```

The ignored historical cache is `local-artifacts/heston-observations-20261109/raw.json`, 14,091,523 bytes, SHA-256 `8c730e1108e3ba0a0f7b2871799880a63eca6ad337be7e4c00de6e2989b59458`. The command verifies size/hash against historical `heston-observations/source.json` and binds all historical design/reader and output hashes in [run.json](run.json). It reuses the standalone historical SPXW reader; no source download or runtime option pricer is involved. Observed command wall time was approximately 0.25 seconds on this machine, including the separately declared revision.

Valuation is **2026-11-09 20:13:37 UTC**, the original pilot's assumed snapshot generation time minus 15 minutes. Exact ACT/365F time ends at 16:00 New York on a regular SPXW expiry; early-close/same-day/non-SPXW/invalid markets are excluded by the historical reader. The raw fields do not establish bid/ask synchronization or freshness. The reader's positive-bid/noncrossed filters and all subsequent quality failures remain recorded.

## Original E-target audit

Choose the nearest actual eligible expiry to each nominal `[5,10,21,42,63,126,252]/252` year target. These are approximate maturity labels, not a business-day change to the clock. Select nearest distinct actual OTM quotes at fixed-16% coordinates `z=log(K/F)/(.16*sqrt(T))`, with target puts at `[-5,-3.5,-2.25,-1.25,-.5,0]` and calls at `[.5,1,2]`. Exclude carry anchors and preserved scoring quotes, require half-spread/mid at most 25%, and flag absolute target distance over 0.35. Keep unsupported nearest quotes visibly flagged; do not interpolate or replace expiries automatically.

| Nominal business-day label | Actual expiry | ACT/365F T | Calendar-day offset from nominal T | Realized z range |
| --- | --- | --- | --- | --- |
| 5 | 16 Nov 2026 | .01926633 | -.210 | -5.017 to +1.958 |
| 10 | 23 Nov 2026 | .03844441 | -.452 | -5.011 to +2.028 |
| 21 | 9 Dec 2026 | .08228003 | -.384 | -5.113 to +1.957 |
| 42 | 8 Jan 2027 | .16447181 | -.801 | -4.737 to +1.875 |
| 63 | 29 Jan 2027 | .22200606 | -10.218 | -5.091 to +1.993 |
| 126 | 30 Apr 2027 | .47120697 | -10.509 | -5.102 to +2.002 |
| 252 | 30 Sep 2027 | .89038505 | -40.009 | -5.147 to +1.486 |

All 63 selected symbols are distinct. The single unsupported strike target is the last +2 call: `SPXW270930C09800000`, strike 9800, realized z **1.485697**, distance **0.514303**. Its bid/ask are positive and its half-spread/mid is about 2.11%; this gap comes from eligible strike availability, not the 25% spread filter. The latest eligible PM expiry is 30 September 2027, **40.009 calendar days short of one year**. Actual quotes therefore do not meet the original endpoint/+2 coverage criteria.

[candidate.json](candidate.json) retains raw selected prices, spreads, symbols, exact maturities, carry and target-distance flags. [summary.json](summary.json) records every target, exclusions, maturity gaps and the old54 comparison. The old54 spans 21.03–324.99 calendar days, with deep puts approximately -3.69 to -4.29 and furthest calls +.93 to +1.07 in the same fixed-16% coordinate. Only one selected contract symbol overlaps old54. This is a substantial physical design change; old labels cannot be relabelled as these 63 contracts.

## Explicit supported-target revision, not silent substitution

At the user's requested follow-up, a separate revision changes **only the final expiry's call target from +2 to +1.5**. Other targets and all actual expiries remain as above. The last call realizes 1.485697, distance .014303 from its revised target, so all 63 quote targets meet the declared distance/quality rule.

See [candidate-supported-revision.json](candidate-supported-revision.json) and [summary-supported-revision.json](summary-supported-revision.json). The original unsupported audit is preserved. The original already retained the nearest actual last call, so both records contain **identical physical contract/price/spread arrays**; the revised declaration explicitly acknowledges that the last call is near +1.5. Its `complete_supported_quote_layout` flag concerns quote-target distance only. It **does not satisfy the original one-year/+2 criteria**, is not yet accepted or repriced, and is not a neural-performance-selected design.

## Carry and common conditioning

The 29 January and 30 September candidate expiries retain the exact old forward/discount estimates and anchor symbols. The comparison also retains all six historical carry panels. Five new panels—16/23 November, 9 December, 8 January and 30 April—use the historical deterministic every-third paired strike within 5% of snapshot spot and at least 15 parity pairs. WLS uses the half-spread-as-SD convention; its reported covariance is conditional, not an empirical quote-error estimate.

The union contains 11 expiry panels. Every anchor symbol is disjoint from old fitted quotes, preserved common raw scoring quotes and the new candidate fitting quotes. The preserved scoring quotes also remain excluded from candidate fitting. A future old/new comparison must disclose this union conditioning; it must not describe the new anchor information as removed or silently change old carry estimates. Future scoring quotes must be drawn outside the same anchor/fitting union.

**Interpretation:** the snapshot supports a clearly declared nearby 63-quote revision through about 10.7 months. It does not supply the literal original target endpoints. Choosing that revision, shortening the scientific horizon or obtaining another dated snapshot is a later explicit grid decision. This audit has no model-accuracy, posterior or release-acceptance claim.

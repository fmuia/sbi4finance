# Reviewed development evidence — 27 October 2026

Read the [results and next work order](../../reviewed-unit-2026-10-27.md). These are development artifacts, not accepted teaching checkpoints or final test results.

- `execution.json`: reviewed plan identity, immutable sources, commands, resources, submission attempts and the Heston case1 gate.
- `accounting.psv`, `unit-accounting-summary.json`: all 48 top-level allocations, counting CPU once. Forty-seven completed and one rough component timed out. No GPU was allocated.
- `heston-reviewed/case-{0,1,2}/`: native four-chain post-warmup draws, sampler metadata, pointwise Fourier checks, unchanged neural draws/comparisons, logs and saved-draw review figures/statistics. All three references pass declared diagnostics; all three neural comparisons fail the proposed parameter screen. Visual review does not prove mode coverage. The separate `case1-gate.json` records review before the other two cases were launched.
- `heston-reviewed/summary.json`: convenience extraction with hashes of retained case artifacts. The original case reports remain authoritative.
- `rough-reviewed/declaration.json`: frozen 44-call cohort, broad prior, policies, runtime/source identity and actual stream audit.
- `rough-reviewed/components/`: all 44 component reports, including the interrupted rung in component13. Do not infer a completed price from partial rung records.
- `rough-reviewed/accounting.json`, `accounting.psv`: scheduler component mapping and original accounting, including the timeout.
- `rough-reviewed/reduction/report.json`: checked reduction; 43 priced plus one censored call. It is intentionally `incomplete`, not an all-pass result.
- `rough-reviewed/analysis.json`: selected-contract shifts, replicate comparisons and separately labelled instrumented/allocated cost scenarios derived from the raw reports.
- `rough-reviewed/performance-note.json`: read-only evidence of system-CPU/frequency confounding; the root cause remains unknown.
- `rough-reviewed/provenance.json`: download and reducer identity. Original banks, model binaries and immutable execution checkouts remain on Oscar.

Reported rough MCSEs and pointwise intervals are empirical regression-residual diagnostics with the limitations explained in the report. Neither a small reported error nor two agreeing repetitions certifies accuracy or nominal coverage. Full covariance is retained; quotations in one vector are correlated.

No failed point was redrawn, no final holdout was opened, and no notebook was executed. Historical 26 October failures and provisional artifacts remain in their separate directory.

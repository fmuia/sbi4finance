# D5 — Optional empirical evaluation

**Status:** proposed. **Scope:** optional E; never a dependency of S/M. Return to the [roadmap](../README.md).

## Purpose

Evaluate a stated market claim (E) using observations excluded from conditioning and tuning. D5 extends D3; it is unnecessary for synthetic teaching, for showing a market-conditioned posterior, or for calculating forecasts and model-price uncertainty. Follow the [S/M/E evidence boundaries](../../inference-scenarios-and-data.md#three-delivery-and-evidence-levels). Market observations do not reveal true model parameters, latent variance or fair prices.

## Dependencies and inputs

- D3 provides actual market conditioning through a validated workflow, with the relevant checkpoint and numerical methods normally integrated in D4.
- Supply separate realized outcomes or observed target quotes, their provenance and an evaluation manifest linking them to conditioning cases.
- Choose one R or O evaluation question before obtaining a large dataset. A compatible pricer is necessary for target-price claims; valid state continuation is necessary for conditional return forecasts.

Evaluation observations are optional assets with their own permitted-use requirements. Their absence blocks the chosen empirical claim, not S or M. This plan does not authorize purchases, account creation or provider contact.

## Implementation sequence

1. **D5.1 Freeze the evaluation design.** Specify forecast horizon or held-out contracts, dates, preprocessing, inclusion rules, reported quantities, baseline, uncertainty estimates and the boundary between exploratory and untouched evaluation cases. Choose metrics appropriate to the prediction: distribution scores/interval coverage for return forecasts, suitable tail-risk assessment for VaR/ES, or residuals relative to observed option quotes/spreads for pricing. Record any subsequent design revision rather than silently reusing test evidence for selection.
2. **D5.2-R Build chronological outcomes.** At each forecast origin, condition only on available history/context. Keep subsequent returns and any realized-volatility proxy outside preprocessing, state estimation, model selection and inference. Preserve corporate-action, calendar and horizon conventions. Account for dependence from overlapping windows/horizons when reporting uncertainty. A single realized loss is not a VaR/ES label; a volatility proxy is not exact latent instantaneous variance.
3. **D5.2-O Build same-time contract holdouts.** Exclude target quotes from the conditioning surface and all fitted transformations. The holdout design must be fixed before training a fixed-grid estimator, or use masking/coordinate support already trained and validated; deleting inputs from an existing fixed-grid checkpoint is invalid. A new unsupported design returns to D0/D2 and PR #3's generation/training work. Audit indirect leakage through forward/discount-curve estimation, variance-curve fitting, smile smoothing, noise calibration and selection using target values. Use independently specified context or estimate it from permitted conditioning data; document the choice. Match valuation time, exercise, settlement and pricing units. Holdouts are other contracts observed at that time, not automatically tomorrow's chain. Separately specify repeated dates for temporal performance claims.
4. **D5.3 Run the fixed workflow.** Save conditioning IDs, checkpoint/numerical versions and predictions before joining evaluation values for scoring. Synthetic training-bank labels remain separate. For O, observed quotes are uncertain trading indications, so report pricing agreement rather than recovery of true fair value. For R, forecasts concern P dynamics; Q pricing posteriors cannot substitute without a specified risk-premium model.
5. **D5.4 Report scope and limitations.** Distinguish one illustrative check from repeated empirical assessment. Summarize missingness/exclusions, relevant baseline comparisons, uncertainty and tested dates/contracts. Separate numerical pricing error, posterior approximation and market discrepancy. Positive results do not establish correct latent parameters or performance in untested regimes.

## Deliverables

Proposed paths:

- `configs/evaluation/<study_id>.json`: frozen design and conditioning/evaluation IDs.
- `data/evaluation/<study_id>/`: held-out values plus provenance manifest, only when redistribution permits; otherwise a documented local location.
- `reports/evaluation/<study_id>/`: reproducible predictions, joined scores, exclusions and evidence statement.
- One offline evaluation entry point using existing loaders and inference interfaces; provider acquisition remains outside it.

## Acceptance and verification

Verify time/contract separation and indirect leakage rules. With the inference/forward random streams held fixed, changing held-out values must not change saved conditioning tensors, context, transforms or predictions. Reject missing or incompatible evaluation records explicitly. Reproduce scores from archived predictions and outcomes. Confirm optional E assets can be removed while S still runs and M can still produce predictions without scoring.

## Exit and deferred scope

Exit with a reproducible report supporting only the declared empirical claim. Broad regime comparisons, trading-strategy backtests, joint P/Q estimation and market latent-state recovery are outside this milestone. No minimum dataset size is asserted to guarantee reliability; match claim strength to the amount and dependence of actual evidence.

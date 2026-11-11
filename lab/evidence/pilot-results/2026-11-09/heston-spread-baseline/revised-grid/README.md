# Revised actual-63 Heston grid study

**In preparation.** The user authorized the revised-grid comparison and explicitly retained the recent broad prior. One new300k bank supports fresh200k-prefix and300k-full fits, each300epochs. The existing old54/200k/300 fit remains the control. Bergomi run preparation and initiation are conditionally authorized once the revised Heston setup works; no model or tutorial asset is accepted by preparation alone.

## Financial task and actual quotes

The [supported quote audit](../grid-coverage/README.md) selects63 actual European SPXW contracts over seven expiries. Nominal standardized log-moneyness is log(K/F)/(.16*sqrt(T)), with put targets[-5,-3.5,-2.25,-1.25,-.5,0] and call targets[.5,1,2]. The last expiry's final call target is explicitly+1.5; the longest actual expiry is about10.7months. These are declared targets/actual quotes, not option deltas or interpolated observations. The new layout changes maturity/wing coverage, quote count and spreads together.

The common conditioned carry union contains11expiry panels. Existing carry estimates remain unchanged where shared; all anchor/fitting/scoring sets stay disjoint. Product horizons stay fixed across the comparison. Model prices and quote noise remain in original price units.

The target is the same six independent physical Uniform parameters as the recent runs: v0/theta[.01,.16],kappa[.1,10],vol-of-vol[.1,3],rho[-.95,-.1],c[.5,2]. The observation law is Y=P(phi)+c*h*Z, with independent standard-Normal quote errors and observed half-spreads h. Half-spread-as-SD remains an explicit convention. No tau/vega component, inferred spread profile or prior-bound transform is introduced.

The [prior-context audit](prior-realism.json) distinguishes this broad training box from a representative market calibration. The original lesson fixed kappa2 and capped vol-of-vol1.5; today's grid study does not broaden the recent prior. Boundary corners are numerical stress cases. The saved old54 SPX references are conditional on one snapshot/model/noise law and do not establish universal market bounds.

## Numerical stability before scale

The original [pricing pilot](preparation/numerical-pilot/run.json) failed at predeclared corner74, v0=theta=.01,kappa10,vol-of-vol3,rho−.95. The whole-surface ladder could not find adjacent valid rungs: one intermediate rung was valid, but later near-zero call values failed the existing price-bound guard. The failure and original source/task remain preserved.

The new [CV task](cv-preparation/task.json) uses the same Heston characteristic function and pricing ladder, with an analytic Black control variate in the Lewis integral. The integrand is the difference between the Black and Heston characteristic functions; adding the analytic Black value gives the same infinite-integral price. This improves numerical cancellation without clipping prices, excluding parameter rows or inflating observation noise. QuantLib also provides Black control-variate Heston integration: [primary implementation](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/analytichestonengine.cpp).

The [separately frozen formal replay](cv-preparation/numerical-pilot/run.json) passes all98 declared points (64random,32corners,2paired synthetic truths), with maximum price refinement6.5011e-5 of the minimum observation SD and maximum absolute likelihood shift4.7061e-6. Shared-CF integration checks establish numerical consistency, not independent financial-model correctness.

The new numerical implementation lives once in the tutorial runtime; lab wiring reuses it. All old source files, labels, checkpoints and reports remain immutable under their original identities.

## Paired bank and training recipe

Always reprice new-grid contracts. Preserve all first200k physical parameter/c rows from the old control, draw100k fresh iid rows from its retained NumPy state, and draw new63-feature observation errors. On the one unchanged contract, reuse the stored standardized error to float64 recomposition precision; do not relabel old54 prices as new63 labels. Native100 and historical512 truths remain paired; historical new63 noise is fresh. Every training row's noise stays fixed across epochs.

Fit one full-rank ZCA transform only on the first10000 new-grid noisy price/half-spread rows. Freeze it for both fits; validation/diagnostic observations never enter fitting. The54-dimensional old whitener is invalid for63 inputs.

| Fresh fit | Distinct rows | Epochs | Presentations | Updates | Tail | Warmup / cosine-decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New63 prefix | 200000 | 300 | 60m | 234600 | 64 | 11730 /222870 |
| New63 full | 300000 | 300 | 90m | 351600 | 224 | 17580 /334020 |

Both use seed9011,nativeBayesFlow2.1/JAX FlowMatching widths128x3,dropout.1,raw targets,inference-variable standardization,batch256,nativeAdamW(weight decay.005,clipnorm1.5),learning rate5e-5 warming to5e-4 then cosine-decaying to0. Each starts fresh. More data also means more updates and regularization exposure; the second comparison is not data-only or equal compute.

Final300 is primary. The first strict native validation-loss minimum is saved and evaluated as a secondary only when distinct. No posterior query of other archived endpoints chooses a result. Every raw posterior sample is retained, including out-of-prior/nonphysical values; no clipping/rejection/rescaling changes the diagnostics.

## Validation and conditional next step

The new target needs [new observation-only reference cases](cv-preparation/conditional-panel.json): broad synthetic, market-like synthetic and actual market. Old54 chains cannot validate a new63 posterior. References use canonical nativePyMC data-only initialization and four sequential chains; smoke outputs are separate and never ready references. The numerical and native end-to-end smoke checks precede production.

Each endpoint receives native100x300 and historical512x512 recovery/coverage/SBC/support, plus4096draws against each target-matched conditional MCMC reference. Missing/unready references remain inconclusive. Reused development panels do not replace fresh confirmation after recipe selection. The reserved2000 pool remains unopened.

One bank and two fits have separate3600-second caps; at most two900-second posterior evaluations per fit. Three references have900-second inner/950-second outer caps with at most two workers concurrently. References may run beside generation/one fit. Failures retain their first records; no retry or extension silently changes the experiment.

The later Bergomi work must independently address irregular actual maturities, price/spread normalization by D*F, its variance-curve assumptions, new labels and rough-specific numerical error. Heston recovery alone cannot certify those components.

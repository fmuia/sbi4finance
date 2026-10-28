# Pricing diagnostic after the report cutoff

The [eight-page inference report](inference-performance-2026-10-28.pdf) preserves the completed posterior fits and the original, early-stopped known-curve 100k pricer. After that report was rendered, a separate pricer-only diagnostic completed at **17:39:34 UTC on 28 October 2026** (Oscar job8060582, exit0:0).

Only early-stopping patience changed from30 to300; the frozen worker, 100k bank, seed, widths, learning-rate schedule and batch size remain the same. Training completed all300 epochs and selected epoch300.

| Pricer | Validation RMSE, quote-noise SD | Worst absolute quote error, quote-noise SD |
| --- | ---: | ---: |
| Known-curve 50k | 0.082865 | 1.310775 |
| Original known-curve 100k | 0.098429 | 1.805862 |
| New full-300 diagnostic | **0.078006** | **0.829154** |

This is an encouraging improvement against the same bank validation labels. It supports investigating optimization before attributing the original regression to data size. At the initial diagnostic cutoff, the candidate had not yet been independently rescored; the subsequent section below records that follow-up. It has not been used for new PyMC references. No posterior was retrained, no original result was overwritten, and no model/cache was promoted.

The diagnostic used133 GPU-seconds and532 CPU-seconds. Cumulative training-pool use remains within the original allowance. Exact model identity, terminal accounting and reports are retained in the [diagnostic evidence](../pilot-results/2026-10-28/known100k-pricer-full300/README.md).

## Subsequent independent rescore

The full300 candidate has now been rescored against the same six fixed benchmark points and saved coarse/doubled-grid Monte Carlo batches. No new simulations, posterior samples or MCMC were run. Its RMS price errors improve over both the 50k and original100k pricers at **all six points on both grids**. For example, case2's generating-parameter RMS falls from .10523 to .08422 quote SD on the bank grid, and from .16414 to .11229 on the doubled grid, relative to50k. All six additional local finite-difference/PyTensor gradient checks pass.

These are point estimates on a small reused development panel, not a global accuracy certificate or an exact-model reference. The existing50k PyMC target and teaching cache remain unchanged. Full results, hashes and runtime are in the [separate rescore evidence](../pilot-results/2026-10-28/rough-emulator-readiness/known100k-full300/README.md).

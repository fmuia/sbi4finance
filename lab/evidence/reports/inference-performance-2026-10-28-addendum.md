# Results after the rendered report cutoff

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

## Joint-curve 100k completion, 29 October

The joint bank is now a checked **100k / 5k / 7k snapshot**. The automatic handoff submitted job 8080084 after verification; its posterior, pricer and validation completed successfully at **01:35:58 UTC / 02:35:58 Berlin time**. The original extension timeout remains recorded and included in spending. Exact model, dataset and terminal records are retained in the [completed comparison](../pilot-results/2026-10-29/joint100k-completed/README.md).

| Measurement | Joint20k | Joint100k |
| --- | ---: | ---: |
| Marginal rank checks, including derived rho × eta | 7/7 pass | 7/7 pass |
| Range of nominal 90% coverage across six inferred parameters | 89.70–92.00% | 89.15–90.15% |
| Pricer validation RMSE, quote-noise SD | .10118 | .08076 |
| Worst absolute quote error, quote-noise SD | 3.88847 | 1.04655 |

The validation observations, sampling seed and draw count are matched. Most posterior diagnostics change modestly; the largest rank discrepancy increases slightly while remaining within the same declared bound. Both posteriors use one training seed and 500 epochs, with 39,500 versus 195,500 optimizer updates. Both joint pricers actually completed 300 epochs; their declared patience changed but neither was cut short. These results do not isolate a causal data-count effect or establish full joint-posterior accuracy.

An independent rescore reuses the same four fixed points and 32 paired Monte Carlo batches. All four RMS estimates improve on both grids. At case 0's generating vector, working/refined errors fall from .18350/.15658 to .08632/.06092 quote SD; at interior case 2 they fall from .03817/.04279 to .02812/.03490. The representative posterior means remain the saved 20k vectors. Four local gradient checks pass. Several errors are at the panel's Monte Carlo noise floor, and pointwise quote discrepancies remain; no global accuracy certificate, new PyMC comparison or artifact promotion follows. [Rescore and precision limits](../pilot-results/2026-10-29/joint100k-independent-rescore/README.md).

The completed training allocation used 1,713 GPU-seconds. Cumulative use of the original training pool is now 5,621 GPU-seconds / 22,484 CPU-seconds, within 8,100 / 32,400. No additional simulation, training or MCMC was run for these comparisons. The rendered PDF retains its 28 October cutoff; this addendum contains the later results.

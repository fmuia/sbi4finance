# Known100k estimator against the fixed known50k reference

**Completed 28 October 2026, with no new MCMC allocation.** Both the 50k and 100k neural estimators pass the proposed parameter comparison screens on **2 of the same 3 fixed development cases**. The 100k estimator improves several discrepancies in case 2, but its Hurst and eta posteriors still fail the screens. This is useful learning evidence, not scientific acceptance.

The [canonical comparison](comparison.json) reuses exactly the [completed known50k PyMC chains](../known50k-current-pymc/README.md): four chains, 8,000 retained draws each, for validation cases 0, 1 and 2. Observed quotes, supplied known curves, priors, logit Jacobians, chain bytes and reference seeds are unchanged. All 5,000 shared validation observations, known curves and example IDs were checked for exact equality between the 50k and 100k bank snapshots. The comparison reads 1,000 saved neural draws per model and case; it performs no neural sampling, financial pricing, training or scheduler submission.

The reference remains the **50k conditional-mean pricer**, SHA-256 `162685962bbbfd8fa4029ae1dc13933ddc490e4222c8581a6f9fab10c8b5199b`. The new 100k pricer is deliberately excluded: its independent price checks regressed, so changing the reference at the same time would obscure this estimator comparison. The 100k neural checkpoint is `39f1a81986a38994dc6eacc520ad95254b34346abc02e758c08779c7c5916fc4`, trained on bank manifest `f1bfa42f7bf2acba3daaffb263b5bdeb60b5b12ebdfc267c5c5cd6eb7e08c313`, using the predeclared final epoch 500.

## Parameter comparisons

Mean errors are absolute differences divided by the reference standard deviation. Width ratios compare central 90% neural and reference intervals. The unchanged proposed screens require mean error at most 0.3 and width ratio between 0.75 and 1.25 for every parameter.

| Case | Parameter | 50k mean error | 100k mean error | 50k width ratio | 100k width ratio |
|---|---|---:|---:|---:|---:|
| 0 | Hurst | 0.055 | 0.024 | 0.967 | 0.955 |
| 0 | eta | 0.056 | 0.030 | 1.086 | 1.054 |
| 0 | rho | 0.173 | 0.140 | 1.119 | 1.081 |
| 1 | Hurst | 0.149 | 0.077 | 1.158 | 1.112 |
| 1 | eta | 0.223 | 0.014 | 1.157 | 1.123 |
| 1 | rho | 0.155 | 0.117 | 1.068 | 1.187 |
| 2 | Hurst | 0.822 | 0.565 | 1.814 | 1.587 |
| 2 | eta | 0.789 | 0.662 | 1.318 | 1.293 |
| 2 | rho | 0.290 | 0.103 | 1.423 | 1.169 |

Cases 0 and 1 pass for both models. Case 2 improves in every listed parameter metric and rho now passes both screens, but Hurst and eta remain too far from the fixed reference in mean and width. The three selected cases do not establish population-wide improvement or isolate a causal effect of training-bank size.

## Downstream rho × eta

The neural estimator still returns separate Hurst, eta and rho samples. Their product is computed downstream. This is descriptive parameter geometry, not a price or acceptance test.

| Case | 50k mean error | 100k mean error | 50k width ratio | 100k width ratio |
|---|---:|---:|---:|---:|
| 0 | 0.200 | 0.177 | 1.036 | 1.068 |
| 1 | 0.009 | 0.112 | 1.065 | 1.125 |
| 2 | 0.103 | 0.259 | 1.451 | 1.188 |

The product comparison shows the trade-off: case 2's width moves closer to the reference while its mean discrepancy increases. Improvement is not uniform across all summaries.

## Scope, uncertainty and cost

The reference targets a deterministic learned conditional-mean pricer with independent Gaussian quote noise and physical uniform priors, including the logit Jacobian. It omits pricing-label Monte Carlo randomness and covariance, emulator uncertainty and bias, and continuous-time grid bias. Convergence and agreement with this reference do not establish agreement with the stochastic simulator or continuous-model posterior.

All three reused reference diagnostics pass for the original parameters and rho × eta. Original diagnostics, chain axes and mean MCSE remain available in the [50k evidence](../known50k-current-pymc/README.md). Reference quantile MCSE and neural Monte Carlo errors are unassessed. Each current estimator contributes 1,000 saved draws per case; matched observations do not imply paired neural random draws. The older 20k comparison used 10,000 neural draws and also changed its reference pricer, so it is not this fixed-reference comparison.

The [fresh accounting receipt](terminal-accounting.psv) confirms **1,544 allocated CPU-seconds** for all existing reference runs: 788 for the original three-case pool and 756 for known50k job 8059665. This comparison adds **zero allocated CPU-seconds, zero GPU use and zero new MCMC jobs**. Metadata and NumPy statistics were processed without an allocated compute job. The original six CPU-hour cap remains unchanged.

The finite [comparison script](../../../../cluster/compare_known100k_fixed50k_reference_20261028.py) uses the frozen `c96a785` consumer implementation. Exact source, bank, checkpoint, saved-panel, reference and chain hashes are recorded in `comparison.json`. Original 50k outputs remain untouched; the selected 100k draw panel is retained on Oscar under `/oscar/scratch/afengler/sbi4finance-scientific-validation-20261028/known100k-fixed50k-reference/`. No reserved test data were evaluated. Both `scientific_acceptance` and `reserved_test_evaluated` remain false.

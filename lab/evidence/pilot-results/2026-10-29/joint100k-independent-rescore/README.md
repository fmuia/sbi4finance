# Joint100k pricer: independent saved-panel rescore

The completed pricer from Oscar job **8080084** was evaluated at the **exact four fixed points** from the existing joint20k pricing panel: validation cases 0 and 2, their generating vectors and their **saved 20k posterior means**. These are not new 100k posterior means. The same 32 Monte Carlo batches were reused, with eight paired working/doubled-grid batches at each point.

All eight RMS point estimates are lower for the later candidate. Values are RMS across 63 contracts in quote-noise standard deviations, shown as **working / doubled grid**.

| Fixed point | Original joint20k pricer | Joint100k candidate |
| --- | ---: | ---: |
| 0 / generating vector | 0.18350 / 0.15658 | 0.08632 / 0.06092 |
| 0 / saved 20k posterior mean | 0.09453 / 0.09045 | 0.06827 / 0.05844 |
| 2 / generating vector | 0.03817 / 0.04279 | 0.02812 / 0.03490 |
| 2 / saved 20k posterior mean | 0.03726 / 0.04168 | 0.02780 / 0.03460 |

The fits differ in training data and optimization: the pricer patience changed from 30 to 300, and the larger dataset also changes optimizer-update counts. This is a comparison of frozen candidates, not a causal data-only result or training-seed replication study. Both joint pricers actually ran all 300 epochs, so neither was truncated by early stopping. The new fit selected epoch 297 by its existing validation-error rule.

Four analytic/finite-difference/compiled-PyTensor gradient checks pass, including the physical-uniform-prior logit Jacobian. The largest scaled PyTensor discrepancy is **4.05e-13**. This establishes arithmetic consistency for the deterministic mean-emulator target, not posterior accuracy or an executed MCMC comparison.

Precision remains important. The saved panel's price-estimate RMS MCSE is approximately .074/.065 quote SD at case 0's generating vector and .043/.045 at interior case 2's generating vector. Several apparent errors are therefore at or below the Monte Carlo noise floor. At the previously flagged case-0 quote 26, the candidate's working-grid error is **−.1396**, pointwise t7 interval **[−.2609, −.0183]**; the doubled-grid value is **−.0652 [−.2563, .1259]**. These reused development-panel intervals do not certify small errors throughout either posterior or prior. Grid shifts and Monte Carlo covariance are unchanged; a doubled grid is not exact truth, and rank-seven batch covariance is not inverted or inserted into a likelihood.

The completed-run, exact model/task/bank/source and unchanged-byte checks are recorded in [extraction.json](extraction.json). Model SHA-256: `be699c31a7950ca3ebd9117275b4f8cb1a8355c96f7ca1e18bcfa36e0d252301`. Bank manifest SHA-256: `eb72588b9b7665e0e416cb5d3a1bcf8cd78a1c97f2b43f18ea8246c749a871d8`. The current `c96a785` pricer evaluator (`e5d8da24…`) is distinct from the unchanged `c773ceb` numerical producer used for the bank and independent price panel.

The local rescore took approximately **1.85 seconds elapsed** on Darwin arm64, using Python 3.12.16, NumPy 2.5.3, SciPy 1.18.1 and PyTensor 3.3.4 with one-thread library settings. Exact diagnostic/evaluator/input hashes, per-quote intervals and covariance are in [readiness.json](readiness.json); [verification.json](verification.json) and [sha256sums.txt](sha256sums.txt) pin the evidence.

No new simulations, posterior sampling, MCMC, training, reserved-test evaluation, cluster allocations or promotion occurred. Original 20k evidence and caches remain unchanged.

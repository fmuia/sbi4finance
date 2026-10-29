# Independent rough-emulator checks

The 50k known-curve emulator improves on the original 20k emulator at **all six fixed benchmark points**, using the same 48 independently generated price batches. No new simulation was needed for this comparison.

Errors below are RMS across the 63 contracts, in quote-noise standard deviations. The parameter points are the **original** generating vectors, 20k neural posterior means and 20k emulator-PyMC means for validation cases 0 and 2. They were held fixed when changing the emulator.

| Fixed point | Original 20k emulator, bank grid | New 50k emulator, bank grid | New 50k emulator, doubled grid |
| --- | ---: | ---: | ---: |
| Case 0, generating vector | .08165 | .03811 | .05348 |
| Case 0, original neural mean | .07635 | .03588 | .05273 |
| Case 0, original PyMC mean | .07725 | .03626 | .05262 |
| Case 2, generating vector | .25359 | .10523 | .16414 |
| Case 2, original neural mean | .16579 | .05189 | .12851 |
| Case 2, original PyMC mean | .19455 | .07426 | .13901 |

More training data and the new fit are buying better approximation at these points. They do not remove the simulator's grid effect. For example, the largest refined-grid quote error at case 2's generating vector is still approximately .5683 quote SD. These are development cases; case 2 remains a retained prior-boundary challenge.

The [paired MSE comparison](known50k-paired-mse.json) also accounts for uncertainty in the reused price batches. Comparing the two squared errors against the same batch cancels its squared-price term. All twelve point/grid contrasts favor the 50k emulator even with the stated Bonferroni/t7 intervals. This remains an exploratory development comparison with the original batch-estimator limitations.

The [local rescore](known50k/readiness.json) includes per-quote t7 intervals and full empirical price covariance. Eight batches give a covariance matrix of rank at most seven: it is preserved, not inverted. The [Linux audit and rescore](linux-audit/) repeat the calculation using the exact current checkpoint consumer.

### Later 100k fit: preserve the regression

The distinct [100k rescore](known100k/readiness.json) passed completed-run, model/task/bank hash and unchanged-file guards before reading the checkpoint. Its six gradient checks pass, but this fit has **larger price errors than the 50k fit at all six fixed benchmark points**. At case 0's generating vector, bank-grid RMS rises from .03811 to .05806; at case 2's generating vector, .10523 to .24264. Global validation-label RMSE also worsens, from .08287 to .09843.

The training records reveal an optimization confound: the 100k pricer stopped after 116 epochs and restored epoch 86; the 50k pricer ran 300 epochs and selected epoch 298. With batch size 256, those selected checkpoints have approximately 33,626 versus 58,408 optimizer updates. The 100k run stopped before the low-learning-rate part of its 300-epoch cosine schedule. This suggests a concrete optimization check; it does not prove that early stopping caused the regression or that more data hurt the model. Preserve both fits. A separately declared pricer-only full-300 run is a proposed next check, not work launched by this validation unit.

## Gradient and PyMC readiness

The existing deterministic mean-emulator target was differentiated independently in prior-box logit coordinates, including its uniform-physical-prior Jacobian. Finite differences and compiled PyTensor automatic differentiation agree at six known-curve benchmark points and four joint-curve points. Local maximum scaled gradient discrepancy was below `5e-13`. Separate tests cover both archived GELU definitions, prior transformations, correlated price errors, the analytic joint curve on refined nodes, finest-grid parity with the actual bank estimator, and checkpoint binding.

The first Linux metadata allocation, **8059387**, failed before pricing in two seconds because the frozen numerical checkout predates a newer reference-loader API. Its failure and cost are retained in [the receipt](failed-audit.json). The reviewed version uses the immutable current `c96a785` pricer consumer under a private import name; actual numerical pricing still uses the unchanged `c773ceb` source, checked by its module hashes. Checkpoint task, fingerprint, file hashes and predictions must agree. The revised audit **8059564 completed 0:0 in 28 seconds**, including all ten compiled gradient checks and a small finest-grid parity test.

This establishes the arithmetic needed for a future native PyMC NUTS route. It does **not** run NUTS, change the existing DEMetropolisZ reference route, or make an emulator posterior an exact-model posterior.

## Joint independent price panel

The [frozen revised plan](joint-price-panel-plan-v2.json) uses joint **validation:0 and validation:2**, with each generating vector and the mean of its already-saved 1,000 final-500 posterior draws. Case 0 is retained despite its low H and near-boundary rho; case 2 is interior. No case search or fresh inference was needed. The original central/high-variance six-case programme remains a separate later check.

The panel has 32 pricing calls: four fixed points, eight independent batches each, 65,536 antithetic paths per batch. Each call compares the working **1280/2016/2016** blocks with **2560/4032/4032**, using the existing Gaussian-channel coupling. It retains raw and controlled prices, full within-batch covariance, paired grid differences and external replicate uncertainty. The coupling has correct marginal innovation laws; it is not an exact singular-Wiener bridge. The analytic initial curve is evaluated on the fine nodes rather than interpolated from the coarse grid.

The retained first batch **8059627 completed 0:0 in 548 allocated seconds**, with 546.2 seconds of pricing, finite correctly shaped outputs and a valid covariance matrix. The remaining array **8059982**, indices 1–31, then completed successfully; final summary **8059983** ran after that entire array's success. Workers used one CPU, one library thread, 16 GiB and 30-minute limits, at most 32 scientific workers. No GPU, bank generation, new MCMC or reserved-test inference evaluation belongs to this phase.

The existing 24 CPU-hour independent-pricing allowance had consumed 5,790 allocated seconds. Including the failed two-second audit, all 32 full worker walls plus cleanup, metadata and final summarization reserve gives **76,352 seconds = 21.2089 hours**, below that same cap. Successful calls and all failed costs remain recorded; the allocation is not raised automatically.

After the successful audit and pilot, completed costs plus all outstanding full wall/cleanup and summary exposure were **73,448 seconds = 20.4022 hours**. Scaling that slow-node pilot suggested approximately 4.9 CPU-hours for the 32 pricing calls. The final actual cost below supersedes both that forecast and the preliminary 2–4-hour estimate.

### Completed panel

**All 32 batches and all 34 revised-phase allocations completed successfully.** Their actual cost was **6,612 CPU-seconds = 1.8367 CPU-hours**, with no GPU. Including the earlier pricing unit and failed two-second metadata attempt gives **12,404 seconds = 3.4456 hours**, below the same 24-hour ceiling. Nodes were materially faster than the retained slow-node pilot on average. [Terminal verification](terminal-verification.json), [full summary](joint-price-panel-completed/summary.json), [compressed price/covariance reduction](joint-price-panel-completed/batch-summaries.npz) and [quote-specific precision readout](joint-price-panel-completed/precision-readout.json) retain the evidence.

| Fixed joint point | Emulator RMS / quote SD, working / doubled grid | Interpretation |
| --- | ---: | --- |
| Case 0, generating vector | .18350 / .15658 | A clear local emulator discrepancy remains. |
| Case 0, current neural mean | .09453 / .09045 | Smaller errors, still requiring qualification. |
| Case 2, generating vector | .03817 / .04279 | At the Monte Carlo noise floor; not a certified small error. |
| Case 2, current neural mean | .03726 / .04168 | The same precision limitation applies. |

At the boundary challenge's generating vector, quote 26 has emulator error **−.5878 quote SD**, pointwise t7 interval **[−.7091, −.4665]** on the working grid, and **−.5133 [−.7044, −.3223]** on the doubled grid. Its largest observed grid movement, **−.2746 [−.7715, .2224]**, is unresolved. At interior case 2, a smaller grid movement is measured at quote 50, **−.03270 [−.05019, −.01522]**, while the price-estimate RMS MCSE of approximately .043–.045 quote SD prevents certifying the apparent .038–.043 emulator RMS. These selected worst-quote intervals are post-hoc and pointwise, not simultaneous accuracy bands.

Full raw/controlled, cross-grid and cross-contract covariance remains on Oscar under `joint-price-panel-v2/results/batch-*/prices.npz`; checksums for every payload are in the terminal receipt. No additional precision batches or references were automatically started.

Three errors remain distinct: emulator interpolation error, random numerical-label error (including shared-path covariance), and time-grid bias. A deterministic emulator Gaussian likelihood omits the bank observation pricer's random error. This panel measures components of that mismatch; it does not justify silently adding a diagonal MCSE correction or passing a fresh stochastic likelihood to ordinary MCMC. A doubled grid is a comparator, not exact truth.

Source: [gradient/rescore diagnostic](../../../../scripts/check_rough_emulator_readiness.py), [paired-panel driver](../../../../scripts/validate_rough_price_panel.py), [fixed-case extraction](../../../../scripts/extract_rough_validation_representatives.py). No original numerical producer is modified.

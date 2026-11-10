# Larger batches for joint Heston noise-scale inference

**Completed, 10 November 2026. Neither larger batch resolved learning of c.** On the same 512 validation datasets, final c posterior-median RMSE is .44743 / .44709 for batches 1024 / 4098, and median central-90% widths are 1.34287 / 1.34244 versus prior width 1.35. The constant prior-median predictor has RMSE .43707. These results remain essentially unchanged from the completed 128/512 runs.

Read the [offline HTML report](report.html), [four-batch c recovery](c-population-recovery.png), [new six-parameter recovery](new-parameter-recovery.png), and [matched MCMC comparison](selected-c-recovery.png). The report includes intermediate endpoints, SBC, conditional comparisons, exact network/rate/dropout code, and observed normalizer counters. All figures and styles are embedded.

| Final result · 512 identical observations | Batch 1024 | Batch 4098 |
| --- | ---: | ---: |
| c posterior-median RMSE | .44743 | .44709 |
| c truth–median correlation | −.03648 | −.04169 |
| Median central-90% c width | 1.34287 | 1.34244 |
| Marginal SBC screens | 6/6 pass | 6/6 pass |
| Both synthetic matched-posterior panels | Fail | Fail |
| Observed training/reload wall time | 1508.55 s | 1058.87 s |

The five physical parameters retain strong unit-line recovery (correlations at least .99286), while c remains almost prior-like. On the same seven converged MCMC cases, c RMSE is .10003 for MCMC versus .47060 / .47585 for the two new networks. That is a small stratified illustration, not population coverage. The unresolved eighth MCMC case remains visible. Passing pooled SBC does not establish that the estimator uses the available information about c.

All eight new checkpoint evaluations and both exact int64 ledgers passed the [independent results audit](results-verification.json). Every one of 100,000 training rows was shown exactly 500 times per branch, with 49,000 / 12,500 added updates. Both runs completed within their original caps; neither was retried or extended. Times were observed on a shared CPU with other training, software tests and sequential evaluations, not in an isolated speed benchmark.

**Specification predeclared 10 November 2026.** The user requested one training run at exact batch size **1024** and one at **4098**. Both start from the same original 60k six-parameter checkpoint as the completed 128/512 comparison, use all 100,000 clean Heston surfaces and train for 500 additional complete passes: **50 million additional noisy examples per arm**. The new [protocol](protocol.json) freezes this comparison before fitting.

| Batch size | Updates per pass | Unpadded final batch | Additional updates | Final optimizer count |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 98 | 672 | 49,000 | 109,000 |
| 4098 | 25 | 1648 | 12,500 | 72,500 |

Only the grouping into batches changes relative to the previous 128/512 study. The full parent state, 100k bank, per-pass row permutations, newly sampled c and Gaussian errors, prior, 54 actual European SPXW contracts, observation law and input representation stay the same. The architecture remains a six-block spline flow with [128,128] conditioner widths and zero effective dropout. The same explicit rate restart at 5e-5 decays by additional examples to 5e-6 over 400 passes, then holds through pass 500. The precision-safe host-float64 target transform is retained without clipping or redrawing.

This compares **equal examples, with different update counts**. It does not simultaneously equalize optimizer steps or compute, and one warm start and paired augmentation stream do not establish across-seed stability. The learning rate is kept fixed across batch sizes to isolate the requested change; it is not independently optimized for each batch.

Checkpoints at passes 25/100/250/500 were evaluated on the same 512 validation observations with the same posterior sample seeds, recovery plots, marginal SBC and existing conditional references. The new two-arm/four-endpoint family retains 384 predeclared approximate Monte Carlo-aware conditional checks. The seven converged references from the [eight-case MCMC study](../mcmc-recovery/README.md) supply an additional descriptive recovery comparison; its unresolved eighth case remains visible. No new MCMC is planned.

The [native preflight](native-batch-preflight.json) records finite updates and gradients at the actual requested batch shapes, production-contract and tail/rate checks, and exact CLI-checkpoint/imported-loader next-update replay. This is software-fixture evidence, not a production fit or a measured RSS bound. Historical producers and artifacts are preserved under their original identities.

At most two direct training workers run concurrently, each capped at 5400 seconds, with a 5700-second whole-training cap and no automatic retries or extensions. Full models, state arrays and posterior payloads stay outside Git; compact records and hashes are retained here. Reserved test arrays remain unopened. Complete the two runs and diagnostics, then reflect before grid or Bergomi work.

The observed float32 standardizer counter is 57,643,328 for batch 1024 and 57,631,768 for batch 4098, versus the stored parent count plus 50m of 57,643,328. The latter counter differs by −11,560 (−.02005%) because of unchanged float32 accumulation at the new grouping. This is separate from the exact int64 exposure ledger; no examples were dropped, and no counters or moments were reset. The [summary](summary.json) and [report manifest](report-manifest.json) bind the observed scalar reads and all compact source/copy records.

[Software verification](verification.json) records 732 passing lab tests plus three separately added report tests, current manifest/claims and the 183-symbol runtime boundary check. The [report verification](report-verification.json) records artifact/data/layout checks. No model is promoted. These runs test the same rate schedule and equal-example budget, with different update counts; they do not rule out other recipes or establish across-seed behavior. Stop for reflection before another experiment.

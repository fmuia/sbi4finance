# PR87 FlowMatching: 300-epoch followup

**In progress.** One 100k/300 fit, followed by one 200k/300 fit only if the first primary fails the existing necessary diagnostic screens. Both scientific specifications are frozen before the first fit. No loss-based or posterior-based checkpoint search follows the declared final/minimum-validation pair.

The [completed 100k/200 comparison](../pr87-100k/README.md) improved recovery but retained physical overcoverage, conditional MCMC errors and raw support violations. This followup keeps the financial task, fixed noisy observations, architecture, dropout, target coordinates, optimizer defaults, batch size, validation data and 10k-fitted whitening. Each fit starts fresh with seed 9011 and a native 300-epoch learning-rate horizon, rather than resuming an optimizer whose rate already reached zero at epoch 200.

| Budget | 100k/300 | Conditional 200k/300 |
| --- | ---: | ---: |
| Distinct noisy training datasets | 100,000 reused exactly | 100,000 preserved + 100,000 fresh |
| Epochs | 300 | 300 |
| Batch size | 256 | 256 |
| Batches per epoch / tail rows | 391 / 160 | 782 / 64 |
| Optimizer updates | 117,300 | 234,600 |
| Dataset presentations | 30,000,000 | 60,000,000 |
| Warmup / decay updates | 5,865 / 111,435 | 11,730 / 222,870 |

The 100k input data are copied byte for byte. For 200k, the fresh rows continue the completed 100k parent's retained post-append NumPy state, with native sequential ordering of all priors followed by all observation noises. The same 100 native validation rows and frozen whitening bytes remain; new data never enter whitening fitting. Models never inherit parent weights. These are practical data/optimization comparisons with distinct learning-rate budgets, not equal-compute attribution tests.

Final epoch 300 is primary; the first strictly minimum native validation objective is a secondary endpoint only when its epoch differs. Archived 100/200/250 checkpoints do not receive posterior queries. Each declared endpoint uses the same native100 x300, historical512 x512 and three existing joint-MCMC cases x4096, with unchanged observations, truths and seeds. Every raw draw is retained, including out-of-prior and nonphysical values; no clipping or interval rescaling is applied.

The primary 100k result alone determines whether 200k runs. The frozen necessary screens check all six marginal coverage/SBC results, finite/physical/prior support and ready conditional-reference agreement; they grant no official acceptance. Secondary results cannot alter this branch. Both generation/training workers are capped at 3600 seconds, each posterior evaluation at 900 seconds, with no retries or further fits. A failed worker stops the campaign.

The [campaign protocol](campaign-protocol.json), [100k protocol](100k-300/protocol.json), [200k protocol](200k-300/protocol.json) and [preservation inventory](preserved-artifacts-before.json) bind the recipe and earlier artifacts. No reserved test pool, new MCMC, grid changes or Bergomi production is opened. Official confirmation and the separate fixed-c baseline remain later gates.

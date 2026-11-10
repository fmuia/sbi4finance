# Larger batches for joint Heston noise-scale inference

**Predeclared 10 November 2026.** The user requested one training run at exact batch size **1024** and one at **4098**. Both start from the same original 60k six-parameter checkpoint as the completed 128/512 comparison, use all 100,000 clean Heston surfaces and train for 500 additional complete passes: **50 million additional noisy examples per arm**. The new [protocol](protocol.json) freezes this comparison before fitting.

| Batch size | Updates per pass | Unpadded final batch | Additional updates | Final optimizer count |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 98 | 672 | 49,000 | 109,000 |
| 4098 | 25 | 1648 | 12,500 | 72,500 |

Only the grouping into batches changes relative to the previous 128/512 study. The full parent state, 100k bank, per-pass row permutations, newly sampled c and Gaussian errors, prior, 54 actual European SPXW contracts, observation law and input representation stay the same. The architecture remains a six-block spline flow with [128,128] conditioner widths and zero effective dropout. The same explicit rate restart at 5e-5 decays by additional examples to 5e-6 over 400 passes, then holds through pass 500. The precision-safe host-float64 target transform is retained without clipping or redrawing.

This compares **equal examples, with different update counts**. It does not simultaneously equalize optimizer steps or compute, and one warm start and paired augmentation stream do not establish across-seed stability. The learning rate is kept fixed across batch sizes to isolate the requested change; it is not independently optimized for each batch.

Checkpoints at passes 25/100/250/500 will be evaluated on the same 512 validation observations with the same posterior sample seeds, recovery plots, marginal SBC and existing conditional references. The new two-arm/four-endpoint family retains 384 predeclared approximate Monte Carlo-aware conditional checks. The seven converged references from the [eight-case MCMC study](../mcmc-recovery/README.md) supply an additional descriptive recovery comparison; its unresolved eighth case remains visible. No new MCMC is planned.

The [native preflight](native-batch-preflight.json) records finite updates and gradients at the actual requested batch shapes, production-contract and tail/rate checks, and exact CLI-checkpoint/imported-loader next-update replay. This is software-fixture evidence, not a production fit or a measured RSS bound. Historical producers and artifacts are preserved under their original identities.

At most two direct training workers run concurrently, each capped at 5400 seconds, with a 5700-second whole-training cap and no automatic retries or extensions. Full models, state arrays and posterior payloads stay outside Git; compact records and hashes are retained here. Reserved test arrays remain unopened. Complete the two runs and diagnostics, then reflect before grid or Bergomi work.

# Paired larger-bank Heston training

**Predeclared development comparison, 10 November 2026.** The user authorized a larger bank, two batch sizes and longer training. The [protocol](protocol.json) binds the exact source checkpoint, existing bank, executable sources, training budgets and evaluation population. Production outcomes will be recorded here after both runs settle.

Both branches restore every saved model, Adam, normalization and auxiliary state array from the same six-target update-060000 checkpoint. They use all100,000 physical training surfaces from the preserved bank, without repricing. Each physical row receives one fresh c~Uniform(.5,2) and54 independent Gaussian errors per pass, under a new augmentation namespace. Shared row permutations and paired noise/scale draws make the example stream identical across branches, regrouped into batches. The model remains `Y_i=P_i(phi)+c*h_i*Z_i`; c is a target only, never an input. Geometry, physical priors, spreads, architecture and effective zero dropout remain unchanged.

| Branch | Complete passes | New examples | Additional Adam updates | Tail batch per pass |
| --- | --- | --- | --- | --- |
| Batch128 | 500 | 50,000,000 | 391,000 | 32 |
| Batch512 | 500 | 50,000,000 | 98,000 | 160 |

The old run used20,000 distinct physical rows and approximately7.64m noisy examples. This follow-up uses100k distinct rows and adds50m examples per branch. Complete passes, examples and optimizer updates are reported separately. At equal examples, the smaller batch receives more updates; this is the primary budget comparison, not an equal-update experiment. One warm start and one paired augmentation stream do not establish stability across seeds.

The parent ended at learning rate5e-7. Both branches explicitly restart at5e-5, cosine-decay by cumulative additional examples to5e-6 over400 complete passes, then retain5e-6 through500 passes. Adam moments and the absolute optimizer counter are preserved. Improvement over the old endpoint therefore combines bank expansion, extra exposure and a rate restart; these are not separately attributed by this two-arm study.

Checkpoints after25/100/250/500 complete passes receive the same512 validation observations and fresh fixed population c/Z as the old study, with512 posterior samples per case and4,096 samples for each of the three saved conditional observations. Existing converged joint references are reused; there is no new MCMC. Recovery includes all six parameters and c width relative to its prior. Full-prior SBC is reported alongside matched means/widths: good pooled ranks alone do not establish that c is learned. The new conditional family is384 checks (2metrics×8quantities×3cases×4endpoints×2arms), with the previous thresholds. The old60k baseline is descriptive and outside this new family. The market application remains separate from synthetic fidelity.

Each direct training worker has a hard5,400-second wall cap; the two-worker campaign has a5,700-second training cap, including startup and cleanup. Failed or capped runs are retained without automatic retry or extension. Reserved test arrays remain unopened. All full models, states, exposure arrays and conditional draws stay in ignored local artifacts; compact run records and checksums are retained here.

After these declared runs and diagnostics, stop for reflection before any grid revision, flexible-input work, asset promotion or Bergomi production.

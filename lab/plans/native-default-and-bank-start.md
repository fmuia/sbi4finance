# Native FlowMatching defaults and the first rough-bank shards

**The cached Heston solver check is complete. Native BayesFlow sampling changes interval widths, but the proposed conditional parameter checks still pass for only one of the three cases. The joint rough bank has 100 verified training surfaces, and the next 5,000 have been submitted.** The [execution ledger](../evidence/pilot-results/2026-10-27/native-bank-start/execution.json) and [evidence guide](../evidence/pilot-results/2026-10-27/native-bank-start/README.md) retain the exact inputs, sources, failures and accounting.

## Heston: use the native solver and keep model quality separate

The shared BF2.1 factory now leaves the FlowMatching solver at BayesFlow's native default for new workflows. Existing checkpoints retain their serialized settings. This check reused the completed 20k-trained, 500-epoch FlowMatching checkpoint, selected at epoch 403, whose stored solver remains **100 uniform Euler steps**. No weights, preprocessing, prior or quote-noise settings changed.

Only call-time sampling settings changed: `workflow.sample(method="tsit5", steps="adaptive", ...)`. The installed native sampling defaults were read and recorded before sampling:

| Setting | Native BF2.1 value |
| --- | --- |
| Method / step selection | Tsit5 / adaptive |
| Minimum / maximum steps | 50 / 10,000 |
| Absolute / relative tolerance | 1e-6 / 1e-4 |

These are solver settings, not a certificate of posterior accuracy. Each case used the same saved quote vector, fresh integer seed `271026 + case_index`, and 10,000-draw shape as its retained Euler comparison. The checkpoint and old Euler draw hashes were frozen before sampling and rechecked on completion. All three native arrays are finite with shape `(10000, 4)`; the stored Euler configuration and checkpoint bytes remain unchanged.

The observations are the three original readiness-bank cases, identified by manifest `e5c12a5f…`. They are distinct from the training/validation collection identified by snapshot manifest `1a7906f…`, despite overlapping short labels such as `validation:0`. This unit did not reopen the 500-observation validation panel or evaluate a reserved test pool.

### What changed

The following uses the same proposed development screen: absolute mean error at most 0.3 reference SD and 90% interval width within 25% of the reference. All parameter checks must pass for a case to pass that screen.

| Saved case | Initial-variance mean error, Euler → native, reference SD | Initial-variance 90% width/reference, Euler → native | All parameter checks with native |
| --- | ---: | ---: | --- |
| 0 | 0.024 → 0.025 | 1.069 → 1.212 | Pass |
| 1 | 1.400 → 1.391 | 1.067 → 1.186 | Fail |
| 2 | 1.492 → 1.451 | 1.715 → 2.014 | Fail |

Native sampling widens the initial-variance intervals by about 11–17%, while the largest parameter-mean change is only 0.0404 reference SD. Case 1's theta width ratio also rises from 1.267 to 1.375. The solver choice therefore affects uncertainty summaries, while the large initial-variance mean discrepancies persist. The overall count remains **1/3**, and the checkpoint remains development evidence.

The two analytic variance products show mixed tail changes. The largest native-versus-Euler q10/q90 movement is 0.000190 annualized variance units, for the forward-variance product in case 1. Some reference-quantile errors shrink and others grow. These mean-curve products do not check leverage, vol-of-vol or path-dependent prices; no product acceptance follows. Full parameter/product metrics and paired changes are retained in the [three native case reports](../evidence/pilot-results/2026-10-27/native-bank-start/heston-native/attempt-8010270).

No native-solver 500-case SBC/coverage result was generated. Earlier validation numbers remain results for their recorded Euler sampling policy. The measured timings also do not establish a solver speedup: this run loaded once and reused compiled code, while the preceding Euler comparisons ran in separate processes.

### The failed guard and actual cost

The first attempt, job 8010024 from source `c773ceb`, failed in our configuration guard before its first posterior sampling call. Deep-copying Keras's tracked configuration traversed its owner and encountered a non-pickleable module. The fix snapshots and compares the plain scalar mapping. A regression test now uses an actual BF2.1 FlowMatching tracked container with the posterior call mocked. The [failed attempt](../evidence/pilot-results/2026-10-27/native-bank-start/heston-native/attempt-8010024) remains intact; it is not evidence about Tsit5.

The corrected job 8010270, source `343cb45`, completed all three cases on one NVIDIA L40S with four allocated CPUs and 24 GiB host memory. Its effective settings, library/source hashes, runtime and input provenance are in [run.json](../evidence/pilot-results/2026-10-27/native-bank-start/heston-native/attempt-8010270/run.json).

| Attempt | Allocated elapsed seconds | Allocated CPU-seconds | Allocated GPU-seconds |
| --- | ---: | ---: | ---: |
| Guard failure 8010024 | 15 | 60 | 15 |
| Corrected run 8010270 | 43 | 172 | 43 |
| **Total** | **58** | **232** | **58** |

That is **0.06444 allocated CPU-hours and 0.01611 GPU-hours**. Reported used CPU time, counted across all threads, is 82.749 seconds. Batch/extern records are not added a second time. GPU kind is recorded directly; raw CPU-frequency accounting counters are not GPU clocks.

The reservation arithmetic needs one correction. The 630-second request gave a planned 885-second worst-case reservation when adding the failed 15 seconds, 180 seconds of KillWait and 60 seconds of cleanup allowance. Slurm rounded the requested limit to 660 seconds, so that reservation was actually **915 seconds**, 15 seconds beyond the original 900-second envelope. Actual allocation use was 58 seconds. Future reservations must use Slurm's rounded limit; the raw request, recipe and accounting are retained rather than rewritten.

## Joint rough bank: a frozen collection, started incrementally

The authorized first bank is **20,000 training / 5,000 validation / 7,000 reserved test surfaces**, using the broad six-parameter joint rough task, complete 63-quote design, working grids `1280/2016/2016`, and existing adaptive precision policy. Root seed is **271200**. No bank training or acceptance evaluation has started.

The frozen producer is source `c773ceb26e7273612b15c69fd63715d131778083` at `/oscar/scratch/afengler/sbi4finance-native-bank-20261027/repo`. Its collection lives under the same root's `collection/`, with collection ID `817de3664968de0c605ae4c73e2c99c26465344294af2fc5412f39fffa4baa50`. The [collection manifest](../evidence/pilot-results/2026-10-27/native-bank-start/collection.json), [job map](../evidence/pilot-results/2026-10-27/native-bank-start/jobs-20k.json.gz) and runtime/source hashes identify this producer independently of the corrected Heston checkout.

### Seed audit before pricing

The [audit](../evidence/pilot-results/2026-10-27/native-bank-start/seed-audit.json.gz) covers all 32,000 split-qualified IDs and every possible block/rung stream, including rungs a particular surface may never need:

- 672,000 possible block/rung streams.
- 704,000 unique reduced base/leaf pricing seeds.
- 768,000 unique initialized PCG64 states, including prior, quote and curve-purpose streams.
- Zero candidate collisions or intersections with the retained inventory of 2,640,086 reduced seeds and 3,400,256 initialized states.

The historical inventory conservatively includes declared but ungenerated IDs from old configurations. This is an exact no-repetition check over the retained identities/states, not a proof of mathematical stream independence or non-overlap. The deployed Linux preflight, job 8010043, reproduced both candidate digests and prepared the collection/job map without generating financial prices.

### Initial production status and expansion limits

**Status at 27 October 2026, 15:37 UTC:** all four jobs in array 8010139 completed and their shards passed independent integrity checks: **100 reusable training surfaces, IDs `train:0` through `train:99`**. There were no failures/timeouts, no reported MCSE/quote-SD values above 0.25 and no final-rung 0.5 exceptions; the maximum was 0.247752. Prior draws and noisy quotes were reconstructed exactly in the original Linux runtime. The [initial-bank summary](../evidence/pilot-results/2026-10-27/native-bank-start/bank-initial/summary.json) retains the manifests, checks and accounting. Array **8010762** was submitted for the next 200 shards, job-map entries 4–203, covering IDs `train:100` through `train:5099`: **5,000 additional training surfaces**. The [launch checks](../evidence/pilot-results/2026-10-27/native-bank-start/bank-wave-4-203/summary.json) confirmed all 200 workers running at 15:42:02 UTC, with the same source/runtime and no reported failure. Submitted work is not a verified row count, and no complete snapshot has been published.

The first 100 surfaces used **5,250 allocated CPU-seconds, or 1.4583 CPU-hours**, excluding the 79-second metadata preflight. Completed bank spending including that preflight is 5,329 CPU-seconds. This initial sample does not establish full-bank throughput or a revised completion forecast. At submission of the next wave, completed spending plus the full outstanding allocation/cleanup exposure and final reserve was **615.913611 CPU-hours**, within the 1,500-hour ceiling.

**Refill at 2026-10-27 18:07 UTC:** the accounting guard found 198 completed shards (4,950 published surfaces), 6 nonterminal workers and no failed jobs. Array **8019485** adds original job-map entries 204–396: **193 shards / 4,825 additional training surfaces**, IDs `train:5100` through `train:9924`. Its submission keeps total nonterminal workers at most 200. Completed spending plus all outstanding/new full limits, cleanup allowances and final integrity reserve is **761.08 CPU-hours**, within the existing ceiling. A subsequent independent check confirmed all 193 new workers running, five earlier workers still running (198 active total), unchanged producer/map/collection identities, and no hard failures or new log error markers. This is a refill of the frozen bank, with no new training or acceptance evaluation. The execution ledger records the original commands, budget check and queue verification.

Further waves remain bounded by **at most 200 nonterminal workers and 1,500 allocated CPU-hours total**, including setup, generation, recovery and final integrity work. Before submission, completed spending plus full queued/running/new time-limit exposure, cleanup allowances and the 1.1-CPU-hour final-integrity reserve must fit under that cap. Successful shards remain in the collection; failed IDs are retained rather than replaced. A full snapshot requires exact coverage and integrity checks, with reserved test outcomes unopened.

The [bank work order](rough-development-bank-work-order.md) contains the spending and preservation rules. This report records the first real start; it does not turn the partial collection or empirical MCSE policy into prior-wide pricing certification.

## Next: reuse the existing emulator, with three small correctness fixes

The bank generator and posterior factory do not call the older `pricer_net` helper. A read-only review identified three repairs before reusing it for the joint task:

1. Use **six parameter inputs only**. Its legacy feature function appends stored curve averages whenever present; those seven auxiliary values must not become conditions for joint calibration.
2. Train and select in the declared **quote-noise error units**. The existing output standardization makes its ordinary MSE a differently weighted objective. Keep any numerical scaling consistent with the intended noise-unit loss.
3. Match the trained and exported activations. Keras currently uses exact GELU while the NumPy evaluator uses a tanh approximation. Check prediction parity before using an exported network as a likelihood.

Reuse the existing checked PyMC deterministic-likelihood wrapper for references, passing only the Gaussian quote likelihood. PyMC supplies prior transforms/Jacobians; the older helper's transformed log-posterior must not be wrapped as another likelihood. Train/select on training and validation only, rather than reuse the legacy harness that opens tests and calls its custom Metropolis sampler. These are concrete baseline adaptations, not grounds for a new framework or architecture campaign. They have not been implemented or fitted in this bank-launch unit; fit and independent-label allocations remain separate from the generation ceiling.

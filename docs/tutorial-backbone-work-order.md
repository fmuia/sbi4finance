# Deliver the tutorial's data and estimator backbone

**Scope correction, 27 October 2026.** The requested outcome is the bank/checkpoint/evidence/cache set that collaborators and participants use. The new 20k Heston runs were development runs below Francesco's existing 50k baseline. Correct that mismatch and restore the known-curve rough lesson before treating the six-target joint extension as the core deliverable. [Central plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md).

## Work that can proceed together

| Work | Concrete next output | Boundary |
| --- | --- | --- |
| Heston data | Nested 50k and 100k training snapshots, retaining the current 20k prefix and 5k/5k validation/test pools. | Frozen original producer; append missing IDs only. A separate proposed **6 allocated CPU-hour** extension ceiling includes setup, workers, cleanup exposure and both integrity/freezes; no GPU. |
| Heston reproduction | One BF2.1 FlowMatching fit on 50k with Francesco's recorded recipe, native sampling, and comparison against the archived checkpoint. | Prepare the exact fit/evaluation manifest and cost before GPU submission. The 100k fit follows as a measured improvement experiment, not an automatic success claim. |
| Known-curve rough software | The existing three-target task works in the shared BF2.1 train/reload/sample/evaluate route, including supplied curves and rho×eta diagnostics. | Reuse the existing task, bank and trainer. First vector baseline; no new storage framework or global fixed-curve task. |
| Known-curve rough baseline | Recover the original compatible bank/model; if unavailable, use a clearly labelled reproduction/control and a shared comparison panel. Then produce the main bank and selected estimator. | Preserve task/prior/noise/numerical identities. Cost the chosen main bank and fits separately; do not charge them to the joint collection's generation ceiling. |
| Existing joint rough jobs | Preserve completed and already submitted work, original IDs and diagnostics for the third lesson. | Existing **1,500 allocated CPU-hour** ceiling. Foundational Heston/known-curve delivery takes priority over further replenishment. Stop expansion on failures under the original rules. |

## Heston extension: reuse rather than regenerate

The live collection is at:

```text
/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/runs/heston-development/
```

Collection ID: `402230db66b9885b59524e7808eebf1b3882c4a271f97eccc7cf970581e85221`; root seed **271028**. Its producer remains frozen at source `9d903db` in the root's `repo/`. The current optimized source is not interchangeable with that producer for appending shards. Use the existing collection `plan → shard → freeze` commands without weakening identity checks.

The original 60 shards contain 20k training, 5k validation and 5k test surfaces. Their recorded generation time totals 1,195.58 process CPU-seconds, around 39.85 ms/surface. Adding 80k training rows is an arithmetic **0.89 process CPU-hours**, before overhead. This supports a small bounded extension; it is not a completion guarantee or the actual allocated cost.

Before pricing, verify frozen source/runtime, existing payload identities and missing ID ranges. Audit the added IDs' streams against the existing collection and retained historical inventories. Preserve the existing holdout bytes. Plan the extension so 50k is a shard boundary, publish both immutable snapshots and record hashes and actual accounting. A timeout or numerical failure remains visible with its original range; no replacement seeds or unrecorded recovery.

## Reproduce the Heston estimator before changing the recipe

Use the original four 256-unit layers, 500 epochs, batch 256, seed 0, fresh quote noise, recorded AdamW/warm-up/cosine optimizer and alpha -0.5, with BayesFlow 2.1's native sampler as requested. The 50k run has about 98,000 updates, compared with 39,500 in the 20k run. Preserve the original checkpoint in its BayesFlow 2.0.14 environment as the control.

First compare both on the same development observations and references. Match the broader original evidence, including 2,000-case calibration and conditional/product comparisons; the three-case 1/3 versus 2/3 diagnosis is insufficient for replacement acceptance. If original observations cannot be recovered, freeze a new shared panel rather than comparing unrelated headline percentages. Keep raw and corrected/audited results separate, and reserve fresh final acceptance data.

The previous new 20k FlowMatching training/evaluation job used 322 allocated GPU-seconds; the archived 50k training recorded 1,927 seconds. These bound a plausible one-fit planning scale, not a guarantee for a new device or larger calibration panel. Freeze the full GPU/evaluation allowance in its run manifest before submitting; this data-extension work order does not silently grant an unbounded training campaign.

## Known-curve rough: reproduce the actual earlier task

The first rough lesson conditions on a known seven-level curve while inferring H, eta and rho. The curve varies across amortization examples and is fixed within each inference. Use `rough_calibration_v1`; the curve is a legitimate observed condition for this route. In `rough_joint_calibration_v1`, the stored curve averages remain auxiliary and must not enter inference.

The original pilot uses root 2612, 1,600/200/400 splits, the supplied-curve pilot prior and grids 80/504/504. Its reports are retained, but its model/raw bank are not tracked and their recorded paths are ephemeral. Recovery should be bounded. If those artifacts cannot be recovered, an explicit controlled re-fit in the old environment and BF2.1 on the same bank provides a comparison; it must not be labelled the original checkpoint or counted as new independent validation.

Start with the existing task/input contract: 63 quote-noise-scaled prices plus log curve levels, fresh noise on prices only, vector conditions, three targets. Include H, eta, rho and rho×eta in the simultaneous calibration screen, using `diagnostics_with_derived_targets["sbc"]` in the evaluator report; its top-level `sbc` remains parameter-only for compatibility. The original 250-case rank screen passed H/eta and failed rho/rho×eta; the delivered estimator must address those failures as well as conditional and product accuracy.

The main-bank planning size is 20k training plus fixed validation/test pools, with incremental growth available. Its final curve prior, grid, holdouts and measured allocated cost must be frozen before submission. Use the old numerical target for a fair reproduction control and independently checked references for any finer delivered target. A pilot curve prior does not cover market curves outside its support; retain an overlap benchmark if adopting Francesco's broader planned context prior.

## Delivery and stopping point

For each of the three routes, produce an immutable bank, selected reloadable estimator, exact recipe, benchmark/reference/product report and small cached notebook fixtures. Connect these to the actual course workflow so a participant can load an observation, sample a posterior, inspect diagnostics and propagate uncertainty without cluster access.

Matching or improving the existing studies is an explicit milestone. A larger bank or a passing software check does not satisfy it. If the matched 50k Heston or known-curve baseline still fails, diagnose that concrete gap before broad architecture screens or further data escalation. Keep PyMC where natural, preserve archived evidence and numerical limitations, and keep sparse chains downstream.

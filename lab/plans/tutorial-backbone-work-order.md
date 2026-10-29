# Deliver the tutorial's data and estimator backbone

**Current work, 28 October:** known-curve 100k and joint20k banks are checked; both intermediate models completed validation. Joint100k generation continues while the known100k fit, architecture pilot, price validation, conditional PyMC work and notebook integration proceed independently. Use the [current parallel work order](backbone-delivery-work-order.md), [rendered report](../evidence/reports/inference-performance-2026-10-28.pdf) and [notebook review guide](tutorial-backbone-review.md). The dated audit and execution notes below are retained history, not current run status.

**Active delivery, 28 October:** [both complete interior corner plots](interior-corner-plots.md) and the [usable Heston handover](full-surface-artifact-handover.md) are available. Actual generation arrays **8032877** (known curve) and **8032878** (joint curve) started with **64 + 120 running workers** and are being refilled by a durable bounded controller. The [delivery work order](backbone-delivery-work-order.md) records these intermediate 50k/20k milestones and the **100k training-surface baseline target for each rough route**, with larger allocation requirements explicit. This supersedes earlier paused/not-running status below; historical outcomes and costs are retained.

**Latest results, 28 October:** both [parallel validation tracks](backbone-validation-2026-10-28.md) completed. The frozen Heston final-500 candidate passes **4/4** rank screens on 2,000 untouched cases and **12/12** analytic development-product quantile checks, including uncertainty. The known-curve rough price panel completed all 48 batches and detected both emulator error and grid effects; the full neural/reference mismatch remains unresolved. Combined cost **2.6917 CPU-hours / .2708 GPU-hours**. No training or bank generation ran. Known-curve data remain 20k/5k/7k; joint generation is paused at 9,925 training rows. Next: package Heston examples, cost the next known-curve 50k milestone and improve/check its posterior and emulator under an explicit numerical scope.

**Scope correction, 27 October 2026.** The requested outcome is the bank/checkpoint/evidence/cache set that collaborators and participants use. The new 20k Heston runs were development runs below Francesco's existing 50k baseline. Correct that mismatch and restore the known-curve rough lesson before treating the six-target joint extension as the core deliverable. [Central plan](NUMERICAL-EXPERIMENT-PLAN.md).

**Audit correction, 28 October:** checkpoint selection discarded a better Heston model. The existing 100k run's final epoch 500 passes **4/4** rank screens on the same 2,000 observations and **3/3** conditional screens, versus **3/4 and 1/3** at selected epoch 402. The archive passes **4/4 and 2/3**. Repeating validation loss over 32 paired seeds also favors final 500, supporting noisy single-epoch selection as a contributor. These final-500 checks are post-hoc development evidence, not acceptance. We fixed a confirmed seed/import-order defect for future runs. Freeze a defensible selection rule and assess the candidate on fresh acceptance data and products before another architecture or data expansion. [Rigorous comparison and evidence](heston-replication-audit.md). The 27 October results and costs below remain the historical selected-checkpoint record.

**Execution update, 27 October:** Heston's nested **50k/100k training snapshots**, both fits and their **2,000-case development evaluations** are complete. The extension preserved the prefix and holdouts and cost **1.0114 CPU-hours**, no GPU. The archive passes four rank checks; both new fits pass three. At 100k, initial-variance KS improves **0.06997 → 0.05084**, still above the **0.03562** band. Conditional screens pass **1/3**, versus **2/3** for the archive. The reviewed evaluation continuation preserved its original timeout; a subsequent preprocessing/readout audit passed and tighter tolerances made negligible differences. Keep the BayesFlow default and diagnose the learned posterior next. Full fit/control/diagnostic cost: **4.9056 CPU-hours / 1.0722 GPU-hours**, inside the existing ceilings. [Comparison and retained failures](heston-backbone-comparison-2026-10-27.md), [data evidence](../evidence/pilot-results/2026-10-27/tutorial-backbone/README.md).

The known-curve **20k train / 5k validation / 7k test bank is complete and checked**, using **95.2475 allocated CPU-hours**, no GPU, below its separate 150-hour ceiling. All 1,280 shards completed without retries. One retained training row, `train:3999`, has reported MCSE/quote-SD **0.3020**, above nominal 0.25 and within the predeclared final-rung 0.5 cap; validation/test rows all meet 0.25. This is empirical precision, not a continuous-price certificate. The first posterior/pricer fits and 2k evaluation completed at **1.12 CPU-hours / 0.28 GPU-hours**. **All four rank checks pass**, including rho and rho×eta; some 50% interval-coverage deviations remain. [Bank check](../evidence/pilot-results/2026-10-27/backbone-runs/known-bank-final/README.md), [fit and coverage report](../evidence/pilot-results/2026-10-27/backbone-runs/known-fit/README.md).

Three fixed-case **PyMC emulator posteriors** completed with passing parameter/rho×eta convergence checks, using **788 CPU-seconds**, no GPU, below their separate 6 CPU-hour ceiling. Their completed neural comparisons pass the proposed parameter rules on **case 0 only**; cases 1/2 fail, including a **2.05 reference-SD eta mean difference** on case 2. These approximate mean-emulator targets omit pricing-label randomness/covariance, so the discrepancy cannot yet be assigned solely to neural error. The immediate next check is independent pricing at posterior representatives for cases 0 and 2, at the bank and doubled grids, before further rough training. Fit plus comparison cost **1.2133 CPU-hours / 0.3033 GPU-hours**, inside its existing pool. [PyMC records](../evidence/pilot-results/2026-10-27/backbone-runs/known-emulator-references/README.md), [conditional comparison](../evidence/pilot-results/2026-10-27/backbone-runs/known-emulator-comparison/README.md).

The three joint-bank timeouts were recovered on other nodes with unchanged source and seeds, and all original parts/failed attempts were preserved. The collection now has **397 checked parts / 9,925 training surfaces**; its validation/test pools and complete 32k snapshot are still pending. Recovery cost **1.1353 CPU-hours** inside its 9.5-hour bound; total joint allocation **283.4164 CPU-hours** remains below the original 1,500-hour ceiling. No wider joint refill was submitted by this unit. [Review and recovery](../evidence/pilot-results/2026-10-27/backbone-runs/joint-stop/recovery-378-380/README.md).

## Work that can proceed together

| Work | Concrete next output | Boundary |
| --- | --- | --- |
| Heston data | Complete: reuse nested 50k and 100k training snapshots, retaining the original 20k prefix and 5k/5k validation/test pools. | Extension used **1.0114 allocated CPU-hours**, no GPU, below its separate 6-hour ceiling. Frozen original producer; future appends must retain IDs and holdouts. |
| Heston reproduction | Frozen candidate's 2k held-out gate and analytic product checks passed. Prepare cached teaching examples and regeneration commands; validate the declared path-product contract before using it. | Keep the native default and earlier failures. One-candidate and analytic-product scope; broader conditional/seed confirmation and notebook packaging remain. No new training required by this unit. |
| Known-curve rough software | The existing three-target task works in the shared BF2.1 train/reload/sample/evaluate route, including supplied curves and rho×eta diagnostics. | Reuse the existing task, bank and trainer. First vector baseline; no new storage framework or global fixed-curve task. |
| Known-curve rough baseline | Independent six-point price panel completed. Cost the next 50k training milestone and posterior/emulator refits, preserving a nested 20k comparison and the fixed price-error benchmark. | Original finite-grid bank remains useful. Emulator and grid errors are measurable; do not treat emulator-PyMC as exact truth or a larger bank as a discretization fix. No expansion submitted in this unit. |
| Joint rough | Recovery complete; preserve all **9,925** training rows. Resume only missing original map entries when proceeding with the remaining third-lesson bank. | Existing **1,500 CPU-hour** ceiling; **283.4164** allocated so far. No replacement seeds, lost successful work or relabelled known-curve data. |

## Heston extension: reuse rather than regenerate

The live collection is at:

```text
/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/runs/heston-development/
```

Collection ID: `402230db66b9885b59524e7808eebf1b3882c4a271f97eccc7cf970581e85221`; root seed **271028**. Its producer remains frozen at source `9d903db` in the root's `repo/`. The current optimized source is not interchangeable with that producer for appending shards. Use the existing collection `plan → shard → freeze` commands without weakening identity checks.

The checked snapshot paths are:

```text
/oscar/scratch/afengler/sbi4finance-heston-backbone-20261027/snapshot-50k/
/oscar/scratch/afengler/sbi4finance-heston-backbone-20261027/snapshot-100k/
```

The original 60 shards contain 20k training, 5k validation and 5k test surfaces. Their recorded generation time totals 1,195.58 process CPU-seconds, around 39.85 ms/surface. Adding 80k training rows is an arithmetic **0.89 process CPU-hours**, before overhead. This supports a small bounded extension; it is not a completion guarantee or the actual allocated cost.

The completed preflight verified frozen source/runtime, existing payload identities and missing ID ranges. The added IDs' stream audit passed against the existing collection and retained historical inventories. Preserve the existing holdout bytes. Plan the extension so 50k is a shard boundary, publish both immutable snapshots and record hashes and actual accounting. A timeout or numerical failure remains visible with its original range; no replacement seeds or unrecorded recovery.

## Reproduce the Heston estimator before changing the recipe

Use the original four 256-unit layers, 500 epochs, batch 256, seed 0, fresh quote noise, recorded AdamW/warm-up/cosine optimizer and alpha -0.5, with BayesFlow 2.1's native sampler as requested. The 50k run has about 98,000 updates, compared with 39,500 in the 20k run. Preserve the original checkpoint in its BayesFlow 2.0.14 environment as the control.

First compare both on the same development observations and references. Match the broader original evidence, including 2,000-case calibration and conditional/product comparisons; the three-case 1/3 versus 2/3 diagnosis is insufficient for replacement acceptance. If original observations cannot be recovered, freeze a new shared panel rather than comparing unrelated headline percentages. Keep raw and corrected/audited results separate, and reserve fresh final acceptance data.

The previous new 20k FlowMatching training/evaluation job used 322 allocated GPU-seconds; the archived 50k training recorded 1,927 seconds. These bound a plausible one-fit planning scale, not a guarantee for a new device or larger calibration panel. Freeze the full GPU/evaluation allowance in its run manifest before submitting; this data-extension work order does not silently grant an unbounded training campaign.

## Known-curve rough: reproduce the actual earlier task

The first rough lesson conditions on a known seven-level curve while inferring H, eta and rho. The curve varies across amortization examples and is fixed within each inference. Use `rough_calibration_v1`; the curve is a legitimate observed condition for this route. In `rough_joint_calibration_v1`, the stored curve averages remain auxiliary and must not enter inference.

The original pilot uses root 2612, 1,600/200/400 splits, the supplied-curve pilot prior and grids 80/504/504. Its reports are retained, but its model/raw bank are not tracked and their recorded paths are ephemeral. Recovery should be bounded. If those artifacts cannot be recovered, an explicit controlled re-fit in the old environment and BF2.1 on the same bank provides a comparison; it must not be labelled the original checkpoint or counted as new independent validation.

Start with the existing task/input contract: 63 quote-noise-scaled prices plus log curve levels, fresh noise on prices only, vector conditions, three targets. Include H, eta, rho and rho×eta in the simultaneous calibration screen, using `diagnostics_with_derived_targets["sbc"]` in the evaluator report; its top-level `sbc` remains parameter-only for compatibility. The original 250-case rank screen passed H/eta and failed rho/rho×eta; the delivered estimator must address those failures as well as conditional and product accuracy.

The main-bank planning size is 20k training plus fixed validation/test pools, with incremental growth available. Its final curve prior, grid, holdouts and measured allocated cost must be frozen before submission. Use the old numerical target for a fair reproduction control and independently checked references for any finer delivered target. A pilot curve prior does not cover market curves outside its support; retain an overlap benchmark if adopting Francesco's broader planned context prior.

The current matched collection is `/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/collection`, ID `da963f7294aee242a80ebf9b47d914294b439aef8b6ee8f7c6d1899766e546f1`. Its producer is frozen at `6163520`; [configuration](../configs/calibration_rough_known_backbone_20k.json). The stream audit found no collisions within the new bank or against the retained Heston, joint and historical inventories. Workers append original ranges only; the controller stops on failure or ambiguous submission and reserves the full outstanding wall limits plus cleanup before adding work. This is a bounded generation script, not new runtime infrastructure.

Its checked snapshot is `/oscar/scratch/afengler/sbi4finance-rough-known-backbone-20261027/snapshot-20k`, manifest SHA `7cf0314dc6371fd045ad6b314dd6dd605db2f3343dbccf4e35b4df9ddf9397eb`. The final allocation reconstructed every prior draw, curve and noisy quote exactly and preserved all initial 100 payload/manifest hashes. Training uses source `845c458` at `/oscar/scratch/afengler/sbi4finance-rough-known-fit-20261027/repo`; outputs are under that root's `runs/known-20k-native`. These distinct producer/trainer revisions are intentional; no bank is regenerated by training.

Use the ATM forward-start call for nontrivial known-curve product uncertainty: its price depends on H, eta and rho. The continuously monitored variance-swap mean rate is fixed by the supplied curve in this lesson, so it is a deterministic control. It becomes uncertain when curve coefficients are inferred in the joint lesson; do not copy Heston's variance-product comparison as though it tested all rough parameters.

## Delivery and stopping point

For each of the three routes, produce an immutable bank, selected reloadable estimator, exact recipe, benchmark/reference/product report and small cached notebook fixtures. Connect these to the actual course workflow so a participant can load an observation, sample a posterior, inspect diagnostics and propagate uncertainty without cluster access.

Matching or improving the existing studies is an explicit milestone. A larger bank or a passing software check does not satisfy it. If the matched 50k Heston or known-curve baseline still fails, diagnose that concrete gap before broad architecture screens or further data escalation. Keep PyMC where natural, preserve archived evidence and numerical limitations, and keep sparse chains downstream.

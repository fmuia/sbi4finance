# Heston backbone comparison — 27 October 2026

> **28 October audit correction:** the results below describe the minimum-loss selected checkpoints. The same 100k run's saved **final epoch 500** passes **4/4** rank screens on the unchanged 2k panel and **3/3** cached conditional screens, compared with **3/4 and 1/3** for selected epoch 402 and **4/4 and 2/3** for the archive. No retraining was needed. The [replication audit](heston-replication-audit.md) records the checkpoint finding, an import-order reproducibility fix and the remaining comparison limits. Checkpoint selection now takes priority over an architecture change; final 500 is a post-hoc development candidate, not an accepted tutorial estimator. The historical selected-checkpoint evidence below is retained unchanged.

**The data-count mismatch is corrected, but the new estimators have not yet matched Francesco's archive.** On the same 2,000 validation observations, the archive passes four rank-calibration screens; the new 50k and 100k fits each pass three. Initial-variance KS improves from **0.06997 to 0.05084** at 100k, still above **0.03562**. Conditional agreement remains weaker than the archive. The completed readout/preprocessing diagnosis supports keeping BayesFlow's default sampler and focusing the next work on the learned posterior. No new Heston estimator is accepted.

## A matched calibration panel

The estimators use the same Heston task, prior, quote-noise law and `validation:0` through `validation:1999`, with 1,000 posterior draws per observation. The archive/new-50k comparison uses the 50k snapshot; the nested 100k snapshot preserves those exact validation parts. The new models used this validation pool for selection; it is development evidence, not untouched acceptance.

| Parameter | Archive KS | New 50k KS | New 100k KS | Archive 95% coverage | New 50k 95% | New 100k 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial variance | 0.02215 | **0.06997** | **0.05084** | 95.00% | 95.85% | 94.75% |
| Theta | 0.02757 | 0.02371 | 0.01879 | 95.00% | 95.55% | 95.40% |
| Vol-of-vol | 0.02189 | 0.03229 | 0.02472 | 94.55% | 95.30% | 95.45% |
| Rho | 0.02387 | 0.01827 | 0.02075 | 95.40% | 95.80% | 95.45% |

The simultaneous four-parameter DKW bound is **0.035620**. The archive passes **4/4** and both new fits **3/4**. More data/updates reduce the initial-variance rank deviation, but do not close the archive gap. Close-to-nominal coverage at one interval level does not settle calibration across the distribution; the absolute KS statistic alone does not identify a signed global bias. [Completed 100k panel](pilot-results/2026-10-27/backbone-runs/heston-100k/validation-continuation/evaluation.json).

The new fit completed its 500-epoch schedule on 50k training surfaces and selected **epoch 445** by minimum validation loss. Reload passed. The archive retains its original BayesFlow 2.0.14 preprocessing and 100-step nonuniform Euler policy; the new fit uses BayesFlow 2.1 and native adaptive Tsit5. [New validation report](pilot-results/2026-10-27/backbone-runs/heston-50k/validation/evaluation.json), [training record](pilot-results/2026-10-27/backbone-runs/heston-50k/checkpoint/training.json), [paired archived control](pilot-results/2026-10-27/backbone-runs/archived-control/attempt-8029251/evaluation.json).

## The three cached conditional comparisons

These are the three original readiness observations, **not** the first three rows of the 2,000-case panel. The old and new comparisons have identical observed quote vectors and reference-file hashes. The retained four-chain PyMC references passed their diagnostics; that does not prove global mode coverage.

The proposed point-estimate screen requires every parameter's absolute mean error to be at most **0.3 reference SD**, and its central 90% interval width/reference width to lie in **[0.75, 1.25]**.

| Case | Initial-variance mean error: archive / 50k / 100k | Initial-variance width ratio: archive / 50k / 100k | New 50k recorded failure |
| --- | ---: | ---: | --- |
| 0 | 0.273 / 0.311 / 0.482 SD | 0.938 / 0.988 / 1.136 | Initial-variance mean, borderline. |
| 1 | 0.192 / 0.075 / 0.067 SD | 1.035 / 1.156 / 1.089 | Theta width ratio **1.277**. |
| 2 | 0.432 / 0.712 / 0.652 SD | 1.158 / 1.308 / 1.355 | Initial-variance mean and width. |

The binary counts are **2/3 for the archive, 0/3 for new 50k and 1/3 for new 100k**. The 100k fit passes case 1; cases 0 and 2 still fail initial-variance mean agreement, and case 2 also fails width agreement. More rows and updates have not established a replacement-quality improvement. The 50k continuous discrepancies matter:

- Case 0's mean error is **0.31077**, only 0.01077 SD beyond the rule. Reference mean MCSE alone is **0.02191 SD**, so this threshold crossing is not a well-resolved distinction. Keep the recorded failure rather than silently upgrading it to a pass.
- Case 1's theta width is **1.27655**, slightly beyond 1.25. Reference quantile MCSE has not been computed, so the uncertainty in that width comparison is unresolved.
- Case 2 retains a substantial initial-variance discrepancy: mean error **0.71229 SD**, against reference mean MCSE **0.01967 SD**, and width ratio **1.30801**. Borderline cases do not explain this remaining problem.

Neural sampling uncertainty, reference quantile uncertainty and numerical posterior stability are not fully assessed by these point estimates. These three cases do not replace the original 50-case benchmark. [Case 0](pilot-results/2026-10-27/backbone-runs/heston-50k/comparisons/case-0/comparison.json), [case 1](pilot-results/2026-10-27/backbone-runs/heston-50k/comparisons/case-1/comparison.json), [case 2](pilot-results/2026-10-27/backbone-runs/heston-50k/comparisons/case-2/comparison.json), [archived conditional control](pilot-results/2026-10-27/followup/archived-heston/run.json).

Analytic variance-product results are mixed too. For case 1's six-to-twelve-month forward-variance rate, absolute q10 error increases from **0.0000372 to 0.0004824 annualized variance units**. Other tails improve. These mean-curve functionals do not test leverage, vol-of-vol or path-dependent prices; product quantile MCSE and economic acceptance remain unassessed. A passing parameter or coverage screen would not by itself settle product quality.

## What the extra data has and has not shown

Against the *same new 20k checkpoint sampled natively*, initial-variance mean errors change from **[0.025, 1.391, 1.451]** to **[0.311, 0.075, 0.712] SD** at 50k. Cases 1 and 2 improve considerably, while case 0 worsens. Case 1's theta width ratio also improves from 1.375 to 1.277, but remains outside the point-estimate rule. [20k native results](native-default-and-bank-start.md).

The historical 20k, 500-case SBC result used **100-step Euler** and passed 2/4 parameters. It cannot be treated as a controlled predecessor to the new **2,000-case native** result: panel size, simultaneous-screen threshold and sampling policy changed. The 20k checkpoint has no native 2,000-case result.

Moreover, 500 epochs on 20k means about **39,500 updates**, versus **98,000 on 50k**; data coverage and optimization effort both increase. Archive-versus-new comparisons also change the training bank realization, framework/flow-time conventions and solver. Architecture, optimizer and alpha -0.5 are substantially matched, but fresh-noise streams and chunk cadence differ. The new 50k and 100k sources, `6163520` and `eb53cea`, have no changes in the core trainer, evaluator or inference modules; the larger run still increases both rows and updates. None of these comparisons isolates a data-only effect. [Earlier findings and recipe](followup-results-2026-10-27.md).

## Actual cost and next milestone

| Work, including retained attempts | Allocated CPU-seconds | GPU-seconds |
| --- | ---: | ---: |
| New 50k fit, 2k validation and three conditional checks, job 8028749 | 5,248 | 1,312 |
| Archived CPU attempt, cancelled with 500 cases committed, job 8028804 | 2,220 | 0 |
| Archived GPU host-configuration failure before sampling, job 8028941 | 52 | 13 |
| Corrected archived continuation, job 8029251 | 212 | 53 |
| **Comparison total** | **7,732 (2.1478 CPU-hours)** | **1,378 (0.3828 GPU-hours)** |

Allocation is top-level elapsed time × allocated CPUs; batch/extern records are not added again. The archived continuation preserves all five committed CPU batches byte-for-byte and completes the remaining 1,500 cases on GPU. This explicitly mixed readout does not support a speedup claim. Both failed/incomplete attempts remain visible. The separate completed Heston bank extension used **3,641 allocated CPU-seconds**, no GPU, and is not charged again here. [New fit accounting](pilot-results/2026-10-27/backbone-runs/heston-50k/accounting.psv), [archived attempt records](pilot-results/2026-10-27/backbone-runs/archived-control/), [data milestone](tutorial-backbone-work-order.md).

The 100k run, job **8030120**, selected epoch **402** and passed exact seeded reload. Training took 488.83 seconds. All three cached comparisons completed, but the 2k validation process exceeded its **900-second internal cap**, leaving no completed broad report. The evaluator retained draws in memory until the whole panel finished, so those unfinished draws are not reusable artifacts. The original job is recorded **FAILED**, its run journal **incomplete**; the trained checkpoint is retained.

The attempt used **5,932 allocated CPU-seconds / 1,483 GPU-seconds**, including the failed validation. The reviewed **evaluation-only continuation**, job **8030765**, completed using the exact checkpoint, 2,000 IDs, sampling seeds and native policy, in a new output directory. It used **3,612 CPU-seconds / 903 GPU-seconds**, performing no training, pricing, MCMC or test evaluation. Its completed panel supplies the table above; the original job remains failed. [Original run and failure](pilot-results/2026-10-27/backbone-runs/heston-100k/run.json), [100k conditional reports](pilot-results/2026-10-27/backbone-runs/heston-100k/comparisons/), [continuation and accounting](pilot-results/2026-10-27/backbone-runs/heston-100k/validation-continuation-execution.json).

The completed 50k/control work, original 100k attempt and continuation total **17,276 CPU-seconds / 3,764 GPU-seconds**. The subsequent diagnostic adds **384 CPU-seconds / 96 GPU-seconds**, for **17,660 CPU-seconds (4.9056 hours) / 3,860 GPU-seconds (1.0722 hours)**, below the pooled **6 CPU-hour / 1.25 GPU-hour ceilings**. The separate bank-generation cost is not included again.

## Readout and preprocessing diagnosis

Job **8030926** audited the three saved observations through the actual adapter and stored standardizers, then paired the native readout with ten-times tighter tolerances using identical latent seeds. All preprocessing checks and 58 frozen hashes passed; native samples reproduced the saved arrays bit-for-bit. The largest mean shift was **0.000676 reference SD** and the largest 90% width change **0.0768%**. Conditional passes stayed **1/3**. This does not explain the remaining errors of roughly 0.48/0.65 SD. [Diagnostic evidence](pilot-results/2026-10-27/backbone-runs/heston-100k/readout-diagnosis/README.md).

Retain BayesFlow's native tutorial default. The next focused experiment should address the learned conditional approximation or optimization variability, using the existing 100k bank. One costed spline-density comparison would answer a different question than more FlowMatching data/epochs; avoid a broad architecture screen. Source inspection found equivalent endpoint weighting after BayesFlow's time reversal, and no obvious adapter or standardization error on the audited inputs; this does not rule out every training issue. One seed per bank does not quantify optimization variability, and minimum stochastic FlowMatching validation loss does not scientifically rank the fits. Final conditional/product checks and untouched acceptance remain required.

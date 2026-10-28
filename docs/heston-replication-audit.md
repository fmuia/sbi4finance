# Heston replication audit — 28 October 2026

> **Follow-up completed:** the candidate was frozen and passed **4/4** rank screens on 2,000 untouched cases plus **12/12** analytic development-product quantile checks, including uncertainty. The [follow-up report](backbone-validation-2026-10-28.md) records that narrower gate and remaining delivery work. The post-hoc development results and limitations in this audit remain unchanged.

**The reported regression partly came from checkpoint selection. The 100k run had already saved a candidate with substantially better performance on the checked development diagnostics than the epoch-402 model we evaluated.** Holding that training run, data, native sampler, observations and PyMC references fixed, its saved final epoch-500 model passes all three conditional screens; the selected model passes one. Francesco's archived model passes two. This is a post-hoc development finding, not final acceptance.

The audit also found genuine reproduction differences, including an import-order reproducibility defect. Matching the labels “seed 0, 500 epochs, four 256-unit blocks” did not reproduce Francesco's experiment. We should have checked the saved final checkpoint before proposing another architecture.

## The strongest direct finding: which checkpoint we evaluated

Training actually completed all 500 epochs. The new 100k run saved epoch 402 as `model.keras` because that epoch had the lowest single-epoch validation FlowMatching loss. Its epoch-500 state remained in protected recovery storage and was not used in the original comparison.

Job `8031924` read both saved models, with no retraining, new prices or MCMC. All rows below use the same three observations, four-chain PyMC references, 10,000 neural draws and within-framework paired latent seeds. Errors are absolute posterior-mean differences in reference standard deviations; widths are central-90% neural/reference ratios.

| Case | Archive, tight adaptive: v0 mean error / width | New 100k selected 402 | Same run, final 500 |
| --- | ---: | ---: | ---: |
| 0 | 0.2610 / 0.9832 | 0.4822 / 1.1364 | **0.0338 / 0.9837** |
| 1 | 0.2025 / 1.0739 | 0.0671 / 1.0889 | **0.0024 / 1.0431** |
| 2 | 0.4373 / 1.2303 | 0.6520 / 1.3555 | **0.2456 / 1.1476** |

All four parameters meet the proposed mean/width rules on **3/3 cases for final 500**, versus **1/3 for selected 402** and **2/3 for the archive**. The archive retains two passes with either its prescribed end-dense Euler readout or tight adaptive Tsit5; changing its sampler does not reverse this conclusion. The new 50k final checkpoint improves from zero to one pass, so the finding is not that final epochs always solve every problem.

These are three previously examined development cases, several near prior boundaries. Reference mean MCSE and neural sampling error remain relevant near thresholds; quantile MCSE is not established. Their improvement does not establish broad calibration or a universal preference for final epochs. [Frozen-checkpoint controls](pilot-results/2026-10-27/heston-replication-audit/checkpoint-controls/README.md).

The unchanged 2,000-case validation panel now confirms the improvement beyond those three cases. Job `8032002` evaluated final 500 with the original 1,000 draws per observation, batch size, seeds and native sampling policy:

| Parameter | Archive KS | New 100k selected 402 KS | Same run, final 500 KS |
| --- | ---: | ---: | ---: |
| Initial variance | .02215 | .05084 | **.01383** |
| Theta | .02757 | .01879 | .02824 |
| Vol-of-vol | .02189 | .02472 | .02563 |
| Rho | .02387 | .02075 | .02118 |
| Parameters inside simultaneous .03562 band | **4/4** | **3/4** | **4/4** |

Final-500 central-95% coverage is **95.80%, 95.65%, 95.10%, 95.70%**, respectively. Some lower-level coverage deviations remain: theta/vol-of-vol at 50% and vol-of-vol at 80% exclude nominal in pointwise, unadjusted Wilson intervals. This is mixed parameter-by-parameter evidence, not uniform superiority over the archive. A passing marginal rank screen does not prove conditional or joint calibration. Its role is explicitly **post-hoc development candidate**, not untouched acceptance. The original selected-402 failures remain retained. [Full final-panel evidence](pilot-results/2026-10-27/heston-replication-audit/final-panel/README.md).

## What changed, and what did not

| Aspect | Francesco's archived run | Our new runs | Audit conclusion |
| --- | --- | --- | --- |
| Financial task | Heston, fixed kappa 2, four inferred parameters, 63 quotes | Identical task, prior bounds, quote noise and Fourier ladder | No model/pricing-law degradation found. |
| Training data | 50k surfaces, root 2610 | 50k/100k nested surfaces, root 271028 | Same distribution, different realizations; not an exact data reproduction. |
| Model capacity | 1,009,172 trainable parameters, 41 tensors | Exactly the same count, shapes and dimensions | No accidental smaller model or masked inputs. |
| Framework | BayesFlow 2.0.14 | BayesFlow 2.1.0 | Population FlowMatching problem equivalent after time reversal; initialization and finite optimization trajectories differ. |
| Initialization | Import BayesFlow, then reset seed 0 | Reset seed 0, then import BayesFlow | Build-only replication shows all 17 random tensors differ, due to import side effects and a new seed generator. Import order corrected for future runs. |
| Stochastic training | Original quote-noise namespace and separate base/time RNGs | Different quote-noise namespace; new base/time generator arrangement | Same intended distributions, different samples. |
| Chunking | 25 epochs | 50 epochs | Dataset reconstruction/shuffle cadence changed: 20 versus 10 constructions. |
| Optimizer | Explicit AdamW, .005 decay, 1.5 clipping, warm-up/cosine | Same explicit function and schedule law | No reset bug. Doubling data also changed update count and cumulative decay. |
| Selection | Minimum stochastic validation loss, epoch 487 | Same nominal criterion, epochs 445/402 | That criterion selected a weaker posterior in the 100k run. |
| Sampling | 100 end-dense Euler steps | Native adaptive Tsit5 | Both policies applied as intended. Readout checks do not explain the poorer selected model. |

The original trainer's recorded SHA matches [commit 18d8902](https://github.com/fmuia/sbi4finance/blob/18d89025a06c71dd170a73983a57cf47a4f9349b/tutorial_final/studies/train_calibration.py), not merely the latest legacy wrapper. The inspected installed framework files match their upstream release tags byte-for-byte. The archived model records Keras 3.15.1; its original training device and precision flags were not recorded, so training timings cannot establish a CPU/GPU explanation. [Source and run summary](pilot-results/2026-10-27/heston-replication-audit/training-summary.json), [data/provenance audit](pilot-results/2026-10-27/heston-replication-audit/heston_data_audit_20261027.json).

## A real reproducibility defect, with a regression test

BayesFlow 2.1 import creates class-level initializers in two score-network classes. Those consume two Python RNG seeds even though this experiment does not use those networks. Its default `MaskHandler` creates a seed generator during network construction, consuming another Python RNG draw and shifting subsequent initializer seeds.

Our new trainer reset the global seed before the library import; Francesco's trainer imported the library first. Build-only CPU experiments confirmed:

- Both versions have **41 trainable tensors**, of which 17 are randomly initialized and 24 are constants.
- All 17 random tensors differ under the nominal seed 0 paths.
- Accounting for the three extra seed draws reproduces **all 41 tensor values exactly** in the build-only comparison.
- Zero-probability masking introduces no extra inputs, missing values or loss scaling. With aligned weights, synthetic forward outputs match bit-for-bit.

This demonstrates initialization drift. It does **not** prove that the new initialization caused the weaker posterior. Equal raw weights also do not align the physical velocity function after time reversal: the corresponding field is `v_new(t,x) = -v_old(1-t,x)`.

The trainer now imports BayesFlow before resetting the training seed. The regression test runs the actual construction path in fresh subprocesses, with and without a prior BayesFlow import. It failed before the fix and passes afterward with all 41 initial tensor hashes equal. A tiny synthetic training/reload check also passes. We have not added magic seed offsets or altered historical model weights. This fixes within-version cold/warm reproducibility; it does not promise identical seeds across framework versions. [Initialization proofs](pilot-results/2026-10-27/heston-replication-audit/initialization/), [regression result](pilot-results/2026-10-27/heston-replication-audit/seed-regression-test.json).

## Why more data did not guarantee a better reported checkpoint

Single-epoch FlowMatching validation includes newly sampled Gaussian noise and interpolation times. Its last 50-epoch standard deviation was approximately **.00543 / .00443 / .00506** for the archive/new 50k/new 100k runs. Their late training-loss means were almost identical: **.20998 / .21053 / .20939**. A single minimum among 500 stochastic estimates is not a reliable ranking of posterior accuracy.

We tested that explanation directly in job `8032144`: **32 paired independent seeds**, each evaluating both saved models on the same 5,000 stored validation observations through BayesFlow's actual stateless validation loss. No gradients or optimizer updates were computed.

| Checkpoint | Mean validation loss over 32 replicates | Replicate SD |
| --- | ---: | ---: |
| Selected epoch 402 | .208772017 | .004356517 |
| Final epoch 500 | **.207083840** | .004349081 |

The paired final-minus-selected difference is **−.001688177**, with Monte Carlo SE **.000053663**. Final 500 therefore also has a lower averaged surrogate loss on this validation set; its better posterior diagnostics do not require accepting a higher average loss. This supports noisy single-epoch selection as a contributor to the checkpoint mistake. It does not establish a universally optimal epoch or an independent acceptance result. The replay uses fixed canonical row order rather than the historical shuffle and explicit independent noise/time seeds, with each model's saved standardizer; those differences are recorded. [Repeated-loss audit](pilot-results/2026-10-27/heston-replication-audit/validation-loss-mc/README.md).

| Quantity | Archive 50k | New 50k | New 100k |
| --- | ---: | ---: | ---: |
| Updates in full 500-epoch schedule | 98,000 | 98,000 | 195,500 |
| Selected epoch | 487 | 445 | 402 |
| Updates at selected checkpoint | 95,452 | 87,220 | 157,182 |
| Nominal zero-gradient AdamW factor at selection | .884167 | .884425 | .784810 |

The decay factors are products of `1 − learning_rate(step) × .005`, not observed weight norms; gradients are omitted. They show that doubling rows at fixed epochs also changes optimization and cumulative regularization. The full-schedule nominal reduction is 11.58% at 50k and 21.78% at 100k. This is another reason the comparison is not a data-only experiment. [Actual schedule calculation](pilot-results/2026-10-27/heston-replication-audit/initialization/audit_heston_adamw_decay.json).

We ruled out optimizer/standardizer resets: the same optimizer persists across chunks, neither run resumed, and stored standardizer counts match the selected epochs × rows exactly. Archived count 24,350,000 equals 487×50,000; new 100k count 40,200,000 equals 402×100,000. Clean and validation preprocessing tensors match in an independent synthetic comparison.

## How strong was the original “ours is worse” evidence?

The **selected-checkpoint v0 regression** was real on the shared 2k observations: archive KS .02215, new 50k .06997, selected 100k .05084, against band .03562. Our 100k theta and rho ranks were actually better than the archive's, so “uniformly worse inference” was already too broad. The panel was used for our validation selection and was fresh to the archive; this did not disadvantage our models.

Francesco's historical 40/50 agreement is a separate experiment, primarily defensive importance reweighting with 2,048 proposals per case and four sampler checks, not 50 independently converged PyMC references. Proposal geometry and chain starts came from neural draws; four importance cases had Pareto k above .7, and the maximum reported sampler R-hat was 1.0111, versus our present 1.01 reference rule. These are different reference procedures and quality screens. His shipped network also followed three rounds of methodological diagnosis, including endpoint weighting and sampler refinement. Those facts do not invalidate the archive's strong shared-panel performance, but they prevent treating all historical headline numbers as identical benchmarks. The current three-observation comparison uses exactly the same PyMC references for both estimators.

## Corrective work order

1. Retain the completed final-500 panel as a separate post-hoc candidate. Preserve all selected-checkpoint failures and the negative/positive controls.
2. Freeze the current candidate before reserved acceptance, and a selection rule before any future training: a fixed-budget endpoint, or a small declared late-checkpoint set compared using repeated/common-randomness loss estimates and a fixed conditional development panel. Do not infer that the last epoch is always best from these cases, or assume lower surrogate loss guarantees better posteriors.
3. Assess the frozen candidate on fresh acceptance data and the broader conditional/product benchmark. Keep the BayesFlow default sampler.
4. Only investigate a new architecture or further data expansion if the resulting gap still warrants it. A same-bank legacy refit and multiple seeds remain possible attribution controls; none was submitted during this audit.

No new production training data, original-pricer calls, MCMC chains or notebook executions were required. The three read-only GPU jobs consumed **4,540 allocated CPU-seconds (1.2611 hours) / 1,135 GPU-seconds (.3153 hours)**: saved-checkpoint controls 672/168, final panel 3,684/921 and repeated loss 184/46. Their allocation ledgers retain separate ceilings; the repeated-loss job shares the original controls pool, whose total actual use is 856 CPU-seconds / 214 GPU-seconds. These are additional audit costs, not another charge for earlier fits or bank generation.

The original raw bank was not recovered; its prior-sample statistics in the audit are explicitly reconstructed from recorded source/seeds, while our bank's sampled rows were checked against actual stored arrays. Only compact evidence is committed; model, bank and draw payloads remain on Oscar.

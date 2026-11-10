# Heston ZCA checkpoint replay

**Completed development diagnosis, 10 November 2026: earlier stopping improves population coverage, but none of the five examined checkpoints is calibrated.** The [offline report](report.html) compares epochs 50, 75, 83, 100 and the preserved final epoch 200. It includes learning curves, parameterwise recovery and selected 90% intervals, all six SBC ECDFs, coverage, matched MCMC recovery and fixed-observation comparisons.

| Epoch | c median RMSE | c truth–median correlation | Median c 90% width | c 90% coverage | Physical 90% coverage range | Marginal SBC within band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | .16817 | .92415 | .47771 | 80.86% | 79.69–84.57% | 2/6 |
| 75 | .16324 | .92803 | .44420 | 79.69% | 76.76–82.23% | 0/6 |
| 83 | .16775 | .92385 | .42807 | 78.91% | 74.22–78.71% | 0/6 |
| 100 | .16988 | .92235 | .41813 | 78.71% | 66.80–73.83% | 0/6 |
| 200 | .16970 | .92328 | .36535 | 74.02% | 54.88–60.94% | 0/6 |

The nominal coverage is 90%. The two within-band marginals at epoch 50 are volatility of volatility and c. c point recovery is already strong at epoch 50 and changes little thereafter; later training narrows intervals while worsening aggregate coverage. This supports the concern about the final checkpoint, but selecting the validation minimum is insufficient: epoch 83 still has substantial undercoverage and all six marginal SBC screens outside their bands.

Within-band SBC means this particular CDF screen does not reject, rather than certifying calibration. Its finite simultaneous band can allow errors at both endpoints of a central interval. Consequently, epoch-50 c can remain inside that band while its nominal-90% coverage is only 80.86%. Coverage and the rank screen should be read together.

## What was held fixed

The [protocol](protocol.json) was committed at `66eba70` before production execution. One fresh replay uses the original 10,000 fixed noisy training rows, 100 fixed validation probe rows, ZCA state, initialization seed 9011, target constraints, six spline blocks with [128,128] conditioners, zero dropout, batch 256 and unpadded 16-row tail. Adam retains the original cosine schedule, 5e-4 to 5e-5 over **8000** updates, although the replay stops at 4000 updates / 100 passes / one million presentations. The original epoch-200 model and evaluation are reused. No observation law, pricing labels, prior, half-spread profile or quote geometry changes.

The [independent result audit](results-verification.json) verifies all four initial state groups, identical preprocessing bytes and initial probes, all 100 original learning-curve prefix records (every field except elapsed time), and native reload/state/seeded-draw checks for all four new checkpoints. Historical early weights were never saved, so equality to those unavailable weight files cannot be checked directly. The same deterministic training algorithm, initialization and observed trajectory are reproduced. All [294 preserved artifacts](preserved-artifacts-before.json) remain unchanged.

The [campaign](campaign.json) completed one replay in 67.78 worker seconds and four evaluations in approximately 9–11 seconds each, within its 900-second training and 300-second evaluation caps. There were no retries or automatic extensions. Full models, state, exposures and conditional draws remain outside Git under the recorded hashes.

[Software and report verification](verification.json) records 748 passing lab tests plus nine additional consumer tests, current manifest/claims/runtime-boundary checks, verified figure/source hashes, and 13 self-contained inline figures. The [review inventory](review-summary.json) binds the final report and dependence summaries. A frozen report consumer initially treated original JSON metadata as executable source files; the separate read-only review consumer corrects that binding distinction and the market wording, leaving the frozen source and all numerical results unchanged. Its narrowly scoped metadata adapter retains hash checks and restores the original hook. The first consumer attempt is preserved outside Git under its recorded hash.

## Population checks and conditional checks answer different questions

Every checkpoint uses the same 512 prior-generated development quote vectors and generating parameters, 512 posterior draws per observation and sampling seeds. The three fixed-observation checks reuse the same existing joint MCMC references and 4096 neural draws per case. The eight selected recovery observations retain seven ready MCMC references; the failed reference at population index 487 is neither replaced nor counted as trustworthy.

Earlier stopping does not uniformly improve the conditional approximation. At epoch 50, the broad synthetic case's physical intervals are 3.73–9.56 times the reference widths and its largest physical mean error is 16.71 reference SDs. At epoch 83, those widths remain 2.92–6.85 times the reference widths, with maximum mean error 6.12 SDs. Several conditional widths move toward MCMC with further training even while population coverage deteriorates. Thus it is inaccurate to describe every posterior as too narrow or to repair this with a single interval multiplier.

**Market interpretation:** network–MCMC agreement for the market observation tests the inference approximation under the specified likelihood and prior. It does not establish market-model adequacy. That requires separate residual/predictive and assumption checks. Large errors divided by tiny reference SDs near a prior boundary also need to be read alongside absolute intervals. The review report makes this distinction explicit and adds descriptive conditional dependence checks from already saved draws; correlation comparisons do not establish full joint correctness.

## Decision and next experiment

No checkpoint is promoted. The minimum at epoch 83 was already visible before this replay, so the checkpoint comparison is post hoc development evidence, not fresh acceptance. The financial one-scale observation model is unchanged; this result does not justify adding tau, changing the prior, widening spreads or changing the architecture to obtain broader intervals.

The next isolated generalization experiment should refresh **c and Gaussian quote errors** across passes while keeping the same 10,000 clean surfaces, frozen ZCA transform, initialization, architecture, batch size, 8000-update learning-rate schedule and total presentation budget. Compare matched epochs, including the previously saved early points and final 200. This tests fixed-noise repetition as a contributor; it is not a guaranteed fix, and the physical training bank remains finite. A larger distinct bank is a separate later comparison if needed.

After settling preprocessing, augmentation and checkpoint selection, the primary five-target c=1 network needs its own fit and validation. Its earlier development model remains unaccepted; the joint-c study cannot grant fixed-c acceptance. Fresh confirmation must follow the frozen selection rule. The asymmetric physical grid and all Bergomi production remain gated. No augmentation fit, new-grid pricing or Bergomi run is launched by this checkpoint diagnosis.

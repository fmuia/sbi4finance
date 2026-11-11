# PR87 FlowMatching: 300-epoch followup

**Completed: both authorized fits improve recovery, but physical posterior calibration remains inadequate for an official accuracy claim.** Final epochs and predeclared validation minima are retained; no checkpoint or asset is promoted.

## Completed results

The [combined offline report](report.html) includes both complete 300-epoch runs and their validation minima. Individual reports are available for [100k](100k-300/report.html) and [200k](200k-300/report.html), with standalone PNG/SVG figures under each run's scientific-figures directory.

All rows below use the same 512 synthetic quote/truth sets and the same posterior sampling and SBC seeds. Physical coverage summarizes five targets; all six are shown separately in the report.

| Endpoint | c median RMSE | c 90% coverage | Physical 90% coverage | SBC inside band | Raw outside prior | Raw nonphysical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k / final200 | 0.17753 | 83.40% | 94.53–99.02% | 3/6 | 15.29% | 1.4786% |
| 100k / final200 | 0.14653 | 89.06% | 94.34–97.85% | 1/6 | 6.21% | 0.1945% |
| 100k / final300 | 0.14570 | 89.06% | 92.97–96.48% | 2/6 | 5.69% | 0.0954% |
| 100k / minimum281 | 0.14576 | 89.06% | 93.16–96.48% | 2/6 | 5.55% | 0.0988% |
| 200k / final300 | 0.14421 | 89.06% | 90.62–95.51% | 4/6 | 4.56% | 0.0431% |
| 200k / minimum272 | 0.14421 | 88.67% | 90.04–95.90% | 3/6 | 4.78% | 0.0530% |

**More data and training improve the approximation, but do not establish official posterior quality.** Compared with 100k/200, the 200k/300 primary narrows physical interval errors and improves physical recovery. c RMSE changes only modestly, from .14653 to .14421, while its 90% coverage stays 89.06%. Four of six population SBC screens pass; initial variance and long-run variance still fail. Their 90% coverages are 95.12% and 95.51%. The other physical coverages are 91.60% for kappa, 90.63% for vol-of-vol and 93.75% for rho.

Conditional comparisons still show approximation errors that pooled recovery cannot expose. On the two ready synthetic cases, physical 90% interval widths range from .91 to 3.00 times their joint-MCMC widths, and several conditional means differ by more than the retained .3-reference-SD convention. Some marginals are close; others remain materially wrong. A uniform interval multiplier would not resolve this pattern. Market comparisons are retained separately from market-model adequacy.

At 200k/300, 11,961 of 262,144 raw sample vectors leave the declared prior; 113 are physically invalid. No raw sample is nonfinite. Native 100-case validation results and selected MCMC comparisons are retained in full, including the unresolved reference at row 487. Nothing is clipped, rejected or rescaled. The minimum-validation epoch 272 has similar substantive errors and does not rescue calibration: c coverage is 88.67%, physical coverage 90.04–95.90%, SBC 3/6, and raw prior violations 4.78%.

The first 100k/300 primary correctly triggered the authorized second fit through the saved [branch proof](branch-proof.json). Its secondary did not influence the decision. The [campaign](campaign.json) completed both fits and all four endpoint evaluations once in 2631.88 seconds, with no retries or budget extensions. Generation/fit took 626.28 seconds for 100k and 1436.85 seconds for 200k; all four posterior evaluations took 133–152 seconds each. These measurements include supervision overhead and concurrent software checks; they are not isolated compute benchmarks.

The [independent audit](independent-audit.json) and its [archived consumer source](independent-audit-consumer.py) verify data/RNG continuation, 200k distinct physical vectors, exact prefixes/validation/whitening, native states, all four endpoints' saved statistics and all 565 protected historical files. [Full lab verification](full-lab-verification.json) records 811 passing tests. The initial discovery errors were confined to three import-order-dependent mocks, repaired only in tests; both logs and the [explanation](test-import-order-fix.json) are retained. Manifest, claims and runtime-boundary release checks pass. [Campaign verification](verification.json) binds the final report, figures, audit and compact training copies.

Stop for reflection after the two authorized fits. Additional epochs/data have helped; they have not alone resolved physical uncertainty, conditional bias or support. A further run should isolate a specific remaining hypothesis, such as training regularization or target coordinates, with sampling-numerics sensitivity assessed separately. No such experiment is launched here. The separate fixed-c teaching baseline still requires its own validation; reserved confirmation, actual-grid changes and Bergomi production remain gated.

## Frozen design and training recipe

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

The network construction remains:

```python
adapter = (
    bf.Adapter()
    .concatenate(parameter_names, into="inference_variables")
    .rename("observations", "inference_conditions")
)
workflow = bf.BasicWorkflow(
    adapter=adapter,
    inference_network=bf.networks.FlowMatching(
        subnet_kwargs={"widths": (128, 128, 128), "dropout": 0.1}
    ),
    standardize="inference_variables",
)
workflow.fit_offline(
    data=train, validation_data=validation, epochs=300, batch_size=256,
    callbacks=[archive_callback],
)
```

Here observations already contain the saved full-rank whitening of price/half-spread inputs. Targets remain the six raw physical coordinates; no bounded transform is added. The archive callback only saves states and logs. Native AdamW keeps weight decay 0.005 and clipnorm 1.5. Its learning rate starts at 0.00005, warms to 0.0005 over the updates listed above, then follows native cosine decay to zero. Dropout 0.1 is training regularization; it is disabled at sampling and does not represent financial quote noise.

The financial model remains independent Gaussian price errors with standard deviation `c * h_i`, where each `h_i` is the selected contract's observed half-spread and `c ~ Uniform(0.5, 2)`. No additional tau/vega error term or spread-profile variation is introduced.

The 100k input data are copied byte for byte. For 200k, the fresh rows continue the completed 100k parent's retained post-append NumPy state, with native sequential ordering of all priors followed by all observation noises. The same 100 native validation rows and frozen whitening bytes remain; new data never enter whitening fitting. Models never inherit parent weights. These are practical data/optimization comparisons with distinct learning-rate budgets, not equal-compute attribution tests. The same AdamW decay coefficient also acts over more optimizer updates.

Final epoch 300 is primary; the first strictly minimum native validation objective is a secondary endpoint only when its epoch differs. Archived 100/200/250 checkpoints do not receive posterior queries. Each declared endpoint uses the same native100 x300, historical512 x512 and three existing joint-MCMC cases x4096, with unchanged observations, truths and seeds. Every raw draw is retained, including out-of-prior and nonphysical values; no clipping or interval rescaling is applied.

The primary 100k result alone determines whether 200k runs. The frozen necessary screens check all six marginal coverage/SBC results, finite/physical/prior support and ready conditional-reference agreement; they grant no official acceptance. Secondary results cannot alter this branch. Both generation/training workers are capped at 3600 seconds, each posterior evaluation at 900 seconds, with no retries or further fits. A failed worker stops the campaign.

The [campaign protocol](campaign-protocol.json), [100k protocol](100k-300/protocol.json), [200k protocol](200k-300/protocol.json) and [preservation inventory](preserved-artifacts-before.json) bind the recipe and earlier artifacts. No reserved test pool, new MCMC, grid changes or Bergomi production is opened. Official confirmation and the separate fixed-c baseline remain later gates.

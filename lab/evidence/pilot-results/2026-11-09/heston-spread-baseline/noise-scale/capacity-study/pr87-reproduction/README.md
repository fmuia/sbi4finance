# One reproduction of the PR87 notebook recipe

**Completed, 10 November 2026. The native FlowMatching recipe improves c coverage relative to the spline fits, but does not give an accurate six-target posterior.** [Open the compact offline report](report.html) (197 KB), or inspect the [standalone scientific figures](scientific-figures/manifest.json).

## How many data points?

The preceding controlled whitening pilot used **10,000 distinct clean surfaces, each with one fixed c and Gaussian quote-error vector**. It repeated those same 10,000 noisy datasets for 200 epochs: two million presentations, not two million independent simulations. Each dataset contains 54 prices and six targets; 540,000 scalar quote entries are not 540,000 independent surfaces. The saved clean bank has 100,000 surfaces, but that whitening pilot deliberately selected a 10,000-row prefix to match the collaborator's small offline example. Its validation loss used 100 fixed datasets. The earlier large-bank fits used unwhitened inputs and do not answer whether more data would help the whitened estimator.

This reproduction follows [PR87](https://github.com/fmuia/sbi4finance/pull/87) at `ddf3619faf48b8f4b0ee56ee841ef24124a508d1`: **10,000 fresh training simulations and 100 fresh validation simulations**, then 200 offline epochs, batch 256, 40 updates per epoch including a 16-row tail. The native simulator draws all priors before observation errors within each split. It generates c and noise once per dataset; shuffling and FlowMatching's latent/time/dropout randomness do not refresh financial observations. The primary endpoint is the final epoch 200 / update 8000, as in the notebook. Earlier checkpoints and the first minimum-validation checkpoint are archival only; none received extra posterior queries.

## Literal model recipe and disclosed reproduction limits

The notebook fits the full covariance to training Y/h only, retaining all 54 dimensions and using an eigenvalue floor of maximum eigenvalue times 1e-8. Here no eigenvalue needs flooring; the covariance condition reduces from approximately 1.118 million to 1. The observations and targets are native float64 arrays, with the native Adapter's default conversion to float32. No extra bounded-target transform is introduced.

```python
adapter = (
    bf.Adapter()
    .concatenate(parameter_names, into="inference_variables")
    .rename("observations", "inference_conditions")
)
workflow = bf.BasicWorkflow(
    adapter=adapter,
    inference_network=bf.networks.FlowMatching(
        subnet_kwargs={"widths": (128,) * 3, "dropout": 0.1}
    ),
    standardize="inference_variables",
)
history = workflow.fit_offline(
    data=training_data, validation_data=validation_data,
    epochs=200, batch_size=256,
)
```

The omitted optimizer is intentional. Locked BayesFlow 2.1.0 resolves its native offline default to AdamW with weight decay .005 and clipnorm 1.5. Its cosine schedule warms from 5e-5 to 5e-4 over 400 updates, then decays to zero over 7600 updates. Raw physical targets are standardized without support constraints. The financial likelihood, six-dimensional Uniform prior, 54 contracts and half-spreads remain unchanged.

The original notebook sets no seed and saves neither its training data nor weights; it does not record complete package versions. Its Python metadata is 3.12.11. This study adds NumPy/Keras seed 9011 and records the current Python 3.12.16, BayesFlow 2.1.0, Keras 3.15.1 and JAX .11.1 CPU environment. It is a faithful statistical-recipe reproduction, not a bitwise recreation of the collaborator's unknown stochastic run. Archival callbacks perform no inference or extra probes during training and verify that serialization leaves model state unchanged.

The source-bound [protocol](protocol.json) was committed at `8bc3721` before execution. One production fit took 57.70 worker seconds including 10,100 checked pricing calls; native training took 39.32 seconds. The [campaign](campaign.json) ran one final-checkpoint evaluation in 129.83 seconds, with no retries. The minimum native validation objective happens to be the final epoch: .263623, with training objective .331326. This is a FlowMatching objective, not density NLL, and cannot be compared numerically with the earlier spline loss. The curve does not show the old spline's large late validation-gap pattern; data count alone does not establish overfitting.

## Calibration and support

| Same historical 512 observations | Spline/ZCA epoch 83 | Spline/ZCA epoch 200 | Native PR87 recipe epoch 200 |
| --- | ---: | ---: | ---: |
| c median RMSE | .16775 | .16970 | .17753 |
| c truth–median correlation | .92385 | .92328 | .91519 |
| c nominal-90% coverage | 78.91% | 74.02% | 83.40% |
| Physical nominal-90% coverage range | 74.22–78.71% | 54.88–60.94% | 94.53–99.02% |
| Marginal SBC within band | 0/6 | 0/6 | 3/6 |

The exact notebook-style first evaluation uses all 100 newly generated validation cases and 300 draws per case. Its physical 90% coverage is 92–98%, c coverage is 89%, and 5/6 SBC screens remain inside their band. That small panel was also monitored during training. The historical panel uses 512 identical saved observations/truths and 512 draws each. Its physical coverage is [99.02,94.53,96.88,96.29,94.53]% and c coverage is 83.40%. The latter's Wilson 95% interval is 79.93–86.37%; the native 100-case c interval is 81.37–93.75%. The different panel results illustrate finite population variation. All these are development diagnostics, not fresh release acceptance.

Raw draws are deliberately neither clipped nor rejected. On the historical panel, **15.29% of six-dimensional sample vectors have at least one coordinate outside the prior**, including 6.32% of c values outside [.5,2]. **1.48% of sample vectors are physically invalid** under the stated positivity/correlation conditions. There are no nonfinite draws. All raw draws contribute to calibration and reference comparisons. Good recovery along the unit line does not remove these support or uncertainty errors.

Three existing fixed-observation references are reused. Broad and market-like c means are close to MCMC (.127 and .153 reference SD error), but their physical 90% intervals are 7.61–18.49 and 3.54–27.49 times reference widths. On the market observation, c's neural 90% interval is [1.2318,1.8208] versus [1.9971,1.99995] for the bounded joint reference. This is conditional inference disagreement under the assumed target; market-model adequacy remains a separate residual/predictive question. The existing eight selected recovery cases retain the failed reference at index 487 without replacement or a confident reference interval.

## What this does and does not establish

This repeats a bundle: new training draws/whitener, FlowMatching, dropout, raw target coordinates and a different optimizer. It cannot isolate a causal FlowMatching effect. c coverage improves relative to the examined spline checkpoints, but point RMSE is slightly worse; physical intervals over-cover in the population and are much wider than MCMC on selected observations. No checkpoint is promoted, including the separate fixed-c=1 teaching target.

Limited diversity of 10,000 fixed noisy simulations is a reasonable hypothesis, not an established explanation. A controlled larger-data experiment on whitened inputs is still needed to answer it. Keep bank diversity, observation refresh, training budget and target constraints as explicit factors; do not silently combine them or treat post hoc clipping as the notebook posterior. One reproduction does not show that FlowMatching generally fails or succeeds on this task.

[Independent audit](independent-audit.json) reconstructs the full prior/noise stream, training-only whitening and all 112 saved state arrays, and verifies that all 406 preserved artifacts remain unchanged. No existing bank or reserved test arrays were used for training; historical development inputs were queried only in evaluation. [Software/report verification](verification.json) records the full lab suite, final focused tests and offline figure checks. All full models, generated datasets and raw posterior draws remain outside Git under their hashes. Grid and Bergomi production remain gated.

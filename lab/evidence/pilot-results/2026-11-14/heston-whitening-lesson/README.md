# Historical whitening comparison for the Heston lesson

`results.json` contains the compact plotting input for the preprocessing lesson:
512 shared generating values of `c`, each arm's 5/50/95% marginal `c` quantiles,
descriptive metrics, experiment metadata and source hashes. It contains no
dataset identifiers, physical-parameter rows, noisy quotes, joint posterior
draws, model payloads or private paths. Viewing it does not require a download.

## What is controlled

The preserved 10 November 2026 `whitening-pilot` is a matched pair on the
**older 54-contract Heston task**. It is separate from the current 63-contract
FlowMatching estimator. Both historical arms use the same six independent
Uniform priors, including `c ~ Uniform(0.5,2)`, and independent Gaussian quote
errors with price SD `c*h_i`. Here `h_i` is the fixed observed price half-spread.
The learned target is `c`, not the Heston volatility-of-variance parameter or a
common raw-price `sigma`.

Both arms receive `X_i=Y_i/h_i`. The control centers and scales each coordinate
by its training mean and SD; it is **not entirely unscaled input**. The ZCA arm
uses a full-covariance, invertible transform. Both transforms are fitted on the
same 10,000 noisy training rows, frozen, and applied exactly once to later data.
No coordinates are dropped and the eigenvalue floor is inactive in this pilot.
The condition number is about 2.20 million after diagonal scaling and about one
after ZCA. Whitening is a coordinate change for the network, not a change to the
financial likelihood, prior or simulated noise.

The two fresh spline CouplingFlows have six blocks, 128-by-128 conditioner
widths and zero effective dropout. All 274 initial state-array summaries match
(84 trainable, 19 non-trainable, 169 optimizer and two metric arrays). Their
contracts differ only in preprocessing mode and fitted-transform fingerprint.
The raw training observations, targets, row ordering, architecture, optimizer,
schedule and seeds are otherwise identical.

Each fit repeats the same 10,000 fixed noisy rows for 200 complete passes:
8,000 optimizer updates and 2,000,000 presentations, batch size 256 with an
unpadded tail of 16. Adam uses cosine decay from `5e-4` to `5e-5` over that
budget. The two final evaluations use the same 512 validation datasets and
512 posterior draws per case. Generating values, raw quote fingerprints and
dataset ordering are matched before selecting only the `c` plotting columns.

## What it shows and does not show

| Same 512 datasets | Diagonal scaling only | Full ZCA whitening |
| --- | ---: | ---: |
| Correlation of `c` truth and posterior median | -0.00116 | 0.92328 |
| Median-estimation RMSE | 0.44203 | 0.16970 |
| Median central-90% interval width | 1.33845 | 0.36535 |
| Fraction of central-90% intervals containing truth | 87.70% | 74.02% |
| Original marginal SBC screens passed | 3/6 | 0/6 |

The diagonal arm retains almost the prior's 90% width of 1.35 for `c`, with
negligible median recovery. Its reasonable-looking marginal coverage does not
demonstrate that it learned `c`. Whitening exposes information about `c`, but
this final whitened checkpoint is **not calibrated**: `c` intervals cover only
74% of truths and physical-parameter interval frequencies are 54.9–60.9%.
All six original marginal SBC screens fail for the whitened arm.

These plots use the predeclared **final epoch-200 / step-8000** checkpoints.
The ZCA validation loss was lowest at epoch 83 (about -6.38217), rising to
about -1.41399 at epoch 200. Earlier weights were not saved. This is consistent
with overfitting, but earlier-checkpoint calibration was not evaluated. The
lesson therefore illustrates improved learnability rather than posterior
acceptance or guaranteed benefits across seeds, architectures and markets.

Metrics are recomputed from the exact float32 marginal-quantile cache used by
the figure; they agree with the original full-precision metrics within rounding.
No point, interval or prior-support violation is clipped or repaired.

## Reproduction and source binding

From `lab/`, point the preparer at the preserved historical study:

```sh
python scripts/prepare_heston_whitening_lesson.py --study "$PRESERVED_WHITENING_PILOT"
```

The preparer verifies the common source protocol, native checkpoint hashes,
completed exposure ledgers, actual identical initial-state summaries, matched
raw training and evaluation bindings, and shared generating values/order. It
copies no old artifact and implements no financial kernel. It uses existing
quantiles and loss curves; no training, repricing or MCMC is launched.

The public source bindings include the protocol, independent pairing audit,
training/evaluation records, checkpoint records, recovery caches, loss curves,
bank identity and current preparation script. Raw inputs stay in their
preserved original locations; the new read-only loader requires only this
public plotting record.

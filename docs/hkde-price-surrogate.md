# HKDE: a learned price map with separate inference and pricing errors

The bounded M5.2 surrogate **passes its validation and new regression test**.
It reproduces the two declared Fourier put-price maps within 3.84e-7 currency
on the tested parameter sets. The previous neural posterior remains provisional:
its worst price-quantile discrepancy is still 0.04387, above 0.03. A faster
forward map preserves that result; it does not repair inference.

Review [notebook07](../notebooks/07_hkde_price_surrogate.py) with
`PYTHONPATH=src pixi run review-hkde-surrogate`. It explains the regression,
runs a small analytic example and lets a reviewer inspect the saved model.
No default cell regenerates a bank, trains a neural network or contacts a provider.

Graphs: [parameter-grid errors](results/hkde-surrogate/plots/parameter-errors.png),
[pricing versus inference errors](results/hkde-surrogate/plots/paired-errors.png),
and [measured pricing costs](results/hkde-surrogate/plots/timing.png).
The [frozen protocol](hkde-surrogate-protocol.md),
[full report](results/hkde-surrogate/report.json), compact banks, coefficients
and check arrays are committed together.

## What was learned

For each fixed contract, the regression learns

\[
 (\theta,\lambda)\longmapsto P^{\mathrm{quad}}(\theta,\lambda).
\]

Here theta is annualized long-run variance and lambda is jump intensity per
year. The domain is `[0.025,0.065] × [0.25,1.0]`; every other model parameter,
initial variance and market context remains fixed. The outputs are European
puts at `(K=95,T=0.75)` and `(K=80,T=1)`, in the currency units of spot 100.

This is supervised approximation of expected discounted payoffs. Parameters
are inputs, unlike the calibration network, where noisy option quotes are
inputs and joint parameter draws are outputs. The regression labels are
deterministic Fourier expectation estimates, not single realized payoffs or
noisy market quotes. The calibration observation generator remains unchanged.

A degree-six tensor Chebyshev polynomial has 49 coefficients per contract.
One float64 least-squares fit used 512 independently drawn parameter pairs,
with both contract labels grouped per pair. Known bounds map the inputs to
`[-1,1]²`; no evaluation data determine the scaling. The fitted design has full
rank 49 and condition number 32.45. The entire model is an 8,114-byte NPZ file;
its metadata binds parameter order, bounds, context and contract order. Reload
predictions match exactly. Unsupported contracts, altered context and
extrapolation are rejected rather than silently approximated.

The polynomial was chosen as a lean baseline for this smooth two-dimensional
task. It does not imply that the same degree, label count or speed applies to
rough models, larger parameter spaces, changing contracts, Greeks or exotics.

## Measured numerical and regression checks

| Check | Largest observed error | Frozen requirement |
|---|---:|---:|
| 256 independent validation pairs | 9.65e-8 currency | ≤0.003 |
| Existing 129×129 endpoint-inclusive stress grid | 3.83e-7 | ≤0.003 |
| 81,920 cached joint neural draws | 3.50e-7 | ≤0.003 |
| 256 new regression-test pairs | 3.76e-7 | ≤0.003 |
| Worst per-contract RMSE, including new test | 3.05e-8 | ≤0.001 |
| Fourier refinement on a fixed 5×5 parameter grid | 7.64e-10 | ≤1e-5 |
| Paired posterior mean shift, either route | 6.43e-9 | ≤0.003 |
| Paired posterior 5/50/95% quantile shift, either route | 2.06e-7 | ≤0.003 |

Put-price bounds pass without clipping. The new regression test was generated
and evaluated only after all validation, refinement and propagation gates passed.
It is separate from the old posterior's final test, which remains unevaluated.
No degree, seed or architecture sweep was used. These checks support this
restricted numerical price map on the tested sets; they are not a proof of
uniform error throughout the rectangle or of continuum accuracy.

## Why the two error sources stay separate

For one observed panel, reuse exactly the same joint draws
`(theta_s,lambda_s)` and compare direct prices `P_s` with surrogate prices
`P_hat_s`. The differences isolate price-map substitution; no extra posterior
sampling noise is introduced by drawing a second parameter sample. We also
compare both price maps on the identical weighted reference grid.

For the matched finite sample, if every `|P_hat_s-P_s| ≤ epsilon`, corresponding
ordered values differ by at most epsilon, and so do the empirical quantiles
used here. This bound concerns the matched sample; it does not certify unseen
parameter points. The paired checks preserve parameter dependence and do not
replace joint draws with independent marginal draws.

| Largest deviation from the reference-grid price quantiles | Currency |
|---|---:|
| Original neural draws + Fourier pricer | 0.0438697547 |
| Same neural draws + surrogate | 0.0438697619 |
| Original inference acceptance limit | 0.03 |

Both inference routes still fail the original limit. The surrogate's independent
accuracy is accepted for its bounded task; the combined neural posterior-price
workflow is not. No new SBC was run: SBC assesses a posterior algorithm,
whereas this increment fits a deterministic regression map.

## Measured cost

| Calculation | Local elapsed time (CPU execution) |
|---|---:|
| Generate/write 512 training pairs, both contracts | 0.162 s |
| Fit 98 coefficients in one solve | 0.00329 s |
| Fresh model-file load + 256 predictions, within the process | 0.000495 s |
| One parameter pair: direct / surrogate, warm median | 0.423 / 0.155 ms |
| 4,096 parameter pairs: direct / surrogate, warm median | 1.177 s / 0.678 ms |
| Complete experiment, including checks and timing repetitions | 5.154 s |

Each warm median uses three calls after one warmup. The reference route actually
recomputes Fourier prices; it does not look up cached labels. The comparison is
against the current per-parameter reference implementation, including each
route's validation. It is not a claim about an optimally vectorized pricer,
end-to-end neural inference, or cold process startup. The observed ratios are
2.72 for one pair and about 1,736 for the 4,096-row batch.

Training-label generation, fitting and saving cost 0.166 s. At the observed
batch saving, that cost is recovered within the first 4,096-row pricing query.
Charging the entire measured run instead gives about five such queries.
These arithmetic comparisons exclude the earlier cached reference/inference
studies and apply only to compatible pricing calls. Neural sampling cost is
unchanged. The reference grid also remains a reusable calibration tool.

## Reproduce or use the saved map

From the repository root in the pinned environment:

```bash
PYTHONPATH=src pixi run hkde-surrogate --out /tmp/new-hkde-surrogate
PYTHONPATH=src pixi run review-hkde-surrogate
```

The output directory must be new. The first command fits once, preserves failed
checks and evaluates the new regression test only after validation passes.
It uses committed original validation artifacts, without loading the neural
checkpoint, generating a market panel or accessing the posterior final test.
The recorded execution uses commit `10f6713`; report hashes pin its numerical
sources and exact inputs.

```python
import numpy as np
from sbi4finance.hkde_surrogate import load_model, predict

model = load_model("docs/results/hkde-surrogate/model.npz")
prices = predict(model, np.array([[0.04, 0.60]]))  # columns follow saved contracts
```

M5.2 now has a tested fixed-contract price component and didactic review.
The [verification record](results/hkde-surrogate/notebook-verification.json)
records a fresh isolated HTML export: 14 cells, zero errors, 9.17 seconds,
with external connections blocked, no market files, raw banks or neural
checkpoint, and the small surrogate artifact available. Strict marimo checks
and 117 software tests pass; the three scientific plots were visually inspected.
M5.1's neural inference discrepancy, the rough reference follow-up and M8's
selected, paced participant workflow remain separate work. Required S evidence
is fully synthetic; it neither requires market observations nor establishes
real-market model adequacy.

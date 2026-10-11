# Bounded time context for the Heston recurrent summary

## Diagnosis and single-change experiment

The original spline candidate passed neither the conditional-posterior nor the
predictive validation gates. Its training loss continued improving while the
best validation loss occurred at epoch 45 of 80. More training alone is not an
established remedy. A code/activation inspection identified a concrete scaling
problem to test first.

BayesFlow's recurrent summary appends a Time2Vec embedding. With no explicit
time channel, it uses integer positions 0 through 251. The saved spline model's
linear time component grows to 235.87 at the final day, whereas returns have
approximately unit global standard deviation after training normalization.
Reconstructing its GRU computation agrees with the saved Keras layer to within
7.3e-7. At the final step, about 81% of update-gate and 88% of reset-gate
preactivations have absolute value greater than five, placing those sigmoids
near saturation. This demonstrates a scale problem, not proof that it caused
every posterior error. The summary still responds to ordering changes.

The [diagnostic script](../scripts/diagnose_heston_summary.py) generates only
validation examples 0–15, reconstructs the saved GRU and compares it to Keras,
and records checkpoint, configuration and source hashes. The
[legacy diagnostic report](results/heston-time-context/summary-legacy.json)
preserves this finding. Reproduce it with:

```bash
pixi run python scripts/diagnose_heston_summary.py \
  --config configs/tasks/heston_inference_followup.json \
  --checkpoint checkpoints/heston-spline-provisional \
  --out /tmp/heston-summary-legacy.json
```

The focused change supplies known observation end times

\[
t_j=\sum_{i=1}^{j}\Delta_i
\]

as a second neural input channel. Training-only global standardization maps
these fixed times to order-one values before `RecurrentNetwork(time_axis=1)`
uses them. For the 252-day design, they range approximately from -1.73 to 1.73
after scaling. Time2Vec now receives these values instead of integer positions.
This is known context, not a new measurement or latent truth. Raw banks remain
decimal log returns `[batch,time,1]`; only the checkpoint's conditioning adapter
constructs `[return,end_time]`. Return amplitudes and the eight trailing moments
remain unchanged. Signed-state support and joint continuation remain unchanged.

The new adapter ID is `heston_returns_time_and_window_moments_v2`. Old saved
checkpoints retain their v1 adapter; frozen evaluation reads the saved ID and
rejects unknown IDs. No simulator change, new bank format or model framework is
needed.

## Protocol fixed before training

- Use the same root-43022 bank, all 32,768 training and 2,048 validation histories,
  spline density estimator, GRU32/summary16, eight window features, seed 871,
  80 epochs and batch size 256 as the preceding spline experiment. Select
  minimum validation loss. Add no squared-return channels or other features.
- Reuse the same first 16 checked validation references and unchanged accuracy
  gates; use 2,048 draws per history and the first 256 validation histories for
  aggregate diagnostics. This is a development comparison of the time encoding.
- If validation passes, freeze the saved candidate, independently check the
  first 16 root-43022 test references and evaluate the reserved 256-history
  final test once. Reference instability blocks neural acceptance. Do not
  modify the candidate or thresholds after viewing final-test results.
- If validation fails, record that result and leave the reserved test unevaluated.
  Do not start an automatic architecture sweep or larger cluster run.

```bash
pixi run heston-train --bank /tmp/heston-followup-bank-v1 \
  --references /tmp/heston-followup-ref-validation-v1 \
  --out /tmp/heston-explicit-time-v1 --density-estimator spline \
  --explicit-time --epochs 80 --batch-size 256
```

The preceding [follow-up](heston-inference-followup.md) documents regeneration of
the shared bank and validation references. Absolute temporary directories above
are examples; use fresh output directories. The [marimo review](../notebooks/05_heston_joint_inference.py)
will display this diagnosis and its measured outcome separately from the old
regression-case illustrations. Notebook execution, synthetic posterior accuracy
and empirical model adequacy remain separate checks.

## Recorded outcome

Training finished in 347.5 seconds including compilation; the saved candidate
had minimum validation loss at epoch 63. This is elapsed local execution time,
not an isolated performance benchmark: small tests/diagnostics ran concurrently
for part of the run. Same-seed checkpoint reload reproduced draws exactly.

The saved model's final linear time component is now **3.09**, versus 235.87.
None of its update/reset preactivations exceeded magnitude five at the four
recorded positions across the same 16 histories. The reconstructed GRU matches
Keras within 1.6e-7. This supports the intended scaling correction; both models
were trained separately, and the result does not identify a universal cause of
inference error.

| Validation check | Raw-index spline | Explicit-time spline | Maximum |
|---|---:|---:|---:|
| Mean RMSE / reference SD | 0.246 | **0.173** | 0.20 |
| Largest absolute mean error / reference SD | 0.792 | **0.455 — fails** | 0.40 |
| Largest relative posterior SD error | 0.461 | 0.289 | 0.30 |
| Largest correlation error | 0.077 | 0.071 | 0.20 |
| Largest relative predictive SD error | 0.147 | 0.088 | 0.15 |
| Largest predictive interval endpoint error / predictive SD | 0.300 | **0.210 — fails** | 0.15 |

Both marginal rank diagnostics now pass (0.0473 for theta, 0.0760 for state,
against 0.0925). Nominal 90% coverage is 91.8% for theta and 86.7% for state;
future-loss interval coverage is 90.2%. These aggregate results do not override
the two failed conditional gates. This candidate **still fails validation**.

Both remaining gate failures occur on `validation:0`, whose reference mean
state is about 0.00644. Its neural state mean remains too high, and its posterior
state SD is about 26% wider than the reference. No other reference case exceeds an individual-error limit in this run. That observation identifies a useful
diagnostic case; it is not a reason to drop or downweight it. The next focused
investigation is conditional approximation in low-variance histories, using
validation and old regression cases before choosing another training change.

**The reserved root-43022 final test remains unevaluated.** No test reference or
neural test report was generated, and neither weights nor thresholds changed
after these results. M3.3 scientific acceptance remains pending. The diagnostic
script uses only validation data; the training bank loader verifies all split
files, but test observations/targets do not enter training, metrics or selection.

The [marimo tutorial](../notebooks/05_heston_joint_inference.py) shows the gate
traces and comparison table, while keeping its original regression-case plots
explicitly historical. Forty software tests pass, including context/label
separation, training-only normalization and both old and new checkpoint paths.

Evidence: [explicit-time diagnostic](results/heston-time-context/summary-explicit-time.json),
[validation report](results/heston-time-context/neural-validation.json),
[per-case checks](results/heston-time-context/validation-checks.npz), and
[saved candidate](../checkpoints/heston-time-context-provisional/README.md).
The [bank manifest](results/heston-inference-followup/bank-manifest.json) and
[reference report](results/heston-inference-followup/reference-validation.json)
are shared with the previous spline experiment; their hashes match exactly.

To reproduce the new activation diagnostic, pass
`--checkpoint checkpoints/heston-time-context-provisional` to the command above
and choose a new output path. The script reads the saved adapter ID.

# One bounded Heston inference follow-up

This experiment continues M3.3 after the original checkpoint failed conditional
test checks. It uses the same two-target physical-measure Heston task, prior,
initialization, daily discrete law, GRU and eight observed-history features.
It changes the posterior density estimator to a spline coupling flow. This
uses conditional density training and deterministic validation loss evaluation;
the original flow-matching objective samples auxiliary noise/times. That is a
reason to test this alternative, not an established explanation of the old
checkpoint's state-mean errors. Loss values from the two objectives are not
comparable measures of posterior accuracy.

## Design fixed before training

- Configuration: `configs/tasks/heston_inference_followup.json`; new root seed
  **43022**, with 32,768 training, 2,048 validation and 256 final-test histories.
  Root 43021's previously inspected test histories are regression cases only.
- Reference: first **16 validation histories**, selected before viewing results,
  checked by the same grid/particle refinement and independent-repeat procedure.
  If reference stability fails, improve the numerical reference before judging
  the neural candidate; retain the failed report and the case selection.
- Candidate: four dual spline coupling blocks, eight bins, 32-by-32 subnets;
  same GRU32/summary16, adapter and normalization. Train once for 80 epochs,
  batch size 256, seed 871; select minimum validation loss.
- Evaluation: 2,048 posterior draws per history; the existing conditional and
  predictive thresholds stay unchanged. Evaluate aggregate ranks/coverage on
  the first 256 validation histories and compare all 16 reference cases.
- Only if validation passes, freeze the candidate and evaluate root 43022's
  final-test split: 256 aggregate cases and the first 16 independently checked
  conditional-reference cases. No final-test diagnostics enter training or
  candidate selection. Record failure without changing thresholds or retuning
  against the same final holdout.

The new root produces independent base examples in every split. This is a
development comparison, not a controlled claim that the estimator architecture
alone causes any observed improvement: the training bank also changed.
No new simulator, storage framework, market data or cluster job is needed.

## Commands

Use new output directories throughout:

```bash
pixi run heston-bank --config configs/tasks/heston_inference_followup.json \
  --out /tmp/heston-followup-bank
pixi run heston-reference --bank /tmp/heston-followup-bank --cases 16 \
  --out /tmp/heston-followup-reference-validation
pixi run heston-train --bank /tmp/heston-followup-bank \
  --references /tmp/heston-followup-reference-validation \
  --out /tmp/heston-followup-trained --density-estimator spline \
  --epochs 80 --batch-size 256
```

The old checkpoint remains under `checkpoints/heston-theta-state-provisional`,
with its unchanged reports and original source hashes. Those hashes describe
commit `9b46719`; the follow-up's source hashes describe the new experiment.
The milestone review notebook retains the old failure as a didactic example.

## Recorded outcome

All 16 validation references passed stability checks. The spline candidate
trained for 80 epochs in 342.7 seconds including compilation, and seeded
checkpoint reload reproduced its draws exactly. It **failed validation**:

| Check | Result | Maximum |
|---|---:|---:|
| Mean RMSE / reference SD | 0.246 | 0.20 |
| Largest absolute mean error / reference SD | 0.792 | 0.40 |
| Largest relative posterior SD error | 46.1% | 30% |
| Largest correlation error | 0.077 | 0.20 |
| Largest relative predictive SD error | 14.7% | 15% |
| Largest predictive interval endpoint error / predictive SD | 0.300 | 0.15 |

The state rank diagnostic also narrowly exceeded its DKW bound (0.0947 versus
0.0925). Marginal 90% coverage was 92.2% for theta and 86.7% for state;
descriptive ten-day future-loss interval coverage was 90.2%. These are
development results, not final-test acceptance or a like-for-like improvement
over the old checkpoint's four test references.

**The new final test split was not evaluated.** No test reference or neural
test report was produced. Its arrays were generated and integrity-checked with
the bank; they did not enter loss, diagnostics or model selection. The current
candidate remains provisional, and M3.3 acceptance remains pending. The broader
validation set caught this failure before another final test was consumed.

The next investigation should examine how the observed-history representation
and training fit support current-state inference, using validation and the old
regression cases. This run does not establish which component caused the error.
Keep the reserved holdout for a candidate that passes the declared checks.

Evidence: [bank manifest](results/heston-inference-followup/bank-manifest.json),
[reference stability](results/heston-inference-followup/reference-validation.json),
[neural validation](results/heston-inference-followup/neural-validation.json),
[per-case checks](results/heston-inference-followup/validation-checks.npz), and
[provisional spline checkpoint](../checkpoints/heston-spline-provisional/README.md).
The [milestone notebook](../notebooks/05_heston_joint_inference.py) distinguishes
this follow-up from the original frozen-test results.

The next [controlled time-context experiment](heston-time-context.md) reuses
this bank and its 16 validation references. It improves the recorded errors but
still fails two gates; the reserved final test remains unevaluated.

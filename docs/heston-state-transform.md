# One bounded experiment: learn the signed state on an asinh scale

## Motivation and unchanged scientific target

The [precision diagnostic](heston-diagnostics.md) found a persistent neural
state-mean discrepancy on a calm validation history. The network does learn
from returns, but its state posterior is too broad there. Global scaling alone
leaves small near-zero states and a long positive range for the same finite
density network to represent.

Test one reversible change of coordinates:

\[
z=\operatorname{asinh}(a_T/c),\qquad
a_T=c\sinh(z),\qquad c=0.01.
\]

Here `a_T` is the signed full-truncation auxiliary variance in annualized variance
units. The fixed scale is one quarter of the known initial variance 0.04; it is
not fitted per history or selected by a scale sweep. Near zero the transform is
approximately linear; for large positive or negative values it grows roughly
logarithmically. It compresses the large-state range while preserving sign and
all information. This may help a finite network learn the conditional shape;
it is a hypothesis, not a guarantee of improved inference.

Negative auxiliary states remain valid, and no clipping, rejection or selective
redrawing is permitted. The prior, simulator, observed returns, joint target and
forecast continuation remain unchanged. Raw banks and `training_data` retain
original-unit targets. The adapter transforms the state before float32 casting
and training-only global normalization; sampling reverses the saved operations.
Both target columns keep their common draw index.

The scalar forward log-Jacobian is

\[
\log\left|\frac{dz}{da}\right|
=-\tfrac12\log(c^2+a^2),
\]

and the inverse log-Jacobian is `log(c) + log(cosh(z))`. The implementation
provides both for density evaluation as well as sampling. Training a normalized
density in transformed coordinates is valid: the target-only Jacobian does not
depend on network weights. **Raw validation-loss values from different target
coordinates cannot be compared directly.** Select an epoch within each run,
then compare posterior/forecast errors in original units. The inverse `sinh`
can amplify extreme neural tails, so finite-output and raw-unit checks remain
essential.

## Protocol fixed before the experiment

1. Add adapter `heston_returns_time_asinh_state_v3`. Preserve both earlier
   adapters and checkpoints. Check signed forward/inverse round trips, exact
   Jacobian against numerical derivatives, training-only transformed moments,
   observation/target separation, and seeded checkpoint reload.
2. Train **one** candidate from scratch on the unchanged root-43022 bank:
   32,768 training and 2,048 validation histories; explicit-time GRU32/summary16,
   eight trailing moments, four spline coupling layers/eight bins/32×32
   subnetworks, seed 871, 80 epochs, batch size 256. Select minimum validation
   loss within this run. Only the state target transform changes relative to
   the explicit-time baseline. Use the same 16 reference cases, 256 aggregate
   validation histories, 2,048 posterior draws and all existing thresholds.
3. If any validation gate fails, retain the full report and provisional
   checkpoint. Do not try other scales, seeds or architectures in this
   increment, and do not evaluate the reserved root-43022 test. Move the next
   implementation increment to M4.1 HKDE while M3.3 remains pending.
4. If validation passes, freeze the selected weights and repeat validation
   with 16,384 posterior draws per history using the same evaluation seed and
   unchanged thresholds. This higher-precision check must also pass; it does
   not replace or erase the first result. If it fails, follow step 3.
5. Only after both validation checks pass, generate/check the first 16
   root-43022 test references using the existing reference-refinement protocol,
   then evaluate the frozen candidate once on all 256 reserved histories with
   16,384 draws and unchanged thresholds. Record every result. Do not retune
   after viewing it. A reference failure blocks acceptance. Acceptance is for
   this bounded discrete Heston task only, not full SVJ or real markets.
6. Add the measured outcome and an explanatory section to notebook 05; verify
   its offline default and publish to the existing 5/10 PR. Update plan #3's
   status without changing the open curriculum choice. The next implementation
   milestone is M4 regardless of this experiment's outcome; expanding the
   returns task requires a separate reviewed experiment.

This is one local run, not a larger training bank, cluster job or tuning
framework. Reusing inspected validation data makes this a development
experiment. Only the final test, if reached under the protocol, supplies new
held-out accuracy evidence.

```bash
pixi run heston-train --bank /tmp/heston-followup-bank-v1 \
  --references /tmp/heston-followup-ref-validation-v1 \
  --out /tmp/heston-asinh-state-v1 --density-estimator spline \
  --asinh-state --epochs 80 --batch-size 256
```

The [shared-bank follow-up](heston-inference-followup.md) explains bank and
reference regeneration. Use a new output directory. A failed scientific gate
makes the command exit nonzero after saving its report; that is distinct from
a software exception. The loader integrity-checks all bank split files, but
test observations and targets do not enter training or development metrics.

## Measured outcome and stop decision

The single run completed in **350.8 seconds**, including compilation, and selected
epoch **49 of 80** by within-run validation loss. Seeded checkpoint reload
reproduced joint draws exactly. Forty-three software tests pass, including
the signed transformation/Jacobian, conditioning separation, transformed
training-only normalization and all earlier checkpoint paths.

| First validation check | Original-state v2 | Asinh-state v3 | Maximum |
|---|---:|---:|---:|
| Mean RMSE / reference SD | 0.1732 | 0.1865 | 0.20 |
| Largest absolute mean error / reference SD | 0.4554 | **0.5268 — fails** | 0.40 |
| Largest relative posterior SD error | 0.2886 | **0.3212 — fails** | 0.30 |
| Largest correlation error | 0.0707 | 0.0878 | 0.20 |
| Largest relative predictive SD error | 0.0881 | 0.1017 | 0.15 |
| Largest predictive interval endpoint error / predictive SD | 0.2099 | **0.2403 — fails** | 0.15 |

Both marginal SBC rank diagnostics pass: deviations 0.0283 for theta and 0.0659
for state, below 0.0925. Nominal 90% coverage is 91.4% and 87.9%; predictive
90% coverage is 89.8%. Improved marginal rank summaries do not override the
conditional failures or establish an independently validated posterior.

The largest state-mean and predictive endpoint errors remain on `validation:0`.
Its neural state mean is 0.00798 versus the reference 0.00644. The new largest
posterior-SD failure is theta on `validation:8` (32.1% too wide). All 16
preselected histories remain in the report. This coordinate change therefore
does not improve the checked conditional approximation under the fixed training
design; it is not evidence that asinh transforms generally fail.

**Stop rule applied:** no higher-draw validation, final-test references, reserved
test evaluation, additional seeds, alternative scales or extra epochs were run.
The v3 checkpoint is retained only as provisional experiment evidence; it does
not replace the better-performing, still-provisional v2 candidate. M3.3 posterior
acceptance remains pending. The next implementation increment is **M4.1 HKDE
law and reference pricing**, which depends on the accepted kernels rather than
this returns-inference gate. The curriculum choice remains open.

Evidence: [complete validation report](results/heston-state-transform/neural-validation.json),
[rank/coverage arrays](results/heston-state-transform/validation-checks.npz),
[checkpoint and adapter contract](../checkpoints/heston-asinh-state-provisional/README.md),
and [marimo review](../notebooks/05_heston_joint_inference.py).

The code/protocol was committed as `a28e39c` before the run. The report records
the exact source, bank, reference and checkpoint hashes; it preserves the full
training history and original-unit metrics. The earlier bank and reference
artifacts are reused unchanged.

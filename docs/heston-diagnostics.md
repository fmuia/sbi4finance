# Which Heston diagnostic answers which question?

The key review plots concern the **latest explicit-time spline checkpoint** and
root-43022 validation data. They are separate from notebook 05's older four-case
regression illustrations. No real observations are required for these checks.

| Plot | Question | Evidence and limit |
|---|---|---|
| [Prior versus posterior](results/heston-diagnostic-review/prior-posterior-comparison.png) | Is the network learning from each history or returning the prior? | Neural and reference means move together across histories, with substantial state contraction. Bars are means ± one SD, not credible intervals. This is descriptive evidence of data updating. |
| [Marginal SBC rank ECDF](results/heston-diagnostic-review/sbc-rank-ecdf.png) | Do true simulated theta/state values have approximately uniform ranks among posterior draws across many histories? | Development SBC on 256 validation histories; both marginal checks pass. Reusing validation for model selection prevents interpreting this as a fresh independent certificate. |
| [Conditional reference errors](results/heston-diagnostic-review/conditional-reference-errors.png) | Does the network approximate the posterior and future-loss distribution for each of the 16 checked histories? | These are reference comparisons, not SBC. The largest mean and predictive-interval errors occur on validation:0. |
| [Recurrent gate saturation](results/heston-diagnostic-review/gru-gate-saturation.png) | Could the time representation explain a learning difficulty? | Engineering diagnosis before/after training. Gate saturation is not a posterior-accuracy criterion. |

The [marimo review](../notebooks/05_heston_joint_inference.py), opened with
`pixi run review-heston`, provides equations, interpretation and the original
development history. The figures also have standalone links for quick review.

## Yes, the rank checks are SBC

For each of 256 independent base simulations, draw theta from its prior and
simulate the entire return/latent-state path. This supplies an observed history
and its true pair `(theta, terminal_state)`. The state is generated jointly
with the data; it is not drawn independently from a separate marginal prior.
Condition the neural posterior on the returns and known context, then draw
2,048 matched theta/state samples. For each target separately, record

\[
R=\sum_{s=1}^{2048}\mathbf{1}\{\vartheta^{(s)}<\vartheta^\star\},
\qquad U_R=\frac{R+U}{2049},\quad U\sim\operatorname{Uniform}(0,1).
\]

Under correct inference with independent posterior draws and no ties, the rank
is discrete uniform. Jitter makes it continuous uniform for the ECDF display.
The plot shows `empirical CDF − uniform CDF`, so departures from zero are easier
to see. This is the standard SBC idea
([Talts et al.](https://arxiv.org/abs/1804.06788)).

The recorded DKW diagnostic uses
`sqrt(log(4 / 0.05) / (2 * 256)) = 0.0925`, a simultaneous bound for the two
targets under the fixed-algorithm, independent-history uniform-rank null. Our
current checks reuse validation data for training/checkpoint/model decisions,
so we label them **development diagnostics**, without claiming that nominal
post-selection significance is retained. The new final holdout remains reserved.

The two observed maximum deviations are 0.0473 for theta and 0.0760 for state.
Both are inside the bound. Central 90% coverage, also reported, is a less
detailed marginal summary and not a substitute for examining ranks.

## Why also compare conditional posteriors?

Marginal rank uniformity does not establish correct joint dependence or an
accurate posterior for every history. An algorithm that ignores the observation
and draws from the prior can even pass parameter-only SBC averaged over the
prior predictive distribution. This limitation, and the importance of test
quantities involving data, are discussed by
[Modrák et al.](https://paulbuerkner.com/publications/pdf/2023__Modrak_et_al__Bayesian_Analysis.pdf).

For this small Heston task we can compute an independent particle/quadrature
reference. Comparing it with the neural posterior directly addresses the
conditional approximation we plan to use for forecasting. The reference is
still numerical: it needs refinement and repeated random seeds. Comparing a
posterior with one true parameter value is a different check; a correct
posterior need not be centered on that truth for each random history.

The mean-error metric is `(neural mean − reference mean) / reference SD`.
The predictive endpoint metric divides the difference between matching loss
quantiles by the reference **loss** SD. Their denominators and financial
meanings differ. The 0.40 and 0.15 limits are declared pilot engineering
tolerances, not universal statistical tests or evidence that exceeding a limit
automatically makes a model financially useless. Inspect absolute effects and
numerical uncertainty as well as the pass/fail labels.

This conditional check matters before calling the posterior validated. It does
not invalidate the simulator/bank infrastructure or block the independent HKDE
branch. The gate-activation diagnosis has already motivated a useful scaling
change; further activation analysis is not a separate acceptance requirement.

## Are we simply reproducing the prior?

No: the checked posteriors move with the observed history and mostly track the
independent reference. The prior-versus-posterior figure shows all 16 selected
histories, including histories where theta remains weakly identified. A broad
posterior can be appropriate; contraction alone is not proof of accuracy.

The theta prior has analytic mean 0.04328 and SD 0.01705. The terminal-state
prior is the unconditional marginal implied by the theta prior, known initial
state and one year of dynamics. It is **not** an independently specified state
prior or stationary distribution. Its estimated mean is 0.04284 and SD 0.02534,
using the 32,768 training-bank simulations without conditioning on their returns.
Only `train.npz` is read for that estimate; its mean Monte Carlo SE is 0.00014.

On the calm `validation:0` history, the original neural evaluation gives theta
mean 0.02776 and SD 0.00686, and state mean 0.00777 and SD 0.00370. Those SDs are
40% and 15% of their respective prior SDs. On `validation:15`, the neural state
mean instead rises to 0.08647. Across the 16 references, a baseline that always
returns the prior mean has reference-normalized mean RMSE **3.265**, versus
**0.173** for the neural candidate. This metric compares posterior means to
reference posterior means, not estimates to one simulated truth per history.

The [comparison record](results/heston-diagnostic-review/prior-comparison.json)
pins the source hashes. The state-prior sample also supplied network training;
this baseline is a descriptive estimate of the model prior, not new independent
validation evidence. Visible data updating rules out simple prior reproduction
on these cases but does not establish full joint or conditional accuracy.

## Bounded diagnostic: sampling precision, with fixed weights

Protocol fixed before running this follow-up:

1. Retain the explicit-time checkpoint, original 16 validation reference cases,
   observations and original failed result. Generate only validation inputs;
   do not load or evaluate the reserved test split.
2. Repeat neural/reference prediction with 16,384 draws and three independent
   seeds (11101, 11201, 11301). Record all 16 cases, absolute loss quantiles and
   normalized errors. Report ranges across repeats and neural-mean Monte Carlo
   standard errors; do not select the most favorable repeat.
3. For the already-identified `validation:0`, refine the reference from
   48 nodes/4,096 particles to 96 nodes/16,384 particles, with independent seeds
   27101 and 27102. Compare both with the existing reference, then compare the
   fixed neural posterior with the refined results.
4. Keep these results as a **post hoc validation diagnostic**, not new SBC,
   new holdout evidence, or a retrospective override of the original gate.
   Make no training or threshold changes in this diagnostic. Use the result
   to decide whether another model change is justified.

This addresses whether we are pursuing a persistent approximation error or
finite-draw/reference noise before spending time on another training run.

### Measured outcome

The [report](results/heston-precision/report.json) records the complete 53.7-second
diagnostic. All 16 histories and all three repeats are retained. Across repeats,
aggregate mean RMSE is 0.171–0.177 reference SD. Against the original reference,
the maximum state-mean error stays at 0.426–0.449 SD; the maximum predictive
endpoint error varies from 0.147 to 0.180 predictive SD. Thus the latter is more
sensitive to finite sampling near its 0.15 tolerance.

The two refined case-0 references differ in state mean by only 0.007 reference
SD. Against the second refinement, neural state-mean error is **0.412–0.435 SD**,
while neural-mean Monte Carlo SE is about **0.0096 SD**. Neural state SD remains
22–24% too large. These comparisons support a persistent conditional
approximation error, rather than draw noise or the tested reference refinement
explaining it. Finite numerical agreement is still not an exact-reference proof.
The error is clearly nonzero, but the smallest excess above the 0.40 engineering
limit is only about 1.3 neural-mean Monte Carlo SEs and excludes reference error.
We do not claim statistically certain rejection at precisely that boundary.

The [case-0 comparison](results/heston-precision/case-0-comparison.png) shows
the first prespecified neural repeat against the second refined reference:

| Quantity | Refined reference | Frozen neural posterior |
|---|---:|---:|
| State mean (annualized variance units) | 0.00647 | 0.00775 |
| State SD | 0.00296 | 0.00366 |
| 10-step future-loss 5% quantile (% initial wealth) | −2.84% | −3.05% |
| 10-step future-loss median (% initial wealth) | −0.31% | −0.31% |
| 10-step future-loss 95% quantile (% initial wealth) | 2.74% | 2.96% |

For this repeat the upper endpoint differs by **0.22 percentage points of
initial wealth**; predictive SD is 8.4% larger. These are concrete effects for
one synthetic history, not a general risk-performance claim. The three-repeat
ranges are descriptive, not confidence intervals. The graph is a conditional
comparison, **not SBC**; no new training or reserved-test evaluation occurred.

This is useful inference and substantial progress within the two-target Heston
pilot. It is not accepted full-SVJ inference. The original failed gate remains
unchanged. A further targeted training change would need its own recorded
protocol and validation, followed by the reserved final test only after passing
development checks. The independent HKDE branch can proceed in parallel.

### Reproduction

The small figures can be regenerated offline from committed artifacts:

```bash
pixi run python scripts/plot_heston_diagnostics.py --out /tmp/heston-diagnostic-figures
pixi run python scripts/plot_heston_precision.py --out /tmp/heston-precision-figure
```

Optionally append `--prior-bank /tmp/heston-followup-bank-v1` to the first
command to regenerate the state-prior summary from the training split alone.
To repeat the precision computation, first regenerate the validation references
as described in [the shared-bank follow-up](heston-inference-followup.md), then:

```bash
pixi run python scripts/check_heston_precision.py \
  --config configs/tasks/heston_inference_followup.json \
  --checkpoint checkpoints/heston-time-context-provisional \
  --references /tmp/heston-followup-ref-validation-v1 \
  --out /tmp/heston-precision-reproduction
```

Use a new output directory. The command generates validation inputs directly
and does not open the test bank. Successful execution means the diagnostic
completed, not that the checkpoint passed acceptance.

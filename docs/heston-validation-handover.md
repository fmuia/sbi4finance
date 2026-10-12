# Agent handover: Heston state and forecast decision, issue #25

Prepared 12 October 2026. This briefing delegates a decision and, only if
justified, one bounded investigation. It does not start another experiment.
See the [parallel-agent allocation](parallel-agent-handover.md) for ownership.

## Task and starting point

Work through [issue #25](https://github.com/fmuia/sbi4finance/issues/25): decide
whether to defer restricted Heston posterior acceptance, retain its current
diagnostic lesson, or investigate one remaining conditional error. The kernels,
banks, reference and forecast pipeline already exist. The better v2 network
learns from returns, but is **provisional**; the later v3 change did not fix its
errors. Full SVJ/SVCJ inference is not accepted.

Evidence review can begin in parallel. Further experimentation depends on the
[curriculum decision #26](https://github.com/fmuia/sbi4finance/issues/26); explicit
deferral is valid completion. No framework, automatic sweep or cluster campaign.

Repository: <https://github.com/fmuia/sbi4finance>. Local root:
`/Users/afengler/Projects/proj_sbi4finance`.
These heads were verified before this documentation update; recheck on starting.

| Purpose | Existing branch / worktree | Verified head |
|---|---|---|
| Narrow Heston implementation base, open [PR #11](https://github.com/fmuia/sbi4finance/pull/11) | `feat/heston-joint-inference` / `sbi4finance-heston-inference` | `ba920bbd628a0913f43984edf0f2718cd755cf6f` |
| Current cross-model handover, open [PR #19](https://github.com/fmuia/sbi4finance/pull/19) | `feat/hkde-offline-tutorial` / `sbi4finance-offline-tutorial` | `b588df33d8fa1456c05bae67438d130d5b00ee1a` |

Use a separate branch/worktree from current #11 for Heston changes; after merges,
use the integration owner’s verified successor. A PR target does not import
later target commits automatically. Do not edit another agent's worktree or merge.
[Integration #27](https://github.com/fmuia/sbi4finance/issues/27) owns dependency
order and combined-tree verification; it can proceed while Heston is deferred.

## Read and inspect first

1. [Diagnostic interpretation](heston-diagnostics.md): SBC versus conditional
   checks, the prior control and the precision investigation.
2. [Explicit-time v2](heston-time-context.md), then
   [asinh-state v3](heston-state-transform.md): frozen designs and retained stops.
3. [Original inference task](heston-joint-inference.md) and
   [spline follow-up](heston-inference-followup.md): numerical reference and
   separate experiment histories.
4. [Notebook05](../notebooks/05_heston_joint_inference.py): didactic review with
   historical and subsequent validation results clearly separated.

Key graphs: [prior versus posterior](results/heston-diagnostic-review/prior-posterior-comparison.png),
[development SBC](results/heston-diagnostic-review/sbc-rank-ecdf.png),
[conditional errors](results/heston-diagnostic-review/conditional-reference-errors.png),
and [refined case-0 state/forecast comparison](results/heston-precision/case-0-comparison.png).
The last graph is a conditional comparison, **not SBC**. Bars in the prior
comparison are means ± one SD, not credible intervals.

## What one example and the inference target mean

The pinned task is `heston_theta_state_v1` under the **physical measure P**.
One observation is a complete sequence of 252 decimal log returns, stored as
`[batch,252,1]`, with positive year fractions `dt_years = 1/252`. No jumps or
measurement noise enter this restricted inference task. The prior is
`theta ~ LogUniform(0.02,0.08)`; drift `mu=0.05`, mean reversion `kappa=2`,
volatility of variance `xi=0.2`, correlation `rho=-0.6` and initial variance
`a_0=0.04` are known. See the
[follow-up task JSON](../configs/tasks/heston_inference_followup.json).

For `v_t=max(a_t,0)`, the declared daily full-truncation Euler law is

\[
r_{t+1}=(\mu-v_t/2)\Delta+\sqrt{v_t\Delta}\,Z^S_t,
\qquad
a_{t+1}=a_t+\kappa(\theta-v_t)\Delta+\xi\sqrt{v_t\Delta}\,Z^v_t,
\]

where the standard normal shocks have correlation `rho`. There is **one**
internal update per observed day. This targets a specified discrete law,
not an exact continuous-time Heston posterior or the separate eight-substep
kernel benchmark. Numerical refinement and inference accuracy are distinct.

The joint target is `p(theta,a_T | r_1:T)`. The terminal **signed auxiliary
state** `a_T` occurs after the final return's update. Its effective variance
`max(a_T,0)` is not an interchangeable restart state. Negative auxiliary values
are valid; clipping, rejection or selective redrawing changes the model. State
truth in synthetic banks is a target, never observed conditioning. Its marginal
prior is induced by theta, initialization and dynamics, not separately chosen.

The reference integrates over `log(theta)` by quadrature and filters the state
with particles. For `v>0`, given the observed return `r` and
`m=(mu-v/2)Delta`, its adapted state transition is

\[
a'\mid a,r,\theta\sim\mathcal N\left(
a+\kappa(\theta-v)\Delta+\xi\rho(r-m),
\xi^2v\Delta(1-\rho^2)\right).
\]

At `a<=0`, the return is the deterministic atom `r=mu*Delta`, followed by
`a'=a+kappa*theta*Delta`. Keep that atom and leverage dependence; adding a
variance floor or independent transition would change the reference problem.
The reference uses finite particles/quadrature and needs precision checks.

Forecasts retain each matched `(theta,a_T)` draw and continue with fresh
innovations. Ten-step relative loss is `L=1-exp(sum future log returns)`.
This produces the predictive mixture over parameter, state and future-path
uncertainty. Its quantiles are not averages of conditional VaRs. Current
diagnostics use a central **90% loss interval**, not a validated 99% VaR or ES.
This P task does not price options or define a P/Q change of measure.

## Evidence and constraints already established

| Stage | Recorded outcome |
|---|---|
| Original flow-matching candidate, root 43021 | Marginal SBC passed on 256 test histories, but four conditional reference cases exposed mean RMSE 0.425 > 0.20, maximum mean error 0.881 > 0.40 and predictive endpoint error 0.166 > 0.15. That test is used; it is regression evidence now. |
| Raw-index spline, new root 43022 | All 16 validation references passed stability checks; network failed conditional gates and the state rank diagnostic. |
| Explicit-time spline v2 | Corrected excessive Time2Vec scale. Mean RMSE 0.1732 passes; maximum mean error **0.4554 > 0.40** and predictive endpoint error **0.2099 > 0.15** fail, both on `validation:0`. Marginal development SBC passes. |
| Fixed-v2 precision diagnostic | Three 16,384-draw repeats and two refined case-0 references retain state-mean error **0.412–0.435 reference SD**, with neural mean SE about 0.0096 SD. Predictive endpoint errors are more draw-sensitive. No training or holdout evaluation occurred. |
| Asinh-state v3 | One fixed transform `asinh(a/0.01)` fails maximum mean error **0.5268**, SD error **0.3212 > 0.30**, and predictive endpoint error **0.2403**. Its SBC passes. Stopped before higher-draw validation or final test; v3 does not replace v2. |

The prior-mean baseline has conditional mean RMSE **3.265**, versus **0.173**
for v2: these cases show data updating. On the refined case-0 graph, reference
versus neural state means are 0.00647 versus 0.00775; the 95th loss percentile
is 2.74% versus 2.96% of initial wealth, a 0.22-percentage-point difference.

The 0.40/0.15 limits are pilot engineering tolerances. The persistent error is
real evidence of approximation limits, but the precision study does not claim
statistically certain rejection exactly at 0.40. Preserve both practical size
and uncertainty; neither silently relax the limit nor portray a small excess
as general financial failure. Passing marginal SBC does not certify conditional
or joint accuracy; current validation SBC is reused development evidence.

Root **43022** has 32,768 train, 2,048 validation and 256 test histories.
Its test arrays were generated and integrity-checked, but **no test reference
or neural test evaluation was run**. Preserve that reserve; all 16 validation
reference cases, including case 0, remain in any comparable diagnostic.

## Ordered subplan and completion criteria

### H25.1 — Decide the value of further work

Review saved reports, graphs and original-unit consequences. Coordinate with
#26 whether the returns route needs an accepted posterior. Recommend **defer**,
**retain as a diagnostic lesson**, or **investigate one mechanism**; a useful
diagnostic lesson need not advertise an accepted risk forecast.

**Deliverable:** a short decision note, exact scientific claim and dependency
on curriculum. Deferral can proceed directly to H25.4 and complete this issue.

### H25.2 — If justified, freeze one bounded development experiment

Name one testable mechanism and the smallest change that examines it. Bind
task, adapter/checkpoint, bank, references, seeds, case selection, budgets,
metrics, numerical precision and stopping rule before execution. Reuse existing
interfaces and source-bound reports; write to a new output directory. Preserve
old failures. A different task-specific criterion requires a prospective,
use-based rationale, not retroactive acceptance of inspected cases.

**Deliverable:** one protocol with a cost bound. No seed/architecture/scale
sweep, automatic bank expansion or test-set preview. A change in simulator law,
prior or target creates a new task and is outside this default increment.

### H25.3 — Evaluate the declared candidate and honor the stop

Check reference precision before judging conditional neural error. Compare
joint means, widths, dependence, signed-state behavior and forecasts, as well
as SBC and the prior control. Keep all prespecified cases and outcomes. If
validation fails, retain provisional status and stop. Save enough arrays to
audit any new statistics; old reports do not store every posterior draw.

The reserved root-43022 test is **not an immediate action**. Only after the
selected candidate passes all its prospective validation prerequisites, freeze
it and its test protocol, check the preselected test references, and evaluate
once. Never use test results to select another candidate or repeat until pass.

**Deliverable:** accepted / provisional / reference-blocked result with scope,
actual cost and an independent arithmetic check of the saved evidence.

### H25.4 — Teach and publish the decision

Extend notebook05 with the decision, equations, graphs and any actual new
findings. Preserve its old failures and explain which plot is SBC versus a
conditional comparison. Verify a fresh-process offline export and rendered
figures with no real-market fixtures; report exactly what ran. Update #25 and
send handover changes to the shared-doc owner rather than concurrently editing
their branch. Deferral is completion of this decision, not posterior acceptance.

Use a numbered follow-up such as `1/1-simulator-followup-heston-validation`,
linking #25, the exact base and dependencies; update the denominator if the
bounded stack changes. Publish for review, **do not merge**. #24 owns HKDE,
#26 curriculum, and #27 integration; do not change their scientific gates.

## Files, commands and implementation guardrails

| Purpose | Existing files |
|---|---|
| Simulator, bank and continuation | `src/sbi4finance/{heston,heston_bank,heston_forecast}.py` |
| Joint inference, transforms, reference | `src/sbi4finance/{heston_inference,heston_transforms,heston_reference}.py` |
| Training / reference / precision scripts | `scripts/{train_heston,check_heston_reference,check_heston_precision}.py` |
| Better provisional baseline | `checkpoints/heston-time-context-provisional/{model.keras,task.json,README.md}` |
| Original and later evidence | `docs/results/heston-{inference,inference-followup,time-context,state-transform,precision}/` |

Initial review commands, from the chosen checkout:

```bash
PYTHONPATH=src pixi run review-heston
PYTHONPATH=src pixi run python -m unittest discover -s tests -p 'test_heston*.py' -v
PYTHONPATH=src pixi run python scripts/plot_heston_diagnostics.py --out /tmp/heston-review-figures
PYTHONPATH=src pixi run python scripts/plot_heston_precision.py --out /tmp/heston-review-precision
```

Choose unused output paths. Plot commands read saved evidence. Do not blindly
copy historical `heston-train` or `--split test` examples. Raw banks/particle
arrays are regenerated assets, not guaranteed at old `/tmp` paths; inspect manifests.

The v2 adapter is `heston_returns_time_and_window_moments_v2`; it appends known
end times and eight observed-history features. The bank stays `[B,T,1]`.
Scaling is global and training-only, never per-window standardization. Load the
saved adapter/task; v3 requires its registered signed-asinh transform. Preserve
paired targets, return amplitudes and observation/latent/future separation.
The pinned environment is Python 3.12, BayesFlow 2.0.14, Keras 3.15.1, JAX
0.11.1; use `KERAS_BACKEND=jax` outside Pixi and the actual checkout's `src`.

All required work is **S**, synthetic teaching/validation; no market acquisition
is needed. Optional real conditioning (**M**) and empirical assessment (**E**)
neither reveal true market latent states nor replace these checks.

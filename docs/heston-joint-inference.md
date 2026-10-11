# A small joint Heston inference pilot

This is the fifth simulator-plan increment, M3.3. It connects the Heston
kernel to a saved synthetic bank, a same-model reference posterior, a small
BayesFlow network and conditional future simulation. The first task infers
**long-run variance and the current continuation state**, with other model
parameters known. It is a restricted Heston pilot; Gaussian-jump inference,
Francesco compatibility and the final tutorial model choice remain separate.

**Status:** the bank, reference, training/reload and forecasting path is
implemented. The frozen checkpoint **failed conditional test checks** and is
shipped as provisional evidence, not an accepted tutorial posterior. M3.3
inference acceptance remains open.

For a guided, visual review, run **`pixi run review-heston`** to open
[`notebooks/05_heston_joint_inference.py`](../notebooks/05_heston_joint_inference.py).
It explains the task, displays the recorded checks, and lets you inspect the
four original reference cases with posterior and future-loss plots. The
[bounded follow-up](heston-inference-followup.md) records subsequent development
separately; it does not replace this original failure report.

The subsequent [time-context experiment](heston-time-context.md) diagnosed
recurrent-gate saturation and improved conditional accuracy by scaling known
times. Two validation gates still fail on a low-variance history, so the new
final test remains unevaluated. The notebook now shows this comparison too.

## What one observation means

One training example is a complete history of 252 daily decimal log returns,
not one return and not an option surface. Under the physical measure P, draw
a long-run variance `theta` from a log-uniform prior on `[0.02, 0.08]`, hold
`drift=0.05`, `kappa=2`, `vol_of_vol=0.2` and `rho=-0.6` fixed, and start from
known variance `0.04`. Each observation spans `1/252` years. No jumps or
measurement noise are added.

The task is recorded in
[`configs/tasks/heston_inference.json`](../configs/tasks/heston_inference.json).
Its numerical law is full-truncation Euler with **one internal update per
observed day**. This is deliberately different from the eight-substep timing
configuration in the preceding increment. The network and inference reference
both target this daily discrete law; neither is silently substituted for a
continuous-time Heston posterior.

The two targets are \((\theta,a_T)\), where \(a_T\) is the signed auxiliary
variance required to restart that discrete law. Its nonnegative reported
variance is \(\max(a_T,0)\); those are not interchangeable continuation states.
In the synthetic bank we know both targets because we generated the history.
For an actual market history they would be unknown, and must never appear
among the network's observed inputs.

This reduced task makes reference validation affordable. It does not show that
SBI is necessary, that all Heston parameters are identified, or that conventional
inference performs poorly. It exercises the parameter/state dependency and
conditional forecasting that the earlier GBM drift example did not have.

## Why the reference matches the simulator

For the current signed auxiliary value \(a\), let \(v=\max(a,0)\),
\(m=(\mu-v/2)\Delta\), and let \(r\) be the next observed log return. If \(v>0\),
the one-step simulator gives

\[
r\mid a\sim\mathcal N(m,v\Delta),
\]
\[
a'\mid a,r,\theta\sim\mathcal N\left(
a+\kappa(\theta-v)\Delta+\xi\rho(r-m),
\;\xi^2v\Delta(1-\rho^2)\right).
\]

The covariance correction \(\xi\rho(r-m)\) matters: the observed return and
the next variance use correlated shocks. Treating the state transition and
return as conditionally independent would discard that leverage information.
The final target is the state **after** the final return's update.

At a fixed candidate `theta`, the fully adapted auxiliary particle filter
starts from the known initial state. At each observation it:

1. Evaluates each previous particle's return likelihood.
2. Adds the log of their average to the estimated log likelihood of the history.
3. Resamples ancestors according to those likelihoods.
4. Draws the next state from the conditional Gaussian above, leaving equal weights.

If \(a\le0\), there is a deterministic return \(r=\mu\Delta\), followed by
\(a'=a+\kappa\theta\Delta\). The reference handles this probability atom
explicitly. It does not add a variance floor, use an approximate equality
tolerance, drop difficult histories or redraw a failed reference. Keeping
drift fixed also fixes the atom's location. These details are especially
important if this task is later extended beyond the current regular pilot.

The remaining unknown static parameter is one-dimensional, so we use
Gauss-Legendre quadrature in \(x=\log\theta\). The prior is uniform in \(x\),
and grid posterior masses are proportional to
\((w_j/2)\widehat p(r_{1:T}\mid\theta_j)\). No extra theta Jacobian belongs
in these masses. A joint posterior draw chooses a grid row with its posterior
probability and then a terminal-state particle **from that same row**.

The conditional proposal is exact for the specified discrete transition;
the finite-particle likelihood and finite-grid posterior remain approximate.
Unbiased likelihood estimation would not make its logarithm or normalized
posterior unbiased. The broader particle-filter construction is described in
[Doucet and Johansen's tutorial](https://www.cs.ubc.ca/~arnaud/doucet_johansen_tutorialPF.pdf).
The equations above are derived directly from our declared simulator.

## Bank, network and checkpoint boundary

The bank uses the existing compact NPZ-plus-JSON pattern. There are independent
train, validation and test base histories, with stable split-qualified IDs,
separate prior/process random streams, source/configuration hashes and a
manifest written only after successful completion. A small fixed generation
batch bounds temporary trajectory memory. There is no new storage framework.

| Saved field | Role |
|---|---|
| `observations [B,252,1]` | Ordered decimal log returns available at the cutoff |
| `dt_years [252]` | Known time context; other fixed context is in the task |
| `parameters [B,1]` | Synthetic theta truth, a supervised target |
| `terminal_state [B,1]` | Synthetic signed current-state truth, a supervised target |
| `example_id [B]` | Independent base identity, never conditioning |

`conditioning_data` accepts only the history, its time grid and the declared
task. A small GRU summarizes the full ordered history. Eight additional
features are deterministic functions of observed returns: their mean and
annualized second moment over the trailing 8, 32, 128 and all observed days.
For shorter test fixtures, each length is capped at the available history.
These features help the small network preserve both recent and longer-run
information; they are not asserted to be sufficient statistics. The full
history remains available to the GRU.

All normalization is learned globally from training batches. There is no
per-history standardization that would erase volatility-level evidence.
Theta is mapped from its open prior interval to the real line by a logit
before float32 casting; the signed state retains an identity transform.
A single two-dimensional flow models their joint posterior. The saved adapter
reverses the transformation and returns paired raw-unit draws. Out-of-support
or nonfinite predictions fail explicitly rather than being clipped or redrawn.

The checkpoint bundle consists of `model.keras`, `task.json` (full task and
adapter identifier), and a README linking its reports. Frozen evaluation rejects differences in
the prior, known parameters, initialization, grid or adapter. A compatible
observation shape alone does not establish checkpoint compatibility.

## Conditional futures answer a different question

For each posterior draw \((\theta^{(s)},a_T^{(s)})\), simulate ten further daily
steps from that exact signed state using independent future shocks. The paired
draw index is preserved throughout. The relative loss of holding the asset is

\[
L^{(s)}=1-\exp\left(\sum_{t=T+1}^{T+10}r_t^{(s)}\right).
\]

The mixture of these draws represents a posterior-predictive distribution.
Its quantiles are not averages of conditional quantiles. This pilot compares
predictive standard deviations and central intervals with the reference and
reports central-interval coverage using one independently generated future
per evaluation history. It does not claim a reliable 99% tail or expected
shortfall estimate from a few thousand draws.

The posterior-predictive calculation receives inferred parameter/state draws.
The bank's true values are used
only to generate separate future outcomes for evaluation, after posterior
inference. The demonstration supplies no option-pricing or Q-calibration
claim and does not link P and Q.

## Reproduce in stages

Every output path below must be new. The bank defaults to 32,768 training,
2,048 validation and 256 test histories (about 72.8 MB of arrays).

```bash
pixi install --locked
pixi run heston-bank --out /tmp/heston-bank
pixi run heston-task-numerics --out /tmp/heston-numerics.json
pixi run heston-reference --bank /tmp/heston-bank --out /tmp/heston-reference-validation
pixi run heston-train --bank /tmp/heston-bank \
  --references /tmp/heston-reference-validation --out /tmp/heston-trained --epochs 80 --batch-size 256
```

The reference command checks the first four validation histories at
24 grid nodes/1,024 particles, 48 nodes/4,096 particles, then an independent
repeat at the finer setting. Predeclared checks limit changes in posterior
means to 0.1 standard deviations, widths to 10%, and correlation to 0.1.
These are diagnostic stability checks, not an exact posterior certificate.

Neural development uses validation data only. Its conditional-reference gate
checks means, widths, dependence and ten-day predictive quantities; marginal
rank and coverage diagnostics are reported separately. A failed checkpoint
remains provisional. Freeze a successful candidate before making test references
and evaluating it without further training:

```bash
pixi run heston-reference --bank /tmp/heston-bank --split test \
  --out /tmp/heston-reference-test
pixi run heston-train --bank /tmp/heston-bank --checkpoint /tmp/heston-trained \
  --references /tmp/heston-reference-test --out /tmp/heston-test-evaluation
```

To evaluate the shipped candidate without retraining, replace
`/tmp/heston-trained` with `checkpoints/heston-theta-state-provisional`.
For that checkpoint the test command writes its results and exits nonzero:
failure of the conditional gate is the recorded outcome, not a setup error.
Independent test histories must not be reused as an untouched holdout after
further development. The bank and particle arrays are regenerated locally;
the repository includes the small checkpoint, reports and per-case test checks.
Exact array hashes describe the recorded environment; other platforms may
differ numerically and need their own checks.

This workflow needs no market files or provider connection. The focused tests
block outbound Python sockets during bank loading, training, checkpoint reload
and posterior sampling. Optional market application would condition compatible
frozen weights on a real window; empirical claims would additionally need
chronologically held-out outcomes. Neither replaces synthetic reference checks.

## Evidence and acceptance

Numerical fidelity, conditional inference and empirical adequacy are separate.
The daily-grid diagnostic checks three prior locations against continuous-time
moments and selected joint transforms, with paired 1/4/8-step refinements and
explicit Monte Carlo uncertainty. Inference comparisons concern the declared
one-step model even when that numerical diagnostic passes.

The initial raw-history-only GRU pilot failed conditional checks despite fairly
reasonable marginal coverage. That motivated the observed-history moment
features. An eight-to-32-thousand training-bank increase then brought validation
within the gates. A 16-to-64-step flow-sampler check changed conditional means
by less than `3e-7` reference standard deviations, so changing the ODE sampling
resolution did not explain the observed inference errors. These were development
experiments; the four validation reference cases and thresholds stayed fixed.

The final candidate was frozen before its first test evaluation. On 256 independent
test histories, marginal 90% coverage was **90.2% for theta and 89.1% for the
state**, and both marginal rank diagnostics passed their DKW check. Those
averages do not settle conditional accuracy. Against the first four test
references, results were:

| Conditional check | Frozen test result | Predeclared maximum | Result |
|---|---:|---:|---|
| Mean RMSE, in reference posterior SD | 0.425 | 0.20 | Fail |
| Largest absolute mean error, in reference posterior SD | 0.881 | 0.40 | Fail |
| Largest relative SD error | 28.4% | 30% | Pass |
| Largest correlation difference | 0.045 | 0.20 | Pass |
| Largest ten-day loss SD error | 11.3% | 15% | Pass |
| Largest central predictive interval endpoint error, in predictive SD | 0.166 | 0.15 | Fail |

State means on `test:0` and `test:2` exceeded the mean-error limit; theta on
`test:1` did too. The test reference itself passed refinement and independent
repeat checks: the largest coarse-to-fine mean change was 0.075 SD, and the
largest fine-repeat change was 0.014 SD. Thus the reported stability check does
not account for the neural discrepancies, although the reference is still finite.
No thresholds or model weights were changed after seeing these test results.

Central 90% future-loss interval coverage was 90.6% over the 256 histories,
using one independent future per history. This is descriptive synthetic
coverage, with roughly 1.9 percentage points nominal binomial standard error;
it neither overrides the failed conditional gate nor establishes tail-risk
accuracy. The numerical daily-grid checks passed at three prior locations,
which is a separate, bounded result.

The recorded local run generated and wrote the 72.8 MB bank in about 1.7 seconds
with the Numba cache already available. Training for 80 epochs took 320.5 seconds
including compilation; the best validation-loss checkpoint was selected.
Frozen-test sampling of 2,048 draws for each of 256 histories took 10.2 seconds
including sampling compilation, and the subsequent predictive-interval loop
took 12.8 seconds. These are task-specific wall times, not a cross-machine
benchmark or a claimed amortization break-even result. This pilot needed no
cluster job.

Recorded evidence:

- [Development decisions and failed earlier trials](results/heston-inference/development-history.json),
  [bank manifest](results/heston-inference/bank-manifest.json), and
  [daily-grid diagnostic](results/heston-inference/discretization.json).
- [Validation reference](results/heston-inference/reference-validation.json) and
  [test reference](results/heston-inference/reference-test.json), with settings,
  seeds, repeat results and source hashes.
- [Neural validation](results/heston-inference/neural-validation.json),
  [frozen test](results/heston-inference/neural-test.json), and
  [per-case test arrays](results/heston-inference/test-checks.npz).
- [Provisional checkpoint and usage boundary](../checkpoints/heston-theta-state-provisional/README.md).

The 36-test suite passes, including limiting-case reference checks, stable bank
identities, observation/target separation, training-only normalization and a
socket-blocked train/save/reload test. Software correctness and a useful
reference do not imply neural accuracy: **the implemented pipeline is ready
for iteration, while M3.3 posterior acceptance is pending**.

The next inference iteration should use a broader validation reference set to
target the conditional state errors and reserve a new independent holdout before
training. The inspected test histories remain a regression set. That is a
bounded next experiment, not a reason to build a new data or model framework.
Broader priors, unknown drift/other dynamics, jumps, missingness, different
return lengths and continuous-time inference require additional reference work.
Neither this pilot nor its failure settles the returns-versus-options curriculum.

# Heston #25: retain the diagnostic lesson; defer further experiments

Decision recorded 12 October 2026 for [issue #25](https://github.com/fmuia/sbi4finance/issues/25).
Retain notebook05 as an **optional diagnostic lesson** and defer further
posterior-acceptance experiments. The restricted Heston posterior remains
**provisional**. This resolves the decision requested by #25; it does not grant
posterior acceptance or select the final curriculum.

The current v2 candidate updates meaningfully from returns and passes marginal
development SBC, but fails its unchanged conditional validation gates. A
fixed-checkpoint precision diagnostic supports a persistent state-mean error
for the calm validation history. The single v3 state-coordinate experiment did
not resolve that error. These results support teaching conditional validation,
joint state continuation and numerical precision. They do not support an
accepted risk forecast, full SVJ/SVCJ inference, continuous-time accuracy or
empirical market adequacy.

## Why stop here?

[Curriculum #26](https://github.com/fmuia/sbi4finance/issues/26) is still open,
with no recorded choice requiring an accepted Heston returns posterior as of
this review. The existing evidence already makes a useful diagnostic lesson.
Another training candidate would consume development effort without an agreed
teaching need. No experiment is selected or frozen in this increment; H25.2
and the candidate-execution portion of H25.3 are therefore not applicable.

This decision does not choose P returns over Q options. #26 may retain this
lesson, omit it, or later justify one bounded investigation. Software integration
under [#27](https://github.com/fmuia/sbi4finance/issues/27) can proceed while
scientific acceptance is deferred. Neither HKDE nor rough-Bergomi gates change.

## Evidence reviewed and independently checked

The task is `heston_theta_state_v1` under P: 252 daily decimal log returns
condition the joint posterior of long-run variance `theta` and the terminal
**signed full-truncation auxiliary state** `a_T`. Other parameters and the
initial state are fixed. The declared simulator takes one Euler update per
observed day. Its posterior and forecasts target that discrete law. Latent
truth never enters conditioning, and forecasts retain matched `(theta,a_T)`
pairs, including valid negative states, with fresh future innovations.

The [saved-evidence arithmetic audit](results/heston-validation-decision/evidence-review.json)
binds the task, bank manifest, reference reports, checkpoints and arrays by
SHA-256. It checks all 16 prespecified validation cases, both targets and both
v2/v3 results. No raw bank or root-43022 test was opened.

| Conditional metric | V2 explicit time | V3 signed asinh | Unchanged limit |
|---|---:|---:|---:|
| Mean RMSE / reference SD | 0.1732 | 0.1865 | 0.20 |
| Maximum absolute mean error / reference SD | **0.4554** | **0.5268** | 0.40 |
| Maximum relative posterior SD error | 0.2886 | **0.3212** | 0.30 |
| Maximum correlation error | 0.0707 | 0.0878 | 0.20 |
| Maximum relative predictive SD error | 0.0881 | 0.1017 | 0.15 |
| Maximum predictive endpoint error / predictive SD | **0.2099** | **0.2403** | 0.15 |

Both marginal development SBC checks pass. V2's ECDF deviations are 0.0473
and 0.0760 against the recorded 0.0925 bound. Validation reuse makes these
development diagnostics, not an independent calibration certificate. Passing
marginal ranks does not establish joint or conditional accuracy. The prior-mean
control has conditional mean RMSE **3.2649**, compared with **0.1732** for v2:
the checked histories show substantial data updating, despite the failures.

All 16 references retain their recorded coarse/fine and independent-repeat
stability passes. The [precision study](heston-diagnostics.md) increased neural
draws to 16,384 across three seeds and refined case 0 twice. Relative to the
second refined reference, state-mean error remains **0.4123–0.4347 SD**, with
neural mean Monte Carlo SE about **0.0096 SD**. The refined reference state means
differ by 0.0070 SD. This supports a persistent discrepancy; it does not prove
statistically certain rejection precisely at the 0.40 engineering boundary.
Predictive endpoint errors are more draw-sensitive and straddle their tolerance.

For the first prespecified repeat and second refined reference:

| Quantity | Refined reference | V2 neural |
|---|---:|---:|
| State mean, annualized variance units | 0.0064694 | 0.0077545 |
| State SD | 0.0029562 | 0.0036583 |
| Ten-day loss 5th percentile, % initial wealth | −2.8373% | −3.0530% |
| Ten-day loss median, % initial wealth | −0.3055% | −0.3059% |
| Ten-day loss 95th percentile, % initial wealth | 2.7375% | 2.9592% |

The upper endpoint difference is **0.2217 percentage points** of initial wealth;
predictive SD is 8.43% larger. These are model-based consequences for one
synthetic history. The central 90% loss interval is neither a validated 99%
VaR/ES result nor a Monte Carlo confidence interval. The tolerances remain
pilot engineering limits, not universal financial-usefulness tests.

The [prior/posterior plot](results/heston-diagnostic-review/prior-posterior-comparison.png)
uses means ± one SD, not credible intervals. The
[rank plot](results/heston-diagnostic-review/sbc-rank-ecdf.png) is development SBC.
The [conditional errors](results/heston-diagnostic-review/conditional-reference-errors.png)
and [refined state/forecast plot](results/heston-precision/case-0-comparison.png)
are conditional comparisons, not SBC. Notebook05 now displays these distinctions
alongside the decision and a bounded saved-evidence exercise.

### What the audit can establish

Mean/SD comparisons, coverage and rank diagnostics are recalculated from saved
validation moments, intervals and ranks. Correlation and predictive-error
maxima are reaggregated from saved per-case summaries: complete posterior and
forecast draws for those evaluations were not committed. Reference stability
is checked from its recorded summaries, not by rebuilding the particle filters.
The state-prior estimate remains the saved training-bank estimate.

The precision archive does contain joint draws and future losses for one
prespecified comparison. Its neural moments and both loss distributions are
recalculated directly. The reported reference state moments use the full
weighted particle/quadrature calculation; the finite reference draw sample has
a slightly different mean and must not be substituted silently. No new
scientific population, simulation bank, training run or acceptance evidence was
generated by this arithmetic review.

## History and conditions for reopening

The original root-43021 test already failed and remains closed regression
evidence. Notebook05 retains its four historical examples for an interactive
demonstration, including new illustrative samples and a recomputed selected
reference. That demonstration does not evaluate the later root-43022 holdout.

Root **43022** retains its 32,768 training / 2,048 validation / 256 test design.
Its test arrays were historically generated and integrity-checked; no test
reference or neural test evaluation has occurred. They remain reserved.
V3's first-stage failure and stop rule remain intact; v2 is still the better
provisional comparison, not an accepted replacement.

Reopen experimental work only if #26 identifies the need and one mechanism is
prospectively specified with task, adapter/checkpoint, bank, references, seeds,
all cases, budget, precision, metrics and stop rule. Preserve signed states,
joint draws, all failures and original thresholds. Any different use-based
criterion needs a prospective rationale. Only after a selected candidate passes
all declared validation prerequisites may its test protocol be frozen, test
references checked, and the reserved test evaluated once. No automatic sweep,
bank expansion or test-driven selection follows this decision.

## Delivery and handback

This follow-up starts at PR #11's verified head
`ba920bbd628a0913f43984edf0f2718cd755cf6f`, on
`feat/heston-validation-decision`, in a separate worktree. It follows the
[handover](https://github.com/fmuia/sbi4finance/blob/c762d7613b57d3ea2632671e773498ae8734b701/docs/heston-validation-handover.md)
and leaves the original/shared worktrees unchanged. Publish as
`1/1-simulator-followup-heston-validation`, based on
`feat/heston-joint-inference`. No merge is requested.

H25.1 is complete with this scoped decision. H25.2 is skipped by that decision;
H25.3 reviews existing evidence without executing another candidate. H25.4
updates notebook05 and verifies its fresh-process offline export. The
[execution record](results/heston-validation-decision/notebook-verification.json)
records exactly what ran and the limits of its isolation. It reuses the pinned
environment; it does not certify a clean dependency installation or #26's
eventual combined participant route.

Verification on 12 October, with artifact identities rechecked on 13 October:
**24 Heston software tests passed** (22.61 seconds); strict marimo validation
passed; the isolated export produced **39 output cells, zero errors and ten
figures** in 13.52 seconds. All ten exported figures were visually inspected.
The tests include tiny training/save/reload smoke checks, distinct from a new
scientific candidate. The optional browser layout inspection was interrupted;
the HTML session outputs and exported figures were checked directly. No clean
installation or browser-layout certification is claimed.

To reproduce the review, use the checkout's pinned Pixi environment and unused
output paths:

```bash
PYTHONPATH=src pixi run python scripts/check_heston_decision.py --out /tmp/heston-decision-new.json
PYTHONPATH=src pixi run python -m unittest discover -s tests -p 'test_heston*.py' -v
PYTHONPATH=src pixi run python scripts/check_heston_decision_review.py --out /tmp/heston-review-new
```

For the shared-doc/integration owner: carry this decision note, its reports,
the two review scripts and notebook05 into the agreed successor tree. Link
the decision from the canonical handover there; preserve its historical
experiment records and provisional label. For #26: the optional diagnostic
lesson is available, further Heston experimentation is deferred, and no
accepted Heston forecast is promised by this increment.

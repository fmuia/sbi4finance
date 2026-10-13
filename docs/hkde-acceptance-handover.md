# Agent handover: HKDE scientific acceptance, issue #24

Prepared 12 October 2026. This is a self-contained briefing for the next agent;
writing it does not start an experiment or change any acceptance decision.

Use the [parallel assignment map](parallel-agent-handover.md) when other agents
are active. This agent owns the HKDE scientific decision and notebook11 evidence;
coordinate participant presentation and data-I4 changes with their owners, and
send shared handover/status updates to the integration coordinator.

**Receiving-agent outcome, 12 October 2026:** the
[decision note](hkde-acceptance-decision.md),
[artifact-bound record](results/hkde-acceptance/decision.json) and extended
[notebook11](../notebooks/11_hkde_calibration_audit.py) retain **provisional
selected-case teaching at 2,048 proposals**. H24.1 and H24.2 are complete;
H24.3 is intentionally not run because no fresh cohort is needed for that
scope. H24.4 records the decision and fresh offline review verification.
Historical acceptance remains false, broader acceptance is deferred, and I4
stays deferred. The briefing below is preserved as the original task context.
Published for review in [PR #29](https://github.com/fmuia/sbi4finance/pull/29),
based on PR #19 at `b588df33d8fa1456c05bae67438d130d5b00ee1a`.
Issue #24 links the decision and actual verification; no PR was merged.

## Task

Work through [issue #24](https://github.com/fmuia/sbi4finance/issues/24): decide
the prospective scientific acceptance policy for the **frozen corrected HKDE
workflow at the 2,048-proposal participant budget**. Review the existing
evidence first. Specify what can be claimed now, whether any additional
evidence is necessary, and the smallest justified next step. Retaining an
explicitly provisional result is a valid completion of the decision task.

This is a scientific decision after a shared calibration alarm. Simulator,
bank, correction and provisional offline teaching delivery already exist.
The user wants solid building blocks and careful reasoning while preserving
maneuverability: no production framework or open-ended experiment campaign.
Every significant increment needs a didactic marimo review of its functionality
and actual findings. The final returns-versus-options curriculum remains open.

## Start here: repository and branch state

Repository: <https://github.com/fmuia/sbi4finance>.
Local project root: `/Users/afengler/Projects/proj_sbi4finance`.
At this handover all PRs below are open and unmerged; recheck current heads
before beginning, since the user intends to review and merge the stacks.

| Purpose | Worktree / branch | Verified head before this documentation update |
|---|---|---|
| Recommended scientific starting point; audit, live lesson and current handover | `sbi4finance-offline-tutorial` / `feat/hkde-offline-tutorial`, [PR #19](https://github.com/fmuia/sbi4finance/pull/19), draft | `06703dd493699335130882ebe4a0c1f5911b05b3` |
| HKDE implementation and original correction study | `sbi4finance-hkde-inference` / `feat/hkde-inference`, [PR #15](https://github.com/fmuia/sbi4finance/pull/15) | `60ebe4d622d6353b37b61ed2fefaa84a55066866` |
| Separate local-data participant integration and provisional demo | `sbi4finance-live-demo` / `feat/hkde-provisional-live`, [PR #22](https://github.com/fmuia/sbi4finance/pull/22) | `663ac4edeaf5935b59da5eb3d6767e13b0806037` |

Use a separate branch/worktree from the current PR #19 head for substantive
#24 work. A PR's target branch does not automatically import that target's later
commits. PR #22 uses a combined review base and has its own notebook/input
integration; do not assume it equals current #19 or edit the data agent's
worktree. [Issue #27](https://github.com/fmuia/sbi4finance/issues/27) owns stack
integration. Do not merge PRs as part of #24.

The [collaborator handover](simulator-handover.md) and
[architecture guide](code-architecture.md) explain the wider simulator ladder.
They are orientation, not a request to reopen every model's scientific gates.

## Read in this order

All relative paths below refer to the recommended PR #19 checkout.

1. [Gate audit](hkde-calibration-gate-audit.md) and
   [notebook11](../notebooks/11_hkde_calibration_audit.py): the central evidence
   and limits of the shared alarm explanation.
2. [Corrected inference](hkde-corrected-inference.md),
   [boundary protocol notes](hkde-correction-boundary-protocol.md) and
   [frozen protocol JSON](../configs/hkde_correction_boundary_protocol.json):
   original estimand, budgets, tolerances and retained stop.
3. [M8 teaching scope](m8-options-candidate.md) and
   [notebook10](../notebooks/10_hkde_corrected_workflow.py): what participants
   actually run, and the separate execution/acceptance statuses.
4. [Data integration's I4 prerequisite](https://github.com/fmuia/sbi4finance/blob/feat/hkde-provisional-live/docs/hkde-teaching-integration-plan.md#i4--formal-live-participant-acceptance-deferred):
   what a scoped positive decision must hand to the data workstream.

The key graphs are the [shared reference/SIR rank shift](results/hkde-correction/plots/reference-pit.png),
[conditional ideal-reference controls](results/hkde-gate-audit/oracle-rank-control.png),
[direct CDF errors](results/hkde-gate-audit/conditional-cdf-errors.png), and
[target-price quantile errors](results/hkde-correction/plots/price-errors.png).

## The actual inference problem

This is a restricted **risk-neutral Q** model, not returns forecasting or the
full nine-parameter HKDE calibration. The unknown pair is long-run annual
variance `theta` and annual `jump_intensity`, independently uniform on
`[0.025,0.065]` and `[0.25,1.0]`. Other parameters are fixed in the
[task JSON](../configs/tasks/hkde_inference_bank.json).

An observation is nine European put quotes in currency, ordered as three
strikes 90/100/110 within each maturity 0.25/0.5/1 year, shape `[batch,9,1]`.
Spot is 100, rate 0.02, dividend 0 and initial variance 0.04. Under the declared
numerical observation model,

\[
y_j=C_j^{\mathrm{Fourier}}(a)+0.25\epsilon_j,
\qquad \epsilon_j\overset{\mathrm{iid}}\sim N(0,1),
\qquad a=(\theta,\lambda).
\]

The saved neural flow is a **proposal**, not an accepted standalone posterior.
The correction draws from

\[
g(a\mid y)=0.8q_{\phi}(a\mid y)+0.1p(a)+0.1h(a),
\qquad
w_j\propto\frac{p(y\mid a_j)p(a_j)}{g(a_j\mid y)}.
\]

Here `h` is a product-arcsine distribution in normalized prior coordinates.
The prior in the numerator stays uniform. Every row uses the full mixture
density in the denominator, regardless of its sampled component. The numerical
likelihood evaluates all nine Fourier prices; this is not purely amortized
inference. Keep joint parameter pairs and their normalized weights intact.

An independently validated surrogate maps each pair to prices of two excluded
puts, `(K=95,T=0.75)` and `(K=80,T=1)`. These weighted price distributions measure
parameter uncertainty in Q model expectations. They are not realized payoff
distributions, tomorrow's market quotes or P-measure loss probabilities.

## What is known, and what remains undecided

| Evidence | Result and interpretation |
|---|---|
| Raw neural flow | Passed marginal SBC but failed a conditional target-price check. It remains provisional. |
| First importance correction, v1 | Conditional accuracy passed, but `stress:2` failed ESS fraction / largest-weight limits at the precision budget. Stopped before calibration or new test. Boundary undercoverage motivated one fixed proposal change. |
| Boundary correction, v2 | All 20 development conditional cases pass at both 2,048 and 8,192 proposals; 512-panel SBC passes. |
| Independent correction test, root 95026 | Conditional/reference/weight checks pass on 16 cases at both budgets; maximum target-price quantile error is 0.01125 currency at 2,048, below 0.03. Across 256 panels, theta SBC KS **0.10325 > 0.09251** fails; intensity KS 0.07203 passes. Historical acceptance remains false. |
| Numerical reference on the same 256 panels | Theta PIT KS **0.10349** at 129 nodes/axis, and 0.10340 at 65. The alarm is shared; close KS values are not a distance between conditional posteriors. |
| Post-hoc ideal-reference controls | 98.85% trigger the theta alarm **conditional on this already-selected cohort**. This is not the population false-positive rate or proof that the shared shift is harmless. |
| Direct conditional CDF audit | Importance-pool error remains measurable. At 2,048, theta/intensity CDF RMS errors are 0.00884/0.00940, versus estimated RMS SEs 0.00976/0.00980. Larger nested pools reduce error; these are 16 cases, not two independent cohorts. |
| Fixed live example, `validation:257` | All six accuracy and both weight checks pass with 2,048 proposals; maximum target-price quantile error 0.00530 currency. It is a selected development example, not a new test. |

The DKW rule itself is mathematically sound. For 256 cases, two marginal
coordinates and family alpha 0.05, its bound is
`sqrt(log(2*2/0.05)/(2*256)) = 0.092513`. The ideal-uniform one-coordinate
false-alarm probability is about 2.3395%; the two-coordinate union bound is
4.6790%. The 98.85% conditional result answers a different question.

Separate four questions throughout the decision:

- **Numerical reference/model validity:** does the reference accurately
  represent the declared numerical posterior? Independent posterior quadrature
  still shares the Fourier price engine; it is not an analytic continuum oracle.
- **Conditional approximation:** how far are this algorithm's joint posterior
  and downstream prices from that reference on a specified panel?
- **Prior-predictive calibration:** does the complete finite-pool algorithm
  behave correctly across newly simulated parameter/panel pairs? Historical
  SBC uses 2,048 proposals and 512 SIR draws per panel; shared-pool resamples
  are not IID exact posterior draws.
- **Participant delivery:** does the intended offline lesson execute using
  compatible artifacts? This already works provisionally; it does not resolve
  the first three questions or establish empirical market adequacy.

## Ordered subplan for the receiving agent

### H24.1 — Establish the claim and evidence inventory

Recheck issue/branch state; review the cited reports and notebook11. Record
which intended claim is being assessed: selected-case teaching, conditional
approximation over a specified domain, or broader prior-predictive calibration.
Bind it to the task, checkpoint, numerical likelihood, proposal, surrogate and
**2,048-proposal** configuration. Preserve the distinction from 8,192-proposal
precision evidence. Summarize what existing evidence establishes and what it
cannot establish. Do not start by rerunning the old failed experiment.

**Deliverable:** a short decision note with the remaining scientific question
and a reasoned recommendation for keeping provisional status or proceeding.

### H24.2 — Specify the prospective decision rule

Before viewing any fresh outcomes, define the relevant conditional/joint/price
tolerances, reference-precision prerequisites, Monte Carlo uncertainty
treatment and calibration decision. State what happens when the reference and
corrected algorithm share an alarm. Distinguish the approximation's practical
accuracy from any claim about general calibration. Keep historical gates and
reports unchanged; explain any new criterion by the intended use, not by making
an observed failure disappear. A new rule must not retroactively relabel old
cases as an independent acceptance cohort.

**Deliverable:** a bounded, source/artifact-bound rule and an explicit judgment
about whether fresh evidence is needed. No mandatory 1,024-panel study follows
from the audit. A documented provisional/deferred decision can proceed directly
to H24.4.

### H24.3 — Only if justified: one prospective replication

If the decision requires fresh evidence, fix its sample size, independent seeds,
conditional case selection, reference precision, proposal/resampling budgets,
metrics, aggregation and stopping rule before execution. Use one new output
directory and retain every outcome. Reuse existing functions and a thin runner;
save sufficient raw draws/weights/ranks to audit the chosen statistics.

Reference failure, calibration failure and conditional failure are distinct
outcomes. No automatic cohort expansion, seed search, proposal change, retraining
or repeat-until-pass loop. An experiment that does not support acceptance is
still a completed experiment. Record actual cost; an old reference or offline
render timing is not a new comparative speed benchmark.

**Deliverable:** the one predeclared run, a reproducible report and an independent
arithmetic/source audit appropriate to the quantities actually saved.

### H24.4 — Record and teach the decision

State accepted / provisional / reference-blocked status and its exact scope,
including exclusions. Extend notebook11 or the existing relevant lesson with
the reasoning, key graphs and outcomes; avoid a new notebook framework.
Check a fresh-process offline export, rendered figures and visible status.
Update the collaborator handover and issue #24 with result links.

If a scoped scientific acceptance is granted, hand its exact task/artifact
hashes, participant budget and criteria to the data workstream's **I4**. That
workstream still owns the actual local-input participant verification; #24 must
not silently mark it complete. Otherwise keep I4 deferred. **I4a**, the
provisional live demonstration, already works and remains distinct.

Use a clearly numbered follow-up PR, for example
`1/1-simulator-followup-hkde-acceptance` for one bounded increment, with #24,
base branch and dependencies in its body. If the stack grows, update the real
denominator rather than using `X`. Publish for review; do not merge.

## Files and commands that matter

| Layer | Existing implementation / evidence |
|---|---|
| Correction and diagnostics | `src/sbi4finance/hkde_correction.py`, `hkde_diagnostics.py`; `scripts/evaluate_hkde_correction.py` |
| Flow, numerical reference and target surrogate | `src/sbi4finance/hkde_inference.py`, `hkde_reference.py`, `hkde_surrogate.py` |
| Frozen assets | `checkpoints/hkde-theta-intensity-provisional/`; `docs/results/hkde-surrogate/model.npz` and `report.json` |
| Original correction evidence | `docs/results/hkde-correction/{validation,test}.json`, corresponding `*-checks.npz` and `*-independent-audit.json`, `new-test-panels.npz`, `test-reference-pit.npz` |
| Gate audit | `scripts/audit_hkde_calibration_gate.py`, `scripts/check_hkde_saved_inputs.py`; `docs/results/hkde-gate-audit/{report,input-consistency,verification}.json`, `checks.npz` |
| Tests | `tests/test_hkde_correction.py`, `test_hkde_correction_run.py`, `test_hkde_gate_audit.py`, plus reference/inference/surrogate tests |
| Existing PR #19 live verification | `docs/results/hkde-live-demo/verification.json` |
| Separate PR #22 entry point / verification | `src/sbi4finance/datasets/hkde_teaching.py`; `docs/results/hkde-provisional-demo/execution.json` in its own checkout |

Useful initial review commands from the recommended checkout:

```bash
PYTHONPATH=src pixi run review-hkde-calibration
PYTHONPATH=src pixi run review-hkde-corrected
PYTHONPATH=src pixi run python -m unittest discover -s tests -p 'test_hkde*.py' -v
```

For a read-only replay of the saved inputs, use
`PYTHONPATH=src pixi run python scripts/check_hkde_saved_inputs.py --repo .`.
It replays existing seeds and reprices selected saved rows, producing a
diagnostic record rather than new acceptance evidence. The old loaders bind
scientific source hashes as well as assets; preserve the historical route and
put any new experiment behind its own explicit protocol.

Notebook11 reads cached diagnostics and performs a small analytic calculation.
Notebook10 in PR #19 runs the single fixed provisional example by default;
PR #22's integrated route defaults to cached review and has an opt-in live mode.
Its `PYTHONPATH=src pixi run demo-hkde-corrected` command enables that mode in
the data checkout; do not overwrite that notebook with PR #19's variant.
Neither is a new population study. Read a script's arguments before running
it; historical `hkde-correct` commands reproduce used cohorts and must not be
advertised as fresh acceptance evidence.

The pinned environment uses Python 3.12, BayesFlow 2.0.14, Keras 3.15.1 and
JAX 0.11.1. Set `KERAS_BACKEND=jax` when invoking Python outside Pixi and keep
`PYTHONPATH=src` tied to the actual checkout. Prior verification reused an
existing pinned environment; a clean installation has not been certified.
PR #19 records 57 HKDE tests, notebook10's 17-cell/zero-error live export
(13.50 seconds total, 5.37 seconds live calculation), and notebook11's
10-cell/zero-error cached export (1.07 seconds). These are snapshot-specific
checks, not guarantees about a new combined tree or timings to add together.

## Preserve these boundaries

- Root **95026**, the corrected HKDE test, is **used and closed**. Keep its
  failed historical acceptance and original arrays immutable. Root **62026**,
  the original raw-neural test, remains untouched; do not consume it implicitly.
- Original all-cohort proposal clouds and individual SIR draws were not saved.
  Existing audits can reconstruct rank-statistic arithmetic from saved ranks,
  but cannot independently reconstruct all missing rank/coverage events.
- Do not accept the raw flow merely because its importance-corrected use is
  accurate. Do not treat ESS as proof of mode coverage or quantile accuracy.
- No new market data is required. This remains **S**, synthetic evidence.
  Real conditioning (**M**) and empirical evaluation (**E**) need their own
  compatible inputs/designs and cannot supply true latent parameter labels.
- [Rough #23 / PR #28](https://github.com/fmuia/sbi4finance/pull/28) is now
  complete: 16/16 references, six of seven neural gates, one eta-spread failure,
  and all 256 rough test cases used. Its candidate remains provisional. It does
  not resolve HKDE acceptance. Historical documents on older branches may still
  describe that test as reserved; use the current handover and measured report.
- [Curriculum #26](https://github.com/fmuia/sbi4finance/issues/26) and integration
  #27 are separate decisions. No P/Q linkage, American/exotic extension, bank
  redesign, cluster campaign or collaborator SVJ reproduction is required here.

## Completion checklist

- [x] Intended claim and 2,048-proposal configuration are explicit and bound to artifacts.
- [x] Shared reference alarm, conditional error, finite-pool noise and delivery are distinguished.
- [x] Continued-use policy is prospective; original failures remain visible. A broader acceptance protocol is explicitly deferred.
- [x] No replication is needed for the selected-case provisional decision; deferral is recorded.
- [x] Decision, limits, didactic marimo review and actual verification are published together in PR #29.
- [x] I4 stays deferred; I4a is not mislabeled acceptance.
- [x] Handover and #24 are synchronized, with no merge or unrelated implementation.

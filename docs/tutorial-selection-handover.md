# Agent handover: tutorial selection and participant review, issue #26

Prepared 12 October 2026. This is a work assignment, not a curriculum decision
or a new scientific experiment. Coordinate through the
[parallel-agent handover](parallel-agent-handover.md).

## Current handback — 12 October 2026

The user chose **Q/options: GBM foundations followed by the provisional
corrected-HKDE diagnostic lesson**, and confirmed that #27 has **not yet**
supplied a combined revision. The [participant route](tutorial-participant-route.md)
records that choice, supported claims, exact sections, modes and a 120-minute
schedule including exercises. The [delivery record](tutorial-delivery-verification.md)
separates fresh setup/current-branch preflight from final combined-tree acceptance.

- H26.1: curriculum choice and provisional teaching scope recorded.
- H26.2: route specification prepared; #27 must retain the agreed #22 local
  panel/cached-default/live-opt-in behavior and incorporate #29's explanation.
- H26.3: **pending #27's pinned combined revision**. Preliminary checks cannot
  discharge clean setup and complete offline execution on that revision.
- H26.4: canonical schedule and MMAR scope corrected; final participant evidence
  and PR closeout remain dependent on H26.3. No PR merges are part of this work.

[PR #29](https://github.com/fmuia/sbi4finance/pull/29) now records #24's decision:
retain 2,048-proposal selected-case teaching as provisional; formal I4 stays
deferred. A provisional curriculum does not need a new acceptance experiment.
The assignment below is retained as the acceptance checklist, not evidence
that its remaining items have already passed.

## Task and boundaries

Complete [issue #26](https://github.com/fmuia/sbi4finance/issues/26): help the
user/team choose the financial question, assemble the smallest honest teaching
path, and verify its participant delivery. The simulator/bank ladder exists;
the final tutorial is still open. More implementation progress on one model
does not authorize replacing the agreed financial question with another.

Prepare the comparison and pacing now. Record the user/team's eventual choice
before treating it as frozen. A deliberately provisional diagnostic lesson is
an admissible choice, provided it is never advertised as an accepted posterior.
Completion does not require every model to pass its scientific gates.

Own the curriculum decision note, presentation/pacing and schedule after
selection. Scientific agents own their inference studies and diagnostic lessons;
the data workstream owns local-input integration and HKDE I4 execution.
[Issue #27](https://github.com/fmuia/sbi4finance/issues/27) owns the combined
repository revision. Do not merge stacks, change scientific criteria, retrain,
consume reserved tests, or replace another agent's notebook wholesale.

## Read and locate the actual code

Repository: <https://github.com/fmuia/sbi4finance>. Local root:
`/Users/afengler/Projects/proj_sbi4finance`. Use a separate worktree for substantive
work. Recheck heads and issue decisions first; these are unmerged snapshots.

| Starting point | Snapshot and purpose |
|---|---|
| `sbi4finance-offline-tutorial`, `feat/hkde-offline-tutorial`, [PR #19](https://github.com/fmuia/sbi4finance/pull/19) | `b588df3` before this note: architecture/handover, Heston review, HKDE notebooks07/10/11 and provisional live candidate. |
| `sbi4finance-rough-heldout`, `feat/rough-heldout-test`, [PR #28](https://github.com/fmuia/sbi4finance/pull/28) | `f1bfa72`: current rough notebook09, completed held-out study, graphs and audit. These additions are not all in #19. |
| `sbi4finance-live-demo`, `feat/hkde-provisional-live`, [PR #22](https://github.com/fmuia/sbi4finance/pull/22) | `663ac4e`: prepared local panel, cached default and opt-in provisional live notebook10. Its combined base contains specific older simulator revisions. |

A PR targeting a branch does not automatically contain that branch's later
commits. Do not claim full participant verification from three separate checkouts.
Ask #27 for the exact combined revision and retained notebook10 behavior before
the final run; branch inspection and curriculum preparation can proceed meanwhile.

Read the [collaborator handover](simulator-handover.md),
[architecture](code-architecture.md), [milestone reviews](milestone-review-notebooks.md),
[M8 candidate](m8-options-candidate.md), and
[HKDE decision brief](hkde-acceptance-handover.md). The original README schedule
totals approximately two hours; notebook10 is only a suggested 30-minute segment.
The development review notebooks collectively are not a paced participant course.

## Financial questions and mathematics

**P means physical or real-world dynamics.** An ordered return history
`r[batch,time,1]` contains decimal log returns, with positive year-fraction time
increments as context. Infer parameters and the model's current continuation
state together, then simulate conditional futures:

\[
p(L_{T+h}\mid r_{1:T})=
\int p(L_{T+h}\mid a,z_T)\,p(a,z_T\mid r_{1:T})\,da\,dz_T.
\]

Here `a` denotes static parameters, `z_T` the state needed to restart the model,
and `L` a future loss under a declared convention. Keep paired parameter/state
draws. In our restricted Heston pilot the targets are long-run variance and the
signed auxiliary terminal state of the full-truncation discretization; replacing
that state with clipped variance changes continuation. This is not Francesco's
full SVJ/SVCJ reproduction. Rough memory needs more than a current variance.
VaR of the predictive mixture is not the average conditional VaR across draws.

**Q means risk-neutral pricing dynamics.** The observation is a panel of noisy
option premiums, `y[batch,contract,1]`, with spot, contracts, carry and model
context. Synthetic generation and inference are

\[
y_j=C_j^Q(a;c)+\epsilon_j,\qquad
p(a\mid y,c)\propto p(y\mid a,c)p(a),\qquad
C_*^Q(a;c)=e^{-rT}\mathbb E^Q[\text{payoff}_*\mid a,c].
\]

Push joint posterior draws through the target price map to describe parameter
uncertainty in model expectations. One payoff is not a price label; price
uncertainty is not a loss/payoff distribution or tomorrow's market option chain.
The numerical pricer supplies the generative map and reference. HKDE's learned
two-target surrogate replaces only supported target evaluations; importance
correction still numerically prices all nine likelihood contracts.

Q calibration cannot furnish P risk probabilities without an additional model
linking the measures. That linkage remains deferred. Rough Bergomi's Q context
includes a forward-variance curve, which can be synthetic for this tutorial.

## Evidence to inspect before proposing the route

| Candidate | Actual status and useful teaching role |
|---|---|
| GBM foundations | Exact drift posterior, prediction and separate European prices; useful analytic starting point, without certifying more complex models. |
| Heston returns | Joint inference learns beyond the prior but retains conditional/state-sensitive failures. The time-context and signed-state follow-ups remain provisional; #25 may investigate further, without being a curriculum prerequisite. |
| Corrected HKDE options | Frozen 2,048-proposal development example passes its case checks; broader historical theta SBC gate failed. #24 owns the prospective scientific decision. A validated target surrogate does not accept the posterior. |
| Rough Bergomi options | #23/#28 complete: 16/16 references and 6/7 neural gates pass, including 256-case SBC and target mean prices. One eta SD is 31.0% too small versus a 25% tolerance. All 256 test cases are used and closed; candidate remains provisional. |
| Original MMAR | Existing example preserved. Its fresh-cascade forecast is not latent-state filtering; do not promise later sections reuse its approximator when another model is chosen. |

Inspect [Heston prior/posterior updating](results/heston-diagnostic-review/prior-posterior-comparison.png)
and [conditional errors](results/heston-diagnostic-review/conditional-reference-errors.png);
[HKDE live joint posterior](results/hkde-live-demo/live-joint.png),
[target prices](results/hkde-live-demo/live-target-prices.png) and
[shared rank alarm](results/hkde-correction/plots/reference-pit.png);
[rough conditional failure](https://github.com/fmuia/sbi4finance/blob/feat/rough-heldout-test/docs/results/rough-heldout-test/plots/conditional-failure.png)
and [rough SBC](https://github.com/fmuia/sbi4finance/blob/feat/rough-heldout-test/docs/results/rough-heldout-test/plots/marginal-sbc.png).
SBC is prior-predictive calibration; it can pass while a particular conditional
posterior is too narrow. Execution, numerical precision and inference validity
are separate findings. Neither implies empirical market adequacy.

## Four ordered milestones

### H26.1 — Present and record the curriculum choice

Prepare a concise P-versus-Q comparison using the actual graphs, financial
question, prerequisites and participant output. Propose one primary example,
optional companions, learning objectives and a time budget including exercises.
Separate what participants will learn from any hoped-for scientific conclusion.
Do not require “frequentist failure,” superior tail-risk forecasts or SBI necessity.

**Done when:** the user/team choice, intended claims and provisional/accepted
teaching scope are recorded. If the choice is pending, preserve both alternatives
and finish independent preparation instead of silently selecting one.

### H26.2 — Freeze the smallest participant route

Select sections from existing reviews, with explicit live versus cached status
for each calculation. Cache expensive training and reference studies with their
provenance. Name CPU assumptions, actual checkpoint/task configuration, loading,
inference, forward calculation and export costs; do not add old timings together.
Keep exercises bounded and failure statuses visible. Coordinate notebook edits
with their owners; avoid a new teaching framework or duplicate scientific lesson.

For HKDE, #19 notebook10 runs live by default; #22 defaults to cached review and
offers `demo-hkde-corrected`. Use the integrated behavior agreed with #27/data.
Keep 2,048 live proposals distinct from 8,192-proposal precision/cache evidence.
I1–I3 and provisional I4a exist; formal I4 needs #24's scoped acceptance plus
the data owner's execution check. A provisional lesson cannot silently clear I4.

**Done when:** a compact route card binds input roles, versions, modes, supported
claims, exercise sequence and required assets to the selected presentation.

### H26.3 — Verify setup, then the full offline synthetic lesson

Use #27's pinned combined revision. Install in a **fresh** locked environment;
package access is allowed during installation. Existing reports reused a pinned
environment and therefore do not certify clean dependency installation.
Then run the complete chosen **S** route offline in a fresh process, with raw
training banks and all real-market fixtures absent, and required synthetic
assets/checkpoints available. Block external networking while permitting local
marimo kernel communication. Export HTML, inspect every graph and exercise,
and record runtime, cell errors, exact revision and remaining setup limitations.

**Done when:** clean setup and isolated execution are separately evidenced on
the actual combined tree. Fix only delivery/presentation defects in this scope;
scientific discrepancies go to their owner and retain their visible status.

### H26.4 — Publish the agreed schedule and handover

Update the canonical README/schedule and selected lesson to match the decision.
Resolve the original MMAR notebook's promise that all later sections use exactly
its approximator. Preserve unused material as optional reviews rather than
deleting it. Link exact reports/checkpoints and the participant execution record.
Report the result to #26 and the coordination handover; do not merge PRs.
Make executable or figure-affecting lesson changes before the final verification,
or repeat the affected checks afterward, so the delivered lesson is the one tested.

**Done when:** one clearly numbered follow-up PR (for example
`1/1-tutorial-followup-participant-route`) contains the chosen path, didactic
review and measured delivery outcome, with scientific limits still explicit.

## Useful commands and evidence boundaries

From the applicable checkout, inspect before executing:

```bash
pixi install --locked
PYTHONPATH=src pixi run review-foundations
PYTHONPATH=src pixi run review-heston
PYTHONPATH=src pixi run review-hkde-corrected
PYTHONPATH=src pixi run review-hkde-calibration
```

`review-rough-inference` exists in #28; `demo-hkde-corrected` exists in #22.
For the HKDE route, strict check and export commands are:

```bash
PYTHONPATH=src pixi run marimo check --strict notebooks/10_hkde_corrected_workflow.py
PYTHONPATH=src pixi run marimo export html notebooks/10_hkde_corrected_workflow.py -o /tmp/participant-review.html
```

These commands alone neither create a fresh environment nor enforce offline
isolation. Historical #19 verification reports 17 cells/zero errors/13.50 s,
including 5.37 s live; #28's cached review reports 26 cells/zero errors/10.72 s.
They are references, not promises for the combined participant route.

S needs no real observations. Optional **M** applies normally frozen weights
to a compatible real window/panel, without known latent truth or requiring future
outcomes merely to make a forecast. Optional **E** evaluates against withheld
later returns or contemporaneous excluded quotes, with leakage/dependence handled.
Many dates are needed for broader backtest claims. Missing M/E assets must skip
visibly; synthetic fallback must never be labelled real. No provider acquisition,
American/exotic extension, two-factor model or new cluster campaign is required.

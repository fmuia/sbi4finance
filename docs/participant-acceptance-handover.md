# Agent handover: conditional HKDE participant acceptance (I4)

Prepared 12 October 2026. This is an assignment brief, not authorization to
rerun scientific studies. See the [parallel-agent allocation](parallel-agent-handover.md).

## Assignment and prerequisite

Own the local-data entry point and eventual **I4 participant verification**
following [plan PR #20](https://github.com/fmuia/sbi4finance/pull/20),
[handoff PR #21](https://github.com/fmuia/sbi4finance/pull/21) and
[provisional-demo PR #22](https://github.com/fmuia/sbi4finance/pull/22).
**I1–I3 and I4a are already complete.** Review them; do not implement them again.
Formal I4 remains deferred until [issue #24](https://github.com/fmuia/sbi4finance/issues/24)
provides an explicit scientific decision covering the actual **2,048-proposal**
workflow and selected artifacts. An 8,192-proposal acceptance does not suffice.
Keeping I4 deferred is a valid outcome; no new issue or experiment follows automatically.
Before implementing I4, confirm that formal HKDE participant delivery is still
part of the selected curriculum or explicitly requested as a separate deliverable.
Scientific acceptance alone should not trigger unused integration work.

The [#24 handover](hkde-acceptance-handover.md) defines that scientific task.
[#26](https://github.com/fmuia/sbi4finance/issues/26) owns curriculum and narrative;
[#27](https://github.com/fmuia/sbi4finance/issues/27) owns the integrated repository
tree and merge review. This agent owns the data boundary and execution evidence,
not the choice of scientific thresholds or final running example.

## Branches and evidence to read

Repository: <https://github.com/fmuia/sbi4finance>. Project root:
`/Users/afengler/Projects/proj_sbi4finance`. Recheck current remote state first.
These are the inspected published/local snapshots; their PRs remain unmerged.

| PR / role | Branch | Head | Review base |
|---|---|---|---|
| #20 plan | `docs/hkde-teaching-integration-plan` | `b9b37877ca92521b287eb3bf2bd92e2ea7093215` | `feat/hkde-offline-walkthrough` |
| #21 I1–I3 | `feat/hkde-teaching-handoff` | `f385225874c875a7cfb1187554fe9f37fc88120c` | `integration/hkde-data-teaching-base` at `0ee09dd9a6b5a5333960cb800e28d1f6dbe7b13d` |
| #22 I4a / recommended starting point | `feat/hkde-provisional-live` | `663ac4edeaf5935b59da5eb3d6767e13b0806037` | `integration/hkde-live-demo-base` at `9cb6d20e0999e3888c3c906b893350e5f229ad3f` |

The #21 base combines simulator #19 at `160bccecea4e806e6efce1e422eabf6347b9fb1b`
and plan/data #20 at `b9b37877ca92521b287eb3bf2bd92e2ea7093215`. The #22 base
combines #21 at `f385225874c875a7cfb1187554fe9f37fc88120c` and simulator audit
#19 at `88202d1cb39bc511ce0c9fcc8caa681d5ee13da1`. Later parent-branch commits
are not automatically included. Coordinate any reconciliation with #27.

Inspect `sbi4finance-live-demo` read-only; start work in your own worktree from
the agreed #22/integrated revision. Do not edit another agent's checkout or copy
notebook10 wholesale from #19. **#19 defaults to its fixed live example; #22
defaults to cached review and enables provisional live execution explicitly.**

Read the [integration plan](https://github.com/fmuia/sbi4finance/blob/feat/hkde-provisional-live/docs/hkde-teaching-integration-plan.md),
[I1–I3 record](https://github.com/fmuia/sbi4finance/blob/feat/hkde-provisional-live/docs/hkde-teaching-integration.md),
and [I4a results](https://github.com/fmuia/sbi4finance/blob/feat/hkde-provisional-live/docs/hkde-provisional-demo.md).
Their older rough-Bergomi “test reserved” wording is historical: #23/PR #28
completed that separate test with a provisional result. It cannot accept HKDE.

## What participants actually infer

One synthetic panel `validation:257` contains nine ordered European put premiums
in currency: strikes 90/100/110 within maturities 0.25/0.5/1 year. The saved
Q task fixes context and all parameters except long-run variance `theta` and
annual `jump_intensity`. Its likelihood is the declared Fourier model price
plus independent Gaussian quote noise of standard deviation 0.25 currency.
The neural network supplies proposals; numerical likelihood weighting corrects
them. A validated surrogate gives prices of excluded puts `(95,.75)` and `(80,1)`.

The local loader returns raw premiums, not neural features. It verifies the
**complete saved task**, units and contract order. The lesson constructs `[9,1]`;
the owning inference adapter receives `[batch,9,1]`. Quote timestamps and other
auxiliary fields, optional synthetic truth, future outcomes and target prices
must not enter conditioning. Keep each joint parameter pair, normalized weight
and both target-price rows aligned. Weighted model-price quantiles describe
parameter uncertainty in Q expectations, not payoff draws or P risk forecasts.

Changing a quote cannot reuse the old cached posterior. Do not fabricate market
labels, interpolate a different panel into this task, clip Gaussian noisy quotes,
or silently replace missing files. The complete required path is synthetic and
offline; optional market conditioning and empirical evaluation are separate.

## Existing result and acceptance boundary

I4a executed the fixed case once with seed `3667540760` and 2,048 proposals.
All eight case checks passed: ESS fraction 0.68777, largest weight 0.002085,
maximum target-price quantile error 0.00530 currency. The local calculation took
5.06 seconds; the isolated two-notebook check took 14.60 seconds. These are one
execution's timings. Seven integration tests and strict marimo checks passed.

The reports distinguish execution and case accuracy from
`scientific_acceptance=false`. The historical corrected test failed theta SBC;
its shared reference alarm is documented, not erased. Cached comparison uses
4,096 raw draws and **8,192 correction proposals**, distinctly labelled. Neither
that cache nor the selected live validation example supplies new population evidence.

## Ordered subplan

### I4.1 — Receive a scoped decision

Read #24's published decision and record its exact task, checkpoint, proposal,
likelihood, surrogate, reference, budget and criteria, with source/artifact hashes.
Confirm that the claim covers the intended 2,048-proposal participant execution.
Confirm with #26/the user that formal HKDE delivery is still in scope.
**Done when:** the prerequisite is explicit, or I4 is documented as deferred.
Do not begin a live acceptance run while that decision remains provisional.

### I4.2 — Make the smallest necessary integration change, only if accepted

Reconcile with the agreed #27 tree; preserve the #22 local observation entry
point and provenance. Current notebook acceptance is derived from historical
validation/test flags, and the checker explicitly requires the historical gate
to stay closed. A new scoped decision therefore needs a narrow, separately
bound acceptance route; do not rewrite old failures, loosen audit hashes, or
flip those flags. Reuse the correction and weighted-summary functions.
**Done when:** incompatible artifacts/task/budget fail early; provisional and
accepted execution remain distinguishable; meaningful boundary tests pass.

### I4.3 — Verify the real participant path once

Execute the accepted 2,048-proposal local-panel workflow in a fresh isolated
process, offline with market fixtures and raw banks absent. Preserve the chosen
case/seed and approved criteria; do not retry seeds or substitute a bigger pool.
Record full elapsed time, paired weighted rows, diagnostics, hashes and failures.
**Done when:** actual outputs meet the approved checks and statuses agree with
the decision. An execution failure is retained and reported, not called acceptance.

### I4.4 — Teach and hand back

Update the existing introductory/advanced marimo sequence with the actual result,
input roles, budget, weighted uncertainty and limits. Inspect rendered figures;
verify cached and intended live behavior separately. Coordinate narrative changes
with #26 and tree/merge changes with #27. Publish one bounded follow-up PR, e.g.
`1/1-data-followup-hkde-participant-acceptance`, with exact dependencies and #24
decision links. **Done when:** decision, execution report, didactic review and
handover agree. Do not merge or relabel I4a as formal I4.

## Files and commands in the #22 checkout

- `src/sbi4finance/datasets/{local,options,hkde_teaching}.py`: local loading/task boundary.
- `data/synthetic/hkde-walkthrough/`: CSV/JSON observation and bound raw cache.
- `notebooks/{hkde_data_walkthrough,10_hkde_corrected_workflow}.py`: connected lessons.
- `scripts/check_hkde_teaching_integration.py`: isolated cached/provisional-live checker.
- `tests/test_hkde_teaching_integration.py`: seven actual-cell boundary/mode tests.
- `docs/results/hkde-provisional-demo/`: immutable execution, cached check, plots and rows.

Initial review commands, with the existing pinned Pixi environment:

```bash
PYTHONPATH=src pixi run hkde-data-walkthrough
PYTHONPATH=src pixi run review-hkde-corrected
PYTHONPATH=src pixi run python -m unittest discover -s tests -p 'test_hkde_teaching*.py' -v
PYTHONPATH=src pixi run hkde-teaching-integration-check --out /tmp/hkde-cached-new.json
```

The historical opt-in `pixi run demo-hkde-corrected` and checker `--live` execute
**I4a**, not a newly accepted I4. Do not run them merely to await #24. Output
paths must be new. Use `PYTHONPATH=src` for the actual checkout; Pixi sets
`KERAS_BACKEND=jax`. Existing verification reused a pinned environment and does
not certify a clean installation. No provider, training bank, retraining campaign,
new adapter framework or fresh scientific cohort belongs to this assignment.

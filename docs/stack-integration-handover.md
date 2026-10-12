# Agent handover: integrate and review the PR stacks, issue #27

Prepared 12 October 2026. See the [parallel assignment map](parallel-agent-handover.md)
for the other agents and ownership boundaries. This brief authorizes no GitHub
merge and starts no experiment.

## Receiving integration checkpoint

See [the combined review](stack-integration-review.md) for the refreshed full
head table, composition choices and exact-revision verification. This original
brief remains the task context; its open-PR table is the preparation snapshot.
The participant follow-up selected provisional Q/HKDE, with #22 local-panel
cached/default and opt-in live modes plus #29's decision. GitHub merges remain
the user's decision. The [final #26 delivery record](tutorial-delivery-verification.md)
now verifies fresh setup and five isolated exports on source `ca63424`;
its evidence commit `8f565121d23b5b3e815be0e2bfde468db8e30211` is incorporated
without changing the verified notebooks or scientific implementation. Browser
layout, interaction and offline runtime remain explicitly unverified. The
software-composition checklist below is complete; those browser checks are a
separate remaining participant-delivery boundary.

## Task and context

Work through [issue #27](https://github.com/fmuia/sbi4finance/issues/27): compose
the completed simulator and data work into one reviewable, tested revision.
Inspect dependencies, resolve routine composition conflicts in an isolated
checkout, verify the actual combined code and lessons, and prepare a clear
integration/merge recommendation. The user will review and decide merges.

Most open PRs contain implemented work; do not assign a new implementation
agent to every old PR or replay every experiment. This agent owns the combined
tree, dependency map, shared entry points and final composition record. Other
agents own model-specific scientific decisions and curriculum content.

The code is deliberately simple: Python functions, explicit JSON tasks,
NPZ banks, frozen checkpoints, small runners and marimo lessons. Its data path
is task -> simulation/pricing -> bank -> task-specific training -> joint
inference -> P futures or Q target prices. The local data work supplies
observation/context/auxiliary roles and adapters to the same task boundary.
It is not a second simulator stack or a new bank framework.

Integrating software does **not** upgrade scientific acceptance. In particular,
the corrected HKDE pipeline, restricted Heston posterior and rough posterior
remain provisional at the recorded heads. Their simulators/banks can still be
reviewed and integrated.

## Starting state and required reading

Repository: <https://github.com/fmuia/sbi4finance>.
Local root: `/Users/afengler/Projects/proj_sbi4finance`.
Remote `main` was `b892fae382b29ac3350246e3be2bc9946e60437f` when checked.
It is the early scaffold, not the accumulated implementation.

Read the [collaborator handover](simulator-handover.md),
[architecture guide](code-architecture.md),
[simulator plan PR #3](https://github.com/fmuia/sbi4finance/pull/3),
[data plan PR #4](https://github.com/fmuia/sbi4finance/pull/4), and the
[data teaching composition record](https://github.com/fmuia/sbi4finance/blob/feat/hkde-provisional-live/docs/hkde-teaching-integration.md).
Then inspect actual branches and changes. A PR's base branch can advance
without those later commits appearing in the PR's head ancestry.

Use a dedicated integration worktree from an explicitly recorded current base
(normally freshly fetched `origin/main`), never another agent's worktree.
Record the selected heads before composing them; refresh this snapshot if the
user has merged or another agent has published since preparation.

### Open-PR snapshot

All listed PRs were open. Only #19 was draft. The N/10 title order is not Git
dependency order. Short hashes are for orientation; save full hashes in the
actual integration record.

| PR | Branch | Head at snapshot | Review base |
|---|---|---|---|
| [#3](https://github.com/fmuia/sbi4finance/pull/3) | `docs/simulator-plan` | `30982c4` | `main` |
| [#4](https://github.com/fmuia/sbi4finance/pull/4) | `docs/data-loading-plan` | `e630104` | `main` |
| [#5](https://github.com/fmuia/sbi4finance/pull/5) | `feat/local-data-artifacts` | `3640c08` | `main` |
| [#6](https://github.com/fmuia/sbi4finance/pull/6) | `feat/return-window-preparation` | `e21898b` | #5 branch |
| [#7](https://github.com/fmuia/sbi4finance/pull/7) | `feat/gbm-reference` | `77760f4` | #3 branch |
| [#8](https://github.com/fmuia/sbi4finance/pull/8) | `feat/gbm-bank-training` | `94a7907` | #7 branch |
| [#9](https://github.com/fmuia/sbi4finance/pull/9) | `feat/gbm-cluster-smoke` | `a4444ee` | #8 branch |
| [#10](https://github.com/fmuia/sbi4finance/pull/10) | `feat/heston-svj-kernels` | `e4b9cbc` | #9 branch |
| [#11](https://github.com/fmuia/sbi4finance/pull/11) | `feat/heston-joint-inference` | `ba920bb` | #10 branch |
| [#12](https://github.com/fmuia/sbi4finance/pull/12) | `feat/hkde-option-bank` | `16b33dd` | #11 branch |
| [#13](https://github.com/fmuia/sbi4finance/pull/13) | `feat/bergomi-option-bank` | `1e49535` | #12 branch |
| [#14](https://github.com/fmuia/sbi4finance/pull/14) | `feat/rough-bergomi-option-bank` | `dfaaee4` | #13 branch |
| [#15](https://github.com/fmuia/sbi4finance/pull/15) | `feat/hkde-inference` | `60ebe4d` | #14 branch |
| [#16](https://github.com/fmuia/sbi4finance/pull/16) | `feat/hkde-panel-handoff` | `3b94743` | #6 branch |
| [#17](https://github.com/fmuia/sbi4finance/pull/17) | `feat/hkde-bank-integration` | `2d46195` | #16 branch |
| [#18](https://github.com/fmuia/sbi4finance/pull/18) | `feat/hkde-offline-walkthrough` | `e5a847b` | #17 branch |
| [#19](https://github.com/fmuia/sbi4finance/pull/19) (draft) | `feat/hkde-offline-tutorial` | `b588df3` | #15 branch |
| [#20](https://github.com/fmuia/sbi4finance/pull/20) | `docs/hkde-teaching-integration-plan` | `b9b3787` | #18 branch |
| [#21](https://github.com/fmuia/sbi4finance/pull/21) | `feat/hkde-teaching-handoff` | `f385225` | `integration/hkde-data-teaching-base` |
| [#22](https://github.com/fmuia/sbi4finance/pull/22) | `feat/hkde-provisional-live` | `663ac4e` | `integration/hkde-live-demo-base` |
| [#28](https://github.com/fmuia/sbi4finance/pull/28) | `feat/rough-heldout-test` | `f1bfa72` | #14 branch |

Special combined review bases:

- #21's base `integration/hkde-data-teaching-base` is
  `0ee09dd9a6b5a5333960cb800e28d1f6dbe7b13d`, combining older #19
  `160bccecea4e806e6efce1e422eabf6347b9fb1b` and #20
  `b9b37877ca92521b287eb3bf2bd92e2ea7093215` (including data #18).
- #22's base `integration/hkde-live-demo-base` is
  `9cb6d20e0999e3888c3c906b893350e5f229ad3f`, combining #21
  `f385225874c875a7cfb1187554fe9f37fc88120c` and older #19
  `88202d1cb39bc511ce0c9fcc8caa681d5ee13da1`.
- #28 is a separate child of #14, not another original N/10 increment.
  Its evaluation/lesson code is absent from #19; handover links are not imports.
- The plan-only branches also received later documentation updates. Their
  current heads are not automatically present in descendant implementations.

## Ordered subplan

### G27.1 — Inventory and inspect before composing

Fetch/read the current PR state. Record full base/head hashes, actual ancestry,
unique patches, and existing test/result records. Read each PR against its
proper review base; do not equate a large stacked diff with unique changes.
Identify what is already represented by #21/#22's combined bases. Include the
current #3/#4 plans, #14/#28 rough work and #19 handover/audit/live additions.

Inspect #28 as completed scientific work: its one retained conditional failure
is part of the result, not a code defect to repair by relaxing a test. Resolve
#19's draft disposition as a review recommendation; do not silently mark it
scientifically accepted. Note the heads of any new #24–#26 follow-up PRs and
decide explicitly whether they belong in this integration batch.

**Deliverable:** one compact dependency/head table and a composition sequence.
No registry, release framework or elaborate new tracking system is needed.

### G27.2 — Build one isolated combined tree

Compose the selected changes in a dedicated local branch. Local merges or
cherry-picks used for review are distinct from merging GitHub PRs/main. Inspect
ancestry first; do not replay the same ancestor patches once per stack leaf.
Resolve conflicts semantically and record meaningful choices. Do not rewrite,
force-push or rebase collaborators' published branches.

Likely shared areas need special attention:

- `pyproject.toml` / `pixi.lock`: preserve commands and compatible pinned
  dependencies from both streams. Regenerate a lock only if composition actually
  requires it; avoid unrelated dependency upgrades.
- `notebooks/10_hkde_corrected_workflow.py`: current #19 defaults to a fixed
  provisional live run; #22 adds the local observation path, cached default
  and opt-in provisional demo. Preserve the data boundary and explicit modes;
  coordinate the participant default with #26 rather than taking one whole
  file and discarding the other's functionality.
- README, plans, notebook guide and handover: retain current scientific
  statuses and both P/Q paths until the curriculum decision is made.
- `notebooks/09_rough_inference.py` and rough scripts/results: take account of
  #28's completed held-out outcome, not only #14's passed development study.
- Source-bound checks: distinguish genuine incompatibility from historical
  provenance. Do not rewrite old report hashes or booleans to fit a new tree.
  Record fresh combined execution evidence separately and preserve original
  assets/results. A necessary scientific source change belongs with its owner.

**Deliverable:** a reviewable combined commit/tree, conflict notes and a clean
diff against the intended integration base. Do not delete unmerged branches.

### G27.3 — Verify this actual revision

Run the combined simulator/data tests in the pinned environment. Existing
small training/reload tests may execute temporary smoke models; they must not
modify published checkpoints or start full studies. Exercise the intended
cached and provisional-live local-panel routes and the relevant marimo reviews.
Check observation/truth separation, task/checkpoint bindings, displayed statuses
and plots. Default lessons must not launch bank generation, training campaigns,
reserved-test evaluation or cluster jobs.

Use isolated offline runs with market fixtures absent. A cached review may omit
raw banks/checkpoints when supported; the live route needs its actual checkpoint
and surrogate. Block external connections while allowing required local kernel
IPC. Record the exact source revision, environment, commands, results and timing.
Inspect figures and cell errors, not only command exit codes.

Useful commands after verifying they exist in the composed task list:

```bash
PYTHONPATH=src pixi run test-simulators
PYTHONPATH=src pixi run review-hkde-calibration
PYTHONPATH=src pixi run review-rough-inference
PYTHONPATH=src pixi run hkde-teaching-integration-check \
  --out /tmp/new-combined-cached.json --artifacts /tmp/new-combined-cached-plots
PYTHONPATH=src pixi run hkde-teaching-integration-check --live \
  --out /tmp/new-combined-live.json --artifacts /tmp/new-combined-live-plots
```

Use fresh output locations. Read the checker first: current #22 verifies an
explicitly **provisional** result and asserts scientific acceptance is false.
Do not change that historical check into an accepted run by flipping a flag.
Any later formal I4 route is a separate scoped input from the I4 owner.

Earlier counts (125, 129, 133, 168, 171) describe different revisions and are
neither additive nor a certificate for this tree. The historical
[171-test record](results/hkde-correction/combined-stack-verification.json)
used older #14/#19 heads and did not include the full current data integration.
A clean participant installation and pacing check is owned by #26; share one
exact combined revision so those checks are not needlessly duplicated.

**Deliverable:** tested combined commit, actual execution report and didactic
review evidence. Fix integration defects; do not retune scientific gates.

### G27.4 — Hand back the review and integration decision

Publish a clearly scoped follow-up PR or update an agreed integration branch,
for example `1/1-integration-plan-combined-review`, with the exact source heads,
unique composition changes, tests and unresolved decisions. Keep the description
about the final tree. Update the canonical dependency/architecture/handover
pages from the other agents' reports at this integration checkpoint.

Give #26 the pinned revision for clean setup and final participant review.
Give #24/#25 the combined implementation context without implying their
scientific issues are resolved. Record what is ready for the user's merge
review and any deliberately deferred PR. **Do not merge GitHub PRs or main**
without the user's subsequent instruction.

## Scientific and data invariants

| Area | Must remain visible after integration |
|---|---|
| HKDE | Corrected root-95026 test used/failed; raw root-62026 test untouched. Shared reference alarm under #24. Raw flow provisional. Formal I4 deferred unless a separately scoped decision arrives. |
| Heston | v2/v3 conditional discrepancies retained; root-43022 test untouched. Signed auxiliary continuation state preserved. #25 optional. |
| Rough | #28 test completed; 16/16 references and 6/7 neural gates pass, eta SD failure retained. All 256 test cases used and closed. No automatic retraining. |
| Data roles | Observations/context are conditioning; synthetic truths, auxiliary diagnostics and future outcomes are not. No fabricated real latent labels. |
| Evidence | S synthetic workflow is complete without market acquisition. Optional M conditioning and E empirical evaluation remain separate. Q pricing does not furnish P risk probabilities. |

## Ownership and acceptance

This agent coordinates shared README/task/lock/status composition and the
integration record. #24 owns HKDE scientific evidence, #25 Heston evidence,
#26 learning objectives/pacing, and the conditional I4 agent the accepted local
participant route. Ask each for a commit, exact files, tests and status delta;
consume those patches instead of editing their working branches. Scientific
acceptance and a curriculum choice are not blanket prerequisites to assembling
the existing provisional implementation.

- [x] Current full heads and actual ancestry recorded, including combined review bases.
- [x] One isolated combined tree preserves existing implementations and immutable results.
- [x] Conflicts are resolved deliberately; no duplicate replay or collaborator branch rewrite.
- [x] Relevant tests and offline marimo modes pass on that exact tree, or failures are explicit.
- [x] Source-bound scientific statuses and cohort histories are preserved.
- [x] #26 receives the pinned revision; review/merge recommendation and docs are synchronized.
- [x] No new scientific study, production framework or unauthorized GitHub merge occurred.

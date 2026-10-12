# Combined simulator and data integration review

This is the composition record for [issue #27](https://github.com/fmuia/sbi4finance/issues/27).
Published for review in [PR #31](https://github.com/fmuia/sbi4finance/pull/31).
The review branch is `integration/issue-27-combined-review`, created in its own
worktree from freshly fetched `origin/main` at
`b892fae382b29ac3350246e3be2bc9946e60437f` on 12 October 2026.
No published branch is rewritten and no GitHub PR or main branch is merged.

The user selected the **Q/options provisional HKDE** participant route through
issue #26. The composed route uses the local CSV/JSON panel, cached default,
explicit 2,048-proposal demo and #29's provisional decision lesson. Formal I4
scientific acceptance remains deferred. P returns and the broader simulator
catalog remain available as separate examples.

## Selected heads and actual dependencies

All PRs in this snapshot are included as implemented work and preserved evidence;
#19 remains draft upstream. The merge recommendation concerns software and
provisional teaching material, not scientific acceptance of the posteriors.
[The machine-readable snapshot](results/stack-integration/heads.json) records
full hashes, unique patch sizes, GitHub's reported base OIDs, current named-base
OIDs and actual merge bases. GitHub's reported base OID for an older PR can lag
the current named branch; the table uses freshly fetched named refs.

| PR | Selected head | Review base branch | Current base head | Base in head ancestry? |
|---|---|---|---|---|
| [#3](https://github.com/fmuia/sbi4finance/pull/3) | `30982c4fab254420d167d10c5053552418fe379e` | `main` | `b892fae382b29ac3350246e3be2bc9946e60437f` | yes |
| [#4](https://github.com/fmuia/sbi4finance/pull/4) | `e63010475fc66834a71deca97370ea3446abcf1d` | `main` | `b892fae382b29ac3350246e3be2bc9946e60437f` | yes |
| [#5](https://github.com/fmuia/sbi4finance/pull/5) | `3640c085b2f7cee884b465c59b8fb767992d22f2` | `main` | `b892fae382b29ac3350246e3be2bc9946e60437f` | yes |
| [#6](https://github.com/fmuia/sbi4finance/pull/6) | `e21898b272669c8db584f5e35ce71e5d12696547` | `feat/local-data-artifacts` | `3640c085b2f7cee884b465c59b8fb767992d22f2` | yes |
| [#7](https://github.com/fmuia/sbi4finance/pull/7) | `77760f44cb41df2ddb18b8c84ecbe9f28d762e61` | `docs/simulator-plan` | `30982c4fab254420d167d10c5053552418fe379e` | **no** |
| [#8](https://github.com/fmuia/sbi4finance/pull/8) | `94a790789676601a4ff8ab460eefc8dcf437cc62` | `feat/gbm-reference` | `77760f44cb41df2ddb18b8c84ecbe9f28d762e61` | yes |
| [#9](https://github.com/fmuia/sbi4finance/pull/9) | `a4444ee0f48a76d50e086e74f403dc0420c90284` | `feat/gbm-bank-training` | `94a790789676601a4ff8ab460eefc8dcf437cc62` | yes |
| [#10](https://github.com/fmuia/sbi4finance/pull/10) | `e4b9cbc8408492f60d3ed2a2691aec11cf753624` | `feat/gbm-cluster-smoke` | `a4444ee0f48a76d50e086e74f403dc0420c90284` | yes |
| [#11](https://github.com/fmuia/sbi4finance/pull/11) | `ba920bbd628a0913f43984edf0f2718cd755cf6f` | `feat/heston-svj-kernels` | `e4b9cbc8408492f60d3ed2a2691aec11cf753624` | yes |
| [#12](https://github.com/fmuia/sbi4finance/pull/12) | `16b33dd41a9be830cf6ad5c574a8efcbf33b7924` | `feat/heston-joint-inference` | `ba920bbd628a0913f43984edf0f2718cd755cf6f` | yes |
| [#13](https://github.com/fmuia/sbi4finance/pull/13) | `1e49535c01fc772c155ee911c30eacfc3640bde8` | `feat/hkde-option-bank` | `16b33dd41a9be830cf6ad5c574a8efcbf33b7924` | yes |
| [#14](https://github.com/fmuia/sbi4finance/pull/14) | `dfaaee4db9d50e23dc05e1c049554a36c39b9bfc` | `feat/bergomi-option-bank` | `1e49535c01fc772c155ee911c30eacfc3640bde8` | yes |
| [#15](https://github.com/fmuia/sbi4finance/pull/15) | `60ebe4d622d6353b37b61ed2fefaa84a55066866` | `feat/rough-bergomi-option-bank` | `dfaaee4db9d50e23dc05e1c049554a36c39b9bfc` | **no** |
| [#16](https://github.com/fmuia/sbi4finance/pull/16) | `3b94743625b87ca3d06fbb41a4dfaad951236fe0` | `feat/return-window-preparation` | `e21898b272669c8db584f5e35ce71e5d12696547` | yes |
| [#17](https://github.com/fmuia/sbi4finance/pull/17) | `2d46195356b74b3c60bdc158ce14a314fb412297` | `feat/hkde-panel-handoff` | `3b94743625b87ca3d06fbb41a4dfaad951236fe0` | yes |
| [#18](https://github.com/fmuia/sbi4finance/pull/18) | `e5a847bb02ff77a927663344aeeaa332691f840d` | `feat/hkde-bank-integration` | `2d46195356b74b3c60bdc158ce14a314fb412297` | yes |
| [#19](https://github.com/fmuia/sbi4finance/pull/19) | `c762d7613b57d3ea2632671e773498ae8734b701` | `feat/hkde-inference` | `60ebe4d622d6353b37b61ed2fefaa84a55066866` | yes |
| [#20](https://github.com/fmuia/sbi4finance/pull/20) | `b9b37877ca92521b287eb3bf2bd92e2ea7093215` | `feat/hkde-offline-walkthrough` | `e5a847bb02ff77a927663344aeeaa332691f840d` | yes |
| [#21](https://github.com/fmuia/sbi4finance/pull/21) | `f385225874c875a7cfb1187554fe9f37fc88120c` | `integration/hkde-data-teaching-base` | `0ee09dd9a6b5a5333960cb800e28d1f6dbe7b13d` | yes |
| [#22](https://github.com/fmuia/sbi4finance/pull/22) | `663ac4edeaf5935b59da5eb3d6767e13b0806037` | `integration/hkde-live-demo-base` | `9cb6d20e0999e3888c3c906b893350e5f229ad3f` | yes |
| [#28](https://github.com/fmuia/sbi4finance/pull/28) | `f1bfa7256cc1ea9e5798e49a95a8befc409fa84e` | `feat/rough-bergomi-option-bank` | `dfaaee4db9d50e23dc05e1c049554a36c39b9bfc` | yes |
| [#29](https://github.com/fmuia/sbi4finance/pull/29) | `45bb08d2b091df7bae49fa250424e122f19e3eca` | `feat/hkde-offline-tutorial` | `c762d7613b57d3ea2632671e773498ae8734b701` | **no** |

[PR #30](https://github.com/fmuia/sbi4finance/pull/30) was published during this
composition: participant head `be4a3baf6d5753337f93c1b95510310aea2d3d69`,
based on #19 `c762d7613b57d3ea2632671e773498ae8734b701`. It is included;
its preliminary delivery record remains historical pending the final #26 check.

The #7, #15 and #29 heads omit later commits on their current named bases.
Neither title numbering nor the displayed base name certifies ancestry.
#21's combined review base `0ee09dd9a6b5a5333960cb800e28d1f6dbe7b13d`
contains #19 `160bccecea4e806e6efce1e422eabf6347b9fb1b` and #20
`b9b37877ca92521b287eb3bf2bd92e2ea7093215`. #22's base
`9cb6d20e0999e3888c3c906b893350e5f229ad3f` contains #21
`f385225874c875a7cfb1187554fe9f37fc88120c` and #19
`88202d1cb39bc511ce0c9fcc8caa681d5ee13da1`.

## Composition and conflict choices

The sequence starts with #22, which already contains the earlier simulator,
data and teaching stack, then merges current #19, #28 (including current #14),
#3, #4 and #29. Ancestor PRs are not replayed individually. The participant
follow-up from #26 is applied after that source composition.

- Notebook10 retains #22's full task/checkpoint match, exact local observation
  identity, observation/truth separation, weighted cached comparison, explicit
  mode selection and PNG rendering. #19 contributes clearer historical-test
  wording, per-metric thresholds and weight diagnostics. The default is cached;
  `demo-hkde-corrected` opts into the fixed provisional calculation.
- The Pixi task list contains both data and rough commands. Package constraints
  and `pixi.lock` are preserved; no dependency upgrade is needed.
- Plan, README, architecture and lesson guidance retain the current rough
  held-out outcome, HKDE decision and Heston restrictions. Historical reports,
  checkpoint bytes and scientific source hashes are not rewritten.
- PR #29 is included because it records the existing provisional decision and
  teaches its limits without changing models or launching a new study. Its
  frozen teaching-file hashes are historical provenance. Combined execution
  is recorded separately while all scientific bindings remain exact.

## Scientific boundaries

| Area | Retained outcome |
|---|---|
| HKDE | Corrected root-95026 test failed theta SBC and is used/closed. Raw root-62026 test remains untouched. Raw flow and correction remain provisional. #29 retains the 2,048-proposal scope; I4 deferred. |
| Heston | v2/v3 conditional failures remain; signed auxiliary continuation state preserved; root-43022 untouched. Optional #25 work is not an integration prerequisite. |
| Rough | 16/16 test references and 6/7 neural gates pass; eta SD error 0.31013 exceeds 0.25. All 256 test cases used/closed. No retraining or gate relaxation. |
| Data | Only observations/context condition inference. Synthetic truth, auxiliary diagnostics and future outcomes stay separate; no real latent labels are invented. |
| Claims | Synthetic Q teaching needs no market data and does not supply P risk probabilities or empirical performance evidence. |

## Execution on the pinned combined revision

Tested commit: `ca63424a140c2e512db97e40f0bb30010da3956a`.
Tree: `0eaff7f14ae721efc74de6bc8784b7fab44f67a9`.
Every execution below started from that clean combined tree. Later review
commits add these records and reconcile documentation; they do not change the
scientific implementation or notebook execution verified here.

| Check | Actual result | Fresh evidence |
|---|---|---|
| Combined simulator/data suite | **225 tests pass**, 37.114 s within unittest (38.652 s process) | [Environment/command](results/stack-integration/tests.json), [log](results/stack-integration/tests.log) |
| Strict marimo | GBM01, local-data, rough09, HKDE10, HKDE11 and changed optional MMAR pass | [Static checks](results/stack-integration/static-checks.json) |
| Local-panel cached route | 8 + 20 output cells, **2 + 4 plots**, 9.255 s; no neural import or live inference | [Report](results/stack-integration/hkde-cached.json) |
| Local-panel provisional live route | 8 + 20 output cells, **2 + 6 plots**, 14.064 s; all eight fixed-case checks pass; scientific acceptance **false** | [Report](results/stack-integration/hkde-live.json), [joint](results/stack-integration/hkde-plots/10_hkde_corrected_workflow-1.png), [target prices](results/stack-integration/hkde-plots/10_hkde_corrected_workflow-2.png) |
| HKDE decision lesson | **15 cells, 0 errors, 5 figures**, 1.087 s export; provisional / 2,048 / I4 deferred visible | [Verification](results/stack-integration/hkde-decision/verification.json), [visual inspection](results/stack-integration/hkde-decision/visual-inspection.json) |
| Rough cached review | **26 cells, 0 errors, 14 figures**, 10.665 s export; all 256 cases used and eta failure visible | [Verification](results/stack-integration/rough/execution.json), [reproduction helper](results/stack-integration/rough/verify_review.py) |
| Preserved input assets | **286 inherited source/config/checkpoint/result files** match selected input blobs; none omitted; every selected head is an ancestor; lock unchanged | [Preservation check](results/stack-integration/preservation.json) |

All fresh graphs were inspected, including the retained failures, joint
parameter dependence and two currency-price distributions. The cached plots
match the corresponding live-route images byte-for-byte; the
[visual record](results/stack-integration/visual-review.json) records the map
and figure hashes. Output-cell counts from `app.run()` differ from complete
export session-cell counts; each report states which it measured.

Execution used the existing pinned macOS arm64 environment: Python 3.12.15,
BayesFlow 2.0.14, Keras 3.15.1, JAX/jaxlib 0.11.1, NumPy 2.5.3, SciPy 1.18.1,
Numba 0.68.0 and marimo 0.25.1. The local `.pixi` symlink reuses an environment
whose editable installation points at another worktree, so every run sets
`PYTHONPATH=src` and verifies the imported package belongs to this checkout or
its isolated copy. No package/lock update occurred. This environment reuse is
separate from #26's fresh-install check.

For an ordinary checkout with its locked environment installed:

```bash
PYTHONPATH=src pixi run test-simulators
PYTHONPATH=src pixi run hkde-teaching-integration-check \
  --out /tmp/new-combined-cached.json --artifacts /tmp/new-combined-cached-plots
PYTHONPATH=src pixi run hkde-teaching-integration-check --live \
  --out /tmp/new-combined-live.json --artifacts /tmp/new-combined-live-plots
PYTHONPATH=src pixi run python scripts/check_hkde_acceptance_review.py \
  --allow-integrated-teaching --out /tmp/new-combined-decision
```

For this shared environment, execution invoked `.pixi/envs/default/bin/python`
directly with `PYTHONPATH=src`, `KERAS_BACKEND=jax` and `MPLBACKEND=Agg`; the JSON
and logs preserve the actual commands and timings. The suite includes existing
tiny temporary training/reload tests; published checkpoints are unchanged.
The notebook checks generate no banks, evaluate no reserved cohort and submit
no cluster job.

### Isolation, provenance and delivery limits

The teaching checker now copies explicit bound assets and scans the isolated
tree: market fixtures, raw banks (including banks below `docs/results`) and
unused checkpoints are absent. Cached mode blocks BayesFlow imports; live mode
loads the exact bound checkpoint/surrogate. Observation equality, full task
matching, weighted quantiles, proposal pairing and scientific flags are checked.
Its output records actual commit/tree/dirty status rather than historical base
constants. Two regressions cover accidental bank copying.

The decision checker retains its strict default. Its explicit
`--allow-integrated-teaching` option verifies **126 current exact bindings** and
**two historical teaching bindings** against the unchanged manifest's frozen
local Git commit. Only notebook10 presentation and the Pixi task table can
be reconciled; dependency/environment settings and all scientific inputs remain
exact. Five tests reject scientific or lock changes, altered dependencies,
unverifiable historical bytes and implicit reconciliation. This option does
not certify replacement notebook execution; the separate cached/live reports do.

Python socket/DNS guards block external connections. The export checks allow
local kernel IPC and block neural-framework imports for cached reviews. The
macOS sandbox initially denied their local listener, so the recorded successful
exports used local IPC permission with the external Python guards still active.
These are explicit Python guards, not a claim of an OS-wide network sandbox.
No raw banks, checkpoints or market fixtures are available in the rough/decision
review copies. The live HKDE copy includes its one necessary checkpoint.

**Standalone offline HTML is not certified by these Python checks.** Marimo
exports may reference CDN browser assets and temporary absolute review links.
The supported participant entry is the installed local marimo route; #26's
[delivery record](tutorial-delivery-verification.md) assesses the four-notebook
route, clean setup and browser limitations against the same source pin.
The pin was sent to #26 immediately after composition, along with the mode and
scope contract. Historical generated SVG whitespace is retained rather than
changing source-bound figure bytes; non-result paths pass `git diff --check`.

## Review and merge recommendation

The combined implementation is ready for the user's software/provisional-teaching
review. All original open stack heads plus #28, #29 and participant commit
`be4a3baf6d5753337f93c1b95510310aea2d3d69` are represented. The original PRs'
focused diffs remain useful for reviewing each contribution; the combined PR
shows their actual composed behavior.

- **#19 disposition:** retain its upstream draft state as the historical M8
  participant-release candidate. Its implemented lesson and handover changes
  are included here for explicitly provisional delivery. Do not mark it
  scientifically accepted to make the software mergeable.
- **#29:** include the provisional decision and notebook11. #24's broader
  calibration question and formal I4 remain deferred; integration resolves
  neither by changing a Boolean.
- **#25:** no new follow-up PR was published at the input snapshot. Heston
  scientific investigation is deliberately outside this batch; existing
  implementation and retained failures are included. The selected options
  route needs no new Heston study.
- **#26:** include the selected two-hour curriculum and its scoped follow-up;
  final clean participant evidence is recorded separately. Optional M market
  acquisition, E empirical evaluation, new cohorts, training campaigns and
  formal I4 are outside this integration.

A merge commit of the combined branch would retain the recorded stack ancestry
and make the review history easiest to follow. The maintainer decides whether
to merge this branch or apply the documented dependency order. Do not squash or
rebase collaborators' branches as an automatic closeout step. No GitHub PR,
main branch or scientific follow-up issue is closed or merged by this work.

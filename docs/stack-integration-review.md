# Combined simulator and data integration review

This is the composition record for [issue #27](https://github.com/fmuia/sbi4finance/issues/27).
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

## Execution evidence and disposition

Execution evidence will be added for the fixed combined source revision before
publication. Earlier test counts and historical notebook timings do not certify
this combined tree. The #26 agent owns fresh installation, pacing and final
participant review against the pinned combined revision.

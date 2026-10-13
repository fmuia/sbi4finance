# Main merge review — 13 October 2026

The user requested merging our completed work onto `main` and checking
Francesco's new contribution. This supplements the earlier
[combined review](stack-integration-review.md); scientific acceptance and the
selected provisional Q/options teaching route remain unchanged.

## Composition

PR #31 at `1d88759e401b172ab1b88cd66bb5899631fb9abf` already composed the
simulator, data, HKDE decision and participant route. The final candidate adds
[PR #32](https://github.com/fmuia/sbi4finance/pull/32): Heston's completed decision,
audit/review scripts, evidence and updated notebook05. It retains the optional
diagnostic lesson and defers further experiments. There were no merge conflicts
or changes to scientific kernels, checkpoints, old results, priors or thresholds.

PR #30's final evidence commit was previously represented by a patch-equivalent
cherry-pick. A normal merge now retains its original ancestry too, changing no
tree content. All **25** of our open PR heads at the preparation snapshot are
ancestors of the candidate, including #31 itself. The
[composition record](results/main-closeout/composition.json) lists their exact
revisions and verifies that all six PR #32 paths match their contributed bytes.
Independent preservation review found no missing or unexpectedly altered
protected input asset.

Use a GitHub **merge commit** for [PR #31](https://github.com/fmuia/sbi4finance/pull/31)
to preserve this history. Prior main was `b892fae382b29ac3350246e3be2bc9946e60437f`;
GitHub's PR record supplies the resulting merge SHA. Keep historical branches
and source pins. Verify original PR dispositions afterward rather than
replaying the stacks or squashing their evidence history.

## Fresh verification

Verified executable source: `3f651df4d6bc23b3b892ff33512ad0442e4fe2e8`, tree
`1cb941efe646d70b020da2afcdc9a1aacc03dd05`. The PR #30 ancestry merge has the
identical tree; subsequent closeout changes are documentation/results only.

| Check | Result |
|---|---|
| Complete simulator/data suite | **225 tests pass**, 37.393 s unittest / 39.452 s process |
| Heston saved-evidence arithmetic audit | Passed; scientific acceptance remains false |
| Updated notebook05 strict marimo | Passed |
| Notebook05 isolated offline export | **39 cells, zero errors, ten figures**, 13.015 s export |
| Visual review | All ten figures inspected at full resolution; no blocking clipping or overlap |

See the [summary](results/main-closeout/summary.json),
[test log](results/main-closeout/tests.log),
[arithmetic audit](results/main-closeout/heston-arithmetic-audit.json) and
[notebook report](results/main-closeout/heston-notebook-ipc/verification.json).
The pinned existing environment was reused with this checkout's `src`. Initial
sandbox denial of localhost IPC happened before notebook execution; a fresh
isolated retry allowed local IPC while retaining external Python socket/DNS
guards. Both records are preserved.

No new scientific candidate or acceptance cohort ran. Unit tests include tiny
existing training/reload fixtures; notebook05 replays four historical root-43021
illustrations. Reserved root-43022 was not evaluated. Unchanged participant,
HKDE and rough lessons retain their earlier source-bound verification; no fresh
rerun of them is claimed. No cluster job or market acquisition occurred.

Browser layout, controls and offline browser runtime remain unverified in the
[participant delivery record](tutorial-delivery-verification.md). Its full-route
verification flag stays false. This visible delivery limit is separate from
merging the implemented code and provisional lessons.

## Francesco's contribution

[Draft PR #33](https://github.com/fmuia/sbi4finance/pull/33), `feat/svj-pipeline`,
is not yet on main at this snapshot. It adds a separate P/returns-to-VaR/ES
application. The [compatibility review](svj-contribution-compatibility.md) finds
no module collision and only README/configuration textual conflicts. Its units,
finite dynamics, targets and checkpoint differ; it cannot silently replace our
Heston kernel or the options lesson.

The draft remains separate and untouched. Later review should address explicit
data conversion/completeness, a synthetic lesson route, flagged-audit presentation
and log-loss versus simple-position-loss units. Its tests and scientific studies
were not rerun here. Our merge neither accepts nor rejects that broader proposal.

## Preserved status

- HKDE: provisional selected-case teaching at 2,048 proposals; corrected
  root-95026 test used/failed, raw root-62026 untouched; formal I4 deferred.
- Heston: decision completed to retain diagnostic teaching and defer more
  experiments; posterior provisional, root-43022 untouched.
- Rough: all 256 test cases used; one eta-spread gate failed; no tuning.
- Tutorial: Q/options selected; setup and isolated Python delivery verified at
  their recorded revisions, with browser limitations still visible.

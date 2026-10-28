# The restructure of 28 October 2026, and how to port work begun before it

The repository was split into `course/`, `tutorial/` and `lab/`, and everything off the critical path was removed.
The removed files are unchanged at commit
[`574b940`](https://github.com/fmuia/sbi4finance/tree/574b940087cd66a7540c5aa080f64e8b33010314), the last commit of
`main` before the split. This page is for branches written against the old layout, such as
[PR #47](https://github.com/fmuia/sbi4finance/pull/47) and [PR #48](https://github.com/fmuia/sbi4finance/pull/48).
Delete it once they are ported.

## Where files went

| Before | After |
|---|---|
| `src/sbi4finance/` (the calibration package and the modules it imports) | `lab/src/sbi4finance/` |
| `scripts/calibration_bank.py`, `scripts/build_surface.py` | `lab/scripts/` |
| `scripts/course/`, and `scripts/plot_design_*.py` | `course/scripts/` |
| `scripts/curriculum/heston_spy_exact.py` | `lab/checks/heston_spy_exact.py` |
| `scripts/curriculum/storyboard.py` | `tutorial/curriculum/storyboard.py` |
| `configs/tasks/calibration_*.json`, `configs/thin/` | `lab/configs/` |
| `cluster/calibration_bank_array.sbatch`, `train_calibration.sbatch`, `study.sbatch`, `thin_coverage_array.sbatch`, `discover_cluster.py` | `lab/cluster/` |
| `tests/test_calibration.py`, `test_thin.py`, `test_bergomi.py`, `test_rough_bergomi.py`, `test_hkde.py`, `test_heston.py`, `test_gbm_bank.py` | `lab/tests/` |
| `pyproject.toml`, `pixi.lock` | `lab/` (the tutorial has its own in `tutorial/`) |
| `tutorial_final/notebooks/` | `tutorial/notebooks/` |
| `tutorial_final/results/`, `tutorial_final/checkpoints/` | `tutorial/assets/results/`, `tutorial/assets/checkpoints/` |
| `tutorial_final/curriculum/data/` | `tutorial/assets/data/` |
| `tutorial_final/curriculum/`, `tutorial_final/TUTORIAL-REVIEW.md` | `tutorial/curriculum/` |
| `tutorial_final/CLAIMS.md`, `tutorial_final/MANIFEST.md` | `tutorial/` |
| `tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md`, `EXPERIMENT-PROTOCOL.md`, `CLUSTER-RUNS.md`, `SPARSE-QUOTE-CASE-PLAN.md` | `lab/plans/` |
| `tutorial_final/studies/` | `lab/studies/` |
| `tutorial_final/checks/` | `lab/checks/`, except the withdrawn `desk_case/`, `rough_spy_is_cases*` and `heston_sparse_free_long.py` (archived) |
| `docs/course/` | `course/` |
| `docs/design/calibration-design-document.md` and its images | `course/design-document/` |
| `docs/mcmc-policy.md` | `lab/plans/mcmc-policy.md` |
| `data/SOURCES.md` | `lab/data/SOURCES.md` |
| `data/SPMO_VOO_2020-01-01_2026-09-04.csv` | `course/scripts/data/` |
| Everything else: `notebooks/`, `checkpoints/`, the other `docs/`, `data/synthetic/`, 40 library modules, 64 scripts, 39 test files, the other configs and cluster jobs, `tutorial_final/PLAN.md` | Archived at `574b940` |

New: `tutorial/src/sbi4f_tutorial/` (the notebooks' runtime), `tutorial/pyproject.toml` and `tutorial/pixi.lock`,
`tutorial/tests/`, `tutorial/assets/quick-checks/`, `tutorial/assets/tasks/` and `tutorial/assets/images/` (copies
made by `lab/studies/manifest.py`), and `lab/tests/test_tutorial_parity.py`.

## Conventions that changed

- Lab commands run from `lab/`, where its pixi manifest is, and cluster jobs are submitted from `lab/`. Inside
  `lab/` documents, paths are relative to `lab/`.
- A study's `sys.path` line is `Path(__file__).resolve().parents[1] / "src"`. A quick check's is the same, with one
  more `parents` level per folder.
- `accept.py --to` is relative to `tutorial/assets/`. `manifest.py` also refreshes the copies listed in its
  `EXPORTS`. `claims.py` reads `tutorial/assets/results/calibration/` and writes `tutorial/CLAIMS.md`.
- The notebooks import `sbi4f_tutorial`, never `sbi4finance`, and read only `tutorial/assets/`. Its modules copy
  lab functions unchanged, and `lab/tests/test_tutorial_parity.py` fails when a copy differs from the lab.

## Porting PR #47 and PR #48

Whichever of the restructure and a pull request reaches `main` second takes the conflicts. A trial merge of PR #48
at `a91851b` (which contains PR #47) into the restructure gave the following.

- **Placed automatically by git:** every change to a moved file, and the new files in `src/sbi4finance/`, `tests/`,
  `configs/` and `cluster/`. They land in `lab/`.
- **Six content conflicts**, all small: `course/forward-pricing-emulation.md`, `lab/plans/CLUSTER-RUNS.md`,
  `lab/plans/NUMERICAL-EXPERIMENT-PLAN.md`, `lab/studies/README.md`, `lab/studies/evaluate_calibration.py` and
  `lab/studies/evaluate_thin.py`. In the two scripts only the usage lines of the docstring conflict. Keep PR #48's
  text and rename its paths: `tutorial_final/studies/` becomes `lab/studies/`, or `studies/` inside `lab/`
  documents.
- **`pyproject.toml` and `pixi.lock` (modify/delete):** move PR #48's `[tool.pixi.feature.core]`,
  `[tool.pixi.feature.reference]` and the three environments into `lab/pyproject.toml`, delete the root copies, and
  re-lock with `cd lab && pixi lock`.
- **Files git places wrongly or not at all.** Move them by hand:
  - new scripts left in `scripts/` go to `lab/scripts/`;
  - PR #48's documents from `docs/` land at the repository root (`*.md`). Living plans, work orders and the status
    page go to `lab/plans/`, and dated result records to `lab/evidence/`;
  - `docs/pilot-results/` lands at `pilot-results/` and goes to `lab/evidence/pilot-results/`;
  - `data/synthetic/full-surface-heston-final500/` goes to `lab/evidence/` until a notebook uses it. Then a lab tool
    exports it into `tutorial/assets/` and `lab/studies/manifest.py` records it. The tutorial reads nothing else.
- **Then:** replace any remaining `tutorial_final`, `docs/` and root-relative paths in the new scripts and
  documents. Run `pixi run test` and `pixi run release-check` in `lab/`, and `pixi run test` in `tutorial/`.

```bash
git switch feat/full-surface-calibration
git merge origin/main                      # once the restructure is on main
git status --short | grep -v ' \(lab\|course\|tutorial\)/'   # files that still need a place
```

The notebook integration of the BayesFlow 2.1 Heston candidate follows the same path as any asset. Accept the
exported checkpoint with `lab/studies/accept.py` (or an export tool writing the same `ACCEPTED.json` record). Copy
the functions of `sbi4finance.calibration.full_surface` that rebuild and sample the network, unchanged, into
`tutorial/src/sbi4f_tutorial/full_surface.py`, add that module to `COPIED_FROM` in
`lab/tests/test_tutorial_parity.py`, and add a check that both return the same draws. Then move the tutorial's
BayesFlow pin to 2.1.

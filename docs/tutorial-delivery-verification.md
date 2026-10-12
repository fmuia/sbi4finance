# Participant delivery preparation for issue #26

The user selected **Q/options: GBM foundations followed by a provisional
corrected-HKDE diagnostic lesson** on 12 October 2026. The
[route card](tutorial-participant-route.md) and [README schedule](../README.md)
record the two-hour curriculum. The user also confirmed that #27 has not yet
supplied a pinned combined revision. **H26.3 remains incomplete.** This record
separates useful preparation from the final participant acceptance check.

## Repository and scientific scope

Worktree: `sbi4finance-tutorial-followup`, branch
`feat/tutorial-followup-participant-route`, based on #19 at
`c762d7613b57d3ea2632671e773498ae8734b701`. No stacks were merged or collaborator
branches changed. Current notebook10 is #19's fixed, live-by-default example.
The selected combined experience requires the following existing contributions:

| Contribution | Required revision | What #27 must preserve |
|---|---|---|
| #19 | `c762d7613b57d3ea2632671e773498ae8734b701` | Scientific history, audit, current handovers and participant figures |
| #22 | `663ac4edeaf5935b59da5eb3d6767e13b0806037` | Local synthetic panel, cached default, explicit `demo-hkde-corrected` live opt-in, observation/truth boundary |
| #29 | `45bb08d2b091df7bae49fa250424e122f19e3eca` | Artifact-bound provisional decision and updated notebook11 explanation |
| #28 | `f1bfa7256cc1ea9e5798e49a95a8befc409fa84e` | Optional rough review with completed test and eta-spread failure retained |
| This follow-up | `feat/tutorial-followup-participant-route` (final commit is recorded in its PR) | Chosen route, pacing/exercises, README and MMAR scope correction; preflight hashes identify executable inputs separately |

These heads identify inputs to #27, not an assertion that this checkout
contains them. #27 owns full stack composition, including the original plans
and later rough/data branches. Do not copy notebook10 wholesale between #19
and #22. Before final verification, reconcile its introductory selected-route
and pacing text with the route card while keeping #22's mode/data controls.
Use the #29 decision explanation without changing frozen scientific sources
or historical evidence bindings.

The [#24 decision in PR #29](https://github.com/fmuia/sbi4finance/blob/45bb08d2b091df7bae49fa250424e122f19e3eca/docs/hkde-acceptance-decision.md)
permits continued provisional selected-case teaching at **2,048 proposals**;
scientific acceptance is false and formal I4 is deferred. A delivery pass
cannot change that decision. No retraining, new cohort, reserved-test use or
market acquisition was undertaken for this follow-up.

## Fresh locked setup: passed on this preliminary branch

The new worktree had **no `.pixi` directory** before installation. Running
`pixi install --locked` with Pixi 0.70.2 created a real environment inside it,
not a symlink to another worktree. The normal package cache was allowed;
this checks fresh environment construction, not a cold-cache download.
The first restricted attempt could not write Pixi's global cache lock and
stopped before dependency installation. The same command passed once normal
package/cache access was permitted. No lock or dependency change was needed.

The [setup record](results/tutorial-participant-preflight/setup.json) binds
`pyproject.toml` and `pixi.lock` and lists the installed versions: Python
3.12.15, BayesFlow 2.0.14, Keras 3.15.1, JAX 0.11.1, marimo 0.25.1 and the
remaining scientific packages. Platform: macOS 26.4.1 / arm64; execution uses
the CPU backend. Installation duration was not instrumented. Linux is present
in the lock but was not installed here. Repeat fresh setup after #27 supplies
the actual combined revision, even if its dependency files are unchanged.

## Current-branch preflight

The narrow [preflight checker](../scripts/check_tutorial_delivery.py) exports
the existing foundations, corrected-HKDE and calibration-audit notebooks in
separate processes. It copies source/configuration and bound synthetic assets
to a new directory, omitting market fixtures, the `data` directory and raw
training banks. The foundations lesson still creates its tiny temporary
64/16/8-example synthetic bank as part of its existing teaching exercise.
No neural training or calibration-study rerun is part of this check.

```bash
pixi install --locked
PYTHONPATH=src pixi run python scripts/check_tutorial_delivery.py \
  --scope preflight --out /tmp/tutorial-preflight-new
```

The output directory must be new. A Python socket/DNS guard blocks external
TCP, UDP and name lookups while permitting loopback and Unix kernel IPC.
This is a Python-level guard, not an operating-system network sandbox. The
checker probes the guard and local communication, checks strict marimo lint,
reads exported cell errors, extracts all PNGs for visual inspection and binds
copied inputs and HTML with SHA256. Its report cannot mark the full selected
route, scientific acceptance or I4 as passed. It does not install dependencies.

The [preliminary execution record](results/tutorial-participant-preflight/verification.json)
reports **21 / 17 / 10 output cells, zero cell errors and 2 / 6 / 3 figures**
for notebooks01/10/11. All strict checks passed and all 11 PNGs were visually
inspected: axes, units, legends and failure/budget annotations are readable.
The existing eight selected-case HKDE checks passed. Its first run in the fresh
environment took **41.95 s inside the live cell**, including **38.84 s** for
imports, artifact checks and checkpoint loading; proposal sampling/density
took 1.26 s, nine-price likelihood/weights 1.85 s and the two-price surrogate
0.00063 s. These overlapping rows must not be summed. Cold setup costs differ
materially from earlier branch timings; no universal runtime is promised.

The first verifier stopped after successful exports because its text parser
stripped a marimo callout's encoded warning. The parser was corrected and the
existing HTML was inspected without repeating inference; notebook11 was then
exported once in 0.95 s. Whole-export times for notebooks01/10 were not retained
after that parser failure. This is **not** an end-to-end successful invocation
of the final checker CLI or a full-route timing. The record preserves those
limits and binds the actual executable inputs; #27's final check must measure
the complete route afresh.

HTML/session data, figures and logs are retained in
`/tmp/sbi4finance-tutorial-preflight-20261012b`; small reports/logs and new figures
are committed here. Identical existing figures are referenced by repository
path and SHA256. Exported HTML uses CDN frontend assets and contains temporary
absolute review links. Consequently **standalone offline browser rendering and
full browser layout are not certified** by this Python export check. The
installed local marimo lesson and exported review artifact must be assessed
separately at final delivery. A shutdown warning about one Python semaphore
was retained; no notebook cell failed. Exercise prompts/answers were reviewed;
the interactive slider activity still belongs in the combined-tree check.

## Remaining final acceptance work, in order

1. Receive #27's exact combined commit and record source/lock/artifact hashes.
   Check that #22's local panel and mode controls and #29's decision notebook
   are actually present; a PR base name is insufficient. Incorporate the
   selected lesson introduction/pacing before exporting.
2. Install `pixi install --locked` in a new environment on that revision.
   Record platform, package versions, command/exit output and limitations
   separately from execution. The setup result above does not substitute.
3. In fresh processes with external networking blocked and local kernel IPC
   allowed, run the complete route: foundations, `hkde_data_walkthrough`,
   notebook10 cached and opt-in live modes, and selected notebook11 sections.
   Remove market fixtures and raw banks; retain only required synthetic assets.
   Use the existing data-boundary/mode checks as well as HTML exports, with
   source paths verified against the isolated copy.
4. Record full-route and per-notebook runtime, checkpoint load, proposal
   sampling/density, nine-price likelihood, two-price forward map and export
   costs from that execution. Keep overlapping timing rows separate; never
   sum old branch reports or substitute 8,192-proposal cache for 2,048 live.
5. Inspect every exported graph, table and exercise, including live/replay
   labels and missing optional M/E inputs. Verify observation/context versus
   truth/future roles and aligned parameter/weight/target rows. Keep failures
   visible; no seed retry or gate change follows a failed example.
6. Link the exact final execution record from README, the selected lesson and
   #26 handover; publish the tested route for review. Repeat affected checks
   after executable or figure changes. Keep #26 open until those results are
   available, and leave formal I4 deferred under the current scientific decision.

The MMAR text change is presentation-only and passes strict marimo checking.
Its optional market-data/checkpoint workflow was not executed as part of the
selected synthetic route. It no longer promises that later sections use the
same approximator, and states the fresh-cascade/latent-state distinction.

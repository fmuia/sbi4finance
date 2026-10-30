# The lab

The lab produces and checks the tutorial's data, estimators and financial conclusions. The participant package
loads published assets and copied functions; it never imports this package. Run the commands below from `lab/`.

```bash
cd lab
pixi install --locked -e core
pixi run --locked -e core test
```

## Start with the supplied data and models

All three complete-surface routes have **100k training banks and completed baseline fits**. Use the
[training-data guide](plans/training-data-guide.md) for archive identities, extraction paths, SHA checks and
executable examples of loading, training and evaluating. Downloads are in the
[GitHub release](https://github.com/fmuia/sbi4finance/releases/tag/tutorial-backbones-2026-10-29) and
[Drive mirror](https://drive.google.com/drive/folders/1Zez6mSu8Y_B7IIjdRYWJlDxdAl49fTLq).

The pretrained-and-cases ZIP is about 38 MB; full bank archives are about 123 MB Heston, 900 MB known-curve rough
and 934 MB joint-curve rough. Extract native banks into one common external directory without flattening paths:
Heston's snapshot and canonical collection are in different top-level directories. Preserve the original manifests,
part files and checksums. A full bank is unnecessary for pretrained inference. Keep large payloads outside Git.

Current validation is development evidence, not automatic model or lesson acceptance. J1 is complete; joint
conditional references J2 and product/long-end comparisons J3 remain pending. See [course/status.md](../course/status.md)
and the [numerical plan](plans/NUMERICAL-EXPERIMENT-PLAN.md) before starting additional work.

## Canonical workflow

Use the current full-surface route for new backbone work. The guide above gives complete command examples;
`--help` on each CLI describes its arguments. Always use a new output directory for a changed fit or evaluation.

| Operation | Entry point |
|---|---|
| Load or stream requested native bank splits and array roles | [`bank_collection.py`](src/sbi4finance/calibration/bank_collection.py): `load_bank`, `iter_snapshot`, `TRAINING_ROLES` |
| Load and sample a matching native posterior | [`full_surface.py`](src/sbi4finance/calibration/full_surface.py): `load_checkpoint`, `sample` |
| Train the amortized posterior | [`scripts/train_full_surface.py`](scripts/train_full_surface.py) |
| Train the separate forward-price emulator | [`scripts/train_full_surface_pricer.py`](scripts/train_full_surface_pricer.py) |
| Evaluate held-out calibration and recovery | [`scripts/evaluate_full_surface.py`](scripts/evaluate_full_surface.py) |
| Initialize, plan, append and freeze a native collection | [`scripts/calibration_bank_collection.py`](scripts/calibration_bank_collection.py) |
| Run a bounded generation campaign when needed | [`scripts/run_bank_campaign.py`](scripts/run_bank_campaign.py); [on-demand instructions](plans/generate-bank-on-demand.md) |
| Reference MCMC, classical comparisons and financial products | [`studies/README.md`](studies/README.md), the relevant task/study and [MCMC policy](plans/mcmc-policy.md) |
| Publish reviewed code and lesson assets | [`copy_to_tutorial.py`](studies/copy_to_tutorial.py), [`accept.py`](studies/accept.py), [`manifest.py`](studies/manifest.py), [`claims.py`](studies/claims.py) |

Reuse successful bank rows and fixed holdouts. An unchanged task can grow by appending missing split-qualified
IDs; a changed task, grid or prior needs a new collection. Training on a new snapshot starts a new fit. Never
regenerate successful data to repair a local download. Retain the task, source/runtime, bank and checkpoint
identities, observation-noise law, Monte Carlo error records and actual sampler settings.

The older [`studies/train_calibration.py`](studies/train_calibration.py) supports the registered study and sparse
routes; it is not the default training command for the new backbone. **Dated cluster launch/recovery controllers
and campaign configurations are historical records bound to their original checkouts, paths, budgets and journals.**
Do not repurpose `cluster/start_*20261028.py`, `start_joint100k_20261029.py` or another frozen controller by changing
its paths. Use the canonical current scripts and a new reviewed configuration/output when fresh work is justified.
The [cluster runbook](plans/CLUSTER-RUNS.md) records earlier commands rather than authorizing their whole programme.

## Environments

| Environment | Purpose | Install |
|---|---|---|
| `core` | Current BayesFlow 2.1.0/JAX workflow; native adaptive Tsit5 for new FlowMatching checkpoints | `pixi install --locked -e core` |
| `core-gpu` | The same with CUDA on Linux | `pixi install --locked -e core-gpu` |
| `core-reference` | Current workflow plus optional PyMC | `pixi install --locked -e core-reference` |
| `default` | Archived BayesFlow 2.0.14 studies/checkpoints | `pixi install --locked` |
| `gpu` | Archived environment with CUDA on Linux | `pixi install --locked -e gpu` |

Use `pixi run --locked -e core ...` for the current backbone. Historical checkpoints retain their matching
runtime and sampling settings. A passing file/runtime test does not establish posterior correctness. Prefer PyMC
for references where it fits naturally; the policy records specialized-method exceptions and required provenance.

## From lab code to a lesson

Put numerical changes here, with the relevant checks. When the tutorial needs a lab function live, copy it
unchanged with `pixi run --locked -e core copy-to-tutorial --add MODULE.NAME`. The registry in that tool and
[`test_tutorial_parity.py`](tests/test_tutorial_parity.py) enforce the copies; the tutorial must not import the lab.

Publish assets through `accept`, `manifest` and `claims`, preserving their acceptance/development roles and
reference provenance. Never hand-edit `tutorial/MANIFEST.md`, the generated claim tables or asset payloads.
After a change, run from this folder:

```bash
pixi run --locked -e core manifest
pixi run --locked -e core claims
pixi run --locked -e core test
pixi run --locked -e core release-check
```

Then run the required [tutorial checks](../tutorial/README.md#author-or-change-a-lesson). For historical studies,
use their recorded environment. [`studies/README.md`](studies/README.md) describes S1–S9 and the publication tools;
[`checks/README.md`](checks/README.md) describes exploratory quick checks, which are not accepted studies.
Course figures are drawn by `course/scripts/` in the lab environment; `pixi run --locked -e core course-check`
checks course-plan numbers against stored results.

## What remains, and what can be trimmed

The [numerical plan](plans/NUMERICAL-EXPERIMENT-PLAN.md#current-next-work-and-repository-trimming) gives the next
lesson work and the trimming order. Check dependencies before removing scripts, helpers or evidence. In particular,
sparse code still has active imports, and some `gbm*`/`svj*` helpers supply utilities, plot style or diagnostics to
supported studies. Their original application being deferred does not make those dependencies removable.

Complete 63-contract surfaces are the current task. Sparse chains, masks and arbitrary quote sets remain a
[separate follow-up](plans/SPARSE-QUOTE-CASE-PLAN.md). This handover starts no new cluster campaign.

# The lab

The lab produces and checks the tutorial's data, estimators and financial conclusions.
It reuses the authoritative numerical runtime under `tutorial/src/sbi4f_tutorial/`;
the participant package never imports the lab. Keep the complete repository
checkout and run the commands below from `lab/`.

```bash
cd lab
pixi install --locked -e core
pixi run --locked -e core test
```

Pixi installs the sibling tutorial runtime as a local dependency without its
participant-only BayesFlow/marimo extra. The lab selects its own current or
historical neural environment. Deploy both `lab/` and `tutorial/` from the same
revision; copying `lab/` alone no longer supplies all executing source.
A source-checkout import checks this boundary before loading numerical modules.
If an old editable install points to another worktree, it stops with the expected
and actual runtime locations; rerun the locked install above in this checkout.

## Start with the supplied data and models

### Review the Heston SPX notebook

The [SPX calibration lesson](notebooks/heston_spx_calibration.py) contains the model,
interactive simulator, live training, posterior checks and pricing examples.

```bash
cd lab
pixi install --locked -e core-reference
pixi run --locked -e core-reference marimo edit notebooks/heston_spx_calibration.py
```

The saved PyMC examples and live training work without a download. To use the
pretrained posterior, follow the [bundle instructions](plans/training-data-guide.md#heston-spx-notebook-and-optional-pretrained-model).
The [lesson run record](evidence/pilot-results/2026-11-11/heston-actual63-notebook/README.md#delivery-scope)
maps its code and results, and explains which research files were left out.

### Existing option and trajectory packages

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

Trajectory inference is the third core application. Its three million-history
banks, native models and preserved outputs are available through the
[training-data guide](plans/training-data-guide.md); the
[path plan](plans/PATH-INFERENCE-PLAN.md) records growing-prefix inference and
its unresolved numerical/calibration questions. The
[collaborator handover plan](../tutorial/curriculum/TUTORIAL-REVIEW.md#g-collaborator-handover-before-the-next-feedback-round)
separates scheduler-free workers, optional deployment wrappers and historical
Oscar controllers. It defines the remaining clean-machine and compute rehearsal;
no other cluster is claimed tested by that plan.

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
| Check runtime ownership and publish lesson assets | [`runtime_boundary.py`](studies/runtime_boundary.py), [`accept.py`](studies/accept.py), [`manifest.py`](studies/manifest.py), [`claims.py`](studies/claims.py) |

Reuse successful bank rows and fixed holdouts. An unchanged task can grow by appending missing split-qualified
IDs when producer source/runtime identities also match; a changed task, grid or prior needs a new collection.
Training on a new snapshot starts a new fit. Never regenerate successful data to repair a local download. Retain the task, source/runtime, bank and checkpoint
identities, observation-noise law, Monte Carlo error records and actual sampler settings.

The older [`studies/train_calibration.py`](studies/train_calibration.py) supports the registered study and sparse
routes; it is not the default training command for the new backbone. **Dated cluster launch/recovery controllers
and campaign configurations are historical records bound to their original checkouts, paths, budgets and journals.**
The [retention ledger](plans/NUMERICAL-EXPERIMENT-PLAN.md#bounded-retirement-batch--1-november-2026)
identifies retired controllers and their pinned source. Do not repurpose frozen historical controllers by changing
their paths. Use the canonical current scripts and a new reviewed configuration/output when fresh work is justified.
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

## Run the workers on a local CPU

The current workers need no SSH login, scheduler or owner-specific account.
The locked platforms are Linux x86-64 and Apple Silicon; Windows, Apple Intel
and other accelerators are not supported by this lock. Use a filesystem with
POSIX locks and hard links for option collections. Obtain private GitHub/Drive
access with your own account when downloading existing artifacts; computation
then uses ordinary filesystem paths. `pip install lab` alone does not install
these scientific dependencies. The supported Pixi install resolves the sibling
tutorial runtime too; a manually managed environment must install the base
package from `../tutorial` as well as this lab package.

From `lab/`, install the current environment and choose a **new**, absolute
output directory. The following sizes are software checks, not calibrated fits.
The shorter path task below is deliberately different from the published
1,260-day task and cannot load its checkpoints.

```bash
pixi install --locked -e core
export KERAS_BACKEND=jax JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SBI_SMOKE=/absolute/caller-owned/new-sbi-smoke
mkdir "$SBI_SMOKE"
pixi run --locked -e core core-check
```

### Heston options: generate, fit, reload and evaluate

```bash
pixi run --locked -e core python scripts/calibration_bank.py config \
  --model heston --train 32 --validation 8 --test 4 --seed 20261101 \
  --out "$SBI_SMOKE/heston-config.json"
pixi run --locked -e core python scripts/calibration_bank.py local \
  --config "$SBI_SMOKE/heston-config.json" --chunk 8 --workers 1 \
  --out "$SBI_SMOKE/heston-bank"
pixi run --locked -e core python scripts/train_full_surface.py \
  --bank "$SBI_SMOKE/heston-bank" --out "$SBI_SMOKE/heston-fit" \
  --family spline --encoder vector --widths 16,16 --epochs 2 \
  --batch-size 8 --chunk-epochs 1 --checkpoint-selection final --seed 20261102
pixi run --locked -e core python scripts/evaluate_full_surface.py \
  --bank "$SBI_SMOKE/heston-bank" --checkpoint "$SBI_SMOKE/heston-fit" \
  --out "$SBI_SMOKE/heston-validation" --split validation \
  --surfaces 8 --draws 32 --batch-size 4 --seed 20261103
```

The trainer verifies a native checkpoint reload. Evaluation independently loads
that checkpoint, samples parameters for the validation quotes and records the
bank/task/model identities in `evaluation.json`. Neither step opens the reserved
test split. Check execution and finite outputs; eight cases do not establish
coverage or posterior accuracy.

### All three trajectory tasks: generate, stream, reload and evaluate

```bash
for sbi_model in heston rough-known rough-joint; do
  pixi run --locked -e core python scripts/path_bank.py generate \
    --model "$sbi_model" --teaching --observations 29 \
    --train 64 --validation 8 --test 4 --chunk 16 --seed 20261101 \
    --out "$SBI_SMOKE/$sbi_model-path-bank"
  pixi run --locked -e core python scripts/path_bank.py check \
    --bank "$SBI_SMOKE/$sbi_model-path-bank"
  pixi run --locked -e core python scripts/train_path_posterior.py \
    --bank "$SBI_SMOKE/$sbi_model-path-bank" --out "$SBI_SMOKE/$sbi_model-path-fit" \
    --streaming --training-limit 64 --cache-shards 2 --epochs 2 --batch-size 16 \
    --hidden-dim 8 --summary-dim 4 --depth 2 --bins 4 --width 16 \
    --prefix-policy uniform_log_mixture --checkpoint-selection final \
    --validation-cases 8 --max-seconds 120 --seed 20261102
  pixi run --locked -e core python scripts/evaluate_path_posterior.py \
    --bank "$SBI_SMOKE/$sbi_model-path-bank" --checkpoint "$SBI_SMOKE/$sbi_model-path-fit" \
    --out "$SBI_SMOKE/$sbi_model-path-validation" --split validation \
    --cases 8 --draws 32 --batch-size 4 --lengths 1 17 29 --seed 20261103
done
```

The evaluator performs the native reload and samples at each declared prefix.
Parameter labels remain targets; masks exclude padding. These development
validation commands do not open tests, change the workshop's frozen selections
or establish inference quality. The existing bounded readiness commands also
exercise actual training, mask forwarding and reload:

```bash
pixi run --locked -e core python scripts/check_core_cluster.py \
  --allow-local --out "$SBI_SMOKE/option-readiness"
pixi run --locked -e core python scripts/check_path_cluster.py \
  --quick --teaching --out "$SBI_SMOKE/path-readiness"
```

`check_core_cluster.py` requires `--allow-local` outside an allocation. It checks
same-node process locks, append-only bank IDs, committed-epoch recovery and
validation sampling; it does not certify a distributed filesystem. The path
check uses short histories and checks all three tasks. Use a new output for each
readiness attempt and preserve a failure record rather than silently replacing it.

### Larger fits and continuation

Use the [training-data guide](plans/training-data-guide.md) for the immutable
option and trajectory downloads, checksum verification, exact extraction trees
and published training recipes. Option snapshots retain their referenced
collection tree. A trajectory bank is an external sharded directory; verify it
with `path_bank.py check`, then use `train_path_posterior.py --streaming
--training-limit 1000000 --cache-shards 2` with a new output. Do not materialize
its million-history training split with `load_split`. A time cap can end a fit
before its planned epochs; inspect completed epochs, optimizer steps and actual
training views before comparing computing devices or data sizes.

For option training, rerunning the identical command/output resumes only from
its committed training state and validated run contract. `calibration_bank.py
local --resume` reuses verified finished shards; it is a different operation.
Trajectory fits record `resume_supported:false`: use a new output for a fresh
fit. `--init-checkpoint` initializes compatible weights with a fresh optimizer;
it is not exact training-state continuation and is not accepted by the streaming
route. Existing option-collection extension additionally requires matching
frozen generator sources/runtime. Loading its data for a new fit does not require
that original generator checkout. Keep heldouts and evaluation guards intact;
changing a task, prior, noise law or numerical grid needs its own task/data/model
identities. None of these execution recipes promotes scientific acceptance.

## Optional single-job Slurm deployment

The same workers run in an allocation. `cluster/core_cpu.sbatch` and
`core_gpu.sbatch` set no account, partition or QOS: choose those in the caller's
`sbatch` flags. Their CPU/GPU/memory/time defaults are resource requests, not
estimates of a new experiment's cost. The wrappers still require an allocation,
use the submitted checkout, select a locked current environment and stop if
GPU readiness fails. Both sibling source directories (`lab/` and `tutorial/`),
the installed runtime and output directories must be visible to the allocated
node. Install the environment before submitting; do not rely on a compute node
downloading packages.
The same-checkout import guard also applies to raw-Python workers: a missing
sibling runtime or an editable install from another worktree fails immediately.
Install from the intended source checkout before submitting; do not bypass the
guard by changing manifests or mixing source revisions.

This builds and prints **one** bounded CPU-readiness submission without invoking
`sbatch`. Replace the site and filesystem placeholders, run from `lab/`, and
review the printed command. `$SBI_RUN` must already exist; its `readiness` child
must be new.

```bash
export SBI_PIXI=/absolute/path/to/pixi
export SBI_RUN=/absolute/shared/caller-owned/new-readiness-run
test -x "$SBI_PIXI" && test -d "$SBI_RUN" && test ! -e "$SBI_RUN/readiness"
sbi_submit=(sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION
  --cpus-per-task=2 --mem=8G --time=00:10:00
  --output="$SBI_RUN/readiness-%j.out"
  cluster/core_cpu.sbatch "$SBI_PIXI" core
  python scripts/check_core_cluster.py --out "$SBI_RUN/readiness")
printf '%q ' "${sbi_submit[@]}"
printf '\n'
```

The account/partition/resource flags precede the batch script and are consumed
by Slurm. Printing is not submission and does not check your site's associations.
If a GPU target is later selected, first run a bounded allocated readiness/small
fit through `core_gpu.sbatch`, with your GPU partition, `--gres=gpu:1` (or a site
GPU type), memory/time requests and `core-gpu` installed. Unset an inherited
`JAX_PLATFORMS=cpu` before that GPU invocation; the wrapper requires an actual
JAX GPU. No alternative cluster or GPU deployment is certified by this recipe.

The multi-job launchers retain their source/budget/duplicate-submission guards.
`submit_path_scaleup.py` is a fixed campaign; `scripts/run_bank_campaign.py` also
submits Slurm jobs. Neither is the default command for a single fresh fit. Dated
Oscar controllers/configurations are frozen campaign provenance, not templates
to repurpose by replacing an owner's paths. Local workers record available
runtime/wall-time/memory information; they do not invent Slurm accounting.

## From lab code to a lesson

The executing lesson runtime is owned under `tutorial/src/sbi4f_tutorial/`.
Edit application helpers in `heston_options/`, `rough_options/` or
`trajectories/`; edit reused numerical components in `shared/`. The lab imports
those same definitions. Its existing public module names remain available for
worker and study compatibility. There is no function-copy step and no second
implementation to maintain. `copy-to-tutorial` is a migration-only command that
explains the new locations; it no longer writes source.

The historical `checks/bs_hierarchy.py` and `checks/long_end/long_end.py`
programs retain their original study inputs/globals. Numerical parity tests
compare their retained calculations with the tutorial versions; they are not
the editing targets for current lessons or production worker APIs.

Run `pixi run --locked -e core runtime-check` to check the dependency boundary
and shared-definition identity. [`test_tutorial_parity.py`](tests/test_tutorial_parity.py)
also exercises the lab and tutorial APIs on the same inputs. The tutorial remains
self-contained and must never import `sbi4finance` or read lab files.

Keep large-bank generation, production training, reference studies and publication
here. Publish assets through `accept`, `manifest` and `claims`, preserving their
acceptance/development roles and reference provenance. Never hand-edit
`tutorial/MANIFEST.md`, generated claim tables or asset payloads. After a shared
runtime or publication change, run from this folder:

```bash
pixi run --locked -e core manifest
pixi run --locked -e core claims
pixi run --locked -e core test
pixi run --locked -e core release-check
```

Then run the required [tutorial checks](../tutorial/README.md#author-or-change-a-lesson).
Moving numerical source changes generator provenance even when its calculations
are identical. Existing banks remain readable/trainable; keep their original
manifests and use the matching historical producer checkout/environment to append
or resume frozen generation. New banks record the new source identity. Existing
model payloads, tasks and published caches are not rewritten by this refactor.

For historical studies, use their recorded environment.
[`studies/README.md`](studies/README.md) describes S1–S9 and publication tools;
[`checks/README.md`](checks/README.md) describes exploratory quick checks, which
are not accepted studies. Course figures are drawn by `course/scripts/` in the
lab environment; `pixi run --locked -e core course-check` checks course-plan
numbers against stored results.

## What remains, and what can be trimmed

The [numerical plan](plans/NUMERICAL-EXPERIMENT-PLAN.md#current-next-work-and-repository-trimming) gives the next
lesson work and the trimming order. Check dependencies before removing scripts, helpers or evidence. In particular,
sparse code still has active imports, and some `gbm*`/`svj*` helpers supply utilities, plot style or diagnostics to
supported studies. Their original application being deferred does not make those dependencies removable.

Complete 63-contract surfaces are the current task. Sparse chains, masks and arbitrary quote sets remain a
[separate follow-up](plans/SPARSE-QUOTE-CASE-PLAN.md). This handover starts no new cluster campaign.

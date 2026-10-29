# The lab

The code that produces every number of the tutorial: the rough-Bergomi and Heston simulators and pricers, the
calibration banks, the networks' training, the reference posteriors, the studies and quick checks, the Oscar jobs,
and the tests. The tutorial loads the results as files; it never imports this package (see the
[repository README](../README.md)).

Run every command from this folder:

```bash
cd lab
pixi install --locked -e core  # current BayesFlow 2.1 numerical backbone
pixi run --locked -e core test # unit tests and tutorial parity tests
```

## Layout

| Path | What |
|---|---|
| `src/sbi4finance/calibration/` | The calibration library: tasks (`tasks.py`), banks (`bank.py`), pricing (`pricing.py`), the network's adapter and training helpers (`inference.py`), the audit (`correction.py`), validation (`validation.py`), the classical route (`classical.py`), the neural pricer (`pricer_net.py`), products (`products.py`), real chains (`chain.py`), Black–Scholes and the design grid (`surface.py`), figures (`plots.py`) |
| `src/sbi4finance/` | The simulators the library uses: `rough_bergomi.py` and `bergomi.py` (hybrid scheme, conditional put estimator), `hkde.py` (the Heston Fourier pricer, used with jumps off) and `heston.py` (Heston paths) |
| `configs/tasks/` | Task and bank configurations: Heston (50,000 / 5,000 / 5,000 smiles, seed 2610), rough Bergomi (100,000 / 5,000 / 7,000, seed 2611), the rough pilot (1,600 / 200 / 400, seed 2612) and the 200-smile mini-pilot |
| `scripts/` | `calibration_bank.py` (`config`, `job-map`, `shard`, `assemble`, `local`): banks, locally or as Oscar arrays. `build_surface.py`: a real chain read onto the design grid (study S8) |
| `studies/` | Studies S1 to S9 and the release tools `accept.py`, `manifest.py`, `claims.py` and `copy_to_tutorial.py`; [studies/README.md](studies/README.md) gives each study's command and test rows |
| `checks/` | Quick checks behind the "why Bayes" material and the course: not accepted studies; [checks/README.md](checks/README.md) |
| `cluster/` | Oscar: lane discovery, bank arrays, network training, one generic study job. Submit every job from `lab/` |
| `tests/` | Unit tests, and `test_tutorial_parity.py`, which checks that the tutorial's copies of this code are unchanged and return the same numbers |
| `plans/` | The [numerical experiment plan](plans/NUMERICAL-EXPERIMENT-PLAN.md), its [protocol](plans/EXPERIMENT-PROTOCOL.md), the [cluster runbook](plans/CLUSTER-RUNS.md), the [sparse-chain follow-up](plans/SPARSE-QUOTE-CASE-PLAN.md) and the [MCMC policy](plans/mcmc-policy.md) |
| `data/SOURCES.md` | Where the raw option chains came from. Raw files are not committed; derived data are |

## From a task to the stage

1. **Task and bank.** A configuration in `configs/tasks/` fixes the 63 contracts, the quote noise, the prior box,
   the forward-variance curve prior and the pricing numerics. `pixi run bank local --config ... --out ...`
   builds a small bank; the full banks run as Oscar arrays (`cluster/calibration_bank_array.sbatch`; the runbook has
   the commands). A bank is regenerated exactly from its configuration and seed, so banks are never committed.
2. **Network.** `studies/train_calibration.py` trains the amortized posterior with fresh quote noise in every
   batch, on Oscar with `cluster/train_calibration.sbatch`.
3. **Checks.** S1 (bank quality), S2 (simulation-based calibration, coverage, contraction), S3 (Heston against the
   exact posterior), S4 (MCMC on a neural pricer), S5 (the audit of each smile with the Monte Carlo likelihood), S6
   (least squares and Gauss–Newton), S7 (products and reserves), S8 (the real smile), S9 (costs). Each study writes
   a new directory and refuses to overwrite one.
4. **Into the tutorial.** `pixi run accept --run RUN --to results/calibration/heston/s2` copies an accepted run
   into `tutorial/assets/` with `ACCEPTED.json` (`--checkpoint RUN --to checkpoints/heston` for a network, with the
   optimizer state removed and same-seed draws checked). Then `pixi run manifest` refreshes the tutorial's copies of
   quick-check outputs, task files and figures, and the hash table of `tutorial/MANIFEST.md`, and `pixi run claims` regenerates
   `tutorial/CLAIMS.md`.
5. **Verify.** `pixi run test` and `pixi run release-check` here; `pixi run test` and `pixi run test-notebooks` in
   `tutorial/`.

Rerun a quick check the same way, then `pixi run manifest`: `tests/test_tutorial_parity.py` fails while a tutorial
copy is stale.

## Rules

- Library code lives here once, with tests. When a notebook needs a computation live, copy the functions unchanged
  into the module of the same name in `tutorial/src/sbi4f_tutorial/` with `pixi run copy-to-tutorial --add
  MODULE.NAME`. `tests/test_tutorial_parity.py` fails when a copy differs from the code here, or returns different
  numbers; after changing a copied function, `pixi run copy-to-tutorial` copies it again.
- Prefer PyMC for reference MCMC where it fits naturally; the [MCMC policy](plans/mcmc-policy.md) lists the
  exceptions and what every chain must record.
- Every result records the task fingerprint, the bank manifest's SHA-256 and, where a network was used, the
  checkpoint's SHA-256. Banks, model payloads and full posterior draws stay outside Git, with their identities in git and versioned downloads in the [training-data guide](plans/training-data-guide.md).
  How much of each run's raw output belongs in git is open (decision D22 in
  [the curriculum overview](../tutorial/curriculum/00-overview.md#10-decisions)).
- `course/scripts/` runs in this environment: `pixi run python ../course/scripts/lecture08.py` redraws a lecture's
  figures (the last figure of `lecture09.py` also needs scikit-learn, which the environment lacks), and
  `pixi run course-check` checks the course plan's numbers against the stored results.

## Environments

| Environment | What | Install |
|---|---|---|
| `core` | Current BayesFlow 2.1.0/JAX backbone and its native adaptive FlowMatching sampler | `pixi install --locked -e core` |
| `core-gpu` | Current backbone with CUDA, Linux only | `pixi install --locked -e core-gpu` |
| `core-reference` | Current backbone plus optional PyMC reference MCMC | `pixi install --locked -e core-reference` |
| `default` | Archived BayesFlow 2.0.14 studies and their historical checkpoints | `pixi install --locked` |
| `gpu` | The same with JAX's CUDA plugin, for Oscar (linux-64 only) | `pixi install -e gpu` |

The current 100k checkpoints use `core`; archived 2.0 networks require their historical environment. The participant package uses 2.1 and gates its historical neural comparisons until a separately checked update. Use `scripts/train_full_surface.py` for current posterior fits; `studies/train_calibration.py` retains the older study route.

## Kept for now, off the critical path

- **The sparse-chain code** (`calibration/thin.py`, `calibration/thin_products.py`, `studies/thin_coverage.py`,
  `studies/evaluate_thin.py`, `studies/thin_real.py`, `checks/thin_chains/`, `configs/thin/`,
  `cluster/thin_coverage_array.sbatch`, `tests/test_thin.py`) is deferred: sparse chains are a separate unit after
  the release. It stays because `studies/train_calibration.py` and `studies/accept.py` still import it. Review this dependency before trimming it in a separate follow-up. The course's Lecture 16 cites its
  EWG results.
- **Helpers from other models.** `gbm.py`, `gbm_bank.py` (small utilities) and `svj/plots.py` (the
  figure style) stay because the library imports a few functions from them; `svj/pmmh.py`, with the
  `svj/filtering.py` and `svj/model.py` it imports, stays because `studies/heston_check.py` and
  `studies/pricer_route.py` take R-hat and ESS from it. Fold those functions into `calibration/` after the
  tutorial, and delete the rest.

## Oscar

`.claude/skills/cluster-setup/` walks a new user through access, lane discovery (`pixi run discover-cluster
--ssh-host oscar`), pixi on Oscar and the first job. The exact commands of every run are in
[plans/CLUSTER-RUNS.md](plans/CLUSTER-RUNS.md).

# sbi4finance

Material for the ICAIF '26 tutorial **Amortized Bayesian Calibration of Volatility Models** (Milan, 14–15 November
2026): the notebooks presented on the day, a sixteen-lecture course behind them, and the lab that produces every
number they show. Inference is amortized simulation-based inference with
[BayesFlow](https://github.com/bayesflow-org/bayesflow).

This page is the handover for anyone joining the work. It says what the tutorial teaches, how the repository is
organized and why, how to set up, how to change a slot, how a number reaches the stage, where the work stands on
28 October 2026, and what is still open. Read it once end to end before your first change.

## What the tutorial teaches

A desk calibrates a volatility model to the day's option quotes, then marks products that nobody quotes: a
forward start, a long-dated put, a cliquet. The calibration is an estimate, so every mark inherits its uncertainty.
The posterior states that uncertainty, and pricing a product under each posterior draw turns it into a price range,
the range the prudent-valuation rules ask for. For a model priced only by simulation, one exact posterior costs
hours; a network trained once on simulated smiles returns it in seconds, and can be checked on thousands of
simulated smiles before anyone trusts it on a real one.

The afternoon works on one data set, the SPY option smile of 21 October 2026 (63 prices on 7 maturities × 9
moneyness points), and three models of increasing cost:

1. **Black–Scholes** (slot 1): one parameter, a posterior on a grid in a millisecond, and a model the smile rejects.
2. **Heston** (slot 4): four parameters and an exact likelihood by Fourier integral, so every step of the network
   can be checked against the exact posterior.
3. **Rough Bergomi** (slot 5, the advanced example): priced only by Monte Carlo, so the network is checked by
   simulation-based calibration and by an audit of each smile.

Slots 2 and 3 explain the machine between them; slot 6 changes the prior and the noise model and tests the model.

## The critical path: Heston, then rough Bergomi

Slot 4 builds the whole BayesFlow workflow on Heston, where an exact likelihood checks every step. Slot 5 keeps the
smile and the code and changes the simulator to rough Bergomi. Everything in `lab/` serves these two examples. The
earlier routes (SVJ and returns, HKDE jumps, the multifractal model, GBM) are archived, and sparse option chains are
a separate unit that starts after the release.

| Slot | Title | Presenter | Min | Notebook (`tutorial/notebooks/`) | State on 28 October |
|---|---|---|---:|---|---|
| 1 | The problem: Bayes for volatility models | Francesco | 15 | `slot1_intro.py` | Drafted; runs offline |
| 2 | What is amortized Bayes? | Alex | 15 | `slot2_amortized_bayes.py` | Drafted; runs offline |
| 3 | Network architectures | Stefan | 10 | `slot3_architectures.py` | Drafted; runs offline |
| 4 | A worked example in BayesFlow: Heston on the SPY smile | Stefan | 20 | `slot4_heston_bayesflow.py` | Drafted; loads the BayesFlow 2.0 Heston network until the 2.1 candidate is exported |
| 5 | Rough Bergomi on the SPY smile: from the posterior to a reserve | Francesco, Alex | 40 | `slot5_calibration.py` | Drafted on pilot results; the full rough-Bergomi network and studies are pending |
| 6 | Changing priors and dealing with misspecified models | Stefan | 15 | – | Curriculum only; no notebook yet |

The slot files in [`tutorial/curriculum/`](tutorial/curriculum/00-overview.md) give each slot's minute map, handoffs,
open decisions and build list. Dates: slot 4 freezes on Friday 30 October; slots 1–3 on Wednesday 4 November;
slots 5 and 6 on Friday 6 November; full timed run-through on Monday 9 November; release tag on Wednesday 11
November. After a freeze, fixes only.

## How the repository is organized

```
course/      The course: sixteen lectures, a superset of the tutorial, with the scripts that draw their
             figures.
tutorial/    What participants install and run: the slot notebooks, their runtime (src/sbi4f_tutorial),
             the frozen networks and results they load (assets/), their tests, and the organizers' curriculum.
lab/         What produces the numbers: simulators and pricers, calibration banks, training, references,
             studies, quick checks, Oscar jobs, tests and plans.
AGENTS.md    Rules for coding agents and people: the boundaries below and the MCMC policy.
```

The boundaries are what keep the tutorial safe to change late and easy to hand to participants:

1. **The tutorial depends on nothing else in the repository.** Notebooks import only `sbi4f_tutorial` and
   third-party packages, and read only `tutorial/assets/`. The tutorial has its own environment
   (`tutorial/pyproject.toml`, `tutorial/pixi.lock`). Its tests fail on any import of the lab, any `sys.path` edit
   and any path that climbs out of `tutorial/`.
2. **Only files cross from the lab to the tutorial, through lab tools.** Trained networks and accepted study
   results enter `tutorial/assets/` through `lab/studies/accept.py`; quick-check outputs, task files and figures as
   copies made by `lab/studies/manifest.py`; the exact Heston posterior on SPY from `lab/checks/heston_spy_exact.py`.
   The one exception is slot 2's toy network, which the notebook trains and saves when the file is missing.
   [`tutorial/MANIFEST.md`](tutorial/MANIFEST.md) lists every asset with its SHA-256 and source. Nobody copies files
   into `tutorial/assets/` by hand.
3. **What the notebooks compute live is copied, not shared.** `tutorial/src/sbi4f_tutorial` (about 1,700 lines)
   holds the lab functions the notebooks run (the pricers, the network code, the loaders and figures), copied
   unchanged into modules of the same names. About 120 lines are the tutorial's own: slot 1's toy example
   (`why_bayes.py`, moved from the lab because only the tutorial uses it), where the assets are, and reading the
   tasks the lab stored. `lab/tests/test_tutorial_parity.py` fails when a copy differs from the lab's
   code or returns different numbers, so the two cannot drift silently.
4. **The course is documents.** Its figures are drawn by `course/scripts/` in the lab environment and committed
   as images; readers need nothing installed.
5. **A number goes on screen only from [`tutorial/CLAIMS.md`](tutorial/CLAIMS.md)**, which `lab/studies/claims.py`
   generates from the accepted results. A pilot value is never presented as a result. Every live cell has a cached
   fallback.

```
lab ──accept.py, manifest.py: files + SHA-256──▶ tutorial/assets ──read by──▶ tutorial/notebooks
lab ──course/scripts draw figures──▶ course/images        course ◀──links only──▶ tutorial
lab/tests/test_tutorial_parity.py ──reads──▶ tutorial/src  (tests only; the tutorial never reads the lab)
```

## Set up

Install [pixi](https://pixi.sh) once (`curl -fsSL https://pixi.sh/install.sh | sh`), then clone the repository.
There are two environments, each defined in its own folder; run `pixi` from that folder.

```bash
cd tutorial
pixi install --locked
pixi run test            # independence, the asset manifest, the runtime; about 10 s
pixi run slot4           # opens slot 4 in the marimo editor
```

```bash
cd lab
pixi install --locked
pixi run test            # library, studies and the tutorial parity tests; about 2 minutes
```

Both environments pin BayesFlow 2.0.14, the version the shipped networks were trained with. The tutorial is
locked for Linux, macOS on Apple silicon and Windows; Windows is not yet tested. Draft
[PR #48](https://github.com/fmuia/sbi4finance/pull/48) adds the lab's BayesFlow 2.1 environments (`core`,
`core-gpu`, `core-reference` with PyMC). The tutorial's pin moves to 2.1 together with the networks, once they are
accepted and exported.

## Working on a slot

1. Read the slot's file in `tutorial/curriculum/` and the lectures of `course/` it draws on.
2. Branch from `main`, for example `git switch -c slot4/sampler-figure`.
3. Edit live: `cd tutorial && pixi run slot4`, or `pixi run edit` for all notebooks. To pair with a coding agent
   on the live kernel, use the marimo-pair skill in `.claude/skills/marimo-pair/`.
4. Inside a notebook, import only `sbi4f_tutorial` and third-party packages, and read files only under
   `ROOT / "assets"`. Put every number on screen through `tutorial/CLAIMS.md`: its study, its file and its status.
   Give every live cell (loading a network, pricing, training) a cached fallback and a time budget.
5. If a notebook needs a new computation, write it in the lab with a test, then copy the functions unchanged into
   the module of the same name in `tutorial/src/sbi4f_tutorial/`: in `lab/`, run
   `pixi run copy-to-tutorial --add MODULE.NAME` (`COPIED_FROM` in `lab/studies/copy_to_tutorial.py` maps each
   module to its lab file). Code only a notebook needs, such as a figure, goes in the notebook, or in the runtime
   with an entry in `WRITTEN_FOR_THE_TUTORIAL` there saying why. When a copied lab function changes,
   `pixi run test` in `lab/` fails and names it, and `pixi run copy-to-tutorial` copies it again.
6. If a notebook needs a new number or figure, produce it in the lab, accept it into `tutorial/assets/` with
   `lab/studies/accept.py`, then run `pixi run manifest` and `pixi run claims` in `lab/` (next section).
7. Before opening a pull request, run in `tutorial/`: `pixi run test`, `pixi run test-notebooks` (every notebook
   headless, a few minutes) and `pixi run check` (marimo lint). If you touched the lab, run `pixi run test` and
   `pixi run release-check` in `lab/` as well.
8. Open one pull request per slot change and ask the slot's presenter to review it.

## Producing numbers in the lab

The pipeline from a task to the stage, with details in [`lab/README.md`](lab/README.md):

1. **Task and bank.** A task configuration in `lab/configs/tasks/` fixes the contracts, the quote noise, the prior
   and the pricing numerics. `lab/scripts/calibration_bank.py` simulates the bank of priced smiles, locally or as
   an Oscar array (`lab/cluster/calibration_bank_array.sbatch`).
2. **Network.** `lab/studies/train_calibration.py` trains the amortized posterior, on Oscar with
   `lab/cluster/train_calibration.sbatch`.
3. **Checks.** Studies S1 to S9 in `lab/studies/`: simulation-based calibration and coverage, the exact Heston
   check, the neural-pricer route, the audit, the classical route, products and reserves, costs. Each study writes
   a new directory.
4. **Acceptance.** `lab/studies/accept.py` copies an accepted run or checkpoint into `tutorial/assets/` with
   `ACCEPTED.json`. Then `pixi run manifest` and `pixi run claims` refresh `tutorial/MANIFEST.md` and
   `tutorial/CLAIMS.md`.

The runbook of the Oscar runs is [`lab/plans/CLUSTER-RUNS.md`](lab/plans/CLUSTER-RUNS.md), and the next campaign is
planned in [`lab/plans/NUMERICAL-EXPERIMENT-PLAN.md`](lab/plans/NUMERICAL-EXPERIMENT-PLAN.md) and its
[protocol](lab/plans/EXPERIMENT-PROTOCOL.md). PyMC is preferred for reference MCMC where it fits naturally
([MCMC policy](lab/plans/mcmc-policy.md)). Banks, model payloads and full posterior draws stay on Oscar; git
records their identities and how they were produced.

## The course

[`course/README.md`](course/README.md) is its index: sixteen lectures from options and Black–Scholes to rough
Bergomi on SPY and a thin chain. It is a superset of the tutorial: the presenters' long version, and take-home
reading for participants. [`course/status.md`](course/status.md) gives the state of every study it cites. To redraw
a lecture's figures, run its script in the lab environment, for example
`cd lab && pixi run python ../course/scripts/lecture08.py`. Every script runs there except the last figure of
`lecture09.py`, which needs scikit-learn; the environment has never included it. `pixi run course-check` checks the
numbers the course plan prints against the stored results.

## Where the work stands on 28 October 2026

| Piece | State | Where |
|---|---|---|
| Heston network | A BayesFlow 2.1 candidate (100,000 training smiles, epoch 500) passed its frozen calibration check on 2,000 unused smiles, and 12 of 12 variance-product checks. It is not yet in the notebooks, which load the BayesFlow 2.0 network trained on 50,000 smiles. | [PR #48](https://github.com/fmuia/sbi4finance/pull/48) (draft) |
| Rough Bergomi, known curve (H, η, ρ given the 7-level forward-variance curve) | 20,000-smile bank complete, first networks trained. Comparisons with reference posteriors still disagree on some smiles, partly from the reference emulator's error and the time grid. The next step is a 50,000-smile bank and refits. | PR #48 |
| Rough Bergomi, inferred curve (six unknowns) | 9,925 of 20,000 training smiles generated; paused while the known-curve checks run. | PR #48 |
| Notebooks of slots 1 to 5 | Run headless and offline in the tutorial environment. Slot 5 shows the pilot's results, with "pending" cards where the full study is missing. | `tutorial/` |
| Notebook of slot 6 | Not started. | `tutorial/curriculum/06-slot6-priors-and-misspecification.md` |
| Sparse option chains (EWG) | Deferred until after the release; its code stays in the lab, marked deferred. | [`lab/plans/SPARSE-QUOTE-CASE-PLAN.md`](lab/plans/SPARSE-QUOTE-CASE-PLAN.md) |

The detailed numerical status of the backbone work is `docs/collaborator-numerical-status.md` on PR #48.

## Open decisions

1. **Slot 5's rough-Bergomi task on 14 November:** the known-curve task (three unknowns) or the inferred-curve task
   (six). Not decided yet; the numerical results above decide it. The slot-5 notebook follows the known-curve pilot.
2. **Agreed:** every network shipped to participants moves to BayesFlow 2.1: the Heston candidate once it is
   packaged, the rough-Bergomi network, and slot 2's toy network, retrained. The tutorial's pin moves with them.
3. **Agreed:** the take-home returns track (SVJ, the multifractal model, GBM from returns, crash-day priors, betas
   across stocks) is not part of the tutorial's release (tagged on 11 November). It stays in the archive and on its branches.
4. **Open:** how much run evidence lives in git. Compact records only (README, summary JSON, key figures), or the
   raw per-case outputs and logs too. PR #48 currently commits about 41 MB of them.
5. **Agreed in principle:** participants get a separate small repository holding `tutorial/` without the
   curriculum, created at the release. It is self-contained, so the copy runs without changes, but six links need
   a target first: the "Evidence and code" footers of the five notebooks point to `tutorial/curriculum/` and
   `course/`, and `tutorial/README.md` points participants to this repository, which is private. Either the course
   is published with the release and the links point there, or the links go.

## Open pull requests

- [PR #47](https://github.com/fmuia/sbi4finance/pull/47) and [PR #48](https://github.com/fmuia/sbi4finance/pull/48)
  (Alex): incremental banks and the BayesFlow 2.1 backbone. Both were written against the layout before the
  restructure; [`lab/plans/RESTRUCTURE.md`](lab/plans/RESTRUCTURE.md) gives the path map and the conflicts a trial
  merge found.
- [PR #37](https://github.com/fmuia/sbi4finance/pull/37), [PR #38](https://github.com/fmuia/sbi4finance/pull/38),
  [PR #40](https://github.com/fmuia/sbi4finance/pull/40), [PR #41](https://github.com/fmuia/sbi4finance/pull/41) and
  [PR #42](https://github.com/fmuia/sbi4finance/pull/42): drafts of the earlier returns and SVJ route, superseded by
  the current curriculum.

## History and the archive

Until 28 October the repository also held the simulator-ladder milestones (GBM, Heston from returns, the HKDE
options route, one-factor Bergomi, the first rough-Bergomi inference pilot), the SVJ and multifractal applications,
their checkpoints, sixty-five handover and review documents and 90 MB of results. They are off the critical path
and were removed. Everything is unchanged at commit
[`574b940`](https://github.com/fmuia/sbi4finance/tree/574b940087cd66a7540c5aa080f64e8b33010314), the last commit of
`main` before the restructure, and links in the documents point there. To read one file, run
`git show 574b940:docs/svj-pipeline.md`; to browse everything, run
`git worktree add ../sbi4finance-archive 574b940`.

## Who to ask

- **Francesco Muia:** owner; the curriculum, the course, slots 1 and 5.
- **Alex (Alexander Fengler):** the numerical backbone (banks, training, references), Oscar, slots 2 and 5.
- **Stefan:** the BayesFlow architecture, slots 3, 4 and 6.
- **Jerry Huang:** role to be agreed (curriculum decision D18).

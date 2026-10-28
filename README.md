# sbi4finance

Material for the ICAIF '26 tutorial **Amortized Bayesian Calibration of Volatility Models** (Milan, 14–15 November
2026). The repository holds three things: a sixteen-lecture course, the notebooks presented on the day, and the lab
that produces every number they show. Inference is amortized simulation-based inference with
[BayesFlow](https://github.com/bayesflow-org/bayesflow).

This page explains what the repository contains, the principles behind its structure, and how to use each part.
Each area has its own guide as well: [`course/README.md`](course/README.md), [`tutorial/README.md`](tutorial/README.md)
and [`lab/README.md`](lab/README.md). [`AGENTS.md`](AGENTS.md) states the rules for every change, by a person or a
coding agent.

## Contents

1. [What the tutorial is about](#what-the-tutorial-is-about)
2. [The repository at a glance](#the-repository-at-a-glance)
3. [Principles](#principles)
4. [Getting started](#getting-started)
5. [Using the tutorial](#using-the-tutorial)
6. [Using the lab](#using-the-lab)
7. [Using the course](#using-the-course)
8. [Contributing](#contributing)
9. [Conventions](#conventions)
10. [Where to find what](#where-to-find-what)
11. [People](#people)

## What the tutorial is about

A desk calibrates a volatility model to the day's option quotes, then marks products that nobody quotes: a
forward start, a long-dated put, a cliquet. The calibration is an estimate, so every mark inherits its uncertainty.
The posterior states that uncertainty, and pricing a product under each posterior draw turns it into a price range,
the range the prudent-valuation rules ask for. For a model priced only by simulation, one exact posterior costs
hours; a network trained once on simulated smiles returns it in seconds, and can be checked on thousands of
simulated smiles before anyone trusts it on a real one.

The tutorial works on one data set, the SPY option smile of 21 October 2026 (63 prices on 7 maturities × 9
moneyness points), with three models of increasing cost:

1. **Black–Scholes** (slot 1): one parameter, a posterior on a grid in a millisecond, and a model the smile rejects.
2. **Heston** (slot 4): four parameters and an exact likelihood by Fourier integral, so every step of the network
   can be checked against the exact posterior.
3. **Rough Bergomi** (slot 5, the advanced example): priced only by Monte Carlo, so the network is checked by
   simulation-based calibration and by an audit of each smile.

Slots 2 and 3 explain the machine between them; slot 6 changes the prior and the noise model and tests the model
itself. The critical path runs through Heston and rough Bergomi: slot 4 builds the whole BayesFlow workflow on
Heston, where an exact likelihood checks every step, and slot 5 keeps the smile and the code and changes the
simulator to rough Bergomi. The lab exists to serve these two examples.

| Slot | Title | Notebook in `tutorial/notebooks/` |
|---|---|---|
| 1 | The problem: Bayes for volatility models | `slot1_intro.py` |
| 2 | What is amortized Bayes? | `slot2_amortized_bayes.py` |
| 3 | Network architectures | `slot3_architectures.py` |
| 4 | A worked example in BayesFlow: Heston on the SPY smile | `slot4_heston_bayesflow.py` |
| 5 | Rough Bergomi on the SPY smile: from the posterior to a reserve | `slot5_calibration.py` |
| 6 | Changing priors and dealing with misspecified models | in preparation |

The [curriculum overview](tutorial/curriculum/00-overview.md) gives each slot's presenter, minutes and handoffs,
the state of the notebooks, the gates and dates, and the decisions taken and open. [`course/status.md`](course/status.md)
gives the state of every study and of the networks being trained.

## The repository at a glance

```
sbi4finance/
├── README.md                    this guide
├── AGENTS.md                    rules for every change; CLAUDE.md points coding agents to it
├── .claude/skills/              agent skills: cluster-setup (Oscar), marimo-pair (live notebooks)
│
├── course/                      the course: documents, read on GitHub or in any Markdown viewer
│   ├── README.md                index: the argument, the lectures, conventions, the symbols table
│   ├── 01-…16-*.md              sixteen lectures, from options and Black–Scholes to rough Bergomi on SPY
│   ├── status.md                the state of every study the course cites
│   ├── v2-plan.md               the course plan: what it argues, and which runs would strengthen it
│   ├── practice-sources.md      sourced statements about calibration practice
│   ├── forward-pricing-emulation.md  a companion note to Lecture 11 on neural pricers
│   ├── images/                  every figure of the lectures, committed
│   └── scripts/                 the scripts that draw the figures, and check_v2_plan.py for the plan's numbers
│
├── tutorial/                    what participants install and run; depends on nothing else here
│   ├── README.md                the participants' guide
│   ├── pyproject.toml, pixi.lock  its own environment
│   ├── notebooks/               the marimo notebooks, one per slot
│   ├── src/sbi4f_tutorial/      the code the notebooks run: copies of lab functions
│   ├── assets/                  every file the notebooks read: networks, accepted results, quick-check
│   │                            outputs, task files, data and figures
│   ├── MANIFEST.md              every asset with its size, SHA-256 and origin
│   ├── CLAIMS.md                every number said on stage, with its study and status
│   ├── tests/                   independence, asset hashes, the runtime, and a headless run of every notebook
│   └── curriculum/              the organizers' plan: the overview, one file per slot, the storyboard
│
└── lab/                         what produces the numbers
    ├── README.md                the lab's guide
    ├── pyproject.toml, pixi.lock  its environments: CPU, and GPU for the cluster
    ├── src/sbi4finance/         the library: simulators and pricers, and calibration/ (tasks, banks,
    │                            inference, checks, products, real chains)
    ├── configs/                 task and bank configurations
    ├── scripts/                 bank generation, and a real chain read onto the design grid
    ├── studies/                 studies S1 to S11, and the tools that copy results into the tutorial
    ├── checks/                  quick checks: computations behind the course, not accepted studies
    ├── cluster/                 Oscar jobs: lane discovery, bank arrays, training, a generic study job
    ├── tests/                   unit tests, and the tests that keep the tutorial's copies exact
    ├── plans/                   the numerical experiment plan and protocol, the cluster runbook, the MCMC policy
    └── data/SOURCES.md          where the raw option chains come from
```

## Principles

The structure answers one question: how can a small team keep changing a tutorial until the week it is given,
without breaking what participants install, and with every number on screen traceable to the run that produced it?
Each principle states the rule and the reason, and, where a test or a tool enforces it, which one.

**1. Three areas for three uses.** The course is read, the tutorial is run, the lab is developed. They have
different readers and change at different speeds: participants install only `tutorial/`, presenters study
`course/`, and the numerical work happens in `lab/`. Separate areas let the lab run experiments every day while the
tutorial stays stable, and let the tutorial be handed to participants as it is.

**2. The lab is the single source of computation.** Simulators, pricers, networks, studies and checks exist once,
in `lab/`, with tests. The tutorial and the course never recompute a result the lab produced: they read its output
files, and the course cites the study behind each number.

**3. The tutorial is self-contained.** Notebooks import only `sbi4f_tutorial` and third-party packages, and read
files only from `tutorial/assets/`. The tutorial has its own environment, so it installs and runs without the lab,
offline, on a laptop. *Enforced by* `tutorial/tests/test_independence.py`, which fails on any import of the lab, any
`sys.path` edit, any path that leaves `tutorial/`, and any asset whose size or SHA-256 differs from `MANIFEST.md`.

**4. Copies, not dependencies, and every copy is checked.** When a notebook needs lab code at run time, the tutorial
holds an exact copy of the lab's functions, under the same names, in modules named like the lab's. When it needs lab
results, it holds copies of the files. Tools make the copies, never hands:

| What the tutorial holds | Made by (run in `lab/`) | Checked by |
|---|---|---|
| Code, in `src/sbi4f_tutorial/` | `pixi run copy-to-tutorial` | `tests/test_tutorial_parity.py`: each copied function must be the lab's, statement for statement, and return the same numbers |
| Accepted study results and trained networks, in `assets/results/` and `assets/checkpoints/` | `pixi run accept`, which writes `ACCEPTED.json` | the hashes in `tutorial/MANIFEST.md`, checked by the tutorial's tests |
| Quick-check outputs, task files and figures | `pixi run manifest` | `pixi run release-check` |
| The exact Heston posterior on SPY, in `assets/data/` | `checks/heston_spy_exact.py` | the hashes in `tutorial/MANIFEST.md` |

A dependency would make participants install the lab, and would let a change in the lab alter the stage without
anyone noticing. A checked copy keeps the code identical and turns every change into a visible step. The code the
tutorial writes for itself (where the assets are, reading the task files, and slot 1's toy example) is listed by
name, with the reason, in `lab/studies/copy_to_tutorial.py`. The one file no lab tool writes is slot 2's toy network,
which the notebook trains and saves when it is missing.

**5. Every number is traceable.** Every result records its task's fingerprint, the SHA-256 of its training bank and,
where a network was used, the SHA-256 of the checkpoint. `ACCEPTED.json` records where an accepted run came from and
the hash of every source file. [`tutorial/CLAIMS.md`](tutorial/CLAIMS.md), generated from the accepted results,
lists every number said on stage with its study, its file and its status. Only a verified claim goes on screen; a
pilot value is labelled as one. A number still to be produced is written ⟨Sn⟩, naming the study that produces it.

**6. Regenerate rather than store.** A bank is regenerated exactly from its configuration and seed. Banks, model
payloads and full posterior draws therefore stay on the cluster, and git keeps their identities and the compact
records: a README, a summary JSON and the key figures of each run.

**7. The stage never waits.** Every live cell of a notebook has a cached fallback and a time budget. Nothing is
downloaded, and nothing is trained unless a button says so. The tutorial's tests run every notebook headless, end
to end.

**8. The course is documents.** Its figures are drawn by `course/scripts/` in the lab environment and committed, so
a reader needs nothing installed. Each lecture names the scripts that drew its figures and the studies behind its
numbers.

**9. Documents are maintained in place.** A plan or status page is updated, not followed by a new one; git keeps
the history.

How the parts connect:

```
lab/configs ─▶ banks ─▶ networks ─▶ studies, checks ── accept, manifest: files and SHA-256 ──▶ tutorial/assets ─┐
                                                                                                                ├─▶ notebooks
lab/src ──────────── copy-to-tutorial: exact copies, checked by tests ──────────────▶ tutorial/src ─────────────┘
lab environment ─▶ course/scripts ─▶ course/images ─▶ lectures
```

The course and the tutorial link to each other. Neither imports the other, and the tutorial never reads the lab.

## Getting started

You need git and [pixi](https://pixi.sh). pixi installs Python and every package from the lock files into the
folder of each area; nothing is installed system-wide.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

```bash
git clone https://github.com/fmuia/sbi4finance.git
cd sbi4finance
```

Each area that runs code has its own environment, defined in its folder; run `pixi` from that folder. Each block
below starts from the repository root.

```bash
cd tutorial
pixi install --locked
pixi run test            # independence, asset hashes and the runtime; about 10 seconds
```

```bash
cd lab
pixi install --locked
pixi run test            # the library, the studies and the tutorial's copies; about 2 minutes
```

The course needs nothing installed. Both environments use Python 3.12 and pin BayesFlow 2.0.14, Keras 3.15.1 and
JAX 0.11.1, the versions the shipped networks were trained with, and set `KERAS_BACKEND=jax`. The lab is locked for
Linux and macOS on Apple silicon; the tutorial also for Windows, which is not yet tested.

## Using the tutorial

```bash
cd tutorial
pixi run slot4           # open one notebook in the marimo editor: slot1 to slot5
pixi run edit            # open the notebook browser
pixi run marimo run notebooks/slot4_heston_bayesflow.py   # a notebook as a read-only app
```

The notebooks are [marimo](https://marimo.io) notebooks: plain Python files, reactive, readable in any editor. Each
reads its inputs through `sbi4f_tutorial.paths`, which points at `tutorial/assets/` (set `SBI4F_TUTORIAL_ASSETS` to
use another copy).

| Folder in `tutorial/assets/` | What it holds |
|---|---|
| `results/calibration/` | Accepted study results, one folder per model and study (`heston/s3/`, `rough-pilot/s5/`, `spy/`, `costs/`, `smile/`), each with its `ACCEPTED.json` |
| `checkpoints/` | Trained networks: `heston/` (the network, its task and its training record) and `slot2-toy/` |
| `quick-checks/` | Outputs of the lab's quick checks that the notebooks show |
| `tasks/` | Task files the lab stored, such as the rough-Bergomi task |
| `data/` | The exact Heston posterior on SPY |
| `images/` | Figures shared with the course and the curriculum |

The runtime, `tutorial/src/sbi4f_tutorial/`, holds the code the notebooks run:

| Module | What it does | Copied from |
|---|---|---|
| `surface.py` | Black–Scholes, implied volatility, the design grid and its quote noise | `lab/src/sbi4finance/calibration/surface.py` |
| `hkde.py`, `gbm.py` | The Heston Fourier pricer, with jumps switched off; Black's formula | `lab/src/sbi4finance/hkde.py`, `gbm.py` |
| `bergomi.py`, `rough_bergomi.py`, `pricing.py` | Rough Bergomi by Monte Carlo: paths, conditional puts, surface prices | the modules of the same names in `lab/src/sbi4finance/` |
| `tasks.py` | Reading a calibration task | `lab/src/sbi4finance/calibration/tasks.py` |
| `inference.py` | The network's adapter, its conditioning and its sampler | `lab/src/sbi4finance/calibration/inference.py` |
| `teaching.py`, `plots.py` | The notebooks' loaders, tables and figures, and the figure style | `lab/src/sbi4finance/calibration/teaching.py`, `svj/plots.py` |
| `bs_hierarchy.py`, `long_end.py` | The sector case and the long end, computed live | `lab/checks/` |
| `why_bayes.py`, `paths.py` | Slot 1's toy example; where the assets are | written for the tutorial |

Three checks cover the tutorial:

```bash
pixi run test             # independence, asset hashes and the runtime
pixi run check            # marimo's lint of every notebook
pixi run test-notebooks   # every notebook headless, end to end; a few minutes
```

## Using the lab

Run every lab command from `lab/`. The [lab's guide](lab/README.md) gives the details of each step.

| Environment | What | Install |
|---|---|---|
| `default` | CPU: BayesFlow on JAX, numba, SciPy | `pixi install --locked` |
| `gpu` | The same with JAX's CUDA plugin, for the cluster (Linux only) | `pixi install -e gpu` |

**From a task to the stage.**

1. **Task and bank.** A configuration in `lab/configs/tasks/` fixes the 63 contracts, the quote noise, the prior and
   the pricing numerics, and seeds the bank. `pixi run bank local --config CONFIG --out DIR` builds a small bank on
   one machine; the full banks run as cluster arrays (`cluster/calibration_bank_array.sbatch`).

   | Configuration | Model | Training / validation / test smiles | Seed |
   |---|---|---|---:|
   | `calibration_heston_bank.json` | Heston, jumps off | 50,000 / 5,000 / 5,000 | 2610 |
   | `calibration_rough_bank.json` | rough Bergomi | 100,000 / 5,000 / 7,000 | 2611 |
   | `calibration_rough_pilot.json` | rough Bergomi, pilot | 1,600 / 200 / 400 | 2612 |
   | `calibration_rough_minipilot.json` | rough Bergomi, a mini-pilot that tests a new prior | 1 / 1 / 198 | 2613 |

2. **Network.** `studies/train_calibration.py` trains the amortized posterior, drawing fresh quote noise in every
   batch; on the cluster with `cluster/train_calibration.sbatch`.
3. **Studies.** Each study answers one question and writes a new directory; none overwrites a result.
   [`studies/README.md`](lab/studies/README.md) gives each study's script, command and test rows.

   | Study | Question |
   |---|---|
   | S1 | The bank: its Monte Carlo error per quote against the budget, and its cost |
   | S2 | Calibration of the network: simulation-based calibration, coverage, recovery, contraction |
   | S3 | Heston: the network against the exact posterior |
   | S4 | The neural-pricer route: a network that learns prices, and MCMC on it |
   | S5 | The audit of each smile by importance sampling with the Monte Carlo likelihood |
   | S6 | The classical route: least squares, Gauss–Newton intervals, their coverage |
   | S7 | Products and reserves; Heston against rough Bergomi on the same smiles |
   | S8 | The real smile: SPY on the design grid, its posterior, audit and predictive check |
   | S9 | The cost of every route on one machine |
   | S10, S11 | The deferred sparse-chain unit: bounds on thin chains, and EWG's posterior |

4. **Into the tutorial.** `pixi run accept --run RUN --to results/calibration/heston/s2` copies an accepted run into
   `tutorial/assets/` with its `ACCEPTED.json`; `pixi run accept --checkpoint RUN --to checkpoints/heston` does the
   same for a network, with the optimizer state removed and same-seed draws checked identical. Then
   `pixi run manifest` refreshes the copied files and the hash table of `tutorial/MANIFEST.md`, and `pixi run claims`
   regenerates `tutorial/CLAIMS.md`.
5. **Verify.** `pixi run test` and `pixi run release-check` here; `pixi run test` and `pixi run test-notebooks` in
   `tutorial/`.

**The tools that move work into the tutorial**, all run from `lab/`:

| Command | What it does |
|---|---|
| `pixi run accept` | Copies an accepted study run or network into `tutorial/assets/`, with `ACCEPTED.json` |
| `pixi run manifest` | Refreshes the copies of quick-check outputs, task files and figures, then the hash table of `tutorial/MANIFEST.md` |
| `pixi run claims` | Regenerates the tables of `tutorial/CLAIMS.md` from the accepted results |
| `pixi run copy-to-tutorial` | Copies changed lab functions into the tutorial's runtime; `--add MODULE.NAME` copies a new one |
| `pixi run release-check` | Changes nothing; fails if any copy, the manifest or the claims are out of date |

**Quick checks.** `lab/checks/` holds computations that are not accepted studies: the exact Heston posterior on
SPY against Gauss–Newton, the local rough-Bergomi posterior, the sector case, the long end, the curve prior. The
course and the "why Bayes" material use them; [`checks/README.md`](lab/checks/README.md) gives their run order. A
number from a quick check reaches the stage only when a study reproduces it.

**The cluster.** Banks, training and the larger studies run on Brown's Oscar cluster. The `cluster-setup` skill in
`.claude/skills/` walks a new user through access, lane discovery (`pixi run discover-cluster --ssh-host oscar`),
pixi on the cluster and the first job, and [`lab/plans/CLUSTER-RUNS.md`](lab/plans/CLUSTER-RUNS.md) gives the exact
command of every run. Submit every job from `lab/`.

**Reference posteriors.** PyMC is preferred for reference MCMC where it fits naturally; the
[MCMC policy](lab/plans/mcmc-policy.md) lists the exceptions and what every chain must record.

**Deferred work.** The sparse-chain code (thin chains such as EWG) is kept working but not extended until its
follow-up is specified in [`lab/plans/SPARSE-QUOTE-CASE-PLAN.md`](lab/plans/SPARSE-QUOTE-CASE-PLAN.md).

## Using the course

Read it on GitHub, starting from its [index](course/README.md), which states the course's argument in four claims and
says which lectures serve each. The lectures build everything the advanced example needs, from option payoffs and
Black–Scholes to rough Bergomi on SPY and a thin chain. The course is a superset of the tutorial: the presenters' long
version, and take-home reading for participants.

- Each figure has a paragraph that says how to read it, and each lecture ends with the scripts that drew its
  figures and its references.
- A number written ⟨Sn⟩ waits for study Sn; [`course/status.md`](course/status.md) gives the state of each study.
- To redraw a lecture's figures, run its script in the lab environment, for example
  `cd lab && pixi run python ../course/scripts/lecture08.py`. The scripts write into `course/images/`. Every script
  runs there except the last figure of `lecture09.py`, which needs scikit-learn, not part of the environment.
- `cd lab && pixi run course-check` checks the numbers the course plan prints against the stored results.

## Contributing

**A change to a slot.**

1. Read the slot's file in `tutorial/curriculum/` and the lectures it draws on.
2. Branch from `main`, for example `git switch -c slot4/sampler-figure`.
3. Edit live with `cd tutorial && pixi run slot4`. To pair with a coding agent on the running notebook, use the
   `marimo-pair` skill in `.claude/skills/`.
4. Inside a notebook, import only `sbi4f_tutorial` and third-party packages, and read files only under
   `ROOT / "assets"`. Put every number on screen through `tutorial/CLAIMS.md`, and give every live cell (loading a
   network, pricing, training) a cached fallback and a time budget.
5. **A new computation** is written in the lab first, with a test, then copied into the module of the same name in
   the tutorial with `pixi run copy-to-tutorial --add MODULE.NAME`. Code that only a notebook needs, such as a
   figure, goes in the notebook, or in the runtime with an entry in `WRITTEN_FOR_THE_TUTORIAL` in
   `lab/studies/copy_to_tutorial.py` that says why.
6. **A new number or figure** is produced in the lab, accepted into `tutorial/assets/` with `pixi run accept`, and
   recorded with `pixi run manifest` and `pixi run claims`.
7. Run the checks below, then open one pull request per slot change and ask the slot's presenter to review it.

**A change to lab code that the tutorial copies.** `pixi run test` in `lab/` fails and names every copy that now
differs. Run `pixi run copy-to-tutorial`, then the tutorial's checks: the copy changes what the notebooks compute.

**Checks before a pull request.**

| If you changed | Run in `tutorial/` | Run in `lab/` |
|---|---|---|
| A notebook or the tutorial's runtime | `pixi run test`, `pixi run check`, `pixi run test-notebooks` | `pixi run test` |
| Lab code | `pixi run test`, and `pixi run test-notebooks` if a copied function changed | `pixi run test`, `pixi run release-check` |
| A study result, a network or a quick check | `pixi run test` | `pixi run release-check` |
| The course | – | `pixi run course-check`, and the lecture's script if a figure changed |

**Coding agents** follow [`AGENTS.md`](AGENTS.md); `CLAUDE.md` points to it. The skills in `.claude/skills/`
cover the cluster and live notebooks.

## Conventions

- **Forward units.** Option prices are divided by the discount factor and by the forward, so the forward is 1, a
  price is a fraction of it, and rates and dividends drop out.
- **The design grid.** 63 contracts: 7 maturities (5, 10, 21, 42, 63, 126 and 252 business days) × 9 standardized
  moneyness points $\delta = k / (\sigma_{\text{ref}}\sqrt{T}) \in \lbrace -2, -1.5, \dots, 2 \rbrace$, with
  $\sigma_{\text{ref}} = 0.20$ and $k$ the log-moneyness. Puts at and below the forward, calls above.
- **Time.** Maturities are business days over 252. Volatility is annualized and written as a fraction (0.20) or a
  percentage (20%); one volatility point is 0.01.
- **Quote noise.** Gaussian, with a standard deviation in implied volatility that rises linearly in $|\delta|$ from
  0.3 volatility points at the money to 1.0 at $|\delta| = 2$, turned into price units by each contract's vega at
  $\sigma_{\text{ref}}$.
- **The models' parameters.** Heston infers $v_0$, $\theta$, $\sigma_v$ and $\rho$ with $\kappa = 2$; rough Bergomi
  infers $H$, $\eta$ and $\rho$ given a forward-variance curve of 7 levels. The [symbols table](course/README.md#symbols)
  collects every symbol.
- **Studies** are named S1 to S11 ([`lab/studies/README.md`](lab/studies/README.md)); each uses fixed rows of the test
  split, so no two studies share a smile. ⟨Sn⟩ marks a number study Sn will produce.
- **Claims** carry a status in `tutorial/CLAIMS.md`: only **verified** ones go on screen; **pilot** values come from
  the 1,600-smile rough-Bergomi pilot bank and are labelled as such.
- **Records.** An accepted result carries `ACCEPTED.json` (its source, the SHA-256 of every source file, every
  change made); `tutorial/MANIFEST.md` lists every asset with its size and SHA-256.
- **Documents and runs.** Plans and status pages are updated in place. Commit compact run records; banks, model
  payloads and full posterior draws stay on the cluster, identified by their SHA-256.

## Where to find what

| To … | Read |
|---|---|
| Understand the tutorial's plan, slot by slot, its dates and its decisions | [`tutorial/curriculum/00-overview.md`](tutorial/curriculum/00-overview.md) and the slot files beside it |
| Install and run the notebooks as a participant | [`tutorial/README.md`](tutorial/README.md) |
| See every number said on stage and its status | [`tutorial/CLAIMS.md`](tutorial/CLAIMS.md) |
| See where each file in `tutorial/assets/` came from | [`tutorial/MANIFEST.md`](tutorial/MANIFEST.md) |
| Learn the theory behind a slot | [`course/README.md`](course/README.md) |
| Know which studies are done and which are pending | [`course/status.md`](course/status.md) |
| Produce or rerun a result | [`lab/README.md`](lab/README.md), [`lab/studies/README.md`](lab/studies/README.md) |
| Run on the cluster | `.claude/skills/cluster-setup/`, [`lab/plans/CLUSTER-RUNS.md`](lab/plans/CLUSTER-RUNS.md) |
| See the next numerical campaign | [`lab/plans/NUMERICAL-EXPERIMENT-PLAN.md`](lab/plans/NUMERICAL-EXPERIMENT-PLAN.md) and its [protocol](lab/plans/EXPERIMENT-PROTOCOL.md) |
| Write a reference chain | [`lab/plans/mcmc-policy.md`](lab/plans/mcmc-policy.md) |
| Know the rules for a change | [`AGENTS.md`](AGENTS.md) |

## People

- **Francesco Muia:** owner; the curriculum, the course, slots 1 and 5.
- **Alexander Fengler (Alex):** the numerical backbone (banks, training, references) and the cluster; slots 2 and 5.
- **Stefan:** the BayesFlow architecture; slots 3, 4 and 6.
- **Jerry Huang:** role to be agreed (curriculum decision D18).

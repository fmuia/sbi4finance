# sbi4finance

Material for the ICAIF '26 tutorial **Amortized Bayesian Calibration of Volatility Models** (Milan, 14–15 November
2026): a sixteen-lecture course, the notebooks presented on the day, and the lab that produces every number they
show. Inference uses [BayesFlow](https://github.com/bayesflow-org/bayesflow).

The tutorial calibrates volatility models to the SPY option smile of 21 October 2026 (63 prices on 7 maturities × 9
moneyness points) and prices, under the posterior, products the smile does not quote. The models are Black–Scholes
(slot 1), Heston (slot 4: exact likelihood, so the network is checked against the exact posterior) and rough
Bergomi (slot 5: Monte Carlo only, checked by simulation-based calibration and a per-smile audit). The critical
path is Heston, then rough Bergomi.

| Slot | Title | Notebook in `tutorial/notebooks/` |
|---|---|---|
| 1 | The problem: Bayes for volatility models | `slot1_intro.py` |
| 2 | What is amortized Bayes? | `slot2_amortized_bayes.py` |
| 3 | Network architectures | `slot3_architectures.py` |
| 4 | A worked example in BayesFlow: Heston on the SPY smile | `slot4_heston_bayesflow.py` |
| 5 | Rough Bergomi on the SPY smile: from the posterior to a reserve | `slot5_calibration.py` |
| 6 | Changing priors and dealing with misspecified models | in preparation |

The current numerical release has 100k training surfaces and completed baseline fits for Heston, known-curve
rough Bergomi and joint-curve rough Bergomi. The new cached examples are synthetic; historical SPY results
retain their own provenance. Start with the [data and training guide](lab/plans/training-data-guide.md)
and [current work order](lab/plans/backbone-delivery-work-order.md). Conditional rough references and
downstream-product checks remain scoped follow-ups.

Presenters, dates and decisions: [`tutorial/curriculum/00-overview.md`](tutorial/curriculum/00-overview.md). State
of the studies and of the training runs: [`course/status.md`](course/status.md).

## Layout

```
course/            the lectures (Markdown), their figures (images/) and the scripts that draw them (scripts/)
tutorial/          what participants install; its own environment
  notebooks/       one marimo notebook per slot
  src/sbi4f_tutorial/  the code the notebooks run: copies of lab functions
  assets/          everything the notebooks read: networks, accepted results, quick-check outputs, data
  MANIFEST.md      every asset with its SHA-256 and origin
  CLAIMS.md        every number said on stage, with its study and status
  curriculum/      the organizers' plan: overview, one file per slot
lab/               what produces the numbers; its own environments (CPU, and GPU for the cluster)
  src/sbi4finance/ simulators, pricers, calibration (tasks, banks, inference, checks, products)
  configs/         task and bank configurations
  studies/         studies S1–S11, and the tools that copy results into the tutorial
  checks/          quick checks behind the course; not accepted studies
  cluster/         Oscar jobs
  plans/           experiment plan and protocol, cluster runbook, MCMC policy
AGENTS.md          rules for every change, for people and coding agents
.claude/skills/    cluster-setup (Oscar), marimo-pair (live notebooks)
```

Guides: [`course/README.md`](course/README.md), [`tutorial/README.md`](tutorial/README.md) (for participants),
[`lab/README.md`](lab/README.md).

## Principles

- **The tutorial depends on nothing else in the repository**, so participants install only `tutorial/`. Notebooks
  import only `sbi4f_tutorial` and third-party packages, and read only `tutorial/assets/`. Enforced by
  `tutorial/tests/test_independence.py`.
- **Computation lives in the lab.** The tutorial holds exact copies of the lab functions it runs, under the same
  names, made by `pixi run copy-to-tutorial`. `lab/tests/test_tutorial_parity.py` fails when a copy's code or
  numbers differ from the lab's. The little code written for the tutorial alone is listed, with the reason, in
  `lab/studies/copy_to_tutorial.py`.
- **Files enter `tutorial/assets/` only through lab tools** (`accept`, `manifest`, `checks/heston_spy_exact.py`),
  with their SHA-256 in `tutorial/MANIFEST.md`, which the tutorial's tests check. Slot 2's explicit toy
  training stays in memory for that session and does not write assets.
- **Every number on screen is in `tutorial/CLAIMS.md`**, with its study, file and status. Only verified claims go on
  screen; pilot values are labelled. ⟨Sn⟩ marks a number that study Sn will produce.
- **The stage never waits.** Every live cell has a cached fallback and a time budget; nothing is downloaded, and
  nothing trains unless a button says so.
- **Large payloads stay outside Git.** Versioned banks and pretrained models are available through the
  [collaborator data guide](lab/plans/training-data-guide.md), with GitHub and Drive mirrors and SHA-256 checksums.
  Git holds compact evidence and teaching caches. Existing prices are reused rather than regenerated.
- **Course figures are committed**, drawn by `course/scripts/` in the lab environment.
- **Plans and status pages are updated in place.**

## Setup

Install [pixi](https://pixi.sh). The tutorial and the lab each have their own environment; run `pixi` from their
folder.

```bash
cd tutorial && pixi install --locked && pixi run test   # about 10 seconds
```

```bash
cd lab && pixi install --locked && pixi run test        # about 2 minutes
```

The participant environment and the lab's `core`/`core-gpu` profiles use Python 3.12, BayesFlow 2.1.0,
Keras 3.15.1 and JAX 0.11.1. The lab default BF2.0.14 profile remains for archived legacy studies; use
`pixi run --locked -e core ...` for the current backbone. Locks cover Linux/macOS, and the tutorial also
Windows (not yet tested). The written course needs nothing installed.

## Tutorial

```bash
cd tutorial
pixi run slot4             # one notebook in the marimo editor: slot1 to slot5
pixi run edit              # the notebook browser
pixi run test              # independence, asset hashes, the runtime
pixi run check             # marimo's lint
pixi run test-notebooks    # every notebook headless; a few minutes
```

Notebooks find their inputs through `sbi4f_tutorial.paths`; `SBI4F_TUTORIAL_ASSETS` points them at another copy.

## Lab

Run every command from `lab/`; `pixi install -e gpu` adds JAX's CUDA plugin for the cluster.

1. **Bank.** A configuration in `configs/tasks/` fixes the contracts, the quote noise, the prior, the pricing
   numerics and the seed. `pixi run bank local --config CONFIG --out DIR` builds a small bank; full banks run as
   cluster arrays.
2. **Network.** `studies/train_calibration.py`, or `cluster/train_calibration.sbatch` on Oscar.
3. **Studies.** Each writes a new directory; commands in [`studies/README.md`](lab/studies/README.md).
4. **Into the tutorial.** `pixi run accept`, then `pixi run manifest` and `pixi run claims`.
5. **Verify.** `pixi run test` and `pixi run release-check`, then the tutorial's checks.

| Configuration | Model | Training / validation / test smiles | Seed |
|---|---|---|---:|
| `calibration_heston_bank.json` | Heston, jumps off | 50,000 / 5,000 / 5,000 | 2610 |
| `calibration_rough_bank.json` | rough Bergomi | 100,000 / 5,000 / 7,000 | 2611 |
| `calibration_rough_pilot.json` | rough Bergomi, pilot | 1,600 / 200 / 400 | 2612 |
| `calibration_rough_minipilot.json` | rough Bergomi, test of a new prior | 1 / 1 / 198 | 2613 |

| Study | Question |
|---|---|
| S1 | The bank's Monte Carlo error per quote, and its cost |
| S2 | The network's calibration: SBC, coverage, recovery, contraction |
| S3 | Heston: the network against the exact posterior |
| S4 | A neural pricer, with MCMC on it |
| S5 | The audit of each smile by importance sampling |
| S6 | Least squares and Gauss–Newton intervals, and their coverage |
| S7 | Products and reserves; Heston against rough Bergomi |
| S8 | SPY: posterior, audit, predictive check |
| S9 | The cost of every route on one machine |
| S10, S11 | Thin chains, deferred |

| Command | What it does |
|---|---|
| `pixi run accept` | Copies an accepted run or network into `tutorial/assets/`, with its `ACCEPTED.json` |
| `pixi run manifest` | Refreshes the copied quick-check outputs, task files and figures, and `tutorial/MANIFEST.md` |
| `pixi run claims` | Regenerates `tutorial/CLAIMS.md` from the accepted results |
| `pixi run copy-to-tutorial` | Refreshes the tutorial's copies of lab functions; `--add MODULE.NAME` adds one |
| `pixi run release-check` | Fails if a copy, the manifest or the claims are out of date; changes nothing |

- **Quick checks** (`checks/`) back the course and slot 1; a number from one reaches the stage only when a study
  reproduces it. Run order: [`checks/README.md`](lab/checks/README.md).
- **Cluster:** Brown's Oscar. The `cluster-setup` skill covers access and the first job;
  [`plans/CLUSTER-RUNS.md`](lab/plans/CLUSTER-RUNS.md) has the command of every run.
- **Reference MCMC:** PyMC where it fits; the exceptions are in the [MCMC policy](lab/plans/mcmc-policy.md).
- **Thin chains** (EWG): kept working, not extended until
  [`plans/SPARSE-QUOTE-CASE-PLAN.md`](lab/plans/SPARSE-QUOTE-CASE-PLAN.md) specifies the follow-up.

## Course

Start from [`course/README.md`](course/README.md). To redraw a lecture's figures, run its script from `lab/`, for
example `pixi run python ../course/scripts/lecture08.py`; it writes to `course/images/`. The last figure of
`lecture09.py` needs scikit-learn, which the environment lacks. `pixi run course-check` in `lab/` checks the
numbers of the course plan against the stored results.

## Contributing

- Branch from `main`, and open one pull request per change, reviewed by the slot's presenter.
- New computation: write it in the lab with a test, then copy it with `pixi run copy-to-tutorial --add MODULE.NAME`.
- New number or figure: produce it in the lab, then run `accept`, `manifest` and `claims`.
- Changed lab code that the tutorial copies: `pixi run test` in `lab/` names the stale copies; run
  `pixi run copy-to-tutorial`, then the tutorial's checks.
- To pair with a coding agent on a running notebook, use the `marimo-pair` skill.

| If you changed | Run in `tutorial/` | Run in `lab/` |
|---|---|---|
| A notebook or the runtime | `test`, `check`, `test-notebooks` | `test` |
| Lab code | `test`, and `test-notebooks` if a copied function changed | `test`, `release-check` |
| A result, a network or a quick check | `test` | `release-check` |
| The course | – | `course-check`, and the lecture's script if a figure changed |

Each entry is `pixi run` followed by the name.

## Conventions

- Prices are in forward units: divided by the discount factor and the forward.
- The design grid has 63 contracts: 7 maturities (5, 10, 21, 42, 63, 126, 252 business days) × 9 points
  $\delta = k / (\sigma_{\text{ref}}\sqrt{T}) \in \lbrace -2, -1.5, \dots, 2 \rbrace$, with $\sigma_{\text{ref}} = 0.20$
  and $k$ the log-moneyness. Puts for $\delta \le 0$, calls above.
- Time is business days over 252; one volatility point is 0.01.
- Quote errors are independent Gaussian in price. Their standard deviations use reference Black–Scholes vega
  times a volatility-equivalent scale rising from 0.3 volatility points at the money to 1.0 at the wings.
- Heston infers $v_0$, $\theta$, $\sigma_v$ and $\rho$ with $\kappa = 2$. Rough Bergomi infers $H$, $\eta$ and
  $\rho$ given a forward-variance curve of 7 levels. Symbols: [course/README.md](course/README.md#symbols).
- Current development uses training and validation. Freeze the model and criteria before reserved-test evaluation;
  historical registered studies retain their recorded split provenance.

## People

- **Francesco Muia:** the curriculum and the course; slots 1 and 5.
- **Alexander Fengler (Alex):** banks, training, reference posteriors and the cluster; slots 2 and 5.
- **Stefan** and **Jerry Huang:** the BayesFlow architecture; slots 3, 4 and 6.

# Amortized Bayesian calibration of volatility models: the notebooks

The notebooks of the ICAIF '26 tutorial (Milan, 14–15 November 2026). A desk calibrates a volatility model to the
day's option quotes and marks products that nobody quotes; the notebooks show how a posterior states the
uncertainty of that calibration, how a network trained on simulations returns the posterior in seconds, and how
to check it. Current cached examples open offline from `assets/` without loading a neural backend.
Fresh model sampling, financial simulations and training are explicit actions.

## Install

With [pixi](https://pixi.sh) (recommended):

```bash
cd tutorial
pixi install --locked
pixi run --locked test # asset hashes, runtime and inference input boundaries
```

The participant environment uses **BayesFlow 2.1.0** and is locked for Linux, macOS on Apple silicon,
and Windows (not runtime-tested here). Archived 2.0 checkpoints remain historical evidence; they are
never loaded implicitly in the 2.1 environment.

With pip, on Python 3.12, for example in a hosted notebook:

```bash
cd tutorial
pip install -e .
export KERAS_BACKEND=jax
```

## Run or review an existing lesson

```bash
pixi run --locked slot1 # opens slot 1 in the marimo editor, in your browser; likewise slot2 ... slot5
pixi run --locked edit  # all notebooks
pixi run --locked backbones # cached 100k examples and explicit native model sampling
```

`pixi run --locked marimo run notebooks/slot4_heston_bayesflow.py` opens a notebook as a read-only app.

| Notebook | Slot | What you do | Computed live |
|---|---|---|---|
| `slot1_intro.py` | 1. The problem: Bayes for volatility models | Meet the afternoon's objects on SPY's quotes: prior, likelihood and posterior for Black–Scholes, the range of a price, a check of the model; see when the posterior is not the classical ellipse, and what it costs | Grid posteriors on one quote; cached Heston and rough-Bergomi posteriors (seconds) |
| `slot2_amortized_bayes.py` | 2. What is amortized Bayes? | Watch a small network learn slot 1's posterior and check it against the exact answer | Exact grid by default; a new 2.1 toy trains only after pressing the button |
| `slot3_architectures.py` | 3. Network architectures | See how flow matching carries noise to Heston's posterior, and what the sampler's steps change | Cached Heston posterior and PyMC reference |
| `slot4_heston_bayesflow.py` | 4. A worked example in BayesFlow: Heston on the SPY smile | Follow the whole workflow, from simulator to checks, against the exact posterior | Cached 100k example; simulation and a small native fit are optional buttons |
| `slot5_calibration.py` | 5. Rough Bergomi on the SPY smile: from the posterior to a reserve | Read rough Bergomi's posterior, its audit, and the price ranges and reserves of products | Cached known/joint-curve examples; Monte Carlo is an explicit action |

`backbone_review.py` adds the current Heston, known-curve rough and joint-curve rough estimators.
The rough examples include validation:0 and three explicitly selected joint-geometry illustrations.
They retain development scope; conditional rough references and product validation remain pending.
Slots 4 and 5 expose these cached examples alongside the historical SPY material. The historical SPY
neural comparison awaits a separately checked 2.1 update. Slot 6 is still in preparation.

Start by reviewing the cached example, its diagnostics and its financial interpretation. No cluster connection
or full bank is needed. For fresh samples, obtain the models archive from the
[training-data guide](../lab/plans/training-data-guide.md) and extract it under
`assets/local-checkpoints/`. Enter the matching native checkpoint directory in the review notebook
and press its live button. The loader checks its task, model SHA256 and sampler settings against the
selected cached example. Model payloads are excluded from git and the shipped asset manifest;
their own native hashes are checked on loading. No bank is needed for pretrained inference.

Optional authoring tools inherit the same 2.1 stack:

```sh
pixi run --locked -e collaboration collaborate notebooks/backbone_review.py
pixi run --locked -e collaboration studio-create  # create a new lesson scaffold once
pixi run --locked -e collaboration studio-build
```

This preserves marimo Pair/MCP, Studio 0.3 and Lens 0.2.3. The old SVJ-specific Studio view is not
reused for the new notebook. Live server sessions and the new Studio scaffold need a separate
interactive review; they were not executed during this integration.

## Author or change a lesson

Start with the [editing map](../README.md#start-editing-here) and the slot's plan in `curriculum/`.
The main content files are `notebooks/slot4_heston_bayesflow.py` and `notebooks/slot5_calibration.py`;
`curriculum/00-overview.md` fixes scope and responsibilities. Edit the explanation, figures and learning sequence
alongside that plan. First reconcile the slot 4/5 plans and prose with the complete-surface Heston → known-curve
rough → joint-curve rough sequence: retain historical SPY provenance and keep thin-chain extensions optional.
State the model, curve conditioning, quote-noise law and evidence status explicitly.

Put new numerical work in the lab. Use its existing `copy-to-tutorial` tool for runtime functions and its
publication tools for assets; [the lab README](../lab/README.md#from-lab-code-to-a-lesson) gives the workflow.
Do not make the participant package import `sbi4finance`, edit `sys.path`, or read lab files. Do not hand-copy a
checkpoint into the shipped assets or edit the generated manifest/claim tables. The ignored
`assets/local-checkpoints/` download cache is the explicit hash-checked exception for optional live loading.

Run these checks from `tutorial/` before a PR:

```bash
pixi run --locked test
pixi run --locked check
pixi run --locked test-notebooks
```

These are validation commands for authors, not actions executed by this handover. Interactive Pair/MCP/Studio
sessions still need their stated review. Runtime success does not promote a development example to an accepted
scientific claim; retain the original reference methods, chain/draw axes, task and checkpoint hashes.

## What is in this folder

| Path | What |
|---|---|
| `notebooks/` | The marimo notebooks: plain Python files, reactive, readable in any editor |
| `src/sbi4f_tutorial/` | The code the notebooks run, copied unchanged from the project's lab: Black–Scholes and implied volatility (`surface.py`), the Heston Fourier pricer (`hkde.py`), rough Bergomi by Monte Carlo (`rough_bergomi.py`, `bergomi.py`, `pricing.py`), the current network's adapter and native sampler (`full_surface.py`), and the notebooks' loaders and figures (`teaching.py`) |
| `assets/` | The trained networks, the accepted study results, quick-check outputs, task files, data and figures that the notebooks load |
| `MANIFEST.md` | Every shipped asset, with its size, SHA-256 and origin; excludes downloaded `local-checkpoints/` |
| `CLAIMS.md` | Every number said on stage, with the study that produced it and its status |
| `tests/` | `pixi run test` checks the files and the code; `pixi run test-notebooks` runs every notebook headless (a few minutes) |

## Where the numbers come from

Stage claims come from the accepted studies in `CLAIMS.md`. The current development replays are
labelled separately and do not promote their models to accepted status. A pilot value is labelled as one and
never quoted as a result. The simulations, training runs and studies behind them, and a sixteen-lecture course with
the full theory, are in the project repository, [fmuia/sbi4finance](https://github.com/fmuia/sbi4finance): the
course in `course/`, the code that produced every asset in `lab/`.

## If something does not work

- **Keras complains about its backend:** set `KERAS_BACKEND=jax`. pixi sets it; with pip, export it before
  starting marimo.
- **The first network call is slow:** it compiles once per session, in a few seconds.
- **A cell says "pending":** the result it shows is not final yet. The notebook says which study produces it.

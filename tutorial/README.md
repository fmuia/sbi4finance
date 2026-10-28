# Amortized Bayesian calibration of volatility models: the notebooks

The notebooks of the ICAIF '26 tutorial (Milan, 14–15 November 2026). A desk calibrates a volatility model to the
day's option quotes and marks products that nobody quotes; the notebooks show how a posterior states the
uncertainty of that calibration, how a network trained on simulations returns the posterior in seconds, and how
to check it. Everything runs on a laptop, offline, from the networks and results in `assets/`. Nothing is
downloaded, and nothing is trained unless you press a button that says so.

## Install

With [pixi](https://pixi.sh) (recommended):

```bash
cd tutorial
pixi install --locked
pixi run test          # about 10 seconds: the files, the code, and one network loaded and sampled
```

The environment is locked for Linux and macOS on Apple silicon, and for Windows (still being tested).

With pip, on Python 3.12, for example in a hosted notebook:

```bash
cd tutorial
pip install -e .
export KERAS_BACKEND=jax
```

## Run

```bash
pixi run slot1         # opens slot 1 in the marimo editor, in your browser; likewise slot2 ... slot5
pixi run edit          # all notebooks
```

`pixi run marimo run notebooks/slot4_heston_bayesflow.py` opens a notebook as a read-only app.

| Notebook | Slot | What you do | Computed live |
|---|---|---|---|
| `slot1_intro.py` | 1. The problem: Bayes for volatility models | Calibrate Black–Scholes to SPY quotes as inference, compare the posterior with the classical interval, price what the quotes do not pin | Grid posteriors, a toy smile, a sector of names (seconds) |
| `slot2_amortized_bayes.py` | 2. What is amortized Bayes? | Watch a small network learn slot 1's posterior and check it against the exact answer | Loads a small network; training it live takes about 30 seconds |
| `slot3_architectures.py` | 3. Network architectures | See how flow matching carries noise to Heston's posterior, and what the sampler's steps change | Loads the Heston network |
| `slot4_heston_bayesflow.py` | 4. A worked example in BayesFlow: Heston on the SPY smile | Follow the whole workflow, from simulator to checks, against the exact posterior | Loads the Heston network; training a small one live takes about a minute |
| `slot5_calibration.py` | 5. Rough Bergomi on the SPY smile: from the posterior to a reserve | Read rough Bergomi's posterior, its audit, and the price ranges and reserves of products | One rough-Bergomi smile by Monte Carlo (seconds) |

Every computed cell has a cached fallback, so a notebook still runs if a live computation is slow or fails.

## What is in this folder

| Path | What |
|---|---|
| `notebooks/` | The marimo notebooks: plain Python files, reactive, readable in any editor |
| `src/sbi4f_tutorial/` | The code the notebooks run: Black–Scholes and implied volatility, the Heston Fourier pricer, a rough-Bergomi Monte Carlo pricer, the network's adapter and sampler. Short, and meant to be read |
| `assets/` | The trained networks, the accepted study results, quick-check outputs, data and figures that the notebooks load |
| `MANIFEST.md` | Every file in `assets/`, with its size, SHA-256 and origin |
| `CLAIMS.md` | Every number said on stage, with the study that produced it and its status |
| `tests/` | `pixi run test` checks the files and the code; `pixi run test-notebooks` runs every notebook headless (a few minutes) |

## Where the numbers come from

Every number on screen comes from an accepted study listed in `CLAIMS.md`. A pilot value is labelled as one and
never quoted as a result. The simulations, training runs and studies behind them, and a sixteen-lecture course with
the full theory, are in the project repository, [fmuia/sbi4finance](https://github.com/fmuia/sbi4finance): the
course in `course/`, the code that produced every asset in `lab/`.

## If something does not work

- **Keras complains about its backend:** set `KERAS_BACKEND=jax`. pixi sets it; with pip, export it before
  starting marimo.
- **The first network call is slow:** it compiles once per session, in a few seconds.
- **A cell says "pending":** the result it shows is not final yet. The notebook says which study produces it.

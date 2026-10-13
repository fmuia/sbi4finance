# SVJ pipeline: from 1,000 daily returns to VaR and ES

This branch implements the end-to-end pipeline of the design document *SVJ advanced example: technical design and narrative* (10 Oct 2026) in `src/sbi4finance/svj`, trains the reference network on CPU, and re-runs the document's feasibility checks against it. Results from the reference training run are being added; until then this page describes the design and how each piece is checked.

## The daily process

Every step is one function call; `pipeline.run_window` chains them and `scripts/svj_daily.py` runs them on a window of the bundled CSV.

1. **Network input.** The last 1,000 daily log returns in percent, as two channels per day: the return and log(y² + 0.01) (`model.network_input`).
2. **Amortized posterior.** The summary network compresses the 1,000 × 2 input into 32 statistics; 50 Euler steps of the flow-matching network carry noise to 1,000 joint draws of the eight parameters and log spot volatility at the last day (`amortized.AmortizedPosterior`, compiled once).
3. **Misspecification test.** The MMD distance between the window's summaries and those of 2,000 simulated series, with a p-value from 500 more (`checks.MisspecificationTest`).
4. **Audit.** A Student t (5 degrees of freedom, scale × 1.2) fitted to the draws proposes 3,000 parameter vectors; each is weighted by prior × particle-filter likelihood / proposal density, and the share of effective draws is the audit score (`audit.audit`). A score below the threshold flags the window, which can go to particle MCMC (`pmmh.run_pmmh`).
5. **Forecast.** 20 paths of 10 days from every posterior draw give the predictive 1- and 10-day 99% VaR and 97.5% ES, beside the plug-in forecast at the posterior median, the posterior of the 1-day conditional VaR and the split of its variance between today's variance and the parameters (`risk.forecast`).

Training happens once, before any of this: `scripts/svj_train.py` simulates every batch fresh from the prior (50 epochs × 400 batches × 128 series).

## Code layout

| Module | What it holds | How it is checked |
|---|---|---|
| `model.py` | Prior in risk-manager coordinates, the daily Euler model with floor and Bernoulli jumps, forward paths, network input | Prior table and prior predictive statistics reproduce the design document; unit tests |
| `filtering.py` | Fully adapted particle filter (JAX, vectorized over parameter vectors) and the grid filter | The mean of L-hat matches the converged grid likelihood; noise falls with particles |
| `amortized.py` | BayesFlow networks, online training, the compiled 50-step Euler sampler | The compiled sampler matches BayesFlow's own Euler integration to 4e-6 |
| `checks.py` | SBC with simultaneous bands, coverage, contraction, misspecification test | An exact posterior passes and an overconfident one fails; false-alarm rate on fresh simulations |
| `audit.py` | Importance sampling with the particle filter | Weights normalize; agreement study against particle MCMC |
| `pmmh.py` | Particle marginal Metropolis–Hastings, rank-normalized R-hat and bulk ESS | Diagnostics on independent, stuck and autocorrelated chains |
| `classical.py` | Approximate ML on the grid-filter likelihood (exact gradient by autodiff, L-BFGS-B, float64) and Wald intervals | Autodiff gradient against finite differences |
| `risk.py` | Predictive and plug-in VaR / ES, conditional VaR, variance split | Normal closed forms; the mixture example of Appendix H.2; the forward-simulation figure of the document |
| `pipeline.py`, `data.py`, `plots.py` | One window end to end; return windows from CSV; figures | Run on VOO and SPMO |

`pixi run test-svj` runs the 19 unit tests in `tests/test_svj.py` in under a minute.

## Results

Pending: the reference network (2.56 million series) is training; validation, agreement with particle MCMC, backtests and the real-data runs follow it.

## Deviations from the design document

| Area | Design document | This branch | Why |
|---|---|---|---|
| Filters | numba (`pf_numba.py`) | JAX: `vmap` over parameter vectors, `lax.scan` over days, float32 | No new dependency: JAX is already the Keras backend. Systematic resampling uses offspring counts and the XLA bit generator, 30–40% faster than the first version |
| Python | 3.13 | 3.12 | The repository's pinned environment; BayesFlow, Keras and JAX versions are the same |
| SBC ranks | Fractional ranks | Randomized ranks (k + U) / (L + 1) | Exactly uniform for an exact posterior; removes false alarms of discrete ranks at the ends of the band |
| Audit threshold | Open decision | 2% efficiency (`--threshold`) | The study flagged windows at 0.2%, 1.5% and 1.6% and passed 3.3% and above |
| Real data | S&P 500 1999–2018 (arch package), WTI to Oct 2026 | VOO and SPMO, 2020 to Sep 2026, from `data/` | The repository's pinned CSV; the document's series are not in the repository |
| Hardware | 2-core cloud VM | 4-core cloud container | Costs are reported as process CPU seconds, which count all threads, and as wall time |

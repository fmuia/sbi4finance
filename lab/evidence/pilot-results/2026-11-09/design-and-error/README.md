# Observation design and error checks · 9 November 2026

Evidence behind [the option-application plan](../../../../plans/OPTION-APPLICATION-PLAN.md).
These are quick checks run in a scratch environment (Laplace/Fisher approximations
and least-squares fits), not production studies or accepted assets. Key numbers
are in [`summary.json`](summary.json).

The noise-reassessment session of the same day (Claude Code, "Tutorial observation
noise reassessment") reached related results with exact likelihood fits and test
networks. Those numbers are quoted in the plan, are marked there as coming from that
session, and are not reproduced here.

## Data

Raw quote files are not committed (see [`lab/data/SOURCES.md`](../../../../data/SOURCES.md)).

| File | Cboe timestamp | Last trade | SHA-256 |
| --- | --- | --- | --- |
| SPY delayed quotes | 2026-11-08 23:53:35 | 2026-11-06 16:00:00 (close) | `81ac26c80ffddb75ccd4eb459f793e28470b5027a6100daf79000be7bdad4429` |
| SPX delayed quotes, intraday | 2026-11-09 16:50:16 | 2026-11-09 11:35:14 | `43dee488077cfde008590cd6eaa2b6db2efb6f511955da614cb5fe7e71d13bce` |
| SPX delayed quotes, close | 2026-11-09 23:20:45 | 2026-11-09 16:14:59 | `209765afc08b1261330f93e81d0b29c551051774f4c2931196f5381fe32fa8a1` |

The SPX close was fetched for the plan and is not analysed here. The checks also
read the tutorial assets `results/calibration/spy/surface.json`,
`data/heston-spy-exact.json` and `backbones/rough-joint100k-validation0/task.json`.
Rates for the live chains use the tutorial's 20 October curve (3-month 4.26%,
2-year 4.79%); the forwards come from parity, so the effect is small.

Environment: Python 3.13, NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.1 in a plain
virtual environment (pixi was not available). The scripts import the tutorial
runtime and the lab package from this checkout by path.

## Files

| File | What it produced |
| --- | --- |
| `heston/common.py` | Shared helpers: grids, the noise conventions, Heston Fourier prices |
| `heston/test_heston_fisher.py` | Heston Laplace widths for the four reviewer grids under the design rule and a consistent vol-point noise, at SPY's 21 October posterior median, the prior centre and 150 prior draws |
| `heston/test_spy_live.py SPY.json [xi_max]` | SPY 6 November read onto the current and proposed grids with the lab's chain code; Heston fits (κ = 2) and variance-swap gaps |
| `heston/eff_noise_dates.py SPY.json` | Effective vol-point noise of the design rule on SPY, 21 October and 6 November |
| `heston/spx_noise_and_model_error.py SPX.json` | SPX half-spreads by moneyness and maturity; Heston (κ free) model error in implied-vol space and its correlation |
| `heston/grid_vs_error_structure.py` | Heston widths on SPX under bid–ask-only, independent and smooth model error, for current, denser and wider grids |
| `heston/region_roles.py` | Which region of the smile carries unique information on each parameter |
| `heston/corr_length_sensitivity.py` | The same comparison for larger and smoother model error |
| `heston/layout_choice.py` | Candidate 63-contract layouts; layout E is the one the plan adopts |
| `heston/*.json` | Small derived outputs (no raw quotes) used by the scripts above |
| `rough/rough_fisher.py`, `analyze.py`, `tables.py`, `timing.py`, `results.json` | Rough Bergomi (joint-curve task) Fisher information for the reviewer grids with common random numbers, 32,768 paths per block |
| `claims/check_grid.py`, `claims/heston_info.py` | Checks of the reviewer's claims against the S8 surface and the exact Heston posterior |

Run from the script's own folder, for example `python -I layout_choice.py`
inside `heston/`. The scripts that read a raw chain take its path as an argument.

## Caveats

- One or two dates per conclusion; SPX model error was measured for Heston only.
- Laplace widths are local, flat-prior approximations. On the current grid they match the
  committed exact Heston posterior on SPY to within 4%.
- The smooth-error results assume a squared-exponential correlation with fixed
  length scales; their robustness was checked over the ranges in `summary.json`.
- Rough Bergomi widths under the design rule on the shifted grids are unreliable
  (Monte Carlo error 40–65 times that noise); the vol-point results are stable to 3%.

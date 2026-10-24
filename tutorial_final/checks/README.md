# Quick checks of 23 and 24 October 2026

Scripts and results behind the "why Bayes" material of slots 1, 4 and 5, and behind the course's version-2
plan (`docs/course/v2-plan.md`). They are quick checks, not
accepted studies: nothing here is registered in `CLAIMS.md`, and a number moves to the stage only once a
study in `tutorial_final/studies/` reproduces it.

| Check | Script | Output | Question |
|:---|:---|:---|:---|
| Heston on SPY | `heston_checks.py`, `heston_sparse_free_long.py` | `heston_checks.json`, `heston_checks-*.npy`, `heston_checks.png` | Where does the exact posterior differ from Gauss–Newton? Full grid, 36 short-dated quotes, a 3 x 5 grid; kappa fixed at 2 and free in [0.25, 8]; a profile in kappa; product ranges against the delta method |
| Rough Bergomi on SPY | `rough_checks.py` | `rough_checks.json`, `rough_checks.png` | The local (Laplace) posterior at the least-squares fit, with xi0 known or uncertain (5 to 20% per level), on three quote sets |
| Black–Scholes sector | `bs_hierarchy.py` | `bs_hierarchy.json`, `bs_hierarchy.png` | Marks for illiquid names: per-name least squares, the desk's proxy, empirical Bayes and hierarchical Bayes |
| The long end | `long_end/long_end.py`, `long_end/long_end_extra.py` | `long_end/long_end.json`, `long_end/long_end_extra.json`, `long_end/long_end.png` | Long-dated SPX-style products under rough Bergomi when xi0 beyond the LEAPs is an assumption: the desk's range against the posterior's |
| Rough Bergomi on SPY, exact | `rough_spy_is.py`, `rough_spy_is_figure.py` | `rough_spy_is/rough_spy_is_seed*.json`, `.npz`, `rough_spy_is.png` | The posterior by importance sampling against Gauss–Newton, on one pricer with common random numbers, two seeds |
| Rough Bergomi when the quotes say less | `rough_spy_is_cases.py`, `rough_spy_is_cases_figure.py` | `rough_spy_is/rough_spy_is_cases.json`, `.npz`, `rough_spy_is_cases.png` | The same on 15 sparse quotes, and on 63 quotes with three times the noise |
| Desk case: front months | `desk_case/` (`desk_case.py`, `desk_case_round2.py`, `validate.py`, `real_market.py`, `timing.py`) | `desk_case/*.json`, `*.npz`, `*.png` | One-year products marked from SPY's front three months by Gauss–Newton, the profile, the desk convention H = 0.10 and the posterior; breach rates on 80 simulated thin chains; the per-draw costs for planning |
| A thin chain | `thin_chains/ewi_chain.py` | `thin_chains/ewi_2026-10-22.json` (derived data only) | EWI's chain of 22 Oct 2026 in implied-volatility units: spreads, two-sided quotes, early-exercise premia, strips, parity |
| A curve prior for the full bank | `curve_prior/curve_prior.py` | `curve_prior/curve_prior.json`, `docs/course/images/v2-curve-prior.png` | The pilot's forward-variance prior against a power law with a log-time random walk, on SPY's curve (gate G0′) |

`figures.py` draws `rough_checks.png`, `heston_checks.png` and `bs_hierarchy.png`; `long_end/long_end_figure.py`
draws `long_end.png`. The Heston draws are thinned to 4,000 per case; the full-grid, kappa-fixed draws are
slot 4's exact posterior in `curriculum/data/heston-spy-exact.npy`. Run order: `rough_checks.py`,
`heston_checks.py` and `heston_sparse_free_long.py`, `bs_hierarchy.py`, then `long_end.py` and
`long_end_extra.py` from `long_end/`.

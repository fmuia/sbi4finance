# Quick checks of 23 and 24 October 2026

Scripts and results behind the "why Bayes" material of slots 1, 4 and 5, and behind the course's version-2
plan (`docs/course/v2-plan.md`). They are quick checks, not
accepted studies: nothing here is registered in `CLAIMS.md`, and a number moves to the stage only once a
study in `tutorial_final/studies/` reproduces it.

| Check | Script | Output | Question |
|:---|:---|:---|:---|
| Heston on SPY | `heston_checks.py`, `heston_sparse_free_long.py` | `heston_checks.json`, `heston_checks-*.npy`, `heston_checks.png` | Where does the exact posterior differ from Gauss–Newton? Full grid, 36 short-dated quotes, a 3 x 5 grid; kappa fixed at 2 and free in [0.25, 8]; a profile in kappa; product ranges against the delta method |
| Rough Bergomi on SPY | `rough_checks.py` | `rough_checks.json`, `rough_checks.png` | The local (Laplace) posterior at the least-squares fit, with xi0 known or uncertain (5 to 20% per level), on three quote sets |
| Black–Scholes sector | `bs_hierarchy.py` | `bs_hierarchy.json`, `bs_hierarchy.png` | Marks for illiquid names: per-name least squares, the proxy (the liquid names' mean), empirical Bayes and hierarchical Bayes |
| The long end | `long_end/long_end.py`, `long_end/long_end_extra.py` | `long_end/long_end.json`, `long_end/long_end_extra.json`, `long_end/long_end.png` | Long-dated SPX-style products under rough Bergomi when xi0 beyond the LEAPs is an assumption: flat extrapolation's range against the posterior's (the JSON keys say "desk" for flat extrapolation) |
| Rough Bergomi on SPY, exact | `rough_spy_is.py`, `rough_spy_is_figure.py` | `rough_spy_is/rough_spy_is_seed*.json`, `.npz`, `rough_spy_is.png` | The posterior by importance sampling against Gauss–Newton, on one pricer with common random numbers, two seeds |
| Rough Bergomi when the quotes say less | `rough_spy_is_cases.py`, `rough_spy_is_cases_figure.py` | `rough_spy_is/rough_spy_is_cases.json`, `.npz`, `rough_spy_is_cases.png` | **Withdrawn.** The same on 15 sparse quotes, and on 63 quotes with three times the noise; every method was given SPY's curve from the full chain |
| Front months | `desk_case/` (`desk_case.py`, `desk_case_round2.py`, `validate.py`, `real_market.py`, `timing.py`) | `desk_case/*.json`, `*.npz`, `*.png` | **Withdrawn as evidence; kept for the record and its costs.** SPY with its later quotes hidden: SPY's noise, and SPY's curve to one year given to every method. The 80-chain validation also had no per-chain k-hat check. Superseded by S10 and S11. Its notebook, `tutorial_final/notebooks/desk_case.py`, presented these results and was removed on 24 October 2026 (it is at commit cb404f7) |
| Thin chains | `thin_chains/thin_chain.py`, `thin_chains/settings.json` | `thin_chains/{EWI,EWG,EWQ}_2026-10-22.json`, `thin_chains/FEZ_2026-10-23.json` (derived data only; raw files: `data/SOURCES.md`) | Four thin chains in implied-volatility units: spreads, two-sided quotes, early-exercise premia, strips, parity; the summary used to choose the thin example |
| The EWG layout | `thin_chains/build_layout.py` | `configs/thin/ewg_layout.json`, `thin_chains/EWG_observation.json` | The design contracts EWG quotes and their noise (the thin network's layout), EWG read onto it, and EWG's cleaned quotes for the audit; carry forward, NYSE trading days, each quote's own vega; diagnostics of the parity forward, the clock and the vega |
| Costs on EWG | `thin_chains/timing.py` | `thin_chains/timing.json` | CPU-seconds of one audit evaluation on EWG's layout and quotes, and of the six thin products |
| A curve prior for the full bank | `curve_prior/curve_prior.py` | `curve_prior/curve_prior.json`, `docs/course/images/v2-curve-prior.png` | The pilot's forward-variance prior against a power law with a log-time random walk, on SPY's curve (gate G0′) |
| The mini-pilot | `scripts/calibration_bank.py local` with `configs/tasks/calibration_rough_minipilot.json`; `studies/bank_quality.py`; `studies/smile_facts.py prior-predictive` | `minipilot/bank-quality.json`, `numerical-error.png`, `prior-predictive.json`, `.png`, `spy-against-prior.json` | Gate G0′: the new prior's cost, its Monte Carlo quality, and SPY's smile statistics in its prior predictive |
| The grid bias | `studies/grid_bias.py` on 12 mini-pilot surfaces | `grid_bias_minipilot/grid-bias.json`, `grid-bias.png` | How far the bank's time grids are from grids twice as fine, in units of the quote noise |

`figures.py` draws `rough_checks.png`, `heston_checks.png` and `bs_hierarchy.png`; `long_end/long_end_figure.py`
draws `long_end.png`. The Heston draws are thinned to 4,000 per case; the full-grid, kappa-fixed draws are
slot 4's exact posterior in `curriculum/data/heston-spy-exact.npy`. Run order: `rough_checks.py`,
`heston_checks.py` and `heston_sparse_free_long.py`, `bs_hierarchy.py`, then `long_end.py` and
`long_end_extra.py` from `long_end/`.

# Quick checks of 23 and 24 October 2026

Scripts and results behind the "why Bayes" material of slots 1, 4 and 5, and behind the course's version-2
plan (`course/v2-plan.md`). They are quick checks, not
accepted studies: nothing here is registered in `tutorial/CLAIMS.md`, and a number moves to the stage only once a
study in `studies/` reproduces it. The outputs the notebooks show are copied to `tutorial/assets/quick-checks/`
by `studies/manifest.py`; rerun it after rerunning a check. Run the scripts from `lab/`, in the lab environment.

| Check | Script | Output | Question |
|:---|:---|:---|:---|
| Heston on SPY | `heston_checks.py`, `heston_sparse_free_long.py` | `heston_checks.json`, `heston_checks-*.npy`, `heston_checks.png` | Where does the exact posterior differ from Gauss–Newton? Full grid, 36 short-dated quotes, a 3 x 5 grid; kappa fixed at 2 and free in [0.25, 8]; a profile in kappa; product ranges against the delta method. `heston_sparse_free_long.py` reruns the 3 x 5 grid with kappa free on longer chains and replaces that case (slot 1 shows it) |
| Rough Bergomi on SPY | `rough_checks.py` | `rough_checks.json`, `rough_checks.png` | The local (Laplace) posterior at the least-squares fit, with xi0 known or uncertain (5 to 20% per level), on three quote sets |
| Black–Scholes sector | `bs_hierarchy.py` | `bs_hierarchy.json`, `bs_hierarchy.png` | Marks for illiquid names: per-name least squares, the proxy (the liquid names' mean), empirical Bayes and hierarchical Bayes |
| The long end | `long_end/long_end.py`, `long_end/long_end_extra.py` | `long_end/long_end.json`, `long_end/long_end_extra.json`, `long_end/long_end.png` | Long-dated SPX-style products under rough Bergomi when xi0 beyond the LEAPs is an assumption: flat extrapolation's range against the posterior's (the JSON keys say "desk" for flat extrapolation) |
| Rough Bergomi on SPY, exact | `rough_spy_is.py`, `rough_spy_is_figure.py` | `rough_spy_is/rough_spy_is_seed*.json`, `.npz`, `rough_spy_is.png` | The posterior by importance sampling against Gauss–Newton, on one pricer with common random numbers, two seeds |
| Thin chains (deferred sparse unit) | `thin_chains/thin_chain.py`, `thin_chains/settings.json` | `thin_chains/{EWI,EWG,EWQ}_2026-10-22.json`, `thin_chains/FEZ_2026-10-23.json` (derived data only; raw files: `data/SOURCES.md`) | Four thin chains in implied-volatility units: spreads, two-sided quotes, early-exercise premia, strips, parity; the summary used to choose the thin example |
| The EWG layout (deferred sparse unit) | `thin_chains/build_layout.py` | `configs/thin/ewg_layout.json`, `thin_chains/EWG_observation.json` | The design contracts EWG quotes and their noise (the thin network's layout), EWG read onto it, and EWG's cleaned quotes for the audit; carry forward, NYSE trading days, each quote's own vega; diagnostics of the parity forward, the clock and the vega |
| Costs on EWG (deferred sparse unit) | `thin_chains/timing.py` | `thin_chains/timing.json` | CPU-seconds of one audit evaluation on EWG's layout and quotes, and of the six thin products |
| A curve prior for the full bank | `curve_prior/curve_prior.py` | `curve_prior/curve_prior.json`, `course/images/v2-curve-prior.png` | The pilot's forward-variance prior against a power law with a log-time random walk, on SPY's curve (gate G0′) |
| The mini-pilot | `scripts/calibration_bank.py local` with `configs/tasks/calibration_rough_minipilot.json`; `studies/bank_quality.py`; `studies/smile_facts.py prior-predictive` | `minipilot/bank-quality.json`, `numerical-error.png`, `prior-predictive.json`, `.png`, `spy-against-prior.json` | Gate G0′: the new prior's cost, its Monte Carlo quality, and SPY's smile statistics in its prior predictive |
| Where the posterior and the classical answer part on complete surfaces (28 Oct) | `why_bayes/*.py` | `why_bayes/README.md` and its JSON and npz files | Heston's S6 smiles by fitted vol-of-vol; a local Gauss–Newton screen and an importance-sampling posterior of PR #48's joint-curve rough task beyond the last quote; SPY's curve against that task's curve family. Three scripts need PR #48's joint task |
| The grid bias | `studies/grid_bias.py` on 12 mini-pilot surfaces | `grid_bias_minipilot/grid-bias.json`, `grid-bias.png` | How far the bank's time grids are from grids twice as fine, in units of the quote noise |

`figures.py` draws `rough_checks.png`, `heston_checks.png` and `bs_hierarchy.png`; `long_end/long_end_figure.py`
draws `long_end.png`. The Heston draws are thinned to 4,000 per case; the full-grid, kappa-fixed draws are
slot 4's exact posterior in `tutorial/assets/data/heston-spy-exact.npy`, written by `heston_spy_exact.py`.
Run order: `heston_spy_exact.py`, `rough_checks.py`, `heston_checks.py` and `heston_sparse_free_long.py`, `bs_hierarchy.py`, then `long_end.py`
and `long_end_extra.py` from `long_end/`.

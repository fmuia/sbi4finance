# SPX notebook, Section 6: exact product prices (19 November 2026)

This record supports the correction of Section 6 of
[`lab/notebooks/heston_spx_calibration.py`](../../../../notebooks/heston_spx_calibration.py).
[`scripts/check_heston_spx_products.py`](../../../../scripts/check_heston_spx_products.py) wrote
`results.json` in 24 minutes on four CPU cores. It reads only committed records and runs no MCMC
or training. Prices are in volatility points: Black-equivalent forward-start volatility for the
forward-start call, max(S_6m/S_3m − 1, 0), and 100 E[√(I_T/T)] for the 3-month volatility swap.

## What was wrong

As presented on 15 November 2026, Section 6 priced 40 posterior draws per data set with
`products.heston_paths`: full-truncation Euler, 252 steps over six months, 4,096 antithetic
paths and common random numbers (seed 20261115).

1. **The Monte Carlo error exceeded the ranges.** Each forward-start price had a standard
   error of about 0.15 points; the 90% ranges were 0.07 (simulation) and 0.12 (SPX) wide. On
   the same 40 draws, Monte Carlo and exact prices correlate 0.32 on the simulation and 0.85
   on SPX, and the Monte Carlo ranges are 1.65 and 1.33 times too wide. The volatility swap
   also carried a time-step bias of +0.07 to +0.11 points, more than its range.
2. **The refit ratio rested on the prior bound c ≤ 2.** On SPX the posterior piles the noise
   multiplier at that bound. The Heston reference of the
   [Bergomi capstone](../../2026-11-14/bergomi-capstone/README.md), with c in [0.5, 50], gives
   c = 14.1 (90%: 12.2–16.6). Quote by quote, its 90% clean-price band is 7.17 times as wide as
   that of the presented posterior ([market predictives](../../../../notebooks/data/heston_market_predictives/README.md);
   interquartile range 6.86–7.35, range 6.39–7.66), and its median prices move by a median of
   0.27 half-spreads. The refit spreads, presented as 18 and 21 times the posterior range,
   shrink by about that factor when c is free.

## What the notebook does now

- It prices both products exactly with
  [`heston_exact_products`](../../../../src/sbi4finance/calibration/heston_exact_products.py).
  Given the variance at the reset, the forward start is a Heston call; integrating it over the
  exact (noncentral chi-square) law of that variance gives one Lewis integral with the CIR
  moment generating function. The volatility swap integrates the CIR Laplace transform of the
  integrated variance. Doubling the Gauss–Legendre nodes moves no price by more than 1e-10
  points. [`tests/test_heston_exact_products.py`](../../../../tests/test_heston_exact_products.py)
  checks the pricers against the tutorial runtime's characteristic function and vanilla price,
  quantile averages of the conditional call, direct integration over the transition density,
  the deterministic-variance limit, and the second-order expansion of the volatility swap.
- It prices every inside-prior posterior draw: all 2,000 per data set. Range ends then carry a
  sampling error of at most about 0.001 points (batch means), and are shown to two decimals.
- It states the refit multiples as conditional on c ≤ 2 and gives the figure for c free.

## Monte Carlo against the exact prices

2^20 antithetic paths per estimate, in 256 chunks with independent streams; the same streams at
every parameter set. Entries are Monte Carlo minus exact, ± one standard error.

| Parameter set | Product | Exact | 252 steps | 1,008 steps | 4,032 steps | As presented (SE) |
|---|---|---:|---:|---:|---:|---:|
| Market-like generating values | Forward start | 15.626 | +0.018 ± 0.010 | +0.007 ± 0.010 | −0.018 ± 0.010 | 15.682 (0.153) |
| | Volatility swap | 15.743 | +0.111 ± 0.005 | +0.021 ± 0.005 | +0.000 ± 0.005 | 15.790 (0.087) |
| SPX posterior median | Forward start | 15.877 | +0.014 ± 0.009 | +0.007 ± 0.009 | −0.015 ± 0.009 | 15.883 (0.140) |
| | Volatility swap | 16.126 | +0.100 ± 0.005 | +0.016 ± 0.005 | −0.001 ± 0.005 | 16.163 (0.083) |
| Refit to the 3 shortest expiries | Forward start | 17.895 | −0.000 ± 0.006 | +0.006 ± 0.006 | −0.007 ± 0.006 | 17.947 (0.095) |
| | Volatility swap | 17.626 | +0.074 ± 0.004 | +0.009 ± 0.004 | +0.001 ± 0.004 | 17.633 (0.067) |
| Refit to the 4 longest expiries | Forward start | 15.604 | +0.013 ± 0.009 | +0.008 ± 0.009 | −0.017 ± 0.009 | 15.697 (0.154) |
| | Volatility swap | 16.290 | +0.085 ± 0.005 | +0.015 ± 0.005 | −0.002 ± 0.005 | 16.349 (0.080) |

Steps are over six months, so the swap's three months see half of them. For the forward start,
every difference is within two standard errors: the Euler bias is below about 0.02 points at
each step count. The swap's bias falls from +0.07–0.11 points at 252 steps, the presented
setting, to zero within 0.002 at 4,032. Differences at one step count move together across
parameter sets because the random numbers are shared.

As an independent route for the forward start, Lewis integrals of the tutorial runtime's
characteristic function (Gauss–Legendre on [0, 25] and [25, 1500]), averaged over 20,000
midpoint quantiles of the exact law of the variance at the reset, differ from the exact prices
by −0.7e-4 to −1.0e-4 points: the error of the midpoint rule.

## Corrected figures

| | As presented (40 draws, Monte Carlo) | Exact, same 40 draws | Exact, all 2,000 draws |
|---|---:|---:|---:|
| Simulation, forward start, 90% range | 15.672–15.741 | 15.614–15.656 | 15.611–15.655 |
| Simulation, volatility swap, 90% range | 15.77–15.80 | 15.738–15.767 | 15.734–15.765 |
| Simulation, true values | 15.682, 15.79 | 15.626, 15.743 | 15.626, 15.743 |
| SPX, forward start, 90% range | 15.865–15.988 | 15.844–15.937 | 15.829–15.926 |
| SPX, volatility swap, 90% range | 16.14–16.21 | 16.100–16.160 | 16.092–16.159 |

Both true values lie inside the exact ranges. The four least-squares refits (c fixed) are
unchanged; their prices, as presented and exact:

| Refit to | Forward start | Volatility swap |
|---|---:|---:|
| All 63 quotes | 15.894 → 15.876 | 16.163 → 16.126 |
| The 3 shortest expiries | 17.947 → 17.895 | 17.633 → 17.626 |
| The 4 longest expiries | 15.697 → 15.604 | 16.349 → 16.290 |
| Quotes near the money | 16.457 → 16.461 | 16.210 → 16.209 |

On SPX the refits spread 2.29 points (forward start) and 1.50 points (swap). Divided by the
posterior 90% width, that is 23.7 and 22.4 with c ≤ 2 (presented: 18.3 and 21.1), and about
3.3 and 3.1 with c free, after dividing by the median band ratio of 7.17. The figures for c free
assume that product ranges widen like the quote bands, as they do to first order around a
nearly unchanged centre; the c-free posterior draws themselves are not public.

## Reproduce

```sh
cd lab
pixi run --locked -e core python scripts/check_heston_spx_products.py
```

`--paths-exponent` and `--steps` change the Monte Carlo budget; `--out` writes elsewhere.
`results.json` records the environment and the SHA-256 of the script, both pricers and the
three input records.

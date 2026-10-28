# Note on this result

`audit.json` was computed on 22 October 2026 with two errors in the audit likelihood, and its numbers must not be
used as final.

1. It used the variance σ_c² + s_c² at the Monte Carlo prices. That counts the Monte Carlo variance twice:
   averaged over the Monte Carlo error it is the likelihood of a noise variance σ_c² + 2 s_c².
2. It priced every proposal of a surface with the same seed (common random numbers). An unbiased likelihood
   estimate gives the exact posterior only if the Monte Carlo errors are independent across proposals; with one
   seed the weights converge to the posterior given that seed's error field, which is shifted (review of 24
   October 2026, finding M1).

The audit now prices each proposal on its own seed, at 32,768 paths, and uses N(y; P̂, diag(σ²) + C − S), with
S the proposal's Monte Carlo covariance and C the bank's covariance of the surface's reference prices
(`src/sbi4finance/calibration/correction.py`, `gaussian_loglik_mc`; `tutorial_final/studies/audit.py`). The
study is to be rerun on the full bank (`tutorial_final/CLUSTER-RUNS.md`).

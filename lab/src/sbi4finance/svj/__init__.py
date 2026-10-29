"""Amortized inference for a stochastic-volatility model with jumps (SVJ), carried into VaR and ES.

Modules, in pipeline order: `model` (prior, daily simulator, forward paths), `filtering` (particle
and grid filters), `amortized` (BayesFlow networks, training, compiled sampler), `checks`
(misspecification test, calibration metrics), `audit` (importance sampling with the filter),
`pmmh` (particle MCMC fallback), `risk` (predictive VaR and ES) and `pipeline` (one window, end to end).
"""

# Fixed-checkpoint Heston market prediction bands

This compact display record compares the actual63 `300k-300-final300` amortizer
with the regenerated PyMC reference used by the SPX notebook. It needs no model
bundle. Full posterior parameters remain private.

Each method contributes 256 deterministically selected paired physical/noise
rows. PyMC selection is balanced over its four stored chains. Price eligibility
uses the first five physical parameters and positive finite raw `c`; it does
not discard `c > 2`. The network's raw support violations remain visible in its
noise marginal and predictive calculation. No projection or parameter
reshuffling occurs. Clean and independently noisy pointwise 5/50/95% price
quantiles are saved in the original 63-contract order and raw price units.

These bands describe viewed calibration quotes, not held-out coverage. Their
finite prediction sample adds Monte Carlo variation. The notebook converts
price residuals to volatility-point equivalents with fixed reference vegas.

Reproduce from `lab/` using the preserved final300 conditional posterior payload:

```sh
python scripts/prepare_heston_market_predictives.py --conditional-draws /private/path/conditional-draws.npz
```

The generator verifies the private payload against the immutable evaluation,
including every parameter's raw support counts and the saved quantiles. It
also verifies the task, observation order, regenerated reference-cache identity
and checked numerical pricer. No training or MCMC runs.

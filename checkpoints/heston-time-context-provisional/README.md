# Provisional Heston checkpoint with explicit time context

This spline candidate **failed two conditional validation gates** on root 43022:
maximum posterior-mean error and maximum predictive-interval endpoint error.
The new final test was not evaluated. This is not an accepted tutorial or
market-forecasting posterior.

`task.json` records the same restricted physical-measure Heston task as the
previous spline experiment and adapter `heston_returns_time_and_window_moments_v2`.
The bank stores decimal log returns `[B,T,1]`; conditioning appends known end
times, and training-only global scaling keeps Time2Vec inputs on an order-one
scale. Use the saved adapter ID when calling `conditioning_data`. Frozen
evaluation through `scripts/train_heston.py --checkpoint ...` does this
automatically. Old checkpoints continue using v1.

The complete architecture and normalization are in `model.keras`. Versions:
BayesFlow 2.0.14, Keras 3.15.1, JAX 0.11.1. See the
[design and outcome](../../docs/heston-time-context.md),
[validation report](../../docs/results/heston-time-context/neural-validation.json)
and [activation diagnostic](../../docs/results/heston-time-context/summary-explicit-time.json)
for the fixed protocol, limitations and artifact/source hashes.

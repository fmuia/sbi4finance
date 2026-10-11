# Provisional Heston joint checkpoint

This is a reproducible **failed conditional-inference pilot**, not an accepted
tutorial or market forecasting checkpoint. It infers long-run variance and
signed current continuation state for the fixed synthetic daily Heston task
in `task.json`.

`model.keras` uses BayesFlow 2.0.14, Keras 3.15.1 and JAX 0.11.1. The task and
adapter identifier must match exactly. It takes raw decimal log returns and
observation-derived features; synthetic truth and future outcomes are never
conditioning. The loader restores the paired, raw-unit outputs and saved
training normalization.

The [guide](../../docs/heston-joint-inference.md) gives regeneration and frozen
evaluation commands. The [validation report](../../docs/results/heston-inference/neural-validation.json)
passed, but the [independent test report](../../docs/results/heston-inference/neural-test.json)
failed conditional mean and predictive-interval gates despite reasonable
marginal coverage. The test command intentionally exits nonzero after saving
those diagnostics. No weights or thresholds were changed after that test.

Model SHA-256:
`947ba17680d4b5acd25564bcd724f66cd91b27e4ae5574a5b3eec7eafaad1860`.
The reports also pin the bank, reference and source hashes. The 73 MB bank and
particle arrays are regenerated rather than stored in Git.

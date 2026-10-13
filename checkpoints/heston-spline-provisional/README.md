# Provisional spline Heston checkpoint

This candidate **failed development validation** on the first 16 reference
histories of the root-43022 bank. Its reserved final-test split was not evaluated.
Do not treat it as an accepted tutorial or market forecasting posterior.

The physical-measure task, adapter, GRU and observed features match the original
Heston pilot. The density estimator is a four-block spline coupling flow with
eight bins and 32-by-32 subnets. The full architecture and normalization are in
`model.keras`; `task.json` specifies the conditioning/target contract. BayesFlow
2.0.14, Keras 3.15.1 and JAX 0.11.1 were used.

See the [experiment design and result](../../docs/heston-inference-followup.md)
and [validation report](../../docs/results/heston-inference-followup/neural-validation.json),
which pins checkpoint/source/bank/reference hashes. The original flow-matching
failure and checkpoint remain available separately.

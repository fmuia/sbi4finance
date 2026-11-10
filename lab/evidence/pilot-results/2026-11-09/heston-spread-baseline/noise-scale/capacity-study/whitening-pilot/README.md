# Training-input whitening for Heston

**Predeclared pilot, 10 November 2026.** [PR #87](https://github.com/fmuia/sbi4finance/pull/87), pinned at `ddf3619faf48b8f4b0ee56ee841ef24124a508d1`, adds a notebook whose saved recovery plot shows c correlation .887. It fits a full-covariance ZCA transform to noisy training quotes divided by their half-spreads. It also changes the network to FlowMatching, dropout, target constraints, optimizer/schedule and dataset exposure. Its appended c-response diagnostic has no executed result. This pilot isolates the input transform within a new matched pair, while preserving bounded targets and the spline architecture.

## Which covariance is this?

For each dataset, the same five Heston parameters determine all 54 clean prices. Across training datasets, those parameters vary, making neighboring quotes move together. The covariance being whitened is across those prior-generated quote vectors. It is not a market-return covariance, Heston's Brownian correlation rho, or an estimated bid–ask error covariance.

Write `X_i = Y_i/h_i = P_i(phi)/h_i + c*Z_i`, with fixed price half-spreads h and conditionally independent standard-Normal Z. Across the training prior,

```
Cov(X) = Cov_phi(P(phi)/h) + E[c^2] I.
```

The first term contains correlations because one parameter vector drives the whole surface. Here `c ~ Uniform(.5,2)`, so `E[c^2]=1.75`. The common uncertain c couples error magnitudes, but creates no off-diagonal covariance of signed errors. Given phi and c, the original price-error covariance remains `c^2 diag(h_i^2)`.

Per-feature standardization rescales each quote independently, leaving correlated combinations badly scaled. Full whitening uses the sample mean mu and covariance C of noisy training X, with

```
C = Q diag(lambda) Q.T
W = Q diag(1/sqrt(max(lambda, max(lambda)*1e-8))) Q.T
U = (X - mu) W.
```

All 54 coordinates are retained. The positive eigenvalue floor limits amplification; it does not truncate components. If the floor is inactive, the sample training covariance of U is approximately I. Centering, fitting and matrix multiplication happen in host float64 before the final network float32 cast.

A financial interpretation is to expose both common surface moves and scaled contrasts between quotes. The new coordinates are numerical features, not individual option prices. The fixed invertible map preserves the original information and exact posterior. It changes neither Heston dynamics, priors, forwards/discounts, the half-spreads, nor c. It does not replace bid–ask weights with an empirical covariance inverse in the financial likelihood.

Crucially, transform the entire noisy vector. Its noise covariance in the new coordinates is `c^2 W.T W`; generating `whiten(clean) + c*Z` or assuming independent unit-variance transformed measurement errors would change the observation model. The reference MCMC remains in its original price likelihood. Fit the matrix on training simulations only, freeze and serialize it, and apply it exactly once at training, validation and prediction. Do not refit it to a new chain. Existing raw-input checkpoints cannot simply be fed whitened data.

## Controlled integration

The new two-arm study uses fresh, identically initialized bounded CouplingFlows: six spline blocks, [128,128] conditioner widths and effective dropout zero. The arms differ only in frozen diagonal versus full ZCA input scaling. Both disable BayesFlow's additional running input standardizer and retain target standardization and the precision-safe bounded target transform.

Both input transforms are fitted to the same 10,000 noisy training rows, with one c and Gaussian vector per physical row. Those fixed noisy rows are repeated for 200 full passes at batch 256, with an unpadded tail of 16: 8000 updates and two million presentations per arm. Explicit Adam uses cosine decay from 5e-4 to 5e-5 over that budget. The clean prices come from the existing bank, with no repricing or reserved-test access. This small fresh offline pair is a controlled comparison within the pilot; the older warm-start/fresh-noise/50m-example runs are contextual references.

The [protocol](protocol.json) freezes task, sources, training-row and preprocessing fingerprints before fitting. Each new checkpoint includes its native model, complete state and separately hash-bound preprocessing artifact. The public loader returns a raw-condition wrapper, so all prediction paths apply the saved transform exactly once. Smoke tests check training-only fitting, inverse reconstruction, identical initialization, source/sidecar tamper rejection, and CLI checkpoint → imported loader exact sampling and next-update state.

Final diagnostics reuse the identical existing 512 validation observations and sample seeds, marginal SBC, three existing conditional datasets, and the unchanged eight-case/seven-ready MCMC panel. No new MCMC is run. Conditional errors are descriptive; this pilot has no release-acceptance family. Each direct worker is capped at 900 seconds, the training campaign at 1000 seconds, with no retries/extensions. Full payloads remain outside Git. Stop for reflection after the pilot and checks; no grid or Bergomi production follows automatically.

# Full-300 known100k pricer: independent saved-panel rescore

The completed diagnostic from Oscar job **8060582** was checked against the **same six fixed development points and 48 saved Monte Carlo batches** used for the earlier pricers. No new simulation, posterior sampling, MCMC, training, reserved-test evaluation or cluster allocation ran. The local reduction took approximately 2.17 seconds elapsed.

The full-300 candidate has lower RMS error point estimates than both the 50k and original 100k fits at every fixed point, on both grids. Entries below are RMS across 63 contracts in quote-noise standard deviations, shown as **working / doubled grid**.

| Fixed point | 50k original | 100k original | 100k full-300 diagnostic |
| --- | ---: | ---: | ---: |
| 0 / generating vector | 0.03811 / 0.05348 | 0.05806 / 0.06408 | 0.03302 / 0.05099 |
| 0 / original neural mean | 0.03588 / 0.05273 | 0.05519 / 0.06170 | 0.03148 / 0.05081 |
| 0 / original PyMC mean | 0.03626 / 0.05262 | 0.05550 / 0.06185 | 0.03165 / 0.05053 |
| 2 / generating vector | 0.10523 / 0.16414 | 0.24264 / 0.22145 | 0.08422 / 0.11229 |
| 2 / original neural mean | 0.05189 / 0.12851 | 0.14565 / 0.15292 | 0.03537 / 0.11150 |
| 2 / original PyMC mean | 0.07426 / 0.13901 | 0.19984 / 0.18766 | 0.05740 / 0.09995 |

“Original neural mean” and “original PyMC mean” are the previously frozen representative vectors from the original 20k study. They were not recomputed or reselected for this diagnostic. Case 2 remains a prior-boundary challenge. The full report retains per-quote t7 intervals and empirical covariance; this table shows point estimates, not new acceptance decisions.

All six independent chain-rule/finite-difference/compiled-PyTensor gradient checks pass, including the uniform-physical-prior logit Jacobian. The largest scaled PyTensor gradient discrepancy is **5.84e-14**. This checks the deterministic mean-emulator target's arithmetic, not posterior accuracy.

Before copying any weights, the source pricer run reported completion, 300 epochs and selected epoch 300. Task, model, bank and checkpoint records agreed, and all copied files were unchanged across the read. Candidate NPZ SHA-256: `069593fc90e47761df7c06657e7a729781dce3680b50105dd4f1d5a094e4756d`; bank manifest SHA-256: `f1bfa42f7bf2acba3daaffb263b5bdeb60b5b12ebdfc267c5c5cd6eb7e08c313`.

Runtime: Python 3.12.16, NumPy 2.5.3, SciPy 1.18.1 and PyTensor 3.3.4 on Darwin arm64, with one-thread library settings. Exact input, evaluator and diagnostic-script hashes are in [the report](readiness.json); [extraction checks](extraction.json), [verification](verification.json) and [file inventory](sha256sums.txt) pin the evidence.

This is a reused development-panel comparison. Pricing-label randomness, emulator approximation and grid effects remain distinct; the doubled grid is not exact truth. No candidate has been promoted. The old 50k/original100k reports, existing PyMC target and teaching caches remain unchanged.

# A bounded coupled-grid Bergomi study

This follows M6.2–M6.3 and M7.2–M7.3a. The simulators, small banks and actual
cluster smoke already exist. The next question is whether changing the asset
and rough-driver grids moves prices materially relative to the bank's quote
noise. Existing independent-run comparisons leave that question poorly
resolved. This experiment changes the comparison, not the model or bank task.

The [machine-readable protocol](../configs/bergomi_refinement_protocol.json)
is frozen before the measured study. It identifies the unchanged task files by
hash, their two parameter priors, the nine contracts and known synthetic curve.
Both are Q tasks; no market observations, neural checkpoint or latent-state
inference is needed.

## Fixed design and stopping rule

For each model, use its four prior corners and midpoint in saved parameter order.
Generate 262,144 independent finest-grid normal paths per case. Use grids of
128, 256, 512 and 1,024 steps over the one-year horizon, with seed
`73026 + 100 * model_index + case_index`. Compare adjacent grids and 128 versus
1,024. Memory chunks of 256 paths do not change path identity. Run each case
once; preserve failed or unresolved results. No seed, path-count, grid or
training sweep follows in this increment.

Each contract's sampling-error budget and grid-difference budget are 10% of its
quote standard deviation, hence **0.025 currency units**. A difference is
resolved within that budget only when

\[
|\widehat\Delta|+4\,\widehat{\operatorname{SE}}(\widehat\Delta)
\le 0.025.
\]

This is deliberately different from the previous discrepancy screen
\(|\widehat\Delta|\le0.025+4\,\widehat{\operatorname{SE}}\), which could pass
because the comparison was noisy. Neither is a continuum-bias bound. Four SEs
are a pointwise engineering screen, not a simultaneous coverage guarantee over
all models, contracts and cases. All marginal price SEs are reported separately.

## What pairing changes

For each independent path index, compute the bounded conditional put value
\(Y_c\) on a coarse grid and \(Y_f\) on a fine grid. The reference pricer
already integrates out the asset's independent Brownian motion. Estimate

\[
\widehat\Delta=\frac1N\sum_{i=1}^N(Y_{f,i}-Y_{c,i}),\qquad
\widehat{\operatorname{SE}}^2=\frac{s^2(Y_f-Y_c)}{N}.
\]

Positive cross-grid covariance reduces difference variance while preserving
both marginal pricing laws. Record the complete joint covariance across grids
and contracts. The counterfactual independent-run variance is
\((s_f^2+s_c^2)/N\), calculated from the same marginal samples; it is not a
second timed independent experiment. Pairing does not reduce the uncertainty
of an individual price estimate by itself.

For the one-factor model, sum fine Brownian increments within each coarse cell
and combine the exact OU innovations with their exponential weights. Project
that coarse OU innovation onto the coarse Brownian increment and standardize
its independent residual. This reproduces coarse factor endpoints exactly while
retaining the coarse asset integration law. Exact factor endpoints do not make
coarse asset integrals exact.

For the rough model, aggregate disjoint blocks of independent normal inputs,
separately in each channel, with a factor \(1/\sqrt m\). Recompute each grid's
hybrid convolution and its own variance normalization. This preserves each
discrete Gaussian model. It is a common-random-number construction, not a claim
that the grid-specific residual channels represent identical continuum Brownian
bridges. Subsampling the fine hybrid driver would change the coarse marginal
law and is not used. The underlying hybrid construction is described by
[Bennedsen, Lunde and Pakkanen](https://arxiv.org/abs/1507.03004).

## Reading the outcome and the next boundary

Report signed price differences, paired SEs, their engineering envelopes,
marginal SEs, variance-reduction factors and actual elapsed time. Inspect every
case and contract; a mean over easy contracts cannot hide a failed difficult
case. A smaller finest-grid correction is useful convergence evidence, but no
convergence rate or Richardson extrapolation is assumed.

The outcome can justify retaining or reconsidering a numerical pilot grid. It
cannot certify uniform prior-wide accuracy or automatically accept a large
training bank. A subsequent inference experiment must explicitly identify its
numerical observation law and its unresolved continuum approximation. Neural
accuracy and empirical model adequacy require separate evidence.

Extend notebook 08 with a small executable pairing example, the saved measured
results, plots and review exercises. Default notebook execution stays offline
and does not launch this full study, bank generation, training or cluster jobs.

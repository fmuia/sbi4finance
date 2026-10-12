# M5.2: fixed-contract supervised pricing experiment

This protocol is frozen before fitting or inspecting the new regression result.
The [machine-readable settings](../configs/hkde_surrogate_protocol.json) define
one bounded synthetic Q experiment. It uses the accepted HKDE Fourier price map
and existing reference artifacts; it does not require an accepted neural
calibration checkpoint. The latter remains provisional after its 0.0439-currency
price-quantile error exceeded 0.03. This experiment can accept only the separate
pricing component, not the combined participant inference workflow.

## Task and lean model

Inputs are the joint parameter pair `(theta, jump_intensity)` within
`[0.025,0.065] × [0.25,1.0]`. Every other HKDE parameter, initial variance, spot
and carry is fixed to the existing task. Outputs are expected discounted prices
of two European puts: `(K=95,T=0.75)` and `(K=80,T=1)`. Contracts and context are
bound to the saved model; a changed contract or extrapolated parameter is rejected.
No quote noise or individual realized payoff is a regression label.

Write `z_d = 2(theta_d - lower_d)/(upper_d-lower_d)-1`. For each contract `c`, fit

\[
\widehat P_c(\theta)=\sum_{i=0}^6\sum_{j=0}^6
\beta_{ijc}T_i(z_1)T_j(z_2),\qquad
T_0(z)=1,\ T_1(z)=z,\ T_{k+1}(z)=2zT_k(z)-T_{k-1}(z).
\]

One float64 least-squares solve estimates 49 coefficients per output from 512
independent uniform parameter pairs. Both contract labels for a pair stay in
the same split. The scaling uses known bounds, not evaluation data. This is
supervised learning of a deterministic price map; it is neither posterior
training nor Bayesian uncertainty in the regression coefficients. A small
polynomial is sufficient to test this two-dimensional responsibility before
adding neural architecture or a general contract encoder.

## Independent checks and price propagation

1. Generate 512 training and 256 separate validation parameter pairs, using
   root 94021 and distinct split streams. Labels use 1,024 Fourier nodes/cutoff
   400. Save the bank and fit once; verify exact reload predictions.
2. Check the complete existing 129×129 reference grid, including boundaries.
   It is a declared stress set, never regression input. Refine Fourier labels
   separately to 2,048 nodes/cutoff 800 on a fixed 5×5 parameter grid.
3. Compare direct and surrogate prices on **identical** cached joint draws
   from the 20 provisional neural posteriors. Compare the 5%, 50%, 95% quantiles
   and means. Also apply identical weights to direct and surrogate prices on
   the reference grid. This isolates substitution error from inference error;
   a pricing surrogate cannot repair an incorrect parameter distribution.
4. Report the unchanged neural-plus-reference-pricer discrepancy and the
   combined neural-plus-surrogate discrepancy side by side. No fresh neural
   checkpoint or SBC run is needed for deterministic price regression.
5. Only if all validation gates pass, generate/evaluate 256 new surrogate test
   parameter pairs. This test is separate from the old posterior final test,
   which remains untouched. No retry or degree/seed sweep follows a failure.

Gates: maximum absolute price error 0.003 currency, RMSE 0.001, maximum paired
mean and quantile shifts 0.003, label-refinement difference 1e-5, design condition
number at most 1,000, and put-bound violations at most 1e-10. Bounds are
`D max(K-F,0) ≤ P ≤ DK`; outputs are checked without clipping. The 0.003
substitution budget is one tenth of the existing 0.03 posterior-price budget.
Relative errors with a 0.1-currency denominator floor are descriptive only.
These finite validation sets do not establish a uniform continuum error bound.

## Cost, reproduction and teaching

Record generation, fitting, serialization/reload and prediction separately.
Warm each pricing route once, then report three repeated calls on the same
4,096 parameter rows. Compare direct pricing with surrogate pricing alone;
cached-label lookup is not a pricer benchmark. Report cold-load-plus-prediction
as well. Inference cost is unchanged and excluded from this component benchmark.
Compute any break-even from this experiment's measured setup cost and query
savings; no speed advantage is assumed.

An offline marimo review will explain the basis, run a small analytic regression
example, show actual error/cost plots and preserve the failed M5.1 inference
result. Banks remain compact NPZ/JSON artifacts. There is no cluster dependency,
market acquisition, package framework, surrogate replacement of calibration
labels, American/exotic claim, or P forecast in this increment.

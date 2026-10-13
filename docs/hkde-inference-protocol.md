# Restricted HKDE posterior: frozen first experiment

This is increment **7/10**, M5.1a–e. It follows the three-model bank/cluster
milestone and leaves the final tutorial's returns-versus-options choice open.
This protocol is fixed before inspecting new reference or neural outcomes.
The [machine-readable settings](../configs/hkde_inference_protocol.json) and
[bank configuration](../configs/tasks/hkde_inference_bank.json) are authoritative.
One architecture, seed and training budget are evaluated; failure remains
useful evidence and does not trigger an automatic tuning sweep.

## The statistical experiment

Infer only long-run annualized variance `theta` and jump intensity `lambda`
from nine noisy European put prices. Independent uniform priors are
`theta ∈ [0.025,0.065]` and `lambda ∈ [0.25,1.0]` per year. The other HKDE
parameters, initial variance, spot, carry and contract ordering are known and
fixed to the existing option-bank task. All data and context are synthetic Q
quantities. Truth labels and clean numerical prices are supervised targets or
diagnostics, never conditioning features.

With the declared deterministic Fourier price vector \(p^{quad}(\vartheta)\),

\[
y_j=p_j^{quad}(\vartheta)+\epsilon_j,
\quad\epsilon_j\overset{iid}{\sim}N(0,0.25^2),\qquad
\pi(\vartheta\mid y)\propto\mathbf1_{\text{prior box}}
\exp\{-\|y-p^{quad}(\vartheta)\|^2/(2\times0.25^2)\}.
\]

The reference targets this **numerical observation law**, not a mathematically
exact continuum-price oracle. Gaussian quote noise can produce inconsistent
or negative quotes; neither training nor evaluation clips those observations.

Two downstream put prices, strike 95 at 0.75 years and strike 80 at one year,
are excluded from conditioning. Each joint posterior parameter draw supplies
one matched pair of reference prices. Their posterior distribution represents
model-price uncertainty, not the distribution of individual future payoffs.

## Independent data roles

Root seed 62026 defines a fresh bank: 8,192 training panels, 768 validation
panels and 256 reserved test panels. The original 24-panel banks are not
training evidence for this experiment.

| Data | Purpose |
|---|---|
| Training split | Fit weights and global normalization. |
| Validation IDs 0–255 | Select the lowest-validation-loss epoch within the single fixed run. |
| Validation IDs 256–767 | Independent diagnostic audit: 512 prior-predictive cases for SBC and coverage. |
| Validation IDs 256–271 | Sixteen fixed conditional-reference cases, also in the diagnostic audit. |
| Four fixed near-corner cases | Parameter fractions `(0.1,0.1)`, `(0.1,0.9)`, `(0.9,0.1)`, `(0.9,0.9)` of the prior box, quote seed 6210. Conditional stress checks only; not SBC cases. |
| Test split | Evaluate only after the fixed checkpoint passes every validation gate. No tuning on these results. |

Loading/storing the reserved test does not authorize its diagnostic evaluation.
No additional observation augmentation crosses a base-panel split. The earlier
Heston results and its reserved final test remain untouched.

## Reference first

Use a uniform two-dimensional grid with tensor trapezoid weights over the
prior rectangle. Calculate its price matrix once and reuse it for every
panel. Normalize posterior weights using log-sum-exp. Compare 65 versus 129
points per axis; use 257 only if the reference checks fail. An unresolved
257-point comparison leaves reference acceptance pending and pauses neural
interpretation, rather than forcing unlimited numerical refinement.

The same 1,024-node/cutoff-400 Fourier law is used on both parameter grids.
Separately compare 2,048 nodes/cutoff 800 at the four full prior corners and
center, including both target contracts. This checks selected pricing
fixtures; it is not a proof of uniform continuum accuracy.

Across the 20 fixed conditional cases, parameter-grid refinement must change
means by at most 0.01 fine-reference SD, SDs by at most 1%, correlation by
0.01, target-price 5/50/95% quantiles by 0.0025 currency, and log evidence by
0.001. The separate price-refinement limit remains 0.001 of quote sigma.
Marginal CDFs integrate marginal densities; a discrete point-mass CDF is not
silently used for parameter credible intervals. Target-price quantiles use
weighted grid masses and must pass their own refinement check.

## One neural baseline

A BayesFlow spline coupling flow (depth 4, widths 64/64, eight bins) takes the
nine ordered currency quotes directly. Each bounded target receives a logit
transform in float64 before conversion to float32. Global normalization learns
from training batches only. There is no summary encoder or per-panel scaling.
Train 80 epochs, batch size 128, seed 6204; reload the best selection-validation
checkpoint and verify identical draws under a repeated sampling seed.

Use 4,096 joint posterior draws per conditional case and 512 draws per SBC
case. Keep the same draw pairing when computing downstream prices.

| Validation quantity | Predeclared gate |
|---|---:|
| Conditional mean RMSE, divided by reference SD | ≤ 0.15 |
| Largest absolute conditional mean error, reference SD units | ≤ 0.40 |
| Largest relative SD error | ≤ 0.25 |
| Largest parameter-correlation difference | ≤ 0.10 |
| Largest downstream 5/50/95% price-quantile error | ≤ 0.03 currency |
| Mean reference mass of inferred 90% parameter intervals | Within 0.03 of 0.90 |
| Marginal SBC | Both randomized-rank ECDF deviations within the joint 5% DKW bound. |

The price gate is 12% of quote sigma, a declared pilot accuracy level rather
than a market standard. Report each case's interval mass, rather than only
an average that could conceal opposing errors. Means and quantiles have
sampling uncertainty: report mean MCSE, order-statistic quantile uncertainty,
and 32 reference-draw repetitions at the same sample size. These contextualize
borderline checks; they do not change the frozen thresholds.

For \(N\) independent SBC cases and two target parameters, use
\(\sqrt{\log(4/0.05)/(2N)}\). Randomize ranks within ties and use
\((R+U)/(L+1)\), with independent uniform \(U\), so the finite draw count is
not compared to a continuous uniform law without adjustment. Coverage and its
sampling uncertainty are reported separately. SBC averages over the prior
predictive experiment and cannot overrule a conditional or joint failure.

The exact uniform-prior means, SDs and zero correlation, plus its grid-based
target-price distribution, provide a **prior-copy baseline**. Compare its
conditional errors with the network's errors and plot prior-to-posterior
change; broad marginal uncertainty alone does not imply the network ignores
the observations.

## Handoff and teaching

Record measured generation, reference construction, training, sampling and
price propagation costs separately. A reusable two-dimensional grid is itself
a cheap inference method; no neural speed advantage is assumed.

Extend notebook 06 with the likelihood, prior/joint-reference comparison,
SBC and conditional error plots, price uncertainty, actual gates and limitations.
Opening the notebook loads cached evidence and a bounded example; it never
launches training or cluster work. Verify offline execution without market
fixtures. A failed baseline can be taught as development evidence but is not
an accepted participant checkpoint. M5.2 pricing-surrogate acceptance remains
a separate subsequent step, and optional market acquisition cannot block S.

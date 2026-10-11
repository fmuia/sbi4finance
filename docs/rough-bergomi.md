# Rough Bergomi: memory, a Gaussian reference and reusable option panels

This implements M7's numerical foundation: a named Gaussian field, an independent
small-grid reference, a production hybrid/convolution generator, and European
pricing through the same bounded estimator as [one-factor Bergomi](bergomi-simulators.md).
The accompanying bank is an explicit numerical pilot. No rough-model neural
posterior, scalar-state historical forecast or empirical market result is claimed.

## The model and what “rough” means

Replace the exponential OU kernel by a power-law kernel:

\[
Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,dW_s,
\quad 0<H<\tfrac12,\quad \operatorname{Var}(Y_t)=t^{2H},
\]
\[
v_t=\xi_0(t)\exp\{\eta Y_t-\tfrac12\eta^2t^{2H}\},\quad
d\log S_t=(r-q_d-\tfrac12v_t)dt+
\sqrt{v_t}(\rho\,dW_t+\sqrt{1-\rho^2}\,dW_t^\perp).
\]

The positive forward-variance curve \(\xi_0\) is known context and has the same
piecewise-linear representation as in the one-factor model. Smaller \(H\) makes
variance fluctuations more irregular. The actual field is a one-sided
Riemann–Liouville/Volterra Gaussian integral starting at zero, **not** ordinary
fractional Brownian motion with its stationary-increment covariance. The kernel
also makes past volatility-driver increments influence future variance with a
long memory. [Original rough Bergomi model](https://doi.org/10.1080/14697688.2015.1099717)

A current variance value is not enough to initialize a future rough path.
`simulate_rough` starts at the original Q origin and explicitly reports
`continuation_state_available=False`. Returned driver paths are useful simulator
diagnostics, but they are not an inferred memory state from observed returns.
No P-measure returns prediction is exposed. A future such task needs a separate
model for P dynamics, initialization and conditionally reconstructed memory.

## Independent reference: include the asset Brownian motion

For positive grid times \(t,u\),

\[
C_{YY}(t,u)=2H\int_0^{\min(t,u)}(t-s)^{H-1/2}(u-s)^{H-1/2}\,ds.
\]

For a Brownian increment on \([a,b]\), truncated at \(t\), its covariance with
\(Y_t\) is

\[
C_{Y,\Delta W}=\frac{\sqrt{2H}}{H+1/2}
\{(t-a)_+^{H+1/2}-(t-b)_+^{H+1/2}\}.
\]

`rough_covariance` evaluates off-diagonal entries by adaptive quadrature after
removing the endpoint singularity; variances and cross-covariances are analytic.
For a uniform step \(h\), conditional Gaussian simulation uses

\[
Y=C_{Y,\Delta W}\,\Delta W/h+LZ,\quad
LL^\top=C_{YY}-C_{Y,\Delta W}C_{Y,\Delta W}^\top/h.
\]

The residual is factorized independently of the hybrid scheme. Material negative
eigenvalues raise an error; only tiny floating-point negatives are rounded to
zero. This method is capped at 128 time steps because its cubic setup cost makes
it a reference, not the large-bank generator. Matching marginal covariance alone
would be insufficient: the same \(W\) determines leverage in the asset equation.

## Production method: one exact singular cell plus convolution

The implementation follows the one-cell hybrid construction: integrate the
singular kernel exactly on the newest interval and approximate it by its
cell-average value on older intervals. This is the \(\kappa=1\) hybrid scheme;
that numerical symbol is unrelated to the OU model's mean-reversion parameter.
[Hybrid method](https://arxiv.org/abs/1507.03004)

Write independent normals for each interval as \(Z_j,B_j\). For step size \(h\),

\[
w_k=h^H\sqrt{2H}\,
\frac{(k+1)^{H+1/2}-k^{H+1/2}}{H+1/2},\quad k=0,1,\ldots,
\]
\[
b=h^H\sqrt{1-\frac{2H}{(H+1/2)^2}},\qquad
Y_{t_i}^{h}=\sum_{j=0}^{i-1}w_{i-1-j}Z_j+bB_{i-1}.
\]

The last-cell combination has exactly the right variance and dependence with
\(\Delta W=\sqrt h Z\). Older weights are cell-average kernels, making their
cross-covariance with every asset-driver increment exact as well. The omitted
within-cell bridge effects on later times leave a small covariance approximation.
We compute the long sum with an FFT convolution, keeping the newest bridge term
separate. This scales as \(O(n\log n)\) per path.

The discrete variance is
\(q_i^h=\sum_{k=0}^{i-1}w_k^2+b^2\). The hybrid model uses
\(v_i^h=\xi_0(t_i)\exp(\eta Y_i^h-\eta^2q_i^h/2)\), ensuring its **own** exact
forward-variance normalization. Substituting \(t_i^{2H}\) while retaining the
approximate driver would introduce another mean bias. This correction does not
make the hybrid process equal to the continuous model: its covariance error and
the asset integration error must still be checked.

`simulate_rough(parameters, curve, dt_years, innovations, method="hybrid")`
requires a uniform positive step and prescribed independent normal channels
`[paths, steps, 3]`. `method="covariance"` invokes the reference. Both use
adapted left-point log asset integration, and both expose latent quantities only
as truth/auxiliary outputs.

## Price labels and the moment problem

`option_panel("rough", ...)` integrates out the independent asset Brownian
motion, averages bounded conditional puts, then obtains calls by put–call
parity. It returns shared-panel sampling covariance and standard errors.
This supplies an estimator with finite variance rather than relying on an
unverified second moment of raw call payoffs.

For rough Bergomi, nonpositive leverage gives the true-martingale pricing domain;
with positive leverage the continuous stock process is a strict local
martingale. Even with negative leverage, sufficiently high positive stock moments
are infinite. Thus a finite option expectation alone does **not** justify raw
call-payoff Monte Carlo error bars. The public API requires \(-1\le\rho\le0\).
[Martingale and moment analysis](https://arxiv.org/abs/1811.10935)

The simulation never silently clips or replaces an overflowed variance draw.
Unsupported numerical magnitudes fail explicitly; silently resampling them
would change the model. Bounded puts are evaluated without exponentiating very
large log asset values. Sampling error, kernel/asset discretization error, and
separately specified quote error remain distinct bank fields/concerns.

## Recorded numerical evidence

```sh
PYTHONPATH=src pixi run python -m unittest discover -s tests -p test_rough_bergomi.py -v
PYTHONPATH=src pixi run python scripts/validate_rough_bergomi.py
```

The [validation artifact](results/rough-bergomi-validation.json) saves parameters,
seeds, contracts, source hashes and complete price-estimator covariance. Three
synthetic scenarios use \((H,\eta,\rho)=(0.1,1,-0.7),(0.07,1.5,-0.9),
(0.3,0.8,-0.3)\), including a rougher, stronger-leverage case. Each prices six
puts with 32,768 paths, hybrid grids 32/64/128 and a 64-step covariance reference.

The frozen screens are a maximum standardized driver-covariance error below
0.02, moment errors below five analytic SEs, and price differences below 0.002
spot units plus four combined MC SEs. All tested fixtures pass:

- Maximum hybrid/continuous standardized Gaussian covariance discrepancy:
  **0.0101**. Asset-driver cross-covariances agree to floating-point precision.
- Independent 100,000-path Gaussian reference sample: maximum covariance error
  **2.54 SE**, asset-driver cross-covariance error **2.49 SE**.
- Forward-variance mean checks: below **0.92 SE** for both schemes.
- Maximum hybrid/covariance-reference price difference at the same grid:
  **0.000787** with spot one.
- Maximum 64-to-128-step hybrid refinement difference: **0.00106** spot units.

The price comparisons use independent random seeds, so their difference SEs
combine the two estimated sampling variances. These are limited numerical
development screens, not simultaneous confidence certificates, continuum-bias
bounds or a guarantee throughout a future parameter prior. The reference has
exact Gaussian **grid** covariance; its asset price still has Euler error.

The [bank-specific screen](results/rough-bergomi-bank-numerics.json) additionally
tests all four corners and the center of the actual prior
\(H\in[0.1,0.3],\eta\in[0.4,1.2]\) with fixed \(\rho=-0.6\), flat forward
variance 0.04, spot 100 and contracts through one year. Each uses 262,144 paths
at both 128 and 512 steps with independent seeds. All pass the predeclared
**0.025 currency units plus four combined MC SEs** refinement screen. The largest
observed difference is **0.0427 currency units**, with combined SE **0.0248**;
the largest 128-step estimator SE is **0.0186**, below the 0.025 bank sampling
limit. Quote-noise SD is 0.25.

The maximum difference exceeds the deterministic 0.025 part of the budget;
sampling uncertainty is still material. **This screen does not prove continuum
bias below 0.025**, and 512-step prices retain approximation error. The saved
bank remains a numerical pilot with sampling and discretization qualifications.
Reproduce the frozen study with:

```sh
PYTHONPATH=src pixi run python scripts/check_bergomi_bank_numerics.py --config configs/tasks/rough_bergomi_option_bank.json --output docs/results/rough-bergomi-bank-numerics.json
```

## Bank readiness and next inference work

This core unlocks a small reproducible Q option-panel bank using generated
curves and contracts. The pilot must retain MC covariance, grid/method settings,
seed identity and its numerical scope. Finite-budget labels are approximate
expectations, not one realized payoff or an exact price oracle. Split and quote
augmentation grouping continue to use the common bank conventions.

Before scaling, choose and validate the actual prior support, contract panel and
price tolerance. Before an inference teaching claim, train a restricted
posterior and assess conditional/reference agreement and price-sensitive
functionals. These posterior gates are distinct from the completed kernel
checks. A finite-factor approximation of this rough kernel and an independently
specified two-factor Bergomi model are different future extensions. Neither is
required to use the current hybrid generator.

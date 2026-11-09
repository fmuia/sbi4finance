# Option application on SPX: error model, observation design and products

Plan of 9 November 2026 for the tutorial of 14 November. Part I says what we will do and
why. Part II gives every step in detail, with the mathematics.

Evidence: [`lab/evidence/pilot-results/2026-11-09/design-and-error/`](../evidence/pilot-results/2026-11-09/design-and-error/README.md)
(this plan's checks). The noise-reassessment session of 9 November is quoted below as
"noise session"; its runs are not committed.

---

# Part I · What we will do and why

## I.1 The goal

Mark products that SPX's listed options do not price directly, consistently with the
listed smile:
- a 6 m → 1 y forward-start call;
- 3 m and 1 y volatility swaps;
- a 1-year down-and-in put.

Give each a range from the full posterior instead of a single point calibration:
- a prudent value at 90%;
- a book-level range computed from the same draws;
- widths that grow when the model misfits the quotes.

Do this under Heston and under rough Bergomi, on the same SPX data, the same design and
the same error model.

Coverage of the ranges is validated in simulation under the stated model and error model.
On real data the true price of a forward start or a barrier is never observed. The
real-data checks are therefore the quantities the market prices independently of the fit:
- the variance swap, replicated model-free from the listed strip;
- the 2-year 90% put, listed on SPX but beyond the design's last maturity.

This mirrors daily practice. A desk calibrates so it can price and hedge what is not
quoted, consistently with what is, and product control attaches a prudent range. Bayes
makes that range part of the calibration. Amortized inference (SBI) makes the posterior
fast enough to compute for every day, and cheap enough to check its coverage on thousands
of simulated days.

## I.2 What is wrong with the current implementation

**The error model.**
- The quote "noise" is 0.3 vol points at the money and 1.0 in the wings, converted to price
  with Black–Scholes vega at 20% vol. That is about 8× SPY's bid–ask (noise session) and
  6–15× SPX's at the money.
- In vol points it equals the stated values only where a quote's own vol is 20%. SPY's
  far calls get 5–7 points; its δ = −2 puts get 0.4–0.9.
- It is therefore neither quote noise nor a stated model error. It is an unestimated,
  independent allowance for misfit, and it alone sets every posterior width.
- The real error is mostly model error. Heston misses SPY and SPX by about 1 vol point
  RMS, 17× SPX's bid–ask, and the misfit is smooth: neighbouring residuals correlate
  0.56–0.80 across strikes and 0.85 across maturities.
- Treated as independent noise, a smooth misfit makes real-data posteriors too narrow.
  Networks trained on the current rule, or on independent error, cover 22–72% and 18–58%
  of the time at a nominal 90% when the real misfit is that size (noise session).

**The design (grid).**
- It is symmetric, ±2 at a fixed 20% vol. It drops the deep put wing, which SPX quotes
  tightly (0.05–0.25 vol points down to −8 at-the-money standard deviations).
- That wing carries the unique information on mean reversion, vol-of-vol, the long-run
  variance level and the variance swap.
- The grid's 9 points replicate only about three quarters of the 1-year variance.
- The far calls look uninformative only because of the 20% vega conversion. Their real
  half-spreads are 0.1–0.4 vol points, and they are what pins the correlation ρ.

**The consequence for the message.** The argument "Heston's variance swaps sit below the
market, so move to rough Bergomi" is confounded twice:
- Heston saw only the grid, while rough Bergomi was handed a forward-variance curve
  replicated from strikes Heston never saw.
- The gap is significant only under independent errors. Under smooth model error the
  bands contain the market from one month to one year (noise session, tentative).

The synthetic main path is internally consistent; what is wrong there is the stated
meaning of the noise.

## I.3 What we will build

One pipeline, applied to both models:

1. **Data.** SPX closing chains of 9–12 November (European, cash-settled, so forwards from
   put–call parity carry no early-exercise bias). From each chain:
   - 63 implied vols on the design, read with the lab's chain code;
   - the half-spread at each point;
   - the held-out market checks (Section 2.6).
2. **Design (layout E).** 7 maturities (5, 10, 21, 42, 63, 126, 252 business days) ×
   9 points at δ ∈ {−5, −3.5, −2.25, −1.25, −0.5, 0, +0.5, +1, +2}, in standard deviations
   of a fixed 16% vol (SPX's at-the-money level). Puts below the money, calls above.
   - Compared with the current grid: 17–35% narrower posteriors on mean reversion,
     long-run variance, vol-of-vol and the variance swap; ρ loses 9–31%.
   - Layouts that drop the near-the-money calls lose ρ by up to 4×.
3. **Error model, in implied-vol units.** Observed vol = model vol + smooth model error
   + quote error.
   - The quote error is each point's measured half-spread.
   - The smooth model error has an unknown size τ, which is inferred with the model
     parameters. Its correlation lengths are fixed from exact fits on the first two days.
4. **Models, both inferring their own mean-variance curve from the same 63 quotes.**
   - Heston with mean reversion free: κ, v0, θ, ξ, ρ, plus τ.
   - Rough Bergomi with the joint curve: H, η, ρ, plus the curve's short level, long level
     and decay, plus τ. Its curve family is Heston-shaped, so the two models get exactly
     the same information about the variance level.
5. **Simulation banks** of clean implied vols:
   - Heston: Fourier prices.
   - Rough Bergomi: Monte Carlo prices to a stated accuracy.

   Errors are drawn fresh during training.
6. **Networks** that output the posterior of all parameters, including τ.
7. **Reference posteriors on every real day and on synthetic cases.**
   - Heston: an exact posterior, by MCMC on the Fourier likelihood.
   - Rough Bergomi: importance sampling from the network with a fixed-seed Monte Carlo
     likelihood.

   The networks must match these references.
8. **Products.** Price every posterior draw with common random numbers. Read off ranges,
   prudent values and book aggregates. Compare with a point calibration, the delta method
   and maturity-drop refits.

## I.4 How we will know it is right

| Check | What it tests |
| --- | --- |
| V1 Prior predictive | Each SPX day lies inside what the prior can produce |
| V2 Pilot exact fits | The error model describes the residuals: whitened residuals uncorrelated, χ²/dof ≈ 1 |
| V3 SBC and coverage on 2,000 simulations | The networks are calibrated on their own simulator |
| V4 Network against reference, on real and synthetic days | The networks are accurate where it matters, which SBC cannot show |
| V5 Predictive check on held-out quotes | The fitted model plus error describes quotes the fit did not use |
| V6 Market checks: model-free variance swap, listed 2-year put | Real-data quantities priced independently of the fit |

If a network fails V4, the real-data results come from the reference posterior, and the
failure is reported.

## I.5 What the results will show, and what they will not

| | Shows |
| --- | --- |
| D1 | One posterior prices every product. Ranges differ by what the quotes pin. |
| D2 | Ranges aggregate coherently across a book, from the joint draws |
| D3 | The 90% ranges contain the true price about 90% of the time, on thousands of simulated days. Only amortization makes this affordable. Compared with the delta method, refits and a point calibration. |
| D4 | "Precise but wrong" comes from the error model: the same day under three error models |
| D5 | How wrong each model is (τ) and where (the residual map) |
| D6 | How marks and ranges move from day to day over the four days |
| D7 | Heston against rough Bergomi on the same information: misfit, predictive checks, market checks, product ranges, and their combination |

**What the ranges are not.** They are parameter uncertainty within each model, under the
stated error model. The model-error term covers misfit on quoted points, not dynamics the
quotes cannot reveal (for example the forward smile). D7 shows how far the two models
disagree. None of this is a complete model-risk reserve.

---

# Part II · Details

## 1. Notation and conventions

- **Time.** Valuation is at a close, time 0. Maturities are counted in business days from
  the valuation session (`chain.trading_days`, NYSE holidays), and T = days/252.
- **Spot, forward, discount.** S_t is the spot. F(T) is the forward for maturity T, D(T)
  the discount factor. With deterministic rates and dividend yield,

  $$S_t = F(t)\,X_t,\qquad X_0 = 1,\qquad \frac{dX_t}{X_t} = \sqrt{V_t}\,dB_t,\qquad B = \rho W + \sqrt{1-\rho^2}\,W^{\perp},$$

  where V is the instantaneous variance, W drives V, and W⊥ is independent of W.
- **Forward units.** A put with strike K and maturity T is worth
  $D(T)\,F(T)\,\mathbb E[(\tilde K - X_T)^+]$ with $\tilde K = K/F(T)$. Its log-moneyness
  is $k = \ln \tilde K$. All model prices are computed in these forward units.
- **Implied vol.** σ_imp(k, T) is the Black–Scholes vol that reproduces the forward-unit
  price (`surface.implied_vol`, bisection). The Black–Scholes vega in forward units is
  $\partial_\sigma P = \sqrt T\,\varphi(d_1)$.
- **Integrated quantities along a path.**

  $$I_t = \int_0^t V_s\,ds,\qquad J_t = \int_0^t \sqrt{V_s}\,dW_s,\qquad J^{\perp}_t = \int_0^t \sqrt{V_s}\,dW^{\perp}_s,$$

  so that

  $$\ln X_t = \rho J_t + \sqrt{1-\rho^2}\,J^{\perp}_t - \tfrac12 I_t,\qquad J^{\perp}_t \mid W \sim \mathcal N(0,\,I_t).$$

  This conditional structure is exact for both models and underlies every Monte Carlo
  estimator below (`lab/src/sbi4finance/calibration/products.py`).
- **Units.** "vp" means vol points (0.01 in decimal vol).

## 2. Data: from the SPX chain to observations

### 2.1 Chains and expiries

- **Source.** Cboe delayed quotes `_SPX.json`, fetched after the close on 9, 10, 11 and
  12 November. The 9 November close is fetched: last trade 16:14:59 ET, SHA-256
  `209765af…a8a1`. Raw files stay outside Git, with each SHA-256 recorded in
  `lab/data/SOURCES.md`.
- **Expiries.** PM-settled SPXW wherever listed. AM-settled SPX (root `SPX` followed by
  the date) only on dates with no SPXW, which supplies the expiries beyond 30 September
  2027. SPXO is a different product and is excluded.
- **Valuation date.** The session of the last trade (`parse_cboe_json` convention).

### 2.2 Discount and forward per expiry

- **Discount.** D(T) = exp(−r(T) T), with r(T) linear in T between that day's 3-month
  and 2-year Treasury constant-maturity yields, converted from bond-equivalent to
  continuous compounding (`chain.rate_curve`).
- **Forward.** From parity on the six pairs nearest the money, given D(T):
  F(T) = median over pairs of K + (C − P)/D(T) (`chain.forward_given_discount`). SPX is
  European, so parity holds without early-exercise terms.
- **Forward between expiries,** used by the barrier product: ln F(t) interpolated
  linearly in t between expiries.

### 2.3 Cleaning (unchanged: `chain.clean_expiration`)

Keep only out-of-the-money quotes: puts with k ≤ 0, calls with k > 0. Drop a quote if:
- its bid is zero;
- its ask is at or below its bid;
- its relative spread (ask − bid)/mid exceeds 50%;
- its mid lies outside the no-arbitrage bounds, so no implied vol exists;
- its implied vol is more than 25% away from the 5-point moving average of its
  neighbours (an isolated outlier).

### 2.4 Implied vols and half-spreads in vol units

For each kept quote: mid price in forward units p = mid/(D F), σ_imp from p, and the
half-spread in vol units

$$h = \frac{(\text{ask}-\text{bid})/2\;/\;(D\,F)}{\sqrt T\,\varphi(d_1(\sigma_{\text{imp}}))},$$

using the vega at the quote's own implied vol. This removes the 20%-vega conversion
criticised in I.2.

### 2.5 Reading onto the design (`chain.build_surface` with `deltas=LAYOUT_E, sigma_ref=0.16`)

1. **Spline per expiry.** A cubic smoothing spline of σ_imp against
   δ = k/(0.16 √T_e), weighted by 1/h, with smoothing factor s = number of quotes.
2. **Choice of expiries.** For a design point (δ, T), with log-moneyness k = δ·0.16·√T,
   use the nearest expiry on each side of T whose cleaned strikes reach k. An expiry
   whose strikes stop short is skipped, never extrapolated.
3. **Interpolation in maturity.** Read each chosen expiry at the same k. Interpolate the
   total implied variance σ²T linearly in T between the two expiries; with one expiry,
   hold its vol.
4. **Half-spread.** Interpolate the half-spread the same way. This gives h_c, without the
   design floor.
5. **Record:** readings from two expiries, from one expiry, and outside quoted strikes.
   The last must be 0.

The output for day t is the vector y_t ∈ ℝ⁶³ of implied vols, plus h_t ∈ ℝ⁶³.

### 2.6 Held-out data, never used in fitting

**(a) Model-free variance swap.** For each expiry whose cleaned strip reaches δ ≤ −8 and
δ ≥ 1.8 (20%-vol units, the existing replication criterion):

$$w(T_e) = 2\int \text{OTM}(k)\,e^{-k}\,dk\qquad(\text{trapezoid over the strip; puts for } k<0,\ \text{calls for } k>0).$$

`build_surface` measures this reach in units of its `sigma_ref`. With `sigma_ref=0.16`,
pass `replication_deltas=(-10, 2.25)` to keep the same strike reach (−8 and 1.8 at 20%).
Interpolate w linearly in T to 21, 63 and 252 business days. The variance-swap vol is
$\sqrt{w(T)/T}$. This is the VIX construction. It is exact for a continuous price path.
Its errors are strike discretization, wing truncation and jumps; report them as a
tolerance, not tune them away.

**(b) The listed 2-year 90% put.** Take the listed expiry nearest 2 years (December 2028)
and the listed strike nearest 0.9 F(T). Record its mid, its half-spread in vol units and
its implied vol.

**(c) Held-out quotes:**
- cleaned quotes on expiries the reading does not use;
- quotes on used expiries with δ < −5 or δ > +2 (16%-vol units).

## 3. Observation design: layout E

- **Definition.** The 63 contracts, maturity-major: T ∈ {5, 10, 21, 42, 63, 126, 252}/252,
  δ ∈ {−5, −3.5, −2.25, −1.25, −0.5, 0, +0.5, +1, +2}, k = δ·0.16·√T, put if δ ≤ 0 else
  call. Strikes e^k in forward units. A fixed 16% scale, not each day's ATM vol:
  - the application is one name over one week;
  - SPX's ATM vol was 15.6–17.1% on 9 November;
  - fixed strikes keep the simulator and the network's input layout independent of the
    data.
- **Why these points.**
  - *Coverage over density.* Under smooth model error, new regions add information and
    points near existing ones add little. Doubling the points in the current range gains
    2–17%; widening to −6…+2 gains 23–49% on κ, θ, vol-of-vol and the variance swap.
  - *Region roles.* Deep puts pin κ, vol-of-vol, θ and the variance swap. ATM pins v0 and
    vol-of-vol. Calls pin ρ. Short maturities pin vol-of-vol and v0. Long maturities pin
    θ, the variance swap and κ.
  - *Keep 0, +0.5, +1, +2.* Without near-the-money calls, ρ degrades by 2.4–4.1× under
    smooth error; with no ATM point, by 2.5×.
  - *Trade-off.* Layout E against the current grid (posterior sd ratio, Heston on SPX):

    | Error model | κ | v0 | θ | ξ | ρ | 1-y var swap |
    | --- | --- | --- | --- | --- | --- | --- |
    | smooth, τ = 1 vp, ℓ = (0.75, 1.15) | 0.65 | 0.92 | 0.72 | 0.68 | 1.09 | 0.75 |
    | smooth, τ = 2.5 vp, ℓ = (1.6, 2.0) | 0.78 | 0.81 | 0.77 | 0.83 | 1.31 | 0.78 |

  - *63 contracts* keeps the existing 7 × 9 code paths. Rough Bergomi's simulation
    blocks already land on these maturities.
- **SPX coverage.** On 9 November, puts were quoted to −12…−16 at-the-money standard
  deviations up to 2 months and about −8 at 6–12 months; calls to +2.2…+3.1. Every
  layout-E point lies inside on every day; this is checked (Section 2.5).
- **Implementation.**
  - A design registry in `tutorial/src/sbi4f_tutorial/shared/surface.py`:
    `DESIGNS = {"canonical_v1": (DESIGN_DAYS, DESIGN_DELTAS, 0.20), "spx_put_heavy_v1": (DESIGN_DAYS, LAYOUT_E, 0.16)}`
    and `design_grid(name)`.
  - Validators compare a task's contracts with its declared design.
    `full_surface.validate_task` currently accepts only the canonical design.
  - Existing tasks and assets are untouched.

## 4. Error model

### 4.1 Observation equation

For a day's vector y of 63 implied vols, with model vols σ(θ) ∈ ℝ⁶³:

$$y = \sigma(\theta) + d + e,\qquad d \sim \mathcal N(0,\ \tau^2 K_\ell),\qquad e \sim \mathcal N(0,\ \operatorname{diag}(h^2)),$$

with the squared-exponential correlation over design coordinates (δ_c, ln T_c):

$$\big(K_\ell\big)_{cc'} = \exp\!\Big(-\frac{(\delta_c-\delta_{c'})^2}{2\ell_\delta^2} - \frac{(\ln T_c-\ln T_{c'})^2}{2\ell_T^2}\Big).$$

- e is the measurement error. h is the half-spread profile in vol units, fixed in the
  task as the median over the fitted days, by design point.
- d is the model error: smooth in strike and maturity, with size τ.

### 4.2 Likelihood

$$\log p(y\mid\theta,\tau) = -\tfrac12\, r^\top\Sigma^{-1} r - \tfrac12\log\det\Sigma - \tfrac{63}{2}\log 2\pi,\qquad r = y-\sigma(\theta),\quad \Sigma = \tau^2 K_\ell + \operatorname{diag}(h^2)\ (+\ \Sigma_{\text{MC}}(\theta)).$$

- Compute it with a Cholesky factor of Σ, adding 10⁻¹⁰ jitter on the diagonal of K.
- Σ_MC is the Monte Carlo covariance of rough Bergomi's implied vols, mapped from price
  covariance by the vegas (Section 5.2). It is zero for Heston.

### 4.3 Why implied-vol units

- The covariance does not depend on θ, so the exact likelihood is a fixed-form Gaussian.
- The network sees the same quantity.
- Cleaned real quotes always invert.

The noise session's price-space form, diag(h_price²) + V(τ_g²K + τ_n²I)V with V the vega
at the model's own vol, is the same model to first order.

### 4.4 Size and correlation lengths

- **Size.** τ is inferred. Prior: log-uniform on [0.2, 5] vp, the same for both models.
  Below 0.2 vp the model error would be comparable to processing error (maturity
  interpolation 0.05–0.25 vp) and bid–ask, which this design does not resolve.
- **Correlation lengths, fixed per model before training,** by profile marginal
  likelihood on the 9 and 10 November closes over ℓ_δ ∈ {0.75, 1, 1.5, 2} and
  ℓ_T ∈ {1, 1.5, 2}:

  $$\hat\ell = \arg\max_\ell \sum_{t\in\{9,10\}} \max_{\theta,\tau}\ \log p(y_t\mid\theta,\tau,\ell).$$

  - Heston uses the Fourier likelihood.
  - Rough Bergomi uses the fixed-seed Monte Carlo pricer with Gauss–Newton steps (the
    classical route of study S6).
  - If rough Bergomi's profile is flat across the grid, it uses Heston's ℓ, and the plan
    says so.
- **Independent term.** An independent model-error term τ_n is added only if the pilot
  needs it: whitened residuals $L^{-1}r$ with lag-1 correlation above 0.3, or χ²/dof
  well above 1 (V2).

### 4.5 What it replaces

This replaces `quote_noise_std` for the two new tasks only. Existing task identities and
assets are unchanged. The new tasks are `heston_spx_iv_v1` and `rough_joint_spx_iv_v1`.

## 5. Models

### 5.1 Heston

$$dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW_t,\qquad V_0 = v_0.$$

- **Inferred parameters** (each uniform on its interval, so the reference and the
  network's logit box need no change):
  - ln κ ∈ [ln 0.5, ln 12];
  - v0 ∈ [0.004, 0.16];
  - θ ∈ [0.004, 0.16];
  - ln ξ ∈ [ln 0.2, ln 4];
  - ρ ∈ [−0.98, −0.2];
  - ln τ ∈ [ln 0.002, ln 0.05].

  This covers the 9 November SPX fit (κ 7.3, ξ 2.5, ρ −0.72), which the current box
  (κ = 2, ξ ≤ 1.5) excludes. The Feller condition is not imposed. Fourier pricing is exact
  without it, and violating it does not invalidate the model.
- **Characteristic function** of ln X_T (`heston_options.model.characteristic_function`),
  with b = κ − ρξiu, d = √(b² + ξ²(u² + iu)), g = (b − d)/(b + d):

  $$\ln\varphi_T(u) = \frac{\kappa\theta}{\xi^2}\Big[(b-d)T - 2\ln\frac{1-g e^{-dT}}{1-g}\Big] + v_0\,\frac{b-d}{\xi^2}\,\frac{1-e^{-dT}}{1-g e^{-dT}}.$$

- **Call price in forward units** (Lewis, half-shifted contour; `european_price`):

  $$C(\tilde K, T) = 1 - \frac{\sqrt{\tilde K}}{\pi}\int_0^\infty \frac{\operatorname{Re}\big[e^{iu\ln(1/\tilde K)}\,\varphi_T(u-\tfrac i2)\big]}{u^2+\tfrac14}\,du,\qquad P = C - 1 + \tilde K.$$

  The integral uses Gauss–Legendre quadrature on [0, U] with N nodes, on the task's
  ladder (N, U) ∈ {(512, 400), (1024, 600), (2048, 1500), (4096, 3000), (8192, 6000)}. A
  rung is accepted when it and the next rung price inside the no-arbitrage bounds and
  agree within 0.01 vp in implied vol. Use the checked pricing speed-ups merged in
  PR #77.
- **Mean variance curve and variance swap:**

  $$\mathbb E[V_t] = \theta + (v_0-\theta)e^{-\kappa t},\qquad K_{\text{var}}(T) = \frac1T\int_0^T\mathbb E[V_t]\,dt = \theta + (v_0-\theta)\frac{1-e^{-\kappa T}}{\kappa T}.$$

### 5.2 Rough Bergomi, joint curve

The model as implemented in `rough_options/simulator.py`:

$$V_t = \xi_0(t)\exp\!\big(\eta\,Y_t - \tfrac12\eta^2\operatorname{Var}(Y_t)\big),\qquad Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW_s,$$

so that E[V_t] = ξ_0(t). The curve is the Heston-shaped family
(`pricing.heston_variance_values`):

$$\xi_0(t) = \xi_s\,e^{-\lambda t} + \xi_l\,(1-e^{-\lambda t}),\qquad K_{\text{var}}(T) = \xi_l + (\xi_s-\xi_l)\frac{1-e^{-\lambda T}}{\lambda T}.$$

- **Same functional form for both models.** The two models infer their variance level
  from the same quotes through the same family, so the variance-swap check (V6) treats
  them equally.
- **Inferred parameters:** H ∈ [0.02, 0.35], η ∈ [0.5, 3.5], ρ ∈ [−0.99, −0.2],
  ξ_s ∈ [0.01, 0.16], ξ_l ∈ [0.01, 0.16], λ ∈ [0.25, 8] (the current joint task's
  bounds), plus ln τ as for Heston.
- **Simulation.** The hybrid scheme (Bennedsen–Lunde–Pakkanen): the singular last cell is
  exact, older cells use averaged kernels, and the convolution is done by FFT. The
  normalizer is the discrete Var(Y_i), so E[V_i] = ξ_0(t_i) holds exactly on the grid. The
  task's blocks:
  - 10 days on 1,280 steps (prices the 5- and 10-day maturities);
  - 3 months on 2,016 steps (21, 42, 63 days);
  - 1 year on 2,016 steps (126, 252 days).
- **Vanilla estimator** (`pricing.rough_block`). Conditional on the W path, ln X_T is
  Gaussian (Section 1), so each path contributes a Black–Scholes price:

  $$P(\tilde K,T) = \mathbb E\Big[\text{BS}_{\text{put}}\big(F' = e^{\rho J_T - \frac12\rho^2 I_T},\ \tilde K,\ w = (1-\rho^2) I_T\big)\Big].$$

  - Antithetic pairs.
  - Two zero-mean controls per maturity, $e^{\rho J_T - \frac12\rho^2 I_T} - 1$ and
    $I_T - \sum_i \xi_0(t_i)\Delta t$, with regression coefficients.
  - The standard error comes from the controlled residuals.
- **Accuracy target.** Implied-vol standard error SE_c = SE_price,c / vega_c(σ_c) must
  satisfy

  $$\text{SE}_c \le 0.25\,\sqrt{h_c^2 + \tau_{\min}^2},\qquad \tau_{\min} = 0.2\ \text{vp},$$

  about 0.05 vp at the money. The path ladder runs 32,768 → 2,097,152. Its stopping rule
  is driven by a per-contract σ_c, set from the first rung's implied vol.
  - The current target, 0.25 × the design noise, is about 0.075 vp at the money but very
    loose in the call wing of low-vol draws. The new target is tighter there.
  - The MC covariance enters training as part of the simulated observation, and enters
    the reference likelihood as Σ_MC (Section 4.2).

## 6. Simulation banks

| | Heston (`heston_spx_iv_v1`) | Rough Bergomi (`rough_joint_spx_iv_v1`) |
| --- | --- | --- |
| Draws | 200,000 from the prior: 180k train, 10k validation, 10k test | 100,000: 90k train, 5k validation, 5k test |
| Pricing | Fourier, quadrature ladder (Section 5.1) | hybrid-scheme MC to the target of Section 5.2 |
| Stored per draw | θ; 63 clean implied vols; quadrature rung; refinement difference | θ; 63 implied vols; their MC standard errors and covariance; paths used per block |
| Failures | draws that cannot be priced inside the bounds are recorded and rejected, never redrawn silently | same |
| Seeds | `SeedSequence([task fingerprint words, example id, purpose])`, independent of batch order | same |
| Cost | about 3–4 CPU-h | the current pilot is 3.3 CPU-h per 1,000 surfaces; the tighter target is estimated at 2–4×, i.e. 660–1,300 CPU-h for 100k. Measure on a 200-surface pilot first. |

- The clean banks do not depend on τ, h or ℓ. Errors are added at training time
  (Section 7).
- New code: `lab/studies/spx_heston/bank.py` and `lab/studies/spx_rough/bank.py`. They
  reuse the tutorial runtime's pricers and write a manifest with SHA-256.

## 7. Networks

- **Input.** The 63 implied vols, standardized per contract with the clean bank's mean and
  sd, stored in the task: z_c = (y_c − μ_c)/s_c.
- **Training augmentation,** drawn fresh for every example in every epoch, extending
  `full_surface.fresh_noise`:
  1. τ = exp(u), u ~ U[ln 0.002, ln 0.05];
  2. d = τ L ε with L L^⊤ = K_ℓ (Cholesky computed once) and ε ~ N(0, I₆₃);
  3. e = h ⊙ ε′ with ε′ ~ N(0, I₆₃) independent of ε;
  4. y = σ_bank + d + e.

  τ (as ln τ) is emitted as an extra target. For rough Bergomi, σ_bank already carries
  its Monte Carlo error.
- **Targets.**
  - Heston: (ln κ, v0, θ, ln ξ, ρ, ln τ).
  - Rough Bergomi: (H, η, ρ, ξ_s, ξ_l, λ, ln τ).

  Both use the logit-box adapter of `shared/full_surface.py`.
- **Architecture.** BayesFlow 2.1, vector encoder, with the family and settings of each
  current backbone.
- **Training budget.** At least 30,000 updates (`lab/scripts/train_full_surface.py` as on
  `main`), several seeds. On the current Heston task this takes about 100 s per fit on
  one L40S. It is the only budget that passed rank and coverage there, and it still
  missed the conditional checks on two of three cases (`NUMERICAL-EXPERIMENT-PLAN.md`,
  P4).
- **Selection.** Choose the configuration and seed by V3 and V4 on synthetic cases only,
  never on the real days.
- **Code.** New task ids keep pricing parameters and τ apart, because
  `tasks.model_parameters` passes every inferred name to the pricer.

## 8. Reference posteriors

### 8.1 Heston: exact

- **Likelihood.** Section 4.2 with Fourier implied vols. Add a new builder next to
  `heston_log_likelihood` in `lab/src/sbi4finance/calibration/reference.py`.
- **Sampler.** PyMC via `sample_deterministic_reference`: uniform priors, the
  log-likelihood as a `Potential`, `DEMetropolisZ`. Four chains, 2,000 tuning steps,
  1,000 draws each.
- **Acceptance.** R̂ ≤ 1.01 and bulk ESS ≥ 400 for every parameter. Record seeds, chain
  structure and versions (MCMC policy).
- **Cost.** About 12,000 likelihood evaluations per chain at 20–50 ms each, i.e. roughly
  15–40 CPU-min per posterior.
- **Cases.** Each SPX day under its own h, plus 8 synthetic cases spread over the prior and
  near SPX.

### 8.2 Rough Bergomi: importance sampling with the network as proposal

- **Proposal.** Draw θ_s ~ q(θ | y) from the network, s = 1…S, with S = 1,000.
- **Weights.**

  $$\tilde w_s = \frac{p(\theta_s)\,\hat L(y\mid\theta_s)}{q(\theta_s\mid y)},\qquad \hat L = \text{Gaussian of Section 4.2 with Monte Carlo implied vols from the fixed-seed pricer and }\Sigma_{\text{MC}}.$$

  The fixed seed makes $\hat L$ a deterministic function of θ (common random numbers).
  The reference is then the exact posterior of that fixed-seed numerical model, which
  differs negligibly from the continuous one when SE ≪ √(h² + τ²).
- **Diagnostics.** ESS = (Σw)²/Σw² ≥ 200, and Pareto-smoothed importance sampling
  k̂ < 0.7. If either fails, report the network as not validated for that case. A second
  proposal, the network widened by a factor 1.5, is tried once.
- **Cost.** 25–50 CPU-s per likelihood evaluation, i.e. 7–14 CPU-h per case.

## 9. Validation

- **V1 · Prior predictive.** For each model, draw 10,000 surfaces from the prior with
  errors. Every SPX day's 63 vols lie inside the 1–99% band at every point.
- **V2 · Pilot fits on 9 and 10 November.** At the fitted (θ, τ, ℓ):
  - whitened residuals u = L⁻¹r have lag-1 autocorrelation below 0.3 along strikes;
  - uᵀu/63 lies within 1 ± 2√(2/63).
- **V3 · SBC and coverage.** On 2,000 test simulations, for each target:
  - the rank of the truth among 127 posterior draws is uniform on 0…127 (χ² test on 16
    bins at the 1% level, Bonferroni across targets);
  - the central 90% intervals cover 87–93%.
- **V4 · Conditional accuracy.** For each reference case, using the repository's limits:
  - (network mean − reference mean)/reference sd within ±0.3 for every parameter;
  - ratio of 90% widths within [0.8, 1.25];
  - τ's median within ±20%;
  - each product's 5%, 50% and 95% quantiles within ±10% of the reference 90% range.
- **V5 · Held-out quotes.** For a held-out point x* with model vol σ*(θ), the predictive
  draw for posterior draw (θ, τ) is

  $$y^* \sim \mathcal N\Big(\sigma^*(\theta) + k_*^\top\Sigma^{-1}r,\ \ \tau^2 - k_*^\top\Sigma^{-1}k_* + h_*^2\Big),\qquad k_* = \tau^2 K_\ell(x^*,\text{grid}),\quad r = y-\sigma(\theta).$$

  The model error at x* is predicted from the grid residuals (kriging); h* is the quote's
  own half-spread. Report the coverage of 90% intervals by region (beyond −5, beyond +2,
  unused expiries). Expect 80–97%; it is not tuned.
- **V6 · Market checks,** reported per day and maturity, inside or outside the 90% band,
  not tuned:
  - the model-free variance-swap vol against the posterior of √K_var(T), at 1 m, 3 m and
    1 y;
  - the listed 2-year 90% put's implied vol against its predictive band from V5. At
    ln T = ln 2, the model-error prediction reverts towards zero with variance up to τ².

**What is reported if a check fails.**
- V3 or V4 fails: real-data results come from the reference posteriors, and the network
  is shown on synthetic data with its V3/V4 numbers.
- V5 or V6 fails: that is a finding about the model, reported as such.

## 10. From posterior draws to product prices

### 10.1 Principle

A product's value is a functional of the model's law: P(θ) = E_θ[discounted payoff]. The
posterior over θ induces a distribution over P (its push-forward). For draws
θ⁽¹⁾…θ⁽ˢ⁾:

$$\hat P^{(s)} = \hat P\big(\theta^{(s)};\ Z\big),\qquad s=1,\dots,S,$$

where Z is one fixed set of random normals shared by every draw (common random numbers).
The spread of $\hat P^{(s)}$ across s is then due to the parameters, not to Monte Carlo
noise. Monte Carlo noise shifts all draws together and is measured separately
(Section 10.9).

- **τ does not enter the products.** It describes how vanillas are observed, not the
  dynamics. The error model affects products only through how it widens and moves the
  posterior of θ.
- **Draws.** S = 1,000: 1,000 network draws per day; for the Heston reference, the 4,000
  MCMC draws thinned to 1,000; for the rough reference, 1,000 draws resampled by the
  importance weights.

### 10.2 Numerical consistency between fit and products

The product pricer must be the same numerical model as the one fitted, or be checked
against it.

- **Heston.** The fit uses Fourier; the products use Monte Carlo (Section 10.3).
  - *Check.* Monte Carlo reprices the 63 grid vanillas within 0.05 vp of Fourier at the
    posterior median, on each day.
  - *If not.* Halve the time step until it does.
- **Rough Bergomi.** The products use the same hybrid scheme and steps per year (2,016)
  as the bank.

### 10.3 Path simulation for products

- **Heston.** Full-truncation Euler for V with 4 sub-steps per business day
  (Δt = 1/1008), as in `products.heston_paths` with finer steps. It stores I and J at every
  business day.
- **Rough Bergomi.** `products.rough_paths` on the hybrid scheme, 2,016 steps per year,
  with I and J stored at every business day, i.e. every 8 steps.
- **Normals.** Paths come in antithetic pairs. The independent normals for J⊥ (one per
  business day) are drawn once and shared by all draws.
- **Horizons.** 1 year for the swaps, the forward start and the barrier; 2 years for the
  2-year put check, with the curve family extending beyond 1 year (stated).

### 10.4 Variance swap

- **Fair strike** (annualized variance, continuous monitoring, no jumps):

  $$K_{\text{var}}(T) = \frac1T\,\mathbb E\Big[\int_0^T V_t\,dt\Big].$$

  Closed form for both models (Sections 5.1 and 5.2). Report √K_var in vol points at 1 m,
  3 m and 1 y.
- **Role.** A check, not a product to mark: compare with the model-free
  $\sqrt{w(T)/T}$ (Section 2.6) in V6 and D4. A traded variance swap uses daily log
  returns with a fixed annualization; the difference from the continuous strike is small
  at these maturities and is stated.

### 10.5 Volatility swap

- **Fair strike:** $K_{\text{vol}}(T) = \mathbb E\big[\sqrt{I_T/T}\,\big]$.
- **Estimator** (`products.volatility_swap`): the sample mean over paths of $\sqrt{I_T/T}$,
  with antithetic pairs and the control $I_T/T - \mathbb E[I_T]/T$. The control's mean is
  exact for rough Bergomi. Under Heston's Euler scheme the discrete mean of I differs from
  the continuous one, so the control is $J_T$, whose mean is zero under any scheme.
- **Sanity check:** the convexity approximation
  $K_{\text{vol}} \approx \sqrt{K_{\text{var}}} - \operatorname{Var}(I_T/T)/(8K_{\text{var}}^{3/2})$.
  The gap between the two is driven by vol-of-vol, which is why this product depends on the
  wings and on the short maturities.
- **Maturities.** 3 months (63 business days) and 1 year (252). Prices in vol points.

### 10.6 Forward-start call, 6 m → 1 y, struck at the forward ratio

- **Payoff** at T₂ = 1 y: $(S_{T_2} - m\,S_{T_1})^+$ with T₁ = 0.5 y and
  m = F(T₂)/F(T₁). Since S_t = F(t)X_t,

  $$\text{Value} = D(T_2)\,F(T_2)\ \mathbb E\big[X_{T_1}\,(X_{T_2}/X_{T_1} - 1)^+\big].$$

- **Conditional formula.** Given the W path, X_{T₁} and X_{T₂}/X_{T₁} are independent,
  because the W⊥ increments over [0, T₁] and [T₁, T₂] are independent. So
  (`products.forward_start_call_spot`, moneyness 1)

  $$\mathbb E\big[\,\cdot\mid W\big] = e^{\rho J_{T_1}-\frac12\rho^2 I_{T_1}}\ \text{BS}_{\text{call}}\big(F' = e^{\rho(J_{T_2}-J_{T_1})-\frac12\rho^2(I_{T_2}-I_{T_1})},\ 1,\ w = (1-\rho^2)(I_{T_2}-I_{T_1})\big).$$

- **Controls:** the forward difference $e^{\rho J_{T_2}-\frac12\rho^2 I_{T_2}} - e^{\rho J_{T_1}-\frac12\rho^2 I_{T_1}}$
  (mean zero) and the integrated-variance control of Section 10.5.
- **Why it is the key exotic.** Its value depends on the forward smile at T₁, which
  today's smile does not determine. Heston's forward skew flattens through mean reversion;
  rough Bergomi's persists. This is where D7's model disagreement is expected.
- **Report** in forward units at T₂ and in index points (× D(T₂) F(T₂)).

### 10.7 Down-and-in put, 1 year, strike 100%, barrier 70%, daily monitoring

- **Payoff** at T = 1 y: (K − S_T)⁺ · 1{min_i S_{t_i} ≤ B}, where K = S_0, B = 0.7 S_0,
  and t_i are the 252 business-day closes.
- **Spot path from the stored quantities.** With ΔI_i = I_{t_i} − I_{t_{i−1}},

  $$\ln X_{t_i} = \rho J_{t_i} + \sqrt{1-\rho^2}\,J^\perp_{t_i} - \tfrac12 I_{t_i},\qquad J^\perp_{t_i} = J^\perp_{t_{i-1}} + \sqrt{\Delta I_i}\;\zeta_i,\qquad \zeta_i\sim\mathcal N(0,1)\ \text{iid}.$$

  This is exact given the simulated variance path. Then S_{t_i} = F(t_i) X_{t_i}, with the
  carry from the parity forwards (Section 2.2). The barrier test is X_{t_i} ≤ B/F(t_i):
  in forward units the barrier is B/F(t_i), which falls as the forward rises with
  positive carry.
- **Estimator.** The mean of D(T)(K − S_T)⁺ 1{hit}, with:
  - antithetic pairs over the variance normals and over ζ;
  - the vanilla put (K − S_T)⁺ as a control variate, with its exact value (Fourier for
    Heston; the conditional estimator of Section 5.2 on the same paths for rough
    Bergomi) and a regression coefficient.

  The per-path identity DIP + DOP = vanilla put is checked.
- **New code** in `lab/src/sbi4finance/calibration/products.py`, with unit tests:
  1. *Black–Scholes limit.* With ξ → 0 and v0 = θ = σ², compare against the closed-form
     continuously monitored down-and-in put evaluated at the shifted barrier
     B e^{−0.5826 σ√Δt} (Broadie–Glasserman–Kou correction for discrete monitoring).
     Pass within 3 standard errors plus 0.5% of price.
  2. *Monitoring frequency.* Halving the monitoring frequency must move the price in the
     direction and by about the size the correction predicts.
  3. *Determinism.* The same seed gives the same price; the antithetic estimator is
     unbiased against a long reference run.
- **Why it matters.** This is the downside barrier of autocallables, the most common
  structured product. It depends on the deep put wing and on spot–vol dynamics (ρ, ξ),
  and it is what the put-heavy design is for.

### 10.8 The 2-year 90% put (check)

- **Model price** at the listed expiry T (≈ 2.1 y) and strike K:
  - Heston: Fourier.
  - Rough Bergomi: the conditional Monte Carlo estimator on 2-year paths, with the curve
    family extended beyond 1 year.

  Then the model implied vol at (ln(K/F(T)), T).
- **Comparison.** The market implied vol ± half-spread against the posterior predictive
  band of Section 9 (V5/V6). This is extrapolation beyond the design's maturities,
  checked against a listed price.

### 10.9 Monte Carlo accuracy

- **Paths.** Choose N per product and model so that the Monte Carlo standard error of the
  price at the posterior median is at most 2% of that product's posterior 90% range.
  Expect N = 20k–100k paths (antithetic) for Heston and 10k–50k for rough Bergomi.
- **Draw-sampling error of quantiles.** Bootstrap over the S draws; for S = 1,000 the 90%
  quantile's probability-level error is about √(0.09/1000) ≈ 0.01.
- **Path error of quantiles.** Rerun all draws with an independent seed and report the
  quantile differences.
- **Requirement.** Both errors at most 2% of the 90% range. Otherwise raise N or S.

### 10.10 Summaries, prudent value and book aggregation

- **Mark (fair value):** the posterior mean, $\bar P = \frac1S\sum_s \hat P^{(s)}$. The
  posterior median and the point-calibration price are also reported.
- **Prudent value at 90%,** on the side that hurts the position: for a long position
  PV = q₀.₁₀, for a short position PV = q₀.₉₀. The parameter-uncertainty reserve is
  |P̄ − PV| (`products.reserve`, with point = P̄).
- **Book.** Positions n_j in products j: short 1 down-and-in put, long 1 forward start,
  short 1 one-year volatility swap. Book value per draw:

  $$B^{(s)} = \sum_j n_j\,\hat P_j^{(s)}.$$

  The book's prudent value is the 10% quantile of B⁽ˢ⁾. Compare the book reserve with the
  sum of standalone reserves; the ratio measures the diversification the joint posterior
  implies. Report the correlation matrix of $\hat P_j^{(s)}$ across draws.

### 10.11 Baselines

- **Point calibration,** as practised: unweighted least squares in implied vol,
  $\hat\theta = \arg\min_\theta \sum_c (y_c - \sigma_c(\theta))^2$, multi-start (8 starts)
  in the prior box. Also the maximum-likelihood version, with τ estimated by the
  likelihood of Section 4.2.
- **Delta method.** At $\hat\theta$, with the Jacobian $\mathcal J = \partial\sigma/\partial\theta$
  (central differences; common random numbers for rough Bergomi) and
  $\hat\Sigma = \hat\tau^2 K_\ell + \operatorname{diag}(h^2)$:

  $$\widehat{\operatorname{Cov}}(\hat\theta) = (\mathcal J^\top\hat\Sigma^{-1}\mathcal J)^{-1},\qquad \operatorname{sd}_j = \sqrt{\nabla P_j^\top\,\widehat{\operatorname{Cov}}\,\nabla P_j},\qquad \text{range } P_j(\hat\theta)\pm1.645\,\operatorname{sd}_j,$$

  with ∇P_j by central differences under common random numbers.
- **Maturity-drop refits.** Refit with each of the 7 maturities removed in turn and take
  the minimum to maximum of the 7 product prices. This is the "recalibrate on other
  instrument sets" practice.

## 11. Demonstrations

| | Computation | Output |
| --- | --- | --- |
| D1 | Per day, model and product: posterior 5/50/95% and mark; point price; delta-method range; refit range; reserve | One table per model; a figure of ranges by product. Vanilla reprices are tight, the forward start is wide, the 2-year put is set against its listed price. |
| D2 | Book reserve against the sum of standalone reserves; correlations between products across draws | Table: book vs sum, diversification ratio |
| D3 | Simulation study. Heston: 1,000 prior draws plus 300 near SPX, 200 posterior draws each. Rough Bergomi: 300 cases, 100 draws each. For each case: truth θ*; observation with errors; network posterior; product distribution; true product price under an independent seed at 4× the paths. | Coverage of the 90% range per product, with binomial ±2 SE (±1.9% for 1,000 cases, ±3.5% for 300), against delta method, refits (300 cases) and point (0%) |
| D4 | One day (9 November), Heston reference under three error models: half-spreads only; plus independent τ; plus smooth τ (adopted) | Variance-swap bands against the model-free value; product ranges as the error model grows |
| D5 | Posterior of τ per day and model; posterior-mean residual map r̄ = y − E[σ(θ) ∣ y] on the 7 × 9 design | Where each model misfits, in vol points |
| D6 | Days 9–12 November: marks and ranges per product; daily point calibrations | Day-to-day changes of the marks against the widths of the ranges. Four days, illustrative. |
| D7 | Heston against rough Bergomi on the same y, h, ℓ and design: τ, V5 coverage and held-out log predictive density, V6 checks, product ranges side by side | Model comparison; stacking weights on held-out quotes, $w = \arg\max_{w\in\Delta}\sum_i \log\sum_m w_m\,p_m(y^*_i\mid y)$, and the weighted mixture of product draws, stated as a combination, not a reserve |

## 12. Corrections to the current text

These are needed whatever else changes. Each item names where the statement is and what
replaces it.

| Where | Now | Replace with |
| --- | --- | --- |
| `tutorial/notebooks/slot4_heston_bayesflow.py:155-158` | noise "specified in implied volatility, 0.3 / 1.0 points" | Section 4's error model; the old rule described as an assumed allowance, about 8× the bid–ask, whose vol-point meaning varies |
| `slot4_heston_bayesflow.py:943-975` | the wing caveat sits in a cell stopped by `mo.stop` | superseded by Sections 2–4 in the rebuilt notebook |
| `slot4_heston_bayesflow.py:1028-1032` | "variance swaps sit below the market … precise, and wrong" | D4, D5 and D7 |
| `slot4` §8 (from line 536), `slot5_calibration.py` §7 (from line 563) | historical SPY workflows | the SPX application of this plan, for both models |
| `slot5_calibration.py:155-163` | the same noise text | as for slot 4 |
| `slot5_calibration.py:650-656` | "variance swaps match SPY's by construction" as the reason to switch models | the curve is inferred from the same quotes for both models; the reasons to compare are the skew's term structure and D7 |
| `tutorial/curriculum/00-overview.md:48`; `04-slot4…md:26, 46, 63, 68`; `05-slot5…md:29, 49, 115, 131`; `06-slot6…md:28, 61` | the variance-swap gap as the case against Heston; "the design's noise" | aligned with the above (verify line numbers when editing) |
| `tutorial/CLAIMS.md` | claims tied to the old rule or the gap | regenerated through `pixi run claims` |
| `lab/src/sbi4finance/calibration/chain.py:147-149, 264-265` | half-spread converted with the quote's own vega, then back to price at 20% vega | not used by the new tasks (Section 2.4 keeps vol units) |
| `tutorial/src/sbi4f_tutorial/shared/inference.py:36-40` | `use_sampler` reportedly reverses its time grid under BayesFlow 2.1 (noise session) | verify; fix if any live path calls it (`full_surface.sample` does not) |

Assets enter `tutorial/assets/` only through the lab tools: `lab/studies/accept.py`, with
a bundle format for each new task, then `pixi run manifest` and `pixi run claims`. The
checks before release: tutorial `pixi run test`, `test-notebooks` and `check`; lab
`pixi run test` and `release-check`.

## 13. Compute

| Item | Estimate |
| --- | --- |
| Heston bank, 200k | 3–4 CPU-h |
| Rough bank, 100k | 660–1,300 CPU-h (pilot of 200 surfaces first) |
| Networks | Heston about 100 s per 30,000-update fit on one L40S; rough Bergomi of the same order; 10–20 fits per model |
| Heston references, 12 posteriors | 3–8 CPU-h |
| Rough references, 12 cases | 85–170 CPU-h |
| Products per day, both models | under 2 CPU-h |
| D3 | Heston about 11 CPU-h; rough Bergomi about 12 CPU-h |
| Data | four SPX closes, about 14 MB each, outside Git |

## 14. Evidence index

- [`design-and-error/summary.json`](../evidence/pilot-results/2026-11-09/design-and-error/summary.json):
  the numbers behind I.2 and Section 3, with their scripts.
- Noise session, 9 November: the SPY error budget, exact fits under three error models,
  test-network coverage, and the variance-swap bands under smooth error. Not committed;
  quoted where used.
- Tutorial assets: `results/calibration/spy/surface.json` (S8);
  `data/heston-spy-exact.json`; `results/calibration/{heston,rough-pilot}/s1/bank-quality.json`
  (costs).
- `lab/plans/NUMERICAL-EXPERIMENT-PLAN.md`, P4: the 8 November Heston training matrix.

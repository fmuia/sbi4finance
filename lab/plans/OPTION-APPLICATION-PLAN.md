# Option application on SPX: error model, observation design and products

Plan of 9 November 2026 for the tutorial of 14 November. Part I says what we will do and
why. Part II gives every step in detail, with the mathematics.

A version in three layers is being written in
[`option-application/`](option-application/README.md). Layers 1 (overview) and 2 (walkthrough) are done.
Layer 3 (full detail) will replace this plan.

Evidence: [`lab/evidence/pilot-results/2026-11-09/design-and-error/`](../evidence/pilot-results/2026-11-09/design-and-error/README.md)
(this plan's checks). Two other pieces of work from 9 November are quoted:
- the noise-reassessment session, quoted as "noise session"; its runs are not committed;
- Alex's European-quote pilot, quoted as "the pilot": the note *How we construct
  observation noise* and its branch `study/heston-european-observations-20261109`
  (evidence through `a1d5c60`; not yet on GitHub).

This plan builds on the pilot for Heston. Section I.6 says what it keeps and what it
changes.

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
quoted, consistently with what is, and product control attaches a prudent range (in the
EU, prudent valuation at 90% confidence for market-price uncertainty and model risk,
Regulation 2016/101). Bayes
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

One pipeline, applied to both models. It starts from the pilot's Heston setup.

1. **Data.** SPX closing chains of 9–12 November (European, cash-settled, so forwards from
   put–call parity carry no early-exercise bias).
2. **Design.** The pilot's frozen put-heavy design: 54 actual SPXW quotes with their
   strikes, expiries, forwards, discounts, half-spreads and market vegas. It is kept if it
   meets the coverage criteria of Section 3.1:
   - deep puts beyond −3 at-the-money standard deviations;
   - an at-the-money point and calls out to about +2;
   - maturities from 1 week to 1 year.

   If it misses one, the missing points are added from layout E (Section 3.3). On the
   design's own day the observations are the quotes themselves. On the other days the
   chain is read at the same locations (Section 3.4).
3. **Error model, in price units with frozen market vegas (the pilot's form):**
   - the quote's half-spread times a known multiplier c;
   - plus a model-error term of size τ, in vol points, converted through vega and
     inferred.

   The pilot draws the model error independently for each quote. Here it is smooth across
   strikes and maturities, because the measured residuals are (Section 4.2).
4. **Models, both inferring their own mean-variance curve from the same quotes.**
   - Heston with κ free and the pilot's priors: v0, θ, κ, ξ, ρ, plus τ.
   - Rough Bergomi with the joint curve: H, η, ρ, plus the curve's short level, long level
     and decay, plus τ. Its curve family is Heston-shaped, so the two models get exactly
     the same information about the variance level.
5. **Simulation banks** of clean prices at the design's contracts:
   - Heston: the pilot's 100,000-surface Fourier bank.
   - Rough Bergomi: a new Monte Carlo bank at the same contracts.

   Errors, τ and c are drawn fresh at every training exposure.
6. **Networks** with the pilot's conditioning (prices over half-spreads, and the log
   measurement scale), which output the posterior of all parameters, including τ.
7. **Reference posteriors on every real day and on synthetic cases.**
   - Heston: an exact posterior, by MCMC on the Fourier likelihood.
   - Rough Bergomi: importance sampling from the network with a fixed-seed Monte Carlo
     likelihood.

   The networks must match these references.
8. **Products.** Price every posterior draw with common random numbers. Read off ranges,
   prudent values and book aggregates. Compare with a point calibration, the delta method
   and expiry-drop refits.

## I.4 How we will know it is right

| Check | What it tests |
| --- | --- |
| V1 Prior predictive | Each SPX day lies inside what the prior can produce |
| V2 Pilot exact fits | The error model describes the residuals: whitened residuals uncorrelated, χ²/dof ≈ 1 |
| V3 SBC and coverage on 2,000 simulations | The networks are calibrated on their own simulator |
| V4 Network against reference, on real and synthetic days | The networks are accurate where it matters, which SBC cannot show |
| V4b Misspecification test | Whether a network that fails on the market case fails because its training errors are independent while the market's residuals are smooth |
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
| D4 | "Precise but wrong" comes from the error model: the same day under several error models |
| D5 | How wrong each model is (τ) and where (the residual map) |
| D6 | How marks and ranges move from day to day over the four days |
| D7 | Heston against rough Bergomi on the same information: misfit, predictive checks, market checks, product ranges, and their combination |

**What the ranges are not.** They are parameter uncertainty within each model, under the
stated error model. The model-error term covers misfit on quoted points, not dynamics the
quotes cannot reveal (for example the forward smile). D7 shows how far the two models
disagree. None of this is a complete model-risk reserve. Hedge ratios under the posterior
are not shown.

## I.6 Relation to the European-quote pilot

| | The pilot | This plan | Why |
| --- | --- | --- | --- |
| Quotes | 54 actual SPXW quotes from one day, frozen | the same, if they meet Section 3.1; read at the same locations on the other days | actual quotes need no interpolation; one network for several days needs fixed locations |
| Error units | price, frozen market vegas, known multiplier c | the same | equivalent to implied-vol units to first order; keeps negative simulated prices without inversion; c gives an error-sensitivity comparison |
| Model error | independent per quote, τ ν_i U_i | smooth: U ~ N(0, K_ℓ) | Heston's residuals correlate 0.56–0.80 between neighbouring strikes and 0.85 across maturities. Networks trained on independent errors cover 18–58% (nominal 90%) when the error is smooth, and the variance-swap gap looks significant only under independent errors (noise session). |
| τ prior | uniform on 0–2 vp | log-uniform on 0.2–5 vp | the smooth amplitude was estimated at 2.1–2.6 vp (noise session), above 2; rough Bergomi's Monte Carlo cost needs a floor above 0 |
| Heston priors, κ free | v0, θ ~ U(0.01, 0.16); κ ~ U(0.1, 10); ξ ~ U(0.1, 3); ρ ~ U(−0.95, −0.1) | the same | the bank exists; the 9 November SPX fit (κ 7.3, ξ 2.5) lies inside; check that posteriors do not pile up at ξ = 3 |
| Pricing tolerance | Fourier refinement within 0.01 h | the same for Heston | right when the measurement term is small |
| Network | the 75k checkpoint fails the reference comparison on the market case (τ mean 0.00143 against 0.00609); non-finite loss at 88k, cause unresolved | the same comparison is the binding check (V4); if it fails, real-data results come from the reference | — |
| Scope | Heston | Heston and rough Bergomi; products | — |

**A possible cause of the pilot's τ gap, and its test (V4b).**
- The pilot's network was trained only on independent errors.
- If the market's residuals are smooth, a smooth residual of about 0.6 vp lies outside
  the training distribution. The network may then read part of the misfit as signal and
  underestimate τ.
- The exact reference is unaffected, because it evaluates the likelihood directly.
- The test, on the pilot's own setup, compares network and reference on:
  1. synthetic cases drawn from the pilot's simulator (independent errors);
  2. synthetic cases with a smooth error of the same size;
  3. the market case.

  It also computes the correlation of neighbouring whitened residuals at the reference
  posterior on the market case.
- Agreement on the first but not on the other two, together with correlated whitened
  residuals, points to the error law rather than to optimization.

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
- **Parameters.** θ denotes either model's parameters; the pilot writes φ for Heston's.
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
criticised in I.2. In the error model (Section 4) h is the half-spread in forward-unit
price; h/ν is the same quantity in vol units.

### 2.5 Reading the chain at the design's locations

On the design's own day the observations are the frozen quotes themselves. On the other
days, read the chain at each design location (k_i, T_i):

1. **Spline per expiry.** A cubic smoothing spline of σ_imp against
   δ = k/(0.16 √T_e), weighted by 1/h, with smoothing factor s = number of quotes
   (`chain.smile_spline`).
2. **Choice of expiries.** Use the nearest expiry on each side of T_i whose cleaned
   strikes reach k_i. An expiry whose strikes stop short is skipped, never extrapolated.
3. **Interpolation in maturity.** Read each chosen expiry at k_i. Interpolate the total
   implied variance σ²T linearly in T between the two expiries; with one expiry, hold its
   vol.
4. **Half-spread.** Interpolate the half-spread the same way.
5. **Record:** readings from two expiries, from one expiry, and outside quoted strikes.
   The last must be 0.
6. **Reading error.** The spline error is about 0.01 vp. Interpolation between expiries
   adds up to 0.05–0.25 vp (noise session; an upper bound from leaving one expiry out).
   It is smooth across strikes, so the smooth model-error term absorbs it. τ on read days
   therefore includes it, and is reported separately from the design's own day.

`chain.build_surface` reads a grid of maturities × deltas. Reading arbitrary (k_i, T_i)
needs a small reader built from the same functions: a new `chain.read_points(rows,
points, rate)`.

The output for each day is the vector of forward-unit prices at the n design locations
(n = 54 for the pilot's design), with their implied vols and half-spreads.

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
- cleaned quotes not in the design: other strikes on the design's expiries, and other
  expiries;
- on read days, also quotes the reading does not use.

## 3. Observation design

### 3.1 Coverage criteria

The coverage principle: cover what the goal depends on, where the name quotes reliably,
within the model's scope. The measured roles of each region (Heston on SPX, smooth error;
rise in posterior sd when the region is removed):

| Region | Pins mostly |
| --- | --- |
| deep puts, below −3 ATM-σ | κ +59%, ξ +51%, θ +33%, 1-year variance swap +27% |
| at the money | v0 +21%, ξ +25% |
| calls above +0.5 | ρ +72% |
| maturities up to 1 month | ξ +108%, v0 +79% |
| maturities from 2 months | θ +303%, variance swap +287%, κ +232% |

A design must therefore contain, in standard deviations of a 16% vol (SPX's
at-the-money level; k = δ·0.16·√T):
1. puts below δ = −3 at short and at long maturities;
2. a point at the money, and calls at about +0.5, +1 and +2. Without the near-the-money
   calls ρ degrades by 2.4–4.1× under smooth error;
3. maturities from 1 week to 1 year, with at least two up to 1 month and two from
   2 months.

Coverage matters more than density. Under smooth error, doubling the points in the
current range gains 2–17%; widening the range gains 23–49% on κ, θ, ξ and the variance
swap.

### 3.2 Adopted design: the pilot's 54 quotes

- The pilot's put-heavy design
  (`lab/evidence/pilot-results/2026-11-09/heston-observations/designs/put_heavy.json` on
  its branch): 54 actual SPXW quotes, frozen with strike, expiry, forward, discount,
  half-spread and market vega.
- Before use, list its points in (δ, T) and check Section 3.1. This plan was written
  without access to the file.
- Each maturity is a whole number of business days from the design's valuation day,
  which is what rough Bergomi's simulation grids need (Section 5.2). Expiries beyond
  1 year need an extra simulation block.
- The kernel of Section 4 uses each point's coordinates δ_i = k_i/(0.16 √T_i) and ln T_i.

### 3.3 Fallback and supplement: layout E

If the pilot's design misses one criterion, add the missing points from layout E. If it
misses several, use layout E itself:
- 7 maturities (5, 10, 21, 42, 63, 126, 252 business days) ×
  δ ∈ {−5, −3.5, −2.25, −1.25, −0.5, 0, +0.5, +1, +2}, with k = δ·0.16·√T and puts at
  δ ≤ 0;
- posterior sd relative to the tutorial's current grid, Heston on SPX:

  | Error model | κ | v0 | θ | ξ | ρ | 1-y var swap |
  | --- | --- | --- | --- | --- | --- | --- |
  | smooth, τ = 1 vp, ℓ = (0.75, 1.15) | 0.65 | 0.92 | 0.72 | 0.68 | 1.09 | 0.75 |
  | smooth, τ = 2.5 vp, ℓ = (1.6, 2.0) | 0.78 | 0.81 | 0.77 | 0.83 | 1.31 | 0.78 |

- read with `chain.build_surface(rows, rate=…, deltas=LAYOUT_E, sigma_ref=0.16)`.

### 3.4 Days after the design's own day

One network serves all four days only if the locations are fixed.
- On days 2–4, read the chain at the design's (k_i, T_i) (Section 2.5).
- Express everything in forward units: divide prices, half-spreads and vegas by that
  day's D(T)F(T).
- For the network's measurement input, set that day's c to the median ratio of its
  half-spreads to the frozen ones.
- Keep the frozen vegas; their day-to-day change is a few percent.

On 9 November, SPX puts were quoted to −12…−16 ATM-σ up to 2 months and about −8 at
6–12 months, and calls to +2.2…+3.1. Every design point must lie inside the quoted
strikes on every day used; this is checked.

## 4. Error model

### 4.1 Observation equation: the pilot's form, with a smooth model error

For the n design quotes, with clean model prices P(φ) in forward units:

$$Y = P(\phi) + c\,h \odot Z + \tau\,\nu \odot (L\,U),\qquad Z,\ U \sim \mathcal N(0, I_n)\ \text{independent},\qquad L L^\top = K_\ell.$$

- **h:** each quote's half-spread in forward units.
- **c:** a known multiplier. c = 1 in the market comparison; c ~ U(0.5, 2) in training
  (the pilot's convention). Treating a half-spread as one Gaussian sd is a convention.
- **ν:** frozen market vegas in forward units, ν_i = √T_i φ(d₁(σ_i^mkt)), in price per
  unit of decimal vol. They are not recomputed for each simulated φ.
- **τ:** the model-error size in decimal vol (0.01 = 1 vp).
- **K_ℓ:** the squared-exponential correlation over (δ_i, ln T_i):

  $$\big(K_\ell\big)_{ij} = \exp\!\Big(-\frac{(\delta_i-\delta_j)^2}{2\ell_\delta^2} - \frac{(\ln T_i-\ln T_j)^2}{2\ell_T^2}\Big).$$

The pilot's model is the special case K_ℓ = I: independent model error.

### 4.2 Why the model error must be smooth

The misfit is a property of the model's smile, not of individual quotes:
- On SPX, Heston's residuals correlate 0.80 between strikes 0.5 ATM-σ apart and 0.85
  between adjacent design maturities (evidence `heston/spx_noise_and_model_error.py`).
- On SPY they correlate 0.56–0.78 (noise session).

Under independent errors these residuals count as many independent pieces of evidence, so
posteriors come out too narrow. A network trained that way covered 18–58% (nominal 90%)
on smooth errors, and the variance-swap gap looked significant (noise session). The
pilot's note makes the same observation: a shared τ "does not create a smooth, signed
model-bias curve".

### 4.3 Likelihood

$$\Sigma(\tau, c) = c^2\operatorname{diag}(h^2) + \tau^2\operatorname{diag}(\nu)\,K_\ell\,\operatorname{diag}(\nu)\ (+\ \Sigma_{\text{MC}}(\phi)),$$

$$\log p(y\mid\phi,\tau) = -\tfrac12\, r^\top\Sigma^{-1} r - \tfrac12\log\det\Sigma - \tfrac n2\log 2\pi,\qquad r = y-P(\phi).$$

- **Computation.** Use a Cholesky factor of Σ, with 10⁻¹⁰ jitter on the diagonal of K.
  The log-determinant term is what stops τ from growing without bound; the pilot's note
  makes the same point.
- **Σ_MC** is rough Bergomi's Monte Carlo price covariance at φ (Section 5.2). It is zero
  for Heston.
- **Vol-unit form.** Dividing the residuals by ν gives
  r/ν ~ N(0, c² diag((h/ν)²) + τ² K_ℓ). This is the same model as an implied-vol
  formulation, to first order.

### 4.4 Size, correlation lengths and the independent part

- **τ prior.** Log-uniform on [0.2, 5] vp, the same for both models.
  - The smooth amplitude was estimated at 2.1–2.6 vp on SPY and SPX (noise session,
    tentative), above the pilot's cap of 2 vp.
  - Below 0.2 vp the model error is comparable to processing error, which this design does
    not resolve. The floor also bounds rough Bergomi's Monte Carlo cost (Section 5.2).
  - The pilot's independent version, with τ ~ U(0, 2 vp), is kept for the comparison in
    D4.
- **Correlation lengths,** fixed per model before training, by profile marginal
  likelihood on the first two days over ℓ_δ ∈ {0.75, 1, 1.5, 2} and ℓ_T ∈ {1, 1.5, 2}:

  $$\hat\ell = \arg\max_\ell \sum_{t\in\{1,2\}} \max_{\phi,\tau}\ \log p(y_t\mid\phi,\tau,\ell).$$

  - Heston uses the Fourier likelihood.
  - Rough Bergomi uses the fixed-seed Monte Carlo pricer with Gauss–Newton steps (the
    classical route of study S6).
  - If rough Bergomi's profile is flat across the grid, it uses Heston's ℓ.
- **Independent part.** Add an independent model-error term τ_n ν ⊙ U′ only if the pilot
  fits need it (V2).

### 4.5 Pricing accuracy

- **Heston:** the pilot's threshold. Fourier rungs must agree within 0.01 h, in float64.
- **Rough Bergomi:** Section 5.2.

### 4.6 What it replaces

This replaces `quote_noise_std` for the two new tasks only. Existing task identities and
assets are unchanged. The new tasks are `heston_spx_v1`, which extends the pilot's task,
and `rough_joint_spx_v1`.

## 5. Models

### 5.1 Heston

$$dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW_t,\qquad V_0 = v_0.$$

- **Inferred parameters:** the pilot's priors, each uniform:
  - v0 and θ on [0.01, 0.16];
  - κ on [0.1, 10];
  - ξ on [0.1, 3];
  - ρ on [−0.95, −0.1];
  - plus ln τ uniform on [ln 0.002, ln 0.05] (Section 4.4).

  This covers the 9 November SPX fit (κ 7.3, ξ 2.5, ρ −0.72), which the tutorial's
  current box (κ = 2, ξ ≤ 1.5) excludes. If a day's posterior piles up against ξ = 3 or
  κ = 10, widen the box and retrain. The Feller condition is not imposed: Fourier pricing
  is exact without it, and violating it does not invalidate the model.
- **Characteristic function** of ln X_T (`heston_options.model.characteristic_function`),
  with b = κ − ρξiu, d = √(b² + ξ²(u² + iu)), g = (b − d)/(b + d):

  $$\ln\varphi_T(u) = \frac{\kappa\theta}{\xi^2}\Big[(b-d)T - 2\ln\frac{1-g e^{-dT}}{1-g}\Big] + v_0\,\frac{b-d}{\xi^2}\,\frac{1-e^{-dT}}{1-g e^{-dT}}.$$

- **Call price in forward units** (Lewis, half-shifted contour; `european_price`):

  $$C(\tilde K, T) = 1 - \frac{\sqrt{\tilde K}}{\pi}\int_0^\infty \frac{\operatorname{Re}\big[e^{iu\ln(1/\tilde K)}\,\varphi_T(u-\tfrac i2)\big]}{u^2+\tfrac14}\,du,\qquad P = C - 1 + \tilde K.$$

  The integral uses Gauss–Legendre quadrature on [0, U] with N nodes, on the task's
  ladder (N, U) ∈ {(512, 400), (1024, 600), (2048, 1500), (4096, 3000), (8192, 6000)}. A
  rung is accepted when it and the next rung price inside the no-arbitrage bounds and
  agree within 0.01 h, the pilot's threshold. Use the checked pricing speed-ups merged in
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
  - 10 days on 1,280 steps (maturities up to 10 business days);
  - 3 months on 2,016 steps (up to 63);
  - 1 year on 2,016 steps (up to 252).
- **Vanilla estimator** (`pricing.rough_block`). Conditional on the W path, ln X_T is
  Gaussian (Section 1), so each path contributes a Black–Scholes price:

  $$P(\tilde K,T) = \mathbb E\Big[\text{BS}_{\text{put}}\big(F' = e^{\rho J_T - \frac12\rho^2 I_T},\ \tilde K,\ w = (1-\rho^2) I_T\big)\Big].$$

  - Antithetic pairs.
  - Two zero-mean controls per maturity, $e^{\rho J_T - \frac12\rho^2 I_T} - 1$ and
    $I_T - \sum_i \xi_0(t_i)\Delta t$, with regression coefficients.
  - The standard error comes from the controlled residuals.
- **Accuracy target,** per contract i, in forward-unit price:

  $$\text{SE}_i \le 0.25\,\sqrt{(c_{\min} h_i)^2 + (\tau_{\min}\,\nu_i)^2},\qquad c_{\min} = 0.5,\quad \tau_{\min} = 0.2\ \text{vp}.$$

  This is a quarter of the smallest total error sd the prior allows, about 0.05 vp at the
  money. The path ladder runs 32,768 → 2,097,152 and stops per contract at this target.
  - The tutorial's current target, 0.25 × its fixed noise, is about 0.075 vp at the money
    but very loose in the call wing of low-vol draws. The new target is tighter there.
  - The MC covariance enters training as part of the simulated observation, and enters
    the reference likelihood as Σ_MC (Section 4.3).
- **Contracts.** The design's quotes (Section 3), in forward units with the design day's
  carry. Maturities in whole business days land on the blocks' grids: one business day is
  128, 32 and 8 steps in the three blocks.

## 6. Simulation banks

| | Heston (`heston_spx_v1`) | Rough Bergomi (`rough_joint_spx_v1`) |
| --- | --- | --- |
| Draws | the pilot's bank: 100,000 surfaces at the design's contracts, reused | 100,000: 90k train, 5k validation, 5k test |
| Pricing | Fourier, rungs within 0.01 h | hybrid-scheme Monte Carlo to the target of Section 5.2 |
| Stored per draw | φ; clean prices; rung; refinement difference | φ; prices; Monte Carlo covariance; paths used per block |
| Failures | recorded and rejected, never redrawn silently | same |
| Seeds | as in the pilot | `SeedSequence([task fingerprint words, example id, purpose])`, independent of batch order |
| Cost | none, it exists. If the design changes (Section 3.3): about 1.3 CPU-h per 100,000 surfaces | 3.3 CPU-h per 1,000 surfaces at today's target. The tighter target is estimated at 2–4× that: 660–1,300 CPU-h. Measure on a 200-surface pilot first. |

- The clean banks do not depend on τ, c, h or ℓ. Errors are added at training time
  (Section 7).
- New code: `lab/studies/spx_rough/bank.py`. It reuses the tutorial runtime's rough
  pricer and writes a manifest with SHA-256.

## 7. Networks

- **Conditions** (the pilot's): the quotes divided by their half-spreads, y_i/h_i, and the
  log measurement scale log(c h_i/(D_i F_i)), both standardized. c is an input, not a
  target.
- **Training augmentation,** fresh at every exposure. This is the pilot's procedure, with
  a smooth model error:
  1. draw ln τ ~ U[ln 0.002, ln 0.05] and c ~ U(0.5, 2);
  2. draw Z, U ~ N(0, I_n), independently;
  3. form Y = P_bank + c h ⊙ Z + τ ν ⊙ (L U), with L the Cholesky factor of K_ℓ,
     computed once.

  For rough Bergomi, P_bank already carries its Monte Carlo error.
- **Targets,** with the logit-box adapter:
  - Heston: (v0, κ, θ, ξ, ρ, ln τ);
  - Rough Bergomi: (H, η, ρ, ξ_s, ξ_l, λ, ln τ).
- **Architecture and training.** The pilot's network and training scripts for Heston
  (`lab/scripts/train_heston_pilot.py` and `train_heston_pilot_campaign.py`, on the
  pilot's branch), and the same settings for rough Bergomi.
  - The pilot's 75k checkpoint fails the reference comparison, and its continuation hit a
    non-finite loss at 88k, cause unresolved. Run V4b first: it says whether the failure
    comes from the error law.
  - On the tutorial's current Heston task, the 8 November matrix needed at least 30,000
    updates to pass rank and coverage, and still missed conditional checks
    (`NUMERICAL-EXPERIMENT-PLAN.md`, P4).
- **Selection.** Choose the configuration and seed by V3 and V4 on synthetic cases only,
  never on the real days.
- **Code.** τ and c are kept apart from the pricing parameters, because
  `tasks.model_parameters` passes every inferred name to the pricer.

## 8. Reference posteriors

### 8.1 Heston: exact

- **Likelihood.** Section 4.3 with Fourier prices: the pilot's normalized likelihood
  (`lab/src/sbi4finance/calibration/heston_pilot.py`), with the smooth covariance.
- **Sampler.** PyMC via `sample_deterministic_reference`: uniform priors, the
  log-likelihood as a `Potential`, `DEMetropolisZ`. Four chains, 2,000 tuning steps and
  1,000 draws each.
- **Acceptance.** R̂ ≤ 1.01 and bulk ESS ≥ 400 for every parameter. Record seeds, chain
  structure and versions (MCMC policy).
- **Cost.** About 12,000 likelihood evaluations per chain at 20–50 ms each, i.e. 15–40
  CPU-min per posterior.
- **Cases.** Each SPX day, plus 8 synthetic cases spread over the prior and near SPX.
- **Comparison point.** Under the pilot's independent model, the market case gives τ mean
  0.00609, with 90% interval [0.00516, 0.00722]. The smooth model's τ is a different
  quantity, the amplitude of a correlated field, and is expected to be larger.

### 8.2 Rough Bergomi: importance sampling with the network as proposal

- **Proposal.** Draw θ_s ~ q(θ | y) from the network, s = 1…S, with S = 1,000.
- **Weights.**

  $$\tilde w_s = \frac{p(\theta_s)\,\hat L(y\mid\theta_s)}{q(\theta_s\mid y)},\qquad \hat L = \text{Gaussian of Section 4.3 with Monte Carlo prices from the fixed-seed pricer and }\Sigma_{\text{MC}}.$$

  The fixed seed makes $\hat L$ a deterministic function of θ (common random numbers).
  The reference is then the exact posterior of that fixed-seed numerical model, which
  differs negligibly from the continuous one when SE ≪ √(h² + τ²).
- **Diagnostics.** ESS = (Σw)²/Σw² ≥ 200, and Pareto-smoothed importance sampling
  k̂ < 0.7. If either fails, report the network as not validated for that case. A second
  proposal, the network widened by a factor 1.5, is tried once.
- **Cost.** 25–50 CPU-s per likelihood evaluation, i.e. 7–14 CPU-h per case.

## 9. Validation

- **V1 · Prior predictive.** For each model, draw 10,000 surfaces from the prior with
  errors. Every SPX day's quotes lie inside the 1–99% band at every design point.
- **V2 · Pilot fits on the first two days.** At the fitted (φ, τ, ℓ), with Σ = LLᵀ:
  - whitened residuals u = L⁻¹r have lag-1 autocorrelation below 0.3 along strikes
    within each expiry;
  - uᵀu/n lies within 1 ± 2√(2/n).

  Apply the same test to the pilot's independent-error reference on its market case.
- **V3 · SBC and coverage.** On 2,000 test simulations, for each target:
  - the rank of the truth among 127 posterior draws is uniform on 0…127 (χ² test on 16
    bins at the 1% level, Bonferroni across targets);
  - the central 90% intervals cover 87–93%.
- **V4 · Conditional accuracy.** For each reference case, using the repository's limits:
  - (network mean − reference mean)/reference sd within ±0.3 for every parameter;
  - ratio of 90% widths within [0.8, 1.25];
  - τ's median within ±20%;
  - each product's 5%, 50% and 95% quantiles within ±10% of the reference 90% range.
- **V4b · Misspecification test** (Section I.6). Here the reference is the exact posterior
  under the network's own error model, evaluated on each case's data. On the pilot's
  independent-error network, compare network and reference on:
  1. synthetic cases from its own simulator;
  2. synthetic cases with a smooth error of the same size;
  3. the market case.

  Agreement on the first but not on the other two, with correlated whitened residuals in
  V2, points to the error law. Repeat all three on the smooth-error network.
- **V5 · Held-out quotes.** For a held-out quote x* with model price P*(φ) and market vega
  ν*, the predictive draw for posterior draw (φ, τ) is

  $$y^* \sim \mathcal N\Big(P^*(\phi) + k_*^\top\Sigma^{-1}r,\ \ \tau^2\nu_*^2 - k_*^\top\Sigma^{-1}k_* + (c\,h_*)^2\Big),\qquad k_* = \tau^2\,\nu_*\,\operatorname{diag}(\nu)\,K_\ell(\text{design}, x^*),\quad r = y-P(\phi).$$

  The model error at x* is predicted from the design's residuals (kriging); h* is the
  quote's own half-spread. Report the coverage of 90% intervals by region (beyond −5, beyond +2,
  unused expiries). Expect 80–97%; it is not tuned.
- **V6 · Market checks,** reported per day and maturity, inside or outside the 90% band,
  not tuned:
  - the model-free variance-swap vol against the posterior of √K_var(T), at 1 m, 3 m and
    1 y;
  - the listed 2-year 90% put's implied vol against its predictive band from V5. At
    ln T = ln 2, the model-error prediction reverts towards zero, with variance up to
    τ²ν*².

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
  - *Check.* Monte Carlo reprices the design's vanillas within 0.05 vp of Fourier at the
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
  $\hat\theta = \arg\min_\theta \sum_i (\sigma^{\text{mkt}}_i - \sigma_i(\theta))^2$, with
  8 starts in the prior box. Also the maximum-likelihood version, with τ estimated by the
  likelihood of Section 4.3.
- **Delta method.** At $\hat\theta$, take the Jacobian of the design prices,
  $\mathcal J = \partial P/\partial\theta$ (central differences; common random numbers for
  rough Bergomi), and
  $\hat\Sigma = c^2\operatorname{diag}(h^2) + \hat\tau^2\operatorname{diag}(\nu)K_\ell\operatorname{diag}(\nu)$.
  Then:

  $$\widehat{\operatorname{Cov}}(\hat\theta) = (\mathcal J^\top\hat\Sigma^{-1}\mathcal J)^{-1},\qquad \operatorname{sd}_j = \sqrt{\nabla P_j^\top\,\widehat{\operatorname{Cov}}\,\nabla P_j},\qquad \text{range } P_j(\hat\theta)\pm1.645\,\operatorname{sd}_j,$$

  with the product gradients ∇P_j taken by central differences under common random
  numbers.
- **Expiry-drop refits.** Refit with each of the design's expiries removed in turn, and
  take the minimum to maximum of the resulting product prices. This is the "recalibrate
  on other instrument sets" practice.

## 11. Demonstrations

| | Computation | Output |
| --- | --- | --- |
| D1 | Per day, model and product: posterior 5/50/95% and mark; point price; delta-method range; expiry-drop refit range; reserve | One table per model; a figure of ranges by product. Vanilla reprices are tight, the forward start is wide, the 2-year put is set against its listed price. |
| D2 | Book reserve against the sum of standalone reserves; correlations between products across draws | Table: book vs sum, diversification ratio |
| D3 | Simulation study. Heston: 1,000 prior draws plus 300 near SPX, 200 posterior draws each. Rough Bergomi: 300 cases, 100 draws each. For each case: truth θ*; observation with errors; network posterior; product distribution; true product price under an independent seed at 4× the paths. | Coverage of the 90% range per product, with binomial ±2 SE (±1.9% for 1,000 cases, ±3.5% for 300), against delta method, refits (300 cases) and point (0%) |
| D4 | One day, Heston reference under four error models: half-spreads only (c = 1); plus independent τ (the pilot's model); plus smooth τ (adopted); the adopted model at c = 0.5 and c = 2 | Variance-swap bands against the model-free value; product ranges as the error model grows |
| D5 | Posterior of τ per day and model; posterior-mean residual map r̄ = y − E[P(φ) ∣ y] on the design, in vol points (r̄/ν) | Where each model misfits |
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
| `tutorial/src/sbi4f_tutorial/shared/{full_surface,full_surface_teaching}.py`; the notebooks' 7 × 9 reshapes | 63-contract checks and plots | accept a declared design of any size (n = 54 for the pilot's) |
| `tutorial/src/sbi4f_tutorial/shared/inference.py:36-40` | `use_sampler` reportedly reverses its time grid under BayesFlow 2.1 (noise session) | verify; fix if any live path calls it (`full_surface.sample` does not) |

Assets enter `tutorial/assets/` only through the lab tools: `lab/studies/accept.py`, with
a bundle format for each new task, then `pixi run manifest` and `pixi run claims`. The
checks before release: tutorial `pixi run test`, `test-notebooks` and `check`; lab
`pixi run test` and `release-check`.

## 13. Tutorial integration

### 13.1 Slot 4: Heston on SPX

The notebook is rewritten in place; `pixi run heston-options` is unchanged.

1. **The job.** What a desk calibrates for each day, and what Bayes and SBI add (I.1).
2. **SPX data and the design.** The smile, where the quotes sit, and which region pins
   which parameter (Section 3.1).
3. **The error budget.** Bid–ask against model error, then the error model of Section 4:
   the pilot's form with the smooth term.
4. **The model.** Heston, its priors and the simulator, with the code visible, and one
   training pair.
5. **The amortized posterior on a synthetic case.** SBC and coverage (V3); the network
   against the exact reference (V4).
6. **SPX, four days.** Posterior, τ, the residual map (D5), and the held-out and market
   checks (V5, V6).
7. **Products.** D1–D4 and D6.
8. **Scope.** What the ranges cover and what they do not (I.5).

Retire §8 (historical SPY) from the live path. Old assets stay until a dependency-checked
retirement batch.

### 13.2 Slot 5: rough Bergomi on SPX

The notebook is rewritten in place; `pixi run rough-options` is unchanged.

1. **Why another model.** SPX's at-the-money skew term structure (`smile_statistics`,
   `skew_power_law`) and Heston's residual map from slot 4.
2. **The model.** Its curve family (Section 5.2), Monte Carlo pricing, and the accuracy
   target.
3. **Inference.** The same design and error model, the network, and the importance-sampling
   reference (Section 8.2).
4. **SPX.** Posterior, τ, residual map, and the held-out and market checks.
5. **Products under rough Bergomi,** then D7: the two models side by side, and their
   combination.
6. **Scope.** What the comparison shows and what it does not.

Retire §7 (historical SPY studies) from the live path.

### 13.3 Slot 6: priors and misspecification

D4 (half-spreads only, independent τ, smooth τ, and c from 0.5 to 2) is the material for
the noise part. A slider maps to c and to the size of τ.

## 14. Limitations and known risks

1. **The smooth error can absorb parameter signal.** A flexible model error and the
   parameters can trade off.
   - The correlation lengths are fixed from the first two days' fits.
   - Wider posteriors are reported as the honest result.
   - V2 and V4 detect problems.
2. **The error law is assumed.** It is Gaussian, with a squared-exponential correlation.
   τ cannot separate structural model error from data or convention problems; the pilot's
   note says the same.
3. **The networks may still miss the references.** The pilot's 75k checkpoint does, and
   so did the 8 November matrix. Real-data results then come from the references.
4. **Rough Bergomi's bank carries Monte Carlo error** inside the simulated observations.
   With the target of Section 5.2, its posteriors become slightly conservative, not
   biased.
5. **Fourier accuracy at extreme κ and ξ,** for short-dated deep puts. Rung checks apply,
   and rejected draws are recorded.
6. **Event risk inside the shortest maturity** (a US CPI release that week). The smooth
   term absorbs it; say so.
7. **Four days** illustrate day-to-day behaviour; they are not a history.
8. **The ranges are within-model parameter uncertainty.** They are not a complete
   model-risk reserve, and hedging under uncertainty is not shown.
9. **The runtime's 63-contract assumptions** must accept a declared design (Section 12).
10. **Raw data.** Cboe serves only the current snapshot, so each close must be fetched
    that evening. Files stay outside Git, recorded by SHA-256.
11. **The public site.** No unreleased results go on `site/` (AGENTS.md).

## 15. Compute

| Item | Estimate |
| --- | --- |
| Heston bank | exists (the pilot's 100k); about 1.3 CPU-h per 100k if the design changes |
| Rough bank, 100k | 660–1,300 CPU-h (pilot of 200 surfaces first) |
| Networks | Heston about 100 s per 30,000-update fit on one L40S; rough Bergomi of the same order; 10–20 fits per model |
| Heston references, 12 posteriors | 3–8 CPU-h |
| Rough references, 12 cases | 85–170 CPU-h |
| Products per day, both models | under 2 CPU-h |
| D3 | Heston about 11 CPU-h; rough Bergomi about 12 CPU-h |
| Data | four SPX closes, about 14 MB each, outside Git |

## 16. Evidence index

- [`design-and-error/summary.json`](../evidence/pilot-results/2026-11-09/design-and-error/summary.json):
  the numbers behind I.2 and Section 3, with their scripts.
- Noise session, 9 November: the SPY error budget, exact fits under three error models,
  test-network coverage, and the variance-swap bands under smooth error. Not committed;
  quoted where used.
- Tutorial assets: `results/calibration/spy/surface.json` (S8);
  `data/heston-spy-exact.json`; `results/calibration/{heston,rough-pilot}/s1/bank-quality.json`
  (costs).
- `lab/plans/NUMERICAL-EXPERIMENT-PLAN.md`, P4: the 8 November Heston training matrix.
- The pilot (Alex, 9 November): the note *How we construct observation noise*, and the
  branch `study/heston-european-observations-20261109`, evidence through `a1d5c60`, not
  yet on GitHub:
  - code: `lab/src/sbi4finance/calibration/heston_pilot.py` and
    `lab/scripts/train_heston_pilot{,_campaign}.py`;
  - evidence: `lab/evidence/pilot-results/2026-11-09/heston-observations/`, i.e. the
    design `designs/put_heavy.json` and, under `training-study/`,
    `continuation-results.json`, `reference-precision.json` and
    `continuation-failure-audit.json`.


---

# Appendix A · Evidence behind the choices

### A.1 The reviewer's grid issue, reassessed

| Claim | Verdict |
| --- | --- |
| Symmetric ±2 at a fixed 20% vol | Right; the noise is tied to 20% too |
| Put quotes reach δ = −8 to −17 | Right in substance: −8.2 to −14.5 in 20%-vol units on SPY, 21 October (−13 to −21 in at-the-money-vol units) |
| A third of the points are far calls with almost no information | 21 of 63 at δ ≥ +1. Their low weight comes from the 20% vega conversion: 5–7 vp of effective noise against half-spreads of 0.1–0.4 vp. |
| Some points have no quotes | Wrong: 0 of 63 lie outside quoted strikes (SPY 21 October, SPY 6 November, SPX 9 November) |
| Re-pricing costs 1.3 CPU-h per 100k Heston surfaces and 330–370 CPU-h for rough Bergomi | Right at today's accuracy targets |
| Fix: scale by at-the-money vol and shift towards puts | The shift is right, but only with a vol-point error model (A.3). Scaling by at-the-money vol alone is neutral on one date (A.5). |

### A.2 Measured error budget

| Source | Size (vp) | From |
| --- | --- | --- |
| Bid–ask, SPX 9 November | 0.02–0.05 at the money; 0.05–0.17 down to −6 ATM-σ; 0.11–0.25 at −6 to −8; 0.10–0.37 for calls at +2 to +3 | evidence |
| Bid–ask, SPY 6 November | 0.04–0.13; 0.26–0.37 at δ = +2 | noise session |
| Spline reading | about 0.01; neighbours correlate +0.38 | noise session |
| Maturity interpolation | 0.05–0.25 (an upper bound) | noise session |
| Early exercise (SPY only) | up to about 1.1 at the 1-year at-the-money put (binomial estimate) | noise session |
| Heston model error | SPY: 0.8–1.2 RMS, about 99% of the variance. SPX: 1.0 RMS, 17× the half-spread. Residual correlation 0.56–0.80 across strikes, 0.85 across maturities. | noise session; evidence |
| The tutorial's rule, in vp | SPY 21 October: δ = −2 0.44 (median); δ = +2 5.4 (median), 7.0 (max). Prior centre: 0.28–0.55. | evidence |
| Fit of the rule | χ² 66.5 on 63 quotes (21 October); 508 (6 November) | noise session |

### A.3 The reviewer's fix under three error models

The grids:
- G1, the proposal: −4…+1 in at-the-money-vol units;
- G2, shift only: the same deltas at 20% vol;
- G3, scale only: ±2 in at-the-money-vol units.

Posterior sd is given relative to the current grid.

- **Under the tutorial's rule:** G1 and G2 look 2–12× tighter for Heston and 5–25× for
  rough Bergomi. Deep puts get only 0.003–0.03 vp of noise, so this is false precision.
  Rough Bergomi's Monte Carlo error at δ = −4 would be 40–65× that noise.
- **Under a consistent, independent vol-point error:**
  - Heston, 150 prior draws, G1: v0 1.07, θ 0.89, ξ 0.83, ρ 1.48, variance swap 0.83.
  - Rough Bergomi, 8 points, G1: H 0.58, η 0.77, ρ 1.41, variance swap 0.68.
  - G3 stays near 1 (0.95–1.0).
- **Under smooth model error (Heston on SPX):** widening to −6…+2 gains 23–49% on κ, θ, ξ
  and the variance swap, while doubling the density gains 2–17%. ρ needs the
  near-the-money calls (Section 3.3).

### A.4 The variance-swap gap

- **SPY, 21 October** (exact Heston posterior under the rule): Heston's variance swaps sit
  0.6–1.7 vp below the replicated values, with bands 0.3–0.6 vp wide.
- **SPY, 6 November,** Heston fitted on the current grid and on a put-heavy grid: the
  1-week shortfall stays (−3.9 and −3.2 vp), and the 1-week deep put misses by 11.5 vp.
  The short-end shortfall is real misfit.
- **Effect of the error model:** under independent τ the bands (0.5–1.1 vp) exclude the gap
  (−0.5 to −1.6 vp); under smooth τ the bands (1.8–3.0 vp) include it from 1 month to
  1 year (noise session, tentative).
- **The comparison was unequal.** In the tutorial, rough Bergomi was handed a curve
  replicated from strikes Heston never saw. The plan gives both models the same quotes and
  compares τ, residual maps, held-out checks and market checks (D5, D7).

### A.5 Coverage by goal and by name; coordinates; alternatives

| Goal | What to cover |
| --- | --- |
| Reprice the hedges | every liquid quote, weighted by the error model |
| Variance or volatility swap | the whole strip into the deep puts |
| Forward start, cliquet | ATM level and skew across the forward dates; short maturities for vol-of-vol |
| Downside barrier, autocallable | long-dated puts down to the barrier |
| Model comparison | where the models differ, with the same data for every model |

| Name | Design | Dominant error |
| --- | --- | --- |
| SPX | wide and put-heavy, 1 week to 1 year | model error; European, so clean forwards |
| SPY | as SPX | plus early exercise on puts, so prefer SPX |
| Thin ETF (EWG: 4 expiries to 6 months; half-spreads 1.2–2 vp) | what the chain has, narrow | quote error; the prior beyond the quotes |
| Single stock | skip earnings expiries | early exercise, dividends, events |

**Coordinates.** This plan uses a fixed 16% scale, not each surface's at-the-money vol.
- On one date, scaling by at-the-money vol adds no information (A.3, G3).
- It makes strikes depend on the data: the simulator must place them from each simulated
  surface's at-the-money vol, and the network must condition on them.
- It pays off only across names and vol regimes.

**Not adopted.**
- *A set encoder over all raw quotes.* It needs variable quote layouts in simulation, and
  is the deferred sparse-chain work (AGENTS.md).
- *A denser symmetric grid.* Under smooth error, density adds little.

### A.6 What stays right in the current implementation

- a fixed observation design shared by simulator, network and reference;
- out-of-the-money quotes in forward units, √T scaling, and maturities from 1 week to
  1 year;
- spline reading without extrapolation (0 points outside quoted strikes);
- numerical accuracy tied to the error scale;
- the recovery, SBC and reference checks. The synthetic main path is internally
  consistent.

# Option application: error model, observation design and products

Plan of 9 November 2026 for the tutorial on Saturday 14 November (Milan, 11:00 CET).
Freeze: Friday 13 November, 18:00 CET. There is no later rebuild: what we present on
Saturday must be principled and correct.

Evidence: [`lab/evidence/pilot-results/2026-11-09/design-and-error/`](../evidence/pilot-results/2026-11-09/design-and-error/README.md)
(this plan's checks) and the noise-reassessment session of 9 November (quoted below as
"noise session"; its runs are not committed).

## 0. Summary

**What is wrong now.**

1. **The quote noise is not quote noise.** The rule is 0.3 vol points at the money and
   1.0 in the wings, converted to price with Black–Scholes vega at 20% vol. That is
   about 8× SPY's bid–ask. In vol points it equals the stated values only where a quote's
   own vol is 20%: SPY's far calls get 5–7 points, the δ = −2 puts 0.4–0.9. In effect
   it is an unestimated, independent allowance for model error.
2. **The real error is model error, and it is smooth.** Heston misses SPY and SPX by
   about 1 vol point RMS, 17× SPX's bid–ask, with neighbouring residuals correlated
   0.56–0.80 across strikes and 0.85 across maturities. Treating it as independent noise
   makes real-data posteriors too narrow. Networks trained on the rule, or on
   independent error, cover 22–72% and 18–58% of the time at a nominal 90% (noise session).
3. **The grid misses the informative put wing.** It is symmetric, ±2 at 20% vol. SPX
   quotes the deep puts tightly, and they carry the unique information on mean
   reversion, vol-of-vol, long-run variance and the variance swap.
4. **The variance-swap argument for rough Bergomi is confounded twice.**
   - Heston sees only the grid, while rough Bergomi's curve comes from strikes down to
     δ = −8.
   - The gap is significant only under independent errors.
5. **The synthetic main path is consistent.** What is wrong there is the stated meaning
   of the noise, not the inference. Every real-data claim is affected.

**What we present instead.**

1. **SPX** (European), closing quotes of 9–12 November.
2. **Heston on SPX, rebuilt on the principles in Section 3:**
   - κ free;
   - a put-heavy 63-quote design;
   - measured half-spreads plus an inferred smooth model error;
   - an exact reference posterior on every real day;
   - an amortized network that must match it.
3. **The goal after fitting** (Section 4.9): mark products that SPX's listed options do
   not price directly, and give each a range from the full posterior.
   - The products are a 6 m → 1 y forward start, 3 m and 1 y volatility swaps and a 1-year
     down-and-in put.
   - The ranges are validated for coverage in simulation.
   - On real data, the checks are what the market prices independently: the variance
     swap and the listed 2-year 90% put beyond the design.
4. **Rough Bergomi remains the Monte Carlo case on synthetic data.** Its error is stated
   as an assumed allowance and its real-data claims are removed. Retraining it with the
   new error model is optional.

| Tier | Content | Why | Effort | Gate |
| --- | --- | --- | --- | --- |
| 1 | Correct every statement: noise meaning, grid scope, variance-swap argument, rough framing, slides | Correctness, whatever else happens | 0.5–1 day, text only | Tue 20:00 |
| 2 | Heston on SPX, rebuilt (Section 4) | Makes sense; shows the goal | About 3 days: two people plus parallel sessions | Wed 20:00: network matches the exact posterior |
| 3 | Rough Bergomi retrained with the new error model on its existing bank | Consistency between the lessons | 1–2 days, Oscar GPU | Start Wed 12:00 only if Tier 2 is on track |

**Decisions needed Tuesday morning:**
1. Scope: Tier 1 + 2 is recommended.
2. Rewrite slot 4 in place (recommended) or add a new notebook.
3. Owners for W1–W8.
4. Oscar time for Tier 3.
5. Who fetches the SPX close each evening (Section 4.1).

## 1. What the exercise is for

**Practice.** A desk calibrates to price and hedge what is not quoted, consistently with
what is. Every day, at the close and often intraday, it:
1. builds forwards and discounting per expiry;
2. marks the implied-vol surface, which prices vanillas;
3. calibrates a dynamic model, because exotics depend on what the surface does not pin
   (the forward smile, path dependence);
4. prices and risk-manages the book.

Product control then verifies the marks and sets reserves. In the EU, prudent valuation
asks for a 90% range for market-price uncertainty and for model risk. Requirements:
reprice the hedges within the bid–ask, stay stable from day to day, take seconds per
name, and explain which quote drives which number. The reserve range is produced ad hoc
today: refits on other instrument sets, other models, bumps. **The calibration set is
chosen from the product.**

**What Bayes adds.**
- The posterior is every parameter set consistent with the quotes. Pricing the product
  under each draw gives its range, and so the reserve.
- The error model replaces ad hoc weights.
- An estimated model error says how badly, and where, the model misfits. That feeds
  model risk but does not cover it.
- Yesterday's posterior can be today's prior.

**What SBI adds and demands.**
- *Adds:* seconds per posterior even when the likelihood is Monte Carlo only; coverage
  that can be checked on thousands of simulated days, which a 90% reserve needs; one
  network across names and days.
- *Demands:* a fixed observation design, and a simulator that reproduces the whole data
  pipeline, error model included. SBC checks the inference given the simulator, not the
  simulator. So a network must also be checked against an exact posterior on real data.

**Hence the exercise.** Heston on SPX is the one case where every step, including the
error model, can be checked against an exact posterior on real data. Rough Bergomi shows
what changes when no cheap exact posterior exists.

## 2. What is wrong now, with evidence

### 2.1 Noise and error

| Finding | Number | Source |
| --- | --- | --- |
| Rule vs bid–ask | median 8.2× SPY's half-spreads (2–36×; about 4× ATM, 15× call wing); 7.3–7.7× SPX's | noise session |
| SPX bid–ask, 9 Nov | 0.02–0.05 vp ATM; 0.05–0.17 down to −6 ATM-σ; 0.11–0.25 at −6 to −8; 0.10–0.37 for calls at +2 to +3 | evidence `heston/spx_noise_and_model_error.py` |
| Rule in vol points, SPY 21 Oct | δ = −2: 0.44 (median); δ = +1.5: 4.9; δ = +2: 5.4 (max 7.0) | evidence `heston/eff_noise_dates.py` |
| Rule in vol points, simulated prior centre (28% vol) | 0.28–0.55 everywhere, wings tighter than stated | evidence `heston/test_heston_fisher.py` |
| Error budget, SPY 6 Nov | bid–ask 0.04–0.13 (0.26–0.37 at δ = +2); spline reading ≈ 0.01; maturity interpolation 0.05–0.25; early exercise up to ≈ 1.1 at the 1-year ATM put; Heston misfit 0.8–1.2 RMS, ≈ 99% of the variance | noise session |
| Heston misfit on SPX, κ free | 1.00 vp RMS = 17× the median half-spread; residual correlation 0.80 (strikes), 0.85 (maturities) | evidence `heston/spx_noise_and_model_error.py` |
| Rule's fit to data | χ² 66.5 on 63 quotes on 21 Oct (right by luck), 508 on 6 Nov (about 3× too tight) | noise session |
| Estimated model error | independent: τ 0.69–0.98 (21 Oct), 1.06–1.45 (6 Nov) vp; smooth: τ_g 2.1–2.6 vp, ℓ_δ ≈ 1.6, ℓ_T 1.5–2.2 (tentative: effective sample size 128–251 of 4,000) | noise session |
| Networks under real misfit | rule-trained: 46–72% coverage at τ = 0.5–1 vp, 22–37% above; independent-τ network: 0.87–0.90 on its own simulations, 18–58% under smooth error; τ estimated 0.29 / 0.76 against exact 0.81 / 1.24 | noise session (smaller test networks) |
| Widths, Heston on SPX (1-year variance-swap sd) | bid–ask only 0.017 vp; independent 1 vp: 0.43; smooth 1 vp: 0.77 | evidence `heston/grid_vs_error_structure.py` |

### 2.2 Grid placement

| Finding | Number | Source |
| --- | --- | --- |
| Current design | 7 maturities (5–252 business days) × δ ∈ {−2, …, +2}, k = δ·0.20·√T | `tutorial/src/sbi4f_tutorial/shared/surface.py:13-17, 74` |
| Where SPY quotes reach, 21 Oct | δ = −8.2 to −14.5 (20%-vol units) on the expiries near the design maturities | tutorial asset `spy/surface.json` |
| Where SPX quotes reach, 9 Nov | puts −12 to −16 ATM-σ up to 2 months, about −8 at 6–12 months; calls +2.2 to +3.1 | evidence |
| What the 9 grid points miss | 1-year variance-swap vol 18.9% from the grid vs 21.8% from the full strip (SPY 21 Oct) | course Lecture 5, §5.9 |
| Which region pins what (Heston, SPX, smooth error; % sd increase if removed) | deep puts: κ 59, vol-of-vol 51, θ 33, variance swap 27; ATM: v0 21, vol-of-vol 25; calls: ρ 72; ≤ 1 month: vol-of-vol 108, v0 79; ≥ 2 months: θ 303, variance swap 287, κ 232; near puts ≤ 5 | evidence `heston/region_roles.py` |
| No missing quotes | 0 of 63 points outside quoted strikes, on SPY 21 Oct, SPY 6 Nov and SPX 9 Nov | asset diagnostics; evidence |

### 2.3 How the two interact

- **The rule decides placement silently.** It gives deep puts 0.003–0.03 vol points of
  noise, so the reviewer's put-shifted grid looked 5–25× better (Heston and rough
  Bergomi). That is false precision. It would also make rough Bergomi's Monte Carlo
  target unattainable: Monte Carlo error 40–65× the noise (evidence `rough/`).
- **Under a consistent vol-point error, shifting puts is a trade-off.** Heston, 150 prior
  draws: vol-of-vol 0.83×, θ 0.89×, variance swap 0.83×, ρ 1.48×. Rough Bergomi: H 0.58×,
  η 0.77×, variance swap 0.68×, ρ 1.41×. Scaling by ATM vol on its own is neutral.
- **Under smooth model error, coverage beats density.** Doubling the points in the
  current range gains 2–17%. Widening to −6…+2 ATM-σ gains 23–49% on κ, θ, vol-of-vol
  and the variance swap, with ρ between 3% better and 12% worse. The ranking holds for
  τ from 0.05 to 2.5 vp and correlation lengths from 0.75 to 2.2. Placement can be
  chosen before the error size is known.
- **Giving Heston the put wing doesn't close its short-end variance-swap gap** (SPY
  6 Nov: −3.2 vs −3.9 vp at one week). It misses the 1-week deep put by 11.5 vp. The
  shortfall is real misfit, but its significance depends on the error model: under
  smooth error the bands (1.8–3.0 vp) contain the market from one month to one year
  (noise session, tentative).

### 2.4 Statements to correct (Tier 1)

| Where | Now | Correction |
| --- | --- | --- |
| `tutorial/notebooks/slot4_heston_bayesflow.py:155-158` | noise "specified in implied volatility, 0.3 / 1.0 points" | an assumed model-error allowance, fixed in price units; its vol-point meaning varies; about 8× the bid–ask; not SPY's spread |
| `slot4_heston_bayesflow.py:943-975` | the loose-wing caveat sits in a cell stopped by `mo.stop` | replaced by Section 4.11 (Tier 2), or moved out of the stopped cell (Tier 1 only) |
| `slot4_heston_bayesflow.py:1028-1032` | "variance swaps sit below the market … precise, and wrong" | replaced by D4/D5 (Tier 2); Tier 1 only: state both confounds |
| `slot4` §8 (from line 536) | historical SPY workflow | retired from the live path (Tier 2); Tier 1 only: label every result as conditional on the rule |
| `tutorial/notebooks/slot5_calibration.py:155-163` | same noise text | as for slot 4 |
| `slot5_calibration.py:650-656` | "variance swaps match SPY's by construction" as a reason to switch | the reason is the skew's power law; the curve is supplied from quotes Heston never sees |
| `slot5` §7 (from line 563) | historical SPY studies | label as conditional on the rule and the supplied curve; drop comparative claims |
| `tutorial/curriculum/00-overview.md:48`; `04-slot4…md:26, 46, 63, 68`; `05-slot5…md:29, 49, 115, 131`; `06-slot6…md:28, 61` | variance-swap gap as the case against Heston; "the design's noise" | align with the above (check line numbers when editing) |
| `tutorial/CLAIMS.md` | claims tied to the rule or the variance-swap gap | through `pixi run claims` only |
| Slides (Overleaf deck) | same statements, if present | same corrections |

### 2.5 The reviewer's issue, reassessed

| Claim | Verdict |
| --- | --- |
| Symmetric ±2 at a fixed 20% vol | Right; the noise is tied to 20% too |
| Put quotes reach δ = −8 to −17 | Right in substance. In 20%-vol units: −8.2 to −14.5 (SPY 21 Oct). In ATM-vol units: −13 to −21. |
| A third of the points are far calls with almost no information | 21/63 at δ ≥ +1. Their low weight comes from the 20% vega conversion (5–7 vp of effective noise against real half-spreads of 0.1–0.4), not from where the points sit. |
| Some have no quotes at all | Wrong: 0 points outside quoted strikes |
| Re-pricing costs | Right: 1.29 CPU-h per 100k Heston surfaces; 331 per 100k rough (370 for the 112k bank) |
| The fix (shift and ATM-scale) | Under the current rule its gain is false precision. Under consistent error it is a trade-off. Under smooth error a put-heavy layout that keeps the calls is right (Section 4.3). |

### 2.6 Defects found by the noise session (both on the SPY path that Tier 2 retires)

- `lab/src/sbi4finance/calibration/chain.py` converts half-spreads to vol with the
  quote's own vega, then back to price with vega at 20% (lines 147–149 and 264–265).
  The SPX pipeline below works in vol units throughout, so this is not fixed in place.
- `tutorial/src/sbi4f_tutorial/shared/inference.py:36-40` (`use_sampler`) reportedly runs
  its time grid backwards under BayesFlow 2.1. Verify; fix it if any live path calls it.
  `full_surface.sample` does not.

## 3. Principles used for the rebuild

1. **Calibrate for the product.** Cover the regions the product's value and hedges
   depend on, and the model features behind them.
2. **A quote's value is sensitivity² ÷ error variance**, both in vol points. Placement and
   the error model are one decision.
3. **Three error sources.** Measurement (bid–ask), processing (forwards, interpolation,
   early exercise) and model error. Which one dominates depends on the name: model error
   on SPX, the quotes on thin chains.
4. **Model error is smooth.** Treat it as correlated. Cover new regions rather than
   densify. Never let a fixed error rule set the widths.
5. **Coverage rule.** Cover a region when it is relevant to the goal, reliably quoted
   every day the network is used, and within the model's scope. If it is relevant but
   not quoted, report extrapolation. If it is quoted but not relevant, hold it out for
   checks.
6. **Validate in four steps.** For SBI, the simulator reproduces the pipeline, error
   included. Then: exact posterior on real data → network matches it → coverage on
   simulations → predictive checks on held-out quotes.
7. **Same information for every model compared.**

## 4. The rebuilt exercise (Tier 2)

### 4.1 Data

- **Source.** Cboe delayed quotes `_SPX.json`, fetched after each close (after 21:30 UTC)
  on 9, 10, 11 and 12 November. The 9 November close is already fetched: last trade
  16:14:59 ET, SHA-256 `209765af…a8a1` (evidence README). Raw files stay outside Git
  (Drive). Record each SHA-256 in `lab/data/SOURCES.md`.
- **Expiries.** SPXW (PM-settled) wherever listed; AM-settled SPX only on dates with no
  SPXW (beyond 30 September 2027). Exclude SPXO.
- **Rates and forwards.** That day's 3-month and 2-year Treasury yields. Forwards from
  put–call parity per expiry; SPX is European, so there is no early-exercise bias.
- **Reading.** `chain.build_surface(rows, rate=…, deltas=LAYOUT_E, sigma_ref=0.16)`, which
  places points at k = δ·0.16·√T. Record the reading counts and require 0 points outside
  quoted strikes. Keep each point's interpolated half-spread in vol units without the
  design floor; that is h_c for that day.
- **Held out, not used for fitting:**
  - the model-free variance-swap level per design maturity, from strips reaching
    δ ≤ −8 (existing replication code);
  - cleaned quotes beyond the grid (below −5 or above +2) and on expiries the reading
    does not use.
- **Code.** New `lab/checks/spx_week/ingest.py` writes one derived JSON per day; no raw
  quotes are written.

### 4.2 Model and prior

- **Heston with κ free.** Parameters: κ, v0, θ, vol-of-vol ξ, ρ. Prior:
  - κ log-uniform on [0.5, 12];
  - v0 and θ uniform on [0.004, 0.16] (6–40% vol);
  - ξ log-uniform on [0.2, 4];
  - ρ uniform on [−0.98, −0.2].

  This covers the 9 November SPX fit (κ 7.3, ξ 2.5, ρ −0.72), which the current box
  (ξ ≤ 1.5, κ = 2) excludes.
- **Model error.** τ log-uniform on [0.1, 5] vol points.
- **Pricing.** `heston_options.model.european_price` with the task's quadrature ladder,
  refined until the change is ≤ 0.01 vp. Record failed draws, then reject them.
- **Prior predictive check (V1).** On each day, SPX's 63 implied vols lie inside the
  1–99% band at every point.

### 4.3 Observation design: layout E

- **Points.** Maturities 5, 10, 21, 42, 63, 126, 252 business days (unchanged).
  δ ∈ {−5, −3.5, −2.25, −1.25, −0.5, 0, +0.5, +1, +2}, k = δ·0.16·√T. Out of the money:
  puts at δ ≤ 0. 63 contracts.
- **Why this layout.** Posterior sd relative to the current grid, Heston on SPX:

  | Error model | κ | v0 | θ | vol-of-vol | ρ | 1-year variance swap |
  | --- | --- | --- | --- | --- | --- | --- |
  | smooth, τ = 1 vp, ℓ = 0.75 / 1.15 | 0.65 | 0.92 | 0.72 | 0.68 | 1.09 | 0.75 |
  | smooth, τ = 2.5 vp, ℓ = 1.6 / 2.0 | 0.78 | 0.81 | 0.77 | 0.83 | 1.31 | 0.78 |

  Layouts that drop the near-the-money calls lose ρ by 2.4–4.1×; one without an ATM point
  loses it by 2.5× (evidence `heston/layout_choice.py`).
- **Why a fixed 16%.** SPX's ATM vol was 15.6–17.1% on 9 November. One name in one week
  does not need data-dependent strikes. State this as a choice for SPX.
- **Why 63 contracts.** The 7 × 9 shape keeps the existing shape-dependent code. The
  cost is up to 31% on ρ compared with a denser wide grid.
- **Code.** Add a design registry to `tutorial/src/sbi4f_tutorial/shared/surface.py`
  (`DESIGNS = {"canonical_v1": …, "spx_put_heavy_v1": …}`, `design_grid(name)`).
  Validators then compare a task's contracts with its declared design;
  `full_surface.validate_task` currently accepts only the canonical design. Existing tasks
  and assets are unchanged.

### 4.4 Error model, in implied-vol space

- **Observation.** y_c = IV_c(θ) + d_c + e_c.
  - e ~ N(0, diag(h²)), with h_c the SPX half-spread at point c. It is the median over the
    days fitted, fixed in the task as a known design constant.
  - d ~ N(0, τ² K), K_cc′ = exp(−(δ_c − δ_c′)² / 2ℓ_δ² − (ln T_c − ln T_c′)² / 2ℓ_T²).
- **Length scales are fixed before training.** Use the profile marginal likelihood on the
  9 and 10 November closes over ℓ_δ ∈ {0.75, 1, 1.5, 2} and ℓ_T ∈ {1, 1.5, 2}.
- **τ is inferred.** It is a network output and a reference parameter.
- **Why implied-vol space.** The covariance does not depend on θ, so the exact likelihood
  is a fixed Gaussian. The network sees the same quantity, and cleaned real quotes always
  invert.
- **Relation to the noise session's covariance.** That form was
  diag(h²) + V(τ_g² K + τ_n² I)V, with V the vega at the model's own vol. This one is the
  same to first order. Add the independent term τ_n only if the pilot fits need it:
  whitened residual lag-1 correlation still above 0.3, or χ²/dof well above 1.
- **Scope.** This replaces `quote_noise_std` for the new task only.

### 4.5 Simulator and bank

- **Clean bank.** For each prior draw: Fourier prices at the 63 contracts, then implied
  vols. 200,000 draws: 180k train, 10k validation, 10k test. About 3–4 CPU-h; Heston costs
  0.013 CPU-h per 1,000 surfaces at 63 contracts, plus inversion.
- **Errors are drawn fresh at training time,** as `full_surface.fresh_noise` does now:
  τ ~ prior, d = τ L z (L the Cholesky factor of K, computed once), e = h ⊙ z′. τ is
  emitted as a target. The clean bank does not depend on ℓ or h, so it can start before
  the pilot fits finish.
- **Code.** New `lab/studies/spx_heston/bank.py` reuses the tutorial pricing. Use
  `SeedSequence` streams and a manifest with SHA-256.

### 4.6 Network

- The BayesFlow 2.1 workflow of `shared/full_surface.py`: vector encoder, the family and
  settings of the current Heston backbone.
- Input: the 63 implied vols, standardized with fixed constants stored in the task.
- Targets: κ, v0, θ, ξ, ρ, τ, with the logit box adapter.
- Training budget: at least 30,000 updates, several seeds and settings, using
  `lab/scripts/train_full_surface.py` as updated on `main`. On the current Heston task,
  30,000 updates take about 100 s per fit on one L40S. It was the only budget to pass
  the rank and coverage checks, and it still missed the conditional checks on two of
  three reference cases (`NUMERICAL-EXPERIMENT-PLAN.md`, P4, 8 November). Record
  `training.json` and the checkpoint SHA-256.
- Bank pricing uses the checked Heston pricing speed-ups merged in PR #77.
- New task id `heston_spx_iv_v1`. Pricing parameters and the error parameter are kept
  apart, because `model_parameters` passes every inferred name to the pricer.

### 4.7 Exact reference

- **Likelihood.** The Gaussian of Section 4.4 over (κ, v0, θ, ξ, ρ, τ), with model implied
  vols from Fourier prices. Add a new builder next to `heston_log_likelihood` in
  `lab/src/sbi4finance/calibration/reference.py`.
- **Sampler.** PyMC via `sample_deterministic_reference` (MCMC policy): 4 chains × 1,000
  draws, R̂ ≤ 1.01, ESS ≥ 400 per parameter. Record seeds, diagnostics and versions.
- **Cases.** 8 synthetic cases, plus each real day as it arrives. About 15–40 CPU-min each.
- **Cross-check.** Importance sampling with the network as proposal; report its ESS.

### 4.8 Validation gates

| Gate | Test | Pass |
| --- | --- | --- |
| V1 | Prior predictive vs each SPX day | every point inside the 1–99% band |
| V2 | Pilot exact fits (9, 10 Nov) | whitened residuals: lag-1 correlation < 0.3 and χ²/dof ≈ 1; ℓ fixed |
| V3 | SBC and coverage on 2,000 held-out simulations | 90% intervals cover 87–93% for all 6 targets; rank histograms pass χ² |
| V4 | Network vs exact on the 8 synthetic cases and each real day | per target, with the repo's existing conditional limits: mean gap within ±0.3 reference sd; 90% width ratio 0.8–1.25; τ within ±20%; product 90% quantiles within ±10% of the range. Expect this to be the binding gate (Section 4.6). |
| V5 | Posterior predictive on held-out quotes, discrepancy kriged from the grid residuals | report coverage of 90% intervals (expect 80–97%); do not tune to it |
| V6 | Market checks held out of the fit: model-free variance swap (1 m, 3 m, 1 y) and the listed 2-year 90% put | report inside or outside the 90% band per day and maturity; do not tune to them |

**If V4 fails:** show real-data posteriors from the exact reference, labelled as such.
Show the network on synthetic data only, and state the V4 result.

### 4.9 After the fit: products and what the full posterior shows

**The goal.** Mark products that SPX's listed options do not price directly, consistently
with the listed smile: a 6 m → 1 y forward-start call, 3 m and 1 y volatility swaps and a
1-year down-and-in put. Give each a range from the full Heston posterior instead of a
single point calibration:
- a prudent value at 90%;
- a book-level range from the same draws;
- widths that grow when the model misfits the quotes.

Coverage of these ranges is validated in simulation, under the stated model and error
model (D3). On real data, the true price of a forward start or barrier is never observed.
The checks there are what the market prices independently and the fit does not use:
- the variance swap, replicated model-free from the strip;
- the 2-year 90% put, listed on SPX but beyond the design's last maturity.

The range is within-model and covers misfit on quoted points only. It feeds model risk
and is not a model-risk reserve.

Price in forward units with common random numbers across draws, using
`lab/src/sbi4finance/calibration/products.py`.

| Product | Depends on | Method |
| --- | --- | --- |
| Variance swap, 1 m, 3 m, 1 y | level and term structure of variance; the put wing | analytic in κ, v0, θ; held-out model-free benchmark |
| Volatility swap, 3 m, 1 y | vol-of-vol (convexity) | `volatility_swap` |
| Forward-start ATM call, 6 m → 1 y (one cliquet period) | the forward smile, which today's smile does not pin | `forward_start_call_spot` |
| Down-and-in put, 1 y, strike 100%, barrier 70%, daily monitoring | downside dynamics, deep puts | new: spot paths from the stored variance paths plus independent normals. Test: in the ξ → 0 limit, against Black–Scholes with the Broadie–Glasserman shift |
| Put, 2 y, 90% (check, not a product to mark) | beyond the design's last maturity, but listed on SPX | Fourier; compared with its listed mid and spread on each day |
| Book: short 1 down-and-in, long 1 forward start, short 1 1-year volatility swap | joint | from the same draws |

**Baselines:**
- point calibration: maximum likelihood with the same covariance;
- delta method: Gauss–Newton covariance, products linearized;
- practitioner refits: drop each maturity in turn, take the min–max of the 7 prices.

**Demonstrations:**

| | Shows | Figure or table |
| --- | --- | --- |
| D1 | One posterior prices every product. Widths differ by what the quotes pin: tight for a vanilla reprice, wide for the forward start; the 2-year put shows extrapolation against a listed price. | 5/50/95% per product against the point, delta-method and refit ranges. The reserve is the 90% quantile minus the point (`products.reserve`). |
| D2 | Coherent aggregation | the book's 90% range against the sum of the standalone ranges |
| D3 | Ranges that can be checked, because amortization makes thousands of posteriors cheap | on 1,000 simulated days (prior) and 300 days near SPX: how often each 90% range contains the true product price. Posterior (target ≈ 90%) against delta method, refits and point (0%). |
| D4 | "Precise but wrong" comes from the error model | one day under three error models (the rule, independent, smooth inferred): variance-swap bands against the model-free value; product ranges as τ grows |
| D5 | How wrong Heston is, and where | the posterior of τ per day, and where the residual sits (short-dated deep puts). This replaces the old variance-swap argument and motivates rough or jump models. |
| D6 | Day-to-day behaviour | posterior medians and ranges on 9–12 November against daily point calibrations. Illustrative only: four days. |

**Scope statement in the notebook.** The range is parameter uncertainty within Heston
under the stated error model. The smooth error term covers misfit on quoted points, not
the forward smile beyond them. It feeds model risk but is not a model-risk reserve.

### 4.10 Rough Bergomi

- **Tier 1, always.**
  - Synthetic tasks unchanged.
  - Noise text: an assumed model-error allowance, with its 20%-vega meaning stated.
  - Drop "variance swaps match by construction" as a reason to switch, and every
    comparative SPY claim.
  - Motivation: SPX's ATM skew term structure from the 9–12 November readings
    (`smile_statistics`, `skew_power_law`).
- **Tier 3, optional.** Retrain the known-curve or joint task with the error model of
  Section 4.4 on the existing 100k bank (canonical grid), then audit one SPX day by
  importance sampling (about 85 CPU-min).
  - Precondition: for every contract, the bank's Monte Carlo SE in vol points is
    ≤ 0.25 × (h_c + τ_min). The far-call wing of low-vol draws may fail this, because the
    bank's accuracy there was set by the rule's loose wing.
  - If it fails: raise τ_min for rough Bergomi, or exclude those contracts, and state it.
- **Not attempted.** Re-pricing rough Bergomi on layout E: 330–370 CPU-h plus training.

### 4.11 What participants see (slot 4, rewritten in place; `pixi run heston-options` unchanged)

1. The job: the daily workflow, and what Bayes and SBI add.
2. SPX data and the design: the smile, the region table, where the 63 quotes sit.
3. The error budget: bid–ask against model error, then the error model.
4. Model, prior, simulator and one training pair, with the code visible.
5. The amortized posterior on a synthetic case.
6. Checks: SBC and coverage; the network against the exact reference.
7. SPX, four days: posterior, τ, predictive check, model-free variance-swap check.
8. Products: D1–D4, D5, D6.
9. What this covers and what it does not.

Retire §8 (historical SPY) from the live path. Old assets remain until a
dependency-checked retirement batch.

## 5. Work plan, Tuesday 10 – Friday 13 November

| Id | Task | Files | Needs | When | Done when |
| --- | --- | --- | --- | --- | --- |
| W0 | Decisions; owners; nightly SPX fetch | `lab/data/SOURCES.md` | — | Tue 09:00 | decisions recorded; Tuesday's close fetched by 22:00 UTC |
| W1 | Tier 1 text corrections | Section 2.4 list; slides | — | Tue | review by a second author |
| W2 | SPX ingest and readings | `lab/checks/spx_week/ingest.py` | 9 Nov close | Tue | 0 points outside quotes; h, held-out sets and model-free variance swaps written |
| W3 | Pilot exact fits; fix ℓ, h, priors | `lab/src/sbi4finance/calibration/reference.py` (new builder) | W2 | Tue–Wed 10:00 | V1, V2 |
| W4 | Runtime: design registry, task schema, implied-vol conditioning, τ augmentation | `tutorial/src/sbi4f_tutorial/shared/{surface,tasks,full_surface}.py`; tests | — | Tue | tutorial `pixi run test`; lab `pixi run test`; `runtime-check` |
| W5 | Clean bank (200k); training | `lab/studies/spx_heston/` | W4 (bank); W3 (training) | Tue night – Wed | V3 |
| W6 | Exact references: 8 synthetic cases and each real day | as W3 | W3, W5 | Wed – Thu | V4 per case |
| W7 | Products: down-and-in pricer with tests; pricing under draws; baselines; D1–D6; D3 coverage study | `lab/src/sbi4finance/calibration/products.py`; `lab/studies/spx_heston/products.py` | W5, W6 | Wed – Thu | Monte Carlo SE ≤ 2% of each 90% range; D3 posterior coverage 87–93% |
| W8 | Notebook and documents | slot 4, slot 5 (Tier 1), curriculum; assets via `lab/studies/accept.py` (add a bundle format for this task); `pixi run manifest`, `pixi run claims` | W1, W5–W7 | Thu – Fri | tutorial `pixi run test`, `test-notebooks`, `check`; lab `pixi run test`, `release-check` |
| W9 | Rehearsal, browser check, freeze | — | W8 | Fri | frozen 18:00 CET |
| W10 | Optional Tier 3 (Section 4.10) | rough tasks | Tier 2 on track | Wed 12:00 – Thu | precondition met; SBC; one audit |

**Parallel work.** W1, W2 and W4 can run in separate sessions on Tuesday. The critical
path is W4 → W5 → W6 → W7 → W8.

## 6. Gates and fallbacks

| When | Gate | If missed |
| --- | --- | --- |
| Tue 20:00 | W1 merged; W4 tests green; bank running; ℓ fixed | stop Tier 2; deliver Tier 1 |
| Wed 20:00 | V3 passed; V4 on two real days | fallback: exact posteriors for SPX, network on synthetic data only, results stated |
| Thu 20:00 | D1–D4 computed; notebook integrated; tests green | cut D5 and D6 to text; keep D1–D4 |
| Fri 18:00 | freeze | present the last green state |

## 7. Compute and data

| Item | Estimate |
| --- | --- |
| Heston clean bank, 200k | 3–4 CPU-h |
| Network training | about 100 s per 30,000-update fit on one L40S (8 November matrix); 10–20 fits for seeds and settings is under an hour |
| Exact references, 12 posteriors | 3–8 CPU-h |
| Products, including D3 (1,000 days × 100 draws × 20k paths) | 1–4 CPU-h |
| Tier 3 | retraining (GPU-hours) plus about 1.5 CPU-h per audited day |
| Data | four SPX closes, about 14 MB each, outside Git |

## 8. Risks

1. **The smooth error absorbs parameter signal.** Fix ℓ from the pilot fits. Report the
   wider posteriors as the honest result. V2 and V4 detect a problem.
2. **The network misses the exact posterior.** The noise session's independent-error
   network did, and so did the best fit of the 8 November training matrix on the current
   task. Train on the smooth error, which is what the data show. Give the network the
   longest budget and several seeds, and pick by V4 on synthetic cases, never on the
   real days. Fallback in Section 6.
3. **Fourier accuracy at extreme κ and ξ** for short-dated deep puts. Use the quadrature
   ladder with refinement checks, and record rejected draws.
4. **Event risk inside the 1-week maturity** (CPI release that week). The smooth term
   absorbs it; say so.
5. **Code changes take longer than planned.** 63 contracts and a registry limit the
   changes; the Tuesday gate catches the rest.
6. **Public site.** No unreleased results go on `site/` (AGENTS.md).

## 9. Evidence index

- [`design-and-error/summary.json`](../evidence/pilot-results/2026-11-09/design-and-error/summary.json):
  this plan's numbers, with the scripts that produced them.
- Noise session, 9 November: error budget, exact fits under three error models,
  test-network coverage and the two defects. Not committed; numbers quoted above.
- Tutorial assets: `results/calibration/spy/surface.json` (S8);
  `data/heston-spy-exact.json`; `results/calibration/{heston,rough-pilot}/s1/bank-quality.json`
  (costs).

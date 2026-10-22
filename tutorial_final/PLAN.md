# Plan: `tutorial_final/` for ICAIF '26

Revision 3, 22 October 2026. Owner: Francesco Muia. Revision 2 (21 Oct) chose the application;
this revision bends the design to the code that already exists and maps every component to it.

`tutorial_final/` will hold what we present in Milan on 14–15 November: the notebooks, the
studies that produce their numbers, and the results they load. The rest of the repository stays
as the research archive. This file is the working plan; a participant README replaces it at
release.

## 1. The decision

**Slot 5 is Bayesian calibration of a simulation-only volatility model to the option smile,
carried into the price of a product the smile does not pin down.** Rough Bergomi calibrated to a
vanilla surface; the posterior propagated into a forward-start option, with the variance swap as
the control; the 90% quantile of the price range as the model-risk reserve. Heston, which has an
exact likelihood through the Fourier pricer already in the repository, plays the role GBM plays
in slot 4: the model on which the network is checked against the exact answer.

The route is GBM (slot 4, under P, from returns) → the smile (slot 5, under Q, from quotes). The
tutorial's title, *Amortized Bayesian Calibration of Volatility Models*, describes it.

The SVJ-from-returns lesson becomes an optional companion. Its audit design is reused (§6).

The [technical design document](../docs/design/calibration-design-document.md) states the
method, the mathematics, the validation plan and the planned evidence in the form of the SVJ
design document; its placeholders ⟨S1⟩–⟨S9⟩ are filled by the studies of §6.4.

## 2. The thesis

1. **Bayes pays when the decision needs a quantile of what the data do not pin down.** A vanilla
   surface identifies the skew (ρη) and its term structure (H) far better than the vol-of-vol η
   alone; the products a desk marks with the calibrated model depend on η alone. The posterior
   turns bid-ask widths into a price range and a reserve. A point calibration returns one number;
   a multi-start returns the modes, not the ridge.
2. **Simulation-based, amortized inference pays when the model is priced by simulation.** For a
   Monte-Carlo-priced model the exact posterior costs hours per surface; the amortized posterior,
   trained once on simulated surfaces, returns it in seconds per underlying, every day, and can
   be validated on thousands of synthetic surfaces.
3. **Where neither holds, we say so.** Heston with a Fourier pricer needs neither; it is the
   check. GBM with 1,000 days of returns needs neither; slot 4 says so.

## 3. Constraints

- **C1, relevance.** Every desk recalibrates daily; exotic marks, hedge ratios and reserves depend
  on the calibration. Prudent valuation (Regulation (EU) 2016/101, Art. 11) asks for a range of
  plausible calibrations with a 90% point; IFRS 13 asks for Level 3 sensitivity ranges; SR 11-7
  (replaced by SR 26-2 in April 2026) asks for model-risk assessment. Rough and path-dependent volatility are the research frontier
  this audience knows, and deep calibration is the method it knows for them.
- **C2, necessity.** Each claim is stated against the strongest alternative (§7).
- **C3, route.** GBM → the smile. SVJ, HKDE jumps, MMAR and the one-factor Bergomi model are
  archive or companions.
- **C4, reuse.** Every component is built from the existing option-bank, pricing, inference,
  correction and surrogate code unless §6 says otherwise. New code is listed and sized there.

## 4. The slot-5 lesson

Working title: **Ten calibrations fit the smile. Which price do you mark?** About 40 minutes
(decision D2 fixes it). Francesco presents.

| Min | Section | On screen | Live or cached |
|---|---|---|---|
| 0–4 | The question. A desk calibrates to today's smile and marks a forward-start option. Ten calibrations fit equally well. Which price, and what reserve? | The SPY smile; the regulation's words | cached |
| 4–9 | GBM under Q. Black–Scholes gives one volatility for every strike; the market does not. P and Q in one slide: slot 4's posterior forecasts, this one prices. | Flat line against the SPY smile; the ATM-skew term structure and its power law | live, seconds |
| 9–16 | Heston: the smile explained. Fourier likelihood, exact posterior, and the slot-4 workflow with a new simulator. The network against the exact posterior. | Posterior pairs, exact against network | exact cached, network live |
| 16–21 | Rough Bergomi: why rough (skew ∝ T^(H−1/2)), why Monte Carlo only, why no exact posterior. | Skew term structure under both models; one priced surface and its cost | cached |
| 21–26 | The amortized posterior for the SPY surface. The ρ–η ridge; H from the term structure. | Posterior pairs over H, η, ρ | live, seconds |
| 26–31 | Is it right? SBC on 2,000 synthetic surfaces; the audit of this surface against the Monte Carlo likelihood; agreement with MCMC on a neural pricer, the deep-calibration route. | SBC; audit efficiency; two posteriors overlaid | SBC cached, audit live |
| 31–35 | The classical route on the same surface: multi-start least squares and a regularized fit; what they return and what they miss. | The modes on the ridge | cached |
| 35–39 | From posterior to prices: variance swap (zero range), volatility swap, forward-start option; the 90% quantile against the point calibration: the reserve. | Three price distributions; the reserve | price map live |
| 39–40 | Across models: Heston and rough Bergomi on the same smile price the forward-start differently. Within-model is a lower bound. Hand-over to slot 6. Where it leads: path-dependent volatility, joint SPX–VIX. | Two price distributions | cached |

Exercises: change the quote-noise model and watch the reserve (a reweighting, no retraining);
drop the two shortest maturities and watch H; calibrate Heston to the rough-Bergomi surface
(misspecification, for slot 6).

## 5. Task specification v1, as built

Everything below is expressed as task JSONs (`configs/tasks/calibration_rough_bank.json`,
`calibration_heston_bank.json`, `calibration_rough_pilot.json`) in `calibration_bank_v1`, a
mirror of the `option_bank_v1` schema that keeps its seed streams, shards, assembler and
fail-closed rules and adds the three extensions marked **[ext]** (§6.2). The archived
option-bank modules are imported, not edited. The rough pilot (`rough_option_panel_v1`: nine
put prices, ρ fixed, flat curve, H ∈ [0.1, 0.3], η ∈ [0.4, 1.2]) was the starting point. v0
differed in three places, all replaced after measurement: a piecewise-linear curve on 8 knots,
2^17 plain Monte Carlo paths with a 0.1 budget, and implied volatilities as conditions. Freeze
on Fri 23 Oct (D4).

| Item | v1 |
|---|---|
| Observation | A currency-price panel `[B, 63, 1]`, as every bank today. Spot 1.0, rate and dividend 0, so prices are fractions of the forward; a real quote is divided by its maturity's forward. The network conditions on each price divided by its quote-noise standard deviation, which is defined for any value including negative noisy quotes; implied vols are computed for plots and for reading the real chain. |
| Contracts (fixed, as the schema requires) | 7 maturities in trading days {5, 10, 21, 42, 63, 126, 252}/252 × 9 strikes K = exp(δ σ_ref √T), δ ∈ {−2, …, 2} in steps of 0.5, σ_ref = 20%. Puts below the forward, calls above (both supported). Maturity-major order. |
| Quote noise | Gaussian in price, per contract **[ext 1]**: `quote_noise_std` becomes a list of 63 values, each the Black–Scholes vega at (σ_ref, K, T) times an implied-vol noise of 0.3 vol points at δ = 0 rising linearly to 1.0 at \|δ\| = 2. Fixed per task; changing it is a reweighting exercise. |
| Rough Bergomi unknowns | Uniform, as `generate_examples` draws: H ∈ (0.02, 0.35), η ∈ (0.5, 3.5), ρ ∈ (−0.99, −0.2). No fixed parameters. |
| Forward-variance curve ξ₀ | Piecewise constant on the 7 intervals between design maturities, through the existing `curve_values`. Drawn per example **[ext 2]**: level σ₀ ~ U(10%, 40%), slope s ~ U(−0.3, 0.6) and a log perturbation u_j ~ U(−0.2, 0.2) per interval, with level σ₀²(1 + s t_mid,j) e^(u_j); the 7 levels are stored in the bank and conditioned on as logarithms. For the real surface the levels come from log-contract replication per expiration, interpolated in total variance at the design maturities. |
| Heston check task | `model: "hkde"` with `jump_intensity` fixed at 0 (the pricer's explicit no-jump branch); unknowns `initial_variance` **[ext 3]**, `theta`, `vol_of_vol`, `rho`, uniform; `kappa` fixed at 2. Heston's forward-variance curve θ + (v₀ − θ)e^(−κt) is parametric: the contrast with Bergomi is a teaching point. Same contracts and noise. |
| Pricing numerics | Rough: the existing conditional-put estimator and hybrid driver, with antithetic pairs and two zero-mean control variates per maturity, in three maturity blocks **(part of ext 2's config)**: {5, 10} d with 80 steps, {21, 42, 63} d with 504 steps (8 per day) and {126, 252} d with 504 steps (2 per day). Paths ladder 2^15 to 2^21 per block until MC SE ≤ 0.25 quote SD on every quote; at the last rung a block is kept up to 0.5 quote SD and stops the shard above. The short block and the longer ladder exist because the first pilot stopped on three corner-of-the-prior surfaces at 2^18. Heston: the Fourier refinement rule on a quadrature ladder (512, 400) … (8192, 6000), refined-minus-base ≤ 0.1 quote SD. |
| Products under the posterior | Variance swap 63 d (control: fixed by ξ₀); volatility swap 63 d; ATM forward-start call, strike set at 63 d, expiry 126 d; one vanilla outside the grid (2 y, 90% put). Priced from `simulate_rough` / `hkde.simulate` paths with common random numbers; a learned price map, validated by the surrogate protocol, serves the live cell on synthetic surfaces; the real surface is priced by MC per posterior draw and cached. |
| Reserve | 90% quantile of the posterior price distribution minus the point-calibration price. |
| Network | Stage B: `bf.networks.FlowMatching` (slot 4's class, TimeMLP 4 × 256; trained with times drawn as τ = u^0.5, toward the data end, and sampled with 100 Euler steps on a grid dense near that end: with BayesFlow's defaults the sharp posterior of Heston's v₀ came out up to 35% too narrow and off-centre) on `inference_conditions` = 63 prices divided by their noise sd + 7 log curve levels (63 for Heston); the existing bounds-constraining adapter. Stage D: `bf.networks.SetTransformer` over tokens (k/√T, T, iv, noise sd, mask) with random masking, so a real chain with missing quotes is handled as trained. |
| Bank | 100,000 train, 5,000 validation, 5,000 test, plus 2,000 reserved for SBC; a fresh quote error in every training batch reuses the priced surfaces at no pricing cost (four fixed draws per surface left the first Heston network overconfident). |

## 6. Reuse map

What exists, what each lesson component takes from it, and the change needed. "As is" means
no code change; "extend" means a small change inside the existing function; "generalize" means
removing a hard-coded task shape; "new" is listed with its size.

### 6.1 Component by component

| Component | Existing code | Change |
|---|---|---|
| Task and bank schema, prior draws, seed streams, manifests, fingerprints | `option_bank.py` (`validate_config`, `generate_examples`, `write_bank`, `load_bank`) | **ext 1–3** (§6.2); otherwise as is |
| Shard generation on Oscar, assembly with checksums | `scripts/generate_option_bank_shard.py`, `assemble_option_bank.py`, `cluster/option_bank_array.sbatch` | as is; two new task JSONs and a job map |
| Rough Bergomi paths | `rough_bergomi.simulate_rough`, `rough_driver` (FFT hybrid), `bergomi.curve_values` | as is |
| Rough European panel pricing with MC covariance | `bergomi.option_panel`, `conditional_put_payoffs` (conditional Black–Scholes estimator, call parity) | as is; called once per maturity block |
| Heston pricing and paths | `hkde.characteristic_function`, `european_price` (no-jump branch exists and is tested), `hkde.simulate` | as is |
| Batch Fourier pricing of parameter rows | `hkde_reference.price_parameters` | generalize: P parameters from the task, v₀ from the row |
| Batch MC pricing of parameter rows (audit likelihood) | `scripts/check_rough_reference.price_parameters` | generalize: 3 parameters, per-row curve; move into the package |
| Quote-noise model | `option_panel.validate_context`, `noisy_panel`; `rough_reference.importance_weights` already takes σ `[C]` | **ext 1** |
| Adapter and workflow builder | `rough_inference.py` (`conditioning_data`, `training_data`, `augmented_training_data`, `build_workflow`) | generalize into `calibration/inference.py`: parameter names, bounds and contract count from the task; curve knots concatenated into the conditions; prices scaled by their noise standard deviations (implied vols for plots); `FlowMatching` in place of `CouplingFlow` |
| Training with selection and reload check | `training_workflow.fit_and_select`, `verify_reload`, `reserve_output_dir`; `scripts/train_rough.py` | as is; the script's accepted-inputs check updated to the new task |
| Exact check by importance reweighting (the audit) | `hkde_correction.propose` (defensive mixture: network + prior + boundary), `correct`, `summarize`, `resample` | generalize into `calibration/correction.py`: P parameters, C contracts, σ `[C]`, a pluggable `price_fn(rows) → [N, C]` (Fourier for Heston, MC for rough). The SVJ `audit.fit_proposal`'s Student-t proposal and `AuditResult.flagged` are the pattern for the per-surface flag |
| Reference diagnostics, weighted summaries, SBC | `rough_reference.weighted_summary`, `reference_diagnostics`, `conditional_metrics`, `randomized_sbc`; `scripts/evaluate_rough.py` (`validation_gates`, `aggregate`) | generalize to P parameters; otherwise as is |
| Per-surface pipeline (prepare → infer → diagnose → correct → fallback → downstream) | `inference_workflow.run_workflow` with `WorkflowHooks`; `hkde_workflow.run_panel` and `HKDEWorkflowConfig` | generalize `run_panel` into `calibration/workflow.py`: downstream = product prices under the weighted draws |
| Learned price map and its validation protocol | `hkde_surrogate.py` (fit, predict, save, load; bound checks) and `scripts/run_hkde_surrogate.py` (held-out error against MC SE, paired propagation, timing) | replace the 2-D Chebyshev basis by a small Keras MLP on 5 inputs (H, η, ρ, σ₀, s); keep the validation protocol and script structure |
| Posterior plots: raw / weighted / reference, efficiency, target-price distributions, PIT | `scripts/plot_hkde_correction.py`, `plot_rough_inference.py`, `svj/plots.py` (style) | as is with new inputs |
| MCMC diagnostics (R-hat, ESS) for the neural-pricer route | `svj/pmmh.rhat_ess` | as is; the random-walk sampler is 60 lines (new, N3) |
| Multi-start calibration structure (starts, bounds, Hessian, bound hits) | `svj/classical.fit_mle`, `wald_standard_errors` | pattern only; the objective is new (N4) |
| Local option-panel artifact (columns, conventions, provenance) | `datasets/options.py`, `datasets/local.py` (`write_artifact`, `load_artifact`) | extend: bid, ask and forward columns as auxiliary; generalize `prepare_hkde_panel` to any task's contract list |
| Cost measurement | `scripts/svj_costs.py` | pattern |
| Notebook text and structure | notebook 10 (local observation → network → why correct → correction → target prices → SBC), notebook 06 (Fourier pricing and bank roles), notebook 08/09 (forward-variance curve, memory kernel, why no continuation state, importance reference, SBC), slot 4 (#37: BayesFlow explanation and diagnostics), slot 5 draft (#38: audit explanation, page structure) | lifted section by section |
| End-to-end template that runs today | notebook 10 with `demo-hkde-corrected`: network proposal → exact Fourier reweighting → two target-price distributions, on the HKDE (θ, λ) task | Stage A0: run it unchanged as the reference implementation while the Heston task is wired |

Not reused: the trained rough checkpoint (`rough-hurst-eta-reference8192-provisional`, nine
puts, two parameters) and the HKDE (θ, λ) checkpoint, except as smoke tests of the loaders; the
SVJ particle filter; the one-factor Bergomi model.

### 6.2 Schema extensions (as built: `calibration/bank.py`, not edits to `option_bank.py`)

Archived results pin the SHA-256 of `option_bank.py`, `option_panel.py` and
`option_bank_shards.py` (for example `docs/results/hkde-inference/bank-manifest.json`), and
the archive's tests refuse to run when they change. The three extensions therefore live in a
mirror module that reuses the archived pricers, seed streams and fail-closed rules by import,
and the archived files are unchanged (a test checks the hashes).

| | Change | Where |
|---|---|---|
| ext 1 | One quote-noise standard deviation per contract; the precision budgets compare per contract | `calibration/tasks.validate_context`, `calibration/bank.generate_examples`, `check_precision` |
| ext 2 | Per-example forward-variance curve: 7 piecewise-constant levels sigma0^2 (1 + s t_mid) e^u, drawn from stream purpose 3 and stored as `context_values [B, 7]`, `context_parameters [B, 2]`; maturity blocks with their own grids; a paths ladder that escalates until MC SE <= 0.25 sigma_c (kept up to 0.5 at the last rung), recorded in `pricing_paths [B, blocks]` | `calibration/tasks.py`, `calibration/pricing.py` |
| ext 3 | `initial_variance` as an inferred Heston parameter; jump intensity fixed at zero | `calibration/tasks.py` |

Banks are written as `calibration_bank_v1`: the option_bank_v1 arrays and roles plus the three
arrays above, compressed. `scripts/calibration_bank.py` (config, job-map, shard, assemble,
local) and `cluster/calibration_bank_array.sbatch` replace the option-bank scripts for them.
The rough pricer adds antithetic paths and two zero-mean control variates per maturity to the
archived conditional estimator; at the prior median this cuts the Monte Carlo error tenfold.

### 6.3 New code

| | Module | Content | Size |
|---|---|---|---|
| N1 | `calibration/surface.py` | Black–Scholes price/vega/implied-vol inversion; design grid; per-maturity smile fit (cubic spline in δ) to read the grid from a chain; forward from put–call parity; ξ₀ by log-contract replication; ATM-skew term structure | ~250 lines |
| N2 | `calibration/products.py` | Payoffs from `simulate_rough` / `hkde.simulate` outputs: variance swap (integrated variance), volatility swap, forward-start call, out-of-grid vanilla via `option_panel`; CRN across posterior draws; MLP price map with the surrogate protocol | ~200 lines |
| N3 | `calibration/pricer_net.py` | NumPy MLP (logit θ, log curve levels) → 63 prices in units of σ_c, trained on the bank's clean `reference_prices` (deep calibration's pricer, free given the bank); vectorized random-walk Metropolis on it; `pmmh.rhat_ess` for diagnostics | ~200 lines |
| N4 | `calibration/classical.py` | Multi-start weighted least squares (trust-region, in logit coordinates) with the exact Heston pricer or the neural pricer; Gauss–Newton covariance and Wald intervals; Tikhonov variant; bound-hit and mode reports | ~150 lines |
| N5 | `calibration/chain.py`, `scripts/build_surface.py` | One SPY end-of-day chain for one date from a vendor file (Alpha Vantage `HISTORICAL_OPTIONS` CSV, a premium endpoint, or Cboe delayed-quotes JSON), cleaning rules, artifact via N1; raw quotes stay out of the repository | ~300 lines |

### 6.4 Studies

Each study is a `pixi` task in `tutorial_final/studies/`, writes to `results/calibration/`, and
starts from the script named.

| ID | Study | Starts from | Output |
|---|---|---|---|
| S1 | Generate the rough and Heston banks on Oscar | `generate_option_bank_shard.py`, `assemble_option_bank.py`, `option_bank_array.sbatch` with the two task JSONs | `bank-manifest.json` |
| S2 | Train the amortized posterior; SBC, recovery, coverage, contraction | `train_rough.py`, `evaluate_rough.py` | `checkpoints/rough_bergomi/`, `validation/` |
| S3 | Heston check: exact posterior (Fourier reweighting of a defensive proposal, and an independent MH chain) against the network on 50 synthetic surfaces and the real one | `evaluate_hkde_correction.py`, `hkde_workflow.run_panel` | `heston-check/` |
| S4 | Neural pricer; MH on it against the amortized posterior on 50 surfaces and the real one | N3, `pmmh.rhat_ess` | `pricer-route/` |
| S5 | Audit: efficiency and max weight on 500 validation surfaces; flag rate; the real surface | `calibration/correction.py`, `check_rough_reference.price_parameters` | `audit/` |
| S6 | Classical route on 200 synthetic surfaces and the real one; Jacobian-interval coverage; modes on the ridge | N4 | `classical/` |
| S7 | Products under the posterior: price distributions, reserves, price-map validation; Heston against rough on the same surface | N2, `run_hkde_surrogate.py` (protocol), `hkde_correction.summarize` | `products/` |
| S8 | Real surface: SPY chain, implied vols, ξ₀, skew term structure; derived data only | N5, N1, `datasets/local.write_artifact` | `data/spy-surface/`, `data/SOURCES.md` |
| S9 | Costs on one machine: one MC likelihood evaluation; exact MCMC with MC pricing inside to a fixed CPU budget, extrapolated to 400 effective draws; network; audit; neural pricer + MH; multi-start | `svj_costs.py` pattern | `costs.json` |

### 6.5 Build order

Each stage is a presentable lesson on its own.

- **Stage A0 (today).** Run notebook 10's pipeline unchanged on the HKDE (θ, λ) task: the
  end-to-end template for proposal → exact reweighting → target prices.
- **Stage A, Heston.** ext 1 and ext 3; the Heston task JSON; `calibration/inference.py` and
  `correction.py` generalized; S3 and S7 on Heston. The lesson's whole arc on a model with an
  exact check.
- **Stage B, rough Bergomi on the grid.** ext 2; the rough task JSON; S1, S2, S4, S5, S6, S7, S9.
  The headline.
- **Stage C, the real surface.** N1, N5, S8 feeding every section.
- **Stage D, varying quote sets.** The set-based summary network; the real chain used as quoted.

## 7. The C2 table: alternatives, and the study that faces each

| Alternative | Gives | Lacks | Faced by |
|---|---|---|---|
| Point calibration with multi-start (desk practice) | The optima | No weights, no range; finds modes, not the ridge | S6 |
| Regularized calibration (Tikhonov; Cont & Tankov) | One stable answer | The uncertainty hides in the penalty | S6 |
| Deep calibration (Horvath, Muguruza & Tomas 2021; Bayer et al. 2019): neural pricer + least squares | A point, fast | No range | S4, S6 |
| Neural pricer + MCMC | A posterior; any noise model; any quote set | A sampler per surface; it is simulation-based inference too | S4: the two posteriors agree, and the lesson says which to use when |
| Exact MCMC with the MC pricer inside | The exact posterior | Hours per surface | S9 measures it |
| Model-free bounds (martingale optimal transport) | Bounds across all models | Too wide to act on | Stated; the posterior says how much of the bound one model occupies |
| Across-model ranges (Cont 2006; Schoutens, Simons & Tistaert 2004) | Model risk | Not within reach of one model's posterior | S7's Heston-against-rough comparison; slot 6 |
| "The posterior width is your noise model" | — | — | Exercise 1 shows it; the lesson states the within-model range as a lower bound |

## 8. Claims (seed of `CLAIMS.md`)

Each claim is on screen only when its study has written the number it quotes.

| Claim | Study |
|---|---|
| The SPY ATM skew follows a power law in T with exponent H − ½ ≈ −0.4 | S8 |
| The surface identifies ρη and H; η alone has a posterior ridge truncated at ρ = −1 | S2, S6 |
| The variance swap's posterior range is zero; the volatility swap's and the forward-start's are not; the reserve for the forward-start is X% of its price | S7 |
| Multi-start least squares lands on N distinct modes along the ridge; the regularized fit picks one by its penalty | S6 |
| The amortized posterior agrees with the exact Heston posterior to within 0.1 posterior sd | S3 |
| The amortized posterior and MCMC on the neural pricer agree within 0.2 posterior sd on 50 surfaces | S4 |
| SBC passes on 2,000 held-out surfaces; the audit flags Y% of validation surfaces and passes the real one | S2, S5 |
| One exact MC-likelihood evaluation costs A CPU-s; the exact posterior extrapolates to B CPU-hours per surface; the network answers in C s; the audit in D s | S9 |
| Heston and rough Bergomi calibrated to the same smile price the forward-start Z% apart | S7 |

## 9. Slot 4: what changes

Slot 4 (#37) stays the BayesFlow lesson on GBM from returns. Edits:

- Merge current `main`.
- Say plainly that at 1,000 days the plug-in forecast equals the predictive one and Bayes adds
  nothing: the slot checks the workflow.
- Show the network's 90% interval for σ next to MLE ± 1.645 se.
- Part B: the network reads log squares, which carry the volatility; report its R² and its error
  share as a range over three seeds.
- Replace the closing table: slot 5 changes the data (a smile, not returns), the measure (Q), the
  unknowns (H, η, ρ), what the network sees (a surface), the check (Heston exact, then the audit)
  and the quantity (a price range and a reserve). Remove the SVJ cost line.
- The VOO ending stays: volatility moves. Add one sentence: the smile is the other rejection of
  this model, and it is slot 5's data.
- Freeze **Fri 30 Oct**.

## 10. The intro

Francesco presents. Decision D1 by **Fri 23 Oct**, after a 5-minute dry talk of each option.

| | Option | Bayes in one example | Fit with the route |
|---|---|---|---|
| E | **One quote, then the smile** (recommended) | One option quote with its bid-ask gives a posterior for σ under Black–Scholes; price a second strike: a range. Two strikes: no σ fits both. The smile is the data that rejects the model. | Direct: closed form, Q-measure, leads into slot 4's model and slot 5's data; built from `calibration/surface.py` and slot 4's `gbm_volatility.py` |
| D | Volatility spine with the beta vignette from #41 | The 50 highest of 500 betas are too high by 0.16; the hedge is $8m short | Strong example, P-measure, a detour from the smile |

Outline for E, about 18 minutes:

1. **The question** (2). A model, a market, and what the market does not tell you. The desk's
   forward-start option and the regulation's "range of plausible valuations".
2. **Bayes in one example** (6). One quote → posterior for σ → price range for another strike.
   Then two quotes: the smile. P against Q in one slide.
3. **When Bayes changes the decision** (4). GBM from 1,000 returns: it does not (slot 4). Heston
   from 63 quotes: the identified directions are narrow and the others are not; the product
   decides whether it matters (slot 5).
4. **When simulation is needed** (4). The 2 × 2 map: likelihood tractable or not × one posterior
   or thousands. Black–Scholes and Heston against rough Bergomi and path-dependent volatility.
5. **The route** (2). Slot 4 under P, slot 5 under Q.

Build Mon 26–Thu 29 Oct; freeze **Wed 4 Nov**. The full beta notebook (#41) goes to `extras/`.

## 11. Decisions

| # | Decision | Recommendation | By |
|---|---|---|---|
| D1 | Intro option | E | Fri 23 Oct |
| D2 | Slot durations, presenters (slot 4), owners of slots 2, 3 and 6, from the acceptance mail | — | Fri 23 Oct |
| D3 | Record the route on #26 and point the `main` README at `tutorial_final/` | Yes | Fri 23 Oct |
| D4 | Freeze the two task JSONs and the protocol (§5) | As written, after one review | Fri 23 Oct |
| D5 | Headline product | ATM forward-start call; variance swap as control; volatility swap second | Fri 23 Oct |
| D6 | Real surface: SPY through the Alpha Vantage connector; derived implied vols committed, raw quotes not | Yes; provenance in `data/SOURCES.md` | Fri 23 Oct |
| D7 | SVJ material: optional companion in `extras/`; #38 closes when harvested there | Yes | Fri 30 Oct |
| D8 | LinkedIn: P5's question becomes "ten calibrations fit the smile; which price do you mark?"; P6 becomes Heston against rough on the same smile; P0 and P5 drop the particle-MCMC cost line | Yes | before P0 |
| D9 | Laptop support: add win-64 to the lock or offer a hosted notebook | Try win-64 first | Fri 30 Oct |
| D10 | Presentation format: marimo app view, Studio, or slides with the notebook beside | Notebooks for 4 and 5 | Fri 30 Oct |

## 12. `tutorial_final/`: layout and the bar for "final"

```
tutorial_final/
  README.md        participant guide (at freeze)
  PLAN.md          this file; removed at release
  CLAIMS.md        every number on stage or posted: value, source, command, status
  MANIFEST.md      what was harvested from where, with sha256 of copied files
  notebooks/       slot1_intro.py, slot4_bayesflow_basics.py, slot5_calibration.py
  extras/          beta notebook (#41); SVJ lesson and pipeline notebook
  studies/         S1–S9
  results/         only the files the notebooks load
  checkpoints/     gbm/, heston/, rough_bergomi/, pricer/
  data/            spy-surface/ (derived), returns CSV, SOURCES.md
  tests/           each notebook runs headless within budget; CLAIMS.md matches results/
```

Rules: library code lives in `src/sbi4finance/` once, with tests (`src/sbi4finance/calibration/`
for the new modules); artifacts are copied, not moved, with their hashes in `MANIFEST.md`;
notebooks resolve paths from `tutorial_final/`.

A slot is final when: it runs headless from a clean `pixi install --locked`, offline, within
budget, on osx-arm64 and linux-64 (win-64 after D9); `marimo check --strict` and its tests pass;
every number on screen is live or loaded from `results/` and listed in `CLAIMS.md`; its rows in
§7 are closed or the claim is gone; core and optional sections are marked; a co-presenter has
reviewed it and the presenter has rehearsed it with a timer; its data's provenance and licence
are recorded.

One pull request per slot ("harvest: slot N"). After a freeze, fixes only.

## 13. Interfaces with slots 2, 3 and 6

One paragraph per owner by Fri 30 Oct: what the slot explains, its key figure, what it assumes.
Until then: slot 2 explains flows, flow matching and summary networks (the set-based summary over
quotes is its natural example of why summaries are learned); slot 3 explains simulation-based
calibration and coverage (slot 5's SBC on synthetic surfaces is its finance example); slot 6
takes priors and misspecification (Heston calibrated to a rough surface; the quote-noise model;
the prior on H; the crash-day seed from #42).

## 14. Timeline

| Week | Slot 5 and studies | Intro | Slot 4 | Folder and logistics | LinkedIn |
|---|---|---|---|---|---|
| Wed 21–Fri 23 Oct | Stage A0; ext 1–3; `calibration/inference.py`, `correction.py`; task JSONs and protocol (D4); S1 job map and first shards on Oscar; N5 fetch | Storyboards E and D; dry talks; D1 | Merge `main` | Folder skeleton; `CLAIMS.md` seeded from §8; D2, D3, D5, D6 | D8 edits; P0 |
| Mon 26–Fri 30 Oct | Stage A: S3, S7 on Heston. Stage B: S2 training and SBC; N3 and S4; S5; N4 and S6; N2 and S7; S9 | Build | Edits; **freeze Fri 30** | Harvest slot 4; D7, D9, D10; win-64 lock | P1 Tue 27, P2 Thu 29 |
| Mon 2–Fri 6 Nov | Stage C: N1, S8 through every section; S7 across-model; Stage D set-based network; notebook complete; **freeze Fri 6** | **Freeze Wed 4** | Harvested | Clean-machine install Thu 5; harvest intro and slot 5 | P3 Tue 3, P4 Thu 5 |
| Mon 9–Fri 13 Nov | Rehearse | Rehearse | Rehearse | Full timed run-through Mon 9; fixes only; release tag Wed 11; participant README | P5 Tue 10, P6 Thu 12, P7 Fri 13 |
| Sat 14–Sun 15 Nov | Tutorial | | | | |

Proposed owners, pending D2: Francesco, slot 5 and the intro; Alexander, ext 1–3, bank
generation, Oscar and the harvest process; Stefan, the set-based summary network and the
BayesFlow architecture; Jerry, slot 3's interface and the SBC study. Everyone reviews one slot
they do not present.

## 15. This week

1. Francesco: review §5 and freeze the two task JSONs and the protocol (D4); storyboard intro
   options E and D and give both as dry talks; decide D1.
2. Team: D2, D3, D5, D6, D8; owners for slots 2, 3 and 6.
3. Stage A0: run notebook 10's pipeline as the template. Then ext 1–3, `calibration/inference.py`
   and `correction.py`, the Heston task, and S3 on synthetic Heston surfaces.
4. S1: write the job map and start the first rough shards on Oscar; measure the per-surface cost
   first and size the array from it.
5. N5: fetch one SPY chain through the connector and build the first artifact.
6. Create the folder skeleton, `CLAIMS.md` and `MANIFEST.md`.
7. Rewrite P0 and P5 in #40 to the calibration question before P0 goes out.

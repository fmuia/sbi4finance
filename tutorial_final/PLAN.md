# Plan: `tutorial_final/` for ICAIF '26

Revision 2, 21 October 2026. Owner: Francesco Muia. Supersedes revision 1 (same day),
which built slot 5 on SVJ from returns.

`tutorial_final/` will hold what we present in Milan on 14–15 November: the notebooks, the
studies that produce their numbers, and the results they load. The rest of the repository stays
as the research archive. This file is the working plan; a participant README replaces it at
release.

## 1. The decision

**Slot 5 is Bayesian calibration of a simulation-only volatility model to the option smile,
carried into the price of a product the smile does not pin down.** Rough Bergomi calibrated to a
vanilla surface; the posterior propagated into a forward-start option, with the variance swap as
the control; the 90% quantile of the price range as the model-risk reserve. Heston, which has an
exact likelihood through its Fourier pricer, plays the role GBM plays in slot 4: the model on
which the network is checked against the exact answer.

The route is GBM (slot 4, under P, from returns) → the smile (slot 5, under Q, from quotes). The
tutorial's title, *Amortized Bayesian Calibration of Volatility Models*, now describes it.

The SVJ-from-returns lesson becomes an optional companion. Its audit and its validation design
are reused in slot 5 (§6).

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
  asks for model-risk assessment. Rough and path-dependent volatility are the research frontier
  this audience knows, and deep calibration is the method it knows for them.
- **C2, necessity.** Each claim is stated against the strongest alternative: multi-start least
  squares, regularized calibration, deep calibration (a neural pricer with least squares), a
  neural pricer with MCMC, model-free bounds, across-model ranges. §7 lists what each gives and
  lacks, and which study faces it.
- **C3, route.** GBM → the smile. SVJ, HKDE jumps, MMAR and the one-factor Bergomi model are
  archive or companions.

## 4. The slot-5 lesson

Working title: **Ten calibrations fit the smile. Which price do you mark?** About 40 minutes
(decision D2 fixes it). Francesco presents.

| Min | Section | On screen | Live or cached |
|---|---|---|---|
| 0–4 | The question. A desk calibrates to today's smile and marks a forward-start option. Ten calibrations fit equally well. Which price, and what reserve? | The SPY smile; the regulation's words | cached |
| 4–9 | GBM under Q. Black–Scholes gives one volatility for every strike; the market does not. P and Q in one slide: slot 4's posterior forecasts, this one prices. | Flat line against the SPY smile; the ATM-skew term structure and its power law | live, seconds |
| 9–16 | Heston: the smile explained. Fourier likelihood, exact posterior by MCMC, and the slot-4 workflow with a new simulator. The network against the exact posterior. | Posterior pairs, exact against network | MCMC cached, network live |
| 16–21 | Rough Bergomi: why rough (skew ∝ T^(H−1/2)), why Monte Carlo only, why no exact posterior. | Skew term structure under both models; one priced surface and its cost | cached |
| 21–26 | The amortized posterior for the SPY surface. The ρ–η ridge; H from the term structure. | Posterior pairs over H, η, ρ | live, seconds |
| 26–31 | Is it right? SBC on 1,000 synthetic surfaces; the audit of this surface against the Monte Carlo likelihood; agreement with MCMC on a neural pricer, the deep-calibration route. | SBC; audit efficiency; two posteriors overlaid | SBC cached, audit live |
| 31–35 | The classical route on the same surface: multi-start least squares and a regularized fit; what they return and what they miss. | The modes on the ridge | cached |
| 35–39 | From posterior to prices: variance swap (zero range), volatility swap, forward-start option; the 90% quantile against the point calibration: the reserve. | Three price distributions; the reserve | price map live |
| 39–40 | Across models: Heston and rough Bergomi on the same smile price the forward-start differently. Within-model is a lower bound. Hand-over to slot 6. Where it leads: path-dependent volatility, joint SPX–VIX. | Two price distributions | cached |

Exercises: change the quote-noise model and watch the reserve; drop the two shortest maturities
and watch H; calibrate Heston to the rough-Bergomi surface (misspecification, for slot 6).

## 5. Task specification v0

Starting values, to be frozen in `configs/calibration_protocol.json` on Fri 23 Oct. The repo's
rough pilot (`rough_option_panel_v1`: nine put prices, ρ fixed at −0.6, flat forward variance,
H ∈ [0.1, 0.3], η ∈ [0.4, 1.2]) is the base; everything below generalizes it.

| Item | v0 |
|---|---|
| Observation | Implied volatilities on a grid of 7 maturities × 9 strikes, in maturity-major order. Maturities 1w, 2w, 1m, 2m, 3m, 6m, 1y. Strikes at log-moneyness k = δ σ_ATM √T for δ ∈ {−2, −1.5, −1, −0.5, 0, 0.5, 1, 1.5, 2}, so every maturity is quoted in its liquid region. |
| Quote noise | Gaussian in implied vol, SD 0.3 vol points at δ = 0 rising linearly to 1.0 at \|δ\| = 2, times a per-surface scale drawn log-normally (median 1, log-sd 0.3) and given to the network as context. |
| Rough Bergomi unknowns | H ~ U(0.02, 0.35); η ~ log-normal(median 1.5, log-sd 0.45); ρ ~ U(−0.99, −0.2). |
| Forward-variance curve ξ₀ | Context, not inferred: piecewise-constant on the 7 maturities; for training, level σ₀ ~ log-normal(median 20%, log-sd 0.35) and a slope factor so that σ(1y)/σ(1w) ∈ [0.7, 1.6]. For the real surface, built from the chain by log-contract replication per maturity. |
| Heston (check model) | v₀, θ, κ, σ_v, ρ with priors in the units of slot 4's prior where they overlap; Fourier pricer from `hkde.py` with jump intensity zero. Heston's forward-variance curve is parametric, θ + (v₀ − θ)e^(−κt): the contrast with Bergomi is a teaching point. |
| Pricing numerics | Hybrid scheme (`rough_bergomi.py`), steps per maturity fixed at 256 regardless of T, 2^17 paths with antithetic variates and the Black–Scholes control variate; MC standard error below 0.1 quote-noise SD on every quote (the pilot's rule). |
| Products under the posterior | Variance swap 3m (control: fixed by ξ₀); volatility swap 3m; ATM forward-start call, strike set at 3m, expiry 6m; one vanilla outside the grid (2y, 90% put) as the extrapolation warm-up. Priced by MC per posterior draw with common random numbers; a learned price map (the `hkde_surrogate.py` design) validated against MC makes the live cell fast. |
| Reserve | 90% quantile of the posterior price distribution minus the point-calibration price, per product. |
| Summary network | Stage B: a small MLP/CNN over the 7 × 9 grid plus context. Stage D: `bf.networks.SetTransformer` over tokens (k/√T, T, iv, noise sd, mask) with random masking in training, so a real chain with missing strikes is handled as trained. |
| Inference network | `bf.networks.FlowMatching`, slot 4's class; three unknowns. |
| Bank | 100,000 training surfaces, 5,000 validation, 5,000 test, generated on Oscar with the existing shard/assemble scripts; 2,000 further surfaces reserved for SBC. |

## 6. What exists and what is built

**Reused from the archive.** Rough Bergomi simulator and FFT hybrid driver; option-bank schema,
shard generation and assembler; Oscar array scripts; Fourier pricer for Heston (via `hkde.py`);
importance correction with an exact likelihood (`hkde_correction.py`); fixed-contract learned
price map (`hkde_surrogate.py`); the SVJ audit (`svj/audit.py`: Student-t proposal, reweighting,
efficiency flag); shared inference and training workflows; the 9 October HKDE figure
(`docs/design/images/01-hkde-option-prices.png`), which previews the lesson's result on a
Heston-with-jumps surface.

**Library work** (`src/sbi4finance/calibration/`, with tests):

| Module | Content |
|---|---|
| `surface.py` | Grid definition, implied-vol inversion, noise model, ξ₀ from a chain by log-contract replication, ATM-skew term structure |
| `heston_q.py` | Heston Q task: prior, Fourier likelihood (JAX), NUTS/adaptive-MH sampler, simulator for BayesFlow |
| `rough_q.py` | Rough Bergomi Q task: prior, ξ₀ context, MC surface pricer with CRN, simulator for BayesFlow, bank adapter |
| `products.py` | Variance swap, volatility swap, forward-start call, out-of-grid vanilla; MC pricers with CRN; learned price map |
| `classical.py` | Multi-start least squares (L-BFGS on the MC pricer with CRN, and on the neural pricer), Jacobian standard errors, Tikhonov-regularized fit |
| `pricer_net.py` | Neural pricer (parameters + ξ₀ → 63 implied vols), the deep-calibration route; MCMC on it |
| `audit.py` | Per-surface importance reweighting of the network's draws with the MC likelihood; efficiency and PSIS k̂ |

**Studies** (`tutorial_final/studies/`, each writes to `results/` and is a `pixi` task):

| ID | Study | Output | Feeds |
|---|---|---|---|
| S1 | Generate the rough-Bergomi bank (Oscar) | `results/calibration/bank-manifest.json` | S2, S4, S5 |
| S2 | Train the amortized posterior; SBC, recovery, coverage, contraction on held-out surfaces | `checkpoints/rough_bergomi/`, `results/calibration/validation/` | §4 min 26–31 |
| S3 | Heston check: exact MCMC against the network on 50 synthetic surfaces and the real one | `results/calibration/heston-check/` | §4 min 9–16 |
| S4 | Neural pricer; MCMC on it against the amortized posterior on 50 surfaces and the real one | `results/calibration/pricer-route/` | C2 row "deep calibration" |
| S5 | Audit: efficiency and k̂ on 500 validation surfaces; flag rate; the real surface | `results/calibration/audit/` | §4 min 26–31 |
| S6 | Classical route: multi-start LS and regularized fit on 200 synthetic surfaces and the real one; coverage of Jacobian intervals; where the modes sit on the ridge | `results/calibration/classical/` | §4 min 31–35 |
| S7 | Products under the posterior: price distributions, reserves, price-map validation against MC; Heston against rough Bergomi on the same surface | `results/calibration/products/` | §4 min 35–40 |
| S8 | Real surface: SPY end-of-day chain for one date through the Alpha Vantage connector (`HISTORICAL_OPTIONS`); cleaning rules; implied vols, ξ₀, skew term structure; derived data only in the repository, provenance recorded | `data/spy-surface/`, `data/SOURCES.md` | §4 throughout |
| S9 | Costs on one machine: one MC likelihood evaluation; the exact MCMC with MC pricing inside, run on one surface to a fixed CPU budget and extrapolated to 400 effective draws; network; audit; neural pricer + MCMC; multi-start LS | `results/calibration/costs.json` | §4 min 16–21, 26–31 |

**Build order.** Each stage is a presentable lesson on its own.

- **Stage A, Heston only.** S3 and S7 on Heston: exact posterior, network, forward-start range,
  reserve. The lesson's whole arc on a model with an exact check.
- **Stage B, rough Bergomi on the fixed grid.** S1, S2, S4, S5, S6, S7, S9. The headline.
- **Stage C, the real surface.** S8 feeding every section.
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
| E | **One quote, then the smile** (recommended) | One option quote with its bid-ask gives a posterior for σ under Black–Scholes; price a second strike: a range. Two strikes: no σ fits both. The smile is the data that rejects the model. | Direct: closed form, Q-measure, leads into slot 4's model and slot 5's data |
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

Build Mon 26–Thu 29 Oct from `gbm_volatility.py` and `calibration/surface.py`; freeze
**Wed 4 Nov**. The full beta notebook (#41) goes to `extras/`.

## 11. Decisions

| # | Decision | Recommendation | By |
|---|---|---|---|
| D1 | Intro option | E | Fri 23 Oct |
| D2 | Slot durations, presenters (slot 4), owners of slots 2, 3 and 6, from the acceptance mail | — | Fri 23 Oct |
| D3 | Record the route on #26 and point the `main` README at `tutorial_final/` | Yes | Fri 23 Oct |
| D4 | Freeze the task protocol v0 (§5) | As written, after one review | Fri 23 Oct |
| D5 | Headline product | ATM forward-start call; variance swap as control; volatility swap second | Fri 23 Oct |
| D6 | Real surface: SPY through the Alpha Vantage connector; derived implied vols committed, raw quotes not | Yes; record provenance in `data/SOURCES.md` | Fri 23 Oct |
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

Rules: library code lives in `src/sbi4finance/` once, with tests; artifacts are copied, not
moved, with their hashes in `MANIFEST.md`; notebooks resolve paths from `tutorial_final/`.

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
| Wed 21–Fri 23 Oct | D4 protocol freeze; S1 bank generation starts on Oscar; `heston_q.py` and the exact sampler (Stage A); S8 fetch | Storyboards E and D; dry talks; D1 | Merge `main` | Folder skeleton; `CLAIMS.md` seeded from §8; D2, D3, D5, D6 | D8 edits; P0 |
| Mon 26–Fri 30 Oct | S3 Heston check; S2 training and SBC; S4 pricer net; S5 audit; S6 classical; S7 products on Heston and rough; S9 timings (Stage B) | Build | Edits; **freeze Fri 30** | Harvest slot 4; D7, D9, D10; win-64 lock | P1 Tue 27, P2 Thu 29 |
| Mon 2–Fri 6 Nov | S8 real-surface results through every section (Stage C); S7 across-model; set-based network (Stage D); notebook complete; **freeze Fri 6** | **Freeze Wed 4** | Harvested | Clean-machine install Thu 5; harvest intro and slot 5 | P3 Tue 3, P4 Thu 5 |
| Mon 9–Fri 13 Nov | Rehearse | Rehearse | Rehearse | Full timed run-through Mon 9; fixes only; release tag Wed 11; participant README | P5 Tue 10, P6 Thu 12, P7 Fri 13 |
| Sat 14–Sun 15 Nov | Tutorial | | | | |

Proposed owners, pending D2: Francesco, slot 5 and the intro; Alexander, bank generation,
Oscar and the harvest process; Stefan, the set-based summary network and the BayesFlow
architecture; Jerry, slot 3's interface and the SBC study. Everyone reviews one slot they do not
present.

## 15. This week

1. Francesco: review §5 and freeze the protocol (D4); storyboard intro options E and D and give
   both as dry talks; decide D1.
2. Team: D2, D3, D5, D6, D8; owners for slots 2, 3 and 6.
3. Start Stage A: `heston_q.py`, the exact sampler, the network on Heston, the forward-start
   range. Start S1 on Oscar. Fetch the SPY chain (S8).
4. Create the folder skeleton, `CLAIMS.md` and `MANIFEST.md`.
5. Rewrite P0 and P5 in #40 to the calibration question before P0 goes out.

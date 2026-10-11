# Inference scenarios and their data contracts

Status: proposed task-to-data specification, 11 October 2026. Read this before choosing a market-data provider.

This reconciles Francesco's HKDE-to-SVJ discussion with [PR #3](https://github.com/fmuia/sbi4finance/pull/3), inspected at `720eb6b`. That discussion contains two applications: Bayesian calibration to option quotes, followed by a proposal to infer stochastic volatility and jumps from historical returns. PR #3 deliberately supports both and leaves the first teaching application open. The [option loading plan](data-loading-plan.md) and [provider comparison](data-source-landscape.md) cover the options branch; they do not settle the curriculum or supply a returns-data specification.

The model name alone does not determine the required data. Specify **model + probability measure + observed channels + inference target + downstream use**. The same stochastic-volatility family can support distinct experiments, with different parameters, training banks and checkpoints.

## Scenario map

| Task | One observed input | Inferred or learned object | Downstream use | Real data required |
|---|---|---|---|---|
| R: Return dynamics and state inference | An ordered return window, with times and any supported masks | Joint posterior over physical-measure parameters and current continuation state | Current volatility estimate; future returns, volatility and losses; VaR/ES | Underlying price/return history; subsequent outcomes only for forecast evaluation |
| O: Option calibration | A cross-section of option quotes across strikes and expiries at one valuation time | Posterior over risk-neutral parameters and any explicitly inferred curve/state inputs | Fitted smiles; same-time held-out option values; uncertainty in another derivative's model price | Contract panel with premiums or documented IVs and consistent pricing inputs |
| P: Supervised pricing surrogate | Parameters/state, pricing inputs and a target contract | A function approximating the reference pricing expectation | Fast valuation of posterior draws from O | None for training; observations used by O remain necessary when conditioning on a real market |
| D: Direct price-posterior inference | The observed panel from O, plus target-contract context | Conditional distribution of target model prices | Price uncertainty without explicitly returning parameter draws | Same panel as O; observed target prices only for empirical validation |

R corresponds to PR #3's M3, O to M4 and the Bergomi options branches M6–M7, and P/D to M5. M5 proposes the parameter-posterior-plus-surrogate route first; D is an optional experiment. Reference pricing of posterior draws is the baseline before either learned-pricing extension. These are proposed capabilities, not a claim that the new loaders or models exist.

## R: Returns, hidden volatility, and future losses

The question is: **given this historical return sequence, which dynamics and current hidden state are plausible, and what could happen next?** Let `y = r[1:T]` be observed returns, `theta_P` physical-measure parameters and `z_T` the continuation state. Infer

```text
p(theta_P, z_T | r[1:T], observation_times, known_context).
```

For a Heston model with independent price jumps, the continuous-time volatility state is current variance. A numerical scheme may need additional retained state for exact continuation. Finite-factor Bergomi needs its factors; rough Bergomi generally needs a conditional memory representation. A network that returns only parameters needs an additional state-inference procedure before conditional forecasting.

This is the later SVJ direction in Francesco's chat. It is distinct from the HKDE paper's risk-neutral option-calibration problem. Exact SVJ equations, initialization, return conventions, targets and checkpoint compatibility must come from his contributed study bundle before claiming to reproduce it. SVCJ, with variance jumps, is a further model choice rather than a synonym for every SVJ experiment.

### Observed format

Preserve a source table of timestamped prices and adjustment/corporate-action information. Prepare an ordered return table such as:

```text
series_id, interval_start, interval_end, dt_years, return_value, observed
```

The manifest declares `return_kind` (log or simple), decimal versus percent units, currency, session/calendar, time-zone and elapsed-time convention, adjustment policy, and whether the modeled process represents price returns or total returns. A log return is `log(P_end / P_start)`; a simple return is `P_end / P_start - 1`. Do not interchange them or silently reuse MMAR's preprocessing for a different model.

As an illustration, 253 eligible price observations yield 252 consecutive returns. This is one inference example, not 252 independent examples. Window length is a checkpoint contract, not an acquisition default. A regular trading-day model must explain how weekends/holidays are treated; an irregular-time model must receive supported time increments. Missing records are not zero returns. Corporate actions or futures roll discontinuities must not silently become model-inferred jumps. Log returns require strictly positive levels; an unsuitable instrument or series needs a different model or transformation.

A simple returns-only experiment does not need option chains, bid/ask quotes, rates/forwards for pricing, vendor implied volatilities, or observed jump labels. Daily closes can suffice. High/low/volume channels are optional observations only if their joint simulation and measurement rules are specified. Per-window standardization that discards the return scale also discards information about volatility unless that scale is retained as context.

### Synthetic training versus real use

For each synthetic example, draw parameters and the declared initial state, simulate **one stochastic history**, then sample/aggregate it as the real observations are measured. Store the resulting returns as conditioning and the true parameters plus necessary terminal state as targets. Hidden paths and jump realizations can support diagnostics, but cannot be given to a returns-only encoder. Do not add a separate quote-noise law; any return measurement noise must be explicitly justified in this task.

For real use, supply the same observed channels and context, with **no true parameter, jump or variance labels**. A measured volatility proxy does not become the latent variance truth. The simulator provides labels for training and recovery tests; the market series provides an unlabelled conditioning example.

Continue each matched posterior parameter/state draw under the physical measure to obtain future returns and losses. Do not redraw a stationary starting variance or independently pair parameter and state marginals. For each simulated future, record the declared horizon and loss definition. Current volatility inference is a filtering question; future volatility inference is a forecast. Reconstructing past states with later observations is smoothing and must not be presented as an available-at-the-time forecast.

### Real-world illustration versus backtest

One archived historical window suffices to show the posterior and a conditional forecast. A subsequent block supplies realized outcomes for an illustrative forecast check. A claim about calibrated VaR/ES or predictive performance requires many chronological forecast origins and outcomes, a fixed horizon and loss definition, no look-ahead in preprocessing/model selection, and treatment of dependence from overlapping windows/horizons. A single realized loss is not a true VaR label.

Predictive VaR/ES are computed from the pooled posterior-predictive loss law. A posterior interval for conditional VaR/ES requires calculating the conditional risk measure for each parameter/state draw. Averaging draw-specific VaRs is generally not the predictive VaR. Likewise, evaluating future volatility requires a declared realized-volatility proxy and its limitations; latent instantaneous variance is not directly observed in ordinary price data. See [PR #3's forecasting definitions](https://github.com/fmuia/sbi4finance/blob/720eb6baa19b19671183759738efb01674e773cf/docs/simulator-plan/notes/03-inference-and-pricing.md#6-forecasting-losses-var-and-expected-shortfall).

## O: Calibrating to an option surface

The question is: **which risk-neutral dynamics explain this set of currently quoted derivative prices?** A chain is a collection of contracts; its cross-section across expiries forms a surface. Neither expiry nor strike is a sequence of future realized observations.

Let `c_j` identify a contract and its pricing inputs, and let `f_Q(theta_Q; c_j)` be its model price or IV. With an explicitly chosen observation law, an illustrative calibration is

```text
y_j = f_Q(theta_Q; c_j) + epsilon_j
p(theta_Q | y, contracts, market_context) proportional to likelihood * prior.
```

Point calibration chooses a parameter vector explaining the quotes; Bayesian calibration keeps a distribution of explanations. Neural posterior training happens offline over many synthetic examples. Feeding one real panel to the trained network is an application of that inference procedure, not supervised training on known market parameters. HKDE's PROJ map is a numerical forward pricer, not the quote likelihood by itself. The paper concerns this option-calibration and downstream-pricing task. [HKDE paper](https://arxiv.org/html/2502.13824v1).

### Observed format

Use a long table with one row per contract at one valuation time:

```text
panel_id, valuation_time, contract_id, strike, expiry, option_type,
exercise_style, settlement_convention, bid, ask, quote_time,
forward, discount_factor
```

Preserve currency, multiplier, root, curve provenance and quality flags as detailed in the [option schema](data-loading-plan.md#shared-panel-schema). A processed-IV representation instead contains clearly labelled decimal IV bid/ask or mid, strikes or documented moneyness, year fractions, forwards and the inversion conventions/inputs. Missing premium or quote-time fields remain null. IV is a transformation of option prices under a convention, not an observation of instantaneous variance. Gaussian price noise and Gaussian IV noise are different observation models.

One real snapshot with several expiries and strikes can illustrate this task. A long history of the underlying alone cannot replace it. Multiple daily surfaces are needed for empirical recalibration-stability studies, or for any expressly modeled time evolution of future quotes. Re-fitting separate days does not by itself train a model that forecasts tomorrow's option chain.

### Synthetic training versus real use

Draw `theta_Q`, select the contract design and market context, calculate the corresponding pricing expectations, then sample the declared quote-error model. The **entire noisy panel is one conditioning example** and the generating parameter vector is its target. For HKDE vanillas, PROJ supplies prices without path Monte Carlo. For rough Bergomi, many paths typically contribute to each surface. Those internal paths are not independent observed surfaces or extra posterior-training labels. Numerical pricing error must be controlled separately from market observation noise.

HKDE and rough Bergomi can use the same market-panel format if both support its contracts. They require different synthetic generators and model-specific targets/checkpoints. In flat rough Bergomi the variance curve is constrained to `xi0(t)=v0`; in a more general version its coefficients must be inferred or its curve supplied as fixed context. The asset forward `F(0,T)` and forward variance `xi0(t)` are different quantities. Changing model flexibility does not automatically change the raw market-file format. [Rough-volatility calibration](https://arxiv.org/abs/1908.08806).

The synthetic and real panel must match in representation, contract coordinates, noise/context assumptions and supported masks. For a fixed-grid network, even a real surface with the same number of quotes can be incompatible if the expiries or moneyness differ. Freeze the selected design before training or explicitly train on varying coordinates and context. A bid/ask spread requires a declared statistical interpretation; it is not automatically a noise standard deviation.

### Downstream pricing and what data it requires

From each posterior draw, compute the value of a specified target contract at the **same valuation time**. A European holdout needs its strike, expiry and payoff. An American put also needs its exercise and dividend/carry assumptions; a barrier needs level, direction, monitoring and rebate rules. Already-running path-dependent contracts also require their accrued state, such as a running average or whether a barrier has been hit. A controlled tutorial can start with new contracts.

Observed target quotes are **not necessary** to demonstrate uncertainty in a model price. They are necessary to claim empirical agreement with that instrument's market price, and should be excluded from the conditioning panel for a held-out check. Synthetic ground-truth prices and an independent converged pricer instead validate implementation under the model. Feeding a contract's quote into calibration and plotting its fitted price is an in-sample check.

A distribution of prices across posterior parameters is uncertainty about conditional pricing expectations. It is not the distribution of one realized payoff, nor a forecast of how tomorrow's traded quotes will move. A pricing model under Q also does not directly supply physical-measure portfolio-loss probabilities. Returns-based risk and option calibration cannot be joined by merely changing stock drift; an explicit P/Q risk-premium model is required. PR #3 defers that joint problem.

## P and D: Two different meanings of learned pricing

PR #3 proposes three pricing routes: posterior draws with a reference pricer; those same draws with a supervised surrogate; and an optional direct price-posterior estimator. Only the latter two add learned pricing models.

| Route | Training input | Training target | What one model query returns |
|---|---|---|---|
| P: Pricing regression | `theta_Q`, any state, market inputs, target contract | Accurate reference price, with solver diagnostics stored separately | Approximate price at the supplied parameters; no parameter posterior |
| D: Direct price posterior | Noisy observed option panel, its context, target-contract specification | Reference target price(s) from the same generating parameter draw | Distribution of target model prices conditional on the observations |

P can be trained entirely from numerical model evaluations; historical market-price labels are unnecessary. D also uses simulator-generated labels: draw parameters, generate an observed panel, and compute the target expectation under those same parameters. A single random payoff is not that reference price. D's scalar marginal posteriors do not establish joint price dependence; portfolio or surface outputs need a coherent joint construction and validation. See [PR #3's three workflows](https://github.com/fmuia/sbi4finance/blob/720eb6baa19b19671183759738efb01674e773cf/docs/simulator-plan/notes/03-inference-and-pricing.md#5-three-connected-pricing-workflows).

For a real example, D consumes the same kind of market panel as O. P is evaluated on posterior draws inferred from that panel. Neither needs a time series of observed exotic prices merely to work. Actual target quotes are an additional evaluation dataset, not a replacement for synthetic training truth.

## Objects, arrays and leakage boundaries

CSV versus Parquet is a storage choice; **sequence versus panel versus pricing-label pair** is the scientific distinction. Keep small real observations as readable tables plus JSON manifests and larger generated banks as PR #3's NumPy shards plus manifests. Let `B` be the number of independent examples, `L` the history length and `M` the observed contract count:

| Bank | Conditioning arrays | Target arrays | Example/split unit |
|---|---|---|---|
| Returns R | Returns `[B,L,1]`, times/context, supported mask | Parameters `[B,d_P]` and named terminal-state arrays | Independent base simulated history; group related windows |
| Option inference O | Prices or IVs `[B,M,1]`, contract coordinates, curves/context, supported mask | Parameters `[B,d_Q]` and explicitly inferred curve/state components | Base parameter–surface draw, before repeated noise augmentation |
| Pricing surrogate P | Parameters/state plus contract and market context | Reference prices | Group related contracts at each base parameter/context draw |
| Direct price inference D | O's conditioning plus target contracts | Reference target price vector `[B,m_target]` | Same base draw as its observed surface |

These are conceptual shapes, not a replacement API for PR #3. Fixed coordinates can live once in the task manifest. Distinct real histories remain separate examples; related windows or assets may still be dependent. Adding time or asset axes changes the modeling problem. Diagnostic fields derived from true parameters, hidden volatility, jumps or future outcomes do not become encoder inputs. Posterior target labels are absent from real inference records.

Training-bank size, days per window, quotes per surface and number of real snapshots are four distinct counts. One million synthetic examples does not imply acquiring one million market surfaces. Conversely, adding more strikes to a single observed surface does not produce more independently labelled training examples.

## Data work to prioritize

1. **Define two task contracts first:** R for chronological return windows and O for dated option panels. Agree the measure, observation channels/units, initialization, targets, fixed/inferred inputs and downstream question. Record these in M0 of PR #3 before fixing provider APIs.
2. **Prepare a small local return fixture and a separate option fixture.** Reuse an existing return cache only after checking its conventions against the chosen task. The options fixture can initially be synthetic; the processed research surface remains an illustrative import with documented missing metadata.
3. **Keep the options acquisition choices conditional on O.** The Cboe/ThetaData/Alpaca comparison solves option-quote acquisition. R's daily-return example can use a vetted historical price export, potentially via yfinance or a stock-data API, without an options subscription. This note does not establish a particular returns source's quality or usage rights.
4. **Specify evaluation separately:** future realized returns for R; same-time excluded quotes for O/D; converged model prices for P. Volatility-proxy observations and joint P/Q datasets are additional experiments, not mandatory fields to invent in either basic loader.
5. **Generate separate compatible synthetic banks.** Keep observation fields the same as those available in the corresponding real example, retain synthetic truth only as target/diagnostic information, and split before related augmentations. Use the manifest and checkpoint contracts of PR #3 rather than a second bank architecture.

This is a scope correction to PR #4: its options work remains useful, while a returns-first tutorial would prioritize the R fixture and forecasting evaluation. No curriculum pivot, model implementation or provider purchase is implied by this document.

# Option data loading plan

Status: proposed implementation plan. Source review: 11 October 2026.

Build the tutorial around local, versioned option panels and our own synthetic training banks. Use the same European market panel for HKDE and rough Bergomi, with separate model-specific training banks. Participants should not need a brokerage account, data subscription, Java service, API key, or network connection to execute the required exercises.

This plan covers acquisition, normalization, validation and loading. It complements the [simulator and inference plan in PR 3](https://github.com/fmuia/sbi4finance/pull/3); it does not implement the proposed interfaces or change the existing MMAR returns notebook. The [source comparison](data-source-landscape.md) records the provider evidence and tradeoffs.

## Source decisions

| Need | First choice | Reason and fallback |
|---|---|---|
| Required training and inference exercises | Our own generated parameter–surface pairs | Known truth, explicit observation law, controlled grid, independently generated validation cases |
| Initial real European illustration | Pinned Gatheral SPX processed-IV file | An accessible research example; retain its limitations and resolve redistribution before bundling |
| One controlled historical raw-quote surface | Cboe DataShop 15:45 EOD product | Small date/symbol order and defined observation time; get an actual quote and usage terms before selecting it |
| Repeated historical surfaces through an API | Evaluate ThetaData at-time quotes first | Direct match to a historical chain at a specified minute; local terminal and actual SPX/SPXW entitlements must pass a sample check |
| Optional capture of a current surface | Alpaca via `alpaca-py`, explicitly using OPRA | Official Python client and documented index-option coverage; current capture is distinct from historical reconstruction |
| Low-friction Yahoo exploration | `yfinance`, outside the required path | Useful exploratory wrapper; no established SPX download or historical as-of chain in this project |

Alpaca is a credible optional acquisition route for both index and stock/ETF options. Its free indicative quotes are modified data, however, and its historical bars/trades do not establish historical bid/ask availability. Do not choose it merely because the product is described as historical options data. [Feed definitions](https://docs.alpaca.markets/us/docs/historical-option-data), [current option client](https://github.com/alpacahq/alpaca-py/blob/master/alpaca/data/historical/option.py).

Implement the common file loader first. Choose at most one remote provider adapter after a bounded acquisition pilot demonstrates that it supplies the missing fields. A one-off authorized export is sufficient for the tutorial; every provider in the comparison does not need an implementation.

## The statistical object being loaded

A panel is a collection of contracts on one underlying at a stated observation time. Let `x` contain their strikes, expiries, exercise/settlement conventions, and fixed pricing inputs. Let `y` contain observed premiums or IVs. Calibration compares `y` with the model's jointly predicted surface `f(theta; x)`.

With a declared Gaussian teaching observation model,

```text
y | theta, x ~ Normal(f(theta; x), Sigma)
p(theta | y, x) ∝ p(y | theta, x) p(theta)
```

The parameter–surface relationship and observation law define the inference task. Training a posterior estimator is separate from fitting one observed panel. Real observations have no known parameter labels; labels come from synthetic generation.

| Model | Parameters and required context |
|---|---|
| HKDE | Five Heston variance parameters and four price-jump parameters; fixed asset forwards and discount factors |
| Flat rough Bergomi | `(v0, eta, rho, H)`, explicitly imposing `xi0(t) = v0` |
| General rough Bergomi | `(eta, rho, H)` plus an initial forward-variance curve that is either fixed context or parameterized and inferred |

The asset forward `F(0,T)` is different from the forward variance `xi0(t) = E_Q[V_t]`. Neither a fixed variance curve nor its uncertainty should enter implicitly through preprocessing. It is not simply squared ATM IV. The posterior is conditional on any estimated inputs treated as fixed. [Calibration formulation](https://arxiv.org/html/1908.08806), [HKDE model](https://arxiv.org/html/2502.13824v1).

Declare whether errors are modeled in premiums or IVs. Bid–ask half-spreads are not automatically standard deviations; the mapping to `Sigma` is an explicit assumption. Numerical pricing error, observation noise and model discrepancy remain separate. Published calibration RMSE and vendor Greeks are not substitutes for an observation model.

## Initial fixtures and acceptance targets

1. Prepare two or three synthetic HKDE demonstration panels with known truth and separately generated validation cases. Add a rough-Bergomi panel when its pricer is validated. Downstream payoff specifications need no observed exotic prices unless empirical validation is claimed.
2. Import the [15 February 2023 SPX research file](https://github.com/jgatheral/QuadraticRoughHeston/blob/76dd1f9122c13333032bd3b8cc1e4d3dcc182009/spx_implied_vol_20230215.csv) as **processed IV**, retaining supplied forwards and year fractions. `Bid`/`Ask` mean annualized decimal IVs. Do not reinterpret `CallMid` as a verified discounted market premium. Metadata gaps must remain explicit; see the [source audit](data-source-landscape.md#research-files-and-synthetic-resources).
3. Target one usable European surface, approximately 6–10 expiries and 10–20 selected strikes per expiry, including ATM and both wings. Start around two weeks to one or two years where liquidity permits. This is a teaching design, not a guarantee of parameter identification. Avoid requiring 0DTE or a long historical panel.
4. Freeze actual contract coordinates before training a fixed-grid estimator. Prefer consistent PM-settled SPXW contracts for a new raw snapshot; do not guess settlement for the processed research file. State any unresolved convention as an illustrative limitation or exclude the affected rows from a stronger empirical claim.
5. Remove genuinely held-out contracts from the conditioning design used in training. For irregular/missing quotes, either train with coordinates/masks or fail clearly. Interpolation creates neither new independent observations nor automatic compatibility with an existing checkpoint.

Success means a reproducible panel, an observed-versus-fitted smile comparison, residuals by maturity/strike, numerical accuracy checks, and uncertainty in parameters and held-out prices. Good price fit alone is not parameter identification, and an option fit concerns risk-neutral dynamics rather than a physical returns forecast.

## Shared panel schema

Use a long-form contract table plus a versioned JSON manifest. Begin with CSV for small transparent fixtures; use Parquet only when volume justifies the dependency. Keep float64 precision in normalized market records and cast explicitly at the network boundary.

Support two representations: `raw_quote` and `processed_iv`. Missing raw fields stay null in processed-IV imports. Do not fabricate premium bid/ask, precise quote timestamps, or settlement metadata to satisfy one rigid table. Validation requirements depend on representation and the claim being made.

| Group | Fields and meaning |
|---|---|
| Identity | Stable panel/row ID, provider symbol, underlying, root, contract ID, currency, multiplier and deliverable when known |
| Contract | Strike, call/put where applicable, exercise style, expiration date, settlement instant/type, year fraction and day-count rule |
| Observation | Premium bid/ask and sizes **or** IV bid/ask; source IV/Greeks retained separately from recomputed values |
| Timing | Requested as-of instant, original quote timestamp when available, timestamp semantics, acquisition start/end, last-trade time in a separate field |
| Context | Discount factor and asset forward by expiry; source/method; spot/dividend inputs if the pricer needs them; fixed/inferred forward-variance treatment |
| Derived fields | Mid, spread, log-forward moneyness, computed IV with convention, quality flags and exclusion reason |
| Manifest | Schema version, source URL/revision or endpoint, request filters/feed, SDK version, retrieval time, raw hashes, normalization version, panel hash, permitted-use evidence |

Preserve original symbols while normalizing them: `^SPX`, `SPX`, and SPXW contract roots are provider-specific identifiers, not interchangeable string substitutions. Quote premium units and per-contract cash values also differ by multiplier. Infer known exercise/settlement rules only from an authoritative product mapping, with its provenance recorded. [Cboe contract specifications](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications).

## Acquisition and validation sequence

1. **Acquire outside notebooks.** Keep credentials in the process environment or existing secret store. Save raw responses privately, record all request parameters, follow pagination completely, and write a manifest only when the acquisition is complete. A partial/rate-limited response is a failed acquisition, not a smaller valid surface.
2. **Establish time semantics.** Distinguish a historical as-of query, a latest quote collected during an interval, and last-standing EOD records. For a historical reconstruction, select eligible quotes at or before the target time and apply a configured quote-age policy where original quote times exist. A sampled output timestamp or last-trade timestamp is not evidence of quote freshness. Preserve unknown age rather than treating it as zero.
3. **Normalize without changing evidence.** Retain missingness and raw values. Check duplicates, dimensions, finite values, positive strikes/maturities and ordered nonnegative premium quotes. A zero bid can be meaningful but need not enter the selected inference panel. Retain excluded rows and reasons.
4. **Resolve pricing inputs.** For European quotes, obtain discount factors and forwards with consistent timing. Matching calls and puts allow `C - P = D(F - K)` as a check or estimation method. Include dividend/carry assumptions; do not substitute a Treasury par yield directly for a zero-coupon curve. OPRA option coverage does not imply a licensed spot-index feed. A forward-normalized European fit may not need spot if the pricer supports it.
5. **Check pricing consistency.** Examine spreads, stale/low-vega contracts, payoff bounds, and strike convexity. Use carry-consistent coordinates for calendar checks; do not impose raw call-price monotonicity across different forwards blindly. Treat parity-related call/put pairs and interpolated observations with compatible statistical assumptions. Flag inconsistencies rather than silently repairing them.
6. **Freeze the chosen panel.** Save contract ordering, chosen representation, curves, quality policy and checksums. Produce smile plots and an exclusion summary offline. The loader rejects unexpected schema, units, grid, or checkpoint-context mismatches.

For Alpaca, set `OptionsFeed.OPRA` explicitly and reject an indicative feed for empirical calibration. Its chain is latest-only; `updated_since` is a freshness filter, not a historical as-of date. Record the capture interval and every returned quote timestamp; SDK pagination does not make the capture atomic. [Chain API](https://docs.alpaca.markets/us/reference/optionchain).

For ThetaData, use its supported at-time resolution and verified time zone; the requested minute and actual contributing quote age are different quantities. Its free EOD report is not a substitute for a 15:45 snapshot. Do not implement an automatic fallback that silently changes provider, feed or observation time after an entitlement error.

## Loading boundaries and dependencies

The proposed boundary is a small loader and normalization functions, not a general provider framework:

```text
src/sbi4finance/datasets/
  option_panels.py              load and validate local panels
  option_sources/               format-specific normalization when needed
scripts/
  prepare_option_panel.py       deterministic raw-to-panel preparation
  acquire_option_panel.py       optional provider acquisition
configs/data/                   filters, field mappings, time and curve conventions
data/options/<panel_id>/        only generated or authorized distributable fixtures
  contracts.csv
  manifest.json
```

Raw licensed files and API caches live in a configured private directory outside the public repository. Add appropriate ignore rules before introducing download scripts. Code, requests and metadata can be versioned without assuming the underlying market records are redistributable. Record permissions for classroom display, derived figures and downloadable data separately; public access or an open-source SDK does not confer all three.

Use the existing Python/NumPy/pandas stack for file normalization, declaring direct dependencies explicitly when code lands. `alpaca-py` or a Theta terminal client belongs in an optional acquisition environment. Student notebooks call a local `load_option_panel(...)`; they do not import provider SDKs or retrieve data on failure. No provider dependency or terminal installation is added by this plan.

Synthetic training banks remain distinct from market panels. Their manifests additionally identify model, parameter order/units, prior, observation law, numerical-generator version, seeds, grid, error estimates and split groups. Split base parameter–surface examples before noise augmentation. Posterior truth and diagnostic quantities must not leak into network conditioning. Reuse the simulator plan's bank and batching interface rather than creating another generator or shard format here.

## American extension

Begin with a synthetic European posterior propagated into an American put on the **same underlying**. An actual SPY chain is optional and belongs to a separately specified American calibration task. SPX index parameters cannot automatically be transferred to SPY ETF options. For real American fitting, require dividend assumptions, an American pricer, correct observation conventions and matching training data. Preserve vendor IVs as vendor outputs, not authoritative European IV inputs.

## Implementation milestones

| Milestone | Deliverable | Acceptance |
|---|---|---|
| D0 Define the data contract | Representations, units, timing semantics, curve treatment and selection policy | Walk a synthetic panel and processed SPX panel through the same statistical task without inventing fields |
| D1 Local import | Generated fixture plus processed-IV importer and manifest | Deterministic round trip; correct decimal IVs, supplied year fractions, null metadata and provenance |
| D2 Validate and freeze | Cleaning report, selected contracts and checkpoint compatibility checks | Numerical spot checks, exclusions explained, stable ordering/hash, invalid or mismatched input fails visibly |
| D3 Acquire one raw surface if needed | Cboe file, ThetaData historical sample, or Alpaca current OPRA capture | Actual access and quote semantics demonstrated; pagination complete; curves/settlement resolved; use rights documented |
| D4 Connect training and teaching | Matching synthetic bank, held-out design and local notebook loader | Independent splits, no target leakage, same observations/context as training, clean offline CPU run |

D1–D2 can proceed with generated fixtures while permissions for the external research file are clarified. A public market fixture is released only with established permission; otherwise the required tutorial stays synthetic and any market illustration follows its permitted access/display arrangement.

Meaningful implementation tests cover unit conversion, expiry/day-count handling, distinct roots, timestamp semantics, missing quote data, interrupted pagination, wrong feeds, integrity checks and incompatible checkpoint grids. Live vendor calls are optional integration checks, never default tests or live-session dependencies. This documentation-only change does not claim these loaders or checks already exist.

## Decisions before paid acquisition

- Choose the observation date and whether we need a historical as-of surface or a current capture.
- Resolve a representative sample's SPX/SPXW mapping, settlement, quote-age information and forward/discount inputs.
- Compare a configured Cboe order with the actual ThetaData entitlement and use terms; consider Alpaca only for the current-capture requirement.
- Establish team access and intended tutorial distribution/display rights. No account creation, purchase, authenticated download or provider contact is part of this PR.

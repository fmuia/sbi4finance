# Data needs for the HKDE and rough-Bergomi tutorial

> Historical context: the curriculum assumptions below describe the 10 October discussion. The [current implementation plan](simulator-plan/README.md) retains this options-data research while leaving the live example and implementation sequence open for review.

Research date: 10 October 2026. This note follows the decision to use HKDE for the live tutorial, rough Bergomi as an extension, and American-put pricing as take-home material. It records data-source checks and a proposed specification; no model has been fitted or trained in this task.

## Recommendation

Build the required tutorial around independently generated synthetic parameter–surface pairs. Add one archived European SPX surface for illustration. The same market surface can be fitted with both models. Start with the processed author snapshot below; use a raw sample to develop cleaning, and obtain a controlled Cboe snapshot only if its additional provenance and fields are needed. Public access and permission to redistribute a dataset are separate checks.

A subscription and years of market history are unnecessary for the proposed cross-sectional illustration. An option surface is many contracts observed at one time. Repeated historical dates are needed for claims about calibration stability through time or backtesting, which are outside the core tutorial.

## Data categories

| Category | Required contents | Purpose |
|---|---|---|
| Synthetic training bank | Parameter draw, contract design, fixed pricing inputs, model prices/IVs, numerical settings | Learn the inverse mapping |
| Synthetic validation/test bank | Independent parameter draws and noise; refined/repeated pricing for selected cases | Recovery, calibration, pricing accuracy, reference-posterior checks |
| Illustrative market surface | One date's European quotes or documented processed IVs, with contract/market inputs | Show how the trained workflow is applied to observations |
| Downstream contract specification | Payoff, strike, expiry, exercise/monitoring/settlement rules | Propagate posterior uncertainty into another model price |

Observed exotic or American prices are not required to demonstrate model-price uncertainty. They are required if we claim empirical validation of that prediction. Historical returns, realized-volatility estimates, and observed jump labels are not inputs to this vanilla-option calibration exercise.

## Sources actually available

### Processed SPX teaching surface: first exploratory illustration

[Jim Gatheral's SPX CSV](https://github.com/jgatheral/QuadraticRoughHeston/blob/main/spx_implied_vol_20230215.csv) was inspected: 7,423 rows, 48 expiries, and columns `Expiry,Texp,Strike,Bid,Ask,Fwd,CallMid`. Of these, 6,749 have finite positive ordered bid/ask IVs. The [consuming code](https://github.com/jgatheral/QuadraticRoughHeston/blob/main/utils.py) treats Bid/Ask as decimal annualized IV, not premiums. Texp matches days/365.25. The [original workshop](https://github.com/jgatheral/RoughVolatilityWorkshop/blob/main/QM2024-1%20Econometrics.ipynb) attributes the surface to OptionMetrics through WRDS at the close on 15 February 2023.

Missing fields include precise quote times, raw premium bid/ask, contract roots/settlement flags, and discount factors. `CallMid` numerically resembles an undiscounted call value; do not treat it as a verified raw midpoint. Begin with an IV-based illustration. Confirm upstream redistribution rights before copying the CSV into the public tutorial. The code repository's MIT license alone does not establish the market-data rights.

Inspected repository revisions: QuadraticRoughHeston `76dd1f9122c13333032bd3b8cc1e4d3dcc182009`; original RoughVolatilityWorkshop `d3661bfba054a101e6be09cb74bb745407c937ba`.

### Free raw-data sample: cleaning and ingestion development

[HistoricalData.net samples](https://historicaldata.net/samples.html) provide a public [options archive](https://historicaldata.net/file/options_sample_2022H2.zip). A partial HTTP download was sufficient to inspect the complete first daily CSV: 1 July 2022 includes 19,552 SPX/SPXW rows across 47 expiries, with premium bid/ask and exercise/settlement fields. Only that daily file was inspected, not the full archive.

The provider's [methodology](https://historicaldata.net/methodology.html) explicitly describes last-standing quotes rather than synchronized snapshots; historical quote-update times are absent. Its calculated IVs use provider-specific rates and conventions. Recompute derived values where necessary. The [terms](https://historicaldata.net/terms.html) allow sample evaluation but restrict redistribution. This is a development candidate, not an already validated public teaching dataset.

### Controlled paid snapshot: Cboe DataShop

[Option EOD Summary](https://datashop.cboe.com/option-eod-summary) supports chosen symbols and a single historical date, including a 15:45 Eastern NBBO snapshot. [Individuals can register](https://datashop.cboe.com/register). Request `^SPX`, preserve the SPX/SPXW root, and obtain matching forwards/discount information. Index levels are absent by default; stock/ETF underlying quotes are included. A separate index-feed license is not required merely to purchase SPX option quotes.

An actual configured price was not obtained. The [$500 minimum](https://datashop.cboe.com/academic-discount) belongs to the academic-discount program, not a verified ordinary-order minimum. Establish the intended research/classroom/public-notebook use under the [applicable agreement](https://datashop.cboe.com/documents/Cboe_LiveVol_DataShop_License_Agreement.pdf). No account or order was created.

### Additional routes, lower priority

- [OptionsDX SPX chains](https://www.optionsdx.com/product/spx-option-chain/) advertise free selections; the [FAQ](https://www.optionsdx.com/faq/) describes checkout without billing for free data. Contents were not obtained. Its documented schema leaves settlement/root questions, so availability alone would not settle suitability.
- [Deribit's public API](https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency) returned populated USDC-option quotes without authentication. [Linear USDC options](https://support.deribit.com/hc/en-us/articles/31424932728093-Linear-USDC-Options) offer a European crypto alternative. This changes the underlying and contract conventions; it is an optional alternative, not needed for the equity tutorial.
- Institutional WRDS/OptionMetrics access is not currently available to the team. Yahoo/yfinance SPX coverage has not been established as a reliable source in this project.

### Synthetic resources

The public rough-Bergomi bank supplies parameter–IV pairs: the [flat file](https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/Data/rBergomiTrainSet.txt.gz) has 40,000 rows with four parameters and 88 IVs; the [term-structure file](https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/Data/TrainrBergomiTermStructure.txt.gz) has 80,000 rows with eleven parameters and 88 IVs. These are model outputs, not market observations. The 11-strike/eight-maturity design starts at 0.1 years. It does not supply simulated trajectories, numerical-error estimates per quote, or American prices. The files are gzip-compressed NumPy arrays despite their names. The authors label the volatility-amplitude parameter nu; verify its scaling before mapping to another implementation's eta. Pricing-network weights in the repository are not posterior checkpoints.

Use it to prototype and cross-check, while generating the final bank ourselves on our chosen design. [McCrickerd's rough-Bergomi code](https://github.com/rmcrkd/rough_bergomi) is a simulation reference; validate discretization and Monte Carlo error before adoption. For HKDE use the authors' [fypy implementation](https://github.com/jkirkby3/fypy) as a pricing reference and generate parameter–surface pairs through PROJ. The collaborator's reported archive remains a separate artifact to obtain and audit; its pasted results are not a ready training bank.

The inspected fypy tree contains numerical test fixtures but no market-chain CSV or labelled HKDE bank. Its Yahoo loader is acquisition code, not an archived observation dataset. MIT-licensed source revisions: NN-StochVol-Calibrations `832690651e628481978b034f5ca08e090267bc4a`; rough_bergomi `6a04ca5d2d0f6a45df28ae7bad9c882d537ef14e`; fypy `0e22a518c4a7c05e8faf415e93a22c5e4fa47574`.

## Exactly what is fitted

At valuation time t0, each quote has a contract description x_j and an observed price or IV y_j. Let f_m(theta; x) be the vector of prices/IVs under model m and parameters theta. The fit compares this vector with the observed surface jointly across strikes and expiries.

| Model | Inferred model parameters | Additional model decision |
|---|---|---|
| HKDE | Initial variance, long-run variance, mean-reversion speed, volatility of variance, correlation, jump intensity, upward-jump probability, two jump-size rates | Valid prior/support and numerical PROJ settings |
| Flat rough Bergomi | Initial variance v0, volatility amplitude eta, correlation rho, roughness H | Impose xi0(t)=v0 |
| General rough Bergomi | eta, rho, H, plus any inferred curve parameters | Supply or jointly infer the initial forward-variance curve xi0(t)=E^Q[V_t] |

The asset forward F(0,T) and the forward-variance curve xi0(t) are different objects. The latter is also not simply squared ATM implied volatility. Fixing an estimated curve gives a posterior conditional on that estimate; it omits curve-estimation uncertainty. A flexible curve increases the number of unknowns. This is material when comparing a flat four-parameter rough-Bergomi fit with nine-parameter HKDE. [Calibration formulation](https://arxiv.org/html/1908.08806), [HKDE model](https://arxiv.org/html/2502.13824v1).

For a simple teaching observation model, choose fixed Sigma and write:

    y | theta, x ~ Normal(f_m(theta; x), Sigma)
    p(theta | y, x) ∝ p(y | theta, x) p(theta)

A point calibration minimizes the corresponding weighted residual sum; Bayesian calibration retains the distribution of plausible parameter explanations. A neural estimator approximates that posterior after offline training. The likelihood is specified by the observation model; PROJ or Monte Carlo supplies the prices entering it.

Gaussian price errors and Gaussian IV errors define different models. A bid–ask half-spread is not automatically a standard deviation. State any spread-to-scale assumption explicitly and vary it in a sensitivity check. Do not infer observation noise from the paper's fit RMSE or from simulation error. Keep model discrepancy, quote uncertainty, and numerical error conceptually separate.

## Minimum market record and preprocessing

For raw-price calibration preserve:

| Field group | Fields |
|---|---|
| Identity | Source, underlying, root, contract identifier, currency, multiplier |
| Contract | Strike, call/put, exercise style, expiry/settlement timestamp, AM/PM/cash/physical conventions |
| Observation | Valuation timestamp/time zone, bid, ask, sizes, quote age when available; volume/OI for quality screening |
| External inputs | Discount factor and asset forward at each expiry, their sources and estimation method; spot if required by the implementation |
| Derived values | Year fraction, mid, spread, log(K/F), IV and inversion convention, selection/missingness flags |
| Provenance | File URL, retrieval date, checksum, applicable use terms, cleaning version |

For European vanillas, deterministic financing and dividends can be summarized through discount factors D and forwards F. Matching calls and puts provide a consistency check or a way to estimate F:

    C(K,T) - P(K,T) = D(0,T) [F(0,T) - K]

Use synchronized quotes with matching settlement. If inputs are estimated and then fixed, label the resulting inference as conditional on them. Do not substitute Treasury par yields directly for an audited zero-coupon discount curve, or an asynchronous index close for synchronized pricing inputs without documenting the approximation.

Remove invalid/crossed quotes, inspect stale/zero bids and low-vega contracts, preserve reasons for exclusions, and check static pricing bounds and consistency. Call and put prices linked by parity are not automatically independent observations. Avoid hiding bad rows by smoothing; if smoothing/interpolation is used, document the resulting statistical assumptions.

SPX/SPXW are European but differ in settlement; select consistent contracts. [Cboe specifications](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications).

## Proposed tutorial-sized design

1. Keep two or three deliberately chosen synthetic surfaces with known parameters, plus independent simulations for diagnostic evaluation.
2. Select one processed real SPX snapshot for the optional illustration. A working target is 6–10 expiries and about 10–20 usable strikes per expiry, roughly 60–200 observations, including ATM and both wings. This is a proposed teaching design, not a guarantee of identification. Short maturities of about two weeks onward and longer maturities up to one or two years provide distinct information without requiring 0DTE machinery.
3. Preserve the selected contracts' actual maturities and forward moneyness. Price both models at these exact coordinates. The existing 210-contract collaborator benchmark can remain a separate synthetic reference.
4. Fix the input design before training a fixed-grid posterior network. A network trained on one maturity/strike grid cannot simply accept another. Alternatively condition explicitly on coordinates, masks and other varying inputs. Interpolated quotes are not new independent information.
5. For a held-out-pricing exercise, remove the held-out contracts from the conditioning design during training. Pricing posterior draws on those contracts then supplies a meaningful prediction.

Training bank size is a separate decision from the number of real snapshots. Start with a measured pricing/inference pilot, then use learning curves and diagnostics to size the full bank on the available CPU/GPU resources. Split by base parameter–surface row before creating noise augmentations. Record seeds, prior, fixed curves, quote law, contract order, pricing accuracy and software revision.

## What would count as a successful result

- A visual overlay of observations/bid–ask ranges and fitted prices/IVs, plus residuals by strike and expiry.
- Numerical pricing error small enough relative to the observation scale and financial conclusions, verified by refinement/repeats.
- Posterior uncertainty and prior sensitivity for parameters and relevant financial quantities; a good price fit does not establish parameter uniqueness.
- Independent synthetic recovery/calibration and credible reference-posterior comparisons, including difficult HKDE modes. Actual market parameters are unobserved.
- Posterior predictive or held-out checks. Distinguish uncertainty in the latent model surface from a wider interval for a newly noisy observation.

The posterior is about risk-neutral pricing parameters under the chosen model. It is not a physical return forecast, and a vanilla fit does not uniquely establish path-dependent dynamics or prove that volatility is rough.

## American take-home and next deliverables

Use synthetic European calibration followed by an American put on the same synthetic underlying. Real American quotes are optional for that illustration. A real SPX calibration does not automatically calibrate SPY: their underlying and dividend economics differ. Direct American calibration requires an appropriate pricer, observation conventions and new training surfaces.

Concrete follow-up deliverables:

1. A small source manifest and cleaned exploratory SPX IV fixture, with provenance and unresolved distribution/convention questions recorded.
2. A common contract/input schema and one frozen quote selection used for both models.
3. Explicit prior, noise model, and rough-Bergomi forward-variance treatment.
4. A measured, accuracy-checked synthetic generation pilot for each model, followed by validated independent training/test banks.
5. A public teaching bundle containing generated data and checkpoints; include market data only under established permissions, otherwise provide the appropriately permitted illustration/access route.

This task established sources and requirements. It did not create accounts, buy data, contact providers, train estimators, or perform full market-data quality validation.

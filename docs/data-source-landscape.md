# Option data sources and tools

Reviewed 11 October 2026 for the options application: HKDE calibration, a rough-Bergomi extension and American-put pricing. The live curriculum remains open in PR #3. Read the [scenario-to-data mapping](inference-scenarios-and-data.md) first: returns inference needs a chronological price/return history, which this options-provider comparison does not evaluate. Here the question is whether a source supplies usable **option bid/ask observations**, not merely option trades, underlying index values, model IVs, or a broadly named historical endpoint.

The recommendation is a local file interface backed by our own synthetic fixtures, with one processed European market example. Cboe is the first controlled one-date export to price; ThetaData is the first historical API alternative to evaluate; Alpaca is an optional current-capture API. See the [loading plan](data-loading-plan.md) for schema, validation and implementation milestones.

Provider acquisition belongs to the optional real-market application and empirical-evaluation levels. No provider or real market file is needed for the complete synthetic teaching core; the [evidence levels](inference-scenarios-and-data.md#three-delivery-and-evidence-levels) state exactly which claims each level supports.

## Comparison

Prices below are advertised individual plans at review time, not quotations for this team's use or redistribution rights. Account eligibility and actual entitlements have not been tested. Open-source clients are software access tools; the data remain governed by provider terms.

| Source or tool | Useful capability | Main qualification | Role |
|---|---|---|---|
| Gatheral SPX research CSV | Existing processed European IV surface and forwards | Incomplete original quote/settlement metadata; third-party data provenance | First exploratory real illustration |
| Cboe DataShop | Historical 15:45 NBBO file, chosen symbol/date | Configured cost and use terms; index levels absent by default | Preferred controlled one-off export |
| ThetaData v3 | Historical at-time option NBBO; index and equity options | Local terminal/Java; actual SPXW mapping and entitlement check | First historical API to evaluate |
| Alpaca / `alpaca-py` | Latest OPRA chain, SPX/SPXW contract data documented | Free feed is modified; historical bid/ask endpoint not established | Optional current capture |
| Massive / official Python client | SPX coverage and historical quote endpoint | Quotes require a higher tier than generic snapshot availability | Alternative if already entitled |
| Databento / Python client | Historical OPRA, definitions and consolidated quote data | As-of reconstruction and schema/timestamp work | Research-scale historical alternative |
| Yahoo / `yfinance` | Unofficial current option-chain wrapper | SPX download unverified; rate limits; no historical as-of chain | Exploration only |
| Tradier | Broker API for current chains | Account dependency; historical OHLC excludes expired options | Useful with existing access, not a historical snapshot solution |
| HistoricalData.net / OptionsDX | Free sample or free selections for file preparation | Timestamp/settlement/rights limitations; uneven verification | Development candidates |
| Deribit public API | European linear USDC crypto option quotes | Changes underlying, payoff/currency conventions and teaching story | Optional different-market extension |

## Alpaca

Alpaca now documents index-option contract data on OPRA, including SPX/SPXW. The missing index offering is the underlying spot level, not all index-option quotes. Trading approval and market-data access are different questions; our account access is untested. [Index-options documentation](https://docs.alpaca.markets/us/docs/index-options).

Basic includes an indicative feed whose quotes are modified derivatives of OPRA, not simply genuine NBBO delayed by 15 minutes. Use it only for labeled connector development. Actual OPRA requires subscribed access. The general options-history start date is February 2024; it does not promise a historical bid/ask surface. [Feed definitions](https://docs.alpaca.markets/us/docs/historical-option-data).

The individual Algo Trader Plus plan is listed at $99/month with OPRA. Options data require authenticated requests; Basic is available for paper and live accounts, so developing a connector need not imply funding a brokerage account. Verify applicable subscriber classification and team eligibility separately. [Plans and authentication](https://docs.alpaca.markets/us/docs/about-market-data-api).

The [chain endpoint](https://docs.alpaca.markets/us/reference/optionchain) returns latest per-contract information with expiry/strike/root filters and pagination. `updated_since` filters freshness, not historical observation time. The official [option client](https://github.com/alpacahq/alpaca-py/blob/master/alpaca/data/historical/option.py) exposes historical bars/trades and latest quotes/snapshots/chains, but no historical option-quotes method. Treat historical quote reconstruction as unsupported by the inspected interface until demonstrated otherwise. The [shared SDK client](https://github.com/alpacahq/alpaca-py/blob/master/alpaca/common/rest.py) follows pagination; capture still spans an interval.

Set `feed=OptionsFeed.OPRA` explicitly. Preserve quote times separately from trade times, capture start/end and raw provider responses. Paid access is not permission to publish a CSV: the [customer agreement](https://files.alpaca.markets/disclosures/library/AcctAppMarginAndCustAgmt.pdf) and [redistribution guidance](https://alpaca.markets/support/redistribute-alpaca-api) restrict dissemination. Alpaca is useful if we want a fresh panel, but is not the chosen route for reconstructing an arbitrary past bid/ask chain.

## Historical acquisition alternatives

### Cboe DataShop

The [EOD product](https://datashop.cboe.com/option-eod-summary) supports a single historical date and selected underlyings, including `^SPX`, with a 15:45 Eastern NBBO snapshot and root identifiers. Stock/ETF underlying bid/ask is included; index levels are absent by default. A separate index-feed subscription is not necessary merely to buy SPX option quotes. [Individual registration](https://datashop.cboe.com/register) is supported.

Get a configured quote rather than assuming a price. The [$500 academic minimum](https://datashop.cboe.com/academic-discount) is specific to that program. The [data agreement](https://datashop.cboe.com/documents/Cboe_LiveVol_DataShop_License_Agreement.pdf) must support the intended research/display/distribution use. This route minimizes API engineering for one archived surface; its economics remain to be compared with an actual alternative offer.

### ThetaData

The v3 [at-time endpoint](https://thetadata.net/docs/operations/option_at_time_quote.html) returns the last OPRA NBBO at a requested time and supports expiry/strike wildcards. Use a documented minute boundary rather than sub-minute whole-chain requests. This is directly suited to a historical cross-section, subject to checking actual contributing quote timestamps and SPX/SPXW symbol mapping.

The [retail pricing](https://www.thetadata.net/pricing) lists Value at $40/month, Standard $80 and Pro $160. Value offers minute-resolution history; Standard adds tick quotes. The [free tier](https://thetadata.net/docs/Articles/Getting-Started/Subscriptions.html) offers one year of EOD history, but the [EOD report](https://thetadata.net/docs/operations/option_history_eod.html) is generated at 17:15 ET and contains last NBBOs; it is not a synchronized 15:45 observation.

V3 requires a locally running Theta Terminal and Java 21+. Keep this in an instructor acquisition environment. [Setup](https://thetadata.net/docs/Articles/Getting-Started/Getting-Started.html). Advertised retail plans are personal-use products, not classroom redistribution permission. Evaluate a small sample before committing to this adapter; a low subscription price alone does not establish total suitability.

### Massive formerly Polygon

The [options FAQ](https://massive.com/knowledge-base/categories/options) documents SPX/index-option coverage. The [latest chain snapshot](https://massive.com/docs/rest/options/snapshots/option-chain-snapshot) is convenient, but quote fields depend on quote entitlement. The [historical quotes endpoint](https://www.massive.com/docs/rest/options/trades-quotes/quotes) is listed under Options Advanced at $199/month, with quote history from 7 March 2022. Do not infer usable bid/ask from a cheaper plan's generic snapshot feature.

Use a date-valid contract universe and per-contract quotes to reconstruct a historical panel. The [official Python client](https://github.com/massive-com/client-python) provides an ordinary hosted-API integration without a local terminal. This is a reasonable choice with existing access or broader data needs; a broad subscription is harder to justify for one tutorial surface. Underlying index data and redistribution remain separate entitlement questions.

### Databento

The [SPX OPRA catalogue](https://databento.com/catalog/opra/OPRA.PILLAR/options/SPX) and [Python client](https://github.com/databento/databento-python) provide a route to historical option data and definitions. Historical [pay-as-you-go access](https://databento.com/blog/introducing-new-opra-pricing-plans) avoids assuming a recurring live subscription. Obtain an authenticated `metadata.get_cost` estimate for the exact symbols, interval and schema before acquisition; no price estimate was obtained here.

Reconstruction requires date-valid contract definitions, symbol mapping and consolidated quotes. Inspect [CBBO timestamp semantics](https://databento.com/docs/schemas-and-data-formats/cbbo): inactive intervals can have no row, and a trade event time must not be mistaken for quote age. Select the appropriate event-level or sampled schema for the required freshness claim. This offers stronger historical research infrastructure at the cost of more ingestion work than a prepared snapshot. OPRA coverage alone does not supply spot-index levels or redistribution permission.

## Exploratory wrappers and free samples

### Yahoo Finance and fypy

[`yfinance`](https://github.com/ranaroussi/yfinance) is an open-source unofficial wrapper. Its [option-chain implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/ticker.py) exposes bid/ask, strike, IV and activity fields. Its date parameter selects expiry, not a past observation date; `lastTradeDate` is not a bid/ask timestamp. The project's ordinary Yahoo SPY/SPX page probes returned HTTP 429 on 11 October, so no populated SPX chain is claimed. Cache a validated acquisition outside the lesson rather than depending on live availability. The software license does not override Yahoo's data-use terms.

[`fypy`'s Yahoo loader](https://github.com/jkirkby3/fypy/blob/master/fypy/market/loader/YahooFinanceSurfaceLoader.py) is useful scaffolding, not an audited shortcut: it uses latest-close spot/date inputs, date-only expiries, European parity assumptions for optional discount fitting, and no American exercise adjustment. Reuse only after convention and implementation checks. The pricing library and the market-data provider remain different components.

### Tradier and institutional access

[Tradier](https://docs.tradier.com/docs/endpoints) offers broker/partner API access to current options, but its [historical endpoint](https://docs.tradier.com/docs/historical-data) is OHLCV and excludes expired options. It does not solve historical bid/ask reconstruction. Existing institutional WRDS/OptionMetrics access would change the acquisition choice; none is currently available to the team. Avoid adding a broker dependency solely to retrieve one tutorial panel.

### Free file candidates and another market

- [HistoricalData.net](https://historicaldata.net/samples.html): an actual first daily CSV from the sample was inspected on 10 October and contains SPX/SPXW and SPY records. [Methodology](https://historicaldata.net/methodology.html) describes last-standing EOD quotes and missing historical quote times. Useful for ingestion development, with [redistribution restrictions](https://historicaldata.net/terms.html); not evidence of a synchronized surface.
- [OptionsDX](https://www.optionsdx.com/product/spx-option-chain/): free selections and checkout are documented, but contents were not acquired. [Field definitions](https://www.optionsdx.com/option-chain-field-definitions/) leave root/AM–PM settlement questions. Keep as a candidate until a sample and permitted use are checked.
- [Deribit](https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency): a normal unauthenticated USDC-option request returned populated quotes during the earlier audit. [Linear USDC contracts](https://support.deribit.com/hc/en-us/articles/31424932728093-Linear-USDC-Options) are a possible European crypto extension, requiring their own payoff, settlement-index, forward and currency conventions. This does not solve the chosen equity example.

## Research files and synthetic resources

The [pinned Gatheral CSV](https://github.com/jgatheral/QuadraticRoughHeston/blob/76dd1f9122c13333032bd3b8cc1e4d3dcc182009/spx_implied_vol_20230215.csv) has 7,423 rows across 48 expiries; 6,749 have finite positive ordered IV bid/ask. The [consuming code](https://github.com/jgatheral/QuadraticRoughHeston/blob/76dd1f9122c13333032bd3b8cc1e4d3dcc182009/utils.py) uses decimal annualized IVs; supplied `Texp` follows days/365.25. The [workshop](https://github.com/jgatheral/RoughVolatilityWorkshop/blob/d3661bfba054a101e6be09cb74bb745407c937ba/QM2024-1%20Econometrics.ipynb) attributes it to OptionMetrics/WRDS at the close on 15 February 2023. Precise quote timestamps, raw premium bid/ask, root/settlement flags and discount factors are missing. `CallMid` numerically resembles an undiscounted call value, but that convention is not established. Use the file as processed IV data and confirm third-party redistribution rights before rebundling.

The public [flat rough-Bergomi bank](https://github.com/amuguruza/NN-StochVol-Calibrations/blob/832690651e628481978b034f5ca08e090267bc4a/Data/rBergomiTrainSet.txt.gz) contains 40,000 rows, four parameters and 88 IVs per row. The [term-structure bank](https://github.com/amuguruza/NN-StochVol-Calibrations/blob/832690651e628481978b034f5ca08e090267bc4a/Data/TrainrBergomiTermStructure.txt.gz) has 80,000 rows and eleven parameters. These are gzip-compressed NumPy arrays, not text despite their names. Their grid starts at 0.1 years. Verify parameter scaling and ordering; these files contain neither observed markets nor American prices, path histories or quote-noise assumptions. Treat supplied neural pricing weights as pricing assets, not posterior checkpoints.

For our final banks, generate prices with validated model engines and add the declared observation noise. [fypy](https://github.com/jkirkby3/fypy) supplies HKDE/PROJ reference code; [rough_bergomi](https://github.com/rmcrkd/rough_bergomi) supplies a simulation reference. The inspected fypy tree did not contain the paper's market-chain CSV or a labelled HKDE training bank. The collaborator's archive is a separate artifact to obtain and audit.

## Evidence and selection gate

Research-file/sample availability was checked through ordinary downloads in the earlier audit. Yahoo probes failed. Alpaca, ThetaData, Massive, Databento and Tradier capabilities above were checked in primary documentation/code, not through authenticated accounts. No provider account, purchase, broker trade, or redistribution permission was obtained in this review.

Select a paid route only after a representative sample demonstrates the required product, date, actual quote feed, settlement metadata, time semantics, cost and permitted use. Preserve the raw provider output and map it into the same local panel interface; provider choice should not silently change the statistical experiment.

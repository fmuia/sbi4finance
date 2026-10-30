# Data sources

Raw quote files are not committed. The repository keeps derived data only:
implied volatilities, prices in forward units, half-spreads and the design-grid readings. This page says where each raw
file came from, so that a collaborator can fetch a file of the same kind.

## Option chains: Cboe delayed quotes

Each URL returns the current delayed-quotes chain as JSON (the format `sbi4finance.calibration.chain.parse_cboe_json`
reads). It does not serve past dates: a file downloaded today holds today's quotes. The files of October 2026 are kept
by Francesco Muia; ask him for a copy rather than downloading a new one if a study must be reproduced exactly. Each
derived file records the SHA-256 of the raw file it came from.

| Ticker | Underlying | URL | File used | Derived files |
| --- | --- | --- | --- | --- |
| SPY | SPDR S&P 500 ETF | https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json | retrieved 2026-10-22 08:39 UTC; quotes of the 2026-10-21 close | `tutorial/assets/results/calibration/spy/surface.json`, and from it `checks/spy_early_exercise/SPY_2026-10-21.json` |
| SPY | SPDR S&P 500 ETF | https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json | retrieved 2026-10-30 01:23 UTC (file timestamp 2026-10-30 00:13:44); quotes of the 2026-10-29 close | `checks/spy_early_exercise/SPY_2026-10-29.json` |
| EWI | iShares MSCI Italy ETF | https://cdn.cboe.com/api/global/delayed_quotes/options/EWI.json | file timestamp 2026-10-22 23:21:36; quotes of the 2026-10-22 close | `checks/thin_chains/EWI_2026-10-22.json` |
| EWG | iShares MSCI Germany ETF | https://cdn.cboe.com/api/global/delayed_quotes/options/EWG.json | file timestamp 2026-10-22 23:21:31; quotes of the 2026-10-22 close | `checks/thin_chains/EWG_2026-10-22.json`, `configs/thin/ewg_layout.json`, `checks/thin_chains/EWG_observation.json` |
| EWQ | iShares MSCI France ETF | https://cdn.cboe.com/api/global/delayed_quotes/options/EWQ.json | file timestamp 2026-10-22 23:21:26; quotes of the 2026-10-22 close | `checks/thin_chains/EWQ_2026-10-22.json` |
| FEZ | SPDR EURO STOXX 50 ETF | https://cdn.cboe.com/api/global/delayed_quotes/options/FEZ.json | file timestamp 2026-10-23 23:34:24; quotes of the 2026-10-23 close | `checks/thin_chains/FEZ_2026-10-23.json` |

To download a chain (from a machine whose network allows it):

```bash
curl -sS -o EWG.json https://cdn.cboe.com/api/global/delayed_quotes/options/EWG.json
```

## Rates and dividends

- Rate: the 3-month Treasury constant-maturity yield (Alpha Vantage `TREASURY_YIELD`, daily, `3month`); 4.22% on
  2026-10-22, used flat. The 2026-10-23 value was not yet published when FEZ was read, so FEZ uses 4.22% too.
- Dividends: Alpha Vantage `DIVIDENDS` for each ETF; the amounts and assumed ex-dates are in
  `checks/thin_chains/settings.json`, and SPY's in `checks/spy_early_exercise/settings.json`.
- SPY's early-exercise check uses S8's Treasury curve for 2026-10-21 and the yields of 2026-10-28 (FRED `DGS3MO`,
  `DGS2`: 4.18% and 4.67%) for 2026-10-29; spots are the closes from Alpha Vantage `TIME_SERIES_DAILY`.

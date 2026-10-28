# What the sources say about calibration practice

Checked 24 October 2026. [Course index](README.md). The course states a practice only when this page sources it.
Everything else is described as the course's own choice. The quotes came through a web tool that returns short
excerpts, so check a quote against the original before it goes on a slide.

| Statement | Status | Source |
| --- | --- | --- |
| Models for exotics are calibrated to the vanilla smile | Documented | Hagan et al. (2002), p. 5, on local volatility: "the most popular method of managing exotic equity and foreign exchange options". Bühler (2006), p. 10: Dupire's local volatility "is today a standard tool for evaluating exotic derivatives". |
| The forward-variance curve comes from variance swaps or their log-contract strip | Documented, with variants | Bergomi (2005) calibrates the curve to variance-swap levels, which vanillas replicate statically. Bayer, Friz and Gatheral (2016, preprint §5.2) use the log-contract strip on an SVI-fitted surface, because variance-swap quotes are poor and wide. El Euch, Gatheral and Rosenbaum (2019) fix the curve "to match at-the-money volatilities". |
| A vol-of-vol with no option market is a choice, not a calibration | Documented | Bergomi (2005): such parameters "cannot be calibrated on market prices"; choosing them "is thus a trading decision". |
| Parameters the smile does not identify are often fixed | Documented for SABR's β and Heston's κ | Hagan et al. (2002), §3.1: β "cannot be determined by fitting a market smile", and is set from the historical backbone or chosen a priori. Bergomi's Société Générale slides (2003): daily S&P 500 Heston fits with 1/κ set to six months. Janek, Kluge, Weron and Wystup (2011), §1.4.1, recommend fixing κ. |
| Fixing H at 0.10 | **Not documented as practice** | Gatheral, Jaisson and Rosenbaum (2018) estimate H "of order 0.1" from realized volatility. Bayer, Friz and Gatheral (2016) use H = 0.07. Gatheral, quoted by Risk.net (2021): "H is typically closer to 0.05". El Euch, Gatheral and Rosenbaum (2019) calibrate H. The course's H = 0.10 is a baseline chosen by analogy with fixing β or κ. |
| Holding the curve flat beyond the last quote | **No single convention** | QuantLib and ORE hold implied volatility flat at a fixed strike (library defaults). An IHS Markit–Milliman paper (2022) calls constant instantaneous variance beyond the last point "one approach often used", for insurance liabilities. FactSet reuses the nearest expiry's coefficients; the JSE fits a power-law decay. The course's flat continuation of the last forward-variance level is one simple rule among these. |
| A thin underlying is marked from a proxy | Allowed, without a recipe | Regulation (EU) 2016/101, Art. 3(3)(a), lists "proxy data based on similar instruments" among the alternatives for an expert-based approach. ORE and FactSet build proxy surfaces. "Index volatility plus a spread" was not found in any reputable source and is not stated. |
| Valuation uncertainty at a 90% point | Documented | Regulation (EU) 2016/101: Art. 9(5) for market price uncertainty, per valuation input; Art. 10(6) for close-out costs; Art. 11(3) for model risk, from "alternative appropriate modelling and calibration approaches". Art. 11(1): uncertainty from calibrating to market-derived parameters belongs to Art. 9, not to model risk. |
| Consensus services give percentile levels | Documented (vendor) | S&P Global, *Totem Service* (2023): consensus mid prices with "explicit 10th and 90th percentile levels for use in prudent valuation calculations"; exotic equity services include autocallables and cliquets. |
| Banks use rough volatility in production | **Not documented** | Risk.net (2019): "nobody is currently using these in production". Risk.net (2021): adopters are "mostly hedge funds and non-bank market-makers"; banks "have been slower movers". The course says only that rough volatility fits index smiles well. |
| Rough models fit SPX smiles best | **Disputed** | Abi Jaber and Li (2024): rough models with H in (0, 1/2) "are inconsistent with the global shape of SPX smiles"; between one week and three months they underperform one-factor Markovian models with as many parameters. The course cites this where it discusses model error. |

## References

- Abi Jaber, E., Li, S. (2024). Volatility models in practice: rough, path-dependent or Markovian? [arXiv:2401.03345](https://arxiv.org/abs/2401.03345)
- Bayer, C., Friz, P., Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*. Preprint: WIAS 2071.
- Bergomi, L. (2003). Dynamic aspects of smile models. Société Générale, slides.
- Bergomi, L. (2005). Smile dynamics II. *Risk*, October.
- Bühler, H. (2006). *Volatility Markets*. PhD thesis.
- Commission Delegated Regulation (EU) 2016/101 (prudent valuation), Articles 3 and 9 to 11. [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0101)
- El Euch, O., Gatheral, J., Rosenbaum, M. (2019). Roughening Heston. *Risk*, May.
- Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*. [arXiv](https://arxiv.org/abs/1410.3394)
- Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002). Managing smile risk. *Wilmott Magazine*.
- Janek, A., Kluge, T., Weron, R., Wystup, U. (2011). FX smile in the Heston model. In *Statistical Tools for Finance and Insurance*, Springer. [arXiv](https://arxiv.org/abs/1010.1617)
- Risk.net (2 December 2019). Rising star in quant finance: Horvath, Muguruza and Tomas.
- Risk.net (1 February 2021). Quants of the year: Jim Gatheral and Mathieu Rosenbaum.
- S&P Global Market Intelligence (2023). *Totem Service*. [PDF](https://cdn.ihsmarkit.com/www/pdf/0624/MI_DVA_2937551_Totem.pdf)
- Baranes, A., Boumezoued, A. (2022). Estimating long-term implied volatility for the valuation of insurance liabilities. IHS Markit and Milliman.

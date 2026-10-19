# Slot 1 (intro): design and evidence

Prepared 19 October 2026. This document records why the intro uses betas across many stocks,
the evidence behind every claim the lesson makes, and the objections it has to survive. The
lesson is [`notebooks/tutorial_1_why_bayesian.py`](../notebooks/tutorial_1_why_bayesian.py); the
numbers come from `scripts/beta_study.py`, with results in
[`docs/results/intro-betas/`](results/intro-betas/).

## 1. What the intro has to do

- **Add one layer only: Bayesian reasoning.** Every posterior in the lesson is exact, in closed
  form; there is no network and no sampler.
- **Stay simple.** The team's reference was PyMC Labs' post on Bayesian computation in finance,
  whose main example is a Bayesian market-model regression with a distribution over a stock's
  beta. The intro starts from that example.
- **Be bulletproof.** The audience includes econometricians, most of them frequentists. Every
  claim is a theorem with its conditions stated, a documented industry practice, or a measured
  result with its uncertainty.
- **Hand over to the method slots.** The intro's structure, one model per name and a prior that
  describes the population of names, is what slots 4 and 5 need for volatility models. Its closed
  form is what they lose, which is the problem amortized inference solves.

## 2. How we chose the example

Two earlier versions were reassessed and dropped as the intro:

1. **"How many jumps a year?"** The jump frequency exists only inside a model, the data barely
   identify it, and showing a Wald interval fail on it invites "you picked a straw man".
2. **"How likely is a 7% loss day next year?"** A decision question, answered with a jump model
   and studied on 1,200 simulated histories. Its results were real but nuanced: the penalized fit
   and a flat-prior posterior came close to Bayes, Bayes lost at quiet truths, and the plug-in was
   fine for a Student-t model. That material is about how much the prior matters, so it moved to
   slot 6 as a seed (#42).

Other candidates we considered:

| Candidate | Verdict |
|---|---|
| Probability of default with zero defaults | Simple and clean (the estimate is 0), but credit risk, far from the agenda. |
| Backtest overfitting (the best of many strategies) | Popular, but a multiple-testing story; false-discovery corrections handle it. |
| Covariance estimation with many stocks | Strong, but multivariate, and Ledoit–Wolf shrinkage is already the standard fix. |
| Calibrating an option model to prices | On topic, but it adds option pricing to the first slot. |
| Expected returns across stocks | The classic James–Stein example, but about returns, not risk. |
| **Betas across stocks** | **Chosen.** Simple (the reference post's example), a theorem behind it, used by every data vendor, measurable on real data, and the hedge is a risk decision. |

## 3. The claims, and what each rests on

| The lesson says | Basis |
|---|---|
| Vendors adjust every beta toward 1: Bloomberg's adjusted beta is 0.67 × raw + 0.33, from two years of weekly returns against the S&P 500 by default, and Value Line adjusts its betas toward 1. | Practice: university library guides to the Bloomberg terminal, and Value Line's own reader guides. The weights come from Blume (1971, 1975). |
| Each least-squares beta is unbiased, yet the highest estimates are too high. | Simulation (§5.1): the 50 highest of 500 are too high by 0.07, 0.16 and 0.26 for daily, weekly and monthly data. |
| Estimating each mean separately is beaten by shrinkage, whatever the truth. | Theorem: Stein (1956), James and Stein (1961), for three or more means with Gaussian errors of known variance; toward the estimates' own average with four or more (Lindley 1962). Efron and Morris (1973) showed it is empirical Bayes. Our estimates have estimated, unequal variances, so the lesson says the simulations show the same result, not that the theorem covers them. |
| The prior can be learned from the cross-section, net of sampling noise. | Vasicek (1973); the correction for noise is Lally's (1998) point. Simulation: the learned prior is as good as the true one (§5.1). |
| A fixed amount of shrinkage is right for one setting at most. | Simulation (§5.1): Bloomberg's rule is worse than least squares with daily data, close to Bayes with weekly data, and short of it with monthly data. |
| Least-squares betas mis-size a hedge. | Simulation (§5.2): hedging the 50 highest-beta stocks leaves the book net short the market by 8 positions in 50 with weekly data, in every one of 200 universes. |
| Both 90% intervals cover; the Bayesian one is narrower. | Simulation (§5.3). The least-squares interval is exact per stock; the Bayesian one is calibrated on average over the universe. |
| On real stocks, Bayes predicts next period's betas better, and betas drift. | Measurement (§5.5), with bootstrap intervals; Levi and Welch (2017) find the same on a much larger sample and shrink a second time for drift. |
| Risk models already shrink. | Practice: MSCI's Barra USE4 applies Bayesian shrinkage to specific risk (Menchero, Orr and Wang 2011); Ledoit–Wolf shrinkage for covariance matrices. |

## 4. Objections, and the answers in the lesson

- **"Shrinkage is frequentist too."** Yes. The James–Stein estimator is frequentist and its
  advantage is a frequentist theorem; the lesson says so. The Bayesian view adds the amount of
  shrinkage for each stock and a distribution to carry into the decision.
- **"The prior is subjective."** Here it is estimated from the other stocks. Exercise 2 shows the
  other side: with only five stocks, an outside prior aimed at the right centre beats one learned
  from the data.
- **"Your simulation assumes the answer."** The theorem does not need the simulation, and the
  real-data test agrees with it.
- **"Bloomberg's rule is good enough."** It is close to Bayes for weekly data on a universe centred
  at 1. It is worse than least squares with daily data, and for a low-beta universe it pushes every
  estimate the wrong way, in simulation (exercise 1) and on the real defensive stocks (§5.5).
- **"Errors average out over many stocks."** Not in a selected portfolio. The hedge example takes
  the 50 highest estimates, and their errors share a sign.
- **"Betas drift, and the market model is too simple."** The drift is measured (§5.5) and the
  lesson cites Levi and Welch's second shrinkage. Model error is slot 6's subject.
- **"This is not about volatility."** The intro teaches the structure the volatility slots reuse;
  the volatility models come in slots 4 and 5.

## 5. Results

All simulated universes have 500 stocks with true betas drawn from N(1, 0.3²), a median
idiosyncratic volatility of 25% a year (log-normal across stocks, sd 0.35) and a market
volatility of 16% a year: 200 universes per setting.

### 5.1 Accuracy, at three data frequencies

| Data | Estimator | Squared error vs least squares | Universes beating least squares | Bias of the 50 highest | Hedge exposure, positions (mean ± sd) |
|---|---|---|---|---|---|
| daily returns, 1 year | Least squares | 1.00 | 0 of 200 | +0.07 | -3.5 ± 1.0 |
|  | Bloomberg's rule | 1.22 | 6 of 200 | -0.12 | +5.8 ± 0.9 |
|  | Bayes, prior learned from the data | 0.83 | 200 of 200 | +0.00 | +0.1 ± 0.8 |
|  | Bayes, true population as prior | 0.83 | 200 of 200 | +0.00 | +0.0 ± 0.8 |
| weekly returns, 2 years | Least squares | 1.00 | 0 of 200 | +0.16 | -8.1 ± 1.7 |
|  | Bloomberg's rule | 0.76 | 200 of 200 | -0.04 | +2.0 ± 1.6 |
|  | Bayes, prior learned from the data | 0.68 | 200 of 200 | +0.00 | +0.0 ± 1.2 |
|  | Bayes, true population as prior | 0.67 | 200 of 200 | +0.00 | -0.0 ± 1.1 |
| monthly returns, 5 years | Least squares | 1.00 | 0 of 200 | +0.26 | -13.2 ± 2.6 |
|  | Bloomberg's rule | 0.62 | 200 of 200 | +0.04 | -2.1 ± 2.2 |
|  | Bayes, prior learned from the data | 0.55 | 200 of 200 | +0.00 | -0.1 ± 1.5 |
|  | Bayes, true population as prior | 0.55 | 200 of 200 | +0.00 | -0.2 ± 1.3 |

### 5.2 The hedge

The book holds one unit of each of the 50 stocks with the highest least-squares beta and sells
each estimator's beta in the market. Its residual market exposure is the sum of true minus
estimated betas, in positions (last column above). With weekly data and \$1m per stock, least
squares leaves about \$8m net short the market on a \$50m book, negative in all 200 universes; at
16% market volatility that is about \$1.3m a year of P&L volatility the risk report does not show.

### 5.3 Intervals

| Data | Median standard error | Weight Bayes keeps on each estimate | Least squares: coverage, width | Bayes: coverage, width |
|---|---|---|---|---|
| daily returns, 1 year | 0.106 | 0.90 | 90%, 0.33 | 90%, 0.31 |
| weekly returns, 2 years | 0.161 | 0.79 | 90%, 0.52 | 90%, 0.45 |
| monthly returns, 5 years | 0.229 | 0.68 | 90%, 0.69 | 90%, 0.56 |

### 5.4 The exercises

Squared error relative to least squares (universes beating least squares), and the bias of the
highest estimates for least squares, Bloomberg's rule and Bayes; weekly data:

| Universe | Bloomberg's rule | Bayes, learned prior | Bayes, true prior | Bias of the highest |
|---|---|---|---|---|
| Utilities: 60 stocks, betas around 0.6 ± 0.2 | 1.15 (48 of 200) | 0.52 (199 of 200) | 0.50 (200 of 200) | +0.16 / +0.17 / -0.01 |
| Five stocks | 0.75 (242 of 400) | 0.86 (248 of 400) | 0.64 (270 of 400) | +0.11 / -0.03 / -0.03 |

### 5.5 Real stocks: Blume's test

Weekly adjusted closes from Alpha Vantage for 20 large US stocks and SPY, November 1999 to
October 2026: ABT, AMGN, CL, D, DUK, ED, GIS, JNJ, KMB, KO, MRK, NEE, PEP, PFE, PG, SO, T, UNH,
VZ and WMT. The free API quota stopped the download there, so the sample is utilities, consumer
staples, health care and telecoms, with an average beta of 0.53. Betas are estimated in
consecutive, non-overlapping windows; each estimator from one window predicts the next window's
least-squares beta. The change in squared error is relative to least squares, with a 90%
bootstrap interval over pairs of windows. `docs/results/intro-betas/window_betas.csv` holds
every window's beta and standard error; the prices are not in the repository
(`scripts/beta_window_betas.py` rebuilds the table from any price file).

| Windows | Pairs of windows (stock-window pairs) | Least squares, mean squared error | Bloomberg's rule | Bayes | Cross-sectional mean alone | Bayes better in | Slope, next on current (noise alone) |
|---|---|---|---|---|---|---|---|
| 1 year | 26 (520) | 0.120 | +1% (-16% to +19%) | -20% (-25% to -15%) | -14% (-25% to -1%) | 23 of 26 | 0.47 (0.54) |
| 2 years | 12 (240) | 0.092 | +0% (-25% to +34%) | -12% (-19% to -5%) | -13% (-32% to +17%) | 10 of 12 | 0.34 (0.69) |

The last column compares the slope of next-window betas on current ones with the slope that
estimation noise alone would give if betas did not change. Over two years the observed slope is
half of it: real betas drift.

## 6. What the intro claims, and what it does not

It claims what §3 lists, at the conditions stated there. It does not claim:

- **That frequentist inference fails.** The least-squares interval is exact for each stock, and
  James–Stein shrinkage is itself frequentist.
- **That Bayes improves every stock.** It improves the total; the lesson's single simulated stock
  is one where the raw estimate was closer to the truth.
- **That the real-data result holds for every universe.** Our sample is 20 defensive stocks;
  Blume (1975) and Levi and Welch (2017) cover far larger samples.
- **That the market model is right.** Slot 6 is about noticing when the model or the prior does
  not fit.

## References

Blume (1971), *Journal of Finance* 26(1); Blume (1975), *Journal of Finance* 30(3); Efron and
Morris (1973), *Journal of the American Statistical Association* 68(341); Efron and Morris (1977),
*Scientific American* 236(5); Jacquier, Polson and Rossi (1994), *Journal of Business and Economic
Statistics* 12(4); James and Stein (1961), *Proceedings of the Fourth Berkeley Symposium* 1; Lally
(1998), *Financial Review* 33(3); Ledoit and Wolf (2004), *Journal of Multivariate Analysis* 88(2);
Levi and Welch (2017), *Journal of Financial and Quantitative Analysis* 52(2); Lindley (1962),
discussion of Stein, *Journal of the Royal Statistical Society B* 24(2); Menchero, Orr and Wang
(2011), *The Barra US Equity Model (USE4): Methodology Notes*, MSCI; Stein (1956), *Proceedings of
the Third Berkeley Symposium* 1; Vasicek (1973), *Journal of Finance* 28, 1233–1239.

# Slot 1 (intro): design and evidence

Prepared 19 October 2026, and revised the same day after two independent adversarial reviews, one
of the claims and statistics and one of the code. This document records why the intro uses betas
across many stocks, the evidence behind every claim the lesson makes, the objections it has to
survive and where its claims stop. The lesson is
[`notebooks/tutorial_1_why_bayesian.py`](../notebooks/tutorial_1_why_bayesian.py); the numbers come
from `scripts/beta_study.py`, with results in [`docs/results/intro-betas/`](results/intro-betas/).

## 1. What the intro has to do

- **Add one layer only: Bayesian reasoning.** Every posterior in the lesson is exact, in closed
  form; there is no network and no sampler.
- **Stay simple.** The team's reference was PyMC Labs' post on Bayesian computation in finance,
  whose main example is a Bayesian market-model regression with a distribution over a stock's
  beta. The intro starts from that example.
- **Be bulletproof.** The audience includes econometricians, most of them frequentists. Every
  claim is a theorem with its conditions stated, a documented industry practice, or a measured
  result with its uncertainty, and the lesson shows where its own method fails.
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
| Vendors adjust every beta toward 1: Bloomberg's adjusted beta is 0.67 × raw + 0.33, from two years of weekly returns against the S&P 500 by default, and Value Line adjusts its betas toward 1. | Practice: university library guides to the Bloomberg terminal; for Value Line, its report glossary ("adjusted for their long-term tendency to converge toward 1.00"), as quoted in BetterInvesting's guide to Value Line reports and in utility rate-case filings. The weights go back to Blume (1971, 1975). |
| Each least-squares beta is unbiased, yet the highest estimates are too high. | Simulation (§5.1): the 50 highest of 500 are too high by 0.07, 0.16 and 0.26 for daily, weekly and monthly data. |
| Pooling beats separate estimates when every estimate has the same noise. | Theorem: Stein (1956), James and Stein (1961): three or more means with Gaussian errors of one common known variance (or a common variance estimated independently) have lower *expected* total squared error under shrinkage, at every truth; toward the estimates' own average with four or more (Lindley 1962). Efron and Morris (1973) showed it is empirical Bayes. |
| With unequal noise, as here, the theorem does not cover the lesson's estimator. | Bock (1975) gives minimax estimators for unequal known variances; the lesson's estimator is not one of them. §5.4 shows a population where it loses to least squares. |
| A frequentist reaches the same estimator. | It is the empirical best linear unbiased predictor of a random-coefficients model (Robinson 1991). |
| The prior can be learned from the cross-section, net of sampling noise. | Vasicek (1973); the correction for noise is Lally's (1998) point. Simulation: the learned prior is as good as the true one in the lesson's population (§5.1). |
| How much to shrink depends on noise against the spread of true betas; a fixed amount suits one ratio at most. | The best fixed weight is τ²/(τ² + mean s²): 0.88, 0.74 and 0.62 in the lesson's population with daily, weekly and monthly data (§5.3). Bloomberg's 0.67 over-shrinks badly with daily data, a little with weekly, and is near the best fixed weight with monthly. The frequency stands in for the ratio. |
| Least-squares betas mis-size a hedge of the highest-beta stocks, and the bias comes from the selection. | Simulation (§5.2): net short 8.1 positions in 50, short in all 200 universes; the same hedge on 50 random stocks is unbiased. |
| Least-squares intervals are exact per stock but miss more of the selected stocks; Bayesian intervals cover the selected stocks but miss more of the extreme ones. | Simulation (§5.3). |
| The pooled prior fails when noise is tied to the beta; a prior per volatility group repairs most of it. | Simulation (§5.4), five populations. |
| On real stocks, Bayes predicts next year's betas better, Bloomberg's rule makes no measurable difference, and betas drift. | Measurement (§5.6), with block-bootstrap intervals and robustness checks; Levi and Welch (2017) find the same on a much larger sample and shrink a second time for drift. |
| Risk models already shrink, and condition the prior on more than the cross-section. | Practice: MSCI's Barra USE4 applies Bayesian shrinkage to specific risk (Menchero, Orr and Wang 2011); Rosenberg and Guy (1976) predicted betas from company fundamentals; Ledoit–Wolf shrinkage for covariance matrices. |

## 4. Objections, and the answers in the lesson

- **"Shrinkage is frequentist too."** Yes. The James–Stein estimator and its theorem are
  frequentist, and the lesson's estimator is also a random-coefficients model's empirical BLUP. The
  lesson says so in the main text: the contrast is one stock at a time against all stocks together.
  The Bayesian view adds the amount of shrinkage for each stock and a distribution to carry into
  the decision.
- **"Stein's theorem needs equal variances."** Yes, and the lesson says the theorem does not cover
  its estimator, then shows a population where that estimator loses to least squares (§5.4).
- **"The prior is subjective."** Here it is estimated from the other stocks. Exercise 2 shows the
  other side: with only five stocks, an outside prior aimed at the right centre beats one learned
  from the data, mostly because the amount of shrinkage is hard to learn from five numbers.
- **"Your simulation assumes the answer."** Partly true, and the lesson shows it. Sections 5–7
  assume the prior's population. Section 8 runs four others. The pooled prior survives skewed
  betas but loses to Bloomberg's rule when noise rises with the beta or its distance from 1, and to
  least squares with a noisy fringe; a normal prior with the true mean and spread fails the same
  way, so the problem is the prior's shape. A prior per volatility group repairs most of it.
- **"Bloomberg's rule is good enough."** It is robust where the pooled prior is not (§5.4), and
  close to the best fixed weight with weekly or monthly data. It over-shrinks badly with daily data,
  and for a low-beta universe it pulls the estimates the wrong way on average: worse than least
  squares in simulation (exercise 1), and no measurable improvement on the real defensive stocks,
  where it moved the estimate away from the next window's beta in 57% (one-year windows) and 66%
  (two-year windows) of cases (§5.6).
- **"The classical interval is exact."** For a stock fixed in advance. For the 50 stocks selected
  for their high estimates it covers 81% with weekly data; the Bayesian interval covers 90% of them
  but only 78% of stocks with extreme true betas (§5.3). Each answers its own question.
- **"Errors average out over many stocks."** Not in a selected portfolio. The hedge example takes
  the 50 highest estimates, and their errors share a sign; for 50 random stocks they do average out.
- **"Betas drift, and the market model is too simple."** The drift is measured (§5.6), also with
  heteroskedasticity-robust standard errors, and the lesson cites Levi and Welch's second
  shrinkage. Model error is slot 6's subject.
- **"Twenty survivor stocks prove nothing."** Agreed for the general claim; the lesson calls the
  real-data section an illustration and cites Blume (1975) and Levi and Welch (2017) for the large
  samples.
- **"This is not about volatility."** The intro teaches the structure the volatility slots reuse;
  the volatility models come in slots 4 and 5.

## 5. Results

Unless stated otherwise, simulated universes have 500 stocks with true betas drawn from
N(1, 0.3²), idiosyncratic volatility log-normal around 25% a year (dispersion 0.35) and a market
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
squares leaves about \$8m net short the market on a \$50m book, short in all 200 universes. The
same hedge on 50 stocks drawn at random is unbiased (+0.1 positions): the bias is the selection.

The book's idiosyncratic risk is about \$2.2m a year. The hidden short is a market position worth
8.1 × 16% ≈ \$1.3m a year on its own; volatilities add in squares, so the true P&L volatility is
about \$2.5m against the \$2.2m a risk report that takes the book as hedged would show.

### 5.3 Intervals and weights

| Data | Median standard error | Best fixed weight | Least squares: coverage of all, of the 50 highest, of extreme betas; width | Bayes: coverage of all, of the 50 highest, of extreme betas; width |
|---|---|---|---|---|
| daily returns, 1 year | 0.099 | 0.88 | 90%, 86%, 90%; 0.33 | 90%, 90%, 84%; 0.31 |
| weekly returns, 2 years | 0.154 | 0.74 | 90%, 81%, 90%; 0.51 | 90%, 90%, 78%; 0.45 |
| monthly returns, 5 years | 0.207 | 0.62 | 90%, 77%, 90%; 0.69 | 90%, 89%, 72%; 0.56 |

"Extreme" means a true beta beyond 1 ± 0.45 (1.5 population sds, about one stock in eight). The
median standard error is the median over universes of each universe's median, and the width the
median over universes of each universe's median width. The best fixed weight is the weight on the
estimate that minimizes expected squared error for a single rule, τ²/(τ² + mean s²); a fixed rule
with it would score that weight as its error relative to least squares, which Bayes beats at every
frequency because each stock gets its own weight.

### 5.4 When the prior is wrong

Weekly data, 500 stocks, 200 universes per population. Squared error relative to least squares
(universes beating least squares), the Bayes bias of the 50 highest, and the hedge exposure:

| Population | Bloomberg's rule | Bayes, one prior | Bayes, a prior per volatility group | Bayes, true mean and spread | Bias of the 50 highest, one prior | Hedge, one prior | Hedge, a prior per group |
|---|---|---|---|---|---|---|---|
| normal betas, noise unrelated to beta | 0.76 (200) | 0.68 (200) | 0.68 (200) | 0.67 | -0.00 | +0.0 | -0.0 |
| right-skewed betas (skewness 0.8) | 0.77 (199) | 0.68 (200) | 0.69 (200) | 0.68 | -0.04 | +1.8 | +1.8 |
| right-skewed betas, noise rises with beta (correlation 0.5) | 0.77 (199) | 0.80 (197) | 0.70 (200) | 0.80 | -0.09 | +4.5 | +1.6 |
| noise rises with distance from 1 (correlation 0.5) | 0.77 (200) | 0.83 (198) | 0.77 (200) | 0.83 | -0.04 | +2.0 | -0.1 |
| a quiet core and a noisy fringe | 0.72 (200) | 1.31 (8) | 0.74 (196) | 1.31 | -0.12 | +6.2 | +3.1 |

The fringe population has 90% of stocks with betas N(1, 0.1²) and 15% idiosyncratic volatility
and 10% at 0.4 or 1.6 with 50%; it is the population form of the reviewer's fixed-beta
counterexample, which gave 1.34 at fixed true betas. The "prior per volatility group" learns one
prior for each fifth of the stocks ranked by standard error. It has a cost when groups are small:
on the 60-stock utilities universe (12 per group) it scores 0.61 against 0.52 for one prior.

### 5.5 The exercises

Squared error relative to least squares (universes beating least squares), and the bias of the
highest estimates for least squares, Bloomberg's rule and Bayes; weekly data:

| Universe | Bloomberg's rule | Bayes, learned prior | Bayes, true prior | Bias of the highest |
|---|---|---|---|---|
| Utilities: 60 stocks, betas around 0.6 ± 0.2, top 10 | 1.15 (48 of 200) | 0.52 (199 of 200) | 0.50 (200 of 200) | +0.16 / +0.17 / -0.01 |
| Five stocks, top 1 | 0.75 (242 of 400) | 0.86 (248 of 400) | 0.64 (270 of 400) | +0.11 / -0.03 / -0.03 |

With five stocks the estimated spread of the true betas is zero (every estimate pooled to the
mean) in 45 of 400 universes, and the Bayesian 90% intervals cover 74%, because they treat the
estimated prior as exact. The loss against the true prior comes mostly from learning the amount
of shrinkage, not the centre: in the code review's decomposition, empirical Bayes scored 0.81
with the true centre and an estimated spread, and 0.68 with the true spread and an estimated
centre.

### 5.6 Real stocks: Blume's test

Weekly adjusted closes from Alpha Vantage for 20 large US stocks and SPY, November 1999 to
October 2026: ABT, AMGN, CL, D, DUK, ED, GIS, JNJ, KMB, KO, MRK, NEE, PEP, PFE, PG, SO, T, UNH,
VZ and WMT. The free API quota stopped the download there, so the sample is utilities, consumer
staples, health care and telecoms that survived to 2026, with an average beta of 0.52. Betas are
estimated in consecutive, non-overlapping windows ending at the latest week; each estimator from
one window predicts the next window's least-squares beta. The change in squared error is relative
to least squares, with a 90% interval from a moving-block bootstrap (Künsch 1989) over consecutive
pairs of windows, in blocks of three, because consecutive pairs share a window.
`docs/results/intro-betas/window_betas.csv` holds every window's beta and standard error; the
prices are not in the repository (`scripts/beta_window_betas.py` rebuilds the table from any price
file). One weekly return spans eleven days, across the market's closure after 11 September 2001.

| Windows | Pairs of windows (stock-window pairs) | Least squares, mean squared error | Bloomberg's rule | Bayes | Every beta set to the average | Bayes better in |
|---|---|---|---|---|---|---|
| 1 year | 26 (520) | 0.116 | +4% (-15% to +23%) | -19% (-24% to -12%) | -11% (-26% to +11%) | 23 of 26 |
| 2 years | 12 (240) | 0.079 | +19% (-8% to +37%) | -13% (-17% to -2%) | -10% (-28% to +25%) | 9 of 12 |

How much the result depends on choices (Bayes' change in error, with its 90% interval):

| Check | 1-year windows | 2-year windows |
|---|---|---|
| As reported | -19% (-24% to -12%) | -13% (-17% to -2%) |
| Windows ending a quarter, a half and three quarters of a window earlier | -24%, -20%, -21%; every interval below zero | -13%, -12%, -14%; one interval reaches +1% |
| Blocks of 1, 2 or 4 pairs | intervals from -25% to -12% | intervals from -19% to -2% |
| Non-overlapping pairs only | -19% (13 pairs) | -16% (6 pairs) |
| Windows anchored at the first week (the code review's run) | -20% (-26% to -14%) | -12% (-24% to +1%) |

The one-year result holds under every check. The two-year gain is 12–16% under every check, but
with 12 pairs its interval touches zero for some alignments, and the cross-sectional average does
about as well; the lesson calls it smaller and less certain. Bloomberg's rule's intervals include
zero everywhere.

Drift. Within each pair of windows, net of the pair's average, next-window betas regress on
current ones with a slope of 0.41 (one-year windows) and 0.43 (two-year windows); estimation noise
alone would give 0.61 and 0.75. The mean squared change in a beta between windows is 1.9 and 2.9
times the sum of the two squared standard errors (1 for constant betas). With
heteroskedasticity-robust (White) standard errors the ratios are 1.7 and 2.4, so understated
standard errors explain only a small part: the betas drift.

## 6. What the intro claims, and what it does not

It claims what §3 lists, at the conditions stated there. It does not claim:

- **That frequentist inference fails.** The least-squares interval is exact for each stock fixed
  in advance, and James–Stein shrinkage is itself frequentist.
- **That Bayes improves every stock.** It improves the total; the lesson's single simulated stock
  is one where the raw estimate was closer to the truth.
- **That Bayes beats a fixed rule in every population.** It does when the prior describes the
  stocks; §5.4 shows populations where Bloomberg's rule wins and one where least squares wins.
- **That Stein's theorem covers the lesson's estimator.** It covers equal noise; ours is unequal.
- **That the real-data result holds for every universe.** Our sample is 20 defensive stocks;
  Blume (1975) and Levi and Welch (2017) cover far larger samples.
- **That the market model is right.** Slot 6 is about noticing when the model or the prior does
  not fit.

## References

Blume (1971), *Journal of Finance* 26(1); Blume (1975), *Journal of Finance* 30(3); Bock (1975),
*Annals of Statistics* 3(1), 209–218; Efron and Morris (1973), *Journal of the American Statistical
Association* 68(341); Efron and Morris (1977), *Scientific American* 236(5); Jacquier, Polson and
Rossi (1994), *Journal of Business and Economic Statistics* 12(4); James and Stein (1961),
*Proceedings of the Fourth Berkeley Symposium* 1; Künsch (1989), *Annals of Statistics* 17(3);
Lally (1998), *Financial Review* 33(3); Ledoit and Wolf (2004), *Journal of Multivariate Analysis*
88(2); Levi and Welch (2017), *Journal of Financial and Quantitative Analysis* 52(2); Lindley
(1962), discussion of Stein, *Journal of the Royal Statistical Society B* 24(2); Menchero, Orr and
Wang (2011), *The Barra US Equity Model (USE4): Methodology Notes*, MSCI; Robinson (1991),
*Statistical Science* 6(1), 15–32; Rosenberg and Guy (1976), *Financial Analysts Journal* 32(3),
60–72 and 32(4), 62–70; Stein (1956), *Proceedings of the Third Berkeley Symposium* 1; Vasicek
(1973), *Journal of Finance* 28, 1233–1239.

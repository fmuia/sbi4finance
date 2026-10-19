# Slot 6 seed: the crash-day example (design and evidence)

Prepared 19 October 2026. This example was designed as the slot-1 intro and reassessed before its
pull request. The team then chose a simpler intro (betas across many stocks), and this material
became the seed of slot 6, on priors and misspecification. It fits there because its results are
about the prior: in this problem the prior decides much of the answer, a flat prior changes the
parameter intervals completely, and an informative prior hurts at truths it finds unlikely. The
sections below keep the original intro framing; slot 6 will reframe them.

The notebook is [`notebooks/tutorial_6_crash_risk.py`](../notebooks/tutorial_6_crash_risk.py);
the study is `scripts/crash_risk_study.py`, with its results in
[`docs/results/crash-risk/`](results/crash-risk/).

## 1. What the intro has to do

- **Add one layer only: Bayesian reasoning.** No network and no sampler: every posterior in the
  lesson is exact, computed on a grid.
- **Answer "why Bayesian?" for a risk audience.** That audience includes econometricians, most
  of them frequentists. The lesson has to survive their best objections, not the weakest ones.
- **Hand over to the method slots with the cost problem.** The exact answer is cheap here and
  unaffordable in slot 5's model.

## 2. Is "how many jumps?" a good question?

The first proposal (in the earlier plan) asked how often jumps happen and showed that Wald
intervals for the jump frequency miscover. Reassessed, that framing has three weaknesses:

1. **A jump frequency is model-internal.** A jump exists only inside a model. Give the same
   returns stochastic volatility and some of those "jumps" become volatility moves; slot 5 shows
   exactly this in April 2025. No risk report quotes a jump frequency.
2. **It invites "you asked an unidentified question".** The data pin down how fat the tail is
   much better than how that fatness splits into frequency and size. A critic can say any method
   must struggle with such a question, and that the failure belongs to the question, not to
   frequentist inference.
3. **The Wald interval is the weakest frequentist answer.** A competent frequentist would use the
   profile likelihood, which is honest there. Showing Wald failing invites "you picked a straw
   man".

The split between few large jumps and many small ones still matters, because it is
decision-relevant: both stories pass a likelihood-ratio test on the past and disagree about the
future. So the reassessed intro asks a **decision question** and uses the jump split as the
**mechanism**:

> **How likely is a day with a loss of 7% or more in the next year?**

This question has four strengths:

- It is observable and model-free as a question: any method, with or without a model, can
  answer it, and its answer can be scored.
- It is what a risk manager actually needs: a crash-day probability feeds stress scenarios,
  limits and capital.
- It is tail-driven: it depends almost entirely on the parameters the data identify worst.
- It matches the hook: VOO's last 1,000 days contain no such day, and March 2020 had three.

The methods are then judged on the decision quantity, with criteria that need no Bayesian
premise: calibration of probability forecasts, proper scoring rules and interval coverage
against the known truth. The jump-frequency intervals stay as a secondary result: they show the
mechanism, and they are the seed for slot 5.

Other questions we considered for the intro:

| Question | Verdict |
|---|---|
| How many jumps a year? | The mechanism, not the headline (above). |
| What is the 1-day 99% VaR or ES? | Driven by the body of the distribution, which 1,000 days pin down: the plug-in is nearly right, so the lesson would be "Bayes changes little". Slot 4 shows it: with 1,000 days, the plug-in 10-day ES falls short of the predictive one by a negligible amount. |
| How precisely do four years pin down the expected return? | A classic (Merton 1980), but a regular problem: the frequentist interval is fine, so it shows uncertainty, not a failure. Kept as slot 4's optional poll. |
| Is there a jump component at all? | A test at the boundary of the parameter space (Chernoff 1954), not a decision. |
| How likely is a 20% drawdown next year? | Driven by the volatility and the drift, not by the tail; path-dependent, so harder to explain in the first slot. |
| **How likely is a 7% loss day next year?** | **Chosen**: a decision, scorable, and in the regime where the parameters the data identify worst decide the answer. The study also reports a 5% loss, to show the effect shrinking as the event becomes less extreme. |

## 3. The frequentist toolbox, exhaustively

Each row says what the method is, whether the study runs it, and why or why not. "Run" means
`sbi4finance.crash_risk.analyse` computes it on every simulated history.

### For the jump frequency (the parameter)

| Method | Status | Notes |
|---|---|---|
| Maximum likelihood with a Wald interval (Hessian, log scale) | run | The textbook output of any optimizer. |
| Profile likelihood-ratio interval (χ²₁) | run | The honest standard for weak identification; computed on the same grid as the posterior. |
| Boundary-corrected likelihood ratio (Chernoff 1954; Self and Liang 1987) | discussed | Applies to testing at a known boundary; near one, the χ²₁ interval above is already the conservative choice. |
| Nonparametric bootstrap, percentile interval | run | 100 resampled histories, each refitted by maximum likelihood. |
| Parametric bootstrap, percentile interval | run | 100 histories simulated from the fit, each refitted. Inconsistent at a boundary for the same reason as the nonparametric one (Andrews 2000). |
| m-out-of-n bootstrap, subsampling (Andrews 2000; Politis, Romano and Wolf 1999) | discussed | Valid at a boundary asymptotically, but needs a tuning choice and many effective observations; ten jump days give it nothing to work with. |
| Weak-identification-robust inference (Andrews and Cheng 2012; Andrews and Mikusheva 2016) | discussed | Valid by construction, so its sets must be wide or unbounded when identification is weak (Dufour 1997, below). The profile region is its practical stand-in. |
| Flag jump days with a threshold, then count them (Mancini 2009 in spirit) | run | Three robust sds, then an exact Poisson interval for the count. The intraday jump tests (Lee and Mykland 2008; Barndorff-Nielsen and Shephard 2006) need high-frequency data the question does not have. |
| Method of moments, GMM, EMM (Press 1967; Gallant and Tauchen 1996; Andersen, Benzoni and Lund 2002) | discussed | The higher moments that identify jumps are dominated by the few jump days, so their sampling variance is huge (design document, D.6). |
| Penalized likelihood: the posterior mode with a Laplace interval | run | The frequentist way to add prior information. Same prior as the Bayesian answer; only the Gaussian approximation differs. |
| Confidence distributions and confidence curves (Schweder and Hjort 2016; Xie and Singh 2013) | discussed | Built from the profile likelihood here, so they return the profile interval above. |
| Bayes with a flat prior on the grid's box | run | Not a frequentist method: the benchmark that separates what the prior adds from what averaging over the likelihood adds. |

### For the crash probability (the decision)

| Method | Status | Notes |
|---|---|---|
| Plug-in at the maximum-likelihood estimate | run | Standard practice. |
| Plug-in at the penalized estimate (posterior mode) | run | Adds the prior to a point estimate. |
| Bootstrap-averaged (bagged) probability and percentile interval | run | The frequentist way to propagate parameter uncertainty (Christoffersen and Gonçalves 2005). |
| Parametric bootstrap predictive and percentile interval | run | The average plug-in over histories simulated from the fit: the bootstrap predictive distribution of Harris (1989). |
| Delta-method interval, logit scale, at the MLE (Wald) and at the mode (Laplace) | run | |
| Profile likelihood-ratio interval for the probability | run | The range of the probability over the 90% likelihood region. |
| Calibrated predictive distributions (Barndorff-Nielsen and Cox 1996; Lawless and Fredette 2005) | discussed | They correct the plug-in to higher order in 1/n in regular models. This one is not regular here (a boundary, weak identification); the parametric bootstrap predictive is the practical version, and it is run. |
| Empirical frequency | run | What historical simulation says. |
| Peaks over threshold: a generalized Pareto tail on the largest 5% of losses | run | The standard model-free tool for exactly this question (McNeil and Frey 2000). The largest 2.5% and 10% are run too, to check the choice of threshold. |
| Profile interval for the peaks-over-threshold tail | discussed | Not needed for the point comparison. |
| The other tail family's plug-in | run | What a practitioner who picked the other model reports: a Student-t fitted to jump-model histories, and the jump model fitted to Student-t histories. |
| A longer history | discussed | The best remedy where the data exist. Desks use one to four years because volatility and correlations change, and slot 5 uses 1,000 days for the same reason. Every method here sees the same 1,000 days. |
| Pooling across names (empirical Bayes; Efron 2010) | discussed | The frequentist route to a prior: estimate it from the cross-section of names. It is the natural next step for a desk with 2,000 names, and supports using a prior rather than arguing against it. |

### The theorem behind the results

When a parameter is unidentified on part of the parameter space, any confidence set with level
1 − α must be unbounded with probability close to 1 − α there. A method that always returns
bounded intervals, such as Wald, then has a confidence level of zero (Dufour 1997; also Gleser
and Hwang 1987). The jump model has exactly this structure:

- as the jump frequency goes to zero, the jump size and mean drop out of the likelihood;
- as the jump size goes to zero with a zero mean, the frequency drops out.

No frequentist method can deliver intervals that are both informative and valid at every truth
here. That is a property of the problem, not of a choice of method. The Bayesian answer does
not escape it either. It changes the guarantee: calibration on average over a stated prior,
instead of coverage at every truth. It also adds information: the prior, in risk-manager units.

## 4. "Frequentist fails because you picked the wrong model"

There are four answers; the study backs the first three:

1. **The model is right by construction.** Every history is simulated from the model being
   fitted, so misspecification cannot be the cause. Misspecification is slot 6's subject, and
   there it hits every method, Bayesian included.
2. **A well-identified tail model is the control, and there the plug-in is fine.** The study
   repeats everything for Student-t returns fitted with the Student-t model, whose two
   parameters 1,000 days identify well. Every likelihood-based answer is then about as good as
   Bayes (§5.6). So the failure in the jump model comes from weak identification: the data
   cannot split the tail into jump frequency and jump size, and that split decides the answer.
   It does not come from fat tails, from the scoring rule, or from a prior that favors Bayes.
3. **Switching to the well-identified model does not help when it is the wrong one.** A
   Student-t fitted to jump-model histories is worse than every jump-model method, and the jump
   model fitted to Student-t histories fails in the same way (§5.1, §5.6). The model-free
   answers, the empirical frequency and peaks over threshold on the largest 2.5%, 5% or 10% of
   losses, are worse still.
4. **This is not an exotic model.** It is Merton's (1976) jump model, the jump part of Bates's
   (1996) model and of slot 5. The difficulty of estimating jumps from daily returns is
   documented well beyond this tutorial (Aït-Sahalia 2004; Eraker, Johannes and Polson 2003).

The objection has a fair core, which the intro keeps: the answer depends on the model, and
choosing between two tail models that fit the same four years is itself a source of uncertainty.
That is slot 6's subject.

## 5. Results

The study ran `scripts/crash_risk_study.py` with seed 2026 on 4 cores: 1,200 histories of
1,000 days, every method on each, with no failures. Unless stated otherwise, the numbers are for
a loss of 7% or more, intervals are 90% for the crash probability and 95% for the jump frequency,
and "±" is two standard errors (binomial for coverage, paired across histories for differences
in excess log loss). A jump-model history took a median of 16 CPU-seconds, most of it the grid
and the 200 refits of the two bootstraps; the whole study took 3.3 CPU-hours, 53 minutes on 4
cores.

### 5.1 Truths drawn from the prior: the crash probability

300 histories, each from its own truth drawn from slot 5's prior. The true probability of a
crash day next year has a median of 20% (12% of truths are below 1%, 25% below 5%): the prior
covers calm names and volatile ones.

| Method | Excess log loss | Minus Bayes (±2 SE, paired) | Median excess loss | Understated 10× | Overstated 10× |
|---|---|---|---|---|---|
| Bayes, predictive | 0.039 |  | 0.019 | 0% | 9% |
| Bayes, flat prior on the grid's box | 0.048 | +0.008 ± 0.005 | 0.014 | 2% | 7% |
| Maximum likelihood, plug-in | 0.099 | +0.060 ± 0.028 | 0.020 | 18% | 1% |
| Posterior mode (penalized likelihood), plug-in | 0.047 | +0.008 ± 0.006 | 0.013 | 4% | 6% |
| Bootstrap average | 0.084 | +0.045 ± 0.023 | 0.019 | 12% | 3% |
| Parametric bootstrap average (Harris) | 0.073 | +0.034 ± 0.018 | 0.015 | 9% | 2% |
| Peaks over threshold, largest 5% | 0.152 | +0.113 ± 0.039 | 0.025 | 28% | 1% |
| Peaks over threshold, largest 2.5% | 0.156 | +0.116 ± 0.040 | 0.032 | 26% | 1% |
| Peaks over threshold, largest 10% | 0.152 | +0.113 ± 0.037 | 0.033 | 28% | 0% |
| A Student-t, plug-in | 0.207 | +0.168 ± 0.038 | 0.055 | 25% | 3% |
| Historical frequency | 0.340 | +0.300 ± 0.080 | 0.066 | 45% | 1% |

- **On a typical history the likelihood-based answers are about equally good**: their median
  excess losses are 0.013–0.020. The model-free answers are worse even there.
- **They differ in how badly they fail.** The plug-in understates the probability by a factor
  of 10 or more in 18% of histories, and its worst miss called a 31% event a one-in-a-million
  one. Both bootstraps, which average refits around the same fit, still do so in about 10%.
  Bayes did so once in 300 histories (a 13.5× miss), and pays by overstating by 10× or more in
  9%, on quiet truths the prior finds unusual.
- **Either ingredient of the Bayesian answer recovers most of the gap.** Averaging over the
  likelihood with a flat prior (+0.008 ± 0.005) and adding the prior to a single point (the
  posterior mode, +0.008 ± 0.006) each close about 85% of the plug-in's excess loss over Bayes.
  Doing both is best, by about three standard errors.
- **Peaks over threshold does not depend on its tail fraction**: the largest 2.5%, 5% and 10%
  of losses give the same picture.

### 5.2 Intervals for the crash probability

| 90% interval | Covers the truth (±2 SE) | Median width (high / low) | Wider than 100× |
|---|---|---|---|
| Bayes, credible | 88% ± 4 | 8.4× | 24% |
| Bayes, flat prior | 85% ± 4 | 9.8× | 28% |
| Wald (delta method at the MLE) | 71% ± 5; 78% of the 91% where it exists | 4.8× | 16% |
| Laplace (delta method at the mode) | 88% ± 4 | 9.2× | 25% |
| Profile likelihood | 86% ± 4 | 8.2× | 28% |
| Bootstrap percentile | 74% ± 5 | 7.8× | 28% |
| Parametric bootstrap percentile | 79% ± 5 | 13.4× | 33% |

The Bayesian, Laplace and profile intervals cover close to 90% at similar widths. The bootstrap
and Wald intervals undercover. Two conventions matter here: Wald counts as a miss where the
Hessian is not positive definite and no interval exists (9% of histories), and the bootstraps use
100 resamples, which costs a percentile interval about two points of coverage, far less than
the shortfall here.

### 5.3 What a low forecast means

| Method | Histories with a forecast below 0.5% | Their average forecast | Their average true probability |
|---|---|---|---|
| Bayes, predictive | 1 | 0.18% | 0.00082% |
| Bayes, flat prior on the grid's box | 14 | 0.27% | 2.4% |
| Maximum likelihood, plug-in | 66 | 0.1% | 4.5% |
| Posterior mode (penalized likelihood), plug-in | 18 | 0.18% | 2.1% |
| Bootstrap average | 53 | 0.11% | 3.6% |
| Parametric bootstrap average (Harris) | 48 | 0.13% | 5% |
| Peaks over threshold, largest 5% | 83 | 0.043% | 5.2% |
| Peaks over threshold, largest 2.5% | 82 | 0.032% | 4.9% |
| Peaks over threshold, largest 10% | 87 | 0.055% | 5.3% |
| A Student-t, plug-in | 71 | 0.14% | 5.9% |
| Historical frequency | 135 | 0% | 7.8% |

When a single-fit method says "below 0.5%", the truth averages 4–8%: these are exactly the
reassuring numbers a risk report would act on. Across all bins, the average true probability
(and the number of histories) for each forecast bin:

| Forecast bin | Bayes | Posterior mode | Maximum likelihood | Bootstrap | Peaks over threshold |
|---|---|---|---|---|---|
| 0–0.5% | 0.00082% (1) | 2.1% (18) | 4.5% (66) | 3.6% (53) | 5.2% (83) |
| 0.5–2% | 2% (12) | 6% (32) | 7.4% (24) | 7.5% (34) | 13% (19) |
| 2–5% | 4.1% (39) | 5.5% (36) | 7.1% (17) | 7.4% (17) | 7.6% (17) |
| 5–10% | 7.6% (52) | 9.1% (32) | 12% (14) | 11% (17) | 15% (18) |
| 10–20% | 15% (49) | 18% (39) | 19% (34) | 19% (33) | 19% (28) |
| 20–100% | 58% (147) | 58% (143) | 58% (145) | 58% (146) | 60% (135) |

The Bayesian forecasts track the truth in every bin; the single-fit ones are too low wherever
they are low.

### 5.4 The jump frequency

| 95% interval for jumps per year | Covers the truth | Median width (high / low) |
|---|---|---|
| Bayes, credible | 94% | 10× |
| Bayes, flat prior | 92% | 145× |
| Wald (delta method at the MLE) | 78%; 85% of the 91% where it exists | 33× |
| Laplace (delta method at the mode) | 93% | 11× |
| Profile likelihood | 93% | 145× |
| Bootstrap percentile | 90% | 132× |
| Parametric bootstrap percentile | 96% | 257× |
| Flag days beyond 3 sd, then count | 60% | 6× |

Without a prior, valid intervals are very wide: the profile likelihood, a flat prior and the
bootstraps all reach a factor of 130 or more at the median, and the profile interval reaches the
edge of the grid in nearly half the histories, as Dufour's theorem predicts. Slot 5's prior
narrows the interval to a factor of 10 at the nominal coverage: for the parameter, the
information is the prior's. Flagging and counting jump days is narrow and wrong.

### 5.5 Truths fixed at four points of the ridge

The frequentist's criterion is the risk at a fixed truth. The study fixes four truths, 100
histories each, all at a 16% volatility: the prior's typical truth, the two ends of the ridge, and
almost no jumps. Excess log loss for each method, then two summaries:

| Truth | True probability | Bayes | Posterior mode | Maximum likelihood | Bootstrap average | Parametric bootstrap | Peaks over threshold | Understated 10×, Bayes / MLE | Coverage, Bayes / profile / bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 a year of −1% ± 3% | 6.8% | 0.027 | 0.041 | 0.138 | 0.118 | 0.095 | 0.122 | 0% / 29% | 97% / 81% / 59% |
| 0.5 a year of −3% ± 6% | 12% | 0.061 | 0.175 | 0.502 | 0.469 | 0.409 | 0.477 | 0% / 52% | 80% / 78% / 43% |
| 12 a year of −0.5% ± 1.5% | 0.18% | 0.049 | 0.022 | 0.009 | 0.009 | 0.009 | 0.014 | 0% / 38% | 72% / 90% / 57% |
| 0.1 a year of −1% ± 3% | 0.28% | 0.018 | 0.001 | 0.014 | 0.013 | 0.011 | 0.012 | 0% / 97% | 99% / 99% / 3% |

- **Where the prior is roughly right, Bayes wins by far.** At the typical truth and at the
  rare-and-large end, its excess loss is a fifth to an eighth of the plug-in's, and the plug-in
  understates by 10× or more in 29% and 52% of histories.
- **At the two quiet truths (a true probability of about 0.2%), Bayes is the worst.** The prior
  expects a few jumps a year of a few percent, so the median Bayesian forecast is 4.9% at the
  frequent-and-small truth and 2.4% at almost no jumps: 27 and 9 times too high. The plug-in wins
  there on excess log loss, although it understates by 10× or more in 38% and 97% of histories:
  in log loss, missing a tiny probability costs little and overstating it costs more.
- **Calibration is a promise on average, not at every truth.** The Bayesian interval covers
  between 72% and 99% across these truths. The profile interval stays closer to its nominal
  level (78–99%), as frequentist theory promises; the bootstrap's does not (3–59%).

### 5.6 Student-t histories

300 histories of Student-t returns, each from its own truth drawn from that family's prior
(ν log-normal around 5, 90% between 2.6 and 15; the same volatility prior), every method refitted
with the Student-t model. Here "the other family" is the jump model.

| Method | Excess log loss | Minus Bayes (±2 SE, paired) | Median excess loss | Understated 10× | Overstated 10× |
|---|---|---|---|---|---|
| Bayes, predictive | 0.012 |  | 0.004 | 0% | 1% |
| Bayes, flat prior on the grid's box | 0.014 | +0.001 ± 0.001 | 0.005 | 1% | 1% |
| Maximum likelihood, plug-in | 0.014 | +0.002 ± 0.001 | 0.005 | 3% | 0% |
| Posterior mode (penalized likelihood), plug-in | 0.012 | +0.000 ± 0.000 | 0.004 | 0% | 1% |
| Bootstrap average | 0.013 | +0.001 ± 0.001 | 0.004 | 1% | 1% |
| Parametric bootstrap average (Harris) | 0.014 | +0.001 ± 0.001 | 0.005 | 1% | 1% |
| Peaks over threshold, largest 5% | 0.109 | +0.097 ± 0.036 | 0.020 | 21% | 2% |
| Peaks over threshold, largest 2.5% | 0.161 | +0.149 ± 0.047 | 0.025 | 25% | 2% |
| Peaks over threshold, largest 10% | 0.075 | +0.063 ± 0.023 | 0.017 | 18% | 1% |
| The jump model, plug-in | 0.081 | +0.068 ± 0.017 | 0.025 | 24% | 1% |
| Historical frequency | 0.327 | +0.315 ± 0.074 | 0.050 | 46% | 1% |

| 90% interval | Covers the truth (±2 SE) | Median width (high / low) | Wider than 100× |
|---|---|---|---|
| Bayes, credible | 88% ± 4 | 3.4× | 4% |
| Bayes, flat prior | 87% ± 4 | 3.6× | 8% |
| Wald (delta method at the MLE) | 90% ± 4 | 3.5× | 7% |
| Laplace (delta method at the mode) | 88% ± 4 (one history has no interval) | 3.3× | 4% |
| Profile likelihood | 83% ± 4 | 3.2× | 6% |
| Bootstrap percentile | 86% ± 4 | 3.7× | 7% |
| Parametric bootstrap percentile | 86% ± 4 | 3.6× | 8% |

Every likelihood-based answer is within 0.002 of Bayes, and the plug-in understates by 10× or
more in only 3% of histories. The model-free answers fail as before, and so does the jump model
fitted to these histories. At two fixed truths (ν = 4 and ν = 8, 100 histories each):

| Truth | True probability | Bayes | Posterior mode | Maximum likelihood | Bootstrap average | Parametric bootstrap | Peaks over threshold | Understated 10×, Bayes / MLE | Coverage, Bayes / profile / bootstrap |
|---|---|---|---|---|---|---|---|---|---|
| ν = 4 | 7.2% | 0.010 | 0.011 | 0.012 | 0.011 | 0.010 | 0.125 | 0% / 0% | 89% / 84% / 83% |
| ν = 8 | 0.52% | 0.004 | 0.003 | 0.004 | 0.004 | 0.004 | 0.020 | 0% / 14% | 89% / 78% / 82% |

### 5.7 A 5% loss

| Method | Excess log loss | Minus Bayes (±2 SE, paired) | Median excess loss | Understated 10× | Overstated 10× |
|---|---|---|---|---|---|
| Bayes, predictive | 0.031 |  | 0.010 | 0% | 1% |
| Bayes, flat prior on the grid's box | 0.043 | +0.012 ± 0.007 | 0.008 | 1% | 0% |
| Posterior mode (penalized likelihood), plug-in | 0.035 | +0.004 ± 0.003 | 0.008 | 0% | 1% |
| Parametric bootstrap average (Harris) | 0.054 | +0.023 ± 0.012 | 0.008 | 4% | 1% |
| Bootstrap average | 0.058 | +0.027 ± 0.014 | 0.008 | 5% | 1% |
| Maximum likelihood, plug-in | 0.069 | +0.037 ± 0.017 | 0.010 | 7% | 1% |
| Peaks over threshold, largest 5% | 0.145 | +0.114 ± 0.055 | 0.015 | 10% | 0% |
| Historical frequency | 0.212 | +0.181 ± 0.067 | 0.020 | 18% | 0% |

| 90% interval | Covers the truth (±2 SE) | Median width (high / low) | Wider than 100× |
|---|---|---|---|
| Bayes, credible | 88% ± 4 | 1.9× | 6% |
| Bayes, flat prior | 83% ± 4 | 1.9× | 7% |
| Wald (delta method at the MLE) | 71% ± 5; 78% of the 91% where it exists | 1.9× | 4% |
| Laplace (delta method at the mode) | 85% ± 4 | 2.0× | 6% |
| Profile likelihood | 80% ± 5 | 1.8× | 8% |
| Bootstrap percentile | 81% ± 4 | 2.0× | 12% |
| Parametric bootstrap percentile | 82% ± 4 | 2.1× | 12% |

The ordering is the same and the gaps are smaller: the problem lives in the tail. Two caveats
on the interval coverage at this threshold:

- **54 of the 300 truths make a 5% loss day nearly certain** (above 99%). There, intervals are
  pinned against 1 and coverage turns on rounding, which costs every method 10–17 of its
  misses (1–2 at a 7% loss).
- **The profile interval covers 80%.** It is the range over the grid's points, which makes it
  somewhat too narrow. We rechecked four of its misses on a grid three times finer in the
  variance and twice in the mean jump. Two were saturated cases, one of which the finer grid
  covered; the other two intervals widened, by a fifth to a third, without reaching the truth.
  The grid explains part of the shortfall, the χ² approximation the rest.

## 6. What the intro claims, and what it does not

Every claim below is for a 7% loss and truths drawn from the prior, unless stated otherwise.

| The notebook says | Evidence |
|---|---|
| A point estimate hides the uncertainty that decides the answer. | On the example history the 95% likelihood region spans crash probabilities from 0.02% to 29%; on VOO, from 0.47% to 28% (notebook, sections 4 and "Back to VOO"). |
| The answers built on one fit fail badly and often. | Understated by 10× or more: maximum likelihood 18%, bootstrap 12%, parametric bootstrap 9% (§5.1). When they forecast below 0.5%, the truth averages 4–5% (§5.3). |
| The model-free answers are worse, not better. | Excess log loss 0.15 for peaks over threshold, whatever its tail fraction, and 0.34 for the historical frequency, against 0.10 for the plug-in (§5.1). |
| Averaging over what the data allow fixes most of it, even with a flat prior. | Flat-prior Bayes: 0.048 against 0.099 for the plug-in and 0.039 for Bayes (§5.1). |
| So does adding the prior to a single point. | Posterior mode: 0.047 (§5.1). |
| The bootstrap does not fix it. | Excess log loss 0.084 and 0.073; 90% intervals cover 74% and 79% (§5.1, §5.2). |
| The Bayesian answer is calibrated, and best on average over the prior. | Lowest excess log loss, by about three standard errors over the next best; one 10× understatement in 300; 90% intervals cover 88% ± 4 (§5.1, §5.2). |
| The profile likelihood is the honest frequentist interval. | 86% ± 4 coverage at the Bayesian interval's width (§5.2). |
| For the jump frequency, valid intervals without a prior are very wide. | Profile, flat prior and bootstraps: a factor of 130–260 at the median; slot 5's prior: a factor of 10 at 94% coverage (§5.4). |
| The failure comes from weak identification, not from fat tails. | With Student-t returns and the Student-t model, every likelihood-based answer is within 0.002 of Bayes and the plug-in understates 10× in 3% of histories (§5.6). |
| Picking the other model does not rescue the plug-in. | A Student-t fitted to jump histories: 0.207, understated 10× in 25%; the jump model fitted to Student-t histories: 0.081 and 24% (§5.1, §5.6). |
| Bayes is not best at every truth. | At the two quiet fixed truths it overstates 9–27× and has the highest excess loss (§5.5). |

What the intro does not claim:

- **That frequentist inference fails.** The profile likelihood is honest here, and the penalized
  fit is nearly as good as Bayes on average. What fails is the single-fit route that dominates
  practice: the plug-in, its bootstrap, and extrapolation from the few largest losses.
- **That Bayes wins at every truth.** Its guarantee is calibration on average over the prior. It
  overstates by 10× or more in 9% of histories, and at truths the prior finds unlikely it can
  lose (§5.5).
- **That the prior is free.** A flat prior on a box is a prior too, and it changes the
  jump-frequency interval completely (§5.4). Slot 6 is about choosing and checking priors.
- **A risk figure for VOO.** The model has no volatility clustering; the VOO numbers illustrate
  the method.
- **That the jump model describes real returns.** Slot 5 adds stochastic volatility; slot 6 asks
  how to notice misspecification.

## References

Aït-Sahalia (2004), *Journal of Financial Economics* 74(3); Andersen, Benzoni and Lund (2002),
*Journal of Finance* 57(3); Andrews (2000), *Econometrica* 68(2); Andrews and Cheng (2012),
*Econometrica* 80(5); Andrews and Mikusheva (2016), *Econometrica* 84(4); Barndorff-Nielsen and
Cox (1996), *Bernoulli* 2(4); Barndorff-Nielsen and Shephard (2006), *Journal of Financial
Econometrics* 4(1); Bates (1996), *Review of Financial Studies* 9(1); Chernoff (1954), *Annals
of Mathematical Statistics* 25(3); Christoffersen and Gonçalves (2005), *Journal of Risk* 7(3);
Dufour (1997), *Econometrica* 65(6); Efron (2010), *Large-Scale Inference*, Cambridge University
Press; Eraker, Johannes and Polson (2003), *Journal of Finance* 58(3); Gallant and Tauchen
(1996), *Econometric Theory* 12(4); Gleser and Hwang (1987), *Annals of Statistics* 15(4);
Harris (1989), *Biometrika* 76(4); Lawless and Fredette (2005), *Biometrika* 92(3); Lee and
Mykland (2008), *Review of Financial Studies* 21(6); Mancini (2009), *Scandinavian Journal of
Statistics* 36(2); McNeil and Frey (2000), *Journal of Empirical Finance* 7(3–4); Merton (1976),
*Journal of Financial Economics* 3(1–2); Merton (1980), *Journal of Financial Economics* 8(4);
Politis, Romano and Wolf (1999), *Subsampling*, Springer; Press (1967), *Journal of Business*
40(3); Schweder and Hjort (2016), *Confidence, Likelihood, Probability*, Cambridge University
Press; Self and Liang (1987), *Journal of the American Statistical Association* 82(398); Xie and
Singh (2013), *International Statistical Review* 81(1).

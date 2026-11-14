# ICAIF '26 tutorial: the LinkedIn series

Prepared 19 October 2026 for the four organizers, and revised the same day after the intro was
reassessed and after a review of every number and claim. Nothing here is scheduled or posted.
Slot titles marked *assumed* need confirming.

## Summary

- **Seven core posts**: P0 on Wednesday 21 Oct, then Tuesdays and Thursdays to 12 Nov. One
  announces the tutorial; each of the other six opens with one slot's question and leaves it
  unanswered.
- **Two optional posts**: logistics on 13 Nov, and the answers after the conference, which keeps
  the promise made by leaving the questions open.
- **Full drafts** are in §5, each with its sources and with what we keep for the session.
- **Decisions needed** (§8): the titles and leads of slots 2–3, our day, time and room,
  whether the repository, private today, goes public before the announcement, and who builds
  slot 6, which P6 commits us to.

## 1. Context

**Conference.** ACM ICAIF '26, 14–17 November 2026, Bocconi University, Milan. Workshops and
tutorials run on Saturday 14 and Sunday 15 November. Our day, time and room are not yet in the
published listings.

**Tutorial.** *Amortized Bayesian Calibration of Volatility Models*, with Stefan Radev, Jerry
Huang, Alexander Fengler and Francesco Muia. Slots 4–6 come from the draft notebooks; slots 2–3
are reconstructed from the proposal.

| Slot | Working title | Status |
|---|---|---|
| 1 | Intro: one beta, then five hundred (Bayes' rule across stocks) | draft [#41](https://github.com/fmuia/sbi4finance/pull/41), with its design and evidence |
| 2 | Amortized inference: normalizing flows and flow matching, summary networks, trade-offs | *assumed* |
| 3 | Validation in a Bayesian workflow, or "why and where it matters for finance" | *assumed* |
| 4 | Basic worked-out example in BayesFlow: constant volatility, exact posterior | draft [#37](https://github.com/fmuia/sbi4finance/pull/37) |
| 5 | Advanced: stochastic volatility with jumps, amortized and audited | draft [#38](https://github.com/fmuia/sbi4finance/pull/38) |
| 6 | Priors and misspecification: when the model, not the network, is wrong | promised by #37/#38; no notebook yet, a seed in [#42](https://github.com/fmuia/sbi4finance/pull/42) (the crash-day example) |

**Slot 1 changed after review.** The first intro asked how often jumps happen; the second asked
how likely a 7% loss day is next year. Both depended on a model the data barely identify, and
the second moved to slot 6 as a seed (#42). The intro now starts from the simplest Bayesian
example in finance, a stock's beta, and adds a classic frequentist result, Stein's: when many
quantities are estimated at once, shrinking the estimates together can beat estimating each on
its own, which is part of why data vendors adjust betas toward 1. P1 follows it. The design
document in #41 (`docs/tutorial-intro-design.md`) has the reasoning, the evidence and the
conditions.

## 2. Goals

1. **Announce the tutorial**: topic, organizers, where and when, and what participants leave
   with (open notebooks, BayesFlow, a workflow they can run).
2. **One question per slot**, each left open, so that people comment, argue and come to Milan.
3. **Collect** the best answers and objections from the comments, address them in the session,
   and credit their authors with their permission. Commenting then pays off for them.
4. **After the conference, post the answers.** Every question post gets a second wave of
   activity.

## 3. How the posts should read

The first drafts followed the usual LinkedIn template: a hook line, a series marker, short
dramatic fragments, a rhetorical question answered in the next line, options behind arrows and a
call to action. That template is what makes a post read as machine-written, so the drafts below
drop it.

- **Write as the person posting**, in the first person, the way you would explain it to a
  colleague. Each author should put the draft in their own words; the drafts fix the facts and
  the question, not the voice.
- **One question per post, asked plainly near the end.** Leave the answer for the session.
- **Avoid the patterns of machine-written text** catalogued in Wikipedia's ["Signs of AI
  writing"](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing): one-line dramatic
  fragments, "not X but Y" contrasts, a question answered by the next line, lists of three for
  their own sake, arrows and decorative symbols, em dashes, and a stock call to action.
- **Mention the tutorial once**, in passing ("part 3 of our ICAIF tutorial"). Put the link in the
  first comment, because posts with links in the body reach fewer people.
- **Two or three hashtags**, at the end: #ICAIF2026 and one or two of #BayesianInference,
  #QuantFinance, #RiskManagement, #ModelRisk, #BayesFlow, #MachineLearning,
  #SimulationBasedInference.
- **One visual** where it helps, portrait 1080×1350, with large type, our plot style and alt text.
- **Length** of 600–1,200 characters (the limit is 3,000).

Two open-source tools help with the check, and neither replaces reading the post aloud:

- [humanizer](https://github.com/blader/humanizer) (MIT) is an agent skill for Claude Code and
  similar tools, built on the Wikipedia guide. It marks the guide's patterns (the count changes
  between versions) and rewrites the text without changing its claims. Give it two or three
  paragraphs of your own writing and it matches your voice.
- [dslop](https://pypi.org/project/dslop/) (Apache-2.0) is a command-line linter
  (`pip install dslop`, then `dslop post.txt`). It flags dashes, "not X, it's Y" and runs of
  short sentences. Its negation rule flags every "not", so treat that one as a hint.

Word-list linters such as [defluff](https://pypi.org/project/defluff/) found nothing in the first
drafts: their vocabulary was clean, and the giveaway was the structure.

## 4. Calendar

Post at 14:00–15:00 Milan time, which is the US East Coast morning. Europe moves to CET on 25
October and the US to EST on 1 November, so the time difference changes during the series. Each
post should come from the person presenting that slot.

| Date | Post | Proposed author | Visual |
|---|---|---|---|
| Wed 21 Oct | P0 Announcement | Francesco; the others reshare with a personal line | title card, or #37's "train once, use every day" diagram |
| Tue 27 Oct | P1 Question 1/6: how far should each beta be pulled? | Francesco | the slot-1 scatter of estimated against true betas, with the 50 highest marked (left panel) |
| Thu 29 Oct | P2 Question 2/6: can a network learn Bayes' rule? | Stefan | the "train once" half of the diagram |
| Tue 3 Nov | P3 Question 3/6: how can a posterior pass the check and still be wrong? | Jerry or Alexander | `docs/results/svj-validation/sbc.png`, uncaptioned, re-exported in portrait |
| Thu 5 Nov | P4 Question 4/6: R² = 0.99, and still off on 30 of 100 stocks | Alexander | slot 4's read-out scatter and error histogram (to export) |
| Tue 10 Nov | P5 Question 5/6: the number that drives risk is hidden | Francesco | `docs/design/images/03-example-path.png`, re-exported in portrait |
| Thu 12 Nov | P6 Question 6/6: which names does the model fit badly? | Alexander or Stefan | `docs/results/svj-rolling/VOO_rolling.png` (top panel, cropped to portrait) |
| Fri 13 Nov | P7 (optional) Logistics: see you this weekend | the slot-1 presenter | title card |
| Tue 17 or 24 Nov | P8 (optional) The answers | all, one carousel | a slide per question |

## 5. Drafts

The numbers come from committed results or the open PRs; each draft names its sources. Re-check
every number against the final notebooks before posting.

### P0: Announcement (Wed 21 Oct)

```text
Stefan Radev, Jerry Huang, Alexander Fengler and I are giving a tutorial at ACM ICAIF '26 in Milan on [day]: "Amortized Bayesian Calibration of Volatility Models".

If your risk process fits a volatility model to each name every morning, the uncertainty in each fit belongs in the VaR as well. Bayesian methods have handled that for stochastic volatility since the 1990s, but they are slow. In our runs, a basic particle MCMC sampler for a stochastic-volatility model with jumps took 12 to 52 CPU-minutes per name. For 2,000 names that is about 48 cores running all day.

The tutorial is about amortized inference: train a network once on simulated markets, and it returns a posterior for each new name in a few seconds. We build it in Python with the open-source BayesFlow library and carry the uncertainty through to VaR and expected shortfall, checking against exact answers wherever they exist. It all runs on a laptop CPU.

Over the next three weeks we'll post one question from each of the six parts and leave the answers for Milan. If you have a view, put it in the comments and we'll pick some up in the session.

#ICAIF2026 #BayesianInference #QuantFinance
```

First comment: the ICAIF '26 page and, once public, the repository. Sources:

- 12 to 52 CPU-minutes: the five particle-MCMC runs in `docs/results/svj-costs/costs.json`
  (median 35 minutes). At the median, 2,000 names take 1,152 CPU-hours a day, which is 48 cores.
  The sampler's random-walk proposal accepts 9–16% of moves, and the results summary notes that a
  tuned proposal would narrow the gap; hence "basic".
- "A few seconds": about 3 CPU-seconds per name, from the same file.
- "Since the 1990s": Jacquier, Polson and Rossi (1994).

### P1: Question 1/6, slot 1 (Tue 27 Oct)

```text
If you look up a stock's beta on Bloomberg, you get the regression estimate and an "adjusted beta", 0.67 times the estimate plus 0.33. The adjustment moves every beta a third of the way toward 1.

In a simulation of 500 stocks with known betas and two years of weekly returns, Bloomberg's default window, the 50 highest regression estimates were 0.16 too high on average. Hedging a $1m position in each of those 50 stocks with the estimated betas left the book about $8m net short the market, in each of 200 simulated universes. On 20 large defensive US stocks from 1999 to 2026, whose betas average about 0.5, pulling every beta toward 1 made no measurable difference to how well it predicted the next period's beta.

How would you decide how far to pull each estimate?

We'll answer it in the first part of our ICAIF tutorial, in Milan.

#ICAIF2026 #QuantFinance #RiskManagement
```

Sources:

- Bloomberg's adjusted beta and its default window (two years of weekly returns against the
  S&P 500): university library guides to the terminal; the weights go back to Blume (1971).
- The simulation: the slot-1 study (#41, `docs/results/intro-betas/summary.json`), 200 universes
  of 500 stocks with true betas around 1 ± 0.3 and two years of weekly returns. The 50 highest
  least-squares betas were too high by 0.16. With $1m in each, the hedged book was net short the
  market by $8.1m on average, and short in all 200 universes.
- The real stocks: Blume's test on weekly returns of 20 large US stocks from defensive sectors
  (average beta 0.52), 1999–2026. The adjustment changed the squared error of next period's
  beta by +4% with one-year windows (90% interval −15% to +23%) and +19% with two-year windows
  (−8% to +37%) against least squares, and beat it in 13 of 26 and 4 of 12 window pairs: no
  measurable difference. The sample is what a free data quota allowed, all survivors from
  defensive sectors, so it speaks to the target (1 against the stocks' own average), not to the
  amount.

Kept for the session: why the highest estimates are too high (the winner's curse), Stein's
theorem and its conditions, how Bayes' rule sets the amount of shrinkage stock by stock, why no
fixed fraction suits every stock (the amount depends on each estimate's noise relative to the
spread of true betas, and the target on the population), and when one pooled prior does worse
than Bloomberg's rule: when a stock's noise is tied to its beta.

### P2: Question 2/6, slot 2 (Thu 29 Oct)

```text
Part 2 of our ICAIF tutorial is about how amortized inference works. The training loop fits in a sentence: draw parameters from a prior, simulate a data set with them, and train a network that, given the data set, turns random noise into those parameters. For the stochastic-volatility model in part 5 we did that 2.56 million times.

Nobody writes down a likelihood. If training goes well, the network's draws for a new data set follow the Bayesian posterior, and they arrive in seconds without MCMC.

Why should training on simulated pairs give the whole posterior, rather than a good point estimate?

#ICAIF2026 #SimulationBasedInference #BayesianInference
```

Sources: 2.56 million simulated series, from #38's notebook (training section). Both slots use
flow matching, which integrates 50 steps per draw, so the post says "seconds" rather than "one
forward pass"; #37's and #38's notebooks use the looser phrase.

Kept for the session: the expected-KL argument, the amortization gap, the role of the prior
and simulation budget, and misspecification. If slot 2 leads with summary networks, ask instead
which eight numbers people would compress 1,000 daily returns into for a volatility model, and
whether the answer changes once the model has jumps and moving volatility.

### P3: Question 3/6, slot 3 (Tue 3 Nov)

This draft assumes slot 3 is validation; adjust it if not.

```text
For a model with a hidden volatility path, the exact posterior takes particle MCMC, 12 to 52 CPU-minutes per name in our runs, so in daily use there is rarely an exact answer to compare a fast approximation with. There is still a check that any correct Bayesian procedure passes, and it needs only the simulator: draw a thousand parameter sets from the prior, simulate a stock from each, compute each stock's posterior, and see where the true value falls in it. If the posteriors are right, those ranks are spread evenly.

That check is part 3 of our ICAIF tutorial, and in Milan we'll spend time on a harder point. A posterior can pass this check and still be wrong for the particular stock you care about. How can that happen?

#ICAIF2026 #BayesianInference #ModelRisk
```

Kept for the session: rank uniformity (SBC), how to read the band, and calibration "on
average" against "for this name" (slot 5's audit). If slot 3 is instead "why and where it
matters for finance", ask after how many names training pays for itself, and what would make
people distrust the cheap answer. Training once costs about 5 CPU-hours (79 minutes on 4 cores),
an exact posterior 12 to 52 CPU-minutes per name (median 35) and the trained network about 3
CPU-seconds, so the answer is about ten names at the median, 6 to 27 across our runs
(`costs.json`; results summary).

### P4: Question 4/6, slot 4 (Thu 5 Nov)

```text
We trained a small network on 256,000 simulated stocks to infer volatility from 1,000 daily returns and their log squares. A linear regression on the network's eight summary numbers recovers the sample log standard deviation with an R² of 0.99. Yet when we compare its volatility estimates with the exact posterior, which this simple model allows, they are off by more than one posterior standard deviation for about 30 of 100 test stocks.

How can a summary that explains 99% of the variance still miss that often?

This is from part 4 of our ICAIF tutorial, where the model is simple enough to check every answer.

#ICAIF2026 #BayesFlow #MachineLearning
```

Sources: `checkpoints/tutorial_gbm/` on #37's branch (256,000 series, 279 s). The R² and the
30 of 100 are computed when #37's notebook runs and are quoted in its description, but no
results file stores them: re-run the notebook and check both before posting. The post says
"and their log squares" because the network's input includes log(y² + 0.01), which carries the
volatility directly; #37's text says the network was never told about standard deviations, which
overstates it.

Kept for the session: the posterior's width (sd of log σ ≈ 0.022 at 1,000 days), so a learned
summary must be accurate to a fraction of it. A plain average of the log squares already gives
an R² of 0.996 for the log sd, and an error under one posterior sd needs an R² above about
0.997, so a high R² says little here.

An optional poll for the same week: how precisely do four years of daily returns pin down the
expected annual return of a stock with 25% volatility (±1 sd)? Options ±1%, ±3%, ±6% and ±12%;
the answer is about 12.5% from the data alone (Merton 1980; #37 panel §6).

### P5: Question 5/6, slot 5 (Tue 10 Nov)

```text
Part 5 of our ICAIF tutorial fits a stochastic-volatility model with jumps to 1,000 daily returns. Volatility is never observed and jump days carry no label, so there are eight parameters to infer plus today's volatility. On a recent window of an S&P 500 ETF, uncertainty about today's volatility made up about 90% of the uncertainty in the next day's VaR.

Particle MCMC gives the exact Bayesian answer, but in our runs it took 12 to 52 CPU-minutes per name. Approximate maximum likelihood takes about 11 CPU-minutes and gives intervals from the Hessian. On 40 simulated series, six fits gave no usable Hessian; of the other 34 intervals at 95%, 26 contained the true jump frequency and 17 the true mean jump. Profile-likelihood or bootstrap intervals are the usual fix, and each needs many more fits per name.

If you had 2,000 names to run before the open, what would you do?

#ICAIF2026 #QuantFinance #RiskManagement
```

Sources:

- Costs: `docs/results/svj-costs/costs.json` (particle MCMC 715 to 3,101 CPU-s over five runs,
  median 2,074; approximate ML median 647 CPU-s).
- Intervals: `docs/results/svj-classical/` (`classical.json` and the per-series files). Six of
  the 40 Hessians were not positive definite, so those fits have no interval; `classical.json`
  counts them as misses (26/40 and 17/40), and the post gives the 34 real intervals instead.
- The 90% share: `state_share` of the one-day VaR in
  `docs/results/svj-daily/VOO_2026-09-03_pmmh.json` (0.89 from the exact posterior; 0.92 from the
  network's).

Kept for the session: amortization, the importance-sampling audit with its 2% flag, and the
predictive ES against the plug-in.

### P6: Question 6/6, slot 6 (Thu 12 Nov)

```text
In April 2025 the 20-day realized volatility of an S&P 500 ETF reached 49%. Our model's spot volatility (the posterior median) never went above 32%, because the model put part of those days down to jumps, which in this model leave volatility unchanged. Real markets also have jumps in volatility. The network had passed its checks on simulated data, and the misspecification test we ran on every window flagged none of them.

The last part of our ICAIF tutorial is about cases like this, where the model rather than the network is wrong. If you run one model every morning on 2,000 names, with one prior meant for a typical stock, what would you check first to find the names it fits badly?

#ICAIF2026 #ModelRisk #BayesianInference
```

Sources: `docs/results/svj-rolling/VOO_rolling.csv`. The 20-day realized volatility peaked at
49% on 28 April 2025, when the posterior median of spot volatility was 22%; the median's own
peak was 32% on 21 April (95th percentile 40%). Realized volatility counts jump days and the
model's spot volatility does not, so part of the gap is built in; the post says why. The
explanation and "no window fails the test" are in `docs/svj-results-summary.md` and #38's
notebook. Slot 6 is not built yet; the post commits us to address both the model and the prior.

### P7: Logistics (optional, Fri 13 Nov)

```text
Our ICAIF tutorial, "Amortized Bayesian Calibration of Volatility Models", is on [day] at [time] in [room] at Bocconi. If you want to follow along, bring a laptop: setup is one command, and after that everything runs offline on a CPU. The instructions are in the first comment.

Thanks to everyone who answered the six questions over the last few weeks. We'll go through some of the answers in the session and, with your permission, credit the people who wrote them.
```

### P8: The answers (optional, after the conference)

A carousel with one slide per question: the question, our answer in two sentences, and the
notebook it lives in. Under each original post, reply with the short answer and, once the
repository is public, the link.

## 6. Engagement

- In the first hour, each co-organizer leaves one substantive comment, such as a hint, an
  objection, a reference or a sub-question. "Great post" is not enough. Early real comments carry
  a post further than anything else.
- Reshare with one personal sentence, not a bare repost.
- Reply to every comment within a day, without giving away the answer. "Close; does a third
  still work with daily data?" works.
- Keep a running list of the answers and objections worth discussing, with links. Before the
  session, pick two or three per slot and credit their authors.
- After the tutorial, reply under each question post with the short answer and, once public, the
  notebook link (P8).
- Tag sparingly: co-organizers, and the official ICAIF and BayesFlow pages if they exist. Check
  the exact page names first.

## 7. Assets and guardrails

| Post | Asset | Work needed |
|---|---|---|
| P0 | title card | names, title, venue, date; no ICAIF logo without permission |
| P1 | slot-1 scatter of estimated against true betas | export the left panel from the notebook; enlarge the type |
| P2 | "train once / use every day" diagram | export from the #37 notebook |
| P3 | SBC ECDF figure | in `docs/results/svj-validation/sbc.png`; re-export in portrait |
| P4 | read-out scatter and error histogram | export from the #37 notebook |
| P5 | example path with the hidden ingredients | in `docs/design/images/03-example-path.png`; re-export in portrait |
| P6 | VOO rolling posterior, top panel | crop `docs/results/svj-rolling/VOO_rolling.png` to portrait |

- **Use our numbers only**, and label simulated results as simulated. A real-data risk figure,
  if one ever appears, is an illustration, not advice.
- **No "frequentists are wrong".** Most of the audience's econometricians are frequentists. In
  slot 1 the least-squares interval is exact for each stock, Stein's result is itself
  frequentist, and the shrinkage estimator is also the empirical best linear unbiased predictor
  of a random-coefficients model. Say "the point-estimate route" or "the usual fit".
- **Name the better classical tool.** When a post shows a fast classical method failing, as P5
  does with Hessian intervals, say what a careful frequentist would use instead and what it
  costs.
- **Don't link the repository while it is private.** Decide first (§8).
- **Employer policies.** Organizers with industry roles check their employer's social-media
  rules before posting.
- **Don't reveal slot 6's answer.** It isn't built yet, and the post only asks the question.

## 8. Decisions and next steps

1. **Slots 2 and 3**: final titles and leads, which also set the post authors.
2. **Our day, time and room** at ICAIF '26, from the acceptance mail.
3. **Repository**: public before P0 (21 Oct), before P7, or after the conference.
4. **Who posts what** (§4), and whether co-organizers repost or write their own versions.
5. **Slot 6**: who builds it, and whether #42's crash-day example goes in. P6 commits us to an
   answer, and P8 to a notebook for it.

| By | Step | Owner |
|---|---|---|
| Tue 20 Oct | Decisions 1–5 | all |
| Wed 21 Oct | P0 | Francesco |
| Mon 26 Oct | Assets for P1–P2 | Francesco, Stefan |
| Tue 27 / Thu 29 Oct | P1, P2 | per §4 |
| Mon 2 Nov | Assets for P3–P4 | authors of P3 and P4 |
| Tue 3 / Thu 5 Nov | P3, P4 | per §4 |
| Tue 10 / Thu 12 Nov | P5, P6 | per §4 |
| Fri 13 Nov | P7; pick the LinkedIn answers to discuss | slot-1 presenter; all |
| Sat–Sun 14–15 Nov | Tutorial | all |
| Tue 17 or 24 Nov | P8 and the replies under P1–P6 | all |

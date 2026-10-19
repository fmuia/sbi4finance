# ICAIF '26 tutorial: the LinkedIn series

Prepared 19 October 2026 for the four organizers, and revised the same day after the intro
was reassessed. Nothing here is scheduled or posted. Slot titles marked *assumed* need
confirming.

## Summary

- **Seven core posts**, Tuesdays and Thursdays from 21 Oct to 12 Nov. One announces the
  tutorial; each of the other six opens with one slot's question and leaves it unanswered.
- **Two optional posts**: logistics on 13 Nov, and the answers after the conference, which keeps
  the promise made by leaving the questions open.
- **Full drafts** are in §5, each with its sources and with what we keep for the session.
- **Decisions needed** (§8): the titles and leads of slots 2–3, our day, time and room, and
  whether the repository, private today, goes public before the announcement.

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
| 6 | Priors and misspecification: when the model, not the network, is wrong | promised by #37/#38; a seed in [#42](https://github.com/fmuia/sbi4finance/pull/42) (the crash-day example) |

**Slot 1 changed after review.** The first intro asked how often jumps happen; the second asked
how likely a 7% loss day is next year. Both depended on a model the data barely identify, and the
second moved to slot 6 as a seed (#42). The intro now starts from the simplest Bayesian example
in finance, a stock's beta, and adds a result no frequentist disputes: estimating every stock's
beta on its own is beaten by shrinking the estimates together, which is why data vendors adjust
betas toward 1. P1 follows it. The design document in #41 (`docs/tutorial-intro-design.md`) has
the reasoning and the evidence.

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
  #QuantFinance, #RiskManagement, #BayesFlow, #SimulationBasedInference.
- **One visual** where it helps, portrait 1080×1350, with large type, our plot style and alt text.
- **Length** of 600–1,200 characters (the limit is 3,000).

Two open-source tools help with the check, and neither replaces reading the post aloud:

- [humanizer](https://github.com/blader/humanizer) (MIT) is an agent skill for Claude Code and
  similar tools, built on the Wikipedia guide. It marks 26 patterns and rewrites the text without
  changing its claims. Give it two or three paragraphs of your own writing and it matches your
  voice.
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
| Tue 27 Oct | P1 Question 1/6: why do vendors adjust every beta? | Francesco | the slot-1 scatter of estimated against true betas, with the 50 highest marked (left panel) |
| Thu 29 Oct | P2 Question 2/6: can a network learn Bayes' rule? | Stefan | the "train once" half of the diagram |
| Tue 3 Nov | P3 Question 3/6: how do you check a posterior you can't compute? | Jerry or Alexander | `docs/results/svj-validation/sbc.png`, uncaptioned |
| Thu 5 Nov | P4 Question 4/6: R² = 0.99, and still off on 30% of stocks | Alexander | slot 4's read-out scatter and error histogram (to export) |
| Tue 10 Nov | P5 Question 5/6: the number that drives risk is hidden | Francesco | `docs/design/images/03-example-path.png` |
| Thu 12 Nov | P6 Question 6/6: network, model, or prior? | Alexander or Stefan | `docs/results/svj-rolling/VOO_rolling.png` (top panel) |
| Fri 13 Nov | P7 (optional) Logistics: see you this weekend | the slot-1 presenter | title card |
| Tue 17 or 24 Nov | P8 (optional) The answers | all, one carousel | a slide per question |

## 5. Drafts

The numbers come from committed results or the open PRs; each draft names its sources. Re-check
every number against the final notebooks before posting.

### P0: Announcement (Wed 21 Oct)

```text
Stefan Radev, Jerry Huang, Alexander Fengler and I are giving a tutorial at ACM ICAIF '26 in Milan, "Amortized Bayesian Calibration of Volatility Models", on [day] at Bocconi.

If you run a risk process, you need a volatility model fitted to every name you hold, every morning, and you should also know how uncertain each fit is, since that uncertainty belongs in the VaR. Bayesian methods have handled the uncertainty part for stochastic volatility since the 90s. They are slow, though. In our runs an exact posterior for a stochastic-volatility model with jumps took about 35 CPU-minutes per name, which for 2,000 names comes to roughly 1,150 CPU-hours a day.

The tutorial is about amortized inference: you train a network once on simulated markets, and it then gives a full posterior for each new name in a few seconds. We'll build this in Python with the open-source BayesFlow library, check it against exact answers where we can compute them, and carry the uncertainty through to VaR and expected shortfall. Everything runs on a laptop CPU.

Over the next three weeks we'll post one question from each of the six parts of the tutorial, without the answers. If you have a view, leave it in the comments and we'll pick some up in the session.

#ICAIF2026 #BayesianInference #QuantFinance
```

First comment: the ICAIF '26 page and, once public, the repository. Sources: 35 CPU-minutes is
the median particle-MCMC cost and "seconds" about 3 CPU-seconds per name
(`docs/results/svj-costs/costs.json`). "Since the 1990s" refers to Jacquier, Polson and Rossi
(1994).

### P1: Question 1/6, slot 1 (Tue 27 Oct)

```text
First of six questions from our ICAIF tutorial, this one from the intro.

If you look up a stock's beta on Bloomberg, you get the regression estimate and an "adjusted beta", which is 0.67 times the estimate plus 0.33. Every beta is moved a third of the way toward 1, deliberately.

Why would a data vendor bias every number it shows you, and is a third the right amount?

Some context for the second part. We simulated 500 stocks with known betas and two years of weekly returns, Bloomberg's default window. The 50 highest regression estimates were 0.16 too high on average, so hedging those stocks with them left the book net short the market by about 16% of its size. On 20 real defensive stocks from 1999 to 2026, whose betas average about 0.5, the adjustment didn't help at all.

We'll go through our answer in Milan. In the meantime I'd like to know what people here use for betas, and why.

#ICAIF2026 #QuantFinance #RiskManagement
```

Sources:

- Bloomberg's adjusted beta and its default window (two years of weekly returns against the
  S&P 500): university library guides to the terminal; the weights go back to Blume (1971).
- The simulation: the slot-1 study (#41, `docs/results/intro-betas/summary.json`), 200 universes
  of 500 stocks with true betas around 1 ± 0.3 and two years of weekly returns. The 50 highest
  least-squares betas were too high by 0.16; the hedged book was net short the market by 8.1 of
  its 50 positions on average, in every universe.
- The real stocks: Blume's test on weekly returns of 20 large US stocks from defensive sectors
  (average beta 0.53), 1999–2026. The adjustment changed the squared error of next period's
  beta by +1% (one-year windows) and +0% (two-year windows) against least squares.

Kept for the session: why the highest estimates are too high (the winner's curse), Stein's
theorem, how Bayes' rule sets the amount of shrinkage stock by stock, and why a fixed third is
right for one data frequency at most.

### P2: Question 2/6, slot 2 (Thu 29 Oct)

```text
Part 2 of our ICAIF tutorial is about how amortized inference works, and the training loop fits in a sentence: draw parameters from a prior, simulate a data set with them, show the data set to a network, and train the network to put high probability on the parameters that generated it. Then repeat a few million times.

At no point do you write down a likelihood or run a sampler. Even so, if training goes well, the network returns the Bayesian posterior for a new data set in a single forward pass.

I'd like to hear how people would explain why that works, as opposed to the network just learning a good point estimate. And for anyone who has used this kind of model in production: what would you want to check before trusting it on real data?

#ICAIF2026 #SimulationBasedInference #BayesianInference
```

Kept for the session: the expected-KL argument, the amortization gap, the role of the prior
and simulation budget, and misspecification. If slot 2 leads with summary networks, ask instead
which eight numbers people would compress 1,000 daily returns into for a volatility model, and
whether the answer changes once the model has jumps and moving volatility.

### P3: Question 3/6, slot 3 (Tue 3 Nov)

This draft assumes slot 3 is validation; adjust it if not.

```text
For most models a risk desk uses, nobody can compute the exact posterior, so there is nothing to compare a fast approximation against. There is still a check that any correct Bayesian procedure has to pass, and all it needs is a simulator: generate a thousand stocks with known parameters, compute each one's posterior, and look at where the true values fall.

That check is the subject of part 3 of our ICAIF tutorial. If the posteriors were too narrow, or biased, what would you expect to see?

There is a harder question we'll spend time on in Milan. A posterior can pass this test and still be wrong for the particular stock you care about. How can that happen?

#ICAIF2026 #BayesianInference #ModelRisk
```

Kept for the session: rank uniformity (SBC), how to read the band, and calibration "on
average" against "for this name" (slot 5's audit). If slot 3 is instead "why and where it
matters for finance", ask after how many names training pays for itself, and what would make
people distrust the cheap answer. Training once costs about 5 CPU-hours (79 minutes on 4 cores),
an exact posterior about 35 CPU-minutes per name and the trained network about 3 CPU-seconds,
so the answer is about ten names (`costs.json`; results summary).

### P4: Question 4/6, slot 4 (Thu 5 Nov)

```text
A result from part 4 of our ICAIF tutorial.

We trained a small network on 256,000 simulated stocks to estimate volatility from 1,000 daily returns, without telling it anything about standard deviations. A linear regression on the eight features it learned recovers the sample log standard deviation with an R² of 0.99. But when we compare its volatility estimates with the exact posterior, they are off by more than one posterior standard deviation for about 30% of the test stocks.

Both numbers are correct. I'd be interested in how people reconcile them, and in the follow-up: for a model where you don't have an exact posterior, how would you notice a problem like this at all?

#ICAIF2026 #BayesFlow #MachineLearning
```

Sources: PR #37's executed-notebook results and `checkpoints/tutorial_gbm/` on that branch
(256,000 series, 279 s). Kept for the session: the posterior's width (sd of log σ ≈ 0.022 at
1,000 days), so a learned summary must be accurate to a fraction of it. An optional poll for
the same week: how precisely do four years of daily returns pin down the expected annual return
of a stock with 25% volatility (±1 sd)? Options ±1%, ±3%, ±6% and ±12%; the answer is about
12.5% from the data alone (Merton 1980; #37 panel §6).

### P5: Question 5/6, slot 5 (Tue 10 Nov)

```text
Part 5 of our ICAIF tutorial uses a stochastic-volatility model with jumps, fitted to 1,000 daily returns. The volatility on any given day is never observed, jump days aren't labelled, and there are nine parameters to estimate. On a recent window of an S&P 500 ETF, uncertainty about today's volatility accounts for about 90% of the uncertainty in the next day's risk number.

You can get the exact Bayesian answer with particle MCMC, but in our runs it took about 35 CPU-minutes per name, so roughly 1,150 CPU-hours a day for 2,000 names. Approximate maximum likelihood is faster (about 11 CPU-minutes per name) and gives a point estimate with Hessian-based intervals. On 40 simulated series, those 95% intervals contained the true jump frequency 26 times and the true mean jump 17 times.

If you ran that desk, which would you use, or would you do something else? And whichever you pick, how would you know, name by name, when not to trust it?

#ICAIF2026 #QuantFinance #RiskManagement
```

Sources: `docs/results/svj-costs/costs.json` (median 2,074 CPU-s; 1,152 CPU-hours; approximate
ML 647 CPU-s), `docs/results/svj-classical/classical.json` (26/40 and 17/40), and PR #38 section 2
(the 90% share). Kept for the session: amortization, the importance-sampling audit with its 2%
flag, and the predictive ES against the plug-in.

### P6: Question 6/6, slot 6 (Thu 12 Nov)

```text
In April 2025 the realized volatility of an S&P 500 ETF reached 49%, while our model's posterior for spot volatility peaked at around 32%. The network had passed its checks on simulated data, and the misspecification test we ran on those windows didn't flag anything.

So was it the network, the model or the prior? That's the subject of the last part of our ICAIF tutorial. The practical side is harder: if you run this every morning on 2,000 names with one prior meant for a typical stock, how do you find the names the model doesn't fit before the risk report goes out? I'd like to know what people would check first.

#ICAIF2026 #ModelRisk #BayesianInference
```

Sources: `docs/svj-results-summary.md` (real data) and PR #38 section 10. Slot 6 is not built
yet; the post commits us to address both the model and the prior.

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
| P3 | SBC ECDF figure | already in `docs/results/svj-validation/sbc.png` |
| P4 | read-out scatter and error histogram | export from the #37 notebook |
| P5 | example path with the hidden ingredients | already in `docs/design/images/03-example-path.png` |
| P6 | VOO rolling posterior, top panel | crop `docs/results/svj-rolling/VOO_rolling.png` |

- **Use our numbers only**, and label simulated results as simulated. A real-data risk figure,
  if one ever appears, is an illustration, not advice.
- **No "frequentists are wrong".** Most of the audience's econometricians are frequentists, and
  in the slot-1 study the profile likelihood and the penalized fit come close to the Bayesian
  answer. Say "the point-estimate route" or "the usual fit".
- **Don't link the repository while it is private.** Decide first (§8).
- **Employer policies.** Organizers with industry roles check their employer's social-media
  rules before posting.
- **Don't reveal slot 6's answer.** It isn't built yet, and the post only asks the question.

## 8. Decisions and next steps

1. **Slots 2 and 3**: final titles and leads, which also set the post authors.
2. **Our day, time and room** at ICAIF '26, from the acceptance mail.
3. **Repository**: public before P0 (21 Oct), before P7, or after the conference.
4. **Who posts what** (§4), and whether co-organizers repost or write their own versions.

| By | Step | Owner |
|---|---|---|
| Tue 20 Oct | Decisions 1–4 | all |
| Wed 21 Oct | P0 | Francesco |
| Mon 26 Oct | Assets for P1–P2 | Francesco, Stefan |
| Tue 27 / Thu 29 Oct | P1, P2 | per §4 |
| Mon 2 Nov | Assets for P3–P4 | |
| Tue 3 / Thu 5 Nov | P3, P4 | per §4 |
| Tue 10 / Thu 12 Nov | P5, P6 | per §4 |
| Fri 13 Nov | P7; pick the LinkedIn answers to discuss | |
| Sat–Sun 14–15 Nov | Tutorial | all |
| Tue 17 or 24 Nov | P8 and the replies under P1–P6 | all |

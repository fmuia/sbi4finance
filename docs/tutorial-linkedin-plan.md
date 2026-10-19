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
| 1 | Intro: what a point estimate hides (how likely is a crash day?) | notebook in its own PR, with its design and evidence |
| 2 | Amortized inference: normalizing flows and flow matching, summary networks, trade-offs | *assumed* |
| 3 | Validation in a Bayesian workflow, or "why and where it matters for finance" | *assumed* |
| 4 | Basic worked-out example in BayesFlow: constant volatility, exact posterior | draft [#37](https://github.com/fmuia/sbi4finance/pull/37) |
| 5 | Advanced: stochastic volatility with jumps, amortized and audited | draft [#38](https://github.com/fmuia/sbi4finance/pull/38) |
| 6 | Priors and misspecification: when the model, not the network, is wrong | promised by #37/#38; not built |

**Slot 1 changed after review.** Its question is now a decision, not a parameter: *how likely
is a day with a loss of 7% or more in the next year?* Four years of returns cannot tell "a few
large jumps" from "many small ones", and the two readings disagree about that probability by
orders of magnitude. The notebook compares the Bayesian answer with every frequentist option we
could think of, on a second tail model too. P1 below follows that framing. The design document
in the notebook PR (`docs/tutorial-intro-design.md`) has the reasoning and the evidence.

## 2. Goals

1. **Announce the tutorial**: topic, organizers, where and when, and what participants leave
   with (open notebooks, BayesFlow, a workflow they can run).
2. **One question per slot**, each left open, so that people comment, argue and come to Milan.
3. **Collect** the best answers and objections from the comments, address them in the session,
   and credit their authors with their permission. Commenting then pays off for them.
4. **After the conference, post the answers.** Every question post gets a second wave of
   activity.

## 3. Rules for every question post

- **Hook in the first lines**, which are all that shows before "…see more": the question,
  concrete and a little surprising.
- **Setup in 3–6 short lines**, with one number or figure from our own results.
- **No solution.** End on two or three open questions and an invitation to answer in the
  comments.
- **Call to action**: "Part *n* of our ICAIF '26 tutorial, Milan, [day]". Put the link in the
  first comment, not the body, because posts with links in the body reach fewer people.
- **Series marker** on the first line: "Question *n*/6 · ICAIF '26 tutorial". Use the same three
  to five hashtags each time: #ICAIF2026, then two to four of #BayesianInference, #QuantFinance,
  #RiskManagement, #BayesFlow, #SimulationBasedInference.
- **One visual**, preferably portrait 1080×1350, with large type, our plot style and alt text.
- **Length** of 700–1,500 characters (the limit is 3,000). No jargon in the hook; one technical
  term at most per paragraph.

## 4. Calendar

Post at 14:00–15:00 Milan time, which is the US East Coast morning. Europe moves to CET on 25
October and the US to EST on 1 November, so the time difference changes during the series. Each
post should come from the person presenting that slot.

| Date | Post | Proposed author | Visual |
|---|---|---|---|
| Wed 21 Oct | P0 Announcement | Francesco; the others reshare with a personal line | title card, or #37's "train once, use every day" diagram |
| Tue 27 Oct | P1 Question 1/6: how likely is a −7% day next year? | Francesco | the slot-1 notebook's likelihood ridge with the crash-probability lines (left panel only) |
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

### P0 — Announcement (Wed 21 Oct)

```text
We're giving a tutorial at ACM ICAIF '26 in Milan: amortized Bayesian inference for volatility models.

Every morning a risk desk needs three numbers for each of thousands of names: a volatility estimate, how sure it is, and a VaR / ES that accounts for both.

Bayesian methods have answered "how sure" for stochastic-volatility models since the 1990s. What they lacked was speed. In our runs, one exact posterior for a stochastic-volatility model with jumps costs about 35 CPU-minutes. Multiply by 2,000 names. Every day.

Amortized inference changes the trade-off. A neural network is trained once, on simulated markets, and then returns the full posterior for a new name in seconds. We'll show how to build one, how to check it name by name, and how to carry its uncertainty into the risk numbers.

Hands-on, in Python, with the open-source BayesFlow library, one layer at a time:
1. Why a point estimate is not enough for tail risk
2. How amortized inference works
3. [slot 3 title]
4. Your first amortized posterior, checked against the exact answer
5. Stochastic volatility with jumps: amortized and audited
6. When the model, not the network, is wrong

With Stefan Radev, Jerry Huang and Alexander Fengler.
ICAIF '26 · Bocconi University, Milan · [day, time, room]

Over the next three weeks we'll post one question from each part, without the answer. Bring yours to Milan.

#ICAIF2026 #BayesianInference #QuantFinance #RiskManagement #BayesFlow
```

First comment: the ICAIF '26 page and, once public, the repository. Sources: 35 CPU-minutes is
the median particle-MCMC cost and "seconds" about 3 CPU-seconds per name
(`docs/results/svj-costs/costs.json`). "Since the 1990s" refers to Jacquier, Polson and Rossi
(1994).

### P1 — Question 1/6, slot 1 (Tue 27 Oct)

```text
Question 1/6 · ICAIF '26 tutorial

How likely is a −7% day in the next year?

The last four years of an S&P 500 ETF contain none. Count them, and the answer is zero. Fit a normal distribution, and it is once in about 9 billion years. March 2020 had three in six trading days.

So allow for jumps, and ask the data again. Two readings are both consistent with the same four years: many small jumps, which put the chance of a −7% day next year at about 0.5%, or fewer, larger ones, which put it at about 28%.

The usual fit picks a single reading and reports its number: 3%. Is that safe? Across 300 simulated histories where we knew the answer, the usual fit understated the chance by a factor of 10 or more in nearly one in five. Its bootstrap interval, built to contain the truth 90% of the time, missed it in about one in four.

So what goes in the risk report?
→ the best fit?
→ the worst case?
→ something else?

Tell us in the comments. Part 1 of our tutorial in Milan answers it.

#ICAIF2026 #QuantFinance #RiskManagement #BayesianInference
```

Sources:

- VOO: the repository's data. The hook figures are on daily log returns, as in the notebooks.
- The two readings: the lowest and highest crash probability inside the 95% likelihood region
  of the jump model on VOO's last 1,000 days, about 19 jumps a year of ±1.8% against about 10 of
  ±3.6% (slot-1 notebook, "Back to VOO"). "Consistent with" means that a likelihood-ratio test
  at the 5% level rejects neither.
- The usual fit: the maximum-likelihood plug-in on the same window, 3.2%.
- The error rates: the slot-1 study, `docs/results/intro-crash-risk/`, 300 histories with
  truths drawn from the prior, a 7% loss. The maximum-likelihood plug-in was below a tenth of
  the true probability in 18% of them, and the bootstrap's 90% percentile interval covered the
  truth in 74%.

Kept for the session: the Bayesian answer and its calibration; what fixes the plug-in (averaging
over what the data allow, and a prior) and why the bootstrap does not; and why no method can be
both informative and valid at every truth here.

### P2 — Question 2/6, slot 2 (Thu 29 Oct)

```text
Question 2/6 · ICAIF '26 tutorial

Can a neural network learn Bayes' rule?

Here is the whole training recipe:
1. Draw parameters from a prior.
2. Simulate a data set from the model.
3. Show the data set to a network, and reward it for putting high probability on the parameters that produced it.
4. Repeat a few million times.

Nobody writes down a likelihood. Nobody runs a sampler.

Yet the claim is that, for a data set it has never seen, the trained network returns the Bayesian posterior, in one forward pass.

Two questions:
1. Why would this recipe give the posterior, and not just a good point estimate?
2. What would have to be true before you trusted it on real data?

Your answers in the comments. Part 2 of our ICAIF '26 tutorial takes both apart.

#ICAIF2026 #SimulationBasedInference #BayesianInference #MachineLearning
```

Kept for the session: the expected-KL argument, the amortization gap, the role of the prior
and simulation budget, and misspecification. If slot 2 leads with summary networks, use this
hook instead: "You must compress 1,000 daily returns into 8 numbers for a volatility model.
Which 8? And if the model has jumps and moving volatility?"

### P3 — Question 3/6, slot 3 (Tue 3 Nov)

This draft assumes slot 3 is validation; adjust it if not.

```text
Question 3/6 · ICAIF '26 tutorial

How do you check a posterior you can't compute?

For the models a risk desk actually runs, there is no exact posterior to compare against. Yet there is a test every correct Bayesian procedure must pass, and it needs nothing but the ability to simulate.

Hint: simulate 1,000 stocks with known parameters, and ask for each one's posterior.

What would you look at?
What pattern would tell you the posterior is too narrow? Biased?

And the harder one: why can a posterior pass that test and still be wrong for the one stock you care about?

Tell us below. Part 3 of our tutorial in Milan works through it.

#ICAIF2026 #BayesianInference #ModelRisk #BayesFlow
```

Kept for the session: rank uniformity (SBC), how to read the band, and calibration "on
average" against "for this name" (slot 5's audit). If slot 3 is instead "why and where it
matters for finance", use this hook: "Training once costs about 5 CPU-hours (79 minutes on 4
cores). An exact posterior costs about 35 CPU-minutes per name, the trained network about 3
CPU-seconds. After how many names does training pay for itself, and what would make you
distrust the cheap answer?" The answer is about ten names (`costs.json`; results summary).

### P4 — Question 4/6, slot 4 (Thu 5 Nov)

```text
Question 4/6 · ICAIF '26 tutorial

R² = 0.99. And still off on 30% of stocks.

We trained a small network on 256,000 simulated stocks to read volatility from 1,000 daily returns. Nobody told it what a standard deviation is.

A plain linear read-out of its 8 learned features recovers the sample (log) standard deviation with R² = 0.99.

And yet, against the exact posterior, its volatility estimate is off by more than one posterior standard deviation on about 30% of test stocks.

Both numbers are right. How?

And the question that matters for everything after: for a model with no exact posterior, how would you even notice?

Part 4 of our ICAIF '26 tutorial is hands-on: you build this workflow in BayesFlow and check every step against the exact answer.

#ICAIF2026 #BayesFlow #QuantFinance #MachineLearning
```

Sources: PR #37's executed-notebook results and `checkpoints/tutorial_gbm/` on that branch
(256,000 series, 279 s). Kept for the session: the posterior's width (sd of log σ ≈ 0.022 at
1,000 days), so a learned summary must be accurate to a fraction of it. An optional companion
poll for the same week: "Four years of daily returns on a stock with 25% volatility. How
precisely do they pin down its expected annual return (±1 sd)?" with options ±1% / ±3% / ±6% /
±12%. The answer is about 12.5% from the data alone (Merton 1980; #37 panel §6).

### P5 — Question 5/6, slot 5 (Tue 10 Nov)

```text
Question 5/6 · ICAIF '26 tutorial

The number that drives tomorrow's risk is one you never observe.

In a stochastic-volatility model with jumps, today's volatility is hidden, jump days carry no label, and nine unknowns must be inferred jointly from 1,000 daily returns. On a real S&P 500 ETF window, today's volatility alone accounts for about 90% of the uncertainty in the 1-day risk number.

The exact Bayesian answer exists. In our runs, particle MCMC delivers it in about 35 CPU-minutes per name. A desk following 2,000 names would need about 1,150 CPU-hours every morning.

The fast classical route, approximate maximum likelihood with a Hessian, takes about 11 CPU-minutes per name and returns one point. On 40 simulated series, its 95% intervals contained the true jump frequency 26 times and the true mean jump 17 times.

You run the desk. What do you do?
A) pay for exactness
B) take the shortcut
C) something else, and then: how do you know when to trust it, name by name?

Part 5 of our tutorial builds our answer, live.

#ICAIF2026 #QuantFinance #RiskManagement #BayesianInference
```

Sources: `docs/results/svj-costs/costs.json` (median 2,074 CPU-s; 1,152 CPU-hours; approximate
ML 647 CPU-s), `docs/results/svj-classical/classical.json` (26/40 and 17/40), and PR #38 section 2
(the 90% share). Kept for the session: amortization, the importance-sampling audit with its 2%
flag, and the predictive ES against the plug-in.

### P6 — Question 6/6, slot 6 (Thu 12 Nov)

```text
Question 6/6 · ICAIF '26 tutorial

April 2025. Realized volatility on an S&P 500 ETF hit 49%. Our model's posterior for spot volatility peaked near 32%.

The network had passed its checks on simulated data. The misspecification test on those windows flagged nothing.

So who was wrong: the network, the model, or the prior?

And the practical version: you'll run this every morning on 2,000 names, each with its own quirks, using one prior chosen for "a typical stock". How do you find the names the model doesn't describe, before the risk report goes out?

That is the last part of our ICAIF '26 tutorial in Milan. What would you check first?

#ICAIF2026 #ModelRisk #BayesianInference #QuantFinance
```

Sources: `docs/svj-results-summary.md` (real data) and PR #38 section 10. Slot 6 is not built
yet; the post commits us to address both the model and the prior.

### P7 — Logistics (optional, Fri 13 Nov)

```text
This weekend at ICAIF '26: "[title]" · [day, time, room], Bocconi University.

To follow along, bring a laptop. Setup takes one command and then everything runs offline on CPU: [instructions or link in comments].

Thanks for every answer to our six questions. We'll discuss some of them live, and we'll credit you.
```

### P8 — The answers (optional, after the conference)

A carousel with one slide per question: the question, our answer in two sentences, and the
notebook it lives in. Under each original post, reply with the short answer and, once the
repository is public, the link.

## 6. Engagement

- In the first hour, each co-organizer leaves one substantive comment, such as a hint, an
  objection, a reference or a sub-question. "Great post" is not enough. Early real comments carry
  a post further than anything else.
- Reshare with one personal sentence, not a bare repost.
- Reply to every comment within a day, without giving away the answer. "Close; what happens if
  the jump days were labelled?" works.
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
| P1 | slot-1 likelihood ridge with crash-probability lines | export the left panel from the notebook; enlarge the type |
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

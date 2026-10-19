# ICAIF '26 tutorial: the intro slot and the LinkedIn plan

Prepared 19 October 2026 for the four organizers. This is a proposal for discussion: nothing
here is scheduled or posted, and every slot title marked *assumed* needs confirming.

## Summary

- **Intro topic, recommended: "What a point estimate hides: few big jumps, or many small
  ones?"** It adds exactly one layer, Bayesian reasoning, with no neural network and no
  sampler. It asks the tutorial's own question (tail risk from 1,000 daily returns) of the
  simplest model in which the standard route visibly breaks. Jumps hidden among ordinary days
  turn the likelihood into a ridge, the maximum-likelihood fit lands on a bound, and Wald
  intervals miscover. The exact posterior fits on a small grid, and its uncertainty is carried
  into a VaR/ES. The intro closes on the cost: the same answer takes about 35 CPU-minutes per
  name once volatility is hidden too. Every later slot picks up a thread from it (§2.5).
- **LinkedIn: seven core posts**, Tuesdays and Thursdays from 21 Oct to 12 Nov. One announces
  the tutorial; each of the other six opens with one slot's question and leaves it unanswered.
  Two optional posts follow: logistics on 13 Nov and the answers after the conference. Full
  drafts are in §3.5.
- **Decisions needed** (§4): the titles and leads of slots 2–3, our day, time and room, and
  whether the repository goes public before the announcement (it is private today).

## 1. Where the tutorial stands

**Conference.** ACM ICAIF '26, 14–17 November 2026, Bocconi University, Milan. Workshops and
tutorials run on Saturday 14 and Sunday 15 November, followed by two main-conference days. The
published listings do not yet show our day, time or room.

**Proposal** (Overleaf, August 2026). Title: *Amortized Bayesian Calibration of Volatility
Models*, subtitle *Uncertainty Quantification, Model Validation, and Propagation to Risk*. It
has four parts: Method, Coverage, Validation and Consequence. The audience is developers, risk
managers, model validators, academics and PhD students. It has two hands-on parts, and the
format discussed was two 90-minute sessions with a 30-minute break. One stated learning
outcome is to "validate the results with simulation-based calibration checks".

**Slots.** Slots 4–6 come from draft PRs #37/#38 and their notebooks. Slots 2–3 are
reconstructed from fragments of the proposal's Schedule and Format section; please correct
them.

| Slot | Working title | Layer it adds | Status |
|---|---|---|---|
| 1 | Intro: what a point estimate hides | Bayesian reasoning | **this plan** |
| 2 | Amortized inference: normalizing flows and flow matching, summary networks (exchangeable, recurrent, transformer), composition and trade-offs | the method | *assumed* from the proposal |
| 3 | Validation in a Bayesian workflow (SBC, misspecification detection), or "why and where it matters for finance" | checking | *assumed*; the proposal has both texts |
| 4 | Basic worked-out example in BayesFlow: constant volatility, exact posterior | the software | draft [#37](https://github.com/fmuia/sbi4finance/pull/37) |
| 5 | Advanced: stochastic volatility with jumps, amortized and audited | a model nobody can solve exactly | draft [#38](https://github.com/fmuia/sbi4finance/pull/38) |
| 6 | Priors and misspecification: when the model, not the network, is wrong | model criticism | promised by #37/#38; not built |

Slots 4–6 follow one question, the opening of #37: every morning a risk desk needs, for each of
thousands of names, a volatility estimate, how sure it is, and a VaR/ES that accounts for both.
The intro should start that question, not a different one.

## 2. The intro topic

### 2.1 Recommendation

**"What a point estimate hides: few big jumps, or many small ones?"**

Take four years of daily returns from a constant-volatility model with occasional normal
jumps. Nobody can see which days were jumps. The likelihood of the jump frequency and size is
then a ridge, running from "a few large jumps" to "many small ones". The design document's
study (`docs/design/svj-design-document.md`, Figure 08 and Appendix D) measured the
consequences over 200 simulated four-year histories with a median of 10 jumps:

- the maximum sits on a bound in 62% of series;
- the Hessian is not positive definite in 17%;
- 95% Wald intervals cover the true jump frequency in **60%** of series, and the true jump-size
  sd in 77%.

The data cannot tell the two explanations apart, and the expected shortfall depends on which
one is true. The likelihood is in closed form, so the exact posterior of all five parameters
fits on one grid: fine over the three jump parameters, coarse over the well-identified drift
and diffusion volatility. The intro can therefore show the Bayesian answer itself, not an
approximation of it. An independent re-implementation (§2.7) reproduces the failure and shows
that its size depends on two settings the notebook must get right.

Two smaller pieces join it:

- **The opening hook.** VOO's last 1,000 trading days contain no day worse than −7%. A normal
  model fitted to that window puts such a day at about once every 8 billion years. March 2020
  had three in six trading days (9, 12 and 16 March).
- **The bridge to the risk number** (design document D.7). When a tail is learned from about
  ten observations, the predictive 99% quantile lies 27% further out than the plug-in one.

### 2.2 Why this one: the candidates compared

These are the criteria, taken from the brief and from the team's earlier discussions:

- add one layer only, so no network and no MCMC;
- use a model nobody argues about;
- show an explicit, measured failure of the standard route;
- ask a finance question, here tail risk;
- seed the later slots;
- avoid the reaction "I could do this without the Bayesian machinery".

The table scores the five candidates against the four criteria that differ between them. The
first two hold for all five.

| Candidate | Explicit failure of the standard route | Seeds later slots | Risk of "I could do this without Bayes" |
|---|---|---|---|
| **Hidden jumps** (recommended) | Yes, measured: bounds in 62% of series, Wald 60% | Directly: slot 5's jump parameters (Wald 26/40 and 17/40) and slot 6's prior | Low (below) |
| Zero crash days (Beta– or Gamma–Poisson on observed tail days) | Only at a zero count (MLE = 0); otherwise Wald works on the log scale (D.5) | Weakly: observed, not hidden, events | High: Clopper–Pearson or the rule of three answers it |
| Plug-in VaR breaches (normal returns, Student-t predictive) | Partly: a plug-in 99% VaR from 32 days of data, roughly the effective window of EWMA with λ = 0.94, is breached 1.45% of the time, not 1% | Slot 4 already teaches it with its slider | High: the frequentist prediction interval gives the same t answer |
| Sharpe-ratio shrinkage and multiple testing | Yes | Slot 6's "prior for your universe" | Medium; it needs a hierarchical model, so a second layer, and leaves the volatility theme |
| Low-default PD (zero defaults, regulatory PD floors) | Yes (MLE = 0) | No: credit, not market risk | High: "most prudent estimation" (Pluto and Tasche 2005) |

The recommended case resists the "without Bayes" objection for three reasons:

- The honest frequentist answer, the profile likelihood-ratio interval, is nearly
  uninformative. On Figure 08's Series A it runs from below 0.1 to beyond 100 jumps a year;
  the Wald interval there is 0.03 to 3.7.
- That interval still relies on a χ² approximation that fails at bounds (Chernoff 1954; Self
  and Liang 1987).
- The bootstrap is inconsistent at a boundary (Andrews 2000).

The Bayesian answer is informative because the prior says, in units a risk manager can check,
that jumps come a few times a year. That is exactly the claim the tutorial needs to make
honestly.

### 2.3 What we claim, and what we do not

- **The target is the point-estimate route, not frequentist statistics.** The route is MLE,
  then a Hessian/Wald interval, then a plug-in forecast. Stefan's note on the proposal applies:
  most economists are frequentists, and MLE does quantify uncertainty, through the information
  matrix or refits. The claim is that the quadratic picture fails in one regime, weakly
  identified tail parameters. It does not fail in general.
- **Say the control result out loud.** If the jump days were observed, the jump count would be
  Poisson, and Wald intervals would cover the truth in 92–95% of series (D.5). The failure
  comes from hiding the jumps, not from their rarity. Saying so earns credibility with
  econometricians.
- **Name the strong frequentist route.** Bates (2006) gives approximate ML for latent affine
  processes. The identification problem belongs to the data, not to the estimator.
- **Own the prior.** The posterior is informative partly because of the prior. State it in
  risk units, show its prior predictive check, and hand its sensitivity to slot 6.
- **Calibration is on average.** Bayesian intervals have nominal coverage on average over the
  prior, not at every true value. D.5 has a case at 0.5 jumps a year where Bayesian coverage
  is 86%. Show the averaged check: it is the simulation-based calibration (SBC) idea that slot
  4 uses to validate a network.

### 2.4 Outline (about 15 minutes)

| Minutes | Beat | Content | Asset |
|---|---|---|---|
| 0–2 | The question | The desk's morning question. VOO's last 1,000 days: 3 moves beyond 5 sd, where a normal model expects one about every 6,900 years. No day worse than −7%, so the historical frequency is 0 and a normal fit says once in 8 billion years. March 2020 had three. What number should the risk model use? | VOO return plot (from #37) |
| 2–5 | Bayes on one slide, and the toolbox | Prior × likelihood → posterior → predictive → decision. A confidence interval is a property of a procedure; a credible interval is a statement given the data. "You already use it": the Kalman filter, Black–Litterman, shrinkage, Bayesian VARs, MCMC for stochastic volatility since Jacquier, Polson and Rossi (1994). Then the toolbox map below. | one diagram, one table |
| 5–11 | The explicit case | The jump model in one line. One history with about 10 hidden jumps: the likelihood ridge, the MLE on a bound and the Wald ellipse (Figure 08, live). Coverage over 200 histories. The profile interval is honest but nearly uninformative. Bayes: a prior in risk units, the exact grid posterior, coverage over truths drawn from the prior (nominal by construction). | Figure 08 recomputed live; cached coverage |
| 11–13 | To the risk number | Push the posterior into a tail quantity: 10-day 99% VaR and 97.5% ES, closed form for this model as a binomial mixture of normals. Plug-in at the MLE against the predictive. The Student-t textbook case (D.7). | one plot, one small table |
| 13–15 | The catch | Add a hidden volatility path and the likelihood becomes a 1,000-dimensional integral. Particle MCMC takes about 35 CPU-minutes per name, about 1,150 CPU-hours a day for 2,000 names. MCMC has been an accurate route for stochastic volatility since Jacquier, Polson and Rossi (1994) and Eraker, Johannes and Polson (2003); the obstacle is what it costs per name, every day. "What if inference were paid once?" leads into slot 2. | cost figure without the amortized row |

The toolbox map for minute 2–5:

| Route | Needs from the model | Cost per new data set | Examples |
|---|---|---|---|
| Closed form or grid | a likelihood; a handful of parameters | milliseconds to seconds | conjugate models; this intro |
| MCMC (HMC/NUTS) | a likelihood and its gradient | seconds to minutes | Stan, PyMC, NumPyro |
| Particle MCMC | a simulator of the hidden state and an observation density | minutes to hours | slot 5's exact route |
| ABC, synthetic likelihood | only a simulator | many simulations per data set | |
| Amortized neural inference | only a simulator | seconds, after one training run | BayesFlow, sbi |

### 2.5 Threads the intro hands to later slots

| The intro plants | Picked up in |
|---|---|
| "Here the posterior is exact; soon it won't be" | Slot 2 (amortization), slot 4 (the exact check), slot 5 (no exact answer) |
| Coverage averaged over the prior | Slots 3–4: SBC as the diagnostic for a network |
| Plug-in against predictive | Slot 4's slider; slot 5's 500-series backtest |
| The jump-parameter ridge | Slot 5: the same ridge in nine dimensions on top of a hidden volatility path (Wald 26/40, 17/40) |
| The prior did real work | Slot 4's exercise 2; slot 6 |
| 35 CPU-minutes per name | Slot 5's cost section |

The model sequence goes jumps → no jumps → jumps with stochastic volatility, and that is
deliberate. The intro uses jumps because that is where point estimates break. Slot 4 strips the
model to its simplest form so that the software can be checked against an exact answer. Slot 5
puts the jumps back inside a model with hidden volatility. Participants meet the jump idea once,
in the intro, and slot 5 only adds stochastic volatility.

### 2.6 Notebook spec: `notebooks/tutorial_1_why_bayesian.py`

Its title would be "Slot 1 · What a point estimate hides". It is marimo, styled like slots 4–5,
with `svj.plots.style()` and the repository's VOO file, and needs no new dependency (scipy and
numpy are already pinned).

| Section | Live or cached | Notes |
|---|---|---|
| The question; VOO's tails | live | the tail counts and the normal-model odds above |
| Bayes on one slide; the toolbox map | static | |
| Simulate one history with about 10 hidden jumps | live | fixed seed, chosen so that the MLE lands on a bound, like Series A; a "how hidden are the jumps?" control switches the diffusion volatility between 16% (VOO-like) and 25% (the SVJ prior's median) |
| One five-dimensional grid; MLE; Wald ellipse; profile interval | live; measure it, target under 30 s | one evaluation gives both the profile picture (the maximum over the inner parameters, as in Figure 08) and the exact posterior (their sum) |
| Coverage over 200 histories, truths drawn from the prior: Wald against Bayes | cached JSON (about an hour offline) | this is the fair comparison, the same design as slot 5's 40-series study; opt-in rerun on fewer histories |
| Prior predictive; posterior marginals | live | the posterior's own coverage is the SBC idea, named and kept for slot 4 |
| 10-day VaR/ES: predictive against plug-in; the t table | live | binomial mixture of normals; closed form |
| The catch: hidden volatility and its cost | static | numbers from `docs/results/svj-costs/costs.json` |
| Exercises | live | (a) flat prior on 0.1–120 jumps a year: what happens to the posterior and the ES? (slot 6 preview); (b) label the jump days and rerun the Wald coverage, which recovers about 95% (the D.5 control) |

Code: start from `scripts/check_intro_jump_toy.py`, committed with this plan (§2.7). Move its
simulator, likelihood, bounded MLE with Hessian and grid posterior into
`src/sbi4finance/jump_toy.py`, and add the five-dimensional grid, the profile interval and the
VaR/ES. A coverage script then writes `docs/results/intro-jumps/`, with `tests/test_jump_toy.py`
and a `review-intro` pixi task. Effort is about one to one and a half days.

**Keep all five parameters free on both sides**, drift, diffusion volatility, jump frequency,
mean jump and jump-size sd, as a practitioner would. Fixing the nuisance parameters at their
true values favours the point-estimate route so much that the failure nearly disappears (§2.7).

### 2.7 Check of the reduced example

`scripts/check_intro_jump_toy.py`, committed with this plan, independently re-implements the
study, because the code behind Figure 08 is not in the repository. Run it with
`pixi run python scripts/check_intro_jump_toy.py`: the full table takes about 20 minutes, and
`--series 20` gives a quick look. Each setting has 200 simulated 1,000-day series, with 2.5 jumps
a year, a jump-size sd of 3%, a mean jump of −1% and a drift of 8%. The diffusion volatility is
16% unless stated. The bounds are the design document's (prior mean ± 6 prior sd), and the prior
is slot 5's. Wald intervals are built in the unbounded coordinates, and a Hessian that is not
positive definite counts as a miss. The exact posterior covers the three jump parameters on a
grid, with the diffusion volatility and the drift at their true values.

| Setting (200 series each) | MLE on a bound | Hessian not PD | Wald 95% coverage, jump frequency / jump sd | Exact posterior, same |
|---|---|---|---|---|
| 2 free (jump frequency, jump sd), truth at the prior median | 2% | 0% | 93% / 96% | 98% / 99%, flattering |
| 3 free (+ mean jump), truth at the prior median | 34% | 1% | 75% / 92% | 100% / 100%, flattering |
| 3 free, **truths drawn from the prior** | 33% | 1% | 86% / 96% | **92% / 99%** |
| 4 free (+ diffusion vol), truth at the prior median | 34% | 1% | 70% / 89% | not run |
| 5 free (+ drift), truth at the prior median | 40% | 3% | 70% / 93% | not run |
| 5 free, **25% diffusion volatility** | 60% | 8% | 64% / 80% | not run |
| Design document, 5 free (settings not recorded) | 62% | 17% | 60% / 77% | |

What follows for the intro:

1. **The failure reproduces**, but its size depends on two settings. The first is how many
   parameters are estimated: with only the two jump parameters free it nearly vanishes (93%).
   The second is how well the jumps hide among ordinary days. Raising the diffusion volatility
   from 16% to 25% moves the five-parameter result from 40% on a bound and 70% coverage to 60%
   and 64%, close to the design document's 62% and 60%. Its study probably used the higher
   volatility; confirm this, and record the settings in the intro's own script.
2. **The exact grid posterior is calibrated where it should be, within Monte Carlo error.**
   With truths drawn from the prior it covers 92% and 99% (nominal 95%; at 200 series the
   band is about 92–98%). The jump-frequency figure sits at the edge of that band; a scratch
   run with another seed gave 96% and 94%. The fixed grid is also coarse for the few prior
   draws with dozens of jumps, so the notebook should centre its grid on each series. This
   averaged check is the SBC idea, and the intro can demonstrate it live.
3. **Do not compare at a single truth.** At the prior median the Bayesian interval
   over-covers (98–100%), because the prior is centred on the truth. The fair comparison draws
   the truths from the prior, as slot 5's 40-series study does.
4. **Quote one source per number.** P1 quotes the design document's 60%. Once the intro
   notebook exists, quote its committed figure everywhere, on stage and in posts.

## 3. LinkedIn plan

### 3.1 Goals

1. Announce the tutorial: topic, organizers, where and when, and what participants leave with
   (open notebooks, BayesFlow, a workflow they can run).
2. One question per slot, each left open, so that people comment, argue and come to Milan.
3. Collect the best answers and objections from the comments, address them in the session, and
   credit the authors with their permission. Commenting then pays off for them.
4. After the conference, post the answers. Every question post then gets a second wave of
   activity, and we keep the promise made by leaving the questions open.

### 3.2 Rules for every question post

- **Hook in the first two lines**, which are all that shows before "…see more": the question,
  concrete and a little surprising.
- **Setup in 3–6 short lines**: one scenario and one number or figure from our own results.
- **The tension**: why the obvious answer is not enough.
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

### 3.3 Calendar

Post at 14:00–15:00 Milan time, which is the US East Coast morning. Europe moves to CET on 25
October and the US to EST on 1 November, so the time difference changes during the series.

| Date | Post | Proposed author | Visual |
|---|---|---|---|
| Wed 21 Oct | P0 Announcement | Francesco; the others reshare with a personal line | title card, or #37's "train once, use every day" diagram |
| Tue 27 Oct | P1 Question 1/6: few big jumps, or many small ones? | Francesco | `docs/design/images/08-jump-frequency-profile-likelihood.png` (Series A row) |
| Thu 29 Oct | P2 Question 2/6: can a network learn Bayes' rule? | Stefan | the same diagram, cropped to "train once" |
| Tue 3 Nov | P3 Question 3/6: how do you check a posterior you can't compute? | Jerry or Alexander | `docs/results/svj-validation/sbc.png`, uncaptioned |
| Thu 5 Nov | P4 Question 4/6: R² = 0.99, and still off on 30% of stocks | Alexander | slot 4's read-out scatter and error histogram (to export) |
| Tue 10 Nov | P5 Question 5/6: the number that drives risk is hidden | Francesco | `docs/design/images/03-example-path.png` |
| Thu 12 Nov | P6 Question 6/6: network, model, or prior? | Alexander or Stefan | `docs/results/svj-rolling/VOO_rolling.png` (top panel) |
| Fri 13 Nov | P7 (optional) Logistics: see you this weekend | the slot-1 presenter | room photo or title card |
| Tue 17 or 24 Nov | P8 (optional) The answers | all, one carousel | a slide per question |

Authors are a suggestion: each post should come from the person presenting that slot.

### 3.4 Engagement playbook

- In the first hour, each co-organizer leaves one substantive comment, such as a hint, an
  objection, a reference or a sub-question. "Great post" is not enough. Early real comments
  carry a post further than anything else.
- Reshare with one personal sentence, not a bare repost.
- Reply to every comment within a day, without giving away the answer. "Close; what happens if
  the jump days were labelled?" works.
- Keep a running list of the answers and objections worth discussing, with links, in the
  repository (or a shared doc). Before the session, pick two or three per slot and credit
  their authors.
- After the tutorial, reply under each question post with the short answer and, once public,
  the notebook link (P8).
- Tag sparingly: co-organizers, and the official ICAIF and BayesFlow pages if they exist.
  Check the exact page names first.

### 3.5 Drafts

The numbers come from committed results or the open PRs; each draft names its sources. Re-check
every number against the final notebooks before posting.

#### P0 — Announcement (Wed 21 Oct)

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
the median particle-MCMC cost (`docs/results/svj-costs/costs.json`); "seconds" is about 3
CPU-seconds per name, the same file; "since the 1990s" refers to Jacquier, Polson and Rossi
(1994).

#### P1 — Question 1/6, slot 1 (Tue 27 Oct)

```text
Question 1/6 · ICAIF '26 tutorial

Few big jumps, or many small ones?

Take four years of daily returns. Somewhere in them are about ten jump days, and nothing marks which ones.

Now ask two questions: how often do jumps happen, and how large are they?

The likelihood answers with a ridge. "A few large jumps" and "many small ones" explain the same history almost equally well. The best fit often sits on the edge of the parameter space, and the familiar ±1.96 standard errors draw an ellipse that misses the shape.

In 200 simulated histories with a known truth, that 95% interval contained the true jump frequency 60% of the time.

It isn't rarity alone: label the jump days, and the textbook interval works again. The trouble starts when the tail is hidden in the data, which is also where your expected shortfall comes from.

So what should a risk model report here?
→ the best fit, with a wider interval?
→ the profile likelihood?
→ something else?

Tell us in the comments. Part 1 of our tutorial in Milan answers it.

#ICAIF2026 #QuantFinance #RiskManagement #BayesianInference
```

Sources: design document, Figure 08, Appendix D.4–D.5 and the 200-series study. Kept for the
session: the posterior with a prior in risk units, its calibration and the predictive ES.

#### P2 — Question 2/6, slot 2 (Thu 29 Oct)

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
and simulation budget, and misspecification. If slot 2 leads with summary networks, the
alternative hook is: "You must compress 1,000 daily returns into 8 numbers for a volatility
model. Which 8? And if the model has jumps and moving volatility?"

#### P3 — Question 3/6, slot 3 (Tue 3 Nov)

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

Kept for the session: rank uniformity (SBC), how to read the ECDF band, and calibration "on
average" against "for this name" (slot 5's audit). If slot 3 is instead "why and where it
matters for finance", use this hook: "Training once costs about 5 CPU-hours (79 minutes on 4
cores). An exact posterior costs about 35 CPU-minutes per name, the trained network about 3
CPU-seconds. After how many names does training pay for itself, and what would make you
distrust the cheap answer?" The answer is about ten names (`costs.json`; results summary).

#### P4 — Question 4/6, slot 4 (Thu 5 Nov)

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

Sources: PR #37's executed-notebook results and `checkpoints/tutorial_gbm/` on that branch.
Kept for the session: the posterior's width (sd of log σ ≈ 0.022 for 1,000 days), so a learned
summary must be accurate to a fraction of it. An optional companion poll for the same week:
"Four years of daily returns on a stock with 25% volatility. How precisely do they pin down its
expected annual return (±1 sd)?" with options ±1% / ±3% / ±6% / ±12%. The answer is about 12.5%
from the data alone (Merton 1980; #37 panel §6).

#### P5 — Question 5/6, slot 5 (Tue 10 Nov)

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

Sources: `docs/results/svj-costs/costs.json` (median 2,074 CPU-s, 1,152 CPU-hours; approximate
ML 647 CPU-s), `docs/results/svj-classical/classical.json`, and PR #38 §2. Kept for the
session: amortization, the importance-sampling audit with its 2% flag, and predictive against
plug-in ES.

#### P6 — Question 6/6, slot 6 (Thu 12 Nov)

```text
Question 6/6 · ICAIF '26 tutorial

April 2025. Realized volatility on an S&P 500 ETF hit 49%. Our model's posterior for spot volatility peaked near 32%.

The network had passed its checks on simulated data. The misspecification test on those windows flagged nothing.

So who was wrong: the network, the model, or the prior?

And the practical version: you'll run this every morning on 2,000 names, each with its own quirks, using one prior chosen for "a typical stock". How do you find the names the model doesn't describe, before the risk report goes out?

That is the last part of our ICAIF '26 tutorial in Milan. What would you check first?

#ICAIF2026 #ModelRisk #BayesianInference #QuantFinance
```

Sources: `docs/svj-results-summary.md` (real data) and PR #38 §10. Slot 6 is not built yet;
the post commits us to address both the model and the prior.

#### P7 — Logistics (optional, Fri 13 Nov)

```text
This weekend at ICAIF '26: "[title]" · [day, time, room], Bocconi University.

To follow along, bring a laptop. Setup takes one command and then everything runs offline on CPU: [instructions or link in comments].

Thanks for every answer to our six questions. We'll discuss some of them live, and we'll credit you.
```

#### P8 — The answers (optional, after the conference)

A carousel with one slide per question: the question, our answer in two sentences, and the
notebook it lives in. Under each original post, reply with the short answer and, once the
repository is public, the link.

### 3.6 Assets to prepare

| Post | Asset | Work needed |
|---|---|---|
| P0 | title card | names, title, venue, date; no ICAIF logo without permission |
| P1 | Figure 08, Series A row | crop; enlarge the type for mobile |
| P2 | "train once / use every day" diagram | export from the #37 notebook |
| P3 | SBC ECDF figure | already in `docs/results/svj-validation/sbc.png` |
| P4 | read-out scatter and error histogram | export from the #37 notebook |
| P5 | example path with the hidden ingredients | already in `docs/design/images/03-example-path.png` |
| P6 | VOO rolling posterior, top panel | crop `docs/results/svj-rolling/VOO_rolling.png` |

### 3.7 Guardrails

- **Use our numbers only**, and label simulated results as simulated. If a real-data risk
  figure ever appears, it is percent log-loss and an illustration, not advice.
- **No "frequentists are wrong".** Say "the point-estimate route" and keep §2.3's control
  results in mind.
- **Don't link the repository while it is private.** Decide first (§4).
- **Employer policies.** Organizers with industry roles check their employer's social-media
  rules before posting.
- **Don't reveal slot 6's answer.** It isn't built yet, and the post only asks the question.

## 4. Decisions for the team

1. **Intro topic:** the recommendation in §2.1, or a runner-up from §2.2.
2. **Free parameters of the intro toy:** see §2.7.
3. **Slots 2 and 3:** final titles, and who leads each slot (this also sets the post authors).
4. **Our day, time and room** at ICAIF '26, from the acceptance mail.
5. **Repository:** public before P0 on 21 October, before P7, or after the conference.
6. **Who posts what** (§3.3), and whether co-organizers repost or write their own versions.

## 5. Next steps

| By | Step | Owner |
|---|---|---|
| Tue 20 Oct | Decisions 1, 3, 4, 5 and 6 | all |
| Wed 21 Oct | P0 | Francesco |
| Mon 26 Oct | Assets for P1–P2 | Francesco, Stefan |
| Tue 27 / Thu 29 Oct | P1, P2 | per §3.3 |
| Fri 30 Oct | Intro notebook draft (§2.6) as a PR | Francesco |
| Mon 2 Nov | Assets for P3–P4 | |
| Tue 3 / Thu 5 Nov | P3, P4 | |
| Tue 10 / Thu 12 Nov | P5, P6 | |
| Fri 13 Nov | P7; pick the LinkedIn answers to discuss | |
| Sat–Sun 14–15 Nov | Tutorial | all |
| Tue 17 or 24 Nov | P8 and the replies under P1–P6 | all |

## References used above

Andrews (2000), *Econometrica* 68(2); Bates (2006), *Review of Financial Studies* 19(3);
Chernoff (1954), *Annals of Mathematical Statistics* 25(3); Jacquier, Polson and Rossi (1994),
*Journal of Business & Economic Statistics* 12(4); Merton (1980), *Journal of Financial
Economics* 8(4); Pluto and Tasche (2005), *Risk* 18(8); Self and Liang (1987), *Journal of the
American Statistical Association* 82(398). The SVJ numbers come from the design document, the
results summary and the files in `docs/results/` cited with each post.

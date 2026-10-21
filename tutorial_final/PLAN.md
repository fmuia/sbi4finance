# Plan: from the drafts to `tutorial_final/`

Draft for team review, 21 October 2026. Owner: Francesco Muia.

`tutorial_final/` will hold only what we present at ICAIF '26 (Milan, 14–15 November): the
notebooks, the studies that produce their numbers and the results they load. The rest of the
repository stays where it is, as a research archive. This file is the plan to get there; at
release a participant README replaces it.

## 1. Goals and constraints

**Goals**

1. Decide the intro (slot 1), which Francesco presents.
2. Finalize slot 4, BayesFlow basics on constant volatility
   ([#37](https://github.com/fmuia/sbi4finance/pull/37)).
3. Finalize slot 5, stochastic volatility with jumps
   ([#38](https://github.com/fmuia/sbi4finance/pull/38)), which Francesco presents.
4. Collect the parts that meet the bar in §7 into `tutorial_final/`.

**Constraints**

- **C1, relevance.** Slots 1, 4 and 5 each answer a question that quants, risk managers and ML
  researchers at ICAIF already ask.
- **C2, necessity.** For every claim we can say when Bayesian inference is needed and when
  simulation-based, amortized inference is needed. We compare against the strongest alternative a
  skeptic would name, not the weakest.
- **C3, route.** GBM, then SVJ. No options for now. HKDE, Heston, Bergomi, rough Bergomi and MMAR
  stay in the archive; the options material can come back later as an optional extra.

Slots 2, 3 and 6 are outside this plan, except where they set an interface (§10).

## 2. The thesis we defend

Every section is judged against three sentences.

1. **Bayesian inference pays when the decision depends on what the data identify poorly:** rare
   events such as jumps, hidden states such as today's volatility, short windows, and many noisy
   estimates followed by a selection.
2. **Simulation-based, amortized inference pays when the likelihood is intractable or costly and
   many posteriors are needed** (names × days), or when the method itself must be validated on
   thousands of cases.
3. **When neither holds, we say so.** GBM with 1,000 days has an exact answer, and its plug-in
   forecast is as good as the predictive one.

The route has one case where neither holds (slot 4, used as a check) and one where both hold
(slot 5).

## 3. Where we stand

| Piece | Where | State | Gaps against C1–C3 |
|---|---|---|---|
| Intro, "one beta, then five hundred" | [#41](https://github.com/fmuia/sbi4finance/pull/41), draft | Built; two adversarial reviews; 15 tests; runs in about 10 s | About betas, not volatility, and slots 4–5 do not reuse its population prior. The real-data section rests on 20 survivor defensive stocks from Alpha Vantage, licence open. Long for an intro. |
| Crash-day example | [#42](https://github.com/fmuia/sbi4finance/pull/42), draft | Built | Now the seed of slot 6; its results are about the prior. Not an intro candidate. |
| Slot 4 | #37, draft, based on `bb3571d` | Runs in under 2 minutes with the saved networks | Quotes "about 35 CPU-minutes per name" for exact SVJ inference (§6). Part B says the network was never told about standard deviations, but it reads log squares. Long for a slot. Needs current `main`. |
| Slot 5 | #38, draft, based on `bb3571d` | Runs in under a minute with cached evidence | The best exact sampler is not measured. No real-data comparison with GARCH or filtered historical simulation. Classical intervals are Wald only. The plug-in comparison uses the posterior median; the MLE plug-in has only 40 series. The audit threshold is our own choice. |
| SVJ evidence | `main`, `docs/results/svj-*` | The design document's seven checks, re-run | See §6. |
| LinkedIn series | [#40](https://github.com/fmuia/sbi4finance/pull/40), draft | The decisions in its §8 were due 20 October; P0 is planned for today | P0 and P5 quote the particle-MCMC cost. P1 depends on the intro. P4 and P5 need frozen slot 4 and slot 5 numbers. |
| Environment | `pyproject.toml` | Locked for osx-arm64 and linux-64 | Windows and Intel-Mac laptops cannot install the locked environment. |

## 4. Decisions

| # | Decision | Recommendation | Needed by | Because |
|---|---|---|---|---|
| D1 | Intro topic (§5) | Option D: the volatility question as the spine, with a short beta example | Fri 23 Oct | P1's assets are due Mon 26 Oct |
| D2 | Tutorial length, slot durations, presenters (who presents slot 4?), owners of slots 2, 3 and 6 | From the acceptance mail | Fri 23 Oct | Trimming slots 4 and 5 needs a time budget |
| D3 | Record the route on [#26](https://github.com/fmuia/sbi4finance/issues/26): GBM → SVJ, options deferred; the `main` README points to `tutorial_final/` | Yes | Fri 23 Oct | The README still advertises the options route |
| D4 | Laptop support: add win-64 (and osx-64) to the lock, or offer a hosted notebook | Try win-64 first | Fri 30 Oct | The clean-machine install test is on Thu 5 Nov |
| D5 | Repository public, and data licences (the VOO/SPMO CSV, the Alpha Vantage betas, the E2 universe) | Public before P7, with no raw prices committed | Fri 30 Oct | The posts cannot link a private repository |
| D6 | Present from marimo (app view or Studio), or from slides with the notebook beside them | Notebooks for slots 4 and 5; the intro either way | Fri 30 Oct | It sets the rehearsal format |
| D7 | Message changes triggered by E1–E3 (§6) | Apply the decision rules | Fri 30 Oct (E1, E3), Mon 2 Nov (E2) | Slot 5 freezes on Fri 6 Nov |

## 5. Workstream A: the intro

**Criteria**

1. Relevance (C1): the room already asks the question.
2. Route (C3): it leads into GBM → SVJ without a detour.
3. Honesty (C2): it says when Bayes is not needed and when SBI is not needed.
4. Comfort: Francesco can give it without notes and answer the five questions below.
5. Cost: it is ready by Wed 4 Nov while slots 4 and 5 finish.

**Options**

| | Option | Relevance | Route | Honesty | Work left |
|---|---|---|---|---|---|
| A | #41 as it is, trimmed | High for quants: Bloomberg's beta, a hedge | Weak: betas | Strong: Stein's scope, the BLUP equivalence | Trim; settle or drop the real-data section |
| B | Crash day (#42) | High | Good: jumps | The prior decides most of the answer | Moved to slot 6; not recommended |
| C | Volatility first: "tomorrow's risk number for 2,000 names: what we don't know, and what it costs to know it" | High for risk | Direct | Built on the thesis | New, about 3 days |
| D | C's spine with a 6-minute beta example from #41 | High | Direct | Strong | About 2 days; reuses #41's verified results |

C's "Bayes in one example" could itself stay on the route: counts of extreme loss days across 500
names, pooled with a Gamma–Poisson prior. That is the betas' winner's-curse structure applied to
tail events. It is new, so it would need its own check.

**Recommendation: D.** It keeps #41's strongest verified result (the winner's curse and the
mis-sized hedge), makes the volatility question the spine (C3), puts the thesis of §2 on screen
by minute 10 (C2) and leaves LinkedIn P1 valid. The full #41 notebook becomes an optional
companion in `extras/`. Choose A if the intro gets 25 minutes or more and the team wants a single
closed-form story. Choose C if the beta example feels like a detour in the dry talk.

Outline for D, about 18 minutes (to adjust after D2):

1. **The question** (2 min). Every morning, VaR and ES for 2,000 names. Three unknowns: today's
   volatility (hidden), the parameters (tails poorly identified) and the model.
2. **Bayes in one example** (6 min). Of 500 betas, the 50 highest are too high by 0.16, and their
   hedge is left about $8m net short on a $50m book. The posterior is a weighted average of the
   estimate and the cross-section. Say up front that empirical Bayes and BLUP give the same
   estimator.
3. **When it matters, and when it doesn't** (4 min). GBM: the plug-in against the predictive ES by
   window length, from slot 4 (negligible at 1,000 days, 2.4% at 60, 6% at 20). SVJ: jump
   parameters that four years cannot pin down, from slot 5.
4. **When simulation-based, amortized inference matters** (4 min). A 2 × 2 map: likelihood
   tractable or not, one posterior or thousands a day. Each cell gets its tool and, for SVJ, its
   measured cost (from E1).
5. **The route** (2 min). GBM, where every step can be checked, then SVJ, where it cannot.

**Steps**

- Wed 21–Thu 22 Oct: storyboard D (eight frames) and a trimmed A. Francesco gives each as a
  5-minute dry talk. Scan the ICAIF '26 programme for overlapping talks and tutorials (30 min).
- Fri 23 Oct: decide (D1); adjust #40's P1 if needed.
- Mon 26–Thu 29 Oct: build the intro notebook, closed form only and under 30 s, from
  `beta_shrinkage.py` (#41) and slot 4's `gbm_volatility.py`.
- Wed 4 Nov: freeze.

**Five questions the intro must survive**

1. "This is James–Stein, or empirical Bayes." Yes. Bayes adds the amount of shrinkage per name and
   a distribution to decide with.
2. "Why not GARCH or filtered historical simulation for VaR?" They are slot 5's benchmarks (E2).
3. "Why not Stan for each name?" Measured in slot 5 (E1).
4. "A network can be wrong for my name." Slot 3 (calibration) and slot 5 (the audit).
5. "Your prior decides the answer." Where the data are weak, yes; that is slot 6.

## 6. The C2 check: the alternatives we must face

| Slot | Claim | The alternative a skeptic will name | What we have | Action |
|---|---|---|---|---|
| 1 | Pulling betas toward the cross-section improves a hedge | James–Stein, empirical Bayes, BLUP; Blume's and Vasicek's adjustments; better data (Dimson betas, daily returns) | #41 covers all of them and says empirical Bayes gives the same estimator | None for the vignette. Drop the real-data part or settle its licence (D5). |
| 4 | The network reproduces the exact posterior | The MLE with its standard error; the conjugate posterior | A grid check against the exact posterior | Add one line: the network's interval for σ next to MLE ± 1.645 se. Say that at 1,000 days Bayes adds nothing and the slot checks the workflow. |
| 5 | The exact posterior costs about 35 CPU-minutes per name | NUTS on the variance path with the jumps summed out (Stan or NumPyro); particle MCMC with a tuned or correlated proposal (Deligiannidis, Doucet and Pitt 2018); particle Gibbs. Plain SV without jumps fits in seconds to a minute with specialised samplers (stochvol), so "SV is slow" is no argument on its own. | One particle-MCMC set-up that accepts 9–16% of moves: 12–52 minutes per name. The design document's run had a median of 6 minutes. | **E1** |
| 5 | Classical intervals fail for the jump parameters | Profile-likelihood intervals, the bootstrap, sets robust to weak identification. #42's own design note calls Wald "the weakest frequentist answer". | Wald only, on 40 series | **E3b** |
| 5 | Posterior intervals cover the truth in 39 of 40 series | "Only on average over your own prior" | Simulation-based calibration; coverage with truths drawn from the prior | **E3a** |
| 5 | Parameter uncertainty belongs in VaR and ES | The plug-in at the MLE; GARCH with bootstrap bands (Christoffersen and Gonçalves 2005) | 500 series: the plug-in at the posterior median breaches +0.06/+0.09 pp more and its Z2 is −0.05/−0.07, both significant. At the MLE: 40 series only, not significant against the predictive. | **E3c** |
| 5 | An SVJ risk number is worth computing | Historical simulation, filtered historical simulation, GARCH(1,1)-t, GJR-GARCH-t ("does anything beat a GARCH(1,1)?", Hansen and Lunde 2005). Realized-volatility models need intraday data; we say they are out of scope. | Historical and filtered historical simulation, on simulated SVJ data only (design document, Appendix A) | **E2** |
| 5 | The audit makes the network trustworthy name by name | Pareto-smoothed importance sampling (PSIS k̂) with a fallback to NUTS, as in the amortized Bayesian workflow (Schmitt et al. 2024) | An efficiency threshold of 2%, chosen by us; 31–157 CPU-s per name | **E4** |
| 5 | The prior suits a typical stock | A prior predictive check against real names | A prior predictive check on simulations only | **E5** |

**Experiments.** E1–E3 are required; E4 and E5 are worth doing if time allows. New study code goes
in `tutorial_final/studies/` and library code in `src/sbi4finance/svj/`. Study-only dependencies
(Stan or NumPyro, `arch`) go in a separate pixi feature, outside the participant environment.

| ID | Question | Method | How the result changes the message | Effort | Due |
|---|---|---|---|---|---|
| E1 | What does the best exact posterior cost per name? | NUTS on the daily SVJ model, with the jump summed out each day, the variance path sampled in a non-centred form and the floor as a constraint. Check agreement with particle MCMC on the four agreement series. Rerun particle MCMC with a tuned proposal. Report CPU-s per 400 effective draws (the minimum over the nine targets), R-hat and agreement in posterior sd. | **5 minutes or more:** keep the message with the new number. **1 to 5 minutes:** lead with scale (2,000 names every day) and with validation on thousands of cases. **Under a minute:** NUTS becomes the exact route and the fallback for flagged names; the case for amortization is the daily universe and cheap validation. In every case, replace "35 CPU-minutes" everywhere. | 1.5 days | Wed 28 Oct |
| E2 | Does SVJ add anything on real data? | Rolling 1- and 10-day 99% VaR and 97.5% ES on a universe of daily series (Kenneth French's industry portfolios, VOO and SPMO, and single stocks if a licensed source exists): historical simulation (250 days), filtered historical simulation (EWMA 0.94), GARCH(1,1)-t, GJR-GARCH-t, the SVJ predictive and the SVJ plug-in, with weekly SVJ refits. Scores: exceedances with Kupiec's and Christoffersen's tests, the Acerbi–Székely Z2 and the FZ0 joint loss (Patton, Ziegel and Chen 2019) with Diebold–Mariano tests. | **SVJ better than GARCH-t:** say so. **Comparable:** claim comparable accuracy, plus what GARCH does not give (parameter and state uncertainty, the split between jumps and volatility). **Worse:** drop accuracy claims, present SVJ as a challenger and validation model, and hand the gap to slot 6. | 2–3 days, Oscar | Mon 2 Nov |
| E3 | Is the comparison of Bayes with the classical route fair? | (a) Coverage at four or five fixed truths, 200 series each, including truths in the prior's tails: minutes with the network. (b) Profile-likelihood intervals for the jump frequency and the mean jump on the 40 classical series, with the grid filter, on Oscar. (c) The plug-in at the MLE against the predictive on 300 or more series, on Oscar. | Rewrite slot 5's first claim with its conditions: calibrated on average over the prior, and where it is not at fixed truths; Wald fails, the profile interval is honest but wide; the plug-in at the MLE against the predictive, on enough series to tell. | 1–2 days, Oscar | Fri 30 Oct |
| E4 | How should the audit decide? | PSIS k̂ next to the efficiency, on the agreement series and the rolling windows; fewer proposals; the grid filter in place of the particle filter. | Adopt k̂ < 0.7 if the two agree. Set the daily policy (audit every name, flagged names only, or a sample) and its cost. | 0.5 day | Fri 30 Oct |
| E5 | Does the prior cover real names? | The prior predictive distribution of volatility, kurtosis, skewness, autocorrelation of absolute returns and the largest loss, against the E2 universe. | One figure in slot 5, or handed to slot 6. | 0.5 day | Mon 2 Nov |

**Known and not planned.** If asked, we say so: realized-volatility models with intraday data;
hand-made summary statistics against the learned summary; jumps in volatility (SVCJ);
dependence between names; the options route.

## 7. `tutorial_final/`: layout and the bar for "final"

```
tutorial_final/
  README.md        participant guide: schedule, setup, what runs live, what is cached (written at freeze)
  PLAN.md          this file; removed at release
  CLAIMS.md        every number said on stage or posted: value, source file, command, status
  MANIFEST.md      what was harvested, from which commit or pull request, with sha256 of copied files
  notebooks/       slot1_intro.py, slot4_bayesflow_basics.py, slot5_svj.py
  extras/          optional companions: the full beta notebook, the original SVJ pipeline notebook
  studies/         scripts that write results/ (E1–E5, and wrappers for the existing SVJ studies)
  results/         only the files the notebooks load
  checkpoints/     gbm/ and svj/ networks, with their training records
  data/            returns used live, with SOURCES.md (provenance and licence)
  tests/           each notebook runs headless within its budget; CLAIMS.md values match results/
```

**Rules**

- Library code stays in `src/sbi4finance/`, in one copy, with its tests. `tutorial_final/` holds
  the lessons and their evidence. `MANIFEST.md` lists the modules each notebook imports.
- Notebooks resolve paths from `tutorial_final/` (`mo.notebook_dir().parent`), so the folder needs
  nothing from the repository except the installed package.
- Artifacts are copied, not moved, with their sha256 in `MANIFEST.md` and a test that checks the
  copies. The archive keeps working.

**A slot is final when all of these hold**

1. It runs headless from a clean `pixi install --locked`, offline and within its time budget, on
   osx-arm64 and linux-64 (and win-64 after D4).
2. `marimo check --strict` and its tests pass.
3. Every number on screen is computed live or loaded from `results/`, and is listed in `CLAIMS.md`
   as verified.
4. Its rows in §6 are closed, or the claim is gone.
5. Core and optional sections are marked, and the core fits the slot.
6. A co-presenter has reviewed it, and the presenter has rehearsed it once with a timer.
7. The provenance and licence of its data are recorded.

**Process.** One pull request per slot ("harvest: slot N") adds the notebook, its results, its
checkpoints and its `CLAIMS.md` rows. After a freeze, only fixes. #37, #38 and #41 close when
harvested; #42 stays open for slot 6.

**Triage of the repository**

| Item | Today | Goes to |
|---|---|---|
| `tutorial_4_bayesflow_basics.py`, `gbm_volatility.py`, `checkpoints/tutorial_gbm/` | #37 | `notebooks/slot4_bayesflow_basics.py`, `checkpoints/gbm/` |
| `tutorial_5_svj_advanced.py`, `svj/tutorial.py`, `checkpoints/svj/`, `docs/results/svj-{validation,agreement,classical,backtest,costs,daily,rolling}/` | #38 and `main` | `notebooks/slot5_svj.py`, `checkpoints/svj/`, `results/svj/` (the loaded files only) |
| `tutorial_1_why_bayesian.py`, `beta_shrinkage.py`, `docs/results/intro-betas/` | #41 | After D1: the vignette in `notebooks/slot1_intro.py`, the full notebook in `extras/` |
| `notebooks/svj_var_pipeline.py`, `data/synthetic/svj-workflow/` | `main` | `extras/`, as a reference; not presented |
| `data/SPMO_VOO_2020-01-01_2026-09-04.csv` | `main` | `data/`, after the provenance check (D5) |
| `docs/design/svj-design-document.md`, `docs/svj-results-summary.md` | `main` | Linked from the README as background reading |
| `tutorial_6_crash_risk.py`, `crash_risk.py` | #42 | Not harvested now; the seed of slot 6 |
| Notebooks 01 (GBM foundations, MMAR), 04–11 and `hkde_data_walkthrough.py`; the HKDE, Heston, Bergomi and rough-Bergomi modules and results | `main` | Archive, not part of the tutorial |

## 8. Slot 4: what is left

- [ ] Merge current `main` into `feat/tutorial-slot4-bayesflow-basics`, which is based on `bb3571d`.
- [ ] With D2's budget, mark the core (steps 1–6, risk, VOO) and the optional parts (Part B's
      details, the exercises, the folds). The core runs live in under 2 minutes.
- [ ] C2: show the network's 90% interval for σ next to MLE ± 1.645 se, and say that at 1,000 days
      Bayes adds nothing and the slot checks the workflow.
- [ ] Replace "about 35 CPU-minutes per name" in step 6 with E1's number, read from `results/`
      rather than typed.
- [ ] Part B: say that the network reads log squares, which carry the volatility (the LinkedIn
      review flagged the current wording). Report R² and the share of stocks off by more than one
      posterior sd as a range over two or three training seeds.
- [ ] Make the references to slot 6 match whatever slot 6 becomes.
- [ ] Time the live-training button on a laptop; keep the saved networks as the default.
- [ ] Ask the owners of slots 2 and 3 what they cover (flows, flow matching, simulation-based
      calibration) and shorten the folds to match.
- [ ] Tests, `marimo check --strict`, a co-presenter's review, a rehearsal and the harvest pull
      request. **Freeze Fri 30 Oct.**

## 9. Slot 5: what is left

- [ ] Merge current `main` (and slot 4's harvest) into `feat/tutorial-slot5-svj-advanced`.
- [ ] A core path for 40 minutes, the design document's length (to confirm with D2): the hidden
      volatility → why there is no formula → the classical answer and its limits (after E3) → the
      same code with a new simulator → calibration on average → the audit (after E4) → the risk
      numbers → the cost (after E1) → real data, with the comparison against standard practice
      (E2) → the hand-over to slot 6. Live: one network posterior, one audit with 1,000 proposals
      and the forecast. Everything else cached.
- [ ] Rewrite the three claims with the results of E1–E3, following the decision rules in §6.
- [ ] One cost table measured on one machine: the network, the audit, the MLE, particle MCMC and
      NUTS, in CPU-seconds and wall time, with the hardware; the training cost and the break-even
      number of names.
- [ ] State the conditions of every simulated result: the daily SVJ model, truths drawn from the
      prior.
- [ ] Keep April 2025 (realized volatility 49%, the posterior near 32%) as the hand-over to slot 6.
- [ ] The VOO data's provenance and licence (D5).
- [ ] Tests, `marimo check --strict`, a co-presenter's review, a rehearsal and the harvest pull
      request. **Freeze Fri 6 Nov.**

## 10. Interfaces with slots 2, 3 and 6

Each owner writes one paragraph by Fri 30 Oct: what the slot explains, its key figure, and what it
assumes the audience has already seen. Until then:

- Slot 2 explains amortized posterior estimation, flows and flow matching, and summary networks.
  Slot 4 shows the code and keeps the mathematics in folds.
- Slot 3 explains simulation-based calibration and coverage. Slot 4 uses BayesFlow's default
  diagnostics; slot 5 adds the audit name by name.
- Slot 6 takes priors and misspecification, including April 2025 and the crash-day seed (#42).
  Slots 4 and 5 promise it; if it has no owner by Fri 30 Oct, they hand over to "what we did not
  cover" instead.

## 11. Timeline

| Week | Intro | Slot 4 | Slot 5 and studies | Folder and logistics | LinkedIn (#40) |
|---|---|---|---|---|---|
| Wed 21–Fri 23 Oct | Storyboards A and D, dry talks, D1 | Merge `main`; time budget | Start E1 and E3a; get E2's data | Create the folder skeleton and seed `CLAIMS.md` from §12; D2, D3 | P0: drop its particle-MCMC cost line, or post after E1 |
| Mon 26–Fri 30 Oct | Build | Edits; **freeze Fri 30** | E1 Wed 28; E3 and E4 Fri 30; E2 running | Harvest slot 4; D4–D6; try a win-64 lock | P1 Tue 27, P2 Thu 29 |
| Mon 2–Fri 6 Nov | **Freeze Wed 4** | Harvested | E2 and E5 Mon 2; rewrite; **freeze Fri 6** | Clean-machine install test Thu 5; harvest the intro and slot 5 | P3 Tue 3, P4 Thu 5 (slot 4's numbers) |
| Mon 9–Fri 13 Nov | Rehearse | Rehearse | Rehearse | Full run-through with a timer Mon 9; fixes only; tag a release Wed 11; participant README | P5 Tue 10 (slot 5's numbers), P6 Thu 12, P7 Fri 13 |
| Sat 14–Sun 15 Nov | Tutorial | | | | |

## 12. Claims at risk (the seed of `CLAIMS.md`)

| Claim, and where it appears | Status | Action |
|---|---|---|
| The exact posterior costs about 35 CPU-minutes per name, and 2,000 names cost 1,152 CPU-hours a day (slot 4, step 6; slot 5, §3 and takeaways; P0, P5) | At risk: one sampler that accepts 9–16% of moves; the design document's run had a median of 6 minutes | E1 |
| The network is about 700 times cheaper than particle MCMC, 26 times with an audit, and training pays for itself within about ten names (slot 5) | At risk; follows E1 | E1, E4 |
| Wald intervals cover the jump frequency in 26 of 40 series and the mean jump in 17; the posterior covers both in 39 (slot 5, §4; P5) | Correct, but against Wald only, and the posterior's coverage is over truths drawn from the prior | E3a, E3b; state the conditions |
| The plug-in breaches its VaR more (+0.06/+0.09 pp) and understates the ES (Z2 −0.05/−0.07) (slot 5, §8) | Correct for the plug-in at the posterior median; at the MLE, not significant on 40 series | E3c |
| Today's volatility carries about 90% of the uncertainty of VOO's 1-day risk number (slot 5, §2; P5) | Source and definition still to record | Add to `CLAIMS.md` with its source |
| The learned summary has R² 0.99 and is still off by more than one posterior sd on about 30% of stocks (slot 4, Part B; P4) | Computed live; depends on the training seed | Report a range over seeds |
| Bloomberg's rule makes no measurable difference on 20 real stocks (slot 1; P1) | A small survivor sample; licence open | The vignette uses simulation; cite Levi and Welch (2017) for real data |

## 13. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| E1 finds NUTS at about the cost of the audit | The per-name cost argument weakens | Lead with scale, daily refits and cheap validation; NUTS becomes the fallback. Decided by Fri 30 Oct. |
| E2 finds GARCH-t as good as SVJ on real data, or better | "Why SVJ?" | Say it. Present SVJ as a challenger and uncertainty model; slot 6 takes misspecification. |
| Slots 2, 3 and 6 have no owner | Slots 4 and 5 explain too much or too little, and promise a slot that does not exist | D2; the interface paragraphs of §10 |
| Windows laptops | Participants cannot install | D4; a hosted notebook as the fallback |
| Data licences | We cannot publish the repository or the posts' figures | Commit derived results only; download raw data at run time |
| E2 or E3 compute runs late | Slot 5 freezes without them | Run on Oscar from day one; cut the universe or the refit frequency first |
| New ideas | The schedule slips | Anything outside this plan goes on a list for after ICAIF |

## 14. This week

1. Francesco: storyboard intro options A and D, give both as dry talks, and decide by Fri 23 Oct.
2. Team: D2 and D3 (a comment on #26), and owners for slots 2, 3 and 6.
3. Start E1 and E3a; collect E2's data.
4. Create the folder skeleton: a README stub, `CLAIMS.md` seeded from §12, and `MANIFEST.md`.
5. Decide whether P0 goes out without its particle-MCMC cost line, or after E1.

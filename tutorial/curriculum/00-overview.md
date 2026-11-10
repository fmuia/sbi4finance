# The ICAIF '26 tutorial: curriculum

**Current application workstream (1 November):** the workshop duration is now
two hours. Refine the existing option and trajectory application notebooks
before allocating minutes or pruning. The developed subplans, dependencies and
first implementation tranche are in the
[application notebook execution plan](TUTORIAL-REVIEW.md). Earlier slot timings
and returns-exclusion decisions below are historical and do not supersede this
new scope. Introduction, Amortized Bayes, Architectures and backbone review are
outside this refinement tranche.

Draft for the organizers' review. 24 October 2026, revision 3.

This folder fixes what the tutorial teaches, in what order and why: this summary, and one file per slot.

For the proposed next numerical campaign, read the [experiment plan](../../lab/plans/NUMERICAL-EXPERIMENT-PLAN.md). The separate [tutorial review](TUTORIAL-REVIEW.md) assesses the learning path, its SBI motivation and the claims that need stronger evidence.

| File | Slot | Presenter | Minutes |
| --- | --- | --- | --- |
| [01-slot1-the-problem.md](01-slot1-the-problem.md) | 1. Introduction: Bayes for volatility models | Francesco | 15 |
| [02-slot2-amortized-bayes.md](02-slot2-amortized-bayes.md) | 2. What is amortized Bayes? | Alex | 15 |
| [03-slot3-architectures.md](03-slot3-architectures.md) | 3. Network architectures | Stefan, Jerry | 10 |
| [04-slot4-heston-in-bayesflow.md](04-slot4-heston-in-bayesflow.md) | 4. A worked example in BayesFlow: Heston on the SPY smile | Stefan, Jerry | 20 |
| [05-slot5-rough-bergomi-to-a-reserve.md](05-slot5-rough-bergomi-to-a-reserve.md) | 5. Rough Bergomi on the SPY smile: from the posterior to a reserve | Francesco, Alex | 40 |
| [06-slot6-priors-and-misspecification.md](06-slot6-priors-and-misspecification.md) | 6. Changing priors and dealing with misspecified models | Stefan, Jerry | 15 |

A number written ⟨Sn⟩ comes from study Sn ([course/status.md](../../course/status.md)) and is not on screen until [CLAIMS.md](../CLAIMS.md) records the final run.

## Executive summary

**What the tutorial teaches.** A calibrated volatility model is an estimate, and the prices it gives for products nobody quotes inherit its uncertainty. The posterior is that uncertainty, and pricing under each posterior draw turns it into the range the prudent-valuation rules ask for. On a liquid smile the posterior and the classical interval agree, as theory predicts. They can part where the quotes do not determine what the product needs: beyond the last quote, which a quick check on SPY shows; on a thin chain, which slot 5 tests; and for names that quote almost nothing, which a simulated sector shows. For a model priced only by simulation, amortization makes the posterior checkable: its coverage can be tested on a thousand simulated chains, and each real chain gets a proposal for an audit. A chain of checks says when to trust it.

**The arc: one smile, three models.** The whole afternoon works on one data set, the SPY option smile of 21 October 2026 (63 prices read from the closing quotes onto a fixed grid of 7 maturities × 9 moneyness points), and one question: which price to mark for a product the smile does not quote, and with what range. Slot 5 ends on a thin chain, EWG of 22 October 2026, whose 61 quotes, read at 27 design points, cover four maturities up to about six months. Three models of increasing realism carry it, each more expensive to invert:

1. **Black–Scholes** in slot 1: one parameter, a posterior on a grid in a fraction of a millisecond, and a model the smile rejects.
2. **Heston** in slot 4: four parameters, an exact likelihood by Fourier integral, so the network can be checked against the exact posterior. It fits SPY, but only with extreme parameters.
3. **Rough Bergomi** in slot 5: three parameters and the market's forward-variance curve, priced only by Monte Carlo, so the exact posterior costs at least 17 CPU-hours by a random-walk chain, or about 85 CPU-minutes by importance sampling from a Gauss–Newton proposal on a fixed-seed pricer, and the network is checked by an audit of each smile.

Slots 2 and 3 explain the machine between them. Slot 6 changes the prior and the noise model, and tests the model.

**What changed from revision 1, and why.**

- **Slot 4 moves from returns to the smile.** It was GBM on VOO returns, under the real-world measure P, followed in slot 5 by option quotes under the pricing measure Q. That switch changed the data, the measure, the model and the question at once. Slot 4 is now Heston on the SPY smile, so slot 5 changes only the simulator. The exact Heston check moves from slot 5 to slot 4.
- **The returns material is left out.** The VOO notebook, the SVJ lesson and the crash-day prior example are not part of the release (decision D21, §6). This widens the gap against the proposal, which calibrates from returns (§8).
- **Slot 1 is rewritten slide by slide**, with one line of argument: the marking procedure, an example of the problem, what the rules ask for, the posterior in one live example, why richer models make it expensive, and the way around.
- **Slots 2, 3 and 6 have proposals**, each built on the same smile, for Alex's, Stefan's and Jerry's review.

**A feasibility run supports the new slot 4.** The exact Heston posterior on SPY ($\kappa$ = 2 fixed, the bank's prior box) was computed on 23 October in 531 CPU-seconds. Seven of eight optimizer starts spread over the prior box reach the same optimum, and four chains from overdispersed starts agree ($\hat R \le 1.007$). The posterior is tight: $\sigma_v$ = 1.14 ± 0.04 and $\rho$ = −0.63 ± 0.02. No draw satisfies the Feller condition: the vol-of-vol is about twice its Feller limit. The posterior prices variance swaps 0.6 to 1.7 volatility points below SPY's replicated strikes from two weeks to one year, with 90% bands 0.3 to 0.6 points wide. This gives slot 4 a real-data ending and slot 5 its reason to change model. Details are in the slot 4 file; the run is redone as study S8-H before its numbers go on screen.

**What has to happen now.**

1. **The cluster runs** R0 to R10 of [`CLUSTER-RUNS.md`](../../lab/plans/CLUSTER-RUNS.md), starting with the full rough bank (gate G1). The curve prior passed gate G0′ on 24 October.
2. **D1 and D16** (§10): confirm option E for slot 1 and the all-smile arc.
3. **Build lists** are at the end of each slot file. Freezes: slot 4 by **Fri 30 Oct**; slots 1, 2 and 3 by **Wed 4 Nov**; slots 5 and 6 by **Fri 6 Nov**. Full timed run-through **Mon 9 Nov**.

## 1. The message

**In three sentences.** On a liquid smile the posterior and the classical interval agree, as theory predicts. They part where the quotes do not determine what the product needs: a thin chain, maturities beyond the last quote, and names that quote almost nothing. Amortization does not make one posterior much cheaper than a good classical proposal does; it makes the method checkable, by coverage on a thousand simulated chains, and usable across many names. The course states the argument in Lecture 6, Section 6.8, and gathers its evidence in [Lecture 16, Section 16.8](../../course/16-a-thin-chain-ewg.md); the plan's Section 2.6 lists the options to strengthen it.

The message has three beats, each carried by named slots.

| Beat | Set up in | Delivered in |
| --- | --- | --- |
| The range matters, and the rules ask for it. | Slot 1 | Slot 5, as a reserve |
| Amortization makes the range checkable: pay once in training, then seconds per smile for the thousands of posteriors a validation needs, and an audit where one smile matters. | Slot 1, as a cost problem | Slots 2 and 3 explain it; slot 4 builds it; slot 5 measures it |
| Trust comes from checks, not from the network. | Slot 2, as a workflow | Slot 4 against an exact posterior; slot 5 by audit; slot 6 on priors and misspecification |

The audience should leave able to say three things about their own model: whether they need this, what it costs, and how they would check it.

**The message under the message.** Doing this in practice is a chain of choices: the model, the prior, the quote-noise model, a simulated bank on a cluster, the network, the validation, the audit of each smile, the pricing of products, the report. The afternoon shows the whole chain built on a real smile, documented, with its failures shown and handled. The audience should conclude that the team can do this for their models. §7 says where that is made explicit (twice) and where it is only shown.

## 2. The arc: one smile, three models

![The three models of the tutorial and what one likelihood evaluation and one exact posterior cost under each](images/ladder.png)

How to read the figure: for each model, the grey bar is the cost of one likelihood evaluation (pricing the quotes once) and the green bar the cost of one exact posterior, in CPU-seconds on a log scale, on one 4-core machine. The hatched bar is a lower bound, for a random-walk chain; the purple diamond is importance sampling from a Gauss–Newton proposal on SPY's 63 quotes, with common random numbers. Heston's bars are at one rung of its quadrature ladder; a checked evaluation costs at least twice as much. The dashed orange line is the amortized network's cost for 1,000 draws, and the dotted red line the estimated cost of auditing one rough-Bergomi smile with 512 proposals, each after a pilot pricing (⟨S5⟩ measures it). Sources: `costs/costs.json` (C8) and `lab/checks/rough_spy_is.py`; the Black–Scholes bars were timed once with `sbi4finance.calibration.surface` on 23 October and are illustrative, not a study result.

| | Black–Scholes | Heston | Rough Bergomi |
| --- | --- | --- | --- |
| Slot | 1 | 4 | 5 |
| Parameters | σ | $v_0$, $\theta$, $\sigma_v$, $\rho$ ($\kappa$ fixed) | $H$, $\eta$, $\rho$, with the curve $\xi_0$ as an input |
| One likelihood evaluation | about 10 µs per price | 20 ms per smile at one quadrature rung | 5.0 CPU-s per smile |
| One exact posterior | 0.2 ms (vectorized grid) | about 4 CPU-min at one rung, twice that with the accuracy checks | at least 17 CPU-h by a random-walk chain; about 85 CPU-min by importance sampling on SPY's 63 quotes |
| What it does on SPY | Cannot fit two quotes | A root-mean-square residual of about one noise unit (the noise is loose in the wings), with a vol-of-vol about twice its Feller limit and variance swaps below SPY's | Its skew follows a power law in maturity, the shape of SPY's; its variance swaps match the market by construction; its fit to SPY is ⟨S8⟩ |
| The check of the network | — (slot 2's toy, against the grid) | The exact posterior by MCMC | SBC, a second method, and an audit of each smile |

## 3. The six slots

![Storyboard: the tutorial and the minute map of each slot, coloured by presenter](images/storyboard.png)

How to read the figure: the top row is the tutorial, 115 minutes of talks in six slots, with widths proportional to minutes. The rows below are the minute maps of the six slots, from their files. Colour is the presenter. A solid block is computed on stage, a light block is loaded from `results/`, and a white block is a slide or code on screen. The figure is drawn by `tutorial/curriculum/storyboard.py`.

| Slot | The question it answers | Example | What it hands to the next slot |
| --- | --- | --- | --- |
| 1 | How sure is the price a calibrated model gives to what the market does not quote, and how is that uncertainty computed and checked? | Beamer: one running example, Heston on simulated quotes (the fits consistent with the quotes, a product's range, the classical range against the posterior, a fixed parameter, joint draws) | The five steps of every notebook, the case for a posterior, and the idea of amortization |
| 2 | How can a network trained on simulations return the posterior, and when does that pay? | A network learns slot 1's posterior (live); the break-even for Heston and rough Bergomi | The object to build, $q_\phi(\theta \mid y)$, and the workflow that checks it |
| 3 | Which networks represent the posterior, and how do they read the data? | Flow matching carrying noise to SPY's Heston posterior; data shapes and their networks | The exact network configuration of slots 4 and 5 |
| 4 | What does the workflow look like in code, when every step can be checked? | Heston on the SPY smile, against the exact posterior | The same code, a new simulator; Heston's defects on SPY |
| 5 | Where the quotes stop, which price is marked, and how large is the reserve? | Rough Bergomi on the SPY smile, audited; Heston against rough; EWG, a thin chain | A range conditional on the prior, the noise model, the curve and the model |
| 6 | What does the answer depend on, and how is a wrong model found out? | The prior on $H$; the Feller condition; the noise multiplier; four detectors | The close |

The argument runs as one chain. Solid arrows are the main handoffs; dashed arrows are threads that skip a slot.

```mermaid
flowchart LR
  S1["1. The problem<br/>Francesco, 15 min<br/>Black-Scholes on two SPY quotes"] -->|"richer models make the posterior expensive"| S2["2. Amortized Bayes<br/>Alex, 15 min<br/>a network learns slot 1's posterior"]
  S2 -->|"what q must be, how to check it"| S3["3. Architectures<br/>Stefan and Jerry, 10 min<br/>flow matching, summary networks"]
  S3 -->|"the configuration"| S4["4. Heston on SPY in BayesFlow<br/>Stefan and Jerry, 20 min<br/>checked against the exact posterior"]
  S4 -->|"same smile and code, new simulator"| S5["5. Rough Bergomi on SPY<br/>Francesco and Alex, 40 min<br/>posterior, audit, reserve"]
  S5 -->|"every range is conditional"| S6["6. Priors and misspecification<br/>Stefan and Jerry, 15 min"]
  S1 -.->|"the posterior is conditional on the model"| S6
  S4 -.->|"Heston's variance swaps below SPY's"| S6
```

**State of the notebooks on 28 October.** The notebooks of slots 1 to 5 run headless and offline in the tutorial's
environment. Slot 4 loads the BayesFlow 2.0 Heston network until the 2.1 candidate is exported (decision D20). Slot 5
shows the pilot's results, with a "pending" card wherever the full study is missing. Slot 6 has no notebook yet.

## 4. Rules for every slot

1. **One owner per concept.** The other slots refer back in one sentence (§5).
2. **Handoffs.** Each slot opens with the previous slot's handoff and closes with its own.
3. **One code path.** Slots 4 and 5 follow the same steps: simulator, adapter, network, training, checks, use. Slot 5 changes the simulator and the checks, and nothing else.
4. **Numbers.** A number goes on screen only from CLAIMS.md, and only from a final run. A final result that misses its criterion is shown as missed ("not met"). A pilot value is never quoted as a result; until the final run exists the slide shows ⟨Sn⟩.
5. **Live cells.** Every live cell has a cached fallback that loads from `results/` if the computation fails or runs past its time budget. A cell that loads a network runs once before the slot starts.
6. **Wording.** Seconds, never milliseconds, for the network; "at least 17 CPU-hours" for the exact rough posterior by a random-walk chain, always with "about 85 CPU-minutes by importance sampling" beside it, a figure measured on a fixed-seed pricer (more with a seed per proposal); "the design's noise", never "SPY's bid–ask". Slot 5's file lists the rest.

## 5. Who owns which concept

| Concept | Owner | Used in |
| --- | --- | --- |
| Calibration as estimation; the posterior; propagation by pricing each draw; conditionality on the model | 1 | 2, 4, 5, 6 |
| The rules: Regulation (EU) 2016/101 Art. 9 and Art. 11, IFRS 13, PRA SS1/23 | 1 | 5 |
| The cost ladder and the 2×2 table | 1 | 2, 5 |
| The amortization identity; the economics and the break-even | 2 | 4, 5 |
| The amortized workflow; SBC, coverage and contraction as concepts | 2 | 4, 5 |
| The amortization gap and its correction by importance sampling | 2 | 4 (exact likelihood), 5 (Monte Carlo likelihood) |
| Normalizing flows and flow matching; the sampler | 3 | 4, 5 |
| Summary networks by data shape; conditions and context variables | 3 | 5 ($\xi_0$ as a condition), 6 (summary-space detection) |
| BayesFlow's objects in code: simulator, adapter, networks, workflow, diagnostics | 4 | 5, 6 |
| The prior predictive check and the posterior predictive check on a smile | 4 | 5, 6 |
| Products under the posterior; the reserve; the classical baseline | 5 | 6 |
| Changing the prior and the noise model; detecting misspecification; across-model reporting | 6 | — |

## 6. The take-home track

The proposal calibrates from returns and propagates into value-at-risk. The afternoon does not. A take-home track was to carry that material:

| Notebook | What it teaches | Draft |
| --- | --- | --- |
| `extras/returns_bayesflow_basics.py` | BayesFlow on 1,000 VOO returns under GBM: the same workflow as slot 4, with VaR and expected shortfall | Branch `feat/tutorial-slot4-bayesflow-basics` |
| `extras/svj_from_returns.py` | Stochastic volatility with jumps from returns, with a particle-filter audit | The SVJ lesson (#38), branch `feat/tutorial-slot5-svj-advanced` |
| `extras/multifractal_returns.py` | The multifractal model of asset returns | None on a branch |
| `extras/crash_day_priors.py` | How much the prior decides the probability of a 7% loss day | Branch `feat/tutorial-slot6-crash-risk` (#42) |
| `extras/betas_across_stocks.py` | Cross-sectional shrinkage of betas | Branch `feat/tutorial-slot1-intro` (option D) |

Decision D21 (28 October): the track is not part of the release.

## 7. Where the consultancy message lands

The message is shown in every slot and said in only two places.

**Shown:**

- real data end to end: the SPY chain of 21 October 2026;
- documentation: the sixteen-lecture course in `course/`, take-home;
- a ledger of claims: every number on screen traced to a study and a file;
- failures shown and handled: the Heston network alone short of the exact-likelihood reference on one smile in five, the rough pilot's SBC failure, the audit's flag rate, the places where the classical route is as good;
- infrastructure: a bank of 112,000 Monte-Carlo-priced smiles generated on a cluster (gate G1), and trained networks shipped as checkpoints (Heston's now, rough Bergomi's after gate G2);
- the people: BayesFlow's lead developer and core developers, and hedge-fund experience on the team; slot 1 spends twenty seconds on this.

**Said, twice:**

1. The last slide of slot 5 (Francesco): "What it takes", the chain of §1, then one line and contacts: "A model only simulation can price, or a range you must report? Talk to the authors: ⟨contacts⟩."
2. The participant README: a short section, "Working with the authors".

## 8. Gaps against the accepted proposal

The audience has read the proposal. Each promise the curriculum does not keep needs a handling, decided before the freeze.

| The proposal promises | The curriculum | Proposed handling |
| --- | --- | --- |
| Calibration from return series (its Figure 1, "simulated pairs (parameters, return series)"), with joint returns-and-options calibration named as the follow-on edition | Every slot works on an option smile | Decision D16. Slot 1 says in one line why the afternoon uses options and that returns are left out (decision D21) |
| Propagation into volatility forecasts and value-at-risk (learning outcome 4) | Propagation into product prices and a reserve | Partly kept; VaR and volatility forecasts are left out (decision D21) |
| Five model families: Heston, jump diffusion, GARCH, rough, multifractal | Black–Scholes, Heston, rough Bergomi; Heston with jumps as one slide | GARCH and the multifractal model are not covered: say so on slot 1's last slide |
| Two notebooks live and one taken home, in student and solved versions, runnable on Colab | Live notebooks in slots 1, 2, 4, 5 and 6 | Decision D10 for the number; decision D17 for the student and solved versions; on Colab, the tutorial installs with pip (`tutorial/README.md`) |
| Half a day, 3.5 hours: principles (1 h with a break), a worked example hands-on (1.5 h with a break), advanced concepts and a second example (1 h with a 15-minute Q&A) | 115 minutes of talks; about 95 minutes of the 3.5 hours unassigned | Decision D12: assign the time to breaks, the Q&A and exercises after slots 4 and 5 |
| A close with "a broader view of the field and recent developments" | Slot 6's last minute | Slot 6's close names recent developments in one slide, or the Q&A opens with them |
| "Trained checkpoints for every example" | Heston's network exists; the rough network and slot 6's networks are pending | Ship each by its slot's freeze, or say which examples train at home |
| Four organizers teaching | Jerry Huang has the same role as Stefan | Decision D18 |
| "Attendees calibrate real market data and sign off on the diagnostics" | Presenters do it on SPY | Decision D12: exercises after slots 4 and 5 |
| LLM-based verification agents, used hands-on | Not in any slot | Decision D11: a two-minute demonstration in slot 4's check step, or drop it and say so |
| "Special emphasis" on learning from series of different scales, dimensions, lengths and degrees of missingness; autoregressive samplers | Not in any slot; the smile is a fixed grid | The largest content gap after returns. Slot 3 can show the set network for real chains as the answer for quotes; time series are not covered. Say so rather than imply coverage |
| "Full posterior distributions in milliseconds" | 1.4 s for 1,000 draws | Say seconds |
| Mis-specification diagnostics (learning outcome 2) | Slot 6 | Kept |
| Read identifiability off the posterior (learning outcome 3) | Slots 4 and 5 | Kept |

## 9. Gates and dates

| Date | What | Owner |
| --- | --- | --- |
| Sat 24 Oct | G0′ passed: the rough bank's curve prior covers SPY's curve | Done |
| Wed 28 Oct | G1: the full rough bank and its quality (S1) | Alex |
| Fri 30 Oct | G2: SBC of the rough network on the full bank (S2); freeze of slot 4; S8-H (Heston on SPY) accepted | Alex; Stefan and Jerry |
| Tue 3 Nov | G3: the audit on held-out smiles, with the corrected likelihood (S5) | Alex |
| Wed 4 Nov | G4: the posterior predictive check on SPY (S8); freezes of slots 1, 2 and 3 | Francesco; Alex; Stefan and Jerry |
| Fri 6 Nov | Freezes of slots 5 and 6 | Francesco, Alex, Stefan, Jerry |
| Mon 9 Nov | Full timed run-through | All |
| Wed 11 Nov | Release tag; participant README | Francesco |

## 10. Decisions

| # | Decision | Recommendation | By |
| --- | --- | --- | --- |
| G0 | Widen the curve prior of the rough bank to cover SPY's forward-variance curve | Done: a power law times a random walk, which passed G0′ | Sat 24 Oct |
| D1 | Slot 1 option | E, as in the slot 1 file; option D to the take-home track | Fri 23 Oct |
| D16 | The arc: every slot on the SPY smile, Heston in slot 4, returns as a take-home track | Yes | Fri 23 Oct |
| D2 | Slot owners and durations | As in the agenda above | Fri 23 Oct |
| D10 | Presentation format | Marimo notebooks for slots 1, 2, 4, 5 and 6, slides for slot 3, a cached fallback in every live cell | Fri 30 Oct |
| D11 | The LLM verification agent promised in the proposal | A two-minute demonstration in slot 4, or drop it and say so | Fri 30 Oct |
| D12 | Hands-on time beyond the talks | Exercises after slots 4 and 5 | Fri 30 Oct |
| D13 | Slot 5's title and presenter split | Proposed: "Where the quotes stop: which price do you mark?", in place of "The smile fits many calibrations" ([v2-plan](../../course/v2-plan.md), Section 9); Alex takes sections 4 and 5 | Fri 30 Oct |
| D14 | Threshold for gate G3 | One smile in three flagged | Fri 30 Oct |
| D15 | Slot 6's sensitivity-aware network and summary-space detector | Build the first; the second if time allows | Fri 30 Oct |
| D17 | Student and solved versions of the notebooks | Solved versions on stage; student versions with the exercises blanked, generated from the solved ones | Fri 30 Oct |
| D18 | Jerry Huang's role | Agreed on 28 Oct: the same as Stefan's | Done |
| D19 | Slot 5's rough-Bergomi task: the known-curve task (three unknowns) or the inferred-curve task (six) | Open: the numerical results decide; the slot 5 notebook follows the known-curve pilot until then | Fri 6 Nov |
| D20 | BayesFlow version of the shipped networks | Agreed on 28 Oct: every network shipped to participants moves to BayesFlow 2.1 (the Heston candidate once it is packaged, the rough-Bergomi network, slot 2's toy network retrained), and the tutorial's pin moves with them | Wed 11 Nov |
| D21 | The take-home returns track (SVJ, the multifractal model, GBM from returns, crash-day priors, betas across stocks) | Agreed on 28 Oct: not part of the release | Done |
| D22 | How much run evidence lives in git | Open: compact records only (README, summary JSON, key figures), or the raw per-case outputs and logs too; the pilot results of the full-surface work are about 41 MB | Before that work merges |
| D23 | What participants receive | Agreed in principle on 28 Oct: a separate small repository holding `tutorial/` without the curriculum, created at the release. Six links need a public target first: the "Evidence and code" footers of the five notebooks point to the curriculum and the course, and the participants' README points to this repository, which is private. Either the course is published with the release, or the links go | Wed 11 Nov |

## 11. Sources

- [course/status.md](../../course/status.md): the studies. [CLAIMS.md](../CLAIMS.md): the status of every number.
- [The course](../../course/README.md): the long version of slots 1, 4 and 5.
- The proposal: *Amortized Bayesian Calibration of Volatility Models*, ICAIF '26 tutorial proposal.
- The feasibility run of 23 October: `lab/checks/heston_spy_exact.py` and [data/heston-spy-exact.json](../assets/data/heston-spy-exact.json).
- Branches with other slot drafts: `feat/tutorial-slot1-intro`, `feat/tutorial-slot4-bayesflow-basics`, `feat/tutorial-slot6-crash-risk`.

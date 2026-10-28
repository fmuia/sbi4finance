# Status of the studies

[Course index](README.md)

24 October 2026. Every result on the full rough-Bergomi bank is pending: runs R0 to R10 of [`lab/plans/CLUSTER-RUNS.md`](../lab/plans/CLUSTER-RUNS.md) produce them, for about 930 CPU-hours and two GPU jobs. "Pilot" is the bank of 2,200 rough-Bergomi surfaces and a network trained on 1,600 of them. Claim IDs and their status are those of [`tutorial/CLAIMS.md`](../tutorial/CLAIMS.md). A quick check is a script in `lab/checks/`, not an accepted study, and is not registered there.

| Item | Question | Result so far | Status | Run | Claims |
| --- | --- | --- | --- | --- | --- |
| Argument, part 1 | Do the posterior and the classical answer agree where the quotes determine what the product needs, and part where they do not? | On SPY's 63 quotes the delta-method reserves are 0.93 to 1.16 times the posterior's, 0.94 to 1.13 at the 90% quantile (Lecture 15, Section 15.4). For the three products that depend mostly on the curve beyond 2.25 years, flat extrapolation gives 0.32, 0.30 and 0 of the posterior's 90% range (Section 15.6) | Thin-chain part pending: S10, S10-E, S11 | R7, R8, R9 | – |
| Argument, part 2 | Does amortization make the posterior checkable: its coverage tested on many simulated chains, and an audit on each real one? | One posterior: at least 16.9 CPU-hours by a chain, about 85 CPU-minutes by importance sampling on SPY with a fixed-seed pricer; the network 3.6 CPU-seconds for 1,000 draws (Lecture 14, Section 14.5) | Validation of the rough network pending: S2, S2-T, S5 | R3, R4, R5 | C8 (verified) |
| S1 | Cost and quality of the training bank: Monte Carlo error $s_c \le 0.25\,\sigma_c$ on every quote | Heston: 60,000 surfaces, largest error 0.056 against a budget of 0.1. Rough pilot: every quote within budget, median 0.04, largest 0.250; 3.3 CPU-hours per 1,000 surfaces. The full bank's curve prior passed gate G0′ on a 200-surface mini-pilot: 11.5 CPU-seconds a surface, no quote above the budget | Heston and pilot done; full rough bank of 112,000 surfaces pending, about 360 CPU-hours | R0, R1, R2 | – |
| S1-G | The bank's time-grid bias, in units of the quote noise | 12 mini-pilot surfaces: a grid twice as fine moves a quote by a median of 0.02 of SPY's design noise sd, 0.15 at the 90th percentile and 1.0 at most | Full bank pending | R2 | – |
| S2 | Calibration and identification of the liquid network: SBC inside the simultaneous band for every target on 2,000 surfaces | Heston: met; 95% coverage 93.5% to 95.2%. Rough pilot, 250 smiles: $H$ and $\eta$ inside the band, $\rho$ and $\rho\eta$ outside; 95% coverage 92.0% to 96.4% | Heston done; full rough network pending (gate G2) | R3 | C2 (pilot, partly); C7 (Heston verified; rough pilot not met) |
| S2-T | The thin network on EWG's layout, which infers the curve: SBC of the ten targets and $\rho\eta$ at noise scales 1, 0.5 and 2 | – | Pending (gate G2-T at $s = 1$) | R4 | – |
| S3 | Agreement with the exact Heston posterior: means within 0.3 sd and 90% widths within 25%, on 50 smiles | Network alone: 40 of 50. Network with its audit: agrees with independent chains on the 4 smiles that have them | Done | – | C5 (not met for the network alone) |
| S4 | Agreement with the neural-pricer route | Heston rehearsal: pricer error 0.03 $\sigma_c$; the pricer route matches the exact posterior on 50 of 50; the network matches the pricer route on 84% of 50 | Heston done; rough pending | R10 | C6 (not met for the network alone) |
| S5 | The audit of the liquid network: efficiency, $\hat k$, flag rate, cost | Rough pilot, 40 smiles: efficiency median 14%; 9 (22.5%) flagged at $\hat k > 0.7$, 13 (32.5%) above the sample-size threshold 0.58; 425 CPU-seconds per smile. That audit priced all proposals of a surface on one seed, which biases the reweighted posterior | Pilot must be rerun; full audit on 200 surfaces, each proposal on its own seed, pending | R5 | C7 (audit part, pilot) |
| S6 | The classical route: Gauss–Newton coverage, optima, regularization | Heston: Gauss–Newton coverage 87% to 92%, network posterior 87% to 90%; one optimum on 97% of smiles. Rough pilot, 8 smiles: one optimum on every smile. Earlier classical fits started at the network's median; the full-bank runs start from the quotes and the prior box | Heston done; full rough pending | R10 | C4 (pilot, not met); C11 (verified) |
| S7 | Products and reserves; Heston against rough Bergomi | Heston and rough pilot, 3 smiles each (Lecture 15, Sections 15.5 and 15.7); forward-start gap between the models 4% to 8% | Heston done; full rough pending | R10 | C3 (pilot); C9 (pilot); C10 (verified) |
| S8 | SPY of 21 October 2026: posterior, audit, predictive check, products | Data done: skew exponent −0.32; every grid point read from quotes. SPY's curve lies outside the pilot's curve prior and inside the full bank's | Posterior, audit, predictive check and products pending (gate G4: $\hat k \le 0.7$) | R6 | C1 (verified) |
| S9 | Costs on one machine, every route | Done with the pilot networks: a rough likelihood evaluation 5.0 CPU-seconds; the exact posterior at least 16.9 CPU-hours by a chain (extrapolated); 1,000 network draws 3.6 CPU-seconds | Training time and full-bank costs pending | R10 | C8 (verified) |
| S10 | Breaches of five methods' one-sided 90% bounds on 1,000 simulated EWG chains, truths from the prior, three laws for the unquoted level | – | Pending, about 180 CPU-hours | R8 | – |
| S10-E | The same on 300 chains whose truths come from EWG's posterior | – | Pending, about 55 CPU-hours; needs S11 | R9 | – |
| S11 | EWG's posterior at noise scales 1, 0.5 and 2: products, classical fits, predictive check | – | Pending | R7 | – |
| Quick check: Heston on SPY | The exact Heston posterior against Gauss–Newton on 63, 36 and 15 quotes (`heston_checks.py`) | Delta-method ranges 0.92 to 1.21 times the exact posterior's (Lecture 11, Section 11.4) | Quick check | – | – |
| Quick check: rough Bergomi on SPY, fixed-seed pricer | The posterior of a fixed-seed pricer by importance sampling from a Gauss–Newton proposal, against Gauss–Newton (`rough_spy_is.py`) | $\hat k = -0.64$ and efficiency 0.45 with 1,024 proposals; Gauss–Newton 90% intervals 1.03 to 1.13 times the posterior's (Lecture 15, Section 15.4) | Quick check | – | – |
| Quick check: local identification | The Laplace posterior with $\xi_0$ known or uncertain, on three quote sets (`rough_checks.py`) | Contraction of $\rho$ on 15 quotes: 0.61 to 0.65 with $\xi_0$ known, 0.12 to 0.16 at 10% per level (Lecture 13, Section 13.11) | Quick check | – | – |
| Quick check: the long end | Flat extrapolation beyond one year against a Laplace posterior with a prior beyond SPY's LEAPs (`long_end/`) | For the three products that depend mostly on the curve beyond 2.25 years, flat gives 0.32, 0.30 and 0 of the posterior's 90% range (Lecture 15, Section 15.6) | Quick check | – | – |
| Quick check: Black–Scholes sector | Pooling across names (`bs_hierarchy.py`) | With 2 liquid and 6 illiquid names, the 90% intervals cover 88.0% under hierarchical Bayes and 59.5% under the proxy (Lecture 8, Section 8.10) | Quick check | – | – |
| Front-month case | SPY's front three months used as a thin market, with an 80-chain validation | – | Withdrawn: SPY's tight noise, and every method given SPY's curve to one year from the full chain (Lecture 16, Section 16.3) | – | – |
| 15-quote and three-times-noise cases | Rough Bergomi on SPY when the quotes say less | – | Withdrawn: every method given SPY's curve from the full chain (Lecture 16, Section 16.3) | – | – |

## The networks and banks in training

28 October 2026. The full-surface work of [PR #48](https://github.com/fmuia/sbi4finance/pull/48) trains the networks the
studies above will run on; its detailed numerical status is
[`docs/collaborator-numerical-status.md`](https://github.com/fmuia/sbi4finance/blob/feat/full-surface-calibration/docs/collaborator-numerical-status.md)
on that branch.

| Piece | State |
| --- | --- |
| Heston network | A BayesFlow 2.1 candidate (100,000 training smiles, epoch 500) passed its frozen calibration check on 2,000 unused smiles, and 12 of 12 variance-product checks. The notebooks load the BayesFlow 2.0 network trained on 50,000 smiles until it is exported. |
| Rough Bergomi, known curve ($H$, $\eta$, $\rho$ given the 7-level forward-variance curve) | The 20,000-smile bank is complete and the first networks are trained. Comparisons with reference posteriors still disagree on some smiles, partly from the reference emulator's error and the time grid. The next step is a 50,000-smile bank and refits. |
| Rough Bergomi, inferred curve (six unknowns) | 9,925 of 20,000 training smiles generated; paused while the known-curve checks run. |
| Sparse option chains (EWG) | Deferred until after the release ([`lab/plans/SPARSE-QUOTE-CASE-PLAN.md`](../lab/plans/SPARSE-QUOTE-CASE-PLAN.md)). |

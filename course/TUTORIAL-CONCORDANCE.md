# Tutorial results in the lectures

Working document for the companion lecture notes. The lectures keep their own order and are not meant to follow the
tutorial slot by slot. This file says where in the existing lectures each tutorial result fits best, and lists the
results the lectures do not yet carry.

Notebooks are those in [`tutorial/notebooks/`](../tutorial/notebooks/). Slot 6 has no notebook, so its rows follow
[its curriculum file](../tutorial/curriculum/06-slot6-priors-and-misspecification.md).

## Where to add what the lectures lack

In lecture order. Each row is a tutorial result that no lecture shows yet, and the section that should take it.

| Lecture section | Tutorial result to add | From |
| --- | --- | --- |
| 8.3 A first posterior | The posterior of $\sigma$ from one quote, SPY's at-the-money one-month price, with width equal to the quote noise over the vega; the 90% range of the one-month put it implies. It is a one-quote opening before the 63 synthetic quotes. | Slot 1, §1 |
| 8.4 A posterior for a wrong model | The two-quote version: the at-the-money price and the put. The two likelihoods do not overlap, the posterior misses the put by many noise units, and only a multiple of the noise reconciles them. It is a lead-in to the 63-quote check. | Slot 1, §2 |
| 8.5 Identification read off the posterior | The noise weight $w$ on the wings (at $w = 0.3$ the valley closes and the two methods agree), and the quote at $x = 2$, the product beyond the data whose range the ellipse misplaces. | Slot 1, §4 |
| 8.11 Where the posterior and the classical answer part | The table "where this happens on real data": Heston on 15 SPY quotes, and rough Bergomi on 15 SPY quotes with $\rho$'s interval past the bound. | Slot 1, §4 |
| 9.2 Neural posterior estimation | The first worked example after the derivation: the Black–Scholes posterior learned from 40,000 simulated pairs (prior 8–40%), against the grid posterior on the SPY quote; then 500 quotes in one call. | Slot 2, §§1–3 |
| 9.5 Sampling with a finite number of steps | The width ratio against the number of Euler steps on SPY's Heston posterior, in the paragraph "Why this matters for the tutorial". | Slot 3, §4 |
| 9, new section before the summary | The seven-step workflow. Lectures 12 and 15 then follow it, and slot 5's closing chain (§10) is the same list for rough Bergomi. This is the one addition to the table of contents. | Slot 2, §5; slot 5, §10 |
| 9, opening or references | The lineage of simulation-based inference: ABC, likelihood-free MCMC, synthetic likelihood, neural posterior estimation, BayesFlow, flow matching. | Slot 2, §6 |
| 11.4 The posterior and the classical interval on SPY | $\kappa$ free on 15 quotes: the Gauss–Newton interval −0.6 to 5.8 includes negative mean reversion, the posterior runs 0.7 to 6.3, skewed, and the 2-year put's range is 21% wider. The exact posterior of $\kappa$ on 63 quotes (about 3.8) and the gain in $\chi^2$ against $\kappa = 2$. | Slot 1, §3 and §4; slot 4, §6 |
| 12.1 What the network reads and returns | The simulator code (`prior()`, `smile()`, `bf.make_simulator`) before `build_workflow`. The flow-matching figure: 2,000 noise points carried to SPY's posterior, shown at several times $\tau$. Add a pointer to it from 9.4. | Slot 4, §1; slot 3, §1 |
| 12.5 Agreement with the exact Heston posterior | SPY as the closing case: network, exact chain and Gauss–Newton side by side ("three routes, one answer"). 11.4 already has the last two; the network column is new. | Slot 4, §5 |
| 12.10 Real chains: quotes on a varying set | The data-shape table: no summary network on the grid, a set network for chains, a time-series network for returns, a learned summary space for misspecification (Schmitt et al. 2024). | Slot 3, §2 |
| 14.4 Who needs many posteriors | The Heston break-even, and SBC on 2,000 smiles priced both ways: exact Heston chains against the network. Figure 14.4 already plots the exact Heston line. | Slot 2, §4 |
| 14.5 The alternatives to the exact chain | What finance uses instead: likelihood expansions (Aït-Sahalia 2002), indirect inference (Gouriéroux, Monfort and Renault 1993), particle MCMC (Andrieu, Doucet and Holenstein 2010). | Slot 2, §6 |
| 16.6 Model error | Slot 6's four detectors as one table: the input check, the posterior predictive check, the audit, and the learned summary space. The fourth is not in the course. | Slot 6 |

**Already in the lectures:**
- slot 1, §§5 and 7 (8.10; Lecture 14);
- slot 3, §3 (12.1);
- slot 4, §§2–4 (12.1–12.5);
- slot 5, §§1–9 (Lectures 6, 13, 14 and 15);
- slot 6's reweighting of the prior (12.9).

These need only a check that the numbers agree with the notebooks.

**No change to the table of contents is needed** beyond the workflow section in Lecture 9. Lecture 16 (EWG) has no
counterpart in the notebooks. The slot 5 curriculum plans one.

## Stale references in the notebooks

These are for the tutorial, not the lectures:

- slot 1, §4 cites "Lecture 9.7" for the toy, which is in 7.7 and 8.5;
- slot 1, §4 cites "Lecture 10, below" for the coverage near the bound, which is in 8.6;
- the model card of slot 4 cites "Lecture 6" for Heston's Fourier pricing, which is in 10.5–10.6.

## Reference: notebook sections against lecture sections

**Match.** *Full*: the lecture section teaches what the notebook section shows, on the same data. *Partial*: the same
idea, but on other data or settings, or spread across sections. *None*: no lecture section covers it.

### Slot 1. The problem: Bayes for volatility models

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · A quote is an observation, so a calibration is an estimate | 5.4, 7.1–7.3, 8.1, 8.3, 8.7; 6.6 for the rules | Partial | The notebook takes a Black–Scholes posterior from SPY quotes and prices a one-month put. 8.3 uses 63 *synthetic* quotes and a prior of 5–60%. |
| 2 · The posterior also judges the model | 8.4 | Full | |
| 3 · With a model that fits a liquid surface, the classical interval is the posterior | 11.3–11.6; Bernstein–von Mises in 8.6 | Full | The 15-quote case is in 11.4, the 200 smiles in 11.6. |
| 4 · Why Bayes, case 1: a posterior that is not an ellipse | 7.7, 8.5 | Partial | The notebook cites "Lecture 9.7"; the toy is in 7.7 and 8.5. The notebook adds a noise slider $w$; the lectures fix $w = 1$, and they do not price the quote at $x = 2$. |
| 5 · Why Bayes, case 2: names that must borrow from their peers | 8.10 | Full | |
| 6 · What the posterior buys a bank | 6.6, 6.7, 8.7; 15.5, 15.6 | Partial | The notebook previews the long end, taught only in Lecture 15. |
| 7 · What the exact posterior costs | 14.1–14.5 | Full | |

### Slot 2. What is amortized Bayes?

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · The toy: slot 1's posterior, learned | 9.1, 9.2 | Partial | The lectures' toys are $y = \theta^3/4 + \dots$ (Figure 9.2) and the toy smile (Figure 9.1). No lecture learns the Black–Scholes posterior from one SPY quote with a prior of 8–40%. |
| 2 · The network against the exact posterior | 9.2 | Partial | As above. |
| 3 · Amortization: many data sets, one call | 9.1, 9.2 | Partial | The lectures do not show the 500-quote, one-call experiment. |
| 4 · When it pays | 14.4 | Partial | 14.4 gives the rough-Bergomi break-even. The notebook's Heston break-even has no lecture section. |
| 5 · The workflow, and SBC | 9.7, 9.8; Lecture 12 as a whole | Partial | The seven-step workflow is never listed as such in a lecture. |
| 6 · Where it comes from | References of Lectures 9 and 14 | Partial | No lecture section reviews the SBI literature against finance's alternatives. |

### Slot 3. Network architectures

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · Flow matching: noise carried to the posterior | 9.3, 9.4 | Partial | The lectures show it on a two-Gaussian mixture. The notebook uses the trained Heston network on SPY. |
| 2 · Reading the data | 12.1, 12.10 | Partial | Set networks appear in 12.10. Time-series and learned-summary networks have no lecture section. |
| 3 · The configuration slot 4 trains | 12.1, 12.2 | Full | |
| 4 · How many steps does the sampler need? | 9.5, 9.6 | Partial | The lectures use a toy target. The notebook's reference is the exact Heston posterior on SPY. |

### Slot 4. A worked example in BayesFlow: Heston on the SPY smile

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · Step 1: a simulator that draws | 10.3, 10.5, 10.6; 12.1 | Partial | No lecture shows `prior()`, `smile()` and `bf.make_simulator`. |
| 2 · Step 2: the bank, the adapter, the network | 12.1, 12.2 | Full | |
| 3 · Step 3: train | 12.2 | Full | |
| 4 · Step 4: check before you trust | 12.3, 12.4 | Full | |
| 5 · Step 5: use it on SPY | 12.5; 11.3 | Full | |
| 6 · What the posterior says about Heston | 10.2 (Feller), 10.4 (variance swap), 11.4, 12.8 | Partial | Spread over three lectures. The mean reversion near 3.8 has no single home. |

### Slot 5. Rough Bergomi on the SPY smile: from the posterior to a reserve

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · The question | 6.1–6.3, 6.6 | Full | |
| 2 · Why rough Bergomi | 13.1, 13.10; 10.8 | Full | |
| 3 · What the exact posterior costs | 14.1–14.3 | Full | |
| 4 · The amortized posterior for SPY | 15.1, 15.3 | Full | |
| 5 · Is it right without an exact answer? | 15.2; 12.6, 12.7; 8.9 | Full | |
| 6 · The classical route on the same smiles | 15.4; 7.6 | Full | |
| 7 · From posterior to prices: the range and the reserve | 15.5; 13.12; 8.7 | Full | |
| 8 · The long end: where the reserve comes from | 15.6 | Full | |
| 9 · Heston against rough Bergomi on the same smile | 15.7 | Full | |
| 10 · What the answer assumes, and what it takes | 8.12, 16.6, 16.7 | Partial | The seven-step chain matches no lecture section. |
| *(none)* | Lecture 16 (EWG) | None | The curriculum gives EWG "section 9" of the slot; the notebook has no EWG section and its section 9 is Heston against rough Bergomi. |

### Slot 6. Changing priors and dealing with misspecified models (curriculum only)

| Curriculum piece | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| Reweighting to a new prior | 12.9 | Full | |
| Sensitivity-aware amortization | 12.9 | Full | One paragraph. |
| The four detectors: input check | 13.4, 15.3 (the curve prior against SPY's curve) | Partial | Not named as a detector. |
| The four detectors: posterior predictive check | 12.8, 15.3 | Full | |
| The four detectors: the audit | 12.7, 15.2 | Full | |
| The four detectors: learned summary space | None | None | Schmitt et al. (2024) is not in the course. |
| Optional: pooling across names | 8.10 | Full | |
| Optional: the prior beyond the last quote | 15.6 | Full | |

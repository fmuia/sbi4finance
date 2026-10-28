# Tutorial and lectures: a section-by-section concordance

Working document for turning the course into companion lecture notes for the tutorial, for arXiv. The aim is that every
section of the tutorial is recognizable in the lectures. This file maps each notebook section to the lecture sections
that teach it, and lists what does not yet match.

State: drafted on the branch stacked on PR #49, before any lecture is edited. Notebooks are those in
[`tutorial/notebooks/`](../tutorial/notebooks/); slot 6 has no notebook, so its rows follow
[its curriculum file](../tutorial/curriculum/06-slot6-priors-and-misspecification.md).

**Match.** *Full*: the lecture section teaches what the notebook section shows, on the same data. *Partial*: the same
idea, but on other data or settings, or spread across sections. *None*: no lecture section covers it.

## Slot 1. The problem: Bayes for volatility models

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · A quote is an observation, so a calibration is an estimate | 5.4, 7.1–7.3, 8.1, 8.3, 8.7; 6.6 for the rules | Partial | The notebook takes a Black–Scholes posterior from SPY quotes and prices a one-month put. 8.3 uses 63 *synthetic* quotes and a prior of 5–60%. |
| 2 · The posterior also judges the model | 8.4 | Full | |
| 3 · With a model that fits a liquid surface, the classical interval is the posterior | 11.3–11.6; Bernstein–von Mises in 8.6 | Full | The 15-quote case is in 11.4, the 200 smiles in 11.6. |
| 4 · Why Bayes, case 1: a posterior that is not an ellipse | 7.7, 8.5 | Partial | The notebook cites "Lecture 9.7"; the toy is in 7.7 and 8.5. The notebook adds a noise slider $w$; the lectures fix $w = 1$, and they do not price the quote at $x = 2$. |
| 5 · Why Bayes, case 2: names that must borrow from their peers | 8.10 | Full | |
| 6 · What the posterior buys a bank | 6.6, 6.7, 8.7; 15.5, 15.6 | Partial | The notebook previews the long end, taught only in Lecture 15. |
| 7 · What the exact posterior costs | 14.1–14.5 | Full | |

## Slot 2. What is amortized Bayes?

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · The toy: slot 1's posterior, learned | 9.1, 9.2 | Partial | The lectures' toys are $y = \theta^3/4 + \dots$ (Figure 9.2) and the toy smile (Figure 9.1). No lecture learns the Black–Scholes posterior from one SPY quote with a prior of 8–40%. |
| 2 · The network against the exact posterior | 9.2 | Partial | As above. |
| 3 · Amortization: many data sets, one call | 9.1, 9.2 | Partial | The lectures do not show the 500-quote, one-call experiment. |
| 4 · When it pays | 14.4 | Partial | 14.4 gives the rough-Bergomi break-even. The notebook's Heston break-even has no lecture section. |
| 5 · The workflow, and SBC | 9.7, 9.8; Lecture 12 as a whole | Partial | The seven-step workflow is never listed as such in a lecture. |
| 6 · Where it comes from | References of Lectures 9 and 14 | Partial | No lecture section reviews the SBI literature against finance's alternatives. |

## Slot 3. Network architectures

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · Flow matching: noise carried to the posterior | 9.3, 9.4 | Partial | The lectures show it on a two-Gaussian mixture. The notebook uses the trained Heston network on SPY. |
| 2 · Reading the data | 12.1, 12.10 | Partial | Set networks appear in 12.10. Time-series and learned-summary networks have no lecture section. |
| 3 · The configuration slot 4 trains | 12.1, 12.2 | Full | |
| 4 · How many steps does the sampler need? | 9.5, 9.6 | Partial | The lectures use a toy target. The notebook's reference is the exact Heston posterior on SPY. |

## Slot 4. A worked example in BayesFlow: Heston on the SPY smile

| Notebook section | Lecture sections | Match | Notes |
| --- | --- | --- | --- |
| 1 · Step 1: a simulator that draws | 10.3, 10.5, 10.6; 12.1 | Partial | No lecture shows `prior()`, `smile()` and `bf.make_simulator`. |
| 2 · Step 2: the bank, the adapter, the network | 12.1, 12.2 | Full | |
| 3 · Step 3: train | 12.2 | Full | |
| 4 · Step 4: check before you trust | 12.3, 12.4 | Full | |
| 5 · Step 5: use it on SPY | 12.5; 11.3 | Full | |
| 6 · What the posterior says about Heston | 10.2 (Feller), 10.4 (variance swap), 11.4, 12.8 | Partial | Spread over three lectures. The mean reversion near 3.8 has no single home. |

## Slot 5. Rough Bergomi on the SPY smile: from the posterior to a reserve

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

## Slot 6. Changing priors and dealing with misspecified models (curriculum only)

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

## What the mapping shows

1. **The lectures never name the tutorial.** Only Lectures 11, 15 and 16 mention a slot, once each. The
   curriculum names the lectures behind each slot, but only by lecture, not by section.
2. **Order.** The tutorial's order is close to the lectures' order from Lecture 6 onward, with two exceptions. Slot 1
   uses Lectures 11 and 14 before Lectures 9, 10 and 13. Slot 3 uses the Heston network of Lecture 12 to explain the
   flow matching of Lecture 9.
3. **The weakest matches are slots 2 and 3.** The lectures teach the same ideas on toy targets. The notebooks use the
   Black–Scholes quote of slot 1 and the Heston network of slot 4.
4. **Material in the tutorial and not the lectures:**
   - the Heston break-even;
   - the seven-step workflow;
   - the literature review of slot 2, section 6;
   - time-series and learned-summary networks;
   - the BayesFlow simulator API;
   - the learned-summary detector.
5. **Material in the lectures and not the tutorial:**
   - Lectures 1–5 (prerequisites);
   - Lecture 16 (EWG);
   - [forward-pricing-emulation.md](forward-pricing-emulation.md).
6. **Stale references:** slot 1, section 4 cites "Lecture 9.7" for the toy, which is in Lectures 7.7 and 8.5.

## Decisions for the review

These are open; nothing on this branch has acted on them yet.

- **Granularity of the link.** Options: a "Tutorial" box at the head of each lecture section, a per-lecture table like
  the ones above, or both. The other direction is "Lecture x.y" pointers in each notebook section.
- **Where the tutorial's examples live.** One option: the lectures adopt the tutorial's data in slots 2 and 3, using
  the SPY quote and the Heston network, and keep the toys as a lead-in.
- **Structure for arXiv.** Options: keep sixteen lectures in four parts, or regroup into a prerequisites part
  (Lectures 1–5) and a part that follows the slots.
- **Format.** Options: keep Markdown, or move to LaTeX for the arXiv build.
- **Lecture 16 and EWG.** Decide whether the tutorial's notebook should gain the EWG section the curriculum plans.

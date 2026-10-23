# Slot 6. Changing priors and dealing with misspecified models

Stefan · 15 minutes · last slot of the afternoon. A proposal for Stefan's review. Part of the [curriculum](00-overview.md).

![Minute map of slot 6](images/slot-6.png)

## The slot in three sentences

**The question.** What does the answer depend on besides the data, and how is it found out when the model cannot produce the data?

**The answer.**

1. An amortized posterior has its prior and its noise model trained in, but both can be changed after training: by reweighting the draws, or by training once with the prior's and the noise's settings as extra conditions, so that a slider changes them.
2. A network answers every input with confidence, so misspecification has to be detected outside it: by checking that the data lie where the prior puts simulations, by re-pricing the posterior over the data, by the audit's weights, and by distances in a learned summary space.
3. When every model is misspecified, the honest report is the posterior within each model and the spread across models.

## Why this slot comes here

Slot 5 ends on four conditions: the prior, the noise model, the forward-variance curve and the model. Slot 6 changes each in turn on the same SPY smile, with the networks of slots 4 and 5, and closes the afternoon.

## Minute plan

| Min | Section | On screen | Live or cached | What is said |
| --- | --- | --- | --- | --- |
| 0–1 | 1. What the answer depends on | Slot 5's last slide: the four conditions | slide | Each condition is a choice. This slot changes each one and shows what moves. |
| 1–6 | 2. Changing the prior without retraining | (a) Reweighting: SPY's rough-Bergomi draws reweighted from the training prior on $H$ to a prior concentrated near the rough-volatility literature's $H \approx 0.1$, with the effective sample size; the forward-start reserve before and after. (b) A Feller-respecting prior for Heston: slot 4's SPY draws truncated at $\sigma_v^2 \le 2\kappa\theta$; the effective sample size collapses. (c) Sensitivity-aware amortization: a Heston network trained once with the prior's scale as a condition; a slider changes the prior and the posterior follows in one pass | (a) and (b) live, instant; (c) live with a network trained before the slot | A posterior's draws can be moved to a new prior by weights, as long as the new prior lives where the old one put mass; the effective sample size says when that fails. When it fails, as for Heston under the Feller condition, the data and the prior disagree, which is itself a finding. Training with the prior's settings as conditions (Elsemüller et al. 2024) makes the sensitivity analysis a slider. |
| 6–8 | 3. The noise model is a choice | The Heston network of (c), with the noise multiplier $c$ as a second condition: $c$ from 0.5 to 3 on a slider, the posterior width and the variance-swap range following | live | The quotes' noise is an assumption, and the posterior's width scales with it. A desk states its noise model and shows the range under alternatives. |
| 8–13 | 4. When the model is wrong: four detectors | One example per detector, all on the tutorial's smiles: (i) the input check: SPY's forward-variance curve against the pilot's curve prior (outside it, gate G0); (ii) the posterior predictive check: Heston's re-priced smiles on SPY, with their over-curved short smiles, and slot 4's variance swaps; (iii) the audit: Heston on a rough-Bergomi pilot smile, $\chi^2$ of 1,347 on 63 quotes, $\hat k$ = 154; (iv) a learned summary space: the distance of SPY and of rough-Bergomi smiles from the Heston training set (Schmitt et al. 2024) | (i)–(iii) cached; (iv) cached if built | The network gives a confident posterior for any input. Each detector looks from outside: at the inputs, at the fit, at the likelihood, or at the summaries. |
| 13–14 | 5. Across models | Slot 5's two forward-start distributions on SPY, Heston and rough Bergomi | cached | When both models are wrong in different ways, report both and the spread between them, judged smile by smile. Model averaging and stacking are the formal versions. |
| 14–15 | 6. Close | What the afternoon showed, in three lines; the take-home track; the repository | slide | The method needs a simulator, a prior and a noise model, and it is only as good as its checks. |

## The pieces, specified

### Reweighting to a new prior

For draws $\theta^{(i)}$ from the posterior under prior $p_0$, the posterior under prior $p_1$ is approximated by the weights

$$w^{(i)} \propto \frac{p_1(\theta^{(i)})}{p_0(\theta^{(i)})}, \qquad \text{ESS} = \frac{\big(\sum_i w^{(i)}\big)^2}{\sum_i \big(w^{(i)}\big)^2},$$

which works when $p_1$ puts its mass inside the posterior's support. The Pareto shape $\hat k$ of slot 5 diagnoses it.

- **Rough Bergomi on SPY.** The training prior on $H$ is replaced by one concentrated near $H \approx 0.1$, the value the rough-volatility literature estimates from realized volatility (Gatheral, Jaisson and Rosenbaum 2018). The slide shows $H$'s posterior and the forward-start reserve under both priors, and the ESS. Needs slot 5's SPY draws (⟨S8⟩).
- **Heston on SPY under the Feller condition.** The new prior is the old one restricted to $\sigma_v^2 \le 2\kappa\theta$. On slot 4's exact draws the Feller ratio $\sigma_v^2 / (2\kappa\theta)$ is never below 3.9 (median 4.9), so none of the 10,000 draws survives and the ESS is zero. Reweighting cannot even approximate the new posterior; only a new chain or a retrained network can, and its best fit to the quotes can be no better than the unconstrained one, which lies outside the new prior. That is the measured form of "the prior and the data disagree", and it is a model diagnosis as much as a prior one.

### Sensitivity-aware amortization

Elsemüller et al. (2024, *Transactions on Machine Learning Research*) train one network with the settings of the prior and the likelihood as extra conditions, so that the posterior under any setting is one pass. The proposal for slot 6 is a Heston network with two conditions:

- the prior's scale $s$: each parameter drawn from its box shrunk around the box's centre by $s \in [0.25, 1]$;
- the noise multiplier $c \in [0.5, 3]$: each quote's noise standard deviation multiplied by $c$.

The bank of slot 4 is regenerated with $(s, c)$ drawn per smile: about 46 CPU-minutes, plus about 32 minutes of training on 4 cores. Built before the slot, it gives sections 2(c) and 3 their sliders.

### The four detectors

| Detector | What it checks | Example on the tutorial's data | Cost |
| --- | --- | --- | --- |
| Input check | That the data and their inputs lie where the prior puts simulations | SPY's forward-variance curve needs a log deviation of 0.36 against the pilot's bound of 0.2 | Seconds |
| Posterior predictive check | That the posterior re-prices the data within its noise | Heston on SPY: over-curved short smiles, and variance swaps 0.6 to 1.7 points below the market | Seconds for Heston, minutes for rough Bergomi |
| The audit | That the network's posterior survives reweighting by the likelihood | Heston on rough pilot smile 291: $\chi^2$ of 1,347 on 63 quotes, $\hat k$ = 154 | Seconds for Heston, about 7 CPU-minutes for rough Bergomi |
| Learned summary space | That the data's learned summaries lie inside the training set's (Schmitt et al. 2024) | SPY and rough-Bergomi smiles against the Heston training set | Needs a Heston network with a summary network trained with the distance term |

The fourth detector is the one slot 3 names. It needs a summary network, which slots 4 and 5 do not use: building it is a choice for Stefan (below).

## Numbers on screen, and their status

| Number | Source | Status |
| --- | --- | --- |
| SPY's curve: 0.36 against the bound of 0.2 | Design document, S8 | stated in the design document; register as a claim |
| Heston on SPY: Feller ratio 4.9; variance-swap gaps | Slot 4's feasibility run | to redo as S8-H |
| Pilot smile 291: $\chi^2$ 1,347, $\hat k$ 154 | `rough-pilot/s7-across/across-models.json` | pilot |
| The reweighted $H$ posterior and reserve | Slot 5's SPY draws | ⟨S8⟩ |
| The sensitivity-aware network's sliders | New network | to build |

## To build, by the freeze on Fri 6 Nov

1. `slot6_priors_and_misspecification.py`: reweighting cells for rough Bergomi and Heston; the sliders of the sensitivity-aware network; the four detectors.
2. The sensitivity-aware Heston network: a bank with $(s, c)$ per smile and one training run.
3. Optionally, a Heston network with a summary network trained with the distance term of Schmitt et al. (2024), for the fourth detector.

## Open choices for Stefan

- Whether the sensitivity-aware network is built, or section 2(c) and section 3 use reweighting only. Reweighting a narrower posterior to a wider one (a larger noise) is unreliable, so the noise slider needs the network.
- Whether to build the summary-space detector, or show the first three detectors only.
- Whether to keep a returns example. The crash-day example (branch `feat/tutorial-slot6-crash-risk`, #42), in which the prior decides much of the answer, is ready as a take-home notebook, and could replace section 2(a) if a returns example is preferred.

## References for the slot

- Elsemüller, L., Olischläger, H., Schmitt, M., Bürkner, P.-C., Köthe, U. and Radev, S. T. (2024). Sensitivity-aware amortized Bayesian inference. *Transactions on Machine Learning Research*. [arXiv:2310.11122](https://arxiv.org/abs/2310.11122)
- Schmitt, M., Bürkner, P.-C., Köthe, U. and Radev, S. T. (2024). Detecting model misspecification in amortized Bayesian inference with neural networks. *Pattern Recognition (DAGM GCPR 2023)*, Springer, 541–557. (The proposal's reference [12].)
- Gatheral, J., Jaisson, T. and Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance* 18(6), 933–949.
- Yao, Y., Vehtari, A., Simpson, D. and Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis* 13(3), 917–1007.

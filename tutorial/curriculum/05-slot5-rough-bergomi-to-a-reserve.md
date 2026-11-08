# Slot 5. Bayesian calibration of rough Bergomi

Francesco and Alex · 40-minute application slot. Part of the
[curriculum](00-overview.md). Exact rehearsal timing is separate from this
content plan. The opening foundations are taught with Beamer; this application
uses the [rough-options notebook](../notebooks/slot5_calibration.py).

## What students should learn

- Construct a parameter–observation simulator and follow the BayesFlow workflow.
- Distinguish Monte Carlo pricing error from assumed quote noise.
- Explain the difference between supplying a variance curve and inferring it.
- Read a joint approximate posterior and check its implications in data space.
- Distinguish a learned pricer plus MCMC from direct amortized inference.
- Identify what the observations establish and what comes from assumptions.

## Core lesson

| Step | Content | Student prompt |
| --- | --- | --- |
| Model | Rough variance, its kernel and the supplied forward-variance curve. Prices are expectations, not observed latent paths. | What does roughness change, and what sets the mean variance? |
| Data | Controlled Monte Carlo mean prices, then independent quote noise. | Which uncertainty would more simulation paths reduce? |
| Two neural routes | Learn parameters → prices and run MCMC, or learn quotes → joint parameter draws. | Which computation repeats for a new market? |
| BayesFlow | Visible adapter, network, fitting and sampling code. The small fit uses fresh pairs and its own outputs. | What are training targets, and what enters inference? |
| Known curve | Current synthetic observation, supplied levels and focused posterior views. | Which parameter relationships remain visible? |
| Predictive check | Explicit three-month repricing of paired draws, with fresh quote noise and a reported precision budget. | Why does adding quote noise change the data spread? |
| Joint curve | Curve coefficients become targets; the generating curve is not supplied. | How does the inference task change? |
| Accuracy | Current aggregate diagnostics and the matching three-case surrogate comparison. | What can one passed check establish? |

The supplied-curve and joint examples use different observations and curve
families. Their widths are not a matched experiment about curve uncertainty.
The small training action demonstrates the API; it does not reproduce the
100,000-market estimator. Default viewing uses saved data and results.

## Additional studies

Students select an extension rather than reading a sequence of experiment
reports. Each starts with a prediction, offers an intervention, and ends with
an explanation prompt.

### 1. Can I use this estimator on a new market?

Use the fixed SPY snapshot from 21 October 2026 as a new supplied curve.
Normalize its levels by the first interval and compare with necessary bounds
under the **current** context prior. Its last/first ratio is about 2.93;
the current design permits at most 2.15.

Change the curve-variation bound in a proposed design. The envelope changes;
the installed estimator does not. A violated bound proves the input is outside
support. Passing this necessary test does not establish every joint constraint.

**Learning outcome:** check the simulation domain before applying the network.
A wider prior needs compatible generated data, training and validation.
No supported SPY posterior is claimed by this exercise.

### 2. Why must posterior draws stay paired?

For the selected current joint example, compute

$$K_{\rm var}(T)=\frac1T\int_0^T\xi_0(t)\,dt$$

for every complete posterior row. Compare the three-month and one-year rates.
Then deliberately shuffle the curve-coefficient columns independently.
Their marginals stay unchanged but their dependence is lost. The shuffled
result is an artificial product-of-marginals calculation, not a posterior.
Its range can widen or narrow depending on the case.

**Learning outcome:** carry complete joint draws into a derived quantity.
An exactly supplied curve gives one fair variance-swap strike, even when
H, eta and rho remain uncertain. This says nothing about certainty of future
realized variance. The known/joint examples are not a controlled width comparison.

### 3. What if two inference routes disagree?

Use the existing comparison whose task, bank and posterior checkpoint match
the current known-curve lesson. The reference holds its 50,000-market pricer
fixed and samples that deterministic surrogate likelihood with PyMC.
Two of three cases meet the recorded location/width screens; H and eta fail
in the third. Show the metrics, then ask students which follow-up checks matter.

**Learning outcome:** small average pricing error and aggregate calibration do
not guarantee conditional posterior agreement. Check the reference's mixing,
local pricing accuracy and the posterior estimator. The surrogate target omits
emulator uncertainty and grid/label bias; agreement is not exact-model acceptance.

### 4. What lies beyond the last quote?

Add an independent tail slope, leaving the curve unchanged through year one:

$$\xi_{\vartheta,\alpha}(t)=
\begin{cases}
\xi_\vartheta(t),&t\le1,\\
\xi_\vartheta(t)e^{\alpha(t-1)},&t>1,
\end{cases}\qquad \alpha\sim\mathcal N(0,\tau^2).$$

All original quotes expire by one year, so their likelihood cannot depend on
alpha. Its posterior therefore remains its prior:
$p(\vartheta,\alpha\mid y)=p(\vartheta\mid y)p(\alpha)$.
Use the existing approximate posterior for vartheta and independent normal
slope draws. Change tau and inspect analytic three-year mean variance rates.
At zero width, recover the original curve continuation exactly.

**Learning outcome:** some narrow intervals reflect a restrictive extrapolation
family. Additional spread here comes from the explicit new prior, not new
information learned by the network. These rates are not future payoff, VaR,
expected-shortfall or exotic-product forecasts.

## Presentation and evidence rules

- Keep the core code visible and match displayed code to executable actions.
- Use original parameter units and explain axes beside each figure.
- Keep full paired rows when transforming samples.
- Keep numerical approximation and statistical uncertainty distinct.
- Use current task/checkpoint evidence. Leave obsolete pilot comparisons,
  timing projections, export-status panels and experiment logs outside the
  student narrative; their source records remain available in the repository.
- Do not infer a new noise model, prior or input design merely by changing the
  conditions passed to an existing estimator.

Exact rough-model references, joint-curve conditional comparisons and further
Monte Carlo product checks remain unfinished. Analytic variance-rate exercises
do not complete those checks. The models' existing accuracy limits remain in
view; no classroom illustration is a new scientific-acceptance decision.

## Code and checks

The main lesson lives in `notebooks/slot5_calibration.py`.
Application helpers are in `src/sbi4f_tutorial/rough_options/`:
`lesson.py` for explicit pricing/training actions, `context_checks.py` for
necessary support bounds and `studies.py` for analytic curve/tail propagation.
Shared numerical definitions remain in `shared/`; lab workflows reuse them.

Run the tutorial tests, marimo check and notebook exports. Check every study
selection, all four current joint examples, prior-width endpoints, the paired/
shuffled switch and reference-case controls. Tail integrals are checked against
quadrature and their zero-slope limits. The
[application plan](TUTORIAL-REVIEW.md#c-refine-the-option-notebooks-in-place)
records source scope and verification. Collaborator rehearsal remains separate.

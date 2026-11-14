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
| Model | Rough variance, its kernel and the supplied forward-variance curve. Decompose Y, variance, finite-step returns and prices with shared random draws and fixed axis presets. | What does roughness change, and what sets the mean variance? |
| Why rough volatility? | Contrast Heston's limiting short-expiry skew with the rough power law; show a dated SPY grid and finite-price illustrations. | What extra time-scale behavior does H make possible? |
| Data | Generate a prior market: paths, payoffs, conditional mean-price estimates, then independent quote noise. Explain Q versus P. | Which uncertainty would more simulation paths reduce? |
| Two neural routes | Decompose the dense pricer and conditional flow-matching network. Fit a small pricer on clean labels and evaluate its Gaussian likelihood. | Which computation repeats for a new market? |
| BayesFlow | Visible adapter, network, fitting and sampling code. Choose a small live fit or optional matching pretrained weights; saved results need neither. | What are training targets, and what enters inference? |
| Check that network | Explicit recovery and rank-ECDF plots for the selected network on 200 stored validation markets, with 200 fresh draws per market. | What can a falling loss tell us, and what does it leave unchecked? |
| Known curve | Current synthetic observation, supplied levels and focused posterior views. | Which parameter relationships remain visible? |
| Predictive check | Explicit three-month repricing of paired draws, with fresh quote noise and a reported precision budget. | Why does adding quote noise change the data spread? |
| Joint curve | Curve coefficients become targets; the generating curve is not supplied. | How does the inference task change? |
| Another contract | Carry the joint posterior into a fair variance-swap strike; optionally compare a least-squares point fit using the matching six-input pricer. | What uncertainty does one plug-in vector leave out? |
| Accuracy | Price-label residuals, predictive checks and optional aggregate diagnostics. | What can one passed check establish? |

The supplied-curve and joint examples use different observations and curve
families. Their widths are not a matched experiment about curve uncertainty.
The small training actions demonstrate the APIs; they do not reproduce the
longer-fit estimators. The pricer's validation rows select an epoch; they are
not an independent final accuracy test. The tiny session pricer is not used for MCMC. Default inference views use saved
data and results; the small forward/data simulators run locally.

The optional network check uses the tutorial fit or the loaded pretrained
workflow. Its **200 input markets are bundled**, with no saved model draws.
They contain 63 quotes each, generated independently under the supplied-curve
task with its original Monte Carlo precision policy. Generating them took
**688 seconds** on one Apple M5 Max CPU worker; sampling the pretrained network
took about **34 seconds**, with loading and plotting measured separately.
Reusing the inputs makes this a fixed validation exercise, not an untouched
final test. Recovery and ranks must be interpreted together; passing a rank
check alone does not show that the estimator learned from the data.
Fresh input generation is an [optional offline step](../../lab/plans/training-data-guide.md#fixed-rough-option-diagnostic-inputs).

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

### 3. What lies beyond the last quote?

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
`explorer.py` for path and architecture figures, `lesson.py` for explicit
pricing/training-data actions, `surrogate.py` for supervised price regression,
`comparison.py` for the Heston/Bergomi skew and smile comparison,
`data_story.py` for payoff/quote generation and pricer validation,
`diagnostics.py` for the fixed 200-market inputs and native BayesFlow checks,
`context_checks.py` for necessary support bounds and `studies.py` for analytic curve/tail propagation and the optional point-fit comparison.
Shared numerical definitions remain in `shared/`; lab workflows reuse them.

Run the tutorial tests, marimo check and notebook exports. Check every study
selection, all four current joint examples, prior-width endpoints, the paired/
shuffled switch, data/noise controls, pricer validation selections, optional native
point fit and source invalidation. Tail integrals are checked against
quadrature and their zero-slope limits. For the 200-market check, verify both
network sources and that changing the source clears the previous plots. The
[application plan](TUTORIAL-REVIEW.md#c-refine-the-option-notebooks-in-place)
records source scope and verification. Collaborator rehearsal remains separate.

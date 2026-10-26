# Calibration from sparse option quotes

The illustrative goal is **joint model and forward-variance curve inference from incomplete option data, followed by uncertainty propagation into an unquoted product**. This is a property of the inference problem, not of a particular ticker. EWG is the current implemented market example; the final example remains to be selected.

Use **Calibration from sparse option quotes** as the learner-facing name. “Thin” can remain a short internal name for existing modules and studies. Here sparse means insufficient strike/maturity coverage for the question being asked; the noise and reliability of the quotes matter too. Quote count alone does not define the problem.

## What the lesson should establish

A participant should be able to identify which model or curve components the quotes constrain, which remain sensitive to the prior or continuation assumptions, and how that distinction affects a product value. A broader posterior is not itself evidence of better inference. An unobserved curve segment does not become learned information just because a neural estimator returns draws for it. Quotes can update that segment indirectly through prior correlations; separate that effect from direct identification of its remaining increment.

The comparison asks how much information survives when useful quotes are unavailable. It does not presume that SBI must outperform a good classical proposal or a price emulator with MCMC. All methods receive the same quotes, noise model and supplied context; posterior methods also use the same joint prior.

## Start with controlled experiments

Use paired synthetic surfaces from the same rough-Bergomi bank, with known parameters and curves. Declare a small development panel of observation layouts before comparing methods:

| Layout | What it isolates |
| --- | --- |
| Full declared grid | The richer-data reference. |
| Missing later maturities | Uncertainty about an unquoted curve segment and products depending on it. |
| Narrow strike coverage or missing wings | Loss of information about smile shape and model parameters. |
| Dispersed missing quotes, with comparable count | Whether coverage matters beyond the number of quotes. |

Vary quote-noise scale separately at 0.5/1/2 where supported. Choose one clear contrast for the live lesson; keep the other layouts as development checks or take-home material. Prefer a controlled subset of the supported grid initially, so the existing clean prices can be reused without new pricing.

**Infer the curve in both the full and sparse arms.** Keep the joint model/curve prior, numerical model and downstream products fixed. Otherwise moving from a supplied-curve liquid task to an unknown-curve sparse task changes both the observations and the inference target. The supplied-curve task remains a useful separate conditional benchmark; it does not isolate the effect of thinning.

Check posterior contraction, conditional reference agreement, coverage, parameter/curve dependence and product uncertainty, including prior sensitivity and deliberately changed continuation assumptions. Use one product affected by the missing information and a well-supported quantity as a control. Choose product horizons from the declared layout; a six-month-to-one-year forward start is a current example, not a universal requirement.

Reusing the bank does not make every layout supported by a trained network. The current thin workflow has a fixed layout and layout-bound checkpoint. Either fit declared layouts separately or implement and validate a mask-conditioned joint estimator over the supported layouts. Budget the richer-data joint control and any extra layout fits explicitly; they are additional to the current fixed-layout training matrix unless one validated estimator covers them.

## Choose a market illustration afterward

Compare a small candidate set on data quality and teaching suitability before inspecting final method rankings. EWG stays in that set, alongside any better-documented dated chain that satisfies the criteria. No replacement ticker is selected by this plan.

Select and freeze one dated chain/layout with:

- Clear missing coverage relevant to the learning question, while retaining enough informative quotes for a nontrivial posterior.
- Reproducible data provenance and coherent treatment of timestamps, forwards, dividends, exercise conventions and quote-noise assumptions.
- A plausible numerical model and feasible reference/audit path, so the example can distinguish weak identification from obvious data or model failure.
- A downstream product that exposes the missing information and a control quantity supported by the quoted region.
- A manageable live story and reproducible cached artifacts.

Record the candidate assessment and selected layout/product before final accuracy comparisons. Choose for an interpretable problem, not for a favourable SBI result. A real chain has no known generating parameter truth; use synthetic experiments for coverage and identification claims. A posterior-conditioned local sensitivity study is not population or broad market validation.

If no market snapshot meets the requirements, the controlled synthetic lesson is still a complete illustration. A market panel is an application of that lesson rather than a prerequisite for it.

## What changes if EWG is replaced

Preserve existing EWG data, figures, source labels and historical study outcomes. They remain evidence about that dated example. The current proposal layout and actual-quote audit are distinct objects and must retain their own coordinates and noise definitions.

Review the new case's layout, preprocessing, quote noise, joint checkpoint, observation audit, downstream products and coverage-study assumptions. The current product/coverage scripts contain assumptions about the last quoted horizon and the curve continuation; changing the ticker can require more than a new layout file. Quotes outside the bank's supported contract grid may require additional pricing or a new declared task. Reassess cost before submitting replacement studies.

The next planning decision is whether the sparse-quote lesson belongs in the release and which claims it should support. Final ticker selection comes after the controlled design. See the [central experiment plan](NUMERICAL-EXPERIMENT-PLAN.md) and [protocol](EXPERIMENT-PROTOCOL.md).

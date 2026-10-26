# Calibration from sparse option quotes

The illustrative goal is **inference from incomplete option data, followed by uncertainty propagation into an unquoted product**. Heston provides a numerically checkable full/sparse control. Rough Bergomi initially uses a Heston-shaped mean-variance curve and jointly infers six unknowns: H, eta, rho, short variance, long variance and curve decay. This is a property of the inference problem, not of a particular ticker. EWG is the current historical rough market example; the final market illustration remains to be selected.

Use **Calibration from sparse option quotes** as the learner-facing name. “Thin” can remain a short internal name for existing modules and studies. Here sparse means insufficient strike/maturity coverage for the question being asked; the noise and reliability of the quotes matter too. Quote count alone does not define the problem.

## What the lesson should establish

A participant should be able to identify which model or curve components the quotes constrain, which remain sensitive to the prior or continuation assumptions, and how that distinction affects a product value. A broader posterior is not itself evidence of better inference. An unobserved curve segment does not become learned information just because a neural estimator returns draws for it. Quotes can update that segment indirectly through prior correlations; separate that effect from direct identification of its remaining increment.

The comparison asks how much information survives when useful quotes are unavailable. It does not presume that SBI must outperform a good classical proposal or a price emulator with MCMC. All methods receive the same quotes, noise model and supplied context; posterior methods also use the same joint prior.

## Start with controlled experiments

Start with paired synthetic surfaces from the Heston bank. Hold the four parameter targets, prior, fixed mean reversion and noise model constant; remove the same declared quote subset from the neural conditions and Fourier-likelihood MCMC. Heston's expected variance curve is determined by its parameters, so there is no separate supplied curve or seven added curve targets. A small fixed-layout quote-subset adapter and a selected-family checkpoint are enough; the current rough-specific thin trainer is not already that implementation. No new clean-price generation is needed.

Then use paired surfaces from the new six-target rough bank, after the [curve-family adequacy check](NUMERICAL-EXPERIMENT-PLAN.md#rough-curve-baseline-and-bounded-flexibility-check) and task/prior freeze. The old seven-level bank is not this task's data and cannot be projected/relabelled without changing prices. Both models need the full grid and one declared sparse layout. Prefer missing later maturities; other patterns below are optional. Reuse the same contract pattern across models when supported, while generating/validating each within its own model/prior. This checks quote handling in Heston before the more expensive rough inference.

| Layout | What it isolates |
| --- | --- |
| Full declared grid | The richer-data reference. |
| Missing later maturities | Uncertainty about an unquoted curve segment and products depending on it. |
| Narrow strike coverage or missing wings | Loss of information about smile shape and model parameters. |
| Dispersed missing quotes, with comparable count | Whether coverage matters beyond the number of quotes. |

Vary quote-noise scale separately at 0.5/1/2 where supported. Choose one clear contrast for the live lesson; keep the other layouts as development checks or take-home material. Prefer a controlled subset of the supported grid initially, so the existing clean prices can be reused without new pricing.

**For rough Bergomi, infer the same six coordinates in both full and sparse arms.** Keep the joint prior, curve family, numerical model and products fixed. A shared initial mean-curve family does not insert Heston's variance dynamics into rough Bergomi; the whole processes remain different. Moving from a supplied-curve task to unknown-curve sparse inference changes more than quote availability. Supplied-curve calibration remains an optional conditional benchmark; its market-derived curve is a plug-in estimate, not an exact independent observation. See the [curve explanation](EXPERIMENT-PROTOCOL.md#what-the-forward-variance-curve-input-means).

Check posterior contraction, conditional reference agreement, coverage, parameter/curve dependence and product uncertainty. The compact curve adds global shape/extrapolation assumptions; a small sparse posterior is not automatically more informative. Near-flat curves may leave decay weakly identified. Use one product affected by missing information and a well-supported quantity as a control. Variance-swap prices now vary with posterior curve coefficients rather than being deterministic supplied-curve controls. Choose product horizons from the layout; a six-month-to-one-year forward start is a current example, not a universal requirement. Existing flat/widened-last-increment studies do not transfer unchanged from the old seven-level prior: any continuation/family sensitivity needs a declared alternative and a correctly labelled target.

Reusing a bank does not make every layout supported by a trained network. Use four selected fixed-layout estimators: full/sparse Heston and full/sparse joint rough, with three training seeds each. The current rough-only thin code is seven-level specific and requires a six-target adapter/prior implementation; no supplied generating curve may leak into conditions. Keep each model's prior matched within its comparison. A mask-conditioned estimator is optional later work. Do not repeat the architecture matrix for sparse layouts.

## Choose a market illustration afterward

Use an available well-documented dated chain if it meets the criteria below. EWG remains a candidate; another suitable example can replace it. Keep this selection bounded rather than building a market-data screening pipeline. Select before inspecting final method rankings. No replacement ticker is selected by this plan.

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

Review the new case's layout, preprocessing, quote noise, joint checkpoint, observation audit, products and coverage-study assumptions. Current product/coverage scripts contain old-prior and curve-continuation assumptions; both the new curve family and a changed ticker require review beyond swapping a layout file. Preserve EWG's original seven-level outcomes as historical evidence. Quotes outside the new bank's supported grid may require additional pricing or a new task. Reassess cost before submitting replacement studies.

The next planning decision is whether the sparse-quote lesson belongs in the release and which claims it should support. Final ticker selection comes after the controlled design. See the [central experiment plan](NUMERICAL-EXPERIMENT-PLAN.md) and [protocol](EXPERIMENT-PROTOCOL.md).

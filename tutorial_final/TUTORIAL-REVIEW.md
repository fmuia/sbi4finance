# Review of the Heston and rough Bergomi tutorial

26 October 2026. Review of the course and curriculum proposed in PR #44 at
`cebfb1f`. This assessment assumes that Heston and rough Bergomi remain the two
core models.

## Verdict

**Keep the broad learning path.** Black–Scholes gives a small problem whose
posterior can be checked directly; Heston makes the complete amortized workflow
checkable against a numerical likelihood; rough Bergomi preserves the financial
question while making pricing and inference substantially more expensive. This
is a relevant application of simulation-based inference (SBI).

The main weakness is the argument for *why amortization is useful*, rather than
the model choice. Bayesian uncertainty, an expensive pricing model and a fast
neural posterior do not by themselves establish that amortized inference is the
best tool for a particular workload. The tutorial will be stronger if it teaches
that decision openly, including situations where another method is sufficient.

The [curriculum overview](curriculum/00-overview.md) provides a coherent sequence.
The sixteen-lecture [course](../docs/course/README.md) should remain supporting
material, with only the concepts needed for the live task introduced on stage.

## Follow-up self-review: keep the tutorial small

The plan had accumulated a tutorial release, a broad architecture benchmark, sparse-chain support and direct rough-inference research. Separate them. The selected core teaching path is complete-grid Heston with four inferred parameters, followed by complete-grid rough Bergomi with six jointly inferred parameters and a Heston-shaped initial mean-variance curve. Retain the bounded four-family/encoder screen and confirm only the two selected configurations across three seeds. Complete data, training, reload, inference, numerical/reference checks, product uncertainty and participant packaging before starting sparse-chain work. No quote-subset adapter, mask-conditioned training or arbitrary-chain support is a prerequisite. One bounded check against the published hump-capable curve family precedes substantial rough generation.

Sparse-chain inference is a separate downstream unit of work that needs its own input/layout, missingness, training and validation specification before implementation. Preserve the existing [case plan](SPARSE-QUOTE-CASE-PLAN.md) as its starting point, rather than promising a sparse lesson in this release. Supplied-curve rough fits, runner-up campaigns, direct rough MCMC and broad reserve-coverage claims remain separate extensions.

The incremental-bank and recovery safeguards address real duplicate-ID, interrupted-write, memory and changed-dataset failure modes. Keep the tested primitives rather than rewriting them to reduce line count, and stop expanding the infrastructure unless a pilot exposes a concrete need. Collection/recovery machinery belongs in instructor tools; participants should use supplied data/checkpoints and follow the inference workflow. The infrastructure's passing mocked tests are not scientific validation. Prioritize actual pricing refinement, independent holdouts, calibrated/conditionally checked posteriors, matched baselines and an honest product/cost figure over more framework work.

The [central plan](NUMERICAL-EXPERIMENT-PLAN.md#tutorial-scope-after-self-review) and [protocol](EXPERIMENT-PROTOCOL.md#training-comparisons) reflect the reduced programme. No new tracking system or additional planning document is needed.

## What already works well

- **One financial question links the examples.** Inferring a calibration and
  propagating it into an unquoted product is a concrete reason to care about a
  joint posterior. It gives the audience something more useful than parameter
  recovery alone.
- **Heston is an effective control experiment.** The numerical likelihood lets
  us investigate the neural approximation separately from the financial model.
  The recorded conditional failures, alongside successful marginal calibration,
  are valuable teaching material.
- **Rough Bergomi supplies a real computational problem.** Repeated Monte Carlo
  pricing makes the trade-off between simulation, training and repeated inference
  tangible. Training data, computation and accuracy all have visible costs.
- **The evidence ledger is unusually useful.** The
  [study status](../docs/course/status.md) and [claims ledger](CLAIMS.md) distinguish
  completed studies, pilot results and pending evidence. Retain this discipline
  when new architectures and BayesFlow versions are added.

## Three arguments that must stay separate

**Why use a posterior?** Weak identification, nonlinear transformations and
uncertain inputs can make one fitted calibration an incomplete description of a
product's value. However, a wider Bayesian interval is not automatically a better
one. Its width depends on the prior and observation model. On well-identified
liquid surfaces, agreement with a classical local interval is a useful result.

**Why use SBI?** Learning an inverse map from simulations is attractive when
likelihood-based inference is expensive or awkward. Here, the option observation
likelihood is specified; its mean requires numerical pricing. Say explicitly that
the difficulty is repeated expensive pricing. This distinguishes the application
from a simulator for which no evaluable observation likelihood is available.

**Why amortize?** The strongest case is a recurring workload: many smiles, dates,
scenarios or posterior queries within the declared task. Making thousands of
validation posteriors affordable is useful, but partly circular as the sole
justification: many of those evaluations are needed to validate the amortizer we
introduced. Put an operational workload and an honest break-even calculation
beside it.

[Slot 2](curriculum/02-slot2-amortized-bayes.md) should compare total costs against
both a good classical proposal with importance sampling and a validated neural
pricer with MCMC. The cost must include bank generation, training, warm-up and any
audit applied to each query. The
[course plan, Section 2.4](../docs/course/v2-plan.md#24-why-simulation-based-inference)
already acknowledges these competitors; make that comparison central on stage.

If either competitor matches the accuracy and total cost of the intended
workload, the tutorial remains a useful demonstration of SBI. The conclusion
becomes “a practical alternative with measured trade-offs,” rather than “SBI is
necessary.” A campaign that can reach either conclusion is more convincing.

## Make the information available to each method visible

The core plan has complete-grid Heston with four targets and complete-grid rough
Bergomi with six targets. Each receives all 63 declared contract prices and its
noise information; missing or invalid entries are not silently filled to pass
this input contract. The rough model has H, eta, rho and three coefficients of
an initial Heston-shaped variance curve, with no supplied curve. Heston already determines its
mean curve from its parameters; using that functional family in rough Bergomi
does not make the stochastic variance processes equivalent. Heston currently
fixes kappa at 2 while rough curve decay is inferred, so match those assumptions
in any controlled cross-model diagnostic and avoid attributing all differences
solely to roughness.

The existing course's supplied-curve and seven-level examples remain historical
or optional comparisons. Holding a market-reconstructed curve fixed gives
conditional intervals that omit curve-estimation uncertainty. In the new joint
task an unquoted segment is constrained globally by the coefficient prior and
exponential shape. Explain that source of information, including extrapolation;
it is not direct evidence learned from unavailable quotes. Near-flat curves can
leave decay weakly identified even when curve bands and product values are stable.

A plug-in curve reconstructed from the same quotes is a conditional modelling
approximation. Do not describe it as additional independent data. Show how
plausible curve perturbations change the result before interpreting the interval
as a full description of calibration uncertainty.

Use matched priors, observed information, noise and numerical accuracy when
comparing methods. Include a small prior/noise sensitivity check in the core
full-grid examples. In the separate sparse-chain unit, the alternative laws for the unquoted curve level in
[Lecture 16, Sections 16.4–16.5](../docs/course/16-a-thin-chain-ewg.md)
are important. Keep the favourable prior-generated case and the deliberately
misspecified cases together; neither alone settles comparative performance.

The option-noise model deserves an early, short explanation. A complete synthetic
design is not a guarantee that a raw market chain supplies every contract.
Any retained market example must label how its complete grid was constructed.
Independent noise
on interpolated grid values is a modelling choice, not a property of the source
quotes. The likelihood should not gain artificial precision merely because a
smoother was read at many points. The actual-quote EWG audit and noise-scale
sensitivity help address this concern, but cannot establish the model's general
market adequacy from one date.

## Tighten claims that could undermine an otherwise good lesson

- **Reference accuracy:** describe Fourier/MCMC results as checked numerical
  references. Label surrogate-MCMC and finite Monte Carlo targets precisely.
  The current rough-MCMC cost extrapolation is not a measured completed chain.
- **Model criticism:** violating Heston's Feller condition does not invalidate
  the model. [Lecture 10, Section 10.2](../docs/course/10-heston.md)
  correctly explains this. Carry that distinction into the live handoff and use
  product or predictive discrepancies as the substantive evidence. Fixing mean
  reversion at two also restricts the Heston comparison. Show the existing
  mean-reversion profile or an appropriate sensitivity control before attributing
  the mismatch to the whole Heston family or arguing that roughness is necessary.
- **Uncertainty:** the proposed product quantile is a conditional, within-model
  uncertainty measure. The course's
  [three ranges](../docs/course/06-marking-a-product-the-smile-does-not-quote.md#67-three-ranges-and-baselines-that-fix-what-the-quotes-do-not)
  explain why it does not cover every source of valuation uncertainty. Avoid
  implying that one posterior quantile automatically establishes a complete
  prudent-valuation or regulatory-compliance procedure.
- **Validation:** SBC, conditional reference agreement, posterior predictive
  checks and importance-weight diagnostics answer different questions. Preserve
  their separate outcomes, including failures; a successful correction must not
  conceal a poor raw amortizer.

## Make the live path smaller than the research programme

The curriculum's 115 minutes of talks can fit a half-day, but the remaining time
needs an explicit allocation. One workable proposal is 60 minutes of exercises,
20 minutes of breaks and 15 minutes of questions: 210 minutes in total.

| Part | Main learner action | Keep outside the essential live path |
|---|---|---|
| Slots 1–2 | Predict how changing one quote changes the toy posterior; compare the exact and learned answers | A survey of every classical SBI method |
| Slot 3 | Distinguish the observation encoder from the posterior estimator | Detailed derivations and every architecture in the benchmark |
| Slot 4 | Trace input → adapter → posterior → reference check → product | Full training and every historical architecture experiment |
| Slot 5 | Predict which product or curve component is weakly identified; inspect the result and its diagnostics | Several products, several market cases and every audit variant at once |
| Slot 6 | Change one prior or noise assumption and explain the effect | A second full workflow or several additional trained detectors |

Use one primary downstream product to carry the complete-grid inference story;
a second product can be a control. Teach identification through posterior
dependence and prior/noise sensitivity while keeping every declared quote
available. A suitable market case can illustrate the workflow if its preprocessing
and observation assumptions are explicit; it is not required to introduce sparse
support. EWG remains historical evidence for the downstream sparse-chain unit,
whose specification starts from the [case plan](SPARSE-QUOTE-CASE-PLAN.md).
Keep architecture comparisons in a compact results
panel and the take-home material. Each participant should finish with one
sentence naming the data, unknowns, inference approximation, validation result
and downstream quantity.

## What would make the tutorial strong enough to release

The learning path is strong enough when a participant can reproduce the Heston
example, interpret the rough example's evidence, and explain when amortization
is worth its cost. Require reproducible checkpoints, scoped reference comparisons,
numerical-error checks, explicit assumptions, a timed rehearsal and a cached
fallback for each expensive demonstration.

Sparse-chain outcomes do not gate this release. If the full rough checkpoint misses
its checks, label the advanced material as an unfinished research case and narrow
the advertised hands-on scope. That does not require replacing the broad
learning path; it requires changing the claims made about it.

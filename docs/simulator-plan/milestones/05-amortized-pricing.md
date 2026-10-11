# M5 Parameter posteriors and amortized pricing

**Status:** acceptance scope; track implementation evidence and didactic completion separately. **Depends on:** accepted [M4.3](04-hkde-and-options.md) option observations and the local bank/loader pipeline in [M2.3](02-banks-and-cluster.md). Read [inference and pricing](../notes/03-inference-and-pricing.md) and [software and data](../notes/04-software-and-data.md).

## Purpose

**Measured update:** increment 7 now includes the first
[restricted HKDE baseline](../../hkde-joint-inference.md). M5.1a–c are delivered:
frozen protocol, refined numerical reference, saved bank and one trained,
reloadable checkpoint. M5.1d ran and failed only the target-price quantile gate
(0.0439 currency versus 0.03); parameter checks and marginal SBC pass. M5.1e
provides the offline review of that outcome. The checkpoint remains provisional,
the reserved final test is untouched, and no tuning sweep follows. M5.1
scientific acceptance remains pending. The [bounded M5.2 price surrogate](../../hkde-price-surrogate.md)
now passes independent validation and its new regression test, with maximum
observed error 3.84e-7 currency. Its successful substitution leaves the original
0.0439 inference discrepancy unchanged. Bergomi banks remain useful
numerical pilots; settling their continuum-price bias is not a dependency of
this HKDE experiment. Heston tuning stays paused with its results preserved,
and the returns-versus-options curriculum choice remains open.

MC-generated Bergomi panels include numerical estimator randomness as well as
quote noise; later reference inference must match that declared law or justify
neglecting numerical error at the selected fidelity. A bank's successful
save/load or marginal SBC cannot replace conditional validation.

Establish the relationship between Bayesian calibration and fast downstream prices. Build parameter-posterior inference first, then a separately assessed supervised pricing surrogate; study direct inference of posterior prices only if a concrete use case justifies it.

## Tasks

1. Train a posterior approximator for the reviewed synthetic option-panel task, including generated curves/context. Record prior, observed context, parameter transformations, initialization targets, bank identity, architecture, and training-only normalization. Optional M applies the normally frozen checkpoint to an unlabelled market panel; it is distinct from network training.
2. Validate against the explicit quote likelihood and trusted reference posterior calculations on fixed cases. Push matched parameter/state draws through the validated pricer to obtain distributions of downstream prices. These are the baseline uncertainty outputs.
3. Define a supervised price-surrogate task with inputs consisting of Q parameters, contract terms, and required market/state context. Labels are validated pricing expectations, not individual realized payoffs or presumed real-market fair values. Set the supported domain and loss/error scales deliberately; tiny prices and tail contracts require more than a single average relative-error score.
4. Train and test the surrogate using grouped splits that keep related contracts/parameter draws from leaking across partitions. Assess posterior price propagation with both exact and surrogate pricers. Keep the calibration observation generator on the validated reference route unless a separate approximation study permits replacement.
5. Optionally design a direct conditional posterior-price model for a fixed target payoff and context. Generate each noisy conditioning panel and reference target model price from the same parameter draw. Its labels and validation must reproduce the relevant push-forward distribution, rather than confusing a conditional price with a future realized payoff.

## Ordered implementation subplan

| Step | Concrete output | Verification and handoff |
|---|---|---|
| M5.1 Parameter posterior and reference prices | Validated calibration checkpoint, same-likelihood posterior reference cases and downstream price distributions from the reference pricer | Accept conditional/joint posterior checks and price propagation. This complete options baseline can enter M8 without waiting for a learned pricer. |
| M5.2 Supervised price surrogate | Separate regression bank/checkpoint, supported-domain specification, error report and measured cost comparison | Compare price errors and propagated quantiles against M5.1. Accept or reject the surrogate independently; its failure does not invalidate the accepted reference-price path. |
| M5.3 Optional direct price posterior | Fixed-target conditional price-posterior experiment and comparison with M5.1 | Verify matched panel/target generation and push-forward agreement. Keep this experiment independent of acceptance of either earlier route. |

M5.1 supplies the reference price map and propagation cases first. M5.2 may
assess the **pricing component independently against an accepted numerical
pricer/reference**, even while the neural posterior remains provisional. In
that case, its fixed posterior draws are stress cases, not accepted inference
evidence; passing the surrogate does not accept M5.1 or unlock its final test.
An accepted combined participant route still requires both components. M5.3
needs its own accepted reference comparison and need not wait for surrogate
acceptance. Plain model-specific modules and compact artifacts suffice;
checkpoint cards identify the route and supported domain.

## Next increment: restricted HKDE baseline

The subplan below remains the frozen acceptance specification. Its first-run
[protocol](../../hkde-inference-protocol.md), [measured result](../../hkde-joint-inference.md)
and graph-linked review preserve both the completed implementation and failed
scientific gate; a later experiment must retain that record.

Reuse `configs/tasks/hkde_option_bank.json`: infer only long-run variance
`theta` and jump intensity `jump_intensity`, with independent uniform priors
on `[0.025, 0.065]` and `[0.25, 1.0]`. Initial variance, the other model
parameters, spot/carry and the nine European put contracts remain known
context. Observations are currency prices with independent additive Gaussian
quote errors of standard deviation **0.25**. This is one bounded S experiment,
not full HKDE calibration, a market study or a curriculum commitment.

Writing the parameter pair as \(\vartheta\), the implemented observation law is

\[
x_j=p_j^{\mathrm{quad}}(\vartheta)+\epsilon_j,
\qquad \epsilon_j\overset{\mathrm{iid}}\sim N(0,0.25^2),
\]

where \(p^{\mathrm{quad}}\) denotes the declared deterministic Fourier
quadrature. Therefore the posterior density on the prior rectangle is
proportional to

\[
\pi(\vartheta)\exp\!\left[-\frac{1}{2(0.25)^2}
\sum_{j=1}^{9}\{x_j-p_j^{\mathrm{quad}}(\vartheta)\}^2\right].
\]

The likelihood is evaluable for this numerical model; the posterior grid and
continuum-price approximation still require refinement checks. No latent-path
sampler, new simulation framework or real-data acquisition is needed for this
two-dimensional baseline. Training pairs contain the noisy panel and known
context as inputs, and the synthetic parameter pair as targets. Clean prices,
pricing diagnostics and truths must not enter observed conditioning.

| Order | Bounded output | Check before moving on |
|---|---|---|
| M5.1a — Freeze the experiment | Existing task/configuration plus one short validation sheet: independent bank/split identities, fixed validation panels, a reserved final test, a few declared downstream European contracts outside the conditioning panel, metrics and tolerances. | Keep the final test unused for architecture, training-budget and acceptance-threshold choices. Express tolerances in posterior/price units with reference uncertainty, and preserve the original fixed cases when a check fails. |
| M5.1b — Build the two-dimensional reference | Evaluate the explicit likelihood on a parameter grid or quadrature; retain normalized weights, joint dependence, marginals and reference target-price distributions for the fixed validation cases. Reuse the grid's price matrix across panels. | Refine both parameter integration and pricing controls. Check posterior normalization, moments/dependence and target-price quantiles at the intended comparison resolution. Record this as a numerical reference, not an analytically exact posterior. Resolve an inadequate reference before interpreting neural differences. |
| M5.1c — Train one small baseline | A measured modest bank, the existing in-memory BayesFlow route, one declared architecture/seed/budget, and a reloadable checkpoint. | Keep normalization training-only; preserve the bank/task/adapter identities and deterministic reload checks. The existing 24-panel smoke bank is an interface test, not an adequate training set. Size this first training run from measured generation cost; do not start an architecture sweep or change Bergomi settings. |
| M5.1d — Check inference and price propagation | Same-case neural/reference comparisons, joint/dependence checks, synthetic SBC and coverage, contraction/data updating relative to the prior, and downstream reference-pricer distributions. | Separate identifiable combinations from weak parameter directions; do not require arbitrary contraction of every marginal. A prior-copy baseline and conditional references distinguish learning from an SBC-only pass. Include posterior sampling uncertainty and grid/pricing uncertainty. Preserve failures; evaluate the untouched final test only after fixing the candidate and its selection rules. |
| M5.1e — Review the complete baseline | Extend notebook06 with the likelihood, grid reference, a bounded checkpoint example, diagnostic/price plots, actual gate outcomes and a brief cost breakdown. | Execute/export offline without market fixtures or automatic training/cluster submission. A provisional result can be reviewed but cannot be relabelled an accepted participant checkpoint. M5.2 surrogate training starts as its own subsequent acceptance step. |

Keep these steps within the existing increment 7 scope. Plain functions, the
shared option bank, small cached reference arrays and the existing review
notebook are sufficient. A finite numerical experiment with unresolved gates
remains useful evidence; it does not require restarting the simulator ladder
or repeatedly tuning the neural model before recording what was learned.

## M5.2 bounded surrogate subplan

The [fixed-contract protocol](../../hkde-surrogate-protocol.md) adds a supervised
degree-six tensor Chebyshev fit for the two existing target puts. Its 49
coefficients per output keep the first two-dimensional price approximation
inspectable; neural regression and variable contract inputs are unnecessary for
this bounded task. The protocol is frozen before fitting.

| Substep | Concrete output | Acceptance boundary |
|---|---|---|
| M5.2a — Separate regression contract | Fixed Q context/target contracts, parameter rectangle, independent train/validation/test streams, numerical expectation labels | No quote errors or single-payoff labels; no extrapolation or arbitrary-contract claim. |
| M5.2b — One fitted map | 512 joint parameter bases, one degree-six least-squares solve and compact reloadable coefficient checkpoint | Group both contract labels per row; require full rank and stable design, exact reload and domain/context rejection. |
| M5.2c — Paired validation | New validation rows, boundary-inclusive reference grid, tighter Fourier labels and matched posterior price propagation | Report regression error separately from the preserved M5.1 inference miss. No degree/seed sweep; maximum substitution error 0.003 currency. |
| M5.2d — New regression holdout and cost | 256 fresh test parameter pairs only after validation passes; actual direct-versus-surrogate timings | This is not the old posterior test. Compare recomputed reference prices, include setup/loading, and separate component speed from end-to-end inference. |
| M5.2e — Offline review | Notebook07 with math, live analytic regression, measured error/cost plots, supported domain and remaining gates | Execute without market data or ML runtime; cached evidence does not rerun generation or training. M8 remains a separate selected participant release. |

## Didactic review deliverable

Notebook06 reviews the parameter posterior and reference pricer; notebook07 reviews the fixed-contract supervised surrogate. A direct-price posterior remains an optional separate experiment. Follow the [common review requirements](../README.md#didactic-review-notebooks-at-every-milestone). Explain each conditional distribution or regression target, then run one bounded synthetic-panel example through the implemented routes. Display inference errors separately from pricing errors and measured costs, with links to exact checkpoint/bank/reference reports. Failed or unimplemented routes remain visibly distinct from the accepted baseline. Exercises should trace a training label through each route and distinguish a distribution of model prices from realized payoff uncertainty.

Accept the sections after offline headless execution/export reproduces the included routes and renders their independent gates. Expensive label generation and retraining are opt-in commands; optional direct-price work cannot block the M5.1 review.

## Deliverables

A validated parameter-posterior workflow, reference posterior price distributions, a supervised pricing prototype and domain/error report, verified M5 review sections, and a short decision on whether direct posterior-price inference adds value. Keep model-posterior and surrogate checkpoints distinct.

## Acceptance and tests

Evaluate conditional posterior agreement and risk/price functionals alongside SBC. Compare surrogate bias, tail errors, bounds/parity where applicable, and resulting posterior-price quantiles with reference calculations. Measure total inference and pricing cost, including preprocessing, amortization cost, and hardware. A faster evaluation is acceptable only at an explicitly reviewed accuracy level.

## Decision gate and deferred scope

Accept or reject the surrogate component against its numerical reference
independently; accept the combined posterior-price route only when both
components pass. Direct posterior-price inference has its own gate and does
not block either path. All three routes support S without market files; E
requires genuinely excluded same-time quotes, including exclusion from smoothing
and input construction. Q price uncertainty is not a forecast of future chain
movements or P losses. American/exotic S outputs need validated target pricers,
not observed target quotes; their implementation remains deferred. Also defer
unverified exactness claims and arbitrary-payoff generalization.

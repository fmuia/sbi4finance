# Numerical experiment plan for Heston and rough Bergomi

Proposed on 26 October 2026 for the course in [PR 44](https://github.com/fmuia/sbi4finance/pull/44), inspected at `cebfb1f`.

The aim is a tutorial in which we can explain **what the network learned, how we checked it, when it saves work, and what its uncertainty means**. We need reusable simulated data, trained networks, trusted comparison posteriors, and a small set of clear figures. More architectures are useful only if they answer those questions.

**Prefer PyMC where it fits naturally**, aiming for a reusable core reference/emulator workflow. Specialized teaching or particle/pseudo-marginal samplers can retain documented custom implementations, and adoption can be deferred when its cost outweighs its value. This guidance is recorded in [AGENTS.md](../AGENTS.md) and [the MCMC policy](../docs/mcmc-policy.md); wholesale sampler migration is not a prerequisite for the campaign.

The recommendation is to grow two reusable numerical banks in stages, compare networks through one training and evaluation workflow, and spend expensive reference computation on a shared collection of cases. A bank is a collection of synthetic option surfaces with their generating parameters and numerical error records. Every network can learn from the same checked snapshot; expansions reuse earlier surfaces without repricing them.

This is a proposal for the next campaign. [The protocol](EXPERIMENT-PROTOCOL.md) specifies the comparisons and acceptance rules. [The tutorial review](TUTORIAL-REVIEW.md) explains the educational strengths and the claims that still need evidence. The existing [cluster runbook](CLUSTER-RUNS.md) remains the source of implementation commands; its environment and checks need updating before this campaign starts.

Implementation has started in a separate worktree: the [full-surface guide](../docs/full-surface-calibration.md) records the available commands and remaining scientific gates. Small generation/training/reload checks are implementation evidence, not completion of this campaign or approval to scale the candidate rough prior.

The [26 October Oscar report](../docs/oscar-pilot-2026-10-26.md) records actual CPU/GPU readiness, the first Heston development bank/fits, and unresolved numerical, curve-family and MCMC gates. In particular, do not scale the candidate rough bank from software readiness alone.

## Tutorial scope after self-review

The core deliverable is two trustworthy, reproducible teaching examples: full-surface Heston, followed by full-surface rough Bergomi with a jointly inferred Heston-shaped initial forward-variance curve. Complete both workflows through data, training, reload, inference, scientific checks and one downstream product. Participants load supplied checkpoints and follow inference → diagnostics → product uncertainty; bank generation, cluster scheduling and recovery remain instructor tools.

**Sparse option chains are a separate downstream unit of work.** The core tutorial does not implement, train or claim support for missing quotes, sparse layouts, changing masks or arbitrary raw chains. Its input is the complete declared 63-contract design, with valid prices and noise metadata. Reject missing/nonfinite entries and unsupported designs explicitly. Market gridding/interpolation, if used, remains a labelled preprocessing approximation; it does not establish missing-data support. The [sparse-quote follow-up](SPARSE-QUOTE-CASE-PLAN.md) starts with specification after the core tutorial is complete.

Keep numerical error checks, independent holdouts, converged references, calibration plus conditional checks, matched comparison inputs and honest cost/approximation labels. The passing infrastructure tests establish data/recovery behaviour; they do not establish pricing accuracy or posterior correctness. The next evidence must come from the numerical pilot and actual end-to-end examples.

Keep architecture variety as a bounded screen: four posterior families with one encoder on Heston, two additional encoder comparisons with one family, and two promising configurations on an intermediate rough bank. Confirm only the selected teaching configurations across three seeds. Million-surface banks are expansion capacity, not a requirement to train every candidate at every size. The full Cartesian benchmark, runner-up seed campaigns and direct rough MCMC are optional follow-ups.

Use the existing tested shard/snapshot and output-protection tools, then stop infrastructure work unless the pilot reveals a concrete problem. Implement only the BayesFlow 2.1 components needed for the complete-surface examples. Sparse adapters, mask-conditioned training, generalized raw-contract support and warm-start machinery are outside the core scope. PyMC adoption remains useful where natural and can be deferred. One central execution plan and the existing run metadata are sufficient; the protocol and deferred case guide are references, not separate software projects.

## What we should have at the end

- A reproducible BayesFlow 2.1 training environment, with safe checkpointing and one configurable workflow.
- One selected full-surface checkpoint per model, with results across three training seeds: six confirmation fits in total. The primary rough configuration jointly infers six unknowns; supplied-curve rough inference is an optional conditional comparison. Preserve completed screen results without requiring runner-up confirmation campaigns.
- A reference collection that lets us compare neural posteriors, MCMC and importance sampling on the same observations.
- Calibration, conditional accuracy, pricing uncertainty and cost figures with clear provenance.
- A bounded live demonstration and a cached version of every result used on stage.

There are two core teaching tasks:

| Task | What is observed | What is inferred | Why it is useful |
| --- | --- | --- | --- |
| Heston | A 63-price option surface | Four parameters; mean reversion is fixed | An inexpensive likelihood makes the neural answer checkable against MCMC. |
| Joint rough Bergomi | A 63-price option surface, with no supplied variance curve | H, eta, rho and three coefficients of a Heston-shaped initial curve: six unknowns | Tests amortization while propagating model and curve uncertainty together. |

Complete inputs do not guarantee that every parameter or product is well identified. Teach posterior dependence, prior/noise sensitivity and downstream uncertainty within these full-surface tasks. Joint curve inference remains part of rough Bergomi; postponing sparse chains does not turn the curve into supplied context. Existing sparse/EWG studies retain their historical labels and do not gate this release.

The forward-variance curve is `xi_0(t) = E^Q[v_t]`, the risk-neutral expected instantaneous variance at future time t. The existing market pipeline estimates it from option strips and holds it fixed, omitting its estimation uncertainty. That remains a **conditional calibration using a market-implied curve estimate**, not an additional exact independent observation. It is now optional rather than the primary rough lesson. The [protocol](EXPERIMENT-PROTOCOL.md#what-the-forward-variance-curve-input-means) describes that construction and the new joint approach.

### Rough curve baseline and bounded flexibility check

Use **rough Bergomi with a Heston-shaped initial forward-variance curve** as the initial joint-calibration baseline:

```text
curve(t) = long_variance
           + (short_variance - long_variance) * exp(-decay * t)

joint unknowns = H, eta, rho, short_variance, long_variance, decay
```

Infer all six from the option quotes. Positive endpoint variances and positive decay give a positive curve. These are the curve's own coefficients, not numbers copied from a Heston fit or additional Heston stochastic dynamics. The same functional family makes the progression easy to teach while reducing the old ten-unknown joint problem to six. Near-flat curves may leave decay weakly identified; assess curve bands and product values rather than demanding precise recovery of every coefficient.

The literature supports this as a transparent simplifying baseline, not a universally standard six-parameter model. [Rømer (2022), Section 3.3](https://doi.org/10.1080/14697688.2022.2081592) uses Heston-derived curves, with perturbations and other shapes, to generate rough-pricer training data; its market calibration is not restricted to three curve coefficients. [Baschetti, Bormetti and Rossi, Section 4.3](https://arxiv.org/html/2306.11061v2#S4.SS3) jointly calibrate a five-coefficient exponential-plus-hump family; the Heston-shaped curve is its no-hump special case.

Before a large rough bank, compare the baseline with that published hump-capable family on one small, predeclared development panel. Check quote fit and downstream valuation/uncertainty against numerical error, using the same observations and error model. Use a suitable numerical pricer or an emulator validated for the candidate family; a baseline-only emulator does not cover a new hump parameter automatically. This is one adequacy check, not another architecture campaign. If the monotone baseline materially distorts the lesson, promote the published family (eight joint unknowns) and freeze that task instead before scaling; otherwise keep the six-unknown baseline and report the restriction.

Heston's current mean-reversion speed is fixed at 2, whereas rough curve decay is inferred. Cross-model differences also reflect those assumptions and the different variance processes. Use matched curve values/decay for any deliberately controlled cross-model diagnostic; do not attribute every difference solely to roughness. Product values beyond quoted maturities also depend on the chosen curve extrapolation.

The tutorial release and the wider benchmark have different completion criteria:

| Work | Required for |
| --- | --- |
| One checked production configuration per retained task, references, diagnostics and participant packaging | The basic tutorial release. |
| The four-family and encoder comparison, including runner-up configurations | The architecture benchmark; unfinished comparisons can remain take-home work. |
| Sparse-chain specification, implementation and evaluation | A separate downstream unit; no sparse checkpoint, market case or reserve-coverage study gates the core tutorial. |
| Direct rough Monte Carlo-based MCMC anchors | A claim of agreement with that direct reference. The pilot may remain research if only emulator-MCMC comparisons are presented and labelled. |

## The work in order

| Step | Work | What we can show or decide afterward | Proposed lead role |
| --- | --- | --- | --- |
| 1. Prepare | Freeze the curve family, prior and six-target rough task after its bounded adequacy check; introduce BayesFlow 2.1 and verify the selected workflow. | A tiny run builds, trains, saves, reloads and samples correctly, with the intended unknowns. | Training and numerical methods |
| 2. Check the numerics | Run the Oscar pilot; measure quadrature, time-grid and Monte Carlo errors against quote noise. | The simulations are accurate enough for the intended claims, with difficult regions identified. | Numerical methods |
| 3. Generate the data | Produce or verify the Heston bank and generate the full rough bank. Record independent data splits and fresh quote-noise rules. | Every architecture receives the same financial experiments. | Numerical methods and infrastructure |
| 4. Compare networks | After checked baselines, run the bounded family/encoder screen and confirm the selected configurations with three seeds. | We can choose reliable tutorial configurations and explain measured trade-offs. | Training |
| 5. Build the references | Create Heston MCMC references; rough neural-pricer MCMC and independent pricing checks. | Agreement plots have a checked, accurately labelled comparison target. Direct rough MCMC is a separate research extension. | Reference methods |
| 6. Freeze and evaluate | Lock the shortlist, thresholds and inference settings; evaluate on untouched acceptance data. | We know which accuracy and calibration claims are supported. | Scientific review |
| 7. Package the lesson | Bind checkpoints and results to notebooks; rehearse the exercises, latency and cached fallbacks. | Collaborators and participants can reproduce the teaching path. | Tutorial delivery |

Steps can overlap after their inputs are fixed. Heston development and reference work can proceed while the rough bank is generated. Final acceptance waits for the shortlist to be frozen.

### Prepare the environment and workflow

Use **BayesFlow 2.1.0** for new candidates. It was released on 23 October and introduces MeanFlow, masking changes and inference fixes. Its saved networks are incompatible with 2.0.x, so archived 2.0.14 checkpoints keep their existing environment and the new tutorial candidates are retrained. Numerical data can be reused when their task and generation protocol match. [Release notes](https://github.com/bayesflow-org/bayesflow/releases/tag/v2.1.0)

Use a small configurable encoder/posterior factory and shared training/inference functions for the selected cases. Reuse existing checkpoint selection, reload checks and metadata rather than building a general workflow framework. The new calibration path currently has its own training/resume implementation; converge the behaviour needed by these examples. Sampling settings must belong to the chosen network family.

Use the output-protection and resume-identity safeguards implemented in the [incremental-bank PR](https://github.com/fmuia/sbi4finance/pull/47): reject nonempty incomplete runs and bind recovery to the original data/task/settings. Preserve and exercise those safeguards when introducing BayesFlow 2.1. Update the runbook's test command: it invokes pytest, which is absent from the current lock; the existing tests use unittest.

Assess one bounded PyMC reference route, starting with deterministic Heston or the price emulator. Prepare its locked environment and migrate only the callers/tests that benefit from it. Choose NUTS only for targets with validated deterministic gradients; use an appropriate gradient-free method for black-box targets. Record the reason when retaining a specialized sampler or deferring adoption. Particle/pseudo-marginal work needs correct state handling in any framework; it can remain a separate implementation.

### Generate enough data without generating it repeatedly

**Freeze the rough task before substantial generation.** A new six-target candidate generator and tiny configuration are now implemented in the [full-surface guide](../docs/full-surface-calibration.md), but its curve/prior/precision choices are not yet accepted. Historical supplied/seven-level rough banks and checkpoints remain separate: projecting their curves onto the new family does not preserve their option prices. Reuse compatible Heston data and the incremental-generation machinery, then generate the chosen complete-surface rough collection after its gates pass. Preserve all clean contract prices for a later compatible sparse study without adding sparse support now. Treat historical throughput estimates as references until the selected task is benchmarked.

| Bank | Initial training tranche | Validation surfaces | Test allocation |
| --- | ---: | ---: | ---: |
| Heston | 50,000 | 5,000 | 5,000 |
| Rough Bergomi | 100,000 | 5,000 | 7,000 |

These counts are inherited starting allocations, not empirically established sufficient sample sizes. Prepare for one million unique training surfaces per model, with learning curves at 50k/200k/1M for Heston and 100k/300k/1M for rough; smaller subsets remain useful for software and architecture screens. Heston generation is cheap enough to expand readily. Rough expansion is decided from development accuracy and measured Oscar throughput. Do not run every architecture at every size: establish data curves with credible baselines, then confirm finalists at the selected size. A later sparse study may reuse compatible clean prices after its observation/training contract is specified.

Use the [incremental-bank workflow](../docs/incremental-calibration-banks.md): one frozen collection per task/root seed/generator identity, append only missing split-qualified surface IDs, and publish immutable snapshots referring to checked shards. Keep the root seed fixed as IDs increase. Existing compatible shards can be adopted without repricing. Validate exact coverage, reject overlapping IDs, and audit the actual derived 64-bit pricing seeds across all included splits. Scheduling, chunk size and worker count must not change a surface. Keep validation/test IDs fixed throughout expansion and record each training run's exact snapshot hash.

Dataset growth starts a new training run; interruption recovery requires the identical snapshot and settings. The projected loader excludes covariance tensors during fitting, avoiding their roughly 32 GB per million surfaces in training memory. It still materializes selected training arrays. Use the validation/development split for learning curves, preserve final acceptance, and distinguish insufficient optimizer training or sampler accuracy from insufficient unique surfaces. Judge conditional agreement, calibration, difficult prior/curve regions and downstream quantities as well as validation loss.

Keep clean prices and numerical-error/covariance records. Add fresh quote noise during training, keeping every version of a base surface in the same split. Additional noisy copies are not independent new surfaces.

Use validation data for architecture choices and development diagnostics. Inventory which existing test cases have already influenced decisions. Reserve genuinely untouched IDs for final acceptance, or generate a small fresh acceptance bank with a new seed when a pool has already been inspected. Previously published Heston and pilot results remain historical evidence. See the [holdout rules](EXPERIMENT-PROTOCOL.md#data-and-holdouts).

### Compare the networks in stages

There are two separate choices: how quotes are represented, and how the conditional posterior is represented.

| Observation representation | What we want to learn |
| --- | --- |
| Fixed vector without learned compression | Whether a small, complete surface needs a summary network at all. |
| DeepSet | Whether permutation-invariant compression of complete quote records helps inference. |
| SetTransformer | Whether attention across strikes and maturities improves inference. |
| Small surface CNN, optional | Whether local grid structure helps enough to justify another comparison. |

Set inputs contain all 63 prices together with contract coordinates and quote noise. Test invariance to reordering complete quote records. The encoder comparison concerns representation of complete surfaces; mask/padding support and missingness augmentation are deferred. Recurrent networks belong more naturally in the returns track. A set encoder alone does not establish support for incomplete chains or new strikes/maturities.

Compare these four posterior families:

| Family | Reason to include it |
| --- | --- |
| Affine coupling | A simple invertible baseline with an inexpensive density. |
| Spline coupling | More flexible transformations for skewed or curved posteriors. |
| FlowMatching | Flexible generation with an explicit numerical-integration cost. |
| MeanFlow | A test of whether few-step inference retains accuracy at lower cost. |

MeanFlow is the method confirmed in discussion; it is experimental in BayesFlow 2.1. Compare its 1/2/4/8-step settings after training the same network. [MeanFlow](https://github.com/bayesflow-org/bayesflow/pull/802)

First get one checked baseline per model. Then screen all four families on Heston with one encoder, compare DeepSet and SetTransformer against the direct-vector baseline with one fixed family, and carry two promising configurations to an intermediate rough bank. This is at most eight short screen fits, reusing identical runs where possible. Confirm the two selected full-surface configurations with three seeds: six confirmation fits, or at most fourteen posterior fits including screens before selected-baseline learning curves. Forward-emulator training is additional. Sparse fits and mandatory supplied-curve rough fits are outside this programme. The [training matrix](EXPERIMENT-PROTOCOL.md#training-comparisons) records it.

### Make the MCMC comparison credible

**Heston:** build a development reference set first, then a final set of 50 predeclared synthetic cases plus SPY. Use a refined Fourier likelihood and four dispersed chains. Check chain convergence and numerical-price stability before treating a posterior as a reference.

**Rough Bergomi:** use two teaching-path checks and one optional research extension:

1. MCMC through a separately trained and validated price emulator on about 50 cases. This is a broad approximation cross-check; shared bank bias can affect both methods.
2. Simulation-based likelihood audits on 50 fixed cases, extending to 200 after checking reliability and cost.
3. Optional research: a direct Monte Carlo-based MCMC pilot on one case, followed by 3–5 predeclared anchor cases only if the tutorial intends to claim agreement with that direct target. Emulator-MCMC results can be taught with their approximation stated and independent pricing checks; they do not require completing this extension.

The current “16.9 CPU-hours per rough posterior” is an extrapolation from Heston mixing, not a measured rough-chain cost. At the stored cost of about five CPU-seconds per price-vector evaluation, four chains of 20,000 steps would use about 111 CPU-hours per case before additional refinement. Pilot mixing and likelihood noise before fixing the larger budget.

Every comparison must state its target. MCMC on an emulator, MCMC using one fixed pricing seed, and MCMC for the simulator averaged over pricing randomness solve different numerical problems. A noisy option-price estimate is not automatically an unbiased likelihood estimate for ideal prices. The [reference protocol](EXPERIMENT-PROTOCOL.md#reference-posteriors) gives the required distinctions.

### Evaluate usefulness as well as agreement

Report raw neural results and corrected results separately. Assess conditional posterior shapes and product quantiles, simulation-based calibration (SBC: repeated checks on synthetic data), interval coverage, contraction relative to the prior, and joint quantities such as correlation times vol-of-vol. Record failures and inconclusive references.

Measure the competing workflows on the same recurring task: for example, 100, 1,000 and 10,000 compatible smiles/scenarios. Include bank generation, training, compilation and any required per-smile audit. Compare against classical-proposal importance sampling and price-emulator MCMC. Validation is a useful workload, but should not be the sole reason offered for introducing an amortizer.

Agreement with classical uncertainty is a good outcome. Examine calibration, conditional agreement and sensitivity to prior, quote noise and product extrapolation even with complete inputs; wider intervals alone do not establish better inference. Missing-quote comparisons belong to the downstream sparse unit.

## Budget and release decisions

| Work | Current planning basis |
| --- | --- |
| Oscar pilot | About 0.6 CPU-hours for 200 rough surfaces. |
| Historical rough bank estimate | About 360 CPU-hours for the previous 112k seven-level design; benchmark the new task before scaling. |
| Expansion capacity, historical-rate scenario | About 13 CPU-hours per million Heston surfaces and 3,200–3,315 per million rough surfaces, extrapolated from stored timings; add holdouts. If the new rough task has similar throughput, 200/400/800 continuously busy cores imply roughly 16–17/8–8.5/4–4.25 hours before overheads. The new-task pilot must establish the actual rate. |
| Historical full study programme | About 930 CPU-hours and two baseline GPU jobs for the earlier bank/thin programme, not the new six-target budget. |
| Architecture comparison | Additional GPU fits; reserve a capped pilot allocation, then budget from measured training throughput. |
| Direct rough references, optional research | Separate allocation only if that extension and its direct-agreement claim are retained. |
| Fresh acceptance data if needed | Additional generation cost; separate from the existing bank budget. |

These are historical planning references: the 360 bank hours were included in the earlier 930-hour programme. The new curve/prior changes generation cost and makes that programme's total nontransferable. Re-estimate from the new-task pilot; budget selected posterior fits, one forward emulator and the bounded curve check separately. CPU-hours describe total work, not elapsed time under an assumed queue allocation.

At the historical rough rate, growth from 100k to 1M would add about 2,880–2,985 CPU-hours while reusing the first tranche. Use this only as an expansion scenario until the new task is benchmarked. The core-count examples are not confirmed Oscar allocations or measured cluster throughput.

The curriculum currently targets 30 October for slot 4, 4 November for slots 1–3, 6 November for slots 5–6, a 9 November rehearsal and an 11 November release. Use those as **claim-freeze dates**: ship the conclusions the results support at that date. Preserve a checked Heston lesson as the fallback. If rough evidence misses its gate, narrow that section to a clearly labelled illustration or remove the unsupported claim; do not relax the criterion to finish the story. Sparse-chain work is deferred regardless of those dates.

The organizers should agree on four decisions before the campaign is queued:

1. The claims and complete-input contract for the two full-surface configurations, narrowing the release if scientific gates fail. Sparse-chain support is excluded and will be specified separately downstream.
2. The named leads for numerical generation, training, references and delivery.
3. Initial CPU/GPU allocations and the maximum expansion allowed after each pilot.
4. The statistical/numerical target and proposed acceptance rules in the protocol.

This planning PR records those proposals. Job submission and spending are a later execution step once the run definitions and allocations are agreed.

## First implementation work

The next implementation work should freeze the Heston-shaped curve prior and implement the six-target rough generator/adapter and numerically consistent curve handling, checking the bounded hump alternative before large generation. Complete the BayesFlow 2.1 environment and minimal factory/sampling route for the complete-input examples, including explicit rejection of incomplete inputs. Reuse the data/recovery safeguards in the [incremental-bank PR](https://github.com/fmuia/sbi4finance/pull/47); do not expand them into a production framework. Freeze task settings and holdout IDs, run the new-task Oscar numerical/throughput pilot, and complete and package one checked full-surface example per model. Assess a bounded PyMC route where useful, then widen data and architecture comparisons only as needed. Start the separately specified sparse unit after core completion, unless an explicit scope decision changes that order.

The execution report should link the exact code revision, task/bank/checkpoint identities, compute use, failures and the claims each result supports. Update [study status](../docs/course/status.md), [claims](CLAIMS.md) and the runbook together when results are accepted for teaching.

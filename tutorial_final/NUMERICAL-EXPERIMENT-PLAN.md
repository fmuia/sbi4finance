# Numerical experiment plan for Heston and rough Bergomi

Proposed on 26 October 2026 for the course in [PR 44](https://github.com/fmuia/sbi4finance/pull/44), inspected at `cebfb1f`.

The aim is a tutorial in which we can explain **what the network learned, how we checked it, when it saves work, and what its uncertainty means**. We need reusable simulated data, trained networks, trusted comparison posteriors, and a small set of clear figures. More architectures are useful only if they answer those questions.

**Prefer PyMC where it fits naturally**, aiming for a reusable core reference/emulator workflow. Specialized teaching or particle/pseudo-marginal samplers can retain documented custom implementations, and adoption can be deferred when its cost outweighs its value. This guidance is recorded in [AGENTS.md](../AGENTS.md) and [the MCMC policy](../docs/mcmc-policy.md); wholesale sampler migration is not a prerequisite for the campaign.

The recommendation is to generate two numerical banks once, compare networks through one training and evaluation workflow, and spend expensive reference computation on a shared collection of cases. A bank is a collection of synthetic option surfaces with their generating parameters and numerical error records. Every network can learn from the same bank.

This is a proposal for the next campaign. [The protocol](EXPERIMENT-PROTOCOL.md) specifies the comparisons and acceptance rules. [The tutorial review](TUTORIAL-REVIEW.md) explains the educational strengths and the claims that still need evidence. The existing [cluster runbook](CLUSTER-RUNS.md) remains the source of implementation commands; its environment and checks need updating before this campaign starts.

## Tutorial scope after self-review

The deliverable is a small set of trustworthy, reproducible teaching examples: full and sparse Heston, followed by full and sparse rough Bergomi with a jointly inferred Heston-shaped initial forward-variance curve. First complete one workflow per model through data, training, reload, inference, scientific checks and one downstream product, then apply the fixed sparse layout. Participants load supplied checkpoints and follow inference → diagnostics → product uncertainty; bank generation, cluster scheduling and recovery remain instructor tools.

Keep numerical error checks, independent holdouts, converged references, calibration plus conditional checks, matched comparison inputs and honest cost/approximation labels. The passing infrastructure tests establish data/recovery behaviour; they do not establish pricing accuracy or posterior correctness. The next evidence must come from the numerical pilot and actual end-to-end examples.

Keep architecture variety as a bounded screen: four posterior families with one encoder on Heston, two additional encoder comparisons with one family, and two promising configurations on an intermediate rough bank. Confirm only the selected teaching configurations across three seeds. Million-surface banks are expansion capacity, not a requirement to train every candidate at every size. The full Cartesian benchmark, runner-up seed campaigns and direct rough MCMC are optional follow-ups.

Use the existing tested shard/snapshot and output-protection tools, then stop infrastructure work unless the pilot reveals a concrete problem. Implement only the BayesFlow 2.1 components needed for the chosen examples. Fixed layouts are sufficient for the sparse lesson; generalized missing-data/raw-contract support and warm-start machinery are outside the immediate scope. PyMC adoption remains useful where natural and can be deferred. One central execution plan and the existing run metadata are sufficient; the protocol and case guide are references, not separate software projects.

## What we should have at the end

- A reproducible BayesFlow 2.1 training environment, with safe checkpointing and one configurable workflow.
- One selected checkpoint for each of the four full/sparse teaching configurations, with results across three training seeds: twelve confirmation fits in total. The primary rough configurations jointly infer six unknowns; supplied-curve rough inference is an optional conditional comparison. Preserve completed screen results without requiring runner-up confirmation campaigns.
- A reference collection that lets us compare neural posteriors, MCMC and importance sampling on the same observations.
- Calibration, conditional accuracy, pricing uncertainty and cost figures with clear provenance.
- A bounded live demonstration and a cached version of every result used on stage.

There are four teaching tasks under two models:

| Task | What is observed | What is inferred | Why it is useful |
| --- | --- | --- | --- |
| Heston | A 63-price option surface | Four parameters; mean reversion is fixed | An inexpensive likelihood makes the neural answer checkable against MCMC. |
| Sparse-quote Heston | A declared subset of the same grid, with its quote-noise model | The same four parameters, with the same prior and fixed mean reversion | Isolates the effect of missing quotes while retaining a cheap numerical likelihood. |
| Joint rough Bergomi | A 63-price option surface, with no supplied variance curve | H, eta, rho and three coefficients of a Heston-shaped initial curve: six unknowns | Tests amortization while propagating model and curve uncertainty together. |
| Sparse-quote rough Bergomi | A declared subset of the same quotes, with their noise levels | The same six unknowns, prior and numerical model | Tests what incomplete coverage leaves uncertain and how that affects product valuation. |

Call the sparse lesson **Calibration from sparse option quotes**. Start with paired full/sparse Heston surfaces: subset the existing bank's prices, keep the same four targets/prior, and compare neural inference with converged Fourier-likelihood MCMC on exactly the observed subset. No new clean-price generation or extra curve targets are needed. Implement a small Heston quote-subset adapter; the current rough-specific thin workflow cannot be assumed to provide this route already.

Then perform the rough comparison with the same six unknowns inferred in both arms. Reuse the new rough bank across its two layouts. EWG is a current market candidate, not the task definition or a required final ticker. The [sparse-quote case plan](SPARSE-QUOTE-CASE-PLAN.md) defines the controls and market-selection criteria.

If the sparse-quote task remains live, it needs a checked checkpoint for its declared layout. Claims about population coverage require the corresponding studies. A curve inferred from a fuller surface must not be supplied to that task as though it were observed. The supplied-curve liquid example is a separate conditional benchmark; comparing it directly with joint sparse inference changes more than quote availability.

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

Heston's current mean-reversion speed is fixed at 2, whereas rough curve decay is inferred. Their full/sparse comparisons are matched within each model, but cross-model differences also reflect those assumptions and the different variance processes. Use matched curve values/decay for any deliberately controlled cross-model diagnostic; do not attribute every difference solely to roughness. Product values beyond quoted maturities also depend on the chosen curve extrapolation.

The tutorial release and the wider benchmark have different completion criteria:

| Work | Required for |
| --- | --- |
| One checked production configuration per retained task, references, diagnostics and participant packaging | The basic tutorial release. |
| The four-family and encoder comparison, including runner-up configurations | The architecture benchmark; unfinished comparisons can remain take-home work. |
| Sparse-quote population coverage studies under declared layouts and continuation laws | Any claim that the corresponding valuation bounds have demonstrated coverage or outperform the alternatives. |
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

Before cluster training, fix two concrete problems in [the training script](studies/train_calibration.py): an existing directory without a resume file can have its files deleted, and resume checks do not bind all data/task/settings identities. Update the runbook's test command: it invokes pytest, which is absent from the current lock; the existing tests use unittest.

Assess one bounded PyMC reference route, starting with deterministic Heston or the price emulator. Prepare its locked environment and migrate only the callers/tests that benefit from it. Choose NUTS only for targets with validated deterministic gradients; use an appropriate gradient-free method for black-box targets. Record the reason when retaining a specialized sampler or deferring adoption. Particle/pseudo-marginal work needs correct state handling in any framework; it can remain a separate implementation.

### Generate enough data without generating it repeatedly

**Freeze and implement the new rough task before substantial generation.** It needs a new task identity, curve-coefficient prior and pricing definition. Current rough configs, banks and checkpoints use supplied/seven-level curves and do not implement this six-target task. Keep them as historical or optional comparisons; projecting their curves onto the new family does not preserve their option prices. Reuse Heston data and the incremental-generation machinery, then generate a new rough collection whose complete prices can serve both full and sparse layouts. Treat existing throughput estimates as planning references until the new-task pilot measures them.

| Bank | Training surfaces | Validation surfaces | Existing test allocation |
| --- | ---: | ---: | ---: |
| Heston | 50,000 | 5,000 | 5,000 |
| Rough Bergomi | 100,000 | 5,000 | 7,000 |

These are starting budgets, not guarantees of accuracy. Test learning curves on nested training subsets before paying for expansion: 5k/20k/50k for Heston and 10k/30k/100k for rough. The rough bank can also supply the fixed thin-layout experiment.

Keep clean prices and numerical-error/covariance records. Add fresh quote noise during training, keeping every version of a base surface in the same split. Additional noisy copies are not independent new surfaces.

Use validation data for architecture choices and development diagnostics. Inventory which existing test cases have already influenced decisions. Reserve genuinely untouched IDs for final acceptance, or generate a small fresh acceptance bank with a new seed when a pool has already been inspected. Previously published Heston and pilot results remain historical evidence. See the [holdout rules](EXPERIMENT-PROTOCOL.md#data-and-holdouts).

### Compare the networks in stages

There are two separate choices: how quotes are represented, and how the conditional posterior is represented.

| Observation representation | What we want to learn |
| --- | --- |
| Fixed vector without learned compression | Whether a small, complete surface needs a summary network at all. |
| DeepSet | Whether a lightweight set encoder handles missing/subsampled quotes well. |
| SetTransformer | Whether attention across strikes and maturities improves inference. |
| Small surface CNN, optional | Whether local grid structure helps enough to justify another comparison. |

Set inputs contain prices together with contract coordinates, quote noise and masks. Tests must show that reordering complete quote records and adding padding leave the answer unchanged. Recurrent networks belong more naturally in the returns track. Supporting arbitrary new strikes and maturities requires corresponding training designs; a set encoder alone does not provide that support.

Compare these four posterior families:

| Family | Reason to include it |
| --- | --- |
| Affine coupling | A simple invertible baseline with an inexpensive density. |
| Spline coupling | More flexible transformations for skewed or curved posteriors. |
| FlowMatching | Flexible generation with an explicit numerical-integration cost. |
| MeanFlow | A test of whether few-step inference retains accuracy at lower cost. |

MeanFlow is the method confirmed in discussion; it is experimental in BayesFlow 2.1. Compare its 1/2/4/8-step settings after training the same network. [MeanFlow](https://github.com/bayesflow-org/bayesflow/pull/802)

First get one checked baseline per model. Then screen all four families on Heston with one encoder, compare DeepSet and SetTransformer against the direct-vector baseline with one fixed family, and carry two promising configurations to an intermediate rough bank. This is at most eight short screen fits, reusing identical runs where possible. Confirm only the four selected full/sparse configurations on the chosen banks with three seeds: twelve confirmation fits, or at most twenty posterior fits including screens before selected-baseline learning curves. Forward-emulator training is additional. Do not repeat the architecture matrix for sparse layouts or add mandatory supplied-curve rough fits. The [training matrix](EXPERIMENT-PROTOCOL.md#training-comparisons) records this programme.

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

On the liquid example, agreement with classical uncertainty is a good outcome. On the thin example, wider intervals alone do not establish better inference. Examine coverage and sensitivity to the assumed prior, quote noise and unquoted curve behaviour.

## Budget and release decisions

| Work | Current planning basis |
| --- | --- |
| Oscar pilot | About 0.6 CPU-hours for 200 rough surfaces. |
| Historical rough bank estimate | About 360 CPU-hours for the previous 112k seven-level design; benchmark the new task before scaling. |
| Historical full study programme | About 930 CPU-hours and two baseline GPU jobs for the earlier bank/thin programme, not the new six-target budget. |
| Architecture comparison | Additional GPU fits; reserve a capped pilot allocation, then budget from measured training throughput. |
| Direct rough references, optional research | Separate allocation only if that extension and its direct-agreement claim are retained. |
| Fresh acceptance data if needed | Additional generation cost; separate from the existing bank budget. |

These are historical planning references: the 360 bank hours were included in the earlier 930-hour programme. The new curve/prior changes generation cost and makes that programme's total nontransferable. Re-estimate from the new-task pilot; budget selected posterior fits, one forward emulator and the bounded curve check separately. CPU-hours describe total work, not elapsed time under an assumed queue allocation.

The curriculum currently targets 30 October for slot 4, 4 November for slots 1–3, 6 November for slots 5–6, a 9 November rehearsal and an 11 November release. Use those as **claim-freeze dates**: ship the conclusions the results support at that date. Preserve a checked Heston lesson as the fallback. If rough or sparse-quote evidence misses its gate, narrow that section to a clearly labelled illustration or remove the unsupported claim; do not relax the criterion to finish the story.

The organizers should agree on four decisions before the campaign is queued:

1. Whether the sparse-quote lesson and a population coverage claim are required for this release; select its market illustration separately.
2. The named leads for numerical generation, training, references and delivery.
3. Initial CPU/GPU allocations and the maximum expansion allowed after each pilot.
4. The statistical/numerical target and proposed acceptance rules in the protocol.

This planning PR records those proposals. Job submission and spending are a later execution step once the run definitions and allocations are agreed.

## First implementation work

The next implementation work should freeze the Heston-shaped curve prior and implement the six-target rough generator/adapter and numerically consistent curve handling, checking the bounded hump alternative before large generation. Complete the BayesFlow 2.1 environment and minimal factory/sampling route for the selected examples. Reuse the data/recovery safeguards in the [incremental-bank PR](https://github.com/fmuia/sbi4finance/pull/47); do not expand them into a production framework. Freeze task settings and holdout IDs, run the new-task Oscar numerical/throughput pilot, and complete one checked example per model before adding its sparse view. Assess a bounded PyMC route where useful, then widen data and architecture comparisons only as needed.

The execution report should link the exact code revision, task/bank/checkpoint identities, compute use, failures and the claims each result supports. Update [study status](../docs/course/status.md), [claims](CLAIMS.md) and the runbook together when results are accepted for teaching.

# Numerical experiment plan for Heston and rough Bergomi

Proposed on 26 October 2026 for the course in [PR 44](https://github.com/fmuia/sbi4finance/pull/44), inspected at `cebfb1f`.

The aim is a tutorial in which we can explain **what the network learned, how we checked it, when it saves work, and what its uncertainty means**. We need reusable simulated data, trained networks, trusted comparison posteriors, and a small set of clear figures. More architectures are useful only if they answer those questions.

**Prefer PyMC where it fits naturally**, aiming for a reusable core reference/emulator workflow. Specialized teaching or particle/pseudo-marginal samplers can retain documented custom implementations, and adoption can be deferred when its cost outweighs its value. This guidance is recorded in [AGENTS.md](../AGENTS.md) and [the MCMC policy](../docs/mcmc-policy.md); wholesale sampler migration is not a prerequisite for the campaign.

The recommendation is to grow two reusable numerical banks in stages, compare networks through one training and evaluation workflow, and spend expensive reference computation on a shared collection of cases. A bank is a collection of synthetic option surfaces with their generating parameters and numerical error records. Every network can learn from the same checked snapshot; expansions reuse earlier surfaces without repricing them.

This is a proposal for the next campaign. [The protocol](EXPERIMENT-PROTOCOL.md) specifies the comparisons and acceptance rules. [The tutorial review](TUTORIAL-REVIEW.md) explains the educational strengths and the claims that still need evidence. The existing [cluster runbook](CLUSTER-RUNS.md) remains the source of implementation commands; its environment and checks need updating before this campaign starts.

## Tutorial scope after self-review

The deliverable is a small set of trustworthy, reproducible teaching examples. First complete one Heston workflow and one rough-Bergomi workflow through data, training, reload, inference, scientific checks and one downstream product. Add one controlled sparse-quote contrast after those work. Participants load supplied checkpoints and follow inference → diagnostics → product uncertainty; bank generation, cluster scheduling and recovery remain instructor tools.

Keep numerical error checks, independent holdouts, converged references, calibration plus conditional checks, matched comparison inputs and honest cost/approximation labels. The passing infrastructure tests establish data/recovery behaviour; they do not establish pricing accuracy or posterior correctness. The next evidence must come from the numerical pilot and actual end-to-end examples.

Keep architecture variety as a bounded screen: four posterior families with one encoder on Heston, two additional encoder comparisons with one family, and two promising configurations on an intermediate rough bank. Confirm only the selected teaching configurations across three seeds. Million-surface banks are expansion capacity, not a requirement to train every candidate at every size. The full Cartesian benchmark, runner-up seed campaigns and direct rough MCMC are optional follow-ups.

Use the existing tested shard/snapshot and output-protection tools, then stop infrastructure work unless the pilot reveals a concrete problem. Implement only the BayesFlow 2.1 components needed for the chosen examples. Fixed layouts are sufficient for the sparse lesson; generalized missing-data/raw-contract support and warm-start machinery are outside the immediate scope. PyMC adoption remains useful where natural and can be deferred. One central execution plan and the existing run metadata are sufficient; the protocol and case guide are references, not separate software projects.

## What we should have at the end

- A reproducible BayesFlow 2.1 training environment, with safe checkpointing and one configurable workflow.
- One selected checkpoint for each retained teaching configuration, with results across three training seeds. Heston and liquid rough require six confirmation fits; any retained full/sparse joint-inference layouts are additional configurations and must be counted explicitly. Preserve completed screen results without requiring runner-up confirmation campaigns.
- A reference collection that lets us compare neural posteriors, MCMC and importance sampling on the same observations.
- Calibration, conditional accuracy, pricing uncertainty and cost figures with clear provenance.
- A bounded live demonstration and a cached version of every result used on stage.

There are three tasks under two models:

| Task | What is observed | What is inferred | Why it is useful |
| --- | --- | --- | --- |
| Heston | A 63-price option surface | Four parameters; mean reversion is fixed | An inexpensive likelihood makes the neural answer checkable against MCMC. |
| Liquid rough Bergomi | The same option design and a supplied forward-variance curve | Hurst parameter, vol-of-vol and correlation | Tests amortization when pricing requires simulation. |
| Sparse-quote rough Bergomi | A declared subset of option quotes, with their coordinates and noise levels | Those three parameters and seven curve levels jointly | Tests what incomplete coverage leaves uncertain and how that affects product valuation. |

Call the third lesson **Calibration from sparse option quotes**. EWG is the current implemented market candidate, not the task definition or a required final ticker. Start with controlled full-versus-sparse synthetic comparisons, inferring the curve in both arms, then choose a market illustration by data quality and teaching suitability. The [sparse-quote case plan](SPARSE-QUOTE-CASE-PLAN.md) defines the learning goal, controls and selection criteria.

If the sparse-quote task remains live, it needs a checked checkpoint for its declared layout. Claims about population coverage require the corresponding studies. A curve inferred from a fuller surface must not be supplied to that task as though it were observed. The supplied-curve liquid example is a separate conditional benchmark; comparing it directly with joint sparse inference changes more than quote availability.

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
| 1. Prepare | Freeze the statistical tasks, introduce BayesFlow 2.1, repair output/resume handling and test the common workflow. | A tiny run builds, trains, saves, reloads and samples correctly. | Training and infrastructure |
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

| Bank | Initial training tranche | Validation surfaces | Existing test allocation |
| --- | ---: | ---: | ---: |
| Heston | 50,000 | 5,000 | 5,000 |
| Rough Bergomi | 100,000 | 5,000 | 7,000 |

These counts are inherited starting allocations, not empirically established sufficient sample sizes. Prepare for one million unique training surfaces per model, with learning curves at 50k/200k/1M for Heston and 100k/300k/1M for rough; smaller subsets remain useful for software and architecture screens. Heston generation is cheap enough to expand readily. Rough expansion is decided from development accuracy and measured Oscar throughput. Do not run every architecture at every size: establish data curves with credible baselines, then confirm finalists at the selected size. The rough bank can also supply the fixed thin-layout experiment.

Use the [incremental-bank workflow](../docs/incremental-calibration-banks.md): one frozen collection per task/root seed/generator identity, append only missing split-qualified surface IDs, and publish immutable snapshots referring to checked shards. Keep the root seed fixed as IDs increase. Existing compatible shards can be adopted without repricing. Validate exact coverage, reject overlapping IDs, and audit the actual derived 64-bit pricing seeds across all included splits. Scheduling, chunk size and worker count must not change a surface. Keep validation/test IDs fixed throughout expansion and record each training run's exact snapshot hash.

Dataset growth starts a new training run; interruption recovery requires the identical snapshot and settings. The projected loader excludes covariance tensors during fitting, avoiding their roughly 32 GB per million surfaces in training memory. It still materializes selected training arrays. Use the validation/development split for learning curves, preserve final acceptance, and distinguish insufficient optimizer training or sampler accuracy from insufficient unique surfaces. Judge conditional agreement, calibration, difficult prior/curve regions and downstream quantities as well as validation loss.

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

First get one checked baseline per core model. Then screen all four families on Heston with one encoder, compare DeepSet and SetTransformer against the direct-vector baseline with one fixed family, and carry two promising configurations to an intermediate rough bank. This is at most eight short screen fits, reusing identical runs where possible. Confirm only selected teaching configurations on the chosen bank sizes with three seeds. The core two-model programme therefore budgets up to eight screens plus six confirmation fits, before selected-baseline learning curves. Additional sparse layouts have explicit extra fits. The [training matrix](EXPERIMENT-PROTOCOL.md#training-comparisons) records this smaller programme; a full benchmark is optional.

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
| Initial rough bank, 112k including holdouts | About 360–370 CPU-hours; benchmark on Oscar and scale the estimate. |
| Expansion capacity | About 13 CPU-hours per million Heston surfaces and 3,200–3,315 per million rough surfaces, extrapolated from stored timings; add holdouts. At 200/400/800 continuously busy cores, one million rough surfaces imply roughly 16–17/8–8.5/4–4.25 hours before queueing, startup, stragglers and assembly. |
| Existing full study programme | About 930 CPU-hours and two baseline GPU jobs, including the bank and thin-chain studies. |
| Architecture comparison | Additional GPU fits; reserve a capped pilot allocation, then budget from measured training throughput. |
| Direct rough references, optional research | Separate allocation only if that extension and its direct-agreement claim are retained. |
| Fresh acceptance data if needed | Additional generation cost; separate from the existing bank budget. |

These costs overlap: the initial bank's roughly 360 hours are already within the 930-hour programme. Bank expansion, the broader architecture campaign and direct rough references are additions, not work covered by the two baseline GPU jobs. Growing from 100k to 1M training surfaces adds roughly 2,880–2,985 CPU-hours at the stored rough rates, without repeating the original tranche. CPU-hours describe total work, not elapsed time under an assumed queue allocation. The core-count examples are arithmetic scenarios, not confirmed Oscar allocations or measured cluster throughput.

The curriculum currently targets 30 October for slot 4, 4 November for slots 1–3, 6 November for slots 5–6, a 9 November rehearsal and an 11 November release. Use those as **claim-freeze dates**: ship the conclusions the results support at that date. Preserve a checked Heston lesson as the fallback. If rough or sparse-quote evidence misses its gate, narrow that section to a clearly labelled illustration or remove the unsupported claim; do not relax the criterion to finish the story.

The organizers should agree on four decisions before the campaign is queued:

1. Whether the sparse-quote lesson and a population coverage claim are required for this release; select its market illustration separately.
2. The named leads for numerical generation, training, references and delivery.
3. Initial CPU/GPU allocations and the maximum expansion allowed after each pilot.
4. The statistical/numerical target and proposed acceptance rules in the protocol.

This planning PR records those proposals. Job submission and spending are a later execution step once the run definitions and allocations are agreed.

## First implementation work

The next implementation work should complete the BayesFlow 2.1 environment and the minimal factory/sampling route needed for one Heston baseline and one rough baseline. Reuse the data/recovery safeguards already available in the [incremental-bank PR](https://github.com/fmuia/sbi4finance/pull/47); do not expand them into a production framework. Freeze task settings and holdout IDs in the existing metadata, run the Oscar numerical/throughput pilot, then complete one checked end-to-end example for each core model. Assess a bounded PyMC route where useful. Widen data and architecture comparisons only after those examples work.

The execution report should link the exact code revision, task/bank/checkpoint identities, compute use, failures and the claims each result supports. Update [study status](../docs/course/status.md), [claims](CLAIMS.md) and the runbook together when results are accepted for teaching.

# Numerical experiment plan for Heston and rough Bergomi

Proposed on 26 October 2026 for the course in [PR 44](https://github.com/fmuia/sbi4finance/pull/44), inspected at `cebfb1f`.

The aim is a tutorial in which we can explain **what the network learned, how we checked it, when it saves work, and what its uncertainty means**. We need reusable simulated data, trained networks, trusted comparison posteriors, and a small set of clear figures. More architectures are useful only if they answer those questions.

The recommendation is to generate two numerical banks once, compare networks through one training and evaluation workflow, and spend expensive reference computation on a shared collection of cases. A bank is a collection of synthetic option surfaces with their generating parameters and numerical error records. Every network can learn from the same bank.

This is a proposal for the next campaign. [The protocol](EXPERIMENT-PROTOCOL.md) specifies the comparisons and acceptance rules. [The tutorial review](TUTORIAL-REVIEW.md) explains the educational strengths and the claims that still need evidence. The existing [cluster runbook](CLUSTER-RUNS.md) remains the source of implementation commands; its environment and checks need updating before this campaign starts.

## What we should have at the end

- A reproducible BayesFlow 2.1 training environment, with safe checkpointing and one configurable workflow.
- One selected checkpoint for each inference task retained in the tutorial, with results across three training seeds and the alternatives preserved. This is nine confirmation fits if all three tasks are retained.
- A reference collection that lets us compare neural posteriors, MCMC and importance sampling on the same observations.
- Calibration, conditional accuracy, pricing uncertainty and cost figures with clear provenance.
- A bounded live demonstration and a cached version of every result used on stage.

There are three tasks under two models:

| Task | What is observed | What is inferred | Why it is useful |
| --- | --- | --- | --- |
| Heston | A 63-price option surface | Four parameters; mean reversion is fixed | An inexpensive likelihood makes the neural answer checkable against MCMC. |
| Liquid rough Bergomi | The same option design and a supplied forward-variance curve | Hurst parameter, vol-of-vol and correlation | Tests amortization when pricing requires simulation. |
| Thin rough Bergomi | EWG-layout quotes with their noise levels | Those three parameters and seven curve levels jointly | Tests what changes when the data do not supply the curve. |

If EWG remains a live inference example, its third task needs a checked checkpoint. Claims about its population coverage require the corresponding coverage studies. A curve inferred from a fuller surface must not be supplied to that task as though it were observed.

The tutorial release and the wider benchmark have different completion criteria:

| Work | Required for |
| --- | --- |
| One checked production configuration per retained task, references, diagnostics and participant packaging | The basic tutorial release. |
| The four-family and encoder comparison, including runner-up configurations | The architecture benchmark; unfinished comparisons can remain take-home work. |
| EWG population reserve-coverage studies | Any claim that the thin-chain bounds have demonstrated coverage or outperform the alternatives. |
| Direct rough Monte Carlo-based MCMC anchors | A claim of agreement with that direct reference. The pilot may remain research if only emulator-MCMC comparisons are presented and labelled. |

## The work in order

| Step | Work | What we can show or decide afterward | Proposed lead role |
| --- | --- | --- | --- |
| 1. Prepare | Freeze the statistical tasks, introduce BayesFlow 2.1, repair output/resume handling and test the common workflow. | A tiny run builds, trains, saves, reloads and samples correctly. | Training and infrastructure |
| 2. Check the numerics | Run the Oscar pilot; measure quadrature, time-grid and Monte Carlo errors against quote noise. | The simulations are accurate enough for the intended claims, with difficult regions identified. | Numerical methods |
| 3. Generate the data | Produce or verify the Heston bank and generate the full rough bank. Record independent data splits and fresh quote-noise rules. | Every architecture receives the same financial experiments. | Numerical methods and infrastructure |
| 4. Compare networks | Screen observation encoders and four posterior families, then train the finalists with three seeds. | We can choose a reliable tutorial configuration and explain the trade-offs. | Training |
| 5. Build the references | Create Heston MCMC references; rough neural-pricer MCMC and simulation audits; pilot direct rough MCMC. | Agreement plots have a checked comparison target, rather than another unchecked network. | Reference methods |
| 6. Freeze and evaluate | Lock the shortlist, thresholds and inference settings; evaluate on untouched acceptance data. | We know which accuracy and calibration claims are supported. | Scientific review |
| 7. Package the lesson | Bind checkpoints and results to notebooks; rehearse the exercises, latency and cached fallbacks. | Collaborators and participants can reproduce the teaching path. | Tutorial delivery |

Steps can overlap after their inputs are fixed. Heston development and reference work can proceed while the rough bank is generated. Final acceptance waits for the shortlist to be frozen.

### Prepare the environment and workflow

Use **BayesFlow 2.1.0** for new candidates. It was released on 23 October and introduces MeanFlow, masking changes and inference fixes. Its saved networks are incompatible with 2.0.x, so archived 2.0.14 checkpoints keep their existing environment and the new tutorial candidates are retrained. Numerical data can be reused when their task and generation protocol match. [Release notes](https://github.com/bayesflow-org/bayesflow/releases/tag/v2.1.0)

Make the observation encoder and posterior family configurable components of the shared training and inference lifecycle. The new calibration path currently has its own training/resume implementation; convergence should include checkpoint selection, reload verification, provenance, diagnostics and failure handling. Sampling settings must belong to the chosen network family.

Before cluster training, fix two concrete problems in [the training script](studies/train_calibration.py): an existing directory without a resume file can have its files deleted, and resume checks do not bind all data/task/settings identities. Update the runbook's test command: it invokes pytest, which is absent from the current lock; the existing tests use unittest.

### Generate enough data without generating it repeatedly

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

Start with small encoder screens, then compare all four families on Heston and an intermediate rough bank. Train the two strongest configurations per retained task on the full banks with three seeds. That is roughly 30–40 fits including screens, before optional learning curves. The [training matrix](EXPERIMENT-PROTOCOL.md#training-comparisons) makes the counts explicit. The tutorial ships selected configurations; the comparison report keeps all completed runs and failures.

### Make the MCMC comparison credible

**Heston:** build a development reference set first, then a final set of 50 predeclared synthetic cases plus SPY. Use a refined Fourier likelihood and four dispersed chains. Check chain convergence and numerical-price stability before treating a posterior as a reference.

**Rough Bergomi:** use three complementary checks:

1. MCMC through a separately trained and validated price emulator on about 50 cases. This is a broad approximation cross-check; shared bank bias can affect both methods.
2. Simulation-based likelihood audits on 50 fixed cases, extending to 200 after checking reliability and cost.
3. A direct Monte Carlo-based MCMC pilot on one case, followed by 3–5 predeclared anchor cases if the pilot is feasible.

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
| Full rough bank | About 360 CPU-hours; benchmark on Oscar and scale the estimate. |
| Existing full study programme | About 930 CPU-hours and two baseline GPU jobs, including the bank and thin-chain studies. |
| Architecture comparison | Additional GPU fits; reserve a capped pilot allocation, then budget from measured training throughput. |
| Direct rough references | Additional CPU allocation, decided after one measured pilot. |
| Fresh acceptance data if needed | Additional generation cost; separate from the existing bank budget. |

These costs overlap: the 360 bank hours are already within the 930-hour programme. The broader architecture campaign and direct rough references are additions, not work covered by the two baseline GPU jobs. CPU-hours describe total work, not elapsed time under an assumed queue allocation.

The curriculum currently targets 30 October for slot 4, 4 November for slots 1–3, 6 November for slots 5–6, a 9 November rehearsal and an 11 November release. Use those as **claim-freeze dates**: ship the conclusions the results support at that date. Preserve a checked Heston lesson as the fallback. If rough or EWG evidence misses its gate, narrow that section to a clearly labelled illustration or remove the unsupported claim; do not relax the criterion to finish the story.

The organizers should agree on four decisions before the campaign is queued:

1. Whether EWG and its coverage claim are required for this release.
2. The named leads for numerical generation, training, references and delivery.
3. Initial CPU/GPU allocations and the maximum expansion allowed after each pilot.
4. The statistical/numerical target and proposed acceptance rules in the protocol.

This planning PR records those proposals. Job submission and spending are a later execution step once the run definitions and allocations are agreed.

## First implementation work

The first implementation PR should provide the BayesFlow 2.1 environment, safe run/resume identities, configurable encoder/posterior factories, family-specific sampling and the tiny save/reload tests. In parallel, freeze the numerical task and holdout ledger. Then run the Oscar pilot, create Heston development references, and start data generation and bounded architecture screens.

The execution report should link the exact code revision, task/bank/checkpoint identities, compute use, failures and the claims each result supports. Update [study status](../docs/course/status.md), [claims](CLAIMS.md) and the runbook together when results are accepted for teaching.

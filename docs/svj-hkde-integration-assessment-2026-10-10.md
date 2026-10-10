# Interfacing with Francesco's SVJ feasibility study

> Historical context: the recommendations below describe the 10 October assessment. The [current implementation plan](simulator-plan/README.md) supersedes its proposed curriculum and sequencing choices, defines shared infrastructure for both applications, and leaves the live example open for review.

Audit date: 10 October 2026. Status: assessment and proposed integration plan, not a team-approved change of curriculum. The collaborator's code, checkpoint, technical document, and numerical result files were not available in the audited repository. Reported experiment results below come from the chat supplied by Alex and have not been reproduced.

## Recommendation

Make importing and validating the existing SVJ experiment the next implementation milestone. Avoid independently rebuilding it or expanding several competing models. If the checks below succeed, use one SVJ-from-returns workflow throughout the live tutorial; retain HKDE European calibration and American-put pricing as an options companion. Keep MMAR as the existing runnable teaching asset/fallback and defer new rough-Bergomi implementation.

This revises the priority of the earlier HKDE plan because the new experiment targets latent-state inference and conditional financial risk directly. It does not establish that HKDE was unsuitable or that SVJ's reported posterior is already accurate enough to teach.

Suggested narrative: infer uncertain dynamics and current volatility from returns, propagate both into future losses, and use amortization to make repeated posterior inference practical. Avoid requiring either "frequentist methods fail" or "SBI is necessary" for the example to be worthwhile.

## What is actually upstream

Verified by remote refs, a temporary bare clone, and authenticated GitHub PR/release inventory; neither working checkout was pulled or modified during inspection.

| Location | Revision / status | Contents |
|---|---|---|
| `fmuia/sbi4finance` upstream `main` | `b892fae382b29ac3350246e3be2bc9946e60437f` | Merge of scaffold PR #2 on 10 October; tree identical to scaffold commit below |
| Local `sbi4finance-scaffold`, branch `scaffold-env-marimo-oscar` | `9a3abf86902ec72aa859bf060b3ec05edd42581d` | Pixi environment, marimo MMAR notebook, simulator, training script, checkpoint, cached SPMO/VOO returns, cluster scripts |
| Local `sbi4finance`, branch `research-notes` | `eca2420510a462ad2fbd6998312a2b96121cffd5` | Tracked README only; `docs/` untracked and local |
| HKDE/SVJ study artifacts | Not found | No model implementation, study checkpoint, posterior-reference draws, result JSON, technical design, or submitted proposal in the inspected upstream tree |

At inspection, upstream had only `main` and the scaffold branch; PR #2 was the only PR and was merged. No releases or pending PRs were listed. This establishes absence from this repository's inspected refs, not absence from Francesco's machine, another repository, or Claude's document storage.

References: [merged PR #2](https://github.com/fmuia/sbi4finance/pull/2), [pinned upstream tree](https://github.com/fmuia/sbi4finance/tree/b892fae382b29ac3350246e3be2bc9946e60437f).

The submitted proposal's alleged commitment to return-series inference could not be verified: its text is not in this tree. The README provides a generic 115-minute schedule without choosing HKDE or SVJ.

## What changed in Francesco's discussion

The initial HKDE investigation asked how vanilla quotes constrain pricing dynamics and exotic prices. The later study asks how historical returns constrain latent volatility and jump dynamics, and what that means for future losses. Rough Heston and MS-GARCH were alternatives in the discussion, not additional implemented deliverables visible here.

| | HKDE option calibration | New SVJ returns experiment |
|---|---|---|
| Probability measure | Risk-neutral Q | Physical P |
| Observations | Cross-section of European prices/IVs plus contract and market inputs | Ordered historical returns with time and data conventions |
| Training observation | Numerically priced surface plus specified quote-error model | Simulated stochastic return sequence |
| Inference target | Pricing parameters conditional on supplied inputs | Apparently eight static parameters plus terminal latent variance; confirm exact definitions |
| Numerical reference | PROJ likelihood plus posterior sampler | Particle likelihood/filter plus particle MCMC |
| Downstream quantity | Another model price | Conditional future-loss distribution, VaR and ES |
| Additional requirement | Price accuracy relative to quote noise | Consistent latent-state filtering, transition law, and forward continuation |

SVJ is a family name. HKDE itself is a stochastic-volatility jump model, specifically with double-exponential price jumps. A Bates-style model typically uses Gaussian log-price jumps; SVCJ additionally involves jumps in variance. Do not equate these names or infer the exact implemented process from the chat. Neither ordinary Heston variance with price jumps nor a Bates variant is rough volatility.

A P-posterior cannot be passed straight into a Q-pricer. If the American notebook is retained, give it its own European/Q calibration context, or explicitly specify and identify the change of measure and risk premia. Sharing a model family does not supply that bridge.

## What the new evidence supports, and what remains open

- The reported 64-minute training run and 1-2-second CPU sampling make an amortized returns example plausible. Reproduce checkpoint loading and warm/cold timings; total risk-forecast time also includes forward simulation.
- The reported first reference case had posterior mean differences of 0.34-0.47 reference SD for several quantities, above the study's stated 0.3 criterion. Complete the reference comparisons and resolve these discrepancies before treating the checkpoint as accepted.
- Marginal SBC over 1,000 prior-predictive datasets is useful but does not establish accurate inference for every dataset or accurate joint dependence between parameters and terminal variance. Assess conditional cases and risk-relevant functionals as well.
- Coverage of 60-68% for nominal 95% jump-parameter intervals is evidence against the tested approximate-ML/Wald implementation. Forty series and an approximate likelihood do not establish failure of all frequentist inference. Compare the same datasets, disclose boundary/Hessian failures, and distinguish prior-averaged from fixed-parameter coverage. A profile/bootstrap check would strengthen broader claims.
- The reported 10-day VaR exceedance of 1.14% for a posterior-median plug-in versus 1.03% for posterior prediction supports a modest difference in that experiment. The posterior-median comparison isolates a plug-in approximation; it is not itself a Bayesian-versus-frequentist comparison. The smaller ML comparison remains less precise.
- An Acerbi-Szekely Z2 value of -0.10 is not by itself an estimate that ES is 10% too low. Z2 combines frequency and severity beyond a predicted tail threshold. Report the score as a score and calculate direct relative error against a carefully defined oracle ES separately.
- Treat the reported traffic-light calculation as an illustrative synthetic calculation until its construction is inspected. A pooled exceedance rate, heterogeneous per-series rates, and overlapping 10-day forecasts lead to different calculations. It does not establish regulatory compliance.
- A likelihood with latent-state integration is computationally difficult, not proof that neural SBI is necessary. Particle MCMC is a likelihood-based alternative and is already the study's reference. Its correctness applies to the specified state-space model under the method's conditions, not automatically to an underlying continuous-time process approximated by Euler steps.
- Cheap Euler simulation still requires checks of positive variance treatment, time-step error, jump-count approximation, and agreement with the filter. No runtime claim removes these numerical requirements.

Primary methodological sources: [particle MCMC](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf), [SBC test quantities](https://doi.org/10.1214/23-BA1404), [Acerbi-Szekely, equations 6-7](https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b). Model-family reference: [HKDE paper](https://arxiv.org/html/2502.13824v1).

## Relevance of the data-needs chat

The earlier HKDE/rough-Bergomi data discussion produced [tutorial-data-needs-2026-10-10.md](tutorial-data-needs-2026-10-10.md).

That note remains a valid options-workstream specification. Its opening assumption that HKDE is the required live example is now a curriculum decision to revisit, not an established common requirement.

| Existing work | Proposed treatment |
|---|---|
| Synthetic training versus market illustration; independent evaluation data; provenance and versioning | Reuse across both tasks |
| SPX sources, bid/ask-IV conventions, forwards, discounting, contract grid | Retain for HKDE/options companion |
| PROJ pricing and quote-noise specification | Retain inside HKDE; do not substitute these for SVJ transitions/return observations |
| American contract/exercise requirements | Retain for a separately identified Q-pricing extension |
| Large option purchases or new HKDE and rough-Bergomi banks | Defer while importing the SVJ study |
| Historical return preparation and rolling evaluation | Add a new returns-specific specification |

The earlier finding that one market surface is enough applies to an illustrative cross-sectional option fit. A return-series fit needs an ordered window; empirical rolling risk validation additionally needs chronological future observations. Synthetic learning still uses simulated sequences, not thousands of purchased market series.

The returns specification should name the actual instrument (index, ETF, spot commodity, or futures contract), source, adjusted-price convention, simple/log return convention, units, timestamps/calendar, gaps, sample cutoff, window length, and forecasting horizon. Futures require an explicit roll treatment; missing observations must not be filled as zero returns. A variance proxy may be optional, but adding it changes the observation model and training target.

Cached SPMO/VOO data in the scaffold are daily simple returns for an existing MMAR example. They are potential inputs to a separately documented returns adapter; they are not the reported S&P 500/WTI study data. Any log conversion and model support must be checked explicitly. Split market evaluation chronologically and avoid leakage from overlapping windows; split simulation seeds/base cases before any augmentation.

## Minimal building blocks to agree before porting

Use small model-specific modules and shared evaluation/reporting utilities. A universal financial-model framework is unnecessary for two notebooks.

1. **Versioned model specification.** Record dynamics, P/Q measure, parameter names/order/support/transforms, units, initial-state law, jump law, discretization and approximation tolerances. Keep a specification identifier in every checkpoint and result.
2. **Observation preparation.** A returns adapter preserves order, time increments and any supported mask; an option adapter preserves contract coordinates and quote/curve conventions. Share provenance metadata, not a forced common observation tensor or encoder.
3. **Reproducible simulation output.** For SVJ, return static parameter labels, return sequences and terminal-state labels with seeds/configuration. Terminal state belongs in the supervised target, never in the network's observed conditioning inputs. Keep diagnostic latent paths separate.
4. **Joint posterior interface.** Provide matched draws of `(theta, v_T)` conditional on the observed sequence, or parameter draws followed by a validated conditional state filter. Preserve dependence; independently pairing parameter and state marginals gives the wrong forecast. Verify network/adapter/backend compatibility before attempting to load the checkpoint.
5. **Forward-risk interface.** Continue each posterior state under its matched parameters. Generate loss scenarios at explicit horizons. Calculate posterior-predictive VaR/ES from the mixture of future losses. Separately expose the posterior distribution of conditional VaR/ES when discussing uncertainty about a risk number: averaging conditional quantiles does not produce the mixture quantile.
6. **Reference and result interface.** Store immutable case IDs, observations, truth when known, reference draws, diagnostic summaries, numerical settings and measured runtime. Compare methods on the same cases and target; capture simulation/training/compile/inference/forward-risk costs separately.

The scaffold already provides packaging, marimo, a BayesFlow adapter/training pattern, cached-data loading, and diagnostics presentation. Its MMAR weights and model-specific transforms cannot become SVJ weights/transforms. Francesco's flow-matching checkpoint may require its original network class and training environment; a switch to the scaffold's coupling flow is a new experiment, not a file conversion.

## Concrete next steps and ownership boundaries

Suggested responsibilities are roles to agree, not assignments sent to collaborators.

| Priority | Deliverable | Suggested boundary | Acceptance |
|---|---|---|---|
| 1 | Original SVJ export | Francesco supplies model/training/evaluation code, checkpoint, environment, technical design, exact data preparation, case/result files, and run order | Load and reproduce one saved case before rewriting anything |
| 2 | Integration branch based on current upstream main | Our side handles packaging and model-specific adapter | Import simulator and checkpoint; inference on one held-out case in a clean CPU environment |
| 3 | Shared model/target specification | Joint scientific review | Simulator, particle filter, training labels and forward forecasts use the same transition and conventions |
| 4 | Reference reproduction and acceptance report | Joint inference review | Complete reported PMCMC cases; inspect joint state/parameter and risk-functional agreement; explain or fix deviations |
| 5 | Small self-contained live notebook | Our side reuses marimo scaffold | Simulate/load one return window, obtain joint posterior, produce one risk plot; cached expensive outputs; timed CPU rehearsal |
| 6 | Returns data manifest and optional real illustration | Data workstream | Correct chronological window, source/conventions recorded, synthetic core independent of access |
| 7 | HKDE options companion | Separate extension track | Preserve existing data note; implement only after the SVJ core decision, or with independent capacity |

The artifact export should include all completed and failed cases, not only selected plots; report actual chain settings, ESS/R-hat, held-out seed generation, and risk-score definitions. The chat was interrupted before the remaining PMCMC, real-data and laptop results were supplied; do not mark them finished without their outputs.

## Decision gate for the two-hour session

Promote SVJ to the single running model when the imported checkpoint runs on attendee hardware, its posterior and conditional forecast checks support the intended claims, and the full 115-minute teaching path rehearses successfully. Use the same SVJ model with selected parameters/figures hidden for the basic segment; participants need not learn every parameter immediately.

If those checks fail, repair or transparently simplify the model and retrain/revalidate, or retain the runnable MMAR fallback with its interpretation corrected. Do not combine complete MMAR, HKDE and SVJ lessons to compensate. The current MMAR notebook still says the finite cascade likelihood is intractable and promises that all advanced work uses the same approximator; both need review when the curriculum is aligned.

Preserve the options research as a separately scoped companion. The previous American-option interest remains represented there, but it no longer follows automatically from the proposed P-measure live posterior.

This audit created only this assessment note. It did not change the existing data-needs note, import an unprovided study, run training, change branches, publish files, or contact Francesco/other chats.

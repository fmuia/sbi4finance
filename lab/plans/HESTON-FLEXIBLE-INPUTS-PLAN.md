# Heston follow-up: geometry, learned summaries and missing quotes

**Status, 9 November 2026: adopted planning sequence; implementation and evidence pending.** This is the separate Heston follow-up to the [noise and grid baseline](HESTON-NOISE-GRID-PLAN.md). That sibling plan is being prepared in a separate PR: merge both documents before executing their dependency, and use its accepted task record rather than treating the link as an already accepted model. This document authorizes no pricing, training, publication or replacement of existing checkpoints by itself. The user has requested parallel planning and implementation tracks; production runs still require the frozen specifications, measured budgets and gates below.

The sequence is **accepted fixed Heston baseline → varying geometry with complete direct inputs → learned summary on the same varied data → structured and random missingness**. Establish the known, fixed Gaussian scale first. Repeat the successful final recipe with one inferred Gaussian scale. Do not add the earlier additive discrepancy parameter `tau`, a Gaussian process, or American exercise to this task.

The accepted baseline may support the Bergomi notebook work while this Heston follow-up proceeds separately. Flexible inputs are not a prerequisite for that port. Neither Heston acceptance nor a working encoder establishes numerical or inferential acceptance for Bergomi; that application keeps its own curve, pricing and inference checks. This plan contains no Bergomi jobs.

## 1. Question, dependencies and scientific scope

The question is whether one estimator can preserve the specified posterior when the observed European contracts move within a declared region, and then when only a declared subset is available. A learned summary is a proposed representation, not evidence of a sufficient statistic. Missingness is an information-removal experiment, not a device for obtaining wide intervals.

Start only after the baseline plan has frozen and checked:

- the actual SPXW quote record, exercise and settlement conventions, observation clock, forwards and discounts;
- the five physical Heston targets, their prior bounds and transformations, numerical pricing controls, and a revised complete design;
- the Gaussian observation law, observed half-spread profile, known-scale baseline, separate inferred-scale task, conditional reference comparisons and failure handling;
- one affected forward-variance quantity and one nearby, well-supported control, with common product horizons and conventions.

Do not silently inherit historical four-target Heston bounds, the old interpolated 63 readings, or a previous engineering prior as the new accepted task. Record the literal accepted names and bounds in each new task manifest. Existing full-input checkpoints retain their original support and must continue rejecting incomplete or incompatible data.

[PR #81](https://github.com/fmuia/sbi4finance/pull/81), pinned to [df2cfc442de07459a8b971f070e3a6f25b0b2020](https://github.com/fmuia/sbi4finance/tree/df2cfc442de07459a8b971f070e3a6f25b0b2020), supplies useful candidate coverage and product questions. Its Fisher/Laplace design comparisons are preliminary, local evidence under their own fit and error assumptions; they neither choose this representation nor establish neural accuracy under the simplified Gaussian law. Adopt its coverage questions selectively. Exotics, barrier solvers, books, prudent-reserve claims, broader discrepancy models and rough-model importance references are outside this follow-up.

## 2. Observation and conditioning contract

For available contract `i`, use the baseline law in price units:

`Y_i = P_i(phi; design, carry) + c * h_i * Z_i`, with independent `Z_i ~ Normal(0, 1)`.

Here `phi` contains the five physical Heston parameters, `h_i > 0` is the declared price half-spread scale, and `c` is one dimensionless Gaussian multiplier. Equating a half-spread to one noise SD remains a stated convention. The main B1–B3 comparisons keep `c` fixed at the accepted known value, normally 1. They do not vary the spread profile at the same time. A later inferred-scale recipe uses the exact baseline prior for `c` (the proposed engineering prior is Uniform[0.5, 2], subject to that baseline's freeze).

In the known-scale task, `c` is fixed task metadata, or an observed condition only if the accepted task explicitly varies it. In the inferred-scale task it is a target: never give the network the generating `c`, `c*h_i`, `log(c*h_i)`, the noise realization, clean prices, or generating parameters. The observed `h_i` remains available. These tasks have different posterior targets; compare physical marginals and products on the same observations without pretending that conditioning on `c` and integrating it out are the same posterior.

Preserve raw records alongside the model input: source and contract IDs, option root, call/put, strike, expiration and settlement time, bid/ask/mid, quote timestamp, half-spread, forward, discount and provenance. One row is one actual contract location. Interpolated or extrapolated prices do not become new independent quotes. Reject American contracts and unsupported settlement mixtures.

The input schema should contain a forward-unit noisy price `Y_i/(D_i*F_i)`, `log(K_i/F_i)`, positive year fraction `T_i`, option type, and normalized declared spread `h_i/(D_i*F_i)`, together with the accepted global carry context. Give coordinate units and normalization explicitly. A display coordinate `z = log(K/F)/(sigma_ref*sqrt(T))` uses a **fixed, declared reference volatility**, not the inferred noise multiplier `c`; it does not replace the true maturity and strike metadata. Do not require implied-volatility inversion of noisy synthetic prices. Apply training-only standardization and the same invertible target transformations and density Jacobians in all comparisons.

## 3. Separate geometry from spread-profile support

Sampling actual contract locations also changes their raw bid/ask spreads. Combining those changes is a valid experiment, but it cannot isolate the effect of geometry. The primary B1 geometry study therefore moves locations while keeping one declared normalized uncertainty profile attached to canonical slots. Store its slot mapping and units. These are **synthetic uncertainty scenarios at actual quote locations**; the raw observed spreads are preserved separately and are not relabelled as that profile.

The fixed profile may be reused across B1, B2 and B3 because it is an explicit controlled assumption. It does not establish support for an arbitrary chain's spread profile. An actual-market application must use its appropriate observed `h_i` and pass a matching support and reference audit. Do not apply the original slot profile to a new chain and describe it as market measurement uncertainty.

Broader spread-profile variation is a separate, deferred extension. If requested, freeze its observed or synthetic profile catalogue, range, geometry dependence and train/development/acceptance split; compare it on the same clean prices and geometry before making generalization claims. A global multiplier `c` changes level, not the relative spread shape. No profile sweep is a mandatory additional campaign here.

## 4. Minimal staged comparison

| Stage | Geometry and availability | Representation | Main comparison |
| --- | --- | --- | --- |
| B0 | Accepted fixed design, complete | Accepted baseline input | Immutable reference task and checkpoint |
| B1 | Declared varying geometry, complete, fixed number `N` | Ordered direct vector containing all quote records and coordinates | Can one estimator condition on moving locations without compression? |
| B2 | Exactly B1's bank, splits, geometry and observations | One learned quote-set summary, same posterior family | Does compression preserve B1's posterior accuracy? |
| B3 | B2's parent geometry with declared masks | Same summary family, explicit availability mask and valid count | Does the estimator handle the supported information removal? |
| C | Successful final geometry/summary/mask recipe | Selected representation; one inferred `c` | Does the accepted recipe remain accurate for the larger target? |

If the B0 adapter does not already encode coordinates in the B1 form, add one fixed-geometry direct-input bridge fit using that form before attributing differences to varying geometry. Reuse it if the baseline already implements the same contract. This is a bridge, not an architecture screen.

Geometry and compression need not launch jointly. B1 deliberately uses no learned summary, so an unsupported geometry or conditioning failure is not hidden inside a new encoder. B2 changes the representation while holding the varied dataset fixed. A fixed-geometry summary probe is optional if a B2 failure needs localization; it is not another mandatory matrix. Do not launch a Cartesian product of posterior families, summaries, dimensions and masks.

### B1: a declared geometry distribution

Freeze a catalogue of eligible actual European contracts from the baseline snapshot, with positive maturity, valid quote metadata and declared quality filters. The first distribution selects exactly `N` distinct records, retaining the baseline's declared maturity bands and strike/wing coverage. The realized catalogue and number `N` follow available quotes; neither a nominal 63-point template nor an exact one-year PM expiry is promised. No AM fallback or interpolated contract is inserted to fill a slot.

Record the selection probabilities, canonical maturity/strike slots, sorting and tie-breaking rules, allowed bounds, and minimum separation between selected records. The direct vector uses stable ordering by full contract metadata, not a price-derived ordering. Include the accepted fixed design as an explicit supported anchor. Freeze a small number of geometry strata, including the shorter-maturity and wing limits; evaluate those strata separately rather than relying only on averages over common designs.

State which coordinates actually vary. Moving strikes at seven fixed expiries does not demonstrate conditioning on new maturities. A finite actual-contract catalogue supports held-out combinations within that catalogue; it does not establish continuous off-catalogue or out-of-range generalization.

Draw geometry independently of `phi`, Gaussian errors and split identity within each declared distribution. The normalized slot spread profile is also independent of `phi`. A geometry selected using a fitted volatility surface or generating parameter is a different conditional task and must not be introduced silently. At inference, the geometry is observed information; the simulator and reference likelihood condition on exactly the supplied coordinates.

Reuse physical-parameter rows and compatible clean prices from the accepted baseline where identities match. New strikes or maturities require clean prices at those locations; storing old values under new coordinates is invalid. Price one shared varied-geometry bank for B1 and B2, retaining numerical diagnostics. A continuous off-catalogue strike/maturity sampler is a later support extension with separate pricing and acceptance checks.

### B2: one controlled summary comparison

Start with one small, declared permutation-invariant summary, such as a 32-dimensional DeepSet, and retain the accepted posterior family. Freeze all effective defaults, including dropout, before training. Keep the posterior network's depth/width, optimizer, batch size, physical draws, quote errors, split IDs and checkpoint schedule aligned with B1 where compatible. Report the extra encoder parameters and compute rather than calling unlike models identical.

Each token contains the complete price/design/spread record. Preserve global conditions separately and report any bypass around the summary. Give the eventual B3 encoder a valid-record count or an aggregation that retains cardinality; mean pooling alone must not make repeated evidence indistinguishable. Do not treat correlated duplicate contract records as independent evidence to test this. Summary dimensionality is a development choice, not a theorem of sufficiency.

On complete sets, reorder whole tokens and verify invariant density/conditional summaries and seeded draws within a declared numerical tolerance. Compression acceptance requires matched conditional references, including difficult parameter/geometry combinations. SBC alone can average over local errors and is not sufficient. A second summary family or larger dimension is considered only after a documented B2 failure and a bounded follow-up decision.

### B3: masks and supported missingness

Start with three limited families on B2's parent designs: (a) remove one or more later maturity bands, (b) remove a declared put- or call-wing region, and (c) random thinning within declared per-expiry coverage and quote-count bounds. Freeze the finite severity levels, mixture probabilities, minimum/maximum valid counts and excluded empty or otherwise unsupported sets before generating outcomes. Include the complete-set anchor in training and evaluation. Keep missingness and the spread profile separate.

Use nested masks and preserve the same physical draw, parent geometry, surviving spread values and surviving Gaussian errors for paired full/restricted observations. Random masks in the primary study are independent of parameters and prices, conditional on the declared geometry. This supports controlled information-removal claims. A liquidity, bid/ask-quality or price-dependent selection rule needs its own simulated or conditioned selection mechanism; random thinning does not validate it.

Absent quotes contribute no likelihood term. Carry a boolean availability mask through attention, aggregation, normalization and any preprocessing; define valid count from it. Padding has no price or economic meaning. Test that replacing masked payloads with arbitrary finite values, adding padding, and reordering whole valid/padded records leave the result unchanged. Reject nonfinite values in available records and unsupported coordinates, masks or counts. Declare whether nonfinite padding is rejected at the boundary; do not rely on multiplication by zero to suppress NaNs.

Full and sparse reference posteriors use only their respective available quotes. Do not rebuild a forward, curve, carry estimate, scale or summary using withheld quote prices and then pass it to the sparse estimator. In the controlled experiment, hold the accepted carry context fixed as external conditioning over the union of supported expiries and disclose that assumption; this measures removal of option-price information conditional on that context. A later operational experiment that re-estimates context must use only permitted information and propagate or explicitly condition on its uncertainty.

## 5. Development data, bounded runs and identities

The following are **initial upper bounds to freeze or reduce after a smoke benchmark**, not instructions to launch every allocation. A candidate shared varied-geometry development bank has 20,000 training rows and 5,000 independent development rows, subject to measured repricing cost. Reserve 2,000 fresh acceptance rows in a disjoint namespace; never use them for model, mask, checkpoint or example selection. Keep acceptance payloads in a separate immutable archive with its own manifest; training/development loaders must neither open the reserved members nor write into their namespace. The later acceptance evaluator receives read-only access after the recipe freezes. Use the same split-level physical draws and geometry IDs across B1/B2/B3; masks and noise streams have recorded independent namespaces. B1/B2 share identical full observations, and B3 selects their surviving errors. Fresh Gaussian augmentation is allowed only with an identical recorded protocol in paired comparisons.

Reuse the fixed-design baseline collection for the optional bridge. A varied bank may expand to at most 100,000 training rows only after an explicit development decision identifies data coverage as an unresolved issue. Keep the physical draw, geometry draw, spread-profile ID, noise seed and mask ID independently reproducible. Splits must separate physical draws as well as observations; merely resampling noise on training parameters is not a fresh acceptance population.

| Component | Initial bound and stop |
| --- | --- |
| Software smoke | At most 128 rows, 5 updates and tiny reference checks; prove shapes, input exclusion, checkpoint reload and counts |
| New pricing pilot | At most 512 physical/geometry rows spanning declared extremes; refine and measure runtime before the shared bank |
| Initial known-scale diagnostics | One seed for B1, one for B2, one for B3; optional one bridge; batch 128, predeclared 2k/10k checkpoints, at most 10,000 updates and 60 minutes wall time per fit, whichever comes first |
| Inferred-scale diagnostic | At most one fit on the selected final recipe with the same initial 2k/10k and 60-minute bounds |
| Later optimization, if justified | Only after code/reference checks and development fidelity evidence, a separately reviewed continuation to at most 50,000 total updates for a selected recipe; no automatic continuation for every stage |
| Seed confirmation | At most two additional seeds for the selected recipe; the same initial diagnostic limits, not a full campaign for every stage |
| Conditional references | Start with 6–8 predeclared development cases spanning geometry, physical boundaries and mask families; at most 12 cases before another review |
| Fresh acceptance | Evaluate only after recipe/selection freeze; one declared population and reference case list, no best-case substitution |

Measure pricing cost, compilation, warm update throughput, memory and reference sampling cost on the actual device. Before each stage, bind those measurements to a numeric wall-time/CPU or GPU allocation, thread count, draw budget and hard stop in a run record. The initial 60-minute fit bound includes compilation; report setup and warm training clocks separately. Bank expansion, another summary, extra seeds, larger update budgets and retries are separate decisions. Code correctness and a checked reference comparison precede any long fit. The table is a count ceiling, not evidence that any allocation is sufficient. Do not extend a falling loss curve indefinitely.

Use one common, predeclared optimizer schedule horizon for checkpoint comparisons within a trajectory. Record exact updates and learning rates, cumulative data exposure and unique rows; a later continuation must explicitly preserve or separately declare its schedule rather than silently restart it. Retain all declared seeds and failed runs. A nonfinite batch, gradient or probe stops training and writes an atomic diagnostic containing the precise failing field and update, input/augmentation IDs and last recoverable optimizer/model/RNG state. Do not overwrite the last valid checkpoint. Never label an absent endpoint as completed. Resume only with verified weights, optimizer moments, standardization, effective randomness and absolute data/augmentation position, or declare the changed continuation protocol.

Every task, bank and checkpoint identity binds source revision and producer hashes, environment/device versions, prior and transformations, raw catalogue/design distribution, clock/carry, observation law, fixed or inferred `c`, spread profile, normalization, geometry/mask support, split IDs, numerical price controls, architecture/defaults, optimizer schedule and checkpoint rule. Keep historical bank identities immutable. Read them normally when compatible; regenerate or extend only under their matching producer rules. Store large banks, draws and models outside Git with SHA-256 manifests; commit compact outcomes and failures.

## 6. Acceptance gates and decision rules

Freeze numerical tolerances, statistical precision targets and case lists in the run record before looking at final comparisons. Inherit a stricter accepted baseline requirement where present. The following are the minimum checks, not substitutes for the complete task identity.

| Gate | Required evidence | Decision on failure |
| --- | --- | --- |
| Baseline dependency | Accepted noise/grid task and reference evidence; declared market-model limitations | Keep this follow-up at specification or smoke work |
| New geometry pricing | Refined Heston Fourier checks on the pilot and all reference geometries; price-error budget at the smallest supported `c*h_i`; derivative/surrogate checks if used | Improve pricing or narrow the declared support transparently before generating production labels |
| Interface | Correct targets and Jacobians; no target/context leakage; complete coordinate records; deterministic support rejection and same-environment checkpoint reload | Fix implementation before a fit |
| Geometry conditioning | Conditional comparisons against refined references across common and rare geometry strata; compare B1 with the anchor/bridge without changing the likelihood | Diagnose encoding/support or optimization; do not add compression to hide a B1 failure |
| Summary | Same-case B1/B2 posterior and product agreement, reference agreement, permutation invariance and measured cost | Retain direct inputs; compression is optional |
| Missingness | Mask/padding invariance; references using only observed quotes; full-set anchor and each declared family/severity checked | Limit claims/support to passing families; do not infer arbitrary-chain readiness |
| Population calibration | Fresh prior-predictive coverage/rank checks with Monte Carlo uncertainty, reported by parameter and geometry/mask stratum | Investigate conditional errors; no acceptance from a pooled plot alone |
| Scientific illustration | Held-out quote predictions, one affected forward-variance quantity and one control, with context and prior sensitivity visible | Report model/assumption limitations; do not widen noise merely to obtain a preferred posterior |
| Fresh acceptance | Frozen recipe, untouched population, reference precision and all retained failures | Accept, retain a clearly scoped limitation, or stop for reflection before another campaign |

Prefer the baseline's validated deterministic PyMC reference workflow where suitable; keep the target, transformation Jacobians and full chain structure explicit. References need adequate exploration (initial goals: rank-normalized R-hat <= 1.01 and bulk/tail ESS >= 400) and sufficient Monte Carlo precision for each comparison. These diagnostics are necessary, not proof of correctness. Recheck numerical likelihood accuracy at the reference posterior; an emulator cannot be its own sole oracle.

Use the implemented Monte Carlo-aware conditional mean and interval-width comparisons, not point estimates alone. Inherit the repository baseline margins: an absolute mean error no greater than 0.3 reference posterior SD and a 90% interval-width ratio in [0.75, 1.25], with uncertainty in both samplers accounted for. Borderline cases are inconclusive until their declared precision budget resolves them; do not move thresholds after observing results. Also inspect joint dependence and the predeclared products, since passing marginals does not establish the joint posterior. Calibration uncertainty and multiple parameter/stratum comparisons must be reported rather than converted into an arbitrary requirement that every finite-sample band hit its nominal value.

Known-scale SBC draws from the fixed-scale task. Inferred-scale SBC draws `c` from its actual prior. A paired comparison at fixed generating `c` is a conditional diagnostic, not an inferred-scale prior-predictive calibration claim. A posterior accumulating at a declared scale bound can indicate inadequate observation/model support; it does not by itself identify neural error or justify increasing that bound. Separate reference agreement from market residual adequacy.

For the information-removal story, preserve a primary mask withholding later expiries and evaluate their actual quote predictions. Inspect forward-variance uncertainty over a predeclared affected interval and a nearby observed control, using analytic Heston quantities where possible. Shared product horizons prevent a change of target across designs. Wider intervals need not occur in every realized dataset, and good model-conditional uncertainty is not automatically a model-risk reserve. Do not choose masks or products after seeing which produces the widest posterior.

## 7. Deliverables, parallel work and reflection

1. **Specification and smoke:** freeze the support/data schema, task identities, budget and reference list; implement small loader/adapter/invariance checks without changing historical core behavior.
2. **B1 geometry:** audit and price one shared bank, fit the direct conditional estimator and pass its gate. This can be prepared while the accepted baseline's notebook port is developed independently.
3. **B2 compression:** fit one summary on exactly B1's data, compare against B1 and references, then retain it only if useful.
4. **B3 missingness:** train declared masks and validate each supported family, full-input anchor and leakage/invariance checks.
5. **C and confirmation:** apply the selected recipe to inferred `c`, retain declared seed results and perform fresh acceptance only after selection freezes.
6. **Reflect before wider support:** publish a compact evidence record with costs, failures and claim limits. Decide whether broader spread profiles, off-catalogue geometry, real liquidity selection or a Bergomi flexible-input extension merits a new specification. No such extension starts automatically.

Notebook integration follows acceptance. Keep live-training and pretrained-loading routes distinct and usable; show the same quote metadata and supported-input checks in each. The baseline notebook keeps its complete fixed-design contract until a separately identified flexible checkpoint and helpers are accepted. Clearly label synthetic uncertainty profiles, controlled masks and market observations. Participants should see the scientific assumptions and consequences, not production debugging machinery. Publish assets only through the lab acceptance/manifest tools, with a new task identity and provenance; preserve the original notebooks/checkpoints as historical results when still supported.

The outcome may be that direct inputs remain preferable, only some mask families work, or the model fails to explain the market despite accurate inference. Those are reviewable results. Completion means a bounded, supported inference contract with evidence, not an arbitrary raw-chain API or a requirement to make every extension succeed.

# Heston follow-up: learned summaries, missing quotes and varying grids

**Status, 12 November 2026: all six sizes, including dimension 2, are complete.** The user requested a separate testing notebook extending fmuia's [SPX calibration notebook, PR #89](https://github.com/fmuia/sbi4finance/pull/89). This carries forward the plan from [PR #83](https://github.com/fmuia/sbi4finance/pull/83), replacing the previous geometry-first, known-scale-first sequence with **fixed-grid compression → mask-aware quote sets → varying observed geometry**. Keep joint estimation of five Heston parameters and `c` throughout; a separate fixed-`c` campaign is unnecessary. The current increment implements the summary experiment and quote-set design helpers. All six matched MLP fits and their final/minimum evaluations completed. Pooled recovery is similar for dimensions 8–64 and deteriorates sharply at 4; every full conditional-reference panel still fails. This is a one-seed screen and does not prove sufficiency. The user's sequence prioritized the smaller summaries to find where performance deteriorates. SetTransformer, masking and moving-grid training remain on hold until that comparison is assessed; their notebook controls remain design previews.

Use the actual63 task and reusable development bank from [PR #85](https://github.com/fmuia/sbi4finance/pull/85), following its [training-data guide](https://github.com/fmuia/sbi4finance/blob/feat/heston-spread-baseline-20261109/lab/plans/training-data-guide.md). Preserve historical producers, data identities, checkpoints and baseline notebook behavior. The former requirement to wait for accepted fixed-layout calibration is not a prerequisite for exploratory comparisons. Scientific acceptance and participant-asset promotion still require their own evidence.

## 1. Questions and tutorial value

| Investigation | Scientific question | Participant activity | Priority |
| --- | --- | --- | --- |
| Learned summary | How much compression preserves useful posterior information? | Compare summary sizes on the same data; identify which targets lose information first | Core |
| Missing quotes | Can one estimator recover the correct posterior for supported subsets, and which observations constrain which parameter combinations? | Hide an expiry or wing and compare uncertainty | Strongest interactive core exercise |
| Varying grid | Can an estimator condition on known strikes/maturities that differ from its anchor layout? | Inspect nearby layouts, supported generalization and its limits | Advanced extension |

All three requests make sense. A small summary is not automatically sufficient. Robustness to missing data means appropriate uncertainty rather than unchanged confidence. Moving contract locations can genuinely change information. Masking changes availability on a layout; varying geometry changes the layout itself.

Heston is useful here because deterministic pricing and an explicit likelihood permit reference comparisons. The baseline remains imperfect: its 300k primary passes five of six marginal SBC screens, initial-variance nominal-90% coverage is about 97.5%, and synthetic physical intervals remain approximately 1.05–2.86 times MCMC widths. Matching it is a relative development result, not a correctness certificate. The market reference also puts `c` near its upper bound; separate that model/noise limitation from neural approximation.

## 2. Shared scientific and data contract

Keep the observation law in price units:

`Y_i = P_i(phi; K_i, T_i, type_i, F_i, D_i) + c * h_i * Z_i`, independently `Z_i ~ Normal(0, 1)`.

The targets are `(initial_variance, kappa, theta, vol_of_vol, rho, c)`, with independent physical-coordinate Uniform bounds `[(0.01,0.16), (0.1,10), (0.01,0.16), (0.1,3), (-0.95,-0.1), (0.5,2)]`. Preserve the noise convention and parameter transformations within paired comparisons. Do not simultaneously change priors, posterior families or discrepancy models.

Freeze the current 63 literal European SPXW contracts, carry, half-spreads and pricing controls as the anchor. Preserve contract IDs, exercise/settlement conventions, expiration times, observation clock, prices, spreads and carry as raw metadata. Interpolated prices are not additional independent observed quotes.

The 300k bank includes raw noisy prices, clean prices, `Y/h` and all six targets. Summary and fixed-grid masking training can reuse it without another large pricing bank. The collaborator loader's `training_inputs` returns **densely whitened** observations: suitable for complete-vector compression, unsuitable for applying masks or moving coordinates. Later experiments must load verified raw arrays. Clean prices and generating targets are simulator/audit inputs, never inference conditions.

The existing 100 validation rows suffice for smoke checks but are small for model selection. Before screening candidates, specify expanded independent validation and fresh confirmation populations; 2,000 cases each is a practical proposal, subject to a small pricing benchmark. Historical 512-case diagnostics and repeatedly inspected references are development evidence. Split underlying parameter draws before geometry/noise/mask augmentation; another augmentation of a training parameter row is not an independent test case. Keep full noise realizations paired across subset comparisons.

## 3. S: learned summaries and an empirical compression limit

**S1 isolates compression.** Insert a small jointly trained MLP summary between the existing frozen 63-feature whitening and unchanged FlowMatching posterior network. Train with the inference objective, not only regression on parameter means. This tests a bottleneck without changing preprocessing or geometry.

Use the complete dimension screen **32, 16, 8, 4 and 64**, in that execution order. Four is deliberately demanding; 64 belongs to the same comparison as the higher-dimensional control, rather than to the later masking agenda. The 32-dimensional run preserves pooled recovery reasonably well relative to the direct baseline, which motivates prioritizing smaller bottlenecks despite unresolved conditional errors. Fix splits, noisy observations, summary hidden layers (two 128-wide SiLU layers), linear summary output, posterior family/head, optimizer schedule, batch size, epochs and effective defaults. A 64-coordinate learned representation is a useful capacity control but does not guarantee an invertible map or the direct model's optimization behavior. Record parameter count and cost. Compare with the current direct checkpoint and at most one fresh matched direct control wherever the split or optimization changes. Do not add intermediate sizes such as 6 or 12 until this initial screen identifies a useful bracket.

Screen one seed; initially confirm only the smallest useful candidate and its adjacent larger-dimensional control with one additional seed each. This caps S1 at eight full fresh fits including the optional matched direct fit. A common seed label does not imply identical initial weights across different architectures; the first screen measures trained-model performance, and independent-seed confirmation is needed before claiming a reproducible breakdown point. Save both declared final and validation-selected checkpoints; do not choose checkpoints from posterior diagnostics. One early 64-dimensional launch was stopped while clarifying priorities; retain its partial files, exclude it from endpoint comparisons, and run the full matched 64-dimensional control after 16/8/4.

Define “minimal viable” as **the smallest tested dimension meeting declared posterior-fidelity criteria for these priors, noise, layouts and cases**. It is not a proof of minimal sufficiency. Six parameters do not imply six sufficient summary coordinates. In particular, `c` needs residual-variation information beyond mean smile shape. PCA or an autoencoder is an optional comparator: reconstruction variance is not inferential information. Learned coordinates need not have individual financial interpretations.

**S2 establishes the quote-set interface and tests the encoder choice.** The primary candidate is a small geometry-aware SetTransformer; DeepSet is the cheaper matched control. Attention can compare contracts across strikes and expiries, potentially helping distinguish smile/term-structure changes from residual variation relevant to `c`. That is a motivation, not evidence of superiority. With 63 tokens, measure actual attention cost before treating it as prohibitive. Compare these two families at dimension 32 using identical token preprocessing, posterior heads and evaluation cases; do not cross every S1 bottleneck dimension with both families. Choose from posterior fidelity and cost, including mask/padding behavior. A smaller successful MLP bottleneck does not automatically transfer to another encoder. The current S1 MLP is a deliberate compression control on the ordered whitened vector, not the selected architecture for flexible inputs.

Each complete quote token carries `(Y/(D*F), log(K/F), T, put_or_call, h/(D*F))`, retaining externally observed carry/context as needed. Set invariance applies to the order of complete records, not to prices exchanged across locations. Use shared tokenwise normalization fitted on training data. Supply valid quote count explicitly, especially with mean pooling. Never supply generating `c`, `c*h`, clean prices or the noise realization.

The saved fixed-layout ZCA is not a supported quote-set preprocessing scheme. Fit one fixed-grid direct-vector bridge using the same tokenwise features/scales and compare the set encoder on identical data. This separates preprocessing from compression. S1 may keep ZCA because it neither masks nor moves quotes.

**Notebook:** select a pretrained summary size, view calibration and contraction for each target, and overlay posteriors for the same synthetic observation. Ask whether noise-scale recovery fails before physical-parameter recovery. A failed compression level is an informative result.

## 4. M: masking and practical identifiability

Use the geometry-aware tokens and selected summary family. Resample declared masks on raw bank observations during training. A proposed training mixture is 25% full grids, 50% random subsets retaining 16–62 quotes, and 25% structured subsets. Freeze exact proportions and supported families before fitting.

| Paired evaluation | Question |
| --- | --- |
| All 63 quotes | Does mask training retain complete-data fidelity? |
| Random subsets of 48, 32 and 16 | What is the effect of quote count? |
| Remove short- or long-dated expiry blocks | What does maturity coverage contribute? |
| Remove a put wing or retain central moneyness | What does smile coverage contribute? |
| Different locations with comparable retained counts | Does location matter beyond cardinality? |

Include representative structured families in training; evaluate unseen instances and label unsupported patterns as stress tests. Sample masks independently of targets and errors conditional on observed geometry. Price- or liquidity-dependent selection is another observation model: random deletion does not establish that robustness.

Exclude quotes **before any mixing of records**. Masking after ZCA leaks withheld values into retained channels. Zero padding alone does not encode absence. Pooling, attention, normalization and global context must ignore invalid payloads. Test invariance to changes in masked values, added padding and whole-record permutations. Carry must remain external observed context, not be reconstructed from hidden test prices. Reject an empty set unless a separately tested prior-return path is provided.

The reference likelihood uses only retained quotes, including normalization: joint `c` requires `-n_observed * log(c)`. `HestonSpreadPilot.log_likelihood(..., mask=...)` already implements the normalized subset likelihood; its simulator can preserve surviving full-grid errors. Verify installed BayesFlow 2.1 summary-mask routing in a smoke test rather than assuming the custom pipeline is correct because an API supports masks.

Assess all six targets, joint correlations, contraction, withheld-price predictions and one fixed Heston expected/forward-variance quantity. Use matched subset MCMC references. Wider reference uncertainty reflects information loss; neural/reference disagreement reflects approximation error. The claim is practical identifiability under this prior, design and noise, not structural identifiability. Individual realized intervals need not widen monotonically with every deletion.

Short expiries helping initial variance, maturity coverage helping mean reversion, and wings helping smile parameters are hypotheses to check. Inspect `c` and its physical-parameter dependence too. If the mask-trained summary fails, a mask-aware direct-vector control can localize the error; it is not a mandatory architecture sweep.

**Notebook:** click quotes, an expiry or a wing on the displayed grid; visibly mark removed points and compare full/subset posteriors and withheld-price predictions. Include equal-count subset presets and saved matched reference overlays. This is the strongest tutorial exercise: uncertainty should change for the right reason.

## 5. G: a fuzzy grid as varying observed geometry

“Fuzzy” here means varying **known** contract coordinates. The estimator conditions on supplied `(K,T,type,F,D,h)`. Latent or measurement-uncertain coordinates would require a different observation model and are deferred.

Initially retain 63 quotes in seven expiry blocks. Vary geometry separately from availability; combine them after each works. Distinguish:

1. **Controlled local jitter first.** Perturb log-moneyness within a declared fraction of adjacent-slot spacing; perturb maturity once per expiry block. Proposed initial bounds are strike jitter up to 10% of local spacing and maturity jitter up to 5% of maturity, capped by neighboring expiry separation and supported carry interpolation. Inspect the anchor before freezing exact bounds. Preserve positive maturities, ordering, type conventions and distinct contracts. Seven shared expiry blocks avoid unnecessary pricing of 63 separate maturities.
2. **Actual-contract catalogue next.** Sample distinct eligible European contracts from the snapshot within declared maturity/wing coverage. This demonstrates irregular combinations within the catalogue, not untested continuous/out-of-range support. Label continuously jittered coordinates as synthetic rather than actual listed contracts.

Draw geometry independently of targets/errors. Recompute forwards/discounts at moved maturities under a declared positive carry-curve interpolation and range; preserve clocks and exercise/settlement conventions. Existing parameter draws may be paired across designs, but **moved contracts must be repriced**. Relabelling old prices is invalid. A later interpolation accelerator would need separate numerical validation.

For a geometry-only comparison, attach the anchor's normalized spread profile `h/(D*F)` to canonical slots as an explicit synthetic uncertainty scenario. Preserve measured anchor spreads separately. Actual-chain generalization needs appropriate observed spread profiles and separate checks: `c` changes their overall level, not relative shape. Do not vary geometry and observed spreads together and attribute every difference to geometry.

Run a pricing pilot of at most 512 cases spanning prior/geometry extremes, with current accuracy controls at minimum supported `c*h`. Measure grouping/caching cost before choosing a bank. Start with one 20k-row development bank plus independent validation/evaluation; expand toward 100k/300k only from measured need and cost. Share it between a geometry-conditioned ordered direct-input control and the selected set encoder.

Evaluate the anchor, unseen in-support layouts, support boundaries and explicitly labelled beyond-support stress layouts. Hold out layout patterns as well as parameter draws. Stratify calibration by geometry and compare with MCMC on the exact moved contracts. Stable posteriors across layouts are not automatically better because information can change. Finally train one combined geometry-plus-mask recipe under a new support identity.

**Notebook:** select a supported nearby layout or shift an expiry/strike band, inspect actual coordinates/prices, and compare inference. Use a few precomputed examples/references for fast feedback; optionally allow live Heston repricing in advanced mode. Display supported regions and identify extrapolation. This is valuable for advanced participants after compression and missingness.

## 6. Common evaluation and decision rules

Freeze splits, cases, checkpoints, budgets and criteria before selection. Retain the baseline conditional yardstick: absolute mean error at most 0.3 reference posterior SD and central-90% width ratio in `[0.75,1.25]`, accounting for Monte Carlo uncertainty. Inspect joint dependence, predictions and variance quantities. Report absolute fidelity separately from change relative to the matched control. Do not relax away the baseline's errors; if no tested size meets the criteria, no viable dimension was demonstrated.

Coverage/SBC alone is insufficient: returning the prior can look calibrated while discarding observations. Check recovery, contraction and conditional-reference agreement together, for every target and mask/geometry stratum. Report finite-sample uncertainty. A fixed generating-`c` panel is not joint-prior SBC.

Reuse the three existing full-grid references for S1/S2. Add a bounded list of masked/jittered references as stages open, initially two synthetic cases and two contrasting subsets/layouts instead of a Cartesian product. Prefer deterministic PyMC references where suitable, preserving sampler provenance and full chains. Initial convergence goals are rank-normalized R-hat <= 1.01 and bulk/tail ESS >= 400, with posterior pricing-precision checks and adequate comparison precision. An unconverged reference is inconclusive.

## 7. Execution sequence and deliverables

1. Define a separate `lab/notebooks/heston_flexible_inputs.py` and configurable experiment runner aligned with PR #89. Reuse canonical pricing/inference helpers and keep the baseline notebook usable.
2. Run S1's sequential bottlenecks on the verified bank, with a matched control where needed; confirm only the useful bracket.
3. Implement tokenwise preprocessing, its direct bridge and mask/padding/permutation checks. Fit the full-set encoder and one mask-trained model; evaluate full/structured cases against references.
4. Pilot and price one varied-geometry bank; compare direct/set conditioning and finally combine successful geometry/masking.
5. Freeze selected recipes, confirm on fresh cases/additional seeds, and package a few pretrained checkpoints/reference panels. Core summary/masking exercises can be ready while geometry remains advanced development work.

The previous direct 300k/300-epoch fit took about 30 minutes on local JAX CPU, plus roughly five minutes for its two endpoint evaluations. This supports feasibility but is not a set-encoder runtime estimate. Benchmark compilation, ten warm epochs, memory and sampling before scheduling a family. Start with a 60-minute compute wall-time ceiling per fit, including compilation, and bind a matched update/schedule budget after the benchmark. If the intended endpoint cannot finish, report it as incomplete rather than treating a short fit as evidence against a representation. One-GPU Oscar jobs can use the separate GPU allowance while Bergomi CPUs continue; keep a local fallback and do not assume immediate starts or GPU speedups. References can run independently on suitable CPUs. Avoid a sweep across every summary, mask, geometry, posterior family and seed.

Save source/environment/task identities, split/augmentation IDs, preprocessing, supported geometry/masks, model/optimizer/RNG states, update counts and checkpoints. Preserve completed outputs if a later stage fails; changed protocols get new identities. Keep large data/models/draws outside Git with checksums and compact records in the existing evidence hierarchy. Reusable execution helpers belong in the canonical tutorial runtime and may be imported by lab workflows; the tutorial never imports the lab. Promote participant assets only through acceptance/manifest tools after adequate evidence.

Completion means bounded, reviewable results. Direct inputs may remain preferable; some mask families may fail; a new layout may expose a model limitation. Report those outcomes rather than require a successful summary or arbitrary-chain API.

## References

- [SPX calibration notebook, PR #89](https://github.com/fmuia/sbi4finance/pull/89), current fixed-grid six-target application.
- [BayesFlow paper](https://arxiv.org/abs/2003.06281), jointly learned summaries and amortized inference.
- [Deep Sets](https://arxiv.org/abs/1703.06114), permutation-invariant functions of complete records.
- [BayesFlow 2.1 BasicEstimator](https://bayesflow.org/v2.1.0/api/bayesflow.estimators.BasicEstimator.html) and [MaskHandler](https://bayesflow.org/v2.1.0/api/bayesflow.masking.MaskHandler.html), structural summary masks; validate against the locked installed version.

## Dimension-two extension requested after the initial screen

The user requested a matched dimension-2 fit to illustrate severe compression.
The fit completed using exactly the initial screen's bank, priors, noise, hidden layers, posterior
head, seed, schedule, epochs and endpoint rules. Keep its actual result regardless
of whether performance is worse than dimension 4. This extends the dimensionality
screen, with masking still deferred. Call any deterioration an empirical
breakdown in this recipe; neither a sharp phase transition nor failure of every
size below 8 follows from these tests (dimension 6 and seed replication are absent).

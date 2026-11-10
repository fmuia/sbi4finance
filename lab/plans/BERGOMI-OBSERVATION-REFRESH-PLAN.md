# Rough Bergomi observation refresh: follow the accepted Heston specification

**Status, 10 November 2026: readiness planning complete; Bergomi production remains gated.** The user authorized parallel planning and preparation. This track ports the accepted **fixed-layout Heston** observation and grid specification to the rough-options lesson **after the Heston noise/grid acceptance gate**. No refreshed Heston specification is treated as accepted by this document. No rough pricing, bank generation, reference sampling, training, asset replacement or production allocation is authorized by this planning PR.

**Heston dependency: [implementation PR #85](https://github.com/fmuia/sbi4finance/pull/85).** The fixed c=1 single-Gaussian `c*h` likelihood is implemented and checked. The initial neural recipe, matched zero-dropout control and exact-state continuation to 50k updates improve development results but still fail conditional-reference checks. Three references are ready; the boundary case reached its time cap and remains inconclusive. The bounded lower-learning-rate comparison from 50k to 55k/60k completed in about 100 seconds: loss improved and the 60k population rank checks passed 5/5, but both endpoint conditional panels still fail. No fixed Heston baseline is accepted; this development round stops for reflection without another fit launch.

The parallel raw-data audit supports 62/63 literal E quote targets and finds no one-year PM expiry. No new-grid labels or fit have been produced, so the Heston grid-production gate remains closed. Flexible-input B1/B2/B3 training has not started and is independently planned. This Bergomi track remains at readiness work until the fixed-layout Heston baseline is accepted; passing Heston software checks does not authorize refreshed rough labels, training or publication. Detailed implementation results remain in PR #85.

The first rough baseline has one independent Gaussian error component, with known spread-based scale. Heston's flexible geometry, learned summaries and masking experiments are independent; they are not prerequisites and do not need to be ported at the same time. Gaussian-process discrepancy, the additive vega-scaled tau model and the multi-day interpolated reader are outside this execution unit.

## 1. Scientific deliverable and dependencies

The deliverable is a reproducible European-option inference task whose data, clean-price simulator, Gaussian likelihood, deterministic calibration, numerical reference and amortized estimator agree on contracts, units, priors and conditions. The lesson must explain what the option information identifies and how remaining uncertainty reaches model fair values. Wide posterior intervals are not an acceptance requirement.

The Heston track must first freeze and identify:

1. One ordered set of actual European SPXW contracts, with their realized coverage and unsupported-target decisions; quote provenance, carry anchors, discount/forward estimates and exact valuation/settlement times.
2. The price half-spread vector and the known-scale Gaussian convention, including the supported scale range and the same-observation sensitivity protocol.
3. A working training/input recipe and evidence of synthetic calibration and conditional agreement with matched Heston references. Model adequacy remains a separate judgment.
4. The selected physical grid and an explicit reflection/acceptance decision. A decreasing loss, passing repository tests or approval of a plan is insufficient.

Port the **accepted scientific contract**, not Heston parameters, posterior draws, Fourier error thresholds or an unaccepted candidate network. If Heston changes the selected contracts or measurement law before acceptance, update this plan's task declaration before rough labels are commissioned. Do not implement competing rough grids while that choice is unresolved.

The candidate coverage discussion includes an approximately 7-by-9 actual-quote layout with short through roughly one-year expiries and fixed-16% log-moneyness targets. It is not a frozen 63-contract rough task. The old 63-contract symmetric grid, the Heston pilot's actual 54 contracts and a revised candidate are distinct physical designs. A coordinate relabelling alone changes no contracts; selecting different strikes or maturities does.

## 2. Observation and input contract

For one frozen design, use

\[
Y_i=P_i(\psi)+c h_i Z_i,\qquad Z_i\stackrel{\mathrm{iid}}\sim N(0,1),\qquad c>0.
\]

Here \(\psi\) contains the rough dynamics and, for joint inference, the curve coefficients. \(h_i\) is the observed **price half-spread**, held fixed within the task; \(c\) is a known dimensionless multiplier. No vega-scaled tau component is present. Half-spread equals one Gaussian SD at \(c=1\) by a declared working convention, not by an empirical theorem about quotes.

- Begin with known \(c=1\), with sensitivity at \(c=.5,1,2\) on the **same observed price vector**. If one estimator supports varying known scale, train over the accepted Heston scale distribution and supply \(c\), or an equivalent lossless representation of the actual SD vector, as an explicit condition. A checkpoint trained at one scale does not automatically support another.
- Keep the frozen spread profile and full ordered design explicit in task/checkpoint metadata. Do not replace a changed spread vector by a median scalar ratio without a separately verified proportionality assumption. A different snapshot is not silently supported by this fixed-design model.
- Use actual quote locations. No smoothing, strike/maturity interpolation or American/AM-settled substitutions enter this baseline. Preserve source timestamps, roots and settlement conventions, and distinguish last-trade time from an unavailable bid/ask update time.
- For the existing forward-unit pricing API, set \(p_i=y_i^{price}/(D_iF_i)\), \(h_i=h_i^{price}/(D_iF_i)\) and strike \(K_i/F_i\). Apply the same conversion to predictions and numerical covariance. Preserve raw quotes and carry estimates alongside these derived inputs. Carry remains conditional unless a separate uncertainty study is declared.
- Synthetic observations receive fresh independent quote errors; clean prices and their numerical uncertainty are stored separately. Reuse a clean physical row across known-scale sensitivity conditions; do not count noise augmentation as additional independent physical parameter locations. Keep all augmentations of one physical row in its original split.
- Keep Gaussian observations as generated, including finite negative values. No IV inversion, positivity truncation, rejection or redraw is permitted merely because a simulated quote is negative.

An optional later **inferred-c** illustration uses the same single variance profile \(c h_i\), an explicit independent prior such as the accepted Heston illustrative \(U(.5,2)\), and a joint posterior over \((\psi,c)\). It hides the generating scale and every feature containing it from inference conditions; the known profile \(h\) remains available. It needs its own target, checkpoint, references, normalization and conditional-scale checks. It is not required for the initial known-c rough baseline, and boundary mass is not evidence of pure measurement noise. Do not substitute an equal average of fixed-c posteriors for marginalizing the joint posterior.

## 3. Curve inference and information experiments

Retain the distinction between the two existing teaching problems:

| Route | Unknowns | Curve information | Interpretation |
| --- | --- | --- | --- |
| Supplied-curve synthetic introduction | H, eta, rho | Explicit known positive curve, varied across training cases and fixed within each inference problem | Conditional teaching task; not evidence that market quotes identify that curve |
| Joint-curve calibration | H, eta, rho and the declared curve coefficients | Generating curve excluded from inference conditions | Main route for studying curve information in market/full-versus-restricted observations |

The current joint family is \(\xi_0(t)=v_{long}+(v_{short}-v_{long})e^{-a t}\), with six ordered targets `hurst, eta, rho, short_variance, long_variance, decay`. Its current bounds are H `[.02,.35]`, eta `[.5,3.5]`, rho `[-.99,-.2]`, variance levels `[.01,.16]` and decay `[.25,8]` per year. These are starting controls to review in the numerical panel, not parameters inherited from Heston or a claim of final market adequacy. Retain difficult low-H/high-eta and high-variance cases; numerical difficulty alone does not justify narrowing the prior. The supplied seven-interval curve law and the analytic joint family remain different tasks.

For a later information-removal experiment, infer the curve jointly from the surviving smile quotes or use truly external, documented curve information. Never supply a curve reconstructed from withheld later-expiry or wing quotes. Common carry anchors can remain a condition, but then the experiment removes smile information conditional on that carry; it does not remove all information at those maturities. ATM information used for design selection must also remain visible as conditioning/selection information.

Start product propagation with analytic expected average variance and a forward average over declared intervals, keeping complete joint posterior rows paired. With an exactly supplied curve these rates have zero H/eta/rho-induced uncertainty by construction; this is a control, not identification evidence. A nontrivial H/eta/rho-sensitive product needs its own bounded payoff-pricing check. Fair-value posterior quantiles are conditional on model/curve/tail assumptions and are not automatically model-risk or regulatory reserves.

## 4. Exact maturities and the rough numerical target

The accepted actual-quote design uses elapsed ACT/365F time to its explicit settlement timestamp. Current rough pricing uses uniform grids from the Q origin and rejects contracts not on grid nodes. Replacing a maturity by a nearest business-day/252 node changes the contract and the variance clock. It cannot be called a coordinate conversion or used to reuse old labels.

Before pricing the candidate, choose and freeze an exact-time strategy. For a bounded first implementation, a block ending at each distinct actual expiry is a possible route: choose an audited uniform step size per horizon and take prices exactly at its endpoint. Review curve-knot handling as well: the current supplied-curve validator requires knots to align on every relevant block, so adding one block per expiry alone does not solve arbitrary supplied knots. An analytic joint curve can be evaluated on each actual simulation grid, but its representation/interpolation and integrated-variance controls still require refinement checks. A new irregular-grid hybrid scheme is a separate numerical project, not a prerequisite chosen casually.

Record the maturity, horizon, step count, curve evaluation rule, hybrid normalizer, antithetic convention, control variates, independent block structure and RNG stream protocol. Starting each maturity block at time zero retains rough history; restarting from current variance or a current driver value is not a valid rough continuation state. Do not change the volatility model's time unit while keeping the old prior/labels under the same identity.

The target hierarchy must be explicit:

1. Continuous-model expected European prices, the scientific ideal.
2. A finite time-grid expectation, with measured discretization effects.
3. A Monte Carlo estimate of that expectation, with price-mean covariance and finite-sample diagnostics.
4. A deterministic learned mean-pricer approximation, if used for references.
5. The amortized posterior approximation conditional on the declared observation law.

Agreement at a lower layer does not certify the earlier layers. Heston's deterministic Fourier audit does not certify any rough layer.

## 5. Measurement-only numerical precision gate

Define the strictest supported measurement SD \(s_{min,i}=.5h_i\), if `.5` is retained as the minimum known multiplier. Before any production bank, predeclare numerical tolerances against this vector, independently of inferred error scales or market residual size.

The initial feasibility target is reported Monte Carlo SE at most \(.25s_{min,i}\) per contract, with **no inherited final-rung .5 exception** for the new task. This is an empirical MC target, not a guarantee about continuous-price bias. Retain unsuccessful rows, IDs and costs; reaching the declared path cap is a failure/stop, not permission to widen noise, discard a row or increase the cap automatically. A different threshold requires an explicit revised numerical specification and sensitivity evidence.

| Error source | Required pilot evidence | What does not suffice |
| --- | --- | --- |
| Monte Carlo sampling | Independent batches; antithetic pairs as statistical units; full price-mean covariance; nominal residual-based SE versus between-batch dispersion | One favorable seed or the within-batch fitted-control residual alone |
| Control variates | Controlled/raw comparison and replicated mean shifts/dispersion, retaining difficult cases | Bounded raw put payoffs do not make every fitted control-variate diagnostic an exact uncertainty certificate |
| Time discretization/curve representation | Paired coarse/fine/finer comparisons at exact contract times, with uncertainty in price differences and curve-integral controls | A doubled grid is not exact truth; a small MCSE does not bound grid bias |
| Surrogate approximation | Independent refined price labels, errors and likelihood effects in strict measurement units, including representative posterior regions | Training/validation MSE or task/model hash matching alone |
| Reference posterior | Matched numerical likelihood, seed/grid/pricer identity, sampler convergence and sensitivity to improved prices | Convergence around an inaccurate fixed pricer |

Predeclare a provisional price-shift screen of `.1 * s_min` for resolved grid shifts and independent surrogate residuals, reporting Monte Carlo confidence intervals and all failing contracts. If uncertainty cannot resolve that scale, report the audit as unresolved. These screens guide a bounded development decision; full-vector likelihood effects and posterior sensitivity remain necessary before scientific acceptance. Do not add pricing variance to observation noise or subtract a noisy SE estimate from its variance without a separately justified target.

Numerical covariance matters even with independent observation error. With \(S_{min}=\operatorname{diag}(s_{min}^2)\), inspect the spectrum and selected contrasts of \(S_{min}^{-1/2}C_{MC}S_{min}^{-1/2}\), paired refinement covariance and accumulated likelihood shifts. Marginal SE bounds alone do not bound every combined direction. A covariance estimated from only a few independent batches can be rank deficient; retain it descriptively rather than inverting it as an exact likelihood correction. A future correlated observation law would need its own whitening/gates; it is not smuggled into this baseline.

For deterministic prediction error \(\delta=\widehat P-P\) and residual \(r=y-P\), record the actual likelihood change \(r^T S^{-1}\delta-\tfrac12\delta^T S^{-1}\delta\) under the same frozen Gaussian covariance. For estimated price shifts, propagate uncertainty instead of treating the estimate as exact. Small individual residuals can accumulate across contracts.

## 6. Bounded implementation sequence

| Phase | Work | Exit evidence |
| --- | --- | --- |
| R0: readiness, now | This source/data audit, task/identity design, exact-time options and tests to implement | Documented gaps and dependency on an accepted Heston specification; zero new numerical jobs |
| R1: representative numerical panel, after Heston gate | Freeze exact rough task; implement isolated fixed-layout/known-c support; audit prices at actual times | Reproducible panel, independent-batch/grid/covariance report and measured cost; no unresolved numerical gate hidden by noise |
| R2: label reuse and cost decision | Compare required contracts, priors and numerical target with preserved clean banks; estimate new/refined rows | Exact reuse ledger, projected CPU/memory/wall time and explicit bounded next allocation before production |
| R3: bounded inference development | Train one posterior and, if needed, one task-bound mean-pricer; establish predeclared conditional references | Separate simulator calibration, conditional posterior agreement, numerical-target sensitivity and market residual checks |
| R4: notebook integration and reflection | Update the existing rough notebook/helpers, cached/live routes, evidence and provenance | Checked live and pretrained paths; accurate teaching claims; explicit accept/fail and reflection before broader work |

For R1, predeclare **12 joint-curve physical rows and four supplied-curve controls** before viewing candidate results: typical, low-H/high-eta, high variance, rising/falling and nearly flat curves, plus boundary-near cases. Declare the exact rows, grid ladder, independent streams and stopping conditions in the run specification. Start with at most four representative rows on up to three grids and four independent batches; extend to the remaining declared rows only within the same cap. Independent batches repeat physical rows and are not new prior draws.

The proposed initial R1 ceiling is **8 allocated CPU-hours, two CPU workers, no GPU**, including setup and failed pricing attempts. It is a future feasibility ceiling, not an allocation launched by this plan; if insufficient, return the incomplete report and a measured cost proposal. Do not automatically escalate path counts beyond the declared ladder, allocate a full bank, fit networks or run a direct-MC MCMC campaign. Reference/production/training budgets and data sizes are decided only after R1, with the Heston gate already satisfied. Historical rough cost ratios, older CPU ceilings and cached Heston timings do not authorize new work.

Development splits, synthetic reference cases and numerical challenge cases must have separate identities from reserved acceptance observations. Freeze the recipe before final calibration/seed confirmation. Keep failed conditional comparisons and unsupported cases visible rather than replacing them with attractive examples.

## 7. References and scientific acceptance

Prefer the repository's PyMC workflow where it fits a **deterministic** validated likelihood. A reference based on a learned mean-pricer targets that specific emulator likelihood, not an exact rough-model posterior. Reusing a seed makes a finite-path pricer deterministic; it does not integrate pricing randomness out or make its posterior exact. Compare independent seeds and refined grids at representative points and posterior regions. Do not call a likelihood with fresh internal Monte Carlo NUTS-compatible or exact pseudo-marginal merely because its prices are unbiased; pseudo-marginal inference requires an appropriate nonnegative unbiased likelihood estimator and preserved stochastic state.

Freeze observation-vector hashes, task/prior/design/spread hashes, curve roles, price units, numerical controls, source/environment versions, reference-pricer/model hashes and sampler state. Verify that market references target the actual observed mids and synthetic references target their declared observations. Preserve chain/draw structure. Apply repository convergence requirements (rank-normalized R-hat at most 1.01 and bulk/tail ESS at least 400) and Monte Carlo-aware conditional comparisons, including means, interval widths, dependence and selected curve products. If importance sampling is used, record proposal law, normalized weights, ESS, tail diagnostics and seed/grid sensitivity; it is an approximate numerical reference, not a substitute for those checks.

| Question | Evidence |
| --- | --- |
| Is inference calibrated under its own simulator? | Fresh synthetic ranks and interval coverage, with declared finite-draw/randomized-rank handling and population scope |
| Does the neural estimator match the specified conditional posterior? | Matched synthetic/market references, joint dependence and product checks with numerical and sampling uncertainty |
| Is the numerical posterior stable enough? | Independent-label, seed, time-grid and surrogate refinement on representative conditional cases |
| Does the model describe market quotes? | Residual structure in half-spread units, raw held-out quotes and prior/curve/noise sensitivity |
| Does removing information matter? | Paired surviving observations under the same priors/carry and no withheld-curve leakage |

SBC cannot validate the market noise convention. Market misfit cannot explain a neural failure on its own simulator. Concentrated physical posteriors at small spread-based noise may make inference harder; simplifying the error model is not a guarantee of easier training. If market residuals exceed plausible quote uncertainty, report that failure instead of silently inflating c or reopening an inferred tau model.

## 8. Read-only readiness audit completed for this plan

Audited checkout: main commit `1dbfa8342d17027f8e4cb83339db93a8d6d2a9b9`, in the dedicated planning branch. The findings below are inspection results, not new pricing accuracy claims.

| Source inspected | Current behavior / reusable component | Required change or limitation for this refresh |
| --- | --- | --- |
| [Shared task validation](../../tutorial/src/sbi4f_tutorial/shared/tasks.py), [lab task builders](../src/sbi4finance/calibration/tasks.py) | Explicit priors/contracts/noise; known versus joint curve roles; rough blocks must hit contract/curve-knot nodes | New versioned actual-layout/spread/known-c task contract. Old builders generate the symmetric fixed-20% grid and fixed synthetic noise; arbitrary ACT/365F times fail alignment checks |
| [Complete-surface adapter](../../tutorial/src/sbi4f_tutorial/shared/full_surface.py) | Reusable bounds/training/sampling/checkpoint structure; joint curve auxiliaries stay hidden | Currently requires the canonical ordered 63 contracts, 63-price arrays and existing target names. Its set encoders also require that full design; they do not provide arbitrary quotes/masks. Add isolated versioned fixed-layout support and known-c conditioning; preserve old loaders |
| [Rough simulator](../../tutorial/src/sbi4f_tutorial/rough_options/simulator.py), [conditional puts and curve helpers](../../tutorial/src/sbi4f_tutorial/rough_options/model.py) | Hybrid driver from the Q origin, covariance-grid test reference, bounded conditional raw puts; existing innovations/curve checks | Uniform-grid API and asset discretization remain. The small covariance reference is capped at 128 steps and is not a full production-price certificate. Preserve mathematical/RNG conventions when adapting times |
| [Joint curve construction](../src/sbi4finance/calibration/tasks.py), [analytic curve values/integrals](../src/sbi4finance/calibration/pricing.py) | Current lab joint builder samples the exponential curve at the union of simulator grid nodes; controls use discrete left sums | Preserve exact node evaluation after changing blocks. Continuous analytic curve integrals and discrete simulator controls are different quantities. PR #74 moves/reuses these helpers in the participant runtime; reconcile that change rather than introduce a second implementation |
| [Shared surface pricing](../../tutorial/src/sbi4f_tutorial/shared/pricing.py) | Conditional European puts, parity calls, antithetic pairs, fitted controls, block-diagonal price-mean covariance, explicit path streams | `rough_surface` gates SE against the supplied old sigma vector and can accept a larger final-rung cap. New numerical gating must use `.5*h` and explicitly reject inherited exceptions; independently audit control diagnostics |
| [Coupled refinement](../src/sbi4finance/bergomi_refinement.py), [price panel validator](../scripts/validate_rough_price_panel.py) | Paired grid differences, raw/controlled comparisons and covariance reporting | Existing panels assume aligned nodes and specific 63-column identities. Reuse methods after exact-task adaptation; rough coupling is common random numbers, not an exact common-Wiener bridge or removal of bias |
| [Mean pricer](../src/sbi4finance/calibration/pricer_net.py), [reference likelihood](../src/sbi4finance/calibration/reference.py), [readiness checker](../scripts/check_rough_emulator_readiness.py) | Task/model binding, deterministic Gaussian normalization, explicit curve features, independent gradient checks | Outputs are scaled by the old quote sigma. New task binding/output scaling and independent strict-noise validation are required; identity checks alone do not establish accuracy |
| [Published joint task](../../tutorial/assets/backbones/rough-joint100k-validation0/task.json), [development configuration](../configs/calibration_rough_joint_development_20k.json), [training-data guide](training-data-guide.md) | Historical complete-grid banks and six-target analytic-curve task remain reproducible controls | Published task uses 63 fixed-20% contracts, blocks 1280/2016/2016, SE target .25 and final cap .5 relative to old noise. Its known and joint banks are different curve laws. No clean-label payload was opened, repriced or certified reusable in this audit |
| [Rough notebook](../../tutorial/notebooks/slot5_calibration.py), [lesson helpers](../../tutorial/src/sbi4f_tutorial/rough_options/lesson.py), [curriculum](../../tutorial/curriculum/05-slot5-rough-bergomi-to-a-reserve.md) | Known-curve introduction followed by joint inference; cached examples, native checkpoint route and minimal live-training scaffold | Narrative and executable examples still describe old noise/63-point tasks. Update together after new acceptance; do not display an old posterior beside new observations or silently reuse old assets |

### What can actually be reused

- Reuse the European dynamics, conditional estimator, curve integrals, antithetics, collection machinery, source hashing and evidence formats where their assumptions still hold.
- Existing raw European quote provenance may be reused if it is the accepted frozen snapshot; carry and quote selection remain traceable. Historical SPY observations are American-contract evidence and are not relabelled SPXW data.
- Existing clean numerical prices may support a new noise law **only where physical rows, curve law, contracts, units and numerical target match, and the stricter precision audit passes**. Keep their original producer/payload identity and create a derived task/consumer binding. A changed likelihood does not magically improve old labels.
- Changed strikes, maturities, clock, curve family, numerical algorithm or physical prior support require new labels for the changed domain, or an explicitly checked matching subset. Preserve common rows/contract columns and augment only missing compatible columns where exact reuse is possible. Never interpolate price columns and call them freshly simulated contracts.
- Posterior networks, forward-pricer output scales, references and cached posteriors belong to their original task. A new task requires new compatible fits/references or an explicitly validated migration; copying files or editing task names is not migration. Old bank extension/resumption still requires its matching producer checkout/environment.

### Preserve PR #74 and runtime ownership

The existing rough lesson work in [PR #74](https://github.com/fmuia/sbi4finance/pull/74) is a separate integration track. The locally available branch `origin/lesson/rough-options-workflow-20261107` was inspected at `1351dc9619225ed90c1327ff2c5f60eacd0702cf`; this is a local branch observation, not a verification of the remote PR's latest state. Its changes include task-bound surrogate loading/comparison, supplied-curve support checks, simulator explanations, local point calibration and analytic joint-curve uncertainty propagation. Preserve these educational improvements and their tests; reconcile against the actual approved PR head before implementation. Do not overwrite that branch or use this planning branch to replace its assets.

The executing runtime remains under `tutorial/src/sbi4f_tutorial/`; lab code imports it. Tutorial code must not import lab modules or read outside `tutorial/`. Keep new bank generation, numerical audits, inference campaigns and publication operators in lab. A reusable fixed-layout module may serve both areas, but broader varying-geometry/masking support remains the independent Heston track.

## 9. Notebook and release checklist

Preserve both **explicit live training** and **pretrained loading** routes for the accepted new task. Cached inspection remains usable without automatically running MC or neural training. Training uses the matching downloaded clean bank and a declared bounded example; loading validates task, design, observation/noise schema and model hashes. Show the simulator-to-price-to-noisy-quote construction and the distinction between forward-pricer learning and posterior learning. Do not expose implementation choices in the participant flow unless they explain an inference decision.

Show supplied versus inferred curve assumptions, known c and its sensitivity, physical/curve priors, realized grid coverage, numerical qualifications and model residuals beside the relevant results. Preserve cached/live and tutorial/backend limitations honestly. Keep illustrations as illustrations, and full posterior product draws paired; state whether bands are pointwise and whether uncertainty is numerical, parameter-induced or predictive.

Promote only through lab acceptance/publication tools, with compact committed evidence and external hashed payloads. Do not hand-edit `tutorial/assets`, `MANIFEST.md` or generated claims tables. Before a later implementation PR, run lab tests, manifest, claims and release-check; run tutorial tests, notebook tests and check when tutorial runtime/notebooks change. Neither passing software checks nor Heston acceptance substitutes for the independent rough scientific gates above.

**Next permitted action after this planning PR:** review the specification and complete read-only preparation while Heston work proceeds. The next numerical action is the separately bounded representative rough panel only after the Heston noise/grid gate and exact rough run specification are accepted. Return its evidence and cost for reflection before a production bank or training allocation.

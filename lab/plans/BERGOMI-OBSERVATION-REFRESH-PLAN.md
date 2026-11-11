# Rough Bergomi observation refresh: joint inference and packed data generation

**Status, 11 November 2026 UTC: the user-authorized joint Bergomi campaign is running on Oscar.** The first 1,000 rows run under Slurm `8822042`; the remaining 299,000 rows are queued under Slurm `8823501` (24 running and 724 pending allocations at submission). The user explicitly superseded both the Heston-success execution prerequisite and the original strict numerical generation gate for this campaign, choosing the historical numerical cost profile with its shortcomings reported. Heston failures remain documented; generation does not establish neural calibration, continuous-model pricing accuracy or scientific acceptance.

## Current execution authorization and training-data target

The current target is **300,000 distinct TRAIN financial parameter/curve rows** for the joint problem. The seven inference targets are `hurst, eta, rho, short_variance, long_variance, decay, c`, with independent `c ~ Uniform(.5,2)`. The expensive clean pricer takes the first six quantities; `c` is added through independent cheap noise augmentation and is never an inference condition. The selected 63 actual European contracts, original physical/curve priors, carry and raw half-spread observation law remain unchanged. The supplied-curve and fixed-c campaigns are dormant; no second 300,000-row campaign is commissioned. Validation/test target sizes are 5,000/7,000, separate from these TRAIN rows and not queued by this work order. Repeated noise draws do not count as extra financial parameter rows.

The new source-bound authorizations are separate from a passed-Heston record and retain `scientific_acceptance:false`, `SBI_training_readiness:false` and `prior_filtering_permitted:false`. Preserve the original source, guards, work orders, failed preflights and stopped runs under their original identities. The user authorized numerical approximation for feasibility; that does not authorize widening market noise, dropping difficult prior regions, changing the financial kernel or claiming calibration before it is measured.

### Active numerical recipe and its limitation

Each exact expiry uses its base uniform grid from time zero (seven blocks; total 8,391 time steps), the canonical hybrid/control-variate pricer, and the original finite adaptive total-path ladder `32768, 65536, 131072, 262144, 524288, 1048576, 2097152`. The global engineering reference is the historical synthetic-noise median **`0.00044092792100661605` in forward-price units**, with preferred nominal MCSE `.25` times that scalar and a final-rung cap `.5` times it. Its historical vector hash is `5c0030a847a54f4e90b29034590af3337426f98467a22fdad5c63ace6f40467c`. This reference allocates compute; it is **not** the market observation SD and does not enter `Y = P + c*h*Z`. It is a declared global approximation to the old per-contract numerical budget, rather than a claim of identical per-contract precision.

The strict proposal in section 5 remains a recorded scientific diagnostic, not a generation prerequisite for this authorized bank. The price target is the finite hybrid base-grid expectation. No per-row refinement is commissioned, and continuous-model discretization bias is uncertified. Keep nominal MC covariance, MCSE/half-spread and MCSE/`(c*h)` diagnostics; same-sample fitted-control covariance is not an independent uncertainty certificate. Rows reaching a numerical or time limit remain in the requested-prior ledger with their flags and partial blocks. A later noise-aware forward-emulator treatment or selected independent refinements may be needed before an official posterior demonstration; labels are useful development progress, not accepted noiseless truth.

### Queued work, resources and durable progress

The first work order covers TRAIN IDs `[0,1000)`, one row per job, 192 one-CPU workers, 8 GiB each, 900 seconds/job and an 860-second row cap. Slurm `8822042` uses the immutable `joint-legacy-budget-1000/plan.json`, SHA256 `1027e17a4a12270724183ad7a51ce26622764107742a5811bc25a0e47bb90f7e`. The completed work order has **996 complete and four row-budget timeouts**. Failed IDs `440,661,664,769` retain their prior truths and completed expiry blocks. The earlier near-complete 995-row snapshot had a completed-row mean of about 108 seconds and a roughly 56-second median; it omitted the unfinished tail and is not a final runtime estimate. Scaling 108 seconds over 300,000 rows at a fully occupied 192 CPUs gives approximately **47 hours**, before queuing, tail failures, retries, setup and validation. It is not an overnight-completion promise.

The remaining work order covers TRAIN IDs `[1000,300000)`: **5,980 shards of 50 rows**, grouped eight independent worker processes per allocation, giving **748 allocations** under one Slurm array `%24`. Each allocation requests one node, eight CPUs, 64 GiB and 12 hours; each child keeps an 860-second row cap and its own shard journal. The maximum live request is 192 CPUs/1,536 GiB, within the discovered per-user 208 CPU/2,500 G entitlement, subject to group load and scheduling. The declared maximum allocated work is **71,808 CPU-hours**, including the four unused CPU positions in the last group; this is the sum of wall caps, not a cost forecast. The full master was launched as **Slurm `8823501`**, with 24 allocations running/724 pending at submission.

The full plan is `joint-legacy-budget-300k/plan.json`, SHA256 `fc246356d5a12e4a0c77413efa3c5f81254676ff80f778ba5e5d503671a38a1b`. The frozen new module/CLI hashes are respectively `494a4761ee54e84449ea87a4e7d946649ddd46f3e3eecea2cb33718147f0879d` and `55f0eb57d24dbc0b4865207cd6066b58dde76007c4e7b0405656c4d640a0fde7`. Both plans and their compact evidence live in the production checkout under `lab/evidence/pilot-results/2026-11-11/bergomi-observation-refresh/`. The full remote root is `/oscar/scratch/afengler/sbi4finance-rough-main-full-20261111`; its `submission.json` hash is `28165f17ca94c30b588fb4e61a3fb304979451f110d7f80122e2ea726267bcb7`.

Every 50-row shard stores `journal.sqlite`, `data.npz`, `status.json` and a worker lease; there is no file per dataset. All requested physical-prior and independently sampled c values are journalled before pricing. Each completed expiry/rung commits durably. A row-local failure is recorded and the worker proceeds to the next row; one dead worker does not cancel its seven siblings. Restarts validate source, task, environment, plan and admission identities and reuse completed work. Failed rows are skipped by default; an explicit reason and a new output identity are required to retry them from the preserved journal. Retain failure history and accepted blocks rather than redrawing a more convenient prior row.

A read-only snapshot/export can expose completed rows **while other rows in the same shard are still running**. It includes all available complete 63-price vectors without filtering on the precision flag, plus the full requested-prior ledger and missing/failed IDs. Final exports retain all 50 requested rows, with validity masks and NaN price fields for failed rows. Early exports are marked development-only and availability-selected: they can support an initial training experiment, but they are not a prior-representative SBC/coverage sample. Never silently train or validate on a precision-passed subset; computational failures may depend on the parameters. A contiguous complete-ID prefix export is available, but missing hard rows must still be disclosed and resolved for representative acceptance studies.

### Read-only export and explicit resumption

The Slurm queue continues independently of this chat. The first full-run startup check found 192 shard journals, 9,600 pre-recorded prior/c draws, 1,965 committed expiry blocks and 140 completed rows with no row errors. A real read-only export from the still-running shard 0 succeeded: rows `1000,1001,1002` had finite arrays of shape `3×7` targets and `3×63` prices, while all 50 requested prior/c rows and masks remained in the snapshot. The receipt is `/oscar/scratch/afengler/sbi4finance-rough-main-full-20261111/early-export-receipt.json`; its immutable export is `snapshots/early-training-1794378894197093766` under that root. These are startup observations, not a later completion count.

On Oscar, use the preserved producer and environment for inspection or resumption:

```sh
export ROUGH_RUN=/oscar/scratch/afengler/sbi4finance-rough-main-full-20261111
export ROUGH_PYTHON=/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python
export PYTHONPATH="$ROUGH_RUN/repo/lab:$ROUGH_RUN/repo/lab/src:$ROUGH_RUN/repo/tutorial/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ROUGH_SCRIPT="$ROUGH_RUN/repo/lab/scripts/rough_actual_layout_overnight_bank.py"
export ROUGH_PLAN="$ROUGH_RUN/preparation/plan.json"
export ROUGH_ADMISSION="$ROUGH_RUN/preparation/admission.json"

# Read-only snapshot while this shard can still be running. Choose a new output path each time.
"$ROUGH_PYTHON" "$ROUGH_SCRIPT" export --plan "$ROUGH_PLAN" \
  --journal "$ROUGH_RUN/payload/shard-000000/journal.sqlite" \
  --out "$ROUGH_RUN/snapshots/early-training-review-001"
```

Adding `--complete-prefix` exports only the contiguous complete-ID prefix of that shard; all requested/missing IDs are still reported. The export does not price new data. An output based on available completed rows is development data until its selection consequences are resolved for validation.

The following worker commands run **inside a matching eight-CPU Slurm allocation**, with its true `SLURM_CPUS_PER_TASK=8`, after the earlier writer has exited. Shard 0 belongs to allocation group 0; use the frozen plan's map for other indices. Do not run two writers on one shard or modify the Slurm CPU environment.

```sh
# Resume the same shard: completed blocks/rows are reused; recorded failed rows are skipped.
"$ROUGH_PYTHON" "$ROUGH_SCRIPT" worker --plan "$ROUGH_PLAN" \
  --admission "$ROUGH_ADMISSION" --index 0 --allocation-group 0 \
  --out "$ROUGH_RUN/payload/shard-000000"

# Explicit failed-row repair in a NEW output identity, preserving the original journal/history.
"$ROUGH_PYTHON" "$ROUGH_SCRIPT" worker --plan "$ROUGH_PLAN" \
  --admission "$ROUGH_ADMISSION" --index 0 --allocation-group 0 \
  --resume-from "$ROUGH_RUN/payload/shard-000000/journal.sqlite" \
  --retry-failed --retry-reason 'Explicit review of retained row-budget failures' \
  --out "$ROUGH_RUN/manual-retries/shard-000000-repair-001"
```

A row-budget timeout preserves already committed blocks, but it is a **recorded failed row**, not an automatic recovery promise. The default restart skips it; the explicit copied-journal repair above reuses its accepted work under the same finite limits and can still fail again. An interruption before a committed result also preserves earlier commits; any unfinished rung may need deterministic replay. A dead child leaves its journal recoverable and does not terminate siblings. No automatic retry/redraw hides failed prior regions.

Current output is on Oscar scratch. Durable per-block writes protect against process interruption; they are not a separate backup against scratch loss or retention policy. Hashed collection and an additional external backup remain subsequent work.

The original 48-component pilot ran as `8820876` and was stopped at the user's request, preserving eight complete components and partial progress. Its measured peak RSS was below 2 GiB, supporting the current conservative 8 GiB/worker request instead of the original static 32 GiB allowance. Its strict spread-based MCSE and grid checks did not pass. The later grid-4/262,144-path 128-row run `8821443` was explicitly stopped when the user chose the feasible historical-cost recipe. These numerical limitations remain evidence; neither stopped run is relabelled successful.

The historical preparation and stricter scientific diagnostics below remain useful provenance. Where they describe known c, supplied-curve controls or the automatic Heston gate, those are preserved or deferred routes, not the active campaign. No Heston failure is transferred as acceptance of a rough posterior.

The Heston dependency remains [PR #85](https://github.com/fmuia/sbi4finance/pull/85). Its supported revision freezes 63 actual contracts, a declared final-call target +1.5 in fixed 16% volatility coordinates and a longest actual tenor of about 10.7 months. The source-bound 200k/300k, 300-epoch comparison is complete; see the [combined saved-data report](https://github.com/fmuia/sbi4finance/blob/feat/heston-spread-baseline-20261109/lab/evidence/pilot-results/2026-11-09/heston-spread-baseline/revised-grid/recovery-bank-independent/report.html). A versioned Black-control-variate representation passed the 98-point numerical panel and all three new target-matched references are ready. These are useful Heston numerical results, not rough-pricing certificates or neural-posterior acceptance. The literal +2/one-year audit and the supported revision retain separate identities.

All rows below describe the declared final epoch 300, on the same 512 historical truth rows. Native validation and both declared secondary endpoints are retained in the report.

| Heston task | Fixed training rows | c median RMSE | c nominal 90% coverage | Historical SBC screens | v₀ nominal 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved old 54 control | 200k | .14421 | 89.06% | 4/6 | 95.12% |
| Revised 63 primary | 200k | .14045 | 88.67% | 5/6 | 97.85% |
| Revised 63 primary | 300k | .13494 | 90.23% | 5/6 | 97.46% |

Point recovery is strong and c improves, but the remaining uncertainty error is material. Both revised primaries fail the v₀ SBC screen and substantially overcover v₀. At 300k, physical conditional 90% widths on the two synthetic cases remain 1.05–2.86 times their target-matched MCMC widths, with remaining mean disagreements. These synthetic comparisons have no raw support violations, so relaxing a strict zero-outside-support rule would not remove the conditional failures. Market conditional discrepancies and c concentration near its prior upper boundary also remain; they do not by themselves identify whether market-model inadequacy, an unusual observation or the noise convention is responsible. The distinct minimum-validation epoch 259 passes only 3/6 historical SBC screens and also fails the necessary diagnostics; it does not rescue the final primary. Training presentations are 60m for 200k and 90m for 300k, so this is not an equal-compute comparison.

Independent raw-draw verification passed for all four declared endpoints, while **every endpoint fails the necessary scientific screens**. Verification establishes faithful saved results, not acceptance or promotion. The original automatic Heston initiation condition remains unmet. The separately authorized rough campaign above proceeds without claiming that condition passed; do not transfer Heston parameters, posterior draws or its fitted whitening matrix.

Earlier fixed-c, inferred-c, large-batch and whitening failures remain part of the Heston evidence. The isolated spline whitening pilot and later native FlowMatching experiments are different recipes and budgets. Neither a favorable plot nor a decreasing objective is an acceptance criterion. Flexible locations/masking remain the separate [PR #83](https://github.com/fmuia/sbi4finance/pull/83) track; this work is one fixed actual layout.

The active rough baseline is the **seven-target joint-curve plus inferred-c task** specified above. The original three-target supplied-curve and six-target fixed-c financial declarations remain unchanged as provenance and dormant controls; the new outer inference blueprint adds independent c without changing clean pricing. Its posterior, preprocessing and validation still need their own evidence. No GP discrepancy, vega-scaled tau or market-derived latent curve is introduced.

## 1. Scientific deliverable and dependencies

The deliverable is a reproducible European-option inference task whose data, clean-price simulator, Gaussian likelihood, deterministic calibration, numerical reference and amortized estimator agree on contracts, units, priors and conditions. The lesson must explain what the option information identifies and how remaining uncertainty reaches model fair values. Wide posterior intervals are not an acceptance requirement.

The frozen financial dependency is the ordered set of actual European SPXW contracts, carry/provenance and half-spread vector. The current rough execution authorization is independent of a passed-Heston diagnostic record. Rough inference must establish its own seven-target calibration, interval coverage, matched conditional references and numerical sensitivity before an official demonstration or publication. Whitening is a candidate input representation; it does not waive those checks. A decreasing loss, strong recovery plot or completed price bank is insufficient on its own.

Port the **frozen financial contract** and its declared observation convention. Rough priors, curve laws, numerical tolerances, networks and scientific acceptance remain separate. A change in selected contracts or measurement law requires a new task identity before rough labels are commissioned; it does not authorize silently relabelling old assets.

The supported 7-by-9 actual-quote layout and its exact maturities are frozen in the source-bound rough preparation. This freezes a numerical-pilot task, not a pricing-accuracy or neural-acceptance claim. The old 63-contract symmetric grid, the Heston pilot's actual 54 contracts and the revised 63-contract task remain distinct physical designs. A coordinate relabelling alone changes no contracts; selecting different strikes or maturities does.

## 2. Observation and input contract

For one frozen design, use

\[
Y_i=P_i(\psi)+c h_i Z_i,\qquad Z_i\stackrel{\mathrm{iid}}\sim N(0,1),\qquad c>0.
\]

Here \(\psi\) contains the rough dynamics and, for joint inference, the curve coefficients. \(h_i\) is the observed **price half-spread**, held fixed within the task; \(c\) is an independently sampled dimensionless multiplier, inferred jointly with the six financial parameters under \(U(.5,2)\). No vega-scaled tau component is present. Half-spread equals one Gaussian SD at \(c=1\) by a declared working convention, not by an empirical theorem about quotes.

- Infer c jointly in the active task; hide the generating c and every feature containing it from inference conditions. The frozen profile h remains known. A later fixed-c sensitivity study can use the same observed price vector, but it requires separately declared consumers and is not a second commissioned bank.
- Keep the frozen spread profile and full ordered design explicit in task/checkpoint metadata. Do not replace a changed spread vector by a median scalar ratio without a separately verified proportionality assumption. A different snapshot is not silently supported by this fixed-design model.
- Use actual quote locations. No smoothing, strike/maturity interpolation or American/AM-settled substitutions enter this baseline. Preserve source timestamps, roots and settlement conventions, and distinguish last-trade time from an unavailable bid/ask update time.
- For the existing forward-unit pricing API, set \(p_i=y_i^{price}/(D_iF_i)\), \(h_i=h_i^{price}/(D_iF_i)\) and strike \(K_i/F_i\). Apply the same conversion to predictions and numerical covariance. Preserve raw quotes and carry estimates alongside these derived inputs. Carry remains conditional unless a separate uncertainty study is declared.
- Synthetic observations receive fresh independent quote errors; clean prices and their numerical uncertainty are stored separately. Reuse a clean physical row across independently generated c/noise augmentations; do not count noise augmentation as additional independent physical parameter locations. Keep all augmentations of one physical row in its original split.
- Keep Gaussian observations as generated, including finite negative values. No IV inversion, positivity truncation, rejection or redraw is permitted merely because a simulated quote is negative.

The inferred-c task needs matching seven-target checkpoints, references, normalization and conditional-scale checks. Boundary mass is not evidence of pure measurement noise; it can also reflect model or numerical misspecification. Do not substitute an equal average of fixed-c posteriors for marginalizing the joint posterior. The historical known-c task metadata inside the frozen financial preparation does not turn c into a condition of the new outer seven-target inference task.

### Training-input covariance and whitening

Whitening is an optional **fixed input representation**, separate from the observation model. For input features \(x_i=Y_i/h_i\), estimate a mean and full covariance only across the declared noisy **training-prior** surfaces. Common changes in volatility level, skew and maturity shape move many quote features together, so this covariance combines parameter-induced variation and simulated quote noise. It is not an estimate of measurement-error covariance at fixed parameters and is not a covariance estimated from market residuals.

If the accepted Heston recipe uses it, compare or port the fixed full-rank transform with all quote features retained; predeclare eigenvalue regularization, precision and ordering. Apply it to the **entire already-noisy vector**, using the identical stored transform at validation and inference. Do not whiten clean prices and then inject new independent errors in transformed coordinates. In raw price units the law remains \(\operatorname{Cov}(Y\mid\psi,c)=c^2\operatorname{diag}(h^2)\); the transformed quote errors generally become correlated. Neither the Gaussian likelihood, its normalization nor the known/inferred role of c changes because the network inputs change.

A rough task needs its **own** training-only mean/covariance/transform fitted for its dynamics, curve law, prior, exact physical design, spread profile and feature units. Do not import the Heston matrix or fit on validation, selected reference, reserved or market observations. Persist a versioned immutable sidecar with training-row and condition hashes, feature order, task identity, eigenvalues/floor, transform and inverse, precision and source hashes; bundle it with every compatible checkpoint and verify application exactly once. For a supplied-curve task, declare which inputs are transformed and preserve the curve's explicit conditional role. A changed task or grid requires a new fitted transform and identity.

A well-conditioned noisy training covariance is not evidence that rough pricing labels are accurate. It cannot legitimize a Monte Carlo floor, erase a grid shift or correct surrogate bias. The separate numerical gate below must examine both errors entering transformed network inputs and errors entering the raw-price likelihood.

## 3. Curve inference and information experiments

Retain the distinction between the existing teaching families, while commissioning only the joint route now:

| Route | Unknowns | Curve information | Interpretation |
| --- | --- | --- | --- |
| Supplied-curve synthetic introduction (dormant) | H, eta, rho | Explicit known positive curve, varied across training cases and fixed within each inference problem | Preserved conditional teaching task; no new campaign commissioned |
| Active joint-curve and noise calibration | H, eta, rho, three curve coefficients and c | Generating curve and c excluded from inference conditions | Main seven-target campaign; expensive clean labels have six inputs |

The current joint family is \(\xi_0(t)=v_{long}+(v_{short}-v_{long})e^{-a t}\), with six ordered targets `hurst, eta, rho, short_variance, long_variance, decay`. Its current bounds are H `[.02,.35]`, eta `[.5,3.5]`, rho `[-.99,-.2]`, variance levels `[.01,.16]` and decay `[.25,8]` per year. These are starting controls to review in the numerical panel, not parameters inherited from Heston or a claim of final market adequacy. Retain difficult low-H/high-eta and high-variance cases; numerical difficulty alone does not justify narrowing the prior. The supplied seven-interval curve law and the analytic joint family remain different tasks.

For a later information-removal experiment, infer the curve jointly from the surviving smile quotes or use truly external, documented curve information. Never supply a curve reconstructed from withheld later-expiry or wing quotes. Common carry anchors can remain a condition, but then the experiment removes smile information conditional on that carry; it does not remove all information at those maturities. ATM information used for design selection must also remain visible as conditioning/selection information.

Start product propagation with analytic expected average variance and a forward average over declared intervals, keeping complete joint posterior rows paired. With an exactly supplied curve these rates have zero H/eta/rho-induced uncertainty by construction; this is a control, not identification evidence. A nontrivial H/eta/rho-sensitive product needs its own bounded payoff-pricing check. Fair-value posterior quantiles are conditional on model/curve/tail assumptions and are not automatically model-risk or regulatory reserves.

## 4. Exact maturities and the rough numerical target

The accepted actual-quote design uses elapsed ACT/365F time to its explicit settlement timestamp. Current rough pricing uses uniform grids from the Q origin and rejects contracts not on grid nodes. Replacing a maturity by a nearest business-day/252 node changes the contract and the variance clock. It cannot be called a coordinate conversion or used to reuse old labels.

The minimal new route keeps the current hybrid estimator on uniform grids from the Q origin, with one block ending exactly at each of the seven actual expiries. Only that expiry’s nine contracts belong to its block. Preserve each raw ACT/365F maturity and settlement timestamp; choose integer step counts and refine them by integer multipliers. No expiry, strike, carry estimate or prior knot is rounded to make a validator pass.

The current `shared/tasks.py` validator requires every prior curve knot to land on each relevant block grid, and `full_surface.validate_task` requires the old ordered 63 contracts. Those are legacy software contracts, not mathematical requirements of the rough model. At the current valuation time `2026-11-09T20:13:37Z`, the actual expiry intervals have a one-second GCD: a single global exact grid would need 28,079,183 steps by the last expiry. Including the old 252-day curve knots would require 220,752,000 steps per year. This demonstrates why merely satisfying the old validator is a poor implementation route; it does not require a one-second rough simulator.

Preserve the original supplied seven-interval forward-variance prior and its original knot times. A new actual-layout adapter can evaluate that exact piecewise-constant curve at every declared block’s left nodes, pass those node values to the unchanged hybrid simulator and measure ordinary time-discretization error when a step crosses a knot. The continuous curve is not moved or smoothed to the new expiries. An explicitly declared interval-average scheme is an alternative discretization to compare, not an unannounced replacement. In both cases the integrated-variance control must be centered with the **same discrete weights used by the asset scheme**. Inserting a continuous curve integral into an unchanged left-point control can introduce bias. Record crossed knots, straddling step widths, discrete-versus-continuous integrals and paired coarse/fine/finer shifts.

The current joint exponential curve also retains its own original financial family and coefficients. Evaluate it directly at the actual simulation nodes using the existing canonical analytic helper; do not switch the supplied-curve family to this analytic family merely to evade alignment checks. A nonuniform-grid hybrid driver is not needed for this first route. Both routes need a new isolated task/adapter identity that decouples financial curve knots, observation times and simulation nodes while leaving old validators/loaders unchanged.

Record the maturity, horizon, step count, curve evaluation rule, hybrid normalizer, antithetic convention, control variates, independent block structure and RNG stream protocol. Starting each maturity block at time zero retains rough history; restarting from current variance or a current driver value is not a valid rough continuation state. Do not change the volatility model's time unit while keeping the old prior/labels under the same identity.

The target hierarchy must be explicit:

1. Continuous-model expected European prices, the scientific ideal.
2. A finite time-grid expectation, with measured discretization effects.
3. A Monte Carlo estimate of that expectation, with price-mean covariance and finite-sample diagnostics.
4. A deterministic learned mean-pricer approximation, if used for references.
5. The amortized posterior approximation conditional on the declared observation law.

Agreement at a lower layer does not certify the earlier layers. Heston's deterministic Fourier audit does not certify any rough layer.

## 5. Historical strict precision proposal and retained scientific diagnostics

The original strict proposal defined \(s_{min,i}=.5h_i\), the smallest observation SD over the supported c prior. It required numerical tolerances against that vector independently of inferred error scales or market residual size. **The user has superseded this as a generation prerequisite**, adopting the historical engineering reference described above. These stricter checks remain useful measurements and later sensitivity targets; their failure must not be hidden by the relaxed generation policy.

The historical feasibility target was reported Monte Carlo SE at most \(.25s_{min,i}\) per contract, with **no inherited final-rung .5 exception** for the new task. This is an empirical MC target, not a guarantee about continuous-price bias. Retain unsuccessful rows, IDs and costs; reaching the declared path cap is a failure/stop, not permission to widen noise, discard a row or increase the cap automatically. The active threshold has its own explicit source-bound policy and user authorization; it does not establish stability of the posterior under more accurate pricing.

| Error source | Required pilot evidence | What does not suffice |
| --- | --- | --- |
| Monte Carlo sampling | Independent batches; antithetic pairs as statistical units; full price-mean covariance; nominal residual-based SE versus between-batch dispersion | One favorable seed or the within-batch fitted-control residual alone |
| Control variates | Controlled/raw comparison and replicated mean shifts/dispersion, retaining difficult cases | Bounded raw put payoffs do not make every fitted control-variate diagnostic an exact uncertainty certificate |
| Time discretization/curve representation | Paired coarse/fine/finer comparisons at exact contract times, with uncertainty in price differences and curve-integral controls | A doubled grid is not exact truth; a small MCSE does not bound grid bias |
| Surrogate approximation | Independent refined price labels, errors and likelihood effects in strict measurement units, including representative posterior regions | Training/validation MSE or task/model hash matching alone |
| Reference posterior | Matched numerical likelihood, seed/grid/pricer identity, sampler convergence and sensitivity to improved prices | Convergence around an inaccurate fixed pricer |

Predeclare a provisional price-shift screen of `.1 * s_min` for resolved grid shifts and independent surrogate residuals, reporting Monte Carlo confidence intervals and all failing contracts. If uncertainty cannot resolve that scale, report the audit as unresolved. These screens guide a bounded development decision; full-vector likelihood effects and posterior sensitivity remain necessary before scientific acceptance. Do not add pricing variance to observation noise or subtract a noisy SE estimate from its variance without a separately justified target.

Numerical covariance matters even with independent observation error. With \(S_{min}=\operatorname{diag}(s_{min}^2)\), inspect the spectrum and selected contrasts of \(S_{min}^{-1/2}C_{MC}S_{min}^{-1/2}\), paired refinement covariance and accumulated likelihood shifts. Marginal SE bounds alone do not bound every combined direction. A covariance estimated from only a few independent batches can be rank deficient; retain it descriptively rather than inverting it as an exact likelihood correction. A future correlated observation law requires a separate likelihood and precision specification; training-input whitening does not introduce that law.

For deterministic prediction error \(\delta=\widehat P-P\) and residual \(r=y-P\), record the actual likelihood change \(r^T S^{-1}\delta-\tfrac12\delta^T S^{-1}\delta\) under the same frozen Gaussian covariance. For estimated price shifts, propagate uncertainty instead of treating the estimate as exact. Small individual residuals can accumulate across contracts.

If an input transform is adopted, also audit its numerical amplification. In column notation let \(D_h=\operatorname{diag}(h)\), let W be the stored full-rank transform of \(Y/h\), and let \(A=W D_h^{-1}\). A price error \(\delta\) enters the network as \(A\delta\), while price-mean covariance becomes \(A C_{MC}A^T\). Record these transformed errors, their covariance/dependent directions and sensitivity to the declared eigenvalue floor, alongside the raw-price likelihood shift above. A small transformed error does not waive a failing raw-likelihood gate, and a small marginal raw SE does not establish harmless transformed error. Audit Monte Carlo, time-grid/curve and surrogate contributions independently; do not fit W to make their failures disappear.

## 6. Current implementation sequence

| Phase | Work | Exit evidence |
| --- | --- | --- |
| R0/R1: preserved preparation and stopped numerical panel | Keep exact-time financial declarations, portable validators, strict-panel failures and measured costs | Immutable source and raw evidence; no false Heston or continuous-price acceptance |
| R2/R3: authorized data generation, running | Complete the separate first 1,000 and packed remaining 299,000 TRAIN requests using the agreed legacy engineering budget | Durable requested-prior ledger, complete/failed masks, MC covariance, runtime/rung statistics and source-bound partial exports |
| R3: development inference alongside generation | Use an explicitly identified partial bank to fit task-specific training transforms and a seven-target posterior/mean-pricer experiment | Disclose availability selection and numerical approximation; keep validation and reserved acceptance observations separate |
| R3: establish numerical and inference limits | Obtain independent refined labels/references at representative points and posterior regions; test c recovery, SBC, coverage and conditional sensitivity | Report both successes and failures under the declared finite-grid/price treatment before scientific acceptance |
| R4: notebook integration and reflection | Update the existing rough notebook/helpers, live/pretrained routes and evidence only for the supported new task | Checked routes and accurate teaching claims, followed by explicit reflection before broader changes |

The earlier 8 CPU-hour/48-component R1 ceiling applied only to that preserved pilot. It is not the full-bank budget. The current finite work orders explicitly queue the 300,000 TRAIN target; no supplied-curve bank, GPU training allocation, direct-MC reference campaign or automatic expansion beyond the finite path ladder is attached. The first 1,000 and remaining 299,000 retain distinct producers/plans and may be combined downstream only through their compatible financial/RNG identities and preserved provenance.

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

## 8. Preserved pre-authorization readiness audit and original conditional route

**Historical scope:** this section records the earlier preparation state. Statements here that no MC/job has started, that known-c support comes first, or that Heston must pass apply to that original route only. Current execution, inferred-c scope and numerical relaxation are recorded above; the old artifacts/commands remain unchanged.

Initial code audit: main commit `1dbfa8342d17027f8e4cb83339db93a8d6d2a9b9`; refreshed read-only inventory on 10 November 2026 (11 November UTC). The refresh read active training-part metadata, the preserved task/snapshot manifests and the actual63-quote design. A prior inspection opened exactly one 25-row TRAIN payload per bank for covariance checks; no validation/test payload arrays were opened. No new rough prices, covariance estimates, whitening fit or training are claimed.

| Source inspected | Current behavior / reusable component | Required change or limitation for this refresh |
| --- | --- | --- |
| [Shared task validation](../../tutorial/src/sbi4f_tutorial/shared/tasks.py), [lab task builders](../src/sbi4finance/calibration/tasks.py) | Explicit priors/contracts/noise; known versus joint curve roles; rough blocks must hit contract/curve-knot nodes | New versioned actual-layout/spread/known-c task contract. Old builders generate the symmetric fixed-20% grid and fixed synthetic noise; arbitrary ACT/365F times fail alignment checks |
| [Complete-surface adapter](../../tutorial/src/sbi4f_tutorial/shared/full_surface.py) | Reusable bounds/training/sampling/checkpoint structure; joint curve auxiliaries stay hidden | Currently requires the canonical ordered 63 contracts, 63-price arrays and existing target names. Its set encoders also require that full design; they do not provide arbitrary quotes/masks. Add isolated versioned fixed-layout support and known-c conditioning; preserve old loaders |
| [Rough simulator](../../tutorial/src/sbi4f_tutorial/rough_options/simulator.py), [conditional puts and curve helpers](../../tutorial/src/sbi4f_tutorial/rough_options/model.py) | Hybrid driver from the Q origin, covariance-grid test reference, bounded conditional raw puts; existing innovations/curve checks | Uniform-grid API and asset discretization remain. The small covariance reference is capped at 128 steps and is not a full production-price certificate. Preserve mathematical/RNG conventions when adapting times |
| [Joint curve construction](../src/sbi4finance/calibration/tasks.py), [analytic curve values/integrals](../src/sbi4finance/calibration/pricing.py) | Current lab joint builder samples the exponential curve at the union of simulator grid nodes; controls use discrete left sums | Preserve exact node evaluation after changing blocks. Continuous analytic curve integrals and discrete simulator controls are different quantities. PR #74 moves/reuses these helpers in the participant runtime; reconcile that change rather than introduce a second implementation |
| [Shared surface pricing](../../tutorial/src/sbi4f_tutorial/shared/pricing.py) | Conditional European puts, parity calls, antithetic pairs, fitted controls, block-diagonal price-mean covariance, explicit path streams | `rough_surface` gates SE against the supplied old sigma vector and can accept a larger final-rung cap. New numerical gating must use `.5*h` and explicitly reject inherited exceptions; independently audit control diagnostics |
| [Coupled refinement](../src/sbi4finance/bergomi_refinement.py), [price panel validator](../scripts/validate_rough_price_panel.py) | Paired grid differences, raw/controlled comparisons and covariance reporting | Existing panels assume aligned nodes and specific 63-column identities. Reuse methods after exact-task adaptation; rough coupling is common random numbers, not an exact common-Wiener bridge or removal of bias |
| [Mean pricer](../src/sbi4finance/calibration/pricer_net.py), [reference likelihood](../src/sbi4finance/calibration/reference.py), [readiness checker](../scripts/check_rough_emulator_readiness.py) | Task/model binding, deterministic Gaussian normalization, explicit curve features, independent gradient checks | Outputs are scaled by the old quote sigma. New task binding/output scaling and independent strict-noise validation are required; identity checks alone do not establish accuracy |
| [Published joint task](../../tutorial/assets/backbones/rough-joint100k-validation0/task.json), [development configuration](../configs/calibration_rough_joint_development_20k.json), [training-data guide](training-data-guide.md) | Historical complete-grid banks and six-target analytic-curve task remain reproducible controls | Published task uses 63 fixed-20% contracts, blocks 1280/2016/2016, SE target .25 and final cap .5 relative to old noise. Its known and joint banks are different curve laws. Only the first 25-row training payload per bank was inspected; no label was repriced or certified reusable |
| [Rough notebook](../../tutorial/notebooks/slot5_calibration.py), [lesson helpers](../../tutorial/src/sbi4f_tutorial/rough_options/lesson.py), [curriculum](../../tutorial/curriculum/05-slot5-rough-bergomi-to-a-reserve.md) | Known-curve introduction followed by joint inference; cached examples, native checkpoint route and minimal live-training scaffold | Narrative and executable examples still describe old noise/63-point tasks. Update together after new acceptance; do not display an old posterior beside new observations or silently reuse old assets |

### Preserved banks, measured costs and actual-layout reuse

Both locally preserved snapshots contain 100,000 training, 5,000 validation and 7,000 reserved-test rows. Their common local prefix is `/Users/afengler/Projects/proj_sbi4finance/local-artifacts/tutorial-backbones-20261029/oscar-scratch/`.

| Snapshot | Prior/context identity | Active training-generation cost | Exact reuse for the new 63 contracts |
| --- | --- | --- | --- |
| `sbi4finance-rough-known-backbone-20261027/snapshot-100k` | root seed 271300; three targets plus supplied original seven-interval `linear_wobble` curve | 4,000 active TRAIN part manifests;298.396743 CPUh, 10.742283 CPU seconds/row | 0 matching maturities and 0 complete physical contracts; no direct price-label reuse |
| `sbi4finance-native-bank-20261027/snapshot-100k` | root seed 271200; six targets and the exponential joint curve | 9,828 active TRAIN part manifests;2,192.451801 CPUh, 78.928265 CPU seconds/row | 0 matching maturities and 0 complete physical contracts; no direct price-label reuse |

Snapshot SHA256s are respectively `f1bfa42f7bf2acba3daaffb263b5bdeb60b5b12ebdfc267c5c5cd6eb7e08c313` and `eb72588b9b7665e0e416cb5d3a1bcf8cd78a1c97f2b43f18ea8246c749a871d8`; task fingerprints are `c99b0dd131896d89d8483a1668a3fde0fb7a0f28794c3db7fe14635cbbd52b65` and `29eca4e1b970c4f13b99573939aa3449c5280a3a4478f7d24beace6539a9889d`. All active TRAIN part-manifest hashes were checked before summing `generation_cpu_seconds`; validation/test payloads were not loaded. These totals omit failed/superseded generation and are historical production measurements, not a new-task budget.

For orientation only, scaling those old 100k averages gives596.8/895.2 CPUh for 200k/300k supplied-curve rows and4,384.9/6,577.4 CPUh for 200k/300k joint rows. The new contracts, seven blocks, stricter spread-based precision and changed numerical allocation invalidate these as predictions for the refresh. Do not allocate such a bank from these extrapolations.

The first TRAIN covariance payloads have shape 25×63×63; the inspected matrices were finite, symmetric and PSD to numerical precision, and their diagonals matched stored MCSE² at about 1e-24. This establishes stored covariance structure for those rows only. It does not certify new labels or convert the old `.25*sigma`/last-rung`.5*sigma` exception into the new strict measurement gate.

The new forward-unit half-spreads `h_price/(D*F)` have min/median/max `6.60650e-6 /3.31241e-5 /2.53321e-4`; median strict supported SD is `.5*h=1.65620e-5`. The old synthetic noise median is `4.40928e-4` (range`7.39312e-5`–`1.74928e-3`). The 26.6× difference between these medians is a scale comparison across different designs, not a per-contract correspondence or a path-cost multiplier. Preserve raw prices, D and F, and transform every price covariance by `diag(1/(D*F))` on both sides. A later rough-only W must also transform MC covariance as `W diag(1/h) C_MC diag(1/h) W.T`; it never changes the raw Gaussian likelihood.

### Reviewable first work order and compute availability

The existing 8 allocated-CPU-hour ceiling is retained. Prepare four initial rows from the predeclared 12 joint/four supplied-curve panel: typical joint, low-H/high-eta joint, high-variance joint and a supplied-curve control. Use the original priors and curve laws. Start at 32,768 total paths (16,384 antithetic pairs), the old first path-ladder rung, with grid multipliers 1, 2, 4 and four independent batches. There is no automatic path escalation to rescue a failing strict-noise check.

For a source-bound paired implementation, the proposed first stage is 48 components: 32 two-grid comparisons (`1→2` and`2→4`, four rows×four independent batches) plus 16 finest-grid replay controls. This is 80 grid-estimator evaluations, not 48 independent physical rows. Reuse the existing tested controlled-CRN helper; it replays the canonical finest marginal and rebuilds each coarse hybrid field. It is not an exact common-Wiener bridge. Retain raw/controlled prices, fitted-control diagnostics, complete 63×63 price-mean covariance, paired-difference covariance, knot-crossing diagnostics and CPU/RSS costs. Replay controls are not independent observations.

A 1 CPU component has a 600-second total wall/allocated-CPU cap, including imports, failures and output. At most two run concurrently; 48×600 seconds is the existing 8 CPUh ceiling. A capped component remains failed/inconclusive with its original identity. Source-bound workers must refuse execution until the predeclared successful Heston working-grid record exists. No array, allocation or rough MC has been submitted by this readiness work. Extending to the remaining 12 declared rows or commissioning labels requires the measured report and a new bounded work order; it is not hidden inside this first-stage cap.

Read-only Oscar queries during this 11 November 2026 UTC readiness session reported 14,453 allocated / 779 idle / 48 other batch CPUs out of 15,280; this user’s Slurm queue was empty. Account/QOS discovery reports `carney-mjfrank-condo2/batch/carney-condo2` with 208 CPU and 2500G per-user limits; `default/batch/normal` with 64 CPU; and a `batch2` lane with 512 CPU. GPU lane limits are entitlements, not free-GPU measurements, and no GPU is requested for R1. Availability is an observation at that time, not a reservation or future guarantee.

The initial read-only execution-host preflight reached `login009`. The historical reference Python at `/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python` exists and reports Python 3.12.16, NumPy 2.5.3 and SciPy 1.18.1. Scratch reports about 404 TB free at filesystem level; this is not a user-quota guarantee. Both historical source roots retain the legacy `src/sbi4finance` / `tutorial_final` layout and were not repurposed. The complete new source has now been staged separately under `/oscar/scratch/afengler/sbi4finance-rough-actual-layout-portable-20261110/repo`; its successful hash/environment preflight is detailed below. No allocation or rough pricing was initiated.

Read-only checks can be repeated without allocating work:

```sh
python cluster/discover_cluster.py --ssh-host oscar --print
ssh oscar 'sinfo -h -p batch -o "%P|%C"'
ssh oscar 'squeue -h -u "$USER" -o "%i|%P|%j|%T|%M|%C"'
```

Existing pilot CLIs (`pilot_rough_numerics.py`, `pilot_rough_working_grid.py`) bind the old complete63 task and are **not** valid launch commands for the new physical design. The isolated actual-layout interfaces and portable declaration now pass pure unit/identity and execution-host checks. The prepared submission file uses `--array=0-47%2 --cpus-per-task=1 --mem=32G --time=00:10:00` on the verified CPU lane, with immutable source/task/gate bindings and no automatic retries. Current lab **and sibling tutorial runtime** are staged at a new remote identity with verified source hashes/versions; an old bank producer checkout is not silently repurposed. The preparer prints worker/submission commands for review and does not submit them.

### Preserved preparation and original conditional commands

The new source preparation exists in the active Heston checkout `sbi4finance-heston-spread-baseline-20261109`: `tutorial/src/sbi4f_tutorial/rough_options/actual_layout.py`, the original `lab/src/sbi4finance/calibration/rough_actual_layout_pilot.py`, and the new versioned `rough_actual_layout_pilot_portable.py` plus `lab/scripts/{prepare,run}_rough_actual_layout_pilot_portable.py`. The old task validators, financial kernels, curve priors and source-bound Heston consumers remain unchanged. Pure tests cover both worker dispatch branches with fake arrays; they do not execute a rough Monte Carlo study.

The active declaration is `lab/evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation-portable/work-order.json`, SHA256 `38455e9565681ea502cac33116f92c27f2d3396eb3ff06c0248b251e9f9edb45`; its sibling `pilot.sbatch` contains the reviewable worker command. It copies the original stored cases and tasks **bit for bit**. The authoritative panel fingerprint is `cbb2c2c31890c6a6ffb45eedf255cb0d7e1ff223c6801c918d11eff7d8fe19c6`. It binds the original work order, actual Heston design/task, both historical rough snapshot manifests, original financial tasks, all 23 new/reused source files, exact components/seeds/paths and NumPy/SciPy versions. No validation/test payload arrays were read. Preparation status remains `prepared_only_no_MC_executed` with zero generated prices.

The original source commit `450631d2664599cefbf2ee26c7ec745257556d52`, preparation and first remote attempt remain immutable. That source-only preflight correctly failed because regenerating known-curve levels/context on x86 differed from ARM by approximately 1–2 last bits, at most `2.7756e-17` in the reported context values. The separately versioned portable validator permits **32 machine epsilons relative, absolute tolerance zero, only when comparing finite host-recomputed derived known-curve levels/context**. All physical parameters, priors, knots, identities, seeds, paths and stored panel fingerprints remain exact. Execution always uses the original stored curve values, never host-recomputed replacements. This fixes a validator portability requirement and changes no financial target or priced case.

Portable implementation commit `3e979d94bc1966f62b917236a9e9963cc37029b7` passed 867 lab tests, including all 13 original preparation and 9 new portability guards. Required manifest, claims and runtime-boundary checks pass; unchanged tutorial runtime also passed 53 tests, seven headless notebook exports and marimo check with two existing empty-cell warnings. The fresh complete source archive is 110,915,578 bytes, SHA256 `db3f4b3bff54ac507fd13cce451aa7534ee3f63ffeb904de143ca0e04dc15daa`; ignored payload banks and local environments were excluded.

On Oscar, source checking **outside the checkout working directory** passed for all 23 files, unchanged stored panel fingerprint, Python 3.12.16, NumPy 2.5.3 and SciPy 1.18.1. Imports resolve to the new source root. The remote preflight record at `/oscar/scratch/afengler/sbi4finance-rough-actual-layout-portable-20261110/preflight/deployment.json` has SHA256 `499ca1d14f9f2f3961a9917c2fa89e2c5c452bcf5dde4c9deb8749b35951e7ae`; compact records are retained under the local portable preparation. Its as-of Slurm query reported 14,283 allocated / 949 idle / 48 other batch CPUs out of 15,280 and an empty user queue. These are availability observations, not reservations. The earlier failed deployment remains visible and was not overwritten.

Static memory inspection found that the **current canonical chunk is 2,048 antithetic pairs**, rather than the historical 128-pair setting. The largest declared refined joint grid has 7,184 steps and FFT size16,384. A live innovation chunk is `[2048,7184,3]` float64 (336.75MiB), not a full16,384-pair path tensor. Conservative accounting for64 float64 path arrays,16 complex128 FFT arrays and16 retained payoff/control copies totals16,147,021,824bytes, about15.0GiB. The new submission therefore requests **32GiB per component**, without changing the numerical kernel, paths, chunk, 600-second wall cap or8CPUh ceiling. This is a static allowance, not a measured peak-RSS claim; Slurm MaxRSS/MaxVMSize remain part of the first pilot's measured evidence.

The completed final epoch 300 Heston evaluations do not satisfy the frozen guard. The preparer refuses to bind a condition unless both complete native100 and historical512 panels pass all six marginal coverage/SBC and raw-support checks, and all three new target-matched references and six-parameter conditional comparisons pass the existing necessary screens. Completion, a good plot or loss alone is insufficient. This strict automatic route may remain closed; an explicit separately reviewed human interpretation would require its own bound authorization record rather than changing this declaration after seeing a failure. The worker rechecks the evaluation's exact SHA and recomputes the same screens before pricing; an execution-host path may differ only if its bytes match the frozen binding.

From the active checkout's `lab/`, these are the concrete **preparation/check commands**, with the cached core-reference Python:

```sh
PYTHONPATH=src:../tutorial/src .pixi/envs/core-reference/bin/python scripts/prepare_rough_actual_layout_pilot_portable.py check-sources --work-order evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation-portable/work-order.json
# Only after the Heston working-grid condition is satisfied:
PYTHONPATH=src:../tutorial/src .pixi/envs/core-reference/bin/python scripts/prepare_rough_actual_layout_pilot_portable.py bind-heston --work-order evidence/pilot-results/2026-11-10/bergomi-observation-refresh/preparation-portable/work-order.json --evaluation <completed-final300-Heston-evaluation.json> --out <NEW-condition.json>
```

No condition file has been created. Source deployment and execution-host verification have passed, while the completed Heston evaluations fail the diagnostic condition. No rough worker or job may start under the agreed gate. After the unchanged bound Heston evaluation and condition JSON exist, stage those exact bytes and validate the same hashes again before submission. A different host path is permitted only for matching bytes. Do not copy a Mac virtual environment or reinterpret a missing Mac absolute path as permission to skip the evaluation hash. The staged worker refuses execution before the condition is met.

Only after those gates, the prepared job command is:

```sh
export ROUGH_REPO_ROOT=/oscar/scratch/afengler/sbi4finance-rough-actual-layout-portable-20261110/repo
export ROUGH_PYTHON=/oscar/scratch/afengler/sbi4finance-full-core-20261026-wy2_j3o4/repo-conditioned-reference/.pixi/envs/core-reference/bin/python
export ROUGH_WORK_ORDER=/oscar/scratch/afengler/sbi4finance-rough-actual-layout-portable-20261110/preparation/work-order.json
export ROUGH_CONDITION=<staged-condition.json>
export ROUGH_HESTON_EVALUATION=<staged-exact-Heston-evaluation.json>
export ROUGH_PAYLOAD_ROOT=<NEW-immutable-pilot-payload-root>
sbatch /oscar/scratch/afengler/sbi4finance-rough-actual-layout-portable-20261110/preparation/pilot.sbatch
```

These commands have **not** been submitted. Each array index runs exactly one declared component; no retry, continuation, automatic path increase or bank/training job is attached. Completed expiry-block payloads and their hashes survive a later component failure. Covariances must be finite and symmetric to the declared32-epsilon relative tolerance, with strictly nonnegative nominal variances; negative variance is never clipped into a passing MCSE. Nominal MCSE passing alone still leaves `numerical_target_established=false`: paired refinement, independent-batch dispersion, likelihood perturbation and measured cost must be assessed together before another work order.

After R1, estimate a fresh bank from observed per-row distributions, failed fractions, path/rung usage, grid allocation and strict-likelihood effects. Use append-only source-bound collection machinery and separated physical-row splits, CPU label workers, then a separately bounded GPU posterior/mean-pricer stage if justified. No production row count, GPU time, surrogate reference budget or prior contraction is fixed by Heston’s cheap Fourier timings or by this plan.

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

Show the active inferred curve/c assumptions, the dormant supplied-curve/known-c distinction, physical/curve priors, realized grid coverage, numerical qualifications and model residuals beside the relevant results. Preserve cached/live and tutorial/backend limitations honestly. Keep illustrations as illustrations, and full posterior product draws paired; state whether bands are pointwise and whether uncertainty is numerical, parameter-induced or predictive.

Promote only through lab acceptance/publication tools, with compact committed evidence and external hashed payloads. Do not hand-edit `tutorial/assets`, `MANIFEST.md` or generated claims tables. Before a later implementation PR, run lab tests, manifest, claims and release-check; run tutorial tests, notebook tests and check when tutorial runtime/notebooks change. Neither passing software checks nor Heston acceptance substitutes for the independent rough scientific gates above.

**Next action:** monitor Slurm `8823501`, retain the first-1,000 results and four failed-row journals, and begin a clearly labelled development training experiment once a useful partial bank is available. Report complete, failed and missing prior IDs, numerical diagnostics, approximate costs and selection limitations together. The user has authorized the full 300,000-row generation target; Heston diagnostic failure no longer blocks execution. Official posterior calibration, rough numerical sensitivity and asset promotion remain separate unfinished work.

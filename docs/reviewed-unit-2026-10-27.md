# Reviewed numerical unit — 27 October 2026

**The reviewed Heston/rough unit is complete. We have usable Heston development references, clear neural errors to improve, and a more realistic rough-generation cost scenario. We do not yet have accepted rough pricing settings.** The unit used **4.97 allocated CPU-hours**, below its 26-hour ceiling, with no GPU training or notebook execution.

This extends Francesco's existing numerical workflow and evidence. Both rough workflows approximate model prices; the added question is whether refinement materially changes the tutorial's posterior and product conclusions. The [central plan](../tutorial_final/NUMERICAL-EXPERIMENT-PLAN.md) still governs the subsequent campaign. All results here are development evidence, not final acceptance.

## Heston: reference problem closed for these three cases

The same three stored observations, priors and noise laws were retained. Each native PyMC DEMetropolisZ run used four chains, 4,000 warmup and 8,000 retained draws per chain, data-only initialization and conditioned proposals. The difficult case ran first; its numerical and visual review passed before the other two were submitted. No neural initialization, post-hoc trimming or relaxed convergence thresholds were used.

| Development observation | Largest parameter R-hat | Smallest bulk / tail ESS | Reference elapsed time |
| --- | ---: | ---: | ---: |
| validation:0 | 1.0034 | 1,703 / 1,617 | 382 s |
| validation:1 | 1.0027 | 1,718 / 1,938 | 361 s |
| validation:2 | 1.0049 | 1,530 / 1,462 | 382 s |

All parameter and analytic variance-product diagnostics meet R-hat ≤1.01 and bulk/tail ESS ≥400. Inspection of all trace figures found overlapping chains and repeated tail visits, with small retained block shifts and boundary concentration still visible. This supports conditional development comparisons; it does not prove stationarity, mode coverage or accuracy over the prior.

The independent 16-point Fourier checks also passed their identity/completeness guards. Across all three cases the largest price change was **0.0000076 quote SD**, and the largest absolute log-likelihood change was **0.000023**. These are selected-point checks, not a full posterior refinement experiment. Saved-draw quantile analysis preserves chain/draw axes; product q10/q90 MCSEs are about **0.8–1.4% of the corresponding reference 90% width**.

The unchanged **20k-trained, 300-epoch BayesFlow 2.1 spline checkpoint fails the proposed parameter agreement screen in all three cases**. This is now an estimator finding rather than an unresolved reference diagnostic:

| Case | Illustrative conditional discrepancy |
| --- | --- |
| 0 | Initial-variance mean differs by 0.76 reference SD; its 90% interval is 1.46× as wide. Rho and vol-of-vol intervals are 1.73× and 1.62× as wide. |
| 1 | Initial-variance mean differs by 1.10 reference SD; its interval is 1.52× as wide. Theta differs by 0.62 SD. |
| 2 | Rho mean differs by 0.41 reference SD. Other parameter means are closer; theta's interval is still 1.23× as wide. |

The screen uses mean error ≤0.3 reference SD and width error ≤25%; it is separate from the stricter later numerical-refinement proposals. Marginal passes do not establish joint agreement. A borderline screen failure, such as case1 rho at 0.309 SD, should not carry the argument: the much larger initial-variance errors already resolve it. Analytic variance-product means/intervals also differ, especially in case1. Product quantiles remain descriptive rather than accepted economic-tolerance results; no forward-start call was priced here.

**Next Heston action:** diagnose on the existing bank before buying more data. Compare the archived course estimator on the same observations where its original runtime is available; check preprocessing/transforms and optimization behavior; then run a bounded longer spline fit and one course-style FlowMatching control on the same 20k data. Preserve the fixed holdouts and use validation only. The three cases here remain development cases.

[Three-case summary and hashes](pilot-results/2026-10-27/heston-reviewed/summary.json), [hard-case manual gate](pilot-results/2026-10-27/heston-reviewed/case1-gate.json). Each case directory retains full chains, neural draws, diagnostics, precision reports, logs and trace/marginal figures. Recreate figures from saved draws with `scripts/plot_reviewed_heston.py`; it performs no MCMC or repricing.

## Rough: useful pilot, material grid dependence

The declaration was frozen on Oscar before pricing: **16 prior-drawn points × two adaptive policies**, plus **three existing anchors × two policies × two independent repetitions**. The broad prior and all seeds were retained. The policies use block steps `1280/2016/2016` and `2560/4032/4032`, through 10/63/252 trading days. These are the reviewed working candidate and doubled comparator, not accepted accuracy levels.

- **Working adaptive calls: 16/16 priced**, all meeting the reported MCSE/quote-SD target of 0.25.
- **Refined adaptive calls: 15/16 priced**, also meeting that target. Point6 timed out at 20 minutes; its current rung and all preceding attempts are retained, with no replacement seed or automatic retry.
- **Fixed anchors: 12/12 priced**, all meeting the reported 0.25 target. No completed call needed the final-rung 0.5 exception.
- The seed audit passed: 708 possible block/rung streams were declared; 185 were observed, with no collision or undeclared reuse. The 32 adaptive calls are **16 paired parameter points**, not 32 independent prior samples.

The three anchors provide more precise grid comparisons than the one-repetition prior panel:

| Anchor | Selected contract | Refined minus working / quote SD | Nominal difference MCSE / quote SD |
| --- | --- | ---: | ---: |
| Low-H/high-eta, case4 | 126-day ATM put | +1.654 | 0.067 |
| High variance, case6 | 21-day call, strike 1.1224 | −0.443 | 0.034 |
| Low-H/high-eta, case4 | Five-day call, index8 | −0.619 | 0.054 |
| High variance, case6 | Five-day call, index8 | −0.273 | 0.032 |

The central anchor's largest absolute movement is 0.105 quote SD. The low-H finding also shows why refining only the shortest block cannot settle the full surface. In the adaptive prior panel, nominal uncertainty is often too large to resolve small grid shifts; one comparison is incomplete. None of these selected price differences measures posterior movement or continuous-model error.

**Recommendation:** retain `1280/2016/2016` as a development candidate and its doubled policy as the comparator, with the observed limitations attached. Do not declare either accurate, narrow the prior to remove difficult cases, or replace the planned posterior/product check with price agreement. Resolve the small reliability/cost follow-up below before freezing a substantial collection.

### What the reported Monte Carlo errors establish

Reported MCSEs come from regression residuals after fitting control variates on the same paths. Unadjusted conditional puts are bounded, but one control—the discrete effective-forward martingale—has infinite second moment in the ideal Gaussian log-Euler scheme. After two steps, its conditional second moment contains `F1^2 * exp(rho^2 * V1 * dt)`; nondegenerate lognormal `V1` gives a divergent expectation. Antithetic averaging does not fix that control's moment.

This is a mathematical observation about the control, **not a proof that the fitted OLS price estimator has infinite variance**: its coefficients depend on the sample and can respond to extremes. It does mean that bounded-payoff arguments do not by themselves justify this estimator's residual MCSE. Floating-point rejection of extreme paths does not supply that justification either.

Conditional pricing, antithetics and fitted controls have established precedent in [McCrickerd and Pakkanen](https://arxiv.org/html/1708.02563v3#S2) and [Rømer](https://doi.org/10.1080/14697688.2022.2081592). That precedent does not establish error-bar coverage for our particular implementation. The frozen pilot remains useful **descriptive grid/precision/cost evidence with approximate error bars**; meeting its reported 0.25 rule is not a proven accuracy or 95%-coverage guarantee. No estimator was changed during these runs.

## Cost: an initial bank is substantial but parallelizable

| Completed unit | Actual allocated CPU-hours | Reported total CPU-hours |
| --- | ---: | ---: |
| Three Heston reference/precision/comparison jobs | 1.358 | 1.114 |
| 44 rough pricing jobs, including timeout | 3.611 | 3.540 |
| Rough declaration | 0.002 | 0.0005 |
| **Total** | **4.971** | **4.655** |

Allocated cost is elapsed time × allocated CPUs. TotalCPU is measured user+system CPU; these columns are alternatives, not additive. Reduction and plotting were local saved-result analysis. No GPU allocation was used.

The **16 fully completed working calls average 132.5 allocated seconds per surface**, including the slow point; median 47.5 seconds. Mean measured CPU is 127.6 seconds. One call accounts for over half the allocated sample cost, so the mean is sensitive to this small panel and runtime conditions. Do not budget from the median or silently drop the slow case.

The refined timeout is especially confounded: about **95% of its CPU was system time**, versus 56% for its working counterpart. Both nodes advertise Cascade Lake CPUs, but frequency counters also differed. These records establish neither an intrinsic grid multiplier nor a particular OS/NUMA/paging cause. A matched short timing diagnostic is warranted; no timeout was rerun in this unit. [Raw performance evidence](pilot-results/2026-10-27/rough-reviewed/performance-note.json).

Conditional on retaining the working task/precision policy and repeating this small sample's mean throughput:

| Nested training size | Fixed validation + reserved test | Total price surfaces | Allocated CPU-hour scenario |
| --- | ---: | ---: | ---: |
| 20k | 5k + 7k | 32k | **1,178 h** |
| 50k | Same 12k | 62k | **2,282 h cumulative** |
| 100k | Same 12k | 112k | **4,122 h cumulative** |

At 200 continuously available one-CPU workers, the first scenario has an ideal work/cores time of **5.9 hours**. Queues, uneven shard times, startup, failures and memory/CPU contention add time. This is a planning scenario, not a completion promise or a confidence bound. All holdout generation is included; later snapshots add only missing training IDs.

For the refined policy, the incomplete cohort's *already spent* time scales to roughly **2,002 allocated CPU-hours per 32k surfaces**. The timed-out completion cost is unknown. This is a lower bound on scaling that cohort's eventual spending, not a population cost bound or a valid complete-bank forecast.

The first-bank work order therefore remains **conditional**, not submission-ready: freeze the empirical precision policy and source after the follow-up, specify a total generation ceiling from the resulting cost evidence, then create the distinct 20k/5k/7k collection and its job map with the existing incremental machinery. Keep the reserved test pool unopened. A production-method change needs a new identity; these pilot prices are not silently adopted into a bank.

The 1,178-hour figure prices generation only. One rough posterior, one forward price emulator, independent labels and at most twelve emulator-reference fits still require separately recorded training/reference limits. Their new-task costs have not been measured, so this report does not attach a misleading complete-campaign total. The existing Heston 300-epoch fit took about 245 seconds for training/reload; that is a useful Heston sizing observation, not a measured rough-emulator cost.

## Next bounded work order for review

These proposed follow-ups are small and can run in parallel. **They have not been launched.**

| Work | Concrete scope | Proposed ceiling / decision |
| --- | --- | --- |
| Rough error-bar reliability and timing | Same three anchors, working grid: 16 independent batches of 32,768 paths per anchor, comparing current controlled and raw bounded conditional-put estimates on shared draws. Record coefficients, reported errors, between-batch scatter and pooled scaling. Add four order-balanced fixed-32,768-path timings at the slow prior point (two per policy), in one allocation. | **2 allocated CPU-hours total**, one numerical thread. No new prior points or automatic reruns. Review material mean differences or scatter inconsistent with reported SE before precision-qualified bank generation. Sixteen batches can expose gross problems; they cannot certify rare-tail coverage. |
| Heston diagnosis and two controlled fits | Check preprocessing/units and the archived estimator where its original runtime is available. On the same 20k snapshot, compare a 1,000-epoch spline fit with a 500-epoch course-style FlowMatching fit, using the same declared training seed and validation selection rule; fixed validation and the three stored reference observations only. | **1 GPU-hour total**: two jobs, at most 30 minutes and four CPUs each; **4 allocated CPU-hours** for those jobs plus **1 CPU-hour** for archived comparison/analysis. Stop at limits. No acceptance-pool inspection or data growth. |

Together the proposed ceiling is **7 allocated CPU-hours plus 1 GPU-hour**. GPU jobs' CPU allocations are included in the seven hours. Settings and new diagnostic streams must be declared before execution; original failed, timed-out and provisional artifacts remain intact.

A clean replication result supports a labelled **empirical** precision policy, not a universal theorem. Adverse or unresolved evidence leaves the bank decision open and calls for a targeted estimator/claim revision, not a new infrastructure project. Once that decision is settled, execute stage C and one baseline posterior/emulator, then the already specified six-case refinement panel. Wider architecture screens and sparse-chain work remain downstream.

## Provenance and retained evidence

Reviewed plan: `9b495cf`; Heston runner: `7223e6d`; rough pilot: `65186b3`. Oscar root: `/oscar/scratch/afengler/sbi4finance-reviewed-unit-20261027-9b495cf`. Existing banks/checkpoints were reused only where identities matched; the locked runtime files match the immutable source checkouts. Oscar's displayed local date is 26 October while this review is dated 27 October; raw timestamps are retained.

The [execution ledger](pilot-results/2026-10-27/execution.json) records commands, resources, source bundles, gates and the rejected stale-dependency submission before any pricing job started. Removing that completed-job dependency changed neither cohort nor settings. [Accounting](pilot-results/2026-10-27/accounting.psv), [accounting summary](pilot-results/2026-10-27/unit-accounting-summary.json), [rough declaration](pilot-results/2026-10-27/rough-reviewed/declaration.json) and [checked reduction](pilot-results/2026-10-27/rough-reviewed/reduction/report.json) retain all outcomes. Scheduler `.batch`/`.extern` CPU is not added to top-level CPU a second time.

Validation: **153 focused integrated tests passed**; the Heston runner also completed a real short four-chain software check through reference, precision and comparison. All three production reference runs and the full checked pilot reduction are retained. Existing pricing-equivalence benchmarks remain applicable because numerical production code was unchanged in this unit. An existing one-ULP historical float-fixture discrepancy remains disclosed; its fixture was not rewritten. No new teaching acceptance or notebook execution is claimed.

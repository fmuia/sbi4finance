# Larger-bank Heston training: two batch sizes

**Completed, 10 November 2026.** Both corrected runs used all **100,000 clean Heston surfaces**, batches of **128 or 512**, and **500 additional complete bank passes**. Each branch saw the same **50 million fresh noisy examples**, starting from the same six-target 60k checkpoint. More training improved the loss and physical-parameter recovery, but **did not resolve learning of c**. The subsequently requested batch-1024/4098 comparison is recorded separately below; grid production and Bergomi work remain deferred.

Read the [offline HTML report](report.html) for parameter recovery, SBC, conditional comparisons, exact network/rate/dropout code and the exploratory input-conditioning diagnostic. All figures and styles are embedded. The [summary](summary.json), [report manifest](report-manifest.json) and [software verification](verification.json) bind the underlying evidence.

| Final result | Batch 128 | Batch 512 |
| --- | ---: | ---: |
| Additional examples | 50,000,000 | 50,000,000 |
| Additional Adam updates | 391,000 | 98,000 |
| Validation probe loss, initial −15.3073 | −16.7367 | −16.3070 |
| c posterior-median RMSE | .44705 | .44788 |
| c truth–median correlation | −.04614 | −.03848 |
| Median central-90% c width; prior 1.35 | 1.34853 | 1.34545 |
| Final marginal SBC screens | 6/6 pass | 5/6 pass |
| Both synthetic matched-posterior panels | Fail | Fail |
| Observed fit/reload wall time | 3,110.78 s | 1,933.80 s |

The constant prior-median predictor has c RMSE .43707 on these same 512 truths. The five Heston parameters still recover strongly: final truth–median correlations range .9943–.99999 for batch 128 and .9936–.99998 for batch 512. Good recovery plots and marginal SBC coexist with missed conditional information. The broad synthetic reference's c width is approximately .377; the final neural width is 3.53× that width for batch 128 and 3.51× for batch 512. Market-like ratios are 3.35× and 3.33×. The separate market case also fails; its reference concentrates at the c=2 prior ceiling, which is not a known true measurement-noise scale.

This tests two training recipes at equal additional examples. The smaller batch receives about four times as many updates. Improvement over the old 20k/60k endpoint combines a fivefold bank expansion, extra training and an explicit rate restart; no separate causal attribution or across-seed claim is made. Times were observed while both runs shared the same machine and software checks/evaluations also ran. No endpoint is promoted as an accepted inferred-c model.

## Executed specification

The observation law remains `Y_i = P_i(phi) + c*h_i*Z_i`, with independent standard-Normal errors conditional on c, frozen price half-spreads h, and independent `c ~ Uniform(.5,2)`. All 54 actual SPXW contracts, five Heston priors, carry and pricing controls remain unchanged. c is a target only; inputs are the 54 ordered noisy prices divided by h. No tau/vega component or varying spread profile is added. Existing clean labels are reused without repricing.

Both runs restore every saved model, Adam, normalization and auxiliary state array. They retain the spline flow: six coupling blocks, twelve conditioners with two 128-unit hidden layers each and effective dropout **zero** in all 24 Dropout layers. The .05 value in the original constructor is superseded by the documented zero-rate control. Both restart learning rate at 5e-5, cosine-decay by additional examples to 5e-6 over 400 passes, then hold 5e-6 through 500. Adam state is preserved. Checkpoints are retained at 25/100/250/500 passes.

Each endpoint uses the same 512 validation observations and sampling seeds as the original 60k study, with 512 posterior samples per surface and 4,096 per saved conditional case. Existing converged joint references are reused; no new MCMC is run. All 384 predeclared approximate Monte Carlo-aware mean/width screens are retained. Reserved test arrays remain unopened and tutorial assets are unchanged.

## Numerical repair and retained failures

The original [paired protocol](protocol.json) stopped both branches during pass 81 on the same valid generated c=1.9999999587495068. The normalized bounded coordinate rounded to 1 inside the backend's float32 logarithm, producing an infinite target logit before an optimizer update. All saved model and Adam arrays remained finite. This is a shared target-preprocessing failure, not evidence of batch-dependent optimizer instability.

The [failure audit](precision-failure-audit.json) preserves both raw batches and last finite states through hashes. The corrected [replay protocol](precision-replay/protocol.json) computes `log(x-lower)-log(upper-x)` in NumPy float64 before casting to float32. This algebraically identical map uses no clipping or redraws and retains the original inverse transform. Ordinary finite roundoff also changes, so the replay is not claimed to reproduce the old trajectory bit for bit. Both branches replay from the same original 60k parent, with identical raw example streams, priors, architecture, schedule and budgets.

The original deadline, 19:48:49.590152UTC, was retained. The corrected training campaign completed before 19:21UTC, without an extension. Original 25-pass evaluations and the [read-only CLI schedule compatibility](evaluation-compatibility-v1.json) remain separately recorded. The corrected producer imports its schedule class consistently and needs no compatibility shim.

## Matched MCMC recovery follow-up

The requested [eight-case MCMC recovery study](mcmc-recovery/README.md) is complete. Seven joint six-parameter fits pass the predeclared convergence and sampled numerical checks; one mixing failure remains visible and is not retried or replaced. On the identical seven ready validation datasets, c posterior-median RMSE is **.10003 for MCMC versus .46699/.46906 for the networks**. MCMC correlation with generating c is .97729, with mean central-90% width .35923 versus 1.36112/1.35404 for the networks. The [recovery report](mcmc-recovery/report.html) and [c plot](mcmc-recovery/c-recovery.png) show the matched comparison and all eight statuses.

This new sampling study supports a failure of the neural approximation to use available scale information on the converged examples. It is a small c-stratified panel, not SBC, a population coverage estimate or a general identifiability proof. All five physical parameters were unknown to MCMC; no truth or neural output was used to initialize it. No further neural, grid or Bergomi work follows automatically.

## Requested larger-batch follow-up

The user requested a further comparison with exact batches **1024 and 4098**. The [new study](large-batches/README.md) uses the same original 60k parent, 100k bank, paired augmentation stream, architecture, zero dropout and rate schedule as the completed 128/512 runs. Each arm receives another 500 complete passes / 50 million examples from that shared parent; the larger batches make 49,000 / 12,500 additional updates. The [new protocol](large-batches/protocol.json) was frozen before launch. These are separate run identities, not continuations of the finished 500-pass branches. Both runs are now complete: c RMSE is .44743/.44709 and median 90% width is 1.34287/1.34244 (prior 1.35); neither resolves c recovery. All eight evaluations and exact exposure counts pass the independent audit. The [four-batch report](large-batches/report.html) includes final/intermediate recovery, SBC, conditional comparisons and the same seven converged MCMC cases. No additional fit or grid/Bergomi production follows automatically.

## Reflection

Increasing the bank, training duration and batch size did not make this estimator use the available scale information. This does not prove that other optimizers, initializations or representations cannot learn c. The references still show that c is informative under the specified likelihood.

A [posthoc training-only covariance check](diagnostics/input-conditioning.json) provides a possible next lead: three correlated directions contain 98.23% of clean-price variance, while noise dominates 33 of 54 much smaller directions. Expected noisy-input covariance has condition number 2.14 million after per-feature standardization. Input-cast rounding is far below the expected noise scale in this check. This is global training-prior geometry, not a causal explanation or local identifiability test. A full-rank covariance whitening experiment could test the conditioning hypothesis while preserving the statistical model; none has been applied or trained here. Clean-only whitening would be nearly singular.

Full models, states, row-exposure arrays and conditional draws stay in ignored local artifacts. Compact byte-identical run exports and payload checksums are retained under precision-replay/training. Software checks pass 703 lab tests plus manifest, claims and the tutorial-runtime boundary checks. The next decision is for reflection; no revised-grid, flexible-input or Bergomi production starts from this result automatically.

# Complete corner plots away from prior boundaries

The primary showcase now uses complete corner plots: every inferred parameter on the diagonal and every pair below it. Boundary examples remain historical diagnostics, not the main demonstration of interesting posterior geometry.

## Known-curve rough Bergomi: complete and interior

[View the full three-parameter corner plot](rough-interior-complete-corner.png).

This is the existing development observation `validation:0`, with all three inferred parameters **H, eta and correlation**. Its seven forward-variance levels are supplied conditions, not additional inferred parameters. Black/gray shows the four-chain PyMC reference; blue shows the saved 20k-trained neural posterior; dashed green lines mark the generating parameters. Blue pair contours are smoothed descriptive 50% and 90% contours. All statistics use the complete 32,000 reference draws and 10,000 neural draws; pair shading is thinned deterministically for display.

The example satisfies the stated interior check: each reference marginal's central **99% interval** lies inside the inner **90% of its prior range**. Its closest such endpoint is still **20.78% of a full prior width** from the nearest bound. None of the saved reference draws lies outside that inner 90% prior box. The prior ranges are H `[.02,.35]`, eta `[.5,3.5]` and rho `[-.99,-.2]`; axes are zoomed to the posterior, not to those full prior ranges.

This case is largely regular and approximately elliptical. It does not demonstrate strong multimodality or a dramatic banana. The reference uses the existing **deterministic pricing emulator plus Gaussian quote noise**, with its documented price/grid/covariance limitations. Its geometry and the neural overlay must not be presented as accepted exact-model rough Bergomi inference. The independent point-pricing checks for case 0 and the known conditional failures in other cases remain retained.

The [plot metadata](rough-interior-complete-corner.json) records all parameters, sample axes, input identities, full prior bounds, generating vector and measured boundary margins. [Verification](rough-interior-verification.json) binds the existing source draws and checked diagnostics. Full draw payloads remain on Oscar/private temporary storage, not GitHub. [Plot source](plot_complete_corners.py) makes no model or inference calls.

## Heston: complete four-parameter interior reference

[View the full four-parameter corner plot](heston-interior-complete-corner.png).

Heston infers **four parameters**: initial variance, long-run variance, vol-of-vol and correlation; kappa remains fixed at 2. Its full four-dimensional corner uses one predeclared midpoint observation, rather than cropping or reusing the earlier boundary examples. Black/gray shows the independently computed Fourier-likelihood PyMC reference; blue shows native sampling from the frozen final-500 amortizer; dashed green lines show truth.

Job **8032768** was submitted once with four CPUs, 16 GiB and a 30-minute limit, no GPU or automatic retry. The separate allocation ceiling is three CPU-hours, including full wall exposure and a 240-second cleanup reserve. Truth `[.085,.085,.8,-.525]`, observation/noise seed, unchanged prior/design, native PyMC settings, frozen candidate and interior criterion were declared **before** observation generation or sampling. The [declaration](manifest.json) and [submission record](execution-submitted.json) retain that prospective state.

The sampler uses existing native PyMC DEMetropolisZ, four independent chains with data-only initialization, 4,000 warmup and 8,000 retained draws each. The existing 16-point Fourier-refinement check follows. An optional 10,000-draw native BayesFlow CPU readout of the frozen final-500 candidate is included in the same allocation. No bank, prior or trained model is changed. If convergence, precision or the interior criterion fails, that outcome remains visible; no automatic case substitution or resampling is allowed.

The completed reference passes its declared chain and interior checks. Maximum R-hat is **1.00297**, minimum bulk ESS **1,908**, minimum tail ESS **2,607**. Every central 99% reference interval lies inside the inner 90% prior box; the closest interval endpoint is **44.45% of a full prior width** from its nearest bound. No reference draw lies outside that inner box.

All sixteen precision points priced. The largest doubled-ladder price shift is **2.97e−6 quote standard deviations**, and the largest log-likelihood shift is **1.23e−5**. This is pointwise numerical sensitivity, not a separate posterior-stability certificate. The neural means differ from the reference by about **.021/.024/.078/.101 reference SD**, with central-90% width ratios **1.016/1.018/1.025/1.040** in task order. These are descriptive development checks, not a newly selected acceptance gate. [Detailed verification](heston-interior-verification.json) retains all marginal CDF distances as well; the largest is about .064 for rho.

The interior posterior is mostly regular and approximately Gaussian, with visible parameter dependence. We do not present it as evidence of strong non-Gaussian geometry. A complete, representative plot and agreement with the independent reference are useful starting points before seeking more weakly identified examples.

The one allocation completed **`0:0` in 476 seconds**, using **1,904 allocated CPU-seconds (.5289 CPU-hours)** and no GPU, below the separate three-hour ceiling. All 24 frozen input/source hashes were rechecked. [Execution/accounting](execution.json), [raw accounting](accounting.psv), [full run journal](run.json), [precision](precision.json) and [raw log](logs/slurm-8032768.out) retain the attempt. Seven payload files, including partial traces, remain on Oscar; [remote-artifacts.json](remote-artifacts.json) records their sizes and hashes.

[Worker](run_heston_interior.py), [one-shot preparation/submission source](prepare_heston_interior.py), [terminal collector](collect_heston_interior.py) and [plot source](plot_complete_corners.py) are one-off evidence helpers, not new tutorial runtime infrastructure. No training, model-test outcomes or extra bank surfaces were used. These midpoint observations and chains are development material, not an untouched conditional acceptance collection.

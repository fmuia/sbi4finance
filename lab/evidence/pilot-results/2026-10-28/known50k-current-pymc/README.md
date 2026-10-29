# Current known50k: fixed-case PyMC continuation

**Completed:** job **8059665 exited COMPLETED/0:0** in 189 seconds on four CPUs, 16 GiB, no GPU. All three fixed-case PyMC references passed the parameter and rho-times-eta convergence diagnostics. Current50k neural/reference parameter checks pass **cases 0 and 1**, but **case 2 still fails**: two of three, versus one of three in the historical 20k study. This is approximate-target development evidence, not scientific acceptance.

| Fixed case | Parameter checks | H mean error / width ratio | eta mean error / width ratio | rho mean error / width ratio |
| --- | --- | --- | --- | --- |
| validation:0 | pass | 0.055 / 0.967 | 0.056 / 1.086 | 0.173 / 1.119 |
| validation:1 | pass | 0.149 / 1.158 | 0.223 / 1.157 | 0.155 / 1.068 |
| validation:2 | fail | 0.822 / 1.814 | 0.789 / 1.318 | 0.290 / 1.423 |

Mean errors are absolute differences in reference posterior standard deviations; width ratios compare central 90% intervals. The pre-existing proposed rules are mean error at most 0.3 SD and relative width error at most 25%. The product's corresponding mean-error/width pairs are 0.200/1.036, 0.009/1.065 and 0.103/1.451. Reference-mean MCSE is retained, including product MCSE/SD of approximately 0.018; reference-quantile MCSE remains unassessed. The eta discrepancy on case 2 decreased from 2.052 SD to 0.789 SD, but both the neural estimator and approximate reference target changed; this does not isolate a causal training-data effect.

The same validation observations **0, 1 and 2**, seven known curve levels, prior and logit transformation are reused from the historical 20k study. Preparation checked the old and current observation/context arrays for exact equality. All old chains and reports remain unchanged.

The unchanged frozen `c96a785` consumer CLI supplies native **PyMC DEMetropolisZ**, four chains, 8,000 retained draws per chain, 4,000 warmup draws, spawn context, data-fit initialization with four starts and a total Powell budget of 2,000 evaluations. Seeds remain 271400, 271401 and 271402. Chain/draw/parameter axes and partial warmup flags are preserved.

The target is the **current50k deterministic conditional-mean pricing emulator plus the declared independent Gaussian quote noise and physical uniform prior/logit Jacobian**. Monte Carlo label randomness/covariance, emulator uncertainty and continuous-model/grid bias are not integrated. Convergence or neural agreement would establish evidence for this approximate target only.

Neural comparisons use the already saved current50k validation panel: **1,000 aligned joint draws per case**, rather than generating new inference. The older 20k comparisons used 10,000 neural draws; the different neural Monte Carlo precision is retained explicitly. The original parameter comparison statistics and proposed 0.3-reference-SD mean / 25% relative width rules are reused, with no new scientific acceptance rule. Rho times eta is computed downstream within each joint draw.

Fresh accounting of old array 8030892 confirms **788 allocated CPU-seconds**. This one additional allocation reserved at most **15,360 CPU-seconds including cleanup**, giving **16,148 seconds**, below the unchanged original **21,600-second / six CPU-hour ceiling**. Its actual cost was **756 CPU-seconds**, giving **1,544 CPU-seconds cumulatively**. No financial simulations, bank generation, GPU allocation or reserved-test inference evaluations were performed.

Current checkpoints and panel are bound in the [exact plan](plan.json). The [terminal summary](terminal-summary.json), [official terminal accounting](terminal-accounting.psv), per-case diagnostics and saved-draw comparisons retain the completed evidence. The [original accounting](prior-accounting.psv), [accepted submission](execution-submitted.json) and [first live progress](first-progress.json) remain historical launch records. Chains stay on Oscar with explicit four-chain axes and hashes in the summary. The finite driver is [run_known50k_references_20261028.py](../../../../cluster/run_known50k_references_20261028.py). Unknown submission or failures stop for review; no automatic retry, seed replacement or resumability claim is made.

# Trajectory development diagnosis

This is a descriptive reanalysis of the six existing 100k/1M trajectory fits,
using original validation histories 0–1023. The primary table uses histories
256–1023: **768 already-inspected development cases**, excluding the first 256
histories used for training-loss monitoring. Full-cohort checks are secondary.
No confirmation or final-test artifact enters this computation or the proposed
next diagnostics. The analyst saw final characterization in earlier work;
consumed tests do not become independent again.

[`summary.json`](summary.json) binds all **27 source files**, the full report and
the analysis script by SHA-256. The full report and script remain outside Git
under `local-artifacts/application-refinement-20261101/` for preservation. Its
192 model/length/parameter rows compare both training budgets. There are also
144 truth-boundary rows and 63 curve-orientation rows. All six source arrays
reproduce their reported full-1024 coverage and physical MSE exactly.

For reproduction on another machine, extract the unchanged
`completed-models-and-diagnostics.tar.gz` from the existing workshop Drive
folder. Import the preserved `diagnose_development.py` as a module, set its
`BASE` to the extracted `run/` directory and `OUT` to a new report path, then
call `run()`. This leaves the source script unchanged; the new report binds the
actual input paths/hashes and creation time. The recorded run's full report and
script are mirrored under `development-diagnosis-20261101/` in the
[workshop folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k).

| Lane | Finding on the primary development cohort | Next diagnostic |
| --- | --- | --- |
| Known-curve rough | At 1,260 days, eta's average posterior-mean error is −0.0750 prior SD, with descriptive case SE 0.00909. At 252 days it is −0.0799 ± 0.0158 prior SD. The pooled 512-draw neural mean Monte Carlo error is much smaller. Long-history drift also shows modest overdispersion: 93.10% coverage and MSE/reported-spread ratio 0.876. | Join existing bank observations to the saved means; stratify eta errors by fixed observed-history summaries at 252/1,260 days. Then establish matched conditional-reference feasibility before choosing a correction. |
| Joint-curve rough | Curve-level recovery is partial. Short-level coverage is 88.02% at 1,260 days; its mean-error/spread discrepancy is modest relative to descriptive SE. Tail misses and bias change with truth-curve orientation. Near-flat decay risk is almost the prior-mean baseline, consistent with weak information. | Stabilize an independent, task-matched short/medium-history conditional reference, retaining curve dependence. Compare actual paired curve-function draws; marginal summaries cannot establish a copula error. |
| Heston | The separately recorded atom-bearing numerical target remains unresolved. This reanalysis estimates neither atom frequency nor new reference accuracy. | Complete the separate numerical-law comparison before specifying a replacement task or fit. |

No causal fit mechanism is established and **no production fit is justified by
this report**. It does not prescribe more data, a broader architecture sweep,
global temperature scaling or an empirical mean shift. Both budgets already use
five million training views and a half-uniform/half-log-uniform prefix policy;
the recorded prefix histograms include probe calls, not just optimizer updates.
One initialization seed per model and differing data-reuse counts do not isolate
the cause of the remaining errors.

The risk baseline here uses the analytic prior mean on the same cohort. It
differs slightly from published evaluator ratios using a simulated prior mean.
The aggregate risk/spread identity accounts for 512 finite neural draws, but is
not a conditional-accuracy certificate. Standard errors are descriptive and
pointwise; nested prefixes are dependent. Truth-decile and curve-orientation
groups are posthoc descriptions: exact Bayesian intervals need not have nominal
coverage conditional on a true-parameter group. Their opposite shrinkage signs
must not be relabelled as incorrect posterior geometry.

The full case summaries contain truths, means, marginal variances and central
90% endpoints. They do not contain full-cohort joint draws, reference errors,
latent trajectories or alternative-seed fits. None of those missing statistics
was reconstructed or asserted from the small teaching panels.

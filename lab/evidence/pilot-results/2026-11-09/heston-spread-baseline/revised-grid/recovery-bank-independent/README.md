# Revised-grid Heston recovery

The [combined report](report.html) retains both fresh fits and all four declared endpoints. The broad prior, single-component price noise and neural recipe remain unchanged. More data and optimization improve c recovery and most physical coverage; initial-variance overcoverage and conditional approximation errors remain material. No checkpoint or tutorial asset is promoted.

| Primary final300 | c median RMSE | c90% coverage | Physical90% coverage | SBC screens | Raw outside prior |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old54 control,200k | 0.14421 | 89.06% | 90.62–95.51% | 4/6 | 4.56% |
| New63,200k | 0.14045 | 88.67% | 91.21–97.85% | 5/6 | 4.30% |
| New63,300k | 0.13494 | 90.23% | 89.26–97.46% | 5/6 | 3.04% |

At300k primary, initial-variance nominal90% coverage is97.46%; the other physical targets span89.26–92.38%. Five marginal SBC screens pass and initial variance fails. Synthetic conditional physical width ratios span1.05–2.86 relative to MCMC, with mean differences up to.96 reference SD. Both synthetic conditional draw sets remain within prior/physical support, so these errors do not depend on the zero-support rule. The actual-market comparison is much worse; its exact c posterior sits against the upper bound2 and cannot establish market adequacy.

The predeclared validation minima are272 for200k and259 for300k. Their population SBC screens pass4/6 and3/6;300k/259 has c RMSE.13555, c90% coverage90.23%,3.10% raw outside prior and synthetic physical widths1.08–3.09 times MCMC. Neither repairs the primary shortcomings. No archival posterior query selected a checkpoint.

The [saved-reference width comparison](reference-width-comparison.json) verifies identical generating physical truths and c=1 in the two synthetic cases. New-grid MCMC physical90% widths are40.5–87.3% of their old-grid widths: these realized new observations generally identify the parameters more tightly. That raises the accuracy required of the amortized posterior without changing the prior. Geometry, spreads and noise realizations differ together; this descriptive two-case comparison does not establish global grid optimality or isolate put placement. The actual-market width changes are mixed and are recorded separately.

**Complete: two fits, four endpoint evaluations and three ready references.** This is a separately declared operational recovery of the [first failed campaign](../campaign.json). It does not resume or overwrite that attempt.

The scientific task is unchanged: actual63 European SPXW contracts; the same broad six-parameter prior; Y=P(phi)+c*h*Z; the checked Black-control-variate Heston representation; one300k bank supporting fresh200k-prefix and300k-full nativeFlowMatching fits, each300epochs; frozen first10k ZCA; final300 primary and native validation minimum secondary.

The original900-second reference cap stopped a nearly completed bank through the controller. This recovery isolates reference failures from generation/training. All three references use a fresh coherent source-bound protocol with2400-second inner and2450-second outer limits, the same seeds, data-only initialization,2k/2k initial and6k/12k escalation schedule, and the same numerical/convergence gates. The already ready broad case remains preserved and is repeated only for protocol consistency. No case is omitted; missing, failed or nonconverged references remain inconclusive.

All original records, sources and payloads are hash-bound before this recovery starts. Numerical generation is repeated from the original retained RNG state because no complete bank payload survived. Bank/training identities, raw samples and evidence remain separate. Reports will disclose both attempts and the discarded generation work.

No Heston acceptance or Bergomi initiation follows from orchestration success. The predeclared recovery/coverage/SBC/support and matched-reference checks still determine the next step.

The recovery completed in4511.73seconds; bank/fit/evaluation workers remained inside their declared caps. Independent reconstruction verifies the300k bank, both fits, all raw endpoint metrics and1017 preserved files. All three new63 references are ready. Final software validation passes867 lab tests,53 tutorial tests and seven current-runtime notebook exports; no reserved2000-case access occurred.

Bergomi preparation is source-staged and preflight-verified on Oscar, including all23 source hashes and the original stored curve panel. No Heston initiation condition is bound and no rough pricing/job was launched. The remaining synthetic conditional errors keep the agreed condition unmet. Stop for reflection on approximation accuracy before another training or production allocation.

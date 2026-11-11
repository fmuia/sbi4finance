# Revised-grid Heston recovery

The [200k report](200k-300/report.html) is complete for primary300 and secondary272. Primary pooled SBC passes5/6 (initial variance fails); physical nominal90% coverage is91.2–97.9%, c coverage88.67%, c RMSE.14045, and4.296% of raw vectors leave prior support. Synthetic conditional physical widths remain1.07–3.31 times matched MCMC widths with mean discrepancies. All three new references are ready, but neither200k endpoint passes the necessary diagnostics. The new300k fit remains the predeclared next comparison. These are partial campaign results, not checkpoint acceptance or a rough-initiation condition.

The [saved-reference width comparison](reference-width-comparison.json) verifies identical generating physical truths and c=1 in the two synthetic cases. New-grid MCMC physical90% widths are40.5–87.3% of their old-grid widths: these realized new observations generally identify the parameters more tightly. That raises the accuracy required of the amortized posterior without changing the prior. Geometry, spreads and noise realizations differ together; this descriptive two-case comparison does not establish global grid optimality or isolate put placement. The actual-market width changes are mixed and are recorded separately.

**Running;300k results pending.** This is a separately declared operational recovery of the [first failed campaign](../campaign.json). It does not resume or overwrite that attempt.

The scientific task is unchanged: actual63 European SPXW contracts; the same broad six-parameter prior; Y=P(phi)+c*h*Z; the checked Black-control-variate Heston representation; one300k bank supporting fresh200k-prefix and300k-full nativeFlowMatching fits, each300epochs; frozen first10k ZCA; final300 primary and native validation minimum secondary.

The original900-second reference cap stopped a nearly completed bank through the controller. This recovery isolates reference failures from generation/training. All three references use a fresh coherent source-bound protocol with2400-second inner and2450-second outer limits, the same seeds, data-only initialization,2k/2k initial and6k/12k escalation schedule, and the same numerical/convergence gates. The already ready broad case remains preserved and is repeated only for protocol consistency. No case is omitted; missing, failed or nonconverged references remain inconclusive.

All original records, sources and payloads are hash-bound before this recovery starts. Numerical generation is repeated from the original retained RNG state because no complete bank payload survived. Bank/training identities, raw samples and evidence remain separate. Reports will disclose both attempts and the discarded generation work.

No Heston acceptance or Bergomi initiation follows from orchestration success. The predeclared recovery/coverage/SBC/support and matched-reference checks still determine the next step.

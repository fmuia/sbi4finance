# Revised-grid Heston recovery

**Running; results pending.** This is a separately declared operational recovery of the [first failed campaign](../campaign.json). It does not resume or overwrite that attempt.

The scientific task is unchanged: actual63 European SPXW contracts; the same broad six-parameter prior; Y=P(phi)+c*h*Z; the checked Black-control-variate Heston representation; one300k bank supporting fresh200k-prefix and300k-full nativeFlowMatching fits, each300epochs; frozen first10k ZCA; final300 primary and native validation minimum secondary.

The original900-second reference cap stopped a nearly completed bank through the controller. This recovery isolates reference failures from generation/training. All three references use a fresh coherent source-bound protocol with2400-second inner and2450-second outer limits, the same seeds, data-only initialization,2k/2k initial and6k/12k escalation schedule, and the same numerical/convergence gates. The already ready broad case remains preserved and is repeated only for protocol consistency. No case is omitted; missing, failed or nonconverged references remain inconclusive.

All original records, sources and payloads are hash-bound before this recovery starts. Numerical generation is repeated from the original retained RNG state because no complete bank payload survived. Bank/training identities, raw samples and evidence remain separate. Reports will disclose both attempts and the discarded generation work.

No Heston acceptance or Bergomi initiation follows from orchestration success. The predeclared recovery/coverage/SBC/support and matched-reference checks still determine the next step.

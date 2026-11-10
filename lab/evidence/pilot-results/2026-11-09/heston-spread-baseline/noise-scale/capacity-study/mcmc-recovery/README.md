# Matched MCMC and neural parameter recovery

**Completed, 10 November 2026: seven of eight joint MCMC fits passed the declared convergence and sampled numerical checks.** On those same seven datasets, the posterior-median recovery RMSE for c is **.10003 for MCMC**, versus **.46699 / .46906 for the batch-128 / batch-512 networks**. MCMC truth–median correlation is .97729; the networks remain almost flat. This demonstrates information about c that the current networks miss on these converged examples. It does not establish performance across the entire prior.

Read the [offline HTML report](report.html), the [c recovery plot](c-recovery.png), and the [six-parameter recovery plot](parameter-recovery.png). Vertical bars are central 90% posterior credible intervals, not Monte Carlo standard errors. The [report manifest](report-manifest.json) binds every input. The [independent reference audit](reference-verification.json), [report verification](report-verification.json), and [software verification](verification.json) retain the checks.

| Method, identical seven ready cases | c median RMSE | c truth–median correlation | Mean central-90% c width |
| --- | ---: | ---: | ---: |
| Joint MCMC | .10003 | .97729 | .35923 |
| Network, batch 128 | .46699 | −.22192 | 1.36112 |
| Network, batch 512 | .46906 | −.15145 | 1.35404 |

All eight selected datasets remain visible. Validation row 487 is **unresolved**, with maximum R-hat 1.394 and minimum bulk/tail ESS 8.47/4.56 after its predeclared escalation. Its unreliable MCMC intervals are omitted from the recovery plots and matched metrics; its neural points and failed diagnostics remain. It is not replaced or retried. For the seven ready fits, the worst R-hat is 1.00487 and the lowest bulk/tail ESS are 1166/1332. Sampled pricing refinements pass on every attempt, including the unresolved case; its failure is mixing, not a failed numerical-precision gate.

## Same observations, all six parameters unknown

The user requested direct MCMC recovery plots after the completed larger-bank experiment. The observation law remains `Y_i = P_i(phi) + c*h_i*Z_i`, with fixed quote-specific price half-spreads, independent standard-Normal Z conditional on c, five unknown Heston parameters and independent `c ~ Uniform(.5,2)`. The same normalized likelihood and independent physical-coordinate Uniform priors are retained, including the `-54*log(c)` normalization term. MCMC uses the deterministic Heston Fourier pricer, with sampled refinement checks; no price surrogate is substituted.

The eight observations were frozen before sampling from the same 512-row validation population used for neural recovery. Stable sorting by generating c and selecting zero-based ranks 32/96/160/224/288/352/416/480 spans c=.5944–1.9052. Selection uses neither physical parameters nor neural errors or MCMC outcomes. Cases are never replaced after sampling. Both final networks' saved 500-pass quantiles are reused on the identical observations, without new neural sampling.

The sampler reads only [panel.json](panel.json): noisy observation vectors, fixed design, prior and products. Generating parameters live separately in [truth-selection.json](truth-selection.json) and are used only for reporting. All five physical Heston parameters and c are inferred jointly. No true parameter, clean price, noise realization or neural draw supplies initialization. Reserved test arrays remain unopened.

## Frozen procedure and scope

The [protocol](protocol.json) freezes native PyMC DEMetropolisZ with four sequential chains, multistart data-only initialization, 2k tuning/2k retained draws per chain and one 6k/12k escalation if the first MCMC diagnostics fail. All eight cases used that escalation. Numerical refinement failure does not trigger escalation. Readiness requires R-hat ≤ 1.01 and bulk/tail ESS ≥ 400 for all six parameters and the existing variance functionals, plus sampled price/likelihood refinement checks.

Each case has a cumulative 900-second inner CPU/wall cap and 950-second outer cap. At most two cases run concurrently, with a 4500-second campaign cap and 7200 seconds of allocated inner CPU; there are no automatic retries or extensions. The [completed campaign](campaign.json) took 1151.51 wall seconds and 2038.03 summed inner CPU seconds. All 16 attempt payloads and 128 sampled refinement points were independently checked. The maximum sampled absolute log-likelihood change on refinement was 7.13e−5.

Eight cases stratified by generating c give a recovery illustration, **not SBC or an estimate of population coverage**. Finite-data posterior medians need not lie exactly on the unit line. Passing chain diagnostics is a practical convergence screen, not proof that every posterior feature is exact. Poor mixing on the eighth case does not establish that c is unidentifiable there.

Full chains and partial-draw payloads remain outside Git in local artifacts, with hashes in the compact records. The existing bank, grid, prior and observation model are unchanged. No neural retraining, revised-grid production or Bergomi work was run. Stop for reflection before another experiment.

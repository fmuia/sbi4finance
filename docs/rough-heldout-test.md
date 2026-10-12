# Rough Bergomi: one reserved-test evaluation

This is the bounded scientific follow-up in [issue #23](https://github.com/fmuia/sbi4finance/issues/23).
The [8,192-row development study](rough-reference-expansion.md) passed all
reference and neural validation gates. We now test that already selected
checkpoint on previously unevaluated synthetic panels. No new training,
reference-bank expansion or market data is needed.

## Protocol before outcomes

The [machine-readable protocol](../configs/rough_heldout_protocol.json) binds
the checkpoint, training record, bank, development results and scientific
source files by SHA256. The starting revision is `dfaaee4`. Before this
protocol was written, only test-file bytes were hashed for identity; test
arrays and outcomes were not inspected. The runner and its tests will also
be committed before executing the held-out study.

1. **Freeze the candidate and experiment.** Retain the exact task, priors,
   numerical law, 8,192 reference rows and trained flow. Use `test:0`–`test:15`
   for conditional/reference/target-mean checks, chosen by index without
   looking at outcomes. Use all 256 existing test cases for marginal SBC.
2. **Check the reference first.** Apply all eight unchanged reference gates
   to all 16 selected panels. Reuse the hash-bound independent target-price
   estimates already computed for the reference rows. Save every result. If
   any gate fails, stop before neural sampling; report reference blockage
   rather than a neural failure or success. Do not expand the bank again.
3. **Evaluate once if the reference passes.** Use 4,096 joint posterior draws
   per conditional case and 512 per SBC case. Apply the unchanged six
   conditional/price tolerances and the original marginal-SBC rule. Use new,
   predeclared sampling/rank/target seeds; never choose a seed from results.
   Save raw audit draws so ranks and coverage can be independently rebuilt.
   Finish all 16 conditional and all 256 SBC cases even if an early neural
   comparison fails. The conditional cases also belong to the SBC cohort;
   they are not 16 additional independent observations.
4. **Report and teach the outcome.** Publish source-bound results, timing,
   reference and neural/prior-control comparisons and update notebook09.
   Verify that review works offline without checkpoints, banks or market
   fixtures. Record a bounded acceptance, provisional result or reference
   blockage; do not require another attempt just to obtain a pass.

The first 16 test cases are a fixed conditional sample, not the worst cases
selected after inspection. The four old stress cases remain development
evidence and are not counted as new held-out observations. Choosing 16
balances conditional coverage and the existing downstream pricing cost; it
does not establish accuracy at every possible panel.

## What is being inferred?

The unknowns are roughness $H\in[0.1,0.3]$ and amplitude
$\eta\in[0.4,1.2]$, with independent uniform priors. The forward variance
curve, leverage, spot, carry and nine European put contracts are fixed known
context. Observations follow the declared finite numerical experiment

$$Y=\widehat C_{512,262144}(H,\eta;U)+0.25\epsilon,$$

where $U$ is fresh pricing randomness and $\epsilon\sim N(0,I_9)$ is quote
noise. The flow approximates $p(H,\eta\mid Y)$ under **Q**, the pricing
measure. It does not infer a P returns model or tomorrow's market prices.

The reference bank stores independent $(\theta_i,U_i)$ rows with clean
numerical price estimates. For observation $y$, weights are

$$w_i(y)\propto\exp\left[-\frac{\|y-\widehat C(\theta_i,U_i)\|^2}
{2(0.25)^2}\right].$$

This averages over the observation pricer's randomness. Adding its saved
covariance to the quote likelihood would count that randomness twice.
The reference is finite importance sampling, not an analytic oracle.
ESS, maximum weight, Monte Carlo precision and agreement between its two
4,096-row blocks check different aspects of reliability; they do not bound
all finite-bank bias or certify mode coverage.

The two target puts remain $(K,T)=(95,0.75)$ and $(80,1)$. We compare
**posterior mean expected prices**, using independent target randomness $V$,
not distributions of realized payoffs or quantiles of noisy price estimates.
The saved reference target estimates are reusable because their rows and
randomness are independent of these test panels. The same reference is shared
across cases, so its errors across cases need not be independent. The neural
target estimates use 128 paths per draw and a fresh seed; empirical variation
across $(\theta,V)$ draws already includes inner pricing noise. The combined
MCSE omits neural and discretization bias.

## Why both SBC and conditional checks?

For each held-out observation, SBC compares its generating parameter with
512 posterior draws and randomizes its rank. With $N=256$ independent panels,
two parameters and family $\alpha=0.05$, the same DKW rule gives

$$\delta=\sqrt{\frac{\log(4/0.05)}{2\cdot256}}\approx0.092513.$$

This is larger than the development bound of 0.065417 because that audit
contained 512 panels. It is the same rule applied to a smaller cohort, not a
threshold relaxed after seeing an outcome. The nominal family level covers
the two marginal rank checks; it does not cover every conditional gate or
every experiment in the project. Coverage intervals are descriptive individual
Wilson intervals, not additional acceptance gates.

An algorithm that ignores the observations and samples the prior can pass
marginal SBC. Therefore we also compare means, spreads, H–eta dependence,
reference mass of neural 90% intervals and downstream target means on the 16
fixed cases. The prior-only control uses the same cases and its own fixed
random stream. A broad H marginal is compatible with weak identification;
compare it with the reference rather than demanding contraction by itself.

## Boundaries and test accounting

The original development cases informed the reference expansion and remain
development evidence. This test does not erase that history. It can support
the selected finite task on the recorded held-out cohort; it cannot establish
continuum pricing accuracy, market adequacy or universal posterior accuracy.
All earlier HKDE and Heston acceptance decisions remain separate.

Once the reference stage evaluates the first 16 test cases, those cases are
used even if it blocks the neural stage. The remaining 240 cases then have
no outcome evaluation, but the original 256-case cohort is no longer wholly
untouched. After a complete neural run all 256 are used. Reproduction with
the same fixed artifacts is not a fresh acceptance attempt.

Results will be added after the frozen runner is checked and executed.

## Execution and review

The one-run command requires a new output directory and the pinned environment:

```bash
PYTHONPATH=src pixi run rough-heldout-test --out /path/to/new-heldout-result
```

It validates the frozen inputs before loading reserved arrays. A scientific
failure saves its results and returns a nonzero exit; that is a stop decision,
not a request to retry. Once results are published, the inexpensive review is
`PYTHONPATH=src pixi run review-rough-inference`, which reads cached reports and
does not execute the study. The new test protocol does not replace the
checkpoint's original training/development protocol.

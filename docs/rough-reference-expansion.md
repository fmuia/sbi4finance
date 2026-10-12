# Rough-Bergomi reference precision: one 4,096 → 8,192 follow-up

This bounded follow-up preserves the [first pilot](rough-joint-inference.md),
which stopped before neural training: 19/20 reference cases passed and
`validation:260` missed ESS and maximum-weight thresholds. The new
[protocol](../configs/rough_reference8192_protocol.json) is specified before
additional rows are generated. Its sole scientific change is the number of
independent prior-reference rows. The training bank, observation law, parameter
box, numerical grid/path counts, 20 cases, quote/target seeds, neural candidate
and every accuracy threshold remain unchanged. Already inspected audit cases
are development evidence, not fresh confirmation. The final test stays reserved.

## Ordered subplan

1. **Extend, without regenerating the prefix.** Keep the original bank intact.
   Reuse its 4,096 training/reference rows and two unused compatibility rows.
   Generate only `train:4096` through `train:8191`, using the same task and root
   seed with new stable IDs. A new bank manifest records the verified source
   bank and every suffix shard. No failed panel is redrawn.
2. **Check the expanded reference once.** Reweight all 8,192 rows for the same
   16 validation and four stress panels. Retain all eight thresholds, including
   ESS ≥800 and maximum weight ≤0.005, plus the existing stress-pricing screen.
   Compare the original and expanded diagnostics and report actual cost. Stop
   on failure; do not automatically double again or introduce a new proposal.
3. **Train only after reference acceptance.** If all reference checks pass,
   run the existing single 80-epoch spline-flow candidate on the unchanged
   1,024 training bases, with eight quote augmentations per base. Evaluate its
   conditional/joint accuracy, prior control, marginal SBC and target means
   using the fixed original settings. A checkpoint reload is not posterior
   acceptance. Leave any neural failure visible and the final test reserved.
4. **Teach and hand over the outcome.** Update notebook09 with the original
   stop, extension provenance, before/after precision and measured outcomes.
   Its default review remains offline and uses cached expensive calculations.

## Why more reference rows can help

A row is an independent parameter/pricing-randomness pair `(theta_i, U_i)`.
For a fixed nine-quote observation `y`, weights satisfy

\[
 w_i(y)\propto\exp\{-\|y-\widehat C(\theta_i,U_i)\|^2/(2\cdot0.25^2)\}.
\]

This integrates pricing randomness in the declared finite Monte Carlo model;
adding its saved covariance to the likelihood would count that noise twice.
Doubling the bank gives more opportunities to represent the small region favored
by the observation. ESS often grows approximately with bank size and Monte Carlo
error often decreases roughly as the square root of size, but neither improvement
is guaranteed for a realized bank. ESS is a concentration diagnostic, not proof
of posterior correctness. Conditional moments, block agreement and target-price
mean precision remain separate checks.

The two diagnostic blocks now contain 4,096 rows each: the original prefix and
the new suffix. Their underlying random streams are disjoint. Because this
extension was chosen after inspecting the prefix, its diagnostics are an adaptive
development check, not a new independent confirmatory experiment. Target prices
still use separate random numbers and estimate posterior mean expectation prices;
no distribution of noisy payoff averages is called a price posterior.

## Reproduction

Use the original pilot bank and the new protocol explicitly; the original
protocol and reports are never overwritten. Generate suffix chunks with
`configs/tasks/rough_reference8192_bank.json`, then assemble with the existing
CLI's `--base-bank` option. Run `scripts/check_rough_reference.py`,
`scripts/train_rough.py` and `scripts/evaluate_rough.py` with
`--protocol configs/rough_reference8192_protocol.json` and new output directories.
Training remains conditional on a matching passed reference report.

This is synthetic Q evidence only. It does not resolve continuum discretization
error, real-market adequacy, rough-memory returns forecasting or curriculum
selection. No real market data or new storage framework is required.
